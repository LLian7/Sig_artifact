import hashlib
import unittest
from math import comb

from val_strict_isp import (
    HashXOF,
    ISPParameters,
    _multiplicity_profile_from_partition_value,
    _sample_uniform_subset,
    blk,
    multiplicity_profile,
    sample_base,
    val_strict_isp,
    window_bounds,
)


def _reference_randbelow_outputs(seed_material: bytes, bounds: list[int]) -> list[int]:
    shake = hashlib.shake_256(seed_material)
    buffer = b""
    offset = 0
    outputs: list[int] = []

    for bound in bounds:
        byte_len = max(1, (bound.bit_length() + 7) // 8)
        upper = 1 << (8 * byte_len)
        threshold = upper - (upper % bound)
        while True:
            end = offset + byte_len
            if len(buffer) < end:
                target = max(end, 32 if len(buffer) < 32 else 2 * len(buffer))
                buffer = shake.digest(target)
            candidate = int.from_bytes(buffer[offset:end], "big")
            offset = end
            if candidate < threshold:
                outputs.append(candidate % bound)
                break

    return outputs


def _reference_sample_uniform_subset(size: int, subset_size: int, seed_material: bytes) -> list[int]:
    shake = hashlib.shake_256(seed_material)
    buffer = b""
    offset = 0
    positions = list(range(size))

    for index in range(subset_size):
        bound = size - index
        threshold = 256 - (256 % bound)
        while True:
            end = offset + 1
            if len(buffer) < end:
                target = max(end, 32 if len(buffer) < 32 else 2 * len(buffer))
                buffer = shake.digest(target)
            candidate = buffer[offset]
            offset = end
            if candidate < threshold:
                swap_index = index + (candidate % bound)
                positions[index], positions[swap_index] = positions[swap_index], positions[index]
                break

    result = positions[:subset_size]
    result.sort()
    return result


def _reference_sample_groups_from_seed_material(
    counts: list[int],
    partition_num: int,
    seed_material: bytes,
) -> list[list[int]]:
    shake = hashlib.shake_256(seed_material)
    buffer = b""
    offset = 0
    groups = [[] for _ in range(partition_num)]

    def randbelow(bound: int) -> int:
        nonlocal buffer, offset
        byte_len = max(1, (bound.bit_length() + 7) // 8)
        upper = 1 << (8 * byte_len)
        threshold = upper - (upper % bound)
        while True:
            end = offset + byte_len
            if len(buffer) < end:
                target = max(end, 32 if len(buffer) < 32 else 2 * len(buffer))
                buffer = shake.digest(target)
            candidate = int.from_bytes(buffer[offset:end], "big")
            offset = end
            if candidate < threshold:
                return candidate % bound

    def decode_subset_rank(rank: int, subset_size: int, universe_size: int) -> list[int]:
        positions: list[int] = []
        remaining = subset_size
        current_rank = rank
        for position in range(universe_size):
            if remaining == 0:
                break
            include_count = comb(universe_size - position - 1, remaining - 1)
            if current_rank < include_count:
                positions.append(position)
                remaining -= 1
            else:
                current_rank -= include_count
        return positions

    for value, count in enumerate(counts):
        if count == 0:
            continue
        rank = randbelow(comb(partition_num, count))
        for position in decode_subset_rank(rank, count, partition_num):
            groups[position].append(value)

    return groups


class ValStrictISPTests(unittest.TestCase):
    def test_isp_parameters_keep_symmetric_window_radius(self) -> None:
        params = ISPParameters(
            hash_len=16,
            max_g_bit=2,
            partition_num=3,
            window_radius=1,
        )

        self.assertEqual(params.window_radius, 1)
        self.assertEqual(window_bounds(params), (1, 3))

    def test_val_strict_isp_applies_window_guard(self) -> None:
        params = ISPParameters(
            hash_len=8,
            max_g_bit=2,
            partition_num=2,
            window_radius=1,
        )

        self.assertIsNotNone(val_strict_isp("00011011", params))
        self.assertIsNone(val_strict_isp("00000000", params))

    def test_hash_xof_randbelow_matches_reference_sequence(self) -> None:
        seed_material = b"val-strict-isp/randbelow"
        bounds = [32, 31, 30, 29, 28, 27, 26, 25, 257, 19, 7, 2, 1]

        reference = _reference_randbelow_outputs(seed_material, bounds)
        xof = HashXOF(seed_material)

        self.assertEqual([xof.randbelow(bound) for bound in bounds], reference)

    def test_sample_uniform_subset_matches_reference_small_size(self) -> None:
        seed_material = b"val-strict-isp/subset"
        xof = HashXOF(seed_material)

        self.assertEqual(
            _sample_uniform_subset(32, 8, xof),
            _reference_sample_uniform_subset(32, 8, seed_material),
        )

    def test_sample_uniform_subset_matches_reference_singleton(self) -> None:
        seed_material = b"val-strict-isp/subset-singleton"
        xof = HashXOF(seed_material)

        self.assertEqual(
            _sample_uniform_subset(19, 1, xof),
            _reference_sample_uniform_subset(19, 1, seed_material),
        )

    def test_sample_base_matches_reference_with_seed_material(self) -> None:
        seed_material = b"val-strict-isp/sample-base"
        block_values = [0, 0, 1, 2, 2, 3]

        self.assertEqual(
            sample_base(
                partition_value=0,
                block_values=block_values,
                partition_num=5,
                max_g_value=4,
                hash_len=8,
                xof_seed_material=seed_material,
            ),
            _reference_sample_groups_from_seed_material([2, 1, 2, 1], 5, seed_material),
        )

    def test_sample_base_matches_reference_for_small_complement_path(self) -> None:
        seed_material = b"val-strict-isp/sample-base-complement"
        block_values = [0, 0, 0, 0, 1]

        self.assertEqual(
            sample_base(
                partition_value=0,
                block_values=block_values,
                partition_num=5,
                max_g_value=4,
                hash_len=8,
                xof_seed_material=seed_material,
            ),
            _reference_sample_groups_from_seed_material([4, 1, 0, 0], 5, seed_material),
        )

    def test_val_strict_isp_matches_sample_base_without_reordering(self) -> None:
        params = ISPParameters(
            hash_len=8,
            max_g_bit=2,
            partition_num=4,
            window_radius=0,
        )
        partition_value = 30

        self.assertEqual(
            val_strict_isp(partition_value, params),
            sample_base(
                partition_value=partition_value,
                block_values=None,
                partition_num=params.partition_num,
                max_g_value=params.max_g_value,
                hash_len=params.hash_len,
                hash_name=params.hash_name,
            ),
        )

    def test_multiplicity_profile_fast_path_matches_w2_reference(self) -> None:
        partition_value = bytes.fromhex("deadbeefcafebabe1122334455667788")

        self.assertEqual(
            _multiplicity_profile_from_partition_value(partition_value, 128, 2, 4),
            multiplicity_profile(blk(partition_value, 128, 2), 4),
        )

    def test_multiplicity_profile_fast_path_matches_w4_reference(self) -> None:
        partition_value = bytes.fromhex("00112233445566778899aabbccddeeff")

        self.assertEqual(
            _multiplicity_profile_from_partition_value(partition_value, 128, 4, 16),
            multiplicity_profile(blk(partition_value, 128, 4), 16),
        )


if __name__ == "__main__":
    unittest.main()
