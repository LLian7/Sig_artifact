import hashlib
import unittest
from itertools import product
from math import comb

from treeaware_isp import (
    HashXOF,
    TreeAwareISPParameters,
    _default_prefix_dict,
    _default_laminar_pattern_masks,
    _decode_laminar_nonempty_masks,
    _laminar_count_with_empty,
    _laminar_nonempty_count,
    _multiplicity_profile_from_partition_value,
    _sample_uniform_subset,
    blk,
    multiplicity_profile,
    sample_base,
    shape_guard,
    shape_statistics,
    route_support,
    score_guard,
    score_value,
    tree_extract,
    tree_cost_pair,
    treeaware_isp,
    verify_score,
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


class TreeAwareISPTests(unittest.TestCase):
    def test_isp_parameters_keep_symmetric_window_radius(self) -> None:
        params = TreeAwareISPParameters(
            hash_len=16,
            max_g_bit=2,
            partition_num=3,
            window_radius=1,
        )

        self.assertEqual(params.window_radius, 1)
        self.assertEqual(window_bounds(params), (1, 3))

    def test_treeaware_isp_applies_window_guard(self) -> None:
        params = TreeAwareISPParameters(
            hash_len=8,
            max_g_bit=2,
            partition_num=2,
            window_radius=1,
        )

        self.assertIsNotNone(treeaware_isp("00011011", params))
        self.assertIsNone(treeaware_isp("00000000", params))

    def test_default_prefix_dict_matches_tree_first_examples(self) -> None:
        self.assertEqual(
            _default_prefix_dict(4),
            ((0, 1, 2, 3), (0, 1), (2, 3)),
        )
        self.assertEqual(
            _default_prefix_dict(8),
            (
                (0, 1, 2, 3, 4, 5, 6, 7),
                (0, 1, 2, 3),
                (4, 5, 6, 7),
                (0, 1),
                (2, 3),
                (4, 5),
                (6, 7),
            ),
        )

    def test_prefix_dict_must_be_dyadic_universe_pattern(self) -> None:
        with self.assertRaises(ValueError):
            TreeAwareISPParameters(
                hash_len=8,
                max_g_bit=2,
                partition_num=4,
                window_radius=1,
                prefix_dict=[[0, 2]],
            )

    def test_hash_xof_randbelow_matches_reference_sequence(self) -> None:
        seed_material = b"treeaware-isp/randbelow"
        bounds = [32, 31, 30, 29, 28, 27, 26, 25, 257, 19, 7, 2, 1]

        reference = _reference_randbelow_outputs(seed_material, bounds)
        xof = HashXOF(seed_material)

        self.assertEqual([xof.randbelow(bound) for bound in bounds], reference)

    def test_sample_uniform_subset_matches_reference_small_size(self) -> None:
        seed_material = b"treeaware-isp/subset"
        xof = HashXOF(seed_material)

        self.assertEqual(
            _sample_uniform_subset(32, 8, xof),
            _reference_sample_uniform_subset(32, 8, seed_material),
        )

    def test_sample_uniform_subset_matches_reference_singleton(self) -> None:
        seed_material = b"treeaware-isp/subset-singleton"
        xof = HashXOF(seed_material)

        self.assertEqual(
            _sample_uniform_subset(19, 1, xof),
            _reference_sample_uniform_subset(19, 1, seed_material),
        )

    def test_sample_base_matches_reference_with_seed_material(self) -> None:
        seed_material = b"treeaware-isp/sample-base"
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
        seed_material = b"treeaware-isp/sample-base-complement"
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

    def test_treeaware_isp_extracts_entropy_budgeted_prefix(self) -> None:
        params = TreeAwareISPParameters(
            hash_len=8,
            max_g_bit=2,
            partition_num=4,
            window_radius=1,
            prefix_dict=[[0, 1, 2, 3]],
            loss_bound=8,
            prefix_limit=1,
        )
        partition_value = "00011011"
        groups = treeaware_isp(partition_value, params)

        self.assertIsNotNone(groups)
        self.assertEqual(groups[0], [0, 1, 2, 3])
        self.assertEqual(
            multiplicity_profile([value for group in groups or [] for value in group], params.max_g_value),
            _multiplicity_profile_from_partition_value(
                partition_value,
                params.hash_len,
                params.max_g_bit,
                params.max_g_value,
            ),
        )
        extracted = tree_extract([1, 1, 1, 1], params)
        self.assertIsNotNone(extracted)
        assert extracted is not None
        self.assertEqual(extracted[0], [[0, 1, 2, 3]])
        self.assertEqual(extracted[1], (0, 0, 0, 0))
        self.assertEqual(extracted[2], 3)

    def test_shape_statistics_helper_remains_public_performance_metadata(self) -> None:
        params = TreeAwareISPParameters(
            hash_len=8,
            max_g_bit=2,
            partition_num=4,
            window_radius=1,
            shape_parms={"max_pair_rows": 0},
        )
        groups = [[0, 1], [2], [], [3]]

        self.assertFalse(shape_guard(groups, params))
        self.assertEqual(shape_statistics(groups, params)["pair_rows"], 1)
        self.assertEqual(shape_statistics(groups, params)["singleton_rows"], 2)

    def test_treeaware_isp_uses_independent_fallback_when_bt_support_is_missing(self) -> None:
        params = TreeAwareISPParameters(
            hash_len=8,
            max_g_bit=2,
            partition_num=4,
            window_radius=1,
            prefix_dict=[[0, 1, 2, 3]],
            loss_bound=0,
            prefix_limit=1,
            bt_block_size=2,
            bt_families={},
        )
        groups = treeaware_isp("00011011", params)

        self.assertIsNotNone(groups)
        self.assertEqual(
            multiplicity_profile([value for group in groups or [] for value in group], params.max_g_value),
            [1, 1, 1, 1],
        )
        self.assertEqual(route_support([1, 1, 1, 1], params), 256)

    def test_treeaware_isp_uses_block_template_route_when_support_loss_is_bounded(self) -> None:
        params = TreeAwareISPParameters(
            hash_len=8,
            max_g_bit=2,
            partition_num=4,
            window_radius=1,
            bt_block_size=2,
            bt_loss_bound=7,
            bt_families={
                2: [
                    {"counts": [1, 1, 0, 0], "realizations": [[[0], [1]]]},
                    {"counts": [0, 0, 1, 1], "realizations": [[[2], [3]]]},
                ]
            },
        )
        groups = treeaware_isp("00011011", params)

        self.assertIn(groups, ([[0], [1], [2], [3]], [[2], [3], [0], [1]]))
        self.assertEqual(route_support([1, 1, 1, 1], params), 2)

    def test_laminar_fast_count_matches_bruteforce(self) -> None:
        for max_g_value, expected_counts, rows in (
            (2, (2, 1), 3),
            (4, (2, 2, 1, 1), 4),
        ):
            masks = _default_laminar_pattern_masks(max_g_value)
            brute_force_count = 0
            for sequence in product(masks, repeat=rows):
                counts = [0] * max_g_value
                for mask in sequence:
                    for value in range(max_g_value):
                        counts[value] += (mask >> value) & 1
                if tuple(counts) == expected_counts:
                    brute_force_count += 1

            self.assertEqual(
                _laminar_count_with_empty(expected_counts, rows),
                brute_force_count,
            )

    def test_laminar_fast_decoder_enumerates_nonempty_family(self) -> None:
        for max_g_value, counts, nonempty_rows in (
            (2, (2, 1), 2),
            (4, (1, 1, 1, 0), 2),
        ):
            masks = tuple(mask for mask in _default_laminar_pattern_masks(max_g_value) if mask)
            expected: set[tuple[int, ...]] = set()
            for sequence in product(masks, repeat=nonempty_rows):
                value_counts = [0] * max_g_value
                for mask in sequence:
                    for value in range(max_g_value):
                        value_counts[value] += (mask >> value) & 1
                if tuple(value_counts) == counts:
                    expected.add(sequence)

            decoded = {
                _decode_laminar_nonempty_masks(counts, nonempty_rows, rank)
                for rank in range(_laminar_nonempty_count(counts, nonempty_rows))
            }
            self.assertEqual(decoded, expected)

    def test_independent_default_route_outputs_valid_isp(self) -> None:
        pattern_family = [[], [0], [1], [2], [3], [0, 1], [2, 3], [0, 1, 2, 3]]
        fast_params = TreeAwareISPParameters(
            hash_len=8,
            max_g_bit=2,
            partition_num=4,
            window_radius=1,
            pattern_family=pattern_family,
        )
        groups = treeaware_isp("00011011", fast_params)

        self.assertIsNotNone(groups)
        self.assertEqual(
            multiplicity_profile([value for group in groups or [] for value in group], fast_params.max_g_value),
            [1, 1, 1, 1],
        )

    def test_vrf_mode_still_uses_profile_routing(self) -> None:
        params = TreeAwareISPParameters(
            hash_len=8,
            max_g_bit=2,
            partition_num=4,
            mode="vrf",
            window_radius=1,
        )

        groups = treeaware_isp("00011011", params)

        self.assertIsNotNone(groups)
        self.assertGreater(route_support([1, 1, 1, 1], params), 0)
        self.assertLess(route_support([1, 1, 1, 1], params), 256)
        self.assertEqual(
            multiplicity_profile([value for group in groups or [] for value in group], params.max_g_value),
            [1, 1, 1, 1],
        )

    def test_entropy_floor_can_force_more_diverse_profile(self) -> None:
        params = TreeAwareISPParameters(
            hash_len=8,
            max_g_bit=2,
            partition_num=4,
            aux_t={"entropy_floor": 4},
            window_radius=1,
        )

        self.assertEqual(route_support([1, 1, 1, 1], params), 24)

    def test_dual_score_guard_can_abort_without_resampling(self) -> None:
        params = TreeAwareISPParameters(
            hash_len=8,
            max_g_bit=2,
            partition_num=4,
            mode="vrf",
            size_threshold=1,
            window_radius=1,
        )

        self.assertIsNone(treeaware_isp("00011011", params))

    def test_profile_routing_uses_public_shape_profile_permutation(self) -> None:
        params = TreeAwareISPParameters(
            hash_len=8,
            max_g_bit=2,
            partition_num=4,
            aux_t={
                "dy_shape_family": [[], [0], [1], [2], [3]],
                "shape_profile": [[0], [1], [2], [3]],
                "entropy_floor": 0,
            },
            window_radius=1,
        )

        groups = treeaware_isp("00011011", params)

        self.assertIsNotNone(groups)
        self.assertEqual(route_support([1, 1, 1, 1], params), 24)
        self.assertEqual(
            sorted(tuple(group) for group in groups or []),
            [(0,), (1,), (2,), (3,)],
        )

    def test_score_helpers_match_manual_costs(self) -> None:
        params = TreeAwareISPParameters(
            hash_len=8,
            max_g_bit=2,
            partition_num=4,
            vrf_threshold=16,
            window_radius=1,
        )
        groups = [[0], [1], [2], [3]]

        selected_cost, complement_cost = tree_cost_pair(groups, params)

        self.assertEqual(score_value(groups, params, "size"), selected_cost + complement_cost)
        self.assertEqual(
            verify_score(groups, params),
            3 * params.block_num - params.forest_root_num - selected_cost + complement_cost,
        )
        self.assertTrue(score_guard(groups, params))

    def test_dual_score_guard_requires_both_thresholds(self) -> None:
        params = TreeAwareISPParameters(
            hash_len=8,
            max_g_bit=2,
            partition_num=4,
            size_threshold=64,
            vrf_threshold=4,
            window_radius=1,
        )
        groups = [[0], [1], [2], [3]]

        self.assertFalse(score_guard(groups, params))
        self.assertTrue(score_guard(groups, params, vrf_threshold=16))

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
