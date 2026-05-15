from __future__ import annotations

import argparse
from functools import lru_cache
import hashlib
import json
import math
import struct
from dataclasses import dataclass
from random import Random
from typing import List, Optional, Sequence, Union

from operation_counter import enabled as counters_enabled, increment

try:
    import _val_strict_isp_native
except ImportError:
    _val_strict_isp_native = None
_NATIVE_BYTES = (
    getattr(_val_strict_isp_native, "val_strict_isp_bytes", None)
    if _val_strict_isp_native is not None
    else None
)
_NATIVE_BYTES_PREFIXED_SEED = (
    getattr(_val_strict_isp_native, "val_strict_isp_bytes_prefixed_seed", None)
    if _val_strict_isp_native is not None
    else None
)
_NATIVE_BYTES_DEFAULT_SEED = (
    getattr(_val_strict_isp_native, "val_strict_isp_bytes_default_seed", None)
    if _val_strict_isp_native is not None
    else None
)
_NATIVE_BYTES_RANDOM = (
    getattr(_val_strict_isp_native, "val_strict_isp_bytes_random", None)
    if _val_strict_isp_native is not None
    else None
)
_NATIVE_BYTES_RANDOM_FAST = (
    getattr(_val_strict_isp_native, "val_strict_isp_bytes_random_fast", None)
    if _val_strict_isp_native is not None
    else None
)
_NATIVE_FIND_FIRST_RANDOM_STREAM = (
    getattr(_val_strict_isp_native, "val_strict_isp_find_first_random_stream", None)
    if _val_strict_isp_native is not None
    else None
)
_NATIVE_PROFILE_COUNTS = (
    getattr(_val_strict_isp_native, "val_strict_isp_profile_counts", None)
    if _val_strict_isp_native is not None
    else None
)
_NATIVE_ACCEPT_CHECK = (
    getattr(_val_strict_isp_native, "val_strict_isp_accept_check", None)
    if _val_strict_isp_native is not None
    else None
)
_NATIVE_ACCEPT_CHECK_FAST = (
    getattr(_val_strict_isp_native, "val_strict_isp_accept_check_fast", None)
    if _val_strict_isp_native is not None
    else None
)
_NATIVE_ACCEPT_CHECK_BATCH_FAST = (
    getattr(_val_strict_isp_native, "val_strict_isp_accept_check_batch_fast", None)
    if _val_strict_isp_native is not None
    else None
)
_NATIVE_PROFILE_COUNTS_BATCH_FAST = (
    getattr(_val_strict_isp_native, "val_strict_isp_profile_counts_batch_fast", None)
    if _val_strict_isp_native is not None
    else None
)
_NATIVE_W4_ACCEPT_CHECK_FAST = (
    getattr(_val_strict_isp_native, "val_strict_isp_w4_accept_check_fast", None)
    if _val_strict_isp_native is not None
    else None
)
_NATIVE_W4_ACCEPT_CHECK_BATCH_FAST = (
    getattr(_val_strict_isp_native, "val_strict_isp_w4_accept_check_batch_fast", None)
    if _val_strict_isp_native is not None
    else None
)
_NATIVE_PREPARE_PLAN = (
    getattr(_val_strict_isp_native, "val_strict_isp_prepare_plan", None)
    if _val_strict_isp_native is not None
    else None
)
_NATIVE_W4_BYTES = (
    getattr(_val_strict_isp_native, "val_strict_isp_w4_bytes", None)
    if _val_strict_isp_native is not None
    else None
)
_NATIVE_W4_BYTES_PREFIXED_SEED = (
    getattr(_val_strict_isp_native, "val_strict_isp_w4_bytes_prefixed_seed", None)
    if _val_strict_isp_native is not None
    else None
)
_NATIVE_W4_BYTES_DEFAULT_SEED = (
    getattr(_val_strict_isp_native, "val_strict_isp_w4_bytes_default_seed", None)
    if _val_strict_isp_native is not None
    else None
)
_NATIVE_W4_BYTES_RANDOM = (
    getattr(_val_strict_isp_native, "val_strict_isp_w4_bytes_random", None)
    if _val_strict_isp_native is not None
    else None
)
_NATIVE_W4_BYTES_RANDOM_FAST = (
    getattr(_val_strict_isp_native, "val_strict_isp_w4_bytes_random_fast", None)
    if _val_strict_isp_native is not None
    else None
)

PartitionValueInput = Union[str, bytes, int]
Groups = List[List[int]]
GroupMasks = bytes | tuple[int, ...]
GroupsOrMasks = Groups | GroupMasks
MessageInput = Union[str, bytes]

SUPPORTED_HASHES = {"shake_128", "shake_256", "sha3_256", "sha3_512"}
DEFAULT_HASH_NAME = "shake_256"
_RAND_BELOW_U8_THRESHOLDS = tuple(0 if bound == 0 else 256 - (256 % bound) for bound in range(257))
_POSITION_TEMPLATES: dict[int, List[int]] = {}
_BINOMIAL_TABLES: dict[int, tuple[tuple[int, ...], ...]] = {}
_SUBSET_RANK_PARAMS: dict[int, tuple[tuple[int, ...], tuple[int, ...], tuple[int, ...]]] = {}
_SAMPLE_BASE_PARAMS: dict[
    int,
    tuple[
        tuple[tuple[int, ...], ...],
        tuple[int, ...],
        tuple[int, ...],
        tuple[int, ...],
        tuple[tuple[int, ...], ...],
        tuple[object | None, ...],
    ],
] = {}
_W2_LOW_BIT_MASKS: dict[int, int] = {}
_INT_UNPACKERS = {
    2: struct.Struct(">H").unpack_from,
    4: struct.Struct(">I").unpack_from,
    8: struct.Struct(">Q").unpack_from,
}
_SMALL_SUBSET_UNRANK_THRESHOLD = 4
_SMALL_SUBSET_DECODE_TABLE_MAX = 32
_NATIVE_MAX_PARTITION_NUM = 64
_NATIVE_MAX_G_VALUE = 64
_NATIVE_MIN_MAX_G_BIT = 2
_NATIVE_MAX_MAX_G_BIT = 6
_NATIVE_W4_MAX_PARTITION_NUM = _NATIVE_MAX_PARTITION_NUM
_SPARSE_PROFILE_MIN_MAX_G_VALUE = 4096
_STREAM_BATCH_TARGET_MISS_PROBABILITY = 0.20
_STREAM_BATCH_MAX_CANDIDATES = 512


def _native_bytes_supported(params: "ISPParameters") -> bool:
    return (
        params._window_valid
        and _NATIVE_MIN_MAX_G_BIT <= params.max_g_bit <= _NATIVE_MAX_MAX_G_BIT
        and params._max_g_value <= _NATIVE_MAX_G_VALUE
        and params.partition_num <= _NATIVE_MAX_PARTITION_NUM
    )


def _sparse_profile_worthwhile(params: "ISPParameters") -> bool:
    return params._max_g_value >= _SPARSE_PROFILE_MIN_MAX_G_VALUE


@lru_cache(maxsize=None)
def _window_accept_probability(
    block_num: int,
    max_g_value: int,
    window_low: int,
    window_high: int,
) -> float:
    if block_num < 0 or max_g_value <= 0 or window_low < 0 or window_high < window_low:
        return 0.0
    if block_num == 0:
        return 1.0 if window_low == 0 else 0.0
    if window_low * max_g_value > block_num or window_high * max_g_value < block_num:
        return 0.0

    high = min(window_high, block_num)
    count_terms = [math.exp(-math.lgamma(count + 1)) for count in range(window_low, high + 1)]
    dp = [0.0] * (block_num + 1)
    dp[0] = 1.0
    for _ in range(max_g_value):
        next_dp = [0.0] * (block_num + 1)
        for total, coefficient in enumerate(dp):
            if coefficient == 0.0:
                continue
            max_count = min(high, block_num - total)
            for count in range(window_low, max_count + 1):
                next_dp[total + count] += coefficient * count_terms[count - window_low]
        dp = next_dp

    coefficient = dp[block_num]
    if coefficient <= 0.0:
        return 0.0
    log_probability = (
        math.lgamma(block_num + 1)
        + math.log(coefficient)
        - block_num * math.log(max_g_value)
    )
    if log_probability <= -745.0:
        return 0.0
    return min(1.0, max(0.0, math.exp(log_probability)))


def _stream_batch_candidates_from_accept_probability(accept_probability: float) -> int:
    if accept_probability <= 0.0:
        return 32
    if accept_probability >= 1.0:
        return 1
    batch = math.ceil(
        math.log(_STREAM_BATCH_TARGET_MISS_PROBABILITY)
        / math.log1p(-accept_probability)
    )
    return max(1, min(_STREAM_BATCH_MAX_CANDIDATES, batch))


def recommended_stream_sampler_bytes(params: "ISPParameters", *, margin_bytes: int = 16) -> int:
    """Return a safe default for drawing ValStrictISP ranks from a shared XOF stream."""

    if margin_bytes < 0:
        raise ValueError("margin_bytes must be non-negative")
    if not params.window_valid:
        return max(32, margin_bytes)

    _, byte_lengths, _ = _subset_rank_parameters(params.partition_num)
    low = max(0, params.window_low)
    high = min(params.partition_num, params.window_high, params.block_num)
    if high < low:
        return max(32, margin_bytes)

    max_rank_bytes = 1
    for count in range(low, high + 1):
        max_rank_bytes = max(max_rank_bytes, byte_lengths[count])
    base_bytes = params.max_g_value * max_rank_bytes
    if params.max_g_value <= 4:
        return max(32, base_bytes + margin_bytes)
    return max(64, base_bytes + 2 * margin_bytes)


@dataclass(frozen=True)
class ISPParameters:
    """Parameters for the ValStrictISP algorithm."""

    hash_len: int
    max_g_bit: int
    partition_num: int
    window_radius: int
    link_threshold: int = -1
    hash_name: str = DEFAULT_HASH_NAME

    def __post_init__(self) -> None:
        if self.hash_len <= 0:
            raise ValueError("hash_len must be positive")
        if self.max_g_bit <= 0:
            raise ValueError("max_g_bit must be positive")
        if self.partition_num <= 0:
            raise ValueError("partition_num must be positive")
        if self.link_threshold < -1:
            raise ValueError("link_threshold must be at least -1")
        if self.hash_len % self.max_g_bit != 0:
            raise ValueError("hash_len must be divisible by max_g_bit")
        if self.hash_name not in SUPPORTED_HASHES:
            raise ValueError(
                f"unsupported hash_name={self.hash_name!r}; choose from {sorted(SUPPORTED_HASHES)}"
            )
        block_num = self.hash_len // self.max_g_bit
        max_g_value = 1 << self.max_g_bit
        avg_floor = block_num // max_g_value
        avg_ceil = (block_num + max_g_value - 1) // max_g_value
        if self.window_radius < 0:
            raise ValueError("window_radius must be non-negative")
        low = avg_floor - self.window_radius
        high = min(avg_ceil + self.window_radius, self.partition_num)
        object.__setattr__(self, "_block_num", block_num)
        object.__setattr__(self, "_max_g_value", max_g_value)
        object.__setattr__(self, "_window_low", low)
        object.__setattr__(self, "_window_high", high)
        object.__setattr__(self, "_window_valid", low >= 0 and high >= low)
        object.__setattr__(self, "_small_partition_fast_path", self.partition_num <= 256)
        object.__setattr__(self, "_sample_base_params", _sample_base_parameters(self.partition_num))
        accept_probability = 0.0
        if low >= 0 and high >= low and max_g_value <= _NATIVE_MAX_G_VALUE and block_num <= max_g_value * self.partition_num:
            accept_probability = _window_accept_probability(block_num, max_g_value, low, high)
        object.__setattr__(self, "_accept_probability", accept_probability)
        object.__setattr__(
            self,
            "_stream_batch_candidates",
            _stream_batch_candidates_from_accept_probability(accept_probability),
        )
        native_prepare = _NATIVE_PREPARE_PLAN
        if (
            native_prepare is not None
            and low >= 0
            and high >= low
            and self.partition_num <= _NATIVE_MAX_PARTITION_NUM
        ):
            native_prepare(self.partition_num, low, high)

    @property
    def block_num(self) -> int:
        return self._block_num

    @property
    def max_g_value(self) -> int:
        return self._max_g_value

    @property
    def window_low(self) -> int:
        return self._window_low

    @property
    def window_high(self) -> int:
        return self._window_high

    @property
    def window_valid(self) -> bool:
        return self._window_valid

    @property
    def small_partition_fast_path(self) -> bool:
        return self._small_partition_fast_path

    @property
    def sample_base_params(
        self,
    ) -> tuple[
        tuple[tuple[int, ...], ...],
        tuple[int, ...],
        tuple[int, ...],
        tuple[int, ...],
        tuple[tuple[int, ...], ...],
        tuple[object | None, ...],
    ]:
        return self._sample_base_params

    @property
    def accept_probability(self) -> float:
        return self._accept_probability

    @property
    def expected_retries(self) -> float:
        return math.inf if self._accept_probability <= 0.0 else 1.0 / self._accept_probability

    @property
    def stream_batch_candidates(self) -> int:
        return self._stream_batch_candidates


def _to_message_bytes(message: MessageInput) -> bytes:
    if isinstance(message, bytes):
        return message
    if isinstance(message, str):
        return message.encode("utf-8")
    raise TypeError("message must be bytes or str")


def _serialize_bitstring(bitstring: str) -> bytes:
    if not bitstring:
        return (0).to_bytes(8, "big")

    byte_len = (len(bitstring) + 7) // 8
    integer_value = int(bitstring, 2)
    return len(bitstring).to_bytes(8, "big") + integer_value.to_bytes(byte_len, "big")


def _serialize_partition_value(partition_value: PartitionValueInput, hash_len: int) -> bytes:
    byte_len = (hash_len + 7) // 8

    if isinstance(partition_value, str):
        return _serialize_bitstring(normalize_partition_value(partition_value, hash_len))

    if isinstance(partition_value, bytes):
        if len(partition_value) != byte_len:
            raise ValueError(
                f"bytes length mismatch: expected {byte_len} bytes for hash_len={hash_len}"
            )
        extra_bits = 8 * byte_len - hash_len
        if extra_bits == 0:
            payload = partition_value
        else:
            payload = (_partition_value_to_int(partition_value, hash_len)).to_bytes(byte_len, "big")
        return hash_len.to_bytes(8, "big") + payload

    if isinstance(partition_value, int):
        integer_value = _partition_value_to_int(partition_value, hash_len)
        return hash_len.to_bytes(8, "big") + integer_value.to_bytes(byte_len, "big")

    raise TypeError("partition_value must be a bitstring, bytes, or non-negative integer")


def _hash_bytes(data: bytes, output_bytes: int, hash_name: str) -> bytes:
    if hash_name == "shake_128":
        increment("hash.backend_calls")
        increment("hash.backend_calls.shake_128")
        increment("isp.hash_backend_calls")
        return hashlib.shake_128(data).digest(output_bytes)
    if hash_name == "shake_256":
        increment("hash.backend_calls")
        increment("hash.backend_calls.shake_256")
        increment("isp.hash_backend_calls")
        return hashlib.shake_256(data).digest(output_bytes)
    if hash_name == "sha3_256":
        increment("hash.backend_calls")
        increment("hash.backend_calls.sha3_256")
        increment("isp.hash_backend_calls")
        return hashlib.sha3_256(data).digest()
    if hash_name == "sha3_512":
        increment("hash.backend_calls")
        increment("hash.backend_calls.sha3_512")
        increment("isp.hash_backend_calls")
        return hashlib.sha3_512(data).digest()
    raise ValueError(f"unsupported hash_name={hash_name!r}")


def hash_message_to_partition_value(
    message: MessageInput,
    hash_len: int,
    hash_name: str = DEFAULT_HASH_NAME,
) -> str:
    """
    Hash an arbitrary message into a bitstring Y in {0,1}^hash_len.

    SHAKE works natively in XOF mode. For fixed-length SHA3 variants, the
    digest is expanded by hashing a domain-separated counter.
    """

    if hash_len <= 0:
        raise ValueError("hash_len must be positive")
    if hash_name not in SUPPORTED_HASHES:
        raise ValueError(
            f"unsupported hash_name={hash_name!r}; choose from {sorted(SUPPORTED_HASHES)}"
        )

    output_bytes = (hash_len + 7) // 8
    message_bytes = _to_message_bytes(message)

    if hash_name.startswith("shake_"):
        digest = _hash_bytes(message_bytes, output_bytes, hash_name)
    else:
        blocks = []
        counter = 0
        while sum(len(block) for block in blocks) < output_bytes:
            block_input = (
                b"ValStrictISP/PartitionValue/"
                + counter.to_bytes(4, "big")
                + message_bytes
            )
            blocks.append(_hash_bytes(block_input, 0, hash_name))
            counter += 1
        digest = b"".join(blocks)[:output_bytes]

    digest_bits = "".join(f"{byte:08b}" for byte in digest)
    return digest_bits[:hash_len]


def normalize_partition_value(partition_value: PartitionValueInput, hash_len: int) -> str:
    """Normalize the partition value into a bitstring of length hash_len."""

    if isinstance(partition_value, str):
        if len(partition_value) != hash_len:
            raise ValueError(
                f"bitstring length mismatch: expected {hash_len}, got {len(partition_value)}"
            )
        if any(bit not in {"0", "1"} for bit in partition_value):
            raise ValueError("bitstring input must contain only '0' and '1'")
        return partition_value

    return f"{_partition_value_to_int(partition_value, hash_len):0{hash_len}b}"


def _partition_value_to_int(partition_value: PartitionValueInput, hash_len: int) -> int:
    if isinstance(partition_value, str):
        if len(partition_value) != hash_len:
            raise ValueError(
                f"bitstring length mismatch: expected {hash_len}, got {len(partition_value)}"
            )
        if any(bit not in {"0", "1"} for bit in partition_value):
            raise ValueError("bitstring input must contain only '0' and '1'")
        return int(partition_value, 2)

    if isinstance(partition_value, bytes):
        expected_bytes = (hash_len + 7) // 8
        if len(partition_value) != expected_bytes:
            raise ValueError(
                f"bytes length mismatch: expected {expected_bytes} bytes for hash_len={hash_len}"
            )
        integer_value = int.from_bytes(partition_value, "big")
        extra_bits = 8 * expected_bytes - hash_len
        if extra_bits:
            integer_value >>= extra_bits
        return integer_value

    if isinstance(partition_value, int):
        if partition_value < 0 or partition_value >= (1 << hash_len):
            raise ValueError("integer partition_value is out of range for the given hash_len")
        return partition_value

    raise TypeError("partition_value must be a bitstring, bytes, or non-negative integer")


def blk(partition_value: PartitionValueInput, hash_len: int, max_g_bit: int) -> List[int]:
    """Parse the partition value into base-2^max_g_bit blocks."""

    if isinstance(partition_value, str):
        bitstring = normalize_partition_value(partition_value, hash_len)
        return [
            int(bitstring[offset : offset + max_g_bit], 2)
            for offset in range(0, hash_len, max_g_bit)
        ]

    integer_value = _partition_value_to_int(partition_value, hash_len)
    block_num = hash_len // max_g_bit
    mask = (1 << max_g_bit) - 1
    return [
        (integer_value >> shift) & mask
        for shift in range((block_num - 1) * max_g_bit, -1, -max_g_bit)
    ]


def multiplicity_profile(block_values: Sequence[int], max_g_value: int) -> List[int]:
    """Compute the labeled multiplicity profile of block_values."""

    counts = [0] * max_g_value
    for value in block_values:
        if value < 0 or value >= max_g_value:
            raise ValueError(f"block value {value} is outside [0, {max_g_value - 1}]")
        counts[value] += 1
    return counts


def _sparse_multiplicity_profile_from_partition_value(
    partition_value: PartitionValueInput,
    hash_len: int,
    max_g_bit: int,
) -> list[tuple[int, int]]:
    sparse_counts: dict[int, int] = {}
    for value in blk(partition_value, hash_len, max_g_bit):
        sparse_counts[value] = sparse_counts.get(value, 0) + 1
    return sorted(sparse_counts.items())


def _multiplicity_profile_from_partition_value(
    partition_value: PartitionValueInput,
    hash_len: int,
    max_g_bit: int,
    max_g_value: int,
) -> List[int]:
    if isinstance(partition_value, bytes):
        if max_g_bit == 2 and max_g_value == 4:
            byte_len = len(partition_value)
            mask = _W2_LOW_BIT_MASKS.get(byte_len)
            if mask is None:
                mask = int.from_bytes(b"\x55" * byte_len, "big")
                _W2_LOW_BIT_MASKS[byte_len] = mask
            extra_bits = 8 * byte_len - hash_len
            if extra_bits:
                mask >>= extra_bits
            integer_value = int.from_bytes(partition_value, "big") >> extra_bits
            low_bits = integer_value & mask
            high_bits = (integer_value >> 1) & mask
            not_low_bits = mask ^ low_bits
            not_high_bits = mask ^ high_bits
            return [
                (not_low_bits & not_high_bits).bit_count(),
                (low_bits & not_high_bits).bit_count(),
                (not_low_bits & high_bits).bit_count(),
                (low_bits & high_bits).bit_count(),
            ]

        if hash_len % 8 == 0 and max_g_bit == 4 and max_g_value == 16:
            counts = [0] * 16
            for byte in partition_value:
                counts[byte >> 4] += 1
                counts[byte & 0x0F] += 1
            return counts

    integer_value = _partition_value_to_int(partition_value, hash_len)
    counts = [0] * max_g_value
    mask = (1 << max_g_bit) - 1
    block_num = hash_len // max_g_bit
    for shift in range((block_num - 1) * max_g_bit, -1, -max_g_bit):
        counts[(integer_value >> shift) & mask] += 1
    return counts


def window_bounds(params: ISPParameters) -> tuple[int, int]:
    """Return the multiplicity window [low, high]."""

    return params.window_low, params.window_high


def _position_template_base(size: int) -> List[int]:
    template = _POSITION_TEMPLATES.get(size)
    if template is None:
        template = list(range(size))
        _POSITION_TEMPLATES[size] = template
    return template


def _position_template(size: int) -> List[int]:
    return _position_template_base(size)[:]


@lru_cache(maxsize=8192)
def _sorted_values_from_mask(mask: int) -> tuple[int, ...]:
    values = []
    current_mask = mask
    while current_mask:
        low_bit = current_mask & -current_mask
        values.append(low_bit.bit_length() - 1)
        current_mask ^= low_bit
    return tuple(values)


def _materialize_group_masks(
    group_masks: Sequence[int],
    winner_shift: int,
    partition_num: int,
) -> Groups:
    return [
        list(_sorted_values_from_mask(group_masks[(winner_shift + offset) % partition_num]))
        for offset in range(partition_num)
    ]


def _compress_group_masks(group_masks: Sequence[int], max_g_value: int) -> GroupMasks:
    if max_g_value <= 8:
        return bytes(group_masks)
    return tuple(group_masks)


def _group_masks_from_groups(groups: Sequence[Sequence[int]], max_g_value: int) -> GroupMasks:
    group_masks = [0] * len(groups)
    for group_index, subgroup in enumerate(groups):
        mask = 0
        for value in subgroup:
            mask |= 1 << value
        group_masks[group_index] = mask
    return _compress_group_masks(group_masks, max_g_value)


class HashXOF:
    """
    Deterministic byte stream derived from H(Y) in XOF mode.

    Python's SHAKE API returns prefixes on demand, so we cache the current
    prefix and expose a stream-like read interface.
    """

    __slots__ = ("_shake", "_buffer", "_offset")

    def __init__(self, seed_material: bytes, hash_name: str = DEFAULT_HASH_NAME) -> None:
        counting = counters_enabled()
        if hash_name == "shake_128":
            self._shake = hashlib.shake_128(seed_material)
            if counting:
                increment("hash.backend_calls")
                increment("hash.backend_calls.shake_128")
        else:
            # We default to SHAKE256 even when the outer hash is SHA3, so the
            # sampler still behaves like an XOF-backed random oracle.
            self._shake = hashlib.shake_256(seed_material)
            if counting:
                increment("hash.backend_calls")
                increment("hash.backend_calls.shake_256")
        if counting:
            increment("isp.xof_instances")
        self._buffer = b""
        self._offset = 0

    def _ensure_buffer(self, end: int) -> None:
        if len(self._buffer) >= end:
            return
        target = max(end, 32 if len(self._buffer) < 32 else 2 * len(self._buffer))
        self._buffer = self._shake.digest(target)

    def read(self, byte_count: int) -> bytes:
        if byte_count < 0:
            raise ValueError("byte_count must be non-negative")
        if byte_count > 0 and counters_enabled():
            increment("isp.xof_output_bytes", byte_count)
            increment("isp.xof_output_bits", 8 * byte_count)
        end = self._offset + byte_count
        self._ensure_buffer(end)
        chunk = self._buffer[self._offset : end]
        self._offset = end
        return chunk

    def read_byte(self) -> int:
        if counters_enabled():
            increment("isp.xof_output_bytes", 1)
            increment("isp.xof_output_bits", 8)
        end = self._offset + 1
        self._ensure_buffer(end)
        value = self._buffer[self._offset]
        self._offset = end
        return value

    def randbelow(self, bound: int) -> int:
        if bound <= 0:
            raise ValueError("bound must be positive")
        if bound <= 256:
            threshold = _RAND_BELOW_U8_THRESHOLDS[bound]
            candidate = self.read_byte()
            while candidate >= threshold:
                candidate = self.read_byte()
            return candidate % bound

        byte_len = max(1, (bound.bit_length() + 7) // 8)
        upper = 1 << (8 * byte_len)
        threshold = upper - (upper % bound)

        candidate = int.from_bytes(self.read(byte_len), "big")
        while candidate >= threshold:
            candidate = int.from_bytes(self.read(byte_len), "big")
        return candidate % bound


def _sample_uniform_subset(size: int, subset_size: int, xof: HashXOF) -> List[int]:
    """
    Sample a uniform subset of [0, size - 1] using a partial Fisher-Yates scan.
    """

    if subset_size < 0 or subset_size > size:
        raise ValueError("subset_size must lie in [0, size]")
    if subset_size == 0:
        return []

    if size <= 256:
        counting = counters_enabled()
        thresholds = _RAND_BELOW_U8_THRESHOLDS
        ensure_buffer = xof._ensure_buffer
        buffer = xof._buffer
        offset = xof._offset
        bytes_drawn = 0

        if subset_size == 1:
            threshold = thresholds[size]
            end = offset + 1
            if len(buffer) < end:
                ensure_buffer(end)
                buffer = xof._buffer
            candidate = buffer[offset]
            offset = end
            bytes_drawn += 1
            while candidate >= threshold:
                end = offset + 1
                if len(buffer) < end:
                    ensure_buffer(end)
                    buffer = xof._buffer
                candidate = buffer[offset]
                offset = end
                bytes_drawn += 1
            xof._offset = offset
            if counting:
                increment("isp.xof_output_bytes", bytes_drawn)
                increment("isp.xof_output_bits", 8 * bytes_drawn)
            return [candidate % size]

        position_template = _position_template_base(size)
        positions = position_template[:]

        for index in range(subset_size):
            bound = size - index
            threshold = thresholds[bound]
            end = offset + 1
            if len(buffer) < end:
                ensure_buffer(end)
                buffer = xof._buffer
            candidate = buffer[offset]
            offset = end
            bytes_drawn += 1
            while candidate >= threshold:
                end = offset + 1
                if len(buffer) < end:
                    ensure_buffer(end)
                    buffer = xof._buffer
                candidate = buffer[offset]
                offset = end
                bytes_drawn += 1
            swap_index = index + (candidate % bound)
            positions[index], positions[swap_index] = positions[swap_index], positions[index]

        xof._offset = offset
        if counting:
            increment("isp.xof_output_bytes", bytes_drawn)
            increment("isp.xof_output_bits", 8 * bytes_drawn)
        result = positions[:subset_size]
        result.sort()
        return result

    position_template = _position_template_base(size)
    positions = position_template[:]
    randbelow = xof.randbelow
    for index in range(subset_size):
        swap_index = index + randbelow(size - index)
        positions[index], positions[swap_index] = positions[swap_index], positions[index]
    result = positions[:subset_size]
    result.sort()
    return result


def _xof_from_partition_value(
    partition_value: PartitionValueInput,
    hash_len: int,
    hash_name: str = DEFAULT_HASH_NAME,
) -> HashXOF:
    seed_material = _xof_seed_material_from_partition_value(
        partition_value=partition_value,
        hash_len=hash_len,
        hash_name=hash_name,
    )
    return _xof_from_seed_material(seed_material, hash_name)


def _xof_seed_material_from_partition_value(
    partition_value: PartitionValueInput,
    hash_len: int,
    hash_name: str = DEFAULT_HASH_NAME,
) -> bytes:
    serialized_y = _serialize_partition_value(partition_value, hash_len)
    hash_name_bytes = hash_name.encode("ascii")
    if counters_enabled():
        increment("isp.sample_seed_hash")
    seed_digest = _hash_bytes(
        b"ValStrictISP/HY/" + hash_name_bytes + b"/" + serialized_y,
        64,
        hash_name,
    )
    seed_material = (
        b"ValStrictISP/SamplePosition/XOF/"
        + hash_name_bytes
        + b"/"
        + seed_digest
    )
    return seed_material


def _xof_from_seed_material(
    seed_material: bytes,
    hash_name: str = DEFAULT_HASH_NAME,
) -> HashXOF:
    return HashXOF(seed_material=seed_material, hash_name=hash_name)


def _binomial_table(universe_size: int) -> tuple[tuple[int, ...], ...]:
    table = _BINOMIAL_TABLES.get(universe_size)
    if table is not None:
        return table

    rows = [(1,)]
    for n in range(1, universe_size + 1):
        previous = rows[-1]
        row = [1] * (n + 1)
        for k in range(1, n):
            row[k] = previous[k - 1] + previous[k]
        rows.append(tuple(row))

    table = tuple(rows)
    _BINOMIAL_TABLES[universe_size] = table
    return table


def _subset_rank_parameters(
    universe_size: int,
) -> tuple[tuple[int, ...], tuple[int, ...], tuple[int, ...]]:
    cached = _SUBSET_RANK_PARAMS.get(universe_size)
    if cached is not None:
        return cached

    binomial_table = _binomial_table(universe_size)
    subset_counts = binomial_table[universe_size]
    byte_lengths = [0] * (universe_size + 1)
    thresholds = [0] * (universe_size + 1)
    for subset_size, bound in enumerate(subset_counts):
        byte_len = max(1, (bound.bit_length() + 7) // 8)
        upper = 1 << (8 * byte_len)
        byte_lengths[subset_size] = byte_len
        thresholds[subset_size] = upper - (upper % bound)

    cached = (subset_counts, tuple(byte_lengths), tuple(thresholds))
    _SUBSET_RANK_PARAMS[universe_size] = cached
    return cached


def _sample_base_parameters(
    universe_size: int,
) -> tuple[
    tuple[tuple[int, ...], ...],
    tuple[int, ...],
    tuple[int, ...],
    tuple[int, ...],
    tuple[tuple[int, ...], ...],
    tuple[object | None, ...],
]:
    cached = _SAMPLE_BASE_PARAMS.get(universe_size)
    if cached is not None:
        return cached

    binomial_table = _binomial_table(universe_size)
    subset_counts, byte_lengths, thresholds = _subset_rank_parameters(universe_size)
    direct_unrank_rows = tuple(
        tuple(
            binomial_table[universe_size - position - 1][remaining - 1]
            if remaining > 0 and remaining - 1 <= universe_size - position - 1
            else 0
            for position in range(universe_size)
        )
        for remaining in range(universe_size + 1)
    )
    unpackers = tuple(_INT_UNPACKERS.get(byte_len) for byte_len in byte_lengths)
    cached = (
        binomial_table,
        subset_counts,
        byte_lengths,
        thresholds,
        direct_unrank_rows,
        unpackers,
    )
    _SAMPLE_BASE_PARAMS[universe_size] = cached
    return cached


def _decode_subset_rank_positions(
    rank: int,
    universe_size: int,
    subset_size: int,
    binomial_table: tuple[tuple[int, ...], ...],
) -> tuple[int, ...]:
    if subset_size == 0:
        return ()

    positions = [0] * subset_size
    n = universe_size
    r = subset_size
    base = 0
    current_rank = rank
    index = 0
    while r > 0:
        total = binomial_table[n][r]
        target = total - current_rank
        low = r
        high = n
        best = n
        while low <= high:
            mid = (low + high) // 2
            if binomial_table[mid][r] >= target:
                best = mid
                high = mid - 1
            else:
                low = mid + 1
        selected_suffix_universe = best
        absolute_position = base + (n - selected_suffix_universe)
        positions[index] = absolute_position
        index += 1
        current_rank = binomial_table[selected_suffix_universe][r] - target
        base = absolute_position + 1
        n = selected_suffix_universe - 1
        r -= 1
    return tuple(positions)


@lru_cache(maxsize=None)
def _subset_decode_tables(
    universe_size: int,
) -> tuple[tuple[tuple[int, ...], ...], ...]:
    if universe_size > _SMALL_SUBSET_DECODE_TABLE_MAX:
        return ()

    binomial_table = _binomial_table(universe_size)
    max_subset_size = min(universe_size, _SMALL_SUBSET_UNRANK_THRESHOLD)
    tables: list[tuple[tuple[int, ...], ...]] = [tuple() for _ in range(universe_size + 1)]
    for subset_size in range(max_subset_size + 1):
        tables[subset_size] = tuple(
            _decode_subset_rank_positions(rank, universe_size, subset_size, binomial_table)
            for rank in range(binomial_table[universe_size][subset_size])
        )
    return tuple(tables)


def _unrank_small_subset_into_groups(
    groups: Groups,
    value: int,
    rank: int,
    universe_size: int,
    subset_size: int,
    binomial_table: tuple[tuple[int, ...], ...],
) -> None:
    n = universe_size
    r = subset_size
    base = 0
    current_rank = rank
    while r > 0:
        total = binomial_table[n][r]
        target = total - current_rank
        low = r
        high = n
        best = n
        while low <= high:
            mid = (low + high) // 2
            if binomial_table[mid][r] >= target:
                best = mid
                high = mid - 1
            else:
                low = mid + 1
        selected_suffix_universe = best
        position = n - selected_suffix_universe
        groups[base + position].append(value)
        current_rank = binomial_table[selected_suffix_universe][r] - target
        base += position + 1
        n = selected_suffix_universe - 1
        r -= 1


def _append_value_excluding_small_subset(
    groups: Groups,
    value: int,
    rank: int,
    universe_size: int,
    included_size: int,
    subset_count: int,
    binomial_table: tuple[tuple[int, ...], ...],
) -> None:
    excluded_size = universe_size - included_size
    excluded_rank = subset_count - 1 - rank
    if excluded_size == 1:
        excluded_positions = (excluded_rank,)
    else:
        n = universe_size
        r = excluded_size
        base = 0
        current_rank = excluded_rank
        decoded = [0] * excluded_size
        index = 0
        while r > 0:
            total = binomial_table[n][r]
            target = total - current_rank
            low = r
            high = n
            best = n
            while low <= high:
                mid = (low + high) // 2
                if binomial_table[mid][r] >= target:
                    best = mid
                    high = mid - 1
                else:
                    low = mid + 1
            selected_suffix_universe = best
            position = base + (n - selected_suffix_universe)
            decoded[index] = position
            index += 1
            current_rank = binomial_table[selected_suffix_universe][r] - target
            base = position + 1
            n = selected_suffix_universe - 1
            r -= 1
        excluded_positions = tuple(decoded)

    group_index = 0
    for excluded_position in excluded_positions:
        while group_index < excluded_position:
            groups[group_index].append(value)
            group_index += 1
        group_index = excluded_position + 1
    while group_index < universe_size:
        groups[group_index].append(value)
        group_index += 1


def _unrank_small_subset_into_group_masks(
    group_masks: list[int],
    group_firsts: list[int],
    group_lasts: list[int],
    value: int,
    value_bit: int,
    rank: int,
    universe_size: int,
    subset_size: int,
    binomial_table: tuple[tuple[int, ...], ...],
) -> None:
    n = universe_size
    r = subset_size
    base = 0
    current_rank = rank
    while r > 0:
        total = binomial_table[n][r]
        target = total - current_rank
        low = r
        high = n
        best = n
        while low <= high:
            mid = (low + high) // 2
            if binomial_table[mid][r] >= target:
                best = mid
                high = mid - 1
            else:
                low = mid + 1
        selected_suffix_universe = best
        position = n - selected_suffix_universe
        absolute_position = base + position
        if group_firsts[absolute_position] < 0:
            group_firsts[absolute_position] = value
        group_lasts[absolute_position] = value
        group_masks[absolute_position] |= value_bit
        current_rank = binomial_table[selected_suffix_universe][r] - target
        base += position + 1
        n = selected_suffix_universe - 1
        r -= 1


def _append_value_excluding_small_subset_group_masks(
    group_masks: list[int],
    group_firsts: list[int],
    group_lasts: list[int],
    value: int,
    value_bit: int,
    rank: int,
    universe_size: int,
    included_size: int,
    subset_count: int,
    binomial_table: tuple[tuple[int, ...], ...],
) -> None:
    excluded_size = universe_size - included_size
    excluded_rank = subset_count - 1 - rank
    if excluded_size == 1:
        excluded_positions = (excluded_rank,)
    else:
        n = universe_size
        r = excluded_size
        base = 0
        current_rank = excluded_rank
        decoded = [0] * excluded_size
        index = 0
        while r > 0:
            total = binomial_table[n][r]
            target = total - current_rank
            low = r
            high = n
            best = n
            while low <= high:
                mid = (low + high) // 2
                if binomial_table[mid][r] >= target:
                    best = mid
                    high = mid - 1
                else:
                    low = mid + 1
            selected_suffix_universe = best
            position = base + (n - selected_suffix_universe)
            decoded[index] = position
            index += 1
            current_rank = binomial_table[selected_suffix_universe][r] - target
            base = position + 1
            n = selected_suffix_universe - 1
            r -= 1
        excluded_positions = tuple(decoded)

    group_index = 0
    for excluded_position in excluded_positions:
        while group_index < excluded_position:
            if group_firsts[group_index] < 0:
                group_firsts[group_index] = value
            group_lasts[group_index] = value
            group_masks[group_index] |= value_bit
            group_index += 1
        group_index = excluded_position + 1
    while group_index < universe_size:
        if group_firsts[group_index] < 0:
            group_firsts[group_index] = value
        group_lasts[group_index] = value
        group_masks[group_index] |= value_bit
        group_index += 1


def _sample_base_fast_packed(
    counts: Sequence[int],
    partition_num: int,
    hash_name: str,
    seed_material: bytes,
    sample_base_params: Optional[
        tuple[
            tuple[tuple[int, ...], ...],
        tuple[int, ...],
        tuple[int, ...],
        tuple[int, ...],
        tuple[tuple[int, ...], ...],
        tuple[object | None, ...],
    ]
    ] = None,
    ) -> tuple[list[int], list[int], list[int]]:
    if hash_name == "shake_128":
        fast_digest = hashlib.shake_128(seed_material).digest
    else:
        # SHA3 variants still expand with SHAKE256 on the sampler side.
        fast_digest = hashlib.shake_256(seed_material).digest

    group_masks = [0] * partition_num
    group_firsts = [-1] * partition_num
    group_lasts = [-1] * partition_num
    if sample_base_params is None:
        sample_base_params = _sample_base_parameters(partition_num)
    binomial_table, subset_count_row, byte_lengths, thresholds, direct_unrank_rows, unpackers = (
        sample_base_params
    )
    small_unrank = _unrank_small_subset_into_group_masks
    complement_unrank = _append_value_excluding_small_subset_group_masks
    group_masks_local = group_masks
    group_firsts_local = group_firsts
    group_lasts_local = group_lasts
    partition_num_local = partition_num
    subset_decode_tables = _subset_decode_tables(partition_num_local)
    buffer = b""
    buffer_len = 0
    offset = 0

    for value, count in enumerate(counts):
        if count == 0:
            continue
        if count > partition_num_local:
            raise ValueError(
                f"cannot place value {value}: multiplicity {count} exceeds partition_num={partition_num_local}"
            )

        subset_count = subset_count_row[count]
        byte_len = byte_lengths[count]
        threshold = thresholds[count]
        unpack = unpackers[count]
        while True:
            end = offset + byte_len
            if buffer_len < end:
                if buffer_len < 32:
                    target = 32 if end < 32 else end
                else:
                    doubled = 2 * buffer_len
                    target = doubled if end < doubled else end
                buffer = fast_digest(target)
                buffer_len = target
            if byte_len == 1:
                candidate = buffer[offset]
            elif unpack is not None:
                candidate = unpack(buffer, offset)[0]
            else:
                candidate = int.from_bytes(buffer[offset:end], "big")
            offset = end
            if candidate < threshold:
                rank = candidate % subset_count
                break

        value_bit = 1 << value
        if subset_decode_tables and count <= _SMALL_SUBSET_UNRANK_THRESHOLD:
            for position in subset_decode_tables[count][rank]:
                if group_firsts_local[position] < 0:
                    group_firsts_local[position] = value
                group_lasts_local[position] = value
                group_masks_local[position] |= value_bit
            continue
        if count == 1:
            if group_firsts_local[rank] < 0:
                group_firsts_local[rank] = value
            group_lasts_local[rank] = value
            group_masks_local[rank] |= value_bit
            continue
        if count <= _SMALL_SUBSET_UNRANK_THRESHOLD:
            small_unrank(
                group_masks_local,
                group_firsts_local,
                group_lasts_local,
                value,
                value_bit,
                rank,
                partition_num_local,
                count,
                binomial_table,
            )
            continue
        if partition_num_local - count <= _SMALL_SUBSET_UNRANK_THRESHOLD:
            complement_unrank(
                group_masks_local,
                group_firsts_local,
                group_lasts_local,
                value,
                value_bit,
                rank,
                partition_num_local,
                count,
                subset_count,
                binomial_table,
            )
            continue

        remaining = count
        current_rank = rank
        include_row = direct_unrank_rows[remaining]
        for position in range(partition_num_local):
            if remaining == 0:
                break
            include_count = include_row[position]
            if current_rank < include_count:
                if group_firsts_local[position] < 0:
                    group_firsts_local[position] = value
                group_lasts_local[position] = value
                group_masks_local[position] |= value_bit
                remaining -= 1
                if remaining:
                    include_row = direct_unrank_rows[remaining]
            else:
                current_rank -= include_count

    return group_masks_local, group_firsts_local, group_lasts_local


def _sample_base_fast_group_masks(
    counts: Sequence[int],
    partition_num: int,
    hash_name: str,
    seed_material: bytes,
    sample_base_params: Optional[
        tuple[
            tuple[tuple[int, ...], ...],
            tuple[int, ...],
            tuple[int, ...],
            tuple[int, ...],
            tuple[tuple[int, ...], ...],
            tuple[object | None, ...],
        ]
    ] = None,
) -> GroupMasks:
    group_masks, _, _ = _sample_base_fast_packed(
        counts=counts,
        partition_num=partition_num,
        hash_name=hash_name,
        seed_material=seed_material,
        sample_base_params=sample_base_params,
    )
    return _compress_group_masks(group_masks, len(counts))


def _sample_base_fast_group_masks_from_items(
    active_counts: Sequence[tuple[int, int]],
    max_g_value: int,
    partition_num: int,
    hash_name: str,
    seed_material: bytes,
    sample_base_params: Optional[
        tuple[
            tuple[tuple[int, ...], ...],
            tuple[int, ...],
            tuple[int, ...],
            tuple[int, ...],
            tuple[tuple[int, ...], ...],
            tuple[object | None, ...],
        ]
    ] = None,
) -> GroupMasks:
    if hash_name == "shake_128":
        fast_digest = hashlib.shake_128(seed_material).digest
    else:
        fast_digest = hashlib.shake_256(seed_material).digest

    group_masks = [0] * partition_num
    if sample_base_params is None:
        sample_base_params = _sample_base_parameters(partition_num)
    binomial_table, subset_count_row, byte_lengths, thresholds, direct_unrank_rows, unpackers = (
        sample_base_params
    )
    subset_decode_tables = _subset_decode_tables(partition_num)
    buffer = b""
    buffer_len = 0
    offset = 0

    for value, count in active_counts:
        if count > partition_num:
            raise ValueError(
                f"cannot place value {value}: multiplicity {count} exceeds partition_num={partition_num}"
            )

        subset_count = subset_count_row[count]
        byte_len = byte_lengths[count]
        threshold = thresholds[count]
        unpack = unpackers[count]
        while True:
            end = offset + byte_len
            if buffer_len < end:
                if buffer_len < 32:
                    target = 32 if end < 32 else end
                else:
                    doubled = 2 * buffer_len
                    target = doubled if end < doubled else end
                buffer = fast_digest(target)
                buffer_len = target
            if byte_len == 1:
                candidate = buffer[offset]
            elif unpack is not None:
                candidate = unpack(buffer, offset)[0]
            else:
                candidate = int.from_bytes(buffer[offset:end], "big")
            offset = end
            if candidate < threshold:
                rank = candidate % subset_count
                break

        value_bit = 1 << value
        if subset_decode_tables and count <= _SMALL_SUBSET_UNRANK_THRESHOLD:
            for position in subset_decode_tables[count][rank]:
                group_masks[position] |= value_bit
            continue
        if count == 1:
            group_masks[rank] |= value_bit
            continue
        if count <= _SMALL_SUBSET_UNRANK_THRESHOLD:
            _unrank_small_subset_into_group_masks(
                group_masks,
                [-1] * partition_num,
                [-1] * partition_num,
                value,
                value_bit,
                rank,
                partition_num,
                count,
                binomial_table,
            )
            continue

        remaining = count
        current_rank = rank
        include_row = direct_unrank_rows[remaining]
        for position in range(partition_num):
            if remaining == 0:
                break
            include_count = include_row[position]
            if current_rank < include_count:
                group_masks[position] |= value_bit
                remaining -= 1
                if remaining:
                    include_row = direct_unrank_rows[remaining]
            else:
                current_rank -= include_count

    return _compress_group_masks(group_masks, max_g_value)


def _sample_base_fast(
    counts: Sequence[int],
    partition_num: int,
    hash_name: str,
    seed_material: bytes,
    sample_base_params: Optional[
        tuple[
            tuple[tuple[int, ...], ...],
            tuple[int, ...],
            tuple[int, ...],
            tuple[int, ...],
            tuple[tuple[int, ...], ...],
            tuple[object | None, ...],
        ]
    ] = None,
) -> Groups:
    if hash_name == "shake_128":
        fast_digest = hashlib.shake_128(seed_material).digest
    else:
        # SHA3 variants still expand with SHAKE256 on the sampler side.
        fast_digest = hashlib.shake_256(seed_material).digest

    groups: Groups = [[] for _ in range(partition_num)]
    if sample_base_params is None:
        sample_base_params = _sample_base_parameters(partition_num)
    binomial_table, subset_count_row, byte_lengths, thresholds, direct_unrank_rows, unpackers = (
        sample_base_params
    )
    small_unrank = _unrank_small_subset_into_groups
    complement_unrank = _append_value_excluding_small_subset
    groups_local = groups
    partition_num_local = partition_num
    subset_decode_tables = _subset_decode_tables(partition_num_local)
    buffer = b""
    buffer_len = 0
    offset = 0

    for value, count in enumerate(counts):
        if count == 0:
            continue
        if count > partition_num_local:
            raise ValueError(
                f"cannot place value {value}: multiplicity {count} exceeds partition_num={partition_num_local}"
            )

        subset_count = subset_count_row[count]
        byte_len = byte_lengths[count]
        threshold = thresholds[count]
        unpack = unpackers[count]
        while True:
            end = offset + byte_len
            if buffer_len < end:
                if buffer_len < 32:
                    target = 32 if end < 32 else end
                else:
                    doubled = 2 * buffer_len
                    target = doubled if end < doubled else end
                buffer = fast_digest(target)
                buffer_len = target
            if byte_len == 1:
                candidate = buffer[offset]
            elif unpack is not None:
                candidate = unpack(buffer, offset)[0]
            else:
                candidate = int.from_bytes(buffer[offset:end], "big")
            offset = end
            if candidate < threshold:
                rank = candidate % subset_count
                break

        if subset_decode_tables and count <= _SMALL_SUBSET_UNRANK_THRESHOLD:
            for position in subset_decode_tables[count][rank]:
                groups_local[position].append(value)
            continue
        if count == 1:
            groups_local[rank].append(value)
            continue
        if count <= _SMALL_SUBSET_UNRANK_THRESHOLD:
            small_unrank(
                groups_local,
                value,
                rank,
                partition_num_local,
                count,
                binomial_table,
            )
            continue
        if partition_num_local - count <= _SMALL_SUBSET_UNRANK_THRESHOLD:
            complement_unrank(
                groups_local,
                value,
                rank,
                partition_num_local,
                count,
                subset_count,
                binomial_table,
            )
            continue

        remaining = count
        current_rank = rank
        include_row = direct_unrank_rows[remaining]
        for position in range(partition_num_local):
            if remaining == 0:
                break
            include_count = include_row[position]
            if current_rank < include_count:
                groups_local[position].append(value)
                remaining -= 1
                if remaining:
                    include_row = direct_unrank_rows[remaining]
            else:
                current_rank -= include_count

    return groups_local


def _sample_base_fast_from_items(
    active_counts: Sequence[tuple[int, int]],
    partition_num: int,
    hash_name: str,
    seed_material: bytes,
    sample_base_params: Optional[
        tuple[
            tuple[tuple[int, ...], ...],
            tuple[int, ...],
            tuple[int, ...],
            tuple[int, ...],
            tuple[tuple[int, ...], ...],
            tuple[object | None, ...],
        ]
    ] = None,
) -> Groups:
    if hash_name == "shake_128":
        fast_digest = hashlib.shake_128(seed_material).digest
    else:
        fast_digest = hashlib.shake_256(seed_material).digest

    groups: Groups = [[] for _ in range(partition_num)]
    if sample_base_params is None:
        sample_base_params = _sample_base_parameters(partition_num)
    binomial_table, subset_count_row, byte_lengths, thresholds, direct_unrank_rows, unpackers = (
        sample_base_params
    )
    subset_decode_tables = _subset_decode_tables(partition_num)
    buffer = b""
    buffer_len = 0
    offset = 0

    for value, count in active_counts:
        if count > partition_num:
            raise ValueError(
                f"cannot place value {value}: multiplicity {count} exceeds partition_num={partition_num}"
            )

        subset_count = subset_count_row[count]
        byte_len = byte_lengths[count]
        threshold = thresholds[count]
        unpack = unpackers[count]
        while True:
            end = offset + byte_len
            if buffer_len < end:
                if buffer_len < 32:
                    target = 32 if end < 32 else end
                else:
                    doubled = 2 * buffer_len
                    target = doubled if end < doubled else end
                buffer = fast_digest(target)
                buffer_len = target
            if byte_len == 1:
                candidate = buffer[offset]
            elif unpack is not None:
                candidate = unpack(buffer, offset)[0]
            else:
                candidate = int.from_bytes(buffer[offset:end], "big")
            offset = end
            if candidate < threshold:
                rank = candidate % subset_count
                break

        if subset_decode_tables and count <= _SMALL_SUBSET_UNRANK_THRESHOLD:
            for position in subset_decode_tables[count][rank]:
                groups[position].append(value)
            continue
        if count == 1:
            groups[rank].append(value)
            continue
        if count <= _SMALL_SUBSET_UNRANK_THRESHOLD:
            _unrank_small_subset_into_groups(
                groups,
                value,
                rank,
                partition_num,
                count,
                binomial_table,
            )
            continue

        remaining = count
        current_rank = rank
        include_row = direct_unrank_rows[remaining]
        for position in range(partition_num):
            if remaining == 0:
                break
            include_count = include_row[position]
            if current_rank < include_count:
                groups[position].append(value)
                remaining -= 1
                if remaining:
                    include_row = direct_unrank_rows[remaining]
            else:
                current_rank -= include_count

    return groups


def _sample_groups_from_seed_material_small(
    counts: Sequence[int],
    partition_num: int,
    seed_material: bytes,
    hash_name: str,
) -> Groups:
    counting = counters_enabled()
    if hash_name == "shake_128":
        shake = hashlib.shake_128(seed_material)
        if counting:
            increment("hash.backend_calls")
            increment("hash.backend_calls.shake_128")
    else:
        shake = hashlib.shake_256(seed_material)
        if counting:
            increment("hash.backend_calls")
            increment("hash.backend_calls.shake_256")
    if counting:
        increment("isp.xof_instances")

    groups: Groups = [[] for _ in range(partition_num)]
    thresholds = _RAND_BELOW_U8_THRESHOLDS
    digest_target = max(32, partition_num * 4)
    buffer = shake.digest(digest_target)
    buffer_len = digest_target
    offset = 0
    bytes_drawn = 0
    partition_threshold = thresholds[partition_num]
    position_template = _position_template_base(partition_num)

    for value, count in enumerate(counts):
        if count == 0:
            continue
        if count > partition_num:
            raise ValueError(
                f"cannot place value {value}: multiplicity {count} exceeds partition_num={partition_num}"
            )

        if count == 1:
            if offset >= buffer_len:
                digest_target *= 2
                buffer = shake.digest(digest_target)
                buffer_len = digest_target
            candidate = buffer[offset]
            offset += 1
            bytes_drawn += 1
            while candidate >= partition_threshold:
                if offset >= buffer_len:
                    digest_target *= 2
                    buffer = shake.digest(digest_target)
                    buffer_len = digest_target
                candidate = buffer[offset]
                offset += 1
                bytes_drawn += 1
            groups[candidate % partition_num].append(value)
            continue

        positions = position_template[:]
        for index in range(count):
            bound = partition_num - index
            threshold = thresholds[bound]
            if offset >= buffer_len:
                digest_target *= 2
                buffer = shake.digest(digest_target)
                buffer_len = digest_target
            candidate = buffer[offset]
            offset += 1
            bytes_drawn += 1
            while candidate >= threshold:
                if offset >= buffer_len:
                    digest_target *= 2
                    buffer = shake.digest(digest_target)
                    buffer_len = digest_target
                candidate = buffer[offset]
                offset += 1
                bytes_drawn += 1
            swap_index = index + (candidate % bound)
            positions[index], positions[swap_index] = positions[swap_index], positions[index]

        for index in range(count):
            groups[positions[index]].append(value)

    if counting:
        increment("isp.xof_output_bytes", bytes_drawn)
        increment("isp.xof_output_bits", 8 * bytes_drawn)
    return groups


def sample_base(
    partition_value: PartitionValueInput,
    block_values: Optional[Sequence[int]],
    partition_num: int,
    max_g_value: int,
    hash_len: int,
    hash_name: str = DEFAULT_HASH_NAME,
    rng: Optional[Random] = None,
    xof_seed_material: Optional[bytes] = None,
    counts: Optional[Sequence[int]] = None,
) -> Groups:
    """
    Sample a legal final state conditioned on the multiplicity profile.

    The implementation uses one global deterministic bit stream derived from
    PartitionValue, then consumes one unbiased rank for each active value and
    decodes that rank into the corresponding subset of positions.
    """

    if counts is None:
        if block_values is None:
            counts = _multiplicity_profile_from_partition_value(
                partition_value=partition_value,
                hash_len=hash_len,
                max_g_bit=(max_g_value.bit_length() - 1),
                max_g_value=max_g_value,
            )
        else:
            counts = multiplicity_profile(block_values, max_g_value)
    counting = counters_enabled()
    (
        binomial_table,
        subset_count_row,
        byte_lengths,
        thresholds,
        direct_unrank_rows,
        unpackers,
    ) = _sample_base_parameters(partition_num)
    subset_decode_tables = _subset_decode_tables(partition_num)
    if rng is None and not counting:
        seed_material = xof_seed_material
        if seed_material is None:
            seed_material = _xof_seed_material_from_partition_value(
                partition_value=partition_value,
                hash_len=hash_len,
                hash_name=hash_name,
            )
        return _sample_base_fast(
            counts=counts,
            partition_num=partition_num,
            hash_name=hash_name,
            seed_material=seed_material,
            sample_base_params=(
                binomial_table,
                subset_count_row,
                byte_lengths,
                thresholds,
                direct_unrank_rows,
                unpackers,
            ),
        )

    if rng is None:
        if xof_seed_material is None:
            xof = _xof_from_partition_value(
                partition_value=partition_value,
                hash_len=hash_len,
                hash_name=hash_name,
            )
        else:
            xof = _xof_from_seed_material(
                seed_material=xof_seed_material,
                hash_name=hash_name,
            )
    else:
        xof = None

    groups: Groups = [[] for _ in range(partition_num)]

    for value, count in enumerate(counts):
        if count == 0:
            continue
        if count > partition_num:
            raise ValueError(
                f"cannot place value {value}: multiplicity {count} exceeds partition_num={partition_num}"
            )

        subset_count = subset_count_row[count]
        if rng is not None:
            rank = rng.randrange(subset_count)
        else:
            if xof is None:
                raise ValueError("xof must be available when rng is not provided")
            rank = xof.randbelow(subset_count)

        if subset_decode_tables and count <= _SMALL_SUBSET_UNRANK_THRESHOLD:
            for position in subset_decode_tables[count][rank]:
                groups[position].append(value)
            continue
        if count == 1:
            groups[rank].append(value)
            continue
        if count <= _SMALL_SUBSET_UNRANK_THRESHOLD:
            _unrank_small_subset_into_groups(
                groups,
                value,
                rank,
                partition_num,
                count,
                binomial_table,
            )
            continue
        if partition_num - count <= _SMALL_SUBSET_UNRANK_THRESHOLD:
            _append_value_excluding_small_subset(
                groups,
                value,
                rank,
                partition_num,
                count,
                subset_count,
                binomial_table,
            )
            continue

        remaining = count
        current_rank = rank
        include_row = direct_unrank_rows[remaining]
        for position in range(partition_num):
            if remaining == 0:
                break
            include_count = include_row[position]
            if current_rank < include_count:
                groups[position].append(value)
                remaining -= 1
                if remaining:
                    include_row = direct_unrank_rows[remaining]
            else:
                current_rank -= include_count

    return groups


sample_position = sample_base


def _val_strict_isp_with_seed_prefix(
    partition_value: bytes,
    params: ISPParameters,
    xof_seed_prefix: bytes,
    return_group_masks: bool = False,
) -> Optional[GroupsOrMasks]:
    hash_len = params.hash_len
    max_g_bit = params.max_g_bit
    partition_num = params.partition_num

    native_w4_prefixed = _NATIVE_W4_BYTES_PREFIXED_SEED
    if (
        not counters_enabled()
        and params._window_valid
        and native_w4_prefixed is not None
        and max_g_bit == 2
        and params._max_g_value == 4
        and partition_num <= _NATIVE_W4_MAX_PARTITION_NUM
    ):
        return native_w4_prefixed(
            partition_value,
            hash_len,
            partition_num,
            params.window_low,
            params.window_high,
            xof_seed_prefix,
            params.hash_name == "shake_128",
            return_group_masks,
        )

    native_prefixed = _NATIVE_BYTES_PREFIXED_SEED
    if (
        not counters_enabled()
        and native_prefixed is not None
        and _native_bytes_supported(params)
    ):
        return native_prefixed(
            partition_value,
            hash_len,
            max_g_bit,
            partition_num,
            params.window_low,
            params.window_high,
            xof_seed_prefix,
            params.hash_name == "shake_128",
            return_group_masks,
        )

    return val_strict_isp(
        partition_value,
        params,
        xof_seed_material=xof_seed_prefix + partition_value,
        return_group_masks=return_group_masks,
    )


def _val_strict_isp_with_random_bytes(
    partition_value: bytes,
    params: ISPParameters,
    random_bytes: bytes,
    *,
    fallback_seed_material: Optional[bytes] = None,
    return_group_masks: bool = False,
) -> Optional[GroupsOrMasks]:
    hash_len = params.hash_len
    max_g_bit = params.max_g_bit
    partition_num = params.partition_num
    max_g_value = params._max_g_value
    window_low = params._window_low
    window_high = params._window_high
    native_random = _NATIVE_BYTES_RANDOM_FAST or _NATIVE_BYTES_RANDOM
    native_w4_random = _NATIVE_W4_BYTES_RANDOM_FAST or _NATIVE_W4_BYTES_RANDOM

    if (
        params._window_valid
        and native_w4_random is not None
        and max_g_bit == 2
        and max_g_value == 4
        and partition_num <= _NATIVE_W4_MAX_PARTITION_NUM
    ):
        if fallback_seed_material is None:
            return native_w4_random(
                partition_value,
                hash_len,
                partition_num,
                window_low,
                window_high,
                random_bytes,
                return_group_masks,
            )
        try:
            return native_w4_random(
                partition_value,
                hash_len,
                partition_num,
                window_low,
                window_high,
                random_bytes,
                return_group_masks,
            )
        except ValueError as exc:
            if fallback_seed_material is None or "insufficient random bytes" not in str(exc):
                raise

    if (
        native_random is not None
        and _native_bytes_supported(params)
    ):
        if fallback_seed_material is None:
            return native_random(
                partition_value,
                hash_len,
                max_g_bit,
                partition_num,
                window_low,
                window_high,
                random_bytes,
                return_group_masks,
            )
        try:
            return native_random(
                partition_value,
                hash_len,
                max_g_bit,
                partition_num,
                window_low,
                window_high,
                random_bytes,
                return_group_masks,
            )
        except ValueError as exc:
            if fallback_seed_material is None or "insufficient random bytes" not in str(exc):
                raise

    if fallback_seed_material is None:
        raise ValueError("insufficient random bytes")
    return val_strict_isp(
        partition_value,
        params,
        xof_seed_material=fallback_seed_material,
        return_group_masks=return_group_masks,
    )


def val_strict_isp(
    partition_value: PartitionValueInput,
    params: ISPParameters,
    rng: Optional[Random] = None,
    xof_seed_material: Optional[bytes] = None,
    return_group_masks: bool = False,
) -> Optional[GroupsOrMasks]:
    """
    Python implementation of the ValStrictISP algorithm.

    Returns:
        A list of partition_num strictly increasing subsequences, or None if
        the multiplicity profile is rejected by the feasibility window.
    """

    hash_len = params.hash_len
    max_g_bit = params.max_g_bit
    max_g_value = params._max_g_value
    partition_num = params.partition_num
    hash_name = params.hash_name
    native_bytes = _NATIVE_BYTES
    native_default_seed = _NATIVE_BYTES_DEFAULT_SEED
    native_w4_default_seed = _NATIVE_W4_BYTES_DEFAULT_SEED
    native_w4 = _NATIVE_W4_BYTES
    fast_sampling = rng is None and not counters_enabled()

    if (
        fast_sampling
        and xof_seed_material is None
        and native_w4_default_seed is not None
        and isinstance(partition_value, bytes)
        and hash_name in {"shake_128", "shake_256"}
        and params._window_valid
        and max_g_bit == 2
        and max_g_value == 4
        and partition_num <= _NATIVE_W4_MAX_PARTITION_NUM
    ):
        return native_w4_default_seed(
            partition_value,
            hash_len,
            partition_num,
            params.window_low,
            params.window_high,
            hash_name == "shake_128",
            return_group_masks,
        )

    if (
        fast_sampling
        and xof_seed_material is None
        and native_default_seed is not None
        and isinstance(partition_value, bytes)
        and hash_name in {"shake_128", "shake_256"}
        and _native_bytes_supported(params)
    ):
        return native_default_seed(
            partition_value,
            hash_len,
            max_g_bit,
            partition_num,
            params.window_low,
            params.window_high,
            hash_name == "shake_128",
            return_group_masks,
        )

    if (
        fast_sampling
        and xof_seed_material is not None
        and params._window_valid
        and native_w4 is not None
        and isinstance(partition_value, bytes)
        and max_g_bit == 2
        and max_g_value == 4
        and partition_num <= _NATIVE_W4_MAX_PARTITION_NUM
    ):
        return native_w4(
            partition_value,
            hash_len,
            partition_num,
            params.window_low,
            params.window_high,
            xof_seed_material,
            hash_name == "shake_128",
            return_group_masks,
        )

    if (
        fast_sampling
        and xof_seed_material is not None
        and isinstance(partition_value, bytes)
        and native_bytes is not None
        and _native_bytes_supported(params)
    ):
        return native_bytes(
            partition_value,
            hash_len,
            max_g_bit,
            partition_num,
            params.window_low,
            params.window_high,
            xof_seed_material,
            hash_name == "shake_128",
            return_group_masks,
        )

    if fast_sampling and _sparse_profile_worthwhile(params):
        if not params._window_valid:
            return None

        active_counts = _sparse_multiplicity_profile_from_partition_value(
            partition_value=partition_value,
            hash_len=hash_len,
            max_g_bit=max_g_bit,
        )
        low = params._window_low
        high = params._window_high
        if low > 0 and len(active_counts) < max_g_value:
            return None
        for _, count in active_counts:
            if count < low or count > high:
                return None

        seed_material = xof_seed_material
        if seed_material is None:
            seed_material = _xof_seed_material_from_partition_value(
                partition_value=partition_value,
                hash_len=hash_len,
                hash_name=hash_name,
            )
        if return_group_masks:
            return _sample_base_fast_group_masks_from_items(
                active_counts=active_counts,
                max_g_value=max_g_value,
                partition_num=partition_num,
                hash_name=hash_name,
                seed_material=seed_material,
                sample_base_params=params.sample_base_params,
            )
        return _sample_base_fast_from_items(
            active_counts=active_counts,
            partition_num=partition_num,
            hash_name=hash_name,
            seed_material=seed_material,
            sample_base_params=params.sample_base_params,
        )

    counts = _multiplicity_profile_from_partition_value(
        partition_value=partition_value,
        hash_len=hash_len,
        max_g_bit=max_g_bit,
        max_g_value=max_g_value,
    )

    if not params._window_valid:
        return None

    low = params._window_low
    high = params._window_high
    for count in counts:
        if count < low or count > high:
            return None

    if fast_sampling:
        seed_material = xof_seed_material
        if seed_material is None:
            seed_material = _xof_seed_material_from_partition_value(
                partition_value=partition_value,
                hash_len=hash_len,
                hash_name=hash_name,
            )
        if (
            native_w4 is not None
            and isinstance(partition_value, bytes)
            and max_g_bit == 2
            and max_g_value == 4
            and partition_num <= _NATIVE_W4_MAX_PARTITION_NUM
        ):
            return native_w4(
                partition_value,
                hash_len,
                partition_num,
                params.window_low,
                params.window_high,
                seed_material,
                hash_name == "shake_128",
                return_group_masks,
            )
        if (
            native_bytes is not None
            and isinstance(partition_value, bytes)
            and _native_bytes_supported(params)
        ):
            return native_bytes(
                partition_value,
                hash_len,
                max_g_bit,
                partition_num,
                params.window_low,
                params.window_high,
                seed_material,
                hash_name == "shake_128",
                return_group_masks,
            )
        if return_group_masks:
            return _sample_base_fast_group_masks(
                counts=counts,
                partition_num=partition_num,
                hash_name=hash_name,
                seed_material=seed_material,
                sample_base_params=params.sample_base_params,
            )
        return _sample_base_fast(
            counts=counts,
            partition_num=partition_num,
            hash_name=hash_name,
            seed_material=seed_material,
            sample_base_params=params.sample_base_params,
        )

    groups = sample_base(
        partition_value=partition_value,
        block_values=None,
        partition_num=partition_num,
        max_g_value=max_g_value,
        hash_len=hash_len,
        hash_name=hash_name,
        rng=rng,
        xof_seed_material=xof_seed_material,
        counts=counts,
    )
    if return_group_masks:
        return _group_masks_from_groups(groups, max_g_value)
    return groups


def is_strictly_increasing(values: Sequence[int]) -> bool:
    return all(left < right for left, right in zip(values, values[1:]))


def verify_output(groups: Sequence[Sequence[int]]) -> bool:
    return all(is_strictly_increasing(group) for group in groups)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the ValStrictISP sampler.")
    parser.add_argument(
        "input_value",
        help="partition value (bitstring/int/hex) or message, depending on --input-mode",
    )
    parser.add_argument("--hash-len", type=int, required=True, help="HashLen")
    parser.add_argument("--max-g-bit", type=int, required=True, help="MaxGBit")
    parser.add_argument("--partition-num", type=int, required=True, help="PartitionNum")
    parser.add_argument(
        "--window-radius",
        type=int,
        required=True,
        help="Symmetric multiplicity window radius",
    )
    parser.add_argument(
        "--link-threshold",
        type=int,
        default=-1,
        help="Deprecated compatibility option; ignored by the simplified-windowed VISP",
    )
    parser.add_argument(
        "--hash-name",
        choices=sorted(SUPPORTED_HASHES),
        default=DEFAULT_HASH_NAME,
        help="Hash/XOF used for message hashing and H(Y)-based sampling",
    )
    parser.add_argument(
        "--input-mode",
        choices=("partition", "message"),
        default="partition",
        help="Interpret the positional input as a partition value Y or as a message to hash",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Optional override for testing; if omitted, sampling randomness comes from H(Y)",
    )
    return parser


def _parse_cli_partition_value(raw: str, hash_len: int) -> PartitionValueInput:
    if raw.startswith("0x"):
        integer_value = int(raw, 16)
        if integer_value >= (1 << hash_len):
            raise ValueError("hex partition value is out of range for the given hash_len")
        return integer_value

    if raw.startswith("0b"):
        bitstring = raw[2:]
        if len(bitstring) != hash_len:
            raise ValueError(
                f"binary literal length mismatch: expected {hash_len}, got {len(bitstring)}"
            )
        return bitstring

    if set(raw) <= {"0", "1"}:
        return raw

    return int(raw, 10)


def _main() -> int:
    args = _build_parser().parse_args()
    params = ISPParameters(
        hash_len=args.hash_len,
        max_g_bit=args.max_g_bit,
        partition_num=args.partition_num,
        window_radius=args.window_radius,
        link_threshold=args.link_threshold,
        hash_name=args.hash_name,
    )
    rng = Random(args.seed) if args.seed is not None else None
    if args.input_mode == "message":
        partition_value = hash_message_to_partition_value(
            message=args.input_value,
            hash_len=params.hash_len,
            hash_name=params.hash_name,
        )
    else:
        partition_value = _parse_cli_partition_value(args.input_value, params.hash_len)

    block_values = blk(partition_value, params.hash_len, params.max_g_bit)
    counts = multiplicity_profile(block_values, params.max_g_value)
    low, high = window_bounds(params)
    groups = val_strict_isp(partition_value, params, rng=rng)

    output = {
        "hash_name": params.hash_name,
        "partition_value": normalize_partition_value(partition_value, params.hash_len),
        "accepted": groups is not None,
        "block_values": block_values,
        "multiplicity_profile": counts,
        "window": {
            "radius": params.window_radius,
            "low": low,
            "high": high,
        },
        "groups": groups,
        "strictly_increasing": verify_output(groups) if groups is not None else False,
    }
    print(json.dumps(output, ensure_ascii=True, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(_main())
