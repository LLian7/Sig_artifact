from __future__ import annotations

import argparse
from functools import lru_cache
import hashlib
import json
import math
import struct
from dataclasses import dataclass, field, replace
from random import Random
from typing import Any, Mapping, List, Optional, Sequence, Union

from operation_counter import enabled as counters_enabled, increment

PartitionValueInput = Union[str, bytes, int]
Groups = List[List[int]]
MessageInput = Union[str, bytes]


@dataclass(frozen=True)
class BTTemplate:
    counts: tuple[int, ...]
    realizations: tuple[tuple[tuple[int, ...], ...], ...]

SUPPORTED_HASHES = {"shake_128", "shake_256", "sha3_256", "sha3_512"}
DEFAULT_HASH_NAME = "shake_256"
TREEAWARE_ISP_DOMAIN = b"TreeAwareISP"
_PARTITION_VALUE_DOMAIN = TREEAWARE_ISP_DOMAIN + b"/PartitionValue/"
_HY_DOMAIN_PREFIX = TREEAWARE_ISP_DOMAIN + b"/HY/"
_SAMPLE_POSITION_XOF_PREFIX = TREEAWARE_ISP_DOMAIN + b"/SamplePosition/XOF/"
TREEAWARE_ISP_KEYEDH_XOF_PREFIX = TREEAWARE_ISP_DOMAIN + b"/SamplePosition/KeyedH/"
_TREE_SAMPLER_DOMAIN = TREEAWARE_ISP_DOMAIN + b"/TreeSampler/"
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
_TREE_COUNT_CACHE: dict[tuple[object, ...], int] = {}
_EXACT_ROUTE_PLAN_CACHE: dict[tuple[object, ...], "_ExactRoutePlan"] = {}
_PROFILE_RULE_CACHE: dict[tuple[object, ...], Optional[tuple[tuple[int, ...], ...]]] = {}
_PROFILE_ROUTE_PLAN_CACHE: dict[tuple[object, ...], Optional["_ProfileRoutePlan"]] = {}
_PROFILE_ROUTE_PLAN_SUCCESS_CACHE: dict[tuple[object, ...], "_ProfileRoutePlan"] = {}
_PROFILE_ROUTE_PLAN_FAIL_FLOOR_CACHE: dict[tuple[object, ...], int] = {}
_DEFAULT_LAMINAR_PATTERN_MASKS: dict[int, tuple[int, ...]] = {}
_LAMINAR_NONEMPTY_ROWS_CACHE: dict[tuple[int, tuple[int, ...]], tuple[int, ...]] = {}
_LAMINAR_NONEMPTY_ITEMS_CACHE: dict[
    tuple[int, tuple[int, ...]],
    tuple[tuple[int, int], ...],
] = {}
_LAMINAR_WITH_EMPTY_COUNT_CACHE: dict[tuple[int, tuple[int, ...]], int] = {}
_W2_LOW_BIT_MASKS: dict[int, int] = {}
_INT_UNPACKERS = {
    2: struct.Struct(">H").unpack_from,
    4: struct.Struct(">I").unpack_from,
    8: struct.Struct(">Q").unpack_from,
}
_SMALL_SUBSET_UNRANK_THRESHOLD = 4
_SMALL_SUBSET_DECODE_TABLE_MAX = 32
_SHAPE_STAT_KEYS = (
    "empty_rows",
    "singleton_rows",
    "pair_rows",
    "half_rows",
    "large_rows",
    "nonempty_rows",
    "full_rows",
)
_SHAPE_LIMIT_ALIASES = {
    "max_empty": "max_empty_rows",
    "max_empties": "max_empty_rows",
    "max_empty_rows": "max_empty_rows",
    "max_singleton": "max_singleton_rows",
    "max_singletons": "max_singleton_rows",
    "max_singleton_rows": "max_singleton_rows",
    "max_pair": "max_pair_rows",
    "max_pairs": "max_pair_rows",
    "max_pair_rows": "max_pair_rows",
    "max_half": "max_half_rows",
    "max_halves": "max_half_rows",
    "max_half_rows": "max_half_rows",
    "max_large": "max_large_rows",
    "max_large_rows": "max_large_rows",
    "max_nonempty": "max_nonempty_rows",
    "max_nonempty_rows": "max_nonempty_rows",
    "max_full": "max_full_rows",
    "max_full_rows": "max_full_rows",
}
_SHAPE_LIMIT_INDEX = {
    "max_empty_rows": 0,
    "max_singleton_rows": 1,
    "max_pair_rows": 2,
    "max_half_rows": 3,
    "max_large_rows": 4,
    "max_nonempty_rows": 5,
    "max_full_rows": 6,
}
_ROW_CLASS_BOTTOM = -1
_ROW_CLASS_EMPTY = 0
_ROW_CLASS_FULL = 1
_ROW_CLASS_HALF_L = 2
_ROW_CLASS_HALF_R = 3
_ROW_CLASS_PAIR = 4
_ROW_CLASS_SINGLE = 5
_ROW_CLASS_OTHER = 6
_ROW_CLASS_NAMES = {
    _ROW_CLASS_BOTTOM: "Bottom",
    _ROW_CLASS_EMPTY: "Empty",
    _ROW_CLASS_FULL: "Full",
    _ROW_CLASS_HALF_L: "HalfL",
    _ROW_CLASS_HALF_R: "HalfR",
    _ROW_CLASS_PAIR: "Pair",
    _ROW_CLASS_SINGLE: "Single",
    _ROW_CLASS_OTHER: "Other",
}
_EXACT_ROUTE_EMPTY = 0
_EXACT_ROUTE_FULL = 1
_EXACT_ROUTE_MIXED = 2
_ROUTE_MODE_JOINT = "joint"
_ROUTE_MODE_SELECTED = "selected"
_ROUTE_MODE_COMPLEMENT = "complement"
_ROUTE_MODE_WEIGHTED_JOINT = "weighted_joint"
_ROUTE_MODE_ALIASES = {
    "joint": _ROUTE_MODE_JOINT,
    "jointmode": _ROUTE_MODE_JOINT,
    "selected": _ROUTE_MODE_SELECTED,
    "sel": _ROUTE_MODE_SELECTED,
    "selmode": _ROUTE_MODE_SELECTED,
    "selectedmode": _ROUTE_MODE_SELECTED,
    "complement": _ROUTE_MODE_COMPLEMENT,
    "comp": _ROUTE_MODE_COMPLEMENT,
    "compmode": _ROUTE_MODE_COMPLEMENT,
    "complementmode": _ROUTE_MODE_COMPLEMENT,
    "weighted_joint": _ROUTE_MODE_WEIGHTED_JOINT,
    "weightedjoint": _ROUTE_MODE_WEIGHTED_JOINT,
}
_ROUTE_REGION_KEY_ALIASES = {
    "threshold": "threshold",
    "tau": "threshold",
    "selected_weight": "selected_weight",
    "lambda_s": "selected_weight",
    "lambda_selected": "selected_weight",
    "complement_weight": "complement_weight",
    "lambda_c": "complement_weight",
    "lambda_complement": "complement_weight",
}
_VISPT_MODE_LEGACY = "legacy"
_VISPT_MODE_SIZE = "size"
_VISPT_MODE_VRF = "vrf"
_VISPT_MODE_ALIASES = {
    "legacy": _VISPT_MODE_LEGACY,
    "size": _VISPT_MODE_SIZE,
    "sizemode": _VISPT_MODE_SIZE,
    "vrf": _VISPT_MODE_VRF,
    "vrfmode": _VISPT_MODE_VRF,
    "verify": _VISPT_MODE_VRF,
    "verification": _VISPT_MODE_VRF,
}
_ROUTE_OBJECTIVE_SIZE = "size"
_ROUTE_OBJECTIVE_VRF = "vrf"
_ROUTE_OBJECTIVE_ALIASES = {
    "size": _ROUTE_OBJECTIVE_SIZE,
    "sizeaware": _ROUTE_OBJECTIVE_SIZE,
    "size_aware": _ROUTE_OBJECTIVE_SIZE,
    "sizemode": _ROUTE_OBJECTIVE_SIZE,
    "vrf": _ROUTE_OBJECTIVE_VRF,
    "verify": _ROUTE_OBJECTIVE_VRF,
    "verification": _ROUTE_OBJECTIVE_VRF,
    "verifyaware": _ROUTE_OBJECTIVE_VRF,
    "verify_aware": _ROUTE_OBJECTIVE_VRF,
    "vrfaware": _ROUTE_OBJECTIVE_VRF,
    "vrf_aware": _ROUTE_OBJECTIVE_VRF,
}
_ROUTE_POLICY_PROFILE = "profile"
_ROUTE_POLICY_FULL_SUPPORT = "full_support"
_ROUTE_POLICY_ALIASES = {
    "profile": _ROUTE_POLICY_PROFILE,
    "profilemode": _ROUTE_POLICY_PROFILE,
    "profile_mode": _ROUTE_POLICY_PROFILE,
    "full_support": _ROUTE_POLICY_FULL_SUPPORT,
    "fullsupport": _ROUTE_POLICY_FULL_SUPPORT,
    "fullsupportmode": _ROUTE_POLICY_FULL_SUPPORT,
    "full_support_mode": _ROUTE_POLICY_FULL_SUPPORT,
    "full-support": _ROUTE_POLICY_FULL_SUPPORT,
    "legal": _ROUTE_POLICY_FULL_SUPPORT,
    "full": _ROUTE_POLICY_FULL_SUPPORT,
}
_SCORE_NAME_SIZE = "size"
_SCORE_NAME_VRF = "vrf"
_SCORE_NAME_ALIASES = {
    "size": _SCORE_NAME_SIZE,
    "sizemode": _SCORE_NAME_SIZE,
    "vrf": _SCORE_NAME_VRF,
    "vrfmode": _SCORE_NAME_VRF,
    "verify": _SCORE_NAME_VRF,
    "verification": _SCORE_NAME_VRF,
}


@dataclass
class _ExactRoutePlan:
    counts: tuple[int, ...]
    root_lengths: tuple[int, ...]
    subtree_items: dict[int, tuple[tuple[tuple[object, ...], int], ...]]
    suffix_tables_by_counts: tuple[
        dict[tuple[int, ...], tuple[tuple[int, int, int], ...]],
        ...
    ]
    support: int
    suffix_support_cache: dict[tuple[int, tuple[int, ...], int, int], int] = field(
        default_factory=dict
    )


@dataclass(frozen=True)
class _ProfileRoutePlan:
    profile: tuple[tuple[int, ...], ...]
    support: int
    support_bits: float
    ordered_rows: tuple[tuple[int, ...], ...]
    ordered_row_counts: tuple[int, ...]


def _pattern_to_mask(pattern: Sequence[int]) -> int:
    mask = 0
    for value in pattern:
        mask |= 1 << value
    return mask


def _default_pattern_family(max_g_value: int) -> tuple[tuple[int, ...], ...]:
    patterns: list[tuple[int, ...]] = [()]
    width = 1
    while width <= max_g_value:
        for start in range(0, max_g_value, width):
            if start % width == 0 and start + width <= max_g_value:
                patterns.append(tuple(range(start, start + width)))
        width *= 2
    return tuple(dict.fromkeys(patterns))


def _normalize_pattern_family(
    pattern_family: Optional[Sequence[Sequence[int]]],
    max_g_value: int,
) -> Optional[tuple[tuple[int, ...], ...]]:
    if pattern_family is None:
        return None

    normalized: list[tuple[int, ...]] = []
    seen: set[tuple[int, ...]] = set()
    for raw_pattern in pattern_family:
        pattern = tuple(int(value) for value in raw_pattern)
        if any(value < 0 or value >= max_g_value for value in pattern):
            raise ValueError("pattern_family entries must lie in [0, max_g_value)")
        if any(left >= right for left, right in zip(pattern, pattern[1:])):
            raise ValueError("pattern_family entries must be strictly increasing")
        if pattern in seen:
            raise ValueError("pattern_family must not contain duplicate patterns")
        seen.add(pattern)
        normalized.append(pattern)

    if not normalized:
        raise ValueError("pattern_family must contain at least one pattern")
    return tuple(normalized)


def _all_row_patterns(max_g_value: int) -> tuple[tuple[int, ...], ...]:
    return tuple(_sorted_values_from_mask(mask) for mask in range(1 << max_g_value))


def _normalize_vispt_mode(mode: Optional[str]) -> str:
    if mode is None:
        return _VISPT_MODE_LEGACY
    normalized = str(mode).strip().lower().replace("-", "_")
    resolved = _VISPT_MODE_ALIASES.get(normalized)
    if resolved is None:
        raise ValueError(f"unsupported mode={mode!r}")
    return resolved


def _normalize_route_policy(route_policy: Optional[str]) -> Optional[str]:
    if route_policy is None:
        return None
    normalized = str(route_policy).strip().lower().replace("-", "_")
    resolved = _ROUTE_POLICY_ALIASES.get(normalized)
    if resolved is None:
        raise ValueError(f"unsupported route_policy={route_policy!r}")
    return resolved


def _normalize_route_objective(
    route_objective: Optional[str],
    mode: str,
) -> str:
    if route_objective is None:
        if mode == _VISPT_MODE_VRF:
            return _ROUTE_OBJECTIVE_VRF
        return _ROUTE_OBJECTIVE_SIZE
    normalized = str(route_objective).strip().lower().replace("-", "_")
    resolved = _ROUTE_OBJECTIVE_ALIASES.get(normalized)
    if resolved is None:
        raise ValueError(f"unsupported route_objective={route_objective!r}")
    return resolved


def _normalize_score_name(
    score_name: Optional[str],
    mode: str,
    score_bound: Optional[int],
) -> Optional[str]:
    if score_name is None:
        if score_bound is None:
            return None
        if mode == _VISPT_MODE_SIZE:
            return _SCORE_NAME_SIZE
        if mode == _VISPT_MODE_VRF:
            return _SCORE_NAME_VRF
        return None
    normalized = str(score_name).strip().lower().replace("-", "_")
    resolved = _SCORE_NAME_ALIASES.get(normalized)
    if resolved is None:
        raise ValueError(f"unsupported score_name={score_name!r}")
    return resolved


def _normalize_score_bound(score_bound: Optional[int | float]) -> Optional[int]:
    if score_bound is None:
        return None
    if score_bound == float("inf"):
        return None
    value = int(score_bound)
    if value < 0:
        raise ValueError("score_bound must be non-negative or infinity")
    return value


def _canonicalize_public_value(value: Any) -> Any:
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if isinstance(value, Mapping):
        return {
            str(key): _canonicalize_public_value(subvalue)
            for key, subvalue in sorted(value.items(), key=lambda item: str(item[0]))
        }
    if isinstance(value, Sequence) and not isinstance(value, (bytes, bytearray, str)):
        return [_canonicalize_public_value(item) for item in value]
    raise TypeError(f"unsupported public parameter type: {type(value)!r}")


def _canonical_aux_mode(aux_mode: Optional[Mapping[str, Any]]) -> tuple[tuple[str, Any], ...]:
    if aux_mode is None:
        return ()
    if not isinstance(aux_mode, Mapping):
        raise TypeError("aux_mode must be a mapping")
    normalized = _canonicalize_public_value(dict(aux_mode))
    assert isinstance(normalized, dict)
    return tuple((str(key), normalized[key]) for key in sorted(normalized))


def _encoded_aux_mode_material(aux_mode: Sequence[tuple[str, Any]]) -> bytes:
    return json.dumps(
        {key: value for key, value in aux_mode},
        ensure_ascii=True,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("ascii")


def _canonical_forest_lengths(leaf_count: int) -> tuple[int, ...]:
    if leaf_count < 0:
        raise ValueError("leaf_count must be non-negative")
    lengths = []
    start = 0
    remaining = leaf_count
    while remaining > 0:
        width = 1 << (remaining.bit_length() - 1)
        while start % width != 0:
            width >>= 1
        lengths.append(width)
        start += width
        remaining -= width
    return tuple(lengths)


def _append_encoded_nonnegative_int(buffer: bytearray, value: int) -> None:
    if value < 0:
        raise ValueError("cannot encode a negative integer")
    byte_len = max(1, (value.bit_length() + 7) // 8)
    if byte_len >= 1 << 16:
        raise ValueError("integer is too large to encode")
    buffer.extend(byte_len.to_bytes(2, "big"))
    buffer.extend(value.to_bytes(byte_len, "big"))


def _append_encoded_bytes(buffer: bytearray, value: bytes) -> None:
    _append_encoded_nonnegative_int(buffer, len(value))
    buffer.extend(value)


def _append_encoded_int_sequence(buffer: bytearray, values: Sequence[int]) -> None:
    _append_encoded_nonnegative_int(buffer, len(values))
    for value in values:
        _append_encoded_nonnegative_int(buffer, int(value))


def _normalize_shape_parms(
    shape_parms: Optional[Mapping[str, Any]],
) -> tuple[tuple[str, int], ...]:
    if shape_parms is None:
        return ()
    if not isinstance(shape_parms, Mapping):
        raise TypeError("shape_parms must be a mapping from shape-limit names to integers")

    normalized: dict[str, int] = {}
    for raw_key, raw_value in shape_parms.items():
        key = _SHAPE_LIMIT_ALIASES.get(str(raw_key))
        if key is None:
            raise ValueError(f"unsupported shape_parms key={raw_key!r}")
        value = int(raw_value)
        if value < 0:
            raise ValueError("shape_parms limits must be non-negative")
        normalized[key] = value
    return tuple(sorted(normalized.items()))


def _encoded_shape_parms_material(shape_parms: Sequence[tuple[str, int]]) -> bytes:
    encoded = bytearray()
    _append_encoded_nonnegative_int(encoded, len(shape_parms))
    for key, value in shape_parms:
        _append_encoded_bytes(encoded, key.encode("ascii"))
        _append_encoded_nonnegative_int(encoded, value)
    return bytes(encoded)


def _shape_delta_from_mask(mask: int, max_g_value: int) -> tuple[int, int, int, int, int, int, int]:
    size = mask.bit_count()
    return (
        1 if size == 0 else 0,
        1 if size == 1 else 0,
        1 if size == 2 else 0,
        1 if max_g_value > 1 and size == max_g_value // 2 else 0,
        1 if size >= 4 else 0,
        1 if size > 0 else 0,
        1 if size == max_g_value else 0,
    )


def _add_shape_delta(
    state: tuple[int, int, int, int, int, int, int],
    delta: tuple[int, int, int, int, int, int, int],
) -> tuple[int, int, int, int, int, int, int]:
    return tuple(left + right for left, right in zip(state, delta))  # type: ignore[return-value]


def _shape_state_allowed(
    state: Sequence[int],
    shape_limits: Sequence[tuple[int, int]],
) -> bool:
    return all(state[index] <= limit for index, limit in shape_limits)


def _shape_statistics_from_groups(
    groups: Sequence[Sequence[int]],
    max_g_value: int,
) -> tuple[int, int, int, int, int, int, int]:
    state = (0, 0, 0, 0, 0, 0, 0)
    for group in groups:
        state = _add_shape_delta(state, _shape_delta_from_mask(_pattern_to_mask(group), max_g_value))
    return state


def _encoded_pattern_family_material(
    pattern_family_unrestricted: bool,
    pattern_family: Optional[Sequence[Sequence[int]]],
) -> bytes:
    encoded = bytearray()
    if pattern_family_unrestricted:
        encoded.append(0)
        return bytes(encoded)

    if pattern_family is None:
        raise ValueError("pattern_family is not available")
    encoded.append(1)
    _append_encoded_nonnegative_int(encoded, len(pattern_family))
    for pattern in pattern_family:
        _append_encoded_int_sequence(encoded, pattern)
    return bytes(encoded)


def _dyadic_interval_patterns(max_g_value: int) -> tuple[tuple[int, ...], ...]:
    if max_g_value <= 0 or max_g_value & (max_g_value - 1):
        raise ValueError("max_g_value must be a positive power of two")
    patterns: list[tuple[int, ...]] = []
    width = 1
    while width <= max_g_value:
        for start in range(0, max_g_value, width):
            patterns.append(tuple(range(start, start + width)))
        width *= 2
    return tuple(patterns)


def _row_pattern_universe(max_g_value: int) -> tuple[tuple[int, ...], ...]:
    full = tuple(range(max_g_value))
    patterns: list[tuple[int, ...]] = [(), full]
    for interval in _dyadic_interval_patterns(max_g_value):
        patterns.append(interval)
    full_set = set(full)
    for interval in _dyadic_interval_patterns(max_g_value):
        complement = tuple(value for value in full if value in full_set - set(interval))
        patterns.append(complement)
    return tuple(dict.fromkeys(patterns))


def _default_prefix_dict(max_g_value: int) -> tuple[tuple[int, ...], ...]:
    """
    Public tree-first prefix dictionary used when none is supplied.

    This follows the paper examples: full row, aligned halves, then one more
    dyadic level when useful.
    """

    min_width = max(2, max_g_value // 4)
    patterns: list[tuple[int, ...]] = []
    width = max_g_value
    while width >= min_width:
        for start in range(0, max_g_value, width):
            patterns.append(tuple(range(start, start + width)))
        width //= 2
    return tuple(dict.fromkeys(patterns))


def _normalize_prefix_dict(
    prefix_dict: Optional[Sequence[Sequence[int]]],
    pattern_family: Optional[Sequence[Sequence[int]]],
    max_g_value: int,
) -> tuple[tuple[int, ...], ...]:
    source = prefix_dict
    if source is None and pattern_family is not None:
        source = [pattern for pattern in pattern_family if pattern]
    if source is None:
        return _default_prefix_dict(max_g_value)

    normalized = _normalize_pattern_family(source, max_g_value)
    if normalized is None:
        return ()
    if any(len(pattern) == 0 for pattern in normalized):
        raise ValueError("prefix_dict must not contain the empty pattern")
    universe = set(_row_pattern_universe(max_g_value))
    if any(pattern not in universe for pattern in normalized):
        raise ValueError("prefix_dict entries must lie in the dyadic row-pattern universe")
    return normalized


def _encoded_prefix_dict_material(prefix_dict: Sequence[Sequence[int]]) -> bytes:
    encoded = bytearray()
    _append_encoded_nonnegative_int(encoded, len(prefix_dict))
    for pattern in prefix_dict:
        _append_encoded_int_sequence(encoded, pattern)
    return bytes(encoded)


def _row_block_count_vector(
    realization: Sequence[Sequence[int]],
    max_g_value: int,
) -> tuple[int, ...]:
    counts = [0] * max_g_value
    for row in realization:
        for value in row:
            counts[value] += 1
    return tuple(counts)


def _normalize_bt_realization(
    raw_realization: Sequence[Sequence[int]],
    *,
    block_len: int,
    max_g_value: int,
) -> tuple[tuple[int, ...], ...]:
    if len(raw_realization) != block_len:
        raise ValueError("BT realization length must match its block length")
    rows = []
    for raw_row in raw_realization:
        row = tuple(int(value) for value in raw_row)
        if any(value < 0 or value >= max_g_value for value in row):
            raise ValueError("BT row patterns must lie in [0, max_g_value)")
        if any(left >= right for left, right in zip(row, row[1:])):
            raise ValueError("BT row patterns must be strictly increasing")
        rows.append(row)
    return tuple(rows)


def _normalize_bt_template(
    raw_template: Any,
    *,
    block_len: int,
    max_g_value: int,
) -> BTTemplate:
    if isinstance(raw_template, Mapping):
        raw_counts = (
            raw_template.get("counts")
            if "counts" in raw_template
            else raw_template.get("count_vector", raw_template.get("template_vec"))
        )
        raw_realizations = raw_template.get(
            "realizations",
            raw_template.get("realization_set"),
        )
    else:
        if not isinstance(raw_template, Sequence) or len(raw_template) != 2:
            raise ValueError("BT template must be a mapping or a (counts, realizations) pair")
        raw_counts, raw_realizations = raw_template

    if raw_realizations is None:
        raise ValueError("BT template is missing realizations")
    realizations = tuple(
        _normalize_bt_realization(
            raw_realization,
            block_len=block_len,
            max_g_value=max_g_value,
        )
        for raw_realization in raw_realizations
    )
    if not realizations:
        raise ValueError("BT template must contain at least one realization")

    computed_counts = _row_block_count_vector(realizations[0], max_g_value)
    for realization in realizations[1:]:
        if _row_block_count_vector(realization, max_g_value) != computed_counts:
            raise ValueError("all BT realizations in a template must cover the same counts")

    if raw_counts is None:
        counts = computed_counts
    else:
        counts = tuple(int(value) for value in raw_counts)
        if len(counts) != max_g_value or any(value < 0 for value in counts):
            raise ValueError("BT template count vector has invalid length or negative entry")
        if counts != computed_counts:
            raise ValueError("BT template count vector does not match its realizations")

    return BTTemplate(counts=counts, realizations=realizations)


def _bt_family_items(raw_bt_families: Any) -> tuple[tuple[int, Any], ...]:
    if raw_bt_families is None:
        return ()
    if isinstance(raw_bt_families, Mapping):
        return tuple((int(block_len), raw_templates) for block_len, raw_templates in raw_bt_families.items())
    if isinstance(raw_bt_families, Sequence):
        return tuple(
            (index, raw_templates)
            for index, raw_templates in enumerate(raw_bt_families, start=1)
        )
    raise TypeError("bt_families must be a mapping or sequence")


def _resolve_bt_block_size(raw_bt_block_size: int, raw_bt_families: Any) -> int:
    if raw_bt_block_size < 0:
        raise ValueError("bt_block_size must be non-negative")
    if raw_bt_block_size > 0 or raw_bt_families is None:
        return raw_bt_block_size
    family_items = _bt_family_items(raw_bt_families)
    return max((block_len for block_len, _raw_templates in family_items), default=0)


def _normalize_bt_families(
    raw_bt_families: Any,
    *,
    max_g_value: int,
    bt_block_size: int,
) -> tuple[tuple[BTTemplate, ...], ...]:
    families: list[list[BTTemplate]] = [[] for _ in range(bt_block_size + 1)]
    seen_by_block_len: dict[int, set[tuple[tuple[int, ...], ...]]] = {
        block_len: set()
        for block_len in range(1, bt_block_size + 1)
    }
    for block_len, raw_templates in _bt_family_items(raw_bt_families):
        if block_len < 1 or block_len > bt_block_size:
            raise ValueError("BT family block length is outside [1, bt_block_size]")
        if raw_templates is None:
            continue
        for raw_template in raw_templates:
            template = _normalize_bt_template(
                raw_template,
                block_len=block_len,
                max_g_value=max_g_value,
            )
            for realization in template.realizations:
                if realization in seen_by_block_len[block_len]:
                    raise ValueError("BT realization sets must be pairwise disjoint")
                seen_by_block_len[block_len].add(realization)
            families[block_len].append(template)
    return tuple(tuple(family) for family in families)


def _encoded_bt_families_material(bt_families: Sequence[Sequence[BTTemplate]]) -> bytes:
    encoded = bytearray()
    _append_encoded_nonnegative_int(encoded, max(0, len(bt_families) - 1))
    for block_len in range(1, len(bt_families)):
        family = bt_families[block_len]
        _append_encoded_nonnegative_int(encoded, block_len)
        _append_encoded_nonnegative_int(encoded, len(family))
        for template in family:
            _append_encoded_int_sequence(encoded, template.counts)
            _append_encoded_nonnegative_int(encoded, len(template.realizations))
            for realization in template.realizations:
                _append_encoded_nonnegative_int(encoded, len(realization))
                for row in realization:
                    _append_encoded_int_sequence(encoded, row)
    return bytes(encoded)


@dataclass(frozen=True)
class TreeAwareISPParameters:
    """Parameters for the TreeAwareISP algorithm."""

    hash_len: int
    max_g_bit: int
    partition_num: int
    aux_t: Optional[Mapping[str, Any]] = None
    route_policy: Optional[str] = None
    size_threshold: Optional[int | float] = None
    vrf_threshold: Optional[int | float] = None
    mode: str = _VISPT_MODE_LEGACY
    aux_mode: Optional[Mapping[str, Any]] = None
    score_name: Optional[str] = None
    score_bound: Optional[int | float] = None
    window_radius: Optional[int] = None
    pattern_family: Optional[Sequence[Sequence[int]]] = None
    window_radius_l: Optional[int] = None
    window_radius_u: Optional[int] = None
    prefix_dict: Optional[Sequence[Sequence[int]]] = None
    loss_bound: int = 0
    prefix_limit: int = 0
    bt_block_size: int = 0
    bt_families: Optional[Any] = None
    bt_loss_bound: int = 0
    # Deprecated compatibility field from the local-shape draft. It is no
    # longer part of the ValStrictISPT sampler.
    shape_parms: Optional[Mapping[str, Any]] = None
    # Deprecated compatibility field.  The tree-aware sampler no longer applies
    # a global TreeScore threshold; tree_score is used only for performance
    # measurement by the surrounding YCSig benchmarks.
    tree_threshold: Optional[int] = None
    link_threshold: int = -1
    hash_name: str = DEFAULT_HASH_NAME
    route_objective: Optional[str] = None

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
        resolved_mode = _normalize_vispt_mode(self.mode)
        resolved_route_policy = _normalize_route_policy(self.route_policy)
        resolved_score_bound = _normalize_score_bound(self.score_bound)
        if resolved_score_bound is None and self.tree_threshold is not None and resolved_mode != _VISPT_MODE_LEGACY:
            resolved_score_bound = self.tree_threshold
        resolved_score_name = _normalize_score_name(
            self.score_name,
            resolved_mode,
            resolved_score_bound,
        )
        resolved_size_threshold = _normalize_score_bound(self.size_threshold)
        resolved_vrf_threshold = _normalize_score_bound(self.vrf_threshold)
        if resolved_score_bound is not None:
            if (
                resolved_score_name == _SCORE_NAME_SIZE
                and resolved_size_threshold is None
            ):
                resolved_size_threshold = resolved_score_bound
            if (
                resolved_score_name == _SCORE_NAME_VRF
                and resolved_vrf_threshold is None
            ):
                resolved_vrf_threshold = resolved_score_bound
        canonical_aux_t = _canonical_aux_mode(
            self.aux_t if self.aux_t is not None else self.aux_mode
        )
        aux_t_map = {key: value for key, value in canonical_aux_t}
        raw_route_objective = self.route_objective
        if raw_route_objective is None:
            raw_route_objective = aux_t_map.get(
                "route_objective",
                aux_t_map.get("objective", aux_t_map.get("obj")),
            )
        resolved_route_objective = _normalize_route_objective(
            None if raw_route_objective is None else str(raw_route_objective),
            resolved_mode,
        )
        entropy_floor_raw = aux_t_map.get("entropy_floor", aux_t_map.get("h_min", 0))
        entropy_floor = int(entropy_floor_raw)
        if entropy_floor < 0:
            raise ValueError("entropy_floor must be non-negative")
        profile_rule_name = str(aux_t_map.get("profile_rule", "dyadic_greedy")).strip().lower()
        block_num = self.hash_len // self.max_g_bit
        max_g_value = 1 << self.max_g_bit
        avg_floor = block_num // max_g_value
        avg_ceil = (block_num + max_g_value - 1) // max_g_value
        if self.window_radius is not None and self.window_radius < 0:
            raise ValueError("window_radius must be non-negative")
        resolved_window_radius_l = (
            self.window_radius
            if self.window_radius_l is None
            else self.window_radius_l
        )
        resolved_window_radius_u = (
            self.window_radius
            if self.window_radius_u is None
            else self.window_radius_u
        )
        if resolved_window_radius_l is None:
            resolved_window_radius_l = 0
        if resolved_window_radius_u is None:
            resolved_window_radius_u = 0
        if resolved_window_radius_l < 0 or resolved_window_radius_u < 0:
            raise ValueError("window radii must be non-negative")
        low = avg_floor - resolved_window_radius_l
        # Cap away from the fully saturated count so the public window matches
        # the ValStrictISPT definition used in the paper.
        high = min(avg_ceil + resolved_window_radius_u, self.partition_num - 1)
        pattern_family = _normalize_pattern_family(self.pattern_family, max_g_value)
        dy_shape_family_source = aux_t_map.get("dy_shape_family")
        if dy_shape_family_source is None:
            dy_shape_family_source = aux_t_map.get("shape_family")
        dy_shape_family = (
            (
                pattern_family
                if pattern_family is not None
                else _default_pattern_family(max_g_value)
            )
            if dy_shape_family_source is None
            else _normalize_pattern_family(dy_shape_family_source, max_g_value)
        )
        dy_shape_index_by_row = (
            None
            if dy_shape_family is None
            else {row: index for index, row in enumerate(dy_shape_family)}
        )
        dy_shape_masks = (
            None
            if dy_shape_family is None
            else tuple(_pattern_to_mask(row) for row in dy_shape_family)
        )
        dy_shape_local_scores = (
            None
            if dy_shape_family is None
            else tuple(_row_local_tree_score(row, max_g_value) for row in dy_shape_family)
        )
        dy_shape_local_cost_pairs = (
            None
            if dy_shape_family is None
            else tuple(_row_local_tree_cost_pair(row, max_g_value) for row in dy_shape_family)
        )
        ordered_dy_shape_indices = (
            ()
            if (
                dy_shape_family is None
                or dy_shape_index_by_row is None
                or dy_shape_local_scores is None
                or dy_shape_local_cost_pairs is None
            )
            else tuple(
                sorted(
                    range(len(dy_shape_family)),
                    key=lambda index: _profile_shape_sort_key(
                        dy_shape_family[index],
                        dy_shape_local_scores[index],
                        dy_shape_local_cost_pairs[index],
                        index,
                        resolved_route_objective,
                        aux_t_map,
                    ),
                )
            )
        )
        ordered_dy_shape_candidates = (
            ()
            if not ordered_dy_shape_indices or dy_shape_family is None
            else tuple(dy_shape_family[index] for index in ordered_dy_shape_indices)
        )
        ordered_dy_shape_candidate_masks = (
            ()
            if not ordered_dy_shape_indices or dy_shape_masks is None
            else tuple(dy_shape_masks[index] for index in ordered_dy_shape_indices)
        )
        ordered_dy_shape_candidates_nonempty = tuple(
            row for row in ordered_dy_shape_candidates if row
        )
        ordered_dy_shape_candidate_masks_nonempty = tuple(
            mask
            for row, mask in zip(ordered_dy_shape_candidates, ordered_dy_shape_candidate_masks)
            if row
        )
        prefix_dict = _normalize_prefix_dict(self.prefix_dict, pattern_family, max_g_value)
        if self.loss_bound < 0:
            raise ValueError("loss_bound must be non-negative")
        if self.prefix_limit < 0:
            raise ValueError("prefix_limit must be non-negative")
        if self.bt_loss_bound < 0:
            raise ValueError("bt_loss_bound must be non-negative")
        bt_block_size = _resolve_bt_block_size(self.bt_block_size, self.bt_families)
        bt_families = _normalize_bt_families(
            self.bt_families,
            max_g_value=max_g_value,
            bt_block_size=bt_block_size,
        )
        if self.tree_threshold is not None and self.tree_threshold < 0:
            raise ValueError("tree_threshold must be non-negative")
        shape_parms = _normalize_shape_parms(self.shape_parms)
        shape_limits = tuple(
            (_SHAPE_LIMIT_INDEX[key], value)
            for key, value in shape_parms
        )
        leaf_universe_size = self.partition_num * max_g_value
        root_intervals = []
        root_offset = 0
        for root_width in _canonical_forest_lengths(leaf_universe_size):
            root_intervals.append((root_offset, root_width))
            root_offset += root_width
        forest_root_num = len(root_intervals)
        leaf_universe_full_mask = (1 << leaf_universe_size) - 1
        has_profile_rule = "shape_profile" in aux_t_map or isinstance(
            aux_t_map.get("shape_profiles"),
            Mapping,
        )
        if resolved_route_policy == _ROUTE_POLICY_PROFILE:
            routing_strategy = _ROUTE_POLICY_PROFILE
        elif resolved_route_policy == _ROUTE_POLICY_FULL_SUPPORT:
            routing_strategy = _ROUTE_POLICY_FULL_SUPPORT
        elif (
            has_profile_rule
            or resolved_mode in {_VISPT_MODE_SIZE, _VISPT_MODE_VRF}
            or self.aux_t is not None
            or self.route_objective is not None
        ):
            routing_strategy = _ROUTE_POLICY_PROFILE
        else:
            routing_strategy = _VISPT_MODE_LEGACY
        if routing_strategy == _ROUTE_POLICY_FULL_SUPPORT and (
            self.prefix_limit != 0
            or bt_block_size != 0
            or self.bt_loss_bound != 0
        ):
            raise ValueError(
                "route_policy='full_support' is incompatible with prefix extraction or BT routing options"
            )
        compatibility_score_name: Optional[str] = None
        compatibility_score_bound: Optional[int] = None
        if (
            compatibility_score_name is None
            and resolved_size_threshold is not None
            and resolved_vrf_threshold is None
        ):
            compatibility_score_name = _SCORE_NAME_SIZE
            compatibility_score_bound = resolved_size_threshold
        elif (
            compatibility_score_name is None
            and resolved_vrf_threshold is not None
            and resolved_size_threshold is None
        ):
            compatibility_score_name = _SCORE_NAME_VRF
            compatibility_score_bound = resolved_vrf_threshold
        object.__setattr__(self, "_block_num", block_num)
        object.__setattr__(self, "_max_g_value", max_g_value)
        object.__setattr__(self, "mode", resolved_mode)
        object.__setattr__(
            self,
            "aux_t",
            None if not canonical_aux_t else dict(aux_t_map),
        )
        object.__setattr__(self, "route_policy", routing_strategy)
        object.__setattr__(self, "route_objective", resolved_route_objective)
        object.__setattr__(
            self,
            "aux_mode",
            None if not canonical_aux_t else dict(aux_t_map),
        )
        object.__setattr__(self, "size_threshold", resolved_size_threshold)
        object.__setattr__(self, "vrf_threshold", resolved_vrf_threshold)
        object.__setattr__(self, "score_name", compatibility_score_name)
        object.__setattr__(self, "score_bound", compatibility_score_bound)
        object.__setattr__(self, "_mode", resolved_mode)
        object.__setattr__(self, "_routing_strategy", routing_strategy)
        object.__setattr__(self, "_route_policy", routing_strategy)
        object.__setattr__(self, "_route_objective", resolved_route_objective)
        object.__setattr__(self, "_aux_t", canonical_aux_t)
        object.__setattr__(self, "_aux_t_map", dict(aux_t_map))
        object.__setattr__(self, "_entropy_floor", entropy_floor)
        object.__setattr__(self, "_profile_rule_name", profile_rule_name)
        object.__setattr__(self, "_aux_mode", canonical_aux_t)
        object.__setattr__(self, "_aux_mode_map", dict(aux_t_map))
        object.__setattr__(self, "_size_threshold", resolved_size_threshold)
        object.__setattr__(self, "_vrf_threshold", resolved_vrf_threshold)
        object.__setattr__(self, "_score_name", compatibility_score_name)
        object.__setattr__(self, "_score_bound", compatibility_score_bound)
        object.__setattr__(
            self,
            "_score_guard_enabled",
            resolved_size_threshold is not None
            or resolved_vrf_threshold is not None,
        )
        object.__setattr__(self, "_leaf_universe_size", leaf_universe_size)
        object.__setattr__(self, "_leaf_universe_full_mask", leaf_universe_full_mask)
        object.__setattr__(self, "_leaf_universe_root_intervals", tuple(root_intervals))
        object.__setattr__(self, "_forest_root_num", forest_root_num)
        object.__setattr__(self, "window_radius_l", resolved_window_radius_l)
        object.__setattr__(self, "window_radius_u", resolved_window_radius_u)
        object.__setattr__(
            self,
            "window_radius",
            max(resolved_window_radius_l, resolved_window_radius_u),
        )
        object.__setattr__(self, "_window_low", low)
        object.__setattr__(self, "_window_high", high)
        object.__setattr__(self, "_window_valid", low >= 0 and high >= low)
        object.__setattr__(self, "_pattern_family", pattern_family)
        object.__setattr__(self, "_dy_shape_family", dy_shape_family)
        object.__setattr__(
            self,
            "_dy_shape_family_set",
            None if dy_shape_family is None else frozenset(dy_shape_family),
        )
        object.__setattr__(self, "_dy_shape_index_by_row", dy_shape_index_by_row)
        object.__setattr__(self, "_dy_shape_masks", dy_shape_masks)
        object.__setattr__(self, "_dy_shape_family_size", 0 if dy_shape_family is None else len(dy_shape_family))
        object.__setattr__(self, "_ordered_dy_shape_candidates", ordered_dy_shape_candidates)
        object.__setattr__(
            self,
            "_ordered_dy_shape_candidate_masks",
            ordered_dy_shape_candidate_masks,
        )
        object.__setattr__(
            self,
            "_ordered_dy_shape_candidates_nonempty",
            ordered_dy_shape_candidates_nonempty,
        )
        object.__setattr__(
            self,
            "_ordered_dy_shape_candidate_masks_nonempty",
            ordered_dy_shape_candidate_masks_nonempty,
        )
        object.__setattr__(
            self,
            "_dy_shape_empty_available",
            bool(dy_shape_index_by_row is not None and () in dy_shape_index_by_row),
        )
        object.__setattr__(self, "_verify_score_base", 3 * block_num - forest_root_num)
        object.__setattr__(
            self,
            "_pattern_masks",
            None if pattern_family is None else tuple(_pattern_to_mask(pattern) for pattern in pattern_family),
        )
        object.__setattr__(self, "_pattern_family_unrestricted", pattern_family is None)
        object.__setattr__(self, "_shape_parms", shape_parms)
        object.__setattr__(self, "_shape_limits", shape_limits)
        object.__setattr__(self, "_shape_guard_default", len(shape_limits) == 0)
        object.__setattr__(self, "_tree_threshold", self.tree_threshold)
        object.__setattr__(self, "_tree_threshold_default", self.tree_threshold is None)
        object.__setattr__(self, "_prefix_dict", prefix_dict)
        object.__setattr__(self, "_prefix_masks", tuple(_pattern_to_mask(pattern) for pattern in prefix_dict))
        object.__setattr__(self, "_loss_bound", self.loss_bound)
        object.__setattr__(self, "_prefix_limit", self.prefix_limit)
        object.__setattr__(self, "_bt_block_size", bt_block_size)
        object.__setattr__(self, "_bt_families", bt_families)
        object.__setattr__(self, "_bt_loss_bound", self.bt_loss_bound)
        object.__setattr__(self, "_small_partition_fast_path", self.partition_num <= 256)
        object.__setattr__(self, "_sample_base_params", _sample_base_parameters(self.partition_num))
        parameter_material = bytearray()
        _append_encoded_bytes(parameter_material, self.hash_name.encode("ascii"))
        _append_encoded_nonnegative_int(parameter_material, self.hash_len)
        _append_encoded_nonnegative_int(parameter_material, self.max_g_bit)
        _append_encoded_nonnegative_int(parameter_material, self.partition_num)
        _append_encoded_bytes(parameter_material, resolved_mode.encode("ascii"))
        _append_encoded_bytes(parameter_material, routing_strategy.encode("ascii"))
        _append_encoded_bytes(parameter_material, resolved_route_objective.encode("ascii"))
        _append_encoded_bytes(parameter_material, _encoded_aux_mode_material(canonical_aux_t))
        _append_encoded_bytes(
            parameter_material,
            b"" if compatibility_score_name is None else compatibility_score_name.encode("ascii"),
        )
        _append_encoded_nonnegative_int(
            parameter_material,
            0 if compatibility_score_bound is None else compatibility_score_bound + 1,
        )
        _append_encoded_nonnegative_int(
            parameter_material,
            0 if resolved_size_threshold is None else resolved_size_threshold + 1,
        )
        _append_encoded_nonnegative_int(
            parameter_material,
            0 if resolved_vrf_threshold is None else resolved_vrf_threshold + 1,
        )
        _append_encoded_nonnegative_int(parameter_material, resolved_window_radius_l)
        _append_encoded_nonnegative_int(parameter_material, resolved_window_radius_u)
        parameter_material.extend(
            _encoded_pattern_family_material(pattern_family is None, pattern_family)
        )
        parameter_material.extend(
            _encoded_pattern_family_material(dy_shape_family is None, dy_shape_family)
        )
        parameter_material.extend(_encoded_prefix_dict_material(prefix_dict))
        _append_encoded_nonnegative_int(parameter_material, self.loss_bound)
        _append_encoded_nonnegative_int(parameter_material, self.prefix_limit)
        _append_encoded_nonnegative_int(parameter_material, bt_block_size)
        parameter_material.extend(_encoded_bt_families_material(bt_families))
        _append_encoded_nonnegative_int(parameter_material, self.bt_loss_bound)
        object.__setattr__(self, "_tree_sampler_parameter_material", bytes(parameter_material))

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
    def forest_root_num(self) -> int:
        return self._forest_root_num

    @property
    def score_guard_enabled(self) -> bool:
        return self._score_guard_enabled


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
                _PARTITION_VALUE_DOMAIN
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


def _multiplicity_profile_from_partition_value(
    partition_value: PartitionValueInput,
    hash_len: int,
    max_g_bit: int,
    max_g_value: int,
) -> List[int]:
    if isinstance(partition_value, bytes) and hash_len % 8 == 0:
        if max_g_bit == 2 and max_g_value == 4:
            byte_len = len(partition_value)
            mask = _W2_LOW_BIT_MASKS.get(byte_len)
            if mask is None:
                mask = int.from_bytes(b"\x55" * byte_len, "big")
                _W2_LOW_BIT_MASKS[byte_len] = mask
            integer_value = int.from_bytes(partition_value, "big")
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

        if max_g_bit == 4 and max_g_value == 16:
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


def window_bounds(params: TreeAwareISPParameters) -> tuple[int, int]:
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
    bitstring = normalize_partition_value(partition_value, hash_len)
    serialized_y = _serialize_bitstring(bitstring)
    if counters_enabled():
        increment("isp.sample_seed_hash")
    hash_name_bytes = hash_name.encode("ascii")
    seed_digest = _hash_bytes(
        _HY_DOMAIN_PREFIX + hash_name_bytes + b"/" + serialized_y,
        64,
        hash_name,
    )
    seed_material = (
        _SAMPLE_POSITION_XOF_PREFIX
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


def _randbelow_from_seed_material_once(
    seed_material: bytes,
    hash_name: str,
    bound: int,
) -> int:
    if bound <= 0:
        raise ValueError("bound must be positive")
    if hash_name == "shake_128":
        fast_digest = hashlib.shake_128(seed_material).digest
    else:
        # Match HashXOF: SHA3-backed samplers expand with SHAKE256.
        fast_digest = hashlib.shake_256(seed_material).digest

    byte_len = max(1, (bound.bit_length() + 7) // 8)
    upper = 1 << (8 * byte_len)
    threshold = upper - (upper % bound)
    buffer = b""
    buffer_len = 0
    offset = 0
    while True:
        end = offset + byte_len
        if buffer_len < end:
            target = max(end, 32 if buffer_len < 32 else 2 * buffer_len)
            buffer = fast_digest(target)
            buffer_len = target
        candidate = int.from_bytes(buffer[offset:end], "big")
        offset = end
        if candidate < threshold:
            return candidate % bound


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


def _node_cover_count(leaf_mask: int, leaf_count: int) -> int:
    if leaf_mask == 0:
        return 0
    full_mask = (1 << leaf_count) - 1
    if leaf_mask == full_mask:
        return 1
    padded_count = 1 << (leaf_count - 1).bit_length()
    padded_full_mask = (1 << padded_count) - 1
    if leaf_mask == padded_full_mask:
        return 1

    total = 0
    stack = [(leaf_mask, padded_count)]
    while stack:
        mask, width = stack.pop()
        if mask == 0:
            continue
        if width == 1:
            total += 1
            continue
        interval_mask = (1 << width) - 1
        if mask == interval_mask:
            total += 1
            continue
        half = width >> 1
        low_mask = mask & ((1 << half) - 1)
        high_mask = mask >> half
        stack.append((high_mask, half))
        stack.append((low_mask, half))
    return total


def tree_cost_pair(
    groups: Sequence[Sequence[int]],
    params: TreeAwareISPParameters,
) -> tuple[int, int]:
    selected_mask = 0
    max_g_value = params._max_g_value
    for row_index, group in enumerate(groups):
        row_base = row_index * max_g_value
        for value in group:
            selected_mask |= 1 << (row_base + value)
    if selected_mask == 0:
        return 0, 1
    if selected_mask == params._leaf_universe_full_mask:
        return 1, 0
    selected_nodes = 0
    complement_nodes = 0
    for root_offset, root_width in params._leaf_universe_root_intervals:
        subtree_selected = (selected_mask >> root_offset) & ((1 << root_width) - 1)
        stack = [(subtree_selected, root_width)]
        while stack:
            subtree_selected, width = stack.pop()
            if subtree_selected == 0:
                complement_nodes += 1
                continue
            subtree_full_mask = (1 << width) - 1
            if subtree_selected == subtree_full_mask:
                selected_nodes += 1
                continue
            if width == 1:
                selected_nodes += 1
                continue
            half = width >> 1
            low_mask = (1 << half) - 1
            stack.append((subtree_selected >> half, half))
            stack.append((subtree_selected & low_mask, half))
    return selected_nodes, complement_nodes


def score_value(
    groups: Sequence[Sequence[int]],
    params: TreeAwareISPParameters,
    score_name: Optional[str] = None,
) -> Optional[int]:
    resolved_score_name = params._score_name if score_name is None else _normalize_score_name(
        score_name,
        params._mode,
        0,
    )
    if resolved_score_name is None:
        return None
    selected_nodes, complement_nodes = tree_cost_pair(groups, params)
    if resolved_score_name == _SCORE_NAME_SIZE:
        return selected_nodes + complement_nodes
    if resolved_score_name == _SCORE_NAME_VRF:
        return (
            3 * params.block_num
            - params.forest_root_num
            - selected_nodes
            + complement_nodes
        )
    raise ValueError(f"unsupported score_name={resolved_score_name!r}")


def verify_score(
    groups: Sequence[Sequence[int]],
    params: TreeAwareISPParameters,
) -> int:
    selected_nodes, complement_nodes = tree_cost_pair(groups, params)
    return params._verify_score_base - selected_nodes + complement_nodes


def leaf_index_set(
    groups: Sequence[Sequence[int]],
    params: TreeAwareISPParameters,
) -> tuple[int, ...]:
    """Return MTLeafIndSet(Groups) in increasing order over the actual leaf universe."""

    selected_indices: list[int] = []
    max_g_value = params.max_g_value
    for group_index, subgroup in enumerate(groups):
        base = group_index * max_g_value
        for block_value in subgroup:
            selected_indices.append(base + int(block_value))
    selected_indices.sort()
    return tuple(selected_indices)


def complement_leaf_index_set(
    groups: Sequence[Sequence[int]],
    params: TreeAwareISPParameters,
) -> tuple[int, ...]:
    """Return the increasing complement of MTLeafIndSet(Groups)."""

    selected = set(leaf_index_set(groups, params))
    leaf_universe_size = params.partition_num * params.max_g_value
    return tuple(
        leaf_index
        for leaf_index in range(leaf_universe_size)
        if leaf_index not in selected
    )


def score_guard(
    groups: Sequence[Sequence[int]],
    params: TreeAwareISPParameters,
    *,
    score_name: Optional[str] = None,
    score_bound: Optional[int | float] = None,
    size_threshold: Optional[int | float] = None,
    vrf_threshold: Optional[int | float] = None,
) -> bool:
    if score_name is not None or score_bound is not None:
        resolved_score_bound = (
            params._score_bound if score_bound is None else _normalize_score_bound(score_bound)
        )
        if resolved_score_bound is None:
            return True
        resolved_score_name = (
            params._score_name
            if score_name is None
            else _normalize_score_name(score_name, params._mode, resolved_score_bound)
        )
        if resolved_score_name is None:
            return False
        current_score = score_value(groups, params, resolved_score_name)
        return current_score is not None and current_score <= resolved_score_bound

    resolved_size_threshold = (
        params._size_threshold
        if size_threshold is None
        else _normalize_score_bound(size_threshold)
    )
    resolved_vrf_threshold = (
        params._vrf_threshold
        if vrf_threshold is None
        else _normalize_score_bound(vrf_threshold)
    )
    if resolved_size_threshold is None and resolved_vrf_threshold is None:
        return True
    selected_nodes, complement_nodes = tree_cost_pair(groups, params)
    if resolved_size_threshold is not None:
        if selected_nodes + complement_nodes > resolved_size_threshold:
            return False
    if resolved_vrf_threshold is not None:
        if (
            params._verify_score_base - selected_nodes + complement_nodes
            > resolved_vrf_threshold
        ):
            return False
    return True


def score_guard_t(
    groups: Sequence[Sequence[int]],
    params: TreeAwareISPParameters,
    *,
    size_threshold: Optional[int | float] = None,
    vrf_threshold: Optional[int | float] = None,
) -> bool:
    """Chapter-aligned alias for the dual tree-aware abort guard."""

    return score_guard(
        groups,
        params,
        size_threshold=size_threshold,
        vrf_threshold=vrf_threshold,
    )


def tree_score(groups: Sequence[Sequence[int]], params: TreeAwareISPParameters) -> int:
    """Public deterministic size score based on the actual leaf-universe cover."""

    selected_nodes, complement_nodes = tree_cost_pair(groups, params)
    return selected_nodes + complement_nodes


def shape_statistics(
    groups: Sequence[Sequence[int]],
    params: TreeAwareISPParameters,
) -> dict[str, int]:
    """Return the local row-shape statistics used by ShapeGuard."""

    values = _shape_statistics_from_groups(groups, params._max_g_value)
    return dict(zip(_SHAPE_STAT_KEYS, values))


def shape_guard(
    groups: Sequence[Sequence[int]],
    params: TreeAwareISPParameters,
) -> bool:
    """Public deterministic local ShapeGuard predicate."""

    state = _shape_statistics_from_groups(groups, params._max_g_value)
    return _shape_state_allowed(state, params._shape_limits)


def _tree_sampler_seed_material(
    partition_value: PartitionValueInput,
    counts: Sequence[int],
    params: TreeAwareISPParameters,
    xof_seed_material: Optional[bytes],
) -> bytes:
    if xof_seed_material is None:
        partition_material = _serialize_bitstring(
            normalize_partition_value(partition_value, params.hash_len)
        )
    else:
        partition_material = xof_seed_material

    encoded = bytearray(_TREE_SAMPLER_DOMAIN)
    _append_encoded_bytes(encoded, partition_material)
    encoded.extend(params._tree_sampler_parameter_material)
    _append_encoded_int_sequence(encoded, counts)
    return bytes(encoded)


def _route_seed_material(
    tag: bytes,
    partition_value: PartitionValueInput,
    counts: Sequence[int],
    params: TreeAwareISPParameters,
    xof_seed_material: Optional[bytes],
    value: Optional[int] = None,
) -> bytes:
    if xof_seed_material is None:
        partition_material = _serialize_bitstring(
            normalize_partition_value(partition_value, params.hash_len)
        )
    else:
        partition_material = xof_seed_material

    encoded = bytearray(_TREE_SAMPLER_DOMAIN)
    _append_encoded_bytes(encoded, tag)
    if value is not None:
        _append_encoded_nonnegative_int(encoded, value)
    _append_encoded_bytes(encoded, partition_material)
    encoded.extend(params._tree_sampler_parameter_material)
    _append_encoded_int_sequence(encoded, counts)
    return bytes(encoded)


def _count_vector_key(counts: Sequence[int]) -> str:
    return ",".join(str(int(count)) for count in counts)


@lru_cache(maxsize=None)
def _log2_factorials(limit: int) -> tuple[float, ...]:
    values = [0.0] * (limit + 1)
    running = 0.0
    for value in range(2, limit + 1):
        running += math.log2(value)
        values[value] = running
    return tuple(values)


def _normalize_shape_profile(
    raw_profile: Any,
    *,
    partition_num: int,
    max_g_value: int,
) -> tuple[tuple[int, ...], ...]:
    if not isinstance(raw_profile, Sequence) or isinstance(raw_profile, (bytes, bytearray, str)):
        raise TypeError("shape_profile must be a sequence of row shapes")
    profile = []
    for raw_row in raw_profile:
        row = tuple(int(value) for value in raw_row)
        if any(value < 0 or value >= max_g_value for value in row):
            raise ValueError("shape_profile entries must lie in [0, max_g_value)")
        if any(left >= right for left, right in zip(row, row[1:])):
            raise ValueError("shape_profile rows must be strictly increasing")
        profile.append(row)
    if len(profile) != partition_num:
        raise ValueError("shape_profile length must equal partition_num")
    return tuple(profile)


@lru_cache(maxsize=None)
def _row_local_tree_score(row: tuple[int, ...], max_g_value: int) -> int:
    selected_nodes, complement_nodes = _row_local_tree_cost_pair(row, max_g_value)
    return selected_nodes + complement_nodes


@lru_cache(maxsize=None)
def _row_local_tree_cost_pair(row: tuple[int, ...], max_g_value: int) -> tuple[int, int]:
    mask = _pattern_to_mask(row)
    full_mask = (1 << max_g_value) - 1
    return (
        _node_cover_count(mask, max_g_value),
        _node_cover_count(full_mask ^ mask, max_g_value),
    )


def _profile_shape_sort_key(
    row: tuple[int, ...],
    local_size_score: int,
    local_cost_pair: tuple[int, int],
    index: int,
    route_objective: str,
    aux_t: Mapping[str, Any],
) -> tuple[float, ...]:
    selected_nodes, complement_nodes = local_cost_pair
    if route_objective == _ROUTE_OBJECTIVE_VRF:
        return (
            complement_nodes - selected_nodes,
            complement_nodes,
            -selected_nodes,
            1 if not row else 0,
            index,
        )
    return (
        1 if not row else 0,
        local_size_score,
        -len(row),
        index,
    )


def _profile_support_bits_from_usage(
    usage: Sequence[int],
    partition_num: int,
) -> float:
    log2_fact = _log2_factorials(partition_num)
    return log2_fact[partition_num] - sum(log2_fact[count] for count in usage)


@lru_cache(maxsize=None)
def _balanced_profile_usage_after_fill(
    usage: tuple[int, ...],
    partition_num: int,
    total_shape_types: int,
) -> tuple[int, ...]:
    counts = sorted(usage)
    if len(counts) < total_shape_types:
        counts = [0] * (total_shape_types - len(counts)) + counts
    rows_left = partition_num - sum(counts)
    if rows_left < 0:
        return ()
    shape_types = len(counts)
    if shape_types == 0:
        return () if rows_left == 0 else ()

    active = 1
    current_level = counts[0]
    while active < shape_types and rows_left > 0:
        next_level = counts[active]
        if next_level < current_level:
            next_level = current_level
        gap = next_level - current_level
        if gap == 0:
            active += 1
            continue
        raise_cost = active * gap
        if rows_left < raise_cost:
            break
        rows_left -= raise_cost
        for index in range(active):
            counts[index] = next_level
        current_level = next_level
        active += 1

    if rows_left > 0:
        width = active if active < shape_types else shape_types
        base_increase, remainder = divmod(rows_left, width)
        base_level = current_level + base_increase
        split = width - remainder
        for index in range(split):
            counts[index] = base_level
        for index in range(split, width):
            counts[index] = base_level + 1
        rows_left = 0

    if active < shape_types:
        for index in range(active, shape_types):
            if counts[index] < current_level:
                counts[index] = current_level
    return tuple(counts)


@lru_cache(maxsize=None)
def _optimistic_profile_support_bits_cached(
    usage: tuple[int, ...],
    partition_num: int,
    total_shape_types: int,
) -> float:
    balanced = _balanced_profile_usage_after_fill(
        usage,
        partition_num,
        total_shape_types,
    )
    if not balanced and partition_num != 0:
        return float("-inf")
    return _profile_support_bits_from_usage(balanced, partition_num)


def _optimistic_profile_support_bits(
    usage: Sequence[int],
    partition_num: int,
    total_shape_types: int,
) -> float:
    return _optimistic_profile_support_bits_cached(
        tuple(int(count) for count in usage),
        partition_num,
        total_shape_types,
    )


def _shape_feasible_with_remaining(
    row: tuple[int, ...],
    row_mask: int,
    remaining_counts: Sequence[int],
    rows_left_after: int,
    must_cover: Sequence[int],
) -> Optional[tuple[int, ...]]:
    for value in row:
        if int(remaining_counts[value]) <= 0:
            return None
    for value in must_cover:
        if ((row_mask >> value) & 1) == 0:
            return None
    next_counts = list(int(count) for count in remaining_counts)
    for value in row:
        next_counts[value] -= 1
    for value in row:
        if next_counts[value] > rows_left_after:
            return None
    return tuple(next_counts)


def _profile_matches_counts(
    profile: Sequence[tuple[int, ...]],
    counts: Sequence[int],
    max_g_value: int,
) -> bool:
    observed = [0] * max_g_value
    for row in profile:
        for value in row:
            observed[value] += 1
    return tuple(observed) == tuple(int(count) for count in counts)


def _explicit_profile_for_counts(
    counts: Sequence[int],
    params: TreeAwareISPParameters,
) -> Optional[tuple[tuple[int, ...], ...]]:
    aux_t = params._aux_t_map
    if "shape_profile" in aux_t:
        profile = _normalize_shape_profile(
            aux_t["shape_profile"],
            partition_num=params.partition_num,
            max_g_value=params.max_g_value,
        )
    else:
        profile_map = aux_t.get("shape_profiles")
        if not isinstance(profile_map, Mapping):
            return None
        raw_profile = profile_map.get(_count_vector_key(counts))
        if raw_profile is None:
            return None
        profile = _normalize_shape_profile(
            raw_profile,
            partition_num=params.partition_num,
            max_g_value=params.max_g_value,
        )
    return profile


def _profile_rule_cache_key(
    counts: Sequence[int],
    params: TreeAwareISPParameters,
) -> tuple[object, ...]:
    return (
        tuple(int(count) for count in counts),
        params.partition_num,
        params.max_g_value,
        params._dy_shape_family,
        params._entropy_floor,
        params._profile_rule_name,
        params._route_objective,
    )


def _profile_rule_base_cache_key(
    counts: Sequence[int],
    params: TreeAwareISPParameters,
) -> tuple[object, ...]:
    return (
        tuple(int(count) for count in counts),
        params.partition_num,
        params.max_g_value,
        params._dy_shape_family,
        params._profile_rule_name,
        params._route_objective,
    )


def _sparse_residual_profile(
    counts: Sequence[int],
    partition_num: int,
) -> Optional[tuple[tuple[int, ...], ...]]:
    if partition_num < 0:
        return None
    if partition_num == 0:
        return () if not any(counts) else None
    if any(int(count) < 0 or int(count) > partition_num for count in counts):
        return None

    rows: list[list[int]] = [[] for _ in range(partition_num)]
    cursor = 0
    for value, raw_count in enumerate(counts):
        count = int(raw_count)
        if count <= 0:
            continue
        for offset in range(count):
            row_index = (cursor + offset) % partition_num
            rows[row_index].append(value)
        cursor = (cursor + count) % partition_num
    return tuple(tuple(row) for row in rows)


def _profile_rule_profile(
    counts: Sequence[int],
    params: TreeAwareISPParameters,
) -> Optional[tuple[tuple[int, ...], ...]]:
    explicit = _explicit_profile_for_counts(counts, params)
    if explicit is not None:
        return explicit

    family = params._dy_shape_family
    if family is None:
        return None
    if params._profile_rule_name in {"sparse_residual", "residual_fallback"}:
        return _sparse_residual_profile(counts, params.partition_num)
    if params._profile_rule_name not in {"dyadic_greedy", "greedy", "default"}:
        return None

    cache_key = _profile_rule_cache_key(counts, params)
    cached = _PROFILE_RULE_CACHE.get(cache_key)
    if cached is not None or cache_key in _PROFILE_RULE_CACHE:
        return cached

    ordered_candidates = params._ordered_dy_shape_candidates_nonempty
    ordered_candidate_masks = params._ordered_dy_shape_candidate_masks_nonempty
    index_by_row = params._dy_shape_index_by_row
    if index_by_row is None:
        return None
    empty_row = ()
    empty_available = params._dy_shape_empty_available
    total_shape_types = params._dy_shape_family_size
    if (
        _optimistic_profile_support_bits(
            tuple(0 for _ in range(total_shape_types)),
            params.partition_num,
            total_shape_types,
        )
        < params._entropy_floor
    ):
        _PROFILE_RULE_CACHE[cache_key] = None
        return None

    @lru_cache(maxsize=None)
    def search(
        rows_left: int,
        remaining_counts: tuple[int, ...],
        usage: tuple[int, ...],
    ) -> Optional[tuple[tuple[int, ...], ...]]:
        if rows_left == 0:
            if any(remaining_counts):
                return None
            if _profile_support_bits_from_usage(usage, params.partition_num) < params._entropy_floor:
                return None
            return ()

        if not any(remaining_counts):
            if not empty_available:
                return None
            final_usage = list(usage)
            final_usage[index_by_row[empty_row]] += rows_left
            if _profile_support_bits_from_usage(final_usage, params.partition_num) < params._entropy_floor:
                return None
            return (empty_row,) * rows_left

        rows_left_after = rows_left - 1
        must_cover = tuple(
            value
            for value, count in enumerate(remaining_counts)
            if count > rows_left_after
        )
        for row, row_mask in zip(ordered_candidates, ordered_candidate_masks):
            next_counts = _shape_feasible_with_remaining(
                row,
                row_mask,
                remaining_counts,
                rows_left_after,
                must_cover,
            )
            if next_counts is None:
                continue
            next_usage = list(usage)
            next_usage[index_by_row[row]] += 1
            if (
                _optimistic_profile_support_bits(
                    next_usage,
                    params.partition_num,
                    total_shape_types,
                )
                < params._entropy_floor
            ):
                continue
            suffix = search(rows_left_after, next_counts, tuple(next_usage))
            if suffix is not None:
                return (row,) + suffix

        if empty_available:
            next_usage = list(usage)
            next_usage[index_by_row[empty_row]] += 1
            if (
                _optimistic_profile_support_bits(
                    next_usage,
                    params.partition_num,
                    total_shape_types,
                )
                >= params._entropy_floor
            ):
                suffix = search(rows_left_after, remaining_counts, tuple(next_usage))
                if suffix is not None:
                    return (empty_row,) + suffix
        return None

    result = search(
        params.partition_num,
        tuple(int(count) for count in counts),
        tuple(0 for _ in family),
    )
    _PROFILE_RULE_CACHE[cache_key] = result
    return result


def _profile_route_plan(
    counts: Sequence[int],
    params: TreeAwareISPParameters,
) -> Optional[_ProfileRoutePlan]:
    cache_key = _profile_rule_cache_key(counts, params)
    base_key = _profile_rule_base_cache_key(counts, params)
    cached = _PROFILE_ROUTE_PLAN_CACHE.get(cache_key)
    if cached is not None or cache_key in _PROFILE_ROUTE_PLAN_CACHE:
        return cached

    cached_success = _PROFILE_ROUTE_PLAN_SUCCESS_CACHE.get(base_key)
    if cached_success is not None and cached_success.support_bits >= params._entropy_floor:
        _PROFILE_ROUTE_PLAN_CACHE[cache_key] = cached_success
        return cached_success
    fail_floor = _PROFILE_ROUTE_PLAN_FAIL_FLOOR_CACHE.get(base_key)
    if fail_floor is not None and params._entropy_floor >= fail_floor:
        _PROFILE_ROUTE_PLAN_CACHE[cache_key] = None
        return None

    profile = _profile_rule_profile(counts, params)
    if profile is None:
        existing_fail_floor = _PROFILE_ROUTE_PLAN_FAIL_FLOOR_CACHE.get(base_key)
        if existing_fail_floor is None or params._entropy_floor < existing_fail_floor:
            _PROFILE_ROUTE_PLAN_FAIL_FLOOR_CACHE[base_key] = params._entropy_floor
        _PROFILE_ROUTE_PLAN_CACHE[cache_key] = None
        return None

    dy_shape_family_set = params._dy_shape_family_set
    if (
        dy_shape_family_set is not None
        and params._profile_rule_name not in {"sparse_residual", "residual_fallback"}
    ):
        for row in profile:
            if row not in dy_shape_family_set:
                _PROFILE_ROUTE_PLAN_CACHE[cache_key] = None
                return None
    if not _profile_matches_counts(profile, counts, params.max_g_value):
        _PROFILE_ROUTE_PLAN_CACHE[cache_key] = None
        return None

    support = _multiset_permutation_count(profile)
    if support <= 0:
        _PROFILE_ROUTE_PLAN_CACHE[cache_key] = None
        return None
    support_bits = _profile_support_bits(profile)
    if support_bits < params._entropy_floor:
        _PROFILE_ROUTE_PLAN_CACHE[cache_key] = None
        return None

    ordered_rows = _profile_canonical_row_order(profile, params)
    multiplicities: dict[tuple[int, ...], int] = {}
    for row in profile:
        multiplicities[row] = multiplicities.get(row, 0) + 1
    plan = _ProfileRoutePlan(
        profile=profile,
        support=support,
        support_bits=support_bits,
        ordered_rows=ordered_rows,
        ordered_row_counts=tuple(multiplicities[row] for row in ordered_rows),
    )
    cached_success = _PROFILE_ROUTE_PLAN_SUCCESS_CACHE.get(base_key)
    if cached_success is None or plan.support_bits > cached_success.support_bits:
        _PROFILE_ROUTE_PLAN_SUCCESS_CACHE[base_key] = plan
    _PROFILE_ROUTE_PLAN_CACHE[cache_key] = plan
    return plan


def _size_mode_profile(
    counts: Sequence[int],
    params: TreeAwareISPParameters,
) -> Optional[tuple[tuple[int, ...], ...]]:
    plan = _profile_route_plan(counts, params)
    return None if plan is None else plan.profile


@lru_cache(maxsize=None)
def _multiset_permutation_count_from_usage(
    usage: tuple[int, ...],
    total_items: int,
) -> int:
    total = math.factorial(total_items)
    for count in usage:
        total //= math.factorial(count)
    return total


def _multiset_permutation_count(items: Sequence[tuple[int, ...]]) -> int:
    multiplicities: dict[tuple[int, ...], int] = {}
    for item in items:
        multiplicities[item] = multiplicities.get(item, 0) + 1
    return _multiset_permutation_count_from_usage(
        tuple(sorted(multiplicities.values())),
        len(items),
    )


def _profile_support_bits(profile: Sequence[tuple[int, ...]]) -> float:
    multiplicities: dict[tuple[int, ...], int] = {}
    for row in profile:
        multiplicities[row] = multiplicities.get(row, 0) + 1
    return _profile_support_bits_from_usage(tuple(multiplicities.values()), len(profile))


def _unrank_shape_profile_permutation(
    profile: Sequence[tuple[int, ...]],
    rank: int,
    ordered_rows: Optional[Sequence[tuple[int, ...]]] = None,
) -> Groups:
    remaining: dict[tuple[int, ...], int] = {}
    for row in profile:
        remaining[row] = remaining.get(row, 0) + 1

    if ordered_rows is None:
        ordered_rows = tuple(sorted(remaining))
    remaining_total = len(profile)
    support = _multiset_permutation_count(profile)
    output: Groups = []
    for _ in range(len(profile)):
        for row in ordered_rows:
            count = remaining.get(row, 0)
            if count == 0:
                continue
            branch_count = (support * count) // remaining_total
            if rank < branch_count:
                remaining[row] = count - 1
                output.append(list(row))
                support = branch_count
                remaining_total -= 1
                break
            rank -= branch_count
        else:
            raise ValueError("rank exceeds the multiset-permutation support")
    return output


def _unrank_profile_route_plan(
    plan: _ProfileRoutePlan,
    rank: int,
) -> Groups:
    remaining_counts = list(plan.ordered_row_counts)
    remaining_total = len(plan.profile)
    support = plan.support
    output: Groups = []
    for _ in range(remaining_total):
        for index, row in enumerate(plan.ordered_rows):
            count = remaining_counts[index]
            if count == 0:
                continue
            branch_count = (support * count) // remaining_total
            if rank < branch_count:
                remaining_counts[index] = count - 1
                output.append(list(row))
                support = branch_count
                remaining_total -= 1
                break
            rank -= branch_count
        else:
            raise ValueError("rank exceeds the multiset-permutation support")
    return output


def _profile_canonical_row_order(
    profile: Sequence[tuple[int, ...]],
    params: TreeAwareISPParameters,
) -> tuple[tuple[int, ...], ...]:
    family = params._dy_shape_family
    present = set(profile)
    if family is None:
        return tuple(sorted(present))
    ordered = [row for row in family if row in present]
    if len(ordered) != len(present):
        ordered.extend(sorted(row for row in present if row not in set(ordered)))
    return tuple(ordered)


def _route_full_support(
    partition_value: PartitionValueInput,
    counts: Sequence[int],
    params: TreeAwareISPParameters,
    rng: Optional[Random],
    xof_seed_material: Optional[bytes],
) -> Groups:
    return _ind_route(
        partition_value=partition_value,
        counts=counts,
        residual_vec=counts,
        residual_rows=params.partition_num,
        params=params,
        rng=rng,
        xof_seed_material=xof_seed_material,
    )


def _route_size(
    partition_value: PartitionValueInput,
    counts: Sequence[int],
    params: TreeAwareISPParameters,
    rng: Optional[Random],
    xof_seed_material: Optional[bytes],
) -> Optional[Groups]:
    plan = _profile_route_plan(counts, params)
    if plan is None:
        return None
    support = plan.support
    if support == 0:
        return None
    if rng is not None:
        rank = rng.randrange(support)
    else:
        seed_material = _route_seed_material(
            b"TISP.prof",
            partition_value,
            counts,
            params,
            xof_seed_material,
        )
        rank = HashXOF(seed_material, params.hash_name).randbelow(support)
    return _unrank_profile_route_plan(plan, rank)


def _support_product(rows: int, counts: Sequence[int]) -> int:
    if rows < 0 or any(count < 0 or count > rows for count in counts):
        return 0
    binomial_table = _binomial_table(rows)
    support = 1
    for count in counts:
        support *= binomial_table[rows][count]
    return support


def tree_extract(
    counts: Sequence[int],
    params: TreeAwareISPParameters,
) -> Optional[tuple[Groups, tuple[int, ...], int]]:
    groups_pre: Groups = []
    residual_vec = tuple(int(count) for count in counts)
    residual_rows = params.partition_num
    initial_support = _support_product(params.partition_num, residual_vec)
    if initial_support == 0:
        return None

    loss_multiplier = 1 << params._loss_bound
    while len(groups_pre) < params._prefix_limit:
        added = False
        for pattern, mask in zip(params._prefix_dict, params._prefix_masks):
            next_rows = residual_rows - 1
            if next_rows < 0:
                continue
            next_vec = tuple(
                residual_vec[value] - ((mask >> value) & 1)
                for value in range(params._max_g_value)
            )
            if any(value < 0 or value > next_rows for value in next_vec):
                continue
            candidate_support = _support_product(next_rows, next_vec)
            if initial_support <= loss_multiplier * candidate_support:
                groups_pre.append(list(pattern))
                residual_vec = next_vec
                residual_rows = next_rows
                added = True
                break
        if not added:
            break

    if any(value > residual_rows for value in residual_vec):
        return None
    return groups_pre, residual_vec, residual_rows


def _residual_block_lengths(residual_rows: int, bt_block_size: int) -> tuple[int, ...]:
    if residual_rows < 0:
        raise ValueError("residual_rows must be non-negative")
    if residual_rows == 0:
        return ()
    if bt_block_size <= 0:
        return ()
    lengths = []
    remaining = residual_rows
    while remaining > 0:
        block_len = min(bt_block_size, remaining)
        lengths.append(block_len)
        remaining -= block_len
    return tuple(lengths)


def _bt_count_function(
    block_lengths: Sequence[int],
    params: TreeAwareISPParameters,
):
    cache: dict[tuple[int, tuple[int, ...]], int] = {}

    def count_from(block_index: int, residual_vec: tuple[int, ...]) -> int:
        cache_key = (block_index, residual_vec)
        cached = cache.get(cache_key)
        if cached is not None:
            return cached
        if block_index == len(block_lengths):
            result = 1 if not any(residual_vec) else 0
            cache[cache_key] = result
            return result

        block_len = block_lengths[block_index]
        total = 0
        for template in params._bt_families[block_len]:
            if any(used > available for used, available in zip(template.counts, residual_vec)):
                continue
            next_vec = tuple(
                available - used
                for available, used in zip(residual_vec, template.counts)
            )
            suffix_count = count_from(block_index + 1, next_vec)
            if suffix_count:
                total += len(template.realizations) * suffix_count
        cache[cache_key] = total
        return total

    return count_from


def _route_mode_and_support(
    residual_vec: Sequence[int],
    residual_rows: int,
    params: TreeAwareISPParameters,
) -> tuple[str, int, int, int, tuple[int, ...]]:
    ind_support = _support_product(residual_rows, residual_vec)
    block_lengths = _residual_block_lengths(residual_rows, params._bt_block_size)
    bt_support = 0
    if block_lengths or residual_rows == 0:
        bt_support = _bt_count_function(block_lengths, params)(0, tuple(residual_vec))
    if (
        bt_support > 0
        and ind_support <= (1 << params._bt_loss_bound) * bt_support
    ):
        return "bt", bt_support, ind_support, bt_support, block_lengths
    return "ind", ind_support, ind_support, bt_support, block_lengths


def route_support(
    counts: Sequence[int],
    params: TreeAwareISPParameters,
) -> int:
    if params._routing_strategy == _ROUTE_POLICY_PROFILE:
        plan = _profile_route_plan(counts, params)
        return 0 if plan is None else plan.support
    if params._routing_strategy == _ROUTE_POLICY_FULL_SUPPORT:
        return _support_product(params.partition_num, counts)
    extracted = tree_extract(counts, params)
    if extracted is None:
        return 0
    _groups_pre, residual_vec, residual_rows = extracted
    _mode, support, _ind_support, _bt_support, _block_lengths = _route_mode_and_support(
        residual_vec,
        residual_rows,
        params,
    )
    return support


def _ind_route(
    partition_value: PartitionValueInput,
    counts: Sequence[int],
    residual_vec: Sequence[int],
    residual_rows: int,
    params: TreeAwareISPParameters,
    rng: Optional[Random],
    xof_seed_material: Optional[bytes],
) -> Groups:
    groups: Groups = [[] for _ in range(residual_rows)]
    sample_base_params = (
        params.sample_base_params
        if residual_rows == params.partition_num
        else _sample_base_parameters(residual_rows)
    )
    binomial_table, subset_count_row, _byte_lengths, _thresholds, direct_unrank_rows, _unpackers = (
        sample_base_params
    )
    subset_decode_tables = _subset_decode_tables(residual_rows)
    counting = counters_enabled()
    for value, raw_count in enumerate(residual_vec):
        count = int(raw_count)
        subset_count = subset_count_row[count]
        if rng is not None:
            rank = rng.randrange(subset_count)
        else:
            seed_material = _route_seed_material(
                b"ind",
                partition_value,
                counts,
                params,
                xof_seed_material,
                value=value,
            )
            if counting:
                rank = HashXOF(seed_material, params.hash_name).randbelow(subset_count)
            else:
                rank = _randbelow_from_seed_material_once(
                    seed_material,
                    params.hash_name,
                    subset_count,
                )
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
                residual_rows,
                count,
                binomial_table,
            )
            continue
        if residual_rows - count <= _SMALL_SUBSET_UNRANK_THRESHOLD:
            _append_value_excluding_small_subset(
                groups,
                value,
                rank,
                residual_rows,
                count,
                subset_count,
                binomial_table,
            )
            continue

        remaining = count
        current_rank = rank
        include_row = direct_unrank_rows[remaining]
        for position in range(residual_rows):
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


def _bt_route(
    partition_value: PartitionValueInput,
    counts: Sequence[int],
    residual_vec: Sequence[int],
    residual_rows: int,
    block_lengths: Sequence[int],
    params: TreeAwareISPParameters,
    rng: Optional[Random],
    xof_seed_material: Optional[bytes],
) -> Optional[Groups]:
    count_from = _bt_count_function(block_lengths, params)
    support = count_from(0, tuple(residual_vec))
    if support == 0:
        return None
    if rng is not None:
        rank = rng.randrange(support)
    else:
        seed_material = _route_seed_material(
            b"bt",
            partition_value,
            counts,
            params,
            xof_seed_material,
        )
        rank = HashXOF(seed_material, params.hash_name).randbelow(support)

    current_vec = tuple(residual_vec)
    groups_res: Groups = []
    for block_index, block_len in enumerate(block_lengths):
        chosen = False
        for template in params._bt_families[block_len]:
            if any(used > available for used, available in zip(template.counts, current_vec)):
                continue
            next_vec = tuple(
                available - used
                for available, used in zip(current_vec, template.counts)
            )
            suffix_count = count_from(block_index + 1, next_vec)
            branch_count = len(template.realizations) * suffix_count
            if branch_count == 0:
                continue
            if rank < branch_count:
                realization_index, rank = divmod(rank, suffix_count)
                groups_res.extend(
                    [list(row) for row in template.realizations[realization_index]]
                )
                current_vec = next_vec
                chosen = True
                break
            rank -= branch_count
        if not chosen:
            return None
    if len(groups_res) != residual_rows or any(current_vec):
        return None
    return groups_res


def _default_laminar_pattern_masks(max_g_value: int) -> tuple[int, ...]:
    cached = _DEFAULT_LAMINAR_PATTERN_MASKS.get(max_g_value)
    if cached is not None:
        return cached
    cached = tuple(_pattern_to_mask(pattern) for pattern in _default_pattern_family(max_g_value))
    _DEFAULT_LAMINAR_PATTERN_MASKS[max_g_value] = cached
    return cached


def _uses_default_laminar_pattern_family(params: TreeAwareISPParameters) -> bool:
    pattern_masks = params._pattern_masks
    if pattern_masks is None:
        return False
    default_masks = _default_laminar_pattern_masks(params._max_g_value)
    return pattern_masks == default_masks


def _laminar_nonempty_count(counts: tuple[int, ...], nonempty_rows: int) -> int:
    """
    Count ordered rows over dyadic interval patterns without the empty row.

    The default tree-aware pattern family is laminar: every non-empty pattern
    is a node in the complete dyadic tree over the alphabet values. Counting
    non-empty rows on that tree, then choosing which outer rows are empty,
    avoids the very large row-by-row DP used for arbitrary pattern families.
    """

    if nonempty_rows < 0 or any(count < 0 for count in counts):
        return 0
    nonempty_counts = _laminar_nonempty_count_row(counts, nonempty_rows)
    if nonempty_rows >= len(nonempty_counts):
        return 0
    return nonempty_counts[nonempty_rows]


def _laminar_nonempty_count_row(
    counts: tuple[int, ...],
    max_rows: Optional[int] = None,
) -> tuple[int, ...]:
    if any(count < 0 for count in counts):
        return ()

    total_count = sum(counts)
    max_rows = total_count if max_rows is None else min(max_rows, total_count)
    if max_rows < 0:
        return ()

    cache_key = (max_rows, counts)
    cached = _LAMINAR_NONEMPTY_ROWS_CACHE.get(cache_key)
    if cached is not None:
        return cached

    row = [0] * (max_rows + 1)
    if total_count == 0:
        row[0] = 1
        cached_row = tuple(row)
        _LAMINAR_NONEMPTY_ROWS_CACHE[cache_key] = cached_row
        return cached_row

    if max_rows < max(counts):
        cached_row = tuple(row)
        _LAMINAR_NONEMPTY_ROWS_CACHE[cache_key] = cached_row
        return cached_row

    if len(counts) == 1:
        if counts[0] <= max_rows:
            row[counts[0]] = 1
        cached_row = tuple(row)
        _LAMINAR_NONEMPTY_ROWS_CACHE[cache_key] = cached_row
        return cached_row

    if len(counts) == 2:
        left_count, right_count = counts
        binomial_table = _binomial_table(max_rows)
        for pair_rows in range(min(left_count, right_count), -1, -1):
            nonempty_rows = left_count + right_count - pair_rows
            if nonempty_rows > max_rows:
                continue
            row[nonempty_rows] = (
                binomial_table[nonempty_rows][pair_rows]
                * binomial_table[nonempty_rows - pair_rows][left_count - pair_rows]
            )
        cached_row = tuple(row)
        _LAMINAR_NONEMPTY_ROWS_CACHE[cache_key] = cached_row
        return cached_row

    half = len(counts) // 2
    left_counts_base = counts[:half]
    right_counts_base = counts[half:]
    binomial_table = _binomial_table(max_rows)

    for node_rows in range(min(counts), -1, -1):
        child_max_rows = max_rows - node_rows
        if child_max_rows < 0:
            continue
        left_counts = tuple(count - node_rows for count in left_counts_base)
        right_counts = tuple(count - node_rows for count in right_counts_base)
        left_items = _laminar_nonempty_row_items(left_counts, child_max_rows)
        right_items = _laminar_nonempty_row_items(right_counts, child_max_rows)
        if not left_items or not right_items:
            continue

        for left_rows, left_count in left_items:
            right_limit = child_max_rows - left_rows
            if right_limit < 0:
                continue
            for right_rows, right_count in right_items:
                if right_rows > right_limit:
                    break
                nonempty_rows = node_rows + left_rows + right_rows
                row[nonempty_rows] += (
                    binomial_table[nonempty_rows][node_rows]
                    * binomial_table[nonempty_rows - node_rows][left_rows]
                    * left_count
                    * right_count
                )

    cached_row = tuple(row)
    _LAMINAR_NONEMPTY_ROWS_CACHE[cache_key] = cached_row
    return cached_row


def _laminar_nonempty_row_items(
    counts: tuple[int, ...],
    max_rows: int,
) -> tuple[tuple[int, int], ...]:
    max_rows = min(max_rows, sum(counts))
    if max_rows < 0:
        return ()
    cache_key = (max_rows, counts)
    cached = _LAMINAR_NONEMPTY_ITEMS_CACHE.get(cache_key)
    if cached is not None:
        return cached

    row = _laminar_nonempty_count_row(counts, max_rows)
    items = tuple((index, value) for index, value in enumerate(row) if value)
    _LAMINAR_NONEMPTY_ITEMS_CACHE[cache_key] = items
    return items


def _laminar_count_with_empty(counts: tuple[int, ...], rows_left: int) -> int:
    if rows_left < 0 or any(count < 0 or count > rows_left for count in counts):
        return 0

    cache_key = (rows_left, counts)
    cached = _LAMINAR_WITH_EMPTY_COUNT_CACHE.get(cache_key)
    if cached is not None:
        return cached

    if not counts:
        result = 1 if rows_left == 0 else 0
        _LAMINAR_WITH_EMPTY_COUNT_CACHE[cache_key] = result
        return result

    nonempty_counts = _laminar_nonempty_count_row(counts, rows_left)
    if not nonempty_counts:
        _LAMINAR_WITH_EMPTY_COUNT_CACHE[cache_key] = 0
        return 0

    nonempty_low = max(counts)
    nonempty_high = min(rows_left, len(nonempty_counts) - 1)
    binomial_table = _binomial_table(rows_left)
    result = 0
    for nonempty_rows in range(nonempty_low, nonempty_high + 1):
        inner_count = nonempty_counts[nonempty_rows]
        if inner_count:
            result += binomial_table[rows_left][nonempty_rows] * inner_count

    _LAMINAR_WITH_EMPTY_COUNT_CACHE[cache_key] = result
    return result


def _unrank_combination_positions(
    universe_size: int,
    subset_size: int,
    rank: int,
) -> tuple[int, ...]:
    if subset_size == 0:
        return ()
    return _decode_subset_rank_positions(
        rank,
        universe_size,
        subset_size,
        _binomial_table(universe_size),
    )


def _remaining_positions(
    universe_size: int,
    selected_positions: tuple[int, ...],
) -> tuple[int, ...]:
    if not selected_positions:
        return tuple(range(universe_size))

    selected_index = 0
    selected_count = len(selected_positions)
    remaining = []
    for position in range(universe_size):
        if selected_index < selected_count and position == selected_positions[selected_index]:
            selected_index += 1
        else:
            remaining.append(position)
    return tuple(remaining)


def _decode_two_leaf_laminar_masks(
    counts: tuple[int, int],
    nonempty_rows: int,
    rank: int,
    value_offset: int,
) -> tuple[int, ...]:
    left_count, right_count = counts
    binomial_table = _binomial_table(nonempty_rows)
    left_mask = 1 << value_offset
    right_mask = 1 << (value_offset + 1)
    pair_mask = left_mask | right_mask

    for pair_rows in range(min(left_count, right_count), -1, -1):
        if left_count + right_count - pair_rows != nonempty_rows:
            continue
        left_only_rows = left_count - pair_rows
        pair_position_count = binomial_table[nonempty_rows][pair_rows]
        left_position_count = binomial_table[nonempty_rows - pair_rows][left_only_rows]
        branch_count = pair_position_count * left_position_count
        if rank >= branch_count:
            rank -= branch_count
            continue

        pair_rank, remaining_rank = divmod(rank, left_position_count)
        pair_positions = _unrank_combination_positions(
            nonempty_rows,
            pair_rows,
            pair_rank,
        )
        remaining_positions = _remaining_positions(nonempty_rows, pair_positions)
        left_relative_positions = _unrank_combination_positions(
            nonempty_rows - pair_rows,
            left_only_rows,
            remaining_rank,
        )
        left_positions = tuple(remaining_positions[index] for index in left_relative_positions)
        result = [right_mask] * nonempty_rows
        for position in pair_positions:
            result[position] = pair_mask
        for position in left_positions:
            result[position] = left_mask
        return tuple(result)

    return ()


def _decode_laminar_nonempty_masks(
    counts: tuple[int, ...],
    nonempty_rows: int,
    rank: int,
    value_offset: int = 0,
) -> tuple[int, ...]:
    row = _laminar_nonempty_count_row(counts, nonempty_rows)
    if nonempty_rows >= len(row) or rank < 0 or rank >= row[nonempty_rows]:
        return ()

    if nonempty_rows == 0:
        return () if not any(counts) else ()
    if len(counts) == 1:
        if counts[0] != nonempty_rows:
            return ()
        return (1 << value_offset,) * nonempty_rows
    if len(counts) == 2:
        return _decode_two_leaf_laminar_masks(
            (counts[0], counts[1]),
            nonempty_rows,
            rank,
            value_offset,
        )

    half = len(counts) // 2
    left_counts_base = counts[:half]
    right_counts_base = counts[half:]
    node_mask = ((1 << len(counts)) - 1) << value_offset
    binomial_table = _binomial_table(nonempty_rows)

    for node_rows in range(min(counts), -1, -1):
        child_max_rows = nonempty_rows - node_rows
        if child_max_rows < 0:
            continue
        left_counts = tuple(count - node_rows for count in left_counts_base)
        right_counts = tuple(count - node_rows for count in right_counts_base)
        left_items = _laminar_nonempty_row_items(left_counts, child_max_rows)
        right_items = _laminar_nonempty_row_items(right_counts, child_max_rows)
        if not left_items or not right_items:
            continue
        right_counts_by_rows = dict(right_items)

        for left_rows, left_count in left_items:
            right_rows = child_max_rows - left_rows
            right_count = right_counts_by_rows.get(right_rows, 0)
            if right_count == 0:
                continue

            node_position_count = binomial_table[nonempty_rows][node_rows]
            left_position_count = binomial_table[child_max_rows][left_rows]
            branch_count = node_position_count * left_position_count * left_count * right_count
            if rank >= branch_count:
                rank -= branch_count
                continue

            per_node_positions = left_position_count * left_count * right_count
            node_rank, rank = divmod(rank, per_node_positions)
            per_left_positions = left_count * right_count
            left_position_rank, rank = divmod(rank, per_left_positions)
            left_rank, right_rank = divmod(rank, right_count)

            node_positions = _unrank_combination_positions(
                nonempty_rows,
                node_rows,
                node_rank,
            )
            child_positions = _remaining_positions(nonempty_rows, node_positions)
            left_relative_positions = _unrank_combination_positions(
                child_max_rows,
                left_rows,
                left_position_rank,
            )
            left_positions = tuple(child_positions[index] for index in left_relative_positions)
            left_relative_set = set(left_relative_positions)
            right_positions = tuple(
                position
                for index, position in enumerate(child_positions)
                if index not in left_relative_set
            )

            left_masks = _decode_laminar_nonempty_masks(
                left_counts,
                left_rows,
                left_rank,
                value_offset,
            )
            right_masks = _decode_laminar_nonempty_masks(
                right_counts,
                right_rows,
                right_rank,
                value_offset + half,
            )
            if len(left_masks) != left_rows or len(right_masks) != right_rows:
                return ()

            result = [0] * nonempty_rows
            for position in node_positions:
                result[position] = node_mask
            for position, mask in zip(left_positions, left_masks):
                result[position] = mask
            for position, mask in zip(right_positions, right_masks):
                result[position] = mask
            return tuple(result)

    return ()


def _tree_sampler_laminar(
    partition_value: PartitionValueInput,
    counts: Sequence[int],
    params: TreeAwareISPParameters,
    rng: Optional[Random],
    xof_seed_material: Optional[bytes],
) -> Optional[Groups]:
    pattern_family = params._pattern_family
    pattern_masks = params._pattern_masks
    if pattern_family is None or pattern_masks is None:
        return None

    rem_vec = tuple(int(count) for count in counts)
    total_count = _laminar_count_with_empty(rem_vec, params.partition_num)
    if total_count == 0:
        return None

    if rng is not None:
        tree_rank = rng.randrange(total_count)
    else:
        seed_material = _tree_sampler_seed_material(
            partition_value=partition_value,
            counts=counts,
            params=params,
            xof_seed_material=xof_seed_material,
        )
        tree_rank = HashXOF(seed_material, params.hash_name).randbelow(total_count)

    nonempty_counts = _laminar_nonempty_count_row(rem_vec, params.partition_num)
    if not nonempty_counts:
        return None
    nonempty_low = max(rem_vec)
    nonempty_high = min(params.partition_num, len(nonempty_counts) - 1)
    binomial_table = _binomial_table(params.partition_num)
    for nonempty_rows in range(nonempty_low, nonempty_high + 1):
        inner_count = nonempty_counts[nonempty_rows]
        if inner_count == 0:
            continue
        position_count = binomial_table[params.partition_num][nonempty_rows]
        branch_count = position_count * inner_count
        if tree_rank >= branch_count:
            tree_rank -= branch_count
            continue

        position_rank, sequence_rank = divmod(tree_rank, inner_count)
        nonempty_positions = _unrank_combination_positions(
            params.partition_num,
            nonempty_rows,
            position_rank,
        )
        nonempty_masks = _decode_laminar_nonempty_masks(
            rem_vec,
            nonempty_rows,
            sequence_rank,
        )
        if len(nonempty_masks) != nonempty_rows:
            return None

        groups: Groups = [[] for _ in range(params.partition_num)]
        for position, mask in zip(nonempty_positions, nonempty_masks):
            groups[position] = list(_sorted_values_from_mask(mask))
        return groups

    return None


def tree_sampler(
    partition_value: PartitionValueInput,
    counts: Sequence[int],
    params: TreeAwareISPParameters,
    rng: Optional[Random] = None,
    xof_seed_material: Optional[bytes] = None,
) -> Optional[Groups]:
    """
    Tree-aware sampler with profile routing and optional public score guards.
    """

    if params._routing_strategy == _ROUTE_POLICY_PROFILE:
        groups = _route_size(
            partition_value,
            counts,
            params,
            rng,
            xof_seed_material,
        )
        if groups is None:
            return None
        if params._score_guard_enabled and not score_guard(groups, params):
            return None
        return groups
    if params._routing_strategy == _ROUTE_POLICY_FULL_SUPPORT:
        groups = _route_full_support(
            partition_value,
            counts,
            params,
            rng,
            xof_seed_material,
        )
        if params._score_guard_enabled and not score_guard(groups, params):
            return None
        return groups

    extracted = tree_extract(counts, params)
    if extracted is None:
        return None
    groups_pre, residual_vec, residual_rows = extracted
    route_mode, _support, _ind_support, _bt_support, block_lengths = _route_mode_and_support(
        residual_vec,
        residual_rows,
        params,
    )
    if route_mode == "bt":
        groups_res = _bt_route(
            partition_value,
            counts,
            residual_vec,
            residual_rows,
            block_lengths,
            params,
            rng,
            xof_seed_material,
        )
        if groups_res is None:
            return None
    else:
        groups_res = _ind_route(
            partition_value,
            counts,
            residual_vec,
            residual_rows,
            params,
            rng,
            xof_seed_material,
        )

    groups = groups_pre + groups_res
    if params._score_guard_enabled and not score_guard(groups, params):
        return None
    return groups


def treeaware_isp(
    partition_value: PartitionValueInput,
    params: TreeAwareISPParameters,
    rng: Optional[Random] = None,
    xof_seed_material: Optional[bytes] = None,
) -> Optional[Groups]:
    """
    Python implementation of the tree-aware TreeAwareISP algorithm.

    Returns:
        A list of partition_num strictly increasing subsequences, or None if
        the multiplicity profile or tree-aware legal family is rejected.
    """

    hash_len = params.hash_len
    max_g_bit = params.max_g_bit
    max_g_value = params._max_g_value
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

    return tree_sampler(
        partition_value=partition_value,
        counts=counts,
        params=params,
        rng=rng,
        xof_seed_material=xof_seed_material,
    )


def route_size(
    partition_value: PartitionValueInput,
    params: TreeAwareISPParameters,
    rng: Optional[Random] = None,
    xof_seed_material: Optional[bytes] = None,
) -> Optional[Groups]:
    """
    Public helper for the routing layer RouteSize in the paper.

    This function performs the histogram window checks and routing step, but it
    does not apply the tree-aware abort guard. The full ValStrictISPT sampler is
    `treeaware_isp(...)`, which additionally applies `score_guard_t(...)`.
    """

    hash_len = params.hash_len
    max_g_bit = params.max_g_bit
    max_g_value = params.max_g_value
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
    return tree_sampler(
        partition_value=partition_value,
        counts=counts,
        params=replace(
            params,
            size_threshold=None,
            vrf_threshold=None,
            score_bound=None,
        ),
        rng=rng,
        xof_seed_material=xof_seed_material,
    )


def is_strictly_increasing(values: Sequence[int]) -> bool:
    return all(left < right for left, right in zip(values, values[1:]))


def verify_output(groups: Sequence[Sequence[int]]) -> bool:
    return all(is_strictly_increasing(group) for group in groups)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the TreeAwareISP sampler.")
    parser.add_argument(
        "input_value",
        help="partition value (bitstring/int/hex) or message, depending on --input-mode",
    )
    parser.add_argument("--hash-len", type=int, required=True, help="HashLen")
    parser.add_argument("--max-g-bit", type=int, required=True, help="MaxGBit")
    parser.add_argument("--partition-num", type=int, required=True, help="PartitionNum")
    parser.add_argument(
        "--aux-t",
        default=None,
        help="JSON object describing AuxT, e.g. DyShapeFamily, shape_profile(s), and EntropyFloor.",
    )
    parser.add_argument(
        "--size-threshold",
        default=None,
        help="Optional public SizeScore threshold; use 'inf' to disable the size guard.",
    )
    parser.add_argument(
        "--vrf-threshold",
        default=None,
        help="Optional public verification-score threshold; use 'inf' to disable the verify guard.",
    )
    parser.add_argument(
        "--mode",
        choices=(_VISPT_MODE_LEGACY, _VISPT_MODE_SIZE, _VISPT_MODE_VRF),
        default=_VISPT_MODE_LEGACY,
        help="Compatibility shorthand for the route objective and score semantics; prefer --route-policy with --aux-t in new ValStrictISPT configurations.",
    )
    parser.add_argument(
        "--route-policy",
        default=None,
        help=(
            "Optional routing policy override: 'profile', 'ProfileMode', "
            "'full_support', or 'FullSupportMode'. Omit to infer the policy "
            "from the public ValStrictISPT parameters."
        ),
    )
    parser.add_argument(
        "--aux-mode",
        default=None,
        help="Deprecated alias for --aux-t.",
    )
    parser.add_argument(
        "--score-name",
        choices=(_SCORE_NAME_SIZE, _SCORE_NAME_VRF),
        default=None,
        help="Deprecated single-score guard selector.",
    )
    parser.add_argument(
        "--score-bound",
        default=None,
        help="Deprecated single-score guard threshold; use 'inf' to disable.",
    )
    parser.add_argument(
        "--window-radius",
        type=int,
        default=None,
        help="Symmetric multiplicity window radius; used for both sides unless overridden",
    )
    parser.add_argument(
        "--window-radius-l",
        type=int,
        default=None,
        help="Lower-side asymmetric multiplicity window radius",
    )
    parser.add_argument(
        "--window-radius-u",
        type=int,
        default=None,
        help="Upper-side asymmetric multiplicity window radius",
    )
    parser.add_argument(
        "--link-threshold",
        type=int,
        default=-1,
        help="Deprecated compatibility option; ignored by the simplified-windowed VISP",
    )
    parser.add_argument(
        "--tree-threshold",
        type=int,
        default=None,
        help="Deprecated compatibility option; TreeScore is no longer constrained by TreeAwareISP",
    )
    parser.add_argument(
        "--shape-parms",
        default=None,
        help="Deprecated compatibility option; ignored by the ValStrictISPT sampler",
    )
    parser.add_argument(
        "--prefix-dict",
        default=None,
        help="JSON list of nonempty prefix row patterns; omitted uses pattern-family or dyadic defaults",
    )
    parser.add_argument(
        "--loss-bound",
        type=int,
        default=0,
        help="Entropy-loss budget for deterministic prefix extraction",
    )
    parser.add_argument(
        "--prefix-limit",
        type=int,
        default=0,
        help="Maximum number of deterministic prefix rows to extract",
    )
    parser.add_argument(
        "--bt-block-size",
        type=int,
        default=0,
        help="Residual block length for block-template routing; 0 disables BT unless families imply a size",
    )
    parser.add_argument(
        "--bt-families",
        default=None,
        help="JSON block-template families keyed by block length",
    )
    parser.add_argument(
        "--bt-loss-bound",
        type=int,
        default=0,
        help="Entropy-loss budget for block-template residual routing",
    )
    parser.add_argument(
        "--pattern-family",
        default=None,
        help="JSON list of row patterns, e.g. '[[],[0],[1],[0,1]]'; omitted uses aligned intervals",
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


def _parse_score_bound_arg(raw: Optional[str]) -> Optional[int | float]:
    if raw is None:
        return None
    normalized = raw.strip().lower()
    if normalized in {"inf", "infinity"}:
        return float("inf")
    return int(raw)


def _main() -> int:
    args = _build_parser().parse_args()
    aux_t = json.loads(args.aux_t) if args.aux_t is not None else None
    aux_mode = json.loads(args.aux_mode) if args.aux_mode is not None else None
    pattern_family = json.loads(args.pattern_family) if args.pattern_family is not None else None
    shape_parms = json.loads(args.shape_parms) if args.shape_parms is not None else None
    prefix_dict = json.loads(args.prefix_dict) if args.prefix_dict is not None else None
    bt_families = json.loads(args.bt_families) if args.bt_families is not None else None
    params = TreeAwareISPParameters(
        hash_len=args.hash_len,
        max_g_bit=args.max_g_bit,
        partition_num=args.partition_num,
        aux_t=aux_t,
        route_policy=args.route_policy,
        size_threshold=_parse_score_bound_arg(args.size_threshold),
        vrf_threshold=_parse_score_bound_arg(args.vrf_threshold),
        mode=args.mode,
        aux_mode=aux_mode,
        score_name=args.score_name,
        score_bound=_parse_score_bound_arg(args.score_bound),
        window_radius=args.window_radius,
        pattern_family=pattern_family,
        window_radius_l=args.window_radius_l,
        window_radius_u=args.window_radius_u,
        prefix_dict=prefix_dict,
        loss_bound=args.loss_bound,
        prefix_limit=args.prefix_limit,
        bt_block_size=args.bt_block_size,
        bt_families=bt_families,
        bt_loss_bound=args.bt_loss_bound,
        shape_parms=shape_parms,
        tree_threshold=args.tree_threshold,
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
    groups = treeaware_isp(partition_value, params, rng=rng)
    route_info = None
    if params._routing_strategy == _ROUTE_POLICY_FULL_SUPPORT:
        route_info = {
            "mode": params._routing_strategy,
            "support": route_support(counts, params),
            "size_threshold": params.size_threshold,
            "vrf_threshold": params.vrf_threshold,
        }
    elif params._routing_strategy == _ROUTE_POLICY_PROFILE:
        route_info = {
            "mode": params._routing_strategy,
            "support": route_support(counts, params),
            "size_threshold": params.size_threshold,
            "vrf_threshold": params.vrf_threshold,
            "aux_t": params.aux_t,
            "aux_mode": params.aux_mode,
        }
    else:
        extracted = tree_extract(counts, params)
        if extracted is not None:
            groups_pre, residual_vec, residual_rows = extracted
            route_mode, support, ind_support, bt_support, block_lengths = _route_mode_and_support(
                residual_vec,
                residual_rows,
                params,
            )
            route_info = {
                "prefix_rows": groups_pre,
                "residual_vector": list(residual_vec),
                "residual_rows": residual_rows,
                "mode": route_mode,
                "support": support,
                "ind_support": ind_support,
                "bt_support": bt_support,
                "block_lengths": list(block_lengths),
            }

    output = {
        "hash_name": params.hash_name,
        "mode": params.mode,
        "route_policy": params.route_policy,
        "routing_strategy": params._routing_strategy,
        "aux_t": params.aux_t,
        "aux_mode": params.aux_mode,
        "size_threshold": params.size_threshold,
        "vrf_threshold": params.vrf_threshold,
        "score_name": params.score_name,
        "score_bound": params.score_bound,
        "partition_value": normalize_partition_value(partition_value, params.hash_len),
        "accepted": groups is not None,
        "block_values": block_values,
        "multiplicity_profile": counts,
        "window": {
            "radius_l": params.window_radius_l,
            "radius_u": params.window_radius_u,
            "low": low,
            "high": high,
        },
        "prefix_dict": [list(pattern) for pattern in params._prefix_dict],
        "loss_bound": params._loss_bound,
        "prefix_limit": params._prefix_limit,
        "bt_block_size": params._bt_block_size,
        "bt_loss_bound": params._bt_loss_bound,
        "route": route_info,
        "shape_statistics": shape_statistics(groups, params) if groups is not None else None,
        "verify_score": verify_score(groups, params) if groups is not None else None,
        "score_guard_passed": score_guard(groups, params) if groups is not None else False,
        "tree_score": tree_score(groups, params) if groups is not None else None,
        "groups": groups,
        "strictly_increasing": verify_output(groups) if groups is not None else False,
    }
    print(json.dumps(output, ensure_ascii=True, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(_main())
