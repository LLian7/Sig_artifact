from __future__ import annotations

import argparse
import json
import math
from dataclasses import asdict, dataclass
from collections import defaultdict
from decimal import Decimal, ROUND_HALF_UP
from functools import lru_cache
from random import Random
from typing import Any, Iterable, List, Optional, Sequence

from merkle_tree import MT
from pprf import PPRF
from treeaware_isp import TreeAwareISPParameters, route_support, treeaware_isp
from yc_sig import YCSig


DEFAULT_CASES = ("case1", "case2")
DEFAULT_SECURITY_TARGETS = (128, 160, 192)
DEFAULT_MAX_G_VALUES = (4, 8, 16, 32)
DEFAULT_HASH_LEN_MAX_FACTOR = 2.5
DEFAULT_PATTERN_FAMILY = "aligned"
_COLOR_EMPTY = 0
_COLOR_FULL = 1
_COLOR_MIXED = 2


@dataclass(frozen=True)
class SearchCell:
    case_name: str
    security_target: int
    max_g_value: int

    @property
    def max_g_bit(self) -> int:
        return self.max_g_value.bit_length() - 1

    @property
    def min_hash_len(self) -> int:
        if self.case_name == "case1":
            return self.security_target
        if self.case_name == "case2":
            return 2 * self.security_target
        raise ValueError(f"unsupported case_name={self.case_name!r}")


@dataclass(frozen=True)
class SearchRow:
    case_name: str
    security_target: int
    max_g_value: int
    partition_num: int
    window_radius: int
    block_num: int
    hash_len: int
    expected_retries: float
    kappa: float
    signature_size_obj: Optional[float] = None
    signature_size_samples: Optional[int] = None
    link_threshold: int = -1
    pattern_family_size: Optional[int] = None
    tree_threshold: Optional[int] = None

    @property
    def max_g_bit(self) -> int:
        return self.max_g_value.bit_length() - 1


def _parse_int_csv(raw: str) -> List[int]:
    return [int(piece.strip()) for piece in raw.split(",") if piece.strip()]


def _parse_optional_int_csv(raw: Optional[str]) -> Optional[List[int]]:
    if raw is None:
        return None
    return _parse_int_csv(raw)


def _parse_case_csv(raw: str) -> List[str]:
    return [piece.strip() for piece in raw.split(",") if piece.strip()]


def _round_half_up(value: float) -> int:
    return int(Decimal(str(value)).quantize(Decimal("1"), rounding=ROUND_HALF_UP))


def _next_divisible(value: int, divisor: int) -> int:
    remainder = value % divisor
    if remainder == 0:
        return value
    return value + divisor - remainder


def _factorials(limit: int) -> List[int]:
    values = [1] * (limit + 1)
    for index in range(2, limit + 1):
        values[index] = values[index - 1] * index
    return values


def aligned_pattern_family(max_g_value: int) -> tuple[tuple[int, ...], ...]:
    if max_g_value <= 0 or max_g_value & (max_g_value - 1):
        raise ValueError("max_g_value must be a positive power of two")
    patterns: list[tuple[int, ...]] = [()]
    width = 1
    while width <= max_g_value:
        for start in range(0, max_g_value, width):
            patterns.append(tuple(range(start, start + width)))
        width *= 2
    return tuple(dict.fromkeys(patterns))


def _pattern_to_mask(pattern: Sequence[int]) -> int:
    mask = 0
    for value in pattern:
        mask |= 1 << value
    return mask


def _pattern_family_from_name(
    name: str,
    max_g_value: int,
) -> tuple[tuple[int, ...], ...]:
    if name == "aligned":
        return aligned_pattern_family(max_g_value)
    if name == "all":
        return tuple(
            tuple(value for value in range(max_g_value) if (mask >> value) & 1)
            for mask in range(1 << max_g_value)
        )
    raise ValueError(f"unsupported pattern_family={name!r}")


def _canonicalize_tree_setup_value(value: Any) -> Any:
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if isinstance(value, (list, tuple)):
        return [_canonicalize_tree_setup_value(item) for item in value]
    if isinstance(value, dict):
        return {
            str(key): _canonicalize_tree_setup_value(subvalue)
            for key, subvalue in sorted(value.items(), key=lambda item: str(item[0]))
        }
    raise TypeError(f"unsupported tree setup value type: {type(value)!r}")


def _tree_setup_cache_key(tree_setup_kwargs: Optional[dict[str, Any]]) -> str:
    normalized = {} if tree_setup_kwargs is None else _canonicalize_tree_setup_value(tree_setup_kwargs)
    return json.dumps(normalized, ensure_ascii=True, sort_keys=True, separators=(",", ":"))


def _tree_setup_kwargs(
    *,
    pattern_family: Optional[Sequence[Sequence[int]]],
    tree_threshold: Optional[int],
    extra_tree_setup_kwargs: Optional[dict[str, Any]],
) -> dict[str, Any]:
    setup_kwargs: dict[str, Any] = {}
    if pattern_family is not None:
        setup_kwargs["pattern_family"] = pattern_family
    if tree_threshold is not None:
        setup_kwargs["tree_threshold"] = tree_threshold
    if extra_tree_setup_kwargs:
        setup_kwargs.update(extra_tree_setup_kwargs)
    return setup_kwargs


@lru_cache(maxsize=None)
def _treeaware_isp_params_cached(
    block_num: int,
    max_g_value: int,
    partition_num: int,
    window_radius: int,
    pattern_family: tuple[tuple[int, ...], ...],
    tree_threshold: Optional[int],
    tree_setup_key: str,
) -> TreeAwareISPParameters:
    max_g_bit = max_g_value.bit_length() - 1
    tree_setup_kwargs = json.loads(tree_setup_key)
    return TreeAwareISPParameters(
        hash_len=block_num * max_g_bit,
        max_g_bit=max_g_bit,
        partition_num=partition_num,
        window_radius=window_radius,
        **_tree_setup_kwargs(
            pattern_family=pattern_family,
            tree_threshold=tree_threshold,
            extra_tree_setup_kwargs=tree_setup_kwargs,
        ),
    )


@lru_cache(maxsize=None)
def _ycsig_setup_cached(
    security_target: int,
    hash_len: int,
    max_g_bit: int,
    partition_num: int,
    window_radius: int,
    link_threshold: int,
    pattern_family: Optional[tuple[tuple[int, ...], ...]],
    tree_threshold: Optional[int],
    tree_setup_key: str,
) -> Any:
    tree_setup_kwargs = json.loads(tree_setup_key)
    return YCSig.SigSetup(
        security_target,
        hash_len=hash_len,
        max_g_bit=max_g_bit,
        partition_size=partition_num,
        window_radius=window_radius,
        link_threshold=link_threshold,
        **_tree_setup_kwargs(
            pattern_family=pattern_family,
            tree_threshold=tree_threshold,
            extra_tree_setup_kwargs=tree_setup_kwargs,
        ),
        salt_bytes=2,
    )


@lru_cache(maxsize=None)
def _histogram_probability_terms_cached(
    block_num: int,
    max_g_value: int,
    low: int,
    high: int,
) -> tuple[tuple[tuple[int, ...], float, float], ...]:
    if low < 0 or high < low:
        return ()
    log_fact_n = math.lgamma(block_num + 1)
    log_space_size = block_num * math.log(max_g_value)
    entries = []
    for counts in _iter_count_vectors(
        max_g_value=max_g_value,
        block_num=block_num,
        low=low,
        high=high,
    ):
        log_hist_mass = (
            log_fact_n
            - log_space_size
            - sum(math.lgamma(count + 1) for count in counts)
        )
        hist_mass = math.exp(log_hist_mass)
        entries.append((counts, hist_mass, hist_mass * hist_mass))
    return tuple(entries)


def _signature_size_object_model_from_groups(
    groups: Sequence[Sequence[int]],
    params: Any,
    *,
    security_target: int,
) -> float:
    selected_indices = []
    for group_index, subgroup in enumerate(groups):
        base = group_index * params.max_g_value
        for block_value in subgroup:
            selected_indices.append(base + block_value)
    if (
        params.pm_ISP.mode == "vrf"
        or getattr(params.pm_ISP, "vrf_threshold", None) is not None
    ):
        selected_index_set = set(selected_indices)
        materialized_indices = [
            alpha for alpha in range(params.leaf_count) if alpha not in selected_index_set
        ]
    else:
        materialized_indices = selected_indices
    bitstrings = tuple(
        format(alpha, f"0{params.leaf_index_bits}b")
        for alpha in materialized_indices
    )
    punctured_count = len(
        PPRF.CanonicalPrefixes(
            params.pm_PPRF,
            bitstrings,
            inputs_normalized=True,
        )
    )
    partial_count = len(MT.CanonicalStatePositions(params.pm_MT, materialized_indices))
    return punctured_count + partial_count + 1.0 + 16.0 / security_target


def _iter_count_vectors(
    *,
    max_g_value: int,
    block_num: int,
    low: int,
    high: int,
) -> Iterable[tuple[int, ...]]:
    counts = [0] * max_g_value

    def visit(index: int, remaining: int) -> Iterable[tuple[int, ...]]:
        if index == max_g_value:
            if remaining == 0:
                yield tuple(counts)
            return
        slots_left = max_g_value - index - 1
        lower = max(low, remaining - slots_left * high)
        upper = min(high, remaining - slots_left * low)
        for count in range(lower, upper + 1):
            counts[index] = count
            yield from visit(index + 1, remaining - count)

    yield from visit(0, block_num)


def _node_cover_count_from_mask(leaf_mask: int, leaf_count: int) -> int:
    if leaf_mask == 0:
        return 0
    full_mask = (1 << leaf_count) - 1
    if leaf_mask == full_mask:
        return 1

    padded_count = 1 << (leaf_count - 1).bit_length()

    def visit(start: int, end: int) -> int:
        width = end - start
        interval_mask = ((1 << width) - 1) << start
        covered = leaf_mask & interval_mask
        if covered == 0:
            return 0
        if covered == interval_mask:
            return 1
        if width == 1:
            return 1
        mid = start + width // 2
        return visit(start, mid) + visit(mid, end)

    return visit(0, padded_count)


def _tree_score_from_group_masks(
    group_masks: Sequence[int],
    *,
    max_g_value: int,
) -> int:
    leaf_count = len(group_masks) * max_g_value
    selected_mask = 0
    for row_index, row_mask in enumerate(group_masks):
        base = row_index * max_g_value
        current = row_mask
        while current:
            low_bit = current & -current
            selected_mask |= 1 << (base + low_bit.bit_length() - 1)
            current ^= low_bit
    full_mask = (1 << leaf_count) - 1
    return _node_cover_count_from_mask(
        selected_mask,
        leaf_count,
    ) + _node_cover_count_from_mask(full_mask ^ selected_mask, leaf_count)


def _mask_count_vector(mask: int, max_g_value: int) -> tuple[int, ...]:
    return tuple(1 if (mask >> value) & 1 else 0 for value in range(max_g_value))


def _add_count_vectors_bounded(
    left: tuple[int, ...],
    right: tuple[int, ...],
    high: int,
) -> Optional[tuple[int, ...]]:
    values = []
    for left_value, right_value in zip(left, right):
        value = left_value + right_value
        if value > high:
            return None
        values.append(value)
    return tuple(values)


@lru_cache(maxsize=None)
def _row_color_counts_by_score_cached(
    partition_num: int,
    tree_threshold: int,
) -> dict[tuple[int, int, int], int]:
    """Count row-level empty/full/mixed shapes by full rows, empty rows, score."""

    padded_rows = 1 << (partition_num - 1).bit_length()

    leaf_maps: dict[int, dict[tuple[int, int, int, int], int]] = {}
    for row_index in range(padded_rows):
        if row_index >= partition_num:
            leaf_maps[row_index] = {}
            continue

        leaf_maps[row_index] = {
            (0, 1, _COLOR_EMPTY, 1): 1,
            (1, 0, _COLOR_FULL, 1): 1,
            (0, 0, _COLOR_MIXED, 0): 1,
        }

    def combine(
        start: int,
        end: int,
    ) -> dict[tuple[int, int, int, int], int]:
        if start >= partition_num:
            return {}
        if end - start == 1:
            return leaf_maps[start]
        mid = (start + end) // 2
        left_map = combine(start, mid)
        right_map = combine(mid, end)
        if not left_map:
            return right_map if end <= partition_num else {
                (full_rows, empty_rows, _COLOR_MIXED, score): ways
                for (full_rows, empty_rows, _color, score), ways in right_map.items()
            }
        if not right_map:
            return left_map if end <= partition_num else {
                (full_rows, empty_rows, _COLOR_MIXED, score): ways
                for (full_rows, empty_rows, _color, score), ways in left_map.items()
            }

        fully_actual = end <= partition_num
        combined: dict[tuple[int, int, int, int], int] = defaultdict(int)
        for (
            left_full,
            left_empty,
            left_color,
            left_score,
        ), left_ways in left_map.items():
            for (
                right_full,
                right_empty,
                right_color,
                right_score,
            ), right_ways in right_map.items():
                if (
                    fully_actual
                    and left_color == right_color
                    and left_color in {_COLOR_EMPTY, _COLOR_FULL}
                ):
                    color = left_color
                    score = 1
                else:
                    color = _COLOR_MIXED
                    score = left_score + right_score
                if score <= tree_threshold:
                    combined[
                        (
                            left_full + right_full,
                            left_empty + right_empty,
                            color,
                            score,
                        )
                    ] += left_ways * right_ways
        return dict(combined)

    root_map = combine(0, padded_rows)
    shape_counts: dict[tuple[int, int, int], int] = defaultdict(int)
    for (full_rows, empty_rows, _color, score), ways in root_map.items():
        shape_counts[(full_rows, empty_rows, score)] += ways
    return dict(shape_counts)


@lru_cache(maxsize=None)
def _partial_pattern_count_tables_cached(
    partition_num: int,
    max_g_value: int,
    partial_pattern_masks: tuple[int, ...],
    tree_threshold: int,
    high: int,
) -> tuple[dict[tuple[int, ...], tuple[int, ...]], ...]:
    zero_counts = (0,) * max_g_value
    if not partial_pattern_masks:
        empty_table = {zero_counts: tuple(1 for _ in range(tree_threshold + 1))}
        return (empty_table,)

    partial_patterns = tuple(
        (
            _mask_count_vector(mask, max_g_value),
            _tree_score_from_group_masks((mask,), max_g_value=max_g_value),
        )
        for mask in partial_pattern_masks
    )
    min_cost = min(cost for _vector, cost in partial_patterns)
    max_mixed_rows = min(partition_num, tree_threshold // min_cost)
    tables: list[dict[tuple[int, ...], tuple[int, ...]]] = []
    states: dict[tuple[tuple[int, ...], int], int] = {(zero_counts, 0): 1}

    for mixed_rows in range(max_mixed_rows + 1):
        by_counts: dict[tuple[int, ...], list[int]] = defaultdict(
            lambda: [0] * (tree_threshold + 1)
        )
        for (counts, score), ways in states.items():
            by_counts[counts][score] += ways

        cumulative_table: dict[tuple[int, ...], tuple[int, ...]] = {}
        for counts, exact_scores in by_counts.items():
            running = 0
            cumulative_scores = []
            for ways in exact_scores:
                running += ways
                cumulative_scores.append(running)
            cumulative_table[counts] = tuple(cumulative_scores)
        tables.append(cumulative_table)

        if mixed_rows == max_mixed_rows:
            break

        next_states: dict[tuple[tuple[int, ...], int], int] = defaultdict(int)
        for (counts, score), ways in states.items():
            for vector, cost in partial_patterns:
                next_score = score + cost
                if next_score > tree_threshold:
                    continue
                next_counts = _add_count_vectors_bounded(counts, vector, high)
                if next_counts is not None:
                    next_states[(next_counts, next_score)] += ways
        states = dict(next_states)

    return tuple(tables)


@lru_cache(maxsize=None)
def _tree_legal_counts_by_vector_cached(
    partition_num: int,
    max_g_value: int,
    pattern_masks: tuple[int, ...],
    tree_threshold: int,
    low: int,
    high: int,
    block_num: int,
) -> dict[tuple[int, ...], int]:
    full_row_mask = (1 << max_g_value) - 1
    partial_pattern_masks = tuple(
        mask for mask in pattern_masks if mask not in {0, full_row_mask}
    )
    row_shape_counts = _row_color_counts_by_score_cached(
        partition_num,
        tree_threshold,
    )
    partial_tables = _partial_pattern_count_tables_cached(
        partition_num,
        max_g_value,
        partial_pattern_masks,
        tree_threshold,
        high,
    )

    legal_counts: dict[tuple[int, ...], int] = {}
    for counts in _iter_count_vectors(
        max_g_value=max_g_value,
        block_num=block_num,
        low=low,
        high=high,
    ):
        legal_count = 0
        for (full_rows, empty_rows, row_score), row_shape_ways in row_shape_counts.items():
            mixed_rows = partition_num - full_rows - empty_rows
            if mixed_rows < 0 or mixed_rows >= len(partial_tables):
                continue
            budget = tree_threshold - row_score
            if budget < 0:
                continue
            residual_counts = tuple(count - full_rows for count in counts)
            if any(count < 0 or count > mixed_rows for count in residual_counts):
                continue
            score_prefixes = partial_tables[mixed_rows].get(residual_counts)
            if score_prefixes is None:
                continue
            partial_ways = score_prefixes[budget]
            if partial_ways:
                legal_count += row_shape_ways * partial_ways
        if legal_count:
            legal_counts[counts] = legal_count
    return legal_counts


@lru_cache(maxsize=None)
def _tree_legal_counts_unthresholded_cached(
    partition_num: int,
    max_g_value: int,
    pattern_masks: tuple[int, ...],
    high: int,
) -> dict[tuple[int, ...], int]:
    pattern_vectors = tuple(
        _mask_count_vector(mask, max_g_value)
        for mask in pattern_masks
    )
    zero = (0,) * max_g_value
    states: dict[tuple[int, ...], int] = {zero: 1}
    for _ in range(partition_num):
        next_states: dict[tuple[int, ...], int] = defaultdict(int)
        for counts, ways in states.items():
            for vector in pattern_vectors:
                next_counts = _add_count_vectors_bounded(counts, vector, high)
                if next_counts is not None:
                    next_states[next_counts] += ways
        states = dict(next_states)
    states.pop(zero, None)
    return states


@lru_cache(maxsize=None)
def _tree_legal_count_cached(
    counts: tuple[int, ...],
    partition_num: int,
    max_g_value: int,
    pattern_masks: tuple[int, ...],
    tree_threshold: Optional[int],
) -> int:
    pattern_vectors = tuple(
        tuple(1 if (mask >> value) & 1 else 0 for value in range(max_g_value))
        for mask in pattern_masks
    )

    if tree_threshold is None:
        @lru_cache(maxsize=None)
        def count_tree(index: int, rem_vec: tuple[int, ...]) -> int:
            rows_left = partition_num - index
            if any(value < 0 or value > rows_left for value in rem_vec):
                return 0
            if index == partition_num:
                return 1 if not any(rem_vec) else 0
            total = 0
            for vector in pattern_vectors:
                next_rem = tuple(left - right for left, right in zip(rem_vec, vector))
                if any(value < 0 for value in next_rem):
                    continue
                total += count_tree(index + 1, next_rem)
            return total

        return count_tree(0, counts)

    @lru_cache(maxsize=None)
    def count_tree_with_score(
        index: int,
        rem_vec: tuple[int, ...],
        prefix_masks: tuple[int, ...],
    ) -> int:
        rows_left = partition_num - index
        if any(value < 0 or value > rows_left for value in rem_vec):
            return 0
        if index == partition_num:
            if any(rem_vec):
                return 0
            return (
                1
                if _tree_score_from_group_masks(prefix_masks, max_g_value=max_g_value)
                <= tree_threshold
                else 0
            )
        total = 0
        for mask, vector in zip(pattern_masks, pattern_vectors):
            next_rem = tuple(left - right for left, right in zip(rem_vec, vector))
            if any(value < 0 for value in next_rem):
                continue
            total += count_tree_with_score(index + 1, next_rem, prefix_masks + (mask,))
        return total

    return count_tree_with_score(0, counts, ())


def _window_bounds(
    block_num: int,
    max_g_value: int,
    partition_num: int,
    window_radius: int,
) -> tuple[int, int]:
    avg_floor = block_num // max_g_value
    avg_ceil = (block_num + max_g_value - 1) // max_g_value
    low = avg_floor - window_radius
    high = min(avg_ceil + window_radius, partition_num)
    return low, high


@lru_cache(maxsize=None)
def _exact_metrics(
    block_num: int,
    max_g_value: int,
    partition_num: int,
    window_radius: int,
) -> tuple[float, float]:
    low, high = _window_bounds(
        block_num,
        max_g_value,
        partition_num,
        window_radius,
    )
    if low < 0 or high < low or high > partition_num:
        return 0.0, float("-inf")

    factorials = _factorials(max(block_num, partition_num))
    log_fact_n = math.lgamma(block_num + 1)
    log_fact_p = math.lgamma(partition_num + 1)
    log_space_size = block_num * math.log(max_g_value)

    acc_weights = [0.0] * (block_num + 1)
    beta_weights = [0.0] * (block_num + 1)
    alpha_weights = [0.0] * (block_num + 1)
    acc_weights[0] = 1.0
    beta_weights[0] = 1.0
    alpha_weights[0] = 1.0

    inv_fact = [0.0] * (high + 1)
    beta = [0.0] * (high + 1)
    alpha = [0.0] * (high + 1)
    for count in range(low, high + 1):
        inv_fact[count] = 1.0 / factorials[count]
        beta[count] = float(factorials[partition_num - count])
        alpha[count] = beta[count] / factorials[count]

    for _ in range(max_g_value):
        next_acc = [0.0] * (block_num + 1)
        next_beta = [0.0] * (block_num + 1)
        next_alpha = [0.0] * (block_num + 1)
        for used in range(block_num + 1):
            if (
                acc_weights[used] == 0.0
                and beta_weights[used] == 0.0
                and alpha_weights[used] == 0.0
            ):
                continue
            upper = min(high, block_num - used)
            for count in range(low, upper + 1):
                next_acc[used + count] += acc_weights[used] * inv_fact[count]
                next_beta[used + count] += beta_weights[used] * beta[count]
                next_alpha[used + count] += alpha_weights[used] * alpha[count]
        acc_weights = next_acc
        beta_weights = next_beta
        alpha_weights = next_alpha

    acceptance_probability = math.exp(
        log_fact_n + math.log(acc_weights[block_num]) - log_space_size
    )
    if acceptance_probability <= 0.0:
        return 0.0, float("-inf")
    if log_space_size == 0.0:
        return acceptance_probability, float("inf")
    if log_space_size < 40.0:
        space_size = math.exp(log_space_size)
        log_inverse_space_pair = -math.log(space_size * (space_size - 1.0))
    else:
        log_inverse_space_pair = -2.0 * log_space_size

    term1 = math.exp(
        2.0 * log_fact_n
        - max_g_value * log_fact_p
        + math.log(alpha_weights[block_num])
        + log_inverse_space_pair
    )
    term2 = math.exp(
        log_fact_n
        - max_g_value * log_fact_p
        + math.log(beta_weights[block_num])
        + log_inverse_space_pair
    )
    ucr_upper = term1 - term2
    if ucr_upper <= 0.0:
        return acceptance_probability, float("inf")
    return acceptance_probability, -math.log2(ucr_upper)


def exact_parameter_metrics(
    block_num: int,
    max_g_value: int,
    partition_num: int,
    window_radius: int,
) -> tuple[float, float]:
    return _exact_metrics(
        block_num,
        max_g_value,
        partition_num,
        window_radius,
    )


@lru_cache(maxsize=None)
def _exact_tree_metrics(
    block_num: int,
    max_g_value: int,
    partition_num: int,
    window_radius: int,
    pattern_family: tuple[tuple[int, ...], ...],
    tree_threshold: Optional[int],
    tree_setup_key: str,
) -> tuple[float, float]:
    params = _treeaware_isp_params_cached(
        block_num,
        max_g_value,
        partition_num,
        window_radius,
        pattern_family,
        tree_threshold,
        tree_setup_key,
    )
    low, high = params._window_low, params._window_high
    if not params._window_valid or high > partition_num:
        return 0.0, float("-inf")

    log_space_size = block_num * math.log(max_g_value)
    space_size = math.exp(log_space_size) if log_space_size < 700.0 else math.inf
    acceptance_probability = 0.0
    ucr_probability = 0.0

    for counts, hist_mass, hist_mass_sq in _histogram_probability_terms_cached(
        block_num,
        max_g_value,
        low,
        high,
    ):
        support = route_support(counts, params)
        if support == 0:
            continue
        acceptance_probability += hist_mass
        if math.isinf(space_size):
            histogram_pair_probability = hist_mass_sq
        else:
            histogram_pair_probability = (
                space_size * hist_mass_sq - hist_mass
            ) / (space_size - 1.0)
        if histogram_pair_probability > 0.0:
            ucr_probability += histogram_pair_probability / support

    if acceptance_probability <= 0.0:
        return 0.0, float("-inf")
    if ucr_probability <= 0.0:
        return acceptance_probability, float("inf")
    return acceptance_probability, -math.log2(ucr_probability)


def exact_tree_parameter_metrics(
    block_num: int,
    max_g_value: int,
    partition_num: int,
    window_radius: int,
    pattern_family: Sequence[Sequence[int]],
    tree_threshold: Optional[int] = None,
    tree_setup_kwargs: Optional[dict[str, Any]] = None,
) -> tuple[float, float]:
    normalized_pattern_family = tuple(tuple(pattern) for pattern in pattern_family)
    return _exact_tree_metrics(
        block_num,
        max_g_value,
        partition_num,
        window_radius,
        normalized_pattern_family,
        tree_threshold,
        _tree_setup_cache_key(tree_setup_kwargs),
    )


def _candidate_metrics(
    *,
    visp_mode: str,
    block_num: int,
    max_g_value: int,
    partition_num: int,
    window_radius: int,
    pattern_family: Optional[Sequence[Sequence[int]]],
    tree_threshold: Optional[int],
    tree_setup_kwargs: Optional[dict[str, Any]] = None,
) -> tuple[float, float]:
    if visp_mode == "base":
        return exact_parameter_metrics(
            block_num,
            max_g_value,
            partition_num,
            window_radius,
        )
    if visp_mode != "tree":
        raise ValueError("visp_mode must be 'base' or 'tree'")
    if pattern_family is None:
        raise ValueError("pattern_family is required when visp_mode='tree'")
    return exact_tree_parameter_metrics(
        block_num,
        max_g_value,
        partition_num,
        window_radius,
        pattern_family,
        tree_threshold,
        tree_setup_kwargs,
    )


def _estimate_signature_size_object_model(
    *,
    visp_mode: str,
    security_target: int,
    hash_len: int,
    max_g_bit: int,
    partition_num: int,
    window_radius: int,
    link_threshold: int,
    pattern_family: Optional[Sequence[Sequence[int]]],
    tree_threshold: Optional[int],
    samples: int,
    seed: int,
    tree_setup_kwargs: Optional[dict[str, Any]] = None,
) -> float:
    if samples <= 0:
        raise ValueError("samples must be positive")

    normalized_pattern_family = (
        None if pattern_family is None else tuple(tuple(pattern) for pattern in pattern_family)
    )
    tree_setup_key = _tree_setup_cache_key(
        tree_setup_kwargs if visp_mode == "tree" else None
    )
    setup = _ycsig_setup_cached(
        security_target,
        hash_len,
        max_g_bit,
        partition_num,
        window_radius,
        link_threshold,
        normalized_pattern_family if visp_mode == "tree" else None,
        tree_threshold if visp_mode == "tree" else None,
        tree_setup_key,
    )
    params = setup.params
    rng = Random(seed)
    total = 0.0
    sample_groups = treeaware_isp
    sample_getrandbits = rng.getrandbits
    isp_params = params.pm_ISP

    for _ in range(samples):
        while True:
            partition_value = sample_getrandbits(hash_len)
            groups = sample_groups(partition_value, isp_params)
            if groups is None:
                continue
            break

        total += _signature_size_object_model_from_groups(
            groups,
            params,
            security_target=security_target,
        )

    return total / samples


def _candidate_row(
    *,
    cell: SearchCell,
    partition_num: int,
    hash_len: int,
    window_radius: int,
    acceptance_probability: float,
    kappa: float,
    signature_size_obj: Optional[float] = None,
    signature_size_samples: Optional[int] = None,
    link_threshold: int = -1,
    pattern_family_size: Optional[int] = None,
    tree_threshold: Optional[int] = None,
) -> SearchRow:
    return SearchRow(
        case_name=cell.case_name,
        security_target=cell.security_target,
        max_g_value=cell.max_g_value,
        partition_num=partition_num,
        window_radius=window_radius,
        block_num=hash_len // cell.max_g_bit,
        hash_len=hash_len,
        expected_retries=(1.0 / acceptance_probability),
        kappa=kappa,
        signature_size_obj=signature_size_obj,
        signature_size_samples=signature_size_samples,
        link_threshold=link_threshold,
        pattern_family_size=pattern_family_size,
        tree_threshold=tree_threshold,
    )


def _iter_hash_lens(cell: SearchCell, hash_len_max: int) -> Iterable[int]:
    start = _next_divisible(cell.min_hash_len, cell.max_g_bit)
    for hash_len in range(start, hash_len_max + 1, cell.max_g_bit):
        yield hash_len


def _bounded_range(
    start: int,
    stop: int,
    lower: Optional[int],
    upper: Optional[int],
) -> range:
    if lower is not None:
        start = max(start, lower)
    if upper is not None:
        stop = min(stop, upper)
    if stop < start:
        return range(0)
    return range(start, stop + 1)


def search_best_row_for_cell(
    *,
    cell: SearchCell,
    objective: str,
    retry_limit: float,
    hash_len_max: int,
    hash_len_min: Optional[int] = None,
    link_threshold: int = -1,
    visp_mode: str = "base",
    pattern_family_name: str = DEFAULT_PATTERN_FAMILY,
    tree_threshold: Optional[int] = None,
    tree_thresholds: Optional[Sequence[int]] = None,
    partition_num_min: Optional[int] = None,
    partition_num_max: Optional[int] = None,
    window_radius_min: Optional[int] = None,
    window_radius_max: Optional[int] = None,
    sig_size_samples: int = 0,
    sig_size_seed: int = 0,
    sig_partition_num_slack: Optional[int] = None,
    tree_setup_kwargs: Optional[dict[str, Any]] = None,
) -> Optional[SearchRow]:
    if objective not in {"partition_num", "sig_size"}:
        raise ValueError("objective must be 'partition_num' or 'sig_size'")

    hash_lens = [
        hash_len
        for hash_len in _iter_hash_lens(cell, hash_len_max)
        if hash_len_min is None or hash_len >= hash_len_min
    ]
    if not hash_lens:
        return None
    pattern_family = (
        _pattern_family_from_name(pattern_family_name, cell.max_g_value)
        if visp_mode == "tree"
        else None
    )
    pattern_family_size = len(pattern_family) if pattern_family is not None else None
    threshold_candidates: Sequence[Optional[int]]
    if visp_mode == "tree":
        if tree_thresholds is not None:
            threshold_candidates = tuple(sorted(set(tree_thresholds)))
        else:
            threshold_candidates = (tree_threshold,)
    else:
        threshold_candidates = (None,)

    if objective == "partition_num":
        min_partition_num = math.ceil(
            (hash_lens[0] // cell.max_g_bit) / cell.max_g_value
        )
        max_partition_num = hash_len_max // cell.max_g_bit
        for partition_num in _bounded_range(
            min_partition_num,
            max_partition_num,
            partition_num_min,
            partition_num_max,
        ):
            for hash_len in hash_lens:
                block_num = hash_len // cell.max_g_bit
                min_required_partition_num = math.ceil(block_num / cell.max_g_value)
                if partition_num < min_required_partition_num:
                    break
                if partition_num > block_num:
                    continue
                max_window_radius = block_num // cell.max_g_value
                for window_radius in _bounded_range(
                    0,
                    max_window_radius,
                    window_radius_min,
                    window_radius_max,
                ):
                    for candidate_tree_threshold in threshold_candidates:
                        acceptance_probability, kappa = _candidate_metrics(
                            visp_mode=visp_mode,
                            block_num=block_num,
                            max_g_value=cell.max_g_value,
                            partition_num=partition_num,
                            window_radius=window_radius,
                            pattern_family=pattern_family,
                            tree_threshold=candidate_tree_threshold,
                            tree_setup_kwargs=tree_setup_kwargs,
                        )
                        if acceptance_probability <= 0.0:
                            continue
                        expected_retries = 1.0 / acceptance_probability
                        if expected_retries > retry_limit:
                            continue
                        if kappa < cell.security_target:
                            continue
                        return _candidate_row(
                            cell=cell,
                            partition_num=partition_num,
                            hash_len=hash_len,
                            window_radius=window_radius,
                            acceptance_probability=acceptance_probability,
                            kappa=kappa,
                            link_threshold=link_threshold,
                            pattern_family_size=pattern_family_size,
                            tree_threshold=candidate_tree_threshold,
                        )
        return None

    feasible_rows: List[SearchRow] = []
    for hash_len in hash_lens:
        block_num = hash_len // cell.max_g_bit
        min_partition_num = math.ceil(block_num / cell.max_g_value)
        for partition_num in _bounded_range(
            min_partition_num,
            block_num,
            partition_num_min,
            partition_num_max,
        ):
            max_window_radius = block_num // cell.max_g_value
            for window_radius in _bounded_range(
                0,
                max_window_radius,
                window_radius_min,
                window_radius_max,
            ):
                for candidate_tree_threshold in threshold_candidates:
                    acceptance_probability, kappa = _candidate_metrics(
                        visp_mode=visp_mode,
                        block_num=block_num,
                        max_g_value=cell.max_g_value,
                        partition_num=partition_num,
                        window_radius=window_radius,
                        pattern_family=pattern_family,
                        tree_threshold=candidate_tree_threshold,
                        tree_setup_kwargs=tree_setup_kwargs,
                    )
                    if acceptance_probability <= 0.0:
                        continue
                    expected_retries = 1.0 / acceptance_probability
                    if expected_retries > retry_limit:
                        continue
                    if kappa < cell.security_target:
                        continue
                    feasible_rows.append(
                        _candidate_row(
                            cell=cell,
                            partition_num=partition_num,
                            hash_len=hash_len,
                            window_radius=window_radius,
                            acceptance_probability=acceptance_probability,
                            kappa=kappa,
                            link_threshold=link_threshold,
                            pattern_family_size=pattern_family_size,
                            tree_threshold=candidate_tree_threshold,
                        )
                    )

    if not feasible_rows:
        return None

    if sig_partition_num_slack is not None:
        min_partition_num = min(row.partition_num for row in feasible_rows)
        feasible_rows = [
            row
            for row in feasible_rows
            if row.partition_num <= min_partition_num + sig_partition_num_slack
        ]

    best_row: Optional[SearchRow] = None
    best_key: Optional[tuple[float, int, float, int, int]] = None
    for index, row in enumerate(feasible_rows):
        signature_size_obj = _estimate_signature_size_object_model(
            visp_mode=visp_mode,
            security_target=row.security_target,
            hash_len=row.hash_len,
            max_g_bit=row.max_g_bit,
            partition_num=row.partition_num,
            window_radius=row.window_radius,
            link_threshold=link_threshold,
            pattern_family=pattern_family,
            tree_threshold=row.tree_threshold,
            samples=sig_size_samples,
            seed=sig_size_seed + index,
            tree_setup_kwargs=tree_setup_kwargs,
        )
        evaluated_row = SearchRow(
            case_name=row.case_name,
            security_target=row.security_target,
            max_g_value=row.max_g_value,
            partition_num=row.partition_num,
            window_radius=row.window_radius,
            block_num=row.block_num,
            hash_len=row.hash_len,
            expected_retries=row.expected_retries,
            kappa=row.kappa,
            signature_size_obj=signature_size_obj,
            signature_size_samples=sig_size_samples,
            link_threshold=link_threshold,
            pattern_family_size=row.pattern_family_size,
            tree_threshold=row.tree_threshold,
        )
        key = (
            signature_size_obj,
            evaluated_row.partition_num,
            evaluated_row.expected_retries,
            evaluated_row.window_radius,
        )
        if best_key is None or key < best_key:
            best_key = key
            best_row = evaluated_row
    return best_row


def search_rows(
    *,
    cases: Sequence[str],
    security_targets: Sequence[int],
    max_g_values: Sequence[int],
    objective: str,
    retry_limit: float,
    hash_len_max_factor: float,
    hash_len_max_absolute: Optional[int],
    hash_len_min_absolute: Optional[int] = None,
    link_threshold: int = -1,
    visp_mode: str = "base",
    pattern_family_name: str = DEFAULT_PATTERN_FAMILY,
    tree_threshold: Optional[int] = None,
    tree_thresholds: Optional[Sequence[int]] = None,
    partition_num_min: Optional[int] = None,
    partition_num_max: Optional[int] = None,
    window_radius_min: Optional[int] = None,
    window_radius_max: Optional[int] = None,
    sig_size_samples: int = 64,
    sig_size_seed: int = 0,
    sig_partition_num_slack: Optional[int] = None,
    tree_setup_kwargs: Optional[dict[str, Any]] = None,
) -> List[SearchRow]:
    rows: List[SearchRow] = []
    for security_target in security_targets:
        for max_g_value in max_g_values:
            for case_name in cases:
                cell = SearchCell(
                    case_name=case_name,
                    security_target=security_target,
                    max_g_value=max_g_value,
                )
                scaled_hash_len_max = max(
                    cell.min_hash_len,
                    math.ceil(hash_len_max_factor * security_target),
                )
                hash_len_caps = [
                    (
                        scaled_hash_len_max
                        if hash_len_max_absolute is None
                        else max(cell.min_hash_len, hash_len_max_absolute)
                    )
                ]
                if hash_len_max_absolute is None:
                    fallback_cap = max(hash_len_caps[0], 4 * security_target)
                    if fallback_cap > hash_len_caps[0]:
                        hash_len_caps.append(fallback_cap)

                row = None
                for hash_len_max in hash_len_caps:
                    row = search_best_row_for_cell(
                        cell=cell,
                        objective=objective,
                        retry_limit=retry_limit,
                        hash_len_max=hash_len_max,
                        hash_len_min=hash_len_min_absolute,
                        link_threshold=link_threshold,
                        visp_mode=visp_mode,
                        pattern_family_name=pattern_family_name,
                        tree_threshold=tree_threshold,
                        tree_thresholds=tree_thresholds,
                        partition_num_min=partition_num_min,
                        partition_num_max=partition_num_max,
                        window_radius_min=window_radius_min,
                        window_radius_max=window_radius_max,
                        sig_size_samples=sig_size_samples,
                        sig_size_seed=sig_size_seed,
                        sig_partition_num_slack=sig_partition_num_slack,
                        tree_setup_kwargs=tree_setup_kwargs,
                    )
                    if row is not None:
                        break
                if row is None:
                    raise RuntimeError(
                        "no feasible row found for "
                        f"{case_name}, k*={security_target}, w={max_g_value}; "
                        "consider increasing --hash-len-max-factor or "
                        "--hash-len-max-absolute"
                    )
                rows.append(row)
    rows.sort(key=lambda row: (row.security_target, row.max_g_value, row.case_name))
    return rows


def render_text(rows: Sequence[SearchRow]) -> str:
    has_tree = any(row.pattern_family_size is not None for row in rows)
    header = (
        "Case | k* | w | P | R | n | HashLen | E[Re] | kappa"
        + (" | TreeT | |PF|" if has_tree else "")
        + (" | SigSize" if any(row.signature_size_obj is not None for row in rows) else "")
    )
    rule = "-" * len(header)
    lines = [header, rule]
    for row in rows:
        line = (
            f"{row.case_name} | "
            f"{row.security_target} | "
            f"{row.max_g_value} | "
            f"{row.partition_num} | "
            f"{row.window_radius} | "
            f"{row.block_num} | "
            f"{row.hash_len} | "
            f"{row.expected_retries:.2f} | "
            f"{row.kappa:.1f}"
        )
        if has_tree:
            tree_threshold = "-" if row.tree_threshold is None else str(row.tree_threshold)
            line += (
                f" | {tree_threshold} | "
                f"{row.pattern_family_size if row.pattern_family_size is not None else '-'}"
            )
        if row.signature_size_obj is not None:
            line += f" | {row.signature_size_obj:.1f}"
        lines.append(line)
    return "\n".join(lines)


def render_latex(rows: Sequence[SearchRow], *, retry_limit: float, objective: str) -> str:
    by_key = {
        (row.case_name, row.security_target, row.max_g_value): row
        for row in rows
    }
    security_targets = sorted({row.security_target for row in rows})
    max_g_values = sorted({row.max_g_value for row in rows})
    if tuple(sorted({row.case_name for row in rows})) != ("case1", "case2"):
        raise ValueError("latex rendering expects both case1 and case2 rows")
    tree_mode = any(row.pattern_family_size is not None for row in rows)

    if tree_mode:
        caption_tail = (
            "The selected rows are required to satisfy "
            f"$\\mathbb{{E}}[\\mathrm{{Re}}]\\le {int(retry_limit)}$ and "
            "verified $\\kappa_T\\ge\\kappa^{*}$ under the exact UCR analysis "
            "of Theorem~\\ref{thm:ucr-tree-visp}. The parameters "
            "$|\\PatternFamily|$ and $\\TreeThreshold$ describe the public "
            "tree-aware routing policy used by $\\TreeSampler$."
        )
    elif objective == "partition_num":
        caption_tail = (
            "For each $(\\kappa^{*},\\MaxGValue)$ pair, the "
            "primary objective is to minimize $\\PartitionNum$ subject to "
            f"$\\mathbb{{E}}[Re]\\le {int(retry_limit)}$ and certified "
            "$\\kappa\\ge \\kappa^{*}$. Among rows attaining the same minimum "
            "$\\PartitionNum$, we report the one with the smallest admissible "
            "$\\HashLen$, breaking remaining ties by the smallest symmetric "
            "radius $\\WindowRadius$. The selected window radius "
            "$\\WindowRadius$ is reported explicitly for each row."
        )
    else:
        caption_tail = (
            "For each $(\\kappa^{*},\\MaxGValue)$ pair, the "
            "primary objective is to minimize the empirical YCSig signature-size "
            "object model induced by the accepted simplified-windowed VISP outputs, subject to "
            f"$\\mathbb{{E}}[Re]\\le {int(retry_limit)}$ and certified "
            "$\\kappa\\ge \\kappa^{*}$. Among rows attaining the same minimum "
            "signature-size score, we prefer the smallest $\\PartitionNum$, then the "
            "smallest $\\mathbb{E}[Re]$, and break any remaining ties by the "
            "smallest symmetric radius $\\WindowRadius$. The selected window "
            "radius $\\WindowRadius$ is reported explicitly for each row."
        )

    lines = [
        r"\begin{table}[t]",
        r"\centering",
        r"\scriptsize" if tree_mode else r"\footnotesize",
        r"\setlength{\tabcolsep}{1.8pt}" if tree_mode else r"\setlength{\tabcolsep}{2pt}",
        (
            r"\caption{Illustrative parameters selected for the "
            + (r"tree-aware TreeAwareISP. " if tree_mode else r"windowed TreeAwareISP. ")
            + r"Case~1 searches admissible $\HashLen\ge \kappa^{*}$ with "
            + r"$\HashLen/\MaxGBit\in\mathbb{N}$, and Case~2 searches admissible "
            + r"$\HashLen\ge 2\kappa^{*}$ with $\HashLen/\MaxGBit\in\mathbb{N}$. "
            + caption_tail
            + "}"
        ),
        r"\label{tab:all-L-treeaware}" if tree_mode else r"\label{tab:all-L-windowed}",
        r"\resizebox{0.999\textwidth}{!}{",
        (
            r"\begin{tabular}{cccc c c c c c c | cccc c c c c c c}"
            if tree_mode
            else r"\begin{tabular}{cccc c c c c c | cccc c c c c c}"
        ),
        r"\toprule",
        (
            r"\multicolumn{10}{c}{\text{Case 1} ($\HashLen \ge \kappa^{*}$)} & \multicolumn{10}{c}{\text{Case 2} ($\HashLen \ge 2\kappa^{*}$)} \\"
            if tree_mode
            else r"\multicolumn{9}{c}{\text{Case 1} ($\HashLen \ge \kappa^{*}$)} & \multicolumn{9}{c}{\text{Case 2} ($\HashLen \ge 2\kappa^{*}$)} \\"
        ),
        r"\cmidrule(lr){1-10}\cmidrule(lr){11-20}" if tree_mode else r"\cmidrule(lr){1-9}\cmidrule(lr){10-18}",
        (
            r"$\MaxGValue$ & $\PartitionNum$ & $\WindowRadius$ & $\TreeThreshold$ & $\PartitionNum\!\cdot\!\MaxGValue$ & $\BlockNum$ & $\HashLen$ & $|\PatternFamily|$ & $\kappa_T$ & $\kappa^{*}$ & $\MaxGValue$ & $\PartitionNum$ & $\WindowRadius$ & $\TreeThreshold$ & $\PartitionNum\!\cdot\!\MaxGValue$ & $\BlockNum$ & $\HashLen$ & $|\PatternFamily|$ & $\kappa_T$ & $\kappa^{*}$ \\"
            if tree_mode
            else r"$\MaxGValue$ & $\PartitionNum$ & $\WindowRadius$ & $\PartitionNum\!\cdot\!\MaxGValue$ & $n$ & $\HashLen$ & $\mathbb{E}[Re]$ & $\kappa$ & $\kappa^{*}$ & $\MaxGValue$ & $\PartitionNum$ & $\WindowRadius$ & $\PartitionNum\!\cdot\!\MaxGValue$ & $n$ & $\HashLen$ & $\mathbb{E}[Re]$ & $\kappa$ & $\kappa^{*}$ \\"
        ),
        r"\midrule",
    ]
    for target_index, security_target in enumerate(security_targets):
        if target_index:
            lines.append(r"\hline")
        lines.append(f"% ------- κ* = {security_target} -------")
        for max_g_value in max_g_values:
            left = by_key[("case1", security_target, max_g_value)]
            right = by_key[("case2", security_target, max_g_value)]
            if tree_mode:
                lines.append(
                    f"{left.max_g_value}  & {left.partition_num:2d} & "
                    f"{left.window_radius:2d} & "
                    f"{left.tree_threshold if left.tree_threshold is not None else 0:2d} & "
                    f"{left.partition_num * left.max_g_value:3d} & "
                    f"{left.block_num:3d} & {left.hash_len:3d} & "
                    f"{left.pattern_family_size or 0:2d} & {left.kappa:5.1f} & {left.security_target} "
                    f"& {right.max_g_value}  & {right.partition_num:2d} & "
                    f"{right.window_radius:2d} & "
                    f"{right.tree_threshold if right.tree_threshold is not None else 0:2d} & "
                    f"{right.partition_num * right.max_g_value:3d} & "
                    f"{right.block_num:3d} & {right.hash_len:3d} & "
                    f"{right.pattern_family_size or 0:2d} & {right.kappa:5.1f} & {right.security_target} \\\\"
                )
            else:
                lines.append(
                    f"{left.max_g_value}  & {left.partition_num:2d} & "
                    f"{left.window_radius:2d} & "
                    f"{left.partition_num * left.max_g_value:3d} & "
                    f"{left.block_num:3d} & {left.hash_len:3d} & "
                    f"{left.expected_retries:5.2f} & {left.kappa:5.1f} & {left.security_target} "
                    f"& {right.max_g_value}  & {right.partition_num:2d} & "
                    f"{right.window_radius:2d} & "
                    f"{right.partition_num * right.max_g_value:3d} & "
                    f"{right.block_num:3d} & {right.hash_len:3d} & "
                    f"{right.expected_retries:5.2f} & {right.kappa:5.1f} & {right.security_target} \\\\"
                )
    lines.extend(
        [
            r"\bottomrule",
            r"\end{tabular}",
            r"}",
            r"\vspace{-2mm}",
            r"\end{table}",
        ]
    )
    return "\n".join(lines)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Search TreeAwareISP YCSig parameter rows using exact "
            "acceptance and theorem-level UCR formulas."
        )
    )
    parser.add_argument(
        "--objective",
        choices=("partition_num", "sig_size"),
        default="partition_num",
        help="Primary search objective.",
    )
    parser.add_argument(
        "--cases",
        default="case1,case2",
        help="Comma-separated cases to search, e.g. case1,case2.",
    )
    parser.add_argument(
        "--security-targets",
        default="128,160,192",
        help="Comma-separated target kappa* values.",
    )
    parser.add_argument(
        "--max-g-values",
        default="4,8,16,32",
        help="Comma-separated MaxGValue choices.",
    )
    parser.add_argument(
        "--retry-limit",
        type=float,
        default=16.0,
        help="Maximum admissible E[Re].",
    )
    parser.add_argument(
        "--hash-len-max-factor",
        type=float,
        default=DEFAULT_HASH_LEN_MAX_FACTOR,
        help="Per-cell search cap as factor * kappa*.",
    )
    parser.add_argument(
        "--hash-len-max-absolute",
        type=int,
        default=None,
        help="Optional absolute HashLen upper bound overriding the factor cap.",
    )
    parser.add_argument(
        "--hash-len-min-absolute",
        type=int,
        default=None,
        help="Optional absolute HashLen lower bound for focused searches.",
    )
    parser.add_argument(
        "--partition-num-min",
        type=int,
        default=None,
        help="Optional inclusive PartitionNum lower bound for focused searches.",
    )
    parser.add_argument(
        "--partition-num-max",
        type=int,
        default=None,
        help="Optional inclusive PartitionNum upper bound for focused searches.",
    )
    parser.add_argument(
        "--window-radius-min",
        type=int,
        default=None,
        help="Optional inclusive WindowRadius lower bound for focused searches.",
    )
    parser.add_argument(
        "--window-radius-max",
        type=int,
        default=None,
        help="Optional inclusive WindowRadius upper bound for focused searches.",
    )
    parser.add_argument(
        "--link-threshold",
        type=int,
        default=-1,
        help="Deprecated compatibility option; ignored by the simplified-windowed VISP search.",
    )
    parser.add_argument(
        "--visp-mode",
        choices=("base", "tree"),
        default="tree",
        help="Use the legacy independent-row-set VISP formula or the TreeAwareISP tree-aware formula.",
    )
    parser.add_argument(
        "--pattern-family",
        choices=("aligned", "all"),
        default=DEFAULT_PATTERN_FAMILY,
        help="Public row-pattern family used when --visp-mode=tree.",
    )
    parser.add_argument(
        "--tree-threshold",
        type=int,
        default=None,
        help="Optional TreeScore upper bound used when --visp-mode=tree.",
    )
    parser.add_argument(
        "--tree-thresholds",
        default=None,
        help="Comma-separated TreeScore thresholds to enumerate when --visp-mode=tree.",
    )
    parser.add_argument(
        "--tree-threshold-min",
        type=int,
        default=None,
        help="Inclusive lower bound for enumerated TreeScore thresholds.",
    )
    parser.add_argument(
        "--tree-threshold-max",
        type=int,
        default=None,
        help="Inclusive upper bound for enumerated TreeScore thresholds.",
    )
    parser.add_argument(
        "--sig-size-samples",
        type=int,
        default=64,
        help="Accepted samples per candidate when objective=sig_size.",
    )
    parser.add_argument(
        "--sig-size-seed",
        type=int,
        default=0,
        help="Base RNG seed for the sig-size estimator.",
    )
    parser.add_argument(
        "--sig-partition-num-slack",
        type=int,
        default=None,
        help=(
            "Optional cap keeping only candidates with partition_num within this "
            "additive slack from the minimum feasible partition_num when "
            "objective=sig_size."
        ),
    )
    parser.add_argument(
        "--tree-setup-json",
        default=None,
        help=(
            "Optional JSON object merged into the tree-mode YCSig/TreeAwareISP setup, "
            "for example '{\"prefix_limit\":32,\"loss_bound\":12,\"prefix_dict\":[[0,1,2,3]]}'."
        ),
    )
    parser.add_argument(
        "--format",
        choices=("json", "text", "latex"),
        default="text",
        help="Output format.",
    )
    return parser


def _main() -> int:
    args = _build_parser().parse_args()
    tree_setup_kwargs = None
    if args.tree_setup_json is not None:
        tree_setup_kwargs = json.loads(args.tree_setup_json)
        if not isinstance(tree_setup_kwargs, dict):
            raise ValueError("--tree-setup-json must decode to a JSON object")
    tree_thresholds = _parse_optional_int_csv(args.tree_thresholds)
    if tree_thresholds is None and (
        args.tree_threshold_min is not None or args.tree_threshold_max is not None
    ):
        if args.tree_threshold_min is None or args.tree_threshold_max is None:
            raise ValueError("--tree-threshold-min and --tree-threshold-max must be used together")
        if args.tree_threshold_max < args.tree_threshold_min:
            raise ValueError("--tree-threshold-max must be at least --tree-threshold-min")
        tree_thresholds = list(range(args.tree_threshold_min, args.tree_threshold_max + 1))
    rows = search_rows(
        cases=_parse_case_csv(args.cases),
        security_targets=_parse_int_csv(args.security_targets),
        max_g_values=_parse_int_csv(args.max_g_values),
        objective=args.objective,
        retry_limit=args.retry_limit,
        hash_len_max_factor=args.hash_len_max_factor,
        hash_len_max_absolute=args.hash_len_max_absolute,
        hash_len_min_absolute=args.hash_len_min_absolute,
        link_threshold=args.link_threshold,
        visp_mode=args.visp_mode,
        pattern_family_name=args.pattern_family,
        tree_threshold=args.tree_threshold,
        tree_thresholds=tree_thresholds,
        partition_num_min=args.partition_num_min,
        partition_num_max=args.partition_num_max,
        window_radius_min=args.window_radius_min,
        window_radius_max=args.window_radius_max,
        sig_size_samples=args.sig_size_samples,
        sig_size_seed=args.sig_size_seed,
        sig_partition_num_slack=args.sig_partition_num_slack,
        tree_setup_kwargs=tree_setup_kwargs,
    )
    if args.format == "json":
        print(json.dumps([asdict(row) for row in rows], ensure_ascii=True, indent=2))
    elif args.format == "latex":
        print(
            render_latex(
                rows,
                retry_limit=args.retry_limit,
                objective=args.objective,
            )
        )
    else:
        print(render_text(rows))
    return 0


if __name__ == "__main__":
    raise SystemExit(_main())
