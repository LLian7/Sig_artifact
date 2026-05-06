from __future__ import annotations

import argparse
import json
import math
import statistics
from dataclasses import asdict, dataclass, replace
from functools import lru_cache
from random import Random
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

from benchmark_ycsig import (
    YCSigBenchmarkCase,
    _derive_key_seed_for_sample,
    _message_for_sample,
    _randomizer_for_sample,
)
from benchmark_ycsig_ops import (
    _aggregate_counter_dicts,
    _build_scheme,
    _case_for_repetition,
    _measure_counters,
    _top_level_backend_hash_calls,
    _top_level_ops,
)
from pprf import PPRF
from search_ycsig_sigsize import _histogram_probability_terms_cached
from treeaware_isp import (
    TreeAwareISPParameters,
    route_support,
    tree_score,
    treeaware_isp,
    verify_score,
)


_TEX_PENDING = r"\Pending"
_TEX_NA = r"\NA"
_TEX_INFINITY = r"$\infty$"
_DEFAULT_RESULTS_CACHE = "vispt_search_results.json"


@dataclass(frozen=True)
class VISPTTableRowSpec:
    case_name: str
    goal: str
    security_target: int
    max_g_value: int
    partition_num: Optional[int] = None
    window_radius: Optional[int] = None
    entropy_floor: Optional[int] = None
    entropy_floor_tex: Optional[str] = None
    size_threshold: Optional[int] = None
    size_threshold_tex: Optional[str] = None
    vrf_threshold: Optional[int] = None
    vrf_threshold_tex: Optional[str] = None
    hash_len: Optional[int] = None
    expected_retries: Optional[float] = None
    kappa: Optional[float] = None

    @property
    def is_pending(self) -> bool:
        return self.partition_num is None

    @property
    def max_g_bit(self) -> int:
        return self.max_g_value.bit_length() - 1

    @property
    def block_num(self) -> Optional[int]:
        if self.hash_len is None:
            return None
        return self.hash_len // self.max_g_bit

    @property
    def leaf_universe_size(self) -> Optional[int]:
        if self.partition_num is None:
            return None
        return self.partition_num * self.max_g_value

    @property
    def label(self) -> str:
        return (
            f"VISPT-{self.case_name}-{self.goal}-"
            f"w{self.max_g_value}-k{self.security_target}"
            + (
                ""
                if self.partition_num is None
                else f"-P{self.partition_num}-R{self.window_radius}-H{self.hash_len}"
            )
        )


@dataclass(frozen=True)
class VISPTMeasuredRow:
    spec: VISPTTableRowSpec
    keygen: float
    sign: float
    verify: float
    sig_size: float
    verify_rate: float
    raw_result: Mapping[str, Any]


@dataclass(frozen=True)
class VISPTSearchCell:
    case_name: str
    goal: str
    security_target: int
    max_g_value: int
    variant: str = "best"


@dataclass(frozen=True)
class VISPTSearchConfig:
    retry_limit: float = 256.0
    retry_slack: float = 64.0
    hash_len_max_factor_case1: float = 1.5
    hash_len_max_factor_case2: float = 1.25
    stop_after_candidates: int = 0
    window_radius_max_size: int = 2
    window_radius_max_verify: int = 1
    partition_slack_verify: int = 8
    partition_head_span_size: int = 32
    partition_tail_span_size: int = 16
    pilot_trials: int = 2048
    pilot_accepted_samples: int = 32
    acceptance_trials: int = 4096
    final_acceptance_trials: int = 32768
    finalist_count: int = 3
    benchmark_samples: int = 8
    benchmark_repetitions: int = 1


@dataclass(frozen=True)
class VISPTSearchCandidate:
    case_name: str
    goal: str
    security_target: int
    max_g_value: int
    partition_num: int
    window_radius: int
    entropy_floor: Optional[int]
    size_threshold: Optional[int]
    vrf_threshold: Optional[int]
    hash_len: int
    expected_retries: float
    kappa: float
    proxy_score: float


GroupsKey = tuple[tuple[int, ...], ...]


def _certified_row(
    *,
    case_name: str,
    goal: str,
    security_target: int,
    max_g_value: int,
    partition_num: int,
    window_radius: int,
    entropy_floor: Optional[int],
    entropy_floor_tex: Optional[str],
    size_threshold: Optional[int],
    size_threshold_tex: Optional[str],
    vrf_threshold: Optional[int],
    vrf_threshold_tex: Optional[str],
    hash_len: int,
    expected_retries: float,
    kappa: float,
) -> VISPTTableRowSpec:
    return VISPTTableRowSpec(
        case_name=case_name,
        goal=goal,
        security_target=security_target,
        max_g_value=max_g_value,
        partition_num=partition_num,
        window_radius=window_radius,
        entropy_floor=entropy_floor,
        entropy_floor_tex=entropy_floor_tex,
        size_threshold=size_threshold,
        size_threshold_tex=size_threshold_tex,
        vrf_threshold=vrf_threshold,
        vrf_threshold_tex=vrf_threshold_tex,
        hash_len=hash_len,
        expected_retries=expected_retries,
        kappa=kappa,
    )


def _pending_row(
    *,
    case_name: str,
    goal: str,
    security_target: int,
    max_g_value: int,
    entropy_floor_tex: Optional[str],
    size_threshold_tex: Optional[str],
    vrf_threshold_tex: Optional[str],
) -> VISPTTableRowSpec:
    return VISPTTableRowSpec(
        case_name=case_name,
        goal=goal,
        security_target=security_target,
        max_g_value=max_g_value,
        entropy_floor_tex=entropy_floor_tex,
        size_threshold_tex=size_threshold_tex,
        vrf_threshold_tex=vrf_threshold_tex,
    )


def vispt_table_sections() -> List[List[VISPTTableRowSpec]]:
    sections: List[List[VISPTTableRowSpec]] = []

    size_case1: List[VISPTTableRowSpec] = []
    for max_g_value in (4, 16, 64, 128, 256):
        if max_g_value == 4:
            size_case1.append(
                _certified_row(
                    case_name="case1",
                    goal="Size",
                    security_target=128,
                    max_g_value=4,
                    partition_num=64,
                    window_radius=1,
                    entropy_floor=115,
                    entropy_floor_tex=None,
                    size_threshold=80,
                    size_threshold_tex=None,
                    vrf_threshold=None,
                    vrf_threshold_tex=_TEX_INFINITY,
                    hash_len=128,
                    expected_retries=55.2,
                    kappa=129.6,
                )
            )
            for security_target in (160, 192):
                size_case1.append(
                    _pending_row(
                        case_name="case1",
                        goal="Size",
                        security_target=security_target,
                        max_g_value=max_g_value,
                        entropy_floor_tex=_TEX_PENDING,
                        size_threshold_tex=_TEX_PENDING,
                        vrf_threshold_tex=_TEX_INFINITY,
                    )
                )
            continue
        for security_target in (128, 160, 192):
            size_case1.append(
                _pending_row(
                    case_name="case1",
                    goal="Size",
                    security_target=security_target,
                    max_g_value=max_g_value,
                    entropy_floor_tex=_TEX_PENDING,
                    size_threshold_tex=_TEX_PENDING,
                    vrf_threshold_tex=_TEX_INFINITY,
                )
            )
    sections.append(size_case1)

    size_case2: List[VISPTTableRowSpec] = []
    for max_g_value in (4, 16, 64, 128, 256):
        if max_g_value == 4:
            size_case2.extend(
                [
                    _certified_row(
                        case_name="case2",
                        goal="Size",
                        security_target=128,
                        max_g_value=4,
                        partition_num=56,
                        window_radius=1,
                        entropy_floor=112,
                        entropy_floor_tex=None,
                        size_threshold=74,
                        size_threshold_tex=None,
                        vrf_threshold=None,
                        vrf_threshold_tex=_TEX_INFINITY,
                        hash_len=256,
                        expected_retries=213.3,
                        kappa=129.2,
                    ),
                    _certified_row(
                        case_name="case2",
                        goal="Size",
                        security_target=128,
                        max_g_value=4,
                        partition_num=64,
                        window_radius=1,
                        entropy_floor=112,
                        entropy_floor_tex=None,
                        size_threshold=72,
                        size_threshold_tex=None,
                        vrf_threshold=None,
                        vrf_threshold_tex=_TEX_INFINITY,
                        hash_len=256,
                        expected_retries=188.3,
                        kappa=129.9,
                    ),
                ]
            )
            for security_target in (160, 192):
                size_case2.append(
                    _pending_row(
                        case_name="case2",
                        goal="Size",
                        security_target=security_target,
                        max_g_value=max_g_value,
                        entropy_floor_tex=_TEX_PENDING,
                        size_threshold_tex=_TEX_PENDING,
                        vrf_threshold_tex=_TEX_INFINITY,
                    )
                )
            continue
        for security_target in (128, 160, 192):
            size_case2.append(
                _pending_row(
                    case_name="case2",
                    goal="Size",
                    security_target=security_target,
                    max_g_value=max_g_value,
                    entropy_floor_tex=_TEX_PENDING,
                    size_threshold_tex=_TEX_PENDING,
                    vrf_threshold_tex=_TEX_INFINITY,
                )
            )
    sections.append(size_case2)

    verify_case1: List[VISPTTableRowSpec] = []
    for max_g_value in (4, 16):
        for security_target in (128, 160, 192):
            verify_case1.append(
                _pending_row(
                    case_name="case1",
                    goal="Verify",
                    security_target=security_target,
                    max_g_value=max_g_value,
                    entropy_floor_tex=_TEX_PENDING,
                    size_threshold_tex=_TEX_INFINITY,
                    vrf_threshold_tex=_TEX_PENDING,
                )
            )
    verify_case1.append(
        _certified_row(
            case_name="case1",
            goal="Verify",
            security_target=128,
            max_g_value=64,
            partition_num=7,
            window_radius=0,
            entropy_floor=None,
            entropy_floor_tex=_TEX_NA,
            size_threshold=None,
            size_threshold_tex=_TEX_INFINITY,
            vrf_threshold=118,
            vrf_threshold_tex=None,
            hash_len=132,
            expected_retries=211.9,
            kappa=131.6,
        )
    )
    for security_target in (160, 192):
        verify_case1.append(
            _pending_row(
                case_name="case1",
                goal="Verify",
                security_target=security_target,
                max_g_value=64,
                entropy_floor_tex=_TEX_NA,
                size_threshold_tex=_TEX_INFINITY,
                vrf_threshold_tex=_TEX_PENDING,
            )
        )
    for security_target in (128, 160, 192):
        verify_case1.append(
            _pending_row(
                case_name="case1",
                goal="Verify",
                security_target=security_target,
                max_g_value=128,
                entropy_floor_tex=_TEX_NA,
                size_threshold_tex=_TEX_INFINITY,
                vrf_threshold_tex=_TEX_PENDING,
            )
        )
    verify_case1.append(
        _certified_row(
            case_name="case1",
            goal="Verify",
            security_target=128,
            max_g_value=256,
            partition_num=5,
            window_radius=0,
            entropy_floor=None,
            entropy_floor_tex=_TEX_NA,
            size_threshold=None,
            size_threshold_tex=_TEX_INFINITY,
            vrf_threshold=106,
            vrf_threshold_tex=None,
            hash_len=128,
            expected_retries=177.0,
            kappa=128.4,
        )
    )
    for security_target in (160, 192):
        verify_case1.append(
            _pending_row(
                case_name="case1",
                goal="Verify",
                security_target=security_target,
                max_g_value=256,
                entropy_floor_tex=_TEX_NA,
                size_threshold_tex=_TEX_INFINITY,
                vrf_threshold_tex=_TEX_PENDING,
            )
        )
    sections.append(verify_case1)

    verify_case2: List[VISPTTableRowSpec] = []
    for max_g_value in (4, 16):
        for security_target in (128, 160, 192):
            verify_case2.append(
                _pending_row(
                    case_name="case2",
                    goal="Verify",
                    security_target=security_target,
                    max_g_value=max_g_value,
                    entropy_floor_tex=_TEX_PENDING,
                    size_threshold_tex=_TEX_INFINITY,
                    vrf_threshold_tex=_TEX_PENDING,
                )
            )
    for max_g_value in (64, 128, 256):
        for security_target in (128, 160, 192):
            verify_case2.append(
                _pending_row(
                    case_name="case2",
                    goal="Verify",
                    security_target=security_target,
                    max_g_value=max_g_value,
                    entropy_floor_tex=_TEX_NA,
                    size_threshold_tex=_TEX_INFINITY,
                    vrf_threshold_tex=_TEX_PENDING,
                )
            )
    sections.append(verify_case2)

    return sections


def vispt_table_rows() -> List[VISPTTableRowSpec]:
    rows: List[VISPTTableRowSpec] = []
    for section in vispt_table_sections():
        rows.extend(section)
    return rows


def vispt_search_sections() -> List[List[VISPTSearchCell]]:
    sections: List[List[VISPTSearchCell]] = []

    sections.append(
        [
            VISPTSearchCell("case1", "Size", security_target, max_g_value)
            for max_g_value in (4, 16, 64, 128, 256)
            for security_target in (128, 160, 192)
        ]
    )
    sections.append(
        [
            VISPTSearchCell("case2", "Size", 128, 4, variant)
            for variant in ("best", "small_leaf")
        ]
        + [
            VISPTSearchCell("case2", "Size", security_target, max_g_value)
            for max_g_value in (4, 16, 64, 128, 256)
            for security_target in (128, 160, 192)
            if not (max_g_value == 4 and security_target == 128)
        ]
    )
    sections.append(
        [
            VISPTSearchCell("case1", "Verify", security_target, max_g_value)
            for max_g_value in (4, 16, 64, 128, 256)
            for security_target in (128, 160, 192)
        ]
    )
    sections.append(
        [
            VISPTSearchCell("case2", "Verify", security_target, max_g_value)
            for max_g_value in (4, 16, 64, 128, 256)
            for security_target in (128, 160, 192)
        ]
    )
    return sections


def _parse_csv_strings(raw: Optional[str]) -> Optional[set[str]]:
    if raw is None:
        return None
    values = {piece.strip() for piece in raw.split(",") if piece.strip()}
    return None if not values else values


def _parse_csv_ints(raw: Optional[str]) -> Optional[set[int]]:
    values = _parse_csv_strings(raw)
    if values is None:
        return None
    return {int(value) for value in values}


def _filter_search_sections(
    sections: Sequence[Sequence[VISPTSearchCell]],
    *,
    cases: Optional[set[str]] = None,
    goals: Optional[set[str]] = None,
    security_targets: Optional[set[int]] = None,
    max_g_values: Optional[set[int]] = None,
) -> List[List[VISPTSearchCell]]:
    filtered_sections: List[List[VISPTSearchCell]] = []
    for section in sections:
        filtered_section = [
            cell
            for cell in section
            if (cases is None or cell.case_name in cases)
            and (goals is None or cell.goal in goals)
            and (security_targets is None or cell.security_target in security_targets)
            and (max_g_values is None or cell.max_g_value in max_g_values)
        ]
        if filtered_section:
            filtered_sections.append(filtered_section)
    return filtered_sections


def select_vispt_rows(
    labels: Optional[Sequence[str]] = None,
) -> List[VISPTTableRowSpec]:
    rows = vispt_table_rows()
    if labels is None:
        return rows
    label_set = set(labels)
    return [row for row in rows if row.label in label_set]


def _cell_min_hash_len(case_name: str, security_target: int) -> int:
    if case_name == "case1":
        return security_target
    if case_name == "case2":
        return 2 * security_target
    raise ValueError(f"unsupported case_name={case_name!r}")


def _hash_len_candidates(
    cell: VISPTSearchCell,
    config: VISPTSearchConfig,
) -> Iterable[int]:
    max_g_bit = cell.max_g_value.bit_length() - 1
    min_hash_len = _cell_min_hash_len(cell.case_name, cell.security_target)
    if cell.case_name == "case1":
        max_hash_len = int(math.floor(min_hash_len * config.hash_len_max_factor_case1))
    elif cell.case_name == "case2":
        max_hash_len = int(math.floor(min_hash_len * config.hash_len_max_factor_case2))
    else:
        raise ValueError(f"unsupported case_name={cell.case_name!r}")

    start = min_hash_len + ((max_g_bit - (min_hash_len % max_g_bit)) % max_g_bit)
    aligned_max = max_hash_len - (max_hash_len % max_g_bit)
    candidates = set(range(start, max(start, aligned_max) + 1, max_g_bit))

    # Always keep known-good static seed rows in range, even if they lie off the
    # default factor cap for this search cell.
    for seed in _static_seed_specs_by_base_key().get(_cell_base_key(cell), ()):
        if seed.hash_len is not None:
            candidates.add(seed.hash_len)

    return tuple(sorted(candidates))


def _partition_candidates(
    cell: VISPTSearchCell,
    block_num: int,
    config: VISPTSearchConfig,
) -> Iterable[int]:
    min_partition_num = math.ceil(block_num / cell.max_g_value)
    if cell.goal == "Verify":
        max_partition_num = min(
            block_num,
            min_partition_num + config.partition_slack_verify,
        )
        return range(min_partition_num, max_partition_num + 1)

    candidates = set(range(min_partition_num, min(min_partition_num + 4, block_num) + 1))
    tail_start = max(min_partition_num, block_num - min(8, config.partition_tail_span_size))
    candidates.update(range(tail_start, block_num + 1))
    for numerator in range(1, 9):
        candidate = math.ceil(block_num * numerator / 8)
        if min_partition_num <= candidate <= block_num:
            candidates.add(candidate)
    return tuple(sorted(candidates))


def _window_radius_candidates(
    cell: VISPTSearchCell,
    block_num: int,
    config: VISPTSearchConfig,
) -> Iterable[int]:
    max_radius = block_num // cell.max_g_value
    cap = config.window_radius_max_verify if cell.goal == "Verify" else config.window_radius_max_size
    return range(0, min(max_radius, cap) + 1)


def _entropy_floor_candidates(
    partition_num: int,
) -> tuple[int, ...]:
    support_bits_cap = math.lgamma(partition_num + 1) / math.log(2.0)
    ratios = (0.0, 0.25, 0.35, 0.45, 0.55)
    values = {max(0, int(round(support_bits_cap * ratio))) for ratio in ratios}
    return tuple(sorted(values))


def _aux_t_for_entropy_floor(entropy_floor: Optional[int]) -> Dict[str, Any]:
    aux_t: Dict[str, Any] = {"profile_rule": "dyadic_greedy"}
    if entropy_floor is not None:
        aux_t["entropy_floor"] = entropy_floor
    return aux_t


@lru_cache(maxsize=None)
def _exact_vispt_support_metrics(
    block_num: int,
    max_g_value: int,
    partition_num: int,
    window_radius: int,
    entropy_floor: Optional[int],
) -> tuple[float, float]:
    max_g_bit = max_g_value.bit_length() - 1
    params = TreeAwareISPParameters(
        hash_len=block_num * max_g_bit,
        max_g_bit=max_g_bit,
        partition_num=partition_num,
        window_radius=window_radius,
        aux_t=_aux_t_for_entropy_floor(entropy_floor),
        mode="size",
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


def _candidate_params(
    *,
    hash_len: int,
    max_g_value: int,
    partition_num: int,
    window_radius: int,
    entropy_floor: Optional[int],
    size_threshold: Optional[int],
    vrf_threshold: Optional[int],
) -> TreeAwareISPParameters:
    return TreeAwareISPParameters(
        hash_len=hash_len,
        max_g_bit=max_g_value.bit_length() - 1,
        partition_num=partition_num,
        window_radius=window_radius,
        aux_t=_aux_t_for_entropy_floor(entropy_floor),
        size_threshold=size_threshold,
        vrf_threshold=vrf_threshold,
        mode="size",
    )


def _estimate_acceptance_probability(
    params: TreeAwareISPParameters,
    *,
    trials: int,
    seed: int,
) -> float:
    acceptance, _ = _estimate_acceptance_with_group_samples(
        params,
        trials=trials,
        seed=seed,
        collect_group_count=0,
    )
    return acceptance


def _estimate_acceptance_with_group_samples(
    params: TreeAwareISPParameters,
    *,
    trials: int,
    seed: int,
    collect_group_count: int,
) -> tuple[float, tuple[GroupsKey, ...]]:
    rng = Random(seed)
    accepted = 0
    group_samples: List[GroupsKey] = []
    for _ in range(trials):
        partition_value = rng.getrandbits(params.hash_len)
        groups = treeaware_isp(partition_value, params)
        if groups is not None:
            accepted += 1
            if len(group_samples) < collect_group_count:
                group_samples.append(tuple(tuple(subgroup) for subgroup in groups))
    acceptance = accepted / trials if trials > 0 else 0.0
    return acceptance, tuple(group_samples)


def _pilot_scores(
    cell: VISPTSearchCell,
    params: TreeAwareISPParameters,
    *,
    trials: int,
    accepted_samples: int,
    seed: int,
) -> List[int]:
    rng = Random(seed)
    scores: List[int] = []
    for _ in range(trials):
        partition_value = rng.getrandbits(params.hash_len)
        groups = treeaware_isp(partition_value, params)
        if groups is None:
            continue
        if cell.goal == "Size":
            scores.append(tree_score(groups, params))
        else:
            scores.append(verify_score(groups, params))
        if len(scores) >= accepted_samples:
            break
    return scores


def _threshold_candidates_from_scores(
    scores: Sequence[int],
    *,
    goal: str,
) -> tuple[int, ...]:
    if not scores:
        return ()
    ordered = sorted(scores)
    quantiles = (
        (0.10, 0.20, 0.30, 0.40, 0.50)
        if goal == "Size"
        else (0.05, 0.10, 0.15, 0.20, 0.25, 0.30)
    )
    thresholds = {ordered[0], ordered[-1]}
    for quantile in quantiles:
        index = min(len(ordered) - 1, max(0, int(round((len(ordered) - 1) * quantile))))
        thresholds.add(ordered[index])
    return tuple(sorted(thresholds))


def _pending_spec_for_cell(cell: VISPTSearchCell) -> VISPTTableRowSpec:
    if cell.goal == "Verify":
        entropy_tex = _TEX_NA if cell.max_g_value >= 64 else _TEX_PENDING
        return _pending_row(
            case_name=cell.case_name,
            goal=cell.goal,
            security_target=cell.security_target,
            max_g_value=cell.max_g_value,
            entropy_floor_tex=entropy_tex,
            size_threshold_tex=_TEX_INFINITY,
            vrf_threshold_tex=_TEX_PENDING,
        )
    return _pending_row(
        case_name=cell.case_name,
        goal=cell.goal,
        security_target=cell.security_target,
        max_g_value=cell.max_g_value,
        entropy_floor_tex=_TEX_PENDING,
        size_threshold_tex=_TEX_PENDING,
        vrf_threshold_tex=_TEX_INFINITY,
    )


@lru_cache(maxsize=1)
def _static_seed_specs_by_base_key() -> Dict[
    tuple[str, str, int, int],
    tuple[VISPTTableRowSpec, ...],
]:
    grouped: Dict[tuple[str, str, int, int], List[VISPTTableRowSpec]] = {}
    for spec in vispt_table_rows():
        if spec.is_pending:
            continue
        base_key = (
            spec.case_name,
            spec.goal,
            spec.security_target,
            spec.max_g_value,
        )
        grouped.setdefault(base_key, []).append(spec)
    return {
        base_key: tuple(rows)
        for base_key, rows in grouped.items()
    }


def _spec_from_candidate(candidate: VISPTSearchCandidate) -> VISPTTableRowSpec:
    if candidate.goal == "Verify":
        return VISPTTableRowSpec(
            case_name=candidate.case_name,
            goal=candidate.goal,
            security_target=candidate.security_target,
            max_g_value=candidate.max_g_value,
            partition_num=candidate.partition_num,
            window_radius=candidate.window_radius,
            entropy_floor=None,
            entropy_floor_tex=_TEX_NA,
            size_threshold=None,
            size_threshold_tex=_TEX_INFINITY,
            vrf_threshold=candidate.vrf_threshold,
            vrf_threshold_tex=None,
            hash_len=candidate.hash_len,
            expected_retries=candidate.expected_retries,
            kappa=candidate.kappa,
        )
    return VISPTTableRowSpec(
        case_name=candidate.case_name,
        goal=candidate.goal,
        security_target=candidate.security_target,
        max_g_value=candidate.max_g_value,
        partition_num=candidate.partition_num,
        window_radius=candidate.window_radius,
        entropy_floor=candidate.entropy_floor,
        entropy_floor_tex=None,
        size_threshold=candidate.size_threshold,
        size_threshold_tex=None,
        vrf_threshold=None,
        vrf_threshold_tex=_TEX_INFINITY,
        hash_len=candidate.hash_len,
        expected_retries=candidate.expected_retries,
        kappa=candidate.kappa,
    )


def _retry_cap(config: VISPTSearchConfig) -> float:
    return config.retry_limit + max(0.0, config.retry_slack)


def _retry_distance(retries: float, target: float) -> float:
    return abs(retries - target)


def _seed_mix(*pieces: int) -> int:
    state = 0x9E3779B185EBCA87
    mask = (1 << 64) - 1
    for piece in pieces:
        state ^= int(piece) + 0x9E3779B97F4A7C15 + ((state << 6) & mask) + (state >> 2)
        state &= mask
    return state


def _cell_base_key(cell: VISPTSearchCell) -> tuple[str, str, int, int]:
    return (
        cell.case_name,
        cell.goal,
        cell.security_target,
        cell.max_g_value,
    )


def _base_key_to_text(base_key: tuple[str, str, int, int]) -> str:
    case_name, goal, security_target, max_g_value = base_key
    return f"{case_name}|{goal}|{security_target}|{max_g_value}"


def _base_key_from_text(raw: str) -> tuple[str, str, int, int]:
    case_name, goal, security_target, max_g_value = raw.split("|")
    return (case_name, goal, int(security_target), int(max_g_value))


def _candidate_leaf_universe_size(candidate: VISPTSearchCandidate) -> int:
    return candidate.partition_num * candidate.max_g_value


def _goal_metric_from_measured(row: VISPTMeasuredRow) -> float:
    return row.sig_size if row.spec.goal == "Size" else row.verify


def _candidate_proxy_sort_key(
    candidate: VISPTSearchCandidate,
    *,
    retry_target: float,
) -> tuple[float, float, int, int, int]:
    return (
        candidate.proxy_score,
        _retry_distance(candidate.expected_retries, retry_target),
        _candidate_leaf_universe_size(candidate),
        candidate.hash_len,
        candidate.partition_num,
    )


def _candidate_leaf_sort_key(
    candidate: VISPTSearchCandidate,
    *,
    retry_target: float,
) -> tuple[int, float, float, int, int]:
    return (
        _candidate_leaf_universe_size(candidate),
        candidate.proxy_score,
        _retry_distance(candidate.expected_retries, retry_target),
        candidate.hash_len,
        candidate.partition_num,
    )


def _measured_row_sort_key(
    row: VISPTMeasuredRow,
    *,
    retry_target: float,
) -> tuple[float, float, int, int, int]:
    return (
        _goal_metric_from_measured(row),
        _retry_distance(row.spec.expected_retries or math.inf, retry_target),
        row.spec.leaf_universe_size or math.inf,
        row.spec.hash_len or math.inf,
        row.spec.partition_num or math.inf,
    )


def _measured_row_small_leaf_sort_key(
    row: VISPTMeasuredRow,
    *,
    retry_target: float,
) -> tuple[int, float, float, int, int]:
    return (
        row.spec.leaf_universe_size or math.inf,
        _goal_metric_from_measured(row),
        _retry_distance(row.spec.expected_retries or math.inf, retry_target),
        row.spec.hash_len or math.inf,
        row.spec.partition_num or math.inf,
    )


def _candidate_acceptance_probability(
    candidate: VISPTSearchCandidate,
    *,
    trials: int,
    seed: int,
) -> float:
    params = _candidate_params(
        hash_len=candidate.hash_len,
        max_g_value=candidate.max_g_value,
        partition_num=candidate.partition_num,
        window_radius=candidate.window_radius,
        entropy_floor=candidate.entropy_floor,
        size_threshold=candidate.size_threshold,
        vrf_threshold=candidate.vrf_threshold,
    )
    return _estimate_acceptance_probability(params, trials=trials, seed=seed)


def _seed_neighborhood_candidates(
    cell: VISPTSearchCell,
    block_num: int,
    config: VISPTSearchConfig,
) -> Optional[tuple[tuple[int, int, Optional[int]], ...]]:
    seeds = _static_seed_specs_by_base_key().get(_cell_base_key(cell))
    if not seeds:
        return None

    candidates: set[tuple[int, int, Optional[int]]] = set()
    min_partition_num = math.ceil(block_num / cell.max_g_value)
    if cell.goal == "Verify":
        max_partition_num = min(block_num, min_partition_num + config.partition_slack_verify)
    else:
        max_partition_num = block_num
    max_radius = min(
        block_num // cell.max_g_value,
        config.window_radius_max_verify if cell.goal == "Verify" else config.window_radius_max_size,
    )

    for seed in seeds:
        partition_offsets = (0, -8, 8)
        radius_values = {
            radius
            for radius in (seed.window_radius,)
            if radius is not None and 0 <= radius <= max_radius
        }
        for partition_offset in partition_offsets:
            partition_num = seed.partition_num + partition_offset
            if partition_num < min_partition_num or partition_num > max_partition_num:
                continue
            for window_radius in radius_values:
                if cell.goal == "Verify":
                    candidates.add((partition_num, window_radius, None))
                    continue
                candidates.add((partition_num, window_radius, seed.entropy_floor))
    return tuple(sorted(candidates))


def _serialize_measured_row(row: VISPTMeasuredRow) -> Dict[str, Any]:
    return {
        "spec": asdict(row.spec),
        "keygen": row.keygen,
        "sign": row.sign,
        "verify": row.verify,
        "sig_size": row.sig_size,
        "verify_rate": row.verify_rate,
        "raw_result": dict(row.raw_result),
    }


def _deserialize_measured_row(payload: Mapping[str, Any]) -> VISPTMeasuredRow:
    spec = VISPTTableRowSpec(**dict(payload["spec"]))
    return VISPTMeasuredRow(
        spec=spec,
        keygen=float(payload["keygen"]),
        sign=float(payload["sign"]),
        verify=float(payload["verify"]),
        sig_size=float(payload["sig_size"]),
        verify_rate=float(payload["verify_rate"]),
        raw_result=dict(payload.get("raw_result", {})),
    )


def load_vispt_results_cache(
    path: str,
) -> Dict[tuple[str, str, int, int], Dict[str, VISPTMeasuredRow]]:
    try:
        with open(path, "r", encoding="utf-8") as handle:
            payload = json.load(handle)
    except FileNotFoundError:
        return {}

    rows_payload = payload.get("rows", {})
    loaded: Dict[tuple[str, str, int, int], Dict[str, VISPTMeasuredRow]] = {}
    for base_key_text, variants_payload in rows_payload.items():
        base_key = _base_key_from_text(base_key_text)
        variants: Dict[str, VISPTMeasuredRow] = {}
        for variant, row_payload in dict(variants_payload).items():
            variants[str(variant)] = _deserialize_measured_row(row_payload)
        if variants:
            loaded[base_key] = variants
    return loaded


def save_vispt_results_cache(
    path: str,
    resolved_rows: Mapping[tuple[str, str, int, int], Mapping[str, VISPTMeasuredRow]],
) -> None:
    payload = {
        "rows": {
            _base_key_to_text(base_key): {
                variant: _serialize_measured_row(row)
                for variant, row in sorted(variants.items())
            }
            for base_key, variants in sorted(resolved_rows.items())
            if variants
        }
    }
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=True, indent=2, sort_keys=True)


def _search_base_cell_candidates(
    cell: VISPTSearchCell,
    config: VISPTSearchConfig,
    *,
    random_seed: int,
) -> List[VISPTSearchCandidate]:
    candidates: Dict[
        tuple[int, int, int, Optional[int], Optional[int], Optional[int]],
        VISPTSearchCandidate,
    ] = {}
    retry_cap = _retry_cap(config)

    for hash_len in _hash_len_candidates(cell, config):
        block_num = hash_len // (cell.max_g_value.bit_length() - 1)
        seeded_triples = _seed_neighborhood_candidates(cell, block_num, config)
        if seeded_triples is None:
            triples: Iterable[tuple[int, int, Optional[int]]] = (
                (partition_num, window_radius, entropy_floor)
                for partition_num in _partition_candidates(cell, block_num, config)
                for window_radius in _window_radius_candidates(cell, block_num, config)
                for entropy_floor in (
                    (None,)
                    if cell.goal == "Verify"
                    else _entropy_floor_candidates(partition_num)
                )
            )
        else:
            triples = seeded_triples

        for partition_num, window_radius, entropy_floor in triples:
                    support_acceptance, support_kappa = _exact_vispt_support_metrics(
                        block_num,
                        cell.max_g_value,
                        partition_num,
                        window_radius,
                        entropy_floor,
                    )
                    if support_acceptance <= 0.0 or support_kappa < cell.security_target:
                        continue
                    support_retries = 1.0 / support_acceptance
                    if support_retries > retry_cap:
                        continue

                    base_params = _candidate_params(
                        hash_len=hash_len,
                        max_g_value=cell.max_g_value,
                        partition_num=partition_num,
                        window_radius=window_radius,
                        entropy_floor=entropy_floor,
                        size_threshold=None,
                        vrf_threshold=None,
                    )
                    seed_base = _seed_mix(
                        random_seed,
                        hash_len,
                        cell.max_g_value,
                        partition_num,
                        window_radius,
                        -1 if entropy_floor is None else entropy_floor,
                    )
                    scores = _pilot_scores(
                        cell,
                        base_params,
                        trials=config.pilot_trials,
                        accepted_samples=config.pilot_accepted_samples,
                        seed=seed_base,
                    )
                    thresholds = _threshold_candidates_from_scores(scores, goal=cell.goal)
                    for threshold in thresholds:
                        filtered_scores = [score for score in scores if score <= threshold]
                        if not filtered_scores:
                            continue
                        size_threshold = threshold if cell.goal == "Size" else None
                        vrf_threshold = threshold if cell.goal == "Verify" else None
                        guarded_params = _candidate_params(
                            hash_len=hash_len,
                            max_g_value=cell.max_g_value,
                            partition_num=partition_num,
                            window_radius=window_radius,
                            entropy_floor=entropy_floor,
                            size_threshold=size_threshold,
                            vrf_threshold=vrf_threshold,
                        )
                        acceptance = _estimate_acceptance_probability(
                            guarded_params,
                            trials=config.acceptance_trials,
                            seed=seed_base ^ threshold,
                        )
                        if acceptance <= 0.0:
                            continue
                        expected_retries = 1.0 / acceptance
                        if expected_retries > retry_cap:
                            continue
                        candidate = VISPTSearchCandidate(
                            case_name=cell.case_name,
                            goal=cell.goal,
                            security_target=cell.security_target,
                            max_g_value=cell.max_g_value,
                            partition_num=partition_num,
                            window_radius=window_radius,
                            entropy_floor=entropy_floor,
                            size_threshold=size_threshold,
                            vrf_threshold=vrf_threshold,
                            hash_len=hash_len,
                            expected_retries=expected_retries,
                            kappa=support_kappa,
                            proxy_score=statistics.fmean(filtered_scores),
                        )
                        candidate_key = (
                            hash_len,
                            partition_num,
                            window_radius,
                            entropy_floor,
                            size_threshold,
                            vrf_threshold,
                        )
                        current = candidates.get(candidate_key)
                        if current is None or _candidate_proxy_sort_key(
                            candidate,
                            retry_target=config.retry_limit,
                        ) < _candidate_proxy_sort_key(
                            current,
                            retry_target=config.retry_limit,
                        ):
                            candidates[candidate_key] = candidate
                            if (
                                config.stop_after_candidates > 0
                                and len(candidates) >= config.stop_after_candidates
                            ):
                                return sorted(
                                    candidates.values(),
                                    key=lambda candidate: _candidate_proxy_sort_key(
                                        candidate,
                                        retry_target=config.retry_limit,
                                    ),
                                )

    return sorted(
        candidates.values(),
        key=lambda candidate: _candidate_proxy_sort_key(
            candidate,
            retry_target=config.retry_limit,
        ),
    )


def _benchmark_selected_candidates(
    candidates: Sequence[VISPTSearchCandidate],
    config: VISPTSearchConfig,
    *,
    random_seed: int,
) -> List[VISPTMeasuredRow]:
    benchmarked: List[VISPTMeasuredRow] = []
    retry_cap = _retry_cap(config)
    for index, candidate in enumerate(candidates):
        params = _candidate_params(
            hash_len=candidate.hash_len,
            max_g_value=candidate.max_g_value,
            partition_num=candidate.partition_num,
            window_radius=candidate.window_radius,
            entropy_floor=candidate.entropy_floor,
            size_threshold=candidate.size_threshold,
            vrf_threshold=candidate.vrf_threshold,
        )
        acceptance, group_samples = _estimate_acceptance_with_group_samples(
            params,
            trials=config.final_acceptance_trials,
            seed=_seed_mix(random_seed, 0xA5A5A5A5, index, candidate.hash_len),
            collect_group_count=max(
                1,
                config.benchmark_samples * config.benchmark_repetitions,
            ),
        )
        if acceptance <= 0.0:
            continue
        expected_retries = 1.0 / acceptance
        if expected_retries > retry_cap:
            continue
        spec = _spec_from_candidate(
            replace(candidate, expected_retries=expected_retries)
        )
        benchmarked.append(
            run_vispt_row_benchmark(
                spec,
                samples=config.benchmark_samples,
                repetitions=config.benchmark_repetitions,
                random_seed=_seed_mix(random_seed, 0x5A5A5A5A, index, spec.hash_len or 0),
                accepted_groups_samples=group_samples,
            )
        )
    return benchmarked


def _select_search_shortlist(
    candidates: Sequence[VISPTSearchCandidate],
    *,
    retry_target: float,
    finalist_count: int,
    include_small_leaf: bool,
) -> List[VISPTSearchCandidate]:
    selected: List[VISPTSearchCandidate] = []
    seen: set[
        tuple[int, int, int, Optional[int], Optional[int], Optional[int]]
    ] = set()

    def add_from(
        ordered: Sequence[VISPTSearchCandidate],
        limit: int,
    ) -> None:
        for candidate in ordered[:limit]:
            key = (
                candidate.hash_len,
                candidate.partition_num,
                candidate.window_radius,
                candidate.entropy_floor,
                candidate.size_threshold,
                candidate.vrf_threshold,
            )
            if key in seen:
                continue
            seen.add(key)
            selected.append(candidate)

    add_from(
        sorted(
            candidates,
            key=lambda candidate: _candidate_proxy_sort_key(
                candidate,
                retry_target=retry_target,
            ),
        ),
        max(finalist_count * 3, finalist_count),
    )
    if include_small_leaf:
        add_from(
            sorted(
                candidates,
                key=lambda candidate: _candidate_leaf_sort_key(
                    candidate,
                    retry_target=retry_target,
                ),
            ),
            max(finalist_count * 3, finalist_count),
        )
    return selected


def _sections_from_resolved_rows(
    search_sections: Sequence[Sequence[VISPTSearchCell]],
    resolved_by_base: Mapping[tuple[str, str, int, int], Mapping[str, VISPTMeasuredRow]],
) -> tuple[List[List[VISPTTableRowSpec]], Dict[str, VISPTMeasuredRow]]:
    section_specs: List[List[VISPTTableRowSpec]] = []
    measured_rows: Dict[str, VISPTMeasuredRow] = {}
    for section in search_sections:
        section_result: List[VISPTTableRowSpec] = []
        for cell in section:
            measured = resolved_by_base.get(_cell_base_key(cell), {}).get(cell.variant)
            if measured is None:
                spec = _pending_spec_for_cell(cell)
            else:
                spec = measured.spec
                measured_rows[spec.label] = measured
            section_result.append(spec)
        section_specs.append(section_result)
    return section_specs, measured_rows


def search_vispt_sections(
    *,
    config: VISPTSearchConfig,
    random_seed: int,
    search_sections: Optional[Sequence[Sequence[VISPTSearchCell]]] = None,
    existing_results: Optional[
        Mapping[tuple[str, str, int, int], Mapping[str, VISPTMeasuredRow]]
    ] = None,
    force_refresh_base_keys: Optional[set[tuple[str, str, int, int]]] = None,
) -> tuple[
    List[List[VISPTTableRowSpec]],
    Dict[str, VISPTMeasuredRow],
    Dict[tuple[str, str, int, int], Dict[str, VISPTMeasuredRow]],
]:
    active_search_sections = (
        vispt_search_sections()
        if search_sections is None
        else [list(section) for section in search_sections]
    )
    resolved_by_base: Dict[
        tuple[str, str, int, int],
        dict[str, VISPTMeasuredRow],
    ] = {
        base_key: dict(variants)
        for base_key, variants in ({} if existing_results is None else existing_results).items()
    }
    refresh_base_keys = set() if force_refresh_base_keys is None else set(force_refresh_base_keys)

    for section_index, section in enumerate(active_search_sections):
        for cell_index, cell in enumerate(section):
            base_key = _cell_base_key(cell)
            resolved_variants = resolved_by_base.get(base_key)
            if resolved_variants is None or base_key in refresh_base_keys:
                base_seed = _seed_mix(
                    random_seed,
                    section_index,
                    cell_index,
                    cell.security_target,
                    cell.max_g_value,
                    1 if cell.case_name == "case1" else 2,
                    1 if cell.goal == "Size" else 2,
                )
                base_candidates = _search_base_cell_candidates(
                    cell,
                    config,
                    random_seed=base_seed,
                )
                shortlist = _select_search_shortlist(
                    base_candidates,
                    retry_target=config.retry_limit,
                    finalist_count=config.finalist_count,
                    include_small_leaf=cell.goal == "Size",
                )
                benchmarked = _benchmark_selected_candidates(
                    shortlist,
                    config,
                    random_seed=base_seed,
                )
                resolved_variants = {}
                if benchmarked:
                    best_row = min(
                        benchmarked,
                        key=lambda row: _measured_row_sort_key(
                            row,
                            retry_target=config.retry_limit,
                        ),
                    )
                    resolved_variants["best"] = best_row
                    small_leaf_row = min(
                        benchmarked,
                        key=lambda row: _measured_row_small_leaf_sort_key(
                            row,
                            retry_target=config.retry_limit,
                        ),
                    )
                    resolved_variants["small_leaf"] = small_leaf_row
                resolved_by_base[base_key] = resolved_variants

    section_specs, measured_rows = _sections_from_resolved_rows(
        active_search_sections,
        resolved_by_base,
    )
    return section_specs, measured_rows, resolved_by_base


def _setup_kwargs_for_spec(spec: VISPTTableRowSpec) -> Dict[str, Any]:
    aux_t: Dict[str, Any] = {"profile_rule": "dyadic_greedy"}
    if spec.entropy_floor is not None:
        aux_t["entropy_floor"] = spec.entropy_floor
    tag = spec.label.encode("utf-8")
    return {
        "aux_t": aux_t,
        "size_threshold": spec.size_threshold,
        "vrf_threshold": spec.vrf_threshold,
        "mode": "size",
        "key_seed": b"key/" + tag,
        "keyed_hash_key_seed": b"hk/" + tag,
        "ads_seed": b"ads/" + tag,
        "tweak_public_seed": b"twh/" + tag,
        "merkle_public_seed": b"mt/" + tag,
        "salt_bytes": 2,
    }


def benchmark_case_for_spec(
    spec: VISPTTableRowSpec,
    *,
    samples: int,
    random_seed: int,
) -> YCSigBenchmarkCase:
    if spec.is_pending:
        raise ValueError("pending rows do not define a benchmark case")
    return YCSigBenchmarkCase(
        name=spec.label,
        security_parameter=spec.security_target,
        hash_len=spec.hash_len,
        max_g_bit=spec.max_g_bit,
        partition_size=spec.partition_num,
        window_radius=spec.window_radius,
        samples=samples,
        signature_extra_hash_values=0.0,
        signature_extra_bits=0,
        random_seed=random_seed,
        setup_kwargs=_setup_kwargs_for_spec(spec),
    )


def _find_partition_unmeasured(
    scheme: Any,
    message: bytes,
    randomizer: bytes,
) -> tuple[int, Sequence[Sequence[int]]]:
    return scheme.FindPartition(message, randomizer)


def run_vispt_row_benchmark(
    spec: VISPTTableRowSpec,
    *,
    samples: int,
    repetitions: int,
    random_seed: int,
    accepted_groups_samples: Optional[Sequence[GroupsKey]] = None,
) -> VISPTMeasuredRow:
    case = benchmark_case_for_spec(spec, samples=samples, random_seed=random_seed)
    if repetitions <= 0:
        raise ValueError("repetitions must be positive")

    keygen_counters_per_rep: List[Dict[str, float]] = []
    sign_core_counters_per_rep: List[Dict[str, float]] = []
    verify_counters_per_rep: List[Dict[str, float]] = []
    sig_bits_avgs: List[float] = []
    sig_obj_avgs: List[float] = []
    verify_rates: List[float] = []

    for repetition in range(repetitions):
        rep_case = _case_for_repetition(case, repetition)
        scheme = _build_scheme(rep_case)

        keygen_sample_counters: List[Dict[str, float]] = []
        sign_core_sample_counters: List[Dict[str, float]] = []
        verify_sample_counters: List[Dict[str, float]] = []
        sig_bits: List[float] = []
        sig_obj: List[float] = []
        verify_successes = 0
        hash_equivalent_bits = (
            rep_case.hash_equivalent_bits or scheme.params.security_parameter
        )

        for sample_index in range(rep_case.samples):
            message = _message_for_sample(rep_case, sample_index)
            key_seed = _derive_key_seed_for_sample(rep_case, sample_index)
            randomizer = _randomizer_for_sample(rep_case, sample_index)
            ks = PPRF.PRFKGen(scheme.params.pm_PPRF, seed=key_seed)

            keypair_holder: Dict[str, Any] = {}

            def do_keygen() -> None:
                keypair_holder["value"] = scheme.SigGen(ks)

            keygen_counter = _measure_counters(do_keygen)
            keypair = keypair_holder["value"]
            keygen_sample_counters.append(keygen_counter)

            if accepted_groups_samples:
                group_index = (
                    repetition * rep_case.samples + sample_index
                ) % len(accepted_groups_samples)
                salt = 0
                groups = accepted_groups_samples[group_index]
            else:
                salt, groups = _find_partition_unmeasured(
                    scheme,
                    message,
                    randomizer,
                )

            signature_holder: Dict[str, Any] = {}

            def do_sign_core() -> None:
                signature_holder["value"] = scheme.SignWithGroups(
                    keypair.secret_key,
                    randomizer,
                    salt,
                    groups,
                )

            sign_core_counter = _measure_counters(do_sign_core)
            signature = signature_holder["value"]
            sign_core_sample_counters.append(sign_core_counter)

            verify_holder: Dict[str, Any] = {}

            def do_verify_core() -> None:
                verify_holder["value"] = scheme.VerifyWithGroups(
                    keypair.public_key,
                    signature,
                    groups,
                )

            verify_counter = _measure_counters(do_verify_core)
            valid = verify_holder["value"]
            verify_successes += int(valid)
            verify_sample_counters.append(verify_counter)

            sig_bits.append(8 * signature.serialized_size())
            sig_obj.append(
                len(signature.punctured_seeds)
                + len(signature.partial_state_values)
                + rep_case.signature_extra_hash_values
                + rep_case.signature_extra_bits / hash_equivalent_bits
            )

        keygen_counters_per_rep.append(_aggregate_counter_dicts(keygen_sample_counters))
        sign_core_counters_per_rep.append(_aggregate_counter_dicts(sign_core_sample_counters))
        verify_counters_per_rep.append(_aggregate_counter_dicts(verify_sample_counters))
        sig_bits_avgs.append(statistics.fmean(sig_bits))
        sig_obj_avgs.append(statistics.fmean(sig_obj))
        verify_rates.append(verify_successes / rep_case.samples)

    keygen_avg = _aggregate_counter_dicts(keygen_counters_per_rep)
    sign_core_avg = _aggregate_counter_dicts(sign_core_counters_per_rep)
    verify_avg = _aggregate_counter_dicts(verify_counters_per_rep)
    keygen_real = _top_level_ops(keygen_avg)
    sign_core_real = _top_level_ops(sign_core_avg)
    verify_real = _top_level_ops(verify_avg)
    expected_retries = 0.0 if spec.expected_retries is None else float(spec.expected_retries)
    sign_total = sign_core_real + expected_retries
    result = {
        "case": case.name,
        "parameters": {
            "security_parameter": case.security_parameter,
            "hash_len": case.hash_len,
            "max_g_bit": case.max_g_bit,
            "max_g_value": 1 << case.max_g_bit,
            "partition_size": case.partition_size,
            "window_radius": case.window_radius,
            "samples": case.samples,
            "repetitions": repetitions,
            "total_experiments": case.samples * repetitions,
            "setup_excluded": True,
            "prf_keygen_excluded": True,
        },
        "operations": {
            "keygen_hash_equivalents_real": keygen_real,
            "retry_hash_equivalents_real": expected_retries,
            "retry_attempt_count_real": expected_retries,
            "retry_attempt_hash_equivalents_real": expected_retries,
            "retry_sampler_hash_equivalents_real": 0.0,
            "retry_sampler_output_bytes_real": 0.0,
            "retry_sampler_output_bits_real": 0.0,
            "sign_core_hash_equivalents_real": sign_core_real,
            "sign_hash_equivalents_real": sign_total,
            "verify_hash_equivalents_real": verify_real,
            "keygen_backend_hash_calls": _top_level_backend_hash_calls(keygen_avg),
            "retry_backend_hash_calls": 0.0,
            "sign_core_backend_hash_calls": _top_level_backend_hash_calls(sign_core_avg),
            "sign_backend_hash_calls": _top_level_backend_hash_calls(sign_core_avg),
            "verify_backend_hash_calls": _top_level_backend_hash_calls(verify_avg),
        },
        "breakdown": {
            "keygen": keygen_avg,
            "retry": {},
            "sign_core": sign_core_avg,
            "sign": sign_core_avg,
            "verify": verify_avg,
        },
        "signature": {
            "avg_signature_bits": statistics.fmean(sig_bits_avgs),
            "avg_signature_hash_equivalents_concrete": statistics.fmean(sig_bits_avgs)
            / float(case.security_parameter),
            "avg_signature_hash_equivalents_object_model": statistics.fmean(sig_obj_avgs),
        },
        "verify_rate": statistics.fmean(verify_rates),
    }
    return VISPTMeasuredRow(
        spec=spec,
        keygen=keygen_real,
        sign=sign_total,
        verify=verify_real,
        sig_size=statistics.fmean(sig_obj_avgs),
        verify_rate=statistics.fmean(verify_rates),
        raw_result=result,
    )


def run_vispt_section_benchmarks(
    *,
    row_specs: Optional[Sequence[VISPTTableRowSpec]] = None,
    samples: int,
    repetitions: int,
    random_seed: int,
) -> Dict[str, VISPTMeasuredRow]:
    active_specs = vispt_table_rows() if row_specs is None else list(row_specs)
    measured: Dict[str, VISPTMeasuredRow] = {}
    certified_rows = [spec for spec in active_specs if not spec.is_pending]
    for index, spec in enumerate(certified_rows):
        measured[spec.label] = run_vispt_row_benchmark(
            spec,
            samples=samples,
            repetitions=repetitions,
            random_seed=random_seed + index,
        )
    return measured


def bootstrap_static_results_cache(
    resolved_rows: Mapping[tuple[str, str, int, int], Mapping[str, VISPTMeasuredRow]],
    *,
    samples: int,
    repetitions: int,
    random_seed: int,
) -> Dict[tuple[str, str, int, int], Dict[str, VISPTMeasuredRow]]:
    bootstrapped: Dict[tuple[str, str, int, int], Dict[str, VISPTMeasuredRow]] = {
        base_key: dict(variants)
        for base_key, variants in resolved_rows.items()
    }
    groups: Dict[tuple[str, str, int, int], List[VISPTTableRowSpec]] = {}
    for spec in vispt_table_rows():
        if spec.is_pending:
            continue
        base_key = (
            spec.case_name,
            spec.goal,
            spec.security_target,
            spec.max_g_value,
        )
        if base_key in bootstrapped:
            continue
        groups.setdefault(base_key, []).append(spec)

    for group_index, (base_key, specs) in enumerate(sorted(groups.items())):
        measured_group = [
            run_vispt_row_benchmark(
                spec,
                samples=samples,
                repetitions=repetitions,
                random_seed=_seed_mix(random_seed, group_index, index, spec.hash_len or 0),
            )
            for index, spec in enumerate(specs)
        ]
        best_row = min(
            measured_group,
            key=lambda row: _measured_row_sort_key(
                row,
                retry_target=256.0,
            ),
        )
        resolved_group: Dict[str, VISPTMeasuredRow] = {"best": best_row}
        if best_row.spec.goal == "Size" and len(measured_group) > 1:
            resolved_group["small_leaf"] = min(
                measured_group,
                key=lambda row: _measured_row_small_leaf_sort_key(
                    row,
                    retry_target=256.0,
                ),
            )
        bootstrapped[base_key] = resolved_group
    return bootstrapped


def _fmt_cost(value: float) -> str:
    rounded = round(value)
    if abs(value - rounded) < 0.05:
        return str(int(rounded))
    return f"{value:.1f}"


def _fmt_fixed(value: float) -> str:
    return f"{value:.1f}"


def _render_int_or_pending(value: Optional[int]) -> str:
    return _TEX_PENDING if value is None else str(value)


def _render_tex_value(value: Optional[int], tex: Optional[str]) -> str:
    if tex is not None:
        return tex
    if value is None:
        return _TEX_PENDING
    return str(value)


def _render_metric_or_pending(value: Optional[float]) -> str:
    return _TEX_PENDING if value is None else _fmt_fixed(value)


def _render_cost_or_pending(value: Optional[float]) -> str:
    return _TEX_PENDING if value is None else _fmt_cost(value)


def _flatten_sections(
    sections: Sequence[Sequence[VISPTTableRowSpec]],
) -> List[VISPTTableRowSpec]:
    return [spec for section in sections for spec in section]


def _measured_rows_in_display_order(
    sections: Sequence[Sequence[VISPTTableRowSpec]],
    measured_rows: Mapping[str, VISPTMeasuredRow],
) -> List[VISPTMeasuredRow]:
    ordered: List[VISPTMeasuredRow] = []
    seen: set[str] = set()
    for spec in _flatten_sections(sections):
        if spec.label in seen or spec.label not in measured_rows:
            continue
        seen.add(spec.label)
        ordered.append(measured_rows[spec.label])
    return ordered


def _filter_measured_rows(
    rows: Sequence[VISPTMeasuredRow],
    *,
    case_name: Optional[str] = None,
    goal: Optional[str] = None,
) -> List[VISPTMeasuredRow]:
    filtered = list(rows)
    if case_name is not None:
        filtered = [row for row in filtered if row.spec.case_name == case_name]
    if goal is not None:
        filtered = [row for row in filtered if row.spec.goal == goal]
    return filtered


def _summary_paragraphs(
    sections: Sequence[Sequence[VISPTTableRowSpec]],
    measured_rows: Mapping[str, VISPTMeasuredRow],
    *,
    search_config: Optional[VISPTSearchConfig] = None,
) -> List[str]:
    ordered_rows = _measured_rows_in_display_order(sections, measured_rows)
    retry_target = 256.0 if search_config is None else search_config.retry_limit
    retry_cap = retry_target if search_config is None else _retry_cap(search_config)
    pending_count = sum(1 for spec in _flatten_sections(sections) if spec.is_pending)

    paragraphs: List[str] = []
    if ordered_rows:
        paragraphs.append(
            r"Table~\ref{tab:vispt-parameter-search} reports the rows retained by the "
            r"current automated search. Each numerical row passes the exact "
            r"profile-support UCR filter with $\KappaT\ge\kappa^{*}$, satisfies the "
            r"guarded retry budget with target "
            + f"${retry_target:.0f}$"
            + r" and cap "
            + f"${retry_cap:.0f}$"
            + r", and is then benchmarked at the concrete $\YCSig+\ValStrictISPT$ level. "
            r"Rows marked $\Pending$ are search cells for which the current grid did not "
            r"retain a benchmarked candidate."
        )
    else:
        paragraphs.append(
            r"Table~\ref{tab:vispt-parameter-search} reports the current search grid. "
            r"No numerical row was retained by the present exact UCR filter and retry "
            r"budget, so all cells remain marked $\Pending$."
        )

    case1_size_rows = _filter_measured_rows(ordered_rows, case_name="case1", goal="Size")
    case2_size_rows = _filter_measured_rows(ordered_rows, case_name="case2", goal="Size")
    case1_verify_rows = _filter_measured_rows(ordered_rows, case_name="case1", goal="Verify")
    case2_verify_rows = _filter_measured_rows(ordered_rows, case_name="case2", goal="Verify")

    if case1_size_rows:
        best = min(
            case1_size_rows,
            key=lambda row: _measured_row_sort_key(row, retry_target=retry_target),
        )
        paragraphs.append(
            r"In Case~1, the best size-oriented row in the current grid uses "
            + f"$\\MaxGValue={best.spec.max_g_value}$, "
            + f"$\\PartitionNum={best.spec.partition_num}$, "
            + f"$\\HashLen={best.spec.hash_len}$, "
            + f"$\\SizeThreshold={best.spec.size_threshold}$, "
            + r"and yields mean signature-size cost about "
            + f"{best.sig_size:.1f}"
            + r" with verification cost about "
            + f"{best.verify:.1f}"
            + r"."
        )

    if case2_size_rows:
        best = min(
            case2_size_rows,
            key=lambda row: _measured_row_sort_key(row, retry_target=retry_target),
        )
        compact = min(
            case2_size_rows,
            key=lambda row: _measured_row_small_leaf_sort_key(row, retry_target=retry_target),
        )
        if best.spec.label == compact.spec.label:
            paragraphs.append(
                r"In Case~2, the best size-oriented row also has the smallest retained "
                r"leaf universe, giving mean signature-size cost about "
                + f"{best.sig_size:.1f}"
                + r" at $\LeafUniverseSize="
                + f"{best.spec.leaf_universe_size}$"
                + r"."
            )
        else:
            paragraphs.append(
                r"In Case~2, the smallest retained signature-size row gives mean cost "
                + f"about {best.sig_size:.1f}"
                + r", while the most compact retained row uses "
                + f"$\\LeafUniverseSize={compact.spec.leaf_universe_size}$"
                + r" and gives mean signature-size cost about "
                + f"{compact.sig_size:.1f}"
                + r"."
            )

    if case1_verify_rows:
        best = min(
            case1_verify_rows,
            key=lambda row: _measured_row_sort_key(row, retry_target=retry_target),
        )
        paragraphs.append(
            r"In Case~1, the best verification-oriented row uses "
            + f"$\\MaxGValue={best.spec.max_g_value}$, "
            + f"$\\PartitionNum={best.spec.partition_num}$, "
            + f"$\\HashLen={best.spec.hash_len}$, "
            + f"$\\VrfThreshold={best.spec.vrf_threshold}$, "
            + r"and gives mean verification cost about "
            + f"{best.verify:.1f}"
            + r" with signature-size score about "
            + f"{best.sig_size:.1f}"
            + r"."
        )
    if case2_verify_rows:
        best = min(
            case2_verify_rows,
            key=lambda row: _measured_row_sort_key(row, retry_target=retry_target),
        )
        paragraphs.append(
            r"In Case~2, the best verification-oriented row retained by the current "
            r"search gives mean verification cost about "
            + f"{best.verify:.1f}"
            + r"."
        )
    elif any(
        spec.goal == "Verify" and spec.case_name == "case2"
        for spec in _flatten_sections(sections)
    ):
        paragraphs.append(
            r"No Case~2 verification-oriented row was retained within the current exact "
            r"UCR filter and retry budget."
        )

    if pending_count > 0:
        paragraphs.append(
            r"The remaining $\Pending$ cells are not negative results. They identify the "
            r"parts of the expanded grid that still need either a wider parameter sweep "
            r"or more aggressive exact guarded counting before benchmark certification."
        )

    return paragraphs


def render_vispt_table_latex(
    sections: Sequence[Sequence[VISPTTableRowSpec]],
    measured_rows: Mapping[str, VISPTMeasuredRow],
) -> str:
    lines = [
        r"\begin{table}[t]",
        r"\centering",
        r"\footnotesize",
        r"\setlength{\tabcolsep}{1.15pt}",
        r"\caption{Expanded parameter candidates and implementation-level costs for",
        r"$\YCSig$ instantiated with $\ValStrictISPT$. Case~1 searches admissible",
        r"$\HashLen\ge\kappa^{*}$ with $\HashLen/\MaxGBit\in\Nn$, and Case~2 searches",
        r"admissible $\HashLen\ge2\kappa^{*}$ with",
        r"$\HashLen/\MaxGBit\in\Nn$. The columns up to $\kappa^{*}$ report the",
        r"$\ValStrictISPT$ parameters and certification quantities. The columns KeyGen,",
        r"Sign, Verify, and Sig. Size report hash-equivalent implementation costs under",
        r"the actual-$\LeafUniverseSize$ canonical cover model. The Sign column includes",
        r"the expected Las Vegas retry cost. Rows marked $\Pending$ are search cells",
        r"for which the current automated sweep did not retain a benchmarked row under",
        r"the present exact UCR filter and retry budget.}",
        r"\label{tab:vispt-parameter-search}",
        r"\resizebox{0.999\textwidth}{!}{",
        r"\begin{tabular}{c c c c c c c c c c c c c c | c c c c}",
        r"\toprule",
        r"Case & Goal & $\MaxGValue$ & $\PartitionNum$ &",
        r"$\WindowRadius$ & $\EntropyFloor$ & $\SizeThreshold$ & $\VrfThreshold$ &",
        r"$\LeafUniverseSize$ & $\BlockNum$ & $\HashLen$ &",
        r"$\mathbb E[Re]$ & $\KappaT$ & $\kappa^{*}$ &",
        r"KeyGen & Sign & Verify & Sig. Size \\",
        r"\midrule",
    ]
    for section_index, section in enumerate(sections):
        for spec in section:
            measured = measured_rows.get(spec.label)
            row = [
                "Case 1" if spec.case_name == "case1" else "Case 2",
                spec.goal,
                str(spec.max_g_value),
                _render_int_or_pending(spec.partition_num),
                _render_int_or_pending(spec.window_radius),
                _render_tex_value(spec.entropy_floor, spec.entropy_floor_tex),
                _render_tex_value(spec.size_threshold, spec.size_threshold_tex),
                _render_tex_value(spec.vrf_threshold, spec.vrf_threshold_tex),
                _render_int_or_pending(spec.leaf_universe_size),
                _render_int_or_pending(spec.block_num),
                _render_int_or_pending(spec.hash_len),
                _render_metric_or_pending(spec.expected_retries),
                _render_metric_or_pending(spec.kappa),
                str(spec.security_target),
                _render_cost_or_pending(None if measured is None else measured.keygen),
                _render_cost_or_pending(None if measured is None else measured.sign),
                _render_cost_or_pending(None if measured is None else measured.verify),
                _render_cost_or_pending(None if measured is None else measured.sig_size),
            ]
            lines.append(" & ".join(row) + r" \\")
        if section_index + 1 < len(sections):
            lines.append(r"\midrule")
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


def render_vispt_section_latex(
    measured_rows: Mapping[str, VISPTMeasuredRow],
    *,
    sections: Optional[Sequence[Sequence[VISPTTableRowSpec]]] = None,
    search_config: Optional[VISPTSearchConfig] = None,
) -> str:
    active_sections = vispt_table_sections() if sections is None else list(sections)

    prose_lines = [
        r"\subsection{Performance Evaluation for $\YCSig$ with $\ValStrictISPT$}",
        r"\label{sec:vispt_performance}",
        "",
        r"Theorem~\ref{thm:ucr-vispt} gives the UCR exponent of the tree-aware",
        r"profile-routed construction $\ValStrictISPT$. The theorem analyzes a pair",
        r"event, namely the non-bottom collision event",
        r"$\{\PFunc(\PartitionValue)=\PFunc(\PartitionValue')\neq\Bottom\}$. By contrast,",
        r"the present subsection studies signature size, verification cost, and retry",
        r"cost through the corresponding single-input acceptance event. Accordingly, this",
        r"subsection uses the one-trial acceptance probability $\AccProbT$ in",
        r"Eq.~\eqref{eq:vispt-acceptance}, rather than the pair-collision probability in",
        r"Theorem~\ref{thm:ucr-vispt}.",
        "",
        r"\paragraph{Evaluation setup.}",
        r"The evaluation uses the profile-restricted routing framework defined in",
        r"Section~\ref{sec:vispt_ucr}. For every accepted histogram $\CountVec$, the",
        r"effective shape family $\DyShapeFamilyOf{\CountVec}$, the profile rule",
        r"$\ProfileRuleT$, the selected profile $\ShapeProfileOf{\CountVec}$, the support",
        r"size $\ShapeSupportSizeOf{\CountVec}$, and the guarded count",
        r"$\GuardCountTOf{\CountVec}$ are the public deterministic objects specified in",
        r"the UCR analysis. The profile-routing layer samples one exact-uniform",
        r"permutation of $\ShapeProfileOf{\CountVec}$, and the score guards act only as",
        r"sample-then-abort filters.",
        "",
        r"The experimental grid varies $\MaxGValue$, $\HashLen$, $\PartitionNum$,",
        r"$\WindowRadius$, $\EntropyFloor$, $\SizeThreshold$, and $\VrfThreshold$. The",
        r"core shape vocabulary is the dyadic profile family from",
        r"Section~\ref{sec:vispt_ucr}: empty and full shapes, aligned dyadic intervals,",
        r"their complements, singletons, and deterministic residual shapes when required",
        r"to realize the exact column sums. The row-local score",
        r"$\RowCostT(\ShapeSet)$ is used only inside $\ProfileRuleT$ to select",
        r"tree-friendly profiles. The final signature-size and verification costs are",
        r"always computed on the actual global universe",
        r"$\LeafUniverseSize=\PartitionNum\cdot\MaxGValue$.",
        "",
        r"\paragraph{Certification workflow.}",
        r"The search follows a UCR-first workflow. For each candidate parameter tuple,",
        r"the exact unguarded profile-support size $\ShapeSupportSizeOf{\CountVec}$ is",
        r"computed for every $\CountVec\in\WinCountSetT$. Because both score guards are",
        r"abort-style filters, enabling a guard can only decrease the non-bottom",
        r"collision mass. Therefore the exact support-only UCR exponent is a",
        r"conservative sufficient certificate for the guarded sampler: if the",
        r"support-only exponent already satisfies $\KappaT\ge\kappa^{*}$, then the",
        r"guarded construction also satisfies the target. After this exact UCR filter,",
        r"Monte Carlo simulation estimates the guarded one-trial acceptance probability",
        r"$\AccProbT$, the retry cost $\mathbb E[Re]=1/\AccProbT$, and the",
        r"implementation-level $\YCSig$ costs.",
        "",
        r"\paragraph{Cost model.}",
        r"For an accepted output $\Groups$, let",
        r"$\SelectedSet=S(\Groups)=\{i\cdot\MaxGValue+\BlockValue:$",
        r"$\BlockValue\in\SubGroup{i}\}\subseteq[\LeafUniverseSize]$, where",
        r"$\LeafUniverseSize=\PartitionNum\cdot\MaxGValue$. Let",
        r"$\CompSelectedSet=[\LeafUniverseSize]\setminus\SelectedSet$,",
        r"$\SelNodeNum=|\MaxCov{\LeafUniverseSize}{\SelectedSet}|$, and",
        r"$\CompNodeNum=|\MaxCov{\LeafUniverseSize}{\CompSelectedSet}|$. The",
        r"signature-size object model is $\SelNodeNum+\CompNodeNum$. The reported",
        r"verification cost uses the complement-signing accounting and is evaluated from",
        r"the same actual-$\LeafUniverseSize$ canonical cover routine as the PPRF and",
        r"Merkle-tree layers.",
        "",
        r"The signing cost includes both the accepted-signature core cost and the",
        r"Las Vegas salt-search cost. In the invocation-based hash-equivalent accounting",
        r"used in this table, one retry costs one domain-separated XOF transcript,",
        r"because $\PartitionValue$ and the route stream can be generated from the same",
        r"XOF state. Thus the reported signing cost is",
        r"$\CT_{\mathsf{sign}}^{\mathsf{tot}}",
        r"=\CT_{\mathsf{sign}}^{\mathsf{core}}+\mathbb E[Re]$. More generally, if an",
        r"implementation charges $\CT_{\mathsf{retry}}$ hash equivalents per trial, then",
        r"$\CT_{\mathsf{sign}}^{\mathsf{tot}}",
        r"=\CT_{\mathsf{sign}}^{\mathsf{core}}",
        r"+\mathbb E[Re]\cdot\CT_{\mathsf{retry}}$.",
        "",
        r"The parameter search follows the same two-case convention as",
        r"Table~\ref{tab:all-L-windowed}. Case~1 searches admissible",
        r"$\HashLen\ge\kappa^{*}$ with $\HashLen/\MaxGBit\in\Nn$, and Case~2 searches",
        r"admissible $\HashLen\ge2\kappa^{*}$ with",
        r"$\HashLen/\MaxGBit\in\Nn$. The expanded grid covers",
        r"$\kappa^{*}\in\{128,160,192\}$ and considers",
        r"$\MaxGValue\in\{4,16,64,128,256\}$. For each candidate, the search ranges over",
        r"$\HashLen$, $\MaxGValue$, $\PartitionNum$, $\WindowRadius$, $\EntropyFloor$,",
        r"$\SizeThreshold$, and $\VrfThreshold$. Every numerical row satisfies the",
        r"tree-aware UCR constraint $\KappaT\ge\kappa^{*}$ from",
        r"Eq.~\eqref{eq:vispt-ucr} and the retry constraint",
        r"$\mathbb E[Re]=1/\AccProbT$. Rows marked as $\Pending$ identify search cells",
        r"for which the current automated sweep did not retain a benchmarked row under",
        r"the present exact UCR filter and retry budget.",
        "",
        render_vispt_table_latex(active_sections, measured_rows),
        "",
    ]
    for paragraph in _summary_paragraphs(
        active_sections,
        measured_rows,
        search_config=search_config,
    ):
        prose_lines.append(paragraph)
        prose_lines.append("")
    return "\n".join(prose_lines)


def render_vispt_section_json(
    measured_rows: Mapping[str, VISPTMeasuredRow],
    *,
    sections: Optional[Sequence[Sequence[VISPTTableRowSpec]]] = None,
) -> str:
    active_sections = vispt_table_sections() if sections is None else list(sections)
    active_specs = _flatten_sections(active_sections)
    payload = {
        "rows": [
            {
                "spec": asdict(spec),
                "measured": None
                if spec.label not in measured_rows
                else {
                    "keygen": measured_rows[spec.label].keygen,
                    "sign": measured_rows[spec.label].sign,
                    "verify": measured_rows[spec.label].verify,
                    "sig_size": measured_rows[spec.label].sig_size,
                    "verify_rate": measured_rows[spec.label].verify_rate,
                    "raw_result": measured_rows[spec.label].raw_result,
                },
            }
            for spec in active_specs
        ]
    }
    return json.dumps(payload, ensure_ascii=True, indent=2)


def render_vispt_section_text(
    measured_rows: Mapping[str, VISPTMeasuredRow],
    *,
    sections: Optional[Sequence[Sequence[VISPTTableRowSpec]]] = None,
) -> str:
    active_sections = vispt_table_sections() if sections is None else list(sections)
    active_specs = _flatten_sections(active_sections)
    lines = [
        "Case | Goal | k* | w | P | R | Hmin | SizeT | VrfT | HashLen | E[Re] | Kappa | KeyGen | Sign | Verify | SigSize"
    ]
    lines.append("-" * len(lines[0]))
    for spec in active_specs:
        measured = measured_rows.get(spec.label)
        lines.append(
            " | ".join(
                [
                    spec.case_name,
                    spec.goal,
                    str(spec.security_target),
                    str(spec.max_g_value),
                    _render_int_or_pending(spec.partition_num),
                    _render_int_or_pending(spec.window_radius),
                    _render_tex_value(spec.entropy_floor, spec.entropy_floor_tex),
                    _render_tex_value(spec.size_threshold, spec.size_threshold_tex),
                    _render_tex_value(spec.vrf_threshold, spec.vrf_threshold_tex),
                    _render_int_or_pending(spec.hash_len),
                    _render_metric_or_pending(spec.expected_retries),
                    _render_metric_or_pending(spec.kappa),
                    _render_cost_or_pending(None if measured is None else measured.keygen),
                    _render_cost_or_pending(None if measured is None else measured.sign),
                    _render_cost_or_pending(None if measured is None else measured.verify),
                    _render_cost_or_pending(None if measured is None else measured.sig_size),
                ]
            )
        )
    return "\n".join(lines)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Search VISPT parameter rows, benchmark YCSig, and render subsection TeX.",
    )
    parser.add_argument(
        "--row-source",
        choices=("search", "static"),
        default="search",
        help="Whether to search the VISPT grid or benchmark the legacy static rows.",
    )
    parser.add_argument("--samples", type=int, default=32, help="Benchmark samples per certified row.")
    parser.add_argument(
        "--repetitions",
        type=int,
        default=3,
        help="Independent repetitions for primitive-operation benchmarks.",
    )
    parser.add_argument(
        "--random-seed",
        type=int,
        default=0,
        help="Deterministic benchmark seed.",
    )
    parser.add_argument(
        "--format",
        choices=("latex", "json", "text"),
        default="latex",
        help="Output format.",
    )
    parser.add_argument(
        "--labels",
        default=None,
        help="Optional comma-separated subset of legacy VISPT row labels when --row-source=static.",
    )
    parser.add_argument(
        "--retry-limit",
        type=float,
        default=256.0,
        help="Target retry budget used to rank retained rows.",
    )
    parser.add_argument(
        "--retry-slack",
        type=float,
        default=64.0,
        help="Allowed slack above --retry-limit during the search filter.",
    )
    parser.add_argument(
        "--pilot-trials",
        type=int,
        default=2048,
        help="Monte Carlo trials for no-guard pilot score sampling.",
    )
    parser.add_argument(
        "--pilot-accepted-samples",
        type=int,
        default=32,
        help="Accepted pilot samples collected before threshold generation.",
    )
    parser.add_argument(
        "--acceptance-trials",
        type=int,
        default=4096,
        help="Monte Carlo trials for guarded acceptance estimation during candidate search.",
    )
    parser.add_argument(
        "--final-acceptance-trials",
        type=int,
        default=32768,
        help="Monte Carlo trials for the final retry estimate of shortlisted candidates.",
    )
    parser.add_argument(
        "--finalist-count",
        type=int,
        default=3,
        help="Number of shortlisted candidates benchmarked per search cell.",
    )
    parser.add_argument(
        "--stop-after-candidates",
        type=int,
        default=0,
        help="Optional early-exit cap on valid search candidates retained per cell (0 keeps exhaustive search).",
    )
    parser.add_argument(
        "--cases",
        default=None,
        help="Optional comma-separated search filter over case names, e.g. case1,case2.",
    )
    parser.add_argument(
        "--goals",
        default=None,
        help="Optional comma-separated search filter over goals, e.g. Size,Verify.",
    )
    parser.add_argument(
        "--security-targets",
        default=None,
        help="Optional comma-separated search filter over kappa targets.",
    )
    parser.add_argument(
        "--max-g-values",
        default=None,
        help="Optional comma-separated search filter over MaxGValue values.",
    )
    parser.add_argument(
        "--results-cache",
        default=_DEFAULT_RESULTS_CACHE,
        help="JSON cache path for searched VISPT rows.",
    )
    parser.add_argument(
        "--no-cache",
        action="store_true",
        help="Disable reading and writing the VISPT search cache.",
    )
    parser.add_argument(
        "--refresh",
        action="store_true",
        help="Refresh the filtered search cells even if they already exist in the cache.",
    )
    parser.add_argument(
        "--skip-static-bootstrap",
        action="store_true",
        help="Do not benchmark the existing static non-pending rows into the cache.",
    )
    parser.add_argument(
        "--cached-only",
        action="store_true",
        help="Render directly from the cache without searching filtered cells.",
    )
    return parser


def _main() -> int:
    args = _build_parser().parse_args()
    labels = None if args.labels is None else [piece.strip() for piece in args.labels.split(",") if piece.strip()]
    active_sections: Sequence[Sequence[VISPTTableRowSpec]]
    measured: Dict[str, VISPTMeasuredRow]
    search_config: Optional[VISPTSearchConfig]

    if args.row_source == "static":
        selected_rows = select_vispt_rows(labels)
        active_sections = [selected_rows]
        measured = run_vispt_section_benchmarks(
            row_specs=selected_rows,
            samples=args.samples,
            repetitions=args.repetitions,
            random_seed=args.random_seed,
        )
        search_config = None
    else:
        search_config = VISPTSearchConfig(
            retry_limit=args.retry_limit,
            retry_slack=args.retry_slack,
            pilot_trials=args.pilot_trials,
            pilot_accepted_samples=args.pilot_accepted_samples,
            acceptance_trials=args.acceptance_trials,
            final_acceptance_trials=args.final_acceptance_trials,
            finalist_count=args.finalist_count,
            stop_after_candidates=args.stop_after_candidates,
            benchmark_samples=args.samples,
            benchmark_repetitions=args.repetitions,
        )
        filtered_search_sections = _filter_search_sections(
            vispt_search_sections(),
            cases=_parse_csv_strings(args.cases),
            goals=_parse_csv_strings(args.goals),
            security_targets=_parse_csv_ints(args.security_targets),
            max_g_values=_parse_csv_ints(args.max_g_values),
        )
        existing_results = (
            {}
            if args.no_cache
            else load_vispt_results_cache(args.results_cache)
        )
        if not args.skip_static_bootstrap:
            existing_results = bootstrap_static_results_cache(
                existing_results,
                samples=args.samples,
                repetitions=args.repetitions,
                random_seed=args.random_seed,
            )
        refresh_base_keys = None
        if args.refresh:
            refresh_base_keys = {
                _cell_base_key(cell)
                for section in filtered_search_sections
                for cell in section
            }
        if args.cached_only:
            active_sections, measured = _sections_from_resolved_rows(
                filtered_search_sections,
                existing_results,
            )
            resolved_results = dict(existing_results)
        else:
            active_sections, measured, resolved_results = search_vispt_sections(
                config=search_config,
                random_seed=args.random_seed,
                search_sections=filtered_search_sections,
                existing_results=existing_results,
                force_refresh_base_keys=refresh_base_keys,
            )
        if not args.no_cache:
            save_vispt_results_cache(args.results_cache, resolved_results)

    if args.format == "json":
        print(render_vispt_section_json(measured, sections=active_sections))
    elif args.format == "text":
        print(render_vispt_section_text(measured, sections=active_sections))
    else:
        print(
            render_vispt_section_latex(
                measured,
                sections=active_sections,
                search_config=search_config,
            )
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(_main())
