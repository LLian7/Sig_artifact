from __future__ import annotations

import argparse
import json
import math
from dataclasses import asdict, dataclass
from decimal import Decimal, ROUND_HALF_UP
from functools import lru_cache
from random import Random
from typing import Iterable, List, Optional, Sequence

from merkle_tree import MT
from pprf import PPRF
from val_strict_isp import ISPParameters, val_strict_isp
from yc_sig import YCSig


DEFAULT_CASES = ("case1", "case2")
DEFAULT_SECURITY_TARGETS = (128, 160, 192)
DEFAULT_MAX_G_VALUES = (4, 8, 16, 32)
DEFAULT_HASH_LEN_MAX_FACTOR = 2.5


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

    @property
    def max_g_bit(self) -> int:
        return self.max_g_value.bit_length() - 1


def _parse_int_csv(raw: str) -> List[int]:
    return [int(piece.strip()) for piece in raw.split(",") if piece.strip()]


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

    if acc_weights[block_num] <= 0.0:
        return 0.0, float("-inf")

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


def _expected_online_prg_cost(leaf_count: int, block_num: int) -> float:
    depth = math.ceil(math.log2(leaf_count))
    total = 0.0
    for level in range(1, depth + 1):
        interval_size = 1 << level
        interval_count = leaf_count // interval_size
        if interval_count == 0:
            continue
        occupied_probability = 1.0 - (1.0 - interval_size / leaf_count) ** block_num
        total += interval_count * occupied_probability
    return total


def _analytic_sign_cost(
    *,
    block_num: int,
    max_g_value: int,
    partition_num: int,
    expected_retries: float,
) -> float:
    leaf_count = partition_num * max_g_value
    return (
        (3 * block_num - 1) / 2.0
        + _expected_online_prg_cost(leaf_count, block_num)
        + expected_retries
    )


def _meets_exact_constraints(
    *,
    acceptance_probability: float,
    kappa: float,
    retry_limit: float,
    security_target: int,
) -> bool:
    if acceptance_probability <= 0.0 or math.isnan(kappa):
        return False
    expected_retries = 1.0 / acceptance_probability
    if expected_retries > retry_limit:
        return False
    return kappa >= security_target


def _estimate_signature_size_object_model(
    *,
    security_target: int,
    hash_len: int,
    max_g_bit: int,
    partition_num: int,
    window_radius: int,
    link_threshold: int,
    samples: int,
    seed: int,
) -> float:
    if samples <= 0:
        raise ValueError("samples must be positive")

    setup = YCSig.SigSetup(
        security_target,
        hash_len=hash_len,
        max_g_bit=max_g_bit,
        partition_size=partition_num,
        window_radius=window_radius,
        link_threshold=link_threshold,
        salt_bytes=2,
    )
    params = setup.params
    rng = Random(seed)
    total = 0.0

    for _ in range(samples):
        while True:
            partition_value = format(rng.getrandbits(hash_len), f"0{hash_len}b")
            groups = val_strict_isp(partition_value, params.pm_ISP)
            if groups is None:
                continue
            break

        signed_indices = []
        for group_index, subgroup in enumerate(groups):
            base = group_index * params.max_g_value
            for block_value in subgroup:
                signed_indices.append(base + block_value)
        bitstrings = tuple(
            format(alpha, f"0{params.leaf_index_bits}b")
            for alpha in signed_indices
        )
        punctured_count = len(
            PPRF.CanonicalPrefixes(
                params.pm_PPRF,
                bitstrings,
                inputs_normalized=True,
            )
        )
        partial_count = len(MT.CanonicalStatePositions(params.pm_MT, signed_indices))
        total += punctured_count + partial_count + 1.0 + 16.0 / security_target

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
    )


def _iter_hash_lens(cell: SearchCell, hash_len_max: int) -> Iterable[int]:
    start = _next_divisible(cell.min_hash_len, cell.max_g_bit)
    for hash_len in range(start, hash_len_max + 1, cell.max_g_bit):
        yield hash_len


def _iter_feasible_rows_for_partition_and_hash_len(
    *,
    cell: SearchCell,
    partition_num: int,
    hash_len: int,
    retry_limit: float,
    link_threshold: int,
) -> Iterable[SearchRow]:
    block_num = hash_len // cell.max_g_bit
    max_window_radius = block_num // cell.max_g_value
    for window_radius in range(max_window_radius + 1):
        acceptance_probability, kappa = exact_parameter_metrics(
            block_num,
            cell.max_g_value,
            partition_num,
            window_radius,
        )
        if not _meets_exact_constraints(
            acceptance_probability=acceptance_probability,
            kappa=kappa,
            retry_limit=retry_limit,
            security_target=cell.security_target,
        ):
            continue
        yield _candidate_row(
            cell=cell,
            partition_num=partition_num,
            hash_len=hash_len,
            window_radius=window_radius,
            acceptance_probability=acceptance_probability,
            kappa=kappa,
            link_threshold=link_threshold,
        )


def _iter_feasible_rows_for_hash_len(
    *,
    cell: SearchCell,
    hash_len: int,
    retry_limit: float,
    link_threshold: int,
) -> Iterable[SearchRow]:
    block_num = hash_len // cell.max_g_bit
    min_partition_num = math.ceil(block_num / cell.max_g_value)
    max_window_radius = block_num // cell.max_g_value
    for partition_num in range(min_partition_num, block_num + 1):
        for window_radius in range(max_window_radius + 1):
            acceptance_probability, kappa = exact_parameter_metrics(
                block_num,
                cell.max_g_value,
                partition_num,
                window_radius,
            )
            if not _meets_exact_constraints(
                acceptance_probability=acceptance_probability,
                kappa=kappa,
                retry_limit=retry_limit,
                security_target=cell.security_target,
            ):
                continue
            yield _candidate_row(
                cell=cell,
                partition_num=partition_num,
                hash_len=hash_len,
                window_radius=window_radius,
                acceptance_probability=acceptance_probability,
                kappa=kappa,
                link_threshold=link_threshold,
            )


def search_best_row_for_cell(
    *,
    cell: SearchCell,
    objective: str,
    retry_limit: float,
    hash_len_max: int,
    link_threshold: int = -1,
    sig_size_samples: int = 0,
    sig_size_seed: int = 0,
    sig_partition_num_slack: Optional[int] = None,
) -> Optional[SearchRow]:
    if objective not in {"partition_num", "sig_size"}:
        raise ValueError("objective must be 'partition_num' or 'sig_size'")

    hash_lens = list(_iter_hash_lens(cell, hash_len_max))
    if not hash_lens:
        return None

    if objective == "partition_num":
        min_partition_num = math.ceil(
            (hash_lens[0] // cell.max_g_bit) / cell.max_g_value
        )
        max_partition_num = hash_len_max // cell.max_g_bit
        for partition_num in range(min_partition_num, max_partition_num + 1):
            for hash_len in hash_lens:
                block_num = hash_len // cell.max_g_bit
                min_required_partition_num = math.ceil(block_num / cell.max_g_value)
                if partition_num < min_required_partition_num:
                    break
                if partition_num > block_num:
                    continue
                if cell.case_name == "case1":
                    feasible_rows = list(
                        _iter_feasible_rows_for_partition_and_hash_len(
                            cell=cell,
                            partition_num=partition_num,
                            hash_len=hash_len,
                            retry_limit=retry_limit,
                            link_threshold=link_threshold,
                        )
                    )
                    if not feasible_rows:
                        continue
                    return min(
                        feasible_rows,
                        key=lambda row: (
                            _analytic_sign_cost(
                                block_num=row.block_num,
                                max_g_value=row.max_g_value,
                                partition_num=row.partition_num,
                                expected_retries=row.expected_retries,
                            ),
                            row.expected_retries,
                            row.window_radius,
                            row.hash_len,
                        ),
                    )
                max_window_radius = block_num // cell.max_g_value
                for window_radius in range(max_window_radius + 1):
                    acceptance_probability, kappa = exact_parameter_metrics(
                        block_num,
                        cell.max_g_value,
                        partition_num,
                        window_radius,
                    )
                    if not _meets_exact_constraints(
                        acceptance_probability=acceptance_probability,
                        kappa=kappa,
                        retry_limit=retry_limit,
                        security_target=cell.security_target,
                    ):
                        continue
                    return _candidate_row(
                        cell=cell,
                        partition_num=partition_num,
                        hash_len=hash_len,
                        window_radius=window_radius,
                        acceptance_probability=acceptance_probability,
                        kappa=kappa,
                        link_threshold=link_threshold,
                    )
        return None

    feasible_rows: List[SearchRow] = []
    for hash_len in hash_lens:
        feasible_rows.extend(
            _iter_feasible_rows_for_hash_len(
                cell=cell,
                hash_len=hash_len,
                retry_limit=retry_limit,
                link_threshold=link_threshold,
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
            security_target=row.security_target,
            hash_len=row.hash_len,
            max_g_bit=row.max_g_bit,
            partition_num=row.partition_num,
            window_radius=row.window_radius,
            link_threshold=link_threshold,
            samples=sig_size_samples,
            seed=sig_size_seed + index,
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
    link_threshold: int,
    sig_size_samples: int,
    sig_size_seed: int,
    sig_partition_num_slack: Optional[int],
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
                        link_threshold=link_threshold,
                        sig_size_samples=sig_size_samples,
                        sig_size_seed=sig_size_seed,
                        sig_partition_num_slack=sig_partition_num_slack,
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
    header = (
        "Case | k* | w | P | R | n | HashLen | E[Re] | kappa"
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
        if row.signature_size_obj is not None:
            line += f" | {row.signature_size_obj:.1f}"
        lines.append(line)
    return "\n".join(lines)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Search ValStrictISP/YCSig parameter rows using the exact acceptance "
            "formula and the theorem-level UCR exponent for the simplified-windowed VISP."
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
        "--link-threshold",
        type=int,
        default=-1,
        help="Deprecated compatibility option; ignored by the simplified-windowed VISP search.",
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
        "--format",
        choices=("json", "text"),
        default="text",
        help="Output format.",
    )
    return parser


def _main() -> int:
    args = _build_parser().parse_args()
    rows = search_rows(
        cases=_parse_case_csv(args.cases),
        security_targets=_parse_int_csv(args.security_targets),
        max_g_values=_parse_int_csv(args.max_g_values),
        objective=args.objective,
        retry_limit=args.retry_limit,
        hash_len_max_factor=args.hash_len_max_factor,
        hash_len_max_absolute=args.hash_len_max_absolute,
        link_threshold=args.link_threshold,
        sig_size_samples=args.sig_size_samples,
        sig_size_seed=args.sig_size_seed,
        sig_partition_num_slack=args.sig_partition_num_slack,
    )
    if args.format == "json":
        print(json.dumps([asdict(row) for row in rows], ensure_ascii=True, indent=2))
    else:
        print(render_text(rows))
    return 0


if __name__ == "__main__":
    raise SystemExit(_main())
