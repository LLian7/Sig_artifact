from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from decimal import Decimal, ROUND_HALF_UP
from typing import Dict, List

from benchmark_ycsig import YCSigBenchmarkCase
from benchmark_ycsig_ops import run_operation_benchmark_case


@dataclass(frozen=True)
class ComparisonCase:
    case_name: str
    security_target: int
    max_g_value: int
    partition_size: int
    window_radius: int
    block_num: int
    hash_len: int

    @property
    def max_g_bit(self) -> int:
        return self.max_g_value.bit_length() - 1

    @property
    def column_key(self) -> str:
        return f"{self.case_name}_k{self.security_target}"

    @property
    def label(self) -> str:
        return (
            f"YCSig-{self.case_name}-w{self.max_g_value}-k{self.security_target}-"
            f"h{self.hash_len}-p{self.partition_size}-r{self.window_radius}"
        )


COMPARISON_CASES: List[ComparisonCase] = [
    ComparisonCase("case1", 128, 4, 32, 2, 64, 128),
    ComparisonCase("case1", 128, 8, 16, 2, 56, 168),
    ComparisonCase("case1", 128, 16, 8, 2, 62, 248),
    ComparisonCase("case1", 128, 32, 5, 1, 40, 200),
    ComparisonCase("case1", 160, 4, 40, 2, 80, 160),
    ComparisonCase("case1", 160, 8, 20, 2, 73, 219),
    ComparisonCase("case1", 160, 16, 10, 2, 73, 292),
    ComparisonCase("case1", 160, 32, 6, 2, 64, 320),
    ComparisonCase("case2", 128, 4, 42, 6, 128, 256),
    ComparisonCase("case2", 128, 8, 17, 3, 86, 258),
    ComparisonCase("case2", 128, 16, 9, 4, 64, 256),
    ComparisonCase("case2", 128, 32, 5, 2, 64, 320),
    ComparisonCase("case2", 160, 4, 52, 5, 160, 320),
    ComparisonCase("case2", 160, 8, 22, 6, 107, 321),
    ComparisonCase("case2", 160, 16, 11, 5, 80, 320),
    ComparisonCase("case2", 160, 32, 6, 2, 64, 320),
]


def _round_half_up(value: float) -> int:
    return int(Decimal(str(value)).quantize(Decimal("1"), rounding=ROUND_HALF_UP))


def _seed_prefix(label: str) -> bytes:
    return label.encode("ascii")


def _make_case(spec: ComparisonCase) -> YCSigBenchmarkCase:
    tag = _seed_prefix(spec.label)
    random_seed = (
        spec.security_target * 1_000_000
        + spec.hash_len * 1_000
        + spec.max_g_value * 100
        + spec.partition_size * 10
        + spec.window_radius
    )
    return YCSigBenchmarkCase(
        name=spec.label,
        security_parameter=spec.security_target,
        hash_len=spec.hash_len,
        max_g_bit=spec.max_g_bit,
        partition_size=spec.partition_size,
        window_radius=spec.window_radius,
        samples=1,
        random_seed=random_seed,
        signature_extra_hash_values=1.0,
        signature_extra_bits=16,
        setup_kwargs={
            "key_seed": b"key/" + tag,
            "keyed_hash_key_seed": b"hk/" + tag,
            "ads_seed": b"ads/" + tag,
            "tweak_public_seed": b"twh/" + tag,
            "merkle_public_seed": b"mt/" + tag,
            "salt_bytes": 2,
        },
    )


def _load_comparison_cases_from_json(path: str) -> List[ComparisonCase]:
    with open(path, "r", encoding="utf-8") as handle:
        raw_rows = json.load(handle)
    if not isinstance(raw_rows, list):
        raise ValueError("params-json must contain a JSON list of parameter rows")

    specs: List[ComparisonCase] = []
    for raw in raw_rows:
        if not isinstance(raw, dict):
            raise ValueError("each params-json row must be a JSON object")
        specs.append(
            ComparisonCase(
                case_name=str(raw["case_name"]),
                security_target=int(raw["security_target"]),
                max_g_value=int(raw["max_g_value"]),
                partition_size=int(raw["partition_num"]),
                window_radius=int(raw["window_radius"]),
                block_num=int(raw["block_num"]),
                hash_len=int(raw["hash_len"]),
            )
        )
    return specs


def run_comparison_table(
    *,
    repetitions: int,
    comparison_cases: List[ComparisonCase] | None = None,
) -> Dict[int, Dict[str, Dict[str, float]]]:
    results: Dict[int, Dict[str, Dict[str, float]]] = {}
    active_cases = COMPARISON_CASES if comparison_cases is None else comparison_cases
    for spec in active_cases:
        result = run_operation_benchmark_case(_make_case(spec), repetitions=repetitions)
        results.setdefault(spec.max_g_value, {})[spec.column_key] = {
            "keygen": float(result["operations"]["keygen_hash_equivalents_real"]),
            "sign": float(result["operations"]["sign_hash_equivalents_real"]),
            "verify": float(result["operations"]["verify_hash_equivalents_real"]),
            "sig_size": float(result["signature"]["avg_signature_hash_equivalents_object_model"]),
        }
    return results


def render_text(
    *,
    comparison_cases: List[ComparisonCase],
    ycsig_results: Dict[int, Dict[str, Dict[str, float]]],
) -> str:
    header = "Case | k* | w | P | (rm,rp) | HashLen | KeyGen | Sign | Verify | SigSize"
    rule = "-" * len(header)
    lines = [header, rule]
    for spec in comparison_cases:
        values = ycsig_results[spec.max_g_value][spec.column_key]
        lines.append(
            (
                f"{spec.case_name} | "
                f"{spec.security_target} | "
                f"{spec.max_g_value} | "
                f"{spec.partition_size} | "
                f"{spec.window_radius} | "
                f"{spec.hash_len} | "
                f"{_round_half_up(values['keygen'])} | "
                f"{_round_half_up(values['sign'])} | "
                f"{_round_half_up(values['verify'])} | "
                f"{_round_half_up(values['sig_size'])}"
            )
        )
    return "\n".join(lines)


def _main() -> int:
    parser = argparse.ArgumentParser(
        description="Recompute the comparison-table YCSig rows using implementation-based counters.",
    )
    parser.add_argument(
        "--repetitions",
        type=int,
        default=5000,
        help="Independent repetitions per parameter set.",
    )
    parser.add_argument(
        "--params-json",
        default=None,
        help="Optional JSON exported by search_ycsig_sigsize.py to override COMPARISON_CASES.",
    )
    parser.add_argument(
        "--format",
        choices=("text", "json"),
        default="text",
        help="Output format.",
    )
    args = parser.parse_args()

    comparison_cases = (
        None
        if args.params_json is None
        else _load_comparison_cases_from_json(args.params_json)
    )
    results = run_comparison_table(
        repetitions=args.repetitions,
        comparison_cases=comparison_cases,
    )
    active_cases = COMPARISON_CASES if comparison_cases is None else comparison_cases
    if args.format == "json":
        print(json.dumps(results, ensure_ascii=True, indent=2))
    else:
        print(render_text(comparison_cases=active_cases, ycsig_results=results))
    return 0


if __name__ == "__main__":
    raise SystemExit(_main())
