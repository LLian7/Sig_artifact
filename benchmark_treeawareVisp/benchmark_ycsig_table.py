from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional

from benchmark_ycsig import YCSigBenchmarkCase, run_benchmark_case_average


@dataclass(frozen=True)
class PaperRow:
    label: str
    regime: str
    security_parameter: int
    hash_len: int
    max_g_value: int
    partition_size: int
    window_radius: int
    keygen: Optional[float]
    sign: Optional[float]
    verify: Optional[float]
    sig_size: Optional[float]


def _make_case(row: PaperRow, samples: int) -> YCSigBenchmarkCase:
    max_g_bit = (row.max_g_value.bit_length() - 1)
    block_num = row.hash_len // max_g_bit
    leaf_count = row.partition_size * row.max_g_value
    expected_retry_overhead = None
    expected_online_prg_cost = None
    if row.keygen is not None and row.sign is not None and row.verify is not None:
        expected_retry_overhead = row.sign + row.verify - row.keygen
        expected_online_prg_cost = (
            row.keygen + 2.0 - (3.0 * block_num + 3.0) / 2.0 - row.verify
        )
    tag = row.label.encode("utf-8")

    return YCSigBenchmarkCase(
        name=row.label,
        security_parameter=row.security_parameter,
        hash_len=row.hash_len,
        max_g_bit=max_g_bit,
        partition_size=row.partition_size,
        window_radius=row.window_radius,
        samples=samples,
        acceptance_mode="exact",
        retry_cost_mode="total",
        signature_extra_hash_values=1.0,
        signature_extra_bits=16,
        analytic_acceptance_probability_override=(
            None
            if expected_retry_overhead is None
            else 1.0 / expected_retry_overhead
        ),
        analytic_retry_overhead_override=expected_retry_overhead,
        analytic_online_prg_cost_override=expected_online_prg_cost,
        analytic_signature_hash_equivalents_override=row.sig_size,
        random_seed=row.security_parameter + row.hash_len + row.max_g_value,
        setup_kwargs={
            "key_seed": b"key/" + tag,
            "keyed_hash_key_seed": b"hk/" + tag,
            "ads_seed": b"ads/" + tag,
            "tweak_public_seed": b"twh/" + tag,
            "merkle_public_seed": b"mt/" + tag,
            "salt_bytes": 2,
        },
    )


def paper_rows() -> List[PaperRow]:
    return [
        PaperRow("YCSig-w4-k128-H>=k", "H>=k", 128, 130, 4, 33, 1, 393.0, 209.7284, 199.2686, 95.8048),
        PaperRow("YCSig-w4-k160-H>=k", "H>=k", 160, 160, 4, 42, 2, 500.0, 250.9478, 260.0856, 121.7240),
        PaperRow("YCSig-w4-k128-H>=2k", "H>=2k", 128, 256, 4, 42, 3, 500.0, 368.5226, 145.8762, 98.3092),
        PaperRow("YCSig-w4-k160-H>=2k", "H>=2k", 160, 320, 4, 53, 3, 631.0, 454.5740, 187.8584, 124.8372),
        PaperRow("YCSig-w8-k128-H>=k", "H>=k", 128, 165, 8, 17, 2, 405.0, 180.6744, 232.5872, 96.7930),
        PaperRow("YCSig-w8-k160-H>=k", "H>=k", 160, 207, 8, 21, 2, 500.0, 226.5684, 285.1164, 119.4896),
        PaperRow("YCSig-w8-k128-H>=2k", "H>=2k", 128, 258, 8, 18, 3, 429.0, 259.9382, 183.1560, 102.2474),
        PaperRow("YCSig-w8-k160-H>=2k", "H>=2k", 160, 321, 8, 22, 3, 524.0, 313.0382, 219.7100, 123.9320),
        PaperRow("YCSig-w16-k128-H>=k", "H>=k", 128, 192, 16, 9, 2, 429.0, 168.3560, 270.2670, 97.1512),
        PaperRow("YCSig-w16-k160-H>=k", "H>=k", 160, 244, 16, 11, 2, 524.0, 207.6674, 324.9810, 120.0470),
        PaperRow("YCSig-w16-k128-H>=2k", "H>=2k", 128, 256, 16, 9, 3, 429.0, 208.7708, 233.0516, 104.2110),
        PaperRow("YCSig-w16-k160-H>=2k", "H>=2k", 160, 320, 16, 11, 3, 524.0, 249.3084, 281.1066, 127.3594),
        PaperRow("YCSig-w32-k128-H>=k", "H>=k", 128, 205, 32, 5, 1, 477.0, 151.6456, 330.7180, 97.8146),
        PaperRow("YCSig-w32-k160-H>=k", "H>=k", 160, 320, 32, 6, 2, 573.0, 220.6466, 360.6018, 129.3582),
        PaperRow("YCSig-w32-k128-H>=2k", "H>=2k", 128, 320, 32, 5, 2, 477.0, 209.9730, 275.4040, 113.4386),
        PaperRow("YCSig-w32-k160-H>=2k", "H>=2k", 160, 320, 32, 6, 2, 573.0, 220.5728, 360.6268, 129.2032),
    ]


def _regime_for_case_name(case_name: str) -> str:
    if case_name == "case1":
        return "H>=k"
    if case_name == "case2":
        return "H>=2k"
    raise ValueError(f"unsupported case_name={case_name!r}")


def _label_from_search_row(raw: Dict[str, object]) -> str:
    case_name = str(raw["case_name"])
    regime = _regime_for_case_name(case_name)
    return (
        f"YCSig-w{int(raw['max_g_value'])}-k{int(raw['security_target'])}-"
        f"{regime}"
    )


def _load_paper_rows_from_json(path: str) -> List[PaperRow]:
    with open(path, "r", encoding="utf-8") as handle:
        raw_rows = json.load(handle)
    if not isinstance(raw_rows, list):
        raise ValueError("params-json must contain a JSON list of parameter rows")

    rows: List[PaperRow] = []
    for raw in raw_rows:
        if not isinstance(raw, dict):
            raise ValueError("each params-json row must be a JSON object")
        rows.append(
            PaperRow(
                label=_label_from_search_row(raw),
                regime=_regime_for_case_name(str(raw["case_name"])),
                security_parameter=int(raw["security_target"]),
                hash_len=int(raw["hash_len"]),
                max_g_value=int(raw["max_g_value"]),
                partition_size=int(raw["partition_num"]),
                window_radius=int(raw["window_radius"]),
                keygen=None,
                sign=None,
                verify=None,
                sig_size=None,
            )
        )
    return rows


def run_paper_table(
    samples: int,
    repetitions: int,
    rows: Optional[Iterable[PaperRow]] = None,
) -> List[Dict[str, object]]:
    results: List[Dict[str, object]] = []
    active_rows = list(paper_rows() if rows is None else rows)
    for row in active_rows:
        case = _make_case(row, samples)
        result = run_benchmark_case_average(case, repetitions)
        result["paper_row"] = {
            "regime": row.regime,
            "security_parameter": row.security_parameter,
            "hash_len": row.hash_len,
            "max_g_value": row.max_g_value,
            "partition_size": row.partition_size,
            "window_radius": row.window_radius,
            "keygen": row.keygen,
            "sign": row.sign,
            "verify": row.verify,
            "sig_size": row.sig_size,
        }
        results.append(result)
    return results


def _format_table(results: Iterable[Dict[str, object]]) -> str:
    header = (
        "Label | KeyGen | Sign | Verify | SigSize | "
        "EmpSig(obj) | AvgAttempts | VerifyRate"
    )
    rule = "-" * len(header)
    lines = [header, rule]
    for result in results:
        analytic = result["analytic"]
        empirical = result["empirical"]
        sig_size_override = analytic.get("signature_hash_equivalents_override")
        lines.append(
            (
                f"{result['case']} | "
                f"{analytic['keygen_hash_equivalents']:.1f} | "
                f"{analytic['sign_hash_equivalents']:.1f} | "
                f"{analytic['verify_hash_equivalents']:.1f} | "
                + (
                    f"{sig_size_override:.1f} | "
                    if sig_size_override is not None
                    else "- | "
                )
                + (
                f"{empirical['avg_signature_hash_equivalents_object_model']:.1f} | "
                f"{empirical['avg_attempts']:.2f} | "
                f"{empirical['verify_success_rate']:.2f}"
                )
            )
        )
    return "\n".join(lines)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Reproduce the YCSig rows from the uploaded comparison table.",
    )
    parser.add_argument(
        "--samples",
        type=int,
        default=64,
        help="Empirical samples per row.",
    )
    parser.add_argument(
        "--repetitions",
        type=int,
        default=1,
        help="Number of independent repetitions to average for each row.",
    )
    parser.add_argument(
        "--format",
        choices=("text", "json"),
        default="text",
        help="Output format.",
    )
    parser.add_argument(
        "--params-json",
        default=None,
        help="Optional JSON rows exported by search_ycsig_sigsize.py.",
    )
    return parser


def _main() -> int:
    args = _build_parser().parse_args()
    rows = None if args.params_json is None else _load_paper_rows_from_json(args.params_json)
    results = run_paper_table(args.samples, args.repetitions, rows=rows)
    if args.format == "json":
        print(json.dumps(results, ensure_ascii=True, indent=2))
    else:
        print(_format_table(results))
    return 0


if __name__ == "__main__":
    raise SystemExit(_main())
