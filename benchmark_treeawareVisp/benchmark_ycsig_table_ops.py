from __future__ import annotations

import argparse
import json
from typing import Dict, Iterable, List

from benchmark_ycsig_ops import run_operation_benchmark_case
from benchmark_ycsig_table import _make_case, paper_rows


def run_paper_table_ops(samples: int, repetitions: int) -> List[Dict[str, object]]:
    results: List[Dict[str, object]] = []
    for row in paper_rows():
        case = _make_case(row, samples)
        result = run_operation_benchmark_case(case, repetitions=repetitions)
        paper_retry = row.sign + row.verify - row.keygen
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
        result["comparison"] = {
            "paper_retry_hash_equivalents": paper_retry,
            "paper_sign_core_hash_equivalents": row.sign - paper_retry,
            "real_retry_attempt_count": result["operations"]["retry_attempt_count_real"],
            "real_retry_attempts_hash_equivalents": result["operations"][
                "retry_attempt_hash_equivalents_real"
            ],
            "real_retry_sampling_hash_equivalents": result["operations"][
                "retry_sampler_hash_equivalents_real"
            ],
            "real_retry_sampling_output_bits": result["operations"][
                "retry_sampler_output_bits_real"
            ],
            "real_signature_hash_equivalents_object_model": result["signature"][
                "avg_signature_hash_equivalents_object_model"
            ],
            "real_signature_hash_equivalents_concrete": result["signature"][
                "avg_signature_hash_equivalents_concrete"
            ],
        }
        results.append(result)
    return results


def _format_table(results: Iterable[Dict[str, object]]) -> str:
    header = (
        "Label | KeyGen(theory) | KeyGen(real) | "
        "Sign(theory) | Sign(real) | "
        "Verify(theory) | Verify(real) | Gap | SigSize(theory) | SigSize(real,obj)"
    )
    rule = "-" * len(header)
    lines = [header, rule]
    for result in results:
        paper_row = result["paper_row"]
        ops = result["operations"]
        sig = result["signature"]
        lines.append(
            (
                f"{result['case']} | "
                f"{paper_row['keygen']:.1f} | "
                f"{ops['keygen_hash_equivalents_real']:.1f} | "
                f"{paper_row['sign']:.1f} | "
                f"{ops['sign_hash_equivalents_real']:.1f} | "
                f"{paper_row['verify']:.1f} | "
                f"{ops['verify_hash_equivalents_real']:.1f} | "
                f"{ops['keygen_sign_relation_gap_real']:.1e} | "
                f"{paper_row['sig_size']:.1f} | "
                f"{sig['avg_signature_hash_equivalents_object_model']:.1f}"
            )
        )
    return "\n".join(lines)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run the YCSig paper rows using real primitive-operation counters.",
    )
    parser.add_argument("--samples", type=int, default=32, help="Samples per row.")
    parser.add_argument("--repetitions", type=int, default=10, help="Independent repetitions.")
    parser.add_argument(
        "--format",
        choices=("text", "json"),
        default="text",
        help="Output format.",
    )
    return parser


def _main() -> int:
    args = _build_parser().parse_args()
    results = run_paper_table_ops(args.samples, args.repetitions)
    if args.format == "json":
        print(json.dumps(results, ensure_ascii=True, indent=2))
    else:
        print(_format_table(results))
    return 0


if __name__ == "__main__":
    raise SystemExit(_main())
