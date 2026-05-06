from __future__ import annotations

import argparse
import json
from typing import Dict, Iterable, List, Optional

from benchmark_ycsig_cycles import run_cycle_benchmark_case
from benchmark_ycsig_table import (
    _load_paper_rows_from_json,
    _make_case,
    PaperRow,
    paper_rows,
)


def run_paper_table_cycles(
    *,
    samples: int,
    repetitions: int,
    cpu_frequency_hz: float | None = None,
    cpu_frequency_ghz: float | None = None,
    measure_memory: bool = False,
    rows: Optional[Iterable[PaperRow]] = None,
) -> List[Dict[str, object]]:
    results: List[Dict[str, object]] = []
    active_rows = list(paper_rows() if rows is None else rows)
    for row in active_rows:
        case = _make_case(row, samples)
        result = run_cycle_benchmark_case(
            case,
            repetitions=repetitions,
            cpu_frequency_hz=cpu_frequency_hz,
            cpu_frequency_ghz=cpu_frequency_ghz,
            measure_memory=measure_memory,
        )
        result["paper_row"] = {
            "regime": row.regime,
            "security_parameter": row.security_parameter,
            "hash_len": row.hash_len,
            "max_g_value": row.max_g_value,
            "partition_size": row.partition_size,
            "window_radius": row.window_radius,
            "sig_size": row.sig_size,
        }
        results.append(result)
    return results


def _format_table(results: Iterable[Dict[str, object]]) -> str:
    header = (
        "Label | KeyGenCycles | RetryCycles(total) | SignCycles | VerifyCycles | SigSize(theory) | SigSize(real,obj) | "
        "TotalExperiments | VerifyRate"
    )
    rule = "-" * len(header)
    lines = [header, rule]
    for result in results:
        cycles = result["cycles"]
        signature = result["signature"]
        params = result["parameters"]
        sig_theory = signature["signature_hash_equivalents_override"]
        lines.append(
            (
                f"{result['case']} | "
                f"{cycles['avg_keygen_cycles']:.1f} | "
                f"{cycles['avg_retry_cycles']:.1f} | "
                f"{cycles['avg_sign_cycles']:.1f} | "
                f"{cycles['avg_verify_cycles']:.1f} | "
                + (
                    f"{sig_theory:.1f} | " if sig_theory is not None else "- | "
                )
                + (
                f"{signature['avg_signature_hash_equivalents_object_model']:.1f} | "
                f"{params['total_experiments']} | "
                f"{result['verify_rate']:.2f}"
                )
            )
        )
    return "\n".join(lines)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Benchmark the uploaded YCSig table rows in estimated CPU cycles.",
    )
    parser.add_argument("--samples", type=int, default=32, help="Samples per repetition.")
    parser.add_argument("--repetitions", type=int, default=40, help="Independent repetitions.")
    parser.add_argument("--cpu-frequency-hz", type=float, default=None, help="Nominal CPU frequency in Hz.")
    parser.add_argument("--cpu-frequency-ghz", type=float, default=None, help="Nominal CPU frequency in GHz.")
    parser.add_argument("--format", choices=("text", "json"), default="text", help="Output format.")
    parser.add_argument(
        "--measure-memory",
        action="store_true",
        help="Also measure peak Python-heap usage with tracemalloc.",
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
    results = run_paper_table_cycles(
        samples=args.samples,
        repetitions=args.repetitions,
        cpu_frequency_hz=args.cpu_frequency_hz,
        cpu_frequency_ghz=args.cpu_frequency_ghz,
        measure_memory=args.measure_memory,
        rows=rows,
    )
    if args.format == "json":
        print(json.dumps(results, ensure_ascii=True, indent=2))
    else:
        print(_format_table(results))
    return 0


if __name__ == "__main__":
    raise SystemExit(_main())
