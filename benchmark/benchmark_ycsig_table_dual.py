from __future__ import annotations

import argparse
import json
from typing import Dict, Iterable, List

from benchmark_ycsig_table_cycles import run_paper_table_cycles
from benchmark_ycsig_table_ops import run_paper_table_ops


def run_paper_table_dual(
    *,
    samples: int,
    repetitions: int,
    cpu_frequency_hz: float | None = None,
    cpu_frequency_ghz: float | None = None,
    cycle_backend: str = "estimated",
    mode: str = "both",
) -> Dict[str, List[Dict[str, object]]]:
    if mode not in {"ops", "cycles", "both"}:
        raise ValueError("mode must be one of {'ops', 'cycles', 'both'}")

    results: Dict[str, List[Dict[str, object]]] = {}
    if mode in {"ops", "both"}:
        results["ops"] = run_paper_table_ops(samples=samples, repetitions=repetitions)
    if mode in {"cycles", "both"}:
        results["cycles"] = run_paper_table_cycles(
            samples=samples,
            repetitions=repetitions,
            cpu_frequency_hz=cpu_frequency_hz,
            cpu_frequency_ghz=cpu_frequency_ghz,
            cycle_backend=cycle_backend,
        )
    return results


def _format_ops(results: Iterable[Dict[str, object]]) -> str:
    header = (
        "Label | KeyGen(theory) | KeyGen(real) | "
        "Sign(theory) | Sign(real) | "
        "Verify(theory) | Verify(real) | Gap | SigSize(theory) | SigSize(real,obj)"
    )
    rule = "-" * len(header)
    lines = ["[Ops]", header, rule]
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


def _format_cycles(results: Iterable[Dict[str, object]]) -> str:
    header = (
        "Label | KeyGenCycles | RetryCycles(total) | SignCycles | VerifyCycles | SigSize(theory) | SigSize(real,obj) | "
        "TotalExperiments | VerifyRate"
    )
    rule = "-" * len(header)
    lines = ["[Cycles]", header, rule]
    for result in results:
        cycles = result["cycles"]
        signature = result["signature"]
        params = result["parameters"]
        lines.append(
            (
                f"{result['case']} | "
                f"{cycles['avg_keygen_cycles']:.1f} | "
                f"{cycles['avg_retry_cycles']:.1f} | "
                f"{cycles['avg_sign_cycles']:.1f} | "
                f"{cycles['avg_verify_cycles']:.1f} | "
                f"{signature['signature_hash_equivalents_override']:.1f} | "
                f"{signature['avg_signature_hash_equivalents_object_model']:.1f} | "
                f"{params['total_experiments']} | "
                f"{result['verify_rate']:.2f}"
            )
        )
    return "\n".join(lines)


def _format_text(results: Dict[str, List[Dict[str, object]]]) -> str:
    sections: List[str] = []
    if "ops" in results:
        sections.append(_format_ops(results["ops"]))
    if "cycles" in results:
        sections.append(_format_cycles(results["cycles"]))
    return "\n\n".join(sections)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run YCSig paper-table benchmarks in dual mode: real primitive counts and CPU cycles measured either by nominal-frequency estimates or xctrace hardware counters.",
    )
    parser.add_argument(
        "--samples",
        type=int,
        default=5,
        help="Samples per repetition. Default is 5.",
    )
    parser.add_argument(
        "--repetitions",
        type=int,
        default=10,
        help="Independent repetitions. Default is 10.",
    )
    parser.add_argument(
        "--mode",
        choices=("ops", "cycles", "both"),
        default="both",
        help="Benchmark mode. Default runs both ops and cycles.",
    )
    parser.add_argument("--cpu-frequency-hz", type=float, default=None, help="Nominal CPU frequency in Hz.")
    parser.add_argument("--cpu-frequency-ghz", type=float, default=None, help="Nominal CPU frequency in GHz.")
    parser.add_argument(
        "--cycle-backend",
        choices=("estimated", "xctrace"),
        default="estimated",
        help="Cycle measurement backend. 'estimated' uses CPU time times a nominal frequency; 'xctrace' uses macOS CPU Counters.",
    )
    parser.add_argument(
        "--format",
        choices=("text", "json"),
        default="text",
        help="Output format.",
    )
    return parser


def _main() -> int:
    args = _build_parser().parse_args()
    results = run_paper_table_dual(
        samples=args.samples,
        repetitions=args.repetitions,
        cpu_frequency_hz=args.cpu_frequency_hz,
        cpu_frequency_ghz=args.cpu_frequency_ghz,
        cycle_backend=args.cycle_backend,
        mode=args.mode,
    )
    if args.format == "json":
        print(json.dumps(results, ensure_ascii=True, indent=2))
    else:
        print(_format_text(results))
    return 0


if __name__ == "__main__":
    raise SystemExit(_main())
