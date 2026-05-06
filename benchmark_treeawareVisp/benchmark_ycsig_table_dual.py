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


def _latex_escape(text: str) -> str:
    escaped = (
        text.replace("\\", r"\textbackslash{}")
        .replace("&", r"\&")
        .replace("%", r"\%")
        .replace("$", r"\$")
        .replace("#", r"\#")
        .replace("_", r"\_")
        .replace("{", r"\{")
        .replace("}", r"\}")
    )
    return escaped.replace(">=", r"$\geq$")


def _latex_number(value: object) -> str:
    if value is None:
        return "--"
    if isinstance(value, int):
        return str(value)
    return f"{float(value):.1f}"


def _latex_gap(value: float) -> str:
    if abs(value) < 1e-12:
        return "0"
    mantissa, exponent = f"{value:.1e}".split("e")
    return rf"${mantissa} \times 10^{{{int(exponent)}}}$"


def _render_latex_table(
    *,
    caption: str,
    label: str,
    column_spec: str,
    headers: List[str],
    rows: List[List[str]],
) -> str:
    lines = [
        r"\begin{table}[t]",
        r"\centering",
        r"\small",
        r"\setlength{\tabcolsep}{4pt}",
        f"\\caption{{{caption}}}",
        f"\\label{{{label}}}",
        f"\\begin{{tabular}}{{{column_spec}}}",
        r"\hline",
        " & ".join(headers) + r" \\",
        r"\hline",
    ]
    lines.extend(" & ".join(row) + r" \\" for row in rows)
    lines.extend(
        [
            r"\hline",
            r"\end{tabular}",
            r"\end{table}",
        ]
    )
    return "\n".join(lines)


def _format_ops_latex(results: Iterable[Dict[str, object]]) -> str:
    rows = list(results)
    first = rows[0]
    params = first["parameters"]
    caption = (
        "YCSig paper-table benchmark in real primitive-operation counts "
        f"(samples={params['samples']}, repetitions={params['repetitions']})."
    )
    latex_rows: List[List[str]] = []
    for result in rows:
        paper_row = result["paper_row"]
        ops = result["operations"]
        sig = result["signature"]
        latex_rows.append(
            [
                _latex_escape(str(result["case"])),
                _latex_number(paper_row["keygen"]),
                _latex_number(ops["keygen_hash_equivalents_real"]),
                _latex_number(paper_row["sign"]),
                _latex_number(ops["sign_hash_equivalents_real"]),
                _latex_number(paper_row["verify"]),
                _latex_number(ops["verify_hash_equivalents_real"]),
                _latex_gap(float(ops["keygen_sign_relation_gap_real"])),
                _latex_number(paper_row["sig_size"]),
                _latex_number(sig["avg_signature_hash_equivalents_object_model"]),
            ]
        )
    return _render_latex_table(
        caption=caption,
        label="tab:ycsig-dual-ops",
        column_spec="lrrrrrrrrrr",
        headers=[
            "Case",
            "KG(th)",
            "KG(real)",
            "S(th)",
            "S(real)",
            "V(th)",
            "V(real)",
            "Gap",
            "Sig(th)",
            "Sig(real)",
        ],
        rows=latex_rows,
    )


def _format_cycles_latex(results: Iterable[Dict[str, object]]) -> str:
    rows = list(results)
    first = rows[0]
    params = first["parameters"]
    freq = params.get("cpu_frequency_ghz")
    frequency_clause = ""
    if freq is not None:
        frequency_clause = f", cpu={float(freq):.2f} GHz"
    caption = (
        "YCSig paper-table benchmark in estimated CPU cycles "
        f"(samples={params['samples']}, repetitions={params['repetitions']}{frequency_clause})."
    )
    latex_rows: List[List[str]] = []
    for result in rows:
        cycles = result["cycles"]
        signature = result["signature"]
        result_params = result["parameters"]
        latex_rows.append(
            [
                _latex_escape(str(result["case"])),
                _latex_number(cycles["avg_keygen_cycles"]),
                _latex_number(cycles["avg_retry_cycles"]),
                _latex_number(cycles["avg_sign_cycles"]),
                _latex_number(cycles["avg_verify_cycles"]),
                _latex_number(signature["signature_hash_equivalents_override"]),
                _latex_number(signature["avg_signature_hash_equivalents_object_model"]),
                _latex_number(result_params["total_experiments"]),
                f"{float(result['verify_rate']):.2f}",
            ]
        )
    return _render_latex_table(
        caption=caption,
        label="tab:ycsig-dual-cycles",
        column_spec="lrrrrrrrr",
        headers=[
            "Case",
            "KG cyc.",
            "Re cyc.",
            "S cyc.",
            "V cyc.",
            "Sig(th)",
            "Sig(real)",
            "N",
            "VR",
        ],
        rows=latex_rows,
    )


def _format_latex(results: Dict[str, List[Dict[str, object]]]) -> str:
    sections: List[str] = []
    if "ops" in results:
        sections.append(_format_ops_latex(results["ops"]))
    if "cycles" in results:
        sections.append(_format_cycles_latex(results["cycles"]))
    return "\n\n".join(sections)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run YCSig paper-table benchmarks in dual mode: real primitive counts and estimated CPU cycles.",
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
        "--format",
        choices=("text", "json", "latex"),
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
        mode=args.mode,
    )
    if args.format == "json":
        print(json.dumps(results, ensure_ascii=True, indent=2))
    elif args.format == "latex":
        print(_format_latex(results))
    else:
        print(_format_text(results))
    return 0


if __name__ == "__main__":
    raise SystemExit(_main())
