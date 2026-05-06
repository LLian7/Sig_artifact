from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from decimal import Decimal, ROUND_HALF_UP
from typing import Dict, Iterable, List, Tuple

from benchmark_ycsig import YCSigBenchmarkCase
from benchmark_ycsig_ops import run_operation_benchmark_case


@dataclass(frozen=True)
class BaselineRow:
    scheme: str
    max_g_value: int
    keygen: str
    sign: str
    verify: str
    sig_size: str


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


CASE_COLUMNS: List[Tuple[str, int]] = [
    ("case1", 128),
    ("case1", 160),
    ("case2", 128),
    ("case2", 160),
]


COMPARISON_CASES: List[ComparisonCase] = [
    ComparisonCase("case1", 128, 4, 33, 1, 65, 130),
    ComparisonCase("case1", 128, 8, 17, 2, 55, 165),
    ComparisonCase("case1", 128, 16, 9, 2, 48, 192),
    ComparisonCase("case1", 128, 32, 5, 1, 41, 205),
    ComparisonCase("case1", 160, 4, 42, 2, 80, 160),
    ComparisonCase("case1", 160, 8, 21, 2, 69, 207),
    ComparisonCase("case1", 160, 16, 11, 2, 61, 244),
    ComparisonCase("case1", 160, 32, 6, 2, 64, 320),
    ComparisonCase("case2", 128, 4, 42, 3, 128, 256),
    ComparisonCase("case2", 128, 8, 18, 3, 86, 258),
    ComparisonCase("case2", 128, 16, 9, 3, 64, 256),
    ComparisonCase("case2", 128, 32, 5, 2, 64, 320),
    ComparisonCase("case2", 160, 4, 53, 3, 160, 320),
    ComparisonCase("case2", 160, 8, 22, 3, 107, 321),
    ComparisonCase("case2", 160, 16, 11, 3, 80, 320),
    ComparisonCase("case2", 160, 32, 6, 2, 64, 320),
]


BASELINES: Dict[int, List[BaselineRow]] = {
    4: [
        BaselineRow("WOTS+C", 4, "319", "182", "160", "64"),
        BaselineRow("TL1C", 4, "449", "258", "193", "65"),
    ],
    8: [
        BaselineRow("WOTS+C", 8, "377", "338", "189", "42"),
        BaselineRow("TL1C", 8, "496", "347", "153", "44"),
    ],
    16: [
        BaselineRow("WOTS+C", 16, "543", "337", "272", "32"),
        BaselineRow("TL1C", 16, "993", "514", "481", "33"),
    ],
    32: [
        BaselineRow("WOTS+C", 32, "824", "1339", "413", "25"),
        BaselineRow("TL1C", 32, "1205", "835", "374", "27"),
    ],
}


EXTRA_BASELINES = [
    (
        "TL1C",
        86,
        ["2416", "2067", "353", "25", "-", "-", "-", "-", "-", "-", "-", "-", "-", "-", "-", "-"],
    ),
    (
        "TL1C",
        56,
        ["-", "-", "-", "-", "2208", "1907", "305", "35", "-", "-", "-", "-", "-", "-", "-", "-"],
    ),
]


BASELINE_VALUES_BY_COLUMN: Dict[int, Dict[str, Dict[str, int]]] = {
    4: {
        "case1_k128": {"keygen": 319, "sign": 182, "verify": 160, "sig_size": 64},
        "case1_k160": {"keygen": 399, "sign": 225, "verify": 200, "sig_size": 80},
        "case2_k128": {"keygen": 639, "sign": 352, "verify": 319, "sig_size": 128},
        "case2_k160": {"keygen": 959, "sign": 519, "verify": 479, "sig_size": 192},
    },
    8: {
        "case1_k128": {"keygen": 377, "sign": 338, "verify": 189, "sig_size": 42},
        "case1_k160": {"keygen": 476, "sign": 322, "verify": 239, "sig_size": 53},
        "case2_k128": {"keygen": 764, "sign": 435, "verify": 382, "sig_size": 85},
        "case2_k160": {"keygen": 1151, "sign": 641, "verify": 575, "sig_size": 128},
    },
    16: {
        "case1_k128": {"keygen": 543, "sign": 337, "verify": 272, "sig_size": 32},
        "case1_k160": {"keygen": 679, "sign": 413, "verify": 340, "sig_size": 40},
        "case2_k128": {"keygen": 1087, "sign": 637, "verify": 543, "sig_size": 64},
        "case2_k160": {"keygen": 1631, "sign": 929, "verify": 815, "sig_size": 96},
    },
    32: {
        "case1_k128": {"keygen": 824, "sign": 1339, "verify": 413, "sig_size": 25},
        "case1_k160": {"keygen": 1055, "sign": 659, "verify": 528, "sig_size": 32},
        "case2_k128": {"keygen": 1682, "sign": 1172, "verify": 842, "sig_size": 52},
        "case2_k160": {"keygen": 2464, "sign": 1396, "verify": 1193, "sig_size": 77},
    },
}


METRICS = [
    ("keygen", "keygen_hash_equivalents_real"),
    ("sign", "sign_hash_equivalents_real"),
    ("verify", "verify_hash_equivalents_real"),
    ("sig_size", "avg_signature_hash_equivalents_object_model"),
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


def _format_display_value(
    *,
    metric: str,
    value: float,
    max_g_value: int,
    column_key: str,
) -> str:
    rounded = _round_half_up(value)
    baseline_values = BASELINE_VALUES_BY_COLUMN[max_g_value][column_key]
    is_best = value < baseline_values[metric]
    text = str(rounded)
    return rf"\textbf{{{text}}}" if is_best else text


def _ycsig_row(
    *,
    max_g_value: int,
    values_by_column: Dict[str, Dict[str, float]],
) -> str:
    cells = [r"$\YCSig$", str(max_g_value)]
    for case_name, security_target in CASE_COLUMNS:
        column_key = f"{case_name}_k{security_target}"
        metric_values = values_by_column[column_key]
        cells.append(
            _format_display_value(
                metric="keygen",
                value=metric_values["keygen"],
                max_g_value=max_g_value,
                column_key=column_key,
            )
        )
        cells.append(
            _format_display_value(
                metric="sign",
                value=metric_values["sign"],
                max_g_value=max_g_value,
                column_key=column_key,
            )
        )
        cells.append(
            _format_display_value(
                metric="verify",
                value=metric_values["verify"],
                max_g_value=max_g_value,
                column_key=column_key,
            )
        )
        cells.append(
            _format_display_value(
                metric="sig_size",
                value=metric_values["sig_size"],
                max_g_value=max_g_value,
                column_key=column_key,
            )
        )
    return " & ".join(cells) + r" \\ \hline"


def _baseline_row(
    *,
    row: BaselineRow,
    values: Iterable[str],
) -> str:
    value_list = list(values)
    formatted_values = [
        rf"\multicolumn{{1}}{{c|}}{{{cell}}}" if index != len(value_list) else cell
        for index, cell in enumerate(value_list, start=1)
    ]
    return f"{row.scheme} & {row.max_g_value} & " + " & ".join(formatted_values) + r" \\ \hline"


def _format_baseline_rows() -> List[str]:
    rows: List[str] = []
    baseline_values = {
        4: {
            "WOTS+C": ["319", "182", "160", "64", "399", "225", "200", "80", "639", "352", "319", "128", "959", "519", "479", "192"],
            "TL1C": ["449", "258", "193", "65", "561", "322", "241", "81", "697", "523", "178", "131", "871", "651", "224", "163"],
        },
        8: {
            "WOTS+C": ["377", "338", "189", "42", "476", "322", "239", "53", "764", "435", "382", "85", "1151", "641", "575", "128"],
            "TL1C": ["496", "347", "153", "44", "611", "435", "180", "55", "969", "699", "274", "88", "1207", "867", "344", "109"],
        },
        16: {
            "WOTS+C": ["543", "337", "272", "32", "679", "413", "340", "40", "1087", "637", "543", "64", "1631", "929", "815", "96"],
            "TL1C": ["993", "514", "481", "33", "1241", "642", "601", "41", "1473", "1043", "434", "66", "1842", "1299", "547", "82"],
        },
        32: {
            "WOTS+C": ["824", "1339", "413", "25", "1055", "659", "528", "32", "1682", "1172", "842", "52", "2464", "1396", "1193", "77"],
            "TL1C": ["1205", "835", "374", "27", "2017", "1026", "993", "33", "2371", "1667", "708", "53", "2977", "2083", "898", "66"],
        },
    }
    for max_g_value in (4, 8, 16, 32):
        for scheme in ("WOTS+C", "TL1C"):
            rows.append(
                _baseline_row(
                    row=BaselineRow(scheme, max_g_value, "", "", "", ""),
                    values=baseline_values[max_g_value][scheme],
                )
            )
    return rows


def render_updated_table(
    *,
    repetitions: int,
    ycsig_results: Dict[int, Dict[str, Dict[str, float]]],
) -> str:
    lines = [
        r"\begin{table}[!htp]",
        r"\vspace{-1mm}",
        r"\footnotesize",
        r"\setlength{\tabcolsep}{2pt}",
        rf"\caption{{Comparison. \small{{Costs are counted in the number of (average) hash-equivalent values/calls. Bold entries indicate that $\YCSig$ outperforms both baselines under the same base, case, security level, and metric. The $\YCSig$ rows are re-measured from the implementation with {repetitions} independent repetitions per parameter set.}}}}",
        r"\label{tab:comparison}",
        r"\resizebox{0.999\textwidth}{!}{",
        r"\begin{tabular}{c|c|cccccccc|cccccccc}",
        r"\toprule",
        r"\multirow{3}{*}{Schemes} & \multirow{3}{*}{$\MaxGValue$} & \multicolumn{8}{c|}{\text{Case} ($\HashLen \ge \kappa^{*}$)} & \multicolumn{8}{c}{\text{Case} ($\HashLen \ge 2\kappa^{*}$)} \\ \cline{3-18}",
        r" &  & \multicolumn{4}{c|}{$\kappa^{*}=128$} & \multicolumn{4}{c|}{$\kappa^{*}=160$} & \multicolumn{4}{c|}{$\kappa^{*}=128$} & \multicolumn{4}{c}{$\kappa^{*}=160$} \\ \cline{3-18}",
        r" &  & \multicolumn{1}{c|}{\begin{tabular}[c]{@{}c@{}}Key\\ Gen\end{tabular}} & \multicolumn{1}{c|}{Sign} & \multicolumn{1}{c|}{Verify} & \multicolumn{1}{c|}{\begin{tabular}[c]{@{}c@{}}Sig.\\ Size\end{tabular}} & \multicolumn{1}{c|}{\begin{tabular}[c]{@{}c@{}}Key\\ Gen\end{tabular}} & \multicolumn{1}{c|}{Sign} & \multicolumn{1}{c|}{Verify} & \begin{tabular}[c]{@{}c@{}}Sig.\\ Size\end{tabular} & \multicolumn{1}{c|}{\begin{tabular}[c]{@{}c@{}}Key\\ Gen\end{tabular}} & \multicolumn{1}{c|}{Sign} & \multicolumn{1}{c|}{Verify} & \multicolumn{1}{c|}{\begin{tabular}[c]{@{}c@{}}Sig.\\ Size\end{tabular}} & \multicolumn{1}{c|}{\begin{tabular}[c]{@{}c@{}}Key\\ Gen\end{tabular}} & \multicolumn{1}{c|}{Sign} & \multicolumn{1}{c|}{Verify} & \begin{tabular}[c]{@{}c@{}}Sig.\\ Size\end{tabular} \\ \hline",
    ]

    lines.extend(_format_baseline_rows()[0:2])
    lines.append(_ycsig_row(max_g_value=4, values_by_column=ycsig_results[4]))
    lines.extend(_format_baseline_rows()[2:4])
    lines.append(_ycsig_row(max_g_value=8, values_by_column=ycsig_results[8]))
    lines.extend(_format_baseline_rows()[4:6])
    lines.append(_ycsig_row(max_g_value=16, values_by_column=ycsig_results[16]))
    lines.extend(_format_baseline_rows()[6:8])
    lines.append(_ycsig_row(max_g_value=32, values_by_column=ycsig_results[32]))

    for scheme, max_g_value, values in EXTRA_BASELINES:
        formatted_values = [
            rf"\multicolumn{{1}}{{c|}}{{{cell}}}" if index != len(values) else cell
            for index, cell in enumerate(values, start=1)
        ]
        lines.append(f"{scheme} & {max_g_value} & " + " & ".join(formatted_values) + r" \\")
        if (scheme, max_g_value) != EXTRA_BASELINES[-1][:2]:
            lines[-1] += " \\hline"

    lines.extend(
        [
            r"",
            r"\bottomrule",
            r"\end{tabular}",
            r"}",
            r"\vspace{-2mm}",
            r"\end{table}",
        ]
    )
    return "\n".join(lines)


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
        choices=("latex", "text", "json"),
        default="latex",
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
    elif args.format == "text":
        print(render_text(comparison_cases=active_cases, ycsig_results=results))
    else:
        print(render_updated_table(repetitions=args.repetitions, ycsig_results=results))
    return 0


if __name__ == "__main__":
    raise SystemExit(_main())
