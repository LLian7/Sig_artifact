from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

from benchmark_val_strict_isp_cycles import run_val_strict_isp_cycle_benchmark
from benchmark_ycsig_table import PaperRow
from benchmark_ycsig_table_cycles import run_paper_table_cycles
from benchmark_ycsig_table_ops import run_paper_table_ops


ROOT = Path(__file__).resolve().parent
OUTPUT_128_160 = ROOT / "implementation_results_128_160_current.json"
OUTPUT_192 = ROOT / "implementation_results_192_current.json"
OUTPUT_VALSTRICTISP = ROOT / "val_strict_isp_cycles_case12_multiw_results_current.json"


@dataclass(frozen=True)
class ValStrictISPSpec:
    case_name: str
    security_target: int
    max_g_value: int
    hash_len: int
    partition_num: int
    window_radius: int


VALSTRICTISP_SPECS = (
    ValStrictISPSpec("case1", 128, 4, 128, 32, 2),
    ValStrictISPSpec("case1", 160, 4, 160, 40, 2),
    ValStrictISPSpec("case1", 192, 4, 192, 48, 2),
    ValStrictISPSpec("case2", 128, 4, 256, 42, 6),
    ValStrictISPSpec("case2", 160, 4, 320, 52, 5),
    ValStrictISPSpec("case2", 192, 4, 384, 62, 3),
    ValStrictISPSpec("case1", 128, 8, 168, 16, 2),
    ValStrictISPSpec("case1", 160, 8, 219, 20, 2),
    ValStrictISPSpec("case1", 192, 8, 288, 24, 3),
    ValStrictISPSpec("case2", 128, 8, 258, 17, 3),
    ValStrictISPSpec("case2", 160, 8, 321, 22, 6),
    ValStrictISPSpec("case2", 192, 8, 384, 26, 6),
    ValStrictISPSpec("case1", 128, 16, 248, 8, 2),
    ValStrictISPSpec("case1", 160, 16, 292, 10, 2),
    ValStrictISPSpec("case1", 192, 16, 384, 12, 3),
    ValStrictISPSpec("case2", 128, 16, 256, 9, 4),
    ValStrictISPSpec("case2", 160, 16, 320, 11, 5),
    ValStrictISPSpec("case2", 192, 16, 384, 12, 3),
    ValStrictISPSpec("case1", 128, 32, 200, 5, 1),
    ValStrictISPSpec("case1", 160, 32, 320, 6, 2),
    ValStrictISPSpec("case1", 192, 32, 320, 7, 2),
    ValStrictISPSpec("case2", 128, 32, 320, 5, 2),
    ValStrictISPSpec("case2", 160, 32, 320, 6, 2),
    ValStrictISPSpec("case2", 192, 32, 385, 7, 2),
)

YCSIG_192_ROWS = (
    PaperRow("YCSig-w4-k192-H>=k", "H>=k", 192, 192, 4, 48, 2, None, None, None, None),
    PaperRow("YCSig-w4-k192-H>=2k", "H>=2k", 192, 384, 4, 62, 3, None, None, None, None),
    PaperRow("YCSig-w8-k192-H>=k", "H>=k", 192, 288, 8, 24, 3, None, None, None, None),
    PaperRow("YCSig-w8-k192-H>=2k", "H>=2k", 192, 384, 8, 26, 6, None, None, None, None),
    PaperRow("YCSig-w16-k192-H>=k", "H>=k", 192, 384, 16, 12, 3, None, None, None, None),
    PaperRow("YCSig-w16-k192-H>=2k", "H>=2k", 192, 384, 16, 12, 3, None, None, None, None),
    PaperRow("YCSig-w32-k192-H>=k", "H>=k", 192, 320, 32, 7, 2, None, None, None, None),
    PaperRow("YCSig-w32-k192-H>=2k", "H>=2k", 192, 385, 32, 7, 2, None, None, None, None),
)


def _merge_ycsig_results(
    cycle_results: Iterable[Dict[str, object]],
    ops_results: Iterable[Dict[str, object]],
) -> List[Dict[str, object]]:
    ops_by_case = {str(result["case"]): result for result in ops_results}
    merged: List[Dict[str, object]] = []
    for cycle_result in cycle_results:
        case_name = str(cycle_result["case"])
        ops_result = ops_by_case[case_name]
        paper_row = cycle_result["paper_row"]
        merged.append(
            {
                "label": case_name,
                "regime": paper_row["regime"],
                "security_parameter": paper_row["security_parameter"],
                "hash_len": paper_row["hash_len"],
                "max_g_value": paper_row["max_g_value"],
                "partition_size": paper_row["partition_size"],
                "window_radius": paper_row["window_radius"],
                "cycles": {
                    "keygen_mcycles": cycle_result["cycles"]["avg_keygen_cycles"] / 1_000_000.0,
                    "sign_mcycles": cycle_result["cycles"]["avg_sign_cycles"] / 1_000_000.0,
                    "verify_mcycles": cycle_result["cycles"]["avg_verify_cycles"] / 1_000_000.0,
                },
                "benchmark": {
                    "cycle_backend": cycle_result["parameters"]["cycle_backend"],
                    "cpu_frequency_ghz": cycle_result["parameters"]["cpu_frequency_ghz"],
                },
                "memory": {
                    "peak_mib": cycle_result["memory"]["avg_peak_memory_mib"],
                },
                "signature": {
                    "sig_size_obj_cycles": cycle_result["signature"]["avg_signature_hash_equivalents_object_model"],
                    "sig_size_obj_ops": ops_result["signature"]["avg_signature_hash_equivalents_object_model"],
                },
                "ops": {
                    "keygen": ops_result["operations"]["keygen_hash_equivalents_real"],
                    "sign": ops_result["operations"]["sign_hash_equivalents_real"],
                    "verify": ops_result["operations"]["verify_hash_equivalents_real"],
                },
            }
        )
    merged.sort(key=lambda row: (row["security_parameter"], row["max_g_value"], row["regime"]))
    return merged


def _split_ycsig_by_security(rows: Iterable[Dict[str, object]]) -> Tuple[List[Dict[str, object]], List[Dict[str, object]]]:
    rows_128_160: List[Dict[str, object]] = []
    rows_192: List[Dict[str, object]] = []
    for row in rows:
        security = int(row["security_parameter"])
        if security in {128, 160}:
            rows_128_160.append(row)
        elif security == 192:
            rows_192.append(row)
        else:
            raise ValueError(f"unexpected security parameter: {security}")
    return rows_128_160, rows_192


def _refresh_ycsig(
    *,
    samples: int,
    repetitions: int,
    cpu_frequency_ghz: float,
    cycle_backend: str,
    xctrace_target_operations: int,
) -> Tuple[List[Dict[str, object]], List[Dict[str, object]]]:
    cycle_results_128_160 = run_paper_table_cycles(
        samples=samples,
        repetitions=repetitions,
        cpu_frequency_ghz=cpu_frequency_ghz,
        cycle_backend=cycle_backend,
        xctrace_target_operations=xctrace_target_operations,
        measure_memory=True,
    )
    ops_results_128_160 = run_paper_table_ops(
        samples=samples,
        repetitions=repetitions,
    )
    cycle_results_192 = run_paper_table_cycles(
        samples=samples,
        repetitions=repetitions,
        cpu_frequency_ghz=cpu_frequency_ghz,
        cycle_backend=cycle_backend,
        xctrace_target_operations=xctrace_target_operations,
        measure_memory=True,
        rows=YCSIG_192_ROWS,
    )
    ops_results_192 = run_paper_table_ops(
        samples=samples,
        repetitions=repetitions,
        rows=YCSIG_192_ROWS,
    )
    rows_128_160 = _merge_ycsig_results(cycle_results_128_160, ops_results_128_160)
    rows_192 = _merge_ycsig_results(cycle_results_192, ops_results_192)
    return rows_128_160, rows_192


def _refresh_valstrictisp(
    *,
    accepted_samples: int,
    rejected_samples: int,
    repetitions: int,
    cpu_frequency_ghz: float,
    cycle_backend: str,
    xctrace_target_operations: int,
    sampler_mode: str,
    random_seed: int,
    max_candidate_attempts: int,
) -> List[Dict[str, object]]:
    rows: List[Dict[str, object]] = []
    total_specs = len(VALSTRICTISP_SPECS)
    for index, spec in enumerate(VALSTRICTISP_SPECS, start=1):
        max_g_value = spec.max_g_value
        max_g_bit = max_g_value.bit_length() - 1
        print(
            (
                f"[ValStrictISP {index}/{total_specs}] "
                f"{spec.case_name} k={spec.security_target} w={max_g_value} "
                f"HashLen={spec.hash_len} PartitionNum={spec.partition_num} "
                f"WindowRadius={spec.window_radius}"
            ),
            file=sys.stderr,
            flush=True,
        )
        result = run_val_strict_isp_cycle_benchmark(
            hash_len=spec.hash_len,
            max_g_bit=max_g_bit,
            partition_num=spec.partition_num,
            window_radius=spec.window_radius,
            hash_name="shake_256",
            accepted_samples=accepted_samples,
            rejected_samples=rejected_samples,
            repetitions=repetitions,
            random_seed=random_seed,
            max_candidate_attempts=max_candidate_attempts,
            cpu_frequency_ghz=cpu_frequency_ghz,
            cycle_backend=cycle_backend,
            xctrace_target_operations=xctrace_target_operations,
            sampler_mode=sampler_mode,
        )
        rows.append(
            {
                "case_name": spec.case_name,
                "security_target": spec.security_target,
                "max_g_value": max_g_value,
                "partition_num": spec.partition_num,
                "window_radius": spec.window_radius,
                "block_num": result["parameters"]["hash_len"] // result["parameters"]["max_g_bit"],
                "hash_len": spec.hash_len,
                "benchmark": {
                    "accepted_samples": accepted_samples,
                    "rejected_samples": rejected_samples,
                    "repetitions": repetitions,
                    "cycle_backend": result["parameters"]["cycle_backend"],
                    "cpu_frequency_ghz": result["parameters"]["cpu_frequency_ghz"],
                    "sampler_mode": result["parameters"]["sampler_mode"],
                },
                "cycles": {
                    "profile_only": result["cycles"]["profile_only"]["avg_cycles"],
                    "accept_check": result["cycles"]["accept_check"]["avg_cycles"],
                    "full_accept": result["cycles"]["full_accept"]["avg_cycles"],
                    "full_reject": result["cycles"]["full_reject"]["avg_cycles"],
                },
            }
        )
    rows.sort(key=lambda row: (row["max_g_value"], row["case_name"], row["security_target"]))
    return rows


def _write_json(path: Path, data: object) -> None:
    path.write_text(json.dumps(data, ensure_ascii=True, indent=2) + "\n", encoding="utf-8")


def _read_json_list(path: Path) -> List[Dict[str, object]]:
    with path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)
    if not isinstance(data, list):
        raise ValueError(f"{path} must contain a JSON list")
    return data


def _build_parser() -> argparse.ArgumentParser:
    default_cycle_backend = "xctrace" if sys.platform == "darwin" else "estimated"
    parser = argparse.ArgumentParser(
        description="Refresh the cached Implementation Benchmarks data files.",
    )
    parser.add_argument("--samples", type=int, default=32, help="YCSig samples per repetition.")
    parser.add_argument("--repetitions", type=int, default=40, help="YCSig repetitions.")
    parser.add_argument("--cpu-frequency-ghz", type=float, default=3.49, help="Nominal CPU frequency in GHz.")
    parser.add_argument(
        "--cycle-backend",
        choices=("estimated", "xctrace"),
        default=default_cycle_backend,
        help="Cycle measurement backend for YCSig and ValStrictISP. Defaults to xctrace on macOS and estimated elsewhere.",
    )
    parser.add_argument(
        "--xctrace-target-operations",
        type=int,
        default=4096,
        help="Minimum operations per YCSig xctrace worker recording; raises amortization for faster metrics.",
    )
    parser.add_argument("--val-accepted-samples", type=int, default=32, help="Accepted ValStrictISP inputs per repetition.")
    parser.add_argument("--val-rejected-samples", type=int, default=8, help="Rejected ValStrictISP inputs per repetition.")
    parser.add_argument("--val-repetitions", type=int, default=10, help="ValStrictISP repetitions.")
    parser.add_argument(
        "--val-sampler-mode",
        choices=("seeded", "random"),
        default="random",
        help="ValStrictISP full-path sampler mode for implementation data; random matches the YCSig stream path.",
    )
    parser.add_argument(
        "--val-xctrace-target-operations",
        type=int,
        default=100_000,
        help="Minimum operations per ValStrictISP xctrace worker recording; use a large value because this path is very fast.",
    )
    parser.add_argument("--random-seed", type=int, default=0, help="Deterministic benchmark seed.")
    parser.add_argument(
        "--max-candidate-attempts",
        type=int,
        default=1_000_000,
        help="Maximum candidate inputs scanned while building ValStrictISP accepted/rejected pools.",
    )
    parser.add_argument("--output-128-160-json", type=Path, default=OUTPUT_128_160)
    parser.add_argument("--output-192-json", type=Path, default=OUTPUT_192)
    parser.add_argument("--output-valstrictisp-json", type=Path, default=OUTPUT_VALSTRICTISP)
    parser.add_argument(
        "--reuse-ycsig-json",
        action="store_true",
        help="Reuse the existing YCSig JSON caches and refresh only the ValStrictISP data.",
    )
    return parser


def _main() -> int:
    args = _build_parser().parse_args()

    if args.reuse_ycsig_json:
        rows_128_160 = _read_json_list(args.output_128_160_json)
        rows_192 = _read_json_list(args.output_192_json)
    else:
        rows_128_160, rows_192 = _refresh_ycsig(
            samples=args.samples,
            repetitions=args.repetitions,
            cpu_frequency_ghz=args.cpu_frequency_ghz,
            cycle_backend=args.cycle_backend,
            xctrace_target_operations=args.xctrace_target_operations,
        )
    valstrictisp_rows = _refresh_valstrictisp(
        accepted_samples=args.val_accepted_samples,
        rejected_samples=args.val_rejected_samples,
        repetitions=args.val_repetitions,
        cpu_frequency_ghz=args.cpu_frequency_ghz,
        cycle_backend=args.cycle_backend,
        xctrace_target_operations=args.val_xctrace_target_operations,
        sampler_mode=args.val_sampler_mode,
        random_seed=args.random_seed,
        max_candidate_attempts=args.max_candidate_attempts,
    )

    _write_json(args.output_128_160_json, rows_128_160)
    _write_json(args.output_192_json, rows_192)
    _write_json(args.output_valstrictisp_json, valstrictisp_rows)
    return 0


if __name__ == "__main__":
    raise SystemExit(_main())
