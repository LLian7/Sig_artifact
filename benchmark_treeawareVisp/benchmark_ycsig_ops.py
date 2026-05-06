from __future__ import annotations

import argparse
import json
import statistics
from dataclasses import replace
from typing import Any, Dict, Iterable, List, Optional, Sequence

from benchmark_ycsig import (
    YCSigBenchmarkCase,
    _derive_key_seed_for_sample,
    _message_for_sample,
    _randomizer_for_sample,
)
from operation_counter import counting_scope, snapshot
from pprf import PPRF
from yc_sig import YCSig


def _build_scheme(case: YCSigBenchmarkCase) -> YCSig:
    setup = YCSig.SigSetup(
        case.security_parameter,
        hash_len=case.hash_len,
        max_g_bit=case.max_g_bit,
        partition_size=case.partition_size,
        window_radius=case.window_radius,
        tweak_hash_name=case.tweak_hash_name,
        keyed_hash_name=case.keyed_hash_name,
        pprf_hash_name=case.pprf_hash_name,
        merkle_hash_name=case.merkle_hash_name,
        **case.setup_kwargs,
    )
    return YCSig(setup.params)


def _seed_variant(value: Any, repetition: int) -> Any:
    if repetition == 0:
        return value
    if isinstance(value, bytes):
        return value + b"/rep/" + repetition.to_bytes(4, "big")
    return value


def _case_for_repetition(case: YCSigBenchmarkCase, repetition: int) -> YCSigBenchmarkCase:
    if repetition == 0:
        return case
    return replace(
        case,
        random_seed=case.random_seed + repetition,
        setup_kwargs={
            key: _seed_variant(value, repetition)
            for key, value in case.setup_kwargs.items()
        },
    )


def _measure_counters(func) -> Dict[str, float]:
    with counting_scope():
        func()
        return snapshot()


def _aggregate_counter_dicts(dicts: Sequence[Dict[str, float]]) -> Dict[str, float]:
    if not dicts:
        return {}

    keys = sorted({key for entry in dicts for key in entry})
    aggregated: Dict[str, float] = {}
    for key in keys:
        aggregated[key] = statistics.fmean(entry.get(key, 0.0) for entry in dicts)
    return aggregated


def _top_level_ops(counter_dict: Dict[str, float]) -> float:
    return sum(
        counter_dict.get(name, 0.0)
        for name in (
            "keyed_hash.eval",
            "pprf.expand",
            "pprf.leaf_output",
            "tweak_hash.eval",
            "isp.sample_seed_hash",
            "isp.xof_instances",
        )
    )


def _top_level_backend_hash_calls(counter_dict: Dict[str, float]) -> float:
    return counter_dict.get("hash.backend_calls", 0.0)


def _keygen_sign_verify_gap(
    *,
    keygen: float,
    retry: float,
    sign: float,
    verify: float,
) -> float:
    return keygen - (sign - retry + verify)


def run_operation_benchmark_case(
    case: YCSigBenchmarkCase,
    *,
    repetitions: int,
) -> Dict[str, Any]:
    if repetitions <= 0:
        raise ValueError("repetitions must be positive")

    keygen_counters_per_rep: List[Dict[str, float]] = []
    retry_counters_per_rep: List[Dict[str, float]] = []
    sign_core_counters_per_rep: List[Dict[str, float]] = []
    sign_counters_per_rep: List[Dict[str, float]] = []
    verify_counters_per_rep: List[Dict[str, float]] = []
    sig_bits_avgs: List[float] = []
    sig_obj_avgs: List[float] = []
    verify_rates: List[float] = []

    for repetition in range(repetitions):
        rep_case = _case_for_repetition(case, repetition)
        scheme = _build_scheme(rep_case)

        keygen_sample_counters: List[Dict[str, float]] = []
        retry_sample_counters: List[Dict[str, float]] = []
        sign_core_sample_counters: List[Dict[str, float]] = []
        sign_sample_counters: List[Dict[str, float]] = []
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

            partition_holder: Dict[str, Any] = {}

            def do_retry() -> None:
                partition_holder["value"] = scheme.FindPartition(message, randomizer)

            retry_counter = _measure_counters(do_retry)
            salt, groups = partition_holder["value"]
            retry_sample_counters.append(retry_counter)

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
            sign_sample_counters.append(
                {
                    key: retry_counter.get(key, 0.0) + sign_core_counter.get(key, 0.0)
                    for key in set(retry_counter) | set(sign_core_counter)
                }
            )

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
        retry_counters_per_rep.append(_aggregate_counter_dicts(retry_sample_counters))
        sign_core_counters_per_rep.append(_aggregate_counter_dicts(sign_core_sample_counters))
        sign_counters_per_rep.append(_aggregate_counter_dicts(sign_sample_counters))
        verify_counters_per_rep.append(_aggregate_counter_dicts(verify_sample_counters))
        sig_bits_avgs.append(statistics.fmean(sig_bits))
        sig_obj_avgs.append(statistics.fmean(sig_obj))
        verify_rates.append(verify_successes / rep_case.samples)

    keygen_avg = _aggregate_counter_dicts(keygen_counters_per_rep)
    retry_avg = _aggregate_counter_dicts(retry_counters_per_rep)
    sign_core_avg = _aggregate_counter_dicts(sign_core_counters_per_rep)
    sign_avg = _aggregate_counter_dicts(sign_counters_per_rep)
    verify_avg = _aggregate_counter_dicts(verify_counters_per_rep)
    keygen_real = _top_level_ops(keygen_avg)
    retry_real = _top_level_ops(retry_avg)
    retry_attempt_count_real = retry_avg.get("ycsig.partition_attempt", 0.0)
    retry_attempt_hash_equivalents_real = retry_avg.get("keyed_hash.eval", 0.0)
    retry_sampler_hash_equivalents_real = (
        retry_avg.get("isp.sample_seed_hash", 0.0)
        + retry_avg.get("isp.xof_instances", 0.0)
    )
    sign_core_real = _top_level_ops(sign_core_avg)
    sign_real = _top_level_ops(sign_avg)
    verify_real = _top_level_ops(verify_avg)

    return {
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
            "retry_hash_equivalents_real": retry_real,
            "retry_attempt_count_real": retry_attempt_count_real,
            "retry_attempt_hash_equivalents_real": retry_attempt_hash_equivalents_real,
            "retry_sampler_hash_equivalents_real": retry_sampler_hash_equivalents_real,
            "retry_sampler_output_bytes_real": retry_avg.get("isp.xof_output_bytes", 0.0),
            "retry_sampler_output_bits_real": retry_avg.get("isp.xof_output_bits", 0.0),
            "sign_core_hash_equivalents_real": sign_core_real,
            "sign_hash_equivalents_real": sign_real,
            "verify_hash_equivalents_real": verify_real,
            "keygen_sign_relation_gap_real": _keygen_sign_verify_gap(
                keygen=keygen_real,
                retry=retry_real,
                sign=sign_real,
                verify=verify_real,
            ),
            "keygen_backend_hash_calls": _top_level_backend_hash_calls(keygen_avg),
            "retry_backend_hash_calls": _top_level_backend_hash_calls(retry_avg),
            "sign_core_backend_hash_calls": _top_level_backend_hash_calls(sign_core_avg),
            "sign_backend_hash_calls": _top_level_backend_hash_calls(sign_avg),
            "verify_backend_hash_calls": _top_level_backend_hash_calls(verify_avg),
        },
        "breakdown": {
            "keygen": keygen_avg,
            "retry": retry_avg,
            "sign_core": sign_core_avg,
            "sign": sign_avg,
            "verify": verify_avg,
        },
        "signature": {
            "avg_signature_bits": statistics.fmean(sig_bits_avgs),
            "avg_signature_hash_equivalents_concrete": statistics.fmean(sig_bits_avgs)
            / (case.hash_equivalent_bits or case.security_parameter),
            "avg_signature_hash_equivalents_object_model": statistics.fmean(sig_obj_avgs),
        },
        "verify_rate": statistics.fmean(verify_rates),
    }


def _format_text(results: Sequence[Dict[str, Any]]) -> str:
    header = (
        "Label | KeyGen(real) | Attempts(real) | Sampler(real) | Retry(real) | Sign(real) | Verify(real) | "
        "Gap | SigSize(real) | VerifyRate"
    )
    rule = "-" * len(header)
    lines = [header, rule]
    for result in results:
        ops = result["operations"]
        sig = result["signature"]
        lines.append(
            (
                f"{result['case']} | "
                f"{ops['keygen_hash_equivalents_real']:.1f} | "
                f"{ops['retry_attempt_count_real']:.2f} | "
                f"{ops['retry_sampler_hash_equivalents_real']:.1f} | "
                f"{ops['retry_hash_equivalents_real']:.1f} | "
                f"{ops['sign_hash_equivalents_real']:.1f} | "
                f"{ops['verify_hash_equivalents_real']:.1f} | "
                f"{ops['keygen_sign_relation_gap_real']:.1e} | "
                f"{sig['avg_signature_hash_equivalents_concrete']:.1f} | "
                f"{result['verify_rate']:.2f}"
            )
        )
    return "\n".join(lines)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Benchmark YCSig by counting real primitive/hash-equivalent operations.",
    )
    parser.add_argument("--name", default="YCSig", help="Benchmark label.")
    parser.add_argument("--security-parameter", type=int, required=True, help="kappa")
    parser.add_argument("--hash-len", type=int, required=True, help="HashLen")
    parser.add_argument("--max-g-bit", type=int, required=True, help="MaxGBit")
    parser.add_argument("--partition-size", type=int, required=True, help="PartitionSize")
    parser.add_argument("--window-radius", type=int, default=None, help="WindowRadius")
    parser.add_argument("--samples", type=int, default=32, help="Samples per repetition.")
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
    case = YCSigBenchmarkCase(
        name=args.name,
        security_parameter=args.security_parameter,
        hash_len=args.hash_len,
        max_g_bit=args.max_g_bit,
        partition_size=args.partition_size,
        window_radius=args.window_radius,
        samples=args.samples,
    )
    result = run_operation_benchmark_case(case, repetitions=args.repetitions)
    if args.format == "json":
        print(json.dumps(result, ensure_ascii=True, indent=2, sort_keys=True))
    else:
        print(_format_text([result]))
    return 0


if __name__ == "__main__":
    raise SystemExit(_main())
