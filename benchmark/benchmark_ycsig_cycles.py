from __future__ import annotations

import argparse
import gc
import json
import math
import os
import statistics
import subprocess
import sys
import time
import tracemalloc
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

from benchmark_ycsig import (
    YCSigBenchmarkCase,
    _derive_key_seed_for_sample,
    _message_for_sample,
    _randomizer_for_sample,
)
from benchmark_true_cycles import measure_worker_cycles_with_xctrace
from crypto_utils import DEFAULT_HASH_NAME
from operation_counter import disabled_scope
from pprf import PPRF
from yc_sig import YCSig


DEFAULT_SINGLE_CASE_NAME = "YCSig-w4-k128-H>=k"
DEFAULT_SINGLE_CASE_SECURITY_PARAMETER = 128
DEFAULT_SINGLE_CASE_HASH_LEN = 130
DEFAULT_SINGLE_CASE_MAX_G_BIT = 2
DEFAULT_SINGLE_CASE_PARTITION_SIZE = 33
DEFAULT_SINGLE_CASE_WINDOW_RADIUS = 1
DEFAULT_CPU_FREQUENCY_GHZ = 3.49
DEFAULT_CYCLE_BACKEND = "estimated"
DEFAULT_XCTRACE_INNER_REPETITIONS = 16
DEFAULT_XCTRACE_TARGET_OPERATIONS = 128
_BYTES_JSON_TAG = "__benchmark_bytes_hex__"
_TUPLE_JSON_TAG = "__benchmark_tuple__"


def _cpu_time_ns() -> int:
    if hasattr(time, "thread_time_ns"):
        return time.thread_time_ns()
    return time.process_time_ns()


def _detect_cpu_frequency_hz() -> Optional[float]:
    env_hz = os.environ.get("CPU_FREQUENCY_HZ")
    if env_hz:
        return float(env_hz)

    env_ghz = os.environ.get("CPU_FREQUENCY_GHZ")
    if env_ghz:
        return float(env_ghz) * 1e9

    if sys.platform == "darwin":
        for name in ("hw.cpufrequency", "hw.cpufrequency_max"):
            try:
                result = subprocess.run(
                    ["sysctl", "-n", name],
                    check=True,
                    capture_output=True,
                    text=True,
                )
                value = result.stdout.strip()
                if value:
                    return float(value)
            except Exception:
                continue
        return None

    cpuinfo_path = "/proc/cpuinfo"
    if os.path.exists(cpuinfo_path):
        mhz_values: List[float] = []
        with open(cpuinfo_path, "r", encoding="utf-8") as handle:
            for line in handle:
                if ":" not in line:
                    continue
                key, value = [item.strip() for item in line.split(":", 1)]
                if key.lower() == "cpu mhz":
                    try:
                        mhz_values.append(float(value))
                    except ValueError:
                        pass
        if mhz_values:
            return statistics.fmean(mhz_values) * 1e6
    return None


def _resolve_cpu_frequency_hz(
    *,
    cpu_frequency_hz: Optional[float] = None,
    cpu_frequency_ghz: Optional[float] = None,
) -> float:
    if cpu_frequency_hz is not None and cpu_frequency_ghz is not None:
        raise ValueError("specify at most one of cpu_frequency_hz and cpu_frequency_ghz")
    if cpu_frequency_hz is not None:
        if cpu_frequency_hz <= 0:
            raise ValueError("cpu_frequency_hz must be positive")
        return cpu_frequency_hz
    if cpu_frequency_ghz is not None:
        if cpu_frequency_ghz <= 0:
            raise ValueError("cpu_frequency_ghz must be positive")
        return cpu_frequency_ghz * 1e9

    detected = _detect_cpu_frequency_hz()
    if detected is None:
        raise ValueError(
            "unable to auto-detect CPU frequency; pass --cpu-frequency-ghz or --cpu-frequency-hz"
        )
    return detected


def _build_scheme(case: YCSigBenchmarkCase):
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
        partition_retry_mode=case.partition_retry_mode,
        partition_sampler_mode=case.partition_sampler_mode,
        partition_stream_sampler_bytes=case.partition_stream_sampler_bytes,
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


def _ns_to_cycles(duration_ns: int, cpu_frequency_hz: float) -> float:
    return duration_ns * cpu_frequency_hz / 1e9


def _resolve_cycle_backend(cycle_backend: str) -> str:
    if cycle_backend not in {"estimated", "xctrace"}:
        raise ValueError("cycle_backend must be one of {'estimated', 'xctrace'}")
    return cycle_backend


def _encode_worker_json_value(value: Any) -> Any:
    if isinstance(value, bytes):
        return {_BYTES_JSON_TAG: value.hex()}
    if isinstance(value, tuple):
        return {_TUPLE_JSON_TAG: [_encode_worker_json_value(item) for item in value]}
    if isinstance(value, list):
        return [_encode_worker_json_value(item) for item in value]
    if isinstance(value, dict):
        return {
            str(key): _encode_worker_json_value(item)
            for key, item in value.items()
        }
    return value


def _decode_worker_json_value(value: Any) -> Any:
    if isinstance(value, list):
        return [_decode_worker_json_value(item) for item in value]
    if isinstance(value, dict):
        if set(value) == {_BYTES_JSON_TAG}:
            return bytes.fromhex(str(value[_BYTES_JSON_TAG]))
        if set(value) == {_TUPLE_JSON_TAG}:
            raw_items = value[_TUPLE_JSON_TAG]
            if not isinstance(raw_items, list):
                raise TypeError("encoded tuple payload must be a list")
            return tuple(_decode_worker_json_value(item) for item in raw_items)
        return {
            key: _decode_worker_json_value(item)
            for key, item in value.items()
        }
    return value


def serialize_case_for_worker_json(case: YCSigBenchmarkCase) -> str:
    return json.dumps(
        _encode_worker_json_value(case.__dict__),
        ensure_ascii=True,
        sort_keys=True,
    )


def deserialize_case_from_worker_json(case_json: str) -> YCSigBenchmarkCase:
    raw = json.loads(case_json)
    if not isinstance(raw, dict):
        raise TypeError("worker case payload must decode to a JSON object")
    return YCSigBenchmarkCase(**_decode_worker_json_value(raw))


def _measure_peak_python_heap_bytes(func) -> int:
    """
    Measure the peak Python-heap footprint of a callable using tracemalloc.

    This intentionally tracks Python-managed allocations only. It is used as an
    implementation-level memory proxy without perturbing the cycle timings,
    which are measured in a separate pass without tracemalloc enabled.
    """

    gc.collect()
    tracemalloc.start()
    try:
        tracemalloc.reset_peak()
        func()
        _, peak = tracemalloc.get_traced_memory()
        return peak
    finally:
        tracemalloc.stop()


def _warm_up_core_algorithms(case: YCSigBenchmarkCase, scheme: YCSig) -> None:
    """
    Prime Python/runtime state before recording cycle measurements.

    This keeps setup/building-block initialization out of the measured keygen
    path and reduces first-call bias in the cycle benchmark.
    """

    warmup_index = case.samples + 1
    message = _message_for_sample(case, warmup_index)
    key_seed = _derive_key_seed_for_sample(case, warmup_index)
    randomizer = _randomizer_for_sample(case, warmup_index)
    ks = PPRF.PRFKGen(scheme.params.pm_PPRF, seed=key_seed)
    keypair = scheme.SigGen(ks)
    salt, groups = scheme.FindPartition(message, randomizer)
    signature = scheme.SignWithGroups(keypair.secret_key, randomizer, salt, groups)
    if not scheme.VerifyWithGroups(keypair.public_key, signature, groups):
        raise AssertionError("warm-up signature failed verification")


def _collect_noncycle_metrics_for_repetition(
    rep_case: YCSigBenchmarkCase,
    *,
    measure_memory: bool,
) -> Dict[str, Any]:
    scheme = _build_scheme(rep_case)
    _warm_up_core_algorithms(rep_case, scheme)

    verify_successes = 0
    sig_bits: List[float] = []
    sig_obj: List[float] = []
    keygen_memories: List[float] = []
    sign_memories: List[float] = []
    verify_memories: List[float] = []
    peak_memories: List[float] = []
    hash_equivalent_bits = rep_case.hash_equivalent_bits or scheme.params.security_parameter

    for sample_index in range(rep_case.samples):
        message = _message_for_sample(rep_case, sample_index)
        key_seed = _derive_key_seed_for_sample(rep_case, sample_index)
        randomizer = _randomizer_for_sample(rep_case, sample_index)
        ks = PPRF.PRFKGen(scheme.params.pm_PPRF, seed=key_seed)
        keypair = scheme.SigGen(ks)
        salt, groups = scheme.FindPartition(message, randomizer)
        signature = scheme.SignWithGroups(
            keypair.secret_key,
            randomizer,
            salt,
            groups,
        )
        valid = scheme.VerifyWithGroups(keypair.public_key, signature, groups)
        verify_successes += int(valid)

        sig_bits.append(8 * signature.serialized_size())
        sig_obj.append(
            len(signature.punctured_seeds)
            + len(signature.partial_state_values)
            + rep_case.signature_extra_hash_values
            + rep_case.signature_extra_bits / hash_equivalent_bits
        )

        if measure_memory:
            keygen_peak = _measure_peak_python_heap_bytes(lambda: scheme.SigGen(ks))

            def do_sign_total() -> None:
                local_salt, local_groups = scheme.FindPartition(message, randomizer)
                scheme.SignWithGroups(
                    keypair.secret_key,
                    randomizer,
                    local_salt,
                    local_groups,
                )

            sign_peak = _measure_peak_python_heap_bytes(do_sign_total)
            verify_peak = _measure_peak_python_heap_bytes(
                lambda: scheme.VerifyWithGroups(keypair.public_key, signature, groups)
            )
            keygen_memories.append(float(keygen_peak))
            sign_memories.append(float(sign_peak))
            verify_memories.append(float(verify_peak))
            peak_memories.append(float(max(keygen_peak, sign_peak, verify_peak)))

    result: Dict[str, Any] = {
        "verify_rate": verify_successes / rep_case.samples,
        "avg_signature_bits": statistics.fmean(sig_bits),
        "avg_signature_hash_equivalents_object_model": statistics.fmean(sig_obj),
    }
    if measure_memory:
        result["memory"] = {
            "avg_peak_keygen_memory_bytes": statistics.fmean(keygen_memories),
            "avg_peak_sign_memory_bytes": statistics.fmean(sign_memories),
            "avg_peak_verify_memory_bytes": statistics.fmean(verify_memories),
            "avg_peak_memory_bytes": statistics.fmean(peak_memories),
        }
    return result


def _measure_ycsig_metric_cycles_xctrace(
    *,
    case: YCSigBenchmarkCase,
    metric: str,
    inner_repetitions: int,
) -> float:
    total_cycles = measure_worker_cycles_with_xctrace(
        [
            "ycsig",
            "--case-json",
            serialize_case_for_worker_json(case),
            "--metric",
            metric,
            "--repetition",
            "0",
            "--inner-repetitions",
            str(inner_repetitions),
        ],
        cwd=str(Path(__file__).resolve().parent),
    )
    return total_cycles / (case.samples * inner_repetitions)


def _xctrace_inner_repetitions(sample_count: int, target_total_operations: int) -> int:
    if sample_count <= 0:
        raise ValueError("sample_count must be positive")
    return max(1, math.ceil(target_total_operations / sample_count))


def run_cycle_benchmark_case(
    case: YCSigBenchmarkCase,
    *,
    repetitions: int,
    cpu_frequency_hz: Optional[float] = None,
    cpu_frequency_ghz: Optional[float] = None,
    cycle_backend: str = DEFAULT_CYCLE_BACKEND,
    xctrace_target_operations: int = DEFAULT_XCTRACE_TARGET_OPERATIONS,
    measure_memory: bool = False,
) -> Dict[str, Any]:
    if repetitions <= 0:
        raise ValueError("repetitions must be positive")
    if xctrace_target_operations <= 0:
        raise ValueError("xctrace_target_operations must be positive")
    resolved_cycle_backend = _resolve_cycle_backend(cycle_backend)
    resolved_frequency_hz: Optional[float]
    if resolved_cycle_backend == "estimated":
        resolved_frequency_hz = _resolve_cpu_frequency_hz(
            cpu_frequency_hz=cpu_frequency_hz,
            cpu_frequency_ghz=cpu_frequency_ghz,
        )
    else:
        resolved_frequency_hz = None

    keygen_cycle_avgs: List[float] = []
    retry_cycle_avgs: List[float] = []
    sign_core_cycle_avgs: List[float] = []
    sign_cycle_avgs: List[float] = []
    verify_cycle_avgs: List[float] = []
    verify_rates: List[float] = []
    sig_bits_avgs: List[float] = []
    sig_obj_avgs: List[float] = []
    keygen_memory_avgs: List[float] = []
    sign_memory_avgs: List[float] = []
    verify_memory_avgs: List[float] = []
    peak_memory_avgs: List[float] = []

    with disabled_scope():
        if resolved_cycle_backend == "estimated":
            for repetition in range(repetitions):
                rep_case = _case_for_repetition(case, repetition)
                assert resolved_frequency_hz is not None
                scheme = _build_scheme(rep_case)
                _warm_up_core_algorithms(rep_case, scheme)

                keygen_cycles: List[float] = []
                retry_cycles: List[float] = []
                sign_core_cycles: List[float] = []
                sign_cycles: List[float] = []
                verify_cycles: List[float] = []
                sig_bits: List[float] = []
                sig_obj: List[float] = []
                verify_successes = 0
                keygen_memories: List[float] = []
                sign_memories: List[float] = []
                verify_memories: List[float] = []
                peak_memories: List[float] = []

                hash_equivalent_bits = (
                    rep_case.hash_equivalent_bits or scheme.params.security_parameter
                )
                for sample_index in range(rep_case.samples):
                    message = _message_for_sample(rep_case, sample_index)
                    key_seed = _derive_key_seed_for_sample(rep_case, sample_index)
                    randomizer = _randomizer_for_sample(rep_case, sample_index)
                    ks = PPRF.PRFKGen(scheme.params.pm_PPRF, seed=key_seed)

                    start_ns = _cpu_time_ns()
                    keypair = scheme.SigGen(ks)
                    end_ns = _cpu_time_ns()
                    keygen_cycles.append(_ns_to_cycles(end_ns - start_ns, resolved_frequency_hz))

                    start_ns = _cpu_time_ns()
                    salt, groups = scheme.FindPartition(message, randomizer)
                    end_ns = _cpu_time_ns()
                    retry_cycle = _ns_to_cycles(end_ns - start_ns, resolved_frequency_hz)
                    retry_cycles.append(retry_cycle)

                    start_ns = _cpu_time_ns()
                    signature = scheme.SignWithGroups(
                        keypair.secret_key,
                        randomizer,
                        salt,
                        groups,
                    )
                    end_ns = _cpu_time_ns()
                    sign_core_cycle = _ns_to_cycles(end_ns - start_ns, resolved_frequency_hz)
                    sign_core_cycles.append(sign_core_cycle)
                    sign_cycles.append(retry_cycle + sign_core_cycle)

                    start_ns = _cpu_time_ns()
                    valid = scheme.VerifyWithGroups(keypair.public_key, signature, groups)
                    end_ns = _cpu_time_ns()
                    verify_cycles.append(_ns_to_cycles(end_ns - start_ns, resolved_frequency_hz))
                    verify_successes += int(valid)

                    if measure_memory:
                        keygen_peak = _measure_peak_python_heap_bytes(lambda: scheme.SigGen(ks))

                        def do_sign_total() -> None:
                            local_salt, local_groups = scheme.FindPartition(message, randomizer)
                            scheme.SignWithGroups(
                                keypair.secret_key,
                                randomizer,
                                local_salt,
                                local_groups,
                            )

                        sign_peak = _measure_peak_python_heap_bytes(do_sign_total)
                        verify_peak = _measure_peak_python_heap_bytes(
                            lambda: scheme.VerifyWithGroups(keypair.public_key, signature, groups)
                        )
                        keygen_memories.append(float(keygen_peak))
                        sign_memories.append(float(sign_peak))
                        verify_memories.append(float(verify_peak))
                        peak_memories.append(float(max(keygen_peak, sign_peak, verify_peak)))

                    sig_bits.append(8 * signature.serialized_size())
                    sig_obj.append(
                        len(signature.punctured_seeds)
                        + len(signature.partial_state_values)
                        + rep_case.signature_extra_hash_values
                        + rep_case.signature_extra_bits / hash_equivalent_bits
                    )

                keygen_cycle_avgs.append(statistics.fmean(keygen_cycles))
                retry_cycle_avgs.append(statistics.fmean(retry_cycles))
                sign_core_cycle_avgs.append(statistics.fmean(sign_core_cycles))
                sign_cycle_avgs.append(statistics.fmean(sign_cycles))
                verify_cycle_avgs.append(statistics.fmean(verify_cycles))
                verify_rates.append(verify_successes / rep_case.samples)
                sig_bits_avgs.append(statistics.fmean(sig_bits))
                sig_obj_avgs.append(statistics.fmean(sig_obj))
                if measure_memory:
                    keygen_memory_avgs.append(statistics.fmean(keygen_memories))
                    sign_memory_avgs.append(statistics.fmean(sign_memories))
                    verify_memory_avgs.append(statistics.fmean(verify_memories))
                    peak_memory_avgs.append(statistics.fmean(peak_memories))
        else:
            for repetition in range(repetitions):
                rep_case = _case_for_repetition(case, repetition)
                noncycle = _collect_noncycle_metrics_for_repetition(
                    rep_case,
                    measure_memory=measure_memory,
                )
                verify_rates.append(noncycle["verify_rate"])
                sig_bits_avgs.append(noncycle["avg_signature_bits"])
                sig_obj_avgs.append(noncycle["avg_signature_hash_equivalents_object_model"])
                if measure_memory:
                    memory = noncycle["memory"]
                    keygen_memory_avgs.append(memory["avg_peak_keygen_memory_bytes"])
                    sign_memory_avgs.append(memory["avg_peak_sign_memory_bytes"])
                    verify_memory_avgs.append(memory["avg_peak_verify_memory_bytes"])
                    peak_memory_avgs.append(memory["avg_peak_memory_bytes"])

            inner_repetitions = _xctrace_inner_repetitions(
                case.samples,
                xctrace_target_operations,
            )
            keygen_cycle = _measure_ycsig_metric_cycles_xctrace(
                case=case,
                metric="keygen",
                inner_repetitions=inner_repetitions,
            )
            retry_cycle = _measure_ycsig_metric_cycles_xctrace(
                case=case,
                metric="retry",
                inner_repetitions=inner_repetitions,
            )
            sign_core_cycle = _measure_ycsig_metric_cycles_xctrace(
                case=case,
                metric="sign_core",
                inner_repetitions=inner_repetitions,
            )
            verify_cycle = _measure_ycsig_metric_cycles_xctrace(
                case=case,
                metric="verify",
                inner_repetitions=inner_repetitions,
            )

            keygen_cycle_avgs.append(keygen_cycle)
            retry_cycle_avgs.append(retry_cycle)
            sign_core_cycle_avgs.append(sign_core_cycle)
            sign_cycle_avgs.append(retry_cycle + sign_core_cycle)
            verify_cycle_avgs.append(verify_cycle)
    result = {
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
            "cpu_frequency_hz": resolved_frequency_hz,
            "cpu_frequency_ghz": (
                None if resolved_frequency_hz is None else resolved_frequency_hz / 1e9
            ),
            "cycle_backend": resolved_cycle_backend,
            "setup_excluded": True,
            "prf_keygen_excluded": True,
            "warmup_runs_per_repetition": 1,
        },
        "cycles": {
            "avg_keygen_cycles": statistics.fmean(keygen_cycle_avgs),
            "avg_retry_cycles": statistics.fmean(retry_cycle_avgs),
            "avg_sign_core_cycles": statistics.fmean(sign_core_cycle_avgs),
            "avg_sign_cycles": statistics.fmean(sign_cycle_avgs),
            "avg_verify_cycles": statistics.fmean(verify_cycle_avgs),
        },
        "cycles_stddev": {
            "avg_keygen_cycles": statistics.pstdev(keygen_cycle_avgs) if repetitions > 1 else 0.0,
            "avg_retry_cycles": statistics.pstdev(retry_cycle_avgs) if repetitions > 1 else 0.0,
            "avg_sign_core_cycles": statistics.pstdev(sign_core_cycle_avgs) if repetitions > 1 else 0.0,
            "avg_sign_cycles": statistics.pstdev(sign_cycle_avgs) if repetitions > 1 else 0.0,
            "avg_verify_cycles": statistics.pstdev(verify_cycle_avgs) if repetitions > 1 else 0.0,
        },
        "signature": {
            "avg_signature_bits": statistics.fmean(sig_bits_avgs),
            "avg_signature_hash_equivalents_concrete": statistics.fmean(sig_bits_avgs)
            / (case.hash_equivalent_bits or case.security_parameter),
            "avg_signature_hash_equivalents_object_model": statistics.fmean(sig_obj_avgs),
            "signature_hash_equivalents_override": case.analytic_signature_hash_equivalents_override,
        },
        "verify_rate": statistics.fmean(verify_rates),
    }
    if measure_memory:
        avg_keygen_memory_bytes = statistics.fmean(keygen_memory_avgs)
        avg_sign_memory_bytes = statistics.fmean(sign_memory_avgs)
        avg_verify_memory_bytes = statistics.fmean(verify_memory_avgs)
        avg_peak_memory_bytes = statistics.fmean(peak_memory_avgs)
        mib = float(1 << 20)
        result["memory"] = {
            "avg_peak_keygen_memory_bytes": avg_keygen_memory_bytes,
            "avg_peak_sign_memory_bytes": avg_sign_memory_bytes,
            "avg_peak_verify_memory_bytes": avg_verify_memory_bytes,
            "avg_peak_memory_bytes": avg_peak_memory_bytes,
            "avg_peak_keygen_memory_mib": avg_keygen_memory_bytes / mib,
            "avg_peak_sign_memory_mib": avg_sign_memory_bytes / mib,
            "avg_peak_verify_memory_mib": avg_verify_memory_bytes / mib,
            "avg_peak_memory_mib": avg_peak_memory_bytes / mib,
        }
    return result


def _format_text(results: Sequence[Dict[str, Any]]) -> str:
    lines: List[str] = []
    for result in results:
        params = result["parameters"]
        cycles = result["cycles"]
        signature = result["signature"]
        lines.append(
            (
                f"{result['case']}: "
                f"kappa={params['security_parameter']}, "
                f"HashLen={params['hash_len']}, "
                f"w={params['max_g_value']}, "
                f"PartitionSize={params['partition_size']}, "
                f"repetitions={params['repetitions']}, "
                f"total_experiments={params['total_experiments']}, "
                + (
                    f"cpu_frequency_ghz={params['cpu_frequency_ghz']:.3f}"
                    if params["cpu_frequency_ghz"] is not None
                    else f"cycle_backend={params['cycle_backend']}"
                )
            )
        )
        lines.append(
            (
                "  Cycles: "
                f"KeyGen={cycles['avg_keygen_cycles']:.1f}, "
                f"Retry={cycles['avg_retry_cycles']:.1f}, "
                f"Sign={cycles['avg_sign_cycles']:.1f}, "
                f"Verify={cycles['avg_verify_cycles']:.1f}"
            )
        )
        if signature["signature_hash_equivalents_override"] is not None:
            lines.append(
                "  SigSize: "
                f"theory={signature['signature_hash_equivalents_override']:.1f}, "
                f"real={signature['avg_signature_hash_equivalents_concrete']:.1f}"
            )
        else:
            lines.append(
                "  SigSize: "
                f"real={signature['avg_signature_hash_equivalents_concrete']:.1f}"
            )
        memory = result.get("memory")
        if memory is not None:
            lines.append(
                "  PeakMem: "
                f"overall={memory['avg_peak_memory_mib']:.2f} MiB, "
                f"kg={memory['avg_peak_keygen_memory_mib']:.2f} MiB, "
                f"sign={memory['avg_peak_sign_memory_mib']:.2f} MiB, "
                f"verify={memory['avg_peak_verify_memory_mib']:.2f} MiB"
            )
    return "\n".join(lines)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Benchmark YCSig in CPU cycles using either nominal-frequency estimates or xctrace hardware counters.",
    )
    parser.add_argument("--config-file", help="Optional JSON file containing a list of benchmark cases.")
    parser.add_argument("--name", default=DEFAULT_SINGLE_CASE_NAME, help="Case label.")
    parser.add_argument(
        "--security-parameter",
        type=int,
        default=DEFAULT_SINGLE_CASE_SECURITY_PARAMETER,
        help="Security parameter kappa.",
    )
    parser.add_argument(
        "--hash-len",
        type=int,
        default=DEFAULT_SINGLE_CASE_HASH_LEN,
        help="HashLen in bits.",
    )
    parser.add_argument("--max-g-bit", type=int, default=DEFAULT_SINGLE_CASE_MAX_G_BIT, help="MaxGBit.")
    parser.add_argument(
        "--partition-size",
        type=int,
        default=DEFAULT_SINGLE_CASE_PARTITION_SIZE,
        help="PartitionSize.",
    )
    parser.add_argument(
        "--window-radius",
        type=int,
        default=DEFAULT_SINGLE_CASE_WINDOW_RADIUS,
        help="WindowRadius.",
    )
    parser.add_argument("--samples", type=int, default=32, help="Samples per repetition.")
    parser.add_argument("--repetitions", type=int, default=10, help="Independent repetitions.")
    parser.add_argument("--random-seed", type=int, default=0, help="Deterministic benchmark seed.")
    parser.add_argument("--message-bytes", type=int, default=None, help="Benchmark message length in bytes.")
    parser.add_argument("--cpu-frequency-hz", type=float, default=None, help="Nominal CPU frequency in Hz.")
    parser.add_argument(
        "--cpu-frequency-ghz",
        type=float,
        default=DEFAULT_CPU_FREQUENCY_GHZ,
        help="Nominal CPU frequency in GHz.",
    )
    parser.add_argument(
        "--cycle-backend",
        choices=("estimated", "xctrace"),
        default=DEFAULT_CYCLE_BACKEND,
        help="Cycle measurement backend. 'estimated' uses CPU time times a nominal frequency; 'xctrace' uses macOS CPU Counters.",
    )
    parser.add_argument(
        "--xctrace-target-operations",
        type=int,
        default=DEFAULT_XCTRACE_TARGET_OPERATIONS,
        help="Minimum operations per xctrace worker recording; raise this to amortize process setup for fast metrics.",
    )
    parser.add_argument(
        "--tweak-hash-name",
        choices=("shake_128", "shake_256", "sha3_256", "sha3_512"),
        default=DEFAULT_HASH_NAME,
        help="Hash backend for the tweakable hash.",
    )
    parser.add_argument(
        "--keyed-hash-name",
        choices=("shake_128", "shake_256", "sha3_256", "sha3_512"),
        default=DEFAULT_HASH_NAME,
        help="Hash backend for KeyedH.",
    )
    parser.add_argument(
        "--pprf-hash-name",
        choices=("shake_128", "shake_256", "sha3_256", "sha3_512"),
        default=DEFAULT_HASH_NAME,
        help="Hash backend for the PPRF.",
    )
    parser.add_argument(
        "--merkle-hash-name",
        choices=("shake_128", "shake_256", "sha3_256", "sha3_512"),
        default=DEFAULT_HASH_NAME,
        help="Hash backend for the Merkle tree.",
    )
    parser.add_argument(
        "--partition-retry-mode",
        choices=("salted", "stream"),
        default="salted",
        help="Use independent salted partition hashes or a Y0-seeded partition XOF stream.",
    )
    parser.add_argument(
        "--partition-sampler-mode",
        choices=("seeded", "stream"),
        default="seeded",
        help="Use ValStrictISP's seeded sampler XOF or draw sampler bytes from the partition stream.",
    )
    parser.add_argument(
        "--partition-stream-sampler-bytes",
        type=int,
        default=None,
        help="Sampler random bytes reserved per retry chunk in partition stream mode (default: auto).",
    )
    parser.add_argument(
        "--format",
        choices=("json", "text"),
        default="json",
        help="Output format.",
    )
    parser.add_argument(
        "--measure-memory",
        action="store_true",
        help="Also measure peak Python-heap usage for KeyGen/Sign/Verify.",
    )
    return parser


def _single_case_from_args(args: argparse.Namespace) -> YCSigBenchmarkCase:
    return YCSigBenchmarkCase(
        name=args.name,
        security_parameter=args.security_parameter,
        hash_len=args.hash_len,
        max_g_bit=args.max_g_bit,
        partition_size=args.partition_size,
        window_radius=args.window_radius,
        samples=args.samples,
        random_seed=args.random_seed,
        message_bytes=args.message_bytes,
        tweak_hash_name=args.tweak_hash_name,
        keyed_hash_name=args.keyed_hash_name,
        pprf_hash_name=args.pprf_hash_name,
        merkle_hash_name=args.merkle_hash_name,
        partition_retry_mode=args.partition_retry_mode,
        partition_sampler_mode=args.partition_sampler_mode,
        partition_stream_sampler_bytes=args.partition_stream_sampler_bytes,
    )


def _load_cases(args: argparse.Namespace) -> List[YCSigBenchmarkCase]:
    if args.config_file:
        with open(args.config_file, "r", encoding="utf-8") as handle:
            raw_cases = json.load(handle)
        if not isinstance(raw_cases, list):
            raise ValueError("config-file must contain a JSON list of case dictionaries")
        return [YCSigBenchmarkCase(**raw_case) for raw_case in raw_cases]
    return [_single_case_from_args(args)]


def _main() -> int:
    parser = _build_parser()
    args = parser.parse_args()
    cases = _load_cases(args)
    results = [
        run_cycle_benchmark_case(
            case,
            repetitions=args.repetitions,
            cpu_frequency_hz=args.cpu_frequency_hz,
            cpu_frequency_ghz=args.cpu_frequency_ghz,
            cycle_backend=args.cycle_backend,
            xctrace_target_operations=args.xctrace_target_operations,
            measure_memory=args.measure_memory,
        )
        for case in cases
    ]
    if args.format == "text":
        print(_format_text(results))
    else:
        print(json.dumps(results, ensure_ascii=True, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(_main())
