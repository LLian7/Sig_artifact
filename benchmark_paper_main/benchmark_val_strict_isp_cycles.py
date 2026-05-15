from __future__ import annotations

import argparse
import json
import statistics
from dataclasses import dataclass
from functools import lru_cache
from typing import Any, Callable, Dict, List, Sequence

from benchmark_true_cycles import measure_worker_cycles_with_xctrace
from benchmark_ycsig_cycles import (
    DEFAULT_CYCLE_BACKEND,
    DEFAULT_CPU_FREQUENCY_GHZ,
    _cpu_time_ns,
    _ns_to_cycles,
    _resolve_cycle_backend,
    _resolve_cpu_frequency_hz,
)
from crypto_utils import DEFAULT_HASH_NAME, derive_parameter
from operation_counter import disabled_scope
from val_strict_isp import (
    ISPParameters,
    _NATIVE_ACCEPT_CHECK,
    _NATIVE_ACCEPT_CHECK_FAST,
    _NATIVE_BYTES_RANDOM,
    _NATIVE_BYTES_RANDOM_FAST,
    _NATIVE_PROFILE_COUNTS,
    _NATIVE_W4_ACCEPT_CHECK_FAST,
    _NATIVE_W4_BYTES_RANDOM,
    _NATIVE_W4_BYTES_RANDOM_FAST,
    _multiplicity_profile_from_partition_value,
    _val_strict_isp_with_random_bytes,
    recommended_stream_sampler_bytes,
    val_strict_isp,
)


PartitionValue = bytes | int
DEFAULT_XCTRACE_INNER_REPETITIONS = 512
DEFAULT_XCTRACE_TARGET_OPERATIONS = 256
SAMPLER_MODES = {"seeded", "random"}


@dataclass(frozen=True)
class _TimedMetric:
    name: str
    avg_cycles: float
    stddev_cycles: float
    min_cycles: float
    max_cycles: float


def _window_accepts(partition_value: PartitionValue, params: ISPParameters) -> bool:
    if (
        isinstance(partition_value, bytes)
        and _NATIVE_W4_ACCEPT_CHECK_FAST is not None
        and params.max_g_bit == 2
        and params.max_g_value == 4
    ):
        return bool(
            _NATIVE_W4_ACCEPT_CHECK_FAST(
                partition_value,
                params.hash_len,
                params.window_low,
                params.window_high,
            )
        )
    if isinstance(partition_value, bytes) and _NATIVE_ACCEPT_CHECK_FAST is not None:
        return bool(
            _NATIVE_ACCEPT_CHECK_FAST(
                partition_value,
                params.hash_len,
                params.max_g_bit,
                params.window_low,
                params.window_high,
            )
        )
    if isinstance(partition_value, bytes) and _NATIVE_ACCEPT_CHECK is not None:
        return bool(
            _NATIVE_ACCEPT_CHECK(
                partition_value,
                params.hash_len,
                params.max_g_bit,
                params.window_low,
                params.window_high,
            )
        )
    counts = _multiplicity_profile_from_partition_value(
        partition_value=partition_value,
        hash_len=params.hash_len,
        max_g_bit=params.max_g_bit,
        max_g_value=params.max_g_value,
    )
    if not params.window_valid:
        return False
    low = params.window_low
    high = params.window_high
    return all(low <= count <= high for count in counts)


def _accept_check(partition_value: PartitionValue, params: ISPParameters) -> bool:
    if (
        isinstance(partition_value, bytes)
        and _NATIVE_W4_ACCEPT_CHECK_FAST is not None
        and params.max_g_bit == 2
        and params.max_g_value == 4
    ):
        return bool(
            _NATIVE_W4_ACCEPT_CHECK_FAST(
                partition_value,
                params.hash_len,
                params.window_low,
                params.window_high,
            )
        )
    if isinstance(partition_value, bytes) and _NATIVE_ACCEPT_CHECK_FAST is not None:
        return bool(
            _NATIVE_ACCEPT_CHECK_FAST(
                partition_value,
                params.hash_len,
                params.max_g_bit,
                params.window_low,
                params.window_high,
            )
        )
    if isinstance(partition_value, bytes) and _NATIVE_ACCEPT_CHECK is not None:
        return bool(
            _NATIVE_ACCEPT_CHECK(
                partition_value,
                params.hash_len,
                params.max_g_bit,
                params.window_low,
                params.window_high,
            )
        )
    counts = _multiplicity_profile_from_partition_value(
        partition_value=partition_value,
        hash_len=params.hash_len,
        max_g_bit=params.max_g_bit,
        max_g_value=params.max_g_value,
    )
    if not params._window_valid:
        return False
    low = params._window_low
    high = params._window_high
    for count in counts:
        if count < low or count > high:
            return False
    return True


def _profile_only(partition_value: PartitionValue, params: ISPParameters) -> Sequence[int]:
    if isinstance(partition_value, bytes) and _NATIVE_PROFILE_COUNTS is not None:
        return _NATIVE_PROFILE_COUNTS(
            partition_value,
            params.hash_len,
            params.max_g_bit,
        )
    return _multiplicity_profile_from_partition_value(
        partition_value=partition_value,
        hash_len=params.hash_len,
        max_g_bit=params.max_g_bit,
        max_g_value=params.max_g_value,
    )


@lru_cache(maxsize=None)
def _random_sampler_bytes(params: ISPParameters) -> bytes:
    return derive_parameter(
        b"ValStrictISP/CycleBenchmark/SamplerBytes/",
        seed=(
            params.hash_len.to_bytes(4, "big")
            + params.max_g_bit.to_bytes(2, "big")
            + params.partition_num.to_bytes(2, "big")
            + params.window_radius.to_bytes(2, "big")
        ),
        output_bits=8 * recommended_stream_sampler_bytes(params),
        hash_name=params.hash_name,
    )


def _full_accept(partition_value: PartitionValue, params: ISPParameters) -> None:
    groups = val_strict_isp(partition_value, params, return_group_masks=True)
    if groups is None:
        raise AssertionError("expected an accepting partition value")


def _full_reject(partition_value: PartitionValue, params: ISPParameters) -> None:
    groups = val_strict_isp(partition_value, params, return_group_masks=True)
    if groups is not None:
        raise AssertionError("expected a rejecting partition value")


def _full_accept_random(partition_value: PartitionValue, params: ISPParameters) -> None:
    if not isinstance(partition_value, bytes):
        raise TypeError("random sampler benchmark requires bytes partition values")
    random_bytes = _random_sampler_bytes(params)
    if (
        params.window_valid
        and params.max_g_bit == 2
        and params.max_g_value == 4
        and params.partition_num <= 64
    ):
        native_w4_random = _NATIVE_W4_BYTES_RANDOM_FAST or _NATIVE_W4_BYTES_RANDOM
        if native_w4_random is not None:
            groups = native_w4_random(
                partition_value,
                params.hash_len,
                params.partition_num,
                params.window_low,
                params.window_high,
                random_bytes,
                True,
            )
            if groups is None:
                raise AssertionError("expected an accepting partition value")
            return

    native_random = _NATIVE_BYTES_RANDOM_FAST or _NATIVE_BYTES_RANDOM
    if (
        native_random is not None
        and params.window_valid
        and 2 <= params.max_g_bit <= 6
        and params.partition_num <= 64
    ):
        groups = native_random(
            partition_value,
            params.hash_len,
            params.max_g_bit,
            params.partition_num,
            params.window_low,
            params.window_high,
            random_bytes,
            True,
        )
        if groups is None:
            raise AssertionError("expected an accepting partition value")
        return

    groups = _val_strict_isp_with_random_bytes(
        partition_value,
        params,
        random_bytes,
        return_group_masks=True,
    )
    if groups is None:
        raise AssertionError("expected an accepting partition value")


def _full_reject_random(partition_value: PartitionValue, params: ISPParameters) -> None:
    if not isinstance(partition_value, bytes):
        raise TypeError("random sampler benchmark requires bytes partition values")
    if _NATIVE_ACCEPT_CHECK_FAST is not None:
        if not _NATIVE_ACCEPT_CHECK_FAST(
            partition_value,
            params.hash_len,
            params.max_g_bit,
            params.window_low,
            params.window_high,
        ):
            return
    groups = _val_strict_isp_with_random_bytes(
        partition_value,
        params,
        _random_sampler_bytes(params),
        return_group_masks=True,
    )
    if groups is not None:
        raise AssertionError("expected a rejecting partition value")


def _build_metric_function(
    metric: str,
    params: ISPParameters,
    sampler_mode: str,
) -> Callable[[PartitionValue], Any]:
    hash_len = params.hash_len
    max_g_bit = params.max_g_bit
    max_g_value = params.max_g_value
    partition_num = params.partition_num
    window_low = params.window_low
    window_high = params.window_high

    if metric == "profile_only" and _NATIVE_PROFILE_COUNTS is not None:
        native_profile = _NATIVE_PROFILE_COUNTS

        def profile_only(partition_value: PartitionValue) -> Any:
            if not isinstance(partition_value, bytes):
                return _profile_only(partition_value, params)
            return native_profile(partition_value, hash_len, max_g_bit)

        return profile_only

    if metric == "accept_check" and _NATIVE_ACCEPT_CHECK_FAST is not None:
        if (
            _NATIVE_W4_ACCEPT_CHECK_FAST is not None
            and max_g_bit == 2
            and max_g_value == 4
        ):
            native_w4_accept = _NATIVE_W4_ACCEPT_CHECK_FAST

            def accept_check_w4(partition_value: PartitionValue) -> bool:
                if not isinstance(partition_value, bytes):
                    return _accept_check(partition_value, params)
                return bool(native_w4_accept(partition_value, hash_len, window_low, window_high))

            return accept_check_w4

        native_accept = _NATIVE_ACCEPT_CHECK_FAST

        def accept_check(partition_value: PartitionValue) -> bool:
            if not isinstance(partition_value, bytes):
                return _accept_check(partition_value, params)
            return bool(native_accept(partition_value, hash_len, max_g_bit, window_low, window_high))

        return accept_check

    if metric == "full_accept" and sampler_mode == "random":
        random_bytes = _random_sampler_bytes(params)
        if (
            params.window_valid
            and max_g_bit == 2
            and max_g_value == 4
            and partition_num <= 64
        ):
            native_w4_random = _NATIVE_W4_BYTES_RANDOM_FAST or _NATIVE_W4_BYTES_RANDOM
            if native_w4_random is not None:

                def full_accept_w4_random(partition_value: PartitionValue) -> None:
                    if not isinstance(partition_value, bytes):
                        raise TypeError("random sampler benchmark requires bytes partition values")
                    groups = native_w4_random(
                        partition_value,
                        hash_len,
                        partition_num,
                        window_low,
                        window_high,
                        random_bytes,
                        True,
                    )
                    if groups is None:
                        raise AssertionError("expected an accepting partition value")

                return full_accept_w4_random

        native_random = _NATIVE_BYTES_RANDOM_FAST or _NATIVE_BYTES_RANDOM
        if (
            native_random is not None
            and params.window_valid
            and 2 <= max_g_bit <= 6
            and partition_num <= 64
        ):

            def full_accept_random(partition_value: PartitionValue) -> None:
                if not isinstance(partition_value, bytes):
                    raise TypeError("random sampler benchmark requires bytes partition values")
                groups = native_random(
                    partition_value,
                    hash_len,
                    max_g_bit,
                    partition_num,
                    window_low,
                    window_high,
                    random_bytes,
                    True,
                )
                if groups is None:
                    raise AssertionError("expected an accepting partition value")

            return full_accept_random

    if metric == "full_reject" and sampler_mode == "random" and _NATIVE_ACCEPT_CHECK_FAST is not None:
        if (
            _NATIVE_W4_ACCEPT_CHECK_FAST is not None
            and max_g_bit == 2
            and max_g_value == 4
        ):
            native_w4_accept = _NATIVE_W4_ACCEPT_CHECK_FAST

            def full_reject_w4_random(partition_value: PartitionValue) -> None:
                if not isinstance(partition_value, bytes):
                    raise TypeError("random sampler benchmark requires bytes partition values")
                if native_w4_accept(partition_value, hash_len, window_low, window_high):
                    raise AssertionError("expected a rejecting partition value")

            return full_reject_w4_random

        native_accept = _NATIVE_ACCEPT_CHECK_FAST

        def full_reject_random(partition_value: PartitionValue) -> None:
            if not isinstance(partition_value, bytes):
                raise TypeError("random sampler benchmark requires bytes partition values")
            if native_accept(partition_value, hash_len, max_g_bit, window_low, window_high):
                raise AssertionError("expected a rejecting partition value")

        return full_reject_random

    if metric == "full_accept":
        selected = _full_accept_random if sampler_mode == "random" else _full_accept
    elif metric == "full_reject":
        selected = _full_reject_random if sampler_mode == "random" else _full_reject
    elif metric == "profile_only":
        selected = _profile_only
    elif metric == "accept_check":
        selected = _accept_check
    else:
        raise ValueError(f"unsupported ValStrictISP metric: {metric}")

    def fallback(partition_value: PartitionValue) -> Any:
        return selected(partition_value, params)

    return fallback


def _candidate_partition_value(index: int, params: ISPParameters, random_seed: int) -> PartitionValue:
    return derive_parameter(
        b"ValStrictISP/CycleBenchmark/Candidate/" + index.to_bytes(8, "big"),
        seed=random_seed.to_bytes(8, "big"),
        output_bits=params.hash_len,
        hash_name=params.hash_name,
    )


def _collect_partition_values(
    *,
    params: ISPParameters,
    accepted_samples: int,
    rejected_samples: int,
    random_seed: int,
    max_candidate_attempts: int,
) -> Dict[str, Any]:
    accepted: List[PartitionValue] = []
    rejected: List[PartitionValue] = []
    attempts = 0

    while len(accepted) < accepted_samples or len(rejected) < rejected_samples:
        if attempts >= max_candidate_attempts:
            raise RuntimeError(
                "unable to collect enough accepted/rejected partition values; "
                "increase --max-candidate-attempts or reduce sample counts"
            )
        partition_value = _candidate_partition_value(attempts, params, random_seed)
        if _window_accepts(partition_value, params):
            if len(accepted) < accepted_samples:
                accepted.append(partition_value)
        else:
            if len(rejected) < rejected_samples:
                rejected.append(partition_value)
        attempts += 1

    return {
        "accepted": accepted,
        "rejected": rejected,
        "attempts": attempts,
        "accepted_found": len(accepted),
        "rejected_found": len(rejected),
    }


def _measure_metric_xctrace(
    *,
    metric: str,
    hash_len: int,
    max_g_bit: int,
    partition_num: int,
    window_radius: int,
    hash_name: str,
    accepted_samples: int,
    rejected_samples: int,
    random_seed: int,
    max_candidate_attempts: int,
    inner_repetitions: int,
    sampler_mode: str,
) -> float:
    sample_count = accepted_samples if metric != "full_reject" else rejected_samples
    total_cycles = measure_worker_cycles_with_xctrace(
        [
            "valstrictisp",
            "--metric",
            metric,
            "--hash-len",
            str(hash_len),
            "--max-g-bit",
            str(max_g_bit),
            "--partition-num",
            str(partition_num),
            "--window-radius",
            str(window_radius),
            "--hash-name",
            hash_name,
            "--accepted-samples",
            str(accepted_samples),
            "--rejected-samples",
            str(rejected_samples),
            "--random-seed",
            str(random_seed),
            "--max-candidate-attempts",
            str(max_candidate_attempts),
            "--inner-repetitions",
            str(inner_repetitions),
            "--sampler-mode",
            sampler_mode,
        ]
    )
    return total_cycles / (sample_count * inner_repetitions)


def _xctrace_inner_repetitions(sample_count: int, target_total_operations: int) -> int:
    if sample_count <= 0:
        raise ValueError("sample_count must be positive")
    return max(1, -(-target_total_operations // sample_count))


def _measure_metrics(
    metric_specs: Sequence[tuple[str, Sequence[PartitionValue], Callable[[PartitionValue], Any]]],
    repetitions: int,
    cpu_frequency_hz: float,
) -> List[_TimedMetric]:
    if repetitions <= 0:
        raise ValueError("repetitions must be positive")
    if not metric_specs:
        raise ValueError("metric_specs must be non-empty")

    per_metric_averages: Dict[str, List[float]] = {}
    for name, values, _ in metric_specs:
        if not values:
            raise ValueError(f"metric {name} received no samples")
        per_metric_averages[name] = []

    for repetition in range(repetitions):
        rotation = repetition % len(metric_specs)
        ordered_specs = list(metric_specs[rotation:]) + list(metric_specs[:rotation])

        # Warm the Python path for every metric before measuring this repetition.
        for _, values, func in ordered_specs:
            func(values[0])

        for name, values, func in ordered_specs:
            start_ns = _cpu_time_ns()
            for partition_value in values:
                func(partition_value)
            end_ns = _cpu_time_ns()
            per_metric_averages[name].append(
                _ns_to_cycles(end_ns - start_ns, cpu_frequency_hz) / len(values)
            )

    return [
        _TimedMetric(
            name=name,
            avg_cycles=statistics.fmean(per_rep_averages),
            stddev_cycles=statistics.pstdev(per_rep_averages) if repetitions > 1 else 0.0,
            min_cycles=min(per_rep_averages),
            max_cycles=max(per_rep_averages),
        )
        for name, per_rep_averages in per_metric_averages.items()
    ]


def run_val_strict_isp_cycle_benchmark(
    *,
    hash_len: int,
    max_g_bit: int,
    partition_num: int,
    window_radius: int,
    hash_name: str,
    accepted_samples: int,
    rejected_samples: int,
    repetitions: int,
    random_seed: int,
    max_candidate_attempts: int,
    cpu_frequency_hz: float | None = None,
    cpu_frequency_ghz: float | None = None,
    cycle_backend: str = DEFAULT_CYCLE_BACKEND,
    xctrace_target_operations: int = DEFAULT_XCTRACE_TARGET_OPERATIONS,
    sampler_mode: str = "seeded",
) -> Dict[str, Any]:
    if xctrace_target_operations <= 0:
        raise ValueError("xctrace_target_operations must be positive")
    if sampler_mode not in SAMPLER_MODES:
        raise ValueError(f"sampler_mode must be one of {sorted(SAMPLER_MODES)}")
    params = ISPParameters(
        hash_len=hash_len,
        max_g_bit=max_g_bit,
        partition_num=partition_num,
        window_radius=window_radius,
        hash_name=hash_name,
    )
    resolved_cycle_backend = _resolve_cycle_backend(cycle_backend)
    resolved_frequency_hz: float | None
    if resolved_cycle_backend == "estimated":
        resolved_frequency_hz = _resolve_cpu_frequency_hz(
            cpu_frequency_hz=cpu_frequency_hz,
            cpu_frequency_ghz=cpu_frequency_ghz,
        )
    else:
        resolved_frequency_hz = None

    sample_sets = _collect_partition_values(
        params=params,
        accepted_samples=accepted_samples,
        rejected_samples=rejected_samples,
        random_seed=random_seed,
        max_candidate_attempts=max_candidate_attempts,
    )

    if resolved_cycle_backend == "estimated":
        assert resolved_frequency_hz is not None
        with disabled_scope():
            metrics = _measure_metrics(
                [
                    (
                    "profile_only",
                    sample_sets["accepted"],
                    _build_metric_function("profile_only", params, sampler_mode),
                    ),
                    (
                    "accept_check",
                    sample_sets["accepted"],
                    _build_metric_function("accept_check", params, sampler_mode),
                    ),
                    (
                    "full_accept",
                    sample_sets["accepted"],
                    _build_metric_function("full_accept", params, sampler_mode),
                    ),
                    (
                    "full_reject",
                    sample_sets["rejected"],
                    _build_metric_function("full_reject", params, sampler_mode),
                    ),
                ],
                repetitions,
                resolved_frequency_hz,
            )
    else:
        per_metric_averages: Dict[str, List[float]] = {
            "profile_only": [],
            "accept_check": [],
            "full_accept": [],
            "full_reject": [],
        }
        for _ in range(repetitions):
            for metric_name in per_metric_averages:
                sample_count = accepted_samples if metric_name != "full_reject" else rejected_samples
                inner_repetitions = _xctrace_inner_repetitions(
                    sample_count,
                    xctrace_target_operations,
                )
                per_metric_averages[metric_name].append(
                    _measure_metric_xctrace(
                        metric=metric_name,
                        hash_len=hash_len,
                        max_g_bit=max_g_bit,
                        partition_num=partition_num,
                        window_radius=window_radius,
                        hash_name=hash_name,
                        accepted_samples=accepted_samples,
                        rejected_samples=rejected_samples,
                        random_seed=random_seed,
                        max_candidate_attempts=max_candidate_attempts,
                        inner_repetitions=inner_repetitions,
                        sampler_mode=sampler_mode,
                    )
                )
        metrics = [
            _TimedMetric(
                name=name,
                avg_cycles=statistics.fmean(per_rep_averages),
                stddev_cycles=statistics.pstdev(per_rep_averages) if repetitions > 1 else 0.0,
                min_cycles=min(per_rep_averages),
                max_cycles=max(per_rep_averages),
            )
            for name, per_rep_averages in per_metric_averages.items()
        ]

    metric_map = {
        metric.name: {
            "avg_cycles": metric.avg_cycles,
            "stddev_cycles": metric.stddev_cycles,
            "min_cycles": metric.min_cycles,
            "max_cycles": metric.max_cycles,
        }
        for metric in metrics
    }
    full_accept_cycles = metric_map["full_accept"]["avg_cycles"]
    full_reject_cycles = metric_map["full_reject"]["avg_cycles"]
    return {
        "parameters": {
            "hash_len": params.hash_len,
            "max_g_bit": params.max_g_bit,
            "max_g_value": params.max_g_value,
            "partition_num": params.partition_num,
            "window_radius": params.window_radius,
            "hash_name": params.hash_name,
            "sampler_mode": sampler_mode,
            "accepted_samples": accepted_samples,
            "rejected_samples": rejected_samples,
            "repetitions": repetitions,
            "random_seed": random_seed,
            "cycle_backend": resolved_cycle_backend,
            "cpu_frequency_hz": resolved_frequency_hz,
            "cpu_frequency_ghz": (
                None if resolved_frequency_hz is None else resolved_frequency_hz / 1e9
            ),
            "window_low": params.window_low,
            "window_high": params.window_high,
        },
        "sample_pool": {
            "candidate_attempts": sample_sets["attempts"],
            "accepted_samples": sample_sets["accepted_found"],
            "rejected_samples": sample_sets["rejected_found"],
        },
        "cycles": metric_map,
        "ratios": {
            "full_accept_over_profile": (
                full_accept_cycles / metric_map["profile_only"]["avg_cycles"]
                if metric_map["profile_only"]["avg_cycles"] > 0
                else 0.0
            ),
            "full_accept_over_accept_check": (
                full_accept_cycles / metric_map["accept_check"]["avg_cycles"]
                if metric_map["accept_check"]["avg_cycles"] > 0
                else 0.0
            ),
            "full_reject_over_accept_check": (
                full_reject_cycles / metric_map["accept_check"]["avg_cycles"]
                if metric_map["accept_check"]["avg_cycles"] > 0
                else 0.0
            ),
        },
    }


def _format_text(result: Dict[str, Any]) -> str:
    params = result["parameters"]
    sample_pool = result["sample_pool"]
    cycles = result["cycles"]
    ratios = result["ratios"]
    return "\n".join(
        [
            (
                "ValStrictISP cycles: "
                f"HashLen={params['hash_len']}, "
                f"MaxGBit={params['max_g_bit']}, "
                f"w={params['max_g_value']}, "
                f"PartitionNum={params['partition_num']}, "
                f"WindowRadius={params['window_radius']}, "
                f"window=[{params['window_low']}, {params['window_high']}], "
                f"accepted_samples={params['accepted_samples']}, "
                f"rejected_samples={params['rejected_samples']}, "
                f"repetitions={params['repetitions']}, "
                + (
                    f"cpu_frequency_ghz={params['cpu_frequency_ghz']:.3f}"
                    if params["cpu_frequency_ghz"] is not None
                    else f"cycle_backend={params['cycle_backend']}"
                )
            ),
            (
                "  SamplePool: "
                f"candidate_attempts={sample_pool['candidate_attempts']}, "
                f"accepted={sample_pool['accepted_samples']}, "
                f"rejected={sample_pool['rejected_samples']}"
            ),
            (
                "  Cycles: "
                f"profile_only={cycles['profile_only']['avg_cycles']:.1f} "
                f"+/- {cycles['profile_only']['stddev_cycles']:.1f}, "
                f"accept_check={cycles['accept_check']['avg_cycles']:.1f} "
                f"+/- {cycles['accept_check']['stddev_cycles']:.1f}, "
                f"full_accept={cycles['full_accept']['avg_cycles']:.1f} "
                f"+/- {cycles['full_accept']['stddev_cycles']:.1f}, "
                f"full_reject={cycles['full_reject']['avg_cycles']:.1f} "
                f"+/- {cycles['full_reject']['stddev_cycles']:.1f}"
            ),
            (
                "  Ratios: "
                f"full_accept/profile={ratios['full_accept_over_profile']:.2f}, "
                f"full_accept/accept_check={ratios['full_accept_over_accept_check']:.2f}, "
                f"full_reject/accept_check={ratios['full_reject_over_accept_check']:.2f}"
            ),
        ]
    )


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Benchmark ValStrictISP in CPU cycles using either nominal-frequency estimates or xctrace hardware counters.",
    )
    parser.add_argument("--hash-len", type=int, default=256, help="HashLen in bits.")
    parser.add_argument("--max-g-bit", type=int, default=16, help="MaxGBit.")
    parser.add_argument("--partition-num", type=int, default=2, help="PartitionNum.")
    parser.add_argument("--window-radius", type=int, default=0, help="WindowRadius.")
    parser.add_argument(
        "--sampler-mode",
        choices=tuple(sorted(SAMPLER_MODES)),
        default="seeded",
        help="Use standalone seeded sampler SHAKE or caller-supplied random bytes for full paths.",
    )
    parser.add_argument(
        "--hash-name",
        choices=("shake_128", "shake_256", "sha3_256", "sha3_512"),
        default=DEFAULT_HASH_NAME,
        help="Hash/XOF backend used by ValStrictISP.",
    )
    parser.add_argument("--accepted-samples", type=int, default=256, help="Accepted inputs per repetition.")
    parser.add_argument("--rejected-samples", type=int, default=64, help="Rejected inputs per repetition.")
    parser.add_argument("--repetitions", type=int, default=40, help="Independent repetitions.")
    parser.add_argument("--random-seed", type=int, default=0, help="Deterministic benchmark seed.")
    parser.add_argument(
        "--max-candidate-attempts",
        type=int,
        default=1_000_000,
        help="Maximum candidate inputs scanned while building accepted/rejected pools.",
    )
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
        help="Minimum operations per xctrace worker recording; raise this to amortize setup for fast metrics.",
    )
    parser.add_argument("--format", choices=("json", "text"), default="json", help="Output format.")
    return parser


def _main() -> int:
    parser = _build_parser()
    args = parser.parse_args()
    result = run_val_strict_isp_cycle_benchmark(
        hash_len=args.hash_len,
        max_g_bit=args.max_g_bit,
        partition_num=args.partition_num,
        window_radius=args.window_radius,
        hash_name=args.hash_name,
        accepted_samples=args.accepted_samples,
        rejected_samples=args.rejected_samples,
        repetitions=args.repetitions,
        random_seed=args.random_seed,
        max_candidate_attempts=args.max_candidate_attempts,
        cpu_frequency_hz=args.cpu_frequency_hz,
        cpu_frequency_ghz=args.cpu_frequency_ghz,
        cycle_backend=args.cycle_backend,
        xctrace_target_operations=args.xctrace_target_operations,
        sampler_mode=args.sampler_mode,
    )
    if args.format == "text":
        print(_format_text(result))
    else:
        print(json.dumps(result, ensure_ascii=True, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(_main())
