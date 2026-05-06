from __future__ import annotations

import argparse
import json
import math
import statistics
import time
from dataclasses import dataclass, field, replace
from decimal import Decimal, localcontext
from random import Random
from typing import Any, Dict, List, Optional, Sequence

from crypto_utils import DEFAULT_HASH_NAME, bits_to_bytes, derive_parameter
from treeaware_isp import treeaware_isp, window_bounds
from yc_sig import YCSig


@dataclass(frozen=True)
class YCSigBenchmarkCase:
    name: str = "YCSig"
    security_parameter: int = 128
    hash_len: Optional[int] = None
    max_g_bit: int = 4
    partition_size: Optional[int] = None
    window_radius: Optional[int] = None
    samples: int = 128
    acceptance_mode: str = "exact"
    acceptance_samples: int = 10000
    retry_cost_mode: str = "total"
    include_verifier_retry_cost: bool = False
    signature_extra_hash_values: float = 1.0
    signature_extra_bits: int = 0
    hash_equivalent_bits: Optional[int] = None
    message_bytes: Optional[int] = None
    random_seed: int = 0
    measure_time: bool = False
    analytic_acceptance_probability_override: Optional[float] = None
    analytic_retry_overhead_override: Optional[float] = None
    analytic_online_prg_cost_override: Optional[float] = None
    analytic_signature_hash_equivalents_override: Optional[float] = None
    tweak_hash_name: str = DEFAULT_HASH_NAME
    keyed_hash_name: str = DEFAULT_HASH_NAME
    pprf_hash_name: str = DEFAULT_HASH_NAME
    merkle_hash_name: str = DEFAULT_HASH_NAME
    setup_kwargs: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.security_parameter <= 0:
            raise ValueError("security_parameter must be positive")
        if self.max_g_bit <= 0:
            raise ValueError("max_g_bit must be positive")
        if self.samples <= 0:
            raise ValueError("samples must be positive")
        if self.acceptance_samples <= 0:
            raise ValueError("acceptance_samples must be positive")
        if self.retry_cost_mode not in {"total", "failures"}:
            raise ValueError("retry_cost_mode must be 'total' or 'failures'")
        if self.acceptance_mode not in {"exact", "monte_carlo"}:
            raise ValueError("acceptance_mode must be 'exact' or 'monte_carlo'")
        if self.signature_extra_hash_values < 0:
            raise ValueError("signature_extra_hash_values must be non-negative")
        if self.signature_extra_bits < 0:
            raise ValueError("signature_extra_bits must be non-negative")


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
        **case.setup_kwargs,
    )
    scheme = YCSig(setup.params)
    keypair = scheme.SigGen(setup.randomness_seed)
    return scheme, keypair


def _message_for_sample(case: YCSigBenchmarkCase, sample_index: int) -> bytes:
    message_bytes = case.message_bytes
    if message_bytes is None:
        message_bytes = max(32, bits_to_bytes(case.hash_len or case.security_parameter))
    seed = case.random_seed.to_bytes(8, "big")
    return derive_parameter(
        b"YCSig/Benchmark/Message/" + sample_index.to_bytes(8, "big"),
        seed=seed,
        output_bits=8 * message_bytes,
        hash_name=case.keyed_hash_name,
    )


def _derive_key_seed_for_sample(case: YCSigBenchmarkCase, sample_index: int) -> bytes:
    return derive_parameter(
        b"YCSig/Benchmark/KeySeed/" + sample_index.to_bytes(8, "big"),
        seed=case.random_seed.to_bytes(8, "big"),
        output_bits=case.security_parameter,
        hash_name=case.pprf_hash_name,
    )


def _randomizer_for_sample(case: YCSigBenchmarkCase, sample_index: int) -> bytes:
    return derive_parameter(
        b"YCSig/Benchmark/Randomizer/" + sample_index.to_bytes(8, "big"),
        seed=case.random_seed.to_bytes(8, "big"),
        output_bits=case.security_parameter,
        hash_name=case.keyed_hash_name,
    )


def _exact_acceptance_probability(
    block_num: int,
    max_g_value: int,
    low: int,
    high: int,
) -> float:
    if low < 0 or high < low:
        return 0.0

    with localcontext() as ctx:
        ctx.prec = max(80, 4 * block_num + 20)

        inv_fact = [Decimal(0)] * (high + 1)
        factorial_value = 1
        inv_fact[0] = Decimal(1)
        for value in range(1, high + 1):
            factorial_value *= value
            inv_fact[value] = Decimal(1) / Decimal(factorial_value)

        dp = [Decimal(0)] * (block_num + 1)
        dp[0] = Decimal(1)
        for _ in range(max_g_value):
            nxt = [Decimal(0)] * (block_num + 1)
            for used, coeff in enumerate(dp):
                if coeff.is_zero():
                    continue
                upper = min(high, block_num - used)
                for count in range(low, upper + 1):
                    nxt[used + count] += coeff * inv_fact[count]
            dp = nxt

        block_factorial = 1
        for value in range(2, block_num + 1):
            block_factorial *= value

        probability = Decimal(block_factorial) * dp[block_num] / (Decimal(max_g_value) ** block_num)
        if probability < 0:
            return 0.0
        if probability > 1:
            return 1.0
        return float(probability)


def _monte_carlo_acceptance_probability(case: YCSigBenchmarkCase, scheme: YCSig) -> float:
    rng = Random(case.random_seed ^ 0xA5A5A5A5)
    accepted = 0
    hash_len = scheme.params.hash_len
    for _ in range(case.acceptance_samples):
        partition_value = format(rng.getrandbits(hash_len), f"0{hash_len}b")
        if treeaware_isp(partition_value, scheme.params.pm_ISP) is not None:
            accepted += 1
    return accepted / case.acceptance_samples


def _retry_overhead(acceptance_probability: float, retry_cost_mode: str) -> float:
    if acceptance_probability <= 0:
        return math.inf
    if retry_cost_mode == "failures":
        return (1.0 - acceptance_probability) / acceptance_probability
    return 1.0 / acceptance_probability


def _expected_online_prg_cost(leaf_count: int, block_num: int) -> float:
    depth = math.ceil(math.log2(leaf_count))
    total = 0.0
    for level in range(1, depth + 1):
        interval_size = 1 << level
        interval_count = leaf_count // interval_size
        if interval_count == 0:
            continue
        occupied_probability = 1.0 - (1.0 - interval_size / leaf_count) ** block_num
        total += interval_count * occupied_probability
    return total


def _sample_empirical_metrics(case: YCSigBenchmarkCase, scheme: YCSig, keypair: Any) -> Dict[str, Any]:
    attempt_counts: List[int] = []
    punctured_seed_counts: List[int] = []
    partial_state_counts: List[int] = []
    serialized_bits: List[int] = []
    sign_ms: List[float] = []
    verify_ms: List[float] = []
    verify_successes = 0

    for sample_index in range(case.samples):
        message = _message_for_sample(case, sample_index)

        sign_start = time.perf_counter()
        signature = scheme.SigSign(keypair.secret_key, message)
        sign_end = time.perf_counter()

        attempt_counts.append(int.from_bytes(signature.salt, "big") + 1)
        punctured_seed_counts.append(len(signature.punctured_seeds))
        partial_state_counts.append(len(signature.partial_state_values))
        serialized_bits.append(8 * signature.serialized_size())

        verify_start = time.perf_counter()
        valid = scheme.SigVrfy(keypair.public_key, message, signature)
        verify_end = time.perf_counter()
        verify_successes += int(valid)

        if case.measure_time:
            sign_ms.append((sign_end - sign_start) * 1000.0)
            verify_ms.append((verify_end - verify_start) * 1000.0)

    hash_equivalent_bits = case.hash_equivalent_bits or scheme.params.security_parameter
    avg_signature_bits = statistics.fmean(serialized_bits)
    avg_punctured = statistics.fmean(punctured_seed_counts)
    avg_partial = statistics.fmean(partial_state_counts)

    metrics: Dict[str, Any] = {
        "samples": case.samples,
        "verify_success_rate": verify_successes / case.samples,
        "avg_attempts": statistics.fmean(attempt_counts),
        "avg_retry_failures": statistics.fmean(value - 1 for value in attempt_counts),
        "avg_punctured_seed_count": avg_punctured,
        "avg_partial_state_count": avg_partial,
        "avg_signature_bits": avg_signature_bits,
        "avg_signature_bytes": avg_signature_bits / 8.0,
        "avg_signature_hash_equivalents_concrete": avg_signature_bits / hash_equivalent_bits,
        "avg_signature_hash_equivalents_object_model": (
            avg_punctured
            + avg_partial
            + case.signature_extra_hash_values
            + case.signature_extra_bits / hash_equivalent_bits
        ),
    }
    if case.measure_time:
        metrics["avg_sign_ms"] = statistics.fmean(sign_ms)
        metrics["avg_verify_ms"] = statistics.fmean(verify_ms)
    return metrics


def run_benchmark_case(case: YCSigBenchmarkCase) -> Dict[str, Any]:
    scheme, keypair = _build_scheme(case)
    params = scheme.params
    low, high = window_bounds(params.pm_ISP)

    if case.analytic_acceptance_probability_override is not None:
        acceptance_probability = case.analytic_acceptance_probability_override
    else:
        if case.acceptance_mode == "exact":
            acceptance_probability = _exact_acceptance_probability(
                params.block_num,
                params.max_g_value,
                low,
                high,
            )
        else:
            acceptance_probability = _monte_carlo_acceptance_probability(case, scheme)

    retry_overhead = (
        case.analytic_retry_overhead_override
        if case.analytic_retry_overhead_override is not None
        else _retry_overhead(acceptance_probability, case.retry_cost_mode)
    )
    expected_prg = (
        case.analytic_online_prg_cost_override
        if case.analytic_online_prg_cost_override is not None
        else _expected_online_prg_cost(params.leaf_count, params.block_num)
    )

    keygen_cost = 3 * params.leaf_count - 2
    sign_cost = (3 * params.block_num - 1) / 2.0 + expected_prg + retry_overhead
    verify_cost = 3 * params.leaf_count - (3 * params.block_num + 3) / 2.0 - expected_prg
    if case.include_verifier_retry_cost:
        verify_cost += retry_overhead

    empirical = _sample_empirical_metrics(case, scheme, keypair)

    return {
        "case": case.name,
        "parameters": {
            "security_parameter": params.security_parameter,
            "hash_len": params.hash_len,
            "max_g_bit": params.max_g_bit,
            "max_g_value": params.max_g_value,
            "block_num": params.block_num,
            "partition_size": params.partition_size,
            "window_radius": params.window_radius,
            "window_low": low,
            "window_high": high,
            "leaf_count": params.leaf_count,
            "pprf_depth": params.leaf_index_bits,
            "hash_equivalent_bits": case.hash_equivalent_bits or params.security_parameter,
            "compact_signature": False,
            "include_verifier_retry_cost": case.include_verifier_retry_cost,
            "retry_cost_mode": case.retry_cost_mode,
        },
        "analytic": {
            "acceptance_mode": case.acceptance_mode,
            "acceptance_probability": acceptance_probability,
            "expected_attempts": math.inf if not math.isfinite(retry_overhead) else 1.0 / acceptance_probability if acceptance_probability > 0 else math.inf,
            "expected_retry_overhead": retry_overhead,
            "expected_online_prg_cost": expected_prg,
            "keygen_hash_equivalents": float(keygen_cost),
            "sign_hash_equivalents": float(sign_cost),
            "verify_hash_equivalents": float(verify_cost),
            "signature_hash_equivalents_override": case.analytic_signature_hash_equivalents_override,
        },
        "empirical": empirical,
    }


def _seed_variant(value: Any, repetition: int) -> Any:
    if repetition == 0:
        return value
    if isinstance(value, bytes):
        return value + b"/rep/" + repetition.to_bytes(4, "big")
    return value


def _case_for_repetition(case: YCSigBenchmarkCase, repetition: int) -> YCSigBenchmarkCase:
    if repetition == 0:
        return case

    setup_kwargs = {
        key: _seed_variant(value, repetition)
        for key, value in case.setup_kwargs.items()
    }
    return replace(
        case,
        random_seed=case.random_seed + repetition,
        setup_kwargs=setup_kwargs,
    )


def _aggregate_numeric_dicts(dicts: Sequence[Dict[str, Any]]) -> Dict[str, float]:
    if not dicts:
        return {}

    keys = dicts[0].keys()
    aggregated: Dict[str, float] = {}
    for key in keys:
        values = [entry[key] for entry in dicts]
        first = values[0]
        if isinstance(first, bool):
            aggregated[key] = statistics.fmean(float(value) for value in values)
        elif isinstance(first, (int, float)):
            aggregated[key] = statistics.fmean(values)
    return aggregated


def _aggregate_stddev_dicts(dicts: Sequence[Dict[str, Any]]) -> Dict[str, float]:
    if len(dicts) <= 1:
        return {}

    keys = dicts[0].keys()
    aggregated: Dict[str, float] = {}
    for key in keys:
        values = [entry[key] for entry in dicts]
        first = values[0]
        if isinstance(first, bool):
            aggregated[key] = statistics.pstdev(float(value) for value in values)
        elif isinstance(first, (int, float)):
            aggregated[key] = statistics.pstdev(values)
    return aggregated


def run_benchmark_case_average(
    case: YCSigBenchmarkCase,
    repetitions: int,
) -> Dict[str, Any]:
    if repetitions <= 0:
        raise ValueError("repetitions must be positive")

    repetition_results = [
        run_benchmark_case(_case_for_repetition(case, repetition))
        for repetition in range(repetitions)
    ]
    first = repetition_results[0]

    aggregated_parameters = dict(first["parameters"])
    aggregated_parameters["repetitions"] = repetitions

    return {
        "case": first["case"],
        "parameters": aggregated_parameters,
        "analytic": _aggregate_numeric_dicts(
            [result["analytic"] for result in repetition_results]
        ),
        "analytic_stddev": _aggregate_stddev_dicts(
            [result["analytic"] for result in repetition_results]
        ),
        "empirical": _aggregate_numeric_dicts(
            [result["empirical"] for result in repetition_results]
        ),
        "empirical_stddev": _aggregate_stddev_dicts(
            [result["empirical"] for result in repetition_results]
        ),
    }


def _case_from_dict(raw: Dict[str, Any]) -> YCSigBenchmarkCase:
    return YCSigBenchmarkCase(**raw)


def _format_text(results: Sequence[Dict[str, Any]]) -> str:
    lines: List[str] = []
    for result in results:
        parameters = result["parameters"]
        analytic = result["analytic"]
        analytic_stddev = result.get("analytic_stddev", {})
        empirical = result["empirical"]
        empirical_stddev = result.get("empirical_stddev", {})
        lines.append(
            (
                f"{result['case']}: "
                f"kappa={parameters['security_parameter']}, "
                f"HashLen={parameters['hash_len']}, "
                f"w={parameters['max_g_value']}, "
                f"PartitionSize={parameters['partition_size']}, "
                f"WindowRadius={parameters['window_radius']}, "
                f"Window=[{parameters['window_low']},{parameters['window_high']}], "
                f"repetitions={parameters.get('repetitions', 1)}"
            )
        )
        lines.append(
            (
                "  Analytic: "
                + f"KeyGen={analytic['keygen_hash_equivalents']:.3f}, "
                + f"Sign={analytic['sign_hash_equivalents']:.3f}, "
                + f"Verify={analytic['verify_hash_equivalents']:.3f}, "
                + (
                    f"SigSize={analytic['signature_hash_equivalents_override']:.3f}, "
                    if analytic.get("signature_hash_equivalents_override") is not None
                    else ""
                )
                + f"p_accept={analytic['acceptance_probability']:.6f}, "
                + f"E[Re]={analytic['expected_retry_overhead']:.3f}"
            )
        )
        lines.append(
            (
                "  Empirical: "
                + f"avg_attempts={empirical['avg_attempts']:.3f}"
                + (
                    f" +- {empirical_stddev['avg_attempts']:.3f}, "
                    if "avg_attempts" in empirical_stddev
                    else ", "
                )
                + f"avg_punctured={empirical['avg_punctured_seed_count']:.3f}, "
                + f"avg_partial={empirical['avg_partial_state_count']:.3f}, "
                + f"sig_eq(obj)={empirical['avg_signature_hash_equivalents_object_model']:.3f}, "
                + f"sig_eq(bits)={empirical['avg_signature_hash_equivalents_concrete']:.3f}, "
                + f"verify_rate={empirical['verify_success_rate']:.3f}"
            )
        )
        if "avg_sign_ms" in empirical:
            lines.append(
                (
                    "  Time: "
                    f"sign={empirical['avg_sign_ms']:.3f} ms, "
                    f"verify={empirical['avg_verify_ms']:.3f} ms"
                )
            )
    return "\n".join(lines)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Benchmark YCSig using analytic cost formulas and empirical sampling.",
    )
    parser.add_argument("--config-file", help="Optional JSON file containing a list of benchmark cases.")
    parser.add_argument("--name", default="YCSig", help="Case label for single-case runs.")
    parser.add_argument("--security-parameter", type=int, default=128, help="Security parameter kappa.")
    parser.add_argument("--hash-len", type=int, default=None, help="HashLen in bits.")
    parser.add_argument("--max-g-bit", type=int, default=4, help="MaxGBit.")
    parser.add_argument("--partition-size", type=int, default=None, help="PartitionSize.")
    parser.add_argument("--window-radius", type=int, default=None, help="WindowRadius.")
    parser.add_argument("--samples", type=int, default=128, help="Number of empirical message samples.")
    parser.add_argument(
        "--repetitions",
        type=int,
        default=1,
        help="Number of independent benchmark repetitions whose outputs are averaged.",
    )
    parser.add_argument(
        "--acceptance-mode",
        choices=("exact", "monte_carlo"),
        default="exact",
        help="How to estimate the TreeAwareISP acceptance probability.",
    )
    parser.add_argument(
        "--acceptance-samples",
        type=int,
        default=10000,
        help="Monte Carlo samples for acceptance estimation when --acceptance-mode=monte_carlo.",
    )
    parser.add_argument(
        "--retry-cost-mode",
        choices=("total", "failures"),
        default="total",
        help="Count retry overhead as total attempts or only failed attempts.",
    )
    parser.add_argument(
        "--include-verifier-retry-cost",
        action="store_true",
        help="Add the partition-search retry overhead to the verification cost, useful for the compact-signature implementation where salt is not stored.",
    )
    parser.add_argument(
        "--signature-extra-hash-values",
        type=float,
        default=1.0,
        help="Extra hash-equivalent objects to add in the signature-size object model, e.g. 1 for the randomizer.",
    )
    parser.add_argument(
        "--signature-extra-bits",
        type=int,
        default=0,
        help="Extra fixed bits to add in the signature-size object model, e.g. 16 for an explicit salt.",
    )
    parser.add_argument(
        "--hash-equivalent-bits",
        type=int,
        default=None,
        help="Bit width used to normalize signature size into hash-equivalent units; defaults to HashLen.",
    )
    parser.add_argument("--message-bytes", type=int, default=None, help="Benchmark message length in bytes.")
    parser.add_argument("--random-seed", type=int, default=0, help="Deterministic seed for benchmark messages.")
    parser.add_argument(
        "--measure-time",
        action="store_true",
        help="Also measure empirical sign/verify wall-clock time.",
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
        "--format",
        choices=("json", "text"),
        default="json",
        help="Output format.",
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
        acceptance_mode=args.acceptance_mode,
        acceptance_samples=args.acceptance_samples,
        retry_cost_mode=args.retry_cost_mode,
        include_verifier_retry_cost=args.include_verifier_retry_cost,
        signature_extra_hash_values=args.signature_extra_hash_values,
        signature_extra_bits=args.signature_extra_bits,
        hash_equivalent_bits=args.hash_equivalent_bits,
        message_bytes=args.message_bytes,
        random_seed=args.random_seed,
        measure_time=args.measure_time,
        tweak_hash_name=args.tweak_hash_name,
        keyed_hash_name=args.keyed_hash_name,
        pprf_hash_name=args.pprf_hash_name,
        merkle_hash_name=args.merkle_hash_name,
    )


def _load_cases(args: argparse.Namespace) -> List[YCSigBenchmarkCase]:
    if args.config_file:
        with open(args.config_file, "r", encoding="utf-8") as handle:
            raw_cases = json.load(handle)
        if not isinstance(raw_cases, list):
            raise ValueError("config-file must contain a JSON list of case dictionaries")
        return [_case_from_dict(raw_case) for raw_case in raw_cases]
    return [_single_case_from_args(args)]


def _main() -> int:
    parser = _build_parser()
    args = parser.parse_args()
    cases = _load_cases(args)
    results = [run_benchmark_case_average(case, args.repetitions) for case in cases]

    if args.format == "text":
        print(_format_text(results))
    else:
        print(json.dumps(results, ensure_ascii=True, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(_main())
