from __future__ import annotations

import argparse
import os
import sys
from collections.abc import Callable

from benchmark_val_strict_isp_cycles import (
    _build_metric_function,
    _collect_partition_values,
)
from benchmark_ycsig import _derive_key_seed_for_sample, _message_for_sample, _randomizer_for_sample
from benchmark_ycsig_cycles import (
    _build_scheme,
    _case_for_repetition,
    _warm_up_core_algorithms,
    deserialize_case_from_worker_json,
)
from operation_counter import disabled_scope
from pprf import PPRF
from val_strict_isp import ISPParameters
from val_strict_isp import _NATIVE_ACCEPT_CHECK_BATCH_FAST
from val_strict_isp import _NATIVE_PROFILE_COUNTS_BATCH_FAST
from val_strict_isp import _NATIVE_W4_ACCEPT_CHECK_BATCH_FAST


def _run_after_optional_sync(args: argparse.Namespace, run: Callable[[], None]) -> None:
    if not getattr(args, "sync_stdin", False):
        run()
        return

    print(f"READY {os.getpid()}", flush=True)
    command = sys.stdin.readline().strip()
    if command != "RUN":
        raise RuntimeError(f"expected RUN on stdin, got {command!r}")
    run()


def _run_ycsig_metric(args: argparse.Namespace) -> None:
    case = deserialize_case_from_worker_json(args.case_json)
    rep_case = _case_for_repetition(case, args.repetition)

    with disabled_scope():
        scheme = _build_scheme(rep_case)
        _warm_up_core_algorithms(rep_case, scheme)

        messages = [_message_for_sample(rep_case, index) for index in range(rep_case.samples)]
        key_seeds = [_derive_key_seed_for_sample(rep_case, index) for index in range(rep_case.samples)]
        randomizers = [_randomizer_for_sample(rep_case, index) for index in range(rep_case.samples)]

        if args.metric == "keygen":
            workloads = [PPRF.PRFKGen(scheme.params.pm_PPRF, seed=seed) for seed in key_seeds]

            def run() -> None:
                for _ in range(args.inner_repetitions):
                    for ks in workloads:
                        scheme.SigGen(ks)

            _run_after_optional_sync(args, run)
            return

        if args.metric == "retry":
            workloads = list(zip(messages, randomizers))

            def run() -> None:
                for _ in range(args.inner_repetitions):
                    for message, randomizer in workloads:
                        scheme.FindPartition(message, randomizer)

            _run_after_optional_sync(args, run)
            return

        if args.metric == "sign_core":
            workloads = []
            for key_seed, message, randomizer in zip(key_seeds, messages, randomizers):
                ks = PPRF.PRFKGen(scheme.params.pm_PPRF, seed=key_seed)
                keypair = scheme.SigGen(ks)
                salt, groups = scheme.FindPartition(message, randomizer)
                workloads.append((keypair.secret_key, randomizer, salt, groups))

            def run() -> None:
                for _ in range(args.inner_repetitions):
                    for secret_key, randomizer, salt, groups in workloads:
                        scheme.SignWithGroups(secret_key, randomizer, salt, groups)

            _run_after_optional_sync(args, run)
            return

        if args.metric == "verify":
            workloads = []
            for key_seed, message, randomizer in zip(key_seeds, messages, randomizers):
                ks = PPRF.PRFKGen(scheme.params.pm_PPRF, seed=key_seed)
                keypair = scheme.SigGen(ks)
                salt, groups = scheme.FindPartition(message, randomizer)
                signature = scheme.SignWithGroups(keypair.secret_key, randomizer, salt, groups)
                workloads.append((keypair.public_key, signature, groups))

            def run() -> None:
                for _ in range(args.inner_repetitions):
                    for public_key, signature, groups in workloads:
                        if not scheme.VerifyWithGroups(public_key, signature, groups):
                            raise AssertionError("precomputed signature failed verification")

            _run_after_optional_sync(args, run)
            return

    raise ValueError(f"unsupported YCSig worker metric: {args.metric}")


def _run_valstrictisp_metric(args: argparse.Namespace) -> None:
    params = ISPParameters(
        hash_len=args.hash_len,
        max_g_bit=args.max_g_bit,
        partition_num=args.partition_num,
        window_radius=args.window_radius,
        hash_name=args.hash_name,
    )
    sample_sets = _collect_partition_values(
        params=params,
        accepted_samples=args.accepted_samples,
        rejected_samples=args.rejected_samples,
        random_seed=args.random_seed,
        max_candidate_attempts=args.max_candidate_attempts,
    )

    metric_map = {
        "profile_only": sample_sets["accepted"],
        "accept_check": sample_sets["accepted"],
        "full_accept": sample_sets["accepted"],
        "full_reject": sample_sets["rejected"],
    }
    values = metric_map[args.metric]
    func = _build_metric_function(args.metric, params, args.sampler_mode)
    func(values[0])

    with disabled_scope():
        if (
            args.metric == "profile_only"
            and args.sampler_mode == "random"
            and _NATIVE_PROFILE_COUNTS_BATCH_FAST is not None
            and 2 <= params.max_g_bit <= 6
        ):

            def run() -> None:
                checksum = _NATIVE_PROFILE_COUNTS_BATCH_FAST(
                    values,
                    params.hash_len,
                    params.max_g_bit,
                    args.inner_repetitions,
                )
                expected = len(values) * args.inner_repetitions * params.block_num
                if checksum != expected:
                    raise AssertionError("profile-only batch checksum mismatch")

            _run_after_optional_sync(args, run)
            return

        if (
            args.metric in {"accept_check", "full_reject"}
            and args.sampler_mode == "random"
            and _NATIVE_W4_ACCEPT_CHECK_BATCH_FAST is not None
            and params.max_g_bit == 2
            and params.max_g_value == 4
        ):

            def run() -> None:
                accepted_total = _NATIVE_W4_ACCEPT_CHECK_BATCH_FAST(
                    values,
                    params.hash_len,
                    params.window_low,
                    params.window_high,
                    args.inner_repetitions,
                )
                if args.metric == "full_reject" and accepted_total != 0:
                    raise AssertionError("expected rejecting partition values")
                if args.metric == "accept_check" and accepted_total != len(values) * args.inner_repetitions:
                    raise AssertionError("expected accepting partition values")

            _run_after_optional_sync(args, run)
            return

        if (
            args.metric in {"accept_check", "full_reject"}
            and args.sampler_mode == "random"
            and _NATIVE_ACCEPT_CHECK_BATCH_FAST is not None
            and 2 <= params.max_g_bit <= 6
        ):

            def run() -> None:
                accepted_total = _NATIVE_ACCEPT_CHECK_BATCH_FAST(
                    values,
                    params.hash_len,
                    params.max_g_bit,
                    params.window_low,
                    params.window_high,
                    args.inner_repetitions,
                )
                if args.metric == "full_reject" and accepted_total != 0:
                    raise AssertionError("expected rejecting partition values")
                if args.metric == "accept_check" and accepted_total != len(values) * args.inner_repetitions:
                    raise AssertionError("expected accepting partition values")

            _run_after_optional_sync(args, run)
            return

        def run() -> None:
            for _ in range(args.inner_repetitions):
                for partition_value in values:
                    func(partition_value)

        _run_after_optional_sync(args, run)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Internal worker used by the xctrace true-cycle benchmark backend.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    ycsig = subparsers.add_parser("ycsig")
    ycsig.add_argument("--case-json", required=True)
    ycsig.add_argument("--metric", choices=("keygen", "retry", "sign_core", "verify"), required=True)
    ycsig.add_argument("--repetition", type=int, required=True)
    ycsig.add_argument("--inner-repetitions", type=int, required=True)
    ycsig.add_argument("--sync-stdin", action="store_true")

    valstrictisp = subparsers.add_parser("valstrictisp")
    valstrictisp.add_argument("--metric", choices=("profile_only", "accept_check", "full_accept", "full_reject"), required=True)
    valstrictisp.add_argument("--hash-len", type=int, required=True)
    valstrictisp.add_argument("--max-g-bit", type=int, required=True)
    valstrictisp.add_argument("--partition-num", type=int, required=True)
    valstrictisp.add_argument("--window-radius", type=int, required=True)
    valstrictisp.add_argument("--hash-name", required=True)
    valstrictisp.add_argument("--accepted-samples", type=int, required=True)
    valstrictisp.add_argument("--rejected-samples", type=int, required=True)
    valstrictisp.add_argument("--random-seed", type=int, required=True)
    valstrictisp.add_argument("--max-candidate-attempts", type=int, required=True)
    valstrictisp.add_argument("--inner-repetitions", type=int, required=True)
    valstrictisp.add_argument("--sampler-mode", choices=("seeded", "random"), default="seeded")
    valstrictisp.add_argument("--sync-stdin", action="store_true")

    return parser


def _main() -> int:
    args = _build_parser().parse_args()
    if args.command == "ycsig":
        _run_ycsig_metric(args)
    else:
        _run_valstrictisp_metric(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(_main())
