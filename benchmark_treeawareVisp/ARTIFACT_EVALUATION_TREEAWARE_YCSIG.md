# TreeAwareISP + YCSig Artifact Evaluation Guide

This document explains how to validate and benchmark the TreeAwareISP + YCSig
artifact in this directory. It is intentionally benchmark-facing: the commands
below produce text or JSON outputs and do not generate paper `.tex` files.

## Scope

The relevant implementation files are:

- `treeaware_isp.py`: TreeAwareISP / ValStrictISPT sampler, routing policies,
  public score guards, and command-line sampler.
- `yc_sig.py`: YCSig implementation. Tree-aware size and verification modes use
  complement signing, so signing opens the complement side and verification
  recomputes the selected side.
- `benchmark_ycsig.py`: analytic hash-equivalent cost model plus empirical
  sampling for KeyGen, Sign, Verify, signature-size, acceptance, and retries.
- `benchmark_ycsig_ops.py`: primitive/hash-equivalent operation-count
  benchmark for a single case.
- `benchmark_ycsig_cycles.py`: estimated CPU-cycle benchmark for a single case
  or a JSON config.
- `benchmark_ycsig_table.py`, `benchmark_ycsig_table_ops.py`,
  `benchmark_ycsig_table_cycles.py`, and `benchmark_ycsig_table_dual.py`:
  convenience entry points for the main YCSig table configurations.
- `treeaware_ycsig_bench_config.json`: TreeAwareISP + YCSig case list.
- `ycsig_table_bench_config.json`: baseline YCSig table case list.

## Environment

Run commands from this directory:

```bash
cd "/Users/zhengyang/D Disk/Google Drive/My paper/2025/Accelerating Hash-based Signatures /benchmark_treeawareVisp"
```

The artifact is Python-based. The core correctness and hash-equivalent
benchmarks use the Python standard library. Cycle numbers are machine-dependent
and should be treated as local performance measurements; pass the local CPU
frequency explicitly when reporting cycle-equivalent numbers.

## Correctness Smoke Tests

Run the TreeAwareISP and YCSig unit tests:

```bash
python3 -m unittest test_treeaware_isp.py test_yc_sig.py
```

This checks the TreeAwareISP sampler, YCSig signing and verification, and the
tree-aware complement-signing behavior used by the benchmark rows.

## TreeAwareISP Sampler Smoke Test

A direct sampler run is useful for checking parameter parsing and public
routing behavior:

```bash
python3 treeaware_isp.py test-message \
  --input-mode message \
  --hash-len 128 \
  --max-g-bit 2 \
  --partition-num 40 \
  --window-radius 2 \
  --mode size
```

The output reports either an accepted group list or `Bottom`, together with
resolved routing metadata.

## TreeAwareISP + YCSig Benchmark Rows

The TreeAwareISP + YCSig benchmark cases are stored in
`treeaware_ycsig_bench_config.json`. A quick smoke run is:

```bash
python3 benchmark_ycsig.py \
  --config-file treeaware_ycsig_bench_config.json \
  --samples 1 \
  --repetitions 1 \
  --format text
```

For more stable empirical averages, increase `--samples` and `--repetitions`.
For machine-readable output, use:

```bash
python3 benchmark_ycsig.py \
  --config-file treeaware_ycsig_bench_config.json \
  --samples 32 \
  --repetitions 10 \
  --format json
```

The benchmark supports two retry-accounting conventions:

- `--retry-cost-mode total`: retry overhead is the expected total number of
  TreeAwareISP attempts.
- `--retry-cost-mode failures`: retry overhead counts only failed attempts.

The default artifact setting is `total`, matching the table convention where
the displayed signing cost includes the Las Vegas retry cost.

## Baseline YCSig Table Benchmarks

To run the baseline YCSig table configuration:

```bash
python3 benchmark_ycsig.py \
  --config-file ycsig_table_bench_config.json \
  --samples 32 \
  --repetitions 10 \
  --format text
```

The convenience table scripts provide separate views:

```bash
python3 benchmark_ycsig_table.py --samples 32 --repetitions 10 --format text
python3 benchmark_ycsig_table_ops.py --samples 32 --repetitions 10 --format text
python3 benchmark_ycsig_table_cycles.py --samples 32 --repetitions 10 --cpu-frequency-ghz 3.49 --format text
python3 benchmark_ycsig_table_dual.py --samples 32 --repetitions 10 --mode both --cpu-frequency-ghz 3.49 --format text
```

Use `--format json` when collecting data for scripts. Avoid `--format latex` in
this artifact directory unless a separate paper-generation step is explicitly
needed.

## Single-Case Benchmarks

For a single hash-equivalent benchmark:

```bash
python3 benchmark_ycsig.py \
  --name smoke-ycsig \
  --security-parameter 128 \
  --hash-len 128 \
  --max-g-bit 2 \
  --partition-size 40 \
  --window-radius 2 \
  --samples 8 \
  --repetitions 3 \
  --format text
```

For real primitive/hash-equivalent operation counts:

```bash
python3 benchmark_ycsig_ops.py \
  --name ops-smoke \
  --security-parameter 128 \
  --hash-len 128 \
  --max-g-bit 2 \
  --partition-size 40 \
  --window-radius 2 \
  --samples 8 \
  --repetitions 3 \
  --format text
```

For local cycle estimates:

```bash
python3 benchmark_ycsig_cycles.py \
  --name cycles-smoke \
  --security-parameter 128 \
  --hash-len 128 \
  --max-g-bit 2 \
  --partition-size 40 \
  --window-radius 2 \
  --samples 8 \
  --repetitions 3 \
  --cpu-frequency-ghz 3.49 \
  --format text
```

## Output Fields

The most important fields are:

- `keygen_hash_equivalents`: analytic KeyGen cost in hash-equivalent units.
- `avg_sign_hash_equivalents`: average signing cost, including retry overhead
  under the selected retry-cost convention.
- `avg_verify_hash_equivalents`: average verification cost.
- `avg_signature_hash_equivalents_object_model`: signature size normalized into
  hash-equivalent objects.
- `acceptance_probability`: one-trial TreeAwareISP acceptance probability.
- `expected_retries`: reciprocal of the acceptance probability under the
  single-input retry model.
- `verify_success_rate`: should be `1.0` in successful benchmark runs.

For TreeAwareISP rows with size or verification guards, the guard is
sample-then-abort. It changes the one-input acceptance probability and expected
retry cost, but it does not weaken the full-support UCR certificate.

## Reproducibility Notes

- Use JSON output for archived benchmark artifacts.
- Keep `samples` and `repetitions` small for smoke tests, then increase them
  for reported numbers.
- Cycle measurements depend on CPU frequency, system load, and interpreter
  version. Report the frequency used with `--cpu-frequency-ghz` or
  `--cpu-frequency-hz`.
- A signing benchmark may raise a retry-budget error if the acceptance
  probability is very low and the runtime retry budget is too small. Use the
  exact acceptance/retry fields from `benchmark_ycsig.py` to select practical
  candidates before running expensive timing measurements.
