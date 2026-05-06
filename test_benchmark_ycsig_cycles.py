import unittest

from benchmark_ycsig import YCSigBenchmarkCase
from benchmark_ycsig_cycles import _build_parser, run_cycle_benchmark_case
from benchmark_ycsig_table_cycles import run_paper_table_cycles
from operation_counter import counting_scope, snapshot


class BenchmarkYCSigCyclesTests(unittest.TestCase):
    def test_cli_defaults_use_runnable_single_case(self) -> None:
        args = _build_parser().parse_args([])

        self.assertEqual(args.name, "YCSig-w4-k128-H>=k")
        self.assertEqual(args.security_parameter, 128)
        self.assertEqual(args.hash_len, 130)
        self.assertEqual(args.max_g_bit, 2)
        self.assertEqual(args.partition_size, 33)
        self.assertEqual(args.window_radius, 1)
        self.assertEqual(args.cpu_frequency_ghz, 3.49)

    def test_run_cycle_benchmark_case(self) -> None:
        case = YCSigBenchmarkCase(
            name="cycle-case",
            security_parameter=128,
            hash_len=8,
            max_g_bit=2,
            partition_size=2,
            window_radius=1,
            samples=2,
            random_seed=5,
        )

        result = run_cycle_benchmark_case(
            case,
            repetitions=2,
            cpu_frequency_ghz=1.0,
        )

        self.assertEqual(result["case"], "cycle-case")
        self.assertEqual(result["parameters"]["total_experiments"], 4)
        self.assertTrue(result["parameters"]["setup_excluded"])
        self.assertTrue(result["parameters"]["prf_keygen_excluded"])
        self.assertGreater(result["cycles"]["avg_keygen_cycles"], 0.0)
        self.assertGreater(result["cycles"]["avg_sign_cycles"], 0.0)
        self.assertGreater(result["cycles"]["avg_verify_cycles"], 0.0)
        self.assertEqual(result["verify_rate"], 1.0)

    def test_run_cycle_benchmark_case_with_memory(self) -> None:
        case = YCSigBenchmarkCase(
            name="cycle-case-memory",
            security_parameter=128,
            hash_len=8,
            max_g_bit=2,
            partition_size=2,
            window_radius=1,
            samples=1,
            random_seed=11,
        )

        result = run_cycle_benchmark_case(
            case,
            repetitions=1,
            cpu_frequency_ghz=1.0,
            measure_memory=True,
        )

        self.assertIn("memory", result)
        self.assertGreater(result["memory"]["avg_peak_memory_bytes"], 0.0)
        self.assertGreater(result["memory"]["avg_peak_sign_memory_bytes"], 0.0)

    def test_run_paper_table_cycles(self) -> None:
        results = run_paper_table_cycles(
            samples=1,
            repetitions=1,
            cpu_frequency_ghz=1.0,
        )
        first = results[0]

        self.assertEqual(first["case"], "YCSig-w4-k128-H>=k")
        self.assertAlmostEqual(
            first["signature"]["signature_hash_equivalents_override"],
            93.0,
        )

    def test_cycle_benchmark_does_not_update_operation_counters(self) -> None:
        case = YCSigBenchmarkCase(
            name="cycle-counters-off",
            security_parameter=128,
            hash_len=8,
            max_g_bit=2,
            partition_size=2,
            window_radius=1,
            samples=1,
            random_seed=7,
        )

        with counting_scope():
            run_cycle_benchmark_case(
                case,
                repetitions=1,
                cpu_frequency_ghz=1.0,
            )
            counters = snapshot()

        self.assertEqual(counters, {})


if __name__ == "__main__":
    unittest.main()
