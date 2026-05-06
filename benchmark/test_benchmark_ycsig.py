import unittest

from benchmark_ycsig import (
    YCSigBenchmarkCase,
    run_benchmark_case,
    run_benchmark_case_average,
)


class BenchmarkYCSigTests(unittest.TestCase):
    def test_run_benchmark_case(self) -> None:
        case = YCSigBenchmarkCase(
            name="small-case",
            security_parameter=128,
            hash_len=8,
            max_g_bit=2,
            partition_size=2,
            window_radius=1,
            samples=4,
            acceptance_mode="exact",
            signature_extra_hash_values=1.0,
            random_seed=7,
        )

        result = run_benchmark_case(case)

        self.assertEqual(result["case"], "small-case")
        self.assertGreater(result["analytic"]["acceptance_probability"], 0.0)
        self.assertGreater(result["analytic"]["keygen_hash_equivalents"], 0.0)
        self.assertGreater(result["analytic"]["sign_hash_equivalents"], 0.0)
        self.assertGreater(result["analytic"]["verify_hash_equivalents"], 0.0)
        self.assertEqual(result["empirical"]["verify_success_rate"], 1.0)
        self.assertGreater(result["empirical"]["avg_signature_bits"], 0.0)
        self.assertGreater(
            result["empirical"]["avg_signature_hash_equivalents_object_model"],
            1.0,
        )

    def test_run_benchmark_case_average(self) -> None:
        case = YCSigBenchmarkCase(
            name="avg-case",
            security_parameter=128,
            hash_len=8,
            max_g_bit=2,
            partition_size=2,
            window_radius=1,
            samples=2,
            acceptance_mode="exact",
            signature_extra_hash_values=1.0,
            random_seed=11,
        )

        result = run_benchmark_case_average(case, repetitions=3)

        self.assertEqual(result["case"], "avg-case")
        self.assertEqual(result["parameters"]["repetitions"], 3)
        self.assertIn("avg_attempts", result["empirical"])
        self.assertIn("avg_attempts", result["empirical_stddev"])
        self.assertGreaterEqual(result["empirical"]["verify_success_rate"], 0.0)
        self.assertLessEqual(result["empirical"]["verify_success_rate"], 1.0)


if __name__ == "__main__":
    unittest.main()
