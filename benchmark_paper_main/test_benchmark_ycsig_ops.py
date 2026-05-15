import unittest
from math import isclose

from benchmark_ycsig import YCSigBenchmarkCase
from benchmark_ycsig_ops import run_operation_benchmark_case
from benchmark_ycsig_table_ops import run_paper_table_ops


class BenchmarkYCSigOpsTests(unittest.TestCase):
    def test_run_operation_benchmark_case(self) -> None:
        case = YCSigBenchmarkCase(
            name="ops-case",
            security_parameter=128,
            hash_len=8,
            max_g_bit=2,
            partition_size=2,
            window_radius=1,
            samples=2,
            random_seed=7,
        )

        result = run_operation_benchmark_case(case, repetitions=2)

        self.assertEqual(result["case"], "ops-case")
        self.assertEqual(result["parameters"]["total_experiments"], 4)
        self.assertTrue(result["parameters"]["setup_excluded"])
        self.assertGreater(result["operations"]["keygen_hash_equivalents_real"], 0.0)
        self.assertGreater(result["operations"]["sign_hash_equivalents_real"], 0.0)
        self.assertGreater(result["operations"]["verify_hash_equivalents_real"], 0.0)
        self.assertGreater(result["operations"]["retry_attempt_count_real"], 0.0)
        self.assertTrue(
            isclose(
                result["operations"]["retry_hash_equivalents_real"],
                result["operations"]["retry_attempt_hash_equivalents_real"]
                + result["operations"]["retry_sampler_hash_equivalents_real"],
                abs_tol=1e-12,
            )
        )
        self.assertTrue(
            isclose(result["operations"]["keygen_sign_relation_gap_real"], 0.0, abs_tol=1e-12)
        )
        self.assertEqual(result["verify_rate"], 1.0)

    def test_run_paper_table_ops(self) -> None:
        results = run_paper_table_ops(samples=1, repetitions=1)
        first = results[0]

        self.assertEqual(first["case"], "YCSig-w4-k128-H>=k")
        self.assertIn("paper_row", first)
        self.assertIn("operations", first)
        self.assertTrue(
            isclose(first["operations"]["keygen_sign_relation_gap_real"], 0.0, abs_tol=1e-12)
        )


if __name__ == "__main__":
    unittest.main()
