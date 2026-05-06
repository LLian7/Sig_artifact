import unittest
from math import isclose

from benchmark_ycsig_table_dual import _format_text, run_paper_table_dual


class BenchmarkYCSigTableDualTests(unittest.TestCase):
    def test_run_dual_benchmark_both_modes(self) -> None:
        results = run_paper_table_dual(
            samples=1,
            repetitions=1,
            cpu_frequency_ghz=1.0,
            mode="both",
        )

        self.assertIn("ops", results)
        self.assertIn("cycles", results)
        self.assertEqual(results["ops"][0]["case"], "YCSig-w4-k128-H>=k")
        self.assertEqual(results["cycles"][0]["case"], "YCSig-w4-k128-H>=k")
        self.assertTrue(
            isclose(
                results["ops"][0]["operations"]["keygen_sign_relation_gap_real"],
                0.0,
                abs_tol=1e-12,
            )
        )
        self.assertTrue(
            isclose(
                results["ops"][0]["signature"]["avg_signature_hash_equivalents_concrete"],
                results["cycles"][0]["signature"]["avg_signature_hash_equivalents_concrete"],
                abs_tol=1e-12,
            )
        )

    def test_format_text_includes_gap_column(self) -> None:
        results = run_paper_table_dual(
            samples=1,
            repetitions=1,
            cpu_frequency_ghz=1.0,
            mode="ops",
        )

        rendered = _format_text(results)

        self.assertIn("Gap", rendered)
        self.assertNotIn("1/p_s(theory)", rendered)
        self.assertNotIn("Attempts(real)", rendered)
        self.assertNotIn("Sampler(real)", rendered)
        self.assertNotIn("Retry(real,total)", rendered)
        self.assertIn("0.0e+00", rendered)

if __name__ == "__main__":
    unittest.main()
