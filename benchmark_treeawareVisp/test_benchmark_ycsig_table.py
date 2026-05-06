import json
import tempfile
import unittest

from benchmark_ycsig_table import _load_paper_rows_from_json, run_paper_table


class BenchmarkYCSigTableTests(unittest.TestCase):
    def test_first_row_matches_paper_numbers(self) -> None:
        results = run_paper_table(samples=1, repetitions=2)
        first = results[0]

        self.assertEqual(first["case"], "YCSig-w4-k128-H>=k")
        self.assertEqual(first["parameters"]["repetitions"], 2)
        self.assertAlmostEqual(first["paper_row"]["keygen"], 393.0)
        self.assertAlmostEqual(first["paper_row"]["sign"], 209.7284)
        self.assertAlmostEqual(first["paper_row"]["verify"], 199.2686)
        self.assertAlmostEqual(
            first["paper_row"]["sig_size"],
            95.8048,
        )

    def test_search_json_rows_can_load(self) -> None:
        with tempfile.NamedTemporaryFile("w+", suffix=".json") as handle:
            json.dump(
                [
                    {
                        "case_name": "case2",
                        "security_target": 128,
                        "max_g_value": 16,
                        "partition_num": 9,
                        "window_radius": 3,
                        "block_num": 64,
                        "hash_len": 256,
                    }
                ],
                handle,
            )
            handle.flush()
            rows = _load_paper_rows_from_json(handle.name)

        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0].label, "YCSig-w16-k128-H>=2k")
        self.assertEqual(rows[0].regime, "H>=2k")
        self.assertEqual(rows[0].partition_size, 9)
        self.assertEqual(rows[0].window_radius, 3)
        self.assertIsNone(rows[0].keygen)
        self.assertIsNone(rows[0].sig_size)


if __name__ == "__main__":
    unittest.main()
