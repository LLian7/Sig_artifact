import json
import tempfile
import unittest

from benchmark_ycsig_table import _load_paper_rows_from_json, paper_rows, run_paper_table


TABLE_ALL_L_WINDOWED_ROWS = {
    ("H>=k", 128, 4): (128, 32, 2, 64),
    ("H>=k", 128, 8): (168, 16, 2, 56),
    ("H>=k", 128, 16): (248, 8, 2, 62),
    ("H>=k", 128, 32): (200, 5, 1, 40),
    ("H>=k", 160, 4): (160, 40, 2, 80),
    ("H>=k", 160, 8): (219, 20, 2, 73),
    ("H>=k", 160, 16): (292, 10, 2, 73),
    ("H>=k", 160, 32): (320, 6, 2, 64),
    ("H>=2k", 128, 4): (256, 42, 6, 128),
    ("H>=2k", 128, 8): (258, 17, 3, 86),
    ("H>=2k", 128, 16): (256, 9, 4, 64),
    ("H>=2k", 128, 32): (320, 5, 2, 64),
    ("H>=2k", 160, 4): (320, 52, 5, 160),
    ("H>=2k", 160, 8): (321, 22, 6, 107),
    ("H>=2k", 160, 16): (320, 11, 5, 80),
    ("H>=2k", 160, 32): (320, 6, 2, 64),
}


class BenchmarkYCSigTableTests(unittest.TestCase):
    def test_first_row_matches_paper_numbers(self) -> None:
        results = run_paper_table(samples=1, repetitions=2)
        first = results[0]

        self.assertEqual(first["case"], "YCSig-w4-k128-H>=k")
        self.assertEqual(first["parameters"]["repetitions"], 2)
        self.assertAlmostEqual(first["paper_row"]["keygen"], 382.0)
        self.assertAlmostEqual(first["paper_row"]["sign"], 199.62)
        self.assertAlmostEqual(first["paper_row"]["verify"], 191.01)
        self.assertAlmostEqual(
            first["paper_row"]["sig_size"],
            93.295,
        )

    def test_paper_rows_match_table_all_l_windowed_parameters(self) -> None:
        rows = paper_rows()
        self.assertEqual(len(rows), len(TABLE_ALL_L_WINDOWED_ROWS))
        for row in rows:
            expected = TABLE_ALL_L_WINDOWED_ROWS[
                (row.regime, row.security_parameter, row.max_g_value)
            ]
            self.assertEqual(
                (row.hash_len, row.partition_size, row.window_radius),
                expected[:3],
            )
            max_g_bit = row.max_g_value.bit_length() - 1
            self.assertEqual(row.hash_len % max_g_bit, 0)
            self.assertEqual(row.hash_len // max_g_bit, expected[3])

    def test_search_json_rows_can_load(self) -> None:
        with tempfile.NamedTemporaryFile("w+", suffix=".json") as handle:
            json.dump(
                [
                    {
                        "case_name": "case2",
                        "security_target": 128,
                        "max_g_value": 16,
                        "partition_num": 9,
                        "window_radius": 4,
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
        self.assertEqual(rows[0].window_radius, 4)
        self.assertIsNone(rows[0].keygen)
        self.assertIsNone(rows[0].sig_size)


if __name__ == "__main__":
    unittest.main()
