import json
import math
import tempfile
import unittest
from unittest import mock

from benchmark_ycsig_table_comparison import (
    COMPARISON_CASES,
    _load_comparison_cases_from_json,
    render_text as render_comparison_text,
)
from search_ycsig_sigsize import (
    SearchCell,
    SearchRow,
    _analytic_sign_cost,
    exact_parameter_metrics,
    search_best_row_for_cell,
    search_rows,
)
from val_strict_isp import ISPParameters, val_strict_isp


TABLE_ALL_L_WINDOWED_COMPARISON_ROWS = {
    ("case1", 128, 4): (32, 2, 64, 128),
    ("case1", 128, 8): (16, 2, 56, 168),
    ("case1", 128, 16): (8, 2, 62, 248),
    ("case1", 128, 32): (5, 1, 40, 200),
    ("case1", 160, 4): (40, 2, 80, 160),
    ("case1", 160, 8): (20, 2, 73, 219),
    ("case1", 160, 16): (10, 2, 73, 292),
    ("case1", 160, 32): (6, 2, 64, 320),
    ("case2", 128, 4): (42, 6, 128, 256),
    ("case2", 128, 8): (17, 3, 86, 258),
    ("case2", 128, 16): (9, 4, 64, 256),
    ("case2", 128, 32): (5, 2, 64, 320),
    ("case2", 160, 4): (52, 5, 160, 320),
    ("case2", 160, 8): (22, 6, 107, 321),
    ("case2", 160, 16): (11, 5, 80, 320),
    ("case2", 160, 32): (6, 2, 64, 320),
}


class SearchYCSigSigSizeTests(unittest.TestCase):
    def test_exact_metrics_returns_zero_for_impossible_profile(self) -> None:
        acceptance_probability, kappa = exact_parameter_metrics(
            5,
            4,
            1,
            0,
        )

        self.assertEqual(acceptance_probability, 0.0)
        self.assertEqual(kappa, float("-inf"))

    def test_exact_metrics_match_bruteforce_valstrictisp_on_toy_instance(self) -> None:
        hash_len = 6
        max_g_bit = 1
        block_num = hash_len // max_g_bit
        max_g_value = 1 << max_g_bit
        partition_num = 3
        window_radius = 1
        params = ISPParameters(
            hash_len=hash_len,
            max_g_bit=max_g_bit,
            partition_num=partition_num,
            window_radius=window_radius,
        )

        outputs = []
        accepted = 0
        space_size = 1 << hash_len
        for partition_value in range(space_size):
            groups = val_strict_isp(partition_value, params)
            if groups is not None:
                accepted += 1
            outputs.append(None if groups is None else tuple(tuple(group) for group in groups))

        collision_count = 0
        for left in range(space_size):
            if outputs[left] is None:
                continue
            for right in range(space_size):
                if left == right:
                    continue
                if outputs[left] == outputs[right]:
                    collision_count += 1

        measured_acceptance_probability, measured_kappa = exact_parameter_metrics(
            block_num,
            max_g_value,
            partition_num,
            window_radius,
        )
        expected_acceptance_probability = accepted / space_size
        expected_ucr_probability = collision_count / (space_size * (space_size - 1))
        expected_kappa = -math.log2(expected_ucr_probability)

        self.assertAlmostEqual(
            measured_acceptance_probability,
            expected_acceptance_probability,
            places=12,
        )
        self.assertAlmostEqual(measured_kappa, expected_kappa, places=12)

    def test_case2_partition_num_search_matches_bruteforce_on_small_cell(self) -> None:
        cell = SearchCell(case_name="case2", security_target=16, max_g_value=4)
        hash_len_max = 40
        row = search_best_row_for_cell(
            cell=cell,
            objective="partition_num",
            retry_limit=16.0,
            hash_len_max=hash_len_max,
        )

        self.assertIsNotNone(row)
        assert row is not None
        brute_force_best = None
        for hash_len in range(cell.min_hash_len, hash_len_max + 1, cell.max_g_bit):
            block_num = hash_len // cell.max_g_bit
            min_partition_num = math.ceil(block_num / cell.max_g_value)
            for partition_num in range(min_partition_num, block_num + 1):
                for window_radius in range(block_num // cell.max_g_value + 1):
                    acceptance_probability, kappa = exact_parameter_metrics(
                        block_num,
                        cell.max_g_value,
                        partition_num,
                        window_radius,
                    )
                    if acceptance_probability <= 0.0:
                        continue
                    expected_retries = 1.0 / acceptance_probability
                    if expected_retries > 16.0 or kappa < cell.security_target:
                        continue
                    candidate_key = (
                        partition_num,
                        hash_len,
                        window_radius,
                    )
                    if brute_force_best is None or candidate_key < brute_force_best:
                        brute_force_best = candidate_key

        self.assertEqual(
            (row.partition_num, row.hash_len, row.window_radius),
            brute_force_best,
        )

    def test_case1_w4_search_prioritizes_partition_footprint_then_block_num_then_sign_cost(self) -> None:
        cell = SearchCell(case_name="case1", security_target=32, max_g_value=4)
        hash_len_max = 64
        row = search_best_row_for_cell(
            cell=cell,
            objective="partition_num",
            retry_limit=16.0,
            hash_len_max=hash_len_max,
        )

        self.assertIsNotNone(row)
        assert row is not None
        brute_force_best_key = None
        brute_force_best_row = None
        for hash_len in range(32, hash_len_max + 1, cell.max_g_bit):
            block_num = hash_len // cell.max_g_bit
            min_partition_num = math.ceil(block_num / cell.max_g_value)
            for partition_num in range(min_partition_num, block_num + 1):
                for window_radius in range(block_num // cell.max_g_value + 1):
                    acceptance_probability, kappa = exact_parameter_metrics(
                        block_num,
                        cell.max_g_value,
                        partition_num,
                        window_radius,
                    )
                    if acceptance_probability <= 0.0 or math.isnan(kappa):
                        continue
                    expected_retries = 1.0 / acceptance_probability
                    if expected_retries > 16.0 or kappa < cell.security_target:
                        continue
                    candidate_key = (
                        partition_num * cell.max_g_value,
                        block_num,
                        _analytic_sign_cost(
                            block_num=block_num,
                            max_g_value=cell.max_g_value,
                            partition_num=partition_num,
                            expected_retries=expected_retries,
                        ),
                        partition_num,
                        expected_retries,
                        window_radius,
                        hash_len,
                    )
                    if brute_force_best_key is None or candidate_key < brute_force_best_key:
                        brute_force_best_key = candidate_key
                        brute_force_best_row = (
                            partition_num,
                            window_radius,
                            block_num,
                            hash_len,
                        )

        self.assertEqual(
            (row.partition_num, row.window_radius, row.block_num, row.hash_len),
            brute_force_best_row,
        )

    def test_case1_w16_search_prioritizes_partition_footprint_then_block_num_then_sign_cost(self) -> None:
        cell = SearchCell(case_name="case1", security_target=32, max_g_value=16)
        hash_len_max = 64
        row = search_best_row_for_cell(
            cell=cell,
            objective="partition_num",
            retry_limit=16.0,
            hash_len_max=hash_len_max,
        )

        self.assertIsNotNone(row)
        assert row is not None
        brute_force_best_key = None
        brute_force_best_row = None
        for hash_len in range(32, hash_len_max + 1, cell.max_g_bit):
            block_num = hash_len // cell.max_g_bit
            min_partition_num = math.ceil(block_num / cell.max_g_value)
            for partition_num in range(min_partition_num, block_num + 1):
                for window_radius in range(block_num // cell.max_g_value + 1):
                    acceptance_probability, kappa = exact_parameter_metrics(
                        block_num,
                        cell.max_g_value,
                        partition_num,
                        window_radius,
                    )
                    if acceptance_probability <= 0.0 or math.isnan(kappa):
                        continue
                    expected_retries = 1.0 / acceptance_probability
                    if expected_retries > 16.0 or kappa < cell.security_target:
                        continue
                    candidate_key = (
                        partition_num * cell.max_g_value,
                        block_num,
                        _analytic_sign_cost(
                            block_num=block_num,
                            max_g_value=cell.max_g_value,
                            partition_num=partition_num,
                            expected_retries=expected_retries,
                        ),
                        expected_retries,
                        window_radius,
                        hash_len,
                    )
                    if brute_force_best_key is None or candidate_key < brute_force_best_key:
                        brute_force_best_key = candidate_key
                        brute_force_best_row = (
                            partition_num,
                            window_radius,
                            block_num,
                            hash_len,
                        )

        self.assertEqual(
            (row.partition_num, row.window_radius, row.block_num, row.hash_len),
            brute_force_best_row,
        )

    def test_partition_num_search_skips_nan_kappa_rows(self) -> None:
        cell = SearchCell(case_name="case1", security_target=32, max_g_value=4)

        def _fake_exact_parameter_metrics(
            block_num: int,
            max_g_value: int,
            partition_num: int,
            window_radius: int,
        ) -> tuple[float, float]:
            self.assertEqual(block_num, 16)
            self.assertEqual(max_g_value, 4)
            self.assertEqual(partition_num, 4)
            if window_radius == 0:
                return 0.5, float("nan")
            return 0.5, 32.0

        with mock.patch(
            "search_ycsig_sigsize.exact_parameter_metrics",
            side_effect=_fake_exact_parameter_metrics,
        ):
            row = search_best_row_for_cell(
                cell=cell,
                objective="partition_num",
                retry_limit=16.0,
                hash_len_max=32,
            )

        self.assertIsNotNone(row)
        assert row is not None
        self.assertEqual((row.partition_num, row.window_radius, row.hash_len), (4, 1, 32))

    def test_comparison_cases_can_load_from_search_json(self) -> None:
        with tempfile.NamedTemporaryFile("w+", suffix=".json") as handle:
            json.dump(
                [
                    {
                        "case_name": "case1",
                        "security_target": 128,
                        "max_g_value": 4,
                        "partition_num": 33,
                        "window_radius": 1,
                        "block_num": 65,
                        "hash_len": 130,
                        "expected_retries": 14.9,
                        "kappa": 128.3,
                    }
                ],
                handle,
            )
            handle.flush()
            cases = _load_comparison_cases_from_json(handle.name)

        self.assertEqual(len(cases), 1)
        self.assertEqual(cases[0].case_name, "case1")
        self.assertEqual(cases[0].partition_size, 33)
        self.assertEqual(cases[0].window_radius, 1)
        self.assertEqual(cases[0].hash_len, 130)

    def test_default_comparison_cases_match_table_all_l_windowed(self) -> None:
        self.assertEqual(len(COMPARISON_CASES), len(TABLE_ALL_L_WINDOWED_COMPARISON_ROWS))
        for case in COMPARISON_CASES:
            expected = TABLE_ALL_L_WINDOWED_COMPARISON_ROWS[
                (case.case_name, case.security_target, case.max_g_value)
            ]
            self.assertEqual(
                (
                    case.partition_size,
                    case.window_radius,
                    case.block_num,
                    case.hash_len,
                ),
                expected,
            )
            self.assertEqual(case.hash_len % case.max_g_bit, 0)
            self.assertEqual(case.hash_len // case.max_g_bit, case.block_num)

    def test_search_rows_expands_hash_len_cap_when_needed(self) -> None:
        expected = SearchRow(
            case_name="case2",
            security_target=128,
            max_g_value=32,
            partition_num=5,
            window_radius=2,
            block_num=64,
            hash_len=320,
            expected_retries=7.2,
            kappa=151.9,
        )

        def _fake_search_best_row_for_cell(**kwargs):
            hash_len_max = kwargs["hash_len_max"]
            if hash_len_max < 320:
                return None
            return expected

        with mock.patch(
            "search_ycsig_sigsize.search_best_row_for_cell",
            side_effect=_fake_search_best_row_for_cell,
        ) as patched:
            rows = search_rows(
                cases=["case2"],
                security_targets=[128],
                max_g_values=[32],
                objective="partition_num",
                retry_limit=16.0,
                hash_len_max_factor=2.1,
                hash_len_max_absolute=None,
                link_threshold=-1,
                sig_size_samples=0,
                sig_size_seed=0,
                sig_partition_num_slack=None,
            )

        self.assertEqual(rows, [expected])
        self.assertEqual(patched.call_count, 2)

    def test_comparison_text_can_render_partial_case_set(self) -> None:
        with tempfile.NamedTemporaryFile("w+", suffix=".json") as handle:
            json.dump(
                [
                    {
                        "case_name": "case1",
                        "security_target": 128,
                        "max_g_value": 32,
                        "partition_num": 5,
                        "window_radius": 1,
                        "block_num": 41,
                        "hash_len": 205,
                    }
                ],
                handle,
            )
            handle.flush()
            cases = _load_comparison_cases_from_json(handle.name)

        rendered = render_comparison_text(
            comparison_cases=cases,
            ycsig_results={
                32: {
                    "case1_k128": {
                        "keygen": 477.0,
                        "sign": 152.0,
                        "verify": 331.0,
                        "sig_size": 98.0,
                    }
                }
            },
        )

        self.assertIn("case1 | 128 | 32 | 5 | 1 | 205 | 477 | 152 | 331 | 98", rendered)


if __name__ == "__main__":
    unittest.main()
