import json
import math
import tempfile
import unittest
from unittest import mock

from benchmark_ycsig_table_comparison import (
    _load_comparison_cases_from_json,
    render_text as render_comparison_text,
)
from search_ycsig_sigsize import (
    SearchCell,
    SearchRow,
    aligned_pattern_family,
    exact_parameter_metrics,
    exact_tree_parameter_metrics,
    render_latex,
    search_best_row_for_cell,
    search_rows,
)
from treeaware_isp import TreeAwareISPParameters, treeaware_isp


class SearchYCSigSigSizeTests(unittest.TestCase):
    def test_exact_metrics_match_bruteforce_valstrictisp_on_toy_instance(self) -> None:
        hash_len = 6
        max_g_bit = 1
        block_num = hash_len // max_g_bit
        max_g_value = 1 << max_g_bit
        partition_num = 3
        window_radius = 1
        params = TreeAwareISPParameters(
            hash_len=hash_len,
            max_g_bit=max_g_bit,
            partition_num=partition_num,
            window_radius=window_radius,
        )

        outputs = []
        accepted = 0
        space_size = 1 << hash_len
        for partition_value in range(space_size):
            groups = treeaware_isp(partition_value, params)
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

    def test_partition_num_search_matches_bruteforce_on_small_cell(self) -> None:
        cell = SearchCell(case_name="case1", security_target=32, max_g_value=4)
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

    def test_tree_metrics_match_base_metrics_for_all_patterns(self) -> None:
        block_num = 6
        max_g_value = 2
        partition_num = 4
        window_radius = 1
        all_patterns = [[], [0], [1], [0, 1]]

        self.assertEqual(aligned_pattern_family(max_g_value), tuple(tuple(p) for p in all_patterns))

        base_acceptance, base_kappa = exact_parameter_metrics(
            block_num,
            max_g_value,
            partition_num,
            window_radius,
        )
        tree_acceptance, tree_kappa = exact_tree_parameter_metrics(
            block_num,
            max_g_value,
            partition_num,
            window_radius,
            all_patterns,
        )

        self.assertAlmostEqual(tree_acceptance, base_acceptance, places=12)
        self.assertAlmostEqual(tree_kappa, base_kappa, places=12)

    def test_tree_metrics_with_prefix_extraction_match_manual_support_formula(self) -> None:
        hash_len = 6
        max_g_bit = 1
        block_num = hash_len // max_g_bit
        max_g_value = 1 << max_g_bit
        partition_num = 4
        window_radius = 1
        pattern_family = [[], [0], [1], [0, 1]]
        tree_setup_kwargs = {
            "prefix_dict": [[0, 1]],
            "loss_bound": 3,
            "prefix_limit": 1,
        }

        measured_acceptance_probability, measured_kappa = exact_tree_parameter_metrics(
            block_num,
            max_g_value,
            partition_num,
            window_radius,
            pattern_family,
            tree_setup_kwargs=tree_setup_kwargs,
        )
        space_size = max_g_value ** block_num
        expected_acceptance_probability = 0.0
        expected_ucr_probability = 0.0
        for counts in ((2, 4), (3, 3), (4, 2)):
            hist_mass = (
                math.factorial(block_num)
                / (max_g_value ** block_num)
                / math.prod(math.factorial(count) for count in counts)
            )
            support = math.comb(3, counts[0] - 1) * math.comb(3, counts[1] - 1)
            expected_acceptance_probability += hist_mass
            histogram_pair_probability = (
                space_size * hist_mass * hist_mass - hist_mass
            ) / (space_size - 1)
            expected_ucr_probability += histogram_pair_probability / support
        expected_kappa = -math.log2(expected_ucr_probability)

        self.assertAlmostEqual(
            measured_acceptance_probability,
            expected_acceptance_probability,
            places=12,
        )
        self.assertAlmostEqual(measured_kappa, expected_kappa, places=12)

    def test_tree_latex_renders_threshold_and_kappa_columns(self) -> None:
        rows = [
            SearchRow("case1", 8, 4, 2, 0, 4, 8, 10.0, 8.9, pattern_family_size=8, tree_threshold=3),
            SearchRow("case2", 8, 4, 3, 1, 8, 16, 8.0, 9.1, pattern_family_size=8, tree_threshold=4),
        ]

        rendered = render_latex(rows, retry_limit=32, objective="partition_num")

        self.assertIn(r"$\TreeThreshold$", rendered)
        self.assertIn(r"$\kappa_T$", rendered)
        self.assertIn("4  &  2 &  0 &  3", rendered)

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
