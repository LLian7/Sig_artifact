import unittest

from benchmark_vispt_section import (
    _effective_hash_len_budget,
    _effective_stop_after_candidates,
    _effective_triple_budget,
    _ordered_hash_len_candidates,
    _partition_candidates,
    _search_seed_specs_for_cell,
    _threshold_acceptance_stats_from_scores,
    _window_radius_candidates,
    VISPTMeasuredRow,
    VISPTSearchCell,
    VISPTSearchConfig,
    VISPTTableRowSpec,
    benchmark_case_for_spec,
    render_vispt_section_latex,
    run_vispt_section_benchmarks,
    search_vispt_sections,
    vispt_table_rows,
)


class BenchmarkVISPTSectionTests(unittest.TestCase):
    def test_default_search_config_prefers_fast_feasible_rows(self) -> None:
        config = VISPTSearchConfig()
        self.assertTrue(config.prefer_fast_feasible)
        self.assertEqual(_effective_stop_after_candidates(config), 1)
        self.assertEqual(_effective_hash_len_budget(config, seeded_only=True), 6)
        self.assertEqual(_effective_hash_len_budget(config, seeded_only=False), 3)
        self.assertEqual(_effective_triple_budget(config, seeded_only=True), 24)
        self.assertEqual(_effective_triple_budget(config, seeded_only=False), 10)
        self.assertEqual(config.finalist_count, 1)

    def test_table_rows_include_five_certified_rows(self) -> None:
        rows = vispt_table_rows()
        certified = [row for row in rows if not row.is_pending]
        self.assertEqual(len(certified), 5)

    def test_benchmark_case_for_certified_size_row_sets_vispt_knobs(self) -> None:
        spec = next(
            row
            for row in vispt_table_rows()
            if row.label == "VISPT-case1-Size-w4-k128-P64-R1-H128"
        )
        case = benchmark_case_for_spec(spec, samples=2, random_seed=7)
        self.assertEqual(case.hash_len, 128)
        self.assertEqual(case.partition_size, 64)
        self.assertEqual(case.window_radius, 1)
        self.assertEqual(case.signature_extra_hash_values, 0.0)
        self.assertEqual(case.signature_extra_bits, 0)
        self.assertEqual(case.setup_kwargs["mode"], "size")
        self.assertEqual(case.setup_kwargs["aux_t"]["entropy_floor"], 115)
        self.assertEqual(case.setup_kwargs["size_threshold"], 80)
        self.assertIsNone(case.setup_kwargs["vrf_threshold"])

    def test_dynamic_seed_specs_include_cached_neighbor_rows(self) -> None:
        seed_spec = VISPTTableRowSpec(
            case_name="case1",
            goal="Size",
            security_target=160,
            max_g_value=4,
            partition_num=176,
            window_radius=1,
            entropy_floor=112,
            size_threshold=117,
            vrf_threshold=None,
            vrf_threshold_tex=r"$\infty$",
            hash_len=352,
            expected_retries=178.1,
            kappa=161.3,
        )
        cached = {
            ("case1", "Size", 160, 4): {
                "best": VISPTMeasuredRow(
                    spec=seed_spec,
                    keygen=1.0,
                    sign=1.0,
                    verify=1.0,
                    sig_size=1.0,
                    verify_rate=1.0,
                    raw_result={},
                )
            }
        }
        specs = _search_seed_specs_for_cell(
            VISPTSearchCell("case2", "Size", 160, 4),
            cached,
        )
        self.assertIn(seed_spec, specs)

    def test_ordered_hash_lengths_prioritize_scaled_seed_neighbors(self) -> None:
        seed_spec = VISPTTableRowSpec(
            case_name="case1",
            goal="Size",
            security_target=160,
            max_g_value=4,
            partition_num=176,
            window_radius=1,
            entropy_floor=112,
            size_threshold=117,
            vrf_threshold=None,
            vrf_threshold_tex=r"$\infty$",
            hash_len=352,
            expected_retries=178.1,
            kappa=161.3,
        )
        ordered = _ordered_hash_len_candidates(
            VISPTSearchCell("case1", "Size", 192, 4),
            VISPTSearchConfig(),
            (seed_spec,),
        )
        self.assertIn(424, ordered)
        self.assertLess(ordered.index(424), ordered.index(256))

    def test_threshold_acceptance_stats_reuse_single_score_sample(self) -> None:
        stats = _threshold_acceptance_stats_from_scores(
            [10, 12, 14, 18],
            [11, 14, 20],
            trials=8,
        )
        self.assertEqual(stats[11], (0.125, 10.0))
        self.assertEqual(stats[14], (0.375, 12.0))
        self.assertEqual(stats[20], (0.5, 13.5))

    def test_seedless_size_partition_candidates_prioritize_large_rows(self) -> None:
        ordered = _partition_candidates(
            VISPTSearchCell("case1", "Size", 128, 4),
            64,
            VISPTSearchConfig(),
        )
        self.assertEqual(ordered[:4], (64, 56, 48, 32))

    def test_size_window_radius_candidates_try_radius_one_first(self) -> None:
        ordered = _window_radius_candidates(
            VISPTSearchCell("case1", "Size", 128, 4),
            64,
            VISPTSearchConfig(),
        )
        self.assertEqual(tuple(ordered), (1, 0, 2))

    def test_run_vispt_section_benchmarks_on_single_row(self) -> None:
        spec = VISPTTableRowSpec(
            case_name="case1",
            goal="Size",
            security_target=16,
            max_g_value=4,
            partition_num=4,
            window_radius=1,
            entropy_floor=0,
            size_threshold=16,
            vrf_threshold=None,
            vrf_threshold_tex=r"$\infty$",
            hash_len=8,
            expected_retries=1.0,
            kappa=16.0,
        )
        measured = run_vispt_section_benchmarks(
            row_specs=[spec],
            samples=1,
            repetitions=1,
            random_seed=0,
        )
        result = measured[spec.label]
        self.assertGreater(result.keygen, 0.0)
        self.assertGreater(result.sign, 0.0)
        self.assertGreater(result.verify, 0.0)
        self.assertGreater(result.sig_size, 0.0)
        self.assertEqual(result.verify_rate, 1.0)
        self.assertEqual(
            result.sign,
            result.raw_result["operations"]["sign_core_hash_equivalents_real"] + spec.expected_retries,
        )

    def test_verify_row_uses_unified_profile_mode_with_vrf_threshold(self) -> None:
        spec = VISPTTableRowSpec(
            case_name="case1",
            goal="Verify",
            security_target=16,
            max_g_value=4,
            partition_num=4,
            window_radius=0,
            entropy_floor=None,
            entropy_floor_tex=r"\NA",
            size_threshold=None,
            size_threshold_tex=r"$\infty$",
            vrf_threshold=12,
            hash_len=8,
            expected_retries=4.0,
            kappa=16.0,
        )
        case = benchmark_case_for_spec(spec, samples=1, random_seed=11)
        self.assertEqual(case.setup_kwargs["mode"], "size")
        self.assertIsNone(case.setup_kwargs["size_threshold"])
        self.assertEqual(case.setup_kwargs["vrf_threshold"], 12)

    def test_search_vispt_sections_finds_toy_row(self) -> None:
        seed_spec = VISPTTableRowSpec(
            case_name="case1",
            goal="Size",
            security_target=16,
            max_g_value=4,
            partition_num=4,
            window_radius=1,
            entropy_floor=0,
            size_threshold=16,
            vrf_threshold=None,
            vrf_threshold_tex=r"$\infty$",
            hash_len=8,
            expected_retries=2.0,
            kappa=16.5,
        )
        sections, measured, _resolved = search_vispt_sections(
            config=VISPTSearchConfig(
                retry_limit=8.0,
                retry_slack=8.0,
                pilot_trials=128,
                pilot_accepted_samples=8,
                acceptance_trials=256,
                final_acceptance_trials=512,
                finalist_count=1,
                benchmark_samples=1,
                benchmark_repetitions=1,
                stop_after_candidates=1,
            ),
            random_seed=0,
            search_sections=[[VISPTSearchCell("case1", "Size", 16, 4)]],
            existing_results={
                ("case1", "Size", 16, 4): {
                    "best": VISPTMeasuredRow(
                        spec=seed_spec,
                        keygen=1.0,
                        sign=1.0,
                        verify=1.0,
                        sig_size=1.0,
                        verify_rate=1.0,
                        raw_result={},
                    )
                }
            },
            force_refresh_base_keys={("case1", "Size", 16, 4)},
        )
        self.assertEqual(len(sections), 1)
        self.assertEqual(len(sections[0]), 1)
        spec = sections[0][0]
        self.assertFalse(spec.is_pending)
        self.assertIn(spec.label, measured)

    def test_render_vispt_section_latex_includes_pending_and_measured_values(self) -> None:
        rows = vispt_table_rows()
        measured = {}
        summary_values = {
            "VISPT-case1-Size-w4-k128-P64-R1-H128": (767.0, 609.4, 212.8, 78.3),
            "VISPT-case2-Size-w4-k128-P56-R1-H256": (669.0, 507.8, 374.5, 73.0),
            "VISPT-case2-Size-w4-k128-P64-R1-H256": (767.0, 572.3, 383.0, 70.3),
            "VISPT-case1-Verify-w64-k128-P7-R0-H132": (1341.0, 1437.4, 115.5, 94.8),
            "VISPT-case1-Verify-w256-k128-P5-R0-H128": (3838.0, 3910.5, 104.5, 89.6),
        }
        for row in rows:
            if row.label not in summary_values:
                continue
            keygen, sign, verify, sig_size = summary_values[row.label]
            measured[row.label] = VISPTMeasuredRow(
                spec=row,
                keygen=keygen,
                sign=sign,
                verify=verify,
                sig_size=sig_size,
                verify_rate=1.0,
                raw_result={},
            )

        rendered = render_vispt_section_latex(measured)
        self.assertIn(r"\subsection{Performance Evaluation for $\YCSig$ with $\ValStrictISPT$}", rendered)
        self.assertIn(r"\Pending", rendered)
        self.assertIn("767", rendered)
        self.assertIn("1437.4", rendered)
        self.assertIn("104.5", rendered)


if __name__ == "__main__":
    unittest.main()
