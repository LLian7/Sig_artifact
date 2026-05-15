import unittest
from unittest.mock import patch

from benchmark_val_strict_isp_cycles import run_val_strict_isp_cycle_benchmark


class BenchmarkValStrictISPCyclesTests(unittest.TestCase):
    def test_xctrace_backend_honors_repetitions(self) -> None:
        with patch(
            "benchmark_val_strict_isp_cycles._measure_metric_xctrace",
            return_value=100.0,
        ) as measure_metric:
            result = run_val_strict_isp_cycle_benchmark(
                hash_len=8,
                max_g_bit=2,
                partition_num=2,
                window_radius=1,
                hash_name="shake_256",
                accepted_samples=1,
                rejected_samples=1,
                repetitions=3,
                random_seed=7,
                max_candidate_attempts=100,
                cycle_backend="xctrace",
                xctrace_target_operations=5,
                sampler_mode="random",
            )

        self.assertEqual(measure_metric.call_count, 12)
        self.assertEqual(result["cycles"]["full_accept"]["avg_cycles"], 100.0)
        self.assertEqual(result["cycles"]["full_accept"]["stddev_cycles"], 0.0)


if __name__ == "__main__":
    unittest.main()
