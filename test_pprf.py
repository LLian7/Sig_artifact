import unittest

from operation_counter import counting_scope, snapshot
from pprf import PPRF, PPRFComputationCache


class PPRFBatchTests(unittest.TestCase):
    def test_exact_domain_key_covers_only_active_points(self) -> None:
        params = PPRF.PRFSetup(
            128,
            message_length=3,
            domain_size=6,
            hash_name="shake_256",
        )
        key = PPRF.PRFKGen(params, seed=b"pprf-exact-domain")

        self.assertEqual(PPRF.DomainSize(key), 6)
        self.assertEqual(
            PPRF.Dom(key),
            frozenset({"000", "001", "010", "011", "100", "101"}),
        )
        self.assertIsNone(PPRF.PRFEval(key, 6))
        self.assertIsNone(PPRF.PRFEval(key, 7))

    def test_eval_many_matches_pointwise_eval(self) -> None:
        params = PPRF.PRFSetup(
            128,
            message_length=4,
            hash_name="shake_256",
        )
        key = PPRF.PRFKGen(params, seed=b"pprf-batch-seed")
        messages = [0, 1, 3, 7, 8, 15]

        pointwise = tuple(PPRF.PRFEval(key, message) for message in messages)
        batched = PPRF.PRFEvalMany(key, messages, cache=PPRFComputationCache())

        self.assertEqual(batched, pointwise)

    def test_batch_puncture_matches_sequential_puncture(self) -> None:
        params = PPRF.PRFSetup(
            128,
            message_length=4,
            hash_name="shake_256",
        )
        key = PPRF.PRFKGen(params, seed=b"pprf-puncture-seed")
        holes = [1, 3, 8, 10]

        sequential = key
        sequential_outputs = []
        for hole in holes:
            sequential_outputs.append(PPRF.PRFEval(key, hole))
            sequential = PPRF.PPRFPunc(sequential, hole)

        batch_key, batch_outputs = PPRF.PunctureAndEvalMany(
            key,
            holes,
            cache=PPRFComputationCache(),
        )

        self.assertEqual(batch_outputs, tuple(sequential_outputs))
        self.assertEqual(batch_key.frontier, sequential.frontier)

    def test_leaf_material_batch_matches_pointwise_material(self) -> None:
        params = PPRF.PRFSetup(
            128,
            message_length=4,
            hash_name="shake_256",
        )
        key = PPRF.PRFKGen(params, seed=b"pprf-material-seed")
        messages = [0, 2, 5, 11, 15]

        pointwise = tuple(PPRF.LeafMaterialEval(key, message) for message in messages)
        batched = PPRF.LeafMaterialMany(key, messages, cache=PPRFComputationCache())

        self.assertEqual(batched, pointwise)

    def test_single_expand_uses_one_shake_call(self) -> None:
        params = PPRF.PRFSetup(
            128,
            message_length=1,
            hash_name="shake_256",
        )
        key = PPRF.PRFKGen(params, seed=b"pprf-expand-seed")

        with counting_scope():
            values = PPRF.PRFEvalMany(key, [0, 1], cache=PPRFComputationCache())
            counters = snapshot()

        self.assertTrue(all(value is not None for value in values))
        self.assertEqual(counters.get("pprf.expand"), 1.0)
        self.assertEqual(counters.get("pprf.leaf_output"), 2.0)
        self.assertEqual(counters.get("hash.backend_calls"), 3.0)
        self.assertEqual(counters.get("hash.backend_calls.shake_256"), 3.0)

    def test_canonical_prefixes_match_expected_cover(self) -> None:
        params = PPRF.PRFSetup(
            128,
            message_length=4,
            hash_name="shake_256",
        )

        self.assertEqual(
            PPRF.CanonicalPrefixes(params, [1, 3, 8, 10]),
            ("0000", "0010", "01", "1001", "1011", "11"),
        )

    def test_canonical_prefixes_respect_exact_domain(self) -> None:
        params = PPRF.PRFSetup(
            128,
            message_length=3,
            domain_size=6,
            hash_name="shake_256",
        )

        self.assertEqual(
            PPRF.CanonicalPrefixes(params, [1, 3]),
            ("000", "010", "10"),
        )


if __name__ == "__main__":
    unittest.main()
