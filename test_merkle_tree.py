import unittest

from merkle_tree import MT
from tweakable_hash import TwH


class MerkleHashedLeafTests(unittest.TestCase):
    def test_sparse_build_with_prehashed_leaves(self) -> None:
        pm_twh = TwH.TweakHSetup(
            128,
            hash_name="shake_256",
            public_seed=b"twh-seed",
        )
        pm_mt = MT.MTSetup(
            128,
            leaf_count=4,
            hash_name="shake_256",
            public_parameter=pm_twh.public_parameter,
        )

        prehashed_leaves = [
            TwH.TweakHEval(pm_twh, b"ads/" + i.to_bytes(2, "big"), b"leaf-" + bytes([i]))
            for i in range(4)
        ]

        full_tree = MT.MTBuild(pm_mt, prehashed_leaves, leaves_hashed=True)
        partial_state = MT.MTIntNGen(
            pm_mt,
            [1, 3],
            {1: prehashed_leaves[1], 3: prehashed_leaves[3]},
            leaves_hashed=True,
        )
        rebuilt_root = MT.MTSparseBuild(
            pm_mt,
            partial_state,
            {0: prehashed_leaves[0], 2: prehashed_leaves[2]},
            leaves_hashed=True,
        )

        self.assertEqual(rebuilt_root, full_tree.root)

    def test_canonical_state_positions_preserve_odd_tree_promotion(self) -> None:
        pm_mt = MT.MTSetup(
            128,
            leaf_count=5,
            hash_name="shake_256",
            public_seed=b"mt-odd-tree-seed",
        )

        self.assertEqual(MT.CanonicalStatePositions(pm_mt, [4]), ((2, 1),))
        self.assertEqual(MT.CanonicalStatePositions(pm_mt, [0, 4]), ((0, 0), (2, 1)))
        self.assertEqual(
            MT.CanonicalStatePositions(pm_mt, [1, 2, 3, 4]),
            ((0, 1), (1, 1), (2, 1)),
        )


if __name__ == "__main__":
    unittest.main()
