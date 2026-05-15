import unittest

from crypto_utils import bits_to_bytes
from crypto_utils import normalize_bitstring
from merkle_tree import MT
from operation_counter import counting_scope, snapshot
from pprf import PPRF
from val_strict_isp import val_strict_isp
from yc_sig import YCSig


class YCSigTests(unittest.TestCase):
    def test_sigsetup_keeps_window_radius(self) -> None:
        setup = YCSig.SigSetup(
            128,
            hash_len=8,
            max_g_bit=2,
            partition_size=2,
            window_radius=1,
        )

        self.assertEqual(setup.params.window_radius, 1)
        self.assertEqual(setup.params.pm_ISP.window_radius, 1)

    def test_sigsetup_ignores_deprecated_link_threshold(self) -> None:
        setup = YCSig.SigSetup(
            128,
            hash_len=8,
            max_g_bit=2,
            partition_size=2,
            window_radius=1,
            link_threshold=3,
        )

        self.assertEqual(setup.params.link_threshold, -1)
        self.assertEqual(setup.params.pm_ISP.link_threshold, -1)

    def test_sign_and_verify_with_shake(self) -> None:
        setup = YCSig.SigSetup(
            128,
            hash_len=8,
            max_g_bit=2,
            partition_size=2,
            window_radius=1,
            tweak_hash_name="shake_256",
            keyed_hash_name="shake_256",
            pprf_hash_name="shake_256",
            merkle_hash_name="shake_256",
            key_seed=b"k-seed",
            keyed_hash_key_seed=b"hk-seed",
            ads_seed=b"ads-seed",
            tweak_public_seed=b"twh-public",
            merkle_public_seed=b"mt-public",
            max_sign_retries=1000,
        )

        scheme = YCSig(setup.params)
        keypair = scheme.SigGen(setup.randomness_seed)
        signature = scheme.SigSign(keypair.secret_key, b"hello yc sig")

        self.assertTrue(scheme.SigVrfy(keypair.public_key, b"hello yc sig", signature))
        self.assertFalse(scheme.SigVrfy(keypair.public_key, b"hello yc sig!", signature))

    def test_sign_and_verify_with_stream_partition_modes(self) -> None:
        setup = YCSig.SigSetup(
            128,
            hash_len=8,
            max_g_bit=2,
            partition_size=2,
            window_radius=1,
            tweak_hash_name="shake_256",
            keyed_hash_name="shake_256",
            pprf_hash_name="shake_256",
            merkle_hash_name="shake_256",
            partition_retry_mode="stream",
            partition_sampler_mode="stream",
            key_seed=b"k-seed-stream",
            keyed_hash_key_seed=b"hk-seed-stream",
            ads_seed=b"ads-seed-stream",
            tweak_public_seed=b"twh-public-stream",
            merkle_public_seed=b"mt-public-stream",
            max_sign_retries=1000,
        )

        scheme = YCSig(setup.params)
        keypair = scheme.SigGen(setup.randomness_seed)
        signature = scheme.SigSign(keypair.secret_key, b"stream partition")

        self.assertTrue(scheme.SigVrfy(keypair.public_key, b"stream partition", signature))
        self.assertFalse(scheme.SigVrfy(keypair.public_key, b"stream partition!", signature))

    def test_sign_and_verify_with_salted_stream_sampler(self) -> None:
        setup = YCSig.SigSetup(
            128,
            hash_len=8,
            max_g_bit=2,
            partition_size=2,
            window_radius=1,
            tweak_hash_name="shake_256",
            keyed_hash_name="shake_256",
            pprf_hash_name="shake_256",
            merkle_hash_name="shake_256",
            partition_retry_mode="salted",
            partition_sampler_mode="stream",
            key_seed=b"k-seed-inline-stream",
            keyed_hash_key_seed=b"hk-seed-inline-stream",
            ads_seed=b"ads-seed-inline-stream",
            tweak_public_seed=b"twh-public-inline-stream",
            merkle_public_seed=b"mt-public-inline-stream",
            max_sign_retries=1000,
        )

        scheme = YCSig(setup.params)
        keypair = scheme.SigGen(setup.randomness_seed)
        signature = scheme.SigSign(keypair.secret_key, b"inline sampler partition")

        self.assertTrue(scheme.SigVrfy(keypair.public_key, b"inline sampler partition", signature))
        self.assertFalse(scheme.SigVrfy(keypair.public_key, b"inline sampler partition!", signature))

    def test_sign_and_verify_with_sha3(self) -> None:
        setup = YCSig.SigSetup(
            128,
            hash_len=16,
            max_g_bit=4,
            partition_size=2,
            window_radius=0,
            tweak_hash_name="sha3_256",
            keyed_hash_name="sha3_256",
            pprf_hash_name="sha3_256",
            merkle_hash_name="sha3_256",
            key_seed=b"k-seed-2",
            keyed_hash_key_seed=b"hk-seed-2",
            ads_seed=b"ads-seed-2",
            tweak_public_seed=b"twh-public-2",
            merkle_public_seed=b"mt-public-2",
            max_sign_retries=5000,
        )

        scheme = YCSig(setup.params)
        keypair = scheme.SigGen(setup.randomness_seed)
        signature = scheme.SigSign(keypair.secret_key, b"sha3-backed message")

        self.assertTrue(
            scheme.SigVrfy(keypair.public_key, b"sha3-backed message", signature)
        )
        self.assertFalse(
            scheme.SigVrfy(keypair.public_key, b"sha3-backed message!", signature)
        )

    def test_sign_and_verify_with_non_byte_aligned_hash_len(self) -> None:
        setup = YCSig.SigSetup(
            128,
            hash_len=9,
            max_g_bit=3,
            partition_size=2,
            window_radius=0,
            tweak_hash_name="shake_256",
            keyed_hash_name="shake_256",
            pprf_hash_name="shake_256",
            merkle_hash_name="shake_256",
            key_seed=b"k-seed-9",
            keyed_hash_key_seed=b"hk-seed-9",
            ads_seed=b"ads-seed-9",
            tweak_public_seed=b"twh-public-9",
            merkle_public_seed=b"mt-public-9",
            max_sign_retries=1000,
        )

        scheme = YCSig(setup.params)
        keypair = scheme.SigGen(setup.randomness_seed)
        signature = scheme.SigSign(keypair.secret_key, b"9-bit hash len")

        self.assertTrue(scheme.SigVrfy(keypair.public_key, b"9-bit hash len", signature))
        self.assertFalse(scheme.SigVrfy(keypair.public_key, b"9-bit hash len!", signature))

    def test_default_output_lengths_follow_security_parameter(self) -> None:
        setup = YCSig.SigSetup(
            128,
            hash_len=256,
            max_g_bit=4,
            partition_size=9,
            window_radius=3,
            tweak_hash_name="shake_256",
            keyed_hash_name="shake_256",
            pprf_hash_name="shake_256",
            merkle_hash_name="shake_256",
            key_seed=b"k-seed-lengths",
            keyed_hash_key_seed=b"hk-seed-lengths",
            ads_seed=b"ads-seed-lengths",
            tweak_public_seed=b"twh-public-lengths",
            merkle_public_seed=b"mt-public-lengths",
            max_sign_retries=1000,
        )

        scheme = YCSig(setup.params)
        keypair = scheme.SigGen(setup.randomness_seed)
        signature = scheme.SigSign(keypair.secret_key, b"length-accounting")

        self.assertEqual(setup.params.pm_TwH.security_parameter, setup.params.security_parameter)
        self.assertEqual(setup.params.pm_PPRF.range_bits, setup.params.security_parameter)
        self.assertEqual(setup.params.pm_PPRF.domain_size, setup.params.leaf_count)
        self.assertEqual(
            len(signature.randomizer),
            bits_to_bytes(setup.params.security_parameter),
        )

    def test_partition_digest_bytes_match_bitstring_path(self) -> None:
        setup = YCSig.SigSetup(
            128,
            hash_len=9,
            max_g_bit=3,
            partition_size=2,
            window_radius=0,
            keyed_hash_name="shake_256",
        )

        scheme = YCSig(setup.params)
        digest, xof_seed_material = scheme._partition_material(0, b"digest path", b"\x23\x45")

        self.assertEqual(
            val_strict_isp(digest, setup.params.pm_ISP, xof_seed_material=xof_seed_material),
            val_strict_isp(
                normalize_bitstring(digest, setup.params.hash_len),
                setup.params.pm_ISP,
                xof_seed_material=xof_seed_material,
            ),
        )

    def test_signature_omits_derivable_structure(self) -> None:
        setup = YCSig.SigSetup(
            128,
            hash_len=8,
            max_g_bit=2,
            partition_size=2,
            window_radius=1,
            tweak_hash_name="shake_256",
            keyed_hash_name="shake_256",
            pprf_hash_name="shake_256",
            merkle_hash_name="shake_256",
            key_seed=b"k-seed-compact",
            keyed_hash_key_seed=b"hk-seed-compact",
            ads_seed=b"ads-seed-compact",
            tweak_public_seed=b"twh-public-compact",
            merkle_public_seed=b"mt-public-compact",
            max_sign_retries=1000,
        )

        scheme = YCSig(setup.params)
        keypair = scheme.SigGen(setup.randomness_seed)
        signature = scheme.SigSign(keypair.secret_key, b"compact signature")

        self.assertEqual(len(signature.salt), setup.params.salt_bytes)
        self.assertFalse(hasattr(signature, "punctured_key"))
        self.assertFalse(hasattr(signature, "partial_state"))

        groups = scheme._groups_for_salt(
            b"compact signature",
            signature.randomizer,
            int.from_bytes(signature.salt, "big"),
        )
        self.assertIsNotNone(groups)
        selected_indices = scheme._groups_to_signed_indices(groups)
        selected_points = scheme._signed_index_bitstrings(selected_indices)

        expected_prefixes = PPRF.CanonicalPrefixes(setup.params.pm_PPRF, selected_points)
        expected_positions = MT.CanonicalStatePositions(setup.params.pm_MT, selected_indices)

        self.assertEqual(len(selected_indices), setup.params.block_num)
        self.assertEqual(len(signature.punctured_seeds), len(expected_prefixes))
        self.assertEqual(len(signature.partial_state_values), len(expected_positions))
        punctured_key = scheme._expand_punctured_key_fast(
            selected_points,
            signature.punctured_seeds,
        )
        self.assertTrue(
            all(
                PPRF.LeafMaterialEval(punctured_key, alpha) is None
                for alpha in selected_indices
            )
        )
        self.assertEqual(
            signature.serialized_size(),
            len(signature.randomizer)
            + len(signature.salt)
            + sum(len(seed) for seed in signature.punctured_seeds)
            + sum(len(value) for value in signature.partial_state_values),
        )

    def test_integer_pprf_fast_paths_match_reference_paths(self) -> None:
        setup = YCSig.SigSetup(
            128,
            hash_len=64,
            max_g_bit=4,
            partition_size=5,
            window_radius=3,
            tweak_hash_name="shake_256",
            keyed_hash_name="shake_256",
            pprf_hash_name="shake_256",
            merkle_hash_name="shake_256",
            key_seed=b"k-seed-fast-path-reference",
            keyed_hash_key_seed=b"hk-seed-fast-path-reference",
            ads_seed=b"ads-seed-fast-path-reference",
            tweak_public_seed=b"twh-public-fast-path-reference",
            merkle_public_seed=b"mt-public-fast-path-reference",
            max_sign_retries=5000,
        )

        scheme = YCSig(setup.params)
        keypair = scheme.SigGen(setup.randomness_seed)
        groups = [
            [0, 1, 2, 3],
            [4, 5, 6],
            [7, 8, 9],
            [10, 11, 12],
            [13, 14, 15],
        ]
        (
            selected_indices,
            selected_points,
            complement_indices,
            complement_points,
        ) = scheme._groups_to_signed_and_complementary_material(groups)

        fast_seeds, fast_ranges, fast_selected_nodes = scheme._puncture_and_reveal_indices_fast(
            keypair.secret_key,
            selected_indices,
        )
        reference_key, reference_selected_nodes = PPRF.PunctureAndRevealLeafMaterialMany(
            keypair.secret_key,
            selected_points,
            inputs_normalized=True,
            inputs_sorted_unique=True,
            inputs_trusted=True,
        )

        self.assertEqual(fast_seeds, PPRF.CompactPuncturedKey(reference_key))
        self.assertEqual(fast_selected_nodes, reference_selected_nodes)
        self.assertEqual(len(fast_ranges), len(fast_seeds))

        fast_complement_nodes = scheme._leaf_material_from_punctured_indices_fast(
            fast_seeds,
            selected_indices,
            complement_indices,
        )
        reference_complement_nodes = PPRF.LeafMaterialMany(
            reference_key,
            complement_points,
            inputs_normalized=True,
            inputs_sorted_unique=True,
            inputs_trusted=True,
        )
        self.assertEqual(fast_complement_nodes, reference_complement_nodes)

    def test_verify_uses_signature_salt_without_retry(self) -> None:
        setup = YCSig.SigSetup(
            128,
            hash_len=8,
            max_g_bit=2,
            partition_size=2,
            window_radius=1,
            tweak_hash_name="shake_256",
            keyed_hash_name="shake_256",
            pprf_hash_name="shake_256",
            merkle_hash_name="shake_256",
            key_seed=b"k-seed-verify",
            keyed_hash_key_seed=b"hk-seed-verify",
            ads_seed=b"ads-seed-verify",
            tweak_public_seed=b"twh-public-verify",
            merkle_public_seed=b"mt-public-verify",
            max_sign_retries=1000,
        )

        scheme = YCSig(setup.params)
        keypair = scheme.SigGen(setup.randomness_seed)
        signature = scheme.SigSign(keypair.secret_key, b"verify once")

        with counting_scope():
            valid = scheme.SigVrfy(keypair.public_key, b"verify once", signature)
            counters = snapshot()

        self.assertTrue(valid)
        self.assertEqual(counters.get("ycsig.partition_attempt", 0.0), 0.0)
        self.assertGreater(counters.get("keyed_hash.eval", 0.0), 0.0)

    def test_find_partition_reuses_partition_digest_for_sampling_seed(self) -> None:
        setup = YCSig.SigSetup(
            128,
            hash_len=8,
            max_g_bit=2,
            partition_size=2,
            window_radius=1,
            tweak_hash_name="shake_256",
            keyed_hash_name="shake_256",
            pprf_hash_name="shake_256",
            merkle_hash_name="shake_256",
            key_seed=b"k-seed-retry",
            keyed_hash_key_seed=b"hk-seed-retry",
            ads_seed=b"ads-seed-retry",
            tweak_public_seed=b"twh-public-retry",
            merkle_public_seed=b"mt-public-retry",
            max_sign_retries=1000,
        )

        scheme = YCSig(setup.params)

        with counting_scope():
            _, groups = scheme.FindPartition(b"retry path", b"\x12")
            counters = snapshot()

        self.assertIsNotNone(groups)
        self.assertNotIn("isp.sample_seed_hash", counters)
        self.assertGreater(counters.get("keyed_hash.eval", 0.0), 0.0)
        self.assertEqual(counters.get("isp.xof_instances", 0.0), 1.0)
        self.assertGreater(counters.get("isp.xof_output_bits", 0.0), 0.0)

    def test_stream_partition_modes_use_one_partition_xof(self) -> None:
        setup = YCSig.SigSetup(
            128,
            hash_len=8,
            max_g_bit=2,
            partition_size=2,
            window_radius=1,
            tweak_hash_name="shake_256",
            keyed_hash_name="shake_256",
            pprf_hash_name="shake_256",
            merkle_hash_name="shake_256",
            partition_retry_mode="stream",
            partition_sampler_mode="stream",
            key_seed=b"k-seed-stream-counters",
            keyed_hash_key_seed=b"hk-seed-stream-counters",
            ads_seed=b"ads-seed-stream-counters",
            tweak_public_seed=b"twh-public-stream-counters",
            merkle_public_seed=b"mt-public-stream-counters",
            max_sign_retries=1000,
        )

        scheme = YCSig(setup.params)

        with counting_scope():
            _, groups = scheme.FindPartition(b"stream counter path", b"\x12")
            counters = snapshot()

        self.assertIsNotNone(groups)
        self.assertGreater(counters.get("ycsig.partition_attempt", 0.0), 0.0)
        self.assertEqual(counters.get("keyed_hash.eval", 0.0), 1.0)
        self.assertEqual(counters.get("hash.backend_calls.shake_256", 0.0), 1.0)
        self.assertNotIn("isp.sample_seed_hash", counters)
        self.assertNotIn("isp.xof_instances", counters)

    def test_salted_stream_sampler_uses_one_xof_per_attempt(self) -> None:
        setup = YCSig.SigSetup(
            128,
            hash_len=8,
            max_g_bit=2,
            partition_size=2,
            window_radius=1,
            tweak_hash_name="shake_256",
            keyed_hash_name="shake_256",
            pprf_hash_name="shake_256",
            merkle_hash_name="shake_256",
            partition_retry_mode="salted",
            partition_sampler_mode="stream",
            key_seed=b"k-seed-salted-stream-counters",
            keyed_hash_key_seed=b"hk-seed-salted-stream-counters",
            ads_seed=b"ads-seed-salted-stream-counters",
            tweak_public_seed=b"twh-public-salted-stream-counters",
            merkle_public_seed=b"mt-public-salted-stream-counters",
            max_sign_retries=1000,
        )

        scheme = YCSig(setup.params)

        with counting_scope():
            _, groups = scheme.FindPartition(b"salted stream counter path", b"\x12")
            counters = snapshot()

        self.assertIsNotNone(groups)
        attempts = counters.get("ycsig.partition_attempt", 0.0)
        self.assertGreater(attempts, 0.0)
        self.assertEqual(counters.get("keyed_hash.eval", 0.0), attempts)
        self.assertEqual(counters.get("hash.backend_calls.shake_256", 0.0), attempts)
        self.assertNotIn("isp.sample_seed_hash", counters)
        self.assertNotIn("isp.xof_instances", counters)

    def test_stream_sampler_too_small_raises_instead_of_hash_fallback(self) -> None:
        setup = YCSig.SigSetup(
            128,
            hash_len=128,
            max_g_bit=2,
            partition_size=32,
            window_radius=2,
            tweak_hash_name="shake_256",
            keyed_hash_name="shake_256",
            pprf_hash_name="shake_256",
            merkle_hash_name="shake_256",
            partition_retry_mode="stream",
            partition_sampler_mode="stream",
            partition_stream_sampler_bytes=1,
            key_seed=b"k-seed-too-small-stream",
            keyed_hash_key_seed=b"hk-seed-too-small-stream",
            ads_seed=b"ads-seed-too-small-stream",
            tweak_public_seed=b"twh-public-too-small-stream",
            merkle_public_seed=b"mt-public-too-small-stream",
            max_sign_retries=1000,
        )

        scheme = YCSig(setup.params)

        with self.assertRaisesRegex(ValueError, "partition_stream_sampler_bytes is too small"):
            scheme.FindPartition(b"too small sampler", b"\x12")

    def test_stream_sampler_auto_default_scales_for_w8(self) -> None:
        setup = YCSig.SigSetup(
            128,
            hash_len=192,
            max_g_bit=3,
            partition_size=16,
            window_radius=2,
            tweak_hash_name="shake_256",
            keyed_hash_name="shake_256",
            pprf_hash_name="shake_256",
            merkle_hash_name="shake_256",
            partition_retry_mode="stream",
            partition_sampler_mode="stream",
            key_seed=b"k-seed-auto-stream-w8",
            keyed_hash_key_seed=b"hk-seed-auto-stream-w8",
            ads_seed=b"ads-seed-auto-stream-w8",
            tweak_public_seed=b"twh-public-auto-stream-w8",
            merkle_public_seed=b"mt-public-auto-stream-w8",
            max_sign_retries=1000,
        )

        self.assertEqual(setup.params.partition_stream_sampler_bytes, 64)
        scheme = YCSig(setup.params)
        salt, groups = scheme.FindPartition(b"auto sampler bytes", b"\x12")
        self.assertGreaterEqual(salt, 0)
        self.assertIsNotNone(groups)


if __name__ == "__main__":
    unittest.main()
