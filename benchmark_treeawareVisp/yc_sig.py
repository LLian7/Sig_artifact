from __future__ import annotations

import hashlib
from dataclasses import dataclass
from math import ceil
from typing import Any, Mapping, Optional, Sequence, Tuple, Union

from crypto_utils import (
    DEFAULT_HASH_NAME,
    bits_to_bytes,
    derive_parameter,
    hash_bytes,
    normalize_bitstring,
    normalize_to_bytes,
    truncate_to_bits,
)
from keyed_hash import KHParameters, KeyedH
from merkle_tree import (
    MT,
    MTParameters,
    MerkleTree,
    _canonical_cover_positions_for_indices,
    _internal_hash,
)
from operation_counter import enabled as counters_enabled, increment
from pprf import (
    PPRF,
    PPRFComputationCache,
    PPRFKey,
    PPRFParameters,
    StoredSeed,
    _canonical_cover_from_sorted_valid_holes,
    _new_pprf_key_unchecked,
)
from tweakable_hash import THParameters, TwH
from treeaware_isp import (
    TREEAWARE_ISP_KEYEDH_XOF_PREFIX,
    TreeAwareISPParameters,
    treeaware_isp,
)


MessageInput = Union[str, bytes, int]
PublicKeyInput = Union[bytes, MerkleTree]


@dataclass(frozen=True)
class YCSigParameters:
    security_parameter: int
    hash_len: int
    max_g_bit: int
    partition_size: int
    window_radius: int
    link_threshold: int
    pm_TwH: THParameters
    pm_PPRF: PPRFParameters
    pm_KH: KHParameters
    hash_key: bytes
    pm_MT: MTParameters
    pm_ISP: TreeAwareISPParameters
    ADS: bytes
    salt_bytes: int
    max_sign_retries: Optional[int] = 1_000_000

    def __post_init__(self) -> None:
        if self.security_parameter <= 0:
            raise ValueError("security_parameter must be positive")
        if self.hash_len <= 0:
            raise ValueError("hash_len must be positive")
        if self.max_g_bit <= 0:
            raise ValueError("max_g_bit must be positive")
        if self.partition_size <= 0:
            raise ValueError("partition_size must be positive")
        if self.window_radius < 0:
            raise ValueError("window_radius must be non-negative")
        if self.link_threshold < -1:
            raise ValueError("link_threshold must be at least -1")
        if self.hash_len % self.max_g_bit != 0:
            raise ValueError("hash_len must be divisible by max_g_bit")
        if self.salt_bytes <= 0:
            raise ValueError("salt_bytes must be positive")
        if len(self.ADS) == 0:
            raise ValueError("ADS must be non-empty")

    @property
    def block_num(self) -> int:
        return self.hash_len // self.max_g_bit

    @property
    def max_g_value(self) -> int:
        return 1 << self.max_g_bit

    @property
    def leaf_count(self) -> int:
        return self.partition_size * self.max_g_value

    @property
    def leaf_index_bits(self) -> int:
        return max(1, (self.leaf_count - 1).bit_length())

    @property
    def alpha_bytes(self) -> int:
        return bits_to_bytes(self.leaf_index_bits)


@dataclass(frozen=True)
class YCSigSetupResult:
    params: YCSigParameters
    randomness_seed: PPRFKey


@dataclass(frozen=True)
class YCSigKeyPair:
    public_key: bytes
    secret_key: PPRFKey


@dataclass(frozen=True)
class YCSigSignature:
    randomizer: bytes
    salt: bytes
    punctured_seeds: Tuple[bytes, ...]
    partial_state_values: Tuple[bytes, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.randomizer, bytes) or len(self.randomizer) == 0:
            raise ValueError("randomizer must be a non-empty byte string")
        if not isinstance(self.salt, bytes) or len(self.salt) == 0:
            raise ValueError("salt must be a non-empty byte string")
        if any(not isinstance(seed, bytes) for seed in self.punctured_seeds):
            raise ValueError("punctured_seeds must be a tuple of byte strings")
        if any(not isinstance(value, bytes) for value in self.partial_state_values):
            raise ValueError("partial_state_values must be a tuple of byte strings")

    def serialized_size(self) -> int:
        return len(self.serialize())

    def serialize(self) -> bytes:
        return (
            self.randomizer
            + self.salt
            + b"".join(self.punctured_seeds)
            + b"".join(self.partial_state_values)
        )


def _new_ycsig_signature_unchecked(
    *,
    randomizer: bytes,
    salt: bytes,
    punctured_seeds: Tuple[bytes, ...],
    partial_state_values: Tuple[bytes, ...],
) -> YCSigSignature:
    signature = object.__new__(YCSigSignature)
    object.__setattr__(signature, "randomizer", randomizer)
    object.__setattr__(signature, "salt", salt)
    object.__setattr__(signature, "punctured_seeds", punctured_seeds)
    object.__setattr__(signature, "partial_state_values", partial_state_values)
    return signature


class YCSigScheme:
    """
    One-time signature from TreeAwareISP with syntax-aligned interfaces:

      YCSig = (SigSetup, SigGen, SigSign, SigVrfy)
    """

    def __init__(self, pm_YCSig: YCSigParameters) -> None:
        self.params = pm_YCSig
        pm_kh = self.params.pm_KH
        self._isp_hash_name_bytes = self.params.pm_ISP.hash_name.encode("ascii")
        self._xof_seed_prefix = TREEAWARE_ISP_KEYEDH_XOF_PREFIX + self._isp_hash_name_bytes + b"/"
        self._partition_hash_prefix = pm_kh.domain_label + self.params.hash_key
        self._partition_hash_name = pm_kh.hash_name
        self._partition_hash_output_bits = pm_kh.output_bits
        self._partition_hash_output_bytes = bits_to_bytes(pm_kh.output_bits)
        self._partition_hash_backend_counter = f"hash.backend_calls.{pm_kh.hash_name}"
        self._partition_hash_fast_backend = None
        self._partition_hash_fast_uses_xof = False
        if pm_kh.output_bits == 8 * self._partition_hash_output_bytes:
            if pm_kh.hash_name == "shake_128":
                self._partition_hash_fast_backend = hashlib.shake_128
                self._partition_hash_fast_uses_xof = True
            elif pm_kh.hash_name == "shake_256":
                self._partition_hash_fast_backend = hashlib.shake_256
                self._partition_hash_fast_uses_xof = True
            elif pm_kh.hash_name == "sha3_256" and self._partition_hash_output_bytes <= hashlib.sha3_256().digest_size:
                self._partition_hash_fast_backend = hashlib.sha3_256
            elif pm_kh.hash_name == "sha3_512" and self._partition_hash_output_bytes <= hashlib.sha3_512().digest_size:
                self._partition_hash_fast_backend = hashlib.sha3_512
        self._alpha_tweaks = tuple(
            self.params.ADS + alpha.to_bytes(self.params.alpha_bytes, "big")
            for alpha in range(self.params.leaf_count)
        )
        self._ots_hash_output_bytes = bits_to_bytes(self.params.pm_TwH.security_parameter)
        self._ots_hash_backend_counter = f"hash.backend_calls.{self.params.pm_TwH.hash_name}"
        self._ots_hash_fast_backend = None
        self._ots_hash_fast_uses_xof = False
        if self.params.pm_TwH.security_parameter == 8 * self._ots_hash_output_bytes:
            if self.params.pm_TwH.hash_name == "shake_128":
                self._ots_hash_fast_backend = hashlib.shake_128
                self._ots_hash_fast_uses_xof = True
            elif self.params.pm_TwH.hash_name == "shake_256":
                self._ots_hash_fast_backend = hashlib.shake_256
                self._ots_hash_fast_uses_xof = True
            elif (
                self.params.pm_TwH.hash_name == "sha3_256"
                and self._ots_hash_output_bytes <= hashlib.sha3_256().digest_size
            ):
                self._ots_hash_fast_backend = hashlib.sha3_256
            elif (
                self.params.pm_TwH.hash_name == "sha3_512"
                and self._ots_hash_output_bytes <= hashlib.sha3_512().digest_size
            ):
                self._ots_hash_fast_backend = hashlib.sha3_512
        self._alpha_hash_prefixes = tuple(
            self.params.pm_TwH.public_parameter + tweak
            for tweak in self._alpha_tweaks
        )
        self._alpha_bitstrings = tuple(
            format(alpha, f"0{self.params.leaf_index_bits}b")
            for alpha in range(self.params.leaf_count)
        )
        self._group_offsets = tuple(
            group_index * self.params.max_g_value
            for group_index in range(self.params.partition_size)
        )
        self._cache_limit = 512
        self._group_material_cache: dict[
            Tuple[Tuple[int, ...], ...],
            Tuple[Tuple[int, ...], Tuple[str, ...], Tuple[int, ...], Tuple[str, ...]],
        ] = {}
        self._signed_group_material_cache: dict[
            Tuple[Tuple[int, ...], ...],
            Tuple[Tuple[int, ...], Tuple[str, ...]],
        ] = {}
        self._cover_positions_cache: dict[
            Tuple[int, ...],
            Tuple[Tuple[int, int], ...],
        ] = {}
        self._punctured_prefix_cache: dict[Tuple[str, ...], Tuple[str, ...]] = {}
        self._punctured_key_cache: dict[
            Tuple[Tuple[str, ...], Tuple[bytes, ...]],
            PPRFKey,
        ] = {}

    @staticmethod
    def SigSetup(
        security_parameter: int,
        *,
        hash_len: Optional[int] = None,
        max_g_bit: int = 4,
        partition_size: Optional[int] = None,
        aux_t: Optional[Mapping[str, Any]] = None,
        route_policy: Optional[str] = None,
        size_threshold: Optional[int | float] = None,
        vrf_threshold: Optional[int | float] = None,
        mode: str = "legacy",
        aux_mode: Optional[Mapping[str, Any]] = None,
        score_name: Optional[str] = None,
        score_bound: Optional[int | float] = None,
        window_radius: Optional[int] = None,
        window_radius_l: Optional[int] = None,
        window_radius_u: Optional[int] = None,
        link_threshold: Optional[int] = None,
        pattern_family: Optional[Sequence[Sequence[int]]] = None,
        prefix_dict: Optional[Sequence[Sequence[int]]] = None,
        loss_bound: int = 0,
        prefix_limit: int = 0,
        bt_block_size: int = 0,
        bt_families: Optional[Any] = None,
        bt_loss_bound: int = 0,
        shape_parms: Optional[Mapping[str, Any]] = None,
        tree_threshold: Optional[int] = None,
        tweak_hash_name: str = DEFAULT_HASH_NAME,
        keyed_hash_name: str = DEFAULT_HASH_NAME,
        pprf_hash_name: str = DEFAULT_HASH_NAME,
        merkle_hash_name: str = DEFAULT_HASH_NAME,
        tweak_output_bits: Optional[int] = None,
        keyed_hash_key_bits: Optional[int] = None,
        pprf_seed_bits: Optional[int] = None,
        pprf_range_bits: Optional[int] = None,
        ads_bits: Optional[int] = None,
        tweak_public_seed: Optional[bytes] = None,
        tweak_public_parameter: Optional[bytes] = None,
        tweak_public_parameter_bits: Optional[int] = None,
        tweak_public_parameter_bytes: Optional[int] = None,
        merkle_public_seed: Optional[bytes] = None,
        merkle_public_parameter: Optional[bytes] = None,
        merkle_public_parameter_bits: Optional[int] = None,
        merkle_public_parameter_bytes: Optional[int] = None,
        keyed_hash_key_seed: Optional[bytes] = None,
        key_seed: Optional[bytes] = None,
        ads_seed: Optional[bytes] = None,
        salt_bytes: int = 8,
        max_sign_retries: Optional[int] = 1_000_000,
        merkle_padding_leaf: bytes = b"",
        merkle_leaf_tweak_label: bytes = b"leaf/",
        merkle_node_tweak_label: bytes = b"node/",
        merkle_padding_tweak_label: bytes = b"padding/",
        pprf_root_label: bytes = b"GGM/PPRF/root/",
        pprf_expand_left_label: bytes = b"GGM/PPRF/expand/0/",
        pprf_expand_right_label: bytes = b"GGM/PPRF/expand/1/",
        pprf_eval_label: bytes = b"GGM/PPRF/eval/",
        keyed_hash_domain_label: bytes = b"KeyedH/",
    ) -> YCSigSetupResult:
        """
        (pm_YCSig, ks) <- SigSetup(1^kappa)
        """

        resolved_hash_len = security_parameter if hash_len is None else hash_len
        if resolved_hash_len % max_g_bit != 0:
            raise ValueError("hash_len must be divisible by max_g_bit")

        block_num = resolved_hash_len // max_g_bit
        max_g_value = 1 << max_g_bit
        min_partition_size = ceil(block_num / max_g_value)
        resolved_partition_size = (
            min_partition_size if partition_size is None else partition_size
        )
        if not (min_partition_size <= resolved_partition_size <= block_num):
            raise ValueError(
                "partition_size must lie in [ceil(block_num / max_g_value), block_num]"
            )

        if window_radius is None:
            window_radius = block_num // max_g_value
        if window_radius < 0:
            raise ValueError("window_radius must be non-negative")
        if link_threshold is not None and link_threshold < -1:
            raise ValueError("link_threshold must be at least -1")
        # Simplified-windowed VISP keeps the legacy link-threshold knob only as a
        # compatibility alias. It does not affect the accepted output family.
        resolved_link_threshold = -1

        leaf_count = resolved_partition_size * max_g_value
        leaf_index_bits = max(1, (leaf_count - 1).bit_length())

        resolved_tweak_output_bits = (
            security_parameter if tweak_output_bits is None else tweak_output_bits
        )
        resolved_pprf_seed_bits = (
            security_parameter if pprf_seed_bits is None else pprf_seed_bits
        )
        resolved_pprf_range_bits = (
            security_parameter if pprf_range_bits is None else pprf_range_bits
        )
        resolved_ads_bits = security_parameter if ads_bits is None else ads_bits

        pm_TwH = TwH.TweakHSetup(
            resolved_tweak_output_bits,
            hash_name=tweak_hash_name,
            public_seed=tweak_public_seed,
            public_parameter=tweak_public_parameter,
            public_parameter_bits=tweak_public_parameter_bits,
            public_parameter_bytes=tweak_public_parameter_bytes,
        )
        pm_PPRF = PPRF.PRFSetup(
            security_parameter,
            message_length=leaf_index_bits,
            domain_size=leaf_count,
            seed_bits=resolved_pprf_seed_bits,
            range_bits=resolved_pprf_range_bits,
            hash_name=pprf_hash_name,
            root_label=pprf_root_label,
            expand_left_label=pprf_expand_left_label,
            expand_right_label=pprf_expand_right_label,
            eval_label=pprf_eval_label,
        )
        pm_KH = KeyedH.KeyedHSetup(
            security_parameter,
            output_bits=resolved_hash_len,
            key_bits=keyed_hash_key_bits or security_parameter,
            hash_name=keyed_hash_name,
            domain_label=keyed_hash_domain_label,
        )
        hash_key = KeyedH.KeyGen(pm_KH, seed=keyed_hash_key_seed)
        pm_MT = MT.MTSetup(
            security_parameter,
            leaf_count=leaf_count,
            hash_name=merkle_hash_name,
            public_seed=merkle_public_seed,
            public_parameter=merkle_public_parameter,
            public_parameter_bits=merkle_public_parameter_bits,
            public_parameter_bytes=merkle_public_parameter_bytes,
            padding_leaf=merkle_padding_leaf,
            leaf_tweak_label=merkle_leaf_tweak_label,
            node_tweak_label=merkle_node_tweak_label,
            padding_tweak_label=merkle_padding_tweak_label,
        )
        pm_ISP = TreeAwareISPParameters(
            hash_len=resolved_hash_len,
            max_g_bit=max_g_bit,
            partition_num=resolved_partition_size,
            aux_t=aux_t,
            route_policy=route_policy,
            size_threshold=size_threshold,
            vrf_threshold=vrf_threshold,
            mode=mode,
            aux_mode=aux_mode,
            score_name=score_name,
            score_bound=score_bound,
            window_radius=window_radius,
            pattern_family=pattern_family,
            window_radius_l=window_radius_l,
            window_radius_u=window_radius_u,
            prefix_dict=prefix_dict,
            loss_bound=loss_bound,
            prefix_limit=prefix_limit,
            bt_block_size=bt_block_size,
            bt_families=bt_families,
            bt_loss_bound=bt_loss_bound,
            shape_parms=shape_parms,
            tree_threshold=tree_threshold,
            link_threshold=resolved_link_threshold,
            hash_name=keyed_hash_name,
        )
        ADS = derive_parameter(
            b"YCSig/ADS/",
            seed=ads_seed,
            output_bits=resolved_ads_bits,
            hash_name=tweak_hash_name,
        )
        ks = PPRF.PRFKGen(pm_PPRF, seed=key_seed)

        pm_YCSig = YCSigParameters(
            security_parameter=security_parameter,
            hash_len=resolved_hash_len,
            max_g_bit=max_g_bit,
            partition_size=resolved_partition_size,
            window_radius=window_radius,
            link_threshold=resolved_link_threshold,
            pm_TwH=pm_TwH,
            pm_PPRF=pm_PPRF,
            pm_KH=pm_KH,
            hash_key=hash_key,
            pm_MT=pm_MT,
            pm_ISP=pm_ISP,
            ADS=ADS,
            salt_bytes=salt_bytes,
            max_sign_retries=max_sign_retries,
        )
        return YCSigSetupResult(params=pm_YCSig, randomness_seed=ks)

    def SigGen(self, ks: PPRFKey) -> YCSigKeyPair:
        """
        (pk, sk) <- SigGen(ks)
        """

        if ks.params != self.params.pm_PPRF:
            raise ValueError("ks was not generated under the configured PPRF parameters")

        cache = PPRFComputationCache()
        key_nodes = PPRF.LeafMaterialMany(
            ks,
            self._alpha_bitstrings,
            cache=cache,
            inputs_normalized=True,
            inputs_sorted_unique=True,
            inputs_trusted=True,
        )
        present_key_nodes = []
        for alpha, key_node in enumerate(key_nodes):
            if key_node is None:
                raise ValueError("master PPRF key unexpectedly failed to evaluate during SigGen")
            present_key_nodes.append(key_node)
        leaves = self._alpha_leaf_hash_many(range(self.params.leaf_count), present_key_nodes)

        pk_tree = MT.MTBuild(self.params.pm_MT, leaves, leaves_hashed=True)
        return YCSigKeyPair(public_key=pk_tree.root, secret_key=ks)

    def SigSign(self, sk: PPRFKey, message: MessageInput) -> YCSigSignature:
        """
        sigma <- SigSign(sk, m)
        """

        if sk.params != self.params.pm_PPRF:
            raise ValueError("sk was not generated under the configured PPRF parameters")

        randomizer = derive_parameter(
            b"YCSig/R/",
            seed=None,
            output_bits=self.params.security_parameter,
            hash_name=self.params.pm_KH.hash_name,
        )

        salt, groups = self._find_first_valid_partition(message, randomizer)
        return self.SignWithGroups(sk, randomizer, salt, groups)

    def SignWithGroups(
        self,
        sk: PPRFKey,
        randomizer: bytes,
        salt: int,
        groups: Sequence[Sequence[int]],
    ) -> YCSigSignature:
        """
        Core signing routine with the partition already fixed.
        """

        if sk.params != self.params.pm_PPRF:
            raise ValueError("sk was not generated under the configured PPRF parameters")
        if salt < 0 or salt >= (1 << (8 * self.params.salt_bytes)):
            raise ValueError("salt is out of range for the configured salt_bytes")

        (
            partial_state_indices,
            punctured_points,
            _,
            _,
        ) = self._groups_to_signature_material(groups)

        cache = PPRFComputationCache()
        punctured_key, key_nodes = PPRF.PunctureAndRevealLeafMaterialMany(
            sk,
            punctured_points,
            cache=cache,
            inputs_normalized=True,
            inputs_sorted_unique=True,
            inputs_trusted=True,
        )
        present_key_nodes = [b""] * len(partial_state_indices)
        for offset, key_node in enumerate(key_nodes):
            if key_node is None:
                raise ValueError("master key unexpectedly failed during signing")
            present_key_nodes[offset] = key_node
        partial_state_leaf_values = self._alpha_leaf_hash_many(
            partial_state_indices,
            present_key_nodes,
        )

        partial_state_values = self._compact_partial_state_values_from_hashed_leaves(
            partial_state_indices,
            partial_state_leaf_values,
        )
        compact_punctured_seeds = PPRF.CompactPuncturedKey(punctured_key)
        self._cache_store(
            self._punctured_key_cache,
            (punctured_points, compact_punctured_seeds),
            punctured_key,
        )
        return _new_ycsig_signature_unchecked(
            randomizer=randomizer,
            salt=salt.to_bytes(self.params.salt_bytes, "big"),
            punctured_seeds=compact_punctured_seeds,
            partial_state_values=partial_state_values,
        )

    def SigVrfy(
        self,
        pk: PublicKeyInput,
        message: MessageInput,
        signature: YCSigSignature,
    ) -> bool:
        """
        b <- SigVrfy(pk, m, sigma)
        """

        if len(signature.randomizer) != bits_to_bytes(self.params.security_parameter):
            return False
        if len(signature.salt) != self.params.salt_bytes:
            return False
        groups = self._groups_for_salt(
            message,
            signature.randomizer,
            int.from_bytes(signature.salt, "big"),
        )
        if groups is None:
            return False
        return self.VerifyWithGroups(pk, signature, groups)

    def VerifyWithGroups(
        self,
        pk: PublicKeyInput,
        signature: YCSigSignature,
        groups: Sequence[Sequence[int]],
    ) -> bool:
        """
        Core verification routine with the partition already fixed.
        """

        (
            partial_state_indices,
            punctured_points,
            expanded_indices,
            expanded_points,
        ) = self._groups_to_signature_material(groups)
        try:
            punctured_key = self._expand_punctured_key_fast(
                punctured_points,
                signature.punctured_seeds,
            )
        except (TypeError, ValueError):
            return False

        cache = PPRFComputationCache()
        key_nodes = PPRF.LeafMaterialMany(
            punctured_key,
            expanded_points,
            cache=cache,
            inputs_normalized=True,
            inputs_sorted_unique=True,
            inputs_trusted=True,
        )
        present_key_nodes = [b""] * len(expanded_indices)
        for offset, key_node in enumerate(key_nodes):
            if key_node is None:
                return False
            present_key_nodes[offset] = key_node
        expanded_leaf_values = self._alpha_leaf_hash_many(
            expanded_indices,
            present_key_nodes,
        )

        try:
            rebuilt_pk = self._rebuild_root_from_compact_partial_state(
                partial_state_indices,
                signature.partial_state_values,
                expanded_indices,
                expanded_leaf_values,
            )
        except (TypeError, ValueError):
            return False
        return rebuilt_pk == self._normalize_public_key(pk)

    def FindPartition(
        self,
        message: MessageInput,
        randomizer: bytes,
    ) -> Tuple[int, Sequence[Sequence[int]]]:
        """
        Return the first salt and the corresponding accepted TreeAwareISP output.
        """

        return self._find_first_valid_partition(message, randomizer)

    def GroupsToSignedIndices(
        self,
        groups: Sequence[Sequence[int]],
    ) -> Tuple[int, ...]:
        """
        Map the ISP groups to the signed global leaf indices.
        """

        return self._groups_to_signed_indices(groups)

    def _partition_value(self, salt: int, message: MessageInput, randomizer: bytes) -> str:
        digest, _ = self._partition_material(salt, message, randomizer)
        return normalize_bitstring(digest, self.params.hash_len)

    def _partition_material(
        self,
        salt: int,
        message: MessageInput,
        randomizer: bytes,
    ) -> Tuple[bytes, bytes]:
        return self._partition_material_from_message_bytes(
            salt,
            normalize_to_bytes(message),
            randomizer,
        )

    def _partition_material_from_message_bytes(
        self,
        salt: int,
        message_bytes: bytes,
        randomizer: bytes,
    ) -> Tuple[bytes, bytes]:
        counting = counters_enabled()
        if counting:
            increment("keyed_hash.eval")
        digest = self._partition_hash_digest(
            self._partition_hash_prefix
            + salt.to_bytes(self.params.salt_bytes, "big")
            + message_bytes
            + randomizer,
            counting,
        )
        return digest, self._xof_seed_prefix + digest

    def _alpha_tweak(self, alpha: int) -> bytes:
        return self._alpha_tweaks[alpha]

    def _alpha_leaf_hash(self, alpha: int, key_node: bytes) -> bytes:
        counting = counters_enabled()
        if counting:
            increment("ycsig.ots_leaf_hash")
            increment("tweak_hash.eval")
        payload = self._alpha_hash_prefixes[alpha] + key_node
        backend = self._ots_hash_fast_backend
        if backend is None:
            return hash_bytes(
                payload,
                output_bits=self.params.pm_TwH.security_parameter,
                hash_name=self.params.pm_TwH.hash_name,
            )
        if counting:
            increment("hash.backend_calls")
            increment(self._ots_hash_backend_counter)
        hash_object = backend(payload)
        if self._ots_hash_fast_uses_xof:
            return hash_object.digest(self._ots_hash_output_bytes)
        raw = hash_object.digest()[: self._ots_hash_output_bytes]
        if self.params.pm_TwH.security_parameter == 8 * self._ots_hash_output_bytes:
            return raw
        return truncate_to_bits(raw, self.params.pm_TwH.security_parameter)

    def _alpha_leaf_hash_many(
        self,
        indices: Sequence[int],
        key_nodes: Sequence[bytes],
    ) -> list[bytes]:
        if len(indices) != len(key_nodes):
            raise ValueError("indices and key_nodes must have the same length")
        if not indices:
            return []

        counting = counters_enabled()
        if counting:
            amount = float(len(indices))
            increment("ycsig.ots_leaf_hash", amount)
            increment("tweak_hash.eval", amount)

        alpha_hash_prefixes = self._alpha_hash_prefixes
        backend = self._ots_hash_fast_backend
        if backend is None:
            return [
                hash_bytes(
                    alpha_hash_prefixes[alpha] + key_node,
                    output_bits=self.params.pm_TwH.security_parameter,
                    hash_name=self.params.pm_TwH.hash_name,
                )
                for alpha, key_node in zip(indices, key_nodes)
            ]

        if counting:
            amount = float(len(indices))
            increment("hash.backend_calls", amount)
            increment(self._ots_hash_backend_counter, amount)

        uses_xof = self._ots_hash_fast_uses_xof
        output_bytes = self._ots_hash_output_bytes
        security_parameter = self.params.pm_TwH.security_parameter
        if uses_xof:
            return [
                backend(alpha_hash_prefixes[alpha] + key_node).digest(output_bytes)
                for alpha, key_node in zip(indices, key_nodes)
            ]

        if security_parameter == 8 * output_bytes:
            return [
                backend(alpha_hash_prefixes[alpha] + key_node).digest()[:output_bytes]
                for alpha, key_node in zip(indices, key_nodes)
            ]
        return [
            truncate_to_bits(
                backend(alpha_hash_prefixes[alpha] + key_node).digest()[:output_bytes],
                security_parameter,
            )
            for alpha, key_node in zip(indices, key_nodes)
        ]

    def _find_first_valid_partition(
        self,
        message: MessageInput,
        randomizer: bytes,
    ) -> Tuple[int, Sequence[Sequence[int]]]:
        message_bytes = normalize_to_bytes(message)
        message_randomizer = message_bytes + randomizer
        counting = counters_enabled()
        pm_isp = self.params.pm_ISP
        salt_bytes = self.params.salt_bytes
        partition_hash_prefix = self._partition_hash_prefix
        partition_hash_backend = self._partition_hash_fast_backend
        partition_hash_backend_counter = self._partition_hash_backend_counter
        partition_hash_output_bits = self._partition_hash_output_bits
        partition_hash_output_bytes = self._partition_hash_output_bytes
        partition_hash_uses_xof = self._partition_hash_fast_uses_xof
        partition_hash_name = self._partition_hash_name
        xof_seed_prefix = self._xof_seed_prefix
        treeaware = treeaware_isp
        salt = 0
        if partition_hash_backend is None:
            if counting:
                while True:
                    if self.params.max_sign_retries is not None and salt >= self.params.max_sign_retries:
                        raise RuntimeError("TreeAwareISP did not accept within the configured retry budget")
                    increment("ycsig.partition_attempt")
                    increment("keyed_hash.eval")
                    digest = hash_bytes(
                        partition_hash_prefix + salt.to_bytes(salt_bytes, "big") + message_randomizer,
                        output_bits=partition_hash_output_bits,
                        hash_name=partition_hash_name,
                    )
                    groups = treeaware(
                        digest,
                        pm_isp,
                        xof_seed_material=xof_seed_prefix + digest,
                    )
                    if groups is not None:
                        return salt, groups
                    salt += 1
            while True:
                if self.params.max_sign_retries is not None and salt >= self.params.max_sign_retries:
                    raise RuntimeError("TreeAwareISP did not accept within the configured retry budget")
                digest = hash_bytes(
                    partition_hash_prefix + salt.to_bytes(salt_bytes, "big") + message_randomizer,
                    output_bits=partition_hash_output_bits,
                    hash_name=partition_hash_name,
                )
                groups = treeaware(
                    digest,
                    pm_isp,
                    xof_seed_material=xof_seed_prefix + digest,
                )
                if groups is not None:
                    return salt, groups
                salt += 1

        if partition_hash_uses_xof:
            if counting:
                while True:
                    if self.params.max_sign_retries is not None and salt >= self.params.max_sign_retries:
                        raise RuntimeError("TreeAwareISP did not accept within the configured retry budget")
                    increment("ycsig.partition_attempt")
                    increment("keyed_hash.eval")
                    increment("hash.backend_calls")
                    increment(partition_hash_backend_counter)
                    digest = partition_hash_backend(
                        partition_hash_prefix + salt.to_bytes(salt_bytes, "big") + message_randomizer
                    ).digest(partition_hash_output_bytes)
                    groups = treeaware(
                        digest,
                        pm_isp,
                        xof_seed_material=xof_seed_prefix + digest,
                    )
                    if groups is not None:
                        return salt, groups
                    salt += 1
            while True:
                if self.params.max_sign_retries is not None and salt >= self.params.max_sign_retries:
                    raise RuntimeError("TreeAwareISP did not accept within the configured retry budget")
                digest = partition_hash_backend(
                    partition_hash_prefix + salt.to_bytes(salt_bytes, "big") + message_randomizer
                ).digest(partition_hash_output_bytes)
                groups = treeaware(
                    digest,
                    pm_isp,
                    xof_seed_material=xof_seed_prefix + digest,
                )
                if groups is not None:
                    return salt, groups
                salt += 1

        if counting:
            while True:
                if self.params.max_sign_retries is not None and salt >= self.params.max_sign_retries:
                    raise RuntimeError("TreeAwareISP did not accept within the configured retry budget")
                increment("ycsig.partition_attempt")
                increment("keyed_hash.eval")
                increment("hash.backend_calls")
                increment(partition_hash_backend_counter)
                digest = partition_hash_backend(
                    partition_hash_prefix + salt.to_bytes(salt_bytes, "big") + message_randomizer
                ).digest()[:partition_hash_output_bytes]
                groups = treeaware(
                    digest,
                    pm_isp,
                    xof_seed_material=xof_seed_prefix + digest,
                )
                if groups is not None:
                    return salt, groups
                salt += 1
        while True:
            if self.params.max_sign_retries is not None and salt >= self.params.max_sign_retries:
                raise RuntimeError("TreeAwareISP did not accept within the configured retry budget")
            digest = partition_hash_backend(
                partition_hash_prefix + salt.to_bytes(salt_bytes, "big") + message_randomizer
            ).digest()[:partition_hash_output_bytes]
            groups = treeaware(
                digest,
                pm_isp,
                xof_seed_material=xof_seed_prefix + digest,
            )
            if groups is not None:
                return salt, groups
            salt += 1

    def _groups_for_salt(
        self,
        message: MessageInput,
        randomizer: bytes,
        salt: int,
    ) -> Optional[Sequence[Sequence[int]]]:
        return self._groups_for_salt_message_bytes(
            normalize_to_bytes(message),
            randomizer,
            salt,
        )

    def _groups_for_salt_message_bytes(
        self,
        message_bytes: bytes,
        randomizer: bytes,
        salt: int,
    ) -> Optional[Sequence[Sequence[int]]]:
        partition_digest, xof_seed_material = self._partition_material_from_message_bytes(
            salt,
            message_bytes,
            randomizer,
        )
        return treeaware_isp(
            partition_digest,
            self.params.pm_ISP,
            xof_seed_material=xof_seed_material,
        )

    def _groups_to_signed_indices(self, groups: Sequence[Sequence[int]]) -> Tuple[int, ...]:
        signed_indices, _ = self._groups_to_signed_indices_and_points(groups)
        return signed_indices

    def _groups_to_signed_indices_and_points(
        self,
        groups: Sequence[Sequence[int]],
    ) -> Tuple[Tuple[int, ...], Tuple[str, ...]]:
        group_key = tuple(tuple(subgroup) for subgroup in groups)
        cached = self._signed_group_material_cache.get(group_key)
        if cached is not None:
            return cached

        full_cached = self._group_material_cache.get(group_key)
        if full_cached is not None:
            signed_material = (full_cached[0], full_cached[1])
            self._cache_store(self._signed_group_material_cache, group_key, signed_material)
            return signed_material

        signed_count = self.params.block_num
        signed_indices = [0] * signed_count
        signed_points = [""] * signed_count
        signed_offset = 0
        alpha_bitstrings = self._alpha_bitstrings
        group_offsets = self._group_offsets

        for group_index, subgroup in enumerate(groups):
            base = group_offsets[group_index]
            for block_value in subgroup:
                alpha = base + block_value
                signed_indices[signed_offset] = alpha
                signed_points[signed_offset] = alpha_bitstrings[alpha]
                signed_offset += 1

        if signed_offset != signed_count:
            raise ValueError("the number of signed indices does not match block_num")
        signed_material = (tuple(signed_indices), tuple(signed_points))
        self._cache_store(self._signed_group_material_cache, group_key, signed_material)
        return signed_material

    def _groups_to_signed_and_complementary_material(
        self,
        groups: Sequence[Sequence[int]],
    ) -> Tuple[Tuple[int, ...], Tuple[str, ...], Tuple[int, ...], Tuple[str, ...]]:
        return self._group_material(groups)

    def _uses_complement_signing(self) -> bool:
        pm_isp = self.params.pm_ISP
        return (
            pm_isp.mode in {"size", "vrf"}
            or pm_isp.size_threshold is not None
            or pm_isp.vrf_threshold is not None
        )

    def _groups_to_signature_material(
        self,
        groups: Sequence[Sequence[int]],
    ) -> Tuple[Tuple[int, ...], Tuple[str, ...], Tuple[int, ...], Tuple[str, ...]]:
        (
            selected_indices,
            selected_points,
            complementary_indices,
            complementary_points,
        ) = self._group_material(groups)
        if self._uses_complement_signing():
            # Tree-aware YCSig signs the complement so verification expands
            # the selected side.
            return (
                complementary_indices,
                complementary_points,
                selected_indices,
                selected_points,
            )
        return (
            selected_indices,
            selected_points,
            complementary_indices,
            complementary_points,
        )

    def _group_material(
        self,
        groups: Sequence[Sequence[int]],
    ) -> Tuple[Tuple[int, ...], Tuple[str, ...], Tuple[int, ...], Tuple[str, ...]]:
        group_key = tuple(tuple(subgroup) for subgroup in groups)
        cached = self._group_material_cache.get(group_key)
        if cached is not None:
            return cached

        signed_count = self.params.block_num
        complementary_count = self.params.leaf_count - signed_count
        signed_indices = [0] * signed_count
        signed_points = [""] * signed_count
        complementary_indices = [0] * complementary_count
        complementary_points = [""] * complementary_count
        signed_offset = 0
        complementary_offset = 0
        alpha_bitstrings = self._alpha_bitstrings
        group_offsets = self._group_offsets
        max_g_value = self.params.max_g_value

        for group_index, subgroup in enumerate(groups):
            base = group_offsets[group_index]
            next_missing = 0
            for block_value in subgroup:
                for missing_value in range(next_missing, block_value):
                    alpha = base + missing_value
                    complementary_indices[complementary_offset] = alpha
                    complementary_points[complementary_offset] = alpha_bitstrings[alpha]
                    complementary_offset += 1
                alpha = base + block_value
                signed_indices[signed_offset] = alpha
                signed_points[signed_offset] = alpha_bitstrings[alpha]
                signed_offset += 1
                next_missing = block_value + 1
            for missing_value in range(next_missing, max_g_value):
                alpha = base + missing_value
                complementary_indices[complementary_offset] = alpha
                complementary_points[complementary_offset] = alpha_bitstrings[alpha]
                complementary_offset += 1

        if signed_offset != signed_count:
            raise ValueError("the number of signed indices does not match block_num")
        if complementary_offset != complementary_count:
            raise ValueError("the number of complementary indices does not match leaf_count - block_num")
        material = (
            tuple(signed_indices),
            tuple(signed_points),
            tuple(complementary_indices),
            tuple(complementary_points),
        )
        self._cache_store(
            self._signed_group_material_cache,
            group_key,
            (material[0], material[1]),
        )
        self._cache_store(self._group_material_cache, group_key, material)
        return material

    def _canonical_cover_positions_cached(
        self,
        signed_indices: Tuple[int, ...],
    ) -> Tuple[Tuple[int, int], ...]:
        cached = self._cover_positions_cache.get(signed_indices)
        if cached is not None:
            return cached
        positions = _canonical_cover_positions_for_indices(self.params.pm_MT, signed_indices)
        self._cache_store(self._cover_positions_cache, signed_indices, positions)
        return positions

    def _punctured_prefixes_cached(
        self,
        punctured_points: Tuple[str, ...],
    ) -> Tuple[str, ...]:
        cached = self._punctured_prefix_cache.get(punctured_points)
        if cached is not None:
            return cached
        prefixes = _canonical_cover_from_sorted_valid_holes(self.params.pm_PPRF, punctured_points)
        self._cache_store(self._punctured_prefix_cache, punctured_points, prefixes)
        return prefixes

    def _cache_store(self, cache: dict, key, value) -> None:
        if len(cache) >= self._cache_limit:
            cache.clear()
        cache[key] = value

    def _expand_punctured_key_fast(
        self,
        punctured_points: Tuple[str, ...],
        stored_seeds: Sequence[bytes],
    ) -> PPRFKey:
        stored_seed_tuple = tuple(stored_seeds)
        cache_key = (punctured_points, stored_seed_tuple)
        cached = self._punctured_key_cache.get(cache_key)
        if cached is not None:
            return cached

        prefixes = self._punctured_prefixes_cached(punctured_points)
        if len(prefixes) != len(stored_seed_tuple):
            raise ValueError("stored_seeds does not match the canonical frontier size")

        frontier = []
        seed_bytes = self.params.pm_PPRF.seed_bytes
        for prefix, seed in zip(prefixes, stored_seed_tuple):
            if len(seed) != seed_bytes:
                raise ValueError("stored seed has incorrect byte length")
            frontier.append(StoredSeed(prefix=prefix, seed=seed))
        punctured_key = _new_pprf_key_unchecked(self.params.pm_PPRF, frontier)
        self._cache_store(self._punctured_key_cache, cache_key, punctured_key)
        return punctured_key

    def _compact_partial_state_values_from_hashed_leaves(
        self,
        signed_indices: Tuple[int, ...],
        signed_leaf_values: Sequence[bytes],
    ) -> Tuple[bytes, ...]:
        positions = self._canonical_cover_positions_cached(signed_indices)
        leaf_values: list[Optional[bytes]] = [None] * self.params.leaf_count
        for alpha, value in zip(signed_indices, signed_leaf_values):
            leaf_values[alpha] = value

        pm_MT = self.params.pm_MT
        level_widths = pm_MT.level_widths
        node_hash_prefixes = pm_MT._node_hash_prefixes
        backend = pm_MT._fast_backend
        uses_xof = pm_MT._fast_uses_xof
        output_bytes = pm_MT.output_bytes
        security_parameter = pm_MT.security_parameter
        hash_name = pm_MT.hash_name
        backend_counter = pm_MT._backend_counter
        counting = counters_enabled()

        def internal_hash(level: int, offset: int, left: bytes, right: bytes) -> bytes:
            if counting:
                increment("merkle.internal_hash")
                increment("tweak_hash.eval")
            payload = node_hash_prefixes[level][offset] + left + right
            if backend is None:
                return hash_bytes(
                    payload,
                    output_bits=security_parameter,
                    hash_name=hash_name,
                )
            if counting:
                increment("hash.backend_calls")
                increment(backend_counter)
            hash_object = backend(payload)
            if uses_xof:
                return hash_object.digest(output_bytes)
            raw = hash_object.digest()[:output_bytes]
            if security_parameter == 8 * output_bytes:
                return raw
            return truncate_to_bits(raw, security_parameter)

        def build(level: int, offset: int) -> bytes:
            if level == 0:
                leaf_value = leaf_values[offset]
                if leaf_value is None:
                    raise ValueError("missing signed leaf required to build the partial state")
                return leaf_value

            child_offset = offset << 1
            left_value = build(level - 1, child_offset)
            right_offset = child_offset + 1
            if right_offset >= level_widths[level - 1]:
                return left_value
            return internal_hash(level, offset, left_value, build(level - 1, right_offset))

        return tuple(build(level, offset) for level, offset in positions)

    def _rebuild_root_from_compact_partial_state(
        self,
        signed_indices: Tuple[int, ...],
        values: Sequence[bytes],
        complementary_indices: Tuple[int, ...],
        complementary_leaf_values: Sequence[bytes],
    ) -> bytes:
        positions = self._canonical_cover_positions_cached(signed_indices)
        if len(positions) != len(values):
            raise ValueError("values does not match the canonical partial-state size")
        if len(complementary_indices) != len(complementary_leaf_values):
            raise ValueError("complementary indices and leaves do not match")

        known_levels: list[list[Optional[bytes]]] = [
            [None] * width for width in self.params.pm_MT.level_widths
        ]
        leaf_values = known_levels[0]
        for alpha, value in zip(complementary_indices, complementary_leaf_values):
            leaf_values[alpha] = value
        for (level, offset), value in zip(positions, values):
            if known_levels[level][offset] is not None:
                raise ValueError("the same position appears in both partial_state and complementary_leaves")
            known_levels[level][offset] = value

        pm_MT = self.params.pm_MT
        level_widths = pm_MT.level_widths
        node_hash_prefixes = pm_MT._node_hash_prefixes
        backend = pm_MT._fast_backend
        uses_xof = pm_MT._fast_uses_xof
        output_bytes = pm_MT.output_bytes
        security_parameter = pm_MT.security_parameter
        hash_name = pm_MT.hash_name
        backend_counter = pm_MT._backend_counter
        counting = counters_enabled()

        def internal_hash(level: int, offset: int, left: bytes, right: bytes) -> bytes:
            if counting:
                increment("merkle.internal_hash")
                increment("tweak_hash.eval")
            payload = node_hash_prefixes[level][offset] + left + right
            if backend is None:
                return hash_bytes(
                    payload,
                    output_bits=security_parameter,
                    hash_name=hash_name,
                )
            if counting:
                increment("hash.backend_calls")
                increment(backend_counter)
            hash_object = backend(payload)
            if uses_xof:
                return hash_object.digest(output_bytes)
            raw = hash_object.digest()[:output_bytes]
            if security_parameter == 8 * output_bytes:
                return raw
            return truncate_to_bits(raw, security_parameter)

        def build(level: int, offset: int) -> bytes:
            stored = known_levels[level][offset]
            if stored is not None:
                return stored
            if level == 0:
                raise ValueError("insufficient information to rebuild the Merkle root")

            child_offset = offset << 1
            left_value = build(level - 1, child_offset)
            right_offset = child_offset + 1
            if right_offset >= level_widths[level - 1]:
                return left_value
            return internal_hash(level, offset, left_value, build(level - 1, right_offset))

        return build(pm_MT.tree_height, 0)

    def _merkle_internal_hash_at(
        self,
        level: int,
        offset: int,
        left: bytes,
        right: bytes,
    ) -> bytes:
        increment("merkle.internal_hash")
        increment("tweak_hash.eval")
        payload = self.params.pm_MT._node_hash_prefixes[level][offset] + left + right
        backend = self.params.pm_MT._fast_backend
        if backend is None:
            return hash_bytes(
                payload,
                output_bits=self.params.pm_MT.security_parameter,
                hash_name=self.params.pm_MT.hash_name,
            )
        increment("hash.backend_calls")
        increment(self.params.pm_MT._backend_counter)
        hash_object = backend(payload)
        if self.params.pm_MT._fast_uses_xof:
            return hash_object.digest(self.params.pm_MT.output_bytes)
        raw = hash_object.digest()[: self.params.pm_MT.output_bytes]
        if self.params.pm_MT.security_parameter == 8 * self.params.pm_MT.output_bytes:
            return raw
        return truncate_to_bits(raw, self.params.pm_MT.security_parameter)

    def _partition_hash_digest(self, payload: bytes, counting: bool) -> bytes:
        backend = self._partition_hash_fast_backend
        if backend is None:
            return hash_bytes(
                payload,
                output_bits=self._partition_hash_output_bits,
                hash_name=self._partition_hash_name,
            )
        if counting:
            increment("hash.backend_calls")
            increment(self._partition_hash_backend_counter)
        hash_object = backend(payload)
        if self._partition_hash_fast_uses_xof:
            return hash_object.digest(self._partition_hash_output_bytes)
        return hash_object.digest()[: self._partition_hash_output_bytes]

    def _signed_index_bitstrings(
        self,
        signed_indices: Sequence[int],
    ) -> Tuple[str, ...]:
        return tuple(self._alpha_bitstrings[alpha] for alpha in signed_indices)

    @staticmethod
    def _normalize_public_key(pk: PublicKeyInput) -> bytes:
        if isinstance(pk, MerkleTree):
            return pk.root
        if isinstance(pk, bytes):
            return pk
        raise TypeError("pk must be a MerkleTree or root bytes")


YCSig = YCSigScheme
