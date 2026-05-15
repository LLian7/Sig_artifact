from __future__ import annotations

import hashlib
from bisect import bisect_left
from dataclasses import dataclass
from math import ceil
from typing import Optional, Sequence, Tuple, Union

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
    _active_domain_prefixes,
    _canonical_cover_from_sorted_valid_holes,
    _new_pprf_key_unchecked,
    _new_stored_seed_unchecked,
)
from tweakable_hash import THParameters, TwH
from val_strict_isp import (
    ISPParameters,
    _NATIVE_BYTES_PREFIXED_SEED,
    _NATIVE_FIND_FIRST_RANDOM_STREAM,
    _NATIVE_MAX_G_VALUE,
    _NATIVE_MAX_MAX_G_BIT,
    _NATIVE_MAX_PARTITION_NUM,
    _NATIVE_MIN_MAX_G_BIT,
    _NATIVE_W4_BYTES_PREFIXED_SEED,
    _NATIVE_W4_MAX_PARTITION_NUM,
    _val_strict_isp_with_random_bytes,
    _val_strict_isp_with_seed_prefix,
    recommended_stream_sampler_bytes,
    val_strict_isp,
)

try:
    import _yc_sig_native
except ImportError:
    _yc_sig_native = None
_NATIVE_PPRF_PUNCTURE_AND_REVEAL = (
    getattr(_yc_sig_native, "pprf_puncture_and_reveal", None)
    if _yc_sig_native is not None
    else None
)
_NATIVE_PPRF_LEAF_MATERIAL = (
    getattr(_yc_sig_native, "pprf_leaf_material", None)
    if _yc_sig_native is not None
    else None
)
_NATIVE_PPRF_LEAF_MATERIAL_DENSE = (
    getattr(_yc_sig_native, "pprf_leaf_material_dense", None)
    if _yc_sig_native is not None
    else None
)
_NATIVE_MERKLE_SPARSE_REBUILD = (
    getattr(_yc_sig_native, "merkle_sparse_rebuild", None)
    if _yc_sig_native is not None
    else None
)
_NATIVE_MERKLE_ROOT_FROM_LEAVES = (
    getattr(_yc_sig_native, "merkle_root_from_leaves", None)
    if _yc_sig_native is not None
    else None
)
_NATIVE_MERKLE_COMPACT_PARTIAL_STATE = (
    getattr(_yc_sig_native, "merkle_compact_partial_state", None)
    if _yc_sig_native is not None
    else None
)
_NATIVE_CANONICAL_COVER_POSITIONS = (
    getattr(_yc_sig_native, "canonical_cover_positions", None)
    if _yc_sig_native is not None
    else None
)


MessageInput = Union[str, bytes, int]
PublicKeyInput = Union[bytes, MerkleTree]
GroupInput = Sequence[Sequence[int]] | bytes | tuple[int, ...]
PARTITION_RETRY_MODES = {"salted", "stream"}
PARTITION_SAMPLER_MODES = {"seeded", "stream"}


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
    pm_ISP: ISPParameters
    ADS: bytes
    salt_bytes: int
    max_sign_retries: Optional[int] = 1_000_000
    partition_retry_mode: str = "salted"
    partition_sampler_mode: str = "seeded"
    partition_stream_sampler_bytes: int = 32

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
        if self.partition_retry_mode not in PARTITION_RETRY_MODES:
            raise ValueError(
                f"partition_retry_mode must be one of {sorted(PARTITION_RETRY_MODES)}"
            )
        if self.partition_sampler_mode not in PARTITION_SAMPLER_MODES:
            raise ValueError(
                f"partition_sampler_mode must be one of {sorted(PARTITION_SAMPLER_MODES)}"
            )
        if (
            (self.partition_retry_mode == "stream" or self.partition_sampler_mode == "stream")
            and self.pm_KH.hash_name not in {"shake_128", "shake_256"}
        ):
            raise ValueError("stream partition modes require a SHAKE keyed hash")
        if self.partition_stream_sampler_bytes <= 0:
            raise ValueError("partition_stream_sampler_bytes must be positive")

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
    One-time signature from ValStrictISP with syntax-aligned interfaces:

      YCSig = (SigSetup, SigGen, SigSign, SigVrfy)
    """

    def __init__(self, pm_YCSig: YCSigParameters) -> None:
        self.params = pm_YCSig
        pm_kh = self.params.pm_KH
        self._isp_hash_name_bytes = self.params.pm_ISP.hash_name.encode("ascii")
        self._xof_seed_prefix = b"ValStrictISP/SamplePosition/KeyedH/" + self._isp_hash_name_bytes + b"/"
        self._partition_hash_prefix = pm_kh.domain_label + self.params.hash_key
        self._partition_stream_prefix = (
            self._partition_hash_prefix
            + b"YCSig/PartitionStream/"
            + self._isp_hash_name_bytes
            + b"/"
        )
        self._partition_hash_name = pm_kh.hash_name
        self._partition_hash_output_bits = pm_kh.output_bits
        self._partition_hash_output_bytes = bits_to_bytes(pm_kh.output_bits)
        self._partition_stream_chunk_bytes = self._partition_hash_output_bytes
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
        self._alpha_hash_prefix_states = None
        if self._ots_hash_fast_backend is not None and self._ots_hash_fast_uses_xof:
            self._alpha_hash_prefix_states = tuple(
                self._ots_hash_fast_backend(prefix)
                for prefix in self._alpha_hash_prefixes
            )
        self._alpha_bitstrings = tuple(
            format(alpha, f"0{self.params.leaf_index_bits}b")
            for alpha in range(self.params.leaf_count)
        )
        self._all_alpha_indices = range(self.params.leaf_count)
        self._group_offsets = tuple(
            group_index * self.params.max_g_value
            for group_index in range(self.params.partition_size)
        )
        self._group_mask_material_templates = None
        if self.params.max_g_value == 4:
            alpha_bitstrings = self._alpha_bitstrings
            group_mask_material_templates = []
            for base in self._group_offsets:
                mask_templates = []
                for mask in range(1 << self.params.max_g_value):
                    signed_indices = tuple(
                        base + block_value
                        for block_value in range(self.params.max_g_value)
                        if mask & (1 << block_value)
                    )
                    complementary_indices = tuple(
                        base + block_value
                        for block_value in range(self.params.max_g_value)
                        if not (mask & (1 << block_value))
                    )
                    mask_templates.append(
                        (
                            signed_indices,
                            tuple(alpha_bitstrings[alpha] for alpha in signed_indices),
                            complementary_indices,
                            tuple(alpha_bitstrings[alpha] for alpha in complementary_indices),
                        )
                    )
                group_mask_material_templates.append(tuple(mask_templates))
            self._group_mask_material_templates = tuple(group_mask_material_templates)
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
        self._punctured_range_cache: dict[
            Tuple[int, ...],
            Tuple[Tuple[int, int, int], ...],
        ] = {}
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
        window_radius: Optional[int] = None,
        link_threshold: Optional[int] = None,
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
        partition_retry_mode: str = "salted",
        partition_sampler_mode: str = "seeded",
        partition_stream_sampler_bytes: Optional[int] = None,
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
        pm_ISP = ISPParameters(
            hash_len=resolved_hash_len,
            max_g_bit=max_g_bit,
            partition_num=resolved_partition_size,
            window_radius=window_radius,
            link_threshold=resolved_link_threshold,
            hash_name=keyed_hash_name,
        )
        if partition_stream_sampler_bytes is None:
            resolved_partition_stream_sampler_bytes = (
                recommended_stream_sampler_bytes(pm_ISP)
                if partition_sampler_mode == "stream"
                else 32
            )
        else:
            resolved_partition_stream_sampler_bytes = partition_stream_sampler_bytes
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
            partition_retry_mode=partition_retry_mode,
            partition_sampler_mode=partition_sampler_mode,
            partition_stream_sampler_bytes=resolved_partition_stream_sampler_bytes,
        )
        return YCSigSetupResult(params=pm_YCSig, randomness_seed=ks)

    def SigGen(self, ks: PPRFKey) -> YCSigKeyPair:
        """
        (pk, sk) <- SigGen(ks)
        """

        if ks.params != self.params.pm_PPRF:
            raise ValueError("ks was not generated under the configured PPRF parameters")

        params = ks.params
        pprf_native_ok = (
            not counters_enabled()
            and params._expand_fast_backend is not None
            and params._expand_fast_uses_xof
            and params.seed_bits == 8 * params._seed_bytes
            and params.message_length <= 63
            and params.hash_name in {"shake_128", "shake_256"}
        )
        provider_ranges = []
        ranges_cover_active_domain = False
        if pprf_native_ok:
            ranges_cover_active_domain = True
            for provider in ks.frontier:
                prefix = provider.prefix
                depth = len(prefix)
                span = 1 << (params.message_length - depth)
                low = (int(prefix, 2) if prefix else 0) * span
                high = low + span
                if high > params.domain_size:
                    ranges_cover_active_domain = False
                    break
                provider_ranges.append((depth, low, high, provider.seed))

        native_dense = _NATIVE_PPRF_LEAF_MATERIAL_DENSE
        present_key_nodes: Sequence[bytes]
        if (
            native_dense is not None
            and pprf_native_ok
            and ranges_cover_active_domain
        ):
            present_key_nodes = native_dense(
                tuple(provider_ranges),
                params._expand_prefix,
                params._seed_bytes,
                params.message_length,
                params.domain_size,
                params.hash_name == "shake_128",
            )
        else:
            present_key_nodes = ()

        if not present_key_nodes:
            cache = PPRFComputationCache()
            key_nodes = PPRF.LeafMaterialMany(
                ks,
                self._alpha_bitstrings,
                cache=cache,
                inputs_normalized=True,
                inputs_sorted_unique=True,
                inputs_trusted=True,
            )
            resolved_key_nodes = []
            for alpha, key_node in enumerate(key_nodes):
                if key_node is None:
                    raise ValueError("master PPRF key unexpectedly failed to evaluate during SigGen")
                resolved_key_nodes.append(key_node)
            present_key_nodes = resolved_key_nodes
        leaves = self._alpha_leaf_hash_many(self._all_alpha_indices, present_key_nodes)

        public_key = self._merkle_root_from_hashed_leaves(leaves)
        return YCSigKeyPair(public_key=public_key, secret_key=ks)

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
        groups: GroupInput,
    ) -> YCSigSignature:
        """
        Core signing routine with the partition already fixed.
        """

        if sk.params != self.params.pm_PPRF:
            raise ValueError("sk was not generated under the configured PPRF parameters")
        if salt < 0 or salt >= (1 << (8 * self.params.salt_bytes)):
            raise ValueError("salt is out of range for the configured salt_bytes")

        (
            selected_indices,
            selected_points,
            complement_indices,
            complement_points,
        ) = self._groups_to_signed_and_complementary_material(groups)
        if len(selected_indices) != self.params.block_num:
            raise ValueError("the number of signed indices does not match block_num")

        compact_punctured_seeds, punctured_ranges, key_nodes = self._puncture_and_reveal_indices_fast(
            sk,
            selected_indices,
        )
        if None in key_nodes:
            raise ValueError("master key unexpectedly failed during signing")
        present_key_nodes = key_nodes
        selected_leaf_values = self._alpha_leaf_hash_many(selected_indices, present_key_nodes)

        partial_state_values = self._compact_partial_state_values_from_hashed_leaves(
            selected_indices,
            selected_leaf_values,
        )
        self._cache_store(self._punctured_range_cache, selected_indices, punctured_ranges)
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
        groups: GroupInput,
    ) -> bool:
        """
        Core verification routine with the partition already fixed.
        """

        (
            selected_indices,
            selected_points,
            complement_indices,
            complement_points,
        ) = self._groups_to_signed_and_complementary_material(groups)
        try:
            key_nodes = self._leaf_material_from_punctured_indices_fast(
                signature.punctured_seeds,
                selected_indices,
                complement_indices,
            )
        except (TypeError, ValueError):
            return False
        if None in key_nodes:
            return False
        present_key_nodes = key_nodes
        complement_leaf_values = self._alpha_leaf_hash_many(
            complement_indices,
            present_key_nodes,
        )

        try:
            rebuilt_pk = self._rebuild_root_from_compact_partial_state(
                selected_indices,
                signature.partial_state_values,
                complement_indices,
                complement_leaf_values,
            )
        except (TypeError, ValueError):
            return False
        return rebuilt_pk == self._normalize_public_key(pk)

    def FindPartition(
        self,
        message: MessageInput,
        randomizer: bytes,
    ) -> Tuple[int, GroupInput]:
        """
        Return the first salt and the corresponding accepted ValStrictISP output.
        """

        return self._find_first_valid_partition(message, randomizer)

    def GroupsToSignedIndices(
        self,
        groups: GroupInput,
    ) -> Tuple[int, ...]:
        """
        Map the ISP groups to the signed global leaf indices.
        """

        return self._groups_to_signed_indices(groups)

    def _partition_value(self, salt: int, message: MessageInput, randomizer: bytes) -> str:
        digest = self._partition_digest_for_salt_message_bytes(
            salt,
            normalize_to_bytes(message),
            randomizer,
        )
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
        digest = self._partition_digest_for_salt_message_bytes(salt, message_bytes, randomizer)
        return digest, self._xof_seed_prefix + digest

    def _partition_digest_for_salt_message_bytes(
        self,
        salt: int,
        message_bytes: bytes,
        randomizer: bytes,
    ) -> bytes:
        if self.params.partition_retry_mode == "stream":
            digest, _ = self._partition_stream_chunk_from_message_randomizer(
                message_bytes + randomizer,
                salt,
            )
            return digest
        if self.params.partition_sampler_mode == "stream":
            digest, _ = self._partition_digest_and_inline_sampler_from_message_bytes(
                salt,
                message_bytes,
                randomizer,
            )
            return digest
        return self._partition_digest_from_message_bytes(salt, message_bytes, randomizer)

    def _partition_digest_from_message_bytes(
        self,
        salt: int,
        message_bytes: bytes,
        randomizer: bytes,
    ) -> bytes:
        counting = counters_enabled()
        if counting:
            increment("keyed_hash.eval")
        return self._partition_hash_digest(
            self._partition_hash_prefix
            + salt.to_bytes(self.params.salt_bytes, "big")
            + message_bytes
            + randomizer,
            counting,
        )

    def _partition_digest_and_inline_sampler_from_message_bytes(
        self,
        salt: int,
        message_bytes: bytes,
        randomizer: bytes,
    ) -> tuple[bytes, bytes]:
        backend = self._partition_hash_fast_backend
        if backend is None or not self._partition_hash_fast_uses_xof:
            raise ValueError("partition stream sampler requires a SHAKE keyed hash")
        counting = counters_enabled()
        if counting:
            increment("keyed_hash.eval")
            increment("hash.backend_calls")
            increment(self._partition_hash_backend_counter)
        stream = backend(
            self._partition_hash_prefix
            + salt.to_bytes(self.params.salt_bytes, "big")
            + message_bytes
            + randomizer
        )
        digest_bytes = self._partition_hash_output_bytes
        output = stream.digest(digest_bytes + self.params.partition_stream_sampler_bytes)
        return (
            output[:digest_bytes],
            output[digest_bytes:],
        )

    def _partition_stream_chunk_from_message_randomizer(
        self,
        message_randomizer: bytes,
        salt: int,
    ) -> tuple[bytes, bytes]:
        if salt < 0:
            raise ValueError("salt must be non-negative")
        backend = self._partition_hash_fast_backend
        if backend is None or not self._partition_hash_fast_uses_xof:
            raise ValueError("partition retry stream requires a SHAKE keyed hash")

        counting = counters_enabled()
        if counting:
            increment("keyed_hash.eval")
            increment("hash.backend_calls")
            increment(self._partition_hash_backend_counter)

        sampler_bytes = (
            self.params.partition_stream_sampler_bytes
            if self.params.partition_sampler_mode == "stream"
            else 0
        )
        chunk_bytes = self._partition_stream_chunk_bytes
        start = sampler_bytes + salt * chunk_bytes
        digest_end = start + self._partition_hash_output_bytes
        stream_output = backend(self._partition_stream_prefix + message_randomizer).digest(digest_end)
        chunk = stream_output[start:digest_end]
        sampler_random_bytes = stream_output[:sampler_bytes] if sampler_bytes else b""
        return (
            chunk[: self._partition_hash_output_bytes],
            sampler_random_bytes,
        )

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
        indices_are_all_alphas = (
            isinstance(indices, range)
            and indices.start == 0
            and indices.step == 1
            and len(indices) == len(alpha_hash_prefixes)
        )
        backend = self._ots_hash_fast_backend
        if backend is None:
            if indices_are_all_alphas:
                return [
                    hash_bytes(
                        prefix + key_node,
                        output_bits=self.params.pm_TwH.security_parameter,
                        hash_name=self.params.pm_TwH.hash_name,
                    )
                    for prefix, key_node in zip(alpha_hash_prefixes, key_nodes)
                ]
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
        prefix_states = self._alpha_hash_prefix_states
        if uses_xof and prefix_states is not None and not counting:
            if indices_are_all_alphas:
                outputs: list[bytes] = []
                for state, key_node in zip(prefix_states, key_nodes):
                    hash_object = state.copy()
                    hash_object.update(key_node)
                    outputs.append(hash_object.digest(output_bytes))
                return outputs
            outputs = []
            for alpha, key_node in zip(indices, key_nodes):
                hash_object = prefix_states[alpha].copy()
                hash_object.update(key_node)
                outputs.append(hash_object.digest(output_bytes))
            return outputs
        if uses_xof:
            if indices_are_all_alphas:
                return [
                    backend(prefix + key_node).digest(output_bytes)
                    for prefix, key_node in zip(alpha_hash_prefixes, key_nodes)
                ]
            return [
                backend(alpha_hash_prefixes[alpha] + key_node).digest(output_bytes)
                for alpha, key_node in zip(indices, key_nodes)
            ]

        if security_parameter == 8 * output_bytes:
            if indices_are_all_alphas:
                return [
                    backend(prefix + key_node).digest()[:output_bytes]
                    for prefix, key_node in zip(alpha_hash_prefixes, key_nodes)
                ]
            return [
                backend(alpha_hash_prefixes[alpha] + key_node).digest()[:output_bytes]
                for alpha, key_node in zip(indices, key_nodes)
            ]
        if indices_are_all_alphas:
            return [
                truncate_to_bits(
                    backend(prefix + key_node).digest()[:output_bytes],
                    security_parameter,
                )
                for prefix, key_node in zip(alpha_hash_prefixes, key_nodes)
            ]
        return [
            truncate_to_bits(
                backend(alpha_hash_prefixes[alpha] + key_node).digest()[:output_bytes],
                security_parameter,
            )
            for alpha, key_node in zip(indices, key_nodes)
        ]

    def _merkle_root_from_hashed_leaves(self, leaves: Sequence[bytes]) -> bytes:
        if len(leaves) != self.params.leaf_count:
            raise ValueError(
                f"expected exactly {self.params.leaf_count} leaves, received {len(leaves)}"
            )

        pm_MT = self.params.pm_MT
        node_hash_prefixes = pm_MT._node_hash_prefixes
        backend = pm_MT._fast_backend
        uses_xof = pm_MT._fast_uses_xof
        output_bytes = pm_MT.output_bytes
        security_parameter = pm_MT.security_parameter
        hash_name = pm_MT.hash_name
        backend_counter = pm_MT._backend_counter
        counting = counters_enabled()

        native_root = _NATIVE_MERKLE_ROOT_FROM_LEAVES
        if (
            native_root is not None
            and not counting
            and backend is not None
            and uses_xof
            and security_parameter == 8 * output_bytes
            and hash_name in {"shake_128", "shake_256"}
        ):
            return native_root(
                leaves,
                pm_MT.level_widths,
                node_hash_prefixes,
                output_bytes,
                pm_MT.tree_height,
                hash_name == "shake_128",
            )

        current_level = list(leaves)
        for level in range(1, pm_MT.tree_height + 1):
            next_width = (len(current_level) + 1) // 2
            next_level = [b""] * next_width
            write_offset = 0
            for offset in range(0, len(current_level), 2):
                if offset + 1 >= len(current_level):
                    next_level[write_offset] = current_level[offset]
                    write_offset += 1
                    continue

                if counting:
                    increment("merkle.internal_hash")
                    increment("tweak_hash.eval")
                payload = (
                    node_hash_prefixes[level][offset // 2]
                    + current_level[offset]
                    + current_level[offset + 1]
                )
                if backend is None:
                    next_level[write_offset] = hash_bytes(
                        payload,
                        output_bits=security_parameter,
                        hash_name=hash_name,
                    )
                else:
                    if counting:
                        increment("hash.backend_calls")
                        increment(backend_counter)
                    hash_object = backend(payload)
                    if uses_xof:
                        next_level[write_offset] = hash_object.digest(output_bytes)
                    else:
                        raw = hash_object.digest()[:output_bytes]
                        next_level[write_offset] = (
                            raw
                            if security_parameter == 8 * output_bytes
                            else truncate_to_bits(raw, security_parameter)
                        )
                write_offset += 1
            current_level = next_level

        return current_level[0]

    def _find_first_valid_partition(
        self,
        message: MessageInput,
        randomizer: bytes,
    ) -> Tuple[int, GroupInput]:
        message_bytes = normalize_to_bytes(message)
        if self.params.partition_retry_mode == "stream":
            return self._find_first_valid_partition_stream(message_bytes, randomizer)

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
        val_strict_prefixed = _val_strict_isp_with_seed_prefix
        native_prefixed = _NATIVE_BYTES_PREFIXED_SEED
        use_native_prefixed = (
            not counting
            and native_prefixed is not None
            and pm_isp._window_valid
            and _NATIVE_MIN_MAX_G_BIT <= pm_isp.max_g_bit <= _NATIVE_MAX_MAX_G_BIT
            and pm_isp.max_g_value <= _NATIVE_MAX_G_VALUE
            and pm_isp.partition_num <= _NATIVE_MAX_PARTITION_NUM
        )
        native_w4_prefixed = _NATIVE_W4_BYTES_PREFIXED_SEED
        use_native_w4_prefixed = (
            not counting
            and native_w4_prefixed is not None
            and pm_isp._window_valid
            and pm_isp.max_g_bit == 2
            and pm_isp.max_g_value == 4
            and pm_isp.partition_num <= _NATIVE_W4_MAX_PARTITION_NUM
        )
        native_hash_len = pm_isp.hash_len
        native_max_g_bit = pm_isp.max_g_bit
        native_partition_num = pm_isp.partition_num
        native_window_low = pm_isp.window_low
        native_window_high = pm_isp.window_high
        native_use_shake128 = pm_isp.hash_name == "shake_128"
        salt = 0
        if partition_hash_backend is None:
            if counting:
                while True:
                    if self.params.max_sign_retries is not None and salt >= self.params.max_sign_retries:
                        raise RuntimeError("ValStrictISP did not accept within the configured retry budget")
                    increment("ycsig.partition_attempt")
                    increment("keyed_hash.eval")
                    digest = hash_bytes(
                        partition_hash_prefix + salt.to_bytes(salt_bytes, "big") + message_randomizer,
                        output_bits=partition_hash_output_bits,
                        hash_name=partition_hash_name,
                    )
                    groups = val_strict_prefixed(digest, pm_isp, xof_seed_prefix, True)
                    if groups is not None:
                        return salt, groups
                    salt += 1
            while True:
                if self.params.max_sign_retries is not None and salt >= self.params.max_sign_retries:
                    raise RuntimeError("ValStrictISP did not accept within the configured retry budget")
                digest = hash_bytes(
                    partition_hash_prefix + salt.to_bytes(salt_bytes, "big") + message_randomizer,
                    output_bits=partition_hash_output_bits,
                    hash_name=partition_hash_name,
                )
                if use_native_w4_prefixed:
                    groups = native_w4_prefixed(
                        digest,
                        native_hash_len,
                        native_partition_num,
                        native_window_low,
                        native_window_high,
                        xof_seed_prefix,
                        native_use_shake128,
                        True,
                    )
                elif use_native_prefixed:
                    groups = native_prefixed(
                        digest,
                        native_hash_len,
                        native_max_g_bit,
                        native_partition_num,
                        native_window_low,
                        native_window_high,
                        xof_seed_prefix,
                        native_use_shake128,
                        True,
                    )
                else:
                    groups = val_strict_prefixed(digest, pm_isp, xof_seed_prefix, True)
                if groups is not None:
                    return salt, groups
                salt += 1

        if partition_hash_uses_xof and self.params.partition_sampler_mode == "stream":
            sampler_bytes = self.params.partition_stream_sampler_bytes
            total_output_bytes = partition_hash_output_bytes + sampler_bytes
            groups_from_random = self._groups_from_partition_digest_and_stream_bytes
            if counting:
                while True:
                    if self.params.max_sign_retries is not None and salt >= self.params.max_sign_retries:
                        raise RuntimeError("ValStrictISP did not accept within the configured retry budget")
                    increment("ycsig.partition_attempt")
                    increment("keyed_hash.eval")
                    increment("hash.backend_calls")
                    increment(partition_hash_backend_counter)
                    output = partition_hash_backend(
                        partition_hash_prefix + salt.to_bytes(salt_bytes, "big") + message_randomizer
                    ).digest(total_output_bytes)
                    digest = output[:partition_hash_output_bytes]
                    groups = groups_from_random(digest, output[partition_hash_output_bytes:])
                    if groups is not None:
                        return salt, groups
                    salt += 1
            while True:
                if self.params.max_sign_retries is not None and salt >= self.params.max_sign_retries:
                    raise RuntimeError("ValStrictISP did not accept within the configured retry budget")
                output = partition_hash_backend(
                    partition_hash_prefix + salt.to_bytes(salt_bytes, "big") + message_randomizer
                ).digest(total_output_bytes)
                digest = output[:partition_hash_output_bytes]
                groups = groups_from_random(digest, output[partition_hash_output_bytes:])
                if groups is not None:
                    return salt, groups
                salt += 1

        if partition_hash_uses_xof:
            if counting:
                while True:
                    if self.params.max_sign_retries is not None and salt >= self.params.max_sign_retries:
                        raise RuntimeError("ValStrictISP did not accept within the configured retry budget")
                    increment("ycsig.partition_attempt")
                    increment("keyed_hash.eval")
                    increment("hash.backend_calls")
                    increment(partition_hash_backend_counter)
                    digest = partition_hash_backend(
                        partition_hash_prefix + salt.to_bytes(salt_bytes, "big") + message_randomizer
                    ).digest(partition_hash_output_bytes)
                    groups = val_strict_prefixed(digest, pm_isp, xof_seed_prefix, True)
                    if groups is not None:
                        return salt, groups
                    salt += 1
            while True:
                if self.params.max_sign_retries is not None and salt >= self.params.max_sign_retries:
                    raise RuntimeError("ValStrictISP did not accept within the configured retry budget")
                digest = partition_hash_backend(
                    partition_hash_prefix + salt.to_bytes(salt_bytes, "big") + message_randomizer
                ).digest(partition_hash_output_bytes)
                if use_native_w4_prefixed:
                    groups = native_w4_prefixed(
                        digest,
                        native_hash_len,
                        native_partition_num,
                        native_window_low,
                        native_window_high,
                        xof_seed_prefix,
                        native_use_shake128,
                        True,
                    )
                elif use_native_prefixed:
                    groups = native_prefixed(
                        digest,
                        native_hash_len,
                        native_max_g_bit,
                        native_partition_num,
                        native_window_low,
                        native_window_high,
                        xof_seed_prefix,
                        native_use_shake128,
                        True,
                    )
                else:
                    groups = val_strict_prefixed(digest, pm_isp, xof_seed_prefix, True)
                if groups is not None:
                    return salt, groups
                salt += 1

        if counting:
            while True:
                if self.params.max_sign_retries is not None and salt >= self.params.max_sign_retries:
                    raise RuntimeError("ValStrictISP did not accept within the configured retry budget")
                increment("ycsig.partition_attempt")
                increment("keyed_hash.eval")
                increment("hash.backend_calls")
                increment(partition_hash_backend_counter)
                digest = partition_hash_backend(
                    partition_hash_prefix + salt.to_bytes(salt_bytes, "big") + message_randomizer
                ).digest()[:partition_hash_output_bytes]
                groups = val_strict_prefixed(digest, pm_isp, xof_seed_prefix, True)
                if groups is not None:
                    return salt, groups
                salt += 1
        while True:
            if self.params.max_sign_retries is not None and salt >= self.params.max_sign_retries:
                raise RuntimeError("ValStrictISP did not accept within the configured retry budget")
            digest = partition_hash_backend(
                partition_hash_prefix + salt.to_bytes(salt_bytes, "big") + message_randomizer
            ).digest()[:partition_hash_output_bytes]
            if use_native_w4_prefixed:
                groups = native_w4_prefixed(
                    digest,
                    native_hash_len,
                    native_partition_num,
                    native_window_low,
                    native_window_high,
                    xof_seed_prefix,
                    native_use_shake128,
                    True,
                )
            elif use_native_prefixed:
                groups = native_prefixed(
                    digest,
                    native_hash_len,
                    native_max_g_bit,
                    native_partition_num,
                    native_window_low,
                    native_window_high,
                    xof_seed_prefix,
                    native_use_shake128,
                    True,
                )
            else:
                groups = val_strict_prefixed(digest, pm_isp, xof_seed_prefix, True)
            if groups is not None:
                return salt, groups
            salt += 1

    def _groups_from_partition_digest_and_stream_bytes(
        self,
        partition_digest: bytes,
        sampler_random_bytes: bytes,
    ) -> Optional[GroupInput]:
        try:
            return _val_strict_isp_with_random_bytes(
                partition_digest,
                self.params.pm_ISP,
                sampler_random_bytes,
                return_group_masks=True,
            )
        except ValueError as exc:
            if "insufficient random bytes" not in str(exc):
                raise
            raise ValueError(
                "partition_stream_sampler_bytes is too small for ValStrictISP; "
                "increase it instead of falling back to an internal sampler hash"
            ) from exc

    def _find_first_valid_partition_stream(
        self,
        message_bytes: bytes,
        randomizer: bytes,
    ) -> Tuple[int, GroupInput]:
        backend = self._partition_hash_fast_backend
        if backend is None or not self._partition_hash_fast_uses_xof:
            raise ValueError("partition retry stream requires a SHAKE keyed hash")

        counting = counters_enabled()
        if counting:
            increment("keyed_hash.eval")
            increment("hash.backend_calls")
            increment(self._partition_hash_backend_counter)

        stream = backend(self._partition_stream_prefix + message_bytes + randomizer)
        sampler_bytes = (
            self.params.partition_stream_sampler_bytes
            if self.params.partition_sampler_mode == "stream"
            else 0
        )
        chunk_bytes = self._partition_stream_chunk_bytes
        digest_bytes = self._partition_hash_output_bytes
        use_sampler_stream = sampler_bytes > 0
        pm_isp = self.params.pm_ISP
        native_find_stream = _NATIVE_FIND_FIRST_RANDOM_STREAM
        use_native_find_stream = (
            use_sampler_stream
            and not counting
            and native_find_stream is not None
            and pm_isp._window_valid
            and _NATIVE_MIN_MAX_G_BIT <= pm_isp.max_g_bit <= _NATIVE_MAX_MAX_G_BIT
            and pm_isp.max_g_value <= _NATIVE_MAX_G_VALUE
            and pm_isp.partition_num <= _NATIVE_MAX_PARTITION_NUM
        )
        salt = 0
        buffer = b""

        if use_native_find_stream:
            initial_batch_candidates = pm_isp.stream_batch_candidates
            if self.params.max_sign_retries is not None:
                if self.params.max_sign_retries <= 0:
                    raise RuntimeError("ValStrictISP did not accept within the configured retry budget")
                initial_batch_candidates = min(
                    initial_batch_candidates,
                    self.params.max_sign_retries,
                )
            buffer = stream.digest(sampler_bytes + initial_batch_candidates * chunk_bytes)
            try:
                result = native_find_stream(
                    buffer,
                    sampler_bytes,
                    pm_isp.hash_len,
                    pm_isp.max_g_bit,
                    pm_isp.partition_num,
                    pm_isp.window_low,
                    pm_isp.window_high,
                    0,
                    initial_batch_candidates,
                    True,
                )
            except ValueError as exc:
                if "insufficient random bytes" not in str(exc):
                    raise
                raise ValueError(
                    "partition_stream_sampler_bytes is too small for ValStrictISP; "
                    "increase it instead of falling back to an internal sampler hash"
                ) from exc
            if result is not None:
                return result
            salt = initial_batch_candidates

            while True:
                if self.params.max_sign_retries is not None and salt >= self.params.max_sign_retries:
                    raise RuntimeError("ValStrictISP did not accept within the configured retry budget")

                if len(buffer) < sampler_bytes + (salt + 1) * chunk_bytes:
                    target = max(sampler_bytes + (salt + 1) * chunk_bytes, 2 * len(buffer))
                    if self.params.max_sign_retries is not None:
                        max_target = sampler_bytes + self.params.max_sign_retries * chunk_bytes
                        if target > max_target:
                            target = max_target
                    buffer = stream.digest(target)

                available_salts = (len(buffer) - sampler_bytes) // chunk_bytes
                candidate_count = available_salts - salt
                if candidate_count <= 0:
                    continue

                try:
                    result = native_find_stream(
                        buffer,
                        sampler_bytes,
                        pm_isp.hash_len,
                        pm_isp.max_g_bit,
                        pm_isp.partition_num,
                        pm_isp.window_low,
                        pm_isp.window_high,
                        salt,
                        candidate_count,
                        True,
                    )
                except ValueError as exc:
                    if "insufficient random bytes" not in str(exc):
                        raise
                    raise ValueError(
                        "partition_stream_sampler_bytes is too small for ValStrictISP; "
                        "increase it instead of falling back to an internal sampler hash"
                    ) from exc
                if result is not None:
                    return result
                salt = available_salts

        xof_seed_prefix = self._xof_seed_prefix
        val_strict_prefixed = _val_strict_isp_with_seed_prefix

        while True:
            if self.params.max_sign_retries is not None and salt >= self.params.max_sign_retries:
                raise RuntimeError("ValStrictISP did not accept within the configured retry budget")
            if counting:
                increment("ycsig.partition_attempt")

            start = sampler_bytes + salt * chunk_bytes
            digest_end = start + digest_bytes
            required_end = digest_end
            if len(buffer) < required_end:
                if len(buffer) == 0:
                    target = max(required_end, sampler_bytes + 8 * chunk_bytes)
                else:
                    target = max(required_end, 2 * len(buffer))
                buffer = stream.digest(target)
            digest = buffer[start:digest_end]
            if use_sampler_stream:
                groups = self._groups_from_partition_digest_and_stream_bytes(
                    digest,
                    buffer[:sampler_bytes],
                )
            else:
                groups = val_strict_prefixed(digest, pm_isp, xof_seed_prefix, True)
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
    ) -> Optional[GroupInput]:
        if self.params.partition_retry_mode == "stream":
            partition_digest, sampler_random_bytes = self._partition_stream_chunk_from_message_randomizer(
                message_bytes + randomizer,
                salt,
            )
            if self.params.partition_sampler_mode == "stream":
                return self._groups_from_partition_digest_and_stream_bytes(
                    partition_digest,
                    sampler_random_bytes,
                )
            return _val_strict_isp_with_seed_prefix(
                partition_digest,
                self.params.pm_ISP,
                self._xof_seed_prefix,
                True,
            )

        if self.params.partition_sampler_mode == "stream":
            partition_digest, sampler_random_bytes = self._partition_digest_and_inline_sampler_from_message_bytes(
                salt,
                message_bytes,
                randomizer,
            )
            return self._groups_from_partition_digest_and_stream_bytes(
                partition_digest,
                sampler_random_bytes,
            )

        partition_digest = self._partition_digest_from_message_bytes(
            salt,
            message_bytes,
            randomizer,
        )
        return _val_strict_isp_with_seed_prefix(
            partition_digest,
            self.params.pm_ISP,
            self._xof_seed_prefix,
            True,
        )

    def _group_key_and_masks(self, groups: GroupInput) -> Tuple[bytes | tuple[int, ...] | tuple[tuple[int, ...], ...], bytes | tuple[int, ...] | None]:
        if isinstance(groups, bytes):
            return groups, groups
        if isinstance(groups, tuple) and (not groups or isinstance(groups[0], int)):
            return groups, groups
        return tuple(tuple(subgroup) for subgroup in groups), None

    def _groups_to_signed_indices(self, groups: GroupInput) -> Tuple[int, ...]:
        signed_indices, _ = self._groups_to_signed_indices_and_points(groups)
        return signed_indices

    def _groups_to_signed_indices_and_points(
        self,
        groups: GroupInput,
    ) -> Tuple[Tuple[int, ...], Tuple[str, ...]]:
        group_key, group_masks = self._group_key_and_masks(groups)
        cached = self._signed_group_material_cache.get(group_key)
        if cached is not None:
            return cached

        full_cached = self._group_material_cache.get(group_key)
        if full_cached is not None:
            signed_material = (full_cached[0], full_cached[1])
            self._cache_store(self._signed_group_material_cache, group_key, signed_material)
            return signed_material

        if group_masks is not None:
            material = self._group_material(groups)
            signed_material = (material[0], material[1])
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
        groups: GroupInput,
    ) -> Tuple[Tuple[int, ...], Tuple[str, ...], Tuple[int, ...], Tuple[str, ...]]:
        return self._group_material(groups)

    def _group_material(
        self,
        groups: GroupInput,
    ) -> Tuple[Tuple[int, ...], Tuple[str, ...], Tuple[int, ...], Tuple[str, ...]]:
        group_key, group_masks = self._group_key_and_masks(groups)
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

        if group_masks is not None:
            group_mask_material_templates = self._group_mask_material_templates
            mask_byte_len = max(1, (max_g_value + 7) // 8)
            if (
                isinstance(group_masks, bytes)
                and mask_byte_len == 1
                and len(group_masks) == self.params.partition_size
                and group_mask_material_templates is not None
            ):
                for group_index, mask in enumerate(group_masks):
                    (
                        group_signed_indices,
                        group_signed_points,
                        group_complementary_indices,
                        group_complementary_points,
                    ) = group_mask_material_templates[group_index][mask]
                    signed_end = signed_offset + len(group_signed_indices)
                    complementary_end = complementary_offset + len(group_complementary_indices)
                    signed_indices[signed_offset:signed_end] = group_signed_indices
                    signed_points[signed_offset:signed_end] = group_signed_points
                    complementary_indices[complementary_offset:complementary_end] = group_complementary_indices
                    complementary_points[complementary_offset:complementary_end] = group_complementary_points
                    signed_offset = signed_end
                    complementary_offset = complementary_end
            else:
                if isinstance(group_masks, bytes):
                    expected_len = self.params.partition_size * mask_byte_len
                    if len(group_masks) != expected_len:
                        raise ValueError("packed group mask length does not match parameters")
                    mask_iter = (
                        int.from_bytes(
                            group_masks[start : start + mask_byte_len],
                            "little",
                        )
                        for start in range(0, expected_len, mask_byte_len)
                    )
                else:
                    mask_iter = iter(group_masks)
                for group_index, mask in enumerate(mask_iter):
                    base = group_offsets[group_index]
                    for block_value in range(max_g_value):
                        alpha = base + block_value
                        if mask & (1 << block_value):
                            signed_indices[signed_offset] = alpha
                            signed_points[signed_offset] = alpha_bitstrings[alpha]
                            signed_offset += 1
                        else:
                            complementary_indices[complementary_offset] = alpha
                            complementary_points[complementary_offset] = alpha_bitstrings[alpha]
                            complementary_offset += 1
        else:
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
        native_cover_positions = _NATIVE_CANONICAL_COVER_POSITIONS
        if native_cover_positions is not None:
            positions = native_cover_positions(
                self.params.pm_MT.level_widths,
                signed_indices,
                self.params.pm_MT.tree_height,
            )
        else:
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

    def _puncture_and_reveal_indices_fast(
        self,
        sk: PPRFKey,
        selected_indices: Tuple[int, ...],
    ) -> Tuple[
        Tuple[bytes, ...],
        Tuple[Tuple[int, int, int], ...],
        Tuple[Optional[bytes], ...],
    ]:
        params = sk.params
        message_length = params.message_length
        full_domain_size = 1 << message_length
        outputs: list[Optional[bytes]] = [None] * len(selected_indices)
        compact_frontier: list[bytes] = []
        frontier_ranges: list[tuple[int, int, int]] = []
        counting = counters_enabled()
        expand_backend = params._expand_fast_backend
        expand_prefix = params._expand_prefix
        expand_uses_xof = params._expand_fast_uses_xof
        expand_output_bytes = params._expand_output_bytes
        backend_counter = params._backend_counter
        seed_bytes = params._seed_bytes
        seed_bits = params.seed_bits
        hash_name = params.hash_name

        native_puncture = _NATIVE_PPRF_PUNCTURE_AND_REVEAL
        if (
            native_puncture is not None
            and not counting
            and expand_backend is not None
            and expand_uses_xof
            and seed_bits == 8 * seed_bytes
            and message_length <= 63
            and hash_name in {"shake_128", "shake_256"}
        ):
            provider_ranges = []
            for provider in sk.frontier:
                prefix = provider.prefix
                depth = len(prefix)
                span = 1 << (message_length - depth)
                low = (int(prefix, 2) if prefix else 0) * span
                high = min(low + span, full_domain_size)
                provider_ranges.append((depth, low, high, provider.seed))
            return native_puncture(
                tuple(provider_ranges),
                selected_indices,
                expand_prefix,
                seed_bytes,
                message_length,
                hash_name == "shake_128",
            )

        if expand_backend is not None and expand_uses_xof:
            if counting:
                def expand_seed(seed: bytes) -> tuple[bytes, bytes]:
                    increment("pprf.expand")
                    increment("hash.backend_calls")
                    increment(backend_counter)
                    expanded = expand_backend(expand_prefix + seed).digest(expand_output_bytes)
                    return expanded[:seed_bytes], expanded[seed_bytes:]
            else:
                def expand_seed(seed: bytes) -> tuple[bytes, bytes]:
                    expanded = expand_backend(expand_prefix + seed).digest(expand_output_bytes)
                    return expanded[:seed_bytes], expanded[seed_bytes:]
        elif expand_backend is not None:
            if counting:
                def expand_seed(seed: bytes) -> tuple[bytes, bytes]:
                    increment("pprf.expand")
                    increment("hash.backend_calls")
                    increment(backend_counter)
                    expanded = expand_backend(expand_prefix + seed).digest()[:expand_output_bytes]
                    return expanded[:seed_bytes], expanded[seed_bytes:]
            else:
                def expand_seed(seed: bytes) -> tuple[bytes, bytes]:
                    expanded = expand_backend(expand_prefix + seed).digest()[:expand_output_bytes]
                    return expanded[:seed_bytes], expanded[seed_bytes:]
        else:
            def expand_seed(seed: bytes) -> tuple[bytes, bytes]:
                if counting:
                    increment("pprf.expand")
                expanded = hash_bytes(
                    expand_prefix + seed,
                    output_bits=2 * seed_bytes * 8,
                    hash_name=hash_name,
                )
                return (
                    truncate_to_bits(expanded[:seed_bytes], seed_bits),
                    truncate_to_bits(expanded[seed_bytes:], seed_bits),
                )

        def visit(
            seed: bytes,
            depth: int,
            low: int,
            high: int,
            start: int,
            end: int,
        ) -> None:
            if start >= end:
                compact_frontier.append(seed)
                frontier_ranges.append((depth, low, high))
                return
            if depth == message_length:
                outputs[start] = seed
                return

            mid_value = (low + high) >> 1
            mid = bisect_left(selected_indices, mid_value, start, end)
            left_seed, right_seed = expand_seed(seed)
            visit(left_seed, depth + 1, low, mid_value, start, mid)
            visit(right_seed, depth + 1, mid_value, high, mid, end)

        for provider in sk.frontier:
            prefix = provider.prefix
            depth = len(prefix)
            span = 1 << (message_length - depth)
            low = (int(prefix, 2) if prefix else 0) * span
            high = min(low + span, full_domain_size)
            start = bisect_left(selected_indices, low)
            end = bisect_left(selected_indices, high, start)
            visit(provider.seed, depth, low, high, start, end)

        return tuple(compact_frontier), tuple(frontier_ranges), tuple(outputs)

    def _leaf_material_from_punctured_indices_fast(
        self,
        stored_seeds: Sequence[bytes],
        selected_indices: Tuple[int, ...],
        complement_indices: Tuple[int, ...],
    ) -> Tuple[Optional[bytes], ...]:
        params = self.params.pm_PPRF
        seed_bytes = params._seed_bytes
        for seed in stored_seeds:
            if len(seed) != seed_bytes:
                raise ValueError("stored seed has incorrect byte length")

        message_length = params.message_length
        outputs: list[Optional[bytes]] = [None] * len(complement_indices)
        counting = counters_enabled()
        expand_backend = params._expand_fast_backend
        expand_prefix = params._expand_prefix
        expand_uses_xof = params._expand_fast_uses_xof
        expand_output_bytes = params._expand_output_bytes
        backend_counter = params._backend_counter
        seed_bits = params.seed_bits
        hash_name = params.hash_name
        frontier_ranges = self._punctured_range_cache.get(selected_indices)
        if frontier_ranges is not None and len(frontier_ranges) != len(stored_seeds):
            frontier_ranges = None

        if expand_backend is not None and expand_uses_xof:
            if counting:
                def expand_seed(seed: bytes) -> tuple[bytes, bytes]:
                    increment("pprf.expand")
                    increment("hash.backend_calls")
                    increment(backend_counter)
                    expanded = expand_backend(expand_prefix + seed).digest(expand_output_bytes)
                    return expanded[:seed_bytes], expanded[seed_bytes:]
            else:
                def expand_seed(seed: bytes) -> tuple[bytes, bytes]:
                    expanded = expand_backend(expand_prefix + seed).digest(expand_output_bytes)
                    return expanded[:seed_bytes], expanded[seed_bytes:]
        elif expand_backend is not None:
            if counting:
                def expand_seed(seed: bytes) -> tuple[bytes, bytes]:
                    increment("pprf.expand")
                    increment("hash.backend_calls")
                    increment(backend_counter)
                    expanded = expand_backend(expand_prefix + seed).digest()[:expand_output_bytes]
                    return expanded[:seed_bytes], expanded[seed_bytes:]
            else:
                def expand_seed(seed: bytes) -> tuple[bytes, bytes]:
                    expanded = expand_backend(expand_prefix + seed).digest()[:expand_output_bytes]
                    return expanded[:seed_bytes], expanded[seed_bytes:]
        else:
            def expand_seed(seed: bytes) -> tuple[bytes, bytes]:
                if counting:
                    increment("pprf.expand")
                expanded = hash_bytes(
                    expand_prefix + seed,
                    output_bits=2 * seed_bytes * 8,
                    hash_name=hash_name,
                )
                return (
                    truncate_to_bits(expanded[:seed_bytes], seed_bits),
                    truncate_to_bits(expanded[seed_bytes:], seed_bits),
                )

        def eval_subtree(
            seed: bytes,
            depth: int,
            low: int,
            high: int,
            start: int,
            end: int,
        ) -> None:
            if start >= end:
                return
            if depth == message_length:
                outputs[start] = seed
                return
            mid_value = (low + high) >> 1
            mid = bisect_left(complement_indices, mid_value, start, end)
            left_seed, right_seed = expand_seed(seed)
            eval_subtree(left_seed, depth + 1, low, mid_value, start, mid)
            eval_subtree(right_seed, depth + 1, mid_value, high, mid, end)

        if frontier_ranges is None:
            frontier_range_list: list[tuple[int, int, int]] = []

            def collect_cover(
                depth: int,
                low: int,
                high: int,
                selected_start: int,
                selected_end: int,
            ) -> None:
                if selected_start >= selected_end:
                    frontier_range_list.append((depth, low, high))
                    return
                if depth == message_length:
                    return

                mid_value = (low + high) >> 1
                selected_mid = bisect_left(selected_indices, mid_value, selected_start, selected_end)
                collect_cover(depth + 1, low, mid_value, selected_start, selected_mid)
                collect_cover(depth + 1, mid_value, high, selected_mid, selected_end)

            for prefix in _active_domain_prefixes(message_length, params.domain_size):
                depth = len(prefix)
                span = 1 << (message_length - depth)
                low = (int(prefix, 2) if prefix else 0) * span
                high = min(low + span, 1 << message_length)
                selected_start = bisect_left(selected_indices, low)
                selected_end = bisect_left(selected_indices, high, selected_start)
                collect_cover(depth, low, high, selected_start, selected_end)
            frontier_ranges = tuple(frontier_range_list)
            self._cache_store(self._punctured_range_cache, selected_indices, frontier_ranges)

        if len(frontier_ranges) != len(stored_seeds):
            raise ValueError("stored_seeds does not match the canonical frontier size")
        native_leaf_material = _NATIVE_PPRF_LEAF_MATERIAL
        if (
            native_leaf_material is not None
            and not counting
            and expand_backend is not None
            and expand_uses_xof
            and seed_bits == 8 * seed_bytes
            and message_length <= 63
            and hash_name in {"shake_128", "shake_256"}
        ):
            return native_leaf_material(
                stored_seeds,
                frontier_ranges,
                complement_indices,
                expand_prefix,
                seed_bytes,
                message_length,
                hash_name == "shake_128",
            )
        for seed, (depth, low, high) in zip(stored_seeds, frontier_ranges):
            start = bisect_left(complement_indices, low)
            end = bisect_left(complement_indices, high, start)
            eval_subtree(seed, depth, low, high, start, end)
        return tuple(outputs)

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
            frontier.append(_new_stored_seed_unchecked(prefix, seed))
        punctured_key = _new_pprf_key_unchecked(self.params.pm_PPRF, frontier)
        self._cache_store(self._punctured_key_cache, cache_key, punctured_key)
        return punctured_key

    def _merkle_internal_hash_function(self):
        pm_MT = self.params.pm_MT
        node_hash_prefixes = pm_MT._node_hash_prefixes
        backend = pm_MT._fast_backend
        uses_xof = pm_MT._fast_uses_xof
        output_bytes = pm_MT.output_bytes
        security_parameter = pm_MT.security_parameter
        hash_name = pm_MT.hash_name
        backend_counter = pm_MT._backend_counter
        counting = counters_enabled()

        if backend is not None and uses_xof:
            if counting:
                def internal_hash(level: int, offset: int, left: bytes, right: bytes) -> bytes:
                    increment("merkle.internal_hash")
                    increment("tweak_hash.eval")
                    increment("hash.backend_calls")
                    increment(backend_counter)
                    return backend(node_hash_prefixes[level][offset] + left + right).digest(output_bytes)
            else:
                def internal_hash(level: int, offset: int, left: bytes, right: bytes) -> bytes:
                    return backend(node_hash_prefixes[level][offset] + left + right).digest(output_bytes)
            return internal_hash

        if backend is not None:
            if security_parameter == 8 * output_bytes:
                if counting:
                    def internal_hash(level: int, offset: int, left: bytes, right: bytes) -> bytes:
                        increment("merkle.internal_hash")
                        increment("tweak_hash.eval")
                        increment("hash.backend_calls")
                        increment(backend_counter)
                        return backend(node_hash_prefixes[level][offset] + left + right).digest()[:output_bytes]
                else:
                    def internal_hash(level: int, offset: int, left: bytes, right: bytes) -> bytes:
                        return backend(node_hash_prefixes[level][offset] + left + right).digest()[:output_bytes]
                return internal_hash

            if counting:
                def internal_hash(level: int, offset: int, left: bytes, right: bytes) -> bytes:
                    increment("merkle.internal_hash")
                    increment("tweak_hash.eval")
                    increment("hash.backend_calls")
                    increment(backend_counter)
                    raw = backend(node_hash_prefixes[level][offset] + left + right).digest()[:output_bytes]
                    return truncate_to_bits(raw, security_parameter)
            else:
                def internal_hash(level: int, offset: int, left: bytes, right: bytes) -> bytes:
                    raw = backend(node_hash_prefixes[level][offset] + left + right).digest()[:output_bytes]
                    return truncate_to_bits(raw, security_parameter)
            return internal_hash

        if counting:
            def internal_hash(level: int, offset: int, left: bytes, right: bytes) -> bytes:
                increment("merkle.internal_hash")
                increment("tweak_hash.eval")
                return hash_bytes(
                    node_hash_prefixes[level][offset] + left + right,
                    output_bits=security_parameter,
                    hash_name=hash_name,
                )
        else:
            def internal_hash(level: int, offset: int, left: bytes, right: bytes) -> bytes:
                return hash_bytes(
                    node_hash_prefixes[level][offset] + left + right,
                    output_bits=security_parameter,
                    hash_name=hash_name,
                )
        return internal_hash

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

        native_partial_state = _NATIVE_MERKLE_COMPACT_PARTIAL_STATE
        if (
            native_partial_state is not None
            and not counting
            and backend is not None
            and uses_xof
            and security_parameter == 8 * output_bytes
            and hash_name in {"shake_128", "shake_256"}
        ):
            return native_partial_state(
                level_widths,
                node_hash_prefixes,
                positions,
                signed_indices,
                signed_leaf_values,
                output_bytes,
                pm_MT.tree_height,
                hash_name == "shake_128",
            )

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
                    raise ValueError("missing leaf required to build the partial state")
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

        pm_MT = self.params.pm_MT
        native_rebuild = _NATIVE_MERKLE_SPARSE_REBUILD
        if (
            native_rebuild is not None
            and not counters_enabled()
            and pm_MT._fast_backend is not None
            and pm_MT._fast_uses_xof
            and pm_MT.security_parameter == 8 * pm_MT.output_bytes
            and pm_MT.hash_name in {"shake_128", "shake_256"}
        ):
            return native_rebuild(
                pm_MT.level_widths,
                pm_MT._node_hash_prefixes,
                positions,
                values,
                complementary_indices,
                complementary_leaf_values,
                pm_MT.output_bytes,
                pm_MT.tree_height,
                pm_MT.hash_name == "shake_128",
            )

        known_levels: list[list[Optional[bytes]]] = [
            [None] * width for width in pm_MT.level_widths
        ]
        leaf_values = known_levels[0]
        for alpha, value in zip(complementary_indices, complementary_leaf_values):
            leaf_values[alpha] = value
        for (level, offset), value in zip(positions, values):
            if known_levels[level][offset] is not None:
                raise ValueError("the same position appears in both partial_state and complementary_leaves")
            known_levels[level][offset] = value

        level_widths = pm_MT.level_widths
        internal_hash = self._merkle_internal_hash_function()

        for level in range(1, pm_MT.tree_height + 1):
            child_level = known_levels[level - 1]
            parent_level = known_levels[level]
            child_width = level_widths[level - 1]
            for offset in range(level_widths[level]):
                if parent_level[offset] is not None:
                    continue

                child_offset = offset << 1
                left_value = child_level[child_offset]
                if left_value is None:
                    continue
                right_offset = child_offset + 1
                if right_offset >= child_width:
                    parent_level[offset] = left_value
                    continue
                right_value = child_level[right_offset]
                if right_value is None:
                    continue
                parent_level[offset] = internal_hash(level, offset, left_value, right_value)

        root = known_levels[pm_MT.tree_height][0]
        if root is None:
            raise ValueError("insufficient information to rebuild the Merkle root")
        return root

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
