from __future__ import annotations

import hashlib
from bisect import bisect_left, bisect_right, insort
from dataclasses import dataclass, field
from typing import Dict, FrozenSet, Iterable, Iterator, List, Optional, Sequence, Tuple, Union

from crypto_utils import (
    DEFAULT_HASH_NAME,
    SUPPORTED_HASHES,
    bitstring_to_bytes,
    bits_to_bytes,
    derive_parameter,
    hash_bytes,
    normalize_bitstring,
    normalize_to_bytes,
    resolve_bit_length,
    truncate_to_bits,
)
from operation_counter import enabled as counters_enabled, increment

MessageInput = Union[str, bytes, int]
SeedInput = Union[str, bytes, int]
PRFValue = bytes


@dataclass(frozen=True)
class PPRFParameters:
    """
    Public parameters pm_PPRF output by PRFSetup(1^kappa).

    MessageSpace_PPRF is the active prefix {0, ..., domain_size - 1} embedded in
    the bitstrings of length message_length. KeySpace_PPRF consists of canonical
    frontier encodings over GGM seeds for that active domain.
    Range_PPRF is {0,1}^{range_bits}. This implementation is binary GGM-based,
    so the branching factor is configurable in the interface but currently must
    be 2.
    """

    security_parameter: int
    message_length: int
    domain_size: int = 0
    hash_name: str = DEFAULT_HASH_NAME
    seed_bits: int = 0
    range_bits: int = 0
    branching_factor: int = 2
    root_label: bytes = b"GGM/PPRF/root/"
    expand_left_label: bytes = b"GGM/PPRF/expand/0/"
    expand_right_label: bytes = b"GGM/PPRF/expand/1/"
    eval_label: bytes = b"GGM/PPRF/eval/"
    _seed_bytes: int = field(init=False, repr=False)
    _range_bytes: int = field(init=False, repr=False)
    _backend_counter: str = field(init=False, repr=False)
    _expand_prefix: bytes = field(init=False, repr=False)
    _expand_fast_backend: Optional[object] = field(init=False, repr=False)
    _expand_fast_uses_xof: bool = field(init=False, repr=False)
    _expand_output_bytes: int = field(init=False, repr=False)
    _leaf_fast_backend: Optional[object] = field(init=False, repr=False)
    _leaf_fast_uses_xof: bool = field(init=False, repr=False)

    def __post_init__(self) -> None:
        if self.security_parameter <= 0:
            raise ValueError("security_parameter must be positive")
        if self.message_length <= 0:
            raise ValueError("message_length must be positive")
        full_domain_size = 1 << self.message_length
        if self.domain_size <= 0:
            object.__setattr__(self, "domain_size", full_domain_size)
        elif self.domain_size > full_domain_size:
            raise ValueError("domain_size must be at most 2^message_length")
        if self.hash_name not in SUPPORTED_HASHES:
            raise ValueError(
                f"unsupported hash_name={self.hash_name!r}; choose from {sorted(SUPPORTED_HASHES)}"
            )
        if self.seed_bits <= 0:
            object.__setattr__(self, "seed_bits", self.security_parameter)
        if self.range_bits <= 0:
            object.__setattr__(self, "range_bits", self.security_parameter)
        if self.branching_factor != 2:
            raise ValueError("the current GGM realization supports branching_factor=2 only")

        seed_bytes = bits_to_bytes(self.seed_bits)
        range_bytes = bits_to_bytes(self.range_bits)
        backend_counter = f"hash.backend_calls.{self.hash_name}"
        expand_output_bytes = 2 * seed_bytes
        expand_prefix = (
            len(self.expand_left_label).to_bytes(2, "big")
            + self.expand_left_label
            + len(self.expand_right_label).to_bytes(2, "big")
            + self.expand_right_label
        )

        expand_fast_backend = None
        expand_fast_uses_xof = False
        leaf_fast_backend = None
        leaf_fast_uses_xof = False
        if self.seed_bits == 8 * seed_bytes:
            if self.hash_name == "shake_128":
                expand_fast_backend = hashlib.shake_128
                expand_fast_uses_xof = True
            elif self.hash_name == "shake_256":
                expand_fast_backend = hashlib.shake_256
                expand_fast_uses_xof = True
            elif self.hash_name == "sha3_256" and expand_output_bytes <= hashlib.sha3_256().digest_size:
                expand_fast_backend = hashlib.sha3_256
            elif self.hash_name == "sha3_512" and expand_output_bytes <= hashlib.sha3_512().digest_size:
                expand_fast_backend = hashlib.sha3_512
        if self.range_bits == 8 * range_bytes:
            if self.hash_name == "shake_128":
                leaf_fast_backend = hashlib.shake_128
                leaf_fast_uses_xof = True
            elif self.hash_name == "shake_256":
                leaf_fast_backend = hashlib.shake_256
                leaf_fast_uses_xof = True
            elif self.hash_name == "sha3_256" and range_bytes <= hashlib.sha3_256().digest_size:
                leaf_fast_backend = hashlib.sha3_256
            elif self.hash_name == "sha3_512" and range_bytes <= hashlib.sha3_512().digest_size:
                leaf_fast_backend = hashlib.sha3_512

        object.__setattr__(self, "_seed_bytes", seed_bytes)
        object.__setattr__(self, "_range_bytes", range_bytes)
        object.__setattr__(self, "_backend_counter", backend_counter)
        object.__setattr__(self, "_expand_prefix", expand_prefix)
        object.__setattr__(self, "_expand_fast_backend", expand_fast_backend)
        object.__setattr__(self, "_expand_fast_uses_xof", expand_fast_uses_xof)
        object.__setattr__(self, "_expand_output_bytes", expand_output_bytes)
        object.__setattr__(self, "_leaf_fast_backend", leaf_fast_backend)
        object.__setattr__(self, "_leaf_fast_uses_xof", leaf_fast_uses_xof)

    @property
    def seed_bytes(self) -> int:
        return self._seed_bytes

    @property
    def range_bytes(self) -> int:
        return self._range_bytes

    @property
    def full_domain_size(self) -> int:
        return 1 << self.message_length


@dataclass(frozen=True, order=True)
class StoredSeed:
    """
    A stored GGM seed for the subtree rooted at prefix.

    The empty prefix "" denotes the whole message space.
    """

    prefix: str
    seed: bytes


@dataclass(frozen=True)
class PPRFKey:
    """
    Canonical punctured key represented by an antichain frontier of stored seeds.
    """

    params: PPRFParameters
    frontier: Tuple[StoredSeed, ...]

    def __post_init__(self) -> None:
        frontier = tuple(sorted(self.frontier, key=lambda node: node.prefix))
        object.__setattr__(self, "frontier", frontier)

        seen_prefixes = set()
        for index, node in enumerate(frontier):
            if node.prefix in seen_prefixes:
                raise ValueError(f"duplicate prefix in frontier: {node.prefix!r}")
            if len(node.prefix) > self.params.message_length:
                raise ValueError("stored prefix is longer than message_length")
            if any(bit not in {"0", "1"} for bit in node.prefix):
                raise ValueError("stored prefixes must be bitstrings")
            if len(node.seed) != self.params.seed_bytes:
                raise ValueError("stored seed has incorrect byte length")
            for parent_index in range(index):
                parent = frontier[parent_index]
                if node.prefix.startswith(parent.prefix) or parent.prefix.startswith(node.prefix):
                    raise ValueError("frontier must be an antichain of prefixes")
            seen_prefixes.add(node.prefix)

    def serialize(self) -> bytes:
        """
        Canonical serialization used by EncLen.
        """

        chunks = [len(self.frontier).to_bytes(4, "big")]
        for node in self.frontier:
            prefix_len = len(node.prefix)
            prefix_bytes = bitstring_to_bytes(node.prefix)
            chunks.append(prefix_len.to_bytes(2, "big"))
            chunks.append(len(prefix_bytes).to_bytes(2, "big"))
            chunks.append(prefix_bytes)
            chunks.append(node.seed)
        return b"".join(chunks)


@dataclass
class PPRFComputationCache:
    """
    Per-operation cache for repeated GGM traversals.

    This is especially useful for YCSig signing and verification where many
    queried leaves share long common prefixes.
    """

    expanded_children: Dict[bytes, Tuple[bytes, bytes]] = field(default_factory=dict)
    leaf_outputs: Dict[bytes, PRFValue] = field(default_factory=dict)


def _new_pprf_key_unchecked(
    params: PPRFParameters,
    frontier: Sequence[StoredSeed],
) -> PPRFKey:
    key = object.__new__(PPRFKey)
    object.__setattr__(key, "params", params)
    object.__setattr__(key, "frontier", tuple(frontier))
    return key


def _expand_seed(
    seed: bytes,
    params: PPRFParameters,
    cache: Optional[PPRFComputationCache] = None,
) -> Tuple[bytes, bytes]:
    if cache is not None:
        cached = cache.expanded_children.get(seed)
        if cached is not None:
            return cached

    counting = counters_enabled()
    if counting:
        increment("pprf.expand")
    backend = params._expand_fast_backend
    if backend is not None:
        if counting:
            increment("hash.backend_calls")
            increment(params._backend_counter)
        hash_object = backend(params._expand_prefix + seed)
        if params._expand_fast_uses_xof:
            expanded = hash_object.digest(params._expand_output_bytes)
        else:
            expanded = hash_object.digest()[: params._expand_output_bytes]
        left = expanded[: params.seed_bytes]
        right = expanded[params.seed_bytes :]
    else:
        # Realize one GGM expansion as a single extendable-output draw and then
        # split the stream into the left/right child seeds.
        expanded = hash_bytes(
            params._expand_prefix + seed,
            output_bits=2 * params.seed_bytes * 8,
            hash_name=params.hash_name,
        )
        left = truncate_to_bits(expanded[: params.seed_bytes], params.seed_bits)
        right = truncate_to_bits(expanded[params.seed_bytes :], params.seed_bits)
    if cache is not None:
        cache.expanded_children[seed] = (left, right)
    return left, right


def _leaf_output(
    seed: bytes,
    params: PPRFParameters,
    cache: Optional[PPRFComputationCache] = None,
) -> PRFValue:
    if cache is not None:
        cached = cache.leaf_outputs.get(seed)
        if cached is not None:
            return cached

    counting = counters_enabled()
    if counting:
        increment("pprf.leaf_output")
    backend = params._leaf_fast_backend
    if backend is not None:
        if counting:
            increment("hash.backend_calls")
            increment(params._backend_counter)
        hash_object = backend(params.eval_label + seed)
        if params._leaf_fast_uses_xof:
            output = hash_object.digest(params.range_bytes)
        else:
            output = hash_object.digest()[: params.range_bytes]
    else:
        output = hash_bytes(
            params.eval_label + seed,
            output_bits=params.range_bits,
            hash_name=params.hash_name,
        )
    if cache is not None:
        cache.leaf_outputs[seed] = output
    return output


def _find_provider(frontier: Sequence[StoredSeed], message_bits: str) -> Optional[int]:
    for index, node in enumerate(frontier):
        if message_bits.startswith(node.prefix):
            return index
    return None


def _normalize_message_list(
    messages: Iterable[MessageInput],
    message_length: int,
    *,
    inputs_normalized: bool = False,
    inputs_trusted: bool = False,
) -> Sequence[str]:
    if not inputs_normalized:
        return [normalize_bitstring(x, message_length) for x in messages]

    if inputs_trusted:
        if isinstance(messages, (list, tuple)):
            return messages
        return tuple(messages)

    normalized_messages = messages if isinstance(messages, list) else list(messages)
    for message_bits in normalized_messages:
        if not isinstance(message_bits, str):
            raise TypeError("normalized PPRF messages must be bitstrings")
        if len(message_bits) != message_length:
            raise ValueError(
                f"bitstring length mismatch: expected {message_length}, got {len(message_bits)}"
            )
    return normalized_messages


def _sorted_target_range_for_prefix(
    targets: Sequence[str],
    prefix: str,
    message_length: int,
) -> Tuple[int, int]:
    suffix_len = message_length - len(prefix)
    lower = prefix + ("0" * suffix_len)
    upper = prefix + ("1" * suffix_len)
    start = bisect_left(targets, lower)
    end = bisect_right(targets, upper, lo=start)
    return start, end


def _frontier_block_size(prefix: str, message_length: int) -> int:
    return 1 << (message_length - len(prefix))


def _messages_are_sorted_unique(messages: Sequence[str]) -> bool:
    return all(messages[index - 1] < messages[index] for index in range(1, len(messages)))


def _is_valid_message_bits(message_bits: str, params: PPRFParameters) -> bool:
    return int(message_bits, 2) < params.domain_size


def _active_domain_prefixes(
    message_length: int,
    domain_size: int,
) -> Tuple[str, ...]:
    full_domain_size = 1 << message_length
    if domain_size == full_domain_size:
        return ("",)

    prefixes: List[str] = []

    def build(prefix: str, start: int, end: int) -> None:
        if start >= domain_size:
            return
        if end <= domain_size:
            prefixes.append(prefix)
            return
        if len(prefix) == message_length:
            prefixes.append(prefix)
            return

        mid = (start + end) // 2
        build(prefix + "0", start, mid)
        build(prefix + "1", mid, end)

    build("", 0, full_domain_size)
    return tuple(prefixes)


def _derive_frontier_from_target_prefixes(
    seed: bytes,
    prefix: str,
    targets: Sequence[str],
    start: int,
    end: int,
    params: PPRFParameters,
    frontier: List[StoredSeed],
    cache: Optional[PPRFComputationCache] = None,
) -> None:
    if start >= end:
        return
    if targets[start] == prefix and start + 1 == end:
        frontier.append(StoredSeed(prefix=prefix, seed=seed))
        return

    split_index = len(prefix)
    mid = start
    while mid < end and targets[mid][split_index] == "0":
        mid += 1

    left_seed, right_seed = _expand_seed(seed, params, cache)
    if start < mid:
        _derive_frontier_from_target_prefixes(
            left_seed,
            prefix + "0",
            targets,
            start,
            mid,
            params,
            frontier,
            cache,
        )
    if mid < end:
        _derive_frontier_from_target_prefixes(
            right_seed,
            prefix + "1",
            targets,
            mid,
            end,
            params,
            frontier,
            cache,
        )


def _puncture_prefix_frontier(
    frontier: List[str],
    hole_bits: str,
    message_length: int,
) -> None:
    provider_index = bisect_right(frontier, hole_bits) - 1
    if provider_index < 0:
        return
    provider = frontier[provider_index]
    if not hole_bits.startswith(provider):
        return

    current_prefix = frontier.pop(provider_index)
    while len(current_prefix) < message_length:
        bit = hole_bits[len(current_prefix)]
        if bit == "0":
            insort(frontier, current_prefix + "1")
            current_prefix += "0"
        else:
            insort(frontier, current_prefix + "0")
            current_prefix += "1"


def _eval_targets_from_seed(
    seed: bytes,
    prefix: str,
    targets: Sequence[str],
    params: PPRFParameters,
    output_map: Dict[str, Optional[PRFValue]],
    cache: Optional[PPRFComputationCache] = None,
    *,
    use_leaf_output: bool = True,
) -> None:
    if not targets:
        return
    if len(prefix) == params.message_length:
        output_map[prefix] = _leaf_output(seed, params, cache) if use_leaf_output else seed
        return

    split_index = len(prefix)
    zero_targets: List[str] = []
    one_targets: List[str] = []
    for target in targets:
        if target[split_index] == "0":
            zero_targets.append(target)
        else:
            one_targets.append(target)

    left_seed, right_seed = _expand_seed(seed, params, cache)
    if zero_targets:
        _eval_targets_from_seed(
            left_seed,
            prefix + "0",
            zero_targets,
            params,
            output_map,
            cache,
            use_leaf_output=use_leaf_output,
        )
    if one_targets:
        _eval_targets_from_seed(
            right_seed,
            prefix + "1",
            one_targets,
            params,
            output_map,
            cache,
            use_leaf_output=use_leaf_output,
        )


def _eval_target_range_from_seed(
    seed: bytes,
    prefix: str,
    targets: Sequence[str],
    start: int,
    end: int,
    params: PPRFParameters,
    output_map: Dict[str, Optional[PRFValue]],
    cache: Optional[PPRFComputationCache] = None,
    *,
    use_leaf_output: bool = True,
) -> None:
    if start >= end:
        return
    if len(prefix) == params.message_length:
        output_map[targets[start]] = _leaf_output(seed, params, cache) if use_leaf_output else seed
        return

    split_index = len(prefix)
    mid = start
    while mid < end and targets[mid][split_index] == "0":
        mid += 1

    left_seed, right_seed = _expand_seed(seed, params, cache)
    if start < mid:
        _eval_target_range_from_seed(
            left_seed,
            prefix + "0",
            targets,
            start,
            mid,
            params,
            output_map,
            cache,
            use_leaf_output=use_leaf_output,
        )
    if mid < end:
        _eval_target_range_from_seed(
            right_seed,
            prefix + "1",
            targets,
            mid,
            end,
            params,
            output_map,
            cache,
            use_leaf_output=use_leaf_output,
        )


def _eval_target_range_from_seed_to_list(
    seed: bytes,
    prefix: str,
    targets: Sequence[str],
    start: int,
    end: int,
    params: PPRFParameters,
    outputs: List[Optional[PRFValue]],
    cache: Optional[PPRFComputationCache] = None,
    *,
    use_leaf_output: bool = True,
) -> None:
    if start >= end:
        return
    if len(prefix) == params.message_length:
        outputs[start] = _leaf_output(seed, params, cache) if use_leaf_output else seed
        return

    split_index = len(prefix)
    mid = start
    while mid < end and targets[mid][split_index] == "0":
        mid += 1

    left_seed, right_seed = _expand_seed(seed, params, cache)
    if start < mid:
        _eval_target_range_from_seed_to_list(
            left_seed,
            prefix + "0",
            targets,
            start,
            mid,
            params,
            outputs,
            cache,
            use_leaf_output=use_leaf_output,
        )
    if mid < end:
        _eval_target_range_from_seed_to_list(
            right_seed,
            prefix + "1",
            targets,
            mid,
            end,
            params,
            outputs,
            cache,
            use_leaf_output=use_leaf_output,
        )


def _puncture_targets_from_seed(
    seed: bytes,
    prefix: str,
    holes: Sequence[str],
    params: PPRFParameters,
    frontier: List[StoredSeed],
    output_map: Dict[str, Optional[PRFValue]],
    cache: Optional[PPRFComputationCache] = None,
    *,
    use_leaf_output: bool = True,
) -> None:
    if not holes:
        frontier.append(StoredSeed(prefix=prefix, seed=seed))
        return

    if len(prefix) == params.message_length:
        if prefix in holes:
            output_map[prefix] = _leaf_output(seed, params, cache) if use_leaf_output else seed
        else:
            frontier.append(StoredSeed(prefix=prefix, seed=seed))
        return

    split_index = len(prefix)
    zero_holes: List[str] = []
    one_holes: List[str] = []
    for hole in holes:
        if hole[split_index] == "0":
            zero_holes.append(hole)
        else:
            one_holes.append(hole)

    left_seed, right_seed = _expand_seed(seed, params, cache)
    _puncture_targets_from_seed(
        left_seed,
        prefix + "0",
        zero_holes,
        params,
        frontier,
        output_map,
        cache,
        use_leaf_output=use_leaf_output,
    )
    _puncture_targets_from_seed(
        right_seed,
        prefix + "1",
        one_holes,
        params,
        frontier,
        output_map,
        cache,
        use_leaf_output=use_leaf_output,
    )


def _puncture_target_range_from_seed(
    seed: bytes,
    prefix: str,
    holes: Sequence[str],
    start: int,
    end: int,
    params: PPRFParameters,
    frontier: List[StoredSeed],
    output_map: Dict[str, Optional[PRFValue]],
    cache: Optional[PPRFComputationCache] = None,
    *,
    use_leaf_output: bool = True,
) -> None:
    if start >= end:
        frontier.append(StoredSeed(prefix=prefix, seed=seed))
        return

    if len(prefix) == params.message_length:
        output_map[holes[start]] = _leaf_output(seed, params, cache) if use_leaf_output else seed
        return

    split_index = len(prefix)
    mid = start
    while mid < end and holes[mid][split_index] == "0":
        mid += 1

    left_seed, right_seed = _expand_seed(seed, params, cache)
    _puncture_target_range_from_seed(
        left_seed,
        prefix + "0",
        holes,
        start,
        mid,
        params,
        frontier,
        output_map,
        cache,
        use_leaf_output=use_leaf_output,
    )
    _puncture_target_range_from_seed(
        right_seed,
        prefix + "1",
        holes,
        mid,
        end,
        params,
        frontier,
        output_map,
        cache,
        use_leaf_output=use_leaf_output,
    )


def _puncture_target_range_from_seed_to_list(
    seed: bytes,
    prefix: str,
    holes: Sequence[str],
    start: int,
    end: int,
    params: PPRFParameters,
    frontier: List[StoredSeed],
    outputs: List[Optional[PRFValue]],
    cache: Optional[PPRFComputationCache] = None,
    *,
    use_leaf_output: bool = True,
) -> None:
    if start >= end:
        frontier.append(StoredSeed(prefix=prefix, seed=seed))
        return

    if len(prefix) == params.message_length:
        outputs[start] = _leaf_output(seed, params, cache) if use_leaf_output else seed
        return

    split_index = len(prefix)
    mid = start
    while mid < end and holes[mid][split_index] == "0":
        mid += 1

    left_seed, right_seed = _expand_seed(seed, params, cache)
    _puncture_target_range_from_seed_to_list(
        left_seed,
        prefix + "0",
        holes,
        start,
        mid,
        params,
        frontier,
        outputs,
        cache,
        use_leaf_output=use_leaf_output,
    )
    _puncture_target_range_from_seed_to_list(
        right_seed,
        prefix + "1",
        holes,
        mid,
        end,
        params,
        frontier,
        outputs,
        cache,
        use_leaf_output=use_leaf_output,
    )


def _canonical_cover_from_holes(
    params: PPRFParameters,
    normalized_holes: Sequence[str],
) -> Tuple[str, ...]:
    valid_holes = sorted(
        {
            hole_bits
            for hole_bits in normalized_holes
            if _is_valid_message_bits(hole_bits, params)
        }
    )
    return _canonical_cover_from_sorted_valid_holes(params, valid_holes)


def _canonical_cover_from_sorted_valid_holes(
    params: PPRFParameters,
    valid_holes: Sequence[str],
) -> Tuple[str, ...]:
    frontier = list(_active_domain_prefixes(params.message_length, params.domain_size))
    for hole_bits in valid_holes:
        _puncture_prefix_frontier(frontier, hole_bits, params.message_length)
    return tuple(frontier)


class GGBPPRF:
    """
    A future-facing Python PPRF module whose method names follow the paper syntax:

      PPRF = (PRFSetup, PRFKGen, PRFEval, PPRFPunc)

    This is a canonical GGM-style puncturable PRF. Unlike the original C++
    prototype, it keeps (prefix, seed) pairs explicitly, so different depths do
    not collide in the punctured-key representation.
    """

    @staticmethod
    def PRFSetup(
        security_parameter: int,
        *,
        message_length: int,
        domain_size: Optional[int] = None,
        seed_bits: Optional[int] = None,
        seed_bytes: Optional[int] = None,
        range_bits: Optional[int] = None,
        range_bytes: Optional[int] = None,
        hash_name: str = DEFAULT_HASH_NAME,
        branching_factor: int = 2,
        root_label: bytes = b"GGM/PPRF/root/",
        expand_left_label: bytes = b"GGM/PPRF/expand/0/",
        expand_right_label: bytes = b"GGM/PPRF/expand/1/",
        eval_label: bytes = b"GGM/PPRF/eval/",
    ) -> PPRFParameters:
        """
        pm_PPRF <- PRFSetup(1^kappa)

        In code we pass kappa as an integer rather than unary 1^kappa.
        """

        resolved_seed_bits = resolve_bit_length(
            explicit_bits=seed_bits,
            explicit_bytes=seed_bytes,
            default_bits=security_parameter,
            label="seed",
        )
        resolved_range_bits = resolve_bit_length(
            explicit_bits=range_bits,
            explicit_bytes=range_bytes,
            default_bits=security_parameter,
            label="range",
        )
        return PPRFParameters(
            security_parameter=security_parameter,
            message_length=message_length,
            domain_size=domain_size or 0,
            hash_name=hash_name,
            seed_bits=resolved_seed_bits,
            range_bits=resolved_range_bits,
            branching_factor=branching_factor,
            root_label=root_label,
            expand_left_label=expand_left_label,
            expand_right_label=expand_right_label,
            eval_label=eval_label,
        )

    @staticmethod
    def PRFKGen(
        pm_PPRF: PPRFParameters,
        seed: Optional[SeedInput] = None,
        *,
        layer: int = 0,
    ) -> PPRFKey:
        """
        k <- PRFKGen(pm_PPRF)

        If seed is omitted, a fresh random master seed is sampled. If seed is
        provided, the root seed is derived deterministically from seed || layer,
        which mirrors the constructor style used in the original C++ file.
        """

        if seed is None:
            root_seed = derive_parameter(
                pm_PPRF.root_label,
                seed=None,
                output_bits=pm_PPRF.seed_bits,
                hash_name=pm_PPRF.hash_name,
            )
        else:
            seed_material = normalize_to_bytes(seed) + layer.to_bytes(4, "big")
            root_seed = hash_bytes(
                pm_PPRF.root_label + seed_material,
                output_bits=pm_PPRF.seed_bits,
                hash_name=pm_PPRF.hash_name,
            )
        active_prefixes = _active_domain_prefixes(
            pm_PPRF.message_length,
            pm_PPRF.domain_size,
        )
        if active_prefixes == ("",):
            return _new_pprf_key_unchecked(pm_PPRF, (StoredSeed(prefix="", seed=root_seed),))

        frontier: List[StoredSeed] = []
        _derive_frontier_from_target_prefixes(
            root_seed,
            "",
            active_prefixes,
            0,
            len(active_prefixes),
            pm_PPRF,
            frontier,
            PPRFComputationCache(),
        )
        return _new_pprf_key_unchecked(pm_PPRF, frontier)

    @staticmethod
    def PRFEval(
        k: PPRFKey,
        x: MessageInput,
        *,
        cache: Optional[PPRFComputationCache] = None,
    ) -> Optional[PRFValue]:
        """
        r <- PRFEval(k, x)

        Returns None as the concrete realization of the symbol bot when x lies
        outside Dom(k), i.e. when x has been punctured.
        """

        x_bits = normalize_bitstring(x, k.params.message_length)
        provider_index = _find_provider(k.frontier, x_bits)
        if provider_index is None:
            return None

        provider = k.frontier[provider_index]
        current_seed = provider.seed
        for bit in x_bits[len(provider.prefix) :]:
            left_seed, right_seed = _expand_seed(current_seed, k.params, cache)
            current_seed = left_seed if bit == "0" else right_seed
        return _leaf_output(current_seed, k.params, cache)

    @staticmethod
    def PRFEvalMany(
        k: PPRFKey,
        messages: Iterable[MessageInput],
        *,
        cache: Optional[PPRFComputationCache] = None,
        inputs_normalized: bool = False,
    ) -> Tuple[Optional[PRFValue], ...]:
        """
        Vectorized PRFEval that shares GGM path expansions across all targets.

        The returned tuple matches the input order. A component is None exactly
        when the corresponding message lies outside Dom(k).
        """

        normalized_messages = _normalize_message_list(
            messages,
            k.params.message_length,
            inputs_normalized=inputs_normalized,
        )
        if not normalized_messages:
            return ()

        unique_messages = tuple(sorted(set(normalized_messages)))
        output_map: Dict[str, Optional[PRFValue]] = {
            message_bits: None for message_bits in unique_messages
        }
        for provider in k.frontier:
            start, end = _sorted_target_range_for_prefix(
                unique_messages,
                provider.prefix,
                k.params.message_length,
            )
            if start >= end:
                continue
            _eval_target_range_from_seed(
                provider.seed,
                provider.prefix,
                unique_messages,
                start,
                end,
                k.params,
                output_map,
                cache,
            )

        return tuple(output_map[message_bits] for message_bits in normalized_messages)

    @staticmethod
    def LeafMaterialEval(
        k: PPRFKey,
        x: MessageInput,
        *,
        cache: Optional[PPRFComputationCache] = None,
    ) -> Optional[bytes]:
        x_bits = normalize_bitstring(x, k.params.message_length)
        provider_index = _find_provider(k.frontier, x_bits)
        if provider_index is None:
            return None

        provider = k.frontier[provider_index]
        current_seed = provider.seed
        for bit in x_bits[len(provider.prefix) :]:
            left_seed, right_seed = _expand_seed(current_seed, k.params, cache)
            current_seed = left_seed if bit == "0" else right_seed
        return current_seed

    @staticmethod
    def LeafMaterialMany(
        k: PPRFKey,
        messages: Iterable[MessageInput],
        *,
        cache: Optional[PPRFComputationCache] = None,
        inputs_normalized: bool = False,
        inputs_sorted_unique: bool = False,
        inputs_trusted: bool = False,
    ) -> Tuple[Optional[bytes], ...]:
        normalized_messages = _normalize_message_list(
            messages,
            k.params.message_length,
            inputs_normalized=inputs_normalized,
            inputs_trusted=inputs_trusted,
        )
        if not normalized_messages:
            return ()

        if inputs_sorted_unique or _messages_are_sorted_unique(normalized_messages):
            outputs: List[Optional[PRFValue]] = [None] * len(normalized_messages)
            for provider in k.frontier:
                start, end = _sorted_target_range_for_prefix(
                    normalized_messages,
                    provider.prefix,
                    k.params.message_length,
                )
                if start >= end:
                    continue
                _eval_target_range_from_seed_to_list(
                    provider.seed,
                    provider.prefix,
                    normalized_messages,
                    start,
                    end,
                    k.params,
                    outputs,
                    cache,
                    use_leaf_output=False,
                )
            return tuple(outputs)

        unique_messages = tuple(sorted(set(normalized_messages)))
        output_map: Dict[str, Optional[PRFValue]] = {
            message_bits: None for message_bits in unique_messages
        }
        for provider in k.frontier:
            start, end = _sorted_target_range_for_prefix(
                unique_messages,
                provider.prefix,
                k.params.message_length,
            )
            if start >= end:
                continue
            _eval_target_range_from_seed(
                provider.seed,
                provider.prefix,
                unique_messages,
                start,
                end,
                k.params,
                output_map,
                cache,
                use_leaf_output=False,
            )

        return tuple(output_map[message_bits] for message_bits in normalized_messages)

    @staticmethod
    def PPRFPunc(
        k: PPRFKey,
        x: MessageInput,
        *,
        cache: Optional[PPRFComputationCache] = None,
    ) -> PPRFKey:
        """
        k' <- PPRFPunc(k, x)

        The output is the canonical minimal frontier for Dom(k) \\ {x}.
        Puncturing an already punctured point is idempotent.
        """

        x_bits = normalize_bitstring(x, k.params.message_length)
        frontier = list(k.frontier)
        provider_index = _find_provider(frontier, x_bits)
        if provider_index is None:
            return k

        provider = frontier.pop(provider_index)
        current_seed = provider.seed
        current_prefix = provider.prefix

        for bit in x_bits[len(current_prefix) :]:
            left_seed, right_seed = _expand_seed(current_seed, k.params, cache)
            if bit == "0":
                frontier.append(StoredSeed(prefix=current_prefix + "1", seed=right_seed))
                current_seed = left_seed
                current_prefix += "0"
            else:
                frontier.append(StoredSeed(prefix=current_prefix + "0", seed=left_seed))
                current_seed = right_seed
                current_prefix += "1"

        frontier.sort(key=lambda node: node.prefix)
        return _new_pprf_key_unchecked(k.params, frontier)

    @staticmethod
    def PunctureAndEvalMany(
        k: PPRFKey,
        holes: Iterable[MessageInput],
        *,
        cache: Optional[PPRFComputationCache] = None,
        inputs_normalized: bool = False,
    ) -> Tuple[PPRFKey, Tuple[Optional[PRFValue], ...]]:
        """
        Batch puncture operation for YCSig-style workloads.

        It returns the canonical punctured key together with the PRF outputs on
        the punctured messages under the original key. Both results are derived
        in one shared traversal of the GGM tree.
        """

        normalized_holes = _normalize_message_list(
            holes,
            k.params.message_length,
            inputs_normalized=inputs_normalized,
        )
        if not normalized_holes:
            return k, ()

        unique_holes = tuple(sorted(set(normalized_holes)))
        output_map: Dict[str, Optional[PRFValue]] = {
            hole_bits: None for hole_bits in unique_holes
        }
        frontier: List[StoredSeed] = []
        for provider in k.frontier:
            start, end = _sorted_target_range_for_prefix(
                unique_holes,
                provider.prefix,
                k.params.message_length,
            )
            if start >= end:
                frontier.append(provider)
                continue
            _puncture_target_range_from_seed(
                provider.seed,
                provider.prefix,
                unique_holes,
                start,
                end,
                k.params,
                frontier,
                output_map,
                cache,
            )

        punctured_key = _new_pprf_key_unchecked(k.params, frontier)
        return punctured_key, tuple(output_map[hole_bits] for hole_bits in normalized_holes)

    @staticmethod
    def PunctureAndRevealLeafMaterialMany(
        k: PPRFKey,
        holes: Iterable[MessageInput],
        *,
        cache: Optional[PPRFComputationCache] = None,
        inputs_normalized: bool = False,
        inputs_sorted_unique: bool = False,
        inputs_trusted: bool = False,
    ) -> Tuple[PPRFKey, Tuple[Optional[bytes], ...]]:
        normalized_holes = _normalize_message_list(
            holes,
            k.params.message_length,
            inputs_normalized=inputs_normalized,
            inputs_trusted=inputs_trusted,
        )
        if not normalized_holes:
            return k, ()

        if inputs_sorted_unique or _messages_are_sorted_unique(normalized_holes):
            outputs: List[Optional[PRFValue]] = [None] * len(normalized_holes)
            frontier: List[StoredSeed] = []
            for provider in k.frontier:
                start, end = _sorted_target_range_for_prefix(
                    normalized_holes,
                    provider.prefix,
                    k.params.message_length,
                )
                if start >= end:
                    frontier.append(provider)
                    continue
                _puncture_target_range_from_seed_to_list(
                    provider.seed,
                    provider.prefix,
                    normalized_holes,
                    start,
                    end,
                    k.params,
                    frontier,
                    outputs,
                    cache,
                    use_leaf_output=False,
                )

            punctured_key = _new_pprf_key_unchecked(k.params, frontier)
            return punctured_key, tuple(outputs)

        unique_holes = tuple(sorted(set(normalized_holes)))
        output_map: Dict[str, Optional[PRFValue]] = {
            hole_bits: None for hole_bits in unique_holes
        }
        frontier: List[StoredSeed] = []
        for provider in k.frontier:
            start, end = _sorted_target_range_for_prefix(
                unique_holes,
                provider.prefix,
                k.params.message_length,
            )
            if start >= end:
                frontier.append(provider)
                continue
            _puncture_target_range_from_seed(
                provider.seed,
                provider.prefix,
                unique_holes,
                start,
                end,
                k.params,
                frontier,
                output_map,
                cache,
                use_leaf_output=False,
            )

        punctured_key = _new_pprf_key_unchecked(k.params, frontier)
        return punctured_key, tuple(output_map[hole_bits] for hole_bits in normalized_holes)

    @staticmethod
    def PPRFGetMsg(k: PPRFKey, i: int) -> str:
        """
        x <- PPRFGetMsg(k, i)

        Returns the i-th message in Dom(k) under lexicographic order on
        MessageSpace_PPRF. The index i is 1-based, mirroring the original C++
        helper GetMsg.
        """

        if i <= 0:
            raise ValueError("i must be a positive 1-based index")

        remaining = i
        for node in sorted(k.frontier, key=lambda item: item.prefix):
            block_size = _frontier_block_size(node.prefix, k.params.message_length)
            if remaining <= block_size:
                suffix_len = k.params.message_length - len(node.prefix)
                suffix_value = remaining - 1
                suffix = f"{suffix_value:0{suffix_len}b}" if suffix_len > 0 else ""
                return node.prefix + suffix
            remaining -= block_size

        raise IndexError("i exceeds |Dom(k)|")

    @staticmethod
    def DomainSize(k: PPRFKey) -> int:
        return sum(_frontier_block_size(node.prefix, k.params.message_length) for node in k.frontier)

    @staticmethod
    def Dom(k: PPRFKey, *, max_size: int = 1 << 20) -> FrozenSet[str]:
        """
        Enumerate Dom(k) exactly for small message spaces.

        This is mainly for testing/auditing. The frontier itself is the scalable
        representation of the domain.
        """

        size = GGBPPRF.DomainSize(k)
        if size > max_size:
            raise ValueError(
                f"Dom(k) has size {size}, which exceeds the configured max_size={max_size}"
            )

        messages = []
        for node in sorted(k.frontier, key=lambda item: item.prefix):
            suffix_len = k.params.message_length - len(node.prefix)
            for suffix_value in range(1 << suffix_len):
                suffix = f"{suffix_value:0{suffix_len}b}" if suffix_len > 0 else ""
                messages.append(node.prefix + suffix)
        return frozenset(messages)

    @staticmethod
    def EncLen(k: PPRFKey) -> int:
        return len(k.serialize())

    @staticmethod
    def MinKey(master_key: PPRFKey, hole_set: Iterable[MessageInput]) -> PPRFKey:
        punctured_key, _ = GGBPPRF.PunctureAndEvalMany(master_key, hole_set)
        return punctured_key

    @staticmethod
    def MinLen(master_key: PPRFKey, hole_set: Iterable[MessageInput]) -> int:
        return GGBPPRF.EncLen(GGBPPRF.MinKey(master_key, hole_set))

    @staticmethod
    def MinKeys(master_key: PPRFKey, hole_set: Iterable[MessageInput]) -> Tuple[PPRFKey, ...]:
        """
        In this concrete canonical implementation the minimal punctured key is unique.
        """

        return (GGBPPRF.MinKey(master_key, hole_set),)

    @staticmethod
    def TestMinKey(
        *args: object,
    ) -> bool:
        """
        TestMinKey(X, k') or TestMinKey(k, X, k').

        The 2-argument form is the verifier-side structural audit suggested in
        the preliminaries: it checks whether the stored frontier is exactly the
        canonical maximal cover of MessageSpace_PPRF \\ X.

        The 3-argument form additionally checks equality against the unique
        canonical punctured key derived from the provided master key.
        """

        if len(args) == 2:
            hole_set, punctured_key = args
            if not isinstance(punctured_key, PPRFKey):
                raise TypeError("the 2-argument form expects TestMinKey(hole_set, punctured_key)")

            normalized_holes = sorted(
                {normalize_bitstring(x, punctured_key.params.message_length) for x in hole_set}
            )
            expected_prefixes = _canonical_cover_from_holes(
                punctured_key.params,
                normalized_holes,
            )
            actual_prefixes = tuple(node.prefix for node in punctured_key.frontier)
            return actual_prefixes == expected_prefixes

        if len(args) == 3:
            master_key, hole_set, punctured_key = args
            if not isinstance(master_key, PPRFKey) or not isinstance(punctured_key, PPRFKey):
                raise TypeError(
                    "the 3-argument form expects TestMinKey(master_key, hole_set, punctured_key)"
                )
            expected = GGBPPRF.MinKey(master_key, hole_set)
            return punctured_key.params == expected.params and punctured_key.frontier == expected.frontier

        raise TypeError("TestMinKey expects either 2 or 3 positional arguments")

    @staticmethod
    def CanonicalPrefixes(
        pm_PPRF: PPRFParameters,
        hole_set: Iterable[MessageInput],
        *,
        inputs_normalized: bool = False,
    ) -> Tuple[str, ...]:
        """
        Return the canonical frontier prefixes for MessageSpace_PPRF \\ hole_set.
        """

        normalized_holes = sorted(
            _normalize_message_list(
                hole_set,
                pm_PPRF.message_length,
                inputs_normalized=inputs_normalized,
            )
        )
        return _canonical_cover_from_holes(pm_PPRF, normalized_holes)

    @staticmethod
    def CompactPuncturedKey(k: PPRFKey) -> Tuple[bytes, ...]:
        """
        Return the canonical compact representation of a punctured key.

        The frontier prefixes are omitted; callers are expected to reconstruct
        them from the hole set.
        """

        return tuple(node.seed for node in k.frontier)

    @staticmethod
    def ExpandPuncturedKey(
        pm_PPRF: PPRFParameters,
        hole_set: Iterable[MessageInput],
        stored_seeds: Sequence[bytes],
        *,
        inputs_normalized: bool = False,
    ) -> PPRFKey:
        """
        Reconstruct a punctured key from the hole set and the frontier seeds.
        """

        prefixes = GGBPPRF.CanonicalPrefixes(
            pm_PPRF,
            hole_set,
            inputs_normalized=inputs_normalized,
        )
        if len(prefixes) != len(stored_seeds):
            raise ValueError("stored_seeds does not match the canonical frontier size")

        frontier = []
        for prefix, seed in zip(prefixes, stored_seeds):
            if len(seed) != pm_PPRF.seed_bytes:
                raise ValueError("stored seed has incorrect byte length")
            frontier.append(StoredSeed(prefix=prefix, seed=seed))
        return _new_pprf_key_unchecked(pm_PPRF, frontier)

    @staticmethod
    def enumerate_outputs(
        k: PPRFKey,
        *,
        max_size: int = 1 << 16,
    ) -> Iterator[Tuple[str, PRFValue]]:
        """
        Helper for tests: iterate over x in Dom(k) and return (x, PRFEval(k, x)).
        """

        for x_bits in sorted(GGBPPRF.Dom(k, max_size=max_size)):
            value = GGBPPRF.PRFEval(k, x_bits)
            if value is None:
                raise AssertionError("Dom(k) contained a punctured point")
            yield x_bits, value


PPRF = GGBPPRF
