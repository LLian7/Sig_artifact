from __future__ import annotations

import hashlib
from bisect import bisect_left
import math
from dataclasses import dataclass, field
from functools import lru_cache
from typing import Dict, Iterable, Mapping, Optional, Sequence, Tuple, Union

from crypto_utils import bits_to_bytes, hash_bytes, truncate_to_bits
from operation_counter import enabled as counters_enabled, increment
from tweakable_hash import DEFAULT_HASH_NAME, SUPPORTED_HASHES, THParameters, TwH


LeafInput = Union[str, bytes, int]
HashValue = bytes
Position = Tuple[int, int]


@dataclass(frozen=True)
class MTParameters:
    """
    Public parameters pm_MT output by MTSetup(1^kappa).

    Python uses 0-based leaf indices {0, ..., leaf_count - 1}. The hash output
    length is exactly kappa bits. The underlying tweakable hash is the
    realization TwH(PP, Tweak, m) = H(PP || Tweak || m).
    """

    security_parameter: int
    leaf_count: int
    th_params: THParameters
    padding_leaf: bytes = b""
    leaf_tweak_label: bytes = b"leaf/"
    node_tweak_label: bytes = b"node/"
    padding_tweak_label: bytes = b"padding/"
    _hash_output_bytes: int = field(init=False, repr=False)
    _backend_counter: str = field(init=False, repr=False)
    _fast_backend: Optional[object] = field(init=False, repr=False)
    _fast_uses_xof: bool = field(init=False, repr=False)

    def __post_init__(self) -> None:
        if self.security_parameter <= 0:
            raise ValueError("security_parameter must be positive")
        if self.leaf_count <= 0:
            raise ValueError("leaf_count must be positive")
        if self.th_params.security_parameter != self.security_parameter:
            raise ValueError("MT security_parameter must match the tweakable hash security parameter")
        if self.th_params.hash_name not in SUPPORTED_HASHES:
            raise ValueError(
                f"unsupported hash_name={self.th_params.hash_name!r}; choose from {sorted(SUPPORTED_HASHES)}"
            )
        hash_output_bytes = bits_to_bytes(self.security_parameter)
        backend_counter = f"hash.backend_calls.{self.th_params.hash_name}"
        fast_backend = None
        fast_uses_xof = False
        if self.security_parameter == 8 * hash_output_bytes:
            if self.th_params.hash_name == "shake_128":
                fast_backend = hashlib.shake_128
                fast_uses_xof = True
            elif self.th_params.hash_name == "shake_256":
                fast_backend = hashlib.shake_256
                fast_uses_xof = True
            elif self.th_params.hash_name == "sha3_256" and hash_output_bytes <= hashlib.sha3_256().digest_size:
                fast_backend = hashlib.sha3_256
            elif self.th_params.hash_name == "sha3_512" and hash_output_bytes <= hashlib.sha3_512().digest_size:
                fast_backend = hashlib.sha3_512
        object.__setattr__(self, "_hash_output_bytes", hash_output_bytes)
        object.__setattr__(self, "_backend_counter", backend_counter)
        object.__setattr__(self, "_fast_backend", fast_backend)
        object.__setattr__(self, "_fast_uses_xof", fast_uses_xof)
        pp = self.th_params.public_parameter
        object.__setattr__(
            self,
            "_leaf_hash_prefixes",
            tuple(
                pp + self.leaf_tweak_label + index.to_bytes(8, "big")
                for index in range(self.leaf_count)
            ),
        )
        object.__setattr__(
            self,
            "_padding_hash_prefixes",
            tuple(
                pp + self.padding_tweak_label + index.to_bytes(8, "big")
                for index in range(self.leaf_count)
            ),
        )
        object.__setattr__(
            self,
            "_node_hash_prefixes",
            tuple(
                tuple(
                    pp + self.node_tweak_label + level.to_bytes(4, "big") + offset.to_bytes(8, "big")
                    for offset in range(width)
                )
                for level, width in enumerate(self.level_widths)
            ),
        )

    @property
    def output_bytes(self) -> int:
        return self._hash_output_bytes

    @property
    def hash_name(self) -> str:
        return self.th_params.hash_name

    @property
    def public_parameter(self) -> bytes:
        return self.th_params.public_parameter

    @property
    def padded_leaf_count(self) -> int:
        return self.leaf_count

    @property
    def tree_height(self) -> int:
        return len(_level_spans(self.leaf_count)) - 1

    @property
    def level_widths(self) -> Tuple[int, ...]:
        return _level_widths(self.leaf_count)


@dataclass(frozen=True)
class MerkleTree:
    params: MTParameters
    leaves: Tuple[bytes, ...]
    levels: Tuple[Tuple[bytes, ...], ...]
    root: bytes


@dataclass(frozen=True)
class PartialStateEntry:
    """
    Entry ((level, offset), value) in the partial internal state ST_MT.

    When level == 0, value is the raw leaf value leaf_i.
    When level >= 1, value is the subtree root hash at that position.
    """

    position: Position
    value: bytes


@dataclass(frozen=True)
class PartialMerkleState:
    params: MTParameters
    entries: Tuple[PartialStateEntry, ...]

    def __post_init__(self) -> None:
        sorted_entries = tuple(
            sorted(self.entries, key=lambda entry: (entry.position[0], entry.position[1]))
        )
        object.__setattr__(self, "entries", sorted_entries)

        seen_positions = set()
        for entry in sorted_entries:
            level, offset = entry.position
            if level < 0 or level > self.params.tree_height:
                raise ValueError("entry level is out of range")
            if offset < 0 or offset >= self.params.level_widths[level]:
                raise ValueError("entry offset is out of range")
            if entry.position in seen_positions:
                raise ValueError("duplicate positions in PartialMerkleState")
            if level >= 1 and len(entry.value) != self.params.output_bytes:
                raise ValueError("internal-state hash length does not match the security parameter")
            seen_positions.add(entry.position)


def _new_partial_state_unchecked(
    params: MTParameters,
    entries: Sequence[PartialStateEntry],
) -> PartialMerkleState:
    state = object.__new__(PartialMerkleState)
    object.__setattr__(state, "params", params)
    object.__setattr__(state, "entries", tuple(entries))
    return state


def _normalize_leaf(leaf: LeafInput) -> bytes:
    if isinstance(leaf, bytes):
        return leaf
    if isinstance(leaf, str):
        return leaf.encode("utf-8")
    if isinstance(leaf, int):
        if leaf < 0:
            raise ValueError("leaf integers must be non-negative")
        width = max(1, (leaf.bit_length() + 7) // 8)
        return leaf.to_bytes(width, "big")
    raise TypeError("leaf must be bytes, str, or non-negative integer")


def _normalize_index_set(index_set: Iterable[int], leaf_count: int) -> Tuple[int, ...]:
    normalized = tuple(sorted(set(index_set)))
    for index in normalized:
        if index < 0 or index >= leaf_count:
            raise ValueError("leaf index is out of range")
    return normalized


def _normalize_indexed_leaves(
    indexed_leaves: Mapping[int, LeafInput],
    leaf_count: int,
) -> Dict[int, bytes]:
    normalized: Dict[int, bytes] = {}
    for index, leaf in indexed_leaves.items():
        if index < 0 or index >= leaf_count:
            raise ValueError("leaf index is out of range")
        normalized[index] = _normalize_leaf(leaf)
    return normalized


def _tweak_hash(pm_MT: MTParameters, tweak: bytes, message: bytes) -> bytes:
    return TwH.TweakHEval(pm_MT.th_params, tweak, message)


def _hash_with_prefix(pm_MT: MTParameters, prefix: bytes, message: bytes) -> bytes:
    counting = counters_enabled()
    if counting:
        increment("tweak_hash.eval")
    backend = pm_MT._fast_backend
    payload = prefix + message
    if backend is None:
        return hash_bytes(
            payload,
            output_bits=pm_MT.security_parameter,
            hash_name=pm_MT.hash_name,
        )
    if counting:
        increment("hash.backend_calls")
        increment(pm_MT._backend_counter)
    hash_object = backend(payload)
    if pm_MT._fast_uses_xof:
        return hash_object.digest(pm_MT.output_bytes)
    raw = hash_object.digest()[: pm_MT.output_bytes]
    if pm_MT.security_parameter == 8 * pm_MT.output_bytes:
        return raw
    return truncate_to_bits(raw, pm_MT.security_parameter)


@lru_cache(maxsize=None)
def _level_spans(leaf_count: int) -> Tuple[Tuple[Tuple[int, int], ...], ...]:
    current = tuple((index, index + 1) for index in range(leaf_count))
    levels = [current]
    while len(current) > 1:
        next_level = []
        for offset in range(0, len(current), 2):
            if offset + 1 < len(current):
                next_level.append((current[offset][0], current[offset + 1][1]))
            else:
                next_level.append(current[offset])
        current = tuple(next_level)
        levels.append(current)
    return tuple(levels)


@lru_cache(maxsize=None)
def _level_widths(leaf_count: int) -> Tuple[int, ...]:
    return tuple(len(level) for level in _level_spans(leaf_count))


def _leaf_tweak_with_label(pm_MT: MTParameters, index: int) -> bytes:
    return pm_MT.leaf_tweak_label + index.to_bytes(8, "big")


def _node_tweak_with_label(pm_MT: MTParameters, position: Position) -> bytes:
    level, offset = position
    return pm_MT.node_tweak_label + level.to_bytes(4, "big") + offset.to_bytes(8, "big")


def _leaf_hash(pm_MT: MTParameters, index: int, leaf: bytes) -> bytes:
    increment("merkle.leaf_hash")
    return _hash_with_prefix(pm_MT, pm_MT._leaf_hash_prefixes[index], leaf)


def _padding_leaf_hash(pm_MT: MTParameters, index: int) -> bytes:
    increment("merkle.padding_hash")
    return _hash_with_prefix(pm_MT, pm_MT._padding_hash_prefixes[index], pm_MT.padding_leaf)


def _internal_hash(pm_MT: MTParameters, position: Position, left: bytes, right: bytes) -> bytes:
    increment("merkle.internal_hash")
    level, offset = position
    return _hash_with_prefix(pm_MT, pm_MT._node_hash_prefixes[level][offset], left + right)


def _children(pm_MT: MTParameters, position: Position) -> Tuple[Position, Optional[Position]]:
    level, offset = position
    if level == 0:
        raise ValueError("leaf positions do not have children")
    child_level = level - 1
    left_child = (child_level, 2 * offset)
    right_offset = 2 * offset + 1
    right_child = (
        (child_level, right_offset)
        if right_offset < pm_MT.level_widths[child_level]
        else None
    )
    return left_child, right_child


def _parent(pm_MT: MTParameters, position: Position) -> Optional[Position]:
    level, offset = position
    if level >= pm_MT.tree_height:
        return None
    return (level + 1, offset // 2)


def _is_ancestor_or_equal(ancestor: Position, descendant: Position) -> bool:
    level_a, offset_a = ancestor
    level_d, offset_d = descendant
    if level_a < level_d:
        return False
    return offset_a == (offset_d >> (level_a - level_d))


def _all_positions(pm_MT: MTParameters) -> Tuple[Position, ...]:
    positions = []
    for level, width in enumerate(pm_MT.level_widths):
        for offset in range(width):
            positions.append((level, offset))
    return tuple(positions)


def _selected_count_in_span(
    normalized_index_set: Sequence[int],
    start: int,
    end: int,
) -> int:
    left = bisect_left(normalized_index_set, start)
    right = bisect_left(normalized_index_set, end, lo=left)
    return right - left


def _canonical_cover_positions_for_indices(
    pm_MT: MTParameters,
    normalized_index_set: Sequence[int],
) -> Tuple[Position, ...]:
    if not normalized_index_set:
        return ()

    spans = _level_spans(pm_MT.leaf_count)
    level_positions = [[] for _ in range(pm_MT.tree_height + 1)]

    def visit(position: Position, start_index: int, end_index: int) -> None:
        if start_index >= end_index:
            return
        level, offset = position
        start, end = spans[level][offset]
        if end_index - start_index == end - start or level == 0:
            level_positions[level].append(position)
            return

        left_child, right_child = _children(pm_MT, position)
        if right_child is None:
            visit(left_child, start_index, end_index)
            return

        left_end = spans[left_child[0]][left_child[1]][1]
        mid_index = bisect_left(
            normalized_index_set,
            left_end,
            start_index,
            end_index,
        )
        if start_index < mid_index:
            visit(left_child, start_index, mid_index)
        if mid_index < end_index:
            visit(right_child, mid_index, end_index)

    visit((pm_MT.tree_height, 0), 0, len(normalized_index_set))
    return tuple(position for positions in level_positions for position in positions)


def _build_subtree_root(
    pm_MT: MTParameters,
    indexed_leaves: Mapping[int, bytes],
    position: Position,
    *,
    leaves_hashed: bool = False,
) -> bytes:
    level, offset = position
    if level == 0:
        if offset not in indexed_leaves:
            raise ValueError("missing leaf required to build the requested subtree")
        if leaves_hashed:
            return indexed_leaves[offset]
        return _leaf_hash(pm_MT, offset, indexed_leaves[offset])

    left_child, right_child = _children(pm_MT, position)
    left_value = _build_subtree_root(
        pm_MT,
        indexed_leaves,
        left_child,
        leaves_hashed=leaves_hashed,
    )
    if right_child is None:
        return left_value
    right_value = _build_subtree_root(
        pm_MT,
        indexed_leaves,
        right_child,
        leaves_hashed=leaves_hashed,
    )
    return _internal_hash(pm_MT, position, left_value, right_value)


class HashBasedMTPIS:
    """
    Hash-based Merkle tree / MTPIS implementation aligned with the preliminaries:

      MT = (MTSetup, MTBuild, MTIntNGen, MTSparseBuild)

    The underlying tweakable hash is realized using SHAKE or SHA3 with output
    length determined by the security parameter kappa.
    """

    @staticmethod
    def MTSetup(
        security_parameter: int,
        *,
        leaf_count: int,
        hash_name: str = DEFAULT_HASH_NAME,
        public_seed: Optional[bytes] = None,
        public_parameter: Optional[bytes] = None,
        public_parameter_bits: Optional[int] = None,
        public_parameter_bytes: Optional[int] = None,
        padding_leaf: bytes = b"",
        leaf_tweak_label: bytes = b"leaf/",
        node_tweak_label: bytes = b"node/",
        padding_tweak_label: bytes = b"padding/",
    ) -> MTParameters:
        """
        pm_MT <- MTSetup(1^kappa)
        """

        th_params = TwH.TweakHSetup(
            security_parameter,
            hash_name=hash_name,
            public_seed=public_seed,
            public_parameter=public_parameter,
            public_parameter_bits=public_parameter_bits,
            public_parameter_bytes=public_parameter_bytes,
        )
        return MTParameters(
            security_parameter=security_parameter,
            leaf_count=leaf_count,
            th_params=th_params,
            padding_leaf=padding_leaf,
            leaf_tweak_label=leaf_tweak_label,
            node_tweak_label=node_tweak_label,
            padding_tweak_label=padding_tweak_label,
        )

    @staticmethod
    def Desc(pm_MT: MTParameters, position: Position) -> Tuple[int, ...]:
        """
        Desc(level, offset) as defined in the preliminaries.
        """

        level, offset = position
        if level < 0 or level > pm_MT.tree_height:
            raise ValueError("position level is out of range")
        if offset < 0 or offset >= pm_MT.level_widths[level]:
            raise ValueError("position offset is out of range")

        start, end = _level_spans(pm_MT.leaf_count)[level][offset]
        return tuple(range(start, end))

    @staticmethod
    def MaxCov(pm_MT: MTParameters, leaf_index_set: Iterable[int], position: Position) -> bool:
        """
        Predicate MaxCov_{I}(position).
        """

        index_set = set(_normalize_index_set(leaf_index_set, pm_MT.leaf_count))
        desc = set(HashBasedMTPIS.Desc(pm_MT, position))
        if not desc or not desc.issubset(index_set):
            return False

        parent = _parent(pm_MT, position)
        if parent is None:
            return True
        parent_desc = set(HashBasedMTPIS.Desc(pm_MT, parent))
        return not parent_desc.issubset(index_set)

    @staticmethod
    def MTBuild(
        pm_MT: MTParameters,
        leaves: Sequence[LeafInput],
        *,
        leaves_hashed: bool = False,
    ) -> MerkleTree:
        """
        Tr <- MTBuild({leaf_i}_{i in [leaf_count]})
        """

        if len(leaves) != pm_MT.leaf_count:
            raise ValueError(
                f"expected exactly {pm_MT.leaf_count} leaves, received {len(leaves)}"
            )

        normalized_leaves = tuple(_normalize_leaf(leaf) for leaf in leaves)
        if leaves_hashed:
            current_level = list(normalized_leaves)
        else:
            current_level = [
                _leaf_hash(pm_MT, index, leaf)
                for index, leaf in enumerate(normalized_leaves)
            ]
        levels = [tuple(current_level)]
        for level in range(1, pm_MT.tree_height + 1):
            next_level = []
            for offset in range(0, len(current_level), 2):
                if offset + 1 < len(current_level):
                    parent_position = (level, offset // 2)
                    parent_value = _internal_hash(
                        pm_MT,
                        parent_position,
                        current_level[offset],
                        current_level[offset + 1],
                    )
                    next_level.append(parent_value)
                else:
                    next_level.append(current_level[offset])
            current_level = next_level
            levels.append(tuple(current_level))

        return MerkleTree(
            params=pm_MT,
            leaves=normalized_leaves,
            levels=tuple(levels),
            root=levels[-1][0],
        )

    @staticmethod
    def MTIntNGen(
        pm_MT: MTParameters,
        leaf_index_set: Iterable[int],
        indexed_leaves: Mapping[int, LeafInput],
        *,
        leaves_hashed: bool = False,
    ) -> PartialMerkleState:
        """
        ST_MT <- MTIntNGen(leaf_count, I, {leaf_i}_{i in I})
        """

        normalized_index_set = _normalize_index_set(leaf_index_set, pm_MT.leaf_count)
        normalized_leaves = _normalize_indexed_leaves(indexed_leaves, pm_MT.leaf_count)
        if set(normalized_index_set) != set(normalized_leaves):
            raise ValueError("indexed_leaves must contain exactly the leaves indexed by leaf_index_set")

        positions = _canonical_cover_positions_for_indices(pm_MT, normalized_index_set)
        entries = []
        for position in positions:
            level, offset = position
            if level == 0:
                entries.append(PartialStateEntry(position=position, value=normalized_leaves[offset]))
                continue
            entries.append(
                PartialStateEntry(
                    position=position,
                    value=_build_subtree_root(
                        pm_MT,
                        normalized_leaves,
                        position,
                        leaves_hashed=leaves_hashed,
                    ),
                )
            )

        return _new_partial_state_unchecked(pm_MT, tuple(entries))

    @staticmethod
    def CanonicalStatePositions(
        pm_MT: MTParameters,
        leaf_index_set: Iterable[int],
    ) -> Tuple[Position, ...]:
        """
        Return the canonical entry positions of ST_MT for the given leaf index set.
        """

        normalized_index_set = _normalize_index_set(leaf_index_set, pm_MT.leaf_count)
        return _canonical_cover_positions_for_indices(pm_MT, normalized_index_set)

    @staticmethod
    def CompactStateValues(partial_state: PartialMerkleState) -> Tuple[bytes, ...]:
        """
        Return the compact Merkle partial state representation without positions.
        """

        return tuple(entry.value for entry in partial_state.entries)

    @staticmethod
    def ExpandPartialState(
        pm_MT: MTParameters,
        leaf_index_set: Iterable[int],
        values: Sequence[bytes],
    ) -> PartialMerkleState:
        """
        Reconstruct a canonical partial state from derivable positions and values.
        """

        positions = HashBasedMTPIS.CanonicalStatePositions(pm_MT, leaf_index_set)
        if len(positions) != len(values):
            raise ValueError("values does not match the canonical partial-state size")

        entries = tuple(
            PartialStateEntry(position=position, value=value)
            for position, value in zip(positions, values)
        )
        return _new_partial_state_unchecked(pm_MT, entries)

    @staticmethod
    def IsMergeable(
        pm_MT: MTParameters,
        partial_state: PartialMerkleState,
        leaf_index_set: Iterable[int],
    ) -> bool:
        """
        Predicate IsMergeable(ST_MT, I).
        """

        if partial_state.params != pm_MT:
            raise ValueError("partial_state was generated under different parameters")

        index_set = set(_normalize_index_set(leaf_index_set, pm_MT.leaf_count))
        entry_positions = {entry.position for entry in partial_state.entries}

        def subtree_is_fully_represented(target: Position) -> bool:
            target_desc = set(HashBasedMTPIS.Desc(pm_MT, target))
            if not target_desc:
                return False

            covered = set()
            for entry in partial_state.entries:
                if _is_ancestor_or_equal(target, entry.position):
                    covered.update(HashBasedMTPIS.Desc(pm_MT, entry.position))
            return covered == target_desc

        for position in _all_positions(pm_MT):
            level, _ = position
            if level == 0:
                continue

            parent_desc = set(HashBasedMTPIS.Desc(pm_MT, position))
            if not parent_desc or not parent_desc.issubset(index_set):
                continue
            if position in entry_positions:
                continue

            left_child, right_child = _children(pm_MT, position)
            if right_child is None:
                if subtree_is_fully_represented(left_child):
                    return True
                continue
            if subtree_is_fully_represented(left_child) and subtree_is_fully_represented(right_child):
                return True

        return False

    @staticmethod
    def MTSparseBuild(
        pm_MT: MTParameters,
        partial_state: PartialMerkleState,
        complementary_leaves: Mapping[int, LeafInput],
        *,
        leaves_hashed: bool = False,
    ) -> bytes:
        """
        root <- MTSparseBuild(ST_MT, {(i, leaf_i)}_{i notin I})
        """

        if partial_state.params != pm_MT:
            raise ValueError("partial_state was generated under different parameters")

        normalized_complement = _normalize_indexed_leaves(complementary_leaves, pm_MT.leaf_count)
        internal_entries: Dict[Position, bytes] = {}
        leaf_values: list[Optional[bytes]] = [None] * pm_MT.leaf_count

        for index, value in normalized_complement.items():
            leaf_values[index] = value

        for entry in partial_state.entries:
            level, offset = entry.position
            if level == 0:
                if leaf_values[offset] is not None:
                    raise ValueError("the same leaf index appears in both partial_state and complementary_leaves")
                leaf_values[offset] = entry.value
            else:
                internal_entries[entry.position] = entry.value

        def build(position: Position) -> bytes:
            stored = internal_entries.get(position)
            if stored is not None:
                return stored

            level, offset = position
            if level == 0:
                leaf_value = leaf_values[offset]
                if leaf_value is None:
                    raise ValueError("insufficient information to rebuild the Merkle root")
                if leaves_hashed:
                    return leaf_value
                return _leaf_hash(pm_MT, offset, leaf_value)

            left_child, right_child = _children(pm_MT, position)
            left_value = build(left_child)
            if right_child is None:
                return left_value
            right_value = build(right_child)
            return _internal_hash(pm_MT, position, left_value, right_value)

        return build((pm_MT.tree_height, 0))

    @staticmethod
    def RootHex(root: bytes) -> str:
        return root.hex()


MT = HashBasedMTPIS
MTPIS = HashBasedMTPIS
