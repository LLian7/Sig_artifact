from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Union

from crypto_utils import (
    DEFAULT_HASH_NAME,
    SUPPORTED_HASHES,
    bits_to_bytes,
    derive_parameter,
    hash_bytes,
    normalize_to_bytes,
    resolve_bit_length,
)
from operation_counter import enabled as counters_enabled, increment


MessageInput = Union[str, bytes, int]
KeyInput = Union[str, bytes, int]


@dataclass(frozen=True)
class KHParameters:
    """
    Public parameters pm_KH output by KeyedHSetup(1^kappa).
    """

    security_parameter: int
    output_bits: int
    key_bits: int
    hash_name: str = DEFAULT_HASH_NAME
    domain_label: bytes = b"KeyedH/"

    def __post_init__(self) -> None:
        if self.security_parameter <= 0:
            raise ValueError("security_parameter must be positive")
        if self.output_bits <= 0:
            raise ValueError("output_bits must be positive")
        if self.key_bits <= 0:
            raise ValueError("key_bits must be positive")
        if self.hash_name not in SUPPORTED_HASHES:
            raise ValueError(
                f"unsupported hash_name={self.hash_name!r}; choose from {sorted(SUPPORTED_HASHES)}"
            )

    @property
    def output_bytes(self) -> int:
        return bits_to_bytes(self.output_bits)

    @property
    def key_bytes(self) -> int:
        return bits_to_bytes(self.key_bits)


class HashBasedKeyedHash:
    """
    Concrete keyed hash:

      KeyedHEval(hk, m) = H(domain_label || hk || m)
    """

    @staticmethod
    def KeyedHSetup(
        security_parameter: int,
        *,
        output_bits: Optional[int] = None,
        output_bytes: Optional[int] = None,
        key_bits: Optional[int] = None,
        key_bytes: Optional[int] = None,
        hash_name: str = DEFAULT_HASH_NAME,
        domain_label: bytes = b"KeyedH/",
    ) -> KHParameters:
        resolved_output_bits = resolve_bit_length(
            explicit_bits=output_bits,
            explicit_bytes=output_bytes,
            default_bits=security_parameter,
            label="output",
        )
        resolved_key_bits = resolve_bit_length(
            explicit_bits=key_bits,
            explicit_bytes=key_bytes,
            default_bits=security_parameter,
            label="key",
        )
        return KHParameters(
            security_parameter=security_parameter,
            output_bits=resolved_output_bits,
            key_bits=resolved_key_bits,
            hash_name=hash_name,
            domain_label=domain_label,
        )

    @staticmethod
    def KeyGen(pm_KH: KHParameters, seed: Optional[KeyInput] = None) -> bytes:
        seed_bytes = None if seed is None else normalize_to_bytes(seed)
        return derive_parameter(
            pm_KH.domain_label + b"key/",
            seed=seed_bytes,
            output_bits=pm_KH.key_bits,
            hash_name=pm_KH.hash_name,
        )

    @staticmethod
    def KeyedHEval(pm_KH: KHParameters, hk: bytes, message: MessageInput) -> bytes:
        if len(hk) != pm_KH.key_bytes:
            raise ValueError("hk has incorrect length for the configured key space")
        payload = pm_KH.domain_label + hk + normalize_to_bytes(message)
        if counters_enabled():
            increment("keyed_hash.eval")
        return hash_bytes(
            payload,
            output_bits=pm_KH.output_bits,
            hash_name=pm_KH.hash_name,
        )


KeyedH = HashBasedKeyedHash
KH = HashBasedKeyedHash
