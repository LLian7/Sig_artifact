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
from operation_counter import increment


TweakInput = Union[str, bytes, int]
MessageInput = Union[str, bytes, int]


@dataclass(frozen=True)
class THParameters:
    """
    Public parameters pm_TwH output by TweakHSetup(1^kappa).

    The public parameter PP is stored explicitly in public_parameter.
    The hash output length is exactly kappa bits.
    """

    security_parameter: int
    hash_name: str = DEFAULT_HASH_NAME
    public_parameter: bytes = b""
    public_parameter_bits: int = 0

    def __post_init__(self) -> None:
        if self.security_parameter <= 0:
            raise ValueError("security_parameter must be positive")
        if self.hash_name not in SUPPORTED_HASHES:
            raise ValueError(
                f"unsupported hash_name={self.hash_name!r}; choose from {sorted(SUPPORTED_HASHES)}"
            )
        if not self.public_parameter:
            raise ValueError("public_parameter must be non-empty")
        if self.public_parameter_bits <= 0:
            object.__setattr__(self, "public_parameter_bits", self.security_parameter)
        if len(self.public_parameter) != bits_to_bytes(self.public_parameter_bits):
            raise ValueError("public_parameter length does not match public_parameter_bits")

    @property
    def output_bytes(self) -> int:
        return bits_to_bytes(self.security_parameter)


class HashBasedTweakableHash:
    """
    Concrete tweakable hash:

      TweakHEval(PP, Tweak, m) = H(PP || Tweak || m)

    where H is SHAKE or SHA3, and the output is truncated to exactly kappa bits.
    """

    @staticmethod
    def TweakHSetup(
        security_parameter: int,
        *,
        hash_name: str = DEFAULT_HASH_NAME,
        public_seed: Optional[bytes] = None,
        public_parameter: Optional[bytes] = None,
        public_parameter_bits: Optional[int] = None,
        public_parameter_bytes: Optional[int] = None,
    ) -> THParameters:
        pp_bits = resolve_bit_length(
            explicit_bits=public_parameter_bits,
            explicit_bytes=public_parameter_bytes,
            default_bits=security_parameter,
            label="public_parameter",
        )
        if public_parameter is not None:
            pp = normalize_to_bytes(public_parameter)
        elif public_seed is not None:
            pp = derive_parameter(
                b"TwH/PP/",
                seed=public_seed,
                output_bits=pp_bits,
                hash_name=hash_name,
            )
        else:
            pp = derive_parameter(
                b"TwH/PP/",
                seed=None,
                output_bits=pp_bits,
                hash_name=hash_name,
            )
        return THParameters(
            security_parameter=security_parameter,
            hash_name=hash_name,
            public_parameter=pp,
            public_parameter_bits=pp_bits,
        )

    @staticmethod
    def TweakHEval(pm_TwH: THParameters, tweak: TweakInput, message: MessageInput) -> bytes:
        tweak_bytes = normalize_to_bytes(tweak)
        message_bytes = normalize_to_bytes(message)
        payload = pm_TwH.public_parameter + tweak_bytes + message_bytes
        increment("tweak_hash.eval")
        return hash_bytes(
            payload,
            output_bits=pm_TwH.security_parameter,
            hash_name=pm_TwH.hash_name,
        )


TwH = HashBasedTweakableHash
TweakH = HashBasedTweakableHash
