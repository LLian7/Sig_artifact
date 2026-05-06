from __future__ import annotations

import hashlib
import secrets
from typing import Optional, Union

from operation_counter import enabled as counters_enabled, increment


BytesLikeInput = Union[str, bytes, int]

SUPPORTED_HASHES = {"shake_128", "shake_256", "sha3_256", "sha3_512"}
DEFAULT_HASH_NAME = "shake_256"


def bits_to_bytes(bit_length: int) -> int:
    if bit_length <= 0:
        raise ValueError("bit_length must be positive")
    return (bit_length + 7) // 8


def truncate_to_bits(data: bytes, bit_length: int) -> bytes:
    byte_length = bits_to_bytes(bit_length)
    truncated = data[:byte_length]
    extra_bits = 8 * byte_length - bit_length
    if extra_bits == 0 or not truncated:
        return truncated

    mask = (0xFF << extra_bits) & 0xFF
    return truncated[:-1] + bytes([truncated[-1] & mask])


def hash_bytes(data: bytes, *, output_bits: int, hash_name: str = DEFAULT_HASH_NAME) -> bytes:
    output_bytes = bits_to_bytes(output_bits)
    byte_aligned = output_bits == 8 * output_bytes
    counting = counters_enabled()

    if hash_name == "shake_128":
        if counting:
            increment("hash.backend_calls")
            increment("hash.backend_calls.shake_128")
        raw = hashlib.shake_128(data).digest(output_bytes)
        return raw if byte_aligned else truncate_to_bits(raw, output_bits)
    if hash_name == "shake_256":
        if counting:
            increment("hash.backend_calls")
            increment("hash.backend_calls.shake_256")
        raw = hashlib.shake_256(data).digest(output_bytes)
        return raw if byte_aligned else truncate_to_bits(raw, output_bits)

    if hash_name == "sha3_256":
        block_fn = hashlib.sha3_256
        digest_size = hashlib.sha3_256().digest_size
    elif hash_name == "sha3_512":
        block_fn = hashlib.sha3_512
        digest_size = hashlib.sha3_512().digest_size
    else:
        raise ValueError(f"unsupported hash_name={hash_name!r}")

    if output_bytes <= digest_size:
        if counting:
            increment("hash.backend_calls")
            increment(f"hash.backend_calls.{hash_name}")
        raw = block_fn(data).digest()[:output_bytes]
        return raw if byte_aligned else truncate_to_bits(raw, output_bits)

    blocks = []
    counter = 1
    if counting:
        increment("hash.backend_calls")
        increment(f"hash.backend_calls.{hash_name}")
    blocks.append(block_fn(data).digest())
    while len(blocks) * digest_size < output_bytes:
        if counting:
            increment("hash.backend_calls")
            increment(f"hash.backend_calls.{hash_name}")
        blocks.append(block_fn(data + counter.to_bytes(4, "big")).digest())
        counter += 1
    raw = b"".join(blocks)[:output_bytes]
    return raw if byte_aligned else truncate_to_bits(raw, output_bits)


def normalize_to_bytes(value: BytesLikeInput) -> bytes:
    if isinstance(value, bytes):
        return value
    if isinstance(value, str):
        return value.encode("utf-8")
    if isinstance(value, int):
        if value < 0:
            raise ValueError("integer inputs must be non-negative")
        width = max(1, (value.bit_length() + 7) // 8)
        return value.to_bytes(width, "big")
    raise TypeError("inputs must be bytes, str, or non-negative int")


def normalize_bitstring(value: BytesLikeInput, bit_length: int) -> str:
    if isinstance(value, str):
        if len(value) != bit_length:
            raise ValueError(
                f"bitstring length mismatch: expected {bit_length}, got {len(value)}"
            )
        if any(bit not in {"0", "1"} for bit in value):
            raise ValueError("bitstring inputs must contain only '0' and '1'")
        return value

    if isinstance(value, bytes):
        expected_bytes = bits_to_bytes(bit_length)
        if len(value) != expected_bytes:
            raise ValueError(
                f"bytes length mismatch: expected {expected_bytes}, got {len(value)}"
            )
        bitstring = "".join(f"{byte:08b}" for byte in value)
        return bitstring[:bit_length]

    if isinstance(value, int):
        if value < 0 or value >= (1 << bit_length):
            raise ValueError("integer input is out of range for the requested bit_length")
        return f"{value:0{bit_length}b}"

    raise TypeError("bitstring inputs must be str, bytes, or non-negative int")


def bitstring_to_bytes(bitstring: str) -> bytes:
    if not bitstring:
        return b""
    return int(bitstring, 2).to_bytes(bits_to_bytes(len(bitstring)), "big")


def derive_parameter(
    label: bytes,
    *,
    seed: Optional[bytes],
    output_bits: int,
    hash_name: str = DEFAULT_HASH_NAME,
) -> bytes:
    if seed is None:
        return truncate_to_bits(secrets.token_bytes(bits_to_bytes(output_bits)), output_bits)
    return hash_bytes(label + seed, output_bits=output_bits, hash_name=hash_name)


def resolve_bit_length(
    *,
    explicit_bits: Optional[int],
    explicit_bytes: Optional[int],
    default_bits: int,
    label: str,
) -> int:
    if explicit_bits is not None and explicit_bits <= 0:
        raise ValueError(f"{label}_bits must be positive")
    if explicit_bytes is not None and explicit_bytes <= 0:
        raise ValueError(f"{label}_bytes must be positive")

    if explicit_bits is not None and explicit_bytes is not None:
        if bits_to_bytes(explicit_bits) != explicit_bytes:
            raise ValueError(
                f"{label}_bits and {label}_bytes are inconsistent: "
                f"{explicit_bits} bits require {bits_to_bytes(explicit_bits)} bytes"
            )
        return explicit_bits

    if explicit_bits is not None:
        return explicit_bits

    if explicit_bytes is not None:
        return 8 * explicit_bytes

    return default_bits
