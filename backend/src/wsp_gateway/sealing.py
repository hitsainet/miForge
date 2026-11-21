"""Endpoint keys sealed to the gateway (feature 016; D15, PF-29, P50, R1B-M8, R2A-9; FTDD §8.5).

The construction (PF-29): an ephemeral X25519 key agreed with the gateway's public key, HKDF-SHA256
over the shared secret, then ChaCha20-Poly1305. The **associated data** is the sha256 of the RFC
8785
form of every routing and price field of the endpoint row (``{id, kind, base_url, model, locality,
serving, output_cap_field, price_*}``), so a row whose URL, model, locality, serving kind, cap field
or ANY price was changed by a database write no longer opens (``wsp_endpoint_key_binding_failed``).

The backend seals (``wsp/endpoints.py``) with the public key only; it holds no private key, so it
cannot open what it stores. The gateway opens, with the private key from its own Secret. Both sides
compute the associated data with :func:`aad_digest`, one function, so they cannot drift.

Blob layout: ``ephemeral public key (32) || nonce (12) || ciphertext and tag``.
"""

from __future__ import annotations

import base64
import hashlib
import os
from decimal import Decimal
from typing import Any

import rfc8785
from cryptography.exceptions import InvalidTag
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric.x25519 import X25519PrivateKey, X25519PublicKey
from cryptography.hazmat.primitives.ciphers.aead import ChaCha20Poly1305
from cryptography.hazmat.primitives.kdf.hkdf import HKDF

INFO = b"miforge-wsp-endpoint-key/v1"
PRICE_FIELDS = (
    "price_in_per_mtok",
    "price_out_per_mtok",
    "price_cache_write_per_mtok",
    "price_cache_read_per_mtok",
)
BOUND_FIELDS = ("id", "kind", "base_url", "model", "locality", "serving", "output_cap_field")
_QUANT = Decimal("0.000001")


class KeyBindingFailed(ValueError):  # noqa: N818 - a refusal, like core.errors.Refusal
    """The blob does not open under this row's associated data (or is malformed)."""


def _price(value: Any) -> str | None:
    """One spelling per price: ``Numeric(12, 6)`` stores six places, so quantise to six."""
    if value is None:
        return None
    return format(Decimal(str(value)).quantize(_QUANT), "f")


def aad_fields(row: dict[str, Any]) -> dict[str, Any]:
    fields: dict[str, Any] = {name: row[name] for name in BOUND_FIELDS}
    fields.update({name: _price(row.get(name)) for name in PRICE_FIELDS})
    return fields


def aad_digest(row: dict[str, Any]) -> bytes:
    """sha256 of the RFC 8785 form of the bound fields (R1B-M8, R2A-9)."""
    return hashlib.sha256(rfc8785.dumps(aad_fields(row))).digest()


def _raw(public: X25519PublicKey) -> bytes:
    return public.public_bytes(serialization.Encoding.Raw, serialization.PublicFormat.Raw)


def _key(shared: bytes, ephemeral: bytes, recipient: bytes) -> bytes:
    return HKDF(algorithm=hashes.SHA256(), length=32, salt=ephemeral + recipient, info=INFO).derive(
        shared
    )


def generate_keypair() -> tuple[str, str]:
    """``(private, public)``, each the base64 of the raw 32 bytes (the operator's one-time step)."""
    private = X25519PrivateKey.generate()
    raw = private.private_bytes(
        serialization.Encoding.Raw, serialization.PrivateFormat.Raw, serialization.NoEncryption()
    )
    return base64.b64encode(raw).decode(), base64.b64encode(_raw(private.public_key())).decode()


def seal(public_key_b64: str, plaintext: bytes, aad: bytes) -> bytes:
    recipient = base64.b64decode(public_key_b64, validate=True)
    if len(recipient) != 32:
        raise ValueError("WSP_GATEWAY_PUBLIC_KEY must be 32 raw bytes, base64-encoded.")
    ephemeral = X25519PrivateKey.generate()
    eph_raw = _raw(ephemeral.public_key())
    shared = ephemeral.exchange(X25519PublicKey.from_public_bytes(recipient))
    nonce = os.urandom(12)
    sealed = ChaCha20Poly1305(_key(shared, eph_raw, recipient)).encrypt(nonce, plaintext, aad)
    return eph_raw + nonce + sealed


def open_sealed(private_key_b64: str, blob: bytes, aad: bytes) -> bytes:
    try:
        private = X25519PrivateKey.from_private_bytes(base64.b64decode(private_key_b64))
        recipient = _raw(private.public_key())
        eph_raw, nonce, sealed = blob[:32], blob[32:44], blob[44:]
        shared = private.exchange(X25519PublicKey.from_public_bytes(eph_raw))
        return ChaCha20Poly1305(_key(shared, eph_raw, recipient)).decrypt(nonce, sealed, aad)
    except (InvalidTag, ValueError) as exc:
        raise KeyBindingFailed("the endpoint key does not open for this row") from exc


def fingerprint(key: str) -> str:
    """The display fingerprint: the first 8 hex characters of the key's sha256 (FR-016.16)."""
    return hashlib.sha256(key.encode()).hexdigest()[:8]
