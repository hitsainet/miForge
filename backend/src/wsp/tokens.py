"""The per-session gateway token (FTDD §8.4; D20, D22, P49).

32 random bytes, minted at push, **stored as its sha256 only**, valid only at the gateway for one
session and only while the run is ``pushing`` or ``running`` and unexpired. It reaches the child on
stdin and is the only secret that ever enters the worker. Reruns and replays get none (R3A-13).
"""

from __future__ import annotations

import hashlib
import hmac
import secrets

TOKEN_BYTES = 32


def mint() -> tuple[str, str]:
    """``(plaintext, sha256)``. Only the sha256 is ever stored or logged."""
    plaintext = secrets.token_urlsafe(TOKEN_BYTES)
    return plaintext, digest(plaintext)


def digest(plaintext: str) -> str:
    return hashlib.sha256(plaintext.encode("utf-8")).hexdigest()


def verify(stored_sha256: str | None, presented: str | None) -> bool:
    """Constant-time comparison of the presented token's digest with the stored one."""
    if not stored_sha256 or not presented:
        return False
    return hmac.compare_digest(stored_sha256, digest(presented))
