"""The secret-named keys miForge never sends out in an answer, and the redaction that enforces it
(operator 2026-11-26: approval answers redact secret-named payload keys on the way out).

**One list.** :data:`PRICE_KEYS` is the WSP endpoint's prices (``wsp.endpoints.PRICE_FIELDS`` is
this tuple). :data:`SECRET_KEYS` adds the key material and credentials. The MCP server's withheld
keys (``mcp_server.tools.wsp.WITHHELD``) cannot import the backend, so a test pins them to this
list rather than copying it (``tests/unit/test_secret_keys.py``).

**What is redacted.** A mapping key is secret-named when, lower-cased, it is in
:data:`SECRET_KEYS` or ends in one of :data:`SECRET_SUFFIXES`. Plural counts stay
(``tokens``, ``thinking_tokens``), and so does a flag naming a secret (``has_key``,
``has_api_key``). Its value is replaced by :data:`REDACTED`, at any depth, inside lists too; the
key itself stays, so a reader sees that something was there.

**Where.** Every approval answer that carries a stored request (``services.approval_service``:
the list, one approval, approve, reject, the batch, the 202 an agent receives). The stored row is
never changed, and ``request_digest`` is computed over the real stored payload at filing, so the
operator's decision and its digest check are unaffected.
"""

from __future__ import annotations

from typing import Any

#: The four prices of a WSP editor endpoint (the operator's, never an agent's).
PRICE_KEYS = (
    "price_in_per_mtok",
    "price_out_per_mtok",
    "price_cache_write_per_mtok",
    "price_cache_read_per_mtok",
)
#: Key material and credentials, by exact name.
CREDENTIAL_KEYS = (
    "api_key",
    "key_fingerprint",
    "key_ciphertext",
    "token",
    "access_token",
    "refresh_token",
    "bearer_token",
    "authorization",
    "secret",
    "password",
)
SECRET_KEYS = frozenset((*PRICE_KEYS, *CREDENTIAL_KEYS))
#: A key ending in one of these is secret-named too (``hub_token``, ``millm_trainer_token``).
SECRET_SUFFIXES = ("_token", "_secret", "_password")
#: What a redacted value reads.
REDACTED = "[redacted]"


def is_secret_key(key: Any) -> bool:
    if not isinstance(key, str):
        return False
    name = key.lower()
    return name in SECRET_KEYS or name.endswith(SECRET_SUFFIXES)


def redacted(value: Any) -> Any:
    """``value`` with every secret-named key's value replaced by :data:`REDACTED`, at any depth.
    A new structure: the input (a stored row's JSON) is never changed."""
    if isinstance(value, dict):
        return {k: (REDACTED if is_secret_key(k) else redacted(v)) for k, v in value.items()}
    if isinstance(value, list | tuple):
        return [redacted(item) for item in value]
    return value
