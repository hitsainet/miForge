"""Editor endpoints: locality, URL rules, registration and sealing (FR-016.16; PF-16, P61, D15).

**Locality is derived from the host, never declared** (PF-16): a body's ``locality`` is ignored, so
a
mis-declaration cannot send sensitive data out. A host is **local** when it is an IP in a private,
loopback or link-local range (RFC 1918, ``fc00::/7`` and the like), ``localhost``, or a cluster name
(``*.svc``, ``*.cluster.local``); everything else is **external**.

**The base URL** must be ``http``/``https`` with no credentials (``providers.settings``' rule:
refuse,
never mask, and never echo the URL), a path that is empty or exactly ``/v1``, no query and no
fragment.
A local host must be in ``WSP_LOCAL_HOST_ALLOWLIST``; an external host must be ``https`` and in
``WSP_EXTERNAL_HOST_ALLOWLIST`` (P50, P61). The gateway re-checks both at connect time.
"""

from __future__ import annotations

import datetime as dt
import ipaddress
import re
import secrets
from decimal import Decimal
from typing import Any
from urllib.parse import urlsplit

from sqlalchemy.orm import Session

from core.errors import refuse
from providers.settings import refuse_credentials_in_base_url
from wsp.models import (
    ACTION_PROTOCOLS,
    DEFAULT_ACTION_PROTOCOL,
    DEFAULT_OUTPUT_CAP,
    DEFAULT_PAGING,
    DEFAULT_THINKING,
    ENDPOINT_KINDS,
    OUTPUT_CAP_FIELDS,
    OUTPUT_CAP_VALUES,
    PAGING_VALUES,
    SERVINGS,
    THINKING_VALUES,
    WspEndpoint,
)
from wsp.settings import WSPSettings

CLUSTER_SUFFIXES = (".svc", ".svc.cluster.local", ".cluster.local")


def _host(base_url: str) -> str:
    host = urlsplit(base_url).hostname
    if not host:
        raise refuse("wsp_endpoint_url_refused", "The endpoint's base_url has no host.")
    return host.lower().rstrip(".")


def derive_locality(base_url: str) -> str:
    """``local`` or ``external``, from the URL's host alone."""
    host = _host(base_url)
    try:
        address = ipaddress.ip_address(host)
    except ValueError:
        if host == "localhost" or host.endswith(CLUSTER_SUFFIXES):
            return "local"
        return "external"
    if address.is_private or address.is_loopback or address.is_link_local:
        return "local"
    return "external"


def _allowlist(value: str) -> set[str]:
    return {h.strip().lower().rstrip(".") for h in value.split(",") if h.strip()}


def validate_base_url(base_url: str, *, local_allowlist: str, external_allowlist: str) -> str:
    """Refuse a base URL outside the rules; return its derived locality."""
    refuse_credentials_in_base_url(base_url, variable="base_url", name="wsp_endpoint")
    parts = urlsplit(base_url)
    if parts.scheme not in ("http", "https"):
        raise refuse("wsp_endpoint_url_refused", "The endpoint's base_url must be http or https.")
    if parts.path not in ("", "/", "/v1", "/v1/"):
        raise refuse(
            "wsp_endpoint_url_refused",
            "The endpoint's base_url path must be empty or exactly /v1.",
            host=parts.hostname,
        )
    if parts.query or parts.fragment:
        raise refuse(
            "wsp_endpoint_url_refused",
            "The endpoint's base_url must carry no query or fragment.",
            host=parts.hostname,
        )
    host = _host(base_url)
    locality = derive_locality(base_url)
    if locality == "local" and host not in _allowlist(local_allowlist):
        raise refuse(
            "wsp_endpoint_url_refused",
            f"Local host {host!r} is not in WSP_LOCAL_HOST_ALLOWLIST.",
            host=host,
        )
    if locality == "external":
        if parts.scheme != "https":
            raise refuse(
                "wsp_endpoint_url_refused", "An external endpoint must use https.", host=host
            )
        if host not in _allowlist(external_allowlist):
            raise refuse(
                "wsp_endpoint_url_refused",
                f"External host {host!r} is not in WSP_EXTERNAL_HOST_ALLOWLIST.",
                host=host,
            )
    return locality


# ------------------------------------------------------------------- registration (6.7; D15)

ENDPOINT_ID = re.compile(r"[a-z0-9][a-z0-9_-]{0,39}\Z")
PRICE_FIELDS = (
    "price_in_per_mtok",
    "price_out_per_mtok",
    "price_cache_write_per_mtok",
    "price_cache_read_per_mtok",
)


def _decimal(body: dict[str, Any], name: str) -> Decimal | None:
    value = body.get(name)
    if value is None:
        return None
    try:
        price = Decimal(str(value))
    except ArithmeticError as exc:
        raise refuse("validation_failed", f"{name} must be a number.") from exc
    if not price.is_finite() or price < 0 or price >= Decimal("1000000"):
        raise refuse("validation_failed", f"{name} must be between 0 and 1,000,000.")
    return price.quantize(Decimal("0.000001"))


# --------------------------------------------- the action protocol and the window (feature 018)

#: The fields the operator may change after registration (``PATCH /wsp/endpoints/{id}``,
#: FR-018.8; ``thinking`` and ``thinking_tokens`` since 2026-11-21). None is sealed
#: (``wsp_gateway.sealing.BOUND_FIELDS``), so a change never breaks the key; each joins the
#: endpoint fingerprint away from its default (F18-D8).
PATCHABLE = (
    "action_protocol",
    "context_window_tokens",
    "thinking",
    "thinking_tokens",
    "output_cap",
    "paging",
)
#: The largest window accepted: an INTEGER column, and no served model comes near it.
MAX_CONTEXT_WINDOW_TOKENS = 10_000_000
#: The engine's reasoning allowance per turn when neither the endpoint nor the run sets one
#: (2026-11-21). The backend freezes it into the run's request; ``test_thinking.py`` pins it equal
#: to ``miwsp_engine.context.DEFAULT_THINKING_TOKENS`` (the backend does not import the worker).
DEFAULT_THINKING_TOKENS = 4_096


def check_action_protocol(protocol: object, *, kind: str, serving: str) -> str:
    """``protocol`` if this endpoint can use it. ``structured`` is miServer's ``response_format``
    json_schema: accepted only for an OpenAI-kind endpoint it serves (FR-018.8). miServer refuses
    structured output on GGUF models and other servers are unverified, so anything else is refused
    by name, never quietly downgraded to ``tool_calls``."""
    if protocol not in ACTION_PROTOCOLS:
        raise refuse(
            "validation_failed", f"action_protocol is one of {', '.join(ACTION_PROTOCOLS)}."
        )
    if protocol == "structured" and (kind != "openai_compatible" or serving != "miserver"):
        raise refuse(
            "wsp_endpoint_protocol_unsupported",
            "action_protocol structured is accepted only for an openai_compatible endpoint "
            "served by miServer (serving: miserver); this endpoint keeps tool_calls.",
            kind=kind,
            serving=serving,
        )
    return str(protocol)


def check_context_window(value: object) -> int | None:
    """``None`` (use the served window, else the engine's default) or a positive integer."""
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, int):
        raise refuse("validation_failed", "context_window_tokens is a positive integer or null.")
    if not 0 < value <= MAX_CONTEXT_WINDOW_TOKENS:
        raise refuse(
            "validation_failed",
            f"context_window_tokens is between 1 and {MAX_CONTEXT_WINDOW_TOKENS:,}.",
        )
    return value


def thinking_supported(kind: str, serving: str) -> bool:
    """``chat_template_kwargs.enable_thinking`` reaches the editor: the gateway admits it for an
    OpenAI-kind endpoint served by miServer only (``wsp_gateway.allowlist.OPENAI_MISERVER_TOP``)."""
    return kind == "openai_compatible" and serving == "miserver"


def check_thinking(value: object, *, kind: str, serving: str) -> str:
    """``value`` if this endpoint can use it (2026-11-21). ``off`` and ``default`` fit every
    endpoint (on one that is not miServer's OpenAI kind, ``off`` sends nothing and the system
    prompt says thinking is off); ``on`` needs ``enable_thinking`` to reach the server, so
    anything else is refused by name, never quietly downgraded to ``off``."""
    if value not in THINKING_VALUES:
        raise refuse("validation_failed", f"thinking is one of {', '.join(THINKING_VALUES)}.")
    if value == "on" and not thinking_supported(kind, serving):
        raise refuse(
            "wsp_endpoint_thinking_unsupported",
            "thinking on is accepted only for an openai_compatible endpoint served by miServer "
            "(serving: miserver), the one kind whose requests may carry "
            "chat_template_kwargs.enable_thinking; this endpoint keeps off or default.",
            kind=kind,
            serving=serving,
        )
    return str(value)


def check_output_cap(value: object) -> str:
    """``on`` (each request names the per-turn output cap) or ``off`` (none; the gateway's
    window-aware default applies). Both fit every endpoint (2026-11-21)."""
    if value not in OUTPUT_CAP_VALUES:
        raise refuse("validation_failed", f"output_cap is one of {', '.join(OUTPUT_CAP_VALUES)}.")
    return str(value)


def check_paging(value: object) -> str:
    """``on`` (``miwsp_engine`` pages bulky results out of the history and keeps its experiment
    ledger) or ``off``. Fits every endpoint; only the engine reads it (2026-11-21)."""
    if value not in PAGING_VALUES:
        raise refuse("validation_failed", f"paging is one of {', '.join(PAGING_VALUES)}.")
    return str(value)


def check_thinking_tokens(value: object, *, window: int | None) -> int | None:
    """``None`` (the engine's default allowance) or a positive integer below the context window
    when the endpoint states one (the engine bounds it by the window again at each session)."""
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, int):
        raise refuse("validation_failed", "thinking_tokens is a positive integer or null.")
    if not 0 < value <= MAX_CONTEXT_WINDOW_TOKENS:
        raise refuse(
            "validation_failed",
            f"thinking_tokens is between 1 and {MAX_CONTEXT_WINDOW_TOKENS:,}.",
        )
    if window is not None and value >= window:
        raise refuse(
            "validation_failed",
            f"thinking_tokens ({value:,}) must be below the context window ({window:,}): the "
            "reasoning, the action and the prompt all fit in it.",
        )
    return value


def update_endpoint(session: Session, endpoint_id: str, body: object) -> WspEndpoint:
    """The operator's PATCH (FR-018.8): ``action_protocol``, ``context_window_tokens``,
    ``thinking`` and/or ``thinking_tokens``, nothing else. Any other key is refused by name. A
    queued or approved run on this endpoint then fails ``wsp_endpoint_changed`` at push when a
    value moved away from its default."""
    if not isinstance(body, dict) or not body:
        raise refuse(
            "validation_failed",
            f"The body is a JSON object naming at least one of {', '.join(PATCHABLE)}.",
        )
    other = sorted(str(k) for k in body if k not in PATCHABLE)
    if other:
        raise refuse(
            "validation_failed",
            f"Only {', '.join(PATCHABLE)} can be changed; refused: {', '.join(other)}. Any other "
            "field is bound to the sealed key: remove the endpoint and register it again.",
            fields=other,
        )
    row = session.get(WspEndpoint, endpoint_id)
    if row is None or row.removed_at is not None:
        raise refuse("wsp_endpoint_unknown", "No such editor endpoint.")
    if "action_protocol" in body:
        row.action_protocol = check_action_protocol(
            body["action_protocol"], kind=row.kind, serving=row.serving
        )
    if "context_window_tokens" in body:
        row.context_window_tokens = check_context_window(body["context_window_tokens"])
    if "thinking" in body:
        row.thinking = check_thinking(body["thinking"], kind=row.kind, serving=row.serving)
    if "output_cap" in body:
        row.output_cap = check_output_cap(body["output_cap"])
    if "paging" in body:
        row.paging = check_paging(body["paging"])
    if "thinking_tokens" in body or "context_window_tokens" in body:
        # Re-checked when the window moves too, so a smaller window cannot strand the allowance.
        row.thinking_tokens = check_thinking_tokens(
            body.get("thinking_tokens", row.thinking_tokens), window=row.context_window_tokens
        )
    session.flush()
    return row


def validate_registration(
    session: Session, body: dict[str, Any], *, settings: WSPSettings
) -> tuple[dict[str, Any], str | None]:
    """Every registration check, storing nothing: the row's fields and the plaintext key.

    The one validator behind both callers: ``register_endpoint`` (the operator's call, and an
    approved agent proposal executing) and the approval gate's ``validate`` hook, which runs it
    before an agent's proposal is filed (``services.approval_gates.validate_wsp_endpoint_proposal``)
    so the operator is never asked to approve a registration that would refuse.
    """
    kind = body.get("kind")
    if kind not in ENDPOINT_KINDS:
        raise refuse("validation_failed", f"kind must be one of {', '.join(ENDPOINT_KINDS)}.")
    serving = body.get("serving", "generic")
    if serving not in SERVINGS:
        raise refuse("validation_failed", f"serving must be one of {', '.join(SERVINGS)}.")
    cap_field = body.get("output_cap_field") or (
        "max_tokens" if kind == "anthropic" else "max_completion_tokens"
    )
    if cap_field not in OUTPUT_CAP_FIELDS or (kind == "anthropic" and cap_field != "max_tokens"):
        raise refuse(
            "validation_failed", "output_cap_field is max_tokens or max_completion_tokens."
        )
    model = body.get("model")
    if not isinstance(model, str) or not 0 < len(model) <= 256:
        raise refuse("validation_failed", "model is a non-empty string of at most 256 characters.")
    base_url = body.get("base_url")
    if not isinstance(base_url, str) or len(base_url) > 512:
        raise refuse("wsp_endpoint_url_refused", "base_url is a URL of at most 512 characters.")
    locality = validate_base_url(
        base_url,
        local_allowlist=settings.wsp_local_host_allowlist,
        external_allowlist=settings.wsp_external_host_allowlist,
    )
    host = _host(base_url)
    if host.startswith("miserver") and serving != "miserver":
        raise refuse(
            "wsp_endpoint_url_refused",
            "A miServer host must be registered with serving: miserver (P28's headers).",
            host=host,
        )
    protocol = check_action_protocol(
        body.get("action_protocol", DEFAULT_ACTION_PROTOCOL), kind=kind, serving=serving
    )
    window = check_context_window(body.get("context_window_tokens"))
    thinking = check_thinking(body.get("thinking", DEFAULT_THINKING), kind=kind, serving=serving)
    thinking_tokens = check_thinking_tokens(body.get("thinking_tokens"), window=window)
    output_cap = check_output_cap(body.get("output_cap", DEFAULT_OUTPUT_CAP))
    paging = check_paging(body.get("paging", DEFAULT_PAGING))
    prices = {name: _decimal(body, name) for name in PRICE_FIELDS}
    if locality == "external" and (
        prices["price_in_per_mtok"] is None or prices["price_out_per_mtok"] is None
    ):
        raise refuse(
            "wsp_endpoint_pricing_required",
            "An external endpoint needs price_in_per_mtok and price_out_per_mtok.",
        )
    endpoint_id = body.get("id") or f"wse_{secrets.token_hex(6)}"
    if not isinstance(endpoint_id, str) or not ENDPOINT_ID.match(endpoint_id):
        raise refuse("validation_failed", "id is 1-40 characters of a-z, 0-9, _ and -.")
    if session.get(WspEndpoint, endpoint_id) is not None:
        raise refuse("validation_failed", "An endpoint with this id exists (ids are not reused).")
    key = body.get("api_key")
    if key is not None and (not isinstance(key, str) or not 0 < len(key) <= 4096):
        raise refuse("validation_failed", "api_key is a non-empty string.")
    if locality == "external" and key is None:
        raise refuse("validation_failed", "An external endpoint needs its api_key.")
    fields: dict[str, Any] = {
        "id": endpoint_id,
        "kind": kind,
        "base_url": base_url,
        "model": model,
        "locality": locality,
        "output_cap_field": cap_field,
        "serving": serving,
        "action_protocol": protocol,
        "context_window_tokens": window,
        "thinking": thinking,
        "thinking_tokens": thinking_tokens,
        "output_cap": output_cap,
        "paging": paging,
        **prices,
    }
    return fields, key


def register_endpoint(
    session: Session, body: dict[str, Any], *, created_by: str, settings: WSPSettings
) -> WspEndpoint:
    """Validate, derive the locality, seal the key to the gateway, and store the row (FR-016.16).

    A ``locality`` in the body is ignored (PF-16). The plaintext key is sealed before the row is
    written and is never stored, returned or logged; the backend holds no private key.
    """
    from wsp_gateway.sealing import aad_digest, fingerprint, seal

    fields, key = validate_registration(session, body, settings=settings)
    row = WspEndpoint(**fields, created_at=dt.datetime.now(dt.UTC), created_by=created_by)
    if key is not None:
        public = settings.wsp_gateway_public_key
        if not public:
            raise refuse(
                "wsp_gateway_key_not_configured",
                "Set WSP_GATEWAY_PUBLIC_KEY (the gateway's public key) before registering a key.",
            )
        try:
            row.key_ciphertext = seal(public, key.encode(), aad_digest(bound_fields(row)))
        except ValueError as exc:  # binascii.Error is a ValueError
            # R3 review fix (OR3-016-5), 2026-11-16: a malformed key was a 500.
            raise refuse(
                "wsp_gateway_key_not_configured",
                "WSP_GATEWAY_PUBLIC_KEY is not a usable key: it must be the gateway's X25519 "
                "public key as one line of standard base64 (32 raw bytes, 44 characters; no PEM "
                "armour, no trailing newline). Copy the public half of the miforge-wsp-gateway "
                "Secret into the backend config again, restart the backend, and register the "
                "endpoint again.",
                reason=type(exc).__name__,
            ) from exc
        row.key_fingerprint = fingerprint(key)
    session.add(row)
    session.flush()
    return row


# ---------------------------------------- an agent's proposal (2026-11-20; FR-018.8 amendment)

#: Where the operator registers, changes and removes editor endpoints: the web app's screen
#: (``frontend/src/config/routes.ts``, row ``wsp-endpoints``). Every refusal that sends the
#: operator there names it in these words.
OPERATOR_SCREEN = "the Editor endpoints screen in miForge's web app (#/wsp/endpoints)"


def check_agent_proposal(body: dict[str, Any]) -> None:
    """What an agent may propose: a LOCAL editor endpoint with NO key and NO prices.

    Runs before anything is stored, twice: in the approval gate's ``validate`` (so a refused
    proposal files no approval) and in the route when an approved proposal executes (so the rule
    holds even for a stored request). Each refusal is its own code and names the operator's
    screen. A null value is the same as an absent one, as the registration reads it; the key's
    value is never echoed.
    """
    if body.get("api_key") is not None:
        raise refuse(
            "wsp_endpoint_agent_key_refused",
            "An agent cannot send an editor endpoint's API key. An endpoint with a key is the "
            f"operator's to register, on {OPERATOR_SCREEN}; the key is typed there once and "
            "sealed. Propose the endpoint without api_key, or ask the operator.",
        )
    priced = sorted(name for name in PRICE_FIELDS if body.get(name) is not None)
    if priced:
        raise refuse(
            "wsp_endpoint_agent_prices_refused",
            "An agent cannot set an editor endpoint's prices: prices decide what a session may "
            f"spend, and they are the operator's, on {OPERATOR_SCREEN}. A local endpoint needs "
            "none; propose it without them.",
            fields=priced,
        )
    declared = body.get("locality")
    base_url = body.get("base_url")
    derived = derive_locality(base_url) if isinstance(base_url, str) and base_url else None
    if declared not in (None, "local") or derived == "external":
        raise refuse(
            "wsp_endpoint_agent_external_refused",
            "An agent can propose a local editor endpoint only (a private, loopback or cluster "
            "host). An external endpoint sends data off this network and needs a key and prices: "
            f"the operator registers it on {OPERATOR_SCREEN}.",
            **({} if derived is None else {"locality": derived}),
        )
    if not isinstance(body.get("id"), str) or not body.get("id"):
        raise refuse(
            "validation_failed",
            "An agent's proposal names its endpoint id (1-40 characters of a-z, 0-9, _ and -), "
            "so the operator approves exactly the endpoint that will be registered.",
        )


def bound_fields(row: WspEndpoint) -> dict[str, Any]:
    """The fields the key is bound to, as the internal application serves them to the gateway."""
    fields: dict[str, Any] = {
        "id": row.id,
        "kind": row.kind,
        "base_url": row.base_url,
        "model": row.model,
        "locality": (
            "external" if "external" in (derive_locality(row.base_url), row.locality) else "local"
        ),
        "serving": row.serving,
        "output_cap_field": row.output_cap_field,
    }
    fields.update({name: getattr(row, name) for name in PRICE_FIELDS})
    return fields


def public_document(row: WspEndpoint) -> dict[str, Any]:
    """What ``GET /wsp/endpoints`` shows: never the key or its ciphertext."""
    doc = bound_fields(row)
    for name in PRICE_FIELDS:
        doc[name] = None if doc[name] is None else format(doc[name], "f")
    doc.update(
        has_key=row.key_ciphertext is not None,
        key_fingerprint=row.key_fingerprint,
        created_at=row.created_at.isoformat(),
        created_by=row.created_by,
        action_protocol=row.action_protocol,
        context_window_tokens=row.context_window_tokens,
        thinking=row.thinking or DEFAULT_THINKING,
        thinking_tokens=row.thinking_tokens,
        output_cap=row.output_cap or DEFAULT_OUTPUT_CAP,
        paging=row.paging or DEFAULT_PAGING,
    )
    return doc
