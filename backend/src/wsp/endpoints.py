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

import sqlalchemy as sa
from sqlalchemy.orm import Session

from core import secret_keys
from core.errors import refuse
from providers.settings import refuse_credentials_in_base_url
from wsp import window_read
from wsp.models import (
    ACTION_PROTOCOLS,
    DEFAULT_ACTION_PROTOCOL,
    DEFAULT_HISTORY_MODE,
    DEFAULT_OUTPUT_CAP,
    DEFAULT_PAGING,
    DEFAULT_THINKING,
    EFFORTS,
    ENDPOINT_KINDS,
    HISTORY_MODES,
    OUTPUT_CAP_FIELDS,
    OUTPUT_CAP_VALUES,
    PAGING_VALUES,
    SERVINGS,
    THINKING_VALUES,
    WspEndpoint,
    WspRun,
)
from wsp.settings import WSPSettings
from wsp_worker import editor_limits

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
#: The endpoint's prices: ``core.secret_keys.PRICE_KEYS``, the one list approval answers redact
#: and the MCP endpoint list withholds (2026-11-26).
PRICE_FIELDS = secret_keys.PRICE_KEYS


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
    "output_ceiling_tokens",
    "history_mode",
)
#: The largest window accepted: an INTEGER column, and no served model comes near it.
MAX_CONTEXT_WINDOW_TOKENS = 10_000_000
#: The engine's reasoning allowance per turn when neither the endpoint nor the run sets one
#: (2026-11-21). The backend freezes it into the run's request; ``test_thinking.py`` pins it equal
#: to ``miwsp_engine.context.DEFAULT_THINKING_TOKENS`` (the backend does not import the worker).
DEFAULT_THINKING_TOKENS = 4_096


def check_action_protocol(protocol: object, *, kind: str, serving: str) -> str:
    """``protocol`` if this endpoint can use it. ``structured`` is a ``response_format``
    json_schema: accepted for any OpenAI-kind endpoint (feature 019 P2, FR-019.11, F19-D8). A
    generic one then runs nothing but its protocol test until that test passes
    (``runs.check_protocol_pass``); miServer keeps 018's rule, no test needed. The Anthropic kind
    is refused by name, never quietly downgraded to ``tool_calls``."""
    if protocol not in ACTION_PROTOCOLS:
        raise refuse(
            "validation_failed", f"action_protocol is one of {', '.join(ACTION_PROTOCOLS)}."
        )
    if protocol == "structured" and kind != "openai_compatible":
        raise refuse(
            "wsp_endpoint_protocol_unsupported",
            "action_protocol structured is accepted only for an openai_compatible endpoint (an "
            "Anthropic endpoint takes tool_calls); this endpoint keeps tool_calls.",
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


def check_output_ceiling(value: object, *, window: int | None) -> int | None:
    """``None`` (32,000) or an integer from 256 to 10,000,000, at most ``window`` when one is
    known: stated, or read at registration (feature 019; FR-019.14, W3-D6)."""
    if value is None:
        return None
    low, high = editor_limits.MIN_OUTPUT_CEILING_TOKENS, editor_limits.MAX_OUTPUT_CEILING_TOKENS
    if isinstance(value, bool) or not isinstance(value, int) or not low <= value <= high:
        raise refuse(
            "validation_failed",
            f"output_ceiling_tokens is null (32,000) or an integer from {low:,} to {high:,}.",
        )
    if window is not None and value > window:
        raise refuse(
            "validation_failed",
            f"output_ceiling_tokens ({value:,}) is at most the context window ({window:,}): one "
            "reply cannot be longer than the window it shares with the prompt.",
            output_ceiling_tokens=value,
            context_window_tokens=window,
        )
    return value


def thinking_supported(kind: str, serving: str) -> bool:
    """``chat_template_kwargs.enable_thinking`` reaches the editor: the gateway admits it for an
    OpenAI-kind endpoint served by miServer only (``wsp_gateway.allowlist.OPENAI_MISERVER_TOP``)."""
    return kind == "openai_compatible" and serving == "miserver"


def hosted_anthropic(kind: str, locality: str) -> bool:
    """Kind ``anthropic`` and locality ``external`` (feature 020's "hosted Anthropic")."""
    return kind == "anthropic" and locality == "external"


def thinking_accepted(kind: str, serving: str, locality: str) -> tuple[str, ...]:
    """The thinking values this endpoint takes (feature 020; FR-020.4, F20-D6; TDD §5.1):

    - **hosted Anthropic:** ``default`` and the efforts (adaptive thinking is always sent;
      ``off`` and ``on`` mean nothing there);
    - **hosted OpenAI-compatible:** ``off``, ``default`` and the efforts, which declare that the
      endpoint takes ``reasoning_effort``;
    - **local:** ``off`` and ``default``, and ``on`` for miServer's OpenAI kind (018's rule).
    """
    if hosted_anthropic(kind, locality):
        return ("default", *EFFORTS)
    accepted: tuple[str, ...] = ("off", "default")
    if thinking_supported(kind, serving):
        accepted += ("on",)
    if locality == "external":
        accepted += EFFORTS
    return accepted


def check_thinking(value: object, *, kind: str, serving: str, locality: str = "local") -> str:
    """``value`` if this endpoint can use it (2026-11-21; feature 020 per kind and locality,
    :func:`thinking_accepted`). Anything else is refused by name, never quietly downgraded:
    ``wsp_endpoint_thinking_unsupported``, naming what this endpoint accepts (TD-1)."""
    if value not in THINKING_VALUES:
        raise refuse("validation_failed", f"thinking is one of {', '.join(THINKING_VALUES)}.")
    accepted = thinking_accepted(kind, serving, locality)
    if value in accepted:
        return str(value)
    if value == "on" and locality != "external":
        raise refuse(
            "wsp_endpoint_thinking_unsupported",
            "thinking on is accepted only for an openai_compatible endpoint served by miServer "
            "(serving: miserver), the one kind whose requests may carry "
            "chat_template_kwargs.enable_thinking; this endpoint keeps off or default.",
            kind=kind,
            serving=serving,
        )
    where = (
        "a hosted Anthropic endpoint"
        if hosted_anthropic(kind, locality)
        else "a hosted endpoint" if locality == "external" else "a local endpoint"
    )
    raise refuse(
        "wsp_endpoint_thinking_unsupported",
        f"thinking {value} is not accepted for {where}; this endpoint accepts "
        f"{', '.join(accepted)}.",
        kind=kind,
        serving=serving,
        locality=locality,
        accepted=list(accepted),
    )


def default_history_mode(locality: str) -> str:
    """``append_only`` for a hosted endpoint, ``paging`` for a local one (FR-020.1, W3-D3)."""
    return "append_only" if locality == "external" else DEFAULT_HISTORY_MODE


def check_history_mode(value: object, *, kind: str, locality: str) -> str:
    """``value`` if this endpoint can use it (FR-020.1, W3-D3): ``paging`` is refused for a
    hosted Anthropic endpoint, whose every thinking block binds the exact earlier conversation
    (``wsp_endpoint_history_mode_refused``, TD-1); every other endpoint takes either."""
    if value not in HISTORY_MODES:
        raise refuse("validation_failed", f"history_mode is one of {', '.join(HISTORY_MODES)}.")
    if value == "paging" and hosted_anthropic(kind, locality):
        raise refuse(
            "wsp_endpoint_history_mode_refused",
            "A hosted Anthropic endpoint keeps history_mode append_only: the Messages API binds "
            "each thinking block to the exact conversation before it, and paging rewrites older "
            "results. Leave history_mode out, or name append_only.",
            kind=kind,
            locality=locality,
        )
    return str(value)


#: FR-020.2 (W3-D2, F20-D8): a hosted endpoint states its window, at registration and at every run.
WINDOW_REQUIRED_MESSAGE = (
    "Set the endpoint's context window to run it: a hosted endpoint must state its window."
)
#: FR-020.3 (F20-D5, F20-D12): a hosted Anthropic endpoint states both cache prices. They are
#: sealed with its key, so an older row can only be registered again.
CACHE_PRICES_MESSAGE = (
    "This endpoint has no cache prices, which are sealed with its key: register it again with "
    "both cache prices to run it."
)


def run_refusal(row: WspEndpoint) -> tuple[str, str] | None:
    """Why a run on ``row`` is refused today, or ``None`` (TD-2): the ONE check behind filing
    (``runs.prepare_start``), push (``dispatch._refusal``) and the endpoint list
    (:func:`public_document`'s ``runs_refused``), so the three can never disagree. In order: a
    hosted endpoint with no window, then a hosted Anthropic endpoint without both cache prices.
    The locality is the stored or the derived one, whichever says external (``bound_fields``)."""
    locality = bound_fields(row)["locality"]
    if locality == "external" and row.context_window_tokens is None:
        return "endpoint_window_required", WINDOW_REQUIRED_MESSAGE
    if hosted_anthropic(row.kind, locality) and (
        row.price_cache_write_per_mtok is None or row.price_cache_read_per_mtok is None
    ):
        return "wsp_endpoint_cache_prices_required", CACHE_PRICES_MESSAGE
    return None


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


def check_thinking_tokens(
    value: object, *, window: int | None, locality: str = "local"
) -> int | None:
    """``None`` (the engine's default allowance) or a positive integer below the context window
    when the endpoint states one (the engine bounds it by the window again at each session). A
    hosted endpoint takes none (feature 020, FR-020.4): its thinking is an effort level."""
    if value is None:
        return None
    if locality == "external":
        raise refuse(
            "wsp_endpoint_thinking_unsupported",
            "thinking_tokens is not accepted for a hosted endpoint: its thinking is an effort "
            "level (default, low, medium or high). Leave thinking_tokens out.",
            locality=locality,
        )
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
    ``thinking``, ``thinking_tokens``, ``output_cap``, ``paging``, (feature 019, FR-019.14)
    ``output_ceiling_tokens`` and/or (feature 020, FR-020.1) ``history_mode``, nothing else.
    Feature 020: thinking is checked with the row's kind and locality, and a hosted row's window
    cannot be cleared (``endpoint_window_required``). Any other key is refused by name. A
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
    locality = bound_fields(row)["locality"]
    if "action_protocol" in body:
        row.action_protocol = check_action_protocol(
            body["action_protocol"], kind=row.kind, serving=row.serving
        )
    if "context_window_tokens" in body:
        window = check_context_window(body["context_window_tokens"])
        if window is None and locality == "external":
            # FR-020.2: a PATCH may set a hosted window, never clear it.
            raise refuse("endpoint_window_required", WINDOW_REQUIRED_MESSAGE, endpoint_id=row.id)
        row.context_window_tokens = window
    if "thinking" in body:
        row.thinking = check_thinking(
            body["thinking"], kind=row.kind, serving=row.serving, locality=locality
        )
    if "history_mode" in body:
        row.history_mode = check_history_mode(
            body["history_mode"], kind=row.kind, locality=locality
        )
    if "output_cap" in body:
        row.output_cap = check_output_cap(body["output_cap"])
    if "paging" in body:
        row.paging = check_paging(body["paging"])
    if "thinking_tokens" in body or "context_window_tokens" in body:
        # Re-checked when the window moves too, so a smaller window cannot strand the allowance.
        row.thinking_tokens = check_thinking_tokens(
            body.get("thinking_tokens", row.thinking_tokens),
            window=row.context_window_tokens,
            locality=locality,
        )
    if "output_ceiling_tokens" in body or "context_window_tokens" in body:
        # Feature 019 (FR-019.14): against the stated window, else the registration read; a PATCH
        # that lowers the window below the stored ceiling is refused, naming both.
        row.output_ceiling_tokens = check_output_ceiling(
            body.get("output_ceiling_tokens", row.output_ceiling_tokens),
            window=row.context_window_tokens or row.registration_window_tokens,
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
    if window is None and locality == "external":
        # Feature 020 (FR-020.2, W3-D2): a hosted endpoint states its window.
        raise refuse("endpoint_window_required", WINDOW_REQUIRED_MESSAGE, locality=locality)
    # Feature 020 (FR-020.4, F20-D6): a hosted Anthropic endpoint defaults to ``default``.
    default_thinking = "default" if hosted_anthropic(kind, locality) else DEFAULT_THINKING
    thinking = check_thinking(
        body.get("thinking", default_thinking), kind=kind, serving=serving, locality=locality
    )
    thinking_tokens = check_thinking_tokens(
        body.get("thinking_tokens"), window=window, locality=locality
    )
    ceiling = check_output_ceiling(body.get("output_ceiling_tokens"), window=window)
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
    if hosted_anthropic(kind, locality) and (
        prices["price_cache_write_per_mtok"] is None or prices["price_cache_read_per_mtok"] is None
    ):
        # Feature 020 (FR-020.3, F20-D5): the 5-minute write price and the read price. Sealed
        # with the key, so they are stated now or the endpoint is registered again.
        raise refuse(
            "wsp_endpoint_cache_prices_required",
            "A hosted Anthropic endpoint needs price_cache_write_per_mtok (the 5-minute cache "
            "write price) and price_cache_read_per_mtok.",
        )
    history_mode = check_history_mode(
        body.get("history_mode", default_history_mode(locality)), kind=kind, locality=locality
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
        "output_ceiling_tokens": ceiling,
        "history_mode": history_mode,
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


def latest_run(session: Session, endpoint_id: str) -> WspRun | None:
    """The newest run on this endpoint that recorded a window (feature 019, F19-D29): one small
    query per endpoint, which the list route makes for each row."""
    return session.scalar(
        sa.select(WspRun)
        .where(WspRun.endpoint_id == endpoint_id, WspRun.window_source.is_not(None))
        .order_by(WspRun.started_at.desc(), WspRun.created_at.desc())
        .limit(1)
    )


def _iso(value: dt.datetime | None) -> str | None:
    """A stored time as aware UTC ISO text: SQLite hands back a naive value for a UTC time."""
    if value is None:
        return None
    return (value if value.tzinfo is not None else value.replace(tzinfo=dt.UTC)).isoformat()


#: The prediction's source for a miServer endpoint that states no window (2026-11-27): a run
#: waits queued until miServer reports a window, and none is assumed (``window_read``).
WINDOW_WAITS = "waits"


def window_if_no_answer(row: WspEndpoint) -> str:
    """What a run does when the endpoint's server gives NO answer to the window read (operator
    decision D1, 2026-11-27): ``stated`` (the stated window is used), ``waits`` (it waits, then
    fails ``editor_not_resident``), or ``not_read`` (a hosted endpoint, which is never read)."""
    if row.context_window_tokens is not None:
        return "stated"
    if runs_locality(row) == "external":
        return "not_read"
    return WINDOW_WAITS


def runs_locality(row: WspEndpoint) -> str:
    """The run-time locality (``runs.endpoint_locality``: derived or stored, whichever says
    external); ``bound_fields``' own reading of it."""
    return str(bound_fields(row)["locality"])


def window_if_read_fails(stated: int | None, *, waits: bool = False) -> dict[str, Any]:
    """``stated``, else ``assumed`` 8,192: a failed push read never falls back to a read. With
    ``waits`` (a miServer endpoint that states no window, ``window_read.awaits_resident``) no
    window is used: the run waits for miServer to report one, then fails ``editor_not_resident``,
    so the source is ``waits`` and ``tokens`` 0, "none" (the answer contract types ``tokens`` an
    integer, and retyping it would be a breaking change, FR-021.19)."""
    if stated is not None:
        return {"tokens": stated, "source": "stated"}
    if waits:
        return {"tokens": 0, "source": WINDOW_WAITS}
    return {"tokens": editor_limits.DEFAULT_WINDOW_TOKENS, "source": "assumed"}


def window_now(stated: int | None, latest: WspRun | None, *, waits: bool = False) -> dict[str, Any]:
    """The next run's expected window: ``stated``; else the latest run's reported read with its
    run and time; else ``assumed`` (``waits``, tokens 0, for a miServer endpoint that states
    none). ``run_id`` and ``at`` are null unless ``reported``."""
    if stated is None and latest is not None and latest.served_context_tokens is not None:
        return {
            "tokens": latest.served_context_tokens,
            "source": "reported",
            "run_id": latest.run_id,
            "at": _iso(latest.started_at or latest.created_at),
        }
    return {**window_if_read_fails(stated, waits=waits), "run_id": None, "at": None}


def window_facts(row: WspEndpoint, latest: WspRun | None) -> dict[str, Any]:
    """The window as the endpoint list serves it (feature 019; FR-019.3, F19-D10, F19-D14): the
    registration read, the latest run's read, the mismatch warning, and two predictions.

    - ``window_now`` (operator, 2026-11-25, finding F-P4-1): what the next run is expected to use:
      ``stated``; else the latest run's reported read (``reported``, with that run's id and time);
      else ``assumed`` 8,192. Never the registration read, which may describe another model
      (F19-D11).
    - ``window_if_read_fails``: what a run uses when its own push read fails (``stated``, else
      ``assumed`` 8,192), the meaning ``window_now`` had until 2026-11-25.

    For a miServer endpoint that states no window (2026-11-27, ``window_read.awaits_resident``)
    the 8,192 fallback is never used: both predictions read ``waits`` with tokens 0 where they
    said ``assumed``, since such a run waits queued for miServer to report a window.

    ``window_if_read_fails.no_answer`` (operator decision D1, 2026-11-27) is the case the two
    fields do not cover for a generic endpoint: a server that gives no answer at all (refused, a
    timeout, a 5xx) makes the run wait like miServer's (``waits``); one that answers without a
    window keeps ``assumed`` (F19-D10), which is what ``tokens``/``source`` then describe.

    A run still decides its own window at its push (``window_read.resolve``); these are readings."""
    stated = row.context_window_tokens
    waits = window_read.awaits_resident(row, "session")
    registration = {
        "tokens": row.registration_window_tokens,
        "read_at": _iso(row.registration_window_read_at),
    }
    latest_doc = None
    if latest is not None:
        latest_doc = {
            "tokens": latest.served_context_tokens,
            "source": latest.window_source,
            "run_id": latest.run_id,
            "at": _iso(latest.started_at or latest.created_at),
        }
    reads = [row.registration_window_tokens]
    if latest is not None:
        reads.append(latest.served_context_tokens)
    above = stated is not None and any(r is not None and r < stated for r in reads)
    return {
        "output_ceiling_effective": editor_limits.ceiling(row.output_ceiling_tokens),
        "registration_window": registration,
        "latest_run_window": latest_doc,
        "window_now": window_now(stated, latest, waits=waits),
        "window_if_read_fails": {
            **window_if_read_fails(stated, waits=waits),
            # D1 (operator, 2026-11-27): the case the two fields above do not cover for a
            # generic endpoint, a server that does not answer at all.
            "no_answer": window_if_no_answer(row),
        },
        "window_warnings": ["window_stated_above_reported"] if above else [],
    }


def public_document(row: WspEndpoint, *, latest: WspRun | None = None) -> dict[str, Any]:
    """What ``GET /wsp/endpoints`` shows: never the key or its ciphertext. ``latest`` is the
    newest run on the endpoint that recorded a window (:func:`latest_run`), passed by the list
    route; a registration's answer has none."""
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
        output_ceiling_tokens=row.output_ceiling_tokens,
        history_mode=row.history_mode or DEFAULT_HISTORY_MODE,
        runs_refused=runs_refused(row),
        **window_facts(row, latest),
        protocol_test=protocol_test_document(row),
    )
    return doc


def runs_refused(row: WspEndpoint) -> dict[str, str] | None:
    """The endpoint list's ``runs_refused`` (FR-020.5, F20-D12; TD-2): what a start on this row
    would be refused with today, from :func:`run_refusal`, else null."""
    refusal = run_refusal(row)
    return None if refusal is None else {"code": refusal[0], "message": refusal[1]}


def protocol_test_document(row: WspEndpoint) -> dict[str, Any]:
    """The latest protocol test as the endpoint list serves it (feature 019 P2, FR-019.16, TDD
    §5.1): its result, run and time, the values it tested, and whether its pass is current, was
    cleared by a change, or is required at all (F19-D8)."""
    from wsp import protocol_tests

    state = protocol_tests.pass_state(row)
    return {
        "result": state.result,
        "run_id": state.run_id,
        "at": _iso(state.at),
        "tested": dict(state.tested),
        "current": state.current,
        "cleared": state.cleared,
        "required": state.required,
    }
