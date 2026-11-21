"""The gateway application: ``POST /v1/messages``, ``POST /v1/chat/completions``, ``GET /health``.

Per request, in this order (FR-016.15; FTDD §5.5):

1. authenticate the session token (``Authorization: Bearer`` or ``x-api-key``), resolving an
   unknown one through ``by-token``; refuse an expired or ended session;
2. fetch the session's ONE endpoint as the backend holds it now (the request's ``model`` is
   overwritten, so a token cannot reach another endpoint); check its URL and host allow-lists and
   derive its locality; refuse an external endpoint unless the session is ``not_sensitive``
   (unknown is sensitive); refuse an editor that is the session's edit target;
3. rebuild the body from the allow-list (``allowlist.py``); refuse streaming;
4. insert the output cap, clamped to what the budget can pay for; for miServer, the empty steering
   set (P47);
5. open the endpoint key (only if the row still matches its sealed associated data);
6. reserve the worst case in USD and tokens in the backend;
7. forward once, with headers built from scratch;
8. meter the response's ``usage`` and settle it; a failed settle WITHHOLDS the response and ends
   the session (fail closed). miServer's steering header other than ``none``, and its
   ``context_length_exceeded``, ``model_not_resident`` and ``model_loading`` answers, end the
   session after that one call (PF-49), except that a ``context_length_exceeded`` is first sent
   ONCE more, settled and reserved anew, at the smaller cap the refusal implies
   (``meter.retry_cap``, fix 2026-11-21); only a second one ends the session.

Nothing secret is logged: records go through :class:`SecretValueFilter`, which removes every
cached key value and every session token seen, because the key-NAME redaction of
``core/logging.py`` does not match header names like ``x-api-key`` (R1B-M22).
"""

from __future__ import annotations

import datetime as dt
import hashlib
import json
import logging
import threading
import time
from collections.abc import Callable
from typing import Any

import httpx
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, Response

from wsp_gateway import allowlist, meter, sealing, upstream
from wsp_gateway.registry import GatewaySession, Registry, RegistryUnavailable
from wsp_gateway.settings import GatewaySettings

LOGGER_NAME = "wsp_gateway"
logger = logging.getLogger(LOGGER_NAME)

#: The gateway's own answers. Codes shared with the backend carry the backend's status
#: (``test_gateway.py`` pins them against ``api.app.REFUSAL_STATUS``).
STATUS = {
    "wsp_gateway_token_invalid": 401,
    "wsp_session_token_invalid": 401,
    "wsp_session_not_active": 409,
    "wsp_sensitive_data_external_endpoint": 403,
    "wsp_endpoint_url_refused": 422,
    "wsp_endpoint_key_binding_failed": 409,
    "wsp_request_field_refused": 422,
    "wsp_streaming_unsupported": 422,
    "wsp_budget_exhausted": 429,
    "wsp_daily_spend_cap_reached": 429,
    "wsp_request_in_flight": 409,
    "wsp_reservation_closed": 409,
    "wsp_editor_is_edit_target": 409,
    "wsp_gateway_ledger_unavailable": 503,
    "wsp_gateway_upstream_failed": 502,
    # Session-ending editor failures (P28, P47, PF-49): never retried.
    "editor_steered": 409,
    "editor_context_exceeded": 409,
    "editor_model_not_resident": 409,
    "editor_model_loading": 409,
}
#: miServer answers that end the session after one call: (status, code) -> failure.
MISERVER_SESSION_ERRORS = {
    (400, "context_length_exceeded"): "editor_context_exceeded",
    (409, "model_not_resident"): "editor_model_not_resident",
    (503, "model_loading"): "editor_model_loading",
}
ROUTES = {"/messages": allowlist.ANTHROPIC, "/chat/completions": allowlist.OPENAI}
#: Set on an answer whose request the gateway sent with thinking off (fix 2026-11-21): its cap
#: left the reasoning allowance under ``meter.MIN_REASONING_TOKENS``. The engine records it in
#: its transcript (``{event: thinking_skipped, why: gateway}``).
THINKING_SKIPPED_HEADER = "X-miForge-Thinking-Skipped"


def _skip_useless_thinking(asked: dict[str, Any], sent: dict[str, Any]) -> bool:
    """When :func:`allowlist.insert_cap` clamped the request's reasoning allowance under
    :data:`meter.MIN_REASONING_TOKENS` (to as little as 1, run ``wsp_855a74f820ff``), send the
    turn with thinking off: no allowance, ``enable_thinking: false``. ``True`` when it did."""
    field = allowlist.REASONING_ALLOWANCE_FIELD
    wanted, kept = asked.get(field), sent.get(field)
    if not isinstance(wanted, int) or not isinstance(kept, int):
        return False
    if kept >= min(wanted, meter.MIN_REASONING_TOKENS):
        return False
    sent.pop(field)
    kwargs = sent.get("chat_template_kwargs")
    sent["chat_template_kwargs"] = {
        **(kwargs if isinstance(kwargs, dict) else {}),
        "enable_thinking": False,
    }
    return True


class GatewayRefusal(Exception):  # noqa: N818 - a refusal, like core.errors.Refusal
    def __init__(self, code: str, message: str, *, recorded: bool = False) -> None:
        self.code = code
        self.message = message
        #: The call is already on the run's call timeline (its settle carried the outcome), so
        #: ``Gateway.handle`` does not record it again as a refusal.
        self.recorded = recorded
        super().__init__(code)


#: Refusals never recorded on a run's call timeline: no live session holds the token, or the
#: backend cannot be reached (the record would be refused, or fail, the same way).
UNRECORDED_REFUSALS = frozenset(
    {
        "wsp_gateway_token_invalid",
        "wsp_session_token_invalid",
        "wsp_session_not_active",
        "wsp_gateway_ledger_unavailable",
    }
)


class SecretValueFilter(logging.Filter):
    """Removes every registered secret VALUE from a record's rendered message (R1B-M22)."""

    def __init__(self) -> None:
        super().__init__()
        self._values: set[str] = set()
        self._lock = threading.Lock()

    def add(self, value: str) -> None:
        if value:
            with self._lock:
                self._values.add(value)

    def scrub(self, text: str) -> str:
        with self._lock:
            values = sorted(self._values, key=len, reverse=True)
        for value in values:
            text = text.replace(value, "***")
        return text

    def filter(self, record: logging.LogRecord) -> bool:
        message = record.getMessage()
        scrubbed = self.scrub(message)
        if record.exc_info:
            record.exc_text = self.scrub(logging.Formatter().formatException(record.exc_info))
            record.exc_info = None
        record.msg, record.args = scrubbed, ()
        return True


class KeyCache:
    """Decrypted keys in memory only, for at most ``ttl`` seconds, keyed by blob and binding."""

    def __init__(self, private_key: str | None, ttl: float, monotonic: Callable[[], float]) -> None:
        self._private = private_key
        self._ttl = ttl
        self._monotonic = monotonic
        self._keys: dict[str, tuple[float, str]] = {}
        self._lock = threading.Lock()

    def open(self, endpoint: dict[str, Any], secrets: SecretValueFilter) -> str | None:
        encoded = endpoint.get("key_ciphertext")
        if encoded is None:
            return None
        import base64

        blob = base64.b64decode(encoded)
        aad = sealing.aad_digest(endpoint)
        slot = hashlib.sha256(blob + aad).hexdigest()
        now = self._monotonic()
        with self._lock:
            cached = self._keys.get(slot)
            if cached is not None and now - cached[0] < self._ttl:
                return cached[1]
        if self._private is None:
            raise GatewayRefusal(
                "wsp_endpoint_key_binding_failed", "The gateway holds no private key."
            )
        try:
            key = sealing.open_sealed(self._private, blob, aad).decode()
        except (sealing.KeyBindingFailed, UnicodeDecodeError) as exc:
            raise GatewayRefusal(
                "wsp_endpoint_key_binding_failed",
                "The endpoint row no longer matches its sealed key; register it again. If every "
                "endpoint with a key is refused this way, the gateway's key pair does not match "
                "the backend's WSP_GATEWAY_PUBLIC_KEY: put the public half of the "
                "miforge-wsp-gateway Secret there, restart the backend, then register the keys "
                "again.",
            ) from exc
        secrets.add(key)
        with self._lock:
            self._keys[slot] = (now, key)
        return key


def _refusal(code: str, message: str) -> JSONResponse:
    """The gateway's own answer. The message ends ``(code: <code>)`` too: an SDK keeps only the
    message in its error text (``openai``'s ``APIStatusError``), which is all ``weightspace``
    records (``result.json``'s ``error``), and the run's intake reads the cause from it
    (``wsp.runs.reported_failure``)."""
    return JSONResponse(
        status_code=STATUS.get(code, 409),
        content={"error": {"code": code, "message": f"{message} (code: {code})"}},
    )


def _token(request: Request) -> str | None:
    auth = request.headers.get("authorization")
    if auth and auth.lower().startswith("bearer "):
        return auth[7:].strip() or None
    return request.headers.get("x-api-key") or None


def _spent(requested: int, body_bytes: int, session: GatewaySession) -> str:
    """Which budget is spent, in the refusal's message: the run's intake labels the stop
    ``token_budget`` or ``cost_budget`` from it (``wsp.runs.budget_stop``). A request whose own
    cap is under the floor spent nothing, and says so."""
    if requested < meter.MIN_OUTPUT_TOKENS:
        return f"The request's own cap is under the gateway's floor of {meter.MIN_OUTPUT_TOKENS}."
    tokens_only = meter.clamp(
        requested,
        body_bytes=body_bytes,
        remaining_usd=None,
        remaining_tokens=session.remaining_tokens(),
        prices=meter.Prices(),
    )
    if tokens_only is None:
        return "The session's token budget is spent."
    return "The session's cost cap is spent."


def _upstream_message(data: Any) -> str | None:
    """An upstream error answer's ``error.message`` (read for the context retry's sizing only,
    never logged or recorded), else ``None``."""
    error = data.get("error") if isinstance(data, dict) else None
    message = error.get("message") if isinstance(error, dict) else None
    return message if isinstance(message, str) else None


def _upstream_code(response: httpx.Response) -> str | None:
    try:
        data = response.json()
    except ValueError:
        return None
    for holder in (data.get("error"), data.get("detail"), data) if isinstance(data, dict) else ():
        if isinstance(holder, dict) and isinstance(holder.get("code"), str):
            return str(holder["code"])
    return None


class Gateway:
    def __init__(
        self,
        settings: GatewaySettings,
        *,
        internal: httpx.Client,
        upstream_client: httpx.Client,
        resolver: upstream.Resolver = upstream.default_resolver,
        clock: Callable[[], dt.datetime] | None = None,
        monotonic: Callable[[], float] = time.monotonic,
    ) -> None:
        self.settings = settings
        self.registry = Registry(internal, monotonic=monotonic)
        self._monotonic = monotonic
        self.ledger = meter.Ledger(internal)
        self.upstream = upstream_client
        self.resolver = resolver
        self.clock = clock or (lambda: dt.datetime.now(dt.UTC))
        private = settings.wsp_gateway_private_key
        self.secrets = SecretValueFilter()
        if private is not None:
            self.secrets.add(private.get_secret_value())
        service = settings.wsp_gateway_service_token
        if service is not None:
            self.secrets.add(service.get_secret_value())
        self.keys = KeyCache(
            None if private is None else private.get_secret_value(),
            settings.wsp_gateway_key_cache_s,
            monotonic,
        )
        for old in [f for f in logger.filters if isinstance(f, SecretValueFilter)]:
            logger.removeFilter(old)
        logger.addFilter(self.secrets)

    # ----------------------------------------------------------------------------- the steps

    def _session(self, request: Request) -> GatewaySession:
        token = _token(request)
        if token is None:
            raise GatewayRefusal("wsp_gateway_token_invalid", "A session token is required.")
        self.secrets.add(token)
        sha = hashlib.sha256(token.encode()).hexdigest()
        try:
            session = self.registry.resolve(sha)
        except RegistryUnavailable as exc:
            raise GatewayRefusal(
                "wsp_gateway_ledger_unavailable", "The backend cannot be reached."
            ) from exc
        if session is None:
            raise GatewayRefusal("wsp_gateway_token_invalid", "This session token is not valid.")
        if session.ended is not None:
            raise GatewayRefusal(
                "wsp_session_not_active", f"This session was ended ({session.ended})."
            )
        if session.expired(self.clock()):
            # The backend re-dates the token when the push completes (fix 2026-11-21): an
            # expiry cached before that is read again once before the token is refused.
            try:
                refreshed = self.registry.refresh_expiry(session)
            except RegistryUnavailable as exc:
                raise GatewayRefusal(
                    "wsp_gateway_ledger_unavailable", "The backend cannot be reached."
                ) from exc
            if refreshed is None or refreshed.expired(self.clock()):
                raise GatewayRefusal("wsp_session_token_invalid", "This session token has expired.")
        return session

    def _endpoint(self, session: GatewaySession, kind: str) -> tuple[dict[str, Any], str]:
        try:
            endpoint = self.registry.endpoint(session.endpoint_id)
        except RegistryUnavailable as exc:
            raise GatewayRefusal(
                "wsp_gateway_ledger_unavailable", "The backend cannot be reached."
            ) from exc
        if endpoint.get("kind") != kind:
            raise GatewayRefusal(
                "wsp_request_field_refused", "This session's endpoint does not serve this route."
            )
        try:
            locality = upstream.check_endpoint(
                endpoint,
                local_allowlist=self.settings.wsp_local_host_allowlist,
                external_allowlist=self.settings.wsp_external_host_allowlist,
            )
        except upstream.EndpointRefused as exc:
            raise GatewayRefusal("wsp_endpoint_url_refused", str(exc)) from exc
        if locality == "external" and session.sensitivity != "not_sensitive":
            raise GatewayRefusal(
                "wsp_sensitive_data_external_endpoint",
                "A session on sensitive (or unknown) data never reaches an external endpoint.",
            )
        if session.edit_target_model is not None and session.edit_target_model == endpoint.get(
            "model"
        ):
            raise GatewayRefusal(
                "wsp_editor_is_edit_target", "The editor model is the model being edited."
            )
        return endpoint, locality

    def _body(
        self, request: Request, raw: bytes, kind: str, endpoint: dict[str, Any]
    ) -> tuple[dict[str, Any], dict[str, str]]:
        try:
            parsed = json.loads(raw)
        except ValueError as exc:
            raise GatewayRefusal("wsp_request_field_refused", "The body is not JSON.") from exc
        try:
            return allowlist.rebuild(
                kind, str(endpoint.get("serving")), parsed, dict(request.headers)
            )
        except allowlist.StreamingRefused as exc:
            raise GatewayRefusal(
                "wsp_streaming_unsupported",
                "Streaming is refused: metering needs the whole answer.",
            ) from exc
        except allowlist.FieldRefused as exc:
            raise GatewayRefusal(
                "wsp_request_field_refused", f"Field refused: {exc.field}."
            ) from exc

    def handle(self, request: Request, raw: bytes, route: str) -> Response:
        """One editor call. A refusal of a live session's call is recorded on its run's call
        timeline (its code, never the request; live progress 2026-11-20), best effort: a record
        that fails never changes the answer."""
        seen: dict[str, GatewaySession] = {}
        try:
            return self._handle(request, raw, route, seen)
        except GatewayRefusal as exc:
            session = seen.get("session")
            if session is not None and not exc.recorded and exc.code not in UNRECORDED_REFUSALS:
                self._record_refusal(session, exc.code)
            raise

    def _record_refusal(self, session: GatewaySession, code: str) -> None:
        try:
            self.ledger.refused(run_id=session.run_id, token_sha256=session.token_sha256, code=code)
        except meter.LedgerRefused:
            logger.info("refusal %s for run %s not recorded", code, session.run_id)

    def _ms(self, started: float) -> int:
        return max(0, int((self._monotonic() - started) * 1000))

    def _handle(
        self, request: Request, raw: bytes, route: str, seen: dict[str, GatewaySession]
    ) -> Response:
        kind = ROUTES[route]
        session = self._session(request)
        seen["session"] = session
        endpoint, locality = self._endpoint(session, kind)
        body, base = self._body(request, raw, kind, endpoint)
        body["model"] = endpoint["model"]
        prices = meter.Prices.of(endpoint)
        cap_field = str(endpoint["output_cap_field"])
        requested = body.get("max_tokens", body.get("max_completion_tokens"))
        # Size the cap against the body as it will be sent (the cap's own digits included).
        probe = json.dumps(
            allowlist.insert_cap(kind, body, cap_field, meter.MAX_OUTPUT_TOKENS),
            separators=(",", ":"),
        ).encode()
        body_bytes = len(probe) + 64
        uncapped = not isinstance(requested, int)
        if not isinstance(requested, int):
            # No cap named: size one to the endpoint's window from the session's own token
            # counts (meter.PromptMeter, fix 2026-11-21), never the 32,000 fallback that once
            # overflowed a 32,768-token miServer window, nor a byte per token.
            requested = meter.uncapped_default(
                meter.window_for(endpoint, session.served_context_tokens),
                prompt_tokens=session.prompt.estimate(body, body_bytes),
            )
        miserver = endpoint.get("serving") == allowlist.MISERVER
        opened: tuple[str | None, str, str | None] | None = None
        retried = False
        while True:
            cap = meter.clamp(
                requested,
                body_bytes=body_bytes,
                remaining_usd=session.remaining_usd(),
                remaining_tokens=session.remaining_tokens(),
                prices=prices,
            )
            if cap is None:
                raise GatewayRefusal("wsp_budget_exhausted", _spent(requested, body_bytes, session))
            sent = allowlist.insert_cap(kind, body, cap_field, cap, uncapped=uncapped)
            skipped = miserver and _skip_useless_thinking(body, sent)
            if miserver:
                sent["steering"] = {"features": []}  # the inline empty set (P47), never intensity 0
            payload = json.dumps(sent, separators=(",", ":")).encode()
            worst_usd, worst_tokens = meter.worst_case(len(payload), cap, prices)
            if opened is None:
                opened = self._open(endpoint, route, locality)
            key, url, host_header = opened
            reservation = self._reserve(session, worst_usd, worst_tokens, locality)
            headers = upstream.headers_for(endpoint, base, key, host_header)
            started = self._monotonic()
            try:
                response = upstream.forward(self.upstream, url, payload, headers)
            except httpx.HTTPError as exc:
                # The request may have been billed: settle at the worst case, never at zero.
                # Unless it was never sent (fix 2026-11-21): a connection that was never made
                # carried no request, so it generated nothing (``meter.NOT_SENT``).
                self._settle(
                    session,
                    reservation,
                    (
                        meter.NOTHING
                        if isinstance(exc, meter.NOT_SENT)
                        else meter.Metered(0, worst_tokens, worst_usd)
                    ),
                    {
                        "outcome": "upstream_failed",
                        "latency_ms": self._ms(started),
                        "refusal_code": "wsp_gateway_upstream_failed",
                    },
                )
                logger.warning(
                    "upstream call failed for run %s: %s", session.run_id, type(exc).__name__
                )
                raise GatewayRefusal(
                    "wsp_gateway_upstream_failed",
                    "The editor endpoint did not answer.",
                    recorded=True,
                ) from exc
            latency_ms = self._ms(started)
            try:
                data = response.json()
            except ValueError:
                data = None
            usage = data.get("usage") if isinstance(data, dict) else None
            try:
                counted = meter.metered(kind, usage, prices) if isinstance(usage, dict) else None
            except ValueError:
                counted = None
            ending = _miserver_failure(response) if miserver else None
            if counted is not None:
                settled = counted
            elif meter.refused_unserved(response.status_code, usage):
                # Fix 2026-11-21 (live run wsp_c3e954d3f21f): a 4xx with no usage refused the
                # request and generated nothing, so it settles at zero, not at the reservation.
                settled = meter.NOTHING
            else:
                # Review OR1-016-8: an answer with no parseable ``usage`` may still have been
                # billed, so it settles at the reservation's worst case, never at zero.
                settled = meter.Metered(0, worst_tokens, worst_usd)
            smaller = (
                meter.retry_cap(_upstream_message(data), cap)
                if ending == "editor_context_exceeded" and not retried
                else None
            )
            if smaller is not None:
                # Fix 2026-11-21: the cap was past miServer's context (the estimate was wrong).
                # Sent ONCE more with the cap the refusal implies, rather than ending the session;
                # this attempt is on the call timeline, its answer never reaches the editor.
                self._settle(
                    session,
                    reservation,
                    settled,
                    {
                        "outcome": "upstream_error",
                        "http_status": response.status_code,
                        "latency_ms": latency_ms,
                        "finish_reason": None,
                        "refusal_code": "editor_context_retried",
                    },
                )
                logger.info(
                    "run %s: cap %d was past the editor's context; retried once at %d",
                    session.run_id,
                    cap,
                    smaller,
                )
                requested, retried = smaller, True
                continue
            # What became of the call, for the run's call timeline: counts and codes only.
            call = {
                "outcome": (
                    "session_ended"
                    if ending is not None
                    else "served" if response.status_code < 400 else "upstream_error"
                ),
                "http_status": response.status_code,
                "latency_ms": latency_ms,
                "finish_reason": meter.finish_reason(kind, data),
                "refusal_code": ending,
            }
            evidence = (
                # The evidence the next uncapped request is sized from (fix 2026-11-21), and
                # persisted with the settle so a restarted gateway recovers it (migration 0033).
                session.prompt.observe(body, len(payload), counted.input_tokens)
                if counted is not None and response.status_code < 400
                else None
            )
            if evidence is not None:
                call.update(
                    sent_bytes=evidence.sent_bytes,
                    prompt_messages=evidence.messages,
                    conversation_sha256=evidence.key,
                )
            self._settle(session, reservation, settled, call)
            if miserver:
                self._miserver_after(session, response, ending)
            if skipped:
                logger.info(
                    "run %s: cap %d left no useful reasoning; thinking off", session.run_id, cap
                )
            return Response(
                content=response.content,
                status_code=response.status_code,
                media_type=response.headers.get("content-type", "application/json"),
                headers={THINKING_SKIPPED_HEADER: "room"} if skipped else None,
            )

    def _open(
        self, endpoint: dict[str, Any], route: str, locality: str
    ) -> tuple[str | None, str, str | None]:
        """The endpoint's key (only if the row still matches its sealed associated data), its URL
        and its Host header."""
        key = self.keys.open(endpoint, self.secrets)
        if locality == "external" and key is None:
            raise GatewayRefusal(
                "wsp_endpoint_key_binding_failed", "An external endpoint needs a sealed key."
            )
        try:
            url, host_header = upstream.target(endpoint, route, locality, self.resolver)
        except upstream.EndpointRefused as exc:
            raise GatewayRefusal("wsp_endpoint_url_refused", str(exc)) from exc
        return key, url, host_header

    def _reserve(
        self, session: GatewaySession, worst_usd: Any, worst_tokens: int, locality: str
    ) -> int:
        try:
            return self.ledger.reserve(
                run_id=session.run_id,
                token_sha256=session.token_sha256,
                worst_case_usd=worst_usd,
                worst_case_tokens=worst_tokens,
                external=locality == "external",
            )
        except meter.LedgerRefused as exc:
            if exc.code == "wsp_session_not_active":
                self.registry.forget(session.token_sha256)
            raise GatewayRefusal(
                exc.code or "wsp_gateway_ledger_unavailable",
                exc.message or "The backend refused the reservation.",
            ) from exc

    def _settle(
        self,
        session: GatewaySession,
        reservation: int,
        usage: meter.Metered,
        call: dict[str, Any] | None = None,
    ) -> None:
        try:
            self.ledger.settle(reservation, usage, call)
        except meter.LedgerRefused as exc:
            self.registry.end(session, "settle_failed")
            logger.warning("settle failed for run %s; the response is withheld", session.run_id)
            raise GatewayRefusal(
                exc.code or "wsp_gateway_ledger_unavailable",
                "The usage could not be recorded; the response is withheld and the session ended.",
            ) from exc
        session.spend_usd += usage.cost_usd
        session.spend_tokens += usage.input_tokens + usage.output_tokens

    def _miserver_after(
        self, session: GatewaySession, response: httpx.Response, failure: str | None
    ) -> None:
        served = response.headers.get("X-miLLM-Model")
        steering = response.headers.get("X-miLLM-Steering")
        if served is not None:
            session.served_models.append(served)
        session.steering.append("<absent>" if steering is None else steering)
        logger.info("run %s editor served by %s, steering %s", session.run_id, served, steering)
        if failure is not None:
            self.registry.end(session, failure)
            # Its settle carried ``session_ended`` and this code: already on the timeline.
            raise GatewayRefusal(
                failure, "The editor session ended; it is not retried.", recorded=True
            )


def _miserver_failure(response: httpx.Response) -> str | None:
    """The session-ending failure a miServer answer carries (P47, PF-49), or ``None``: one of its
    session errors, or a served answer whose steering header is not ``none``."""
    failure = MISERVER_SESSION_ERRORS.get((response.status_code, _upstream_code(response) or ""))
    steering = response.headers.get("X-miLLM-Steering")
    if failure is None and response.status_code < 400 and steering != "none":
        failure = "editor_steered"
    return failure


def create_gateway_app(gateway: Gateway) -> FastAPI:
    app = FastAPI(title="miForge WSP gateway", docs_url=None, redoc_url=None, openapi_url=None)
    app.state.gateway = gateway

    @app.exception_handler(GatewayRefusal)
    def _refused(_request: Request, exc: GatewayRefusal) -> JSONResponse:
        logger.info("refused %s", exc.code)
        return _refusal(exc.code, exc.message)

    @app.exception_handler(Exception)
    def _unhandled(_request: Request, exc: Exception) -> JSONResponse:
        logger.error("unhandled %s in the gateway", type(exc).__name__)
        return JSONResponse(
            status_code=500, content={"error": {"code": "internal_error", "message": "error"}}
        )

    @app.get("/health")
    def health() -> dict[str, Any]:
        return {"status": "ok", "sessions": len(gateway.registry)}

    @app.post("/v1/messages")
    async def messages(request: Request) -> Response:
        raw = await request.body()
        return await _run(request, raw, "/messages")

    @app.post("/v1/chat/completions")
    async def chat(request: Request) -> Response:
        raw = await request.body()
        return await _run(request, raw, "/chat/completions")

    async def _run(request: Request, raw: bytes, route: str) -> Response:
        from starlette.concurrency import run_in_threadpool

        return await run_in_threadpool(gateway.handle, request, raw, route)

    return app
