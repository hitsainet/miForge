"""The WSP internal application: the GATEWAY's routes only (feature 016; FTDD §5.4; P38, P49, P61).

A separate FastAPI factory, never mounted on the public app (``create_app`` has no
``/wsp-internal/`` path), served by the ``wsp-internal`` container on port 8001 behind a ClusterIP
Service no Ingress names. The worker cannot reach it (FR-016.13) and it has no worker route.

Every route requires the gateway SERVICE token (``WSP_GATEWAY_SERVICE_TOKEN``), compared as sha256
values with ``hmac.compare_digest``. It is not the session token the worker's child presents to the
gateway: that one arrives here only as its sha256, inside a request body or path.
"""

from __future__ import annotations

import base64
import datetime as dt
import hashlib
import hmac
import time
from decimal import Decimal
from typing import Annotated, Any

import sqlalchemy as sa
from fastapi import APIRouter, Body, Depends, FastAPI, Header
from fastapi.exceptions import RequestValidationError
from sqlalchemy.orm import Session
from starlette.exceptions import HTTPException as StarletteHTTPException

from api.deps import SessionDep
from core.errors import Refusal, refuse
from wsp import gateway_ledger, protocol_tests, runs
from wsp.models import DEFAULT_THINKING, WspEndpoint, WspRun
from wsp.settings import get_wsp_settings

PREFIX = "/wsp-internal/v1/gateway"
#: A token ``by-token`` could not resolve stays unresolved for this long (FTDD §5.4).
NEGATIVE_CACHE_S = 5.0
_negative: dict[str, float] = {}


def require_service_token(authorization: Annotated[str | None, Header()] = None) -> None:
    secret = get_wsp_settings().wsp_gateway_service_token
    presented = (authorization or "").removeprefix("Bearer ").strip()
    if secret is None or not presented:
        raise refuse("wsp_gateway_token_invalid", "The gateway service token is required.")
    expected = hashlib.sha256(secret.get_secret_value().encode()).hexdigest()
    if not hmac.compare_digest(expected, hashlib.sha256(presented.encode()).hexdigest()):
        raise refuse("wsp_gateway_token_invalid", "The gateway service token is not valid.")


gateway_router = APIRouter(prefix=PREFIX, dependencies=[Depends(require_service_token)])
# Every route takes the session as the shared `api.deps.SessionDep` (``scope="function"``): the
# transaction COMMITS BEFORE the answer is sent. With FastAPI's default scope the gateway could
# send its settle before its reserve had committed, and the settle found no open reservation:
# the response was withheld and the session ended (016-P2, 2026-11-19: 1 request in 15-40 over
# loopback HTTP on PostgreSQL).


def _session_entry(session: Session, run: WspRun) -> dict[str, Any]:
    usd, tokens_used = gateway_ledger.spent(session, run.run_id)
    budgets = run.request.get("budgets") or {}
    expires = run.token_expires_at
    return {
        "run_id": run.run_id,
        "token_sha256": run.gateway_token_sha256,
        "endpoint_id": run.endpoint_id,
        "sensitivity": run.sensitivity_effective,
        "caps": {"cost_usd": budgets.get("cost_usd"), "tokens": budgets.get("tokens")},
        "expires_at": None if expires is None else runs._iso(expires),
        # Fix 2026-11-22: the grace the expiry adds to the time budget (``dispatch``), so the
        # gateway bounds each call's deadline by the supervisor's kill, not by the token's expiry.
        "grace_s": get_wsp_settings().wsp_session_grace_s,
        "spend": {"usd": float(usd), "tokens": tokens_used},
        # 2026-11-21: the window the server reported at push (every serving since 019); the
        # gateway's uncapped default uses it after the resolved and the stated windows.
        "served_context_tokens": run.served_context_tokens,
        # Feature 019 (FR-019.2, FR-019.15): the window and the output ceiling the push resolved,
        # the figures the run row and the worker's spec.editor carry (meter.window_for first; the
        # gateway's every sizing is bounded by the ceiling). Null on a run pushed before 019.
        "window_tokens": run.window_tokens,
        "output_ceiling_tokens": run.output_ceiling_tokens,
        # Feature 019 P2 (FR-019.13): the endpoint's own protocol test may send response_format.
        "protocol_test": run.protocol_test_endpoint_id is not None,
        # 2026-11-21 (migration 0033): the last served calls' prompt evidence, so a restarted
        # gateway sizes the next uncapped request from the upstream's counts (meter.PromptMeter).
        "prompt_evidence": gateway_ledger.prompt_evidence(session, run.run_id),
    }


def _active_query() -> Any:
    return sa.select(WspRun).where(
        WspRun.status.in_(gateway_ledger.ACTIVE),
        WspRun.style.not_in(gateway_ledger.NO_GATEWAY_STYLES),
        WspRun.gateway_token_sha256.is_not(None),
    )


@gateway_router.get("/sessions")
def list_sessions(session: SessionDep) -> dict[str, Any]:
    """Active sessions with spend: the gateway rebuilds its registry from this on start."""
    rows = session.scalars(_active_query()).all()
    return {"sessions": [_session_entry(session, r) for r in rows]}


@gateway_router.get("/sessions/by-token/{token_sha256}")
def session_by_token(token_sha256: str, session: SessionDep) -> dict[str, Any]:
    """A token the gateway has not seen; 404 for anything not live (cached for 5 s)."""
    missed = _negative.get(token_sha256)
    if missed is not None and time.monotonic() - missed < NEGATIVE_CACHE_S:
        raise _no_session()
    run = session.scalar(_active_query().where(WspRun.gateway_token_sha256 == token_sha256))
    if run is None or not gateway_ledger.session_active(run, token_sha256, gateway_ledger.utcnow()):
        _negative[token_sha256] = time.monotonic()
        raise _no_session()
    return _session_entry(session, run)


def _no_session() -> Refusal:
    """FTDD §5.4: an unresolvable token answers 404 (``wsp_run_not_found``)."""
    return refuse("wsp_run_not_found", "No live session for this token.")


@gateway_router.get("/endpoints/{endpoint_id}")
def get_endpoint(endpoint_id: str, session: SessionDep) -> dict[str, Any]:
    row = runs.live_endpoint(session, endpoint_id)
    if row is None:
        raise refuse("wsp_endpoint_unknown", "No such editor endpoint.")
    return endpoint_document(row)


def endpoint_document(row: WspEndpoint) -> dict[str, Any]:
    def price(value: Decimal | None) -> str | None:
        return None if value is None else format(value, "f")

    return {
        "id": row.id,
        "kind": row.kind,
        "base_url": row.base_url,
        "model": row.model,
        "locality": runs.endpoint_locality(row),
        "serving": row.serving,
        "output_cap_field": row.output_cap_field,
        # The gateway sizes a request that names no cap to it (meter.uncapped_default). Not
        # in the key's sealed associated data: the operator may PATCH it (FR-018.8).
        "context_window_tokens": row.context_window_tokens,
        # Feature 019 P2 (FR-019.13): read fresh on every request, so a pass cleared mid-session
        # (a protocol PATCH, a failed re-test) refuses that session's next structured request.
        "response_format_admitted": protocol_tests.admitted(row),
        # Feature 020 (FR-020.16, TDD §5.4): the gateway admits ``reasoning_effort`` for a hosted
        # OpenAI-compatible endpoint whose thinking is that effort, read fresh on every request.
        "thinking": row.thinking or DEFAULT_THINKING,
        "price_in_per_mtok": price(row.price_in_per_mtok),
        "price_out_per_mtok": price(row.price_out_per_mtok),
        "price_cache_write_per_mtok": price(row.price_cache_write_per_mtok),
        "price_cache_read_per_mtok": price(row.price_cache_read_per_mtok),
        "key_ciphertext": (
            None if row.key_ciphertext is None else base64.b64encode(row.key_ciphertext).decode()
        ),
    }


@gateway_router.post("/reserve")
def reserve(body: Annotated[dict[str, Any], Body()], session: SessionDep) -> dict[str, Any]:
    try:
        reservation_id = gateway_ledger.reserve(
            session,
            run_id=str(body["run_id"]),
            token_sha256=str(body["token_sha256"]),
            worst_case_usd=Decimal(str(body["worst_case_usd"])),
            worst_case_tokens=int(body["worst_case_tokens"]),
            external=bool(body["external"]),
            settings=get_wsp_settings(),
        )
    except (KeyError, ValueError, ArithmeticError) as exc:
        raise refuse(
            "validation_failed",
            "reserve takes run_id, token_sha256, worst_case_usd, "
            "worst_case_tokens and external.",
        ) from exc
    return {"reservation_id": reservation_id}


@gateway_router.post("/settle")
def settle(body: Annotated[dict[str, Any], Body()], session: SessionDep) -> dict[str, Any]:
    try:
        gateway_ledger.settle(
            session,
            reservation_id=int(body["reservation_id"]),
            usage=dict(body["usage"]),
            # What became of the call (live progress, 2026-11-20): counts and codes, checked in
            # gateway_ledger.call_columns. Optional, so a gateway from before it still settles.
            call=body.get("call"),
        )
    except (KeyError, ValueError, TypeError) as exc:
        raise refuse("validation_failed", "settle takes reservation_id and usage.") from exc
    return {"settled": True, "at": dt.datetime.now(dt.UTC).isoformat()}


@gateway_router.post("/refused")
def refused(body: Annotated[dict[str, Any], Body()], session: SessionDep) -> dict[str, Any]:
    """A call the gateway refused before anything was spent, for the run's call timeline (live
    progress, 2026-11-20): its code only, never the request. Refused for a session that is not
    live with this token (``wsp_session_not_active``)."""
    try:
        seq = gateway_ledger.record_refusal(
            session,
            run_id=str(body["run_id"]),
            token_sha256=str(body["token_sha256"]),
            code=str(body["code"]),
        )
    except (KeyError, TypeError) as exc:
        raise refuse("validation_failed", "refused takes run_id, token_sha256 and code.") from exc
    return {"recorded": True, "seq": seq}


def create_internal_app() -> FastAPI:
    """The internal application: ``gateway_router`` and the shared refusal handlers only."""
    from api.app import (
        http_exception_handler,
        refusal_handler,
        unhandled_handler,
        validation_handler,
    )

    app = FastAPI(title="miForge WSP internal", docs_url=None, redoc_url=None, openapi_url=None)
    app.include_router(gateway_router)
    app.add_exception_handler(Refusal, refusal_handler)
    app.add_exception_handler(RequestValidationError, validation_handler)
    app.add_exception_handler(StarletteHTTPException, http_exception_handler)
    app.add_exception_handler(Exception, unhandled_handler)
    return app
