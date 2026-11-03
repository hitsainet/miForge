# Origin: miDataworks (Onegaishimas/miDataworks) backend/src/core/agent_origin.py @ a055a38
# Mode: adapt (docs/REUSE.md). Kept: Actor, the header pattern, a gate that wraps the route
# function and records an executor, the fail-closed predicate, the x-approval-action marker.
# Changed: sync routes and sessions; a session injected for a route that has none; a
# `validate=` step that runs the route's own refusal code before anything is stored; an
# `escalate=` predicate (gpu_start -> hotupdate_attach by payload); Response parameters
# replayed; lower-case refusal codes through core.errors (012 FTID section 3.1).
"""Agent origin and the approval gate  (F12; FR-012.9 - .16, .22; MR-6 - MR-10, MR-15).

**Origin.** A request carrying ``X-Miforge-Agent: agent:<name>`` is an agent's; any other
request is the operator's. There is no login (suite decision, 2026-10-06): the header is how
an agent identifies itself, and the MCP server always sends its own (``agent:forge-mcp``).
A malformed or empty header is refused ``invalid_agent_identity`` (400) on every route by
:class:`AgentIdentityMiddleware`, so a bad identity cannot be used even on a read.

**The gate binds agents that identify themselves, not adversaries.** Anyone who can reach
``/api`` can already act as the operator, exactly as before this feature and exactly as in
miDataworks (BRD-MIFORGE-MCP-001 section 7).

**The gate.** :func:`requires_approval_when_agent` wraps a route function. The operator's
call runs it unchanged. An agent's call, when the class predicate holds, runs the route's own
validation (``validate=``: an invalid request is refused with the route's own refusal and
nothing is stored), stores a pending approval and answers ``202``. Approving runs the stored
request through :data:`EXECUTORS` - the original function, in-process, exactly once
(``services.approval_service``).
"""

from __future__ import annotations

import functools
import inspect
import logging
import re
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass
from typing import Annotated, Any, Literal, get_args, get_origin, get_type_hints

from fastapi import Depends, Request, Response
from fastapi.encoders import jsonable_encoder
from fastapi.params import Depends as DependsParam
from fastapi.responses import JSONResponse
from starlette.types import ASGIApp, Receive, Scope, Send

from api.deps import session_dependency
from core.errors import Refusal, refuse

log = logging.getLogger(__name__)

AGENT_HEADER = "X-Miforge-Agent"
AGENT_PATTERN = r"^agent:[a-z0-9][a-z0-9-]{0,47}$"
_AGENT_VALUE = re.compile(AGENT_PATTERN)

#: The four approval actions (FR-012.16), each with the sentence ``GET /approvals/actions``
#: serves. Pinned equal to ``models.approval.APPROVAL_ACTION_NAMES`` (the CHECK constraint).
APPROVAL_ACTIONS: dict[str, str] = {
    "gpu_start": (
        "Starts work on the GPU: a training run, a chain's first stage, an adapter merge or an "
        "out-of-cadence evaluation. Holds the card for as long as the work runs."
    ),
    "hotupdate_attach": (
        "Starts training on the weights miLLM is serving (the optional hot-update mode): the "
        "run attaches to the served model and writes into it in place. One approval covers "
        "the whole attached run (D11)."
    ),
    "delete": "Deletes a pack version or a pack draft. A deleted artifact cannot be recovered.",
    "config_write": (
        "Changes or clears a runtime setting (checkpoint pruning, sibling URLs, log level, "
        "hot-update timeouts and tolerances). Enabling pruning deletes checkpoints."
    ),
}

#: The OpenAPI extension keys the gate publishes and every guard reads.
X_APPROVAL_ACTION = "x-approval-action"
X_APPROVAL_ESCALATION = "x-approval-escalation"

#: Names of the parameters the gate injects into a wrapped route's signature.
ACTOR_PARAM = "mf_gate_actor"
SESSION_PARAM = "mf_gate_session"


@dataclass(frozen=True)
class Actor:
    """Who is calling, as far as the request says."""

    origin: Literal["operator", "agent"]
    agent: str | None = None
    #: Set while an approved request executes, so the action can name its approval.
    approval_id: str | None = None


#: The actor an approved request is running AS, while its executor runs (013 G2). Unset for
#: every direct call: the operator's request and an agent's ungated read.
_executing: ContextVar[Actor | None] = ContextVar("mf_executing_actor", default=None)


@contextmanager
def executing_as(actor: Actor) -> Iterator[None]:
    """Run the block as ``actor`` - the approval executor's identity for a stored request.

    The executor replays only a route's value parameters (FTID 012 section 3.1), so a route
    cannot be told who asked through its signature. Without this, an agent's approved setting
    write recorded ``updated_by = "operator"`` (ENH-003 section 3.4, gap 2). Reset in
    ``finally``: a later call in the same thread never inherits an agent's identity.
    """
    token = _executing.set(actor)
    try:
        yield
    finally:
        _executing.reset(token)


def executing_actor() -> Actor | None:
    """The approved request's requester while its executor runs; ``None`` otherwise."""
    return _executing.get()


def _identity_refusal() -> Refusal:
    return refuse(
        "invalid_agent_identity",
        f"{AGENT_HEADER} must look like agent:<name> - lower case letters, digits and "
        f"hyphens, at most 48 characters after 'agent:'. Omit the header for an operator "
        f"request.",
        header=AGENT_HEADER,
        pattern=AGENT_PATTERN,
    )


def get_actor(request: Request) -> Actor:
    """FastAPI dependency: the caller, from the agent header."""
    raw = request.headers.get(AGENT_HEADER)
    if raw is None:
        return Actor(origin="operator")
    if not _AGENT_VALUE.fullmatch(raw):
        raise _identity_refusal()
    return Actor(origin="agent", agent=raw)


class AgentIdentityMiddleware:
    """Refuse a malformed agent header on **every** HTTP request (FR-012.22).

    ``get_actor`` sees only the routes that declare it; this sees all of them, so an agent
    cannot read with an identity the gate would refuse. Pure ASGI, so streamed responses and
    the Socket.IO wrapper outside the app are untouched.
    """

    def __init__(self, app: ASGIApp) -> None:
        self.app = app

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope.get("type") == "http":
            wanted = AGENT_HEADER.lower().encode("latin-1")
            for name, value in scope.get("headers", []):
                if name.lower() == wanted:
                    if not _AGENT_VALUE.fullmatch(value.decode("latin-1")):
                        response = JSONResponse(
                            status_code=400, content=_identity_refusal().envelope()
                        )
                        await response(scope, receive, send)
                        return
                    break
        await self.app(scope, receive, send)


# --------------------------------------------------------------------------------------
# The gate
# --------------------------------------------------------------------------------------

#: ``values`` -> bool. Values are the route's path, query and body parameters as parsed.
Predicate = Callable[[dict[str, Any]], bool]
#: ``(values, session)`` -> anything the summary wants; raises the route's own Refusal.
Validator = Callable[[dict[str, Any], Any], Any]
#: ``(values, validated)`` -> the operator-facing summary block.
Summariser = Callable[[dict[str, Any], Any], dict[str, Any]]


@dataclass(frozen=True)
class Executor:
    """How to run a stored request once the operator approves it."""

    key: str
    action: str
    escalation: str | None
    fn: Callable[..., Any]
    session_param: str | None
    response_param: str | None
    #: name -> the annotation to rebuild the stored JSON value with (Annotated stripped).
    value_params: dict[str, Any]


#: Every gated route, keyed by ``module.qualname``. Read by the approval service on approve.
EXECUTORS: dict[str, Executor] = {}


def base_type(hint: Any) -> Any:
    """``Annotated[X, ...]`` -> ``X``; anything else unchanged."""
    if get_origin(hint) is Annotated:
        return get_args(hint)[0]
    return hint


def _is_session_dependency(hint: Any, default: Any) -> bool:
    if isinstance(default, DependsParam) and default.dependency is session_dependency:
        return True
    if get_origin(hint) is Annotated:
        return any(
            isinstance(meta, DependsParam) and meta.dependency is session_dependency
            for meta in get_args(hint)[1:]
        )
    return False


def _is_dependency(hint: Any, default: Any) -> bool:
    if isinstance(default, DependsParam):
        return True
    if get_origin(hint) is Annotated:
        return any(isinstance(meta, DependsParam) for meta in get_args(hint)[1:])
    return False


def to_jsonable(value: Any) -> Any:
    """A route value parameter as the JSON stored in the approval and replayed later."""
    return jsonable_encoder(value)


def requires_approval_when_agent(
    action: str,
    *,
    when: Predicate | None = None,
    escalate: tuple[str, Predicate] | None = None,
    validate: Validator | None = None,
    summary: Summariser | None = None,
) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
    """Gate a route for agent-originated calls.

    Args:
        action: the approval action this route's agent calls wait for.
        when: the class predicate; the call is gated only when it returns true. A predicate
            that **raises fails closed** (gated). Absent: always gated.
        escalate: ``(other_action, predicate)`` - when the predicate holds, the request is
            stored under ``other_action`` instead (``gpu_start`` -> ``hotupdate_attach`` for an
            attach-mode submission, FR-012.15). A predicate that raises escalates.
        validate: the route's **own** refusal code, run before anything is stored
            (FR-012.11). Its return value is handed to ``summary``.
        summary: the operator-facing block shown in the Approvals view (FR-012.18).

    The route may declare a session (``Annotated[Session, Depends(session_dependency)]``)
    and a ``Response``; every other parameter must be a plain value (path, query, body) so
    the stored request can be replayed exactly. Anything else is refused at import time.
    """
    if action not in APPROVAL_ACTIONS:
        raise ValueError(f"{action!r} is not a registered approval action")
    if escalate is not None and escalate[0] not in APPROVAL_ACTIONS:
        raise ValueError(f"{escalate[0]!r} is not a registered approval action")

    def decorate(fn: Callable[..., Any]) -> Callable[..., Any]:
        sig = inspect.signature(fn)
        hints = get_type_hints(fn, include_extras=True)
        session_param: str | None = None
        response_param: str | None = None
        value_params: dict[str, Any] = {}
        for name, param in sig.parameters.items():
            hint = hints.get(name, Any)
            if _is_session_dependency(hint, param.default):
                session_param = name
            elif base_type(hint) is Response:
                response_param = name
            elif _is_dependency(hint, param.default):
                raise TypeError(
                    f"{fn.__qualname__}: parameter {name!r} is a dependency the approval "
                    f"executor cannot replay"
                )
            else:
                value_params[name] = base_type(hint)

        key = f"{fn.__module__}.{fn.__qualname__}"
        EXECUTORS[key] = Executor(
            key=key,
            action=action,
            escalation=None if escalate is None else escalate[0],
            fn=fn,
            session_param=session_param,
            response_param=response_param,
            value_params=value_params,
        )

        @functools.wraps(fn)
        def wrapper(**kwargs: Any) -> Any:
            actor: Actor = kwargs.pop(ACTOR_PARAM)
            gate_session = kwargs.pop(SESSION_PARAM, None)
            if actor.origin != "agent":
                return fn(**kwargs)
            values = {name: kwargs[name] for name in value_params}
            decided = decide(values)
            if decided is None:
                return fn(**kwargs)
            session = kwargs[session_param] if session_param is not None else gate_session
            validated = validate(values, session) if validate is not None else None
            from services import approval_service

            approval = approval_service.create_pending(
                session,
                action=decided,
                executor_key=key,
                payload={name: to_jsonable(values[name]) for name in value_params},
                requested_by=actor.agent or "agent:unknown",
                summary=summary(values, validated) if summary is not None else {},
            )
            return JSONResponse(status_code=202, content=approval_service.pending_body(approval))

        def decide(values: dict[str, Any]) -> str | None:
            """The action this agent call waits for, or ``None`` (runs ungated)."""
            if when is not None:
                try:
                    gated = bool(when(values))
                except Exception:
                    # FAIL CLOSED: a predicate that cannot decide gates the call.
                    log.exception("approval predicate raised; gating", extra={"code": key})
                    gated = True
                if not gated:
                    return None
            if escalate is not None:
                try:
                    escalated = bool(escalate[1](values))
                except Exception:
                    log.exception("escalation predicate raised; escalating", extra={"code": key})
                    escalated = True
                if escalated:
                    return escalate[0]
            return action

        # FastAPI reads this signature. The route's annotations are RESOLVED here: the
        # endpoint modules use `from __future__ import annotations`, so their annotations are
        # strings that FastAPI would otherwise evaluate against THIS module's globals, where
        # the route's own types do not exist.
        params = [
            p.replace(annotation=hints.get(name, p.annotation))
            for name, p in sig.parameters.items()
        ]
        params.append(
            inspect.Parameter(
                ACTOR_PARAM,
                inspect.Parameter.KEYWORD_ONLY,
                annotation=Annotated[Actor, Depends(get_actor)],
            )
        )
        if session_param is None:
            from sqlalchemy.orm import Session

            params.append(
                inspect.Parameter(
                    SESSION_PARAM,
                    inspect.Parameter.KEYWORD_ONLY,
                    annotation=Annotated[Session, Depends(session_dependency)],
                )
            )
        # Parameters with defaults must follow those without; keyword-only ones may sit in
        # any order, which is why both injected parameters are keyword-only.
        ordered = [p.replace(kind=inspect.Parameter.KEYWORD_ONLY) for p in params]
        wrapper.__signature__ = sig.replace(  # type: ignore[attr-defined]
            parameters=ordered,
            return_annotation=hints.get("return", sig.return_annotation),
        )
        del wrapper.__wrapped__  # FastAPI must read the signature above, not the original's
        wrapper.__mf_approval_action__ = action  # type: ignore[attr-defined]
        wrapper.__mf_approval_escalation__ = (  # type: ignore[attr-defined]
            None if escalate is None else escalate[0]
        )
        wrapper.__mf_executor_key__ = key  # type: ignore[attr-defined]
        wrapper.__mf_decide__ = decide  # type: ignore[attr-defined]
        return wrapper

    return decorate


def mark_gated_routes(app: Any) -> int:
    """Publish each gated route's action in the OpenAPI document. Returns how many.

    Run on the assembled app (``api.app.create_app``). The MCP server's descriptions, the
    contract generator and the coverage guards all read ``x-approval-action``; a gated route
    without it would be presented to an agent as one that runs at once.
    """
    # ⚠ NOT `app.routes`: from FastAPI 0.14x an included router is ONE opaque entry there,
    # and this loop marked ZERO routes while every gated tool then read as "runs at once"
    # (deps sync 2026-10-21; `api/route_table.py` says why).
    from api.route_table import declared_route, mounted_api_routes, set_openapi_extra

    marked = 0
    for route in mounted_api_routes(app):
        action = getattr(route.endpoint, "__mf_approval_action__", None)
        if not action:
            continue
        extra = {**(declared_route(route).openapi_extra or {}), X_APPROVAL_ACTION: action}
        escalation = getattr(route.endpoint, "__mf_approval_escalation__", None)
        if escalation:
            extra[X_APPROVAL_ESCALATION] = escalation
        set_openapi_extra(route, extra)
        marked += 1
    return marked


def refuse_agent(actor: Actor) -> None:
    """``403 agent_cannot_decide`` for an agent; nothing for the operator (FR-012.13)."""
    if actor.origin == "agent":
        raise refuse(
            "agent_cannot_decide",
            "An agent cannot approve or reject a request; the operator decides in miForge's "
            "Approvals view.",
            agent=actor.agent,
        )
