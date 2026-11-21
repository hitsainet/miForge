"""WSP editor endpoints: list, register, change, remove (feature 016; FR-016.16; FTDD §5.2, §8.5;
D15), and the protocol test (feature 018; FR-018.8).

The operator does all four in the web app's **Editor endpoints** screen (``#/wsp/endpoints``,
2026-11-20), which calls these routes as the operator. ``GET`` lists the rows without the key or
its ciphertext. The key is sealed to the gateway's public key before the row is written; the
request body is never logged, and the backend holds no private key, so it cannot open what it
stored.

**An agent may PROPOSE a registration** (2026-11-20; ``forge_wsp_register_endpoint``): ``POST``
is gated by ``requires_approval_when_agent("wsp_endpoint_register")``. A proposal with a key,
any price, or an external base URL or locality is refused before anything is filed
(``wsp.endpoints.check_agent_proposal``, run by the gate's ``validate`` and again by the route
when the approved proposal executes); a local, keyless one waits for the operator's approval and
then runs this route once. The operator's own call is unchanged: no gate, every field.

``PATCH`` and ``DELETE`` stay the OPERATOR's alone: an agent gets ``403
wsp_endpoint_operator_only`` from a router dependency, the pattern of ``purge.py``'s
``operator_only``. A PATCH moves the endpoint fingerprint, so it fails every queued or approved run
on that endpoint at push (``wsp_endpoint_changed``), runs the operator already approved; an agent
that wants another protocol proposes a second endpoint instead (decision recorded in
``0xcc/reviews/wsp_endpoints_ui_mcp_2026-11-20.md``).
"""

from __future__ import annotations

import datetime as dt
from typing import Annotated, Any

import sqlalchemy as sa
from fastapi import APIRouter, Body, Depends, Header, Request

from api.agent_origin import executing_actor, get_actor, requires_approval_when_agent
from api.deps import SessionDep
from api.v1.endpoints.wsp_runs import approval_still_holds
from core.errors import refuse
from services import approval_gates
from wsp import endpoints, protocol_test, runs
from wsp.models import WspEndpoint
from wsp.settings import get_wsp_settings


def operator_only(request: Request) -> None:
    """``403 wsp_endpoint_operator_only`` for any request that identifies as an agent."""
    actor = get_actor(request)
    if actor.origin == "agent":
        raise refuse(
            "wsp_endpoint_operator_only",
            "Changing or removing an editor endpoint is the operator's, on "
            f"{endpoints.OPERATOR_SCREEN}. Agents can list endpoints, and can propose a local "
            "endpoint with no key and no prices (forge_wsp_register_endpoint), which waits for "
            "the operator's approval.",
            agent=actor.agent,
        )


router = APIRouter(prefix="/wsp", tags=["wsp"])
operator_router = APIRouter(prefix="/wsp", tags=["wsp"], dependencies=[Depends(operator_only)])


@router.get("/endpoints", summary="List editor endpoints (never the key or its ciphertext)")
def list_endpoints(session: SessionDep) -> dict[str, Any]:
    rows = session.scalars(
        sa.select(WspEndpoint).where(WspEndpoint.removed_at.is_(None)).order_by(WspEndpoint.id)
    ).all()
    return {"endpoints": [endpoints.public_document(r) for r in rows]}


@router.post(
    "/endpoints",
    status_code=201,
    summary="Register an editor endpoint (an agent's proposal waits for the operator)",
)
@requires_approval_when_agent(
    "wsp_endpoint_register",
    validate=approval_gates.validate_wsp_endpoint_proposal,
    summary=approval_gates.summarise_wsp_endpoint_proposal,
)
def register_endpoint(
    body: Annotated[dict[str, Any], Body()], session: SessionDep
) -> dict[str, Any]:
    """The operator's registration runs at once with every field. An agent's call never reaches
    here directly: the gate refuses a key, prices or an external endpoint, and files the rest
    as ``wsp_endpoint_register``; approving it runs this function once, as the agent, and the
    agent rule is applied again here so a stored request cannot carry what filing refused."""
    actor = executing_actor()
    created_by = "operator"
    if actor is not None and actor.origin == "agent":
        endpoints.check_agent_proposal(body)
        created_by = actor.agent or "agent:unknown"
    row = endpoints.register_endpoint(
        session, body, created_by=created_by, settings=get_wsp_settings()
    )
    return endpoints.public_document(row)


@operator_router.patch(
    "/endpoints/{endpoint_id}",
    summary=(
        "Change an editor endpoint's action protocol, context window or thinking (operator only)"
    ),
)
def update_endpoint(
    endpoint_id: str, body: Annotated[dict[str, Any], Body()], session: SessionDep
) -> dict[str, Any]:
    """Feature 018 (FR-018.8): ``action_protocol`` and ``context_window_tokens``, and since
    2026-11-21 ``thinking`` and ``thinking_tokens`` (``wsp.endpoints.PATCHABLE``). None is sealed,
    so the key still opens; a value away from its default joins the endpoint fingerprint, so a
    queued or approved run on this endpoint fails ``wsp_endpoint_changed`` at push."""
    return endpoints.public_document(endpoints.update_endpoint(session, endpoint_id, body))


@operator_router.delete("/endpoints/{endpoint_id}", summary="Remove an editor endpoint")
def remove_endpoint(endpoint_id: str, session: SessionDep) -> dict[str, Any]:
    row = session.get(WspEndpoint, endpoint_id)
    if row is None or row.removed_at is not None:
        raise refuse("wsp_endpoint_unknown", "No such editor endpoint.")
    row.removed_at = dt.datetime.now(dt.UTC)
    return {"id": endpoint_id, "removed_at": row.removed_at.isoformat()}


@router.post(
    "/endpoints/{endpoint_id}/protocol-test",
    status_code=201,
    summary="Test an editor endpoint's action protocol: a short inspect run on a built-in bundle",
)
@requires_approval_when_agent(
    "wsp_session",
    when=approval_gates.wsp_protocol_test_needs_approval,
    escalate=("wsp_session_external", approval_gates.wsp_protocol_test_is_external),
    validate=approval_gates.validate_wsp_protocol_test,
    summary=approval_gates.summarise_wsp_protocol_test,
)
def protocol_test_run(
    endpoint_id: str,
    body: Annotated[dict[str, Any], Body()],
    session: SessionDep,
    x_miforge_agent: Annotated[str | None, Header()] = None,
) -> dict[str, Any]:
    """Feature 018 (FR-018.8, F18-D7; FTDD §5.3): queue the protocol test of the endpoint's
    CURRENT ``action_protocol``, through the normal run path and its approval (none for a local
    endpoint; ``wsp_session_external`` for an external one). Its list row shows
    ``refused_turns``. The body is ``{caller_ref?}``; the run's ``caller_ref`` is
    ``protocol_test:<endpoint_id>[:<caller_ref>]``."""
    actor = executing_actor()
    caller = actor.agent if actor is not None and actor.agent else (x_miforge_agent or "operator")
    prepared = protocol_test.prepare(session, endpoint_id, body, caller)
    row = runs.create_run(session, prepared, approval_id=approval_still_holds(session, prepared))
    # 2026-11-21: the thinking the test runs with (the endpoint's, frozen in its request).
    return {
        "run_id": row.run_id,
        "status": row.status,
        "thinking": prepared.frozen.request.get("thinking"),
    }
