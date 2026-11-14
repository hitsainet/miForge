"""The gate decorator's own rules, on a throwaway app  (F12; FR-012.9, .15, .22; FTID 3.1).

The routes it guards are tested through the mounted app (``test_approval_coverage.py``); this
file pins the decorator's rules that no single route exercises: a predicate that raises fails
CLOSED, an escalation that raises escalates, a dependency the executor cannot replay is refused
at import, and ``mark_gated_routes`` publishes both extensions.
"""

from __future__ import annotations

from typing import Annotated, Any

import pytest
from fastapi import APIRouter, Depends, FastAPI, Response
from fastapi.testclient import TestClient
from sqlalchemy.orm import Session

from api.agent_origin import (
    EXECUTORS,
    X_APPROVAL_ACTION,
    X_APPROVAL_ESCALATION,
    mark_gated_routes,
    requires_approval_when_agent,
)
from api.deps import session_dependency

AGENT = {"X-Miforge-Agent": "agent:test"}


def _boom(values: dict[str, Any]) -> bool:
    raise RuntimeError("cannot decide")


def other() -> int:
    """A dependency the executor could not rebuild from a stored request."""
    return 1


def _decide(fn: Any, values: dict[str, Any]) -> str | None:
    return fn.__mf_decide__(values)  # type: ignore[no-any-return]


def test_a_predicate_that_raises_gates_the_call() -> None:
    @requires_approval_when_agent("gpu_start", when=_boom)
    def route(x: int) -> dict[str, Any]:
        return {"x": x}

    assert _decide(route, {"x": 1}) == "gpu_start"


def test_a_false_predicate_lets_the_call_through() -> None:
    @requires_approval_when_agent("hotupdate_attach", when=lambda values: False)
    def route(x: int) -> dict[str, Any]:
        return {"x": x}

    assert _decide(route, {"x": 1}) is None


def test_an_escalation_that_raises_escalates() -> None:
    @requires_approval_when_agent("gpu_start", escalate=("hotupdate_attach", _boom))
    def route(x: int) -> dict[str, Any]:
        return {"x": x}

    assert _decide(route, {"x": 1}) == "hotupdate_attach"


def test_an_unknown_action_is_refused_at_import() -> None:
    with pytest.raises(ValueError):
        requires_approval_when_agent("launch_missiles")
    with pytest.raises(ValueError):
        requires_approval_when_agent("gpu_start", escalate=("launch_missiles", bool))


def test_a_dependency_the_executor_cannot_replay_is_refused_at_import() -> None:
    with pytest.raises(TypeError, match="cannot replay"):

        @requires_approval_when_agent("delete")
        def route(x: Annotated[int, Depends(other)]) -> dict[str, Any]:
            return {"x": x}


def test_the_executor_records_session_response_and_values() -> None:
    @requires_approval_when_agent("delete")
    def recorded_route(
        thing_id: str,
        response: Response,
        session: Annotated[Session, Depends(session_dependency)],
        limit: int = 3,
    ) -> dict[str, Any]:
        return {}

    executor = EXECUTORS[recorded_route.__mf_executor_key__]  # type: ignore[attr-defined]
    assert executor.session_param == "session"
    assert executor.response_param == "response"
    assert executor.value_params == {"thing_id": str, "limit": int}


def test_mark_gated_routes_publishes_both_extensions_and_the_operator_runs_at_once() -> None:
    router = APIRouter()
    calls: list[int] = []

    @router.post("/things/{n}")
    @requires_approval_when_agent("gpu_start", escalate=("hotupdate_attach", lambda v: v["n"] > 5))
    def make(n: int) -> dict[str, Any]:
        calls.append(n)
        return {"n": n}

    app = FastAPI()
    app.include_router(router)
    assert mark_gated_routes(app) == 1
    operation = app.openapi()["paths"]["/things/{n}"]["post"]
    assert operation[X_APPROVAL_ACTION] == "gpu_start"
    assert operation[X_APPROVAL_ESCALATION] == "hotupdate_attach"
    response = TestClient(app).post("/things/3")
    assert response.status_code == 200 and response.json() == {"n": 3}
    assert calls == [3]


@pytest.mark.parametrize("value", ["", "agent:", "Agent:x", "forge", "agent:-x"])
def test_get_actor_refuses_a_malformed_identity_itself(value: str) -> None:
    """Defence in depth: the dependency refuses even where the middleware is not in front."""
    from starlette.requests import Request

    from api.agent_origin import get_actor
    from core.errors import Refusal

    request = Request({"type": "http", "headers": [(b"x-miforge-agent", value.encode("latin-1"))]})
    with pytest.raises(Refusal) as caught:
        get_actor(request)
    assert caught.value.code == "invalid_agent_identity"


def test_get_actor_reads_an_agent_and_an_operator() -> None:
    from starlette.requests import Request

    from api.agent_origin import get_actor

    agent = Request({"type": "http", "headers": [(b"x-miforge-agent", b"agent:forge-mcp")]})
    assert get_actor(agent).origin == "agent" and get_actor(agent).agent == "agent:forge-mcp"
    assert get_actor(Request({"type": "http", "headers": []})).origin == "operator"
