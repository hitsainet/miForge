"""G5 and G6: every gated route is known, and an agent call to each one waits  (F12).

**G5** (FR-012.21, MR-14). ``GATED_ROUTES`` is the expected set. It must EQUAL the routes the
LIVE OpenAPI document marks with ``x-approval-action`` - so a lost decorator, or a new gated
route nobody listed, fails here - and each entry names the test that sends an agent call to
that route; this file checks by AST that the test exists, so deleting it turns this red. The
tool side of the same question (a tool classed gpu/attach/delete/config that reaches an
ungated route) is G9, ``tests/unit/test_mcp_gpu_tools_need_approval.py``.

**G6** (FR-012.9, .15). One waits-test per gated route, below and in ``test_approvals.py`` /
``test_settings_api.py``: an agent call is answered ``202`` with the action, and the thing the
route would have done - a run row, a chain row, a lease, an enqueue, a merge, an evaluation, a
delete, a settings write - did not happen.
"""

from __future__ import annotations

import ast
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from api.route_table import mounted_routes
from tests.support.approval_world import (
    AGENT,
    World,
    chain_spec,
    definition,
    submit_body,
)

TESTS = Path(__file__).resolve().parents[1]

#: (METHOD, template) -> (action, escalation, test file under tests/, test function)
GATED_ROUTES: dict[tuple[str, str], tuple[str, str | None, str, str]] = {
    ("POST", "/api/v1/runs"): (
        "gpu_start",
        "hotupdate_attach",
        "integration/test_approval_coverage.py",
        "test_an_agent_submit_waits",
    ),
    ("POST", "/api/v1/runs/define"): (
        "gpu_start",
        "hotupdate_attach",
        "integration/test_approvals.py",
        "test_an_agent_define_is_stored_and_nothing_runs",
    ),
    ("POST", "/api/v1/chains"): (
        "gpu_start",
        "hotupdate_attach",
        "integration/test_approval_coverage.py",
        "test_an_agent_chain_waits",
    ),
    ("POST", "/api/v1/runs/{run_id}/continue"): (
        "hotupdate_attach",
        None,
        "integration/test_approval_coverage.py",
        "test_an_agent_continue_submit_waits",
    ),
    ("POST", "/api/v1/runs/{run_id}/evaluate"): (
        "gpu_start",
        None,
        "integration/test_approval_coverage.py",
        "test_an_agent_evaluate_now_waits",
    ),
    ("POST", "/api/v1/adapters/{name}/merge"): (
        "gpu_start",
        None,
        "integration/test_approval_coverage.py",
        "test_an_agent_merge_waits",
    ),
    ("DELETE", "/api/v1/packs/{pack_id}/versions/{version}"): (
        "delete",
        None,
        "integration/test_approval_coverage.py",
        "test_an_agent_pack_delete_waits",
    ),
    ("DELETE", "/api/v1/pack-drafts/{draft_id}"): (
        "delete",
        None,
        "integration/test_approval_coverage.py",
        "test_an_agent_draft_delete_waits",
    ),
    ("PUT", "/api/v1/settings/{key}"): (
        "config_write",
        None,
        "integration/test_settings_api.py",
        "test_an_agent_setting_write_waits",
    ),
    ("DELETE", "/api/v1/settings/{key}"): (
        "config_write",
        None,
        "integration/test_settings_api.py",
        "test_an_agent_setting_reset_waits",
    ),
}


def marked_routes() -> dict[tuple[str, str], tuple[str, str | None]]:
    """``x-approval-action`` / ``x-approval-escalation`` from the LIVE OpenAPI document."""
    from api.app import create_app

    out: dict[tuple[str, str], tuple[str, str | None]] = {}
    for path, operations in create_app().openapi()["paths"].items():
        for method, operation in operations.items():
            action = operation.get("x-approval-action") if isinstance(operation, dict) else None
            if action:
                out[(method.upper(), path)] = (action, operation.get("x-approval-escalation"))
    return out


# ------------------------------------------------------------------------------------ G5


def test_the_marked_routes_equal_the_expected_gated_routes() -> None:
    live = marked_routes()
    assert live, "the live OpenAPI document marks no route at all - the scan saw nothing"
    expected = {
        key: (action, escalation) for key, (action, escalation, _, _) in GATED_ROUTES.items()
    }
    assert live == expected, (
        f"marked but not expected: {sorted(set(live) - set(expected))}; expected but not "
        f"marked (a lost decorator): {sorted(set(expected) - set(live))}; differing: "
        f"{sorted(k for k in set(live) & set(expected) if live[k] != expected[k])}"
    )


def test_every_action_is_attached_to_a_route() -> None:
    from api.agent_origin import APPROVAL_ACTIONS

    attached = {action for action, _ in marked_routes().values()} | {
        escalation for _, escalation in marked_routes().values() if escalation
    }
    assert attached == set(APPROVAL_ACTIONS)


def test_every_gated_route_has_a_live_executor() -> None:
    """The decorator recorded an executor for each gated route; approve reads it by key."""
    from fastapi.routing import APIRoute

    from api.agent_origin import EXECUTORS
    from api.app import create_app

    app = create_app()
    keys = {
        # `{name:path}` is the router's converter; the OpenAPI document drops it.
        (method, route.path.replace(":path}", "}")): getattr(
            route.endpoint, "__mf_executor_key__", None
        )
        for route in mounted_routes(app)
        if isinstance(route.original_route, APIRoute)
        for method in route.methods
    }
    for route in GATED_ROUTES:
        key = keys.get(route)
        assert key is not None and key in EXECUTORS, route


@pytest.mark.parametrize("route", sorted(GATED_ROUTES))
def test_each_gated_route_has_its_waits_test(route: tuple[str, str]) -> None:
    _, _, rel, name = GATED_ROUTES[route]
    tree = ast.parse((TESTS / rel).read_text())
    names = {
        n.name for n in ast.walk(tree) if isinstance(n, ast.FunctionDef | ast.AsyncFunctionDef)
    }
    assert names, f"{rel} defines no function at all - the scan read the wrong file"
    assert name in names, f"{rel} has no {name}; the agent gate on {route} is untested"


# ------------------------------------------------------------------------------------ G6


def _waits(response: Any, action: str) -> dict[str, Any]:
    assert response.status_code == 202, response.text
    body = response.json()
    assert body["status"] == "pending", body
    assert body["action"] == action, body
    return dict(body)


def test_an_agent_submit_waits(world: World) -> None:
    body = _waits(world.client.post("/api/v1/runs", json=submit_body(), headers=AGENT), "gpu_start")
    assert body["summary"]["route"] == "POST /api/v1/runs"
    assert world.count("mf_runs") == 0 and world.enqueued == [] and world.leases == []


def test_an_agent_chain_waits(world: World) -> None:
    body = _waits(
        world.client.post("/api/v1/chains", json=chain_spec(), headers=AGENT), "gpu_start"
    )
    assert body["summary"]["stage_count"] == 2
    assert body["summary"]["stage_0"]["step_budget"] == 40
    assert world.count("mf_chains") == 0 and world.count("mf_runs") == 0
    assert world.enqueued == [] and world.leases == []


@pytest.fixture
def a_run(world: World) -> str:
    from tests.support.jobs import insert_run

    with world.session() as session:
        insert_run(session, "run-existing", status="failed", last_step=12)
    return "run-existing"


@pytest.fixture
def stub_plan(monkeypatch: pytest.MonkeyPatch) -> list[str]:
    """The continue plan, recorded. Its own behaviour is F5's and F11's tests'."""
    from services.checkpoints import resume as resume_module

    planned: list[str] = []

    def plan(run_id: str, **kwargs: Any) -> Any:
        planned.append(run_id)
        return SimpleNamespace(
            step=10,
            optimizer_state="restored",
            reason="trainer state present",
            as_dict=lambda: {
                "run_id": run_id,
                "name": f"base:{run_id}:step-10",
                "step": 10,
                "optimizer": {"state": "restored", "reason": "trainer state present"},
            },
        )

    monkeypatch.setattr(resume_module, "continue_plan", plan)
    return planned


def test_an_agent_continue_submit_waits(
    world: World, a_run: str, stub_plan: list[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    from services import run_continue

    submitted: list[str] = []
    monkeypatch.setattr(
        run_continue, "submit_continue", lambda session, run_id, **kw: submitted.append(run_id)
    )
    body = _waits(
        world.client.post(f"/api/v1/runs/{a_run}/continue", json={"submit": True}, headers=AGENT),
        "hotupdate_attach",
    )
    assert body["summary"]["from_checkpoint"] == f"base:{a_run}:step-10"
    assert stub_plan == [a_run], "the gate did not run the route's own continue plan"
    assert submitted == [], "the continue was submitted before the operator approved it"


def test_an_agent_evaluate_now_waits(
    world: World, a_run: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    from api.v1.endpoints import scoring

    evaluated: list[str] = []
    monkeypatch.setattr(
        scoring.entrypoint, "evaluate_run", lambda **kw: evaluated.append(kw["run_id"])
    )
    body = _waits(
        world.client.post(
            f"/api/v1/runs/{a_run}/evaluate", json={"kinds": ["test"]}, headers=AGENT
        ),
        "gpu_start",
    )
    assert body["summary"]["kinds"] == ["test"] and body["summary"]["at_step"] == 12
    assert evaluated == []


def test_an_agent_merge_waits(world: World, monkeypatch: pytest.MonkeyPatch) -> None:
    from services import runtypes_service

    monkeypatch.setattr(
        runtypes_service, "preflight_merge", lambda session, name: {"adapter": name}
    )
    name = "base:run-a:adapter-40"
    _waits(
        world.client.post(
            f"/api/v1/adapters/{name}/merge", params={"run_id": "run-a", "step": 41}, headers=AGENT
        ),
        "gpu_start",
    )
    assert world.merges == []


def test_an_agent_pack_delete_waits(world: World) -> None:
    from tests.support import pack_fixtures as fx

    with world.session() as session:
        fx.commit(session, fx.definition(slots=fx.valid_slots()))
    body = _waits(
        world.client.delete("/api/v1/packs/notary-md/versions/1", headers=AGENT), "delete"
    )
    assert body["summary"]["target"] == "pack notary-md@1"
    assert world.client.get("/api/v1/packs/notary-md/versions/1").status_code == 200


def test_an_agent_draft_delete_waits(world: World) -> None:
    created = world.client.post("/api/v1/pack-drafts", json={"name": "legal-draft"})
    assert created.status_code in (200, 201), created.text
    draft_id = created.json()["draft_id"]
    _waits(world.client.delete(f"/api/v1/pack-drafts/{draft_id}", headers=AGENT), "delete")
    assert world.client.get(f"/api/v1/pack-drafts/{draft_id}").status_code == 200


# ---------------------------------------------------- FR-012.15: the class is the request's


def test_load_submit_is_gpu_start_and_attach_submit_is_hotupdate_attach(
    world: World,
) -> None:
    load = world.client.post("/api/v1/runs", json=submit_body(), headers=AGENT)
    attach = world.client.post("/api/v1/runs", json=submit_body(mode="attach"), headers=AGENT)
    assert _waits(load, "gpu_start")
    body = _waits(attach, "hotupdate_attach")
    assert "attach" in body["summary"], "an attach request must name its miLLM target"


def test_an_attach_define_and_an_attach_chain_escalate(world: World) -> None:
    _waits(
        world.client.post("/api/v1/runs/define", json=definition(mode="attach"), headers=AGENT),
        "hotupdate_attach",
    )
    _waits(
        world.client.post("/api/v1/chains", json=chain_spec(first_mode="attach"), headers=AGENT),
        "hotupdate_attach",
    )


def test_a_continue_plan_is_answered_and_stores_nothing(
    world: World, a_run: str, stub_plan: list[str]
) -> None:
    response = world.client.post(f"/api/v1/runs/{a_run}/continue", json={}, headers=AGENT)
    assert response.status_code == 200, response.text
    assert response.json()["plan"]["step"] == 10
    assert world.count("mf_approvals") == 0


# --------------------------------------------- FR-012.11: an invalid request is not stored


def test_an_invalid_agent_definition_is_refused_and_not_stored(
    world: World, monkeypatch: pytest.MonkeyPatch
) -> None:
    from core.errors import refuse
    from services import run_definition as validator

    def too_big(definition: Any) -> Any:
        raise refuse("budget_exceeded", "This run does not fit on the card.", needed_bytes=1)

    monkeypatch.setattr(validator, "_live_budget", too_big)
    response = world.client.post("/api/v1/runs/define", json=definition(), headers=AGENT)
    assert response.status_code != 202, response.text
    error = response.json()["error"]
    assert error["code"] == "run_definition_invalid"
    assert "budget_exceeded" in [failure["code"] for failure in error["details"]["failures"]]
    assert world.count("mf_approvals") == 0


def test_an_agent_delete_of_a_missing_pack_is_refused_and_not_stored(
    world: World,
) -> None:
    response = world.client.delete("/api/v1/packs/absent/versions/1", headers=AGENT)
    assert response.status_code == 404, response.text
    assert world.count("mf_approvals") == 0


# ------------------------------------------------------- D7: stopping is never gated


def test_an_agent_cancel_runs_at_once(world: World) -> None:
    from tests.support.jobs import insert_run

    with world.session() as session:
        insert_run(session, "run-live", status="running")
    response = world.client.post("/api/v1/runs/run-live/cancel", headers=AGENT)
    assert response.status_code == 202, response.text
    assert "approval_id" not in response.json()
    assert response.json()["cancel_pending"] is True
    assert world.count("mf_approvals") == 0


def test_the_gated_routes_are_exactly_the_routes_the_ledger_classes_as_waiting() -> None:
    """G5's link to the inventory: FTDD section 5.2's classes live in the ledger's class
    column (``tests/support/mcp_ledger.py``), and the routes those tools reach are this file's
    ``GATED_ROUTES`` - so a route the inventory says waits cannot be missing here, nor the
    reverse."""
    from tests.support.mcp_gate_probe import route_for, template
    from tests.support.mcp_ledger import EXPECTED_CALLS, GATED_CLASSES

    reached = set()
    for method, path, _args, _recorded, klass in EXPECTED_CALLS.values():
        if klass in GATED_CLASSES:
            route = route_for(method, path)
            assert route is not None, (method, path)
            reached.add((method, template(route)))
    assert reached, "the ledger classes no tool as waiting - the scan read nothing"
    assert reached == set(GATED_ROUTES)
