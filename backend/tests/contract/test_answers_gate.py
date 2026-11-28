"""The gate's pending 202 is declared exactly where the gate is (021 FR-021.7; FT3, FTASKS 5.3).

- The contract keys whose ``202`` can be the pending approval EQUAL the routes the live app marks
  with an approval action (``__mf_approval_action__``, which ``mark_gated_routes`` publishes as
  ``x-approval-action``), and those carry the mark in the contract. A gated route without
  ``gated_responses`` stops the writer (``answers_contract._own_answers``).
- Every gated route's ``202`` accepts the real pending body (``approval_service.pending_body``)
  of an approval filed for its action, and an agent's call over HTTP answers it.
- No route's own answer declares ``approval_id``, the field that tells the pending branch of a
  ``PendingOr<Own>`` union from the route's own 202 (M14b).
- The four routes whose own success is a 202 declare the union; the other nine the pending model.
"""

from __future__ import annotations

from datetime import timedelta
from typing import Any

import pytest

from api.route_table import mounted_api_routes
from schemas import answers_contract
from schemas import contract_tools as tools
from tests.support import answer_validator
from tests.support.approval_world import AGENT, World, submit_body

PENDING = "#/$defs/ApprovalPending"
#: The gated routes whose own success is also a 202, and their own model (FTID §3.6).
OWN_202 = {
    "POST /api/v1/runs": "RunSubmitRunAnswer",
    "POST /api/v1/runs/define": "RunDefineRunAnswer",
    "POST /api/v1/chains": "ChainCreateChainAnswer",
    "POST /api/v1/runs/{run_id}/continue": "ContinueSubmittedAnswer",
}


@pytest.fixture(scope="module")
def contract() -> dict[str, Any]:
    import json

    return dict(json.loads(answers_contract.CONTRACT_PATH.read_text(encoding="utf-8")))


def _marked() -> dict[str, tuple[str, str | None]]:
    """Each gated route of the live app -> (action, escalation)."""
    from api.app import create_app

    out: dict[str, tuple[str, str | None]] = {}
    for ctx in mounted_api_routes(create_app()):
        action = getattr(ctx.endpoint, "__mf_approval_action__", None)
        if action:
            escalation = getattr(ctx.endpoint, "__mf_approval_escalation__", None)
            for method in ctx.methods or ():
                out[f"{method} {ctx.path_format}"] = (action, escalation)
    return out


def _pending_routes(contract: dict[str, Any]) -> set[str]:
    defs = contract["$defs"]
    return {
        key
        for key, entry in contract["responses"].items()
        if any(m.get("$ref") == PENDING for m in tools.members(entry.get("202", {}), defs))
    }


def test_the_pending_202_is_declared_exactly_on_the_gated_routes(contract: dict[str, Any]) -> None:
    marked = _marked()
    assert len(marked) == 13, sorted(marked)
    assert _pending_routes(contract) == set(marked)
    for key, (action, escalation) in marked.items():
        entry = contract["responses"][key]
        assert entry["x-approval-action"] == action, key
        assert entry.get("x-approval-escalation") == escalation, key


def test_the_own_202_routes_declare_the_union_and_the_rest_the_pending_model(
    contract: dict[str, Any],
) -> None:
    defs = contract["$defs"]
    for key in _marked():
        members = [m["$ref"] for m in tools.members(contract["responses"][key]["202"], defs)]
        if key in OWN_202:
            assert members == [PENDING, "#/$defs/" + OWN_202[key]], (key, members)
        else:
            assert members == [PENDING], (key, members)


def test_no_own_answer_declares_approval_id(contract: dict[str, Any]) -> None:
    defs = contract["$defs"]
    for key in _marked():
        entry = contract["responses"][key]
        for status, schema in entry.items():
            if not status[:1].isdigit():
                continue
            for member in tools.members(schema, defs):
                if member.get("$ref") == PENDING:
                    continue
                props = tools.deref(member, defs).get("properties") or {}
                assert "approval_id" not in props, (key, status, member)


def _approval(action: str) -> Any:
    from models.approval import Approval
    from services import approval_service

    now = approval_service.now()
    return Approval(
        id="apr_0123456789abcdef",
        action=action,
        executor_key="api.v1.endpoints.runs_acquisition.submit_run",
        summary={"route": "a gated route", "nested": {"a": [1]}},
        payload={"body": {}},
        request_digest="0" * 64,
        requested_by="agent:forge-mcp",
        status="pending",
        created_at=now,
        expires_at=now + timedelta(hours=23),
    )


@pytest.mark.parametrize("key", sorted(_marked()))
def test_each_gated_202_accepts_the_real_pending_body(key: str) -> None:
    from services import approval_service

    action, escalation = _marked()[key]
    check = answer_validator.answer_check()
    for filed in (action, escalation):
        if filed:
            body = approval_service.pending_body(_approval(filed))
            assert check.check(key, 202, body) == [], (key, filed)
    planted = {**approval_service.pending_body(_approval(action)), "planted": True}
    assert check.check(key, 202, planted), f"{key}: the pending 202 accepts an undeclared key"


def test_an_agents_call_answers_the_declared_pending_202(world: World) -> None:
    """Over HTTP: an agent's submit and an agent's draft delete (the union route and a 204
    route) answer the pending approval, which validates."""
    check = answer_validator.answer_check()
    answer = world.client.post("/api/v1/runs", json=submit_body(), headers=AGENT)
    assert answer.status_code == 202, answer.text
    assert answer.json()["status"] == "pending"
    assert check.check("POST /api/v1/runs", 202, answer.json()) == []
    created = world.client.post("/api/v1/pack-drafts", json={"name": "legal-draft"})
    assert created.status_code in (200, 201), created.text
    draft = created.json()["draft_id"]
    answer = world.client.delete(f"/api/v1/pack-drafts/{draft}", headers=AGENT)
    assert answer.status_code == 202, answer.text
    assert check.check("DELETE /api/v1/pack-drafts/{draft_id}", 202, answer.json()) == []
    # The operator's own call answers the 204 with no body, which the entry lists.
    answer = world.client.delete(f"/api/v1/pack-drafts/{draft}")
    assert answer.status_code == 204 and not answer.content
    entry = check.contract["responses"]["DELETE /api/v1/pack-drafts/{draft_id}"]
    assert entry["x-no-body"] == ["204"]
