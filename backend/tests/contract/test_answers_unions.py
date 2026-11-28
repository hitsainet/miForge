"""The contract's unions (021 FR-021.8; FTDD §5, FTASKS 5.4).

- The continue route answers three ways: the plan (200), the submitted continue (202) and, for an
  agent's submit, the pending approval (202). The plan and the pending approval come from the
  route here and validate (the submitted continue comes from the real launcher in
  ``tests/integration/test_hotupdate_resume.py``, under the validator); a body of one branch
  carrying another's marker fails.
- An approval's ``result`` is checked by ``action``: one ``if``/``then`` branch per action a gated
  route files, each the history form of those routes' own answers or null (miServer TD14). The
  branch set equals the actions on the live routes, and a result of one action's route does not
  pass as another's.
- No ``$ref`` sits beside an ``anyOf`` or ``oneOf`` (a reader would have to merge them).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from schemas import answers_contract
from schemas import contract_tools as tools
from tests.support import answer_validator
from tests.support.approval_world import AGENT, World

CONTINUE = "POST /api/v1/runs/{run_id}/continue"
APPROVAL = "GET /api/v1/approvals/{approval_id}"


@pytest.fixture(scope="module")
def contract() -> dict[str, Any]:
    return dict(json.loads(answers_contract.CONTRACT_PATH.read_text(encoding="utf-8")))


@pytest.fixture
def a_run(world: World) -> str:
    from tests.support.jobs import insert_run

    with world.session() as session:
        insert_run(session, "run-existing", status="failed", last_step=12)
    return "run-existing"


@pytest.fixture
def real_plan(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """The continue plan as ``services.checkpoints.resume.ResumePlan`` builds it (its own
    behaviour is F5's and F11's tests'): a real ``ResumePlan``, so ``as_dict`` is production."""
    from services.checkpoints import resume as resume_module

    def plan(run_id: str, **kwargs: Any) -> Any:
        return resume_module.ResumePlan(
            run_id=run_id,
            source_run_id=run_id,
            step=10,
            directory=tmp_path / "step-10",
            manifest={"weights_sha256": "c" * 64},
            base_model="base",
            parent=f"base:{run_id}:step-10",
            optimizer_state="restored",
            reason="trainer state present",
            steps_with_state=(10,),
            config_hash_changed=("a" * 64, "b" * 64),
            is_branch=False,
        )

    monkeypatch.setattr(resume_module, "continue_plan", plan)


def test_the_continue_route_declares_its_three_branches(contract: dict[str, Any]) -> None:
    defs, entry = contract["$defs"], contract["responses"][CONTINUE]
    assert [m["$ref"] for m in tools.members(entry["200"], defs)] == ["#/$defs/ContinuePlanAnswer"]
    assert [m["$ref"] for m in tools.members(entry["202"], defs)] == [
        "#/$defs/ApprovalPending",
        "#/$defs/ContinueSubmittedAnswer",
    ]


def test_each_continue_branch_comes_from_the_route_and_validates(
    world: World, a_run: str, real_plan: None
) -> None:
    check = answer_validator.answer_check()
    plan = world.client.post(f"/api/v1/runs/{a_run}/continue", json={})
    assert plan.status_code == 200, plan.text
    assert check.check(CONTINUE, 200, plan.json()) == []

    pending = world.client.post(
        f"/api/v1/runs/{a_run}/continue", json={"submit": True}, headers=AGENT
    )
    assert pending.status_code == 202 and pending.json()["status"] == "pending", pending.text
    assert check.check(CONTINUE, 202, pending.json()) == []

    # A body of one branch with another's marker fails. (The submitted branch itself comes from
    # the real launcher in tests/integration/test_hotupdate_resume.py, checked by the validator.)
    crossed = {**pending.json(), "submitted": True}
    assert check.check(CONTINUE, 202, crossed), "a pending body marked submitted passed"
    assert check.check(CONTINUE, 202, plan.json()), "the plan passed as a 202"
    assert check.check(CONTINUE, 200, pending.json()), "the pending approval passed as the plan"


def _branches(contract: dict[str, Any], name: str) -> dict[str, list[Any]]:
    out: dict[str, list[Any]] = {}
    for branch in contract["$defs"][name].get("allOf", []):
        action = branch["if"]["properties"]["action"]["const"]
        out[action] = branch["then"]["properties"]["result"]["anyOf"]
    return out


@pytest.mark.parametrize("name", ["ApprovalViewDoc", "ApproveBatchResult"])
def test_the_result_branches_equal_the_actions_the_gated_routes_file(
    contract: dict[str, Any], name: str
) -> None:
    from tests.contract.test_answers_gate import _marked

    filed = {a for pair in _marked().values() for a in pair if a}
    branches = _branches(contract, name)
    assert set(branches) == filed
    assert all(v[-1] == {"type": "null"} for v in branches.values())
    # History form: no `required` at any depth of a branch's variants.
    defs = contract["$defs"]
    for variants in branches.values():
        for variant in variants:
            ref = variant.get("$ref", "")
            target = defs[ref[len(tools.DEFS_REF) :]] if ref else variant
            assert not any("required" in s for s in tools.walk(target)), variant
    assert "HISTORY FORM" in contract["$defs"][name]["description"]


def _view(action: str, result: Any) -> dict[str, Any]:
    from services import approval_service
    from tests.contract.test_answers_gate import _approval

    row = _approval(action)
    row.status, row.result, row.decided_by = "executed", result, "operator"
    row.decided_at = row.created_at
    body = approval_service.as_payload(row)
    return {**body, "approval": approval_service.approval_view(row, approval_service.now())}


def test_a_result_is_checked_by_its_action() -> None:
    from fastapi import Response

    from services.approval_service import _result_of

    check = answer_validator.answer_check()
    deleted = _result_of(Response(status_code=204))
    assert deleted == {"status_code": 204}
    assert check.check(APPROVAL, 200, _view("delete", deleted)) == []
    submitted = {"run_id": "run-1", "status": "pending"}
    assert check.check(APPROVAL, 200, _view("gpu_start", {"run_id": "run-1"})) == []
    assert check.check(APPROVAL, 200, _view("delete", submitted)), "a run passed as a delete"
    assert check.check(APPROVAL, 200, _view("gpu_start", deleted)), "a delete passed as a run"
    assert check.check(APPROVAL, 200, _view("delete", None)) == []


def test_no_ref_sits_beside_a_union(contract: dict[str, Any]) -> None:
    beside = [
        s
        for s in tools.walk([contract["responses"], contract["$defs"]])
        if "$ref" in s and ("anyOf" in s or "oneOf" in s)
    ]
    assert not beside, beside[:3]


def test_a_secret_named_key_in_a_stored_result_is_the_redaction_marker(
    contract: dict[str, Any],
) -> None:
    """Every approval answer redacts a secret-named key (``core.secret_keys``), so in a history
    copy such a property is the marker, not the live type (review round 1, finding 9)."""
    from core.secret_keys import REDACTED, is_secret_key

    seen = 0
    for name, body in contract["$defs"].items():
        if not name.endswith("__history"):
            continue
        for schema in tools.walk(body):
            for prop, sub in (schema.get("properties") or {}).items():
                if is_secret_key(prop):
                    seen += 1
                    assert sub.get("const") == REDACTED, (name, prop, sub)
    assert seen, "no history copy holds a secret-named key: the test would prove nothing"
