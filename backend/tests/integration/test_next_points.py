"""``next_steps`` where the run and pack workflows continue (017 FR-017.7, .8; FTDD §6).

Each of the eleven points is driven over the transport through the production app
(``tests/support/approval_world.py`` stands in for the card, the enqueue and the validator's
non-Hub steps), and the two refusals are triggered. Every ``next_steps`` validates as a list of
the vendored contract's ``next_step``, names a tool in the LIVE MCP registry, and no answer
carries a key ``next`` (miForge adds only the contract's new key).
"""

from __future__ import annotations

import asyncio
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from tests.support.approval_world import AGENT, World, definition
from tests.support.result_language import validate


@pytest.fixture(scope="module")
def live_tools() -> set[str]:
    from mcp_server.config import MCPSettings
    from mcp_server.server import build_server

    mcp, _ = build_server(MCPSettings(auth_token="x" * 32))
    return {t.name for t in asyncio.run(mcp.list_tools())}


def _steps(body: dict[str, Any], live: set[str]) -> list[str]:
    """Assert the contract and the registry; return the tool names in order."""
    assert "next" not in body, body
    steps = body["next_steps"]
    assert isinstance(steps, list) and 1 <= len(steps) <= 5
    for step in steps:
        validate("next_step", step)
        assert step["tool"] in live, step["tool"]
    return [step["tool"] for step in steps]


def _refusal_steps(response: Any, code: str, live: set[str]) -> list[str]:
    error = response.json()["error"]
    assert error["code"] == code, error
    return _steps(error["details"], live)


# ------------------------------------------------------------------ points 1 - 3: validate


def test_1_a_valid_definition_points_to_define(world: World, live_tools: set[str]) -> None:
    body = world.client.post("/api/v1/runs/validate", json=definition()).json()
    assert body["ok"] is True, body
    assert _steps(body, live_tools) == ["forge_define_run"]


def test_2_an_invalid_definition_points_back_to_validate(
    world: World, live_tools: set[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    from core.errors import refuse
    from services import run_definition as validator

    def absent(*_: Any, **__: Any) -> Any:
        raise refuse("midataworks_not_configured", "miDataworks is not configured.")

    monkeypatch.setattr(validator, "_live_services", absent)
    body = world.client.post("/api/v1/runs/validate", json=definition()).json()
    assert body["ok"] is False, body
    assert _steps(body, live_tools) == ["forge_validate_run"]


def test_3_an_estimate_that_fits_points_to_validate(
    world: World, live_tools: set[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    from api.v1.endpoints import runtypes as endpoints

    card = SimpleNamespace(uuid="GPU-x", name="test card", free_memory_bytes=10**13)
    monkeypatch.setattr(endpoints, "resolve_submit_device", lambda request: card)
    monkeypatch.setattr(endpoints, "_candidate_cards", lambda request, device: [])
    response = world.client.post(
        "/api/v1/budget/estimate",
        json={
            "run_type": "change",
            "config": {"learning_rate": 1e-5, "warmup_steps": 0, "batch_size": 1},
            "param_count": 1000,
        },
    )
    assert response.status_code == 200, response.text
    body = response.json()
    assert body["verdict"] == "fits", body
    assert _steps(body, live_tools) == ["forge_validate_run"]


# ------------------------------------------------------------------ points 4 - 6: the run


@pytest.mark.parametrize(
    ("status", "expected"),
    [
        ("pending", ["forge_list_steps", "forge_get_run_stop_conditions"]),
        ("running", ["forge_list_steps", "forge_get_run_stop_conditions"]),
        ("completed", ["forge_list_checkpoints", "forge_list_scores"]),
        ("failed", ["forge_list_checkpoints", "forge_list_scores"]),
    ],
)
def test_4_5_a_run_read_points_by_status(
    world: World, live_tools: set[str], status: str, expected: list[str]
) -> None:
    from tests.support.jobs import insert_run

    with world.session() as session:
        insert_run(session, "run-r", status=status)
        session.commit()
    body = world.client.get("/api/v1/runs/run-r").json()
    assert _steps(body, live_tools) == expected
    assert all(s["args"] == {"run_id": "run-r"} for s in body["next_steps"])


def test_6_an_executed_approval_that_started_a_run_points_to_it(
    world: World, live_tools: set[str]
) -> None:
    filed = world.client.post("/api/v1/runs/define", json=definition("run-x"), headers=AGENT)
    approval_id = filed.json()["approval_id"]
    assert world.client.post(f"/api/v1/approvals/{approval_id}/approve").status_code == 200
    body = world.client.get(f"/api/v1/approvals/{approval_id}").json()
    assert body["status"] == "executed"
    assert _steps(body, live_tools) == ["forge_get_run"]
    assert body["next_steps"][0]["args"] == {"run_id": "run-x"}


# ------------------------------------------------------------------ points 7 - 10: packs


def _slot(world: World, tmp: Path, origin: str, suffix: str) -> dict[str, object]:
    from tests.support import pack_fixtures as fx

    with world.session() as session:
        _part_id, provenance = fx.import_external_part(
            session,
            parts_root=world.root / "parts",
            export_root=tmp / f"export-{suffix}",
            origin=origin,
            content_sha256=suffix[0] * 64,
        )
        session.commit()
    return fx.external_slot(
        origin=provenance["source"],
        content_sha256=provenance["content_sha256"],
        retrieved_at=provenance["retrieved_at"],
    )


def test_7_to_10_the_draft_flow_points_forward(
    world: World, live_tools: set[str], tmp_path: Path
) -> None:
    from packs.slots import SlotKind

    client = world.client
    created = client.post("/api/v1/pack-drafts", json={}).json()
    draft_id = created["draft_id"]
    assert _steps(created, live_tools) == ["forge_commit_draft_step"]  # point 7
    assert created["next_steps"][0]["args"] == {"draft_id": draft_id, "number": 1}

    step1 = client.put(
        f"/api/v1/pack-drafts/{draft_id}/steps/1",
        json={"name": "notary-md", "domain": "maryland notary", "domain_given": True},
    ).json()
    assert _steps(step1, live_tools) == ["forge_commit_draft_step"]  # point 8
    assert step1["next_steps"][0]["args"]["number"] == 2

    response = client.get(f"/api/v1/pack-drafts/{draft_id}/validity")
    assert response.status_code == 200, response.text
    review = response.json()
    assert review["validity"]["state"] != "valid"
    assert _steps(review, live_tools) == ["forge_commit_draft_step"]  # point 9, missing
    assert review["next_steps"][0]["args"]["number"] == 2  # the prompt set's step

    prompts = _slot(world, tmp_path, "/data/prompts.jsonl", "aprompt")
    tests = _slot(world, tmp_path, "/data/test.jsonl", "btest")
    retention = _slot(world, tmp_path, "/data/retain.jsonl", "cretain")
    client.put(
        f"/api/v1/pack-drafts/{draft_id}/steps/2",
        json={"slots": {SlotKind.PROMPT_SET.value: prompts}},
    )
    for skipped in (3, 4):
        client.put(f"/api/v1/pack-drafts/{draft_id}/steps/{skipped}", json={"skip": True})
    client.put(
        f"/api/v1/pack-drafts/{draft_id}/steps/5", json={"slots": {SlotKind.TEST_SET.value: tests}}
    )
    last = client.put(
        f"/api/v1/pack-drafts/{draft_id}/steps/6",
        json={"slots": {SlotKind.RETENTION_SET.value: {**retention, "floor": 0.72}}},
    ).json()
    assert _steps(last, live_tools) == ["forge_review_draft"]  # point 8, after the last step

    review = client.get(f"/api/v1/pack-drafts/{draft_id}/validity").json()
    assert review["validity"]["state"] == "valid", review
    assert _steps(review, live_tools) == ["forge_commit_draft"]  # point 9, complete

    committed = client.post(f"/api/v1/pack-drafts/{draft_id}/commit")
    assert committed.status_code == 201, committed.text
    assert _steps(committed.json(), live_tools) == ["forge_get_pack_validity"]  # point 10
    args = committed.json()["next_steps"][0]["args"]

    validity = client.get(f"/api/v1/packs/{args['pack_id']}/versions/{args['version']}/validity")
    assert validity.json()["state"] == "valid"
    assert _steps(validity.json(), live_tools) == ["forge_validate_run"]  # point 10


# ------------------------------------------------------------------ point 11: plans


def test_11_branch_and_continue_plans_point_to_their_submission(
    world: World, live_tools: set[str]
) -> None:
    from tests.support.checkpoints import write_chain
    from tests.support.jobs import insert_run

    with world.session() as session:
        insert_run(session, "run-p", status="completed")
        session.commit()
    write_chain(world.root / "checkpoints", run_id="run-p", steps=(4,))
    cont = world.client.post("/api/v1/runs/run-p/continue", json={})
    assert cont.status_code == 200, cont.text
    assert _steps(cont.json(), live_tools) == ["forge_continue_run"]
    branch = world.client.post("/api/v1/runs/run-p/checkpoints/4/branch", json={"run_id": "run-q"})
    assert branch.status_code == 200, branch.text
    assert _steps(branch.json(), live_tools) == ["forge_define_run"]


# ------------------------------------------------------------------ the two refusals


def test_an_invalid_define_teaches_validate(
    world: World, live_tools: set[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    from core.errors import refuse
    from services import run_definition as validator

    def absent(*_: Any, **__: Any) -> Any:
        raise refuse("midataworks_not_configured", "miDataworks is not configured.")

    monkeypatch.setattr(validator, "_live_services", absent)
    response = world.client.post("/api/v1/runs/define", json=definition())
    assert _refusal_steps(response, "run_definition_invalid", live_tools) == ["forge_validate_run"]


def test_archiving_a_live_run_teaches_get_run(world: World, live_tools: set[str]) -> None:
    from tests.support.jobs import insert_run

    with world.session() as session:
        insert_run(session, "run-live", status="running")
        session.commit()
    response = world.client.post("/api/v1/runs/run-live/archive")
    assert _refusal_steps(response, "run_not_archivable", live_tools) == ["forge_get_run"]
