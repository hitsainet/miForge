"""The Approvals screen's fixture is what the gate really stores  (F12 task 5.5; FR-012.18).

``frontend/e2e/fixtures/approvals.json`` feeds the Playwright spec and the Vitest screen test.
It is not hand-written: this test sends one agent request of each action through the mounted
app - a run definition (``gpu_start``), an attach definition (``hotupdate_attach``), a draft
delete (``delete``) and a setting change (``config_write``) - reads ``GET /api/v1/approvals``
and compares it with the committed file, so a change in what the gate stores turns this red
before the screen silently renders a shape it no longer receives.

Only the values that differ per run are normalised: the approval ids, the timestamps.

``MIFORGE_WRITE_UI_FIXTURES=1 pytest backend/tests/integration/test_ui_approval_fixtures.py``
rewrites the file; review the frontend diff afterwards.
"""

from __future__ import annotations

import json
import os
from typing import Any

from tests.support.approval_world import AGENT, World, definition
from tests.support.db import REPO_ROOT

FIXTURE = REPO_ROOT / "frontend" / "e2e" / "fixtures" / "approvals.json"
WRITE = os.environ.get("MIFORGE_WRITE_UI_FIXTURES") == "1"
CREATED = "2026-10-19T12:00:00+00:00"
EXPIRES = "2026-10-20T12:00:00+00:00"


def _normalised(listing: dict[str, Any]) -> dict[str, Any]:
    rows = sorted(listing["approvals"], key=lambda row: row["action"])
    out = []
    for index, row in enumerate(rows, start=1):
        out.append(
            {
                **row,
                "approval_id": f"apr_fixture_{index}",
                "created_at": CREATED,
                "expires_at": EXPIRES,
            }
        )
    return {**listing, "approvals": out}


def test_the_e2e_approvals_are_what_the_gate_stores(world: World) -> None:
    client = world.client
    assert client.post("/api/v1/runs/define", json=definition(), headers=AGENT).status_code == 202
    attach = definition("run-attach", mode="attach")
    assert client.post("/api/v1/runs/define", json=attach, headers=AGENT).status_code == 202
    draft = client.post("/api/v1/pack-drafts", json={"name": "legal-draft"}).json()["draft_id"]
    assert client.delete(f"/api/v1/pack-drafts/{draft}", headers=AGENT).status_code == 202
    put = client.put(
        "/api/v1/settings/checkpoint_lifecycle_enabled", json={"value": True}, headers=AGENT
    )
    assert put.status_code == 202

    listing = client.get("/api/v1/approvals").json()
    assert listing["pending"] == 4
    answer = _normalised(listing)
    # The draft id is minted per run; the fixture names a fixed one.
    for row in answer["approvals"]:
        if row["action"] == "delete":
            row["summary"] = {**row["summary"], "target": "pack draft draft-fixture"}
            row["payload"] = {"draft_id": "draft-fixture"}
            row["request_digest"] = "sha256:" + "0" * 64
    document = json.dumps(answer, indent=2, sort_keys=True) + "\n"
    if WRITE:
        FIXTURE.parent.mkdir(parents=True, exist_ok=True)
        FIXTURE.write_text(document)
    assert FIXTURE.is_file(), f"Generate it: MIFORGE_WRITE_UI_FIXTURES=1 pytest {__file__}"
    assert json.loads(FIXTURE.read_text()) == json.loads(document), (
        f"{FIXTURE} no longer matches what the gate stores: regenerate it with "
        "MIFORGE_WRITE_UI_FIXTURES=1 and review the frontend diff."
    )
