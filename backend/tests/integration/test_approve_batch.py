"""Approve every waiting settings change with one operator action  (ENH-006 D79, 2026-10-29).

Operator: "When approving one or more settings changes, I want to approve all outstanding
settings changes with one approval request." ``POST /api/v1/approvals/approve-batch`` with
``{"approval_ids": [...]}``.

Everything here runs through the mounted app on a migrated database, with the real settings
routes as the executors. **What a request did is counted in ``mf_setting_changes``** - the
append-only history every executed settings write appends exactly one row to, naming the
approval it ran for - so "each executed once" is an observation of the effect, not of a stub.

The four properties the operator relies on, each with its own test:

1. each named request is approved **through the single-approve path**: executed once, recorded
   on its own row, decided by the operator, all carrying one ``batch_id``;
2. anything that is not a waiting settings change refuses the **whole** batch (409, naming
   every such id and why) **before** anything is approved;
3. a request that fails while the batch runs is recorded and does not stop the others;
4. an agent cannot do it (403 ``agent_cannot_decide``), exactly like a single approve.
"""

from __future__ import annotations

from typing import Any

import sqlalchemy as sa

from tests.support.approval_world import AGENT, World, definition

URL = "/api/v1/approvals/approve-batch"


def _ask_set(world: World, key: str, value: Any) -> str:
    response = world.client.put(f"/api/v1/settings/{key}", json={"value": value}, headers=AGENT)
    assert response.status_code == 202, response.text
    assert response.json()["action"] == "config_write"
    return str(response.json()["approval_id"])


def _ask_reset(world: World, key: str) -> str:
    response = world.client.delete(f"/api/v1/settings/{key}", headers=AGENT)
    assert response.status_code == 202, response.text
    assert response.json()["action"] == "config_write"
    return str(response.json()["approval_id"])


def _changes(world: World) -> list[tuple[str, str, str | None, str]]:
    """``(key, action, approval_id, actor)`` for every recorded settings change, in order."""
    with world.engine.connect() as conn:
        rows = conn.execute(
            sa.text("SELECT key, action, approval_id, actor FROM mf_setting_changes ORDER BY id")
        ).fetchall()
    return [(row[0], row[1], row[2], row[3]) for row in rows]


def _row(world: World, approval_id: str) -> dict[str, Any]:
    response = world.client.get(f"/api/v1/approvals/{approval_id}")
    assert response.status_code == 200, response.text
    return dict(response.json())


def _served(world: World, key: str) -> Any:
    return world.client.get("/api/v1/settings").json()["settings"][key]["value"]


class TestEachIsApprovedThroughTheSinglePath:
    def test_three_settings_changes_are_each_executed_once_with_one_batch_id(
        self, world: World
    ) -> None:
        # A reset needs an override to clear; the operator sets one first (one history row).
        assert world.client.put("/api/v1/settings/log_level", json={"value": "DEBUG"}).status_code
        before = _changes(world)
        assert len(before) == 1
        ids = [
            _ask_set(world, "checkpoint_lifecycle_enabled", True),
            _ask_set(world, "checkpoint_lifecycle_keep_last_n", 3),
            _ask_reset(world, "log_level"),
        ]
        assert _changes(world) == before, "a request ran before the operator approved it"

        response = world.client.post(URL, json={"approval_ids": ids})
        assert response.status_code == 200, response.text
        body = response.json()
        batch_id = body["batch_id"]
        assert batch_id.startswith("bat_")
        assert body["count"] == 3
        # One result per id, in the order given, each executed.
        assert [item["approval_id"] for item in body["results"]] == ids
        assert [item["status"] for item in body["results"]] == ["executed"] * 3
        assert [item["ok"] for item in body["results"]] == [True, True, True]

        # Each executed EXACTLY once: one new history row per request, naming it.
        after = _changes(world)[len(before) :]
        assert after == [
            ("checkpoint_lifecycle_enabled", "set", ids[0], "agent:forge-mcp"),
            ("checkpoint_lifecycle_keep_last_n", "set", ids[1], "agent:forge-mcp"),
            ("log_level", "reset", ids[2], "agent:forge-mcp"),
        ]
        # ...and the values are in force.
        assert _served(world, "checkpoint_lifecycle_enabled") is True
        assert _served(world, "checkpoint_lifecycle_keep_last_n") == 3
        assert _served(world, "log_level") == "INFO"

        # Recorded individually: each row decided by the operator, with the shared batch id.
        for one in ids:
            row = _row(world, one)
            assert row["status"] == "executed"
            assert row["decided_by"] == "operator"
            assert row["decided_at"] is not None
            assert row["batch_id"] == batch_id
        # History (the listing the Approvals screen reads) shows each of them.
        listed = {row["approval_id"]: row for row in world.approvals()}
        assert {listed[one]["batch_id"] for one in ids} == {batch_id}

    def test_a_second_batch_of_the_same_ids_is_refused_and_runs_nothing(self, world: World) -> None:
        ids = [
            _ask_set(world, "checkpoint_lifecycle_enabled", True),
            _ask_set(world, "checkpoint_lifecycle_keep_last_n", 3),
        ]
        assert world.client.post(URL, json={"approval_ids": ids}).status_code == 200
        once = _changes(world)
        assert len(once) == 2
        again = world.client.post(URL, json={"approval_ids": ids})
        assert again.status_code == 409, again.text
        assert again.json()["error"]["code"] == "approval_batch_refused"
        assert _changes(world) == once, "a request ran twice"

    def test_a_single_approve_still_carries_no_batch_id(self, world: World) -> None:
        one = _ask_set(world, "checkpoint_lifecycle_enabled", True)
        decided = world.client.post(f"/api/v1/approvals/{one}/approve")
        assert decided.json()["status"] == "executed"
        assert decided.json()["batch_id"] is None


class TestRefusedBeforeAnythingRuns:
    def _assert_nothing_approved(self, world: World, ids: list[str]) -> None:
        for one in ids:
            assert _row(world, one)["status"] == "pending", one
            assert _row(world, one)["batch_id"] is None
        assert _changes(world) == []
        assert world.count("mf_settings") == 0

    def test_a_request_that_is_not_a_settings_change_refuses_the_batch(self, world: World) -> None:
        settings = [
            _ask_set(world, "checkpoint_lifecycle_enabled", True),
            _ask_set(world, "checkpoint_lifecycle_keep_last_n", 3),
        ]
        run = world.client.post("/api/v1/runs/define", json=definition(), headers=AGENT)
        assert run.status_code == 202
        gpu = run.json()["approval_id"]

        # The GPU request sits LAST: a loop that approved as it checked would have run the two
        # settings changes before reaching it.
        response = world.client.post(URL, json={"approval_ids": [*settings, gpu]})
        assert response.status_code == 409, response.text
        error = response.json()["error"]
        assert error["code"] == "approval_batch_refused"
        assert error["details"]["refused"] == [
            {"approval_id": gpu, "reason": "not_a_settings_change"}
        ]
        assert gpu in error["message"]
        self._assert_nothing_approved(world, settings)
        assert _row(world, gpu)["status"] == "pending"
        assert world.enqueued == [] and world.leases == []

    def test_every_refused_id_is_named_unknown_decided_and_expired(self, world: World) -> None:
        good = _ask_set(world, "checkpoint_lifecycle_enabled", True)
        decided = _ask_set(world, "checkpoint_lifecycle_keep_last_n", 3)
        rejected = world.client.post(f"/api/v1/approvals/{decided}/reject", json={"reason": "no"})
        assert rejected.status_code == 200
        stale = _ask_set(world, "log_level", "DEBUG")
        with world.engine.begin() as conn:
            conn.execute(
                sa.text(
                    "UPDATE mf_approvals SET expires_at = :past WHERE id = :id",
                ),
                {"past": "2000-01-01 00:00:00", "id": stale},
            )

        response = world.client.post(
            URL, json={"approval_ids": [good, "apr_missing", decided, stale]}
        )
        assert response.status_code == 409, response.text
        assert response.json()["error"]["details"]["refused"] == [
            {"approval_id": "apr_missing", "reason": "not_found"},
            {"approval_id": decided, "reason": "not_pending (rejected)"},
            {"approval_id": stale, "reason": "not_pending (expired)"},
        ]
        self._assert_nothing_approved(world, [good])

    def test_an_empty_or_repeated_list_is_refused(self, world: World) -> None:
        one = _ask_set(world, "checkpoint_lifecycle_enabled", True)
        empty = world.client.post(URL, json={"approval_ids": []})
        assert empty.status_code == 422, empty.text
        repeated = world.client.post(URL, json={"approval_ids": [one, one]})
        assert repeated.status_code == 422, repeated.text
        assert repeated.json()["error"]["code"] == "approval_batch_invalid"
        assert repeated.json()["error"]["details"]["duplicates"] == [one]
        self._assert_nothing_approved(world, [one])


class TestOneFailingDoesNotStopTheOthers:
    def test_a_request_whose_executor_refuses_is_failed_and_the_rest_run(
        self, world: World
    ) -> None:
        assert world.client.put("/api/v1/settings/log_level", json={"value": "DEBUG"}).status_code
        first = _ask_set(world, "checkpoint_lifecycle_enabled", True)
        reset = _ask_reset(world, "log_level")
        last = _ask_set(world, "checkpoint_lifecycle_keep_last_n", 3)
        # The operator clears the override by hand, so the waiting reset has nothing to clear:
        # its own route refuses it when it runs.
        assert world.client.delete("/api/v1/settings/log_level").status_code == 200
        before = _changes(world)

        response = world.client.post(URL, json={"approval_ids": [first, reset, last]})
        assert response.status_code == 200, response.text
        results = response.json()["results"]
        assert [item["status"] for item in results] == ["executed", "failed", "executed"]
        assert results[1]["ok"] is False
        assert results[1]["error"]["code"] == "setting_not_overridden"
        assert [change[2] for change in _changes(world)[len(before) :]] == [first, last]
        batch_id = response.json()["batch_id"]
        assert {_row(world, one)["batch_id"] for one in (first, reset, last)} == {batch_id}
        assert _row(world, reset)["error"]["code"] == "setting_not_overridden"

    def test_a_request_decided_while_the_batch_runs_is_reported_and_the_rest_run(
        self, world: World, monkeypatch
    ) -> None:
        """Another decision gets to the second request between the check and its turn."""
        from services import approval_service

        first = _ask_set(world, "checkpoint_lifecycle_enabled", True)
        raced = _ask_set(world, "checkpoint_lifecycle_keep_last_n", 3)
        last = _ask_set(world, "log_level", "DEBUG")
        real = approval_service.approve

        def racing(session: Any, approval_id: str, **kwargs: Any) -> Any:
            if approval_id == raced:
                approval_service.reject(session, raced, reason="decided elsewhere")
            return real(session, approval_id, **kwargs)

        monkeypatch.setattr(approval_service, "approve", racing)
        response = world.client.post(URL, json={"approval_ids": [first, raced, last]})
        assert response.status_code == 200, response.text
        results = response.json()["results"]
        assert [item["status"] for item in results] == ["executed", "rejected", "executed"]
        assert results[1]["error"]["code"] == "approval_not_pending"
        assert [change[2] for change in _changes(world)] == [first, last]


class TestOnlyTheOperatorDecides:
    def test_an_agent_is_refused_and_nothing_runs(self, world: World) -> None:
        ids = [
            _ask_set(world, "checkpoint_lifecycle_enabled", True),
            _ask_set(world, "checkpoint_lifecycle_keep_last_n", 3),
        ]
        response = world.client.post(URL, json={"approval_ids": ids}, headers=AGENT)
        assert response.status_code == 403, response.text
        assert response.json()["error"]["code"] == "agent_cannot_decide"
        for one in ids:
            assert _row(world, one)["status"] == "pending"
        assert _changes(world) == []

    def test_the_route_is_in_the_live_openapi_and_has_no_mcp_tool(self, world: World) -> None:
        paths = world.client.get("/api/openapi.json").json()["paths"]
        assert "post" in paths["/api/v1/approvals/approve-batch"]
        from tests.support.mcp_ledger import ROUTE_EXEMPT

        assert ("POST", "/api/v1/approvals/approve-batch") in ROUTE_EXEMPT
