"""A WSP run's live progress (2026-11-20): the heartbeat the driver stores, and the progress the
API serves beside the run document (``GET /wsp/runs/{id}/progress``, and a compact form on the list
rows).

Every case runs the real driver (``dispatch.drive`` over the fake worker) and reads through the
production app's routes (``create_app``), so removing the heartbeat store, the route registration
or the list field is red here, not only in a unit test of ``wsp.progress``.
"""

from __future__ import annotations

import datetime as dt
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest
import sqlalchemy as sa
from fastapi.testclient import TestClient

from tests.integration.wsp.test_dispatch import World, _running, build_world
from tests.integration.wsp.test_run_cause import _client
from tests.integration.wsp.wsp_db import insert
from wsp import dispatch, runs, schemas
from wsp.settings import get_wsp_settings


@pytest.fixture()
def world(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Iterator[World]:
    built = build_world(tmp_path, monkeypatch)
    yield built
    built.engine.dispose()
    get_wsp_settings.cache_clear()


@pytest.fixture()
def client(world: World, monkeypatch: pytest.MonkeyPatch) -> TestClient:
    return _client(world, monkeypatch)


def _at(value: Any) -> dt.datetime:
    """A stored time as an aware datetime: SQLite's raw row holds text, PostgreSQL a datetime."""
    parsed = dt.datetime.fromisoformat(value) if isinstance(value, str) else value
    return parsed if parsed.tzinfo is not None else parsed.replace(tzinfo=dt.UTC)


def _usage(world: World, run_id: str, seq: int, **values: Any) -> None:
    with world.engine.begin() as conn:
        insert(
            conn,
            "mf_wsp_usage",
            run_id=run_id,
            seq=seq,
            endpoint_id="ep-local",
            at=dt.datetime.now(dt.UTC),
            **{"input_tokens": 100, "output_tokens": 50, "cost_usd": 0, **values},
        )


# --------------------------------------------------------------------------- the heartbeat


def test_each_heartbeat_stores_the_workers_state_and_elapsed_time(world: World) -> None:
    run_id, worker = _running(world)
    worker.elapsed = 42.7
    beat = dt.datetime(2026, 11, 20, 12, 0, 5, tzinfo=dt.UTC)
    world.drive(worker, now=beat)
    row = world.run(run_id)
    assert (row["worker_state"], row["worker_elapsed_s"]) == ("running", 42)
    assert _at(row["last_status_at"]) == beat
    # The next heartbeat overwrites the last one.
    worker.elapsed = 61.0
    world.drive(worker, now=beat + dt.timedelta(seconds=19))
    row = world.run(run_id)
    assert row["worker_elapsed_s"] == 61
    assert _at(row["last_status_at"]) == beat + dt.timedelta(seconds=19)


@pytest.mark.parametrize(
    ("state", "elapsed", "stored"),
    [
        ("<script>alert(1)</script>", 5, ("unknown", 5)),
        ("running", -3, ("running", None)),
        ("running", float("nan"), ("running", None)),
        ("running", float("inf"), ("running", None)),
        ("running", True, ("running", None)),
        ("running", "60", ("running", None)),
        ("running", 10**9, ("running", None)),
        (None, 1, ("unknown", 1)),
    ],
)
def test_a_hostile_heartbeat_is_bounded_before_it_is_stored(
    world: World, state: Any, elapsed: Any, stored: tuple[Any, Any]
) -> None:
    """The worker is untrusted (P8/P9): its state and elapsed time are bounded on the way in."""
    assert dispatch.worker_heartbeat({"state": state, "elapsed_s": elapsed}) == {
        "worker_state": stored[0],
        "worker_elapsed_s": stored[1],
    }
    run_id, worker = _running(world)
    worker.elapsed = elapsed
    real = worker.status

    def status() -> dict[str, Any]:
        return {**real(), "state": state}

    worker.status = status  # type: ignore[method-assign]
    world.drive(worker)
    row = world.run(run_id)
    assert (row["worker_state"], row["worker_elapsed_s"]) == stored


# ---------------------------------------------------------------------- the progress route


def test_a_running_runs_progress_has_its_heartbeat_spend_and_budgets(
    world: World, client: TestClient
) -> None:
    run_id, worker = _running(world)
    worker.elapsed = 12.0
    world.drive(worker)
    _usage(world, run_id, 0, outcome="served", http_status=200, latency_ms=900)
    _usage(world, run_id, 1, outcome="served", http_status=200, latency_ms=1100)
    _usage(
        world,
        run_id,
        2,
        input_tokens=0,
        output_tokens=0,
        outcome="refused",
        refusal_code="wsp_budget_exhausted",
    )
    answer = client.get(f"/api/v1/wsp/runs/{run_id}/progress")
    assert answer.status_code == 200, answer.text
    p = answer.json()
    assert p["run_id"] == run_id and p["status"] == "running" and p["live"] is True
    assert p["clock_from"] == "started_at" and p["elapsed_s"] >= 0
    assert p["worker"]["state"] == "running" and p["worker"]["elapsed_s"] == 12
    assert p["worker"]["last_heartbeat_at"] is not None and p["worker"]["heartbeat_age_s"] >= 0
    metered = p["metered"]
    assert (metered["calls"], metered["refused"]) == (2, 1)  # a refusal is not a call
    assert (metered["input_tokens"], metered["output_tokens"], metered["tokens"]) == (200, 100, 300)
    assert metered["last_call_at"] is not None
    budgets = {b["name"]: b for b in p["budgets"]}
    assert budgets["tokens"]["spent"] == 300 and budgets["tokens"]["cap"] == 1000
    assert budgets["tokens"]["fraction"] == 0.3
    assert budgets["time_s"]["cap"] == 60 and budgets["time_s"]["spent"] is not None
    assert budgets["cost_usd"]["cap"] is None and budgets["cost_usd"]["fraction"] is None
    steps = budgets["steps"]
    assert steps["estimate"] is True and steps["unit"] == "editor calls"
    # The local default since 2026-11-21 (freeze.default_budgets): a safety stop of 100 steps.
    assert (steps["spent"], steps["cap"]) == (2, 100)
    assert p["queue"] is None
    # The run document itself is untouched: still valid miforge.wsp-run/v1, no progress in it.
    document = client.get(f"/api/v1/wsp/runs/{run_id}").json()
    schemas.validate_run(document)
    assert "progress" not in document and "worker" not in document
    # The run's reported requests count the metered calls, never the refusal.
    with world.factory() as session:
        assert runs.metered_usage(session, run_id)["requests"] == 2


def test_a_queued_runs_progress_says_its_place_in_the_queue(
    world: World, client: TestClient
) -> None:
    active, _worker = _running(world)
    first, second = world.queue(), world.queue()
    p = client.get(f"/api/v1/wsp/runs/{second}/progress").json()
    assert p["status"] == "queued" and p["clock_from"] == "created_at" and p["worker"] is None
    assert p["queue"] == {"position": 2, "queued": 2, "active_run_id": active}
    assert client.get(f"/api/v1/wsp/runs/{first}/progress").json()["queue"]["position"] == 1


def test_a_finished_run_reports_final_figures_and_is_not_live(
    world: World, client: TestClient
) -> None:
    run_id, _worker = _running(world)
    _usage(world, run_id, 0)
    with world.factory() as session:
        runs.transition(session, run_id, "running", "cancelled", stop_reason="cancelled")
        session.commit()
    p = client.get(f"/api/v1/wsp/runs/{run_id}/progress").json()
    assert p["live"] is False and p["finished_at"] is not None
    assert p["worker"]["heartbeat_age_s"] is None  # no age for a run nothing watches any more
    assert p["metered"]["calls"] == 1


def test_an_unknown_runs_progress_is_404(client: TestClient) -> None:
    answer = client.get("/api/v1/wsp/runs/wsp_ffffffffffff/progress")
    assert answer.status_code == 404 and answer.json()["error"]["code"] == "wsp_run_not_found"


# ---------------------------------------------------------------------------- the list rows


def test_list_rows_carry_compact_progress_for_live_runs_only(
    world: World, client: TestClient
) -> None:
    running, worker = _running(world)
    worker.elapsed = 3.0
    world.drive(worker)
    _usage(world, running, 0)
    queued = world.queue()
    with world.engine.begin() as conn:  # a finished run beside them
        conn.execute(
            sa.text(
                "UPDATE mf_wsp_runs SET status = 'failed', stop_reason = 'error', "
                "finished_at = created_at WHERE run_id = :r"
            ),
            {"r": world.queue()},
        )
    rows = {r["run_id"]: r for r in client.get("/api/v1/wsp/runs").json()["runs"]}
    assert len(rows) == 3
    live = rows[running]["progress"]
    assert live["worker_state"] == "running" and live["calls"] == 1 and live["tokens"] == 150
    assert live["queue_position"] is None and live["heartbeat_age_s"] is not None
    assert {b["name"] for b in live["budgets"]} == {"tokens", "cost_usd", "time_s", "steps"}
    assert rows[queued]["progress"]["queue_position"] == 1
    assert rows[queued]["progress"]["clock_from"] == "created_at"
    (finished,) = [r for r in rows.values() if r["status"] == "failed"]
    assert finished["progress"] is None
