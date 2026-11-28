"""A fresh WSP worker pod per session (operator 2026-11-26, option B of the 019 P4 restart debt),
through the real driver and intake (``dispatch.drive`` over fake workers) and the production
Celery task:

- after a session is acked, the driver asks for the worker's pod to be replaced: ONCE, with the
  worker's index and the time its boot was first seen; also after it recycles a stale idle boot;
  never when the ack itself failed;
- a refused or failing replacement is logged and the run is unaffected (today's container restart
  is the fallback);
- the pool reads the worker ``restarting`` (not ``not_answering``) until its new pod answers, and
  nothing is claimed on it until a fresh boot answers ``/ready``;
- the production task builds the recycler from ``WSP_WORKER_POD_NAME`` and the pod name it deletes
  is ``miforge-wsp-worker-<index>``.
"""

from __future__ import annotations

import datetime as dt
import json
import logging
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pytest
import sqlalchemy as sa

from tests.integration.wsp.test_dispatch import (
    FakeWorker,
    World,
    _drive_task,
    _running,
    archive,
    build_world,
)
from wsp import pool
from wsp.dispatch import WorkerUnreachableError
from wsp.models import WspWorkerBoot
from wsp.settings import get_wsp_settings


@dataclass
class Recycler:
    """Records every replacement asked for; answers ``outcome`` or raises ``error``."""

    outcome: str = "deleted"
    error: Exception | None = None

    def __post_init__(self) -> None:
        self.calls: list[tuple[int, dt.datetime]] = []

    def recycle(self, index: int, served_since: dt.datetime) -> str:
        self.calls.append((index, served_since))
        if self.error is not None:
            raise self.error
        return self.outcome


@pytest.fixture()
def world(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Iterator[World]:
    built = build_world(tmp_path, monkeypatch)
    yield built
    built.engine.dispose()
    get_wsp_settings.cache_clear()


def _finished(worker: FakeWorker, run_id: str) -> None:
    worker.state = "finished"
    worker.data = archive(
        worker.boot,
        run_id,
        **{
            "result.json": json.dumps({"stop_reason": "submitted"}).encode(),
            "transcript.jsonl": b'{"role": "user", "content": "t"}\n',
        },
    )


def _first_seen(world: World, boot: str) -> dt.datetime:
    with world.factory() as session:
        row = session.get(WspWorkerBoot, boot)
        assert row is not None
        return row.first_seen_at


def _pool(world: World, now: dt.datetime | None = None) -> dict[str, Any]:
    with world.factory() as session:
        return pool.pool_document(session, 1, world.settings, now=now)


def test_an_acked_session_asks_once_for_its_workers_pod_with_its_boots_time(
    world: World,
) -> None:
    run_id, worker = _running(world)
    _finished(worker, run_id)
    recycler = Recycler()
    actions = world.drive(worker, recycle_pod=recycler)
    assert f"acked:{run_id}" in actions
    assert "pod_replaced:0:deleted" in actions
    assert recycler.calls == [(0, _first_seen(world, worker.boot))]
    assert world.run(run_id)["status"] == "completed"
    # The next pass sees the old pod still answering on the ended boot, exiting as a real
    # supervisor does after a valid ack: nothing more is asked.
    worker.state = "exiting"
    world.drive(worker, recycle_pod=recycler)
    assert len(recycler.calls) == 1


@pytest.mark.parametrize(
    "make",
    [lambda: Recycler(outcome="refused:get_403"), lambda: Recycler(error=RuntimeError("down"))],
    ids=["refused", "raises"],
)
def test_a_refused_replacement_never_touches_the_run(
    world: World, make: Any, caplog: pytest.LogCaptureFixture
) -> None:
    recycler = make()
    run_id, worker = _running(world)
    _finished(worker, run_id)
    with caplog.at_level(logging.WARNING, logger="wsp.dispatch"):
        actions = world.drive(worker, recycle_pod=recycler)
    assert f"acked:{run_id}" in actions
    assert world.run(run_id)["status"] == "completed"
    assert len(recycler.calls) == 1
    word = [a for a in actions if a.startswith("pod_replaced:0:")]
    assert word == [
        f"pod_replaced:0:{recycler.outcome if recycler.error is None else 'refused:RuntimeError'}"
    ]
    if recycler.error is not None:
        failed = [
            r for r in caplog.records if getattr(r, "code", None) == "wsp_worker_pod_delete_failed"
        ]
        assert len(failed) == 1 and failed[0].worker_index == 0  # type: ignore[attr-defined]


def test_a_failed_ack_asks_for_nothing(world: World) -> None:
    run_id, worker = _running(world)
    _finished(worker, run_id)

    def lost(*_: Any) -> None:
        raise WorkerUnreachableError("ack lost")

    worker.ack = lost  # type: ignore[method-assign]
    recycler = Recycler()
    actions = world.drive(worker, recycle_pod=recycler)
    assert f"ack_failed:{run_id}" in actions
    assert recycler.calls == []


def test_without_a_recycler_nothing_is_asked_and_the_session_still_ends(world: World) -> None:
    run_id, worker = _running(world)
    _finished(worker, run_id)
    actions = world.drive(worker)
    assert f"acked:{run_id}" in actions
    assert not [a for a in actions if a.startswith("pod_replaced")]


def test_a_recycled_stale_idle_boot_replaces_the_pod_too(world: World) -> None:
    """A claim whose push never happened leaves the worker idle on a consumed boot; the driver
    sends it /recycle (the supervisor exits) and replaces its pod."""
    run_id, worker = _running(world)
    with world.engine.begin() as conn:
        conn.execute(
            sa.text("UPDATE mf_wsp_runs SET status = 'failed' WHERE run_id = :r"), {"r": run_id}
        )
    worker.state = "idle"
    recycler = Recycler()
    actions = world.drive(worker, recycle_pod=recycler)
    assert f"recycled:{worker.boot}" in actions
    assert recycler.calls == [(0, _first_seen(world, worker.boot))]


def test_the_pool_reads_restarting_until_a_fresh_pod_answers_and_only_then_claims(
    world: World,
) -> None:
    run_id, worker = _running(world)
    _finished(worker, run_id)
    world.drive(worker, recycle_pod=Recycler())
    # The old pod is going: still answering on the ended boot, then silent.
    worker.state = "exiting"
    nxt = world.queue()
    actions = world.drive(worker, recycle_pod=Recycler())
    assert not [a for a in actions if "pushed" in a or "claimed" in a], actions
    body = _pool(world)
    assert [(w["index"], w["state"]) for w in body["workers"]] == [(0, "restarting")]
    assert (body["restarting"], body["not_answering"], body["idle"]) == (1, 0, 0)
    worker.unreachable = True
    world.drive(worker, recycle_pod=Recycler())
    assert _pool(world)["workers"][0]["state"] == "restarting"
    assert world.run(nxt)["status"] == "queued"
    # The StatefulSet's new pod answers with a fresh boot: idle, and the queued run is claimed.
    fresh = FakeWorker()
    actions = world.drive(fresh, recycle_pod=Recycler())
    assert world.run(nxt)["status"] in ("pushing", "running"), actions
    assert world.run(nxt)["boot_id"] == fresh.boot


def test_a_silence_past_the_restart_window_is_not_answering(world: World) -> None:
    run_id, worker = _running(world)
    _finished(worker, run_id)
    world.drive(worker, recycle_pod=Recycler())
    worker.unreachable = True
    world.drive(worker)
    now = dt.datetime.now(dt.UTC)
    assert _pool(world, now)["workers"][0]["state"] == "restarting"
    late = now + dt.timedelta(seconds=pool.RESTART_WINDOW_S + 5)
    body = _pool(world, late)
    assert body["workers"][0]["state"] == "not_answering"
    assert (body["restarting"], body["not_answering"]) == (0, 1)


def test_an_unconsumed_silent_worker_is_still_not_answering(world: World) -> None:
    world.drive(FakeWorker(unreachable=True))
    assert _pool(world)["workers"][0]["state"] == "not_answering"


def test_the_production_task_replaces_miforge_wsp_worker_index_after_the_ack(
    world: World, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Through ``wsp.tasks.drive``: ``WSP_WORKER_POD_NAME`` builds the Kubernetes recycler, and
    after the ack it is asked for ``miforge-wsp-worker-0`` exactly once."""
    from wsp.pod_recycler import KubernetesPodRecycler

    asked: list[tuple[str, str]] = []

    def recycle(self: KubernetesPodRecycler, index: int, served_since: dt.datetime) -> str:
        asked.append((self.namespace, self.pod_name(index)))
        return "deleted"

    monkeypatch.setattr(KubernetesPodRecycler, "recycle", recycle)
    monkeypatch.setenv("WSP_WORKER_POD_NAME", "miforge-wsp-worker-{index}")
    monkeypatch.setenv("WSP_WORKER_POD_NAMESPACE", "miforge")
    run_id = world.queue()
    worker, answer = _drive_task(world, monkeypatch)
    assert world.run(run_id)["status"] == "running", answer
    _finished(worker, run_id)
    _, answer = _drive_task_with(world, monkeypatch, worker)
    assert f"acked:{run_id}" in answer["actions"]
    assert "pod_replaced:0:deleted" in answer["actions"]
    assert asked == [("miforge", "miforge-wsp-worker-0")]


def test_the_production_task_without_a_pod_name_replaces_nothing(
    world: World, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv("WSP_WORKER_POD_NAME", raising=False)
    run_id = world.queue()
    worker, _ = _drive_task(world, monkeypatch)
    _finished(worker, run_id)
    _, answer = _drive_task_with(world, monkeypatch, worker)
    assert f"acked:{run_id}" in answer["actions"]
    assert not [a for a in answer["actions"] if a.startswith("pod_replaced")]


def _drive_task_with(world: World, monkeypatch: pytest.MonkeyPatch, worker: Any) -> Any:
    """One more production pass on the SAME fake worker (``_drive_task`` makes a new one)."""
    from workers.celery_app import celery_app

    monkeypatch.setattr("wsp.worker_client.HttpWorker", lambda url: worker)
    get_wsp_settings.cache_clear()
    return worker, celery_app.tasks["miforge.wsp.drive"].run()
