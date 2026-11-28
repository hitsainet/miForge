"""An idle worker's boot is retired by the driver, through its pod, before the supervisor's own
idle recycle ends its container (2026-11-26, the residue of the 019 P4 restart debt).

F3 (``wsp.pod_recycler``) replaces a worker's pod after every ack and every ``/recycle`` the driver
sends. One exit stayed outside it: the supervisor's own idle recycle (``IDLE_RECYCLE_S``, 3,600 s),
which the driver never saw. Live on 2026-11-26 both worker pods were 120 minutes old with two
container restarts each, every one ``Completed``, exit 0, exactly 3,600 s after the last. Each such
exit (a) is a container restart, not a fresh pod; (b) left the pool reading ``not_answering`` for
the pass that met the gap, because the boot was never consumed; and (c) could take a run with it:
a claim read ``idle`` seconds before the recycle, and its push met the next boot and failed
``worker_push_failed``.

Now ``/ready`` carries ``idle_recycle_in_s`` and a pass, finding an idle, unconsumed boot within
``dispatch.RETIRE_BEFORE_S`` of it, consumes the boot (so nothing is claimed on it and the pool
reads ``restarting``), sends ``/recycle`` and replaces the pod, as after an ack. Through the real
driver over fake workers, and the production Celery task.
"""

from __future__ import annotations

import datetime as dt
import math
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pytest
import sqlalchemy as sa

from tests.integration.wsp.test_dispatch import FakeWorker, World, build_world
from tests.integration.wsp.test_fresh_pod_per_session import Recycler, _drive_task_with
from tests.support.wsp_drive import resident_window, slots
from wsp import dispatch, pool
from wsp.models import WspWorkerBoot
from wsp.settings import get_wsp_settings


@dataclass
class ClockedWorker(FakeWorker):
    """A fake worker whose ``/ready`` carries the supervisor's ``idle_recycle_in_s``; the key is
    absent when ``left`` is the ``ABSENT`` marker (an image from before this change)."""

    left: Any = "ABSENT"

    def ready(self) -> dict[str, Any]:
        answer = super().ready()
        if self.left != "ABSENT":
            answer["idle_recycle_in_s"] = self.left
        return answer

    def close(self) -> None:  # the production task closes every client
        return


NEAR = 120.0  # inside the margin: the pass retires it


@pytest.fixture()
def world(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Iterator[World]:
    built = build_world(tmp_path, monkeypatch)
    yield built
    built.engine.dispose()
    get_wsp_settings.cache_clear()


def _boot_row(world: World, boot: str) -> WspWorkerBoot | None:
    with world.factory() as session:
        row = session.get(WspWorkerBoot, boot)
        if row is not None:
            session.expunge(row)
        return row


def _claim_only(world: World, worker: Any, recycle_pod: Any = None) -> list[str]:
    """One pass whose enqueued push never runs (the push task has not started yet)."""
    return dispatch.drive(
        world.factory,
        slots(worker),
        world.settings,
        read_sensitivity=world.read,
        files_for=lambda run, bundle: [],
        enqueue_push=lambda _run: None,
        recycle_pod=recycle_pod,
        read_window=resident_window,  # 2026-11-27: a model is resident, so the run is claimed
    ).actions


def _pool(world: World, now: dt.datetime | None = None) -> dict[str, Any]:
    with world.factory() as session:
        return pool.pool_document(session, 1, world.settings, now=now)


def test_a_boot_near_its_idle_recycle_is_retired_through_its_pod_and_never_claimed(
    world: World,
) -> None:
    run_id = world.queue()
    worker = ClockedWorker(left=NEAR)
    recycler = Recycler()
    actions = world.drive(worker, recycle_pod=recycler)
    # The boot is consumed with no run, then recycled, then its pod replaced: once each.
    assert f"retired:{worker.boot}" in actions, actions
    assert f"recycled:{worker.boot}" in actions, actions
    row = _boot_row(world, worker.boot)
    assert row is not None and row.run_id is None
    assert [c for c in worker.calls if c[0] in ("recycle", "push")] == [("recycle", worker.boot)]
    assert recycler.calls == [(0, row.first_seen_at)]
    assert "pod_replaced:0:deleted" in actions
    # Nothing was claimed on a boot about to end.
    assert world.run(run_id)["status"] == "queued"
    assert not [a for a in actions if a.startswith("claimed")], actions


def test_the_pool_reads_restarting_through_the_replacement_and_the_new_pod_takes_the_run(
    world: World,
) -> None:
    run_id = world.queue()
    worker = ClockedWorker(left=NEAR)
    world.drive(worker, recycle_pod=Recycler())
    body = _pool(world)
    assert [(w["index"], w["state"]) for w in body["workers"]] == [(0, "restarting")]
    assert (body["restarting"], body["not_answering"], body["idle"]) == (1, 0, 0)
    # The old pod is exiting, then silent: still restarting, and the run still waits.
    worker.state, worker.left = "exiting", None
    world.drive(worker, recycle_pod=Recycler())
    assert _pool(world)["workers"][0]["state"] == "restarting"
    worker.unreachable = True
    world.drive(worker, recycle_pod=Recycler())
    assert _pool(world)["workers"][0]["state"] == "restarting"
    assert world.run(run_id)["status"] == "queued"
    # The StatefulSet's new pod answers on a fresh boot with its hour ahead: claimed there.
    fresh = ClockedWorker(left=3599.0)
    world.drive(fresh, recycle_pod=Recycler())
    assert world.run(run_id)["status"] in ("pushing", "running")
    assert world.run(run_id)["boot_id"] == fresh.boot
    assert _pool(world)["workers"][0]["state"] == "busy"


@pytest.mark.parametrize(
    "left",
    [
        "ABSENT",
        None,
        dispatch.RETIRE_BEFORE_S + 1,
        3600.0,
        True,
        "10",
        math.nan,
        math.inf,
        -math.inf,
    ],
    ids=["absent", "null", "outside-margin", "fresh", "bool", "string", "nan", "inf", "-inf"],
)
def test_a_boot_with_its_time_ahead_or_no_reading_is_claimed_as_before(
    world: World, left: Any
) -> None:
    """An image from before this change sends no ``idle_recycle_in_s``; a worker is untrusted, so
    anything but a finite number is no reading. Either way the pass behaves as it did."""
    run_id = world.queue()
    worker = ClockedWorker(left=left)
    recycler = Recycler()
    actions = world.drive(worker, recycle_pod=recycler)
    assert not [a for a in actions if a.startswith(("retired", "recycled"))], actions
    assert recycler.calls == []
    assert world.run(run_id)["status"] in ("pushing", "running"), actions
    assert world.run(run_id)["boot_id"] == worker.boot


def test_the_margin_edge_retires(world: World) -> None:
    worker = ClockedWorker(left=dispatch.RETIRE_BEFORE_S)
    assert f"retired:{worker.boot}" in world.drive(worker, recycle_pod=Recycler())


@pytest.mark.parametrize("state", ["receiving", "running", "finished", "exiting"])
def test_only_an_idle_boot_is_retired(world: World, state: str) -> None:
    worker = ClockedWorker(left=NEAR, state=state)
    actions = world.drive(worker, recycle_pod=Recycler())
    assert not [a for a in actions if a.startswith("retired")], actions
    assert _boot_row(world, worker.boot) is None


def test_a_claimed_boot_waiting_for_its_push_is_left_to_its_push(world: World) -> None:
    """A boot already consumed by a claim belongs to its run: retiring it would fail that push.
    The claim refused it within the margin in the first place."""
    run_id = world.queue()
    worker = ClockedWorker(left=dispatch.RETIRE_BEFORE_S + 60)
    _claim_only(world, worker)
    assert world.run(run_id)["status"] == "pushing"
    worker.left = NEAR  # the push has not arrived yet; the supervisor is still idle
    actions = _claim_only(world, worker, recycle_pod=Recycler())
    assert not [a for a in actions if a.startswith(("retired", "recycled"))], actions
    assert world.run(run_id)["status"] == "pushing"


def test_without_a_recycler_the_boot_is_still_retired_and_recycled(world: World) -> None:
    """Dev and the local launcher replace no pod: ``/recycle`` ends the process, as the idle
    recycle would have, but the pass knows it and claims nothing on it."""
    run_id = world.queue()
    worker = ClockedWorker(left=NEAR)
    actions = world.drive(worker)
    assert f"recycled:{worker.boot}" in actions
    assert not [a for a in actions if a.startswith("pod_replaced")]
    assert world.run(run_id)["status"] == "queued"


def test_a_lost_recycle_is_retried_by_the_next_pass(world: World) -> None:
    """The boot is consumed before ``/recycle``: if that call is lost, the next pass finds an idle
    worker on a consumed boot no run owns, and the stale-boot rule recycles it and replaces the
    pod."""
    worker = ClockedWorker(left=NEAR)
    sent: list[str] = []

    def lost(boot_id: str) -> None:
        sent.append(boot_id)
        raise dispatch.WorkerUnreachableError("lost")

    worker.recycle = lost  # type: ignore[method-assign]
    recycler = Recycler()
    actions = world.drive(worker, recycle_pod=recycler)
    assert f"retired:{worker.boot}" in actions
    assert recycler.calls == []
    del worker.recycle  # the next call reaches the worker
    actions = world.drive(worker, recycle_pod=recycler)
    assert f"recycled:{worker.boot}" in actions
    assert len(recycler.calls) == 1 and sent == [worker.boot]


def test_two_passes_retire_a_boot_once(world: World) -> None:
    worker = ClockedWorker(left=NEAR)
    recycler = Recycler()
    world.drive(worker, recycle_pod=recycler)
    worker.state = "exiting"
    actions = world.drive(worker, recycle_pod=recycler)
    assert not [a for a in actions if a.startswith(("retired", "recycled"))], actions
    assert len(recycler.calls) == 1
    with world.engine.connect() as conn:
        n = conn.execute(sa.text("SELECT count(*) FROM mf_wsp_worker_boots")).scalar_one()
    assert n == 1


def test_the_production_task_retires_and_replaces_miforge_wsp_worker_index(
    world: World, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Through ``wsp.tasks.drive``: the pass the beat runs retires a boot near its idle recycle,
    sends ``/recycle`` once and asks for ``miforge-wsp-worker-0`` once; the queued run waits."""
    from wsp import tasks
    from wsp.pod_recycler import KubernetesPodRecycler

    asked: list[tuple[str, str]] = []

    def recycle(self: KubernetesPodRecycler, index: int, served_since: dt.datetime) -> str:
        asked.append((self.namespace, self.pod_name(index)))
        return "deleted"

    monkeypatch.setattr(KubernetesPodRecycler, "recycle", recycle)
    monkeypatch.setenv("WSP_WORKER_POD_NAME", "miforge-wsp-worker-{index}")
    monkeypatch.setenv("WSP_WORKER_POD_NAMESPACE", "miforge")
    monkeypatch.setattr("db.session.get_session_factory", lambda: world.factory)
    monkeypatch.setattr(tasks, "run_files", lambda run, bundle: [])
    monkeypatch.setattr(tasks, "read_sensitivity", lambda run, bundle: "not_sensitive")
    enqueued: list[str] = []
    monkeypatch.setattr(tasks, "enqueue_push", enqueued.append)
    run_id = world.queue()
    worker = ClockedWorker(left=NEAR)
    _, answer = _drive_task_with(world, monkeypatch, worker)
    assert f"retired:{worker.boot}" in answer["actions"], answer
    assert "pod_replaced:0:deleted" in answer["actions"], answer
    assert [c for c in worker.calls if c[0] == "recycle"] == [("recycle", worker.boot)]
    assert asked == [("miforge", "miforge-wsp-worker-0")]
    assert enqueued == [] and world.run(run_id)["status"] == "queued"


def test_a_pass_lands_inside_the_margin_and_the_margin_is_inside_the_hour() -> None:
    """The beat must run a pass well inside the margin, or a worker passes from outside the margin
    to its own recycle between two passes; and the margin must leave the worker most of its hour."""
    from workers.celery_app import WSP_DRIVE_INTERVAL_SECONDS
    from wsp_worker import supervisor

    assert WSP_DRIVE_INTERVAL_SECONDS * 10 <= dispatch.RETIRE_BEFORE_S
    assert dispatch.RETIRE_BEFORE_S <= supervisor.IDLE_RECYCLE_S / 6
