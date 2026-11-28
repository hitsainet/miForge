"""The pool (feature 019 P3; D-4 to D-10, FTASKS 16.4-16.9): the real driver over several workers.

Every case runs ``dispatch.drive`` (the pass, which only claims) and ``dispatch.push_claimed`` (the
push task's body) on fake workers, through ``tests.support.wsp_drive``. One test clock is shared
by the driver and the workers, so a push can be held "for minutes" while other passes run.
The two-supervisor case (D-3) is in ``test_dispatch_two_supervisors.py``.
"""

from __future__ import annotations

import datetime as dt
import logging
import secrets
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pytest
import sqlalchemy as sa

from tests.integration.wsp.test_dispatch import FakeWorker, World, archive
from tests.integration.wsp.test_dispatch import world as world
from tests.integration.wsp.wsp_db import NOW, insert
from tests.support.wsp_drive import drive_inline, slots
from wsp import dispatch, runs

TRANSCRIPT = b'{"role": "user", "text": "repair it"}\n'


@dataclass
class Clock:
    at: dt.datetime = field(default_factory=lambda: dt.datetime(2026, 11, 25, 6, 0, tzinfo=dt.UTC))

    def __call__(self) -> dt.datetime:
        return self.at

    def advance(self, seconds: float) -> None:
        self.at += dt.timedelta(seconds=seconds)


def _generic(world: World, endpoint_id: str, base_url: str | None = None) -> str:
    with world.engine.begin() as conn:
        insert(
            conn,
            "mf_wsp_endpoints",
            id=endpoint_id,
            kind="openai_compatible",
            base_url=base_url or f"http://{endpoint_id}.lan:8000/v1",
            model="m",
            locality="local",
            # Feature 020 (FR-020.2): a ``.lan`` host derives external, so it states a window.
            context_window_tokens=32_768,
            output_cap_field="max_completion_tokens",
            serving="generic",
            created_at=NOW,
            created_by="operator",
        )
    return endpoint_id


def _miserver(world: World, endpoint_id: str, base_url: str) -> str:
    with world.engine.begin() as conn:
        insert(
            conn,
            "mf_wsp_endpoints",
            id=endpoint_id,
            kind="openai_compatible",
            base_url=base_url,
            model="m",
            locality="local",
            context_window_tokens=32_768,  # as _generic: an ``llm.lan`` host derives external
            output_cap_field="max_tokens",
            serving="miserver",
            created_at=NOW,
            created_by="operator",
        )
    return endpoint_id


def drive(
    world: World,
    workers: list[Any],
    *,
    pool_size: int = 1,
    clock: Callable[[], dt.datetime] = dispatch.utcnow,
    **kwargs: Any,
) -> list[str]:
    return drive_inline(
        world.factory,
        workers,
        world.settings,
        read_sensitivity=world.read,
        pool_size=pool_size,
        now=clock,
        **kwargs,
    )


def _finish(worker: FakeWorker, run_id: str) -> None:
    worker.state = "finished"
    worker.data = archive(worker.boot, run_id, **{"transcript.jsonl": TRANSCRIPT})


def _restart(worker: FakeWorker) -> None:
    """What the supervisor does after its ack: it exits and comes back idle on a fresh boot."""
    worker.boot, worker.state, worker.pushed, worker.data = (
        secrets.token_hex(16),
        "idle",
        None,
        None,
    )


def _status(world: World, run_id: str) -> str:
    return str(world.run(run_id)["status"])


def _index(world: World, run_id: str) -> int | None:
    value = world.run(run_id)["worker_index"]
    return None if value is None else int(value)


# ------------------------------------------------------------- D-4: the key (FR-019.8)


def test_a_busy_key_never_holds_back_another_and_its_own_waits(world: World) -> None:
    """Queued A1, A2 and B1, with A1 active: B1 is claimed, and A2 only after A1 ends."""
    a, b = _generic(world, "gen-a"), _generic(world, "gen-b")
    a1, a2, b1 = world.queue(a), world.queue(a), world.queue(b)
    w0, w1 = FakeWorker(), FakeWorker()
    assert drive(world, [w0, w1], pool_size=2) == [f"pushed:{a1}"]
    assert drive(world, [w0, w1], pool_size=2) == [f"pushed:{b1}"]
    assert (_index(world, a1), _index(world, b1)) == (0, 1)
    assert _status(world, a2) == "queued"
    assert drive(world, [w0, w1], pool_size=2) == ["worker_busy"]
    _finish(w0, a1)
    actions = drive(world, [w0, w1], pool_size=2)
    assert actions[:2] == [f"collected:{a1}:completed", f"acked:{a1}"], actions
    _restart(w0)
    assert drive(world, [w0, w1], pool_size=2) == [f"pushed:{a2}"]
    assert _index(world, a2) == 0
    assert [c[1] for c in w0.calls if c[0] == "push"] == [a1, a2]
    assert [c[1] for c in w1.calls if c[0] == "push"] == [b1]


@pytest.mark.parametrize(
    "second_url",
    [
        "http://llm.lan:8000/v1",
        "http://llm.lan:8000",
        "http://LLM.lan:8000/",
        "http://llm.lan:8000/v1/",
    ],
)
def test_two_miserver_rows_on_one_server_run_one_at_a_time(world: World, second_url: str) -> None:
    first = world.queue(_miserver(world, "ms-a", "http://llm.lan:8000/v1"))
    second = world.queue(_miserver(world, "ms-b", second_url))
    w0, w1 = FakeWorker(), FakeWorker()
    assert drive(world, [w0, w1], pool_size=2) == [f"pushed:{first}"]
    assert drive(world, [w0, w1], pool_size=2) == []  # the key is busy; nothing is said for it
    assert _status(world, second) == "queued" and "push" not in w1.names()
    assert world.run(first)["schedule_key"] == "miserver:http://llm.lan:8000"


def test_two_generic_rows_on_one_base_url_run_both(world: World) -> None:
    url = "http://vllm.lan:8000/v1"
    first = world.queue(_generic(world, "gen-a", url))
    second = world.queue(_generic(world, "gen-b", url))
    w0, w1 = FakeWorker(), FakeWorker()
    assert drive(world, [w0, w1], pool_size=2) == [f"pushed:{first}"]
    assert drive(world, [w0, w1], pool_size=2) == [f"pushed:{second}"]
    assert (world.run(first)["schedule_key"], world.run(second)["schedule_key"]) == (
        "endpoint:gen-a",
        "endpoint:gen-b",
    )


# --------------------------------------------- D-5: keyless runs count toward the pool (F19-D7)


def _replay(world: World, tmp_path: Path) -> str:
    """A real replay (F19-D7): of a completed session, with no endpoint and so no key."""
    from tests.integration.wsp.test_replay_and_rerun import _queue_replay, _session

    original, _ = _session(world, tmp_path)
    replay = _queue_replay(world, original)
    row = world.run(replay)
    assert (row["style"], row["endpoint_id"]) == ("replay", None)
    return replay


def _drive_files(world: World, workers: list[Any], **kwargs: Any) -> list[str]:
    from wsp import tasks

    return drive(world, workers, files_for=tasks.run_files, **kwargs)


def test_a_keyless_run_waits_for_a_free_worker_at_pool_size_one(
    world: World, tmp_path: Path
) -> None:
    replay = _replay(world, tmp_path)
    active = world.queue(_generic(world, "gen-a"))
    with world.engine.begin() as conn:  # the endpoint run is older, so it is claimed first
        conn.execute(
            sa.text("UPDATE mf_wsp_runs SET created_at = :t WHERE run_id = :r"),
            {"t": NOW - dt.timedelta(days=1), "r": active},
        )
    w0, w1 = FakeWorker(), FakeWorker()
    assert _drive_files(world, [w0, w1], pool_size=1) == [f"pushed:{active}"]
    assert _drive_files(world, [w0, w1], pool_size=1) == ["worker_busy"]
    assert _status(world, replay) == "queued"


def test_a_keyless_run_takes_the_free_worker_at_pool_size_two(world: World, tmp_path: Path) -> None:
    replay = _replay(world, tmp_path)
    active = world.queue(_generic(world, "gen-a"))
    with world.engine.begin() as conn:
        conn.execute(
            sa.text("UPDATE mf_wsp_runs SET created_at = :t WHERE run_id = :r"),
            {"t": NOW - dt.timedelta(days=1), "r": active},
        )
    w0, w1 = FakeWorker(), FakeWorker()
    assert _drive_files(world, [w0, w1], pool_size=2) == [f"pushed:{active}"]
    assert _drive_files(world, [w0, w1], pool_size=2) == [f"pushed:{replay}"]
    row = world.run(replay)
    assert (row["schedule_key"], row["worker_index"]) == (None, 1)


def test_a_running_keyless_run_holds_its_worker_against_the_pool(
    world: World, tmp_path: Path
) -> None:
    """The control's case: a keyless run that is ACTIVE counts, so at pool size 1 a queued
    endpoint run waits behind it."""
    replay = _replay(world, tmp_path)
    w0, w1 = FakeWorker(), FakeWorker()
    assert _drive_files(world, [w0, w1], pool_size=1) == [f"pushed:{replay}"]
    waiting = world.queue(_generic(world, "gen-b"))
    assert _drive_files(world, [w0, w1], pool_size=1) == ["worker_busy"]
    assert _status(world, waiting) == "queued"


# ------------------------------------------------ D-6: the effective pool (F19-D24, D31)


def test_a_maximum_of_three_at_setting_two_claims_two_and_never_a_third(world: World) -> None:
    queued = [world.queue(_generic(world, f"gen-{i}")) for i in range(3)]
    workers = [FakeWorker(), FakeWorker(), FakeWorker()]
    assert drive(world, workers, pool_size=2) == [f"pushed:{queued[0]}"]
    assert drive(world, workers, pool_size=2) == [f"pushed:{queued[1]}"]
    assert drive(world, workers, pool_size=2) == ["worker_busy"]
    assert drive(world, workers, pool_size=2) == ["worker_busy"]
    assert _status(world, queued[2]) == "queued" and "push" not in workers[2].names()


def test_at_setting_one_one_claim(world: World) -> None:
    queued = [world.queue(_generic(world, f"gen-{i}")) for i in range(2)]
    workers = [FakeWorker(), FakeWorker(), FakeWorker()]
    assert drive(world, workers, pool_size=1) == [f"pushed:{queued[0]}"]
    assert drive(world, workers, pool_size=1) == ["worker_busy"]


def test_lowering_the_setting_with_two_active_fails_neither(world: World) -> None:
    queued = [world.queue(_generic(world, f"gen-{i}")) for i in range(3)]
    workers = [FakeWorker(), FakeWorker()]
    drive(world, workers, pool_size=2)
    drive(world, workers, pool_size=2)
    for worker in workers:
        worker.calls = []
    assert drive(world, workers, pool_size=1) == ["worker_busy"]
    assert [_status(world, r) for r in queued] == ["running", "running", "queued"]
    assert [w.names().count("status") for w in workers] == [1, 1]  # both still polled


def test_a_maximum_below_the_setting_uses_the_maximum(world: World) -> None:
    queued = [world.queue(_generic(world, f"gen-{i}")) for i in range(2)]
    only = FakeWorker()
    assert drive(world, [only], pool_size=3) == [f"pushed:{queued[0]}"]
    assert drive(world, [only], pool_size=3) == ["worker_busy"]


# ------------------------------------------------ D-7: one worker's failure is its own


def _two_running(world: World, clock: Clock) -> tuple[str, str, FakeWorker, FakeWorker]:
    first = world.queue(_generic(world, "gen-a"))
    second = world.queue(_generic(world, "gen-b"))
    w0, w1 = FakeWorker(), FakeWorker()
    drive(world, [w0, w1], pool_size=2, clock=clock)
    drive(world, [w0, w1], pool_size=2, clock=clock)
    assert (_index(world, first), _index(world, second)) == (0, 1)
    return first, second, w0, w1


def test_a_silent_worker_fails_only_its_own_run(world: World) -> None:
    clock = Clock()
    first, second, w0, w1 = _two_running(world, clock)
    w1.unreachable = True
    clock.advance(world.settings.wsp_heartbeat_lapse_s + 1)
    actions = drive(world, [w0, w1], pool_size=2, clock=clock)
    assert actions == [f"lost:{second}"], actions
    assert world.run(second)["stop_reason"] == "worker_lost"
    assert _status(world, first) == "running"


def test_a_restart_under_a_run_fails_only_that_run(world: World) -> None:
    clock = Clock()
    first, second, w0, w1 = _two_running(world, clock)
    w0.status_boot = secrets.token_hex(16)
    actions = drive(world, [w0, w1], pool_size=2, clock=clock)
    assert actions == [f"lost:{first}"], actions
    assert _status(world, second) == "running"


# ------------------------------------- D-8: a claimed worker is live before its push


def test_a_claimed_but_unpushed_worker_is_not_recycled(world: World) -> None:
    run_id = world.queue(_generic(world, "gen-a"))
    worker = FakeWorker()
    enqueued: list[str] = []
    passed = dispatch.drive(
        world.factory,
        slots([worker]),
        world.settings,
        read_sensitivity=world.read,
        files_for=lambda run, bundle: [],
        enqueue_push=enqueued.append,
    )
    assert passed.actions == [f"claimed:{run_id}:0"] and enqueued == [run_id]
    # The push task has not run yet: the worker is idle on a boot the claim consumed.
    again = dispatch.drive(
        world.factory,
        slots([worker]),
        world.settings,
        read_sensitivity=world.read,
        files_for=lambda run, bundle: [],
        enqueue_push=enqueued.append,
    )
    assert ("recycle", worker.boot) not in worker.calls, again.actions
    assert _status(world, run_id) == "pushing" and enqueued == [run_id]


def test_one_push_at_a_time_a_second_claim_waits_for_the_first_push(world: World) -> None:
    """F19-D35: while a run is ``pushing`` no other is claimed, even with a free worker and a
    free key; the next claim follows once that push has ended."""
    first = world.queue(_generic(world, "gen-a"))
    second = world.queue(_generic(world, "gen-b"))
    w0, w1 = FakeWorker(), FakeWorker()
    enqueued: list[str] = []

    def one_pass() -> list[str]:
        return dispatch.drive(
            world.factory,
            slots([w0, w1]),
            world.settings,
            pool_size=2,
            read_sensitivity=world.read,
            files_for=lambda run, bundle: [],
            enqueue_push=enqueued.append,
        ).actions

    assert one_pass() == [f"claimed:{first}:0"]
    assert one_pass() == ["push_in_flight"]
    assert _status(world, second) == "queued" and enqueued == [first]
    assert (
        dispatch.push_claimed(world.factory, w0, world.settings, first, files_for=lambda r, b: [])
        == "pushed"
    )
    assert one_pass() == [f"claimed:{second}:1"]


# ---------------------- D-9: a long push holds back nobody else's poll or result (F19-D19)


@dataclass
class HeldPushWorker(FakeWorker):
    """A push that lasts ``held_s`` on the shared clock and, while it lasts, lets ``during`` run
    (the other passes the beat fires while this push's task is busy), reporting progress."""

    clock: Clock = field(default_factory=Clock)
    held_s: float = 0.0
    during: Callable[[], None] = lambda: None
    tokens: list[str] = field(default_factory=list)

    def push(
        self,
        document: dict[str, Any],
        files: list[tuple[str, Path]],
        *,
        on_progress: Callable[[], None] | None = None,
    ) -> dict[str, Any]:
        gateway = document.get("gateway") or {}
        if gateway.get("token"):
            self.tokens.append(gateway["token"])
        step = 30.0
        spent = 0.0
        while spent < self.held_s:
            self.clock.advance(step)
            spent += step
            if on_progress is not None:
                on_progress()
            self.during()
        return super().push(document, files)


def test_a_push_held_past_the_lapse_and_the_pull_timeout_loses_no_other_result(
    world: World, caplog: pytest.LogCaptureFixture
) -> None:
    """Worker 0 runs a session; worker 1 receives a push held longer than the heartbeat lapse and
    the result pull timeout (each 300 s). The beat's passes go on meanwhile: worker 0's session
    is polled, its result pulled, taken in and acked, and the held push is never failed. At
    32b6fa8 the push ran inside the pass, under its lock, so every one of those passes was
    ``lock_busy`` and worker 0's result waited behind the push (the control)."""
    clock = Clock()
    settings = world.settings
    held_s = max(settings.wsp_heartbeat_lapse_s, settings.wsp_result_pull_timeout_s) + 120
    other = world.queue(_generic(world, "gen-b"))
    w0 = FakeWorker()
    inner: list[list[str]] = []

    def other_passes() -> None:
        if w0.state == "running":
            _finish(w0, other)
        inner.append(drive(world, [w0, w1], pool_size=2, clock=clock))

    w1 = HeldPushWorker(clock=clock, held_s=held_s, during=other_passes)
    assert drive(world, [w0, w1], pool_size=2, clock=clock) == [f"pushed:{other}"]
    assert _index(world, other) == 0
    long = world.queue(_generic(world, "gen-a"))
    with caplog.at_level(logging.DEBUG):
        actions = drive(world, [w0, w1], pool_size=2, clock=clock)
    assert actions == [f"pushed:{long}"], actions
    assert len(inner) >= held_s // 30
    flat = [a for pass_ in inner for a in pass_]
    assert [a for a in flat if a.startswith(("collected:", "acked:"))] == [
        f"collected:{other}:completed",
        f"acked:{other}",
    ], inner
    assert "lock_busy" not in flat
    row = world.run(other)
    assert (row["status"], row["result_sha256"] is not None) == ("completed", True)
    assert (runs.run_dir(world.settings, other) / "transcript.jsonl").read_bytes() == TRANSCRIPT
    # The held push was failed by neither the lease nor a lapse, and landed on its own worker.
    assert _status(world, long) == "running" and _index(world, long) == 1
    assert not [a for a in flat if long in a]
    # D-11's second half: the session token is in no log line of this run.
    assert w1.tokens and all(t not in caplog.text for t in w1.tokens)


# --------------------------------------------------- D-10: the push lease (F19-D19)


def test_a_push_killed_mid_push_fails_after_the_lease_and_is_never_pushed_again(
    world: World,
) -> None:
    clock = Clock()
    run_id = world.queue(_generic(world, "gen-a"))
    worker = FakeWorker()
    enqueued: list[str] = []
    dispatch.drive(
        world.factory,
        slots([worker]),
        world.settings,
        read_sensitivity=world.read,
        files_for=lambda run, bundle: [],
        enqueue_push=enqueued.append,
        now=clock,
    )
    assert _status(world, run_id) == "pushing"  # the push task died before it ran
    lease = world.settings.wsp_push_lease_s
    clock.advance(lease - 1)
    assert drive(world, [worker], clock=clock) == [], "a push inside its lease is left alone"
    clock.advance(2)
    actions = drive(world, [worker], clock=clock)
    assert actions[0] == f"orphan_failed:{run_id}", actions
    row = world.run(run_id)
    assert (row["status"], row["stop_reason"]) == ("failed", "error") or row["status"] == "failed"
    assert "worker_push_failed" in str(row["failure"])
    # A late push task for it finds nothing to push.
    assert (
        dispatch.push_claimed(
            world.factory, worker, world.settings, run_id, files_for=lambda r, b: [], now=clock
        )
        == "not_pushing"
    )
    assert "push" not in worker.names()


def test_a_slow_push_that_reports_progress_is_not_failed(world: World) -> None:
    clock = Clock()
    lease = world.settings.wsp_push_lease_s
    run_id = world.queue(_generic(world, "gen-a"))
    seen: list[list[str]] = []
    worker = HeldPushWorker(clock=clock, held_s=lease * 2)
    worker.during = lambda: seen.append(drive(world, [worker], clock=clock))
    assert drive(world, [worker], clock=clock) == [f"pushed:{run_id}"]
    assert seen and not any(a.startswith("orphan_failed") for p in seen for a in p), seen
    assert _status(world, run_id) == "running"
