"""The WSP driver (feature 016; FR-016.11; FTASKS 5.2, 5.2a; P41, P60, P61, R3A-10).

Against a fake supervisor that records every call, plus one pass end to end against the REAL
supervisor started by ``LocalProcessLauncher`` through the real HTTP client (the wiring).
"""

from __future__ import annotations

import datetime as dt
import hashlib
import io
import json
import secrets
import tarfile
import time
from collections.abc import Callable, Iterator
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pytest
import sqlalchemy as sa
from sqlalchemy.orm import sessionmaker

from tests.integration.wsp.wsp_db import NOW, engine_at_head, insert
from tests.support.wsp_drive import RESIDENT_WINDOW, drive_inline
from tests.support.wsp_fixtures import enable_wsp, make_bundle
from wsp import dispatch, runs
from wsp.dispatch import WorkerUnreachableError
from wsp.settings import WSPSettings, get_wsp_settings

LOCAL = "http://miserver-backend.miserver.svc:8000/v1"
EXTERNAL = "https://api.anthropic.com"


def _at(value: Any) -> dt.datetime:
    """A stored time as an aware datetime: SQLite's raw row holds text, PostgreSQL a datetime."""
    parsed = dt.datetime.fromisoformat(value) if isinstance(value, str) else value
    return parsed if parsed.tzinfo is not None else parsed.replace(tzinfo=dt.UTC)


def _j(value: Any) -> Any:
    """A JSON column as a value: SQLite returns its text, PostgreSQL the decoded object."""
    return json.loads(value) if isinstance(value, str | bytes) else value


def archive(boot_id: str, run_id: str, **members: bytes) -> bytes:
    record = {
        "boot_id": boot_id,
        "run_id": run_id,
        "exit_code": 0,
        "deadline_hit": False,
        "unsafe_reason": None,
    }
    files = {"supervisor.json": json.dumps(record).encode(), **members}
    buffer = io.BytesIO()
    with tarfile.open(fileobj=buffer, mode="w", format=tarfile.USTAR_FORMAT) as tar:
        for name, data in files.items():
            info = tarfile.TarInfo(name)
            info.size = len(data)
            tar.addfile(info, io.BytesIO(data))
    return buffer.getvalue()


@dataclass
class FakeWorker:
    calls: list[tuple[Any, ...]] = field(default_factory=list)
    boot: str = field(default_factory=lambda: secrets.token_hex(16))
    state: str = "idle"
    pushed: dict[str, Any] | None = None
    data: bytes | None = None
    unreachable: bool = False
    push_fails: bool = False
    status_boot: str | None = None
    #: What ``GET /status`` reports as ``elapsed_s`` (live progress); absent when ``None``.
    elapsed: Any = None

    def _check(self) -> None:
        if self.unreachable:
            raise WorkerUnreachableError("down")

    def ready(self) -> dict[str, Any]:
        self.calls.append(("ready",))
        self._check()
        return {"boot_id": self.boot, "state": self.state}

    def push(
        self,
        document: dict[str, Any],
        files: list[tuple[str, Path]],
        *,
        on_progress: Callable[[], None] | None = None,
    ) -> dict[str, Any]:
        self.calls.append(("push", document["run_id"], document["boot_id"]))
        self._check()
        if self.push_fails:
            raise WorkerUnreachableError("lost response")
        self.pushed, self.state = document, "running"
        return {"state": "running"}

    def status(self) -> dict[str, Any]:
        self.calls.append(("status",))
        self._check()
        assert self.pushed is not None
        answer = {
            "boot_id": self.status_boot or self.boot,
            "run_id": self.pushed["run_id"],
            "state": self.state,
        }
        if self.elapsed is not None:
            answer["elapsed_s"] = self.elapsed
        return answer

    def result(self) -> bytes:
        self.calls.append(("result",))
        self._check()
        assert self.data is not None
        return self.data

    def ack(self, boot_id: str, run_id: str, result_sha256: str) -> None:
        self.calls.append(("ack", boot_id, run_id))

    def recycle(self, boot_id: str) -> None:
        self.calls.append(("recycle", boot_id))

    def names(self) -> list[str]:
        return [c[0] for c in self.calls]


@dataclass
class World:
    engine: sa.Engine
    factory: sessionmaker[Any]
    settings: WSPSettings
    body: dict[str, Any]
    reads: list[str] = field(default_factory=list)
    sensitivity: str | Exception = "not_sensitive"

    def queue(self, endpoint_id: str = "ep-local", kind: str = "repair") -> str:
        with self.factory() as session:
            prepared = runs.prepare_start(
                session, {**self.body, "endpoint_id": endpoint_id, "kind": kind}, "operator"
            )
            run = runs.create_run(session, prepared, approval_id=None)
            session.commit()
            return run.run_id

    def read(self, run: Any, bundle: Any) -> str:
        self.reads.append(run.run_id)
        if isinstance(self.sensitivity, Exception):
            raise self.sensitivity
        return self.sensitivity

    def drive(
        self,
        worker: Any,
        now: dt.datetime | Callable[[], dt.datetime] | None = None,
        **kwargs: Any,
    ) -> list[str]:
        """One pass, then the pushes it enqueued (``tests.support.wsp_drive``); ``now`` is a fixed
        time or a clock (called each time the driver reads it)."""
        kwargs.setdefault("files_for", lambda run, bundle: [])
        clock: Callable[[], dt.datetime]
        if now is None:
            clock = dispatch.utcnow
        elif isinstance(now, dt.datetime):
            fixed = now
            clock = lambda: fixed  # noqa: E731
        else:
            clock = now
        return drive_inline(
            self.factory, worker, self.settings, read_sensitivity=self.read, now=clock, **kwargs
        )

    def run(self, run_id: str) -> dict[str, Any]:
        with self.engine.connect() as conn:
            row = conn.execute(
                sa.text("SELECT * FROM mf_wsp_runs WHERE run_id = :r"), {"r": run_id}
            ).mappings()
            return dict(row.one())

    def boots(self) -> int:
        with self.engine.connect() as conn:
            return int(
                conn.execute(sa.text("SELECT COUNT(*) FROM mf_wsp_worker_boots")).scalar_one()
            )


@pytest.fixture()
def world(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Iterator[World]:
    built = build_world(tmp_path, monkeypatch)
    yield built
    built.engine.dispose()
    get_wsp_settings.cache_clear()


def build_world(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, **bundle: Any) -> World:
    """The world, on a fixture bundle made with ``bundle`` (``make_bundle``'s keywords); the
    caller disposes of its engine."""
    enable_wsp(monkeypatch)
    monkeypatch.setenv("WSP_BUNDLE_ROOT", str(tmp_path / "wsp"))
    get_wsp_settings.cache_clear()
    fx = make_bundle(tmp_path / "bundle", **{"rule": "mean", **bundle})
    engine = engine_at_head(tmp_path)
    with engine.begin() as conn:
        insert(
            conn,
            "mf_wsp_bundles",
            bundle_sha256=fx.manifest["bundle_sha256"],
            owner="miworkbench",
            bundle_id=fx.manifest["bundle_id"],
            bundle_kind="linear_probe",
            artifact_kind="probe",
            artifact_id="p",
            detector_digest=None,
            model="m",
            layer=12,
            sensitivity_value="not_sensitive",
            manifest=json.dumps(fx.manifest),
            bytes=1,
            import_status="imported",
            received_at=NOW,
            received_by="operator",
        )
        for endpoint_id, url, locality in (
            ("ep-local", LOCAL, "local"),
            ("ep-ext", EXTERNAL, "external"),
        ):
            insert(
                conn,
                "mf_wsp_endpoints",
                id=endpoint_id,
                kind="openai_compatible",
                base_url=url,
                model="m",
                locality=locality,
                price_in_per_mtok=3 if locality == "external" else None,
                price_out_per_mtok=15 if locality == "external" else None,
                # Feature 020 (FR-020.2): a hosted endpoint states its window.
                context_window_tokens=200_000 if locality == "external" else None,
                output_cap_field="max_tokens",
                serving="generic" if locality == "external" else "miserver",
                created_at=NOW,
                created_by="operator",
            )
    body = {
        "bundle": {
            "owner": "miworkbench",
            "bundle_id": fx.manifest["bundle_id"],
            "bundle_sha256": fx.manifest["bundle_sha256"],
        },
        "mode": "session",
        "task": "t",
        "budgets": {"tokens": 1000, "time_s": 60},
        "primary_sets": ["eval"],
        "sensitivity": "not_sensitive",
    }
    return World(
        engine, sessionmaker(bind=engine, expire_on_commit=False), get_wsp_settings(), body
    )


# ----------------------------------------------------------------------- the push


def test_a_queued_run_is_pushed_once_to_a_fresh_boot(world: World) -> None:
    run_id = world.queue()
    worker = FakeWorker()
    assert world.drive(worker) == [f"pushed:{run_id}"]
    run = world.run(run_id)
    assert run["status"] == "running" and run["boot_id"] == worker.boot
    assert run["gateway_token_sha256"] is not None and world.boots() == 1
    # 019 P3: one /ready per worker per pass (the pool read), then the push task's push. The push
    # needs no second /ready: the supervisor refuses a push whose boot is not its own.
    assert worker.names() == ["ready", "push"]


def test_a_boot_seen_before_is_never_pushed_to_even_when_idle(world: World) -> None:
    first = world.queue()
    worker = FakeWorker()
    world.drive(worker)
    with world.factory() as session:
        runs.transition(session, first, "running", "cancelled", stop_reason="cancelled")
        session.commit()
    second = world.queue()
    worker.state, worker.calls = "idle", []
    actions = world.drive(worker)
    assert ("push", second, worker.boot) not in worker.calls
    assert worker.calls.count(("recycle", worker.boot)) == 1, actions
    assert f"boot_consumed:{worker.boot}" in actions
    assert world.run(second)["status"] == "queued"


def test_every_refusal_check_comes_before_anything_is_consumed(world: World) -> None:
    """P41 (R1B-X1): an external run whose bundle is now sensitive fails; no boot is consumed."""
    run_id = world.queue("ep-ext")
    world.sensitivity = "sensitive"
    worker = FakeWorker()
    actions = world.drive(worker)
    assert actions == [f"refused:{run_id}:wsp_sensitivity_raised"]
    assert world.reads == [run_id] and world.boots() == 0
    assert "push" not in worker.names()
    run = world.run(run_id)
    assert run["status"] == "failed" and run["gateway_token_sha256"] is None


def test_an_unreadable_sensitivity_refuses_an_external_run(world: World) -> None:
    run_id = world.queue("ep-ext")
    world.sensitivity = RuntimeError("miWorkbench is down")
    assert world.drive(FakeWorker()) == [f"refused:{run_id}:wsp_sensitivity_raised"]


def test_the_production_reader_with_no_sibling_refuses_an_external_run(
    world: World, monkeypatch: pytest.MonkeyPatch
) -> None:
    """OR1-016-1: the real ``wsp.tasks.read_sensitivity``, not an injected one. With no
    ``MIWORKBENCH_API_URL`` it reads ``unknown``, and an external run is refused before a boot."""
    from wsp import tasks

    monkeypatch.delenv("MIWORKBENCH_API_URL", raising=False)
    get_wsp_settings.cache_clear()
    run_id = world.queue("ep-ext")
    worker = FakeWorker()
    actions = drive_inline(
        world.factory, worker, world.settings, read_sensitivity=tasks.read_sensitivity
    )
    assert actions == [f"refused:{run_id}:wsp_sensitivity_raised"]
    assert "push" not in worker.names() and world.boots() == 0


def test_a_local_run_is_never_held_back_by_the_sensitivity_read(world: World) -> None:
    run_id = world.queue("ep-local")
    world.sensitivity = "sensitive"
    assert world.drive(FakeWorker()) == [f"pushed:{run_id}"]
    assert world.reads == []


def test_a_changed_endpoint_fails_the_run_before_the_push(world: World) -> None:
    run_id = world.queue()
    with world.engine.begin() as conn:
        conn.execute(sa.text("UPDATE mf_wsp_endpoints SET model = 'other' WHERE id = 'ep-local'"))
    assert world.drive(FakeWorker()) == [f"refused:{run_id}:wsp_endpoint_changed"]
    assert world.boots() == 0


@pytest.mark.parametrize(
    "change",
    [
        {"action_protocol": "structured"},
        {"context_window_tokens": 32768},
        # 2026-11-21: either thinking field, away from its default.
        {"thinking": "on"},
        {"thinking": "default"},
        {"thinking_tokens": 2048},
        # 2026-11-21 (migration 0034): paging off.
        {"paging": "off"},
    ],
    ids=["protocol", "window", "thinking-on", "thinking-default", "thinking-tokens", "paging-off"],
)
def test_the_operators_patch_fails_a_queued_run_at_push(
    world: World, change: dict[str, Any]
) -> None:
    """018 FR-018.8 / F18-D8: a PATCH away from a default moves the endpoint fingerprint, so a run
    frozen before it fails ``wsp_endpoint_changed`` before a boot is consumed."""
    from wsp import endpoints

    run_id = world.queue()
    with world.factory() as session:
        endpoints.update_endpoint(session, "ep-local", change)
        session.commit()
    assert world.drive(FakeWorker()) == [f"refused:{run_id}:wsp_endpoint_changed"]
    assert world.boots() == 0


def test_a_patch_that_keeps_the_defaults_does_not_fail_a_queued_run(world: World) -> None:
    """The control: the same PATCH at the default values leaves the fingerprint alone."""
    from wsp import endpoints

    run_id = world.queue()
    with world.factory() as session:
        endpoints.update_endpoint(
            session,
            "ep-local",
            {
                "action_protocol": "tool_calls",
                "context_window_tokens": None,
                "thinking": "off",
                "thinking_tokens": None,
            },
        )
        session.commit()
    assert world.drive(FakeWorker()) == [f"pushed:{run_id}"]


# ----------------------------------------- the worker's /work (production defect 2026-11-20)

GIB = 1024**3


def _sparse(path: Path, size: int) -> Path:
    with path.open("wb") as handle:
        handle.truncate(size)  # sparse: the size is real to stat(), no disk is used
    return path


def test_a_push_larger_than_the_workers_work_is_refused_before_a_boot(
    world: World, tmp_path: Path
) -> None:
    """A 35 GB push into an 8 GiB /work was evicted mid-stream. The driver now compares the files
    it would push with the worker's /work and fails the run by name, before consuming a boot."""
    run_id = world.queue()
    big = _sparse(tmp_path / "practice.bin", 57 * GIB)
    worker = FakeWorker()
    actions = world.drive(worker, files_for=lambda run, bundle: [("practice.bin", big)])
    assert actions == [f"refused:{run_id}:wsp_bundle_too_large_for_worker"]
    assert "push" not in worker.names() and world.boots() == 0
    run = world.run(run_id)
    failure = _j(run["failure"])
    assert run["status"] == "failed" and failure["code"] == "wsp_bundle_too_large_for_worker"
    assert str(57 * GIB) in failure["message"] and "WSP_WORKER_WORK_BYTES" in failure["message"]


def test_a_replay_larger_than_the_workers_work_is_refused_too(world: World, tmp_path: Path) -> None:
    """The size check comes before the replay/rerun source checks, so it covers every style."""
    run_id = world.queue()
    with world.engine.begin() as conn:
        conn.execute(
            sa.text(
                "UPDATE mf_wsp_runs SET style = 'replay', endpoint_id = NULL WHERE run_id = :r"
            ),
            {"r": run_id},
        )
    big = _sparse(tmp_path / "practice.bin", 57 * GIB)
    actions = world.drive(FakeWorker(), files_for=lambda run, bundle: [("practice.bin", big)])
    assert actions == [f"refused:{run_id}:wsp_bundle_too_large_for_worker"]


def test_a_push_that_fits_is_pushed_with_the_watchdog_below_the_volume(
    world: World, tmp_path: Path
) -> None:
    """The pushed spec carries ``work_max_bytes`` = WSP_WORKER_WORK_BYTES less the 4 GiB margin,
    so the supervisor's watchdog fires ``worker_disk_limit`` before the kubelet evicts the pod."""
    run_id = world.queue()
    small = tmp_path / "a.bin"
    small.write_bytes(b"x" * 1000)
    worker = FakeWorker()
    actions = world.drive(worker, files_for=lambda run, bundle: [("a.bin", small)])
    assert actions == [f"pushed:{run_id}"]
    assert worker.pushed is not None
    assert worker.pushed["spec"]["work_max_bytes"] == 60 * GIB


def test_the_pushed_watchdog_follows_the_setting(
    world: World, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("WSP_WORKER_WORK_BYTES", str(100 * GIB))
    get_wsp_settings.cache_clear()
    world.settings = get_wsp_settings()
    run_id = world.queue()
    worker = FakeWorker()
    assert world.drive(worker) == [f"pushed:{run_id}"]
    assert worker.pushed is not None and worker.pushed["spec"]["work_max_bytes"] == 96 * GIB


def test_the_worker_reads_the_pushed_watchdog(world: World) -> None:
    """The backend's document, parsed by the worker's own ``parse_push``: the supervisor's
    watchdog bound is the pushed value, not its 8 GiB default."""
    from wsp_worker import spec

    run_id = world.queue()
    worker = FakeWorker()
    world.drive(worker)
    assert worker.pushed is not None and worker.pushed["run_id"] == run_id
    push = spec.parse_push(json.dumps(worker.pushed).encode())
    assert push.work_max_bytes == 60 * GIB != spec.DEFAULT_WORK_MAX_BYTES


def test_a_lost_push_response_fails_the_run_and_it_is_never_pushed_again(world: World) -> None:
    run_id = world.queue()
    worker = FakeWorker(push_fails=True)
    assert world.drive(worker) == [f"push_failed:{run_id}"]
    run = world.run(run_id)
    assert run["status"] == "failed" and _j(run["failure"])["code"] == "worker_push_failed"
    worker.push_fails, worker.calls = False, []
    world.drive(worker)
    assert "push" not in worker.names()


def test_a_pass_that_does_not_hold_the_lock_does_nothing(world: World) -> None:
    world.queue()
    worker = FakeWorker()
    with dispatch.single_flight(world.factory) as held:
        assert held
        assert world.drive(worker) == ["lock_busy"]
    assert worker.calls == []


# ----------------------------------------------------------------------- the running run


def _running(world: World) -> tuple[str, FakeWorker]:
    run_id = world.queue()
    worker = FakeWorker()
    world.drive(worker)
    worker.calls = []
    return run_id, worker


def test_a_finished_run_is_pulled_taken_in_and_acked_once(world: World) -> None:
    run_id, worker = _running(world)
    worker.state = "finished"
    worker.data = archive(
        worker.boot,
        run_id,
        **{"transcript.jsonl": b'{"role": "user", "text": "<script>x</script>"}\n'},
    )
    actions = world.drive(worker)
    assert actions == [f"collected:{run_id}:completed", f"acked:{run_id}"]
    assert [c for c in worker.calls if c[0] in ("result", "ack")] == [
        ("result",),
        ("ack", worker.boot, run_id),
    ]
    run = world.run(run_id)
    assert run["status"] == "completed" and run["acked_at"] is not None
    assert run["result_sha256"] is not None and run["transcript_sha256"] is not None


def test_an_intake_failure_still_acks(world: World) -> None:
    run_id, worker = _running(world)
    worker.state = "finished"
    worker.data = archive(worker.boot, run_id, **{"patch/../../etc": b"x"})
    actions = world.drive(worker)
    assert actions == [f"intake_refused:{run_id}", f"acked:{run_id}"]
    run = world.run(run_id)
    assert run["status"] == "failed"
    assert _j(run["failure"])["code"] == "wsp_result_invalid"


def test_another_boot_id_on_status_is_worker_lost_at_once(world: World) -> None:
    run_id, worker = _running(world)
    worker.status_boot = secrets.token_hex(16)
    assert world.drive(worker) == [f"lost:{run_id}"]
    assert world.run(run_id)["stop_reason"] == "worker_lost"


def test_unreachable_past_the_lapse_is_worker_lost(world: World) -> None:
    run_id, worker = _running(world)
    worker.unreachable = True
    later = dispatch.utcnow() + dt.timedelta(seconds=world.settings.wsp_heartbeat_lapse_s + 1)
    assert world.drive(worker, now=later) == [f"lost:{run_id}"]


def test_unreachable_within_the_lapse_waits(world: World) -> None:
    run_id, worker = _running(world)
    worker.unreachable = True
    assert world.drive(worker, now=dispatch.utcnow()) == []
    assert world.run(run_id)["status"] == "running"


def test_past_the_workers_lifetime_with_no_result_is_time_budget(world: World) -> None:
    """A worker that answers is waited for up to its own lifetime (budget + grace + pull timeout,
    fix 2026-11-21): past budget + grace it is still waited for, past the lifetime it is not."""
    run_id, worker = _running(world)
    started = dispatch.utcnow()
    grace = world.settings.wsp_session_grace_s
    lifetime = dispatch.worker_lifetime_s(60, world.settings)
    assert lifetime == 60 + grace + world.settings.wsp_result_pull_timeout_s
    assert world.drive(worker, now=started + dt.timedelta(seconds=60 + grace + 1)) == []
    assert world.run(run_id)["status"] == "running"
    later = started + dt.timedelta(seconds=lifetime + 1)
    assert world.drive(worker, now=later) == [f"time_budget:{run_id}"]
    assert world.run(run_id)["stop_reason"] == "time_budget"


# ------------------------------------------- the deadline is the worker's (fix 2026-11-21)


@dataclass
class Clock:
    """A test clock the fake worker moves forward while it receives the push."""

    at: dt.datetime = field(
        default_factory=lambda: dt.datetime(2026, 11, 20, 23, 47, tzinfo=dt.UTC)
    )

    def __call__(self) -> dt.datetime:
        return self.at

    def advance(self, seconds: float) -> None:
        self.at += dt.timedelta(seconds=seconds)


@dataclass
class SlowPushWorker(FakeWorker):
    """A push that takes ``push_s`` of the clock (run wsp_cb092cedad97: 301 s for 34.9 GB)."""

    clock: Clock = field(default_factory=Clock)
    push_s: float = 301.0

    def push(
        self,
        document: dict[str, Any],
        files: list[tuple[str, Path]],
        *,
        on_progress: Callable[[], None] | None = None,
    ) -> dict[str, Any]:
        self.clock.advance(self.push_s)
        return super().push(document, files)


def _usage(world: World, run_id: str, calls: int) -> None:
    with world.engine.begin() as conn:
        for seq in range(calls):
            insert(
                conn,
                "mf_wsp_usage",
                run_id=run_id,
                seq=seq,
                endpoint_id="ep-local",
                input_tokens=1000,
                output_tokens=200,
                cost_usd=0,
                at=NOW,
            )


def _killed_at_deadline(boot: str, run_id: str, transcript: bytes) -> bytes:
    """The worker's archive for a session its supervisor killed at the time budget."""
    data = archive(boot, run_id, **{"transcript.jsonl": transcript})
    members: dict[str, bytes] = {}
    with tarfile.open(fileobj=io.BytesIO(data)) as tar:
        for member in tar.getmembers():
            handle = tar.extractfile(member)
            assert handle is not None
            members[member.name] = handle.read()
    record = json.loads(members["supervisor.json"])
    record.update(deadline_hit=True, exit_code=-9)
    members["supervisor.json"] = json.dumps(record).encode()
    buffer = io.BytesIO()
    with tarfile.open(fileobj=buffer, mode="w", format=tarfile.USTAR_FORMAT) as tar:
        for name, body in members.items():
            info = tarfile.TarInfo(name)
            info.size = len(body)
            tar.addfile(info, io.BytesIO(body))
    return buffer.getvalue()


def test_a_slow_push_does_not_cost_the_session_its_result(world: World) -> None:
    """The incident (run wsp_cb092cedad97), replayed: a 301 s push, then a session killed at its
    own time budget. The driver measured budget + grace from the START of the push, so it failed
    the run 116 s before the worker's own deadline, and the result, the transcript and the
    counters were discarded unread when the worker finished. Measured from the push's
    completion, the driver waits, collects the deadline-hit result with its transcript, and the
    run's usage is the gateway's 18 calls."""
    clock = Clock()
    worker = SlowPushWorker(clock=clock)
    run_id = world.queue()
    pushed_from = clock()
    assert world.drive(worker, now=clock) == [f"pushed:{run_id}"]
    row = world.run(run_id)
    running_at = pushed_from + dt.timedelta(seconds=301)
    assert _at(row["pushed_at"]) == running_at
    _usage(world, run_id, 18)
    grace = world.settings.wsp_session_grace_s
    # The old deadline (push start + budget + grace) passes with the worker still running.
    clock.at = pushed_from + dt.timedelta(seconds=60 + grace + 5)
    assert clock.at < running_at + dt.timedelta(seconds=60)
    assert world.drive(worker, now=clock) == []
    assert world.run(run_id)["status"] == "running"
    # The supervisor kills the session at running + budget and serves what it wrote.
    clock.at = running_at + dt.timedelta(seconds=60 + 3)
    worker.state = "finished"
    transcript = b'{"role": "user", "content": "go"}\n{"role": "assistant", "content": "ok"}\n'
    worker.data = _killed_at_deadline(worker.boot, run_id, transcript)
    assert world.drive(worker, now=clock) == [
        f"collected:{run_id}:failed",
        f"acked:{run_id}",
    ]
    row = world.run(run_id)
    assert (row["status"], row["stop_reason"]) == ("failed", "time_budget")
    assert row["transcript_sha256"] == hashlib.sha256(transcript).hexdigest()
    usage = _j(row["usage"])
    assert (usage["requests"], usage["input_tokens"], usage["output_tokens"]) == (18, 18000, 3600)


def test_the_gateway_token_is_dated_from_the_push_completion(world: World) -> None:
    clock = Clock()
    worker = SlowPushWorker(clock=clock)
    run_id = world.queue()
    started = clock()
    world.drive(worker, now=clock)
    row = world.run(run_id)
    budget_and_grace = dt.timedelta(seconds=60 + world.settings.wsp_session_grace_s)
    assert _at(row["token_expires_at"]) == started + dt.timedelta(seconds=301) + budget_and_grace
    assert _at(row["started_at"]) == started  # the run's own clock still starts at the push


def test_a_run_the_driver_fails_keeps_the_gateways_usage(world: World) -> None:
    """``requests: 0`` on a failed run that the gateway had metered 18 calls for: every terminal
    transition now stores the gateway's records."""
    run_id, worker = _running(world)
    _usage(world, run_id, 3)
    worker.status_boot = secrets.token_hex(16)
    assert world.drive(worker) == [f"lost:{run_id}"]
    usage = _j(world.run(run_id)["usage"])
    assert (usage["requests"], usage["input_tokens"], usage["output_tokens"]) == (3, 3000, 600)
    with world.factory() as session:
        document = runs.run_document(runs.get_run(session, run_id))
    assert document["usage"]["requests"] == 3


def test_a_cancelled_run_keeps_the_gateways_usage(world: World) -> None:
    run_id, _worker = _running(world)
    _usage(world, run_id, 2)
    with world.factory() as session:
        runs.cancel(session, run_id)
        session.commit()
    assert _j(world.run(run_id)["usage"])["requests"] == 2


def test_unreachable_past_budget_and_grace_but_within_the_lapse_is_time_budget(
    world: World,
) -> None:
    """C20c, the unreachable branch: a worker that stops answering after the budget and grace
    are spent ends the run ``time_budget`` without waiting for the (longer) heartbeat lapse."""
    run_id, worker = _running(world)
    worker.unreachable = True
    after = 60 + world.settings.wsp_session_grace_s + 1
    assert after < world.settings.wsp_heartbeat_lapse_s  # the lapse cannot be the reason
    later = dispatch.utcnow() + dt.timedelta(seconds=after)
    assert world.drive(worker, now=later) == [f"time_budget:{run_id}"]
    assert world.run(run_id)["stop_reason"] == "time_budget"


# ----------------------------------------------------------------------- orphans


def test_a_pushing_run_at_the_start_of_a_pass_is_an_orphan(world: World) -> None:
    """R3A-10: a driver killed between the commit and the push's answer."""
    run_id = world.queue()
    with world.factory() as session:
        runs.transition(session, run_id, "queued", "pushing", boot_id=None)
        session.commit()
    assert world.drive(FakeWorker())[0] == f"orphan_failed:{run_id}"
    failure = _j(world.run(run_id)["failure"])
    assert failure["code"] == "worker_push_failed"


def test_a_cancelled_running_runs_worker_is_pulled_unread_and_acked(world: World) -> None:
    """OVN-wsp-forge-24: the boot of a cancelled run is an orphan once it finishes."""
    run_id, worker = _running(world)
    with world.factory() as session:
        runs.cancel(session, run_id)
        session.commit()
    worker.state, worker.data = "finished", archive(worker.boot, run_id)
    actions = world.drive(worker)
    assert f"discarded:{worker.boot}" in actions
    assert ("ack", worker.boot, run_id) in worker.calls
    assert world.run(run_id)["status"] == "cancelled"


# ----------------------------------------------------------------------- the wiring


def test_one_pass_end_to_end_against_the_real_supervisor(world: World) -> None:
    """The real HTTP client against the real supervisor (``LocalProcessLauncher``)."""
    from tests.support.wsp_supervisor import SCRIPTED
    from wsp.launcher import LocalProcessLauncher
    from wsp.worker_client import HttpWorker

    run_id = world.queue()
    launcher = LocalProcessLauncher(environment="ci", test_script=str(SCRIPTED))
    with launcher.session() as local:
        client = HttpWorker(local.url)
        try:
            assert world.drive(client) == [f"pushed:{run_id}"]
            deadline = dispatch.utcnow() + dt.timedelta(seconds=30)
            while client.status()["state"] != "finished":
                assert dispatch.utcnow() < deadline, "the session never finished"
                time.sleep(0.2)
            actions = world.drive(client)
        finally:
            client.close()
        assert actions == [f"collected:{run_id}:completed", f"acked:{run_id}"]
        assert local.process.wait(timeout=30) == 0  # the ack ends the supervisor
    run = world.run(run_id)
    assert run["status"] == "completed" and run["acked_at"] is not None


# ----------------------------------------------------------------------- 5.2a (wsp-forge4)


def test_a_killed_drivers_orphan_is_failed_and_its_worker_recycled(world: World) -> None:
    """R3A-10: the driver died after the commit (boot consumed, run ``pushing``) and before the
    push's answer. The next pass fails the run ``worker_push_failed`` and sends the consumed,
    idle supervisor ``/recycle`` exactly once; the run is never pushed again."""
    from wsp.models import WspWorkerBoot

    run_id = world.queue()
    worker = FakeWorker()
    with world.factory() as session:
        session.add(WspWorkerBoot(boot_id=worker.boot, first_seen_at=NOW, run_id=run_id))
        session.flush()
        runs.transition(session, run_id, "queued", "pushing", boot_id=worker.boot)
        session.commit()
    actions = world.drive(worker)
    assert actions[:2] == [f"orphan_failed:{run_id}", f"recycled:{worker.boot}"]
    assert [c for c in worker.calls if c[0] in ("push", "recycle")] == [("recycle", worker.boot)]
    assert _j(world.run(run_id)["failure"])["code"] == "worker_push_failed"


def test_the_driver_abandons_a_hung_supervisor_at_its_timeout(
    world: World, monkeypatch: pytest.MonkeyPatch
) -> None:
    """5.2a: a supervisor that accepts and never answers costs the pass one read timeout; the
    pass ends ``worker_unreachable`` and the run stays queued, unconsumed."""
    import socket as socket_mod

    from wsp import worker_client

    hung = socket_mod.socket()
    hung.bind(("127.0.0.1", 0))
    hung.listen(4)
    monkeypatch.setattr(worker_client, "READ_S", 0.5)
    monkeypatch.setattr(worker_client, "CONNECT_S", 0.5)
    run_id = world.queue()
    client = worker_client.HttpWorker(f"http://127.0.0.1:{hung.getsockname()[1]}")
    started = time.monotonic()
    try:
        actions = world.drive(client)
    finally:
        client.close()
        hung.close()
    assert time.monotonic() - started < 10
    assert actions == ["worker_unreachable"]
    assert world.run(run_id)["status"] == "queued" and world.boots() == 0


@pytest.mark.parametrize(
    ("value", "says"),
    [
        (
            "sensitive",
            "now sensitive, so an external session is refused. Start it again on a local",
        ),
        ("unknown", "could not be read from miWorkbench"),
        (RuntimeError("down"), "could not be read from miWorkbench"),
    ],
)
def test_the_sensitivity_refusal_says_which_and_what_to_do(
    world: World, value: Any, says: str
) -> None:
    """R3 review fix (OR3-016-5), 2026-11-16: a sensitive bundle and an unread value are told
    apart, and the unread one points at the log line naming its cause."""
    run_id = world.queue("ep-ext")
    world.sensitivity = value
    world.drive(FakeWorker())
    message = _j(world.run(run_id)["failure"])["message"]
    assert says in message, message
    assert ("wsp_sensitivity_unread" in message) == (value != "sensitive")


# ----------------------------------------------- feature 018: the engine's counters (8.5, 8.6)

#: Feature 019 P2: the protocol the run was pushed with (``ep-local``'s default), which the engine
#: block must name (another is ``wsp_result_invalid``: ``test_an_engine_that_ran_another_...``).
ENGINE_BLOCK = {
    "action_protocol": "tool_calls",
    "handoffs": 2,
    "refused_turns": 3,
    "kernel_restarts": 1,
}
COUNTER_COLUMNS = (
    "forward_passes_used",
    "action_protocol",
    "handoffs",
    "refused_turns",
    "kernel_restarts",
)


def _finish(world: World, result: dict[str, Any]) -> tuple[str, FakeWorker, list[str]]:
    run_id, worker = _running(world)
    worker.state = "finished"
    worker.data = archive(
        worker.boot,
        run_id,
        **{
            "result.json": json.dumps(result).encode(),
            "transcript.jsonl": b'{"role": "user", "content": "t"}\n',
        },
    )
    return run_id, worker, world.drive(worker)


def _listed(world: World, run_id: str) -> dict[str, Any]:
    from api.v1.endpoints.wsp_runs import list_runs

    with world.factory() as session:
        rows = list_runs(session, None, 50)["runs"]
    (row,) = [r for r in rows if r["run_id"] == run_id]
    return row


def test_the_engines_counters_reach_the_run_row_and_the_list(world: World) -> None:
    """FR-018.11a: the run's columns equal its ``result.json``; the list rows serve them with the
    provider; ``run_document`` stays schema-valid (no counter in it)."""
    from wsp import schemas
    from wsp.models import WspBundle

    run_id, _worker, actions = _finish(
        world, {"stop_reason": "answered", "forward_used": 1, "engine": ENGINE_BLOCK}
    )
    assert actions[0] == f"collected:{run_id}:completed", actions
    run = world.run(run_id)
    assert {c: run[c] for c in COUNTER_COLUMNS} == {"forward_passes_used": 1, **ENGINE_BLOCK}
    listed = _listed(world, run_id)
    assert {c: listed[c] for c in COUNTER_COLUMNS} == {"forward_passes_used": 1, **ENGINE_BLOCK}
    assert listed["provider"] == "miwsp_engine"  # the provider the run was frozen with
    with world.factory() as session:
        row = runs.get_run(session, run_id)
        bundle = session.get(WspBundle, row.bundle_sha256)
        document = runs.run_document(row, None if bundle is None else bundle.manifest)
    schemas.validate_run(document)
    assert document["forward_passes_used"] == 1
    assert not set(ENGINE_BLOCK) & set(document)


def test_the_optional_engine_counters_reach_the_run_row_and_the_list(world: World) -> None:
    """2026-11-21: ``reasoning_turns`` and ``repeated_turns`` (the repeat guard), stored by the
    intake and served in the list rows beside ``refused_turns``; still never in the document."""
    block = {**ENGINE_BLOCK, "reasoning_turns": 2, "repeated_turns": 5}
    run_id, _worker, actions = _finish(world, {"stop_reason": "answered", "engine": block})
    assert actions[0] == f"collected:{run_id}:completed", actions
    run = world.run(run_id)
    assert (run["reasoning_turns"], run["repeated_turns"]) == (2, 5)
    listed = _listed(world, run_id)
    assert (listed["reasoning_turns"], listed["repeated_turns"]) == (2, 5)


PAGING_COUNTERS = {"pages_out": 9, "recalls": 2, "ledger_rows": 31, "peak_prompt_tokens": 8675}


def test_the_paging_counters_reach_the_run_row_and_the_list(world: World) -> None:
    """2026-11-21 (migration 0034): the engine block's paging counters, stored by the intake and
    served in the list rows (with the frozen ``paging``); never in the document."""
    from wsp import schemas
    from wsp.models import WspBundle

    block = {**ENGINE_BLOCK, **PAGING_COUNTERS}
    run_id, _worker, actions = _finish(world, {"stop_reason": "submitted", "engine": block})
    assert actions[0] == f"collected:{run_id}:completed", actions
    run = world.run(run_id)
    assert {k: run[k] for k in PAGING_COUNTERS} == PAGING_COUNTERS
    listed = _listed(world, run_id)
    assert {k: listed[k] for k in PAGING_COUNTERS} == PAGING_COUNTERS
    assert listed["paging"] is None  # frozen only when off
    with world.factory() as session:
        row = runs.get_run(session, run_id)
        bundle = session.get(WspBundle, row.bundle_sha256)
        document = runs.run_document(row, None if bundle is None else bundle.manifest)
    schemas.validate_run(document)
    assert not set(PAGING_COUNTERS) & set(document)


def test_a_result_with_no_engine_block_stores_forward_used_and_null_counters(
    world: World,
) -> None:
    """A result with no ``engine`` key (a program rerun's, and every run of the retired
    ``weightspace``) is not refused: its ``forward_used`` reaches ``forward_passes_used``
    (written by nothing before 018) and the engine columns stay null, but for the protocol the run
    was pushed with (feature 019 P2), which the push wrote and the intake keeps."""
    run_id, _worker, actions = _finish(world, {"stop_reason": "submitted", "forward_used": 4})
    assert actions[0] == f"collected:{run_id}:completed", actions
    run = world.run(run_id)
    assert {c: run[c] for c in COUNTER_COLUMNS} == {
        "forward_passes_used": 4,
        **dict.fromkeys(ENGINE_BLOCK),
        "action_protocol": "tool_calls",
    }


@pytest.mark.parametrize(
    "result",
    [
        {"engine": {k: v for k, v in ENGINE_BLOCK.items() if k != "kernel_restarts"}},
        {"engine": {**ENGINE_BLOCK, "extra": 0}},
        {"engine": None},
        {"engine": [ENGINE_BLOCK]},
        {"engine": {**ENGINE_BLOCK, "action_protocol": "free_text"}},
        {"engine": {**ENGINE_BLOCK, "handoffs": -1}},
        {"engine": {**ENGINE_BLOCK, "refused_turns": True}},
        {"engine": {**ENGINE_BLOCK, "kernel_restarts": "1"}},
        {"engine": {**ENGINE_BLOCK, "handoffs": 2**31}},
        {"engine": {**ENGINE_BLOCK, "refused_turns": 1.5}},
        {"forward_used": -1},
        {"forward_used": "2"},
    ],
    ids=[
        "three-keys",
        "five-keys",
        "null",
        "list",
        "protocol",
        "negative",
        "bool",
        "string",
        "overflow",
        "float",
        "forward-negative",
        "forward-string",
    ],
)
def test_a_malformed_engine_block_fails_the_run_wsp_result_invalid(
    world: World, result: dict[str, Any]
) -> None:
    """FTASKS 8.6: the run fails ``wsp_result_invalid`` with nothing stored, and is still acked."""
    run_id, worker, actions = _finish(world, {"stop_reason": "answered", **result})
    assert actions[0] == f"intake_refused:{run_id}", actions
    run = world.run(run_id)
    assert run["status"] == "failed" and _j(run["failure"])["code"] == "wsp_result_invalid"
    assert {c: run[c] for c in COUNTER_COLUMNS} == {
        **dict.fromkeys(COUNTER_COLUMNS),
        "action_protocol": "tool_calls",  # the push's (019 P2); the refused block stored nothing
    }
    assert ("ack", worker.boot, run_id) in worker.calls


# ------------------------------------------- feature 018: spec.editor and the served window (9.2)
# ------------------------------------------- feature 019: the window every push resolves (W-3)


@dataclass
class WindowReader:
    """A recording ``read_window``: the endpoints it was asked about, and its answer."""

    answer: Any = 32768
    asked: list[str] = field(default_factory=list)

    def __call__(self, endpoint: Any) -> Any:
        self.asked.append(endpoint.id)
        if isinstance(self.answer, Exception):
            raise self.answer
        return self.answer


def _patch_endpoint(world: World, endpoint_id: str, change: dict[str, Any]) -> None:
    from wsp import endpoints

    with world.factory() as session:
        endpoints.update_endpoint(session, endpoint_id, change)
        session.commit()


def _session_entry(world: World, run_id: str) -> dict[str, Any]:
    from api.wsp_internal_app import _session_entry as entry_of

    with world.factory() as session:
        return entry_of(session, runs.get_run(session, run_id))


WINDOW_COLUMNS = (
    "window_tokens",
    "window_source",
    "served_context_tokens",
    "output_ceiling_tokens",
    "output_ceiling_lowered",
)


def _window_columns(world: World, run_id: str) -> dict[str, Any]:
    row = world.run(run_id)
    values = {c: row[c] for c in WINDOW_COLUMNS}
    lowered = values["output_ceiling_lowered"]
    values["output_ceiling_lowered"] = None if lowered is None else bool(lowered)
    return values


def test_spec_editor_carries_the_protocol_and_the_reported_window(world: World) -> None:
    """FR-018.8 / FR-019.2: the push's ``spec.editor`` carries the endpoint's protocol and the
    window its server reported, read once, for this endpoint, at this push; the run row and the
    gateway's session entry carry the same figures."""
    _patch_endpoint(world, "ep-local", {"action_protocol": "structured"})
    run_id = world.queue()
    worker, reader = FakeWorker(), WindowReader(32768)
    assert world.drive(worker, read_window=reader) == [f"pushed:{run_id}"]
    assert worker.pushed is not None
    editor = worker.pushed["spec"]["editor"]
    assert {k: editor[k] for k in ("action_protocol", "context_window_tokens")} == {
        "action_protocol": "structured",
        "context_window_tokens": None,
    }
    assert {k: editor[k] for k in WINDOW_COLUMNS} == {
        "window_tokens": 32768,
        "window_source": "reported",
        "served_context_tokens": 32768,
        "output_ceiling_tokens": 32000,
        "output_ceiling_lowered": False,
    }
    assert editor["window_warnings"] == []
    # 2026-11-27: a miServer endpoint that states no window is read at the claim (is a model
    # resident?) and again at the push, which resolves its own read (F19-D11).
    assert reader.asked == ["ep-local", "ep-local"]
    assert _window_columns(world, run_id) == {k: editor[k] for k in WINDOW_COLUMNS}
    entry = _session_entry(world, run_id)
    assert (entry["window_tokens"], entry["output_ceiling_tokens"]) == (32768, 32000)
    assert entry["served_context_tokens"] == 32768


def test_an_endpoint_at_its_defaults_with_nothing_read_is_assumed(world: World) -> None:
    """F19-D11: nothing read gives 8,192 ``assumed`` with its warning, and the ceiling stays at
    32,000, not lowered (F19-D24 reading 9). A GENERIC endpoint since 2026-11-27: a miServer one
    that states no window waits for its model instead (``test_editor_not_resident.py``)."""
    with world.engine.begin() as conn:
        conn.execute(
            sa.text("UPDATE mf_wsp_endpoints SET serving = 'generic' WHERE id = 'ep-local'")
        )
    run_id = world.queue()
    worker = FakeWorker()
    assert world.drive(worker) == [f"pushed:{run_id}"]  # a generic server reports nothing here
    assert worker.pushed is not None
    editor = worker.pushed["spec"]["editor"]
    assert (editor["action_protocol"], editor["context_window_tokens"]) == ("tool_calls", None)
    assert {k: editor[k] for k in WINDOW_COLUMNS} == {
        "window_tokens": 8192,
        "window_source": "assumed",
        "served_context_tokens": None,
        "output_ceiling_tokens": 32000,
        "output_ceiling_lowered": False,
    }
    assert editor["window_warnings"] == ["window_assumed"]
    assert _window_columns(world, run_id) == {k: editor[k] for k in WINDOW_COLUMNS}


def test_a_stated_window_is_still_read_and_a_mismatch_is_recorded(world: World) -> None:
    """FR-019.2, F19-D10: the stated window wins, the server is read anyway, and a stated window
    above the reported one is recorded with the mismatch warning. The ceiling is lowered to the
    REPORTED window, never to the stated one."""
    _patch_endpoint(world, "ep-local", {"context_window_tokens": 65536})
    run_id = world.queue()
    worker, reader = FakeWorker(), WindowReader(32768)
    assert world.drive(worker, read_window=reader) == [f"pushed:{run_id}"]
    assert reader.asked == ["ep-local"]
    assert worker.pushed is not None
    editor = worker.pushed["spec"]["editor"]
    assert (editor["window_tokens"], editor["window_source"]) == (65536, "stated")
    assert editor["served_context_tokens"] == 32768
    assert editor["window_warnings"] == ["window_stated_above_reported"]
    assert _window_columns(world, run_id)["window_source"] == "stated"
    assert _session_entry(world, run_id)["window_tokens"] == 65536


def test_a_stated_window_below_the_report_has_no_warning(world: World) -> None:
    _patch_endpoint(world, "ep-local", {"context_window_tokens": 16384})
    run_id = world.queue()
    worker, reader = FakeWorker(), WindowReader(32768)
    assert world.drive(worker, read_window=reader) == [f"pushed:{run_id}"]
    assert worker.pushed is not None
    editor = worker.pushed["spec"]["editor"]
    assert (editor["window_tokens"], editor["window_source"]) == (16384, "stated")
    assert editor["window_warnings"] == []
    assert (editor["output_ceiling_tokens"], editor["output_ceiling_lowered"]) == (32000, False)


def test_a_ceiling_above_the_reported_window_is_lowered_for_that_run_only(world: World) -> None:
    """F19-D12: a ceiling of 32,768 on a server reporting 16,384 is 16,384 "lowered" for the run;
    the endpoint keeps 32,768. A report of 65,536 leaves 32,768, not lowered."""
    _patch_endpoint(world, "ep-local", {"output_ceiling_tokens": 32768})
    run_id = world.queue()
    worker = FakeWorker()
    assert world.drive(worker, read_window=WindowReader(16384)) == [f"pushed:{run_id}"]
    assert worker.pushed is not None
    editor = worker.pushed["spec"]["editor"]
    assert (editor["output_ceiling_tokens"], editor["output_ceiling_lowered"]) == (16384, True)
    assert _session_entry(world, run_id)["output_ceiling_tokens"] == 16384
    with world.engine.connect() as conn:
        kept = conn.execute(
            sa.text("SELECT output_ceiling_tokens FROM mf_wsp_endpoints WHERE id = 'ep-local'")
        ).scalar_one()
    assert kept == 32768


def test_a_hosted_endpoint_is_read_by_the_reader_it_is_given(world: World) -> None:
    """The driver asks its reader about every endpoint; it is the production reader
    (``window_read.read_window``) that never reads a hosted one (FR-019.4, its own test). Since
    020 (FR-020.2) a hosted endpoint states its window, so the run uses it, ``stated``."""
    run_id = world.queue("ep-ext")
    worker, reader = FakeWorker(), WindowReader(None)
    assert world.drive(worker, read_window=reader) == [f"pushed:{run_id}"]
    assert worker.pushed is not None
    editor = worker.pushed["spec"]["editor"]
    assert (editor["window_tokens"], editor["window_source"]) == (200_000, "stated")
    assert reader.asked == ["ep-ext"]


@pytest.mark.parametrize(
    "answer",
    [RuntimeError("the server is down"), 0, -1, True, "32768", None],
    ids=["raises", "zero", "negative", "bool", "string", "none"],
)
def test_a_push_never_fails_on_the_window_read(world: World, answer: Any) -> None:
    """A GENERIC endpoint's failed or malformed read is ``assumed`` (F19-D11). Since D1
    (operator, 2026-11-27) a generic endpoint that states no window is read at the claim too
    (only a server that gives no answer at all waits, ``test_editor_not_resident``), so the
    reader is asked twice: the claim's read and the push's."""
    with world.engine.begin() as conn:
        conn.execute(
            sa.text("UPDATE mf_wsp_endpoints SET serving = 'generic' WHERE id = 'ep-local'")
        )
    run_id = world.queue()
    worker, reader = FakeWorker(), WindowReader(answer)
    assert world.drive(worker, read_window=reader) == [f"pushed:{run_id}"]
    assert worker.pushed is not None
    editor = worker.pushed["spec"]["editor"]
    assert (editor["served_context_tokens"], editor["window_source"]) == (None, "assumed")
    assert reader.asked == ["ep-local", "ep-local"]


@pytest.mark.parametrize(
    "answer",
    [RuntimeError("miServer is down"), 0, -1, True, "32768", None],
    ids=["raises", "zero", "negative", "bool", "string", "none"],
)
def test_a_miserver_read_that_gives_no_window_leaves_the_run_queued(
    world: World, answer: Any
) -> None:
    """2026-11-27 (run ``wsp_4fbf5a18758e``): the same answers on a miServer endpoint that states
    no window mean no model is resident: the run waits queued, nothing is consumed or pushed, and
    no window is assumed."""
    run_id = world.queue()
    worker, reader = FakeWorker(), WindowReader(answer)
    assert world.drive(worker, read_window=reader) == [f"editor_not_resident:{run_id}"]
    assert worker.pushed is None and world.boots() == 0
    row = world.run(run_id)
    assert (row["status"], row["window_source"]) == ("queued", None)
    assert reader.asked == ["ep-local"]


class WindowServer:
    """A generic OpenAI-compatible server on 127.0.0.1 that answers ``/v1/models`` and
    ``/props`` with the recorded answers (``tests.support.window_answers``), and records the paths
    it was asked for."""

    def __init__(self, models: Any, props: Any = None, status: int = 200) -> None:
        import threading
        from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

        self.answers = {"/v1/models": models, "/props": props}
        self.status = status
        self.paths: list[str] = []
        outer = self

        class Handler(BaseHTTPRequestHandler):
            def do_GET(self) -> None:
                outer.paths.append(self.path)
                body = outer.answers.get(self.path.split("?")[0])
                data = json.dumps(body).encode()
                self.send_response(outer.status if body is not None else 404)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(data)))
                self.end_headers()
                self.wfile.write(data)

            def log_message(self, *args: Any) -> None:
                return

        self.httpd = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.url = f"http://127.0.0.1:{self.httpd.server_address[1]}/v1"
        threading.Thread(target=self.httpd.serve_forever, daemon=True).start()

    def close(self) -> None:
        self.httpd.shutdown()
        self.httpd.server_close()


def _generic_endpoint(world: World, base_url: str, model: str, **extra: Any) -> str:
    with world.engine.begin() as conn:
        insert(
            conn,
            "mf_wsp_endpoints",
            id="ep-gen",
            kind="openai_compatible",
            base_url=base_url,
            model=model,
            locality="local",
            output_cap_field="max_completion_tokens",
            serving="generic",
            created_at=NOW,
            created_by="operator",
            **extra,
        )
    return "ep-gen"


class _ClosingWorker(FakeWorker):
    closed: int = 0

    def close(self) -> None:
        self.closed += 1


def _drive_task(world: World, monkeypatch: pytest.MonkeyPatch) -> tuple[_ClosingWorker, Any]:
    """One pass through the PRODUCTION Celery task (``wsp.tasks.drive``): its own reader, its
    own settings; only the worker client and the session factory are this world's."""
    from workers.celery_app import celery_app
    from wsp import tasks

    worker = _ClosingWorker()
    monkeypatch.setattr("wsp.worker_client.HttpWorker", lambda url: worker)
    monkeypatch.setattr("db.session.get_session_factory", lambda: world.factory)
    monkeypatch.setattr(tasks, "run_files", lambda run, bundle: [])
    monkeypatch.setattr(tasks, "read_sensitivity", lambda run, bundle: "not_sensitive")
    # 2026-11-27: the one HTTP read the production reader makes that no test server answers here,
    # miServer's ``/api/health/inference``: a model is resident (``RESIDENT_WINDOW``), so a run on
    # a miServer endpoint that states no window is claimed instead of waiting for one.
    monkeypatch.setattr(tasks, "read_served_window", lambda endpoint: RESIDENT_WINDOW)
    enqueued: list[str] = []
    monkeypatch.setattr(tasks, "enqueue_push", enqueued.append)
    get_wsp_settings.cache_clear()
    celery_app.loader.import_default_modules()
    answer = celery_app.tasks["miforge.wsp.drive"].run()
    # 019 P3: then the PRODUCTION push task for each claim, as the solo wsp_push consumer would;
    # it is where the window is read. Each claim action reads as the push's outcome.
    outcomes = {
        run_id: celery_app.tasks["miforge.wsp.push"].run(run_id)["outcome"] for run_id in enqueued
    }
    answer["actions"] = [
        f"{outcomes[a.split(':')[1]]}:{a.split(':')[1]}" if a.startswith("claimed:") else a
        for a in answer["actions"]
    ]
    return worker, answer


def test_the_production_pass_reads_a_generic_servers_window(
    world: World, monkeypatch: pytest.MonkeyPatch
) -> None:
    """W-3 through ``wsp.tasks.drive``: a generic endpoint behind a vLLM-shaped server is pushed
    with the window it reports; the row, ``spec.editor`` and the gateway's session entry agree."""
    from tests.support import window_answers as wa

    server = WindowServer(wa.VLLM_MODELS)
    try:
        endpoint_id = _generic_endpoint(world, server.url, wa.VLLM_MODEL)
        run_id = world.queue(endpoint_id)
        worker, answer = _drive_task(world, monkeypatch)
    finally:
        server.close()
    assert answer == {"actions": [f"pushed:{run_id}"]}
    assert server.paths == ["/v1/models", "/v1/models"]  # the claim's read (D1) and the push's
    assert worker.pushed is not None and worker.closed == 2  # the pass's client and the push task's
    editor = worker.pushed["spec"]["editor"]
    expected = {
        "window_tokens": wa.VLLM_WINDOW,
        "window_source": "reported",
        "served_context_tokens": wa.VLLM_WINDOW,
        "output_ceiling_tokens": 32000,
        "output_ceiling_lowered": False,
    }
    assert {k: editor[k] for k in WINDOW_COLUMNS} == expected
    assert _window_columns(world, run_id) == expected
    entry = _session_entry(world, run_id)
    assert (entry["window_tokens"], entry["output_ceiling_tokens"]) == (wa.VLLM_WINDOW, 32000)


def test_the_production_pass_falls_through_to_llamacpps_props(
    world: World, monkeypatch: pytest.MonkeyPatch
) -> None:
    from tests.support import window_answers as wa

    server = WindowServer(wa.LLAMACPP_MODELS, wa.LLAMACPP_PROPS)
    try:
        endpoint_id = _generic_endpoint(world, server.url, wa.LLAMACPP_MODEL)
        run_id = world.queue(endpoint_id)
        worker, _ = _drive_task(world, monkeypatch)
    finally:
        server.close()
    # The claim's read (D1) and the push's, each falling through to /props.
    assert server.paths == ["/v1/models", f"/props?model={wa.LLAMACPP_MODEL}"] * 2
    assert worker.pushed is not None
    editor = worker.pushed["spec"]["editor"]
    assert (editor["window_tokens"], editor["window_source"]) == (
        wa.LLAMACPP_SLOT_WINDOW,
        "reported",
    )
    assert world.run(run_id)["window_tokens"] == wa.LLAMACPP_SLOT_WINDOW


def test_a_failed_push_read_is_assumed_whatever_the_registration_read_was(
    world: World, monkeypatch: pytest.MonkeyPatch
) -> None:
    """F19-D11 through the production pass: the endpoint's registration read was 32,768, the push
    read states no window (401: a keyed server the backend sends no key), so the run is 8,192
    ``assumed`` and 32,768 appears in no window column and nowhere in ``spec.editor``. Until D1
    (operator, 2026-11-27) this read failed with a 500; a 5xx is now no answer, and the run
    waits instead (``test_editor_not_resident.test_d1_a_5xx_is_no_answer...``)."""
    from tests.support import window_answers as wa

    server = WindowServer(wa.VLLM_MODELS, wa.LLAMACPP_PROPS, status=401)
    try:
        endpoint_id = _generic_endpoint(
            world,
            server.url,
            wa.VLLM_MODEL,
            registration_window_tokens=32768,
            registration_window_read_at=NOW,
        )
        run_id = world.queue(endpoint_id)
        worker, _ = _drive_task(world, monkeypatch)
    finally:
        server.close()
    assert worker.pushed is not None
    editor = worker.pushed["spec"]["editor"]
    assert (editor["window_tokens"], editor["window_source"]) == (8192, "assumed")
    assert editor["window_warnings"] == ["window_assumed"]
    columns = _window_columns(world, run_id)
    assert 32768 not in columns.values()
    assert 32768 not in [v for v in editor.values() if isinstance(v, int)]


def test_the_workers_lifetime_is_measured_from_the_push_completion(world: World) -> None:
    """Review 2026-11-21 (018 thinking round), control M6 survived: measuring the poll's clock
    from ``started_at`` (the push's START) instead of ``pushed_at`` left every test green,
    because the incident test above stays inside the worker's lifetime on either clock. Here a
    301 s push is followed by a worker that answers, unfinished, one second before its lifetime
    from the push's completion: it is still waited for. From the push's start it would be 300 s
    past its lifetime, and the run would be failed ``time_budget`` with its result unread."""
    clock = Clock()
    worker = SlowPushWorker(clock=clock)
    run_id = world.queue()
    assert world.drive(worker, now=clock) == [f"pushed:{run_id}"]
    running_at = _at(world.run(run_id)["pushed_at"])
    lifetime = dispatch.worker_lifetime_s(60, world.settings)
    clock.at = running_at + dt.timedelta(seconds=lifetime - 1)
    assert world.drive(worker, now=clock) == []
    assert world.run(run_id)["status"] == "running"
    clock.at = running_at + dt.timedelta(seconds=lifetime + 1)
    assert world.drive(worker, now=clock) == [f"time_budget:{run_id}"]


def test_a_run_with_every_019_column_set_is_a_valid_run_document(world: World) -> None:
    """W-9 (FR-019.21): a run pushed with a stated window above its server's report and a
    lowered ceiling (every 019 column set) completes, reads back as a schema-valid
    ``miforge.wsp-run/v1`` with no 019 key in it, and its list row serves the fields."""
    from wsp import schemas
    from wsp.models import WspBundle

    _patch_endpoint(world, "ep-local", {"context_window_tokens": 65536})
    _patch_endpoint(world, "ep-local", {"output_ceiling_tokens": 32768})
    run_id = world.queue()
    worker = FakeWorker()
    assert world.drive(worker, read_window=WindowReader(16384)) == [f"pushed:{run_id}"]
    assert _window_columns(world, run_id) == {
        "window_tokens": 65536,
        "window_source": "stated",
        "served_context_tokens": 16384,
        "output_ceiling_tokens": 16384,
        "output_ceiling_lowered": True,
    }
    worker.state = "finished"
    worker.data = archive(
        worker.boot,
        run_id,
        **{
            "result.json": json.dumps({"stop_reason": "answered"}).encode(),
            "transcript.jsonl": b'{"role": "user", "content": "t"}\n',
        },
    )
    assert world.drive(worker)[0] == f"collected:{run_id}:completed"
    with world.factory() as session:
        row = runs.get_run(session, run_id)
        bundle = session.get(WspBundle, row.bundle_sha256)
        document = runs.run_document(row, None if bundle is None else bundle.manifest)
    schemas.validate_run(document)
    assert not set(WINDOW_COLUMNS) & set(document) and "window_warnings" not in document
    listed = _listed(world, run_id)
    assert {
        k: listed[k]
        for k in (
            "window_tokens",
            "window_source",
            "window_reported_tokens",
            "output_ceiling_tokens",
            "output_ceiling_lowered",
            "window_warnings",
        )
    } == {
        "window_tokens": 65536,
        "window_source": "stated",
        "window_reported_tokens": 16384,
        "output_ceiling_tokens": 16384,
        "output_ceiling_lowered": True,
        "window_warnings": ["window_stated_above_reported"],
    }


def test_a_pre_019_run_serves_nulls_and_no_warning(world: World) -> None:
    """FTASKS 5.4: a run written before 019 (no window columns) reads "not recorded"."""
    from wsp import progress

    run_id = world.queue()
    listed = _listed(world, run_id)
    assert {k: listed[k] for k in ("window_tokens", "window_source", "window_warnings")} == {
        "window_tokens": None,
        "window_source": None,
        "window_warnings": [],
    }
    with world.factory() as session:
        full = progress.run_progress(session, runs.get_run(session, run_id))
    assert (full["window_source"], full["window_warnings"]) == (None, [])


def test_a_ceiling_patch_fails_a_queued_run_at_push(world: World) -> None:
    """W-5 / FR-019.14: the ceiling joins the fingerprint when set, so the operator's PATCH fails
    a run queued before it ``wsp_endpoint_changed``, and pushes nothing."""
    run_id = world.queue()
    _patch_endpoint(world, "ep-local", {"output_ceiling_tokens": 4096})
    worker = FakeWorker()
    assert world.drive(worker) == [f"refused:{run_id}:wsp_endpoint_changed"]
    assert worker.pushed is None
    assert _j(world.run(run_id)["failure"])["code"] == "wsp_endpoint_changed"


def test_a_hosted_run_through_the_production_pass_is_assumed_and_not_refused(
    world: World, monkeypatch: pytest.MonkeyPatch
) -> None:
    """FR-019.4 (FTASKS 4.8): a run on an external endpoint is not read (the production reader
    makes no request) and no 019 check refuses it. Since 020 (FR-020.2) the endpoint states its
    window, which the run records ``stated``; a hosted row without one cannot run at all
    (``endpoint_window_required``, ``test_a_run_queued_before_the_deploy_fails_at_push_by_name``).
    """
    run_id = world.queue("ep-ext")
    worker, answer = _drive_task(world, monkeypatch)
    assert answer == {"actions": [f"pushed:{run_id}"]}
    assert worker.pushed is not None
    editor = worker.pushed["spec"]["editor"]
    assert (editor["window_tokens"], editor["window_source"]) == (200_000, "stated")
    assert _window_columns(world, run_id)["window_source"] == "stated"


def test_the_push_logs_its_window_once_with_its_code(
    world: World, caplog: pytest.LogCaptureFixture
) -> None:
    """FR-019.2 (review control X1): each push logs ONE ``wsp_window`` line naming the run, the
    window, its source and the warnings, from named fields only."""
    import logging

    _patch_endpoint(world, "ep-local", {"context_window_tokens": 65536})
    run_id = world.queue()
    with caplog.at_level(logging.INFO, logger="wsp.dispatch"):
        assert world.drive(FakeWorker(), read_window=WindowReader(32768)) == [f"pushed:{run_id}"]
    lines = [r for r in caplog.records if getattr(r, "code", None) == "wsp_window"]
    assert len(lines) == 1
    line = lines[0]
    assert (line.run_id, line.window_tokens, line.window_source, line.warnings) == (  # type: ignore[attr-defined]
        run_id,
        65536,
        "stated",
        ["window_stated_above_reported"],
    )


def test_an_engine_that_ran_another_protocol_than_pushed_is_refused(world: World) -> None:
    """Feature 019 P2 (TID §3.6): the push writes the endpoint's protocol to the run; an engine
    block naming another fails the run ``wsp_result_invalid`` (the engine ran something it was not
    pushed with), and the pushed protocol stays on the row."""
    run_id, worker, actions = _finish(
        world,
        {"stop_reason": "answered", "engine": {**ENGINE_BLOCK, "action_protocol": "structured"}},
    )
    assert actions[0] == f"intake_refused:{run_id}", actions
    run = world.run(run_id)
    assert run["status"] == "failed" and _j(run["failure"])["code"] == "wsp_result_invalid"
    assert "pushed with" in _j(run["failure"])["message"]
    assert run["action_protocol"] == "tool_calls"
    assert ("ack", worker.boot, run_id) in worker.calls


# ------------------------------------------- feature 019 P2: the pass rule at push (S-3)


def _generic_structured_passed(world: World) -> None:
    from wsp.models import WspEndpoint

    with world.factory() as session:
        row = session.get(WspEndpoint, "ep-local")
        assert row is not None
        row.serving, row.action_protocol = "generic", "structured"
        row.protocol_test_result = "passed"
        row.protocol_test_run_id = "wsp_000000000999"
        row.protocol_test_at = NOW
        row.protocol_tested_base_url = row.base_url
        row.protocol_tested_model = row.model
        row.protocol_tested_action_protocol = "structured"
        session.commit()


def test_a_run_queued_on_a_pass_is_pushed_with_its_protocol(world: World) -> None:
    _generic_structured_passed(world)
    run_id = world.queue()
    worker = FakeWorker()
    assert world.drive(worker) == [f"pushed:{run_id}"]
    assert world.run(run_id)["action_protocol"] == "structured"
    assert worker.pushed is not None
    assert worker.pushed["spec"]["editor"]["action_protocol"] == "structured"


def test_a_pass_that_clears_while_the_run_waits_refuses_it_at_push(world: World) -> None:
    """FR-019.11's wiring: a failed re-test after the run was queued; the push check refuses it
    and nothing is pushed (the fingerprint does not move: the record is outside it)."""
    _generic_structured_passed(world)
    run_id = world.queue()
    with world.engine.begin() as conn:
        conn.execute(sa.text("UPDATE mf_wsp_endpoints SET protocol_test_result = 'failed'"))
    worker = FakeWorker()
    assert world.drive(worker) == [f"refused:{run_id}:wsp_endpoint_protocol_untested"]
    assert worker.pushed is None


def test_a_protocol_test_is_pushed_without_a_pass(world: World, monkeypatch) -> None:
    from wsp import protocol_test

    monkeypatch.setenv("WSP_PROVIDERS_AVAILABLE", "miwsp_engine")
    get_wsp_settings.cache_clear()
    with world.engine.begin() as conn:
        conn.execute(
            sa.text(
                "UPDATE mf_wsp_endpoints SET serving = 'generic', action_protocol = 'structured' "
                "WHERE id = 'ep-local'"
            )
        )
    with world.factory() as session:
        prepared = protocol_test.prepare(session, "ep-local", {}, "operator", get_wsp_settings())
        run_id = runs.create_run(session, prepared, approval_id=None).run_id
        session.commit()
    worker = FakeWorker()
    assert world.drive(worker) == [f"pushed:{run_id}"]


# --------------------------------------------- feature 020: filing, push and the list (H-4, H-6)


def _hosted_anthropic(world: World, endpoint_id: str, **values: Any) -> None:
    row: dict[str, Any] = {
        "id": endpoint_id,
        "kind": "anthropic",
        "base_url": EXTERNAL,
        "model": "claude-x",
        "locality": "external",
        "price_in_per_mtok": 3,
        "price_out_per_mtok": 15,
        "price_cache_write_per_mtok": 3.75,
        "price_cache_read_per_mtok": 0.3,
        "output_cap_field": "max_tokens",
        "serving": "generic",
        "context_window_tokens": 200_000,
        "thinking": "default",
        "history_mode": "append_only",
        "created_at": NOW,
        "created_by": "operator",
    }
    row.update(values)
    with world.engine.begin() as conn:
        insert(conn, "mf_wsp_endpoints", **row)


def _queued(world: World) -> int:
    with world.engine.connect() as conn:
        return int(
            conn.execute(
                sa.text("SELECT COUNT(*) FROM mf_wsp_runs WHERE status = 'queued'")
            ).scalar_one()
        )


def _queue_before_020(world: World, monkeypatch: pytest.MonkeyPatch, endpoint_id: str) -> str:
    """A run filed before the 020 deploy: frozen with no run_refusal, as the code then was."""
    from wsp import endpoints

    with monkeypatch.context() as m:
        m.setattr(endpoints, "run_refusal", lambda _row: None)
        return world.queue(endpoint_id)


@pytest.mark.parametrize(
    ("values", "code"),
    [
        ({"context_window_tokens": None}, "endpoint_window_required"),
        ({"price_cache_write_per_mtok": None}, "wsp_endpoint_cache_prices_required"),
        ({"price_cache_read_per_mtok": None}, "wsp_endpoint_cache_prices_required"),
    ],
)
def test_a_start_on_a_row_that_refuses_runs_queues_nothing(
    world: World, values: dict[str, Any], code: str
) -> None:
    """FR-020.2, FR-020.3, F20-D12: refused at filing, by name, before anything is frozen."""
    from core.errors import Refusal

    _hosted_anthropic(world, "ep-old", **values)
    with pytest.raises(Refusal) as refused:
        world.queue("ep-old")
    assert refused.value.code == code
    assert _queued(world) == 0


@pytest.mark.parametrize(
    ("values", "code"),
    [
        ({"context_window_tokens": None}, "endpoint_window_required"),
        ({"price_cache_read_per_mtok": None}, "wsp_endpoint_cache_prices_required"),
    ],
)
def test_a_run_queued_before_the_deploy_fails_at_push_by_name(
    world: World, monkeypatch: pytest.MonkeyPatch, values: dict[str, Any], code: str
) -> None:
    """A run frozen before 020 on a row that now refuses runs (its fingerprint unchanged) fails
    at push by name and is never pushed; no boot is consumed."""
    _hosted_anthropic(world, "ep-old", **values)
    run_id = _queue_before_020(world, monkeypatch, "ep-old")
    worker = FakeWorker()
    assert world.drive(worker) == [f"refused:{run_id}:{code}"]
    run = world.run(run_id)
    assert run["status"] == "failed" and _j(run["failure"])["code"] == code
    assert "push" not in worker.names() and world.boots() == 0


def test_the_list_serves_the_same_refusal_as_filing_and_push(world: World) -> None:
    from wsp import endpoints

    _hosted_anthropic(world, "ep-old", context_window_tokens=None)
    with world.factory() as session:
        row = session.get(endpoints.WspEndpoint, "ep-old")
        assert row is not None
        doc = endpoints.public_document(row)
        assert doc["runs_refused"]["code"] == "endpoint_window_required"
        assert (doc["runs_refused"]["code"], doc["runs_refused"]["message"]) == (
            endpoints.run_refusal(row)
        )


def test_the_three_callers_of_the_one_check_are_wired() -> None:
    """TD-2, by AST: ``prepare_start`` and ``dispatch._refusal`` call ``run_refusal``, and the
    list's ``public_document`` calls ``runs_refused``, which calls it."""
    import ast
    import inspect
    import textwrap

    from wsp import endpoints

    def calls(fn: Any) -> set[str]:
        tree = ast.parse(textwrap.dedent(inspect.getsource(fn)))
        return {
            n.func.attr if isinstance(n.func, ast.Attribute) else getattr(n.func, "id", "")
            for n in ast.walk(tree)
            if isinstance(n, ast.Call)
        }

    assert "run_refusal" in calls(runs.prepare_start)
    assert "run_refusal" in calls(dispatch._refusal)
    assert "runs_refused" in calls(endpoints.public_document)
    assert "run_refusal" in calls(endpoints.runs_refused)


def test_a_patch_to_append_only_fails_a_queued_run_at_push(world: World) -> None:
    """FR-020.1's AC: the history mode joins the fingerprint away from paging."""
    run_id = world.queue()
    _patch_endpoint(world, "ep-local", {"history_mode": "append_only"})
    assert world.drive(FakeWorker()) == [f"refused:{run_id}:wsp_endpoint_changed"]


def test_spec_editor_carries_the_history_mode_and_the_locality(world: World) -> None:
    """H-6 (FR-020.6, TD-5): the push's ``spec.editor`` for a paging and an append-only run."""
    local = world.queue()
    worker = FakeWorker()
    assert world.drive(worker) == [f"pushed:{local}"]
    assert worker.pushed is not None
    editor = worker.pushed["spec"]["editor"]
    assert (editor["history_mode"], editor["locality"]) == ("paging", "local")
    with world.factory() as session:
        runs.transition(session, local, "running", "cancelled", stop_reason="cancelled")
        session.commit()
    _hosted_anthropic(world, "ep-hosted")
    hosted = world.queue("ep-hosted")
    assert world.run(hosted)["request"] is not None
    assert _j(world.run(hosted)["request"])["paging"] == "off"  # F20-D4: the only paging value
    second = FakeWorker()
    assert world.drive(second) == [f"pushed:{hosted}"]
    assert second.pushed is not None
    editor = second.pushed["spec"]["editor"]
    assert (editor["history_mode"], editor["locality"]) == ("append_only", "external")


def test_the_gateways_endpoint_document_carries_the_thinking(world: World) -> None:
    """H-6 (FR-020.16, TDD §5.4): the gateway reads the effort a hosted endpoint declares."""
    from api.wsp_internal_app import endpoint_document

    _hosted_anthropic(world, "ep-hosted", thinking="medium")
    with world.factory() as session:
        row = runs.live_endpoint(session, "ep-hosted")
        assert row is not None
        assert endpoint_document(row)["thinking"] == "medium"
        local = runs.live_endpoint(session, "ep-local")
        assert local is not None
        assert endpoint_document(local)["thinking"] == "off"
