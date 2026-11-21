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

    def push(self, document: dict[str, Any], files: list[tuple[str, Path]]) -> dict[str, Any]:
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
        """One pass; ``now`` is a fixed time or a clock (called each time the driver reads it)."""
        kwargs.setdefault("files_for", lambda run, bundle: [])
        clock: Callable[[], dt.datetime]
        if now is None:
            clock = dispatch.utcnow
        elif isinstance(now, dt.datetime):
            fixed = now
            clock = lambda: fixed  # noqa: E731
        else:
            clock = now
        return dispatch.drive(
            self.factory, worker, self.settings, read_sensitivity=self.read, now=clock, **kwargs
        ).actions

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
    assert worker.names() == ["ready", "ready", "push"]  # orphan check, then the push's ready


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
    actions = dispatch.drive(
        world.factory,
        worker,
        world.settings,
        read_sensitivity=tasks.read_sensitivity,
        files_for=lambda run, bundle: [],
    ).actions
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

    def push(self, document: dict[str, Any], files: list[tuple[str, Path]]) -> dict[str, Any]:
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

ENGINE_BLOCK = {
    "action_protocol": "structured",
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
    assert listed["provider"] == "weightspace"  # the provider the run was frozen with
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
    """A ``weightspace`` result (no ``engine`` key) is not refused: its ``forward_used`` reaches
    ``forward_passes_used`` (written by nothing before 018) and the engine columns stay null."""
    run_id, _worker, actions = _finish(world, {"stop_reason": "submitted", "forward_used": 4})
    assert actions[0] == f"collected:{run_id}:completed", actions
    run = world.run(run_id)
    assert {c: run[c] for c in COUNTER_COLUMNS} == {
        "forward_passes_used": 4,
        **dict.fromkeys(ENGINE_BLOCK),
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
    assert {c: run[c] for c in COUNTER_COLUMNS} == dict.fromkeys(COUNTER_COLUMNS)
    assert ("ack", worker.boot, run_id) in worker.calls


# ------------------------------------------- feature 018: spec.editor and the served window (9.2)


@dataclass
class WindowReader:
    """A recording ``read_served_window``: the endpoints it was asked about, and its answer."""

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


def test_spec_editor_carries_the_protocol_and_miservers_served_window(world: World) -> None:
    """FR-018.8 / FR-018.12: the push's ``spec.editor`` carries the endpoint's protocol and the
    window miServer reported, read once, for this endpoint, at this push."""
    _patch_endpoint(world, "ep-local", {"action_protocol": "structured"})
    run_id = world.queue()
    worker, reader = FakeWorker(), WindowReader(32768)
    assert world.drive(worker, read_served_window=reader) == [f"pushed:{run_id}"]
    assert worker.pushed is not None
    editor = worker.pushed["spec"]["editor"]
    assert (
        editor["action_protocol"],
        editor["context_window_tokens"],
        editor["served_context_tokens"],
    ) == ("structured", None, 32768)
    assert reader.asked == ["ep-local"]
    # 2026-11-21: the same figure is stored on the run, so the gateway's session has it too.
    assert world.run(run_id)["served_context_tokens"] == 32768
    from api.wsp_internal_app import _session_entry

    with world.factory() as session:
        entry = _session_entry(session, runs.get_run(session, run_id))
    assert entry["served_context_tokens"] == 32768


def test_an_endpoint_at_its_defaults_pushes_tool_calls_and_no_windows(world: World) -> None:
    run_id = world.queue()
    worker = FakeWorker()
    assert world.drive(worker) == [f"pushed:{run_id}"]  # the default reader reads nothing
    assert worker.pushed is not None
    editor = worker.pushed["spec"]["editor"]
    assert (
        editor["action_protocol"],
        editor["context_window_tokens"],
        editor["served_context_tokens"],
    ) == ("tool_calls", None, None)


def test_an_operator_window_is_pushed_and_miserver_is_not_asked(world: World) -> None:
    _patch_endpoint(world, "ep-local", {"context_window_tokens": 16384})
    run_id = world.queue()
    worker, reader = FakeWorker(), WindowReader(32768)
    assert world.drive(worker, read_served_window=reader) == [f"pushed:{run_id}"]
    assert worker.pushed is not None
    editor = worker.pushed["spec"]["editor"]
    assert (editor["context_window_tokens"], editor["served_context_tokens"]) == (16384, None)
    assert reader.asked == []


def test_a_generic_endpoint_is_never_asked_for_a_served_window(world: World) -> None:
    run_id = world.queue("ep-ext")
    worker, reader = FakeWorker(), WindowReader(32768)
    assert world.drive(worker, read_served_window=reader) == [f"pushed:{run_id}"]
    assert worker.pushed is not None
    assert worker.pushed["spec"]["editor"]["served_context_tokens"] is None
    assert reader.asked == []


@pytest.mark.parametrize(
    "answer",
    [RuntimeError("miServer is down"), 0, -1, True, "32768", None],
    ids=["raises", "zero", "negative", "bool", "string", "none"],
)
def test_a_push_never_fails_on_the_served_window(world: World, answer: Any) -> None:
    run_id = world.queue()
    worker, reader = FakeWorker(), WindowReader(answer)
    assert world.drive(worker, read_served_window=reader) == [f"pushed:{run_id}"]
    assert worker.pushed is not None
    assert worker.pushed["spec"]["editor"]["served_context_tokens"] is None
    assert reader.asked == ["ep-local"]


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
