"""A cancel lands while a run is running, and its progress is visible meanwhile  (F11 18.5).

Found live in F11 window 5 (2026-10-19): during an attached run the worker's session sat
``idle in transaction`` holding the ``mf_runs`` row lock, so

* ``POST /runs/{id}/cancel`` blocked until the run ended (3 m 45 s and 1 m 32 s live), then
  answered ``202`` with ``cancel_requested_at: null`` - a stop that stopped nothing;
* ``last_step`` stayed at its pre-run value in ``GET /runs`` for the whole run;
* ``heartbeat_at`` was uncommitted for the whole run, so a run longer than the janitor's
  900 s lapse threshold would have been reaped as dead while it trained.

**The root cause** was a shared heartbeat throttle with one committing call site among
five. The loop's start, the attached and F5 saves and the score recorder all touch the beat
in the run's session WITHOUT committing; only the step record committed, and only when its
own touch was the one that wrote. A touch that writes resets the 60 s throttle, so when the
60 s mark fell inside a save the save's uncommitted touch won - and at the live cadence
(50 steps ~6.6 s, save ~9 s, so 60 s mod 15.6 s lands in a save) it then won every minute.
The loop-start beat's row lock was never released. Fixed by one short transaction per step
(the step record commits every step, with ``last_step``) and a commit at the loop's start
and at the end of every boundary (``training/loop.py``), so the outcome no longer depends
on which site wins the throttle. The same pattern was on the non-attached path (F5's save
and the score recorder touch without committing), so the run tests here run both ways.

**What makes this able to see it**: the run is driven in a real thread through the
production driver (:func:`training.run_driver.drive_recorded_run` -> ``tasks.drive_run`` ->
F6's loop -> the real step record), with its OWN connection, while the API's cancel route and
the janitor run on OTHER connections - the shape production has. The suite's in-memory
SQLite (``StaticPool``, one shared connection) sees uncommitted rows and cannot show a lock
wait, which is why every earlier test passed over this. The engine is
:func:`tests.support.db.migrated_engine`: PostgreSQL when ``MIFORGE_TEST_DATABASE_URL`` is
set (CI's ``postgres`` job, the engine production runs), a file-backed SQLite database
otherwise (separate connections, real locking). Where a phase must be in progress when the
cancel is sent, it is held on a :class:`threading.Event`, never on a sleep.

Doubled, and why: ``attach`` - the CUDA-IPC rebuild (``rebuild.build`` / ``build_module``)
needs a card, so it returns a tiny torch module, and the R-02.2 verifier is off (pinned by
``test_hotupdate_verify_wiring.py``); the miLLM side is the conformance-checked fake surface,
and the attached save is the REAL :class:`AttachedSaver` (wrapped only to hold it).
``load`` - the acquisition returns the tiny module (a real load needs a card), and F5's
writer is replaced because the tiny module has no ``save_pretrained``; the replacement ends
with the same ``heartbeat.touch(session)`` the real service ends with
(``services/checkpoints/service.py``), which is the line that matters here.
"""

from __future__ import annotations

import re
import threading
import time
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any, ClassVar

import pytest
import sqlalchemy as sa
from fastapi.testclient import TestClient
from sqlalchemy.orm import Session, sessionmaker

from api.app import create_app
from api.deps import session_dependency
from core.config import reset_settings
from models.run import Run, Step
from services import run_definition as validator
from tests.support.db import migrated_engine
from tests.support.fake_weights_surface import FakeWeightsSurface
from tests.support.fixtures_hotupdate import configure
from training import acquisition, run_driver, tasks
from training.acquisition.base import (
    AcquiredModel,
    AcquisitionMode,
    AcquisitionRecord,
    SourceKind,
)
from workers import janitor

from .test_continuity_across_a_resume import _cfg, _fixed_init_model, _FixedCycleSource

RUN = "f11-cancel-mid-run"
#: The run's own bound. Large enough that the run is still going when the cancel lands, and
#: finite so a cancel that does NOT land ends the test in seconds rather than hanging it.
TOTAL = 400
#: Per-step delay in the batch source, so the run has a wall-clock duration to cancel into.
STEP_DELAY_S = 0.02
#: How long the cancel route may take. Live it took 225 s; a lock-free UPDATE takes
#: milliseconds. Generous for a loaded CI runner, and two orders below the defect.
CANCEL_BOUND_S = 2.0
#: Production's heartbeat throttle. Where a test needs a beat mid-run it says so and sets 1.
PRODUCTION_BEAT_S = 60
MODES = ("attach", "load")
POSTGRES_ONLY = "needs PostgreSQL's row locks (MIFORGE_TEST_DATABASE_URL); CI's postgres job"


def _definition(mode: str, save_every: int) -> dict[str, Any]:
    cfg = _cfg()
    source = {"mode": "attach"} if mode == "attach" else {"mode": "load", "path": "/models/tiny"}
    return {
        "run_id": RUN,
        "model_source": source,
        "run_type": "change",
        "run_configuration": {
            "learning_rate": cfg.learning_rate,
            "warmup_steps": cfg.warmup_steps,
            "batch_size": cfg.batch_size,
            "gradient_accumulation": 1,
            "optimizer_state_precision": "32bit",
        },
        "signal_source": {"source": "corpus"},
        "parts": [{"part_id": "part-corpus", "kind": "corpus"}],
        "cadences": {"test_every": 100000, "retention_every": 100000, "save_every": save_every},
        "stop_conditions": [{"condition": "step_count", "config": {"total": TOTAL}}],
        "domain": "legal",
        "seed": cfg.seed,
    }


class _PackDouble:
    class _Definition:
        domain = "legal"

    definition = _Definition()

    def payload(self) -> dict[str, Any]:
        return {"pack_id": "legal", "version": "1", "retention_floor": None, "parts": []}

    def part_records(self) -> list[Any]:
        return []


class _Base:
    device_uuid = "GPU-test"
    tensors: ClassVar[dict[str, Any]] = {}


class Gate:
    """Holds a phase open: ``entered`` is set when the phase reaches it, ``release`` lets it go.

    Bounded waits everywhere, so a defect ends the test with a message rather than a hang.
    """

    def __init__(self) -> None:
        self.entered = threading.Event()
        self.release = threading.Event()

    def hold(self) -> None:
        self.entered.set()
        assert self.release.wait(30), "the test never released the held phase"


class SlowSource(_FixedCycleSource):
    """The fixed cycle, a little slower, recording when each batch was handed out.

    ``hold_after_save`` holds the first batch requested after a save on a gate - the step
    AFTER a save, which is where an uncommitted save-time write would be held.
    """

    def __init__(self) -> None:
        super().__init__(cursor=0)
        self.handed_at: list[float] = []
        self.hold_after_save: Gate | None = None
        self.saved = threading.Event()

    def next(self) -> Any:
        if self.hold_after_save is not None and self.saved.is_set():
            gate, self.hold_after_save = self.hold_after_save, None
            gate.hold()
        time.sleep(STEP_DELAY_S)
        self.handed_at.append(time.monotonic())
        return super().next()


class Rig:
    """One run, defined through the real launch, driven by the real driver on demand."""

    def __init__(
        self,
        *,
        engine: Any,
        scope: Callable[[], Any],
        client: TestClient,
        log_root: Path,
        surface: FakeWeightsSurface | None,
        acquirer: Any,
    ) -> None:
        self.engine = engine
        self.scope = scope
        self.client = client
        self.log_root = log_root
        self.surface = surface
        self.acquirer = acquirer
        self.source = SlowSource()
        #: Held at step 0's baseline evaluation when set.
        self.hold_evaluation: Gate | None = None
        #: Held inside the save at this step, when set.
        self.hold_save_at: tuple[int, Gate] | None = None
        #: Seconds a save takes before it writes (and touches the beat), when set.
        self.save_delay_s = 0.0
        self.saves: list[int] = []
        self.result: dict[str, Any] | None = None
        self.error: BaseException | None = None
        self._thread: threading.Thread | None = None

    # ---------------------------------------------------------------- the save hooks
    def on_save(self, step: int) -> None:
        """Called by both modes' save at its START, before anything is written."""
        if self.save_delay_s:
            time.sleep(self.save_delay_s)
        if self.hold_save_at is not None and self.hold_save_at[0] == step:
            self.hold_save_at[1].hold()

    def after_save(self, step: int) -> None:
        self.saves.append(step)
        self.source.saved.set()

    # ---------------------------------------------------------------- driving
    def evaluate(self, step: int) -> None:
        if step == 0 and self.hold_evaluation is not None:
            self.hold_evaluation.hold()

    def start(self) -> None:
        def work() -> None:
            try:
                with self.scope() as session:
                    self.result = run_driver.drive_recorded_run(
                        RUN,
                        session=session,
                        session_scope=self.scope,
                        batch_source=self.source,
                        objectives=None,
                        evaluate=self.evaluate,
                        log_root=self.log_root,
                    )
            except BaseException as exc:  # reported by the test, never swallowed
                self.error = exc

        self._thread = threading.Thread(target=work, daemon=True)
        self._thread.start()

    def finish(self) -> dict[str, Any]:
        assert self._thread is not None
        self._thread.join(timeout=60)
        assert not self._thread.is_alive(), "the run did not end within 60 s"
        if self.error is not None:
            raise self.error
        assert self.result is not None
        return self.result

    def stop_and_finish(self) -> None:
        """Teardown for a test that did not end its run: cancel it and wait."""
        self.client.post(f"/api/v1/runs/{RUN}/cancel")
        if self._thread is not None:
            self._thread.join(timeout=60)

    # ---------------------------------------------------------------- reading
    def row(self) -> Any:
        table = Run.__table__
        with self.scope() as session:
            return session.execute(
                sa.select(
                    table.c.status,
                    table.c.last_step,
                    table.c.heartbeat_at,
                    table.c.cancel_requested_at,
                    table.c.stop_reason,
                ).where(table.c.id == RUN)
            ).one()

    def visible_steps(self) -> int:
        with self.scope() as session:
            return int(
                session.execute(
                    sa.select(sa.func.count()).select_from(Step).where(Step.run_id == RUN)
                ).scalar_one()
            )

    def cancel(self) -> tuple[Any, float]:
        started = time.monotonic()
        response = self.client.post(f"/api/v1/runs/{RUN}/cancel")
        return response, time.monotonic() - started


def wait_for(predicate: Callable[[], bool], *, timeout: float, what: str) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return
        time.sleep(0.05)
    raise AssertionError(f"timed out after {timeout} s waiting for {what}")


def cancel_while_held(rig: Rig, gate: Gate) -> tuple[Any, float]:
    """Send the cancel while ``gate`` holds its phase, releasing the phase on a timer.

    The timer is what keeps a regression from DEADLOCKING the test instead of failing it: if
    the worker holds the run row while it waits on the gate, the cancel waits on the worker
    and the worker waits on the gate - so the gate is let go after the bound and the
    cancel's elapsed time says what happened.
    """
    releaser = threading.Timer(CANCEL_BOUND_S + 1.0, gate.release.set)
    releaser.start()
    try:
        return rig.cancel()
    finally:
        releaser.cancel()
        gate.release.set()


def assert_accepted_promptly(response: Any, elapsed: float, rig: Rig) -> None:
    assert response.status_code == 202, response.text
    body = response.json()
    assert elapsed < CANCEL_BOUND_S, f"the cancel route took {elapsed:.2f} s (a lock wait)"
    assert body["cancel_pending"] is True
    assert body["cancel_requested_at"] is not None, body
    # The committed value another connection reads, not a value the route made up.
    assert rig.row().cancel_requested_at is not None


@pytest.fixture
def make_rig(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Iterator[Callable[..., Rig]]:
    built: list[Rig] = []

    def build(mode: str, *, beat_s: int = PRODUCTION_BEAT_S, save_every: int = 100000) -> Rig:
        configure(
            monkeypatch,
            CHECKPOINT_ROOT=str(tmp_path / "checkpoints"),
            TRAINER_STATE_ROOT=str(tmp_path / "trainer-state"),
            CHECKPOINT_READ_ROOTS=str(tmp_path / "millm-checkpoints"),
            HEARTBEAT_MIN_INTERVAL_SECONDS=str(beat_s),
        )
        engine = migrated_engine(tmp_path / "db")
        factory = sessionmaker(bind=engine, expire_on_commit=False)

        @contextmanager
        def scope() -> Iterator[Session]:
            """``db.session.session_scope``'s shape over THIS engine: a new connection each."""
            session = factory()
            try:
                yield session
                session.commit()
            except BaseException:
                session.rollback()
                raise
            finally:
                session.close()

        stubs = {
            "gate": lambda *a, **k: _PackDouble(),
            "resolve_parts": lambda *a, **k: {"source": "parts", "parts": []},
            "services": lambda *a, **k: {"needed": []},
        }
        for name, fn in (
            ("_live_gate", stubs["gate"]),
            ("_live_parts", stubs["resolve_parts"]),
            ("_live_services", stubs["services"]),
        ):
            monkeypatch.setattr(validator, name, fn)

        model = _fixed_init_model()
        surface: FakeWeightsSurface | None = None
        acquirer: Any = None
        holder: dict[str, Rig] = {}
        if mode == "attach":
            from training.hotupdate import attach as attach_module
            from training.hotupdate import rebuild
            from training.hotupdate import save as save_module

            surface = FakeWeightsSurface()
            acquirer = attach_module.AttachAcquirer(transport=surface.transport())
            # THROUGH THE REGISTRY: `acquire_for_run` resolves the attach mode by name.
            monkeypatch.setitem(acquisition._REGISTRY, AcquisitionMode.ATTACH, acquirer)
            monkeypatch.setattr(rebuild, "build", lambda response, **kw: _Base())
            monkeypatch.setattr(rebuild, "check_build_identity", lambda response: None)
            monkeypatch.setattr(attach_module, "build_module", lambda response, base: model)
            monkeypatch.setattr(
                attach_module.AttachAcquirer,
                "_tokenizer_loader",
                lambda self, resident, live, *directory: (lambda: None),
            )
            monkeypatch.setattr(attach_module.AttachAcquirer, "_verifier", lambda self, *a: None)
            real_save = save_module.AttachedSaver.__call__

            def held_save(self: Any, **kwargs: Any) -> Any:
                step = int(kwargs["fields"].step)
                holder["rig"].on_save(step)
                outcome = real_save(self, **kwargs)  # the real save, beat touch included
                holder["rig"].after_save(step)
                return outcome

            monkeypatch.setattr(save_module.AttachedSaver, "__call__", held_save)
        else:
            from training.runtypes import change as change_module

            record = AcquisitionRecord(
                mode=AcquisitionMode.LOAD,
                source_kind=SourceKind.LOCAL_DIR,
                source_path="/models/tiny",
                starting_weights_sha256="a" * 64,
                base_model="local:tiny@abcdef0123456789",
                parent=None,
                param_count=1000,
            )
            monkeypatch.setattr(
                tasks, "acquire_for_run", lambda spec: AcquiredModel(model, None, record)
            )
            monkeypatch.setattr(tasks, "verify_checkpoint_source", lambda spec, **kw: None)

            def f5_save(self: Any, ctx: Any) -> None:
                holder["rig"].on_save(int(ctx.step))
                # F5's service ends with exactly this (services/checkpoints/service.py).
                if ctx.heartbeat is not None:
                    ctx.heartbeat.touch(ctx.session)
                holder["rig"].after_save(int(ctx.step))

            monkeypatch.setattr(change_module.ChangeRunType, "save", f5_save)

        from schemas.run_definition import RunDefinition
        from services import run_launch

        with scope() as session:
            run_launch.launch(
                RunDefinition(**_definition(mode, save_every)),
                session=session,
                enqueue=lambda run_id: None,
                budget=lambda *a, **k: {"param_count": 10, "gpu_uuid": None, "mode": mode},
                **stubs,
            )

        app = create_app()

        def request_session() -> Iterator[Session]:
            with scope() as session:
                yield session

        app.dependency_overrides[session_dependency] = request_session
        rig = Rig(
            engine=engine,
            scope=scope,
            client=TestClient(app, raise_server_exceptions=False),
            log_root=tmp_path / "runs",
            surface=surface,
            acquirer=acquirer,
        )
        holder["rig"] = rig
        built.append(rig)
        return rig

    yield build
    for rig in built:
        for gate in (rig.hold_evaluation, rig.source.hold_after_save):
            if gate is not None:
                gate.release.set()
        if rig.hold_save_at is not None:
            rig.hold_save_at[1].release.set()
        if rig.acquirer is not None and rig.acquirer.heartbeat is not None:
            rig.acquirer.heartbeat.stop()
        rig.engine.dispose()
    reset_settings()


# =========================================================================== visibility


@pytest.mark.parametrize("mode", MODES)
def test_last_step_and_step_rows_advance_on_another_connection_during_the_run(
    make_rig: Callable[..., Rig], mode: str
) -> None:
    """At production's 60 s throttle - no beat lands mid-test, so only per-step commits show."""
    rig = make_rig(mode)
    rig.start()
    try:
        wait_for(lambda: (rig.row().last_step or 0) >= 5, timeout=20, what="last_step 5")
        first = rig.row()
        assert first.status == "running"
        assert first.heartbeat_at is not None
        assert rig.visible_steps() >= 5, "step rows are not visible mid-run"
        wait_for(
            lambda: (rig.row().last_step or 0) > first.last_step + 5,
            timeout=20,
            what="last_step to keep advancing",
        )
        assert rig.row().status == "running"
    finally:
        rig.stop_and_finish()


@pytest.mark.parametrize("mode", MODES)
def test_the_beat_is_visible_mid_run_when_a_save_wins_the_throttle(
    make_rig: Callable[..., Rig], mode: str
) -> None:
    """The live shape: a save every step that outlasts the step, so the SAVE's touch is the one
    that writes the beat - and it must be committed, not left in the run's transaction."""
    rig = make_rig(mode, beat_s=1, save_every=1)
    rig.save_delay_s = 0.3
    rig.start()
    try:
        wait_for(lambda: len(rig.saves) >= 1, timeout=20, what="the first save")
        seen = {rig.row().heartbeat_at}
        deadline = time.monotonic() + 8
        while time.monotonic() < deadline and len(seen) < 4:
            seen.add(rig.row().heartbeat_at)
            time.sleep(0.1)
        assert len(seen) >= 4, f"the beat moved {len(seen) - 1} times in 8 s: {sorted(seen)}"
    finally:
        rig.stop_and_finish()


@pytest.mark.parametrize("mode", MODES)
def test_the_janitor_does_not_reap_a_long_healthy_run(
    make_rig: Callable[..., Rig], mode: str
) -> None:
    """Sweeping past the lapse threshold (2 s here, 900 s live) while saves win the throttle."""
    lapse = 2
    rig = make_rig(mode, beat_s=1, save_every=1)
    rig.save_delay_s = 0.3
    rig.start()
    try:
        wait_for(lambda: len(rig.saves) >= 1, timeout=20, what="the first save")
        started = time.monotonic()
        report = None
        while time.monotonic() - started < lapse + 2.5:
            with rig.scope() as session:
                report = janitor.sweep(session, lapse_seconds=lapse)
            assert RUN not in report.reaped, "the janitor reaped a healthy run"
            time.sleep(0.25)
        assert report is not None and RUN in report.alive
        assert rig.row().status == "running"
    finally:
        rig.stop_and_finish()


@pytest.mark.parametrize("mode", MODES)
def test_the_loop_start_beat_is_visible_on_another_connection_before_the_first_step(
    make_rig: Callable[..., Rig], monkeypatch: pytest.MonkeyPatch, mode: str
) -> None:
    """F11 18.5c: the loop's start beat is COMMITTED before the step-0 baseline, so a reader on
    another connection sees it - and the longest gap the worker stores is the gap readers saw.

    Window 6 recorded the start beat as "not visible" (a 79.3 s visible gap against a stored 72 s).
    Its own monitor says otherwise: the API on other connections read the loop-start beat
    (13:07:03.464741Z) from the reading at 13:07:04.8 to the one at 13:08:14.8, and the stored
    72 s is that beat to the step-100 save's (13:08:15.907Z), 72.44 s rounded. 79.3 s was measured
    from the DRIVER's earlier beat (13:06:56.634Z). This pins what was observed: the beat the
    loop writes at its start is the one another session reads while the baseline runs, on the
    attach path as on the load path, and the stored gap equals the visible one.
    """
    from workers import heartbeat as heartbeat_module
    from workers.run_record import as_aware

    writes: list[Any] = []
    real_touch = heartbeat_module.Heartbeat.touch

    def recording_touch(self: Any, session: Any, run_id: str | None = None, **kw: Any) -> bool:
        wrote = real_touch(self, session, run_id, **kw)
        if wrote:
            writes.append(self.last_write_at)
        return wrote

    monkeypatch.setattr(heartbeat_module.Heartbeat, "touch", recording_touch)
    rig = make_rig(mode)
    rig.hold_evaluation = Gate()
    rig.start()
    try:
        assert rig.hold_evaluation.entered.wait(20), "the baseline was never reached"
        # Inside the step-0 baseline, before any step: the driver's beat, then the loop's.
        assert len(writes) == 2, writes
        driver_beat, loop_beat = writes
        assert loop_beat > driver_beat
        seen = rig.row()
        assert seen.status == "running"
        assert rig.source.handed_at == [], "a step started before the beat was read"
        # ANOTHER connection reads the loop's beat - not the driver's earlier one.
        assert as_aware(seen.heartbeat_at) == loop_beat, (seen.heartbeat_at, writes)
        with rig.scope() as session:
            stored_gap = session.execute(
                sa.select(Run.__table__.c.longest_heartbeat_gap_seconds).where(
                    Run.__table__.c.id == RUN
                )
            ).scalar_one()
        assert stored_gap == round((loop_beat - driver_beat).total_seconds())
    finally:
        rig.hold_evaluation.release.set()
        rig.stop_and_finish()


# =========================================================================== the cancel


@pytest.mark.parametrize("mode", MODES)
def test_a_cancel_mid_run_is_accepted_promptly_and_the_run_stops_within_one_step(
    make_rig: Callable[..., Rig], mode: str
) -> None:
    rig = make_rig(mode)
    rig.start()
    wait_for(lambda: len(rig.source.handed_at) >= 8, timeout=20, what="eight batches")
    if rig.surface is not None:
        assert rig.surface.call_count("detach") == 0

    response, elapsed = rig.cancel()
    batches_at_cancel = len(rig.source.handed_at)
    assert_accepted_promptly(response, elapsed, rig)

    result = rig.finish()
    assert result["status"] == "completed"
    assert result["stop_reason"] == "operator_stop"
    # WITHIN ONE STEP: the step in flight when the request landed may finish (a step is never
    # interrupted, R-02.31), and the next step's boundary check honours the stop.
    assert len(rig.source.handed_at) - batches_at_cancel <= 1, (
        len(rig.source.handed_at),
        batches_at_cancel,
    )
    assert result["last_step"] < TOTAL
    row = rig.row()
    assert (row.status, row.stop_reason, row.last_step) == (
        "completed",
        "operator_stop",
        result["last_step"],
    )
    if rig.surface is not None:
        # DETACHED, ONCE, carrying the attachment it held.
        assert rig.surface.call_count("detach") == 1, rig.surface.ops()[-10:]
        [headers] = rig.surface.headers_of("detach")
        assert "attachment-0001" in headers.values(), headers
        assert rig.surface.attachment_id is None


@pytest.mark.parametrize("mode", MODES)
def test_a_cancel_during_the_step_zero_baseline_lands(
    make_rig: Callable[..., Rig], mode: str
) -> None:
    """Pins the loop's start commit: the start beat (and, attached, the attachment record)
    would otherwise hold the row through a baseline that can take minutes."""
    rig = make_rig(mode)
    rig.hold_evaluation = Gate()
    rig.start()
    assert rig.hold_evaluation.entered.wait(20), "the baseline was never reached"
    response, elapsed = cancel_while_held(rig, rig.hold_evaluation)
    assert_accepted_promptly(response, elapsed, rig)
    result = rig.finish()
    assert result["stop_reason"] == "operator_stop"
    assert rig.source.handed_at == [], "a step started after the stop was recorded"


@pytest.mark.parametrize("mode", MODES)
def test_a_cancel_during_a_save_lands(make_rig: Callable[..., Rig], mode: str) -> None:
    """Pins the per-step commit: the step's own row writes (``last_step``) must not be held
    through the save that follows it in the same boundary."""
    rig = make_rig(mode, save_every=3)
    gate = Gate()
    rig.hold_save_at = (3, gate)
    rig.start()
    assert gate.entered.wait(20), "the save at step 3 was never reached"
    response, elapsed = cancel_while_held(rig, gate)
    assert_accepted_promptly(response, elapsed, rig)
    result = rig.finish()
    assert result["stop_reason"] == "operator_stop"
    assert result["last_step"] <= 4


@pytest.mark.parametrize("mode", MODES)
def test_a_cancel_during_the_step_after_a_save_lands(
    make_rig: Callable[..., Rig], mode: str
) -> None:
    """Pins the end-of-boundary commit: a save's beat (written here, because the save outlasts
    the 1 s throttle) must not be held through the next step."""
    rig = make_rig(mode, beat_s=1, save_every=3)
    rig.save_delay_s = 1.2
    gate = Gate()
    rig.source.hold_after_save = gate
    rig.start()
    assert gate.entered.wait(20), "the step after the save was never reached"
    response, elapsed = cancel_while_held(rig, gate)
    assert_accepted_promptly(response, elapsed, rig)
    assert rig.finish()["stop_reason"] == "operator_stop"


def test_a_cancel_after_the_run_finished_is_refused_not_accepted(
    make_rig: Callable[..., Rig],
) -> None:
    rig = make_rig("attach")
    rig.start()
    wait_for(lambda: len(rig.source.handed_at) >= 2, timeout=20, what="two batches")
    rig.cancel()
    rig.finish()
    again = rig.client.post(f"/api/v1/runs/{RUN}/cancel")
    assert again.status_code == 400, again.text
    assert again.json()["error"]["code"] == "run_not_stoppable"


# ====================================================== the route against a held row lock


def _postgres(rig: Rig) -> None:
    if rig.engine.dialect.name != "postgresql":
        pytest.skip(POSTGRES_ONLY)


#: Copy that names a document, a requirement or a feature instead of telling the operator
#: something. The stop route's refusals are shown on the run view as they are; the first cut
#: of ``run_row_busy`` said "(F11 18.5)" to the operator. Mirrors ``test_new_run_form_copy``.
_JARGON = re.compile(
    r"\bF\d{1,2}\b|\bR-\d{2}\.\d+|\bFR-\d{3}\.\d+|\b(?:ADR|BRD|PRD|TDD|DEF|AMB)-\d+|§\s?\d|\*\*"
)


def _assert_plain_copy(error: dict[str, Any]) -> None:
    match = _JARGON.search(error["message"])
    assert match is None, f"{match.group(0)!r} in operator copy: {error['message']!r}"


def _hold_row(rig: Rig, **values: Any) -> Any:
    """Another transaction's uncommitted UPDATE of the run row - the lock the worker held."""
    connection = rig.engine.connect()
    transaction = connection.begin()
    connection.execute(sa.update(Run.__table__).where(Run.__table__.c.id == RUN).values(**values))
    return connection, transaction


def test_a_stop_whose_update_finds_the_run_finished_is_refused_not_202_null(
    make_rig: Callable[..., Rig],
) -> None:
    """Window 5's answer: the UPDATE waited, the run finished meanwhile, the predicate matched
    no row - and the route said 202 ``cancel_pending: true`` with ``cancel_requested_at: null``.
    """
    rig = make_rig("attach")
    _postgres(rig)
    from workers.run_record import utcnow

    connection, transaction = _hold_row(rig, status="completed", finished_at=utcnow())
    answer: dict[str, Any] = {}
    sender = threading.Thread(target=lambda: answer.update(response=rig.cancel()[0]))
    sender.start()
    time.sleep(0.5)  # the request is now waiting on the held row
    transaction.commit()
    connection.close()
    sender.join(timeout=20)
    response = answer["response"]
    assert response.status_code == 400, response.text
    error = response.json()["error"]
    assert error["code"] == "run_not_stoppable"
    assert error["details"]["status"] == "completed"
    _assert_plain_copy(error)
    assert rig.row().cancel_requested_at is None


def test_a_stop_held_past_its_lock_bound_is_refused_503_not_left_hanging(
    make_rig: Callable[..., Rig],
) -> None:
    from services import stop_service

    rig = make_rig("attach")
    _postgres(rig)
    connection, transaction = _hold_row(rig, last_step=1)
    releaser = threading.Timer(stop_service.LOCK_WAIT_SECONDS + 4, transaction.rollback)
    releaser.start()
    try:
        response, elapsed = rig.cancel()
    finally:
        releaser.cancel()
        if transaction.is_active:
            transaction.rollback()
        connection.close()
    assert elapsed < stop_service.LOCK_WAIT_SECONDS + 2, elapsed
    assert response.status_code == 503, response.text
    assert response.json()["error"]["code"] == "run_row_busy"
    _assert_plain_copy(response.json()["error"])
    assert rig.row().cancel_requested_at is None
