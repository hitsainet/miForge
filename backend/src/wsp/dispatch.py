"""The WSP driver: one pass of ``miforge.wsp.drive``, and the push of a claimed run (feature 016;
FTDD §7; FR-016.11; P41, P60; feature 019 P3: FR-019.6-FR-019.9, F19-D17-D19, D33, D35).

A pass holds a single-flight lock for its whole length (a PostgreSQL advisory transaction lock; on
SQLite, where every test runs in one process, a process lock), so two beats never claim twice. It
never pushes: since 019 P3 a claimed run is pushed by its own Celery task (``wsp.tasks.push`` ->
:func:`push_claimed`), outside the lock, so a long push never holds back another worker's poll and
result (F19-D19). Per pass, in this order:

0. **lapsed pushes** (F19-D19): a ``pushing`` run whose push lease (``push_heartbeat_at``) is older
   than ``WSP_PUSH_LEASE_S``, or was never written, is failed ``worker_push_failed``;
1. **every provisioned worker** (``WorkerSlot``): one ``GET /ready``, recorded for the pool route
   (``wsp.pool``); a worker showing a consumed boot that no ``pushing``, ``running`` or
   ``collecting`` run owns is freed: ``idle`` gets ``POST /recycle``, ``finished`` is pulled,
   discarded unread and acked (OVN-wsp-forge-24); an idle, unconsumed boot within
   ``RETIRE_BEFORE_S`` of its supervisor's idle recycle is consumed first, so it is recycled and
   its pod replaced the same way and never claimed (2026-11-26);
2. **every ``running`` run**, on ITS worker (``worker_index``; null before 019, read as 0):
   ``GET /status``; another ``boot_id`` fails it ``worker_lost``; on ``finished``: ``GET /result``
   -> intake -> ``POST /ack`` (an intake failure still acks); unreachable past
   ``WSP_HEARTBEAT_LAPSE_S`` -> ``worker_lost``; past budget + grace, or answering but unfinished
   past the worker's own lifetime -> ``time_budget``. Every clock runs from ``pushed_at``;
3. **claims**: while fewer workers are held than the effective pool (``wsp_pool_size``, at most
   the maximum) and no run is ``pushing`` (F19-D35), the oldest queued run whose scheduling key is
   not active passes **every refusal check** (the sensitivity read for an external endpoint, the
   bundle's import status, the push's size against ``/work``, the endpoint fingerprint, the
   protocol-test pass) and is claimed on the lowest idle worker with a fresh boot, in ONE
   transaction (boot row, key, worker index, lease, ``pushing``); the push task is enqueued after
   the commit, carrying the run id alone. A refusal leaves the boot unconsumed. A run whose
   miServer endpoint states no window is claimed only once miServer reports one (a model is
   resident, ``window_read.awaits_resident``): until then it stays queued, the next run is
   considered, and after ``wsp_editor_resident_wait_s`` it fails ``editor_not_resident``
   (2026-11-27, run ``wsp_4fbf5a18758e``).
"""

from __future__ import annotations

import datetime as dt
import hashlib
import json
import logging
import math
import threading
from collections.abc import Callable, Iterator, Sequence
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol

import sqlalchemy as sa
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session, sessionmaker

from core.errors import Refusal
from wsp import capacity, endpoints, pool, runs, scheduling, tokens, window_read
from wsp.models import (
    DEFAULT_ACTION_PROTOCOL,
    DEFAULT_HISTORY_MODE,
    HOLDING_STATES,
    WORKER_STATES,
    WspBundle,
    WspEndpoint,
    WspRun,
    WspWorker,
    WspWorkerBoot,
)
from wsp.pod_recycler import PodRecycler
from wsp.protocol_test import BUILTIN_OWNER
from wsp.settings import WSPSettings

log = logging.getLogger(__name__)

#: ``hashtext('miforge.wsp.drive')`` would do; a literal keeps it obvious and stable.
ADVISORY_KEY = 0x57535044  # "WSPD"
_LOCAL_LOCK = threading.Lock()
#: How long before its supervisor's own idle recycle (``idle_recycle_in_s`` in ``/ready``) an idle,
#: unconsumed boot is retired by the driver (2026-11-26, the residue of the 019 P4 restart debt).
#: Retired, it is consumed with no run, so nothing is claimed on a boot about to end (a push that
#: met the next boot failed ``worker_push_failed``) and the pool reads it ``restarting``; then the
#: stale-boot rule sends ``/recycle`` and replaces its pod, as after an ack. A claim therefore
#: leaves its push this long to be accepted. Well above the beat's 5 s pass, and a small part of
#: the hour (``test_idle_boot_retired_by_the_driver.py``).
RETIRE_BEFORE_S = 300.0


class WorkerUnreachableError(Exception):
    """The worker did not answer (connect, read or protocol failure)."""


class Worker(Protocol):
    def ready(self) -> dict[str, Any]: ...
    def push(
        self,
        document: dict[str, Any],
        files: list[tuple[str, Path]],
        *,
        on_progress: Callable[[], None] | None = None,
    ) -> dict[str, Any]: ...
    def status(self) -> dict[str, Any]: ...
    def result(self) -> bytes: ...
    def ack(self, boot_id: str, run_id: str, result_sha256: str) -> None: ...
    def recycle(self, boot_id: str) -> None: ...


SensitivityReader = Callable[[WspRun, WspBundle], str]
#: The window the endpoint's server reports now, or ``None`` (feature 019, FR-019.1;
#: ``wsp.window_read.read_window_answer`` in production, passed by ``wsp.tasks.push`` for the push
#: and, since 2026-11-27, by ``wsp.tasks.drive`` for the claim's read; this comment named only
#: ``drive`` while only ``push`` passed it). Since D1 it may answer ``NOT_ANSWERED``.
WindowReader = Callable[[WspEndpoint], "window_read.WindowAnswer"]


def no_window(endpoint: WspEndpoint) -> int | None:
    """The default reader: nothing is read, so every run is ``assumed`` unless it states one."""
    return None


#: A queued run on a miServer endpoint that states no window, failed when miServer reported no
#: window (no model resident) for ``wsp_editor_resident_wait_s`` (2026-11-27).
EDITOR_NOT_RESIDENT = "editor_not_resident"


def not_resident_message(
    endpoint: WspEndpoint, waited_s: float, *, unanswered: bool = False
) -> str:
    """``editor_not_resident``'s message, in the suite's error layout: what happened, then
    ``fix:``, then ``next:`` (a JSON list of ``{tool, args, why}``). ``unanswered`` (D1): the
    server gave no answer at all to the window read."""
    if unanswered:
        return _unanswered_message(endpoint, waited_s)
    steps = [
        {
            "tool": "miserver_load_model",
            "args": {},
            "why": f"load {endpoint.model} in miServer (the operator approves the load there)",
        },
        {
            "tool": "forge_wsp_list_endpoints",
            "args": {},
            "why": f"check that endpoint {endpoint.id} now expects a reported window (window_now)",
        },
        {"tool": "forge_wsp_start_run", "args": {}, "why": "start the run again"},
    ]
    return (
        f"miServer reported no context window for endpoint {endpoint.id} for {int(waited_s)} s "
        "(GET /api/health/inference carried no context_length): no model was resident, miServer "
        "did not answer, or the resident model's config states no window. miForge does not guess "
        "a window for a miServer endpoint, so the run was never pushed and nothing was spent.\n"
        f"fix: load {endpoint.model} in miServer and start the run again; if it is resident and "
        "its config states no window, set the endpoint's context window (context_window_tokens) "
        "and start the run again.\n"
        f"next: {json.dumps(steps, separators=(',', ':'))}"
    )


def _unanswered_message(endpoint: WspEndpoint, waited_s: float) -> str:
    """D1 (operator, 2026-11-27): a server that gave no answer to the window read for the wait."""
    where = window_read.origin(endpoint.base_url)
    steps = [
        {
            "tool": "forge_wsp_list_endpoints",
            "args": {},
            "why": f"check endpoint {endpoint.id}'s base URL ({where}) and its window",
        },
        {"tool": "forge_wsp_start_run", "args": {}, "why": "start the run again"},
    ]
    return (
        f"The editor's server for endpoint {endpoint.id} ({where}) gave no answer to the window "
        f"read for {int(waited_s)} s (connection refused, a timeout or a 5xx on every request): "
        "it is not up yet, or miForge's backend cannot reach it. miForge does not guess a window "
        "for a server that does not answer, so the run was never pushed and nothing was spent.\n"
        "fix: start the server (or correct the endpoint's base URL) and start the run again; if "
        "it serves but cannot be read from the backend, set the endpoint's context window "
        "(context_window_tokens) and start the run again.\n"
        f"next: {json.dumps(steps, separators=(',', ':'))}"
    )


#: Intake refusals that are the run's own failure code, not a malformed result.
INTAKE_RUN_CODES = ("wsp_inspect_changed_weights", "wsp_change_limit_exceeded")
#: What every intake refusal tells the operator to do (review OR3-016-5).
INTAKE_NEXT_STEP = (
    "Start the run again; if it is refused the same way, the editor (or its program) is writing a "
    "damaged or disallowed result, so change the task or the editor endpoint."
)


@dataclass
class Pass:
    """What one pass did, for the task's log line and the tests."""

    actions: list[str] = field(default_factory=list)

    def say(self, action: str) -> None:
        self.actions.append(action)


def utcnow() -> dt.datetime:
    return dt.datetime.now(dt.UTC)


def _aware(value: dt.datetime | None) -> dt.datetime | None:
    if value is not None and value.tzinfo is None:
        return value.replace(tzinfo=dt.UTC)
    return value


@contextmanager
def single_flight(factory: sessionmaker[Session]) -> Iterator[bool]:
    """``True`` when this pass holds the driver's lock for its whole length."""
    session = factory()
    try:
        if session.get_bind().dialect.name == "postgresql":
            held = bool(
                session.execute(
                    sa.text("SELECT pg_try_advisory_xact_lock(:k)"), {"k": ADVISORY_KEY}
                ).scalar()
            )
            yield held
            return
        held = _LOCAL_LOCK.acquire(blocking=False)
        try:
            yield held
        finally:
            if held:
                _LOCAL_LOCK.release()
    finally:
        session.rollback()  # releases the transaction-scoped advisory lock
        session.close()


def _fail(
    session: Session, run: WspRun, frm: str | tuple[str, ...], code: str, message: str, **extra: Any
) -> bool:
    stop = extra.pop("stop_reason", "worker_lost" if code == "worker_lost" else "error")
    if code == "time_budget":
        stop = "time_budget"
    return runs.transition(
        session,
        run.run_id,
        frm,
        "failed",
        outcome="failed",
        stop_reason=stop,
        failure={"code": code, "message": message},
        **extra,
    )


# ----------------------------------------------------------------------------- the pass


@dataclass(frozen=True)
class WorkerSlot:
    """One provisioned worker: its index (``0..max-1``), its address and its client (FR-019.7)."""

    index: int
    address: str
    client: Worker


@dataclass
class _Seen:
    """What one pass read from one worker's ``/ready``."""

    slot: WorkerSlot
    ready: dict[str, Any] | None
    boot: str | None
    consumed: bool


def drive(
    factory: sessionmaker[Session],
    workers: Sequence[WorkerSlot],
    settings: WSPSettings,
    *,
    pool_size: int = 1,
    read_sensitivity: SensitivityReader,
    files_for: Callable[[WspRun, WspBundle], list[tuple[str, Path]]],
    enqueue_push: Callable[[str], None],
    now: Callable[[], dt.datetime] = utcnow,
    recycle_pod: PodRecycler | None = None,
    read_window: WindowReader = no_window,
    backoff: window_read.NoAnswerBackoff | None = None,
) -> Pass:
    """One pass of the pool (feature 019 P3, FR-019.6-FR-019.9; TID §3.8). It only CLAIMS: the
    push runs in its own task (``push_claimed``, ``wsp.tasks.push``), outside the lock, so a long
    push never holds back another run's poll or result (F19-D19).

    ``recycle_pod`` (operator 2026-11-26, a fresh pod per session): after every ack or
    ``/recycle`` that ends a worker's boot, the worker's pod is replaced (``wsp.pod_recycler``).
    ``None`` replaces nothing, and the container restarts as before.

    ``read_window`` (2026-11-27): the claim's residency read for a miServer endpoint that states
    no window (``wsp.window_read.read_window`` in production, passed by ``wsp.tasks.drive``); the
    default reads nothing, so such a run waits and then fails ``editor_not_resident``.

    ``backoff`` (debt D5, 2026-11-28): the claim's hold on an endpoint whose server gave no answer
    (``wsp.tasks.CLAIM_BACKOFF`` in production); ``None`` reads at every pass."""
    done = Pass()
    with single_flight(factory) as held:
        if not held:
            done.say("lock_busy")
            return done
        _lapsed_pushes(factory, settings, now, done)
        seen = [_read_worker(factory, slot, now) for slot in workers]
        _forget_unprovisioned(factory, len(workers))
        for view in seen:
            _retire_idle_boot(factory, view, now, done)
            _release_stale_boot(factory, view, done, recycle_pod)
        by_index = {slot.index: slot for slot in workers}
        with factory() as session:
            running = session.scalars(
                sa.select(WspRun).where(WspRun.status == "running").order_by(WspRun.pushed_at)
            ).all()
            for run in running:
                slot = by_index.get(run.worker_index or 0)
                if slot is None:
                    continue  # its worker is no longer provisioned; the lapse rule ends it
                acked = _poll(session, slot.client, run, settings, now, done)
                session.commit()
                if acked:
                    _replace_pod(session, recycle_pod, slot.index, run.boot_id, done)
        _claim(
            factory,
            seen,
            settings,
            scheduling.effective_pool(pool_size, len(workers)),
            read_sensitivity,
            files_for,
            enqueue_push,
            now,
            done,
            read_window,
            backoff,
        )
    return done


def _lapsed_pushes(
    factory: sessionmaker[Session],
    settings: WSPSettings,
    now: Callable[[], dt.datetime],
    done: Pass,
) -> None:
    """F19-D19: a ``pushing`` run whose push lease lapsed (no progress for ``wsp_push_lease_s``,
    or none ever written) is failed ``worker_push_failed`` and never pushed again."""
    current = now()
    lease = dt.timedelta(seconds=settings.wsp_push_lease_s)
    with factory() as session:
        for run in session.scalars(sa.select(WspRun).where(WspRun.status == "pushing")).all():
            beat = _aware(run.push_heartbeat_at)
            if beat is not None and current - beat <= lease:
                continue
            if _fail(
                session,
                run,
                "pushing",
                "worker_push_failed",
                f"The push stopped reporting progress for {settings.wsp_push_lease_s} s; the run "
                "is never pushed again.",
            ):
                log.warning(
                    "wsp push lease lapsed", extra={"code": "wsp_push_lapsed", "run_id": run.run_id}
                )
                done.say(f"orphan_failed:{run.run_id}")
        session.commit()


def _read_worker(
    factory: sessionmaker[Session], slot: WorkerSlot, now: Callable[[], dt.datetime]
) -> _Seen:
    """One ``/ready`` per worker per pass, recorded for the pool route (``pool.record_ready``)."""
    try:
        ready: dict[str, Any] | None = slot.client.ready()
    except WorkerUnreachableError:
        ready = None
    with factory() as session:
        pool.record_ready(session, slot.index, slot.address, ready, now())
        boot = None if ready is None else ready.get("boot_id")
        boot = boot if isinstance(boot, str) else None
        consumed = boot is not None and session.get(WspWorkerBoot, boot) is not None
        session.commit()
    return _Seen(slot, ready, boot, consumed)


def _forget_unprovisioned(factory: sessionmaker[Session], provisioned: int) -> None:
    """A worker row at or above the maximum is deleted, so a lowered maximum leaves no ghost."""
    with factory() as session:
        session.execute(sa.delete(WspWorker).where(WspWorker.worker_index >= provisioned))
        session.commit()


def _replace_pod(
    session: Session,
    recycle_pod: PodRecycler | None,
    index: int,
    boot: str | None,
    done: Pass,
) -> None:
    """Replace worker ``index``'s pod once its boot has ended (operator 2026-11-26). The pod that
    served ``boot`` existed when the boot was first seen, so that time tells the recycler which pod
    is the old one. Never raises: a refusal is the recycler's log line and an outcome word."""
    if recycle_pod is None or boot is None:
        return
    row = session.get(WspWorkerBoot, boot)
    served_since = utcnow() if row is None else row.first_seen_at
    try:
        outcome = recycle_pod.recycle(index, served_since)
    except Exception as exc:  # the protocol says it never raises; a run never fails on it
        log.warning(
            "wsp worker pod not replaced; its container restarts instead",
            extra={"code": "wsp_worker_pod_delete_failed", "worker_index": index},
            exc_info=True,
        )
        outcome = f"refused:{type(exc).__name__}"
    done.say(f"pod_replaced:{index}:{outcome}")


def _retiring(ready: dict[str, Any] | None) -> bool:
    """An idle worker whose supervisor will end its boot within :data:`RETIRE_BEFORE_S`. The worker
    is untrusted and may predate the reading: anything but a finite number is no reading."""
    if ready is None or ready.get("state") != "idle":
        return False
    left = ready.get("idle_recycle_in_s")
    if isinstance(left, bool) or not isinstance(left, int | float) or not math.isfinite(left):
        return False
    return left <= RETIRE_BEFORE_S


def _retire_idle_boot(
    factory: sessionmaker[Session],
    view: _Seen,
    now: Callable[[], dt.datetime],
    done: Pass,
) -> None:
    """Consume an idle, unconsumed boot near its idle recycle (:data:`RETIRE_BEFORE_S`), with no
    run, and mark the view consumed: the claim then passes it over and :func:`_release_stale_boot`
    recycles it and replaces its pod in this same pass. Consumed first, so a lost ``/recycle``
    is retried by the stale-boot rule on the next pass."""
    if view.consumed or view.boot is None or not _retiring(view.ready):
        return
    with factory() as session:
        session.add(WspWorkerBoot(boot_id=view.boot, first_seen_at=now(), run_id=None))
        try:
            session.commit()
        except IntegrityError:  # consumed meanwhile: it is just as retired
            session.rollback()
    view.consumed = True
    done.say(f"retired:{view.boot}")


def _release_stale_boot(
    factory: sessionmaker[Session],
    view: _Seen,
    done: Pass,
    recycle_pod: PodRecycler | None = None,
) -> None:
    """A worker showing a CONSUMED boot that no live run owns is freed: ``idle`` gets ``/recycle``,
    ``finished`` is pulled, discarded and acked (OVN-wsp-forge-24). ``pushing`` counts as live
    (TP-1): between its claim and its push a claimed worker is idle on a consumed boot."""
    if view.ready is None or not view.consumed or view.boot is None:
        return
    worker, boot = view.slot.client, view.boot
    with factory() as session:
        live = session.scalar(
            sa.select(WspRun).where(WspRun.boot_id == boot, WspRun.status.in_(HOLDING_STATES))
        )
        if live is not None:
            return
        state = view.ready.get("state")
        try:
            if state == "idle":
                worker.recycle(boot)
                done.say(f"recycled:{boot}")
            elif state == "finished":
                data = worker.result()
                owner = session.scalar(sa.select(WspRun).where(WspRun.boot_id == boot))
                worker.ack(boot, "" if owner is None else owner.run_id, _sha(data))
                done.say(f"discarded:{boot}")
            else:
                return
        except WorkerUnreachableError:
            return
        _replace_pod(session, recycle_pod, view.slot.index, boot, done)


def _poll(
    session: Session,
    worker: Worker,
    run: WspRun,
    settings: WSPSettings,
    now: Callable[[], dt.datetime],
    done: Pass,
) -> bool:
    """Poll one running run on its worker; ``True`` when its result was pulled and ACKED (the
    worker's boot has ended, so its pod may be replaced)."""
    current = now()
    # Fix 2026-11-21 (run wsp_cb092cedad97): ``pushed_at`` is when the push COMPLETED, the moment
    # the worker's supervisor starts the session's time budget (``running_mono``). Measured from
    # the start of a 301 s push instead, this deadline fell before the worker's, and the run was
    # failed with its result, transcript and counters still on the worker.
    pushed = _aware(run.pushed_at) or current
    elapsed = (current - pushed).total_seconds()
    budget = float((run.request.get("budgets") or {}).get("time_s") or 0)
    try:
        status = worker.status()
    except WorkerUnreachableError:
        seen = _aware(run.last_status_at) or pushed
        if (current - seen).total_seconds() > settings.wsp_heartbeat_lapse_s:
            _fail(session, run, "running", "worker_lost", "The worker stopped answering.")
            done.say(f"lost:{run.run_id}")
        elif elapsed > budget + settings.wsp_session_grace_s:
            _fail(session, run, "running", "time_budget", "No result within budget and grace.")
            done.say(f"time_budget:{run.run_id}")
        return False
    if status.get("boot_id") != run.boot_id or status.get("run_id") != run.run_id:
        _fail(session, run, "running", "worker_lost", "The worker restarted under the run.")
        done.say(f"lost:{run.run_id}")
        return False
    runs.transition(
        session,
        run.run_id,
        "running",
        "running",
        last_status_at=current,
        **worker_heartbeat(status),
    )
    if status.get("state") != "finished":
        # A worker that answers is waited for up to its OWN lifetime (``spec.Push.lifetime_s``:
        # budget + grace + pull timeout from the session's start): it kills the session at the
        # budget and serves what it wrote, so a slow kill or result build is still collected.
        if elapsed > worker_lifetime_s(budget, settings):
            _fail(
                session,
                run,
                "running",
                "time_budget",
                "No result within the worker's lifetime (budget, grace and pull timeout).",
            )
            done.say(f"time_budget:{run.run_id}")
        return False
    try:
        data = worker.result()
    except WorkerUnreachableError:
        return False  # the lapse above catches a worker that never serves it
    sha = _sha(data)
    if not runs.transition(
        session, run.run_id, "running", "collecting", result_pulled_at=current, result_sha256=sha
    ):
        return False
    # R3 review fix (OR3-016-1), 2026-11-16: the intake is the boundary where hostile bytes meet
    # the database. It runs inside a savepoint, so whatever it wrote is undone when it fails, and
    # ANY failure - a named refusal or an exception nobody foresaw (a parser error, a constraint
    # violation) - fails the run here with its reason and still acks, so the worker is freed now
    # instead of the run sitting `running` until the pull timeout calls it `worker_lost`.
    try:
        with session.begin_nested():
            outcome = runs.intake_result(session, run, data, settings)
            # The run's usage is the gateway's records, stored by every terminal transition
            # (``runs.transition``), after any open reservation is closed at its worst case.
            runs.transition(session, run.run_id, "collecting", outcome.status, **outcome.columns())
        done.say(f"collected:{run.run_id}:{outcome.status}")
    except Refusal as refusal:
        error = refusal.envelope()["error"]
        # A named run refusal (an inspect that changed weights, a broken change limit) keeps its
        # code; every other intake failure is wsp_result_invalid, naming the check.
        code = refusal.code if refusal.code in INTAKE_RUN_CODES else "wsp_result_invalid"
        _fail(session, run, "collecting", code, f"{error['message']} {INTAKE_NEXT_STEP}")
        done.say(f"intake_refused:{run.run_id}")
    except Exception as exc:
        log.warning(
            "wsp intake could not check a result",
            extra={"code": "wsp_intake_error", "run_id": run.run_id, "error": type(exc).__name__},
            exc_info=True,
        )
        _fail(
            session,
            run,
            "collecting",
            "wsp_result_invalid",
            f"The worker's result could not be checked ({type(exc).__name__}); it is refused and "
            f"nothing from it was kept. The backend log line wsp_intake_error names the cause. "
            f"{INTAKE_NEXT_STEP}",
        )
        done.say(f"intake_refused:{run.run_id}")
    try:
        worker.ack(str(run.boot_id), run.run_id, sha)  # an intake failure still acks
        session.execute(
            sa.update(WspRun).where(WspRun.run_id == run.run_id).values(acked_at=current)
        )
        done.say(f"acked:{run.run_id}")
    except WorkerUnreachableError:
        done.say(f"ack_failed:{run.run_id}")
        return False
    return True


def worker_lifetime_s(budget_s: float, settings: WSPSettings) -> float:
    """The worker's own lifetime for a session (``wsp_worker.spec.Push.lifetime_s``, measured
    from the session's start): past it the worker exits by itself, so nothing is left to wait for.
    """
    return (
        budget_s + float(settings.wsp_session_grace_s) + float(settings.wsp_result_pull_timeout_s)
    )


#: The longest elapsed time stored from a heartbeat: ten days, far past any time budget. A larger
#: (or negative, or non-finite) value is the worker's, untrusted, and is stored null.
MAX_WORKER_ELAPSED_S = 864_000


def worker_heartbeat(status: dict[str, Any]) -> dict[str, Any]:
    """The worker's ``GET /status`` as the run row stores it (live progress, 2026-11-20).

    The worker is untrusted (P8/P9): a state outside ``WORKER_STATES`` is stored ``unknown``, and
    an elapsed time that is not a finite number in ``[0, MAX_WORKER_ELAPSED_S]`` is stored null.
    Nothing else from the answer is kept."""
    state = status.get("state")
    elapsed = status.get("elapsed_s")
    seconds: int | None = None
    if (
        isinstance(elapsed, int | float)
        and not isinstance(elapsed, bool)
        and math.isfinite(elapsed)
        and 0 <= elapsed <= MAX_WORKER_ELAPSED_S
    ):
        seconds = int(elapsed)
    return {
        "worker_state": state if isinstance(state, str) and state in WORKER_STATES else "unknown",
        "worker_elapsed_s": seconds,
    }


def _claim(
    factory: sessionmaker[Session],
    seen: Sequence[_Seen],
    settings: WSPSettings,
    effective: int,
    read_sensitivity: SensitivityReader,
    files_for: Callable[[WspRun, WspBundle], list[tuple[str, Path]]],
    enqueue_push: Callable[[str], None],
    now: Callable[[], dt.datetime],
    done: Pass,
    read_window: WindowReader = no_window,
    backoff: window_read.NoAnswerBackoff | None = None,
) -> None:
    """The claims (TID §3.8 step 4): while fewer workers are held than the effective pool, no run
    is ``pushing`` (F19-D35: pushes run one at a time) and an idle worker on a fresh boot is free,
    the oldest queued run whose key is not active is checked and claimed on the lowest such worker
    (F19-D33). A busy key never holds back another (FR-019.8); a keyless run (a replay, a rerun)
    takes any free worker (F19-D7). The push itself is enqueued after the commit.

    A run that waits for its miServer model (``window_read.awaits_resident``) is read once per
    endpoint per pass AFTER every refusal check, and skipped while the read is empty
    (:func:`_await_resident`), so a waiting run neither consumes a boot nor holds back the runs
    behind it. While ``backoff`` holds an endpoint that gave no answer (debt D5), the read is not
    made and the run waits on that no answer, so its wait still ends on time."""
    with factory() as session:
        queued = session.scalars(
            sa.select(WspRun)
            .where(WspRun.status == "queued")
            .order_by(WspRun.created_at)
            .limit(200)
        ).all()
        if not queued:
            return  # nothing waits, so nothing to say
        # A holding run from before 019 has no worker index: it holds worker 0.
        held = {
            0 if index is None else int(index)
            for index in session.scalars(
                sa.select(WspRun.worker_index).where(WspRun.status.in_(HOLDING_STATES))
            ).all()
        }
        if len(held) >= effective:
            if seen and any(v.ready is not None for v in seen):
                done.say("worker_busy")
            return
        if session.scalar(sa.select(WspRun.run_id).where(WspRun.status == "pushing").limit(1)):
            done.say("push_in_flight")
            return
        free = [
            view
            for view in seen
            if view.slot.index not in held
            and view.ready is not None
            and view.ready.get("state") == "idle"
            and view.boot is not None
            and not view.consumed
        ]
        if not free:
            reachable = [view for view in seen if view.ready is not None]
            if not reachable:
                done.say("worker_unreachable")
            elif all(view.consumed for view in reachable):
                done.say(f"boot_consumed:{reachable[0].boot}")
            else:
                done.say("worker_busy")
            return
        active_keys = set(
            session.scalars(
                sa.select(WspRun.schedule_key).where(
                    WspRun.status.in_(("pushing", "running")), WspRun.schedule_key.is_not(None)
                )
            ).all()
        )
        windows: dict[str, window_read.WindowAnswer] = {}
        for run in queued:
            endpoint = session.get(WspEndpoint, run.endpoint_id) if run.endpoint_id else None
            key = None if endpoint is None else scheduling.key_for(endpoint)
            if key is not None and key in active_keys:
                continue
            bundle = session.get(WspBundle, run.bundle_sha256)
            files = [] if bundle is None else files_for(run, bundle)
            refusal = _refusal(session, run, bundle, read_sensitivity, files, settings)
            if refusal is not None:
                code, message = refusal
                _fail(session, run, "queued", code, message, stop_reason="error")
                session.commit()
                done.say(f"refused:{run.run_id}:{code}")
                continue
            if endpoint is not None and window_read.may_wait(endpoint, run.style):
                if endpoint.id not in windows:
                    windows[endpoint.id] = _claim_read(read_window, endpoint, backoff, now())
                read = windows[endpoint.id]
                if window_read.waits(endpoint, run.style, read):
                    _await_resident(
                        session,
                        run,
                        endpoint,
                        settings,
                        now(),
                        done,
                        read is window_read.NOT_ANSWERED,
                    )
                    continue
            view = free[0]
            current = now()
            try:
                session.add(
                    WspWorkerBoot(boot_id=view.boot, first_seen_at=current, run_id=run.run_id)
                )
                session.flush()
                moved = runs.transition(
                    session,
                    run.run_id,
                    "queued",
                    "pushing",
                    boot_id=view.boot,
                    schedule_key=key,
                    worker_index=view.slot.index,
                    push_heartbeat_at=current,
                    started_at=current,
                    # Feature 019 P2 (TID §3.6): the protocol this run is PUSHED with; the intake
                    # refuses an engine block that names another, and a protocol test records it.
                    action_protocol=(
                        None
                        if endpoint is None
                        else endpoint.action_protocol or DEFAULT_ACTION_PROTOCOL
                    ),
                )
                session.flush()
            except IntegrityError:
                session.rollback()
                log.info(
                    "wsp claim conflict", extra={"code": "wsp_claim_conflict", "run_id": run.run_id}
                )
                done.say(f"claim_conflict:{run.run_id}")
                return
            if not moved:
                session.rollback()
                return
            session.commit()
            enqueue_push(run.run_id)
            done.say(f"claimed:{run.run_id}:{view.slot.index}")
            return  # F19-D35: one run in ``pushing`` at a time


def _claim_read(
    read: WindowReader,
    endpoint: WspEndpoint,
    backoff: window_read.NoAnswerBackoff | None,
    current: dt.datetime,
) -> window_read.WindowAnswer:
    """The claim's read of ``endpoint``: none while ``backoff`` holds it after a no answer (debt
    D5; the no answer stands), else one live read, recorded in ``backoff``."""
    if backoff is not None and backoff.held(endpoint, current):
        return window_read.NOT_ANSWERED
    answer = _read(read, endpoint)
    if backoff is not None:
        backoff.record(endpoint, answer, current)
    return answer


def _await_resident(
    session: Session,
    run: WspRun,
    endpoint: WspEndpoint,
    settings: WSPSettings,
    current: dt.datetime,
    done: Pass,
    unanswered: bool = False,
) -> None:
    """One claim's read that leaves ``run`` waiting (2026-11-27; D1 for ``unanswered``, a server
    that gave no answer at all): the first starts its wait
    (``editor_wait_since``); within ``wsp_editor_resident_wait_s`` it stays queued; past it, it
    fails ``editor_not_resident`` with nothing consumed."""
    since = _aware(run.editor_wait_since)
    if since is None:
        session.execute(
            sa.update(WspRun)
            .where(WspRun.run_id == run.run_id, WspRun.status == "queued")
            .values(editor_wait_since=current)
            .execution_options(synchronize_session=False)
        )
        session.commit()
        log.info(
            "wsp run waits for its editor model",
            extra={
                "code": "wsp_editor_not_resident",
                "run_id": run.run_id,
                "endpoint_id": endpoint.id,
                "wait_s": settings.wsp_editor_resident_wait_s,
            },
        )
        done.say(f"editor_not_resident:{run.run_id}")
        return
    waited = (current - since).total_seconds()
    if waited < settings.wsp_editor_resident_wait_s:
        done.say(f"editor_not_resident:{run.run_id}")
        return
    _fail(
        session,
        run,
        "queued",
        EDITOR_NOT_RESIDENT,
        not_resident_message(endpoint, waited, unanswered=unanswered),
        stop_reason="error",
    )
    session.commit()
    log.warning(
        "wsp run failed: its editor model never became resident",
        extra={"code": EDITOR_NOT_RESIDENT, "run_id": run.run_id, "endpoint_id": endpoint.id},
    )
    done.say(f"refused:{run.run_id}:{EDITOR_NOT_RESIDENT}")


#: The push heartbeat's least interval (TID §13): at most one UPDATE every 10 s per push.
PUSH_HEARTBEAT_EVERY_S = 10.0


def push_claimed(
    factory: sessionmaker[Session],
    worker: Worker,
    settings: WSPSettings,
    run_id: str,
    *,
    read_window: WindowReader = no_window,
    files_for: Callable[[WspRun, WspBundle], list[tuple[str, Path]]],
    now: Callable[[], dt.datetime] = utcnow,
) -> str:
    """Push one claimed run to its worker (feature 019 P3, F19-D19; TID §3.8). Holds no lock. The
    window is read and resolved here (FR-019.1, FR-019.2), the token minted here: its plaintext
    lives only in this frame and the push document, never on the broker or in a log (TP-3).
    Every transition is conditional, so a push racing the lease rule cannot revive a failed run.
    Returns ``pushed``, ``push_failed``, ``lapsed``, ``not_pushing`` or ``requeued``: a run whose
    miServer endpoint states no window and whose read here is empty (the model went away after
    the claim's read) goes back to ``queued`` with a new wait, never to an assumed window
    (2026-11-27); its boot stays consumed (P49), so the next pass recycles that worker."""
    with factory() as session:
        run = session.get(WspRun, run_id)
        if run is None or run.status != "pushing" or run.boot_id is None:
            return "not_pushing"
        boot = run.boot_id
        bundle = session.get(WspBundle, run.bundle_sha256)
        if bundle is None:
            _fail(session, run, "pushing", "wsp_bundle_not_imported", "The run's bundle is gone.")
            session.commit()
            return "push_failed"
        files = files_for(run, bundle)
        endpoint = session.get(WspEndpoint, run.endpoint_id) if run.endpoint_id else None
        resolved = None
        values: dict[str, Any] = {}
        if endpoint is not None:
            # Feature 019 (FR-019.1, FR-019.2, FR-019.15): read the window once for EVERY
            # endpoint (a stated one too, so a mismatch is found; a hosted one is never read), and
            # resolve the window, source and ceiling the row, ``spec.editor`` and the gateway's
            # session entry all carry.
            try:
                resolved = window_read.resolve(
                    endpoint, _read(read_window, endpoint), style=run.style
                )
            except window_read.NotResidentError:
                current = now()
                moved = runs.transition(
                    session,
                    run.run_id,
                    "pushing",
                    "queued",
                    boot_id=None,
                    schedule_key=None,
                    worker_index=None,
                    push_heartbeat_at=None,
                    started_at=None,
                    action_protocol=None,
                    editor_wait_since=current,
                )
                session.commit()
                log.info(
                    "wsp push found no resident editor model; the run is queued again",
                    extra={
                        "code": "wsp_editor_not_resident",
                        "run_id": run.run_id,
                        "endpoint_id": endpoint.id,
                        "at": "push",
                    },
                )
                return "requeued" if moved else "lapsed"
            values = {
                "served_context_tokens": resolved.reported,
                "window_tokens": resolved.window_tokens,
                "window_source": resolved.window_source,
                "output_ceiling_tokens": resolved.ceiling,
                "output_ceiling_lowered": resolved.ceiling_lowered,
            }
            log.info(
                "wsp window",
                extra={
                    "code": "wsp_window",
                    "run_id": run.run_id,
                    "window_tokens": resolved.window_tokens,
                    "window_source": resolved.window_source,
                    "warnings": list(resolved.warnings),
                },
            )
        current = now()
        plaintext: str | None = None
        if runs.takes_gateway(run.style):  # a replay or a rerun gets no token (R3A-13)
            plaintext, token_sha = tokens.mint()
            values["gateway_token_sha256"] = token_sha
            values["token_expires_at"] = current + dt.timedelta(
                seconds=float(run.request["budgets"]["time_s"]) + settings.wsp_session_grace_s
            )
        values["push_heartbeat_at"] = current
        if not runs.transition(session, run.run_id, "pushing", "pushing", **values):
            session.rollback()
            return "lapsed"
        session.commit()
        document = push_document(
            run,
            boot,
            plaintext,
            settings,
            files,
            manifest=bundle.manifest,
            editor=(None if endpoint is None else editor_document(endpoint, resolved)),
            replay_of_mode=_replay_of_mode(session, run),
        )
        last = [current]

        def heartbeat() -> None:
            # On the injected clock, as every other time here: the lease is judged against
            # ``push_heartbeat_at``, so the beat and the lease must read one clock.
            at = now()
            if (at - last[0]).total_seconds() < PUSH_HEARTBEAT_EVERY_S:
                return
            last[0] = at
            with factory() as beat:
                beat.execute(
                    sa.update(WspRun)
                    .where(WspRun.run_id == run_id, WspRun.status == "pushing")
                    .values(push_heartbeat_at=at)
                    .execution_options(synchronize_session=False)
                )
                beat.commit()

        try:
            worker.push(document, files, on_progress=heartbeat)
        except (WorkerUnreachableError, OSError) as exc:
            _fail(session, run, "pushing", "worker_push_failed", f"The push failed: {exc}")
            session.commit()
            return "push_failed"
        # Fix 2026-11-21: the session's clock starts HERE, when the push has completed, as the
        # supervisor's does (``running_mono``); the gateway token is re-dated on the same clock.
        running_at = now()
        extra: dict[str, Any] = {}
        if plaintext is not None:
            extra["token_expires_at"] = running_at + dt.timedelta(
                seconds=float(run.request["budgets"]["time_s"]) + settings.wsp_session_grace_s
            )
        moved = runs.transition(
            session,
            run.run_id,
            "pushing",
            "running",
            pushed_at=running_at,
            last_status_at=running_at,
            **extra,
        )
        session.commit()
        return "pushed" if moved else "lapsed"


def _refusal(
    session: Session,
    run: WspRun,
    bundle: WspBundle | None,
    read_sensitivity: SensitivityReader,
    files: list[tuple[str, Path]],
    settings: WSPSettings,
) -> tuple[str, str] | None:
    if bundle is None or bundle.import_status != "imported":
        return "wsp_bundle_not_imported", "The run's bundle is no longer imported."
    # Production defect 2026-11-20: a push larger than the worker's /work is refused here, for
    # every style, instead of being streamed until the kubelet evicts the worker mid-push.
    too_large = capacity.refusal(capacity.files_bytes(files), settings)
    if too_large is not None:
        return too_large
    if not runs.takes_gateway(run.style):
        return _source_refusal(session, run, files)
    endpoint = session.get(WspEndpoint, run.endpoint_id) if run.endpoint_id else None
    if endpoint is None or endpoint.removed_at is not None:
        return "wsp_endpoint_unknown", "The run's editor endpoint was removed."
    if runs.endpoint_fingerprint(endpoint) != run.endpoint_fingerprint:
        return "wsp_endpoint_changed", "The editor endpoint changed after the run was frozen."
    # Feature 020 (FR-020.2, FR-020.3, F20-D12; TD-2): the filing check again, so a run queued
    # before the deploy (or before a hosted window was cleared) fails at push by name.
    refused = endpoints.run_refusal(endpoint)
    if refused is not None:
        log.info("wsp_run_refused", extra={"code": refused[0], "run_id": run.run_id, "at": "push"})
        return refused
    if run.protocol_test_endpoint_id is None:
        # Feature 019 P2 (FR-019.11): the pass can clear while the run waits (a failed re-test).
        try:
            runs.check_protocol_pass(endpoint)
        except Refusal as exc:
            return exc.code, exc.message
    if runs.endpoint_locality(endpoint) == "external" and bundle.owner == BUILTIN_OWNER:
        # Feature 018: the protocol test's built-in bundle is generated by miForge and holds no
        # real data; miWorkbench does not know it, so its stored value stands (an import can
        # never create a row with this owner: ``bundles.parse_request``).
        now_value = bundle.sensitivity_value
        if now_value != "not_sensitive":
            return "wsp_sensitivity_raised", "The built-in bundle is not marked not_sensitive."
    elif runs.endpoint_locality(endpoint) == "external":
        # P41: re-read at push time; unreachable reads as unknown, which is sensitive.
        try:
            now_value = read_sensitivity(run, bundle)
        except Exception as exc:
            log.warning(
                "bundle sensitivity read failed",
                extra={"code": "wsp_sensitivity_unread", "run_id": run.run_id},
                exc_info=exc,
            )
            now_value = "unknown"
        # R3 review fix (OR3-016-5), 2026-11-16: say which it is and what to do.
        if now_value == "sensitive":
            return (
                "wsp_sensitivity_raised",
                "The bundle is now sensitive, so an external session is refused. Start it again "
                "on a local endpoint.",
            )
        if now_value != "not_sensitive":
            return (
                "wsp_sensitivity_raised",
                "The bundle's sensitivity could not be read from miWorkbench, so an external "
                "session is refused (an unread value counts as sensitive). The backend log line "
                "wsp_sensitivity_unread names the cause (MIWORKBENCH_API_URL unset, miWorkbench "
                "unreachable, or its HTTP status); fix it and start the run again, or start it "
                "on a local endpoint.",
            )
    return None


def _file_sha(files: list[tuple[str, Path]], name: str) -> str | None:
    path = dict(files).get(name)
    if path is None or not path.is_file() or path.is_symlink():
        return None
    return str(_entry(name, path)["sha256"])


def _source_refusal(
    session: Session, run: WspRun, files: list[tuple[str, Path]]
) -> tuple[str, str] | None:
    """A replay pushes the ORIGINAL transcript, a rerun the STORED program: each checked here
    against the digest the database holds, before anything is consumed (R1B-A7)."""
    if run.style == "replay":
        original = session.get(WspRun, run.replay_of) if run.replay_of else None
        if original is None or original.transcript_sha256 is None:
            return "wsp_run_not_found", "The replayed run or its transcript is gone."
        if _file_sha(files, TRANSCRIPT_NAME) != original.transcript_sha256:
            return (
                "wsp_result_invalid",
                "The stored transcript no longer matches the original run's transcript_sha256.",
            )
        return None
    if _file_sha(files, PROGRAM_NAME) != run.program_sha256:
        return "wsp_program_not_found", "The stored program no longer matches its sha256."
    return None


#: The extra file a replay or a rerun pushes, and its role.
TRANSCRIPT_NAME, PROGRAM_NAME = "transcript.jsonl", "program.py"
SOURCE_ROLES = {
    "replay": (TRANSCRIPT_NAME, "transcript"),
    "program_rerun": (PROGRAM_NAME, "program"),
}


def push_document(
    run: WspRun,
    boot_id: str,
    token: str | None,
    settings: WSPSettings,
    files: list[tuple[str, Path]],
    *,
    manifest: dict[str, Any] | None = None,
    editor: dict[str, Any] | None = None,
    replay_of_mode: str | None = None,
) -> dict[str, Any]:
    """The first multipart part of ``POST /session`` (FTDD §5.4a).

    ``spec`` also carries the bundle ``manifest`` (the worker checks every pushed file and tensor
    against it) and the ``editor`` (kind, model, serving and prices: what the adapter needs to build
    its client; never a key, which only the gateway can open).
    """
    gateway = None
    if runs.takes_gateway(run.style) and token is not None and settings.wsp_gateway_url:
        ip, port = gateway_address(settings.wsp_gateway_url)
        gateway = {"ip": ip, "port": port, "token": token}
    budgets = run.request["budgets"]
    # Each file carries its manifest role: the worker's loader matches name, role and sha256.
    roles = {f["name"]: f["role"] for f in (manifest or {}).get("files", [])}
    if run.style in SOURCE_ROLES:
        name, role = SOURCE_ROLES[run.style]
        roles[name] = role
    return {
        "boot_id": boot_id,
        "run_id": run.run_id,
        "mode": run.style,
        "spec": {
            "time_budget_s": float(budgets["time_s"]),
            "grace_s": float(settings.wsp_session_grace_s),
            "pull_timeout_s": float(settings.wsp_result_pull_timeout_s),
            "log_max_bytes": int(settings.wsp_child_log_max_bytes),
            # The disk watchdog's bound, below the `work` volume (wsp.capacity).
            "work_max_bytes": capacity.watchdog_bytes(settings),
            "request": run.request,
            "manifest": manifest,
            "editor": editor,
            **({"replay_of_mode": replay_of_mode} if replay_of_mode else {}),
        },
        "files": [_entry(name, path, roles.get(name, "bundle")) for name, path in files],
        "gateway": gateway,
    }


def _replay_of_mode(session: Session, run: WspRun) -> str | None:
    if run.style != "replay" or not run.replay_of:
        return None
    original = session.get(WspRun, run.replay_of)
    return None if original is None else original.style


def _read(read: WindowReader, endpoint: WspEndpoint) -> window_read.WindowAnswer:
    """The reader's answer for ``endpoint``. A push never fails on it: ``NOT_ANSWERED`` passes
    through (D1); anything else but a positive integer, or an exception from an injected
    reader, is ``None`` (the production reader, ``window_read.read_window_answer``, never
    raises)."""
    try:
        value = read(endpoint)
    except Exception as exc:
        log.warning(
            "wsp window read failed",
            extra={"code": "wsp_window_unread", "endpoint_id": endpoint.id},
            exc_info=exc,
        )
        return None
    if value is window_read.NOT_ANSWERED:
        return value
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        return None
    return value


def editor_document(
    endpoint: WspEndpoint, resolved: window_read.Resolved | None = None
) -> dict[str, Any]:
    """What the worker's adapter needs to build its LLM client: no key, no URL (the gateway's).

    Feature 018 (FR-018.8, FR-018.12): the endpoint's ``action_protocol`` and
    ``context_window_tokens``. Feature 019 (FR-019.2, FR-019.15): the push's resolution, the
    ``window_tokens`` the engine uses first, its ``window_source`` and ``window_warnings``, the
    ``output_ceiling_tokens`` (after F19-D12's lowering) and whether it was lowered, and the
    window the server reported at this push (``served_context_tokens``). An engine from before
    019 reads the stated window, then the reported one: the same figure. Feature 020 (FR-020.6,
    TD-5): the endpoint's ``history_mode`` and its ``locality``."""

    def f(value: Any) -> float | None:
        return None if value is None else float(value)

    window: dict[str, Any] = {"served_context_tokens": None}
    if resolved is not None:
        window = {
            "served_context_tokens": resolved.reported,
            "window_tokens": resolved.window_tokens,
            "window_source": resolved.window_source,
            "window_warnings": list(resolved.warnings),
            "output_ceiling_tokens": resolved.ceiling,
            "output_ceiling_lowered": resolved.ceiling_lowered,
        }
    return {
        "kind": endpoint.kind,
        "model": endpoint.model,
        "serving": endpoint.serving,
        "action_protocol": endpoint.action_protocol or DEFAULT_ACTION_PROTOCOL,
        "context_window_tokens": endpoint.context_window_tokens,
        **window,
        # Feature 020 (FR-020.6, TD-5): how the engine keeps the conversation, and where the
        # editor runs (thinking fields and the output cap are keyed on kind and locality, TD-6).
        "history_mode": endpoint.history_mode or DEFAULT_HISTORY_MODE,
        "locality": runs.endpoint_locality(endpoint),
        "prices": {
            "input": f(endpoint.price_in_per_mtok),
            "output": f(endpoint.price_out_per_mtok),
            "cache_write": f(endpoint.price_cache_write_per_mtok),
            "cache_read": f(endpoint.price_cache_read_per_mtok),
        },
    }


def _entry(name: str, path: Path, role: str = "bundle") -> dict[str, Any]:
    digest = hashlib.sha256()
    size = 0
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
            size += len(chunk)
    return {"name": name, "role": role, "sha256": digest.hexdigest(), "bytes": size}


def _sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def gateway_address(url: str) -> tuple[str, int]:
    """The gateway Service's IP and port: the worker has no DNS, so the push carries a literal."""
    import socket
    from urllib.parse import urlsplit

    parts = urlsplit(url)
    if parts.hostname is None:
        raise WorkerUnreachableError(f"WSP_GATEWAY_URL has no host: {url!r}")
    port = parts.port or (443 if parts.scheme == "https" else 80)
    try:
        info = socket.getaddrinfo(parts.hostname, port, type=socket.SOCK_STREAM)
    except OSError as exc:
        raise WorkerUnreachableError(f"the gateway's address cannot be resolved: {exc}") from exc
    return str(info[0][4][0]), port
