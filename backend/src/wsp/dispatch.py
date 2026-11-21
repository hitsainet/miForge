"""The WSP driver: one pass of ``miforge.wsp.drive`` (feature 016; FTDD §7; FR-016.11; P41, P60).

A pass holds a single-flight lock for its whole length (a PostgreSQL advisory transaction lock; on
SQLite, where every test runs in one process, a process lock), so two beats never push twice.

Per pass, in this order:

0. **orphans** (R3A-10): a run found ``pushing`` at the start of a pass is failed
   ``worker_push_failed`` (a pass always moves its own run out of ``pushing`` before it releases the
   lock). Then, when ``/ready`` shows a boot already in ``mf_wsp_worker_boots`` and no live run
   owns it: ``idle`` gets ``POST /recycle``; ``finished`` is pulled, discarded unread and acked
   (this is also how a cancelled running run's worker is released, OVN-wsp-forge-24);
1. a ``running`` run: ``GET /status``; another ``boot_id`` fails it ``worker_lost`` at once; on
   ``finished``: ``GET /result`` → intake → ``POST /ack`` (an intake failure still acks);
   unreachable past ``WSP_HEARTBEAT_LAPSE_S`` → ``worker_lost``; unreachable past budget + grace,
   or answering but unfinished past the worker's own lifetime (budget + grace + pull timeout),
   → ``time_budget``. Every clock runs from ``pushed_at``, when the push completed and the
   worker's session began (fix 2026-11-21);
2. else the oldest ``queued`` run: ``GET /ready`` → **every refusal check** (the push-time
   sensitivity read for an external endpoint, the bundle's import status, the push's size against
   the worker's ``/work`` (``wsp.capacity``), the endpoint fingerprint)
   → ONE transaction (boot insert, token, ``pushing``) → ``POST /session`` → ``running``, or
   ``failed`` (``worker_push_failed``) on any push error, so a run is never pushed twice. A refusal
   leaves the boot unconsumed. There is no gateway registration (P61, R3A-7).
"""

from __future__ import annotations

import datetime as dt
import hashlib
import logging
import math
import threading
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol

import sqlalchemy as sa
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session, sessionmaker

from core.errors import Refusal
from wsp import capacity, runs, tokens
from wsp.models import (
    DEFAULT_ACTION_PROTOCOL,
    WORKER_STATES,
    WspBundle,
    WspEndpoint,
    WspRun,
    WspWorkerBoot,
)
from wsp.protocol_test import BUILTIN_OWNER
from wsp.settings import WSPSettings

log = logging.getLogger(__name__)

#: ``hashtext('miforge.wsp.drive')`` would do; a literal keeps it obvious and stable.
ADVISORY_KEY = 0x57535044  # "WSPD"
_LOCAL_LOCK = threading.Lock()


class WorkerUnreachableError(Exception):
    """The worker did not answer (connect, read or protocol failure)."""


class Worker(Protocol):
    def ready(self) -> dict[str, Any]: ...
    def push(self, document: dict[str, Any], files: list[tuple[str, Path]]) -> dict[str, Any]: ...
    def status(self) -> dict[str, Any]: ...
    def result(self) -> bytes: ...
    def ack(self, boot_id: str, run_id: str, result_sha256: str) -> None: ...
    def recycle(self, boot_id: str) -> None: ...


SensitivityReader = Callable[[WspRun, WspBundle], str]
#: The context window miServer reports for its resident model, or ``None`` (feature 018,
#: FR-018.12; ``wsp.tasks.read_served_window`` in production).
ServedWindowReader = Callable[[WspEndpoint], int | None]


def no_served_window(endpoint: WspEndpoint) -> int | None:
    """The default reader: nothing is read (the engine falls back to its own default)."""
    return None


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


def drive(
    factory: sessionmaker[Session],
    worker: Worker,
    settings: WSPSettings,
    *,
    read_sensitivity: SensitivityReader,
    files_for: Callable[[WspRun, WspBundle], list[tuple[str, Path]]],
    now: Callable[[], dt.datetime] = utcnow,
    read_served_window: ServedWindowReader = no_served_window,
) -> Pass:
    done = Pass()
    with single_flight(factory) as held:
        if not held:
            done.say("lock_busy")
            return done
        _orphans(factory, worker, done)
        with factory() as session:
            running = session.scalar(sa.select(WspRun).where(WspRun.status == "running"))
            if running is not None:
                _poll(session, worker, running, settings, now, done)
                session.commit()
                return done
        _push_next(
            factory, worker, settings, read_sensitivity, files_for, now, done, read_served_window
        )
    return done


def _orphans(factory: sessionmaker[Session], worker: Worker, done: Pass) -> None:
    with factory() as session:
        for run in session.scalars(sa.select(WspRun).where(WspRun.status == "pushing")).all():
            if _fail(
                session,
                run,
                "pushing",
                "worker_push_failed",
                "The push did not complete; the run is never pushed again.",
            ):
                done.say(f"orphan_failed:{run.run_id}")
        session.commit()
        try:
            ready = worker.ready()
        except WorkerUnreachableError:
            return
        boot = ready.get("boot_id")
        if not isinstance(boot, str) or session.get(WspWorkerBoot, boot) is None:
            return
        live = session.scalar(
            sa.select(WspRun).where(WspRun.boot_id == boot, WspRun.status == "running")
        )
        if live is not None:
            return
        state = ready.get("state")
        try:
            if state == "idle":
                worker.recycle(boot)
                done.say(f"recycled:{boot}")
            elif state == "finished":
                data = worker.result()
                owner = session.scalar(sa.select(WspRun).where(WspRun.boot_id == boot))
                worker.ack(boot, "" if owner is None else owner.run_id, _sha(data))
                done.say(f"discarded:{boot}")
        except WorkerUnreachableError:
            return


def _poll(
    session: Session,
    worker: Worker,
    run: WspRun,
    settings: WSPSettings,
    now: Callable[[], dt.datetime],
    done: Pass,
) -> None:
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
        return
    if status.get("boot_id") != run.boot_id or status.get("run_id") != run.run_id:
        _fail(session, run, "running", "worker_lost", "The worker restarted under the run.")
        done.say(f"lost:{run.run_id}")
        return
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
        return
    try:
        data = worker.result()
    except WorkerUnreachableError:
        return  # the lapse above catches a worker that never serves it
    sha = _sha(data)
    if not runs.transition(
        session, run.run_id, "running", "collecting", result_pulled_at=current, result_sha256=sha
    ):
        return
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


def _push_next(
    factory: sessionmaker[Session],
    worker: Worker,
    settings: WSPSettings,
    read_sensitivity: SensitivityReader,
    files_for: Callable[[WspRun, WspBundle], list[tuple[str, Path]]],
    now: Callable[[], dt.datetime],
    done: Pass,
    read_served_window: ServedWindowReader = no_served_window,
) -> None:
    with factory() as session:
        run = session.scalar(
            sa.select(WspRun).where(WspRun.status == "queued").order_by(WspRun.created_at).limit(1)
        )
        if run is None:
            return
        try:
            ready = worker.ready()
        except WorkerUnreachableError:
            done.say("worker_unreachable")
            return
        boot = ready.get("boot_id")
        if ready.get("state") != "idle" or not isinstance(boot, str):
            done.say("worker_busy")
            return
        if session.get(WspWorkerBoot, boot) is not None:
            # A consumed boot: it never gets a second session (P60). The orphan step of this
            # pass already sent it /recycle when it was idle and no live run owned it.
            done.say(f"boot_consumed:{boot}")
            return
        # ---- every refusal check, before anything is consumed (R3A-7) -------------------
        bundle = session.get(WspBundle, run.bundle_sha256)
        files = [] if bundle is None else files_for(run, bundle)
        refusal = _refusal(session, run, bundle, read_sensitivity, files, settings)
        if refusal is not None:
            code, message = refusal
            _fail(session, run, "queued", code, message, stop_reason="error")
            session.commit()
            done.say(f"refused:{run.run_id}:{code}")
            return
        assert bundle is not None
        # The window miServer serves for the endpoint's model (2026-11-21): read once, before the
        # pushing transaction, so the gateway's session entry carries the same figure the push's
        # ``spec.editor`` gives the engine (an uncapped request is sized to it).
        endpoint = session.get(WspEndpoint, run.endpoint_id) if run.endpoint_id else None
        served = None if endpoint is None else served_window(read_served_window, endpoint)
        # ---- one transaction: boot, token, pushing --------------------------------------
        # A replay or a program rerun gets no token and no gateway (R3A-13).
        current = now()
        plaintext: str | None = None
        token_sha: str | None = None
        expires: dt.datetime | None = None
        if runs.takes_gateway(run.style):
            plaintext, token_sha = tokens.mint()
            expires = current + dt.timedelta(
                seconds=float(run.request["budgets"]["time_s"]) + settings.wsp_session_grace_s
            )
        try:
            session.add(WspWorkerBoot(boot_id=boot, first_seen_at=current, run_id=run.run_id))
            session.flush()
        except IntegrityError:
            session.rollback()
            done.say(f"boot_reused:{boot}")
            return
        if not runs.transition(
            session,
            run.run_id,
            "queued",
            "pushing",
            boot_id=boot,
            gateway_token_sha256=token_sha,
            token_expires_at=expires,
            started_at=current,
            served_context_tokens=served,
        ):
            session.rollback()
            return
        session.commit()
        document = push_document(
            run,
            boot,
            plaintext,
            settings,
            files,
            manifest=bundle.manifest,
            editor=(None if endpoint is None else editor_document(endpoint, served)),
            replay_of_mode=_replay_of_mode(session, run),
        )
        try:
            worker.push(document, files)
        except (WorkerUnreachableError, OSError) as exc:
            _fail(session, run, "pushing", "worker_push_failed", f"The push failed: {exc}")
            session.commit()
            done.say(f"push_failed:{run.run_id}")
            return
        # Fix 2026-11-21: the session's clock starts HERE, when the push has completed, as the
        # supervisor's does (``running_mono``); a 34.9 GB push took 301 s of the budget's
        # deadline when it started at ``current``. The gateway token is re-dated on the same
        # clock, so the editor's calls are not refused for the push's length before the budget
        # ends (the gateway re-reads an expiry it holds as passed: ``wsp_gateway.app``).
        running_at = now()
        extra: dict[str, Any] = {}
        if token_sha is not None:
            extra["token_expires_at"] = running_at + dt.timedelta(
                seconds=float(run.request["budgets"]["time_s"]) + settings.wsp_session_grace_s
            )
        runs.transition(
            session,
            run.run_id,
            "pushing",
            "running",
            pushed_at=running_at,
            last_status_at=running_at,
            **extra,
        )
        session.commit()
        done.say(f"pushed:{run.run_id}")


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


def served_window(read: ServedWindowReader, endpoint: WspEndpoint) -> int | None:
    """miServer's served window for ``endpoint`` (FR-018.12), read only when it can matter: a
    miServer endpoint whose operator set no ``context_window_tokens`` (which the engine prefers).
    A push never fails on it: any failure, or anything but a positive integer, is ``None``."""
    if endpoint.serving != "miserver" or endpoint.context_window_tokens is not None:
        return None
    try:
        value = read(endpoint)
    except Exception as exc:
        log.warning(
            "served window read failed",
            extra={"code": "wsp_served_window_unread", "endpoint_id": endpoint.id},
            exc_info=exc,
        )
        return None
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        return None
    return value


def editor_document(
    endpoint: WspEndpoint, served_context_tokens: int | None = None
) -> dict[str, Any]:
    """What the worker's adapter needs to build its LLM client: no key, no URL (the gateway's).

    Feature 018 (FR-018.8, FR-018.12): the endpoint's ``action_protocol`` and
    ``context_window_tokens``, and the window miServer reported at this push
    (``served_context_tokens``), which the engine uses when the endpoint sets none."""

    def f(value: Any) -> float | None:
        return None if value is None else float(value)

    return {
        "kind": endpoint.kind,
        "model": endpoint.model,
        "serving": endpoint.serving,
        "action_protocol": endpoint.action_protocol or DEFAULT_ACTION_PROTOCOL,
        "context_window_tokens": endpoint.context_window_tokens,
        "served_context_tokens": served_context_tokens,
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
