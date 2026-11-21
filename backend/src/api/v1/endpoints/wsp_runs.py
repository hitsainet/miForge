"""WSP runs: start, list, get, transcript, cancel, replay; programs: get, rerun (feature 016;
FTDD §5.2, §5.3, §7; FR-016.21, FR-016.23).

``POST /wsp/runs`` is GATED for agents (FR-016.37): ``wsp_session``, escalated to
``wsp_session_external`` when the stored endpoint is external; a local inspect runs at once. The
gate's validate hook and this route run the same ``wsp.runs.prepare_start``. When an approved
request executes, the route re-derives the action and the endpoint fingerprint and refuses
``wsp_approval_action_stale`` / ``wsp_endpoint_changed`` when either moved since filing.

A started run is only stored ``queued``: the driver (``miforge.wsp.drive``) pushes it to the worker.
A replay and a program rerun are ungated writes (PF-13): no LLM, no gateway, no token (R3A-13).
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Annotated, Any

import sqlalchemy as sa
from fastapi import APIRouter, Body, Header, Query
from sqlalchemy.orm import Session

from api.agent_origin import executing_actor, requires_approval_when_agent
from api.deps import SessionDep
from core.errors import refuse
from models.approval import Approval
from services import approval_gates
from wsp import progress, runs
from wsp.models import RUN_STATUSES, WspProgram, WspRun
from wsp.settings import get_wsp_settings

router = APIRouter(prefix="/wsp", tags=["wsp"])

MAX_PAGE = 200


@router.post("/runs", status_code=201, summary="Start a WSP session run (queued for the worker)")
@requires_approval_when_agent(
    "wsp_session",
    when=approval_gates.wsp_run_needs_approval,
    escalate=("wsp_session_external", approval_gates.wsp_run_is_external),
    validate=approval_gates.validate_wsp_run,
    summary=approval_gates.summarise_wsp_run,
)
def start_run(
    body: Annotated[dict[str, Any], Body()],
    session: SessionDep,
    x_miforge_agent: Annotated[str | None, Header()] = None,
) -> dict[str, Any]:
    actor = executing_actor()
    caller = actor.agent if actor is not None and actor.agent else (x_miforge_agent or "operator")
    prepared = runs.prepare_start(session, body, caller)
    approval_id = approval_still_holds(session, prepared)
    row = runs.create_run(session, prepared, approval_id=approval_id)
    return {"run_id": row.run_id, "request_sha256": row.request_sha256, "status": row.status}


def approval_still_holds(session: Session, prepared: runs.Prepared) -> str | None:
    """When an approved request executes: its id, after re-deriving the action and the endpoint
    fingerprint (``wsp_approval_action_stale`` / ``wsp_endpoint_changed`` when either moved since
    filing). ``None`` for a call that needed no approval. Shared with the protocol test (018)."""
    actor = executing_actor()
    approval_id: str | None = None
    if actor is not None and actor.approval_id is not None:
        approval = session.get(Approval, actor.approval_id)
        if approval is None:  # pragma: no cover - the executor holds the row it is executing
            raise refuse("approval_not_found", "The executing approval is gone.")
        if prepared.frozen.approval_action != approval.action:
            raise refuse(
                "wsp_approval_action_stale",
                "The run's approval action changed since the request was filed (the endpoint's "
                "locality or the run's kind moved); file the request again.",
                approved=approval.action,
                now=prepared.frozen.approval_action,
            )
        if (approval.summary or {}).get("endpoint_fingerprint") != prepared.fingerprint:
            raise refuse(
                "wsp_endpoint_changed",
                "The editor endpoint changed since the request was filed; file it again.",
                endpoint_id=prepared.endpoint.id,
            )
        approval_id = approval.id
    return approval_id


@router.get("/runs", summary="WSP runs, newest first")
def list_runs(
    session: SessionDep,
    status: Annotated[str | None, Query()] = None,
    limit: Annotated[int, Query(ge=1, le=MAX_PAGE)] = 50,
) -> dict[str, Any]:
    query = sa.select(WspRun).where(WspRun.deleted_at.is_(None))
    if status is not None:
        if status not in RUN_STATUSES:
            raise refuse("validation_failed", f"status is one of {list(RUN_STATUSES)}.")
        query = query.where(WspRun.status == status)
    rows = session.scalars(query.order_by(WspRun.created_at.desc()).limit(limit)).all()
    # Live progress (2026-11-20): a live row carries its compact progress, a finished one null.
    live = progress.list_progress(session, rows)
    return {
        "runs": [
            {
                "run_id": r.run_id,
                "status": r.status,
                "kind": r.operation,
                "mode": r.style,
                "caller": r.caller,
                "caller_ref": r.caller_ref,
                "endpoint_id": r.endpoint_id,
                "stop_reason": r.stop_reason,
                "outcome": r.outcome,
                "created_at": runs._iso(r.created_at),
                # Feature 018 (FR-018.11a): miForge's own counters, in these rows only, never in
                # run_document (miforge.wsp-run/v1 is closed and vendored by miWorkbench).
                "provider": r.provider_id,
                # Fix 2026-11-20 (the web UI's WSP runs view): why it stopped, what it used and
                # when it ended, in the list too. miForge's own rows: not miforge.wsp-run/v1.
                "provider_version": r.provider_version,
                "failure": r.failure,
                "usage": _usage(r.usage),
                "started_at": None if r.started_at is None else runs._iso(r.started_at),
                "finished_at": None if r.finished_at is None else runs._iso(r.finished_at),
                "action_protocol": r.action_protocol,
                "handoffs": r.handoffs,
                "refused_turns": r.refused_turns,
                "kernel_restarts": r.kernel_restarts,
                # 2026-11-21: the thinking frozen in the request, and whether reasoning came
                # back (``reasoning_returned``: null unless thinking was on and a result was read).
                "thinking": (r.request or {}).get("thinking"),
                "thinking_tokens": (r.request or {}).get("thinking_tokens"),
                "output_cap": (r.request or {}).get("output_cap", "on"),
                "reasoning_turns": r.reasoning_turns,
                # The repeat guard's count (2026-11-21), beside refused_turns.
                "repeated_turns": r.repeated_turns,
                # Paging (2026-11-21): the frozen setting (null: not paged) and its counters.
                "paging": (r.request or {}).get("paging"),
                "pages_out": r.pages_out,
                "recalls": r.recalls,
                "ledger_rows": r.ledger_rows,
                "peak_prompt_tokens": r.peak_prompt_tokens,
                "reasoning_returned": reasoning_returned(r),
                "forward_passes_used": r.forward_passes_used,
                "progress": live.get(r.run_id),
            }
            for r in rows
        ]
    }


def reasoning_returned(row: WspRun) -> bool | None:
    """Whether the editor's replies carried reasoning, for a run that asked for it: ``None``
    unless the frozen thinking is ``on`` and the engine counted (``reasoning_turns``)."""
    if (row.request or {}).get("thinking") != "on" or row.reasoning_turns is None:
        return None
    return row.reasoning_turns > 0


def _usage(usage: dict[str, Any] | None) -> dict[str, Any]:
    """The metered usage as ``run_document`` serves it (requests, tokens, USD)."""
    u = dict(usage or {})
    return {
        "requests": int(u.get("requests", 0)),
        "input_tokens": int(u.get("input_tokens", 0)),
        "output_tokens": int(u.get("output_tokens", 0)),
        "cost_usd": float(u.get("cost_usd", 0.0)),
    }


@router.get("/runs/{run_id}", summary="One WSP run as miforge.wsp-run/v1")
def get_run(run_id: str, session: SessionDep) -> dict[str, Any]:
    """The live document, with its candidate, lineage chain and stored verdict
    (``evaluate.document_of``, as ``/candidates/{id}/run-document`` serves it). Until 2026-11-20
    this route built it without the candidate, so a run whose outcome was ``candidate`` read
    ``candidate_id: null`` and ``verdict: null``."""
    from wsp import evaluate

    return evaluate.document_of(session, runs.get_run(session, run_id))


@router.get("/runs/{run_id}/progress", summary="A WSP run's live progress (miForge's own read)")
def get_progress(run_id: str, session: SessionDep) -> dict[str, Any]:
    """Live progress, 2026-11-20: the worker's last heartbeat, the gateway's metered usage so far,
    each budget's spend against its cap, and a queued run's place in the queue. Final figures
    once the run is terminal (``live: false``). Beside ``miforge.wsp-run/v1``, never in it."""
    return progress.run_progress(session, runs.get_run(session, run_id))


@router.get("/runs/{run_id}/calls", summary="A WSP run's editor calls: counts and codes only")
def get_calls(
    run_id: str,
    session: SessionDep,
    after_seq: Annotated[int | None, Query(ge=-1)] = None,
    limit: Annotated[int, Query(ge=1, le=progress.MAX_CALLS_PAGE)] = progress.MAX_CALLS_PAGE,
) -> dict[str, Any]:
    """One row per editor call the gateway saw, oldest first (live progress, 2026-11-20): its
    sequence number, time, tokens, cost, latency, outcome, upstream status, finish reason and
    refusal code. **Never the prompt or the answer**: the ledger stores no text. ``after_seq``
    reads only the calls after that one, so a live page polls for what is new."""
    return progress.run_calls(
        session, runs.get_run(session, run_id), after_seq=after_seq, limit=limit
    )


@router.get("/runs/{run_id}/transcript", summary="The editor's transcript, served as data")
def get_transcript(run_id: str, session: SessionDep) -> dict[str, Any]:
    """The transcript the intake stored and hashed (FR-016.19). Every line is DATA: a hostile
    transcript is never interpreted. ``null`` before the run has a result. A run that hit its
    deadline or ended abnormally keeps its transcript as written (review OR2-016-3): a line that
    is not one JSON object (a cut last line) is served as ``{"raw": <text>}`` and ``partial`` is
    true."""
    row = runs.get_run(session, run_id)
    if row.transcript_sha256 is None:
        return {"run_id": run_id, "transcript_sha256": None, "entries": [], "partial": False}
    path = runs.run_dir(get_wsp_settings(), run_id) / "transcript.jsonl"
    raw = _read(path)
    if hashlib.sha256(raw).hexdigest() != row.transcript_sha256:
        raise refuse(
            "wsp_result_invalid",
            "The stored transcript no longer matches its recorded sha256: the file under "
            "WSP_BUNDLE_ROOT/runs was changed after the intake stored it, so it is not served. "
            "Restore it from a backup of WSP_BUNDLE_ROOT; until then the run's transcript is "
            "lost.",
            run_id=run_id,
        )
    entries: list[dict[str, Any]] = []
    partial = False
    for line in raw.decode("utf-8", "replace").splitlines():
        if not line.strip():
            continue
        entry = _entry(line)
        partial = partial or entry is None
        entries.append({"raw": line} if entry is None else entry)
    return {
        "run_id": run_id,
        "transcript_sha256": row.transcript_sha256,
        "entries": entries,
        "partial": partial,
    }


def _entry(line: str) -> dict[str, Any] | None:
    """One stored line's JSON object, or ``None`` when it is not one."""
    try:
        entry = json.loads(line)
    except ValueError:
        return None
    return entry if isinstance(entry, dict) else None


def _read(path: Path) -> bytes:
    try:
        return path.read_bytes()
    except OSError as exc:
        raise refuse("wsp_result_invalid", "The stored transcript cannot be read.") from exc


@router.post("/runs/{run_id}/cancel", summary="Cancel a queued or running WSP run")
def cancel_run(run_id: str, session: SessionDep) -> dict[str, Any]:
    row = runs.cancel(session, run_id)
    return {"run_id": row.run_id, "status": row.status, "stop_reason": row.stop_reason}


def _caller(x_miforge_agent: str | None) -> str:
    return x_miforge_agent or "operator"


@router.post(
    "/runs/{run_id}/replay",
    status_code=201,
    summary="Replay a completed run from its transcript (no LLM, no gateway)",
)
def replay_run(
    run_id: str,
    session: SessionDep,
    x_miforge_agent: Annotated[str | None, Header()] = None,
) -> dict[str, Any]:
    row = runs.create_replay(session, run_id, _caller(x_miforge_agent))
    return {"run_id": row.run_id, "request_sha256": row.request_sha256, "status": row.status}


@router.get("/programs/{program_sha256}", summary="A stored program, served as data")
def get_program(program_sha256: str, session: SessionDep) -> dict[str, Any]:
    """The program's text is DATA: never executed by the backend."""
    program = session.get(WspProgram, program_sha256)
    if program is None:
        raise refuse("wsp_program_not_found", "No stored program with that sha256.")
    path = runs.program_path(get_wsp_settings(), program_sha256)
    try:
        raw = path.read_bytes()
    except OSError as exc:
        raise refuse("wsp_program_not_found", "The stored program cannot be read.") from exc
    if hashlib.sha256(raw).hexdigest() != program_sha256:
        raise refuse(
            "wsp_result_invalid",
            "The stored program no longer matches its sha256: the file under "
            "WSP_BUNDLE_ROOT/programs was changed after it was stored, so it is not served (and a "
            "rerun of it is refused). Restore it from a backup of WSP_BUNDLE_ROOT.",
        )
    return {
        "program_sha256": program_sha256,
        "first_run_id": program.first_run_id,
        "bytes": program.bytes,
        "created_at": runs._iso(program.created_at),
        "source": raw.decode("utf-8", errors="replace"),
        "reruns": runs.program_reruns(session, program_sha256),
    }


@router.post(
    "/programs/{program_sha256}/rerun",
    status_code=201,
    summary="Rerun a stored program on a named support set (no LLM, no gateway)",
)
def rerun_program(
    program_sha256: str,
    session: SessionDep,
    body: Annotated[dict[str, Any] | None, Body()] = None,
    x_miforge_agent: Annotated[str | None, Header()] = None,
) -> dict[str, Any]:
    support = ((body or {}).get("support_set") or {}).get("kind", "practice_train")
    row = runs.create_rerun(session, program_sha256, str(support), _caller(x_miforge_agent))
    return {"run_id": row.run_id, "request_sha256": row.request_sha256, "status": row.status}
