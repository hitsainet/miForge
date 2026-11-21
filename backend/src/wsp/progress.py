"""A WSP run's live progress and its call timeline (live progress, 2026-11-20; the web UI's WSP run
pages).

Both are miForge's own reads, served beside the run document and never inside it:
``miforge.wsp-run/v1`` is a closed contract vendored byte-identically by miWorkbench, so nothing
here touches it.

**Where each figure comes from** (no new source of truth):

- the worker's last state, elapsed time and heartbeat time: the run row, written by the driver on
  every ``GET /status`` it already makes (``dispatch.worker_heartbeat``; the worker is untrusted,
  so the values are bounded on the way in);
- editor calls, tokens, cost and the time of the last call: the gateway's ledger
  (``mf_wsp_usage``), the same rows :func:`wsp.runs.metered_usage` reads at the end of a run, plus
  any open reservation at its worst case;
- each budget's cap: the run's frozen request (``request.budgets``). Time is counted from the push,
  as the driver's ``time_budget`` check counts it. Steps are not metered while a session runs:
  they are shown as editor calls against ``max_steps``, labelled an estimate.

**Counts and codes only.** The call timeline serves what ``mf_wsp_usage`` stores, and that table
holds no prompt, answer or upstream message (``WspUsage``; ``test_wsp_calls.py`` pins both the
column set and the served field set).
"""

from __future__ import annotations

import datetime as dt
from collections.abc import Iterable
from decimal import Decimal
from typing import Any

import sqlalchemy as sa
from sqlalchemy.orm import Session

from wsp.models import TERMINAL_STATUSES, WspReservation, WspRun, WspUsage

#: The longest page of the call timeline.
MAX_CALLS_PAGE = 500


def utcnow() -> dt.datetime:
    return dt.datetime.now(dt.UTC)


def _aware(value: dt.datetime | None) -> dt.datetime | None:
    if value is not None and value.tzinfo is None:
        return value.replace(tzinfo=dt.UTC)
    return value


def _iso(value: dt.datetime | None) -> str | None:
    aware = _aware(value)
    return None if aware is None else aware.astimezone(dt.UTC).isoformat().replace("+00:00", "Z")


def _seconds(start: dt.datetime | None, end: dt.datetime) -> float | None:
    begun = _aware(start)
    return None if begun is None else max(0.0, round((end - begun).total_seconds(), 1))


def _not_refused() -> Any:
    return sa.or_(WspUsage.outcome.is_(None), WspUsage.outcome != "refused")


def _metered(session: Session, run_ids: Iterable[str]) -> dict[str, dict[str, Any]]:
    """Per run: metered calls, refused calls, tokens, cost and the last call's time."""
    ids = list(run_ids)
    out: dict[str, dict[str, Any]] = {
        r: {
            "calls": 0,
            "refused": 0,
            "input_tokens": 0,
            "output_tokens": 0,
            "cost_usd": 0.0,
            "last_call_at": None,
        }
        for r in ids
    }
    if not ids:
        return out
    refused = sa.case((WspUsage.outcome == "refused", 1), else_=0)
    for run_id, calls, refusals, tokens_in, tokens_out, cost, last in session.execute(
        sa.select(
            WspUsage.run_id,
            sa.func.count(WspUsage.id) - sa.func.coalesce(sa.func.sum(refused), 0),
            sa.func.coalesce(sa.func.sum(refused), 0),
            sa.func.coalesce(sa.func.sum(WspUsage.input_tokens), 0),
            sa.func.coalesce(sa.func.sum(WspUsage.output_tokens), 0),
            sa.func.coalesce(sa.func.sum(WspUsage.cost_usd), 0),
            sa.func.max(WspUsage.at),
        )
        .where(WspUsage.run_id.in_(ids))
        .group_by(WspUsage.run_id)
    ).all():
        out[run_id] = {
            "calls": int(calls),
            "refused": int(refusals),
            "input_tokens": int(tokens_in),
            "output_tokens": int(tokens_out),
            "cost_usd": float(Decimal(str(cost))),
            "last_call_at": _iso(last),
        }
    return out


def _open(session: Session, run_ids: Iterable[str]) -> dict[str, dict[str, Any]]:
    """Per run: the open reservation (a call in flight) at its worst case, or nothing."""
    out: dict[str, dict[str, Any]] = {}
    ids = list(run_ids)
    if not ids:
        return out
    for run_id, usd, tokens, since in session.execute(
        sa.select(
            WspReservation.run_id,
            sa.func.sum(WspReservation.worst_case_usd),
            sa.func.sum(WspReservation.worst_case_tokens),
            sa.func.min(WspReservation.created_at),
        )
        .where(WspReservation.run_id.in_(ids), WspReservation.settled_at.is_(None))
        .group_by(WspReservation.run_id)
    ).all():
        out[run_id] = {
            "worst_case_tokens": int(tokens),
            "worst_case_usd": float(Decimal(str(usd))),
            "since": _iso(since),
        }
    return out


def _queue(session: Session) -> tuple[list[str], str | None]:
    """The queued runs in the order the driver takes them (oldest first), and the active one."""
    queued = list(
        session.scalars(
            sa.select(WspRun.run_id)
            .where(WspRun.status == "queued", WspRun.deleted_at.is_(None))
            .order_by(WspRun.created_at, WspRun.run_id)
        ).all()
    )
    active = session.scalar(
        sa.select(WspRun.run_id).where(WspRun.status.in_(("pushing", "running", "collecting")))
    )
    return queued, active


def _budget(
    name: str, spent: float | int | None, cap: Any, *, unit: str, estimate: bool = False
) -> dict[str, Any]:
    number = None if cap is None else float(cap)
    return {
        "name": name,
        "unit": unit,
        "spent": spent,
        "cap": None if number is None else (int(number) if number.is_integer() else number),
        "fraction": (
            None
            if number is None or number <= 0 or spent is None
            else round(min(1.0, float(spent) / number), 4)
        ),
        "estimate": estimate,
    }


def progress_of(
    row: WspRun,
    metered: dict[str, Any],
    open_: dict[str, Any] | None,
    queue: tuple[list[str], str | None],
    now: dt.datetime,
) -> dict[str, Any]:
    """One run's progress from its row, its ledger figures and the queue (no query here)."""
    live = row.status not in TERMINAL_STATUSES
    end = now if live else (_aware(row.finished_at) or now)
    clock_from = "started_at" if row.started_at is not None else "created_at"
    budgets = dict((row.request or {}).get("budgets") or {})
    tokens = metered["input_tokens"] + metered["output_tokens"]
    heartbeat = _aware(row.last_status_at)
    queued, active = queue
    position = queued.index(row.run_id) + 1 if row.run_id in queued else None
    return {
        "run_id": row.run_id,
        "status": row.status,
        "live": live,
        "now": _iso(now),
        "created_at": _iso(row.created_at),
        "started_at": _iso(row.started_at),
        "finished_at": _iso(row.finished_at),
        "clock_from": clock_from,
        "elapsed_s": _seconds(row.started_at or row.created_at, end),
        "worker": (
            None
            if heartbeat is None and row.worker_state is None
            else {
                "state": row.worker_state,
                "elapsed_s": row.worker_elapsed_s,
                "last_heartbeat_at": _iso(heartbeat),
                "heartbeat_age_s": (
                    None if heartbeat is None or not live else _seconds(heartbeat, now)
                ),
            }
        ),
        "metered": {**metered, "tokens": tokens},
        "in_flight": open_,
        "budgets": [
            _budget("tokens", tokens, budgets.get("tokens"), unit="tokens"),
            _budget("cost_usd", metered["cost_usd"], budgets.get("cost_usd"), unit="USD"),
            _budget(
                "time_s",
                None if row.pushed_at is None else _seconds(row.pushed_at, end),
                budgets.get("time_s"),
                unit="s",
            ),
            _budget(
                "steps",
                metered["calls"],
                budgets.get("max_steps"),
                unit="editor calls",
                estimate=True,
            ),
        ],
        "queue": (
            None
            if position is None
            else {"position": position, "queued": len(queued), "active_run_id": active}
        ),
    }


def run_progress(session: Session, row: WspRun, now: dt.datetime | None = None) -> dict[str, Any]:
    """``GET /wsp/runs/{id}/progress``: the full progress object, live or final."""
    now = now or utcnow()
    return progress_of(
        row,
        _metered(session, [row.run_id])[row.run_id],
        _open(session, [row.run_id]).get(row.run_id),
        _queue(session) if row.status == "queued" else ([], None),
        now,
    )


def compact(full: dict[str, Any]) -> dict[str, Any]:
    """The list row's form: enough for a row's one line of progress."""
    worker = full["worker"] or {}
    queue = full["queue"] or {}
    return {
        "elapsed_s": full["elapsed_s"],
        "clock_from": full["clock_from"],
        "worker_state": worker.get("state"),
        "heartbeat_age_s": worker.get("heartbeat_age_s"),
        "calls": full["metered"]["calls"],
        "refused": full["metered"]["refused"],
        "tokens": full["metered"]["tokens"],
        "cost_usd": full["metered"]["cost_usd"],
        "last_call_at": full["metered"]["last_call_at"],
        "queue_position": queue.get("position"),
        "budgets": [
            {k: b[k] for k in ("name", "spent", "cap", "fraction", "estimate")}
            for b in full["budgets"]
        ],
    }


def list_progress(
    session: Session, rows: Iterable[WspRun], now: dt.datetime | None = None
) -> dict[str, dict[str, Any]]:
    """The compact progress of every LIVE row (a finished row has none), in four queries."""
    now = now or utcnow()
    live = [r for r in rows if r.status not in TERMINAL_STATUSES]
    if not live:
        return {}
    ids = [r.run_id for r in live]
    metered = _metered(session, ids)
    open_ = _open(session, ids)
    queue = _queue(session)
    return {
        r.run_id: compact(progress_of(r, metered[r.run_id], open_.get(r.run_id), queue, now))
        for r in live
    }


#: The fields of one call, exactly. Counts and codes only: adding a field here is a review.
CALL_FIELDS = (
    "seq",
    "at",
    "outcome",
    "http_status",
    "latency_ms",
    "finish_reason",
    "refusal_code",
    "input_tokens",
    "output_tokens",
    "cost_usd",
)


def run_calls(
    session: Session, row: WspRun, *, after_seq: int | None, limit: int
) -> dict[str, Any]:
    """``GET /wsp/runs/{id}/calls``: one row per call the gateway saw, oldest first, after
    ``after_seq`` (exclusive) when given, so a live page reads only what is new."""
    query = sa.select(WspUsage).where(WspUsage.run_id == row.run_id)
    if after_seq is not None:
        query = query.where(WspUsage.seq > after_seq)
    found = session.scalars(query.order_by(WspUsage.seq).limit(limit + 1)).all()
    page = found[:limit]
    calls = [
        {
            "seq": u.seq,
            "at": _iso(u.at),
            "outcome": u.outcome,
            "http_status": u.http_status,
            "latency_ms": u.latency_ms,
            "finish_reason": u.finish_reason,
            "refusal_code": u.refusal_code,
            "input_tokens": u.input_tokens,
            "output_tokens": u.output_tokens,
            "cost_usd": float(Decimal(str(u.cost_usd))),
        }
        for u in page
    ]
    return {
        "run_id": row.run_id,
        "status": row.status,
        "live": row.status not in TERMINAL_STATUSES,
        "calls": calls,
        "more": len(found) > limit,
    }
