"""The worker pool as the driver last read it, and the pool route's document (feature 019 P3;
FR-019.18; F19-D13, D24 reading 2, D32; TID §3.9).

The pass writes one ``mf_wsp_workers`` row per provisioned worker (:func:`record_ready`) inside its
lock, so the route reads the database and never calls a worker (ADR-007). A worker is BUSY while a
run in ``pushing``, ``running`` or ``collecting`` holds it; RESTARTING when the boot it last
answered with has ended (its session was acked, or the driver retired it as idle before its
supervisor's idle recycle, so its pod is being replaced or its container restarted, operator
2026-11-26) and either it still shows that boot or its silence is within
:data:`RESTART_WINDOW_S` of the end; NOT ANSWERING when its last read failed otherwise; else IDLE.
Only an IDLE worker, one answering ``/ready`` on a fresh boot, is ever claimed
(``dispatch._claim``). ``shared_base_urls`` names active runs that share one server: under the key
index only generic rows can (FR-019.8, F19-D6).
"""

from __future__ import annotations

import datetime as dt
from typing import Any

import sqlalchemy as sa
from sqlalchemy.orm import Session

from wsp import scheduling
from wsp.models import (
    HOLDING_STATES,
    WORKER_STATES,
    WspEndpoint,
    WspRun,
    WspWorker,
    WspWorkerBoot,
)
from wsp.settings import WSPSettings

BOOT_MAX = 32
#: How long after its boot ended a silent worker reads RESTARTING rather than NOT ANSWERING: a
#: replaced pod answers within seconds, and a container restart (the fallback when the pod could
#: not be replaced) within the kubelet's 300 s back-off cap. Silent past this, it is a fault.
RESTART_WINDOW_S = 600
ADDRESS_MAX = 256


#: The largest disk figure kept from a worker's ``/ready``: a signed 64-bit column's maximum.
DISK_BYTES_MAX = 2**63 - 1


def disk_bytes(value: Any) -> int | None:
    """A worker-reported byte count, kept only as a non-negative integer the column can hold (a
    bool, a float, a string or a negative number is ``None``): the worker is untrusted."""
    if isinstance(value, bool) or not isinstance(value, int) or not 0 <= value <= DISK_BYTES_MAX:
        return None
    return value


def record_ready(
    session: Session,
    index: int,
    address: str,
    answer: dict[str, Any] | None,
    now: dt.datetime,
) -> WspWorker:
    """Upsert worker ``index``'s row from its ``/ready`` (``None``: it did not answer). The worker
    is untrusted: a state outside ``WORKER_STATES`` is ``unknown``, a boot id is bounded, and its
    ``/work`` disk figures (2026-11-26) are kept only as :func:`disk_bytes` admits them. A worker
    that did not answer keeps its last figures, dated by ``answered_at``."""
    row = session.get(WspWorker, index)
    if row is None:
        row = WspWorker(worker_index=index, address=address[:ADDRESS_MAX], state="unknown")
        session.add(row)
    row.address = address[:ADDRESS_MAX]
    row.read_at = now
    if answer is None:
        row.state = "not_answering"
    else:
        state = answer.get("state")
        row.state = state if isinstance(state, str) and state in WORKER_STATES else "unknown"
        boot = answer.get("boot_id")
        row.boot_id = boot[:BOOT_MAX] if isinstance(boot, str) else None
        row.work_free_bytes = disk_bytes(answer.get("work_free_bytes"))
        row.work_total_bytes = disk_bytes(answer.get("work_total_bytes"))
        row.answered_at = now
    session.flush()
    return row


def _iso(value: dt.datetime | None) -> str | None:
    if value is None:
        return None
    return (value if value.tzinfo is not None else value.replace(tzinfo=dt.UTC)).isoformat()


def _ended_at(session: Session, boot: str | None) -> dt.datetime | None:
    """When ``boot``'s session ended (acked, else finished), or ``None`` while it has not: the
    boot is unconsumed, or its run still holds it."""
    seen = None if boot is None else session.get(WspWorkerBoot, boot)
    if seen is None:
        return None
    run = session.scalar(sa.select(WspRun).where(WspRun.boot_id == boot))
    if run is not None and run.status in HOLDING_STATES:
        return None
    end = None if run is None else (run.acked_at or run.finished_at)
    return _aware(end or seen.first_seen_at)


def _aware(value: dt.datetime | None) -> dt.datetime | None:
    if value is None:
        return None
    return value if value.tzinfo is not None else value.replace(tzinfo=dt.UTC)


def worker_word(session: Session, row: WspWorker, holder: str | None, now: dt.datetime) -> str:
    """busy, restarting, not_answering or idle (the module docstring)."""
    if holder is not None:
        return "busy"
    silent = row.answered_at is None or (row.read_at is not None and row.answered_at < row.read_at)
    ended = _ended_at(session, row.boot_id)
    if ended is not None:
        if not silent:
            return "restarting"  # still showing the ended boot: it is about to exit
        current = _aware(now) or now
        if (current - ended).total_seconds() <= RESTART_WINDOW_S:
            return "restarting"
    return "not_answering" if silent else "idle"


def pool_document(
    session: Session, pool_size: int, wsp: WSPSettings, now: dt.datetime | None = None
) -> dict[str, Any]:
    """``GET /api/v1/wsp/pool`` (TDD §5.3)."""
    now = now or dt.datetime.now(dt.UTC)
    maximum = wsp.wsp_worker_pool_max
    holders = {
        (0 if r.worker_index is None else int(r.worker_index)): r.run_id
        for r in session.scalars(sa.select(WspRun).where(WspRun.status.in_(HOLDING_STATES))).all()
    }
    rows = session.scalars(
        sa.select(WspWorker)
        .where(WspWorker.worker_index < maximum)
        .order_by(WspWorker.worker_index)
    ).all()
    workers = []
    counts = {"busy": 0, "idle": 0, "restarting": 0, "not_answering": 0}
    for row in rows:
        run_id = holders.get(row.worker_index)
        word = worker_word(session, row, run_id, now)
        counts[word] += 1
        workers.append(
            {
                "index": row.worker_index,
                "state": word,
                "worker_state": row.state,
                "run_id": run_id,
                "boot_id": row.boot_id,
                "read_at": _iso(row.read_at),
            }
        )
    queued = session.execute(
        sa.select(WspRun.endpoint_id, sa.func.count())
        .where(WspRun.status == "queued")
        .group_by(WspRun.endpoint_id)
        .order_by(WspRun.endpoint_id)
    ).all()
    shared: dict[str, dict[str, list[str]]] = {}
    for run, endpoint in session.execute(
        sa.select(WspRun, WspEndpoint)
        .join(WspEndpoint, WspEndpoint.id == WspRun.endpoint_id)
        .where(WspRun.status.in_(("pushing", "running")))
        .order_by(WspRun.run_id)
    ).all():
        group = shared.setdefault(
            scheduling.origin_of(endpoint.base_url), {"endpoint_ids": [], "run_ids": []}
        )
        if endpoint.id not in group["endpoint_ids"]:
            group["endpoint_ids"].append(endpoint.id)
        group["run_ids"].append(run.run_id)
    return {
        "pool_size": int(pool_size),
        "maximum": maximum,
        "effective": scheduling.effective_pool(pool_size, maximum),
        **counts,
        "workers": workers,
        "queued_by_endpoint": [
            {"endpoint_id": endpoint_id, "queued": int(n)}
            for endpoint_id, n in queued
            if endpoint_id is not None
        ],
        "queued_without_endpoint": sum(int(n) for endpoint_id, n in queued if endpoint_id is None),
        "shared_base_urls": [
            {"base_url": origin, **group}
            for origin, group in sorted(shared.items())
            if len(group["run_ids"]) >= 2
        ],
    }


def worker_disk(session: Session, wsp: WSPSettings) -> list[dict[str, Any]]:
    """Each provisioned worker's ``/work`` filesystem as its last answered ``/ready`` reported it
    (2026-11-26, for ``GET /host/resources``): free and total bytes (null until it reports them),
    and when it answered. A worker is never called here (ADR-007): the driver's pass wrote these."""
    rows = session.scalars(
        sa.select(WspWorker)
        .where(WspWorker.worker_index < wsp.wsp_worker_pool_max)
        .order_by(WspWorker.worker_index)
    ).all()
    return [
        {
            "index": row.worker_index,
            "boot_id": row.boot_id,
            "free_bytes": row.work_free_bytes,
            "total_bytes": row.work_total_bytes,
            "reported_at": _iso(row.answered_at),
        }
        for row in rows
    ]
