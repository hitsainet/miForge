"""A run's batch records  (FR-002.28, FR-002.30, FR-002.32).

Two read routes - the list and one record - which are what F9's step log renders and what
F6's stop conditions read.

**The list comes from the index and the detail comes from the log.** That is not an
inconsistency, it is ADR-004: the run log is authoritative and the index is a rebuildable
projection over it, so the paginated listing reads the projection (one query, keyset
ordered) and a single record is served from the **truth**. A detail route that read the
index would be serving a projection of a record it could simply read.

The file is named ``signal_batches.py`` rather than ``batches.py``: ``batches`` is a noun
several features could claim, and this is F2's surface over F2's records.
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Annotated, Any

from fastapi import APIRouter, Depends, Query
from sqlalchemy.orm import Session

from api.deps import session_dependency
from core.errors import refuse
from training import run_log
from training.signals import batch_index
from training.signals.settings import get_signal_settings

router = APIRouter(tags=["batches"])

#: Newest-first, which is the mockup's step-log order and the shape a long run needs.
DEFAULT_PAGE = 50


def _log_root() -> Path:
    """Where the authoritative run logs live. F1's one answer, not a second one."""
    from training.tasks import run_log_root

    return run_log_root()


@router.get("/runs/{run_id}/batches", summary="A run's batch records, newest first")
def list_batches(
    run_id: str,
    session: Annotated[Session, Depends(session_dependency)],
    limit: Annotated[int, Query(ge=1)] = DEFAULT_PAGE,
    before: Annotated[datetime | None, Query()] = None,
) -> dict[str, Any]:
    """Paginated, newest first, keyset-ordered on the batch's own timestamp.

    Keyset rather than offset: a run writes batches while the operator is paging, and an
    offset would silently skip or repeat them as the tail grows.
    """
    settings = get_signal_settings()
    if limit > settings.batch_page_size_max:
        raise refuse(
            "batch_page_too_large",
            f"A page of {limit} exceeds the configured maximum of "
            f"{settings.batch_page_size_max}. The cap is named rather than silently applied: "
            f"a response quietly shorter than the page asked for reads as the end of the "
            f"run.",
            limit=limit,
            maximum=settings.batch_page_size_max,
        )
    rows = batch_index.list_batches(session, run_id=run_id, limit=limit, before=before)
    return {
        "run_id": run_id,
        "batches": [_row_payload(row) for row in rows],
        "count": len(rows),
        "limit": limit,
        "next_before": rows[-1]["created_at"].isoformat() if len(rows) == limit else None,
    }


@router.get(
    "/runs/{run_id}/batches/{batch_sha256}",
    summary="One batch record, by the hash the step record cites",
)
def get_batch(run_id: str, batch_sha256: str) -> dict[str, Any]:
    """One record, **read from the authoritative run log** (ADR-004).

    Addressed by the content hash because that hash is the batch's identity and the only
    join between a step and the material it consumed (R-02.14).
    """
    path = run_log.run_log_path(_log_root(), run_id)
    for block in run_log.read_blocks(path, block="batch"):
        record = block.get("batch") or {}
        if record.get("batch_sha256") == batch_sha256:
            return {
                "run_id": run_id,
                "step": block.get("step"),
                "ts": block.get("ts"),
                "batch": record,
            }
    raise refuse(
        "batch_not_found",
        f"Run {run_id!r} has no batch with hash {batch_sha256!r} in its run log. The log is "
        f"the authoritative record: if the index lists it and the log does not, the index is "
        f"stale and a reindex will drop it.",
        run_id=run_id,
        batch_sha256=batch_sha256,
    )


def _row_payload(row: dict[str, Any]) -> dict[str, Any]:
    """One index row on the wire.

    **Every number names its scale and sample** (ADR-012): the acceptance rate never travels
    without the candidate count it was taken over, and the three verdict counts are all
    present so a pending candidate is never read as a rejection.
    """
    return {
        "batch_sha256": row["batch_sha256"],
        "step": row["step"],
        "source": row["source"],
        "resolved_model": row["resolved_model"],
        "parts": row["part_refs"],
        "prompt_ids": row["prompt_ids"],
        "n_prompts": row["n_prompts"],
        "verdicts": {
            "accept": row["n_accept"],
            "reject": row["n_reject"],
            "undecided": row["n_undecided"],
        },
        "selection_health": {
            "candidates_generated": row["candidates_generated"],
            "candidates_admitted": row["candidates_admitted"],
            "acceptance_rate": row["acceptance_rate"],
            "exhausted": row["exhausted"],
        },
        "selector_config_sha256": row["selector_config_sha256"],
        "grader_config_sha256": row["grader_config_sha256"],
        "created_at": row["created_at"].isoformat(),
    }
