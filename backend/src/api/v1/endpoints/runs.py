"""``GET /api/v1/runs/{run_id}`` - the run record, carrying the liveness block.

Foundation task 6.4a's *publish* half. **This module is the foundation's floor, not the
feature's ceiling:** F1 owns this route and F6 extends it (F6 FR-006.10, AC-006.46), adding
``stop_reason``, ``round``, the condition set, the projected footprint and the rest of the
record. What the foundation puts here is the part neither of them can invent -
``heartbeat`` - and the reason it is here rather than filed as a note is ADR-016's first
rule: *a capability is not shipped until a test fails when its wiring is removed.* A
liveness serialiser with no route calling it is a mechanism declared and not wired, which
is the defect this project has shipped three times.

**Why the block is served and not computed in the client** (F9's finding F-15): a client
that computes liveness has to know the rule, and then there are two copies of it. The two
copies that were actually written and withdrawn were *"the socket is connected, so the run
is live"* - which shows a stalled run as live - and *"the heartbeat is older than 60 s, so
the run is stalled"* - which shows a **healthy** run as stalled through every test pass and
every save, because 60 s is a minimum interval between writes, not a maximum gap. So the
server publishes the age, the run's own longest observed gap, **and the janitor's
threshold**, and the client compares.

**A note for whoever extends this.** Keep ``heartbeat`` on the response. F6 carries a
mutation control for dropping it (M-18) and this feature's reachability test asserts the
key is present with every one of its sub-keys; the block going missing is not a cosmetic
regression, it is the run screen losing its only honest liveness input.
"""

from __future__ import annotations

from typing import Annotated, Any

import sqlalchemy as sa
from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session

from api.deps import session_dependency
from core.errors import refuse
from workers.heartbeat import liveness_of
from workers.run_record import RUNS, as_aware

router = APIRouter(prefix="/runs", tags=["runs"])


def run_payload(record: dict[str, Any], outcome: dict[str, Any] | None = None) -> dict[str, Any]:
    """Serialise a run record. **The only producer of the ``heartbeat`` block.**

    Separated from the route so F1 and F6 extend one function rather than each building
    their own response dict - which is how one of them ends up without the block.
    """
    cancel_requested_at = as_aware(record.get("cancel_requested_at"))
    return {
        # `run_id` on the wire, `id` in the column. The wire name is what every manifest,
        # log line and socket room already calls it, and a response field called `id`
        # beside a `run_id` in the same JSON document would be two names for one value.
        "run_id": record["id"],
        "status": record["status"],
        "run_type": record.get("run_type"),
        "last_step": record.get("last_step"),
        # The timestamp, not a boolean, because the useful question is not only "was a
        # stop asked for" but "how long ago" - which is what tells an operator whether the
        # worker is shutting down or ignoring them.
        "cancel_requested_at": (
            None if cancel_requested_at is None else cancel_requested_at.isoformat()
        ),
        "failure_code": record.get("failure_code"),
        # What the refusal said and its details (run_driver.failure_text). Without it the
        # only diagnostic for a failed run was a code (F11 window 2, 2026-10-17).
        "failure_message": record.get("failure_message"),
        "heartbeat": liveness_of(record),
        # §5.10 (F11 16.4): where this run attached, or null for every other mode. Built
        # from the base URL ONLY - `attachment_id` authorises holder-only calls and never
        # reaches a response (migration 0017; asserted by test_run_attachment_block.py).
        "attachment": _attachment_block(record),
        # How this run acquired its model, off the recorded definition - `null` for a row
        # with none (F1's pre-R-02.53 path). The run view offers "Continue" for an `attach`
        # run (D52), and must not infer the mode from the attachment block, which outlives
        # the attachment.
        "acquisition_mode": _acquisition_mode(record),
        # Archive (operator 2026-10-20): when the run was hidden from the default lists, or
        # null. The run view opens an archived run like any other and offers Unarchive.
        "archived_at": _iso_or_none(record.get("archived_at")),
        # ---- F6's extension (FR-006.10, AC-006.46; R-02.24 - R-02.27, R-02.53) --------
        # The loop's own record: which condition ended the run, how far through the
        # curriculum it is, and what it will consume. `stop_reason` has **no column** on
        # `mf_runs` - the DDL is owed - so it is read from the recorded definition's
        # resolved block where the loop put it, and from the run log where it is
        # authoritative. Absent keys are never used for "unknown": each is present and
        # null, because *a missing key is indistinguishable from a healthy one*.
        **_loop_block(record, outcome),
    }


def _iso_or_none(value: Any) -> str | None:
    aware = as_aware(value)
    return None if aware is None else aware.isoformat()


def _acquisition_mode(record: dict[str, Any]) -> str | None:
    submitted = record.get("definition_submitted")
    source = submitted.get("model_source") if isinstance(submitted, dict) else None
    if not isinstance(source, dict):
        return None
    mode = source.get("mode")
    # A recorded definition with no `mode` was submitted with the default, which is `load`
    # (R-02.46) - stated rather than reported as unknown.
    return str(mode) if mode else "load"


def _attachment_block(record: dict[str, Any]) -> dict[str, Any] | None:
    """The run's §5.10 attachment, for the shell's attachment chip. ``None`` = not attached.

    The live half is ``mf_runs.hotupdate_status`` (migration 0020): miLLM's status as the
    TRAINER last observed it, written by ``training/hotupdate`` into the run record it
    already heartbeats (operator decision 2026-10-16). This route only reads a column - it
    never asks miLLM, so the core path gains no sibling call and no second importer of the
    fenced package. Absent an observation every field is ``null`` ("not reported"), never a
    guessed ``clean``; ``observed_at`` says how old the observation is.
    """
    base_url = record.get("hotupdate_base_url")
    if not base_url:
        return None
    observed = record.get("hotupdate_status")
    status = observed if isinstance(observed, dict) else {}
    return {
        "serving_base_url": str(base_url),
        "resident_model": status.get("resident_model"),
        "weights_state": status.get("weights_state"),
        "holder": status.get("holder"),
        "stale": status.get("stale"),
        "attached": status.get("attached"),
        "observed_at": status.get("observed_at"),
    }


def _loop_block(record: dict[str, Any], outcome: dict[str, Any] | None = None) -> dict[str, Any]:
    """F6's fields on the run response.

    Read out of ``definition_resolved`` rather than from columns of their own, because
    ``mf_runs`` has none yet and the resolved definition is where the loop records the
    cadences, the condition set and the projected footprint (FR-006.61). The DDL for the
    index columns is reported; nothing is inferred and nothing is defaulted.
    """
    resolved = record.get("definition_resolved") or {}
    loop_block = resolved.get("loop") or {}
    # `stop_reason`, `round` and `all_fired` come from the **run log's last boundary
    # block**, not from a column: `mf_runs` has none and the DDL is owed. The log is
    # authoritative either way (ADR-004), and the reason is a function of the block - the
    # array carries every condition's `triggered` and the precedence tuple is
    # deterministic. `None` here means the caller passed no outcome, which is a different
    # fact from "the run has not ended" and is why every key is present and null.
    ended = dict(outcome or {})
    if not ended.get("stop_reason") and record.get("stop_reason"):
        # The log has no boundary block (a run stopped before it started), so the row's
        # columns (migration 0016, written by run_state.mark_completed) answer instead.
        ended["stop_reason"] = record.get("stop_reason")
        ended["stop_detail"] = record.get("stop_detail")
    return {
        "run_type": record.get("run_type"),
        "stop_reason": ended.get("stop_reason"),
        "stop_detail": ended.get("stop_detail"),
        "all_fired": ended.get("all_fired") or [],
        "round": ended.get("round"),
        "stop_conditions": loop_block.get("stop_conditions"),
        "cadences": loop_block.get("cadences"),
        "retention_exposure_steps": loop_block.get("retention_exposure_steps"),
        "footprint": resolved.get("footprint"),
        "seed": resolved.get("seed"),
        "domain": resolved.get("domain"),
    }


@router.get("/{run_id}", summary="One run's record, with its liveness block")
def get_run(
    run_id: str, session: Annotated[Session, Depends(session_dependency)]
) -> dict[str, Any]:
    """Return the run record, or refuse by name when there is no such run."""
    row = session.execute(
        sa.select(
            RUNS.c.id,
            RUNS.c.status,
            RUNS.c.run_type,
            RUNS.c.last_step,
            RUNS.c.cancel_requested_at,
            RUNS.c.failure_code,
            RUNS.c.failure_message,
            RUNS.c.stop_reason,
            RUNS.c.stop_detail,
            RUNS.c.heartbeat_at,
            RUNS.c.longest_heartbeat_gap_seconds,
            # F6: the resolved definition carries the loop's own record (R-02.53).
            RUNS.c.definition_resolved,
            # The submitted definition's model source names the acquisition mode.
            RUNS.c.definition_submitted,
            # §5.10: where the run attached. NOT `attachment_id` - a credential in all but name.
            RUNS.c.hotupdate_base_url,
            RUNS.c.hotupdate_status,
            RUNS.c.archived_at,
        ).where(RUNS.c.id == run_id)
    ).one_or_none()

    if row is None:
        # Refused by name. A 404 with an empty body would leave the caller unable to tell
        # "no such run" from "this route does not exist", and those have different fixes.
        raise refuse(
            "run_not_found",
            f"No run with id {run_id!r}. Check the identifier.",
            run_id=run_id,
        )

    record = dict(row._mapping)
    # Read from the log, not from a column. One file, tailed - and the alternative was
    # leaving `stop_reason` permanently null on the most-watched screen in the product
    # while the value sat in the authoritative record a directory away.
    from services.run_launch import last_logged_step, terminal_outcome

    if record.get("last_step") is None:
        # The column has no production writer; the log is the record (ADR-004).
        record["last_step"] = last_logged_step(run_id)
    return run_payload(record, terminal_outcome(run_id))
