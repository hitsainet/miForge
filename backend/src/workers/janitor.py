"""The janitor: reap lapsed runs, and distinguish *lapsed* from *slow*  (task 6.5; ADR-007).

A worker can die without telling anyone - OOM-killed, node drained, pod evicted mid-step.
Its ``mf_runs`` row stays ``running`` forever, holding a GPU lease and showing on the run
list as live work. Something has to notice, and the only evidence available is the
heartbeat column.

**The distinction this module is judged on.** ADR-007 asks it to tell *lapsed* from
*slow*, and the two look identical from every angle except one:

* **slow** - the heartbeat is recent, but no step has been recorded for twenty minutes.
  That is a generation-bound test pass, or a 2.4 GB save, or a model load. **Not reaped.**
  Step staleness is *not* an input to this decision, and adding it would reproduce
  miStudio's 5.8-hour failure with the sign flipped.
* **lapsed** - the heartbeat itself has not moved for longer than
  ``JANITOR_LAPSE_SECONDS``. **Reaped**, with ``failure_code = worker_lost``.

Socket activity is not an input either, and cannot be: the janitor runs in a different
process from the socket server and has no view of it. That asymmetry *is* ADR-007's
precedent - *"progress reported only over the socket is invisible to the janitor"* - and
the correct response is for the loop to write the column, not for the janitor to learn
about sockets.

**A reap must set ``finished_at``.** ``mf_runs`` carries a check constraint,
``terminal_iff_finished``, asserting that a terminal status and a non-null ``finished_at``
are the same fact. 4.2's reasoning is that *"a terminal row with no finished_at is how a
janitor ends up unable to tell a finished run from one it should reap"* - so the constraint
exists to protect this module from itself, and a reap that set only the status would raise
rather than corrupt the record. Noticed by reading 4.2's constraints; the first draft of
this module set the status alone.

**The threshold has no measured value yet.** ``JANITOR_LAPSE_SECONDS`` is provisional at
900 s; the measurement that sets it is task 6.4b's longest observed gap, recorded per run
and published on the run response. Nothing else in the product can produce it. Until it
exists the threshold is a guess with a margin, and that is stated rather than hidden.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime, timedelta

import sqlalchemy as sa
from sqlalchemy.orm import Session

from core.config import get_settings
from workers.run_record import (
    ERROR_WORKER_LOST,
    RUNS,
    STATE_FAILED,
    STATE_RUNNING,
    TERMINAL_STATES,
    as_aware,
    rowcount_of,
    utcnow,
)

log = logging.getLogger(__name__)


@dataclass
class SweepReport:
    """What one sweep did, and what it deliberately did not do.

    The two "declined" fields are the point. A janitor that reports only its kills is
    unauditable: the question asked after an incident is always *"why did it not reap
    this"* or *"why did it reap that"*, and only one of those is answerable from a list of
    kills.
    """

    reaped: list[str] = field(default_factory=list)
    #: ``running`` rows whose heartbeat is present and fresh. Alive, or slow - and this
    #: module cannot and must not tell those apart.
    alive: list[str] = field(default_factory=list)
    #: ``running`` rows with ``heartbeat_at IS NULL``. **Reported, never reaped.** The row
    #: is genuinely ambiguous: a worker that died immediately after claiming the run looks
    #: exactly like a worker two seconds into its first model load. Reaping on a guess
    #: would kill live runs at startup; ignoring it silently would hide an immortal row.
    #: :meth:`workers.heartbeat.Heartbeat.start` is what keeps this list empty in
    #: practice, by writing once at loop entry before any work.
    running_without_heartbeat: list[str] = field(default_factory=list)
    lapse_threshold_seconds: int = 0
    swept_at: datetime | None = None

    @property
    def reaped_count(self) -> int:
        return len(self.reaped)


def sweep(
    session: Session,
    *,
    lapse_seconds: int | None = None,
    now: datetime | None = None,
) -> SweepReport:
    """One pass: reap every ``running`` row whose heartbeat has lapsed.

    Args:
        session: the session to work in. The caller commits.
        lapse_seconds: override the configured threshold. Tests use it.
        now: injectable clock.

    Returns:
        A :class:`SweepReport` naming what was reaped, what was left alive, and what was
        declined for ambiguity.
    """
    settings = get_settings()
    threshold = settings.janitor_lapse_seconds if lapse_seconds is None else lapse_seconds
    moment = now or utcnow()
    cutoff = moment - timedelta(seconds=threshold)

    report = SweepReport(lapse_threshold_seconds=threshold, swept_at=moment)

    # Only `running` rows are candidates.
    #
    # A `queued` row has never heartbeated by definition - it is waiting for a worker to
    # pick it up - so reaping on heartbeat age would reap every run that waited longer
    # than the threshold in the queue. That is the opposite of this module's job.
    # A terminal row is a historical record and is never touched.
    candidates = session.execute(
        sa.select(RUNS.c.id, RUNS.c.heartbeat_at).where(RUNS.c.status == STATE_RUNNING)
    ).all()

    for run_id, heartbeat_at in candidates:
        if heartbeat_at is None:
            report.running_without_heartbeat.append(str(run_id))
            log.warning(
                "janitor declined to reap a running run with no heartbeat ever written",
                extra={
                    "code": "janitor_declined_no_heartbeat",
                    "run_id": str(run_id),
                    "reason": "indistinguishable from a worker still in its first step",
                },
            )
            continue

        beat = as_aware(heartbeat_at)
        assert beat is not None
        if beat > cutoff:
            # Fresh. Alive, or slow. NOTE: step staleness is not consulted here and must
            # not be - a long save is slow, not dead.
            report.alive.append(str(run_id))
            continue

        age = (moment - beat).total_seconds()
        # The terminal predicate is in the WHERE clause for the same reason it is in the
        # heartbeat's: a run that completes between the SELECT above and this UPDATE must
        # not be overwritten as failed. A completed run marked `worker_lost` is a lie that
        # survives in the record.
        result = session.execute(
            sa.update(RUNS)
            .where(RUNS.c.id == run_id)
            .where(RUNS.c.status.notin_(tuple(TERMINAL_STATES)))
            .values(
                status=STATE_FAILED,
                failure_code=ERROR_WORKER_LOST,
                failure_message=(
                    f"No heartbeat for {round(age)}s, past the {threshold}s lapse "
                    f"threshold. The worker is presumed gone. Continue from the last "
                    f"checkpoint (R-02.22) rather than retrying the interrupted step."
                ),
                # REQUIRED by the `terminal_iff_finished` check constraint: a terminal
                # status without a finish time is rejected by the database. Setting the
                # status alone raises here rather than writing a row the run list cannot
                # interpret.
                finished_at=moment,
                updated_at=moment,
            )
        )
        if rowcount_of(result) == 0:
            log.info(
                "janitor skipped a run that reached a terminal state during the sweep",
                extra={"code": "janitor_skipped_raced", "run_id": str(run_id)},
            )
            continue

        report.reaped.append(str(run_id))
        log.warning(
            "janitor reaped a run whose heartbeat lapsed",
            extra={
                "code": ERROR_WORKER_LOST,
                "run_id": str(run_id),
                "heartbeat_age_seconds": round(age, 1),
                "lapse_threshold_seconds": threshold,
            },
        )

    log.info(
        "janitor sweep complete",
        extra={
            "reaped": report.reaped_count,
            "alive": len(report.alive),
            "running_without_heartbeat": len(report.running_without_heartbeat),
            "lapse_threshold_seconds": threshold,
        },
    )
    return report
