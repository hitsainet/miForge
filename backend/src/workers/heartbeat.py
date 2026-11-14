"""The database heartbeat  (tasks 6.4, 6.4a, 6.4b; ADR-007).

**The database is the truth and the socket is a convenience.** ADR-007 records the
precedent in one sentence: *"progress reported only over the socket is invisible to the
janitor,"* which is how miStudio reaped a live 5.8-hour job as "worker lost." A run that
emits a thousand socket events and never writes this column is, to the janitor,
indistinguishable from a dead worker - and the janitor is right to treat it that way,
because it cannot see the socket at all.

Three properties, each of which has already cost this project or a sibling something:

1. **Throttled to 60 s** (``HEARTBEAT_MIN_INTERVAL_SECONDS``). One ``UPDATE`` per step on
   a loop doing hundreds of steps a minute is write amplification for no information.
2. **It refuses to move a terminal row.** Once a run is ``completed``, ``failed`` or
   ``cancelled`` its row is a historical record. A late heartbeat from a worker that has
   not noticed would resurrect it, and the run list would show a finished run as alive.
   The refusal is a predicate *inside the UPDATE*, not a check before it, so two writers
   racing cannot slip between the read and the write. F6's mutation M-4 breaks this.
3. **It records the longest gap between writes** (task 6.4b), derived from the persisted
   column rather than from in-process memory, so the measurement survives a restart.

**60 s IS A MINIMUM INTERVAL BETWEEN WRITES, NOT A MAXIMUM GAP.** This is the single
sentence most worth carrying out of this module. The loop's own sub-operations exceed it
legitimately - a generation-bound test pass over hundreds of prompts, a 2.4 GB save - so
*any* liveness threshold derived from 60 s reports a healthy run as stalled, on the
most-watched screen in the product, in the false-positive direction of the very failure
that screen exists to catch. It has been tried and withdrawn twice (F9's finding F-15).
The threshold a reader may use is the **janitor's own**, which is why :func:`liveness_of`
publishes it beside the age instead of leaving the client to choose.
"""

from __future__ import annotations

import logging
from datetime import datetime
from typing import Any

import sqlalchemy as sa
from sqlalchemy.orm import Session

from core.config import get_settings
from core.errors import refuse
from workers.run_record import (
    HEARTBEAT_COLUMN,
    HEARTBEAT_LONGEST_GAP_COLUMN,
    RUNS,
    TERMINAL_STATES,
    as_aware,
    require_run,
    rowcount_of,
    utcnow,
)

log = logging.getLogger(__name__)

__all__ = [
    "HEARTBEAT_COLUMN",
    "HEARTBEAT_LONGEST_GAP_COLUMN",
    "Heartbeat",
    "liveness_of",
]


class Heartbeat:
    """The heartbeat for one run.

    Constructed once at task entry and carried through the loop, so the throttle has
    somewhere to remember the last write. The *measurement* does not depend on that
    memory - it is taken from the persisted column - so a heartbeat reconstructed after a
    restart extends the same gap record rather than starting a new one.

    Args:
        run_id: the run this heartbeat beats for.
        min_interval_seconds: override the 60 s throttle. Tests use it; production does
            not, and a feature that finds itself wanting to is describing a different
            mechanism.
    """

    def __init__(self, run_id: str, *, min_interval_seconds: int | None = None) -> None:
        if not run_id:
            raise refuse(
                "heartbeat_run_id_required",
                "A heartbeat must name the run it beats for; an unattributed heartbeat "
                "cannot be distinguished from a dead worker.",
            )
        self.run_id = run_id
        self._min_interval = (
            get_settings().heartbeat_min_interval_seconds
            if min_interval_seconds is None
            else min_interval_seconds
        )
        self._last_write_at: datetime | None = None
        #: Writes made by *this object*. Not persisted: ``mf_runs`` carries no write
        #: counter, and F6's per-boundary ``heartbeat: {writes, longest_gap_ms}`` block is
        #: counted by F6 in its own run-log record. Exposed for that counting and for
        #: tests, never as the authority.
        self.writes = 0

    @property
    def min_interval_seconds(self) -> int:
        return self._min_interval

    @property
    def last_write_at(self) -> datetime | None:
        """When this object last actually wrote. ``None`` before its first write."""
        return self._last_write_at

    def start(self, session: Session) -> bool:
        """Write unconditionally, at loop entry, before any work.

        **Not an optimisation - a correctness requirement for the janitor.** A ``running``
        row whose ``heartbeat_at`` is ``NULL`` is ambiguous: it may be a worker that died
        a millisecond after claiming the run, or a worker two seconds into its first model
        load. The janitor therefore *reports* such a row and refuses to reap it
        (:mod:`workers.janitor`) rather than guessing. Calling this first is what keeps
        that ambiguous case empty in practice.
        """
        return self.touch(session, force=True)

    def touch(
        self,
        session: Session,
        run_id: str | None = None,
        *,
        force: bool = False,
    ) -> bool:
        """Advance the heartbeat if the throttle allows, and return whether it wrote.

        ``run_id`` is accepted positionally because F6's documents spell the call
        ``heartbeat.touch(run_id)``. It is **checked, not honoured**: a heartbeat object
        for one run asked to beat for another is a bug in the caller, and writing the other
        row would attribute one run's liveness to a different one.

        Returns:
            ``True`` if the column moved. ``False`` if the throttle suppressed the write,
            or if the row is terminal and the write was refused.

        Raises:
            Refusal: ``run_not_found`` when there is no such row - an absent row is a wrong
                identifier, not a finished run.
        """
        if run_id is not None and run_id != self.run_id:
            raise refuse(
                "heartbeat_run_id_mismatch",
                f"This heartbeat belongs to run {self.run_id!r} and was asked to beat for "
                f"{run_id!r}. Construct a Heartbeat per run.",
                heartbeat_run_id=self.run_id,
                requested_run_id=run_id,
            )

        now = utcnow()
        if (
            not force
            and self._last_write_at is not None
            and (now - self._last_write_at).total_seconds() < self._min_interval
        ):
            return False

        # The row must exist. Checked before the UPDATE because an UPDATE matching no row
        # is indistinguishable from one refused by the terminal-status predicate, and those
        # two facts need different answers.
        status = require_run(session, self.run_id)

        previous = as_aware(
            session.execute(
                sa.select(RUNS.c.heartbeat_at).where(RUNS.c.id == self.run_id)
            ).scalar_one()
        )
        # None when this is the first write: there is no gap yet, which is not the same
        # fact as a gap of zero. The column stays NULL rather than becoming 0.
        gap_seconds = None if previous is None else max(0, round((now - previous).total_seconds()))

        values: dict[str, Any] = {"heartbeat_at": now}
        if gap_seconds is not None:
            # The maximum, kept in ONE statement. A read-compute-write would lose a
            # concurrent writer's larger value, and NULL has to be handled explicitly
            # because `NULL < 5` is NULL, not true - so a first gap would never be stored.
            values["longest_heartbeat_gap_seconds"] = sa.case(
                (RUNS.c.longest_heartbeat_gap_seconds.is_(None), gap_seconds),
                (gap_seconds > RUNS.c.longest_heartbeat_gap_seconds, gap_seconds),
                else_=RUNS.c.longest_heartbeat_gap_seconds,
            )

        # THE GUARD. The terminal-status predicate lives in the WHERE clause, so a row that
        # became terminal between the read above and this statement is still refused. A
        # `if status in TERMINAL_STATES: return False` before the UPDATE would be a
        # check-then-act race, and what it races with is a run finishing - which is not
        # rare, it is what every run does.
        result = session.execute(
            sa.update(RUNS)
            .where(RUNS.c.id == self.run_id)
            .where(RUNS.c.status.notin_(tuple(TERMINAL_STATES)))
            .values(**values)
        )

        if rowcount_of(result) == 0:
            # Refused, not failed. The run is terminal and its row is a historical record.
            log.warning(
                "heartbeat refused: the run row is terminal",
                extra={
                    "code": "heartbeat_refused_terminal_row",
                    "run_id": self.run_id,
                    "status": status,
                },
            )
            return False

        self._last_write_at = now
        self.writes += 1
        log.debug(
            "heartbeat",
            extra={"run_id": self.run_id, "gap_seconds": gap_seconds, "forced": force},
        )
        return True


# ------------------------------------------------------------------------ the API payload


def liveness_of(record: dict[str, Any], *, now: datetime | None = None) -> dict[str, Any]:
    """The liveness block for a run response  (**task 6.4a's publish half**).

    F6's ``GET /api/v1/runs/{run_id}`` carries this block (FR-006.10, AC-006.46) and F9's
    run screen reads it. It exists because the two alternatives were both written and both
    withdrawn:

    * a screen that trusts socket connectivity shows a **stalled run as live** - liveness
      derived from a transport, which F9's FR-009.4 forbids;
    * a screen that derives a threshold from the 60 s throttle shows a **healthy run as
      stalled** through every test pass and every save, because 60 s is a minimum interval
      between writes and the loop's sub-operations exceed it.

    So the server publishes the age, the longest gap this run has actually shown, **and the
    janitor's own threshold**. The client compares; it does not choose. Two thresholds are
    two answers to "is this run alive", and the one the operator sees would disagree with
    the one that reaps. ``lapsed`` is computed here for the same reason: a client that
    computes it has to know the rule, and then there are two copies of the rule.

    Args:
        record: a row mapping carrying at least ``status``; ``heartbeat_at`` and
            ``longest_heartbeat_gap_seconds`` are read when present.
        now: injectable clock for tests.

    Returns:
        ``{"heartbeat_at", "age_seconds", "longest_gap_seconds", "lapse_threshold_seconds",
        "lapsed"}``. **No key is ever absent** - a missing key is indistinguishable from a
        healthy one, which is the failure this block exists to prevent.

        ``age_seconds`` and ``longest_gap_seconds`` are ``None`` only when the underlying
        column is ``NULL``, and ``None`` is reported rather than rendered as zero: a run
        that has never beaten is not a fresh run, and a run with no observed gap is not a
        run with a gap of nothing.
    """
    settings = get_settings()
    moment = now or utcnow()
    beat = as_aware(record.get(HEARTBEAT_COLUMN))
    threshold = settings.janitor_lapse_seconds
    status = record.get("status")

    age: float | None = None if beat is None else max(0.0, (moment - beat).total_seconds())
    gap = record.get(HEARTBEAT_LONGEST_GAP_COLUMN)

    # A terminal run is not "stalled" - it is finished, and its heartbeat is *expected* to
    # be old. Reporting a completed run as lapsed because nobody beat for it since is the
    # same category of false positive as a threshold derived from the 60 s throttle.
    lapsed = False if status in TERMINAL_STATES else (age is not None and age > threshold)

    return {
        HEARTBEAT_COLUMN: beat.isoformat() if beat is not None else None,
        "age_seconds": None if age is None else round(age, 3),
        "longest_gap_seconds": None if gap is None else int(gap),
        # Published so no reader derives a second threshold. Provisional until 6.4b's
        # longest-observed gaps exist across a full run of each type; the value lives in
        # settings, not here.
        "lapse_threshold_seconds": threshold,
        "lapsed": lapsed,
    }
