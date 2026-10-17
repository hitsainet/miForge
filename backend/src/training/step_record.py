"""Writing a step down, in the order that makes the record true  (FR-001.24 - FR-001.27).

**The order is the requirement**, and it is the one thing in this module that a mutation
test must catch:

1. **the run log** - authoritative (ADR-004);
2. **``mf_steps``** - the index over it, rebuildable;
3. **the socket** - a projection that may be missed entirely;
4. **the heartbeat** - throttled to 60 s, refusing to move a terminal row.

A crash between 1 and 2 costs a ``reindex``. A crash between 2 and 1 would cost **the
record**, and the index would then be the only evidence of a step that the authoritative log
denies ever happened. Inverting the order changes nothing observable until the machine dies,
which is exactly why the test asserts the sequence rather than the outcome.

The socket is last and fire-and-forget. ADR-007 states the precedent in one sentence:
*progress reported only over the socket is invisible to the janitor*, which is how miStudio
reaped a live 5.8-hour job as "worker lost."
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol, cast

import sqlalchemy as sa

from core.errors import refuse
from training import run_log
from training.run_config import RunConfig
from training.step import StepOutcome

log = logging.getLogger(__name__)

#: The event name on the run's Socket.IO room. One event per completed step, carrying the
#: step block exactly as written - so F9 types against ``step-record-v1.json`` rather than
#: against this module's Python.
STEP_EVENT = "step"


def _steps_table() -> sa.Table:
    """``mf_steps`` as a ``Table``.

    Cast for the same reason :mod:`workers.run_record` casts its own: ``Step.__table__`` is
    typed ``FromClause`` on the declarative base, which is not what ``insert()`` and
    ``delete()`` accept under MyPy strict. One cast, in one place, rather than an ignore at
    each call site - and importing the model lazily keeps this module importable while the
    ORM layer is in flight.
    """
    from models.run import Step

    return cast(sa.Table, Step.__table__)


class Emitter(Protocol):
    """Publishes a run event. The foundation's ``api.sockets.emit_run_event``, injected.

    Injected rather than imported at the call site so the ordering test can observe *when*
    the emit happened relative to the durable write, and so a worker with no Redis reachable
    still completes a step - the socket is a convenience.
    """

    def __call__(self, run_id: str, event: str, payload: dict[str, Any]) -> Any: ...


@dataclass
class StepWriter:
    """Writes step records for one run, in the required order.

    Args:
        run_id: bound for the run's lifetime (ADR-021), on every line and every log entry.
        log_root: where run logs live. ``<log_root>/<run_id>/run.jsonl``.
        heartbeat: the foundation's :class:`workers.heartbeat.Heartbeat`, or ``None`` in a
            test that is not exercising liveness.
        emit: the socket emitter, or ``None`` to write no projection.
    """

    run_id: str
    log_root: Path
    heartbeat: Any | None = None
    emit: Emitter | None = None
    #: The order each side effect actually happened in, for this run. **Not telemetry** -
    #: it is what lets the ordering test assert the sequence instead of the result, which
    #: is the only way to catch an inversion whose consequences appear solely after a crash.
    effects: list[str] = field(default_factory=list)

    @property
    def log_path(self) -> Path:
        return run_log.run_log_path(self.log_root, self.run_id)

    def write(
        self,
        *,
        session: Any,
        step: int,
        outcome: StepOutcome,
        cfg: RunConfig,
    ) -> dict[str, Any]:
        """Record one completed step. Returns the block as written.

        Raises:
            Refusal: ``step_record_invalid`` when the assembled block does not validate.
                **Nothing is written and the step fails** - including the index row, which
                is why validation happens before the first side effect rather than between
                two of them.
        """
        record = run_log.step_block(
            run_id=self.run_id,
            step=step,
            loss=outcome.loss,
            objective=outcome.objective,
            learning_rate=cfg.learning_rate,
            wall_time_ms=outcome.wall_time_ms,
            grad_norm=outcome.grad_norm,
            batch_sha256=outcome.batch_sha256,
            # §5.10's two fields. `None` on the default path; F11's stepper sets them on the
            # outcome. Before 2026-10-16 nothing passed them, so an attached run's record
            # carried null timings even when a bracket was held.
            quiesce_hold_ms=outcome.quiesce_hold_ms,
            commit_duration_ms=outcome.commit_duration_ms,
        )

        # 1. DURABLE, AND FIRST. `append` validates before it writes, so an invalid block
        #    leaves no trace anywhere rather than an index row with no record behind it.
        run_log.append(self.log_path, record)
        self.effects.append("log")

        # 2. The index. One row, in the caller's transaction.
        if session is not None:
            self._upsert(session, record)
            self.effects.append("index")

        # 3. The projection. Fire-and-forget, after the record exists.
        if self.emit is not None:
            self.emit(self.run_id, STEP_EVENT, record)
            self.effects.append("socket")

        # 4. Liveness. Throttled to 60 s by the Heartbeat itself and refusing to move a
        #    terminal row; this does not re-implement either rule.
        if self.heartbeat is not None and session is not None:
            self.heartbeat.touch(session)
            self.effects.append("heartbeat")

        log.info(
            "step recorded",
            extra={
                "code": "step_recorded",
                "run_id": self.run_id,
                "step": step,
                "loss": outcome.loss,
                "objective": outcome.objective,
                "grad_norm": outcome.grad_norm,
                "wall_time_ms": round(outcome.wall_time_ms, 3),
                # ADR-012: every number names its scale and sample. A loss without its
                # effective batch size is not a measurement.
                "effective_batch_size": cfg.effective_batch_size,
            },
        )
        return record

    def _upsert(self, session: Any, record: dict[str, Any]) -> None:
        """Write (or replace) the ``mf_steps`` row for this step.

        Replace rather than insert-only because ``reindex`` reruns over the same log, and an
        insert-only path would make a rebuild fail on its second run - which would make
        ADR-016's equivalence test pass once and never again.
        """
        row = run_log.step_rows_from_block(record)
        table = _steps_table()
        session.execute(
            sa.delete(table)
            .where(table.c.run_id == row["run_id"])
            .where(table.c.step == row["step"])
        )
        session.execute(sa.insert(table).values(**row))

    def last_recorded_step(self) -> int | None:
        """The highest step in the **log**, or ``None``.

        Read from the log rather than from the index, because this answers "what is the last
        step that actually happened" - the question asked after a crash, when the index may
        be behind or absent. E-9: *the last recorded step is the last completed one.*
        """
        steps = [record["step"] for record in run_log.read_blocks(self.log_path, block="step")]
        return max(steps) if steps else None


def rebuild_step_index(session: Any, *, log_root: Path, run_id: str) -> int:
    """Rebuild ``mf_steps`` for one run from its run log. Returns the row count.

    **ADR-016 #4 applied to steps.** ``services/reindex.py`` rebuilds the checkpoint and
    adapter indexes from on-disk *manifests*; a step has no manifest and its authoritative
    copy is the run log, so this is the rebuild for ``mf_steps``. Delete-then-insert for the
    whole run, so a rebuild is idempotent and a row the log no longer contains disappears -
    an upsert-only rebuild would leave a step that was rolled back visible forever.
    """
    table = _steps_table()
    path = run_log.run_log_path(log_root, run_id)
    rows = run_log.step_rows(path)
    session.execute(sa.delete(table).where(table.c.run_id == run_id))
    for row in rows:
        session.execute(sa.insert(table).values(**row))
    if not rows and not path.is_file():
        raise refuse(
            "run_log_absent",
            f"No run log at {path}, so {run_id}'s steps cannot be rebuilt. The log is the "
            f"authoritative record (ADR-004): without it the steps are gone, and an empty "
            f"index would assert that the run took none.",
            run_id=run_id,
            path=str(path),
        )
    return len(rows)
