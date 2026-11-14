"""Cooperative cancellation  (task 6.3; ADR-006, ADR-007).

**Never ``revoke(terminate=True)``.** ADR-006 fixes every worker at ``--pool=solo -c 1``,
and that choice has two consequences this module exists to live with:

* ``revoke(terminate=True)`` signals a **pool child**, and the solo pool has none. The
  call returns successfully and nothing happens - the worst shape of failure, because the
  operator sees a cancellation that was accepted.
* **A busy solo worker never reads the control queue.** So even a non-terminating revoke
  is not delivered until the task it was meant to stop has finished.

Therefore a stop request is a **column on the run row** - ``mf_runs.cancel_requested_at``
- and the worker re-reads it at every step boundary. ``workers/`` never imports Celery's
control API; a guard test walks the AST of every module under ``backend/src/`` and refuses
a ``revoke`` call carrying a truthy ``terminate``.

**The column is a timestamp, not a flag**, which is 4.2's choice and the better one: a flag
answers *"should I stop"*, a timestamp also answers *"when was it asked"* - and that second
answer is what makes a slow shutdown distinguishable from an ignored request. It follows
that a second stop request must **not** overwrite the first one's timestamp, or the only
record of how long the worker took to notice is destroyed by the operator asking again.

**The boundary, and why the step is the unit.** A step either completes and is recorded,
or has not started - never half (ADR-007). R-02.31 is the reason it must be the step and
not something finer: *an interrupted step is never retried*, because re-applying a
non-idempotent AdamW update to a half-written parameter set compounds the damage rather
than repairing it. So the only safe place to stop is between two steps.

**The read must be in its own fresh transaction.** A loop that holds one long transaction
sees the snapshot from when that transaction began, so the flag written by the API after
that moment is invisible for the whole run. F6 calls breaking this mutation M-2, and it
fails *silently* - the run simply never stops, and nothing in the logs says why.
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from contextlib import AbstractContextManager
from datetime import datetime

import sqlalchemy as sa
from sqlalchemy.orm import Session

from workers.run_record import (
    RUNS,
    TERMINAL_STATES,
    as_aware,
    default_session_scope,
    require_run,
    rowcount_of,
    utcnow,
)

log = logging.getLogger(__name__)


class RunCancelled(BaseException):
    """Raised at a step boundary when a stop has been requested.

    **A ``BaseException`` subclass, deliberately.** The training loop is full of
    ``except Exception`` - around a provider call, a tokenizer, a file write - and any one
    of them would swallow a cancellation derived from ``Exception``, turning an operator
    stop into a run that logs a warning and keeps going. Inheriting from ``BaseException``
    means the only thing that can catch this is code that names it.

    It carries the ``run_id`` because the handler that converts it into a terminal row and
    a ``stop_reason`` needs to know which run it is terminating, and reading that from
    ambient state is how the wrong row gets moved.
    """

    def __init__(self, run_id: str, reason: str = "operator_stop") -> None:
        self.run_id = run_id
        self.reason = reason
        super().__init__(f"run {run_id} cancelled: {reason}")


def stop_requested(session: Session, run_id: str) -> bool:
    """Whether a stop has been requested for this run, read from the given session.

    The low-level read. Prefer :class:`CancellationToken`, which owns the fresh-transaction
    discipline; this is here for the API layer, which is already inside a request-scoped
    transaction of its own.
    """
    require_run(session, run_id)
    return (
        session.execute(
            sa.select(RUNS.c.cancel_requested_at).where(RUNS.c.id == run_id)
        ).scalar_one()
        is not None
    )


def stop_requested_at(session: Session, run_id: str) -> datetime | None:
    """*When* the stop was requested, or ``None``.

    Separate from :func:`stop_requested` because the two questions have different readers:
    the worker only needs the boolean, while an operator asking "why is this still running"
    needs the time. Collapsing them to the boolean is what a flag column would have forced.
    """
    require_run(session, run_id)
    value = session.execute(
        sa.select(RUNS.c.cancel_requested_at).where(RUNS.c.id == run_id)
    ).scalar_one()
    return as_aware(value)


def request_stop(session: Session, run_id: str, *, now: datetime | None = None) -> bool:
    """Record a stop request. Returns whether a stop is now pending for this run.

    **Idempotent, and it preserves the first request's timestamp.** Asking twice returns
    ``True`` both times and moves the column only once: the recorded moment is when the
    operator *first* asked, which is the number that says how long the worker took to
    notice. Overwriting it on every press would make a worker that ignored a request for
    ten minutes look like one that stopped promptly.

    Refuses on a terminal row by predicate, for the same reason the heartbeat does: a stop
    against a finished run is a no-op, and reporting it as accepted would leave the operator
    waiting for a transition that will never come.

    This is the *recording* half only. F6 owns the requesting identity's audit label and
    the route; the foundation owns the column and the worker's reading of it.
    """
    status = require_run(session, run_id)
    if status in TERMINAL_STATES:
        log.info(
            "stop request ignored: the run row is already terminal",
            extra={"code": "stop_ignored_terminal_row", "run_id": run_id, "status": status},
        )
        return False

    result = session.execute(
        sa.update(RUNS)
        .where(RUNS.c.id == run_id)
        .where(RUNS.c.status.notin_(tuple(TERMINAL_STATES)))
        # The null predicate is what makes this idempotent *in SQL* rather than by a prior
        # read: two operators pressing stop together both match, but only the first finds
        # the column null, so the later write affects no row.
        .where(RUNS.c.cancel_requested_at.is_(None))
        .values(cancel_requested_at=now or utcnow())
    )
    if rowcount_of(result) == 0:
        # NO ROW MOVED: either a request was already outstanding (the collapse - still
        # pending), or the row went terminal between the read above and the UPDATE. The
        # second is not hypothetical: in F11 window 5 the UPDATE waited on the worker's row
        # lock for the whole run, found the row `completed` when the lock was released, and
        # the stop was reported pending with `cancel_requested_at: null` (18.5). Re-read
        # rather than assume - this statement sees what the UPDATE saw.
        current = session.execute(
            sa.select(RUNS.c.status, RUNS.c.cancel_requested_at).where(RUNS.c.id == run_id)
        ).one()
        if current.status in TERMINAL_STATES:
            log.info(
                "stop request ignored: the run finished before the request landed",
                extra={
                    "code": "stop_ignored_terminal_row",
                    "run_id": run_id,
                    "status": current.status,
                },
            )
            return False
        return current.cancel_requested_at is not None
    return True


class CancellationToken:
    """Checked at a step boundary; raises :class:`RunCancelled` when a stop is requested.

    Each :meth:`check` opens its **own short transaction**, which is the whole point of
    the class. The alternative - reading the flag from the session the loop is already
    holding - returns a snapshot taken before the stop was written, so the loop never sees
    it. That failure is invisible: no error, no log line, just a run that does not stop.

    Args:
        run_id: the run being stepped.
        session_scope: a callable returning a context manager yielding a ``Session``.
            Defaults to ``db.session.session_scope``. Injected so the tests - and any
            caller that must not depend on the process-wide engine - can supply their own.
    """

    def __init__(
        self,
        run_id: str,
        *,
        session_scope: Callable[[], AbstractContextManager[Session]] | None = None,
    ) -> None:
        self.run_id = run_id
        self._scope = session_scope or default_session_scope
        self.checks = 0

    def is_cancelled(self) -> bool:
        """Read the flag in a fresh transaction. Does not raise."""
        self.checks += 1
        with self._scope() as session:
            return stop_requested(session, self.run_id)

    def check(self) -> None:
        """Raise :class:`RunCancelled` if a stop has been requested.

        Call this **between** steps, never inside one. Calling it mid-step would unwind
        through a partially applied optimizer update, which is the state R-02.31 forbids
        reaching.
        """
        if self.is_cancelled():
            log.info(
                "cancellation observed at a step boundary",
                extra={"code": "run_cancelled", "run_id": self.run_id},
            )
            raise RunCancelled(self.run_id)
