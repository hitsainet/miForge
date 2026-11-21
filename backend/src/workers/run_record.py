"""The run record as a protocol  (tasks 6.3-6.7; ADR-006, ADR-007, ADR-008).

ADR-007's title is the design: *the run record is the protocol.* Every mechanism in this
package - the heartbeat, cooperative cancellation, the janitor, the GPU lease - reaches a
running job through **columns on a row**, never through a Celery control message, because
ADR-006's ``--pool=solo -c 1`` means a busy worker never reads the control queue and
``revoke(terminate=True)`` has no pool child to signal.

**One declaration of the schema, and it is not this one.** Foundation task 4.2 owns
``mf_runs`` and ``mf_leases`` - the migration, the ORM classes, the check constraints. This
module imports those tables and builds Core statements against them. It does **not**
re-declare them, and the distinction is the reason this file was rewritten: it originally
carried its own ``sa.Table`` definitions as a defensive seam, written while 4.2 was in
flight, and when 4.2 landed **six of its seven column names were wrong** - ``run_id`` for
``id``, ``state`` for ``status``, a boolean ``stop_requested`` for a
``cancel_requested_at`` timestamp, ``error_code`` for ``failure_code``, milliseconds for
seconds, and a ``heartbeat_writes`` column that does not exist. Every test of it passed,
because the tests built *its* tables. **A fixture that agrees with the code by construction
is how a suite stays green over a real bug** (ADR-016), and a seam tested only against its
own declaration is the purest form of that.

So the aliases below are the *only* translation layer, and each is a plain re-export of a
real column. If 4.2 renames one, this module stops importing and every test in this
package goes red at once - which is the behaviour wanted.

**Three facts from 4.2 that change how this package must behave**, each of which would
have been a silent defect:

1. ``TERMINAL_STATUSES`` includes **``cancelled``**, not just ``completed`` and ``failed``.
   A terminal set missing a member means the heartbeat happily moves a cancelled row.
2. ``mf_runs`` carries a check constraint ``terminal_iff_finished``: a terminal status
   **requires** ``finished_at``. So the janitor must set it when it reaps, or the write
   raises.
3. ``mf_leases`` is **append-and-release**, not a row per card: exclusivity is a *partial
   unique index* over the unreleased rows. A lapsed lease is still unreleased, so it still
   blocks - which is why :mod:`workers.lease` reclaims before it inserts.
"""

from __future__ import annotations

from collections.abc import Callable
from contextlib import AbstractContextManager
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any, cast

import sqlalchemy as sa

from core.errors import refuse
from models.lease import Lease
from models.run import RUN_STATUSES, TERMINAL_STATUSES, Run

if TYPE_CHECKING:  # pragma: no cover - typing only
    from sqlalchemy.orm import Session

# ---------------------------------------------------------------------------- the tables

#: ``mf_runs``, as 4.2 declares it. Core rather than ORM because every write in this
#: package is a conditional ``UPDATE`` whose ``WHERE`` clause carries the policy - the
#: terminal-state guard, the lease predicate - and an ORM attribute assignment cannot
#: express a predicate that two racing writers are serialised by.
# ``cast`` because SQLAlchemy annotates ``__table__`` as ``FromClause``, which is true of
# an ORM class in general but not of these two: both are mapped to a real ``Table``, and
# without the cast every ``update()`` and every ``==`` predicate below is untyped under
# MyPy strict - which would hide a genuine type error in the one place this package cannot
# afford one, the WHERE clauses that carry the terminal-row and lease guards.
RUNS: sa.Table = cast(sa.Table, Run.__table__)
LEASES: sa.Table = cast(sa.Table, Lease.__table__)

# --------------------------------------------------------------------------- the columns
# Re-exported under the names this package uses, so there is exactly one place where a
# 4.2 rename has to be noticed.

#: The run's primary key. **``id``, not ``run_id``** - the ORM's own spelling.
RUN_ID: sa.Column[Any] = RUNS.c.id
RUN_STATUS: sa.Column[Any] = RUNS.c.status

#: Task 6.4a: **the heartbeat column is ``mf_runs.heartbeat_at``.** Named in 4.2 and
#: carried here so the janitor, the liveness block and F9's run screen all read one name.
HEARTBEAT_AT: sa.Column[Any] = RUNS.c.heartbeat_at

#: Task 6.4b: the longest gap actually observed between heartbeat *writes*, **in whole
#: seconds**, nullable.
#:
#: Two things about this column are load-bearing. **Seconds, not milliseconds** - F6's
#: per-boundary block uses ``longest_gap_ms`` and that is a different measurement at a
#: different granularity; this one exists to give ``JANITOR_LAPSE_SECONDS`` a defensible
#: value, so it is in the unit that threshold is expressed in and directly comparable with
#: ``age_seconds`` on the same response. And **nullable means "no gap has been observed
#: yet"**, which is not the same fact as zero: a run that has written once has no gap to
#: report, while a run showing 0 has written twice in under a second. Collapsing them would
#: make a brand-new run look like the smoothest run in the system.
HEARTBEAT_LONGEST_GAP_SECONDS: sa.Column[Any] = RUNS.c.longest_heartbeat_gap_seconds

#: Cooperative cancellation (task 6.3). **A timestamp, not a flag** - 4.2's choice, and the
#: better one: the flag answers "should I stop", the timestamp also answers "when was it
#: asked", which is what makes a slow shutdown distinguishable from an ignored request.
CANCEL_REQUESTED_AT: sa.Column[Any] = RUNS.c.cancel_requested_at

FAILURE_CODE: sa.Column[Any] = RUNS.c.failure_code
FAILURE_MESSAGE: sa.Column[Any] = RUNS.c.failure_message
FINISHED_AT: sa.Column[Any] = RUNS.c.finished_at

#: **The aliases above are names, not operands.** Every statement in this package writes
#: its predicates as ``RUNS.c.<column> == value`` rather than using an alias, for a reason
#: found by running the linter: Ruff's ``SIM300`` ("yoda condition") fix sees
#: ``RUN_ID == run_id`` as a constant on the left and **reverses the operands** to
#: ``run_id == RUN_ID``. That still works at runtime - Python falls back to the column's
#: reflected ``__eq__`` - so the tests stayed green, and only MyPy noticed, typing the
#: result as ``bool`` instead of a SQL expression. A lint autofix that silently rewrites a
#: WHERE clause is not something to leave standing on the clauses carrying the terminal-row
#: and lease guards.
#:
#: Column *names* as strings, for the API layer and for anything reading a row mapping.
HEARTBEAT_COLUMN = HEARTBEAT_AT.name
HEARTBEAT_LONGEST_GAP_COLUMN = HEARTBEAT_LONGEST_GAP_SECONDS.name

# -------------------------------------------------------------------------- the lifecycle

#: From 4.2 / F6 FR-006.13. Re-exported rather than restated: a second copy of a state set
#: is a second answer, and the one this package would get wrong is the terminal set.
STATE_PENDING = "pending"
STATE_RUNNING = "running"
STATE_COMPLETED = "completed"
STATE_FAILED = "failed"
STATE_CANCELLED = "cancelled"

#: **Includes ``cancelled``.** Imported, not written out, because the cost of omitting a
#: member is a heartbeat that resurrects a cancelled run on the run list.
TERMINAL_STATES: frozenset[str] = frozenset(TERMINAL_STATUSES)

assert STATE_RUNNING in RUN_STATUSES, "4.2 renamed the running status"
assert TERMINAL_STATES <= RUN_STATUSES, "4.2's terminal set is not a subset of its statuses"

#: What the janitor writes when it reaps a run whose heartbeat has lapsed. The wording is
#: miStudio's, kept on purpose: *"worker lost"* is the label under which it reaped a live
#: 5.8-hour job, and keeping the name makes the precedent searchable in this codebase.
ERROR_WORKER_LOST = "worker_lost"


def utcnow() -> datetime:
    """Timezone-aware UTC now.

    One function so every timestamp this package writes is comparable with every other. A
    naive datetime compared against an aware one raises in Python and compares *wrong* in
    SQL - and the value being compared here is whether a run is alive.
    """
    return datetime.now(UTC)


def as_aware(value: datetime | None) -> datetime | None:
    """Treat a naive timestamp read back from the database as UTC.

    SQLite - which the tests use - discards the offset, so a value written as aware comes
    back naive. Postgres with ``timestamptz`` does not. Normalising on read makes the gap
    arithmetic identical on both, instead of correct on one and raising on the other.
    """
    if value is None:
        return None
    return value if value.tzinfo is not None else value.replace(tzinfo=UTC)


# ------------------------------------------------------------------- the session provider

#: A callable returning a context manager that yields a ``Session`` and commits on exit -
#: the shape :func:`db.session.session_scope` has.
SessionScope = Callable[[], AbstractContextManager["Session"]]


def default_session_scope() -> AbstractContextManager[Session]:
    """``db.session.session_scope()``.

    Imported at call time so that importing this package does not build an engine. 4.1
    makes the same promise from its own side - *"importing this module must not open a
    socket"* - and this keeps the promise true for anything that imports a worker.
    """
    from db.session import session_scope

    return session_scope()


def require_run(session: Session, run_id: str) -> str:
    """The run's status, refusing by name when the row does not exist.

    An absent row is a different fact from a terminal row and the two must not be
    collapsed: terminal means *stop writing*, absent means *this id is wrong*. Collapsing
    them lets a typo'd ``run_id`` look like a finished run, and the heartbeat then reports
    success for a run nobody is tracking.
    """
    status = session.execute(
        sa.select(RUNS.c.status).where(RUNS.c.id == run_id)
    ).scalar_one_or_none()
    if status is None:
        raise refuse(
            "run_not_found",
            f"No mf_runs row for run id {run_id!r}. Check the identifier; nothing was " f"written.",
            run_id=run_id,
        )
    return str(status)


def rowcount_of(result: Any) -> int:
    """Rows affected by a DML statement.

    A one-line helper because ``Session.execute`` is typed as returning ``Result``, which
    has no ``rowcount`` - only ``CursorResult`` does. Every guard in this package decides
    *refused or written* from this number, so it is worth one named accessor rather than a
    ``type: ignore`` at each of the eight call sites.
    """
    return int(cast("sa.CursorResult[Any]", result).rowcount)


def row_as_dict(row: Any) -> dict[str, Any]:
    """A result row as a plain mapping, for the API layer to serialise."""
    return dict(row._mapping)
