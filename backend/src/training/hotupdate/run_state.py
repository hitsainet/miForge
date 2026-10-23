"""Where the attachment id lives  (FR-011.5; TDD §4.2; **MUT-12**).

**The id must survive a worker restart and cannot be re-fetched.** miLLM returns it once and
no read path returns it again - not status, not health, not a log line (miLLM Feature 031
FR-31.1.6) - so losing it loses the ability to commit, and the symptom appears a step later
as a refused commit with no visible connection to the restart. A module global or a
request-scoped variable loses it on the first restart. It therefore belongs on the run row.

## THE TWO COLUMNS DO NOT EXIST YET, AND THAT IS REPORTED RATHER THAN WORKED AROUND

``mf_runs`` has no ``attachment_id`` and no ``hotupdate_base_url``. F11 does not write
Alembic revisions - the migration graph is serialised through the orchestrator, because two
revisions off one parent break ``alembic upgrade head`` for everyone - so the exact DDL is
in this feature's report and **this module refuses by name** until it lands:
``hotupdate_run_columns_missing``, naming both columns.

**Why a refusal rather than a fallback.** The obvious workaround is to keep the id in memory
"for now". That is MUT-12 shipped on purpose: it works in every test, works in production
until the first worker restart, and then produces a refused commit whose cause nothing
explains. ADR-018's rule applies to storage as much as to values - *a fabricated durability
is worse than a refusal, because it is silent.*

``tests/unit/test_hotupdate_run_state.py`` carries the **tripwire**: it asserts the columns
are absent today, so the moment the migration lands that test goes red and says to delete
itself. The round trip is proved against a table carrying the columns, so the code is
exercised rather than merely written.
"""

from __future__ import annotations

from typing import Any

import sqlalchemy as sa

from core.errors import refuse

#: The two columns, by name. Both nullable, both null for every other acquisition mode.
ATTACHMENT_ID_COLUMN = "attachment_id"
BASE_URL_COLUMN = "hotupdate_base_url"
REQUIRED_COLUMNS: tuple[str, ...] = (ATTACHMENT_ID_COLUMN, BASE_URL_COLUMN)

#: The DDL this feature owes, as the report states it. Kept here as data so the refusal can
#: hand it to whoever hits it instead of sending them to a document.
OWED_DDL = (
    "ALTER TABLE mf_runs ADD COLUMN hotupdate_base_url VARCHAR(255);\n"
    "ALTER TABLE mf_runs ADD COLUMN attachment_id VARCHAR(128);"
)


def _runs_table() -> sa.Table:
    """The production table object, so this module and the ORM cannot disagree.

    ``__table__`` is typed ``FromClause`` by SQLAlchemy's stubs although a declarative
    model's is always a ``Table``; the cast keeps MyPy strict happy without widening this
    module's own signatures, which the update and select below need to be a ``Table`` for.
    """
    from typing import cast

    from models.run import Run

    return cast(sa.Table, Run.__table__)


def columns_present(table: sa.Table | None = None) -> bool:
    """Whether ``mf_runs`` can hold §5.10's run state yet."""
    columns = (table if table is not None else _runs_table()).c
    return all(name in columns for name in REQUIRED_COLUMNS)


def _require_columns(table: sa.Table) -> None:
    missing = [name for name in REQUIRED_COLUMNS if name not in table.c]
    if missing:
        raise refuse(
            "hotupdate_run_columns_missing",
            f"The optional hot-update mode needs {missing} on mf_runs and they are not "
            f"there, so the attachment id cannot be persisted - and it cannot be re-fetched "
            f"either: miLLM returns it once and no read path returns it again. Refusing "
            f"rather than holding it in memory: that works until the first worker restart "
            f"and then produces a refused commit nothing explains. Run the database "
            f"migrations; every other acquisition mode is unaffected.",
            missing=missing,
            ddl=OWED_DDL,
        )


def remember(
    session: Any,
    run_id: str,
    *,
    attachment_id: str,
    base_url: str | None,
    table: sa.Table | None = None,
) -> None:
    """Persist the attachment id **before any other call is made** (FR-011.5).

    Raises:
        Refusal: ``hotupdate_run_columns_missing`` until the migration lands;
            ``hotupdate_run_not_found`` when the row is not there - which would mean the id
            was held by nothing, and a commit later refused for a reason nobody could trace.
    """
    runs = table if table is not None else _runs_table()
    _require_columns(runs)
    result = session.execute(
        sa.update(runs)
        .where(runs.c.id == run_id)
        .values(**{ATTACHMENT_ID_COLUMN: attachment_id, BASE_URL_COLUMN: base_url})
    )
    if result.rowcount == 0:
        raise refuse(
            "hotupdate_run_not_found",
            f"Run {run_id!r} has no row, so the attachment id has nowhere to live. Refusing "
            f"rather than continuing: the id cannot be re-fetched, so a commit would be "
            f"refused later for a reason nothing connects to this moment.",
            run_id=run_id,
        )


def recall(
    session: Any, run_id: str, *, table: sa.Table | None = None
) -> tuple[str | None, str | None]:
    """``(attachment_id, base_url)`` for a run, read back after a worker restart.

    ``None`` for the id is a statement that this run holds no attachment - which is the
    normal answer for every other acquisition mode - and **never** a reason to attach again:
    a second attach while one is active is refused ``409 ALREADY_ATTACHED``, and the holder
    may be mid-commit (R-02.34).
    """
    runs = table if table is not None else _runs_table()
    _require_columns(runs)
    row = session.execute(
        sa.select(runs.c[ATTACHMENT_ID_COLUMN], runs.c[BASE_URL_COLUMN]).where(runs.c.id == run_id)
    ).first()
    if row is None:
        return None, None
    return row[0], row[1]


def forget(session: Any, run_id: str, *, table: sa.Table | None = None) -> None:
    """Clear the id after a clean detach, so a finished run holds no live capability."""
    runs = table if table is not None else _runs_table()
    _require_columns(runs)
    session.execute(
        sa.update(runs).where(runs.c.id == run_id).values(**{ATTACHMENT_ID_COLUMN: None})
    )


STATUS_COLUMN = "hotupdate_status"


def status_record(body: Any, *, holds_attachment: bool) -> dict[str, Any]:
    """The run-record form of miLLM's status body (R-01.20), as the chip renders it.

    ``holder`` is the attaching identity **only when it is not this run** - a chip that told
    the operator "held by trainer" about their own run would read as contention.
    """
    from datetime import UTC, datetime

    data = body if isinstance(body, dict) else {}
    resident = data.get("resident_model")
    name = resident.get("name") if isinstance(resident, dict) else None
    # miLLM's holder is the `attachment` OBJECT (role, label, state), read by one helper.
    from training.hotupdate import state

    holder = state.holder_of(data)
    return {
        "attached": bool(data.get("attached")),
        "weights_state": data.get("weights_state"),
        "resident_model": name if isinstance(name, str) and name else None,
        "holder": None if holds_attachment or not holder else holder,
        "stale": state.attachment_stale(data),
        "observed_at": datetime.now(UTC).isoformat(),
    }


def record_status(
    session: Any, run_id: str, record: dict[str, Any], *, table: sa.Table | None = None
) -> None:
    """Write the observed attachment state into the run row (migration 0020).

    The API serves it read-only; nothing in the core path imports this package to read it.
    """
    runs = table if table is not None else _runs_table()
    if STATUS_COLUMN not in runs.c:
        raise refuse(
            "hotupdate_run_columns_missing",
            "mf_runs has no hotupdate_status column; run migration 0020.",
            missing=[STATUS_COLUMN],
        )
    session.execute(sa.update(runs).where(runs.c.id == run_id).values({STATUS_COLUMN: record}))
