"""A test that asks for two databases gets two  (``tests.support.db``).

**The defect this pins was invisible on SQLite and only CI could show it.**
``migrated_engine(tmp_path / "db-here")`` and ``migrated_engine(tmp_path / "db-there")``
mean *"a second installation, empty"* — F7's export/import criterion says so in terms:
*"'elsewhere' is a second pack root and a second database, both empty, so the import is
about the document that travelled rather than about rows that happened to be shared."*

With ``MIFORGE_TEST_DATABASE_URL`` set, ``database_url`` returned the configured URL for
**every** requested path. So the two engines were one database; the second
``migrated_engine`` call downgraded it to base and re-upgraded it, destroying the rows the
first half of the test had just written; and the failure surfaced as
``MultipleResultsFound`` and a ``pk_mf_contract_parts`` unique violation — two symptoms
that say nothing about the cause. On SQLite the two paths are two files and the test
passed, which is why it had to be CI that found it.

These assertions hold on **both** engines, and that is the point: on SQLite they confirm
the property that was always true, and on PostgreSQL they are the regression test. The
negative control is recorded rather than left to be imagined: reverting
``_configured_url_for`` to ``return configured`` makes
``test_a_row_written_in_one_is_invisible_in_the_other`` fail on PostgreSQL and still pass
on SQLite. Run, not imagined: 2 failed on PostgreSQL, 0 on SQLite.
"""

from __future__ import annotations

from pathlib import Path

import sqlalchemy as sa

from tests.support.db import migrated_engine, session_for


def _count_runs(engine: sa.Engine) -> int:
    with engine.connect() as conn:
        return int(conn.execute(sa.text("SELECT count(*) FROM mf_runs")).scalar_one())


def _insert_run(engine: sa.Engine, run_id: str) -> None:
    from datetime import UTC, datetime

    from models.run import Run

    with session_for(engine) as session:
        session.add(
            Run(
                id=run_id,
                status="pending",
                run_type="change",
                created_at=datetime.now(UTC),
                updated_at=datetime.now(UTC),
            )
        )


def test_two_requested_paths_are_two_distinct_databases(tmp_path: Path) -> None:
    here = migrated_engine(tmp_path / "db-here")
    there = migrated_engine(tmp_path / "db-there")
    try:
        assert str(here.url) != str(there.url), (
            f"both engines point at {here.url}. A test asking for a second, empty database "
            f"got the first one, and the second migration wiped it."
        )
    finally:
        here.dispose()
        there.dispose()


def test_a_row_written_in_one_is_invisible_in_the_other(tmp_path: Path) -> None:
    """The behavioural form. The URLs differing is the mechanism; this is the consequence."""
    here = migrated_engine(tmp_path / "db-here")
    there = migrated_engine(tmp_path / "db-there")
    try:
        _insert_run(here, "run-here")
        assert _count_runs(here) == 1
        assert _count_runs(there) == 0, (
            "a row written to the first database is visible in the second, so the two are "
            "one database and F7's 'second installation' proves nothing about what "
            "travelled in the exported document"
        )
        _insert_run(there, "run-there")
        assert _count_runs(here) == 1, (
            "writing to the second database changed the first - the same collision, in the "
            "other direction"
        )
    finally:
        here.dispose()
        there.dispose()


def test_the_SAME_path_asked_twice_is_RE_MIGRATED_not_reallocated(tmp_path: Path) -> None:
    """The other half of the contract, and it is a reset rather than a new database.

    ``migrated_engine`` downgrades a stamped database to base before upgrading it, so a
    second call on the same path hands back the **same** database, **empty**. That is
    deliberate - its docstring says *"a reused database would otherwise keep the previous
    test's rows, and a reindex-equivalence test seeded twice compares a superset with a
    subset"* - and it is asserted here because the sibling-allocation above must not have
    changed it. Allocating a fresh database for a repeat of one path would be a different
    bug with the same symptom.

    The practical consequence for a test author: build the engine **once** and pass it
    around. Calling ``migrated_engine`` again on the same path discards your rows.
    """
    first = migrated_engine(tmp_path / "db")
    _insert_run(first, "run-1")
    assert _count_runs(first) == 1
    second = migrated_engine(tmp_path / "db")
    try:
        assert str(first.url) == str(second.url), (
            "the same path was given two databases, so a test that rebuilt its engine "
            "would silently start talking to a different one"
        )
        assert _count_runs(second) == 0, (
            "the same path asked twice kept its rows. The reset is what stops a shared CI "
            "database leaking one test's rows into the next."
        )
    finally:
        first.dispose()
        second.dispose()
