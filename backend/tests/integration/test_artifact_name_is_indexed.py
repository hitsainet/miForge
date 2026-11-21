"""``name`` is indexed on ``mf_checkpoints`` and ``mf_adapters``  (F9 task 11.6; FTID section 4).

**This closes the surviving half of F9's task 11.6, and it closes it by measurement rather
than by adding an index.** The finding read *"Index ``name`` on both artifact tables **if
foundation 4.2 has not**, since ``name`` is the join key (FTDD T10)"* - and foundation 4.2
**has**: ``name`` is the single-column PRIMARY KEY of both tables, which is a unique btree
on both engines. :mod:`models.checkpoint` says so in words - *"this is the join key
R-02.29's tree walks, so it is the primary key and therefore indexed by construction rather
than by remembering to"* - and that claim had nothing asserting it. It does now.

**A second ``CREATE INDEX ix_mf_checkpoints_name (name)`` was deliberately NOT added.** It
would be a second btree over a column that already carries a unique one; neither planner
can prefer it (a unique index on ``(name)`` covers every predicate a non-unique index on
``(name)`` covers and is strictly more selective), so it would cost a write on every
``reindex`` insert and buy no read. An index nothing can choose reads as protection while
protecting nothing - the same shape as the duplicate-code check F8 deleted rather than kept
"for safety".

**Why the finding looked true, which is the part worth pinning.** SQLAlchemy's
``Inspector.get_indexes`` **omits the index that backs the primary key**, on SQLite *and*
on PostgreSQL. So anyone who enumerates ``get_indexes("mf_checkpoints")`` sees
``base_name``/``domain_state``/``parent``/``run_id_step`` and concludes, reasonably and
wrongly, that ``name`` is unindexed. :func:`indexes_leading_on` therefore reads the primary
key constraint **as well as** the index list, and
:meth:`TestTheReasonTheFindingLookedTrue` pins the omission so that a future reader meets
the explanation instead of re-filing the finding.

Three further notes on method:

* the assertions run against the **live migrated schema**, never against
  ``Base.metadata``. A model that declares an index its migration never created would pass
  a model-level check and leave production unindexed - which is the drift
  ``test_db_migration.py`` exists for, and this file does not want a second, weaker copy of
  it;
* :meth:`TestTheIndex.test_the_planner_really_uses_an_index_for_a_lookup_by_name` asks the
  **query planner**, with rows present, because a declared index that the planner will not
  use is a declaration, not an index. This project's false-measurement family is four
  members long and every one of them needed a positive check that the measurement happened;
* :meth:`TestTheCheckCanSayNo` is that positive check for this file: a table with a ``name``
  column and nothing indexing it, asserted to come back unindexed. Without it, a helper
  that silently returned "indexed" for everything would look exactly like a clean pass.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
import sqlalchemy as sa

from core.config import reset_settings
from tests.support import db as support
from tests.support import tree as forest

#: The two tables F9's tree composer joins on ``name`` (FTID section 4 note 1: one SELECT
#: per artifact table, joined in memory, so the lookups are by ``name``).
ARTIFACT_TABLES = ("mf_checkpoints", "mf_adapters")


def indexes_leading_on(inspector: Any, table: str, column: str) -> list[tuple[str, str]]:
    """Every index on ``table`` whose **leading** column is ``column``.

    Returns ``(index name, how it was declared)`` pairs. Two things are deliberate:

    **The primary key is read, not only the index list.** See the module docstring: the
    inspector hides the PK-backing index on both engines.

    **"Leading column", not "the only column".** A btree on ``(name, step)`` serves
    ``WHERE name = ?`` just as a btree on ``(name)`` does, so a composite key leading on
    ``name`` satisfies what task 11.6 asked for. Requiring an exact single-column match
    would make this test fail for a schema that was still correct.
    """
    found: list[tuple[str, str]] = []
    primary = inspector.get_pk_constraint(table) or {}
    if list(primary.get("constrained_columns") or [])[:1] == [column]:
        found.append((primary.get("name") or f"<unnamed pk on {table}>", "primary key"))
    for index in inspector.get_indexes(table):
        if list(index.get("column_names") or [])[:1] == [column]:
            found.append((index["name"], "index"))
    return found


def index_used_for_name_lookup(engine: sa.Engine, table: str) -> str:
    """The planner's plan for ``WHERE name = ?``, as one string.

    ``enable_seqscan`` is turned off for the PostgreSQL case, inside the transaction so it
    reverts. That is not a thumb on the scale: on a table of a handful of rows PostgreSQL
    will sequentially scan whatever indexes exist, so leaving it on would assert the
    planner's **cost model** rather than the question being asked, which is whether an
    index exists that *can* serve the predicate. SQLite's planner picks the index at this
    size already and needs no equivalent.
    """
    with engine.begin() as connection:
        if engine.dialect.name == "postgresql":
            connection.execute(sa.text("SET LOCAL enable_seqscan = off"))
            statement = f"EXPLAIN SELECT name FROM {table} WHERE name = 'a-name'"
        else:
            statement = f"EXPLAIN QUERY PLAN SELECT name FROM {table} WHERE name = 'a-name'"
        rows = connection.execute(sa.text(statement)).all()
    return " | ".join(" ".join(str(cell) for cell in row) for row in rows)


#: What an index-using plan looks like on each engine. ``Index Only Scan`` and
#: ``USING COVERING INDEX`` appear because the projection above is the indexed column
#: itself, which is the shape the tree composer's ``SELECT name`` lookups take.
INDEX_IN_PLAN = (
    "Index Scan",
    "Index Only Scan",
    "Bitmap Index Scan",
    "USING INDEX",
    "USING COVERING INDEX",
)


@pytest.fixture()
def rig(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Any:
    """A migrated database with **real artifacts indexed into it**.

    Rows come from ``forest``'s real writers and from ``reindex`` reading the manifests
    back off the disk, not from a hand-built insert: a fixture whose fields agree with the
    reader by construction is how a suite stays green over a real bug (ADR-016), and the
    planner question below is only honest against rows the production path produced.
    """
    checkpoints = tmp_path / "checkpoints"
    checkpoints.mkdir()
    monkeypatch.setenv("CHECKPOINT_ROOT", str(checkpoints))
    reset_settings()
    engine = support.migrated_engine(tmp_path / "db")
    base = forest.root_checkpoint(checkpoints, run_id="run-base", step=40, domain="support")
    forest.branch(checkpoints, parent=base.name, run_id="run-base", step=80)
    forest.adapter(checkpoints, base=base, run_id="run-adapter", step=50)
    forest.indexed(engine, checkpoints, run_ids=("run-base", "run-adapter"))
    try:
        yield engine
    finally:
        reset_settings()


class TestTheIndex:
    """Task 11.6's remnant: ``name`` must be indexed on both artifact tables."""

    @pytest.mark.parametrize("table", ARTIFACT_TABLES)
    def test_name_is_indexed(self, rig: sa.Engine, table: str) -> None:
        """**The property, not the mechanism.** Today it is the primary key; an explicit
        index leading on ``name`` would satisfy this too. What must never happen is
        neither, because every tree walk in R-02.29 resolves a ``parent`` or a
        ``base_name`` to a ``name``."""
        found = indexes_leading_on(sa.inspect(rig), table, "name")
        assert found, (
            f"{table} has no index leading on `name`, which is the join key every "
            f"R-02.29 tree walk resolves against. F9 task 11.6 asked for one; it was "
            f"satisfied by the PRIMARY KEY, so losing it means the primary key moved."
        )

    @pytest.mark.parametrize("table", ARTIFACT_TABLES)
    def test_it_is_the_primary_key_that_does_it_so_no_second_index_is_owed(
        self, rig: sa.Engine, table: str
    ) -> None:
        """The reason task 11.6's ``if foundation 4.2 has not`` is already satisfied.

        Stated as an assertion rather than as a comment so that a change of identity - a
        composite key, a surrogate id - turns this red and sends the next reader back to
        the question of whether an explicit ``ix_<table>_name`` is owed after all.
        """
        found = indexes_leading_on(sa.inspect(rig), table, "name")
        assert found == [(f"pk_{table}", "primary key")], (
            f"{table}'s indexes leading on `name` are {found}. If the primary key moved, "
            f"task 11.6's question reopens and an explicit `ix_{table}_name` may now be "
            f"owed. If a second index on `name` was added beside the primary key, it is a "
            f"duplicate btree no planner can prefer - measured on both engines, see this "
            f"module's docstring - so delete it or amend this test with the measurement "
            f"that justifies it. Either way `test_name_is_indexed` above still passes: "
            f"this assertion is about the mechanism, that one is about the property."
        )

    @pytest.mark.parametrize("table", ARTIFACT_TABLES)
    def test_the_planner_really_uses_an_index_for_a_lookup_by_name(
        self, rig: sa.Engine, table: str
    ) -> None:
        """A declared index the planner will not use is a declaration, not an index."""
        plan = index_used_for_name_lookup(rig, table)
        assert any(marker in plan for marker in INDEX_IN_PLAN), (
            f"the planner scanned {table} rather than using an index for `name = ?`; "
            f"plan was: {plan}"
        )

    def test_the_rows_are_really_there(self, rig: sa.Engine) -> None:
        """The positive check that the fixture above did something.

        An empty table makes the planner question vacuous on SQLite and makes the whole
        file a check nobody performed - which is the construction-green failure this
        project's harness was written after.
        """
        with rig.connect() as connection:
            counts = {
                table: connection.execute(sa.text(f"SELECT count(*) FROM {table}")).scalar()
                for table in ARTIFACT_TABLES
            }
        assert counts == {"mf_checkpoints": 2, "mf_adapters": 1}, counts


class TestTheCheckCanSayNo:
    """The negative control, permanent rather than run once by hand."""

    def test_a_table_with_no_index_on_name_comes_back_unindexed(self, rig: sa.Engine) -> None:
        """``indexes_leading_on`` must be able to return nothing.

        Built in the same migrated database, with the same inspector, so the control
        differs from the real checks in exactly one way: this table has no index on
        ``name``. A helper that returned "indexed" unconditionally would pass every
        assertion above and would be indistinguishable from a correct one.
        """
        with rig.begin() as connection:
            connection.execute(
                sa.text("CREATE TABLE ctl_unindexed_name (name VARCHAR(512), step INTEGER)")
            )
        try:
            assert indexes_leading_on(sa.inspect(rig), "ctl_unindexed_name", "name") == []
        finally:
            with rig.begin() as connection:
                connection.execute(sa.text("DROP TABLE ctl_unindexed_name"))

    def test_the_control_table_really_is_dropped(self, rig: sa.Engine) -> None:
        """A ``finally`` that silently did nothing looks exactly like a clean test, and on
        the shared PostgreSQL a leaked table outlives the session - the defect
        ``test_tree_api.py`` records for a dropped index table."""
        assert "ctl_unindexed_name" not in sa.inspect(rig).get_table_names()


class TestTheReasonTheFindingLookedTrue:
    """A tripwire on the library behaviour that made task 11.6 look open.

    If a future SQLAlchemy starts reporting the PK-backing index from ``get_indexes``, this
    goes red and says so, and :func:`indexes_leading_on` can lose half its body. Until
    then it is the evidence that reading ``get_indexes`` alone is what produces the false
    finding, so nobody has to rediscover it.
    """

    @pytest.mark.parametrize("table", ARTIFACT_TABLES)
    def test_get_indexes_alone_would_report_name_as_unindexed(
        self, rig: sa.Engine, table: str
    ) -> None:
        listed = sa.inspect(rig).get_indexes(table)
        assert "name" not in {
            column for index in listed for column in (index.get("column_names") or [])
        }, (
            "get_indexes now reports the primary key's index - indexes_leading_on no "
            "longer needs to read get_pk_constraint separately"
        )
        assert sa.inspect(rig).get_pk_constraint(table)["constrained_columns"] == ["name"]
