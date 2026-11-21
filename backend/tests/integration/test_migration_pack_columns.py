"""Migration ``0015``'s three pack columns, round-tripped **on a populated table**.

Task 4.5. ``0015`` adds ``mf_packs.is_example``, ``mf_packs.derived_from_pack_id`` and
``mf_pack_parts.addressed_through_pack_id`` — the three capabilities the pack artifact cannot
express, because ``pack-definition-v1.json``'s top-level object and its ``PackSlot`` are both
closed to unknown keys.

**Why a file of its own rather than a line in ``test_db_migration.py``.** That file compares
the migrated schema with the models, which is the drift check; it says nothing about what
happens to **rows**. Every other test in the suite builds its schema against an *empty*
database (``tests/support/db.py``), and this project has already shipped two migration
defects that only rows could show: ``0014``'s UPDATE-then-swap-the-constraint, which raises
the moment one row exists, and the ``copy_from`` rebuild that silently lost the table's one
index. So this file seeds at ``0014``, upgrades, writes the three values, downgrades and
upgrades again, and asserts four separate properties of that round trip:

1. the pack rows **survive** the downgrade;
2. ``mf_packs``'s CHECK constraint ``invalid_names_a_reason`` survives it — SQLAlchemy does
   not reflect SQLite CHECK constraints, so a ``batch_alter_table`` left to reflect would
   have rebuilt the table without it while reporting success, which is why ``0015`` uses
   plain ``add_column`` and why that choice is asserted rather than trusted;
3. both partial indexes **come back** on the re-upgrade. The index half is not decoration:
   the one time a rebuild in this project lost an index, the hand verification *printed*
   ``indexes kept: []`` and the reader went past it;
4. the three values are **gone** — which is the property ``0015``'s docstring warns about,
   and the reason its warning does not say "export before downgrading": an export carries
   the closed document, which has none of these fields, so the prescribed remedy would have
   remedied nothing.

Property 4 is asserted *positively*, as a loss, rather than left as an unstated consequence.
A test that only checked "the rows are still there" would pass over exactly the data loss an
operator needs to be warned about.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import sqlalchemy as sa
from alembic import command
from alembic.config import Config

from packs.registry import DB_ONLY_PACK_FIELDS, DB_ONLY_PART_FIELDS
from tests.support import db as support

#: The revision immediately before the three columns, and the one that adds them.
BEFORE = "0014_merges"
WITH_COLUMNS = "0015_pack_cols"

#: The columns ``0015`` adds, per table. Taken from the two ``DB_ONLY_*`` tuples the
#: production rebuild reads rather than restated here: a second list would drift, and the
#: whole point of the pair is that ``rebuild_index`` and this test mean the same three
#: columns. The assertion below pins that they are the three this revision adds.
PACK_COLUMNS = DB_ONLY_PACK_FIELDS
PART_COLUMNS = DB_ONLY_PART_FIELDS

#: Two packs and one part. ``p2`` exists so ``derived_from_pack_id`` can name a real pack
#: rather than a dangling id - this test is about the round trip, not about the registry's
#: rendering of an unresolvable derivation.
SEED_PACKS = (("p1", "alpha", "1"), ("p2", "beta", "1"))


def _config(url: str) -> Config:
    cfg = Config(str(support.REPO_ROOT / "alembic.ini"))
    cfg.set_main_option("script_location", str(support.REPO_ROOT / "backend" / "alembic"))
    cfg.attributes["configure_logger"] = False
    cfg.cmd_opts = support._XArgs(url)  # type: ignore[assignment]
    return cfg


def _columns(engine: sa.Engine, table: str) -> set[str]:
    return {column["name"] for column in sa.inspect(engine).get_columns(table)}


def _indexes(engine: sa.Engine, table: str) -> set[str]:
    return {index["name"] or "" for index in sa.inspect(engine).get_indexes(table)}


def _table_sql(engine: sa.Engine, table: str) -> str:
    """The stored DDL, for the one thing reflection cannot see.

    Reflection is not used for the CHECK constraint, deliberately: SQLAlchemy does not
    reflect SQLite CHECK constraints, which is the blind spot that makes a reflecting rebuild
    dangerous in the first place. A reflection-based assertion here would report the
    constraint as absent in both the good and the broken case and prove nothing.
    """
    with engine.connect() as conn:
        return (
            conn.exec_driver_sql(
                "SELECT sql FROM sqlite_master WHERE type='table' AND name=?", (table,)
            ).scalar()
            or ""
        )


@pytest.fixture()
def seeded(tmp_path: Path) -> tuple[Config, str]:
    """A database at ``0014`` holding two packs and a part - rows **before** the columns."""
    url = support.database_url(tmp_path)
    if not url.startswith("sqlite"):
        pytest.skip("reads sqlite_master directly; the PostgreSQL form is a separate test")
    cfg = _config(url)
    command.upgrade(cfg, BEFORE)
    engine = support.make_engine(url)
    with engine.begin() as conn:
        for identifier, name, version in SEED_PACKS:
            conn.exec_driver_sql(
                "INSERT INTO mf_packs (id, name, version, valid, invalid_reason, created_at) "
                "VALUES (?, ?, ?, 1, NULL, '2026-10-14T00:00:00')",
                (identifier, name, version),
            )
        conn.exec_driver_sql(
            "INSERT INTO mf_pack_parts "
            "(pack_id, part_kind, source, ref_type, ref_name, ref_version) "
            "VALUES ('p1', 'prompt-set', 'midataworks', 'prompt_set', 'core', '1')"
        )
    engine.dispose()
    return cfg, url


def _fill(engine: sa.Engine) -> None:
    """Write the three values, which only exist once ``0015`` has run."""
    with engine.begin() as conn:
        conn.exec_driver_sql(
            "UPDATE mf_packs SET is_example = 1, derived_from_pack_id = 'p2' WHERE id = 'p1'"
        )
        conn.exec_driver_sql("UPDATE mf_pack_parts SET addressed_through_pack_id = 'p2'")


class TestTheColumnsAreAdditiveOverExistingRows:
    def test_the_three_columns_are_the_two_db_only_tuples(self) -> None:
        """The lists the production rebuild carries are the columns this revision adds.

        Asserted rather than assumed, because the two must not drift: if a fourth db-only
        column is ever added, ``rebuild_index`` carrying it and this test knowing about it
        are the same fact, and a test with its own hand-written list would keep passing
        while the new column was erased by every reindex.
        """
        assert PACK_COLUMNS == ("is_example", "derived_from_pack_id")
        assert PART_COLUMNS == ("addressed_through_pack_id",)

    def test_the_upgrade_adds_the_columns_and_keeps_every_row(self, seeded) -> None:
        cfg, url = seeded
        command.upgrade(cfg, WITH_COLUMNS)
        engine = support.make_engine(url)
        try:
            assert set(PACK_COLUMNS) <= _columns(engine, "mf_packs")
            assert set(PART_COLUMNS) <= _columns(engine, "mf_pack_parts")
            with engine.connect() as conn:
                packs = conn.exec_driver_sql("SELECT id FROM mf_packs ORDER BY id").fetchall()
                # `is_example` is NOT NULL with a server default, so an existing row must come
                # back `false` rather than null: "not an example" is an assertion, and a null
                # there would be an omission the column's own comment rules out.
                flags = conn.exec_driver_sql(
                    "SELECT id, is_example, derived_from_pack_id FROM mf_packs ORDER BY id"
                ).fetchall()
            assert [row[0] for row in packs] == ["p1", "p2"]
            assert [(row[0], bool(row[1]), row[2]) for row in flags] == [
                ("p1", False, None),
                ("p2", False, None),
            ]
        finally:
            engine.dispose()

    def test_the_round_trip_keeps_the_rows_the_constraint_and_the_indexes(self, seeded) -> None:
        """Seed → upgrade → write → downgrade → upgrade, with rows present throughout."""
        cfg, url = seeded
        command.upgrade(cfg, WITH_COLUMNS)
        engine = support.make_engine(url)
        try:
            _fill(engine)
            before_indexes = _indexes(engine, "mf_packs") | _indexes(engine, "mf_pack_parts")
            assert "ix_mf_packs_derived_from" in before_indexes
            assert "ix_mf_pack_parts_addressed_through" in before_indexes

            command.downgrade(cfg, BEFORE)

            # The rows, which is what makes this a round trip rather than a schema check.
            with engine.connect() as conn:
                assert conn.exec_driver_sql(
                    "SELECT id, name, version FROM mf_packs ORDER BY id"
                ).fetchall() == [("p1", "alpha", "1"), ("p2", "beta", "1")]
                assert conn.exec_driver_sql(
                    "SELECT pack_id, part_kind FROM mf_pack_parts"
                ).fetchall() == [("p1", "prompt-set")]

            # The columns are gone and nothing else is.
            assert set(PACK_COLUMNS) & _columns(engine, "mf_packs") == set()
            assert set(PART_COLUMNS) & _columns(engine, "mf_pack_parts") == set()

            # The CHECK constraint a reflecting rebuild would have dropped in silence.
            assert "ck_mf_packs_invalid_names_a_reason" in _table_sql(engine, "mf_packs")
            parts_sql = _table_sql(engine, "mf_pack_parts")
            assert "ck_mf_pack_parts_part_is_pinned" in parts_sql
            assert "ck_mf_pack_parts_source_known" in parts_sql

            # The table's own unique indexes, which are not this revision's and must survive.
            assert "uq_mf_packs_name_version" in _indexes(engine, "mf_packs")
            assert "uq_mf_pack_parts_pack_id_part_kind" in _indexes(engine, "mf_pack_parts")

            command.upgrade(cfg, "head")

            after = _indexes(engine, "mf_packs") | _indexes(engine, "mf_pack_parts")
            assert "ix_mf_packs_derived_from" in after
            assert "ix_mf_pack_parts_addressed_through" in after
            assert "uq_mf_packs_name_version" in after
            assert "uq_mf_pack_parts_pack_id_part_kind" in after
        finally:
            engine.dispose()


class TestTheDowngradeIsLossyAndNothingUndoesIt:
    """The property ``0015``'s docstring warns about, asserted rather than implied."""

    def test_the_three_values_do_not_come_back(self, seeded) -> None:
        cfg, url = seeded
        command.upgrade(cfg, WITH_COLUMNS)
        engine = support.make_engine(url)
        try:
            _fill(engine)
            command.downgrade(cfg, BEFORE)
            command.upgrade(cfg, "head")
            with engine.connect() as conn:
                pack = conn.exec_driver_sql(
                    "SELECT is_example, derived_from_pack_id FROM mf_packs WHERE id = 'p1'"
                ).one()
                part = conn.exec_driver_sql(
                    "SELECT addressed_through_pack_id FROM mf_pack_parts WHERE pack_id = 'p1'"
                ).one()
            # Not "may be lost" - lost. `p1` was an example derived from `p2` with a part
            # addressed through `p2`, and after the round trip it is an unmarked pack with no
            # derivation and no cross-pack addressing. This is the data loss the operator is
            # warned about, and it is why the warning names two SELECTs rather than an export.
            assert bool(pack[0]) is False
            assert pack[1] is None
            assert part[0] is None
        finally:
            engine.dispose()

    def test_an_export_does_not_carry_any_of_the_three(self) -> None:
        """Why the warning does **not** say "export before downgrading".

        The task asked for those words. An export is the closed pack document, and the three
        columns exist *because* that document cannot hold them - so an operator who exported,
        downgraded and re-imported would have followed the instruction exactly and still lost
        every Example pill and every derivation record. Read out of the schema rather than
        out of a Python model, because the schema is what closes the objects.
        """
        import json

        schema = json.loads(
            (support.REPO_ROOT / "docs" / "schemas" / "pack-definition-v1.json").read_text()
        )
        definition = schema["$defs"]["PackDefinitionV1"]
        slot = schema["$defs"]["PackSlot"]

        # Closed, which is the mechanism: an unknown key does not merely go unread, it is
        # refused, so there is no "write it in anyway" escape.
        assert definition["additionalProperties"] is False
        assert slot["additionalProperties"] is False

        for column in PACK_COLUMNS:
            assert column not in definition["properties"]
            # And not under the name the document would plausibly have used either.
            assert column.removesuffix("_pack_id") not in definition["properties"]
        for column in PART_COLUMNS:
            assert column not in slot["properties"]
            assert column.removesuffix("_pack_id") not in slot["properties"]


class TestTheWarningIsWhereAnOperatorWillReadIt:
    """A tripwire on the docstring, because the docstring is the deliverable of task 4.5.

    Prose is not usually worth asserting. This paragraph is, for one reason: it is the only
    place the lossy downgrade is stated, there is no code path that can report it at the
    moment it matters (``alembic downgrade`` does not consult the application), and a
    refactor that trimmed the docstring would remove the warning with the suite green.
    """

    def test_the_revision_warns_before_the_downgrade_and_names_the_two_queries(self) -> None:
        source = (
            support.REPO_ROOT
            / "backend"
            / "alembic"
            / "versions"
            / "0015_pack_example_derivation_and_cross_pack.py"
        ).read_text()
        assert "EXPORT BEFORE DOWNGRADING" in source
        # The remedy that works, named concretely: two SELECTs over the three columns.
        assert "SELECT id, is_example, derived_from_pack_id FROM mf_packs" in source
        assert "SELECT pack_id, part_kind, addressed_through_pack_id FROM mf_pack_parts" in source
        # And the reason the task's own wording was not used verbatim, kept in the file so a
        # later reader does not "restore" a remedy that does not remedy anything.
        assert "an export would not have saved any of this" in source.lower()
