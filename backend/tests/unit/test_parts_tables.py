"""The migration and the table declarations must agree  (F10 task 5.x; ADR-003).

**The drift this catches is the one that stayed green through a whole feature.**
``workers/run_record.py`` records it: a module carried its own ``sa.Table`` definitions
while another task owned the real ones, and when the real ones landed *"six of its seven
column names were wrong"* - ``run_id`` for ``id``, a boolean for a timestamp, milliseconds
for seconds, and a column that did not exist. **Every test of it passed, because the tests
built its tables.**

F10 owns these four tables, so it declares them once
(:mod:`services.parts.tables`) and migrates them once
(``0010_f010_parts_and_review_binding``). This file is what keeps the two honest: it reads
the **migrated** database through an inspector and compares it with the declaration.

The foundation's ``test_db_migration.py`` compares against the union of every declared
``MetaData`` and so covers this too; this file is narrower and names F10, so a failure points
at the right place rather than at "the migration".
"""

from __future__ import annotations

from pathlib import Path

import pytest
import sqlalchemy as sa
from sqlalchemy.exc import IntegrityError

from db.base import TABLE_PREFIX
from services.parts import tables as f010
from tests.support import db as support


@pytest.fixture(scope="module")
def engine(tmp_path_factory):
    return support.migrated_engine(tmp_path_factory.mktemp("f010-schema"))


@pytest.fixture(scope="module")
def inspector(engine):
    return sa.inspect(engine)


class TestTheFourTablesExistInTheMigratedDatabase:
    @pytest.mark.parametrize("table", sorted(f010.ALL_TABLES))
    def test_the_table_was_created(self, inspector, table: str):
        assert inspector.has_table(table)

    @pytest.mark.parametrize("table", sorted(f010.ALL_TABLES))
    def test_its_columns_match_the_declaration(self, inspector, table: str):
        """Column for column. **This is the assertion that would have caught the six wrong
        names**, and the reason it compares names rather than types is that a type mismatch
        is caught by the foundation's autogenerate comparison while a *missing* column is
        the one that silently breaks a read."""
        migrated = {column["name"] for column in inspector.get_columns(table)}
        declared = set(f010.metadata.tables[table].columns.keys())
        assert migrated == declared, {
            "only in the database": sorted(migrated - declared),
            "only in the declaration": sorted(declared - migrated),
        }

    def test_the_declaration_is_not_on_the_base_metadata(self):
        """Foundation 4.2's boundary rule: *a table only one feature reads belongs in that
        feature's own migration*, and its base-schema ratchet asserts ``Base.metadata`` holds
        exactly the seven base tables. So F10's tables carry their **own** ``MetaData``."""
        from db.base import Base

        assert not set(f010.ALL_TABLES) & set(Base.metadata.tables)

    def test_every_table_carries_the_mf_prefix(self):
        """ADR-003. The base-schema ratchet cannot see these tables, so the prefix needs its
        own check here - an unprefixed table collides in a shared database and is found in
        production."""
        offenders = [name for name in f010.ALL_TABLES if not name.startswith(TABLE_PREFIX)]
        assert offenders == [], offenders

    def test_the_naming_convention_is_the_shared_one(self):
        """Unnamed CHECK and UNIQUE constraints cannot be dropped by a later migration - so a
        migration that reverses on PostgreSQL would be irreversible on the SQLite the unit
        suite runs against."""
        from db.base import NAMING_CONVENTION

        assert f010.metadata.naming_convention == NAMING_CONVENTION


class TestEveryTimestampIsTimezoneAware:
    @pytest.mark.parametrize(
        ("table", "column"),
        [
            ("mf_contract_parts", "imported_at"),
            ("mf_review_bindings", "read_back_cursor"),
            ("mf_review_bindings", "created_at"),
            ("mf_review_bindings", "updated_at"),
            ("mf_review_candidate_refs", "decided_at"),
            ("mf_review_candidate_refs", "created_at"),
        ],
    )
    def test_the_column_is_declared_with_a_timezone(self, table: str, column: str):
        """**A naive column silently shifts the ``since`` watermark, and a shifted watermark
        loses decisions permanently.** The loss is invisible: the read returns fewer rows and
        nothing says why, so a candidate stays pending forever while a human believes they
        answered it.

        Asserted on the **declaration**, because SQLite reports every ``DATETIME`` the same
        way whether or not the type carries ``timezone=True`` - so an inspector check here
        would pass against the mutation and only fail on PostgreSQL.
        """
        declared = f010.metadata.tables[table].columns[column].type
        assert isinstance(declared, sa.DateTime)
        assert declared.timezone is True


class TestTheConstraintsThatCarryRequirements:
    def test_an_external_part_cannot_carry_a_contract_reference(self, engine):
        """R-02.40 **at the database**, so the prohibition is not a convention three call
        sites are trusted to honour."""
        with support.session_for(engine) as session, pytest.raises(IntegrityError):
            session.execute(
                sa.insert(f010.CONTRACT_PARTS).values(
                    id="part-x",
                    part_kind="corpus",
                    source=f010.SOURCE_EXTERNAL,
                    ref_type="corpus",
                    ref_name="n",
                    ref_version="1",
                    content_kind="rows",
                    local_path="/tmp/x",
                    manifest_sha256="a" * 64,
                    row_count=0,
                    imported_at=support.utc_now(),
                )
            )
            session.flush()

    def test_a_partial_contract_reference_is_refused(self, engine):
        """Two of three fields names nothing resolvable, and the schema's own
        ``ContractReference`` requires all three."""
        with support.session_for(engine) as session, pytest.raises(IntegrityError):
            session.execute(
                sa.insert(f010.CONTRACT_PARTS).values(
                    id="part-y",
                    part_kind="corpus",
                    source=f010.SOURCE_CONTRACT,
                    ref_type="corpus",
                    ref_name=None,
                    ref_version="1",
                    content_kind="rows",
                    local_path="/tmp/y",
                    manifest_sha256="a" * 64,
                    row_count=0,
                    imported_at=support.utc_now(),
                )
            )
            session.flush()

    def test_several_parts_with_no_reference_can_coexist(self, engine):
        """**Why the unique index is partial.** An external part and a conformant version
        carrying a null ``contract_reference`` (E-21) both have null reference columns, and a
        non-partial unique index would let only one of them exist."""
        with support.session_for(engine) as session:
            for number in range(3):
                session.execute(
                    sa.insert(f010.CONTRACT_PARTS).values(
                        id=f"part-null-{number}",
                        part_kind="corpus",
                        source=f010.SOURCE_EXTERNAL,
                        content_kind="rows",
                        local_path=f"/tmp/{number}",
                        manifest_sha256=f"{number}" * 64,
                        row_count=0,
                        external_provenance={"source": "x", "content_sha256": "a" * 64},
                        imported_at=support.utc_now(),
                    )
                )
            session.flush()

    def test_only_an_override_carries_a_label(self, engine):
        """miDataworks enforces the same rule on the way in (*"Only an override carries a
        label"*), so a label beside an ``accept`` here would mean miForge recorded something
        miDataworks never said."""
        with support.session_for(engine) as session:
            session.execute(
                sa.insert(f010.REVIEW_BINDINGS).values(
                    id="rvb-1",
                    run_id="run-x",
                    queue_id="rq1",
                    question="q",
                    label_set=["a", "b"],
                    posted=0,
                    decided=0,
                    pending=0,
                    created_at=support.utc_now(),
                    updated_at=support.utc_now(),
                )
            )
            with pytest.raises(IntegrityError):
                session.execute(
                    sa.insert(f010.REVIEW_CANDIDATE_REFS).values(
                        binding_id="rvb-1",
                        step=1,
                        external_id="x1",
                        decision="accept",
                        override_label="good",
                        created_at=support.utc_now(),
                    )
                )
                session.flush()

    def test_the_decision_column_holds_midataworks_vocabulary_plus_none(self):
        """**What was read**, never miForge's verdict value. ``none`` means *no decision yet*
        - a stored value rather than an absence, because an absence invites a default."""
        assert f010.DECISION_VALUES == ("accept", "override", "flag", "reject", "none")
        assert f010.DECISION_NONE == "none"

    def test_an_unknown_decision_value_is_refused_by_the_check_constraint(self, engine):
        with support.session_for(engine) as session:
            session.execute(
                sa.insert(f010.REVIEW_BINDINGS).values(
                    id="rvb-2",
                    run_id="run-y",
                    queue_id="rq2",
                    question="q",
                    label_set=["a", "b"],
                    posted=0,
                    decided=0,
                    pending=0,
                    created_at=support.utc_now(),
                    updated_at=support.utc_now(),
                )
            )
            with pytest.raises(IntegrityError):
                session.execute(
                    sa.insert(f010.REVIEW_CANDIDATE_REFS).values(
                        binding_id="rvb-2",
                        step=1,
                        external_id="x2",
                        decision="pending",
                        created_at=support.utc_now(),
                    )
                )
                session.flush()


class TestTheMigrationIsTheOnlyPlaceTheseTablesAreCreated:
    def test_nothing_calls_create_all_on_this_metadata(self):
        """``create_all`` builds tables from the declaration, so a test using it would test
        the models against themselves and a drifted migration would still pass."""
        import ast

        src = Path(__file__).resolve().parents[2] / "src"
        offenders = [
            str(path.relative_to(src))
            for path in src.rglob("*.py")
            if any(
                isinstance(node, ast.Call)
                and isinstance(node.func, ast.Attribute)
                and node.func.attr == "create_all"
                for node in ast.walk(ast.parse(path.read_text("utf-8")))
            )
        ]
        assert offenders == [], offenders

    def test_the_migration_file_names_all_four_tables(self):
        versions = Path(__file__).resolve().parents[2] / "alembic" / "versions"
        migration = (versions / "0010_f010_parts_and_review_binding.py").read_text("utf-8")
        for table in f010.PART_INDEX_TABLES + f010.REVIEW_STATE_TABLES:
            assert f'"{table}"' in migration, f"{table} is declared but never migrated"
        # Group C (archive, operator 2026-10-20) arrived later, in its own revision.
        archive = (versions / "0022_archive.py").read_text("utf-8")
        for table in f010.ARCHIVE_STATE_TABLES:
            assert f'"{table}"' in archive, f"{table} is declared but never migrated"
        assert set(f010.ALL_TABLES) == set(
            f010.PART_INDEX_TABLES + f010.REVIEW_STATE_TABLES + f010.ARCHIVE_STATE_TABLES
        )
