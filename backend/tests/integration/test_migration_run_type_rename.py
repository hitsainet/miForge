"""Migration ``0014`` renames a ``run_type`` value **on a populated table**.

**Why this file exists: a defect that every other test in the suite is blind to.**

``0014`` renames the permitted ``run_type`` value ``full`` to ``change``. It was first
written the obvious way - ``UPDATE mf_runs SET run_type = 'change' WHERE run_type = 'full'``,
then swap the CHECK constraint - and that is wrong, because the UPDATE writes a value the
constraint still in force **forbids**. It raises ``CHECK constraint failed`` the moment a
single row exists.

It passed anyway. Every other test builds its schema with ``alembic upgrade head`` against an
**empty** database (``tests/support/db.py``), so the UPDATE matched zero rows and the
constraint was never tested against anything. The migration's own comment said the UPDATE was
written *"because 'there are no rows' is an assumption about every installation, not a
fact"* - and the code underneath it depended on exactly that assumption.

It surfaced only in the **downgrade**, and only because the hand verification happened to
insert rows before running it. The upgrade half was latent and identical, and would have
failed on the first installation with any history at all.

So: this file upgrades to ``0013``, **seeds rows with the old spelling**, and only then
upgrades to head. That ordering is the entire point of the file. The fix is a three-step
rename - widen the constraint to permit both spellings, rewrite the rows, narrow it to the
new set - and the two-step version fails ``test_the_upgrade_rewrites_existing_rows`` below,
which was re-run against the broken migration as a negative control to prove it bites.

**The second thing it pins** is that the ``mf_runs`` rebuild preserves what SQLite cannot
reflect. Changing a named CHECK constraint on SQLite requires a table copy, and SQLAlchemy
does not reflect SQLite CHECK constraints, so a ``batch_alter_table`` left to reflect rebuilds
the table **without** the other two CHECK constraints - dropping two invariants while
reporting success. ``0014`` therefore declares the table in full as ``copy_from``. That
declaration omitted the table's one index on the first attempt and the rebuild silently lost
it; ``test_db_migration.py`` caught that, and the index assertion here is the cheaper, more
direct statement of the same requirement.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import sqlalchemy as sa
from alembic import command
from alembic.config import Config

from tests.support import db as support

#: The revision immediately before the rename, and the one after it.
BEFORE = "0013_scoring"
AFTER = "head"

#: Every CHECK constraint ``mf_runs`` carries. The two that are not about ``run_type`` are
#: the ones a reflecting rebuild would have dropped in silence, which is why they are named
#: here rather than left to a general "the schema still matches" assertion.
RUN_CHECKS = (
    "ck_mf_runs_terminal_iff_finished",
    "ck_mf_runs_status_known",
    "ck_mf_runs_run_type_known",
)

#: Seeded at ``0013`` with the OLD spelling. ``create`` and ``adapter`` are included so the
#: test can tell a rename from a blanket overwrite: a migration that set every row to
#: ``change`` would satisfy an assertion that only looked at the renamed rows.
SEED = (
    ("r-full-1", "full"),
    ("r-create", "create"),
    ("r-adapter", "adapter"),
    ("r-full-2", "full"),
)


def _config(url: str) -> Config:
    cfg = Config(str(support.REPO_ROOT / "alembic.ini"))
    cfg.set_main_option("script_location", str(support.REPO_ROOT / "backend" / "alembic"))
    cfg.attributes["configure_logger"] = False
    cfg.cmd_opts = support._XArgs(url)  # type: ignore[assignment]
    return cfg


def _table_sql(engine: sa.Engine, table: str) -> str:
    """The stored DDL. Read from ``sqlite_master`` because the point is what SQLite *kept*.

    Reflection is deliberately not used: SQLAlchemy does not reflect SQLite CHECK
    constraints, which is the very blind spot that makes the rebuild dangerous, so a
    reflection-based assertion would report every constraint as absent and prove nothing.
    """
    with engine.connect() as conn:
        return (
            conn.exec_driver_sql(
                "SELECT sql FROM sqlite_master WHERE type='table' AND name=?", (table,)
            ).scalar()
            or ""
        )


def _indexes(engine: sa.Engine, table: str) -> set[str]:
    with engine.connect() as conn:
        return {
            row[0]
            for row in conn.exec_driver_sql(
                "SELECT name FROM sqlite_master WHERE type='index' AND tbl_name=?", (table,)
            )
            if not row[0].startswith("sqlite_")
        }


def _run_types(engine: sa.Engine) -> list[tuple[str, str]]:
    with engine.connect() as conn:
        return [
            (row[0], row[1])
            for row in conn.exec_driver_sql("SELECT id, run_type FROM mf_runs ORDER BY id")
        ]


@pytest.fixture()
def seeded(tmp_path: Path) -> tuple[Config, str]:
    """A database at revision ``0013`` holding rows with the pre-rename spelling.

    Not :func:`support.migrated_engine`, which goes straight to head - the whole defect is
    invisible unless the rows exist *before* the rename runs.
    """
    url = support.database_url(tmp_path)
    if not url.startswith("sqlite"):
        pytest.skip("reads sqlite_master directly; the PostgreSQL form is a separate test")
    cfg = _config(url)
    command.upgrade(cfg, BEFORE)
    engine = support.make_engine(url)
    with engine.begin() as conn:
        for run_id, run_type in SEED:
            conn.exec_driver_sql(
                "INSERT INTO mf_runs (id, status, run_type) VALUES (?, ?, ?)",
                (run_id, "pending", run_type),
            )
    engine.dispose()
    return cfg, url


class TestTheRenameSurvivesExistingRows:
    def test_the_upgrade_rewrites_existing_rows(self, seeded) -> None:
        """**The regression.** Fails against the two-step UPDATE-then-swap migration.

        Proven, not assumed: the two-step form was restored and this test went red with
        ``CHECK constraint failed: ck_mf_runs_run_type_known`` before the three-step fix was
        put back.
        """
        cfg, url = seeded
        command.upgrade(cfg, AFTER)
        engine = support.make_engine(url)
        try:
            assert _run_types(engine) == [
                ("r-adapter", "adapter"),
                ("r-create", "create"),
                ("r-full-1", "change"),
                ("r-full-2", "change"),
            ], (
                "the rename did not rewrite the rows it was supposed to, or rewrote rows it "
                "was not. Both 'full' rows become 'change'; 'create' and 'adapter' are "
                "untouched - a migration that overwrote everything would pass an assertion "
                "that only checked the renamed pair."
            )
        finally:
            engine.dispose()

    def test_the_downgrade_puts_them_back_exactly(self, seeded) -> None:
        """The round trip. This is the half that failed first and exposed the upgrade's
        identical latent bug."""
        cfg, url = seeded
        command.upgrade(cfg, AFTER)
        command.downgrade(cfg, BEFORE)
        engine = support.make_engine(url)
        try:
            assert _run_types(engine) == sorted(SEED)
        finally:
            engine.dispose()

    def test_upgrade_downgrade_upgrade_is_stable(self, seeded) -> None:
        """A migration that only works once is a migration that cannot be rolled back
        during an incident, which is the moment it is most needed."""
        cfg, url = seeded
        command.upgrade(cfg, AFTER)
        command.downgrade(cfg, BEFORE)
        command.upgrade(cfg, AFTER)
        engine = support.make_engine(url)
        try:
            assert {run_type for _, run_type in _run_types(engine)} == {
                "change",
                "create",
                "adapter",
            }
        finally:
            engine.dispose()


class TestTheRebuildKeepsWhatSqliteCannotReflect:
    def test_all_three_check_constraints_survive(self, seeded) -> None:
        """The two unrelated CHECK constraints are the silent-loss case.

        A rebuild that dropped ``terminal_iff_finished`` would let a completed run carry no
        ``finished_at``, which is how a janitor ends up unable to tell a finished run from
        one it should reap - and nothing would report it, because the migration succeeded.
        """
        cfg, url = seeded
        command.upgrade(cfg, AFTER)
        engine = support.make_engine(url)
        try:
            ddl = _table_sql(engine, "mf_runs")
            missing = [name for name in RUN_CHECKS if name not in ddl]
            assert not missing, (
                f"the mf_runs rebuild lost {missing}. The copy_from table in migration 0014 "
                f"must declare every constraint the table carries: SQLAlchemy does not "
                f"reflect SQLite CHECK constraints, so anything left out is dropped in "
                f"silence while the migration reports success."
            )
        finally:
            engine.dispose()

    def test_the_index_survives(self, seeded) -> None:
        """It did not, on the first attempt. The hand verification printed
        ``indexes kept: []`` and that line was read past."""
        cfg, url = seeded
        command.upgrade(cfg, AFTER)
        engine = support.make_engine(url)
        try:
            assert "ix_mf_runs_status_created_at" in _indexes(engine, "mf_runs")
        finally:
            engine.dispose()

    def test_the_columns_added_by_later_revisions_survive(self, seeded) -> None:
        """``copy_from`` must describe the table as it **is**, not as ``0001`` left it.

        ``save_requested_at`` and ``save_request_count`` arrive in ``0012``. Omitting them
        from the declaration would drop the operator-requested-save mechanism on the rebuild.
        """
        cfg, url = seeded
        command.upgrade(cfg, AFTER)
        engine = support.make_engine(url)
        try:
            ddl = _table_sql(engine, "mf_runs")
            for column in ("save_requested_at", "save_request_count"):
                assert column in ddl, f"the rebuild dropped {column}, added by revision 0012"
            assert "fk_mf_runs_pack_id_mf_packs" in ddl, "the rebuild dropped the pack FK"
        finally:
            engine.dispose()


class TestTheConstraintActuallyPermitsTheNewVocabulary:
    @pytest.mark.parametrize(
        ("run_type", "permitted"),
        [
            ("change", True),
            ("create", True),
            ("adapter", True),
            # ~~Permitted and nothing should write it~~ — **REMOVED by migration `0016`, and
            # this row is the tripwire that fired.** It read `("merge", True)` with the note
            # *"asserted so that removing it from the CHECK is a deliberate decision rather
            # than a side effect"*, and the removal was exactly that decision.
            #
            # `0014` left it permitted *"because F6 is not built, so nothing has yet proved
            # no submit path produces one."* F6 is built and it is proved three ways:
            # `run_launch.launch` is the only path that creates an `mf_runs` row,
            # `RunDefinition.run_type` is validated through F8's registry, and
            # `workers/merge_tasks` writes `mf_merges` and never a run row.
            ("merge", False),
            # The old spelling must be gone, or both vocabularies are live at once and the
            # rename achieved nothing.
            ("full", False),
            ("bogus", False),
        ],
    )
    def test_the_permitted_set_is_exactly_the_new_vocabulary(
        self, seeded, run_type: str, permitted: bool
    ) -> None:
        cfg, url = seeded
        command.upgrade(cfg, AFTER)
        engine = support.make_engine(url)
        try:
            with engine.begin() as conn:
                try:
                    conn.exec_driver_sql(
                        "INSERT INTO mf_runs (id, status, run_type) VALUES (?, ?, ?)",
                        (f"probe-{run_type}", "pending", run_type),
                    )
                    accepted = True
                except sa.exc.IntegrityError:
                    accepted = False
            assert accepted is permitted
        finally:
            engine.dispose()
