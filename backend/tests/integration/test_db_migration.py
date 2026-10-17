"""The migration and the models must agree  (task 4.2; ADR-003).

**The drift this catches is invisible to every other test in the suite.** Each other test
builds its tables from the migration *or* talks to the models, and either way the two are
never compared - so a column added to a model without a migration passes everything while
production's database lacks the column, and a column dropped from a model but left in the
migration leaves a ghost nobody can account for.

Autogenerate's own comparison is used rather than a hand-written column list, because a
hand-written list is a third description of the schema and would drift from both.

**"The models" means every declared table, not only the base schema's.** This is the one
non-obvious thing in the file and it was added when F10 landed the first feature-owned
tables. Two foundation rules meet here and, read literally, they jointly forbade any
feature from owning a table at all:

* ``test_models_base_schema.test_the_base_schema_is_exactly_these_seven_tables`` asserts
  ``Base.metadata`` holds **exactly** the seven base-schema tables, with the message *"a
  table only one feature reads belongs in that feature's own migration"* - so a feature's
  tables must **not** be on ``Base.metadata``;
* this file asserted the migrated database equals ``Base.metadata`` **exactly** - so a
  table in the database that is not on ``Base.metadata`` is a failure.

Both are right about what they are protecting, and the contradiction is only in the word
"models". So the comparison is made against the union: ``Base.metadata`` plus each
feature's own metadata, declared in :data:`FEATURE_METADATA`. The base-schema ratchet keeps
meaning what it says, and this test still catches the drift it exists for - a column added
to a feature's table declaration without a migration, or left in a migration after being
dropped from the declaration.

**A feature adding tables adds one line to :data:`FEATURE_METADATA`.** If it does not, this
test goes red, which is the behaviour wanted: a table created by a migration that no
declaration describes is a table nothing can be checked against.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import sqlalchemy as sa
from alembic.autogenerate import compare_metadata
from alembic.migration import MigrationContext

from db.base import NAMING_CONVENTION, Base
from services.chains.tables import metadata as f006_metadata
from services.parts.tables import metadata as f010_metadata
from services.scoring.tables import metadata as f004_metadata
from tests.support import db as support
from training.runtypes.tables import metadata as f008_metadata
from training.signals.batch_index import metadata as f002_metadata

#: Every feature-owned ``MetaData``. See the module docstring.
FEATURE_METADATA: tuple[sa.MetaData, ...] = (
    # F10 (R-02.36, R-02.37): mf_contract_parts, mf_part_splits, mf_review_bindings,
    # mf_review_candidate_refs - declared in services/parts/tables.py, migrated by
    # 0010_f010_parts_and_review_binding.py.
    f010_metadata,
    # F4 (R-02.16-R-02.18): mf_scores, mf_score_items - declared in
    # services/scoring/tables.py, migrated by 0013_scoring_tables.py. Added 2026-10-14 when
    # that migration landed, which is the one line this tuple's docstring asks for: the
    # migration created two tables and the comparison had nothing to check them against, so
    # it correctly reported them as tables to REMOVE.
    f004_metadata,
    # F8 (R-02.45): mf_merges - declared in training/runtypes/tables.py, migrated by
    # 0014_merges_and_run_type_vocabulary.py. Added 2026-10-14 with that migration. Unlike
    # the others this table is NOT purely an index: a refused or interrupted merge writes no
    # checkpoint and so leaves no manifest for `reindex` to rebuild from, which is the whole
    # reason it exists.
    f008_metadata,
    # F6 (R-02.27): mf_chains, mf_chain_stages - declared in services/chains/tables.py,
    # migrated by 0016_loop_index_columns_and_chains.py. Added 2026-10-14 with that
    # migration. **These tables precede their writer** and that is stated rather than
    # disguised: F6 shipped R-02.27's types and its safety-critical advance filter (three
    # layers, three red controls) and deliberately not the orchestration, reporting it
    # blocked on this DDL. What is owed on top is chain_service, the
    # miforge.maintenance.chain_sweep sweeper and four routes.
    f006_metadata,
    # F2 (R-02.8): mf_batches - declared in training/signals/batch_index.py, migrated by
    # 0019_batch_index_table.py. **This omission was the defect, not a tidy-up.** Until
    # 2026-10-14 this tuple did not name F2's metadata and no revision created the table,
    # so the two halves of the comparison agreed by both being blind: nothing described
    # mf_batches and nothing created it. Six integration fixtures called
    # `batch_index.metadata.create_all(engine)` and made it exist for the tests alone, so
    # the suite was green while a batch write on a production database would have failed
    # with `relation "mf_batches" does not exist`. The general rule is now asserted
    # mechanically by tests/unit/test_every_table_has_a_migration.py, which is what found
    # it.
    f002_metadata,
)


def all_declared_metadata() -> sa.MetaData:
    """``Base.metadata`` plus every feature's, as one ``MetaData`` to compare against.

    Built by copying rather than by mutating either source: a test that added a feature's
    tables **into** ``Base.metadata`` would make the base-schema ratchet pass or fail
    depending on whether this module had been imported first, which is the worst kind of
    test coupling - order-dependent and invisible.
    """
    combined = sa.MetaData(naming_convention=NAMING_CONVENTION)
    for source in (Base.metadata, *FEATURE_METADATA):
        for table in source.tables.values():
            table.to_metadata(combined)
    return combined


@pytest.fixture()
def engine(tmp_path: Path):
    return support.migrated_engine(tmp_path / "db")


def test_the_migration_produces_exactly_the_models_schema(engine) -> None:
    with engine.connect() as connection:
        context = MigrationContext.configure(
            connection,
            opts={"compare_type": True, "compare_server_default": True, "render_as_batch": True},
        )
        diff = compare_metadata(context, all_declared_metadata())
    assert diff == [], (
        "the migration and the models disagree. Generate a migration for the difference "
        f"rather than editing one of the two to match: {diff}"
    )


def test_the_migration_reverses(engine) -> None:
    """A downgrade that was never run is a downgrade that does not work.

    SQLite cannot drop a column, which is why ``render_as_batch`` is on in ``env.py``; a
    downgrade asserted only against PostgreSQL would hide that the unit suite's engine
    cannot reverse the migration at all.
    """
    from alembic import command
    from alembic.config import Config

    # render_as_string(hide_password=False), NOT str(engine.url): SQLAlchemy MASKS the
    # password in __str__, so str() yields 'miforge:***@...' and Alembic then tries to
    # authenticate with the literal '***'. On SQLite there is no password, so this is
    # invisible locally and fails only on the PostgreSQL leg of CI - found by the agent
    # that wrote backend-tests.yml, which ran a real PostgreSQL 15 instead of reasoning
    # about it. Guarded by test_the_migration_url_carries_a_usable_password below.
    url = engine.url.render_as_string(hide_password=False)
    cfg = Config(str(support.REPO_ROOT / "alembic.ini"))
    cfg.set_main_option("script_location", str(support.REPO_ROOT / "backend" / "alembic"))
    cfg.attributes["configure_logger"] = False
    cfg.cmd_opts = support._XArgs(url)  # type: ignore[assignment]
    command.downgrade(cfg, "base")

    with engine.connect() as connection:
        remaining = set(all_declared_metadata().tables) & set(
            engine.dialect.get_table_names(connection)  # type: ignore[attr-defined]
        )
    assert remaining == set(), remaining


def test_the_migration_url_carries_a_usable_password() -> None:
    """The URL handed to Alembic must not be the masked form.

    Asserts the PROPERTY rather than the call, so it holds against any future way of
    building the URL. ``str(engine.url)`` renders the password as ``***``; Alembic then
    authenticates with that literal and fails with "password authentication failed" -
    which looks like a credentials problem in CI and is actually a rendering bug.

    SQLite has no password, so this class of defect is invisible on the default engine and
    appears only on the PostgreSQL leg. That asymmetry is the reason for the test: a
    failure mode that only one engine can see needs something that checks the shape on
    both.
    """
    from sqlalchemy import make_url

    url = make_url("postgresql+psycopg://miforge:s3cret@localhost:5432/miforge")
    rendered = url.render_as_string(hide_password=False)
    assert "***" not in rendered, "the masked form would be passed to Alembic verbatim"
    assert "s3cret" in rendered
    # And the shape this replaced really does mask, so the test is not vacuous.
    assert "***" in str(url)
