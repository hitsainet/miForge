"""``mf_batches`` — F2's batch index, declared since F2 landed and created by no migration.

Revision ID: 0019_batches
Revises: 0018_scores_jsonb
Create Date: 2026-10-14

## A table the code declares and production does not have

``training/signals/batch_index.py`` declares ``BATCHES`` on F2's own metadata, following
the pattern ``services/scoring/tables.py`` (migration ``0013``),
``services/parts/tables.py`` (``0010``), ``training/runtypes/tables.py`` (``0014``) and
``services/chains/tables.py`` (``0016``) all use — **except that no migration ever created
it.** Found by diffing every ``sa.Table("mf_…")`` declared under ``src/`` against every
``op.create_table`` in this directory: 17 declared, 16 created, this one missing.

**In the cluster that is the compose cold-start defect again**, which is worth naming
because it is the third time this exact failure shape has appeared: ``/api/health`` reports
postgres ``ok`` — it checks connectivity, not schema — while every batch write 500s on
``relation "mf_batches" does not exist``. Health green and the feature wholly broken.

## Why 3,700 green tests said nothing about it

**Six test files call ``batch_index.metadata.create_all(engine)`` in their own fixtures.**
That is reasonable-looking test setup and it silently replaced the missing migration, so the
table existed everywhere it was asserted about and nowhere it was used. It also caused a
second, quieter defect: ``tests/support/db.migrated_engine`` resets a shared PostgreSQL by
``downgrade base`` + ``upgrade head``, and a table **outside the Alembic graph is not
dropped by that**, so its rows survived into the next test. On CI's serial PostgreSQL job
``test_three_rounds_each_produce_a_recorded_batch`` read **10 rows where it asserted 3** —
five tests in that file use ``run_id="run-1"`` — and F7's standalone test hit
``MultipleResultsFound`` the same way. Both were invisible on SQLite, where every test gets
a fresh file.

## The JSONB variant, deliberately

``part_refs`` and ``prompt_ids`` use ``db.base.JsonColumn``, which is
``JSON().with_variant(postgresql.JSONB(), "postgresql")``. This revision declares the same
variant locally, as ``0001``/``0010``/``0014``/``0016`` do. ``0013`` wrote plain
``sa.JSON()`` instead and the divergence was invisible on SQLite and a ``modify_type`` diff
on PostgreSQL — ``0018`` exists to repair exactly that, one revision earlier. Getting it
right here costs nothing; getting it wrong costs another migration.

## No surrogate key, and no foreign key to ``mf_runs``

``(run_id, batch_sha256)`` **is** the identity: the hash is what a step record cites and
what joins a step to the material it consumed. And the FK is omitted for the reason F4's
``SCORES``, F10's ``REVIEW_BINDINGS`` and F6's ``STAGES`` all omit theirs — those tables are
on feature-local metadata while ``mf_runs`` is on ``Base.metadata``, so a cross-metadata
constraint can only be emitted from a migration and is never resolved by a ``create_all``
over one metadata alone. ``run_id`` is a plain column, as ``mf_scores.run_id`` is.
"""

from __future__ import annotations

import sqlalchemy as sa
from alembic import op
from sqlalchemy import Text
from sqlalchemy.dialects import postgresql

revision: str = "0019_batches"
down_revision: str | None = "0018_scores_jsonb"
branch_labels: str | None = None
depends_on: str | None = None

#: The same type ``db.base.JsonColumn`` resolves to. See the module docstring.
JSON_COLUMN = sa.JSON().with_variant(postgresql.JSONB(astext_type=Text()), "postgresql")


def upgrade() -> None:
    op.create_table(
        "mf_batches",
        # `(run_id, batch_sha256)` IS the identity - no surrogate key.
        sa.Column("run_id", sa.String(length=64), nullable=False),
        sa.Column("batch_sha256", sa.String(length=64), nullable=False),
        # NULLABLE, and not an error state: a batch can exist with no step - assembled and
        # then discarded by a cancellation, or consumed by a step that failed.
        sa.Column("step", sa.Integer(), nullable=True),
        sa.Column("source", sa.String(length=32), nullable=False),
        # NULL for a corpus batch, and that null is the statement "nothing was generated"
        # rather than a missing value.
        sa.Column("resolved_model", sa.String(length=256), nullable=True),
        sa.Column("part_refs", JSON_COLUMN, nullable=False),
        sa.Column("prompt_ids", JSON_COLUMN, nullable=False),
        sa.Column("n_prompts", sa.Integer(), nullable=False),
        # THREE verdict counts, never two: `undecided` is a real outcome of an asynchronous
        # human-review selector, and a two-column tally would force every pending candidate
        # to be miscounted as accepted or rejected.
        sa.Column("n_accept", sa.Integer(), nullable=False),
        sa.Column("n_reject", sa.Integer(), nullable=False),
        sa.Column("n_undecided", sa.Integer(), nullable=False),
        sa.Column("candidates_generated", sa.Integer(), nullable=False),
        sa.Column("candidates_admitted", sa.Integer(), nullable=False),
        sa.Column("acceptance_rate", sa.Float(), nullable=False),
        sa.Column("exhausted", sa.Boolean(), nullable=False),
        sa.Column("selector_config_sha256", sa.String(length=64), nullable=True),
        sa.Column("grader_config_sha256", sa.String(length=64), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.PrimaryKeyConstraint("run_id", "batch_sha256", name=op.f("pk_mf_batches")),
    )
    # The paginated newest-first listing is the only hot read.
    op.create_index("ix_mf_batches_run_id_created_at", "mf_batches", ["run_id", "created_at"])
    # F9's and F6's filters.
    op.create_index("ix_mf_batches_run_id_source", "mf_batches", ["run_id", "source"])


def downgrade() -> None:
    op.drop_index("ix_mf_batches_run_id_source", table_name="mf_batches")
    op.drop_index("ix_mf_batches_run_id_created_at", table_name="mf_batches")
    op.drop_table("mf_batches")
