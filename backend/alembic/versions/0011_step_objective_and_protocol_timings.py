"""mf_steps gains objective and the two section-5.10 protocol timings.

Revision ID: 0011_step_obj
Revises: 0010_f010
Create Date: 2026-10-14

**Why these three columns, and why `objective` is the load-bearing one.**

F1 built the step without them because `backend/src/models/` is not a feature agent's to
edit, and it deliberately did **not** smuggle them into `batch_source_hashes` - all three
are preserved in the authoritative run log meanwhile (ADR-004: the log is the record, this
table is an index over it).

* **`objective`** - *a loss figure is interpretable against another only when both say which
  objective produced it.* F1 dispatches on `batch.objective` through F2's registry rather
  than hard-coding a cross-entropy, so a run's objective can legitimately change between
  curriculum stages. F9 draws loss curves from this table, and a curve that silently splices
  two objectives together looks exactly like a regression. This column is what stops that.

* **`quiesce_hold_ms` / `commit_duration_ms`** - **section 5.10 only**, and nullable for a
  reason that is itself a recorded requirement defect: R-02.14 demands both on *every* step
  record, which was written when every step held a commit bracket. Standalone there is no
  quiesce and no bracket, so demanding them would make the core product unable to satisfy
  its own requirement. They are present-and-null on every step F1 writes; F11 fills them.

  `commit_duration_ms` is **not local telemetry**. miLLM Feature 31's acceptance gate is set
  from its p99.9 under concurrent serving, so this column is a cross-repo product
  obligation - which is why it gets an index rather than being left to a file scan.

**A note on the migration chain.** This revises `0010_f010`, the head at the time, **not**
the base. Two migrations branching off `0001` give every developer multiple heads and
`alembic upgrade head` fails for all of them. Later migrations chain off the then-current
head; the Alembic revision graph is shared state and is serialised through the coordinator
rather than written by concurrent feature agents.
"""

from __future__ import annotations

import sqlalchemy as sa
from alembic import op

revision: str = "0011_step_obj"
down_revision: str | None = "0010_f010"
branch_labels: str | None = None
depends_on: str | None = None

#: Partial on PostgreSQL: the column is null on every standalone step, and an index over
#: mostly-null rows is mostly wasted. SQLite has no partial-index support through this
#: dialect keyword and simply builds a full index, which is correct if less tidy - the
#: suite's default engine, so the behaviour difference is worth stating rather than hiding.
_COMMIT_INDEX = "ix_mf_steps_commit_duration"


def upgrade() -> None:
    # Additive and nullable, so this is safe against an existing table with rows: no
    # backfill, no default, no rewrite. A step recorded before this migration honestly has
    # no objective recorded, and NULL says exactly that - inventing one would be the
    # fabricated value ADR-018 forbids.
    op.add_column("mf_steps", sa.Column("objective", sa.String(length=64), nullable=True))
    op.add_column("mf_steps", sa.Column("quiesce_hold_ms", sa.Float(), nullable=True))
    op.add_column("mf_steps", sa.Column("commit_duration_ms", sa.Float(), nullable=True))

    op.create_index(
        _COMMIT_INDEX,
        "mf_steps",
        ["run_id", "commit_duration_ms"],
        unique=False,
        postgresql_where=sa.text("commit_duration_ms IS NOT NULL"),
    )


def downgrade() -> None:
    op.drop_index(_COMMIT_INDEX, table_name="mf_steps")
    op.drop_column("mf_steps", "commit_duration_ms")
    op.drop_column("mf_steps", "quiesce_hold_ms")
    op.drop_column("mf_steps", "objective")
