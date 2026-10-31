"""Archive: hide finished runs and unwanted parts from the default lists (operator 2026-10-20).

Revision ID: 0022_archive
Revises: 0021_approvals_and_settings
Create Date: 2026-10-20

**Archive is visibility, never deletion.** Nothing here removes a row, a checkpoint, a step
or any trainer state; unarchiving restores the default listing exactly.

* ``mf_runs.archived_at`` - nullable; null means "listed". ``mf_runs`` is a record table
  (ADR-004), so the column lives on the row. Additive; no rebuild.
* ``mf_part_archive`` - one row per archived part id. **A table of its own, not a column on
  ``mf_contract_parts``**, because that table is an *index* that ``reindex_parts`` empties
  and rebuilds from the manifests on disk: a column there would silently unarchive every
  part on the next reindex. Part ids are derived deterministically from the contract
  reference (``index.part_id_for``), so a row here keeps meaning the same part across a
  rebuild. **No foreign key to ``mf_contract_parts``, deliberately**: ``ON DELETE CASCADE``
  would erase the archive on every reindex, and ``RESTRICT`` would make the reindex fail.
"""

from __future__ import annotations

import sqlalchemy as sa
from alembic import op

revision: str = "0022_archive"
down_revision: str | None = "0021_approvals_and_settings"
branch_labels: str | None = None
depends_on: str | None = None


def upgrade() -> None:
    op.add_column("mf_runs", sa.Column("archived_at", sa.DateTime(timezone=True), nullable=True))
    op.create_table(
        "mf_part_archive",
        sa.Column("part_id", sa.String(64), nullable=False),
        sa.Column("archived_at", sa.DateTime(timezone=True), nullable=False),
        sa.PrimaryKeyConstraint("part_id", name="pk_mf_part_archive"),
    )


def downgrade() -> None:
    # Losing these loses only which items were hidden; every run and part is still there.
    op.drop_table("mf_part_archive")
    op.drop_column("mf_runs", "archived_at")
