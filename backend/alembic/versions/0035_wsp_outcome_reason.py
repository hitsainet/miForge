"""WSP outcome reason: why a run whose patch changed weights is ``no_change`` (2026-11-23;
``0xcc/reviews/wsp_outcome_reason_2026-11-23.md``).

Revision ID: 0035_wsp_outcome_reason
Revises: 0034_wsp_paging
Create Date: 2026-11-23

* ``mf_wsp_runs.outcome_reason`` - the reason ``wsp.runs.qualify`` gave for keeping a verified
  patch as evidence instead of a candidate (FR-016.30, PF-20), e.g. "practice-validation AUROC
  0.990293 is below the parent's 0.990332". Null on every other run, and on every run finished
  before this migration: nothing is backfilled, because the reason was never stored.

Additive and nullable: neither engine rebuilds a table.
"""

from __future__ import annotations

import sqlalchemy as sa
from alembic import op

revision: str = "0035_wsp_outcome_reason"
down_revision: str | None = "0034_wsp_paging"
branch_labels: str | None = None
depends_on: str | None = None


def upgrade() -> None:
    op.add_column("mf_wsp_runs", sa.Column("outcome_reason", sa.String(300), nullable=True))


def downgrade() -> None:
    # Loses every stored reason; the runs' outcomes are unchanged.
    op.drop_column("mf_wsp_runs", "outcome_reason")
