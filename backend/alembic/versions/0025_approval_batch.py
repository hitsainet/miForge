"""Approvals: the batch a request was approved in (ENH-006 D79, 2026-10-29).

Revision ID: 0025_approval_batch
Revises: 0024_setting_changes
Create Date: 2026-10-29

``mf_approvals.batch_id`` - nullable. The operator can approve every waiting settings change
with one action (``POST /api/v1/approvals/approve-batch``). Each request is still approved,
executed and recorded on its own row exactly as a single approve records it; the rows one
action decided share this id, so History can say they were approved together. ``NULL`` means
the request was decided on its own (every row written before this revision, and every reject).

Additive, no rebuild, no backfill: a request decided before this revision was decided on its
own, and nothing is invented for it. **No index**: nothing filters by batch; the id is read off
a row the listing already returns.
"""

from __future__ import annotations

import sqlalchemy as sa
from alembic import op

revision: str = "0025_approval_batch"
down_revision: str | None = "0024_setting_changes"
branch_labels: str | None = None
depends_on: str | None = None


def upgrade() -> None:
    op.add_column("mf_approvals", sa.Column("batch_id", sa.String(40), nullable=True))


def downgrade() -> None:
    # Losing it loses only which decisions were made together; every row and outcome stays.
    op.drop_column("mf_approvals", "batch_id")
