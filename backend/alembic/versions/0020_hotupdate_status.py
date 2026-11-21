"""F11 16.4 / operator Q1: the attachment's live state, written by the trainer, read by the UI.

Revision ID: 0020_hotupdate_status
Revises: 0019_batches
Create Date: 2026-10-16

Operator decision (2026-10-16): the trainer writes its attachment's state into the run record
it already heartbeats, and a read-only route serves it - so the attachment chip renders the
real ``weights_state``, resident model, holder and staleness, **without a second importer of
``training/hotupdate``** (the API reads a column; it does not ask miLLM).

One nullable JSON column, ``hotupdate_status``: ``{weights_state, resident_model, holder,
stale, attached, observed_at}`` exactly as last observed from miLLM's status endpoint. Null
for every acquisition mode but §5.10, and null until the first observation - "not reported",
never a guessed ``clean``. Additive; no rebuild; reindex unaffected (run state, not artifact).
"""

from __future__ import annotations

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

revision: str = "0020_hotupdate_status"
down_revision: str | None = "0019_batches"
branch_labels: str | None = None
depends_on: str | None = None


def upgrade() -> None:
    op.add_column(
        "mf_runs",
        sa.Column(
            "hotupdate_status",
            sa.JSON().with_variant(postgresql.JSONB(), "postgresql"),
            nullable=True,
        ),
    )


def downgrade() -> None:
    # Losing it loses only the last observation; the trainer writes a fresh one.
    op.drop_column("mf_runs", "hotupdate_status")
