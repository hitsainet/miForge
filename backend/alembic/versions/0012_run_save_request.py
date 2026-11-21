"""mf_runs gains the save-request flag.

Revision ID: 0012_save_req
Revises: 0011_step_obj
Create Date: 2026-10-14

FR-005.1 says the save request is *"a flag on the ``mf_runs`` row"*. F5 built it as a
separate table on its own ``MetaData`` because ``backend/src/models/`` is not a feature
agent's to edit - correct of it to deviate rather than reach, and it said so and offered
the straight reading. This is that straight reading.

**It is also the better engineering, not merely the literal one.** The loop reads the
``mf_runs`` row at every step boundary already - for the heartbeat and for
``cancel_requested_at`` - so a separate table would add a join to the hottest path in the
product in order to carry one timestamp.

A **timestamp** rather than a boolean, matching ``cancel_requested_at``: it answers "when
was it asked", which is what distinguishes a slow save from an ignored request. Two
requests before the next boundary collapse, because one save is what was wanted;
``save_request_count`` keeps the fact that two were asked for.

Chained off ``0011_step_obj``, the then-head. Migrations are serialised through the
coordinator while feature agents run concurrently: two revisions off one parent give
everyone multiple heads, and it fails on someone else's clean clone.
"""

from __future__ import annotations

import sqlalchemy as sa
from alembic import op

revision: str = "0012_save_req"
down_revision: str | None = "0011_step_obj"
branch_labels: str | None = None
depends_on: str | None = None


def upgrade() -> None:
    # Nullable timestamp: NULL means "no save requested", which is the normal state and an
    # honest one. The count carries a server default so existing rows get 0 rather than
    # NULL - a count that is NULL would have to be special-cased at every read.
    op.add_column(
        "mf_runs", sa.Column("save_requested_at", sa.DateTime(timezone=True), nullable=True)
    )
    op.add_column(
        "mf_runs",
        sa.Column("save_request_count", sa.Integer(), nullable=False, server_default="0"),
    )


def downgrade() -> None:
    op.drop_column("mf_runs", "save_request_count")
    op.drop_column("mf_runs", "save_requested_at")
