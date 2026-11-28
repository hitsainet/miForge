"""WSP editor wait: a queued run waits for its miServer model instead of assuming a window
(2026-11-27, run ``wsp_4fbf5a18758e``; ``0xcc/reviews/wsp_window_read_not_resident_2026-11-27.md``).

Revision ID: 0042_wsp_editor_wait
Revises: 0041_wsp_worker_disk
Create Date: 2026-11-27

* ``mf_wsp_runs.editor_wait_since`` - when the driver's claim first found the run's miServer
  endpoint (one that states no window) reporting no window, i.e. no model resident. The claim
  skips such a run and fails it ``editor_not_resident`` once ``WSP_EDITOR_RESIDENT_WAIT_S`` has
  passed since this time (``wsp.dispatch``). Null on every run that never waited, so nothing is
  backfilled; the column is not part of any run document.

Additive and nullable: neither engine rebuilds a table, and the downgrade simply drops it (a run
queued across the downgrade starts its wait again).
"""

from __future__ import annotations

import sqlalchemy as sa
from alembic import op

revision: str = "0042_wsp_editor_wait"
down_revision: str | None = "0041_wsp_worker_disk"
branch_labels: str | None = None
depends_on: str | None = None


def upgrade() -> None:
    op.add_column(
        "mf_wsp_runs", sa.Column("editor_wait_since", sa.DateTime(timezone=True), nullable=True)
    )


def downgrade() -> None:
    op.drop_column("mf_wsp_runs", "editor_wait_since")
