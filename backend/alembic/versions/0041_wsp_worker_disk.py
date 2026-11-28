"""WSP worker disk: each worker's ``/work`` filesystem, free and total (2026-11-26, the 019 P4 MCP
gap: ``forge_host_resources`` had no disk reading, so FTASKS 21.1 was read by hand with ``df``).

Revision ID: 0041_wsp_worker_disk
Revises: 0040_wsp_submitted_check
Create Date: 2026-11-26

* ``mf_wsp_workers.work_free_bytes`` and ``work_total_bytes`` - what the worker's ``/ready``
  reported for the filesystem under its ``/work`` emptyDir (``wsp_worker.supervisor.work_disk``),
  written by the driver's pass (``wsp.pool.record_ready``) and served by ``GET /host/resources``.

Additive and nullable: neither engine rebuilds a table. The rows are a cache the pass rewrites
every few seconds, so the downgrade simply drops both columns.
"""

from __future__ import annotations

import sqlalchemy as sa
from alembic import op

revision: str = "0041_wsp_worker_disk"
down_revision: str | None = "0040_wsp_submitted_check"
branch_labels: str | None = None
depends_on: str | None = None

COLUMNS = ("work_free_bytes", "work_total_bytes")


def upgrade() -> None:
    for name in COLUMNS:
        op.add_column("mf_wsp_workers", sa.Column(name, sa.BigInteger(), nullable=True))


def downgrade() -> None:
    for name in reversed(COLUMNS):
        op.drop_column("mf_wsp_workers", name)
