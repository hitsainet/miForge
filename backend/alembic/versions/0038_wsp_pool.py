"""WSP worker pool (feature 019, phase P3; FR-019.6-FR-019.8, FR-019.18; TDD §4.3; F19-D17, D19, D32).

Revision ID: 0038_wsp_pool
Revises: 0037_wsp_protocol_pass
Create Date: 2026-11-25

* ``mf_wsp_runs.schedule_key``, ``worker_index`` (CHECK ``>= 0``), ``push_heartbeat_at``: written at
  the claim (``wsp.dispatch``); null on every queued row and on every run before 019.
* ``uq_mf_wsp_runs_one_active`` (one active run in all, 016) is replaced by
  ``uq_mf_wsp_runs_one_active_key`` (one per scheduling key) and ``uq_mf_wsp_runs_one_per_worker``
  (one run holding each worker), both partial unique indexes on PostgreSQL and SQLite alike.
* ``mf_wsp_workers``: what the driver last read from each provisioned worker (the pool route).

**The downgrade refuses with more than one active run**, naming them: the old index cannot be built
over two. SQLite cannot drop a column an index names, so the indexes go first.
"""

from __future__ import annotations

import sqlalchemy as sa
from alembic import op

revision: str = "0038_wsp_pool"
down_revision: str | None = "0037_wsp_protocol_pass"
branch_labels: str | None = None
depends_on: str | None = None

ACTIVE = "status IN ('pushing', 'running')"
HOLDING = "status IN ('pushing', 'running', 'collecting')"
COLUMNS: tuple[tuple[str, str, str], ...] = (
    ("mf_wsp_runs", "schedule_key", "VARCHAR(600)"),
    (
        "mf_wsp_runs",
        "worker_index",
        "INTEGER CONSTRAINT ck_mf_wsp_runs_worker_index_non_negative CHECK (worker_index >= 0)",
    ),
)
#: The lease's timestamp through Alembic (each engine its own type; 019 FTASKS 3.2).
HEARTBEAT = ("mf_wsp_runs", "push_heartbeat_at")
STATES = "('idle', 'receiving', 'running', 'finished', 'exiting', 'unknown', 'not_answering')"


def upgrade() -> None:
    for table, column, ddl in COLUMNS:
        op.execute(f"ALTER TABLE {table} ADD COLUMN {column} {ddl}")
    op.add_column(HEARTBEAT[0], sa.Column(HEARTBEAT[1], sa.DateTime(timezone=True), nullable=True))
    op.drop_index("uq_mf_wsp_runs_one_active", table_name="mf_wsp_runs")
    key = sa.text(f"{ACTIVE} AND schedule_key IS NOT NULL")
    op.create_index(
        "uq_mf_wsp_runs_one_active_key",
        "mf_wsp_runs",
        ["schedule_key"],
        unique=True,
        postgresql_where=key,
        sqlite_where=key,
    )
    held = sa.text(f"{HOLDING} AND worker_index IS NOT NULL")
    op.create_index(
        "uq_mf_wsp_runs_one_per_worker",
        "mf_wsp_runs",
        ["worker_index"],
        unique=True,
        postgresql_where=held,
        sqlite_where=held,
    )
    op.create_table(
        "mf_wsp_workers",
        sa.Column("worker_index", sa.Integer, nullable=False, autoincrement=False),
        sa.Column("address", sa.String(256), nullable=False),
        sa.Column("state", sa.String(16), nullable=False),
        sa.Column("boot_id", sa.String(32), nullable=True),
        sa.Column("read_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("answered_at", sa.DateTime(timezone=True), nullable=True),
        sa.CheckConstraint(f"state IN {STATES}", name=op.f("ck_mf_wsp_workers_state_known")),
        sa.CheckConstraint("worker_index >= 0", name=op.f("ck_mf_wsp_workers_index_non_negative")),
        sa.PrimaryKeyConstraint("worker_index", name=op.f("pk_mf_wsp_workers")),
    )


def downgrade() -> None:
    bind = op.get_bind()
    active = bind.execute(sa.text(f"SELECT run_id FROM mf_wsp_runs WHERE {ACTIVE}")).scalars().all()
    if len(active) > 1:
        raise RuntimeError(
            f"0038 downgrade refused: {len(active)} runs are active ({', '.join(active)}); the "
            "one-active index cannot be built over them. Let them finish first."
        )
    op.drop_table("mf_wsp_workers")
    op.drop_index("uq_mf_wsp_runs_one_per_worker", table_name="mf_wsp_runs")
    op.drop_index("uq_mf_wsp_runs_one_active_key", table_name="mf_wsp_runs")
    op.create_index(
        "uq_mf_wsp_runs_one_active",
        "mf_wsp_runs",
        [sa.text("(1)")],
        unique=True,
        postgresql_where=sa.text(ACTIVE),
        sqlite_where=sa.text(ACTIVE),
    )
    op.drop_column(*HEARTBEAT)
    for table, column, _ddl in reversed(COLUMNS):
        op.drop_column(table, column)
