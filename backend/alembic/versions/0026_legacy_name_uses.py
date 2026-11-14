"""Legacy names: which old ``MILLM_*`` environment names are in use (feature 015, FR-015.6).

Revision ID: 0026_legacy_name_uses
Revises: 0025_approval_batch
Create Date: 2026-11-08

miLLM becomes miServer (miSuite ENH-009). Until suite phase P7 miForge reads ``MISERVER_*`` first
and the old ``MILLM_*`` name as a fallback, and P7 may remove the fallback only after seven days
in which no process used an old name. That evidence must survive a restart, so each process
records what it saw here and ``GET /api/health`` serves the union of the live record and these
rows from the last 24 hours.

- ``mf_legacy_name_uses``: one row per ``(kind, old_name, state)``, ``last_seen`` advanced at
  most once a minute per process. ``kind`` admits the five kinds of the suite's health block;
  miForge writes only ``env``.
- ``mf_legacy_name_write_errors``: one row per reporter (``api@host``, ``train@pod``,
  ``beat@host``) whose last write FAILED, cleared by its next success. A failed write must read
  as "unmeasured", never as "nothing in use" (CF-25 m2).

No value of any variable is stored (ADR-020). Not an index table: reindex never drops it. The
CHECK constraints are spelled out here so this revision describes the schema as it was on this
date. Nothing is backfilled.
"""

from __future__ import annotations

import sqlalchemy as sa
from alembic import op

revision: str = "0026_legacy_name_uses"
down_revision: str | None = "0025_approval_batch"
branch_labels: str | None = None
depends_on: str | None = None


def upgrade() -> None:
    op.create_table(
        "mf_legacy_name_uses",
        sa.Column("kind", sa.String(16), nullable=False),
        sa.Column("old_name", sa.String(64), nullable=False),
        sa.Column("state", sa.String(16), nullable=False),
        sa.Column("new_name", sa.String(64), nullable=True),
        sa.Column("first_seen", sa.DateTime(timezone=True), nullable=False),
        sa.Column("last_seen", sa.DateTime(timezone=True), nullable=False),
        sa.CheckConstraint(
            "kind IN ('env', 'host', 'header', 'tool', 'category')",
            name=op.f("ck_mf_legacy_name_uses_kind_known"),
        ),
        sa.CheckConstraint(
            "state IN ('in_use', 'also_set', 'conflict', 'retired_but_set')",
            name=op.f("ck_mf_legacy_name_uses_state_known"),
        ),
        sa.PrimaryKeyConstraint("kind", "old_name", "state", name=op.f("pk_mf_legacy_name_uses")),
    )
    op.create_index("ix_mf_legacy_name_uses_last_seen", "mf_legacy_name_uses", ["last_seen"])
    op.create_table(
        "mf_legacy_name_write_errors",
        sa.Column("reporter", sa.String(128), nullable=False),
        sa.Column("error", sa.String(64), nullable=False),
        sa.Column("failed_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("cleared_at", sa.DateTime(timezone=True), nullable=True),
        sa.PrimaryKeyConstraint("reporter", name=op.f("pk_mf_legacy_name_write_errors")),
    )


def downgrade() -> None:
    op.drop_table("mf_legacy_name_write_errors")
    op.drop_index("ix_mf_legacy_name_uses_last_seen", table_name="mf_legacy_name_uses")
    op.drop_table("mf_legacy_name_uses")
