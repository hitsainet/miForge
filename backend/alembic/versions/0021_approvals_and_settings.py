"""F12: the approval queue and the runtime settings overrides.

Revision ID: 0021_approvals_and_settings
Revises: 0020_hotupdate_status
Create Date: 2026-10-19

Two record tables (ADR-004: no artifact on disk carries either, so ``reindex`` never touches
them). ``mf_approvals`` holds an agent's request until the operator decides it
(FR-012.9 - .16); ``mf_settings`` holds the operator's runtime overrides for the allow-listed
keys (FR-012.8, D5).

The CHECK constraints are spelled out here rather than imported from the models, so this
revision describes the schema as it was on this date even if the models later change. The
``executed``/``result`` and ``failed``/``error`` pairs hold in both directions, the rule
``mf_merges`` uses. JSON columns are JSONB on PostgreSQL, as ``0019`` and every revision
after ``0018`` declare them.
"""

from __future__ import annotations

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

revision: str = "0021_approvals_and_settings"
down_revision: str | None = "0020_hotupdate_status"
branch_labels: str | None = None
depends_on: str | None = None

_JSON = sa.JSON().with_variant(postgresql.JSONB(), "postgresql")


def upgrade() -> None:
    op.create_table(
        "mf_approvals",
        sa.Column("id", sa.String(40), nullable=False),
        sa.Column("action", sa.String(32), nullable=False),
        sa.Column("executor_key", sa.String(200), nullable=False),
        sa.Column("summary", _JSON, nullable=False),
        sa.Column("payload", _JSON, nullable=False),
        sa.Column("request_digest", sa.String(64), nullable=False),
        sa.Column("requested_by", sa.String(64), nullable=False),
        sa.Column("status", sa.String(16), nullable=False),
        sa.Column("result", _JSON, nullable=True),
        sa.Column("error", _JSON, nullable=True),
        sa.Column("decided_by", sa.String(64), nullable=True),
        sa.Column("reason", sa.String(2000), nullable=True),
        sa.Column(
            "created_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.func.now()
        ),
        sa.Column("decided_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("expires_at", sa.DateTime(timezone=True), nullable=False),
        sa.PrimaryKeyConstraint("id", name="pk_mf_approvals"),
        sa.CheckConstraint(
            "status IN ('pending', 'executing', 'executed', 'rejected', 'expired', 'failed')",
            name="ck_mf_approvals_status_known",
        ),
        sa.CheckConstraint(
            "action IN ('gpu_start', 'hotupdate_attach', 'delete', 'config_write')",
            name="ck_mf_approvals_action_known",
        ),
        sa.CheckConstraint(
            "(status = 'executed') = (result IS NOT NULL)",
            name="ck_mf_approvals_executed_iff_result",
        ),
        sa.CheckConstraint(
            "(status = 'failed') = (error IS NOT NULL)",
            name="ck_mf_approvals_failed_iff_error",
        ),
    )
    op.create_index("ix_mf_approvals_status_created_at", "mf_approvals", ["status", "created_at"])
    op.create_table(
        "mf_settings",
        sa.Column("key", sa.String(64), nullable=False),
        sa.Column("value", _JSON, nullable=True),
        sa.Column(
            "updated_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.func.now()
        ),
        sa.Column("updated_by", sa.String(64), nullable=False),
        sa.PrimaryKeyConstraint("key", name="pk_mf_settings"),
    )


def downgrade() -> None:
    op.drop_table("mf_settings")
    op.drop_index("ix_mf_approvals_status_created_at", table_name="mf_approvals")
    op.drop_table("mf_approvals")
