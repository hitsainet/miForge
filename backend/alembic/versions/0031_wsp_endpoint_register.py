"""The ``wsp_endpoint_register`` approval action: an agent's proposal of a local editor endpoint
(2026-11-20; ``forge_wsp_register_endpoint``).

Revision ID: 0031_wsp_endpoint_register
Revises: 0030_wsp_live_progress
Create Date: 2026-11-20

``mf_approvals``' action CHECK widens to admit ``wsp_endpoint_register``: an agent's
``POST /wsp/endpoints`` for a LOCAL endpoint with no key and no prices waits for the operator
under that action, and approving it registers the endpoint once. Nothing else changes; no row is
written.

The ``0028_wsp_provider`` pattern, unchanged: PostgreSQL drops and re-creates the CHECK in place;
SQLite rebuilds the table inside ``batch_alter_table`` with ``copy_from`` describing the table as it
stands (SQLAlchemy does not reflect SQLite CHECKs, so a reflected rebuild would drop the other three
CHECKs and the index while reporting success). ``test_migration_0031.py`` pins both.

**The downgrade refuses while any approval uses the new action**, and names the count: narrowing
the CHECK under such a row would fail half-way on PostgreSQL and silently drop it on a SQLite
rebuild.
"""

from __future__ import annotations

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

revision: str = "0031_wsp_endpoint_register"
down_revision: str | None = "0030_wsp_live_progress"
branch_labels: str | None = None
depends_on: str | None = None

#: The action CHECK as 0028 left it, and as this revision widens it.
OLD_ACTIONS = (
    "action IN ('gpu_start', 'hotupdate_attach', 'delete', 'config_write', "
    "'wsp_session', 'wsp_session_external')"
)
NEW_ACTIONS = (
    "action IN ('gpu_start', 'hotupdate_attach', 'delete', 'config_write', "
    "'wsp_session', 'wsp_session_external', 'wsp_endpoint_register')"
)
NEW_ACTION = "wsp_endpoint_register"

#: 0028's names: the action CHECK under the model's name, the other three still double-prefixed
#: (0021's naming defect, recorded as debt there).
DOUBLE = "ck_mf_approvals_ck_mf_approvals_"
ACTION_CHECK = "ck_mf_approvals_action_known"

_JSON = sa.JSON().with_variant(postgresql.JSONB(), "postgresql")


def _mf_approvals(actions: str) -> sa.Table:
    """``mf_approvals`` as it stands at ``0030`` (0021, 0025's ``batch_id``, 0028's action CHECK),
    for ``copy_from``."""
    return sa.Table(
        "mf_approvals",
        sa.MetaData(),
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
        sa.Column("batch_id", sa.String(40), nullable=True),
        sa.PrimaryKeyConstraint("id", name="pk_mf_approvals"),
        sa.CheckConstraint(
            "status IN ('pending', 'executing', 'executed', 'rejected', 'expired', 'failed')",
            name=f"{DOUBLE}status_known",
        ),
        sa.CheckConstraint(actions, name=ACTION_CHECK),
        sa.CheckConstraint(
            "(status = 'executed') = (result IS NOT NULL)",
            name=f"{DOUBLE}executed_iff_result",
        ),
        sa.CheckConstraint(
            "(status = 'failed') = (error IS NOT NULL)",
            name=f"{DOUBLE}failed_iff_error",
        ),
        sa.Index("ix_mf_approvals_status_created_at", "status", "created_at"),
    )


def _set_actions(*, from_actions: str, to_actions: str) -> None:
    if op.get_bind().dialect.name == "postgresql":
        op.drop_constraint(op.f(ACTION_CHECK), "mf_approvals", type_="check")
        op.create_check_constraint(op.f(ACTION_CHECK), "mf_approvals", to_actions)
        return
    with op.batch_alter_table("mf_approvals", copy_from=_mf_approvals(from_actions)) as batch:
        batch.drop_constraint(op.f(ACTION_CHECK), type_="check")
        batch.create_check_constraint(op.f(ACTION_CHECK), to_actions)


def upgrade() -> None:
    _set_actions(from_actions=OLD_ACTIONS, to_actions=NEW_ACTIONS)


def downgrade() -> None:
    in_use = (
        op.get_bind()
        .execute(sa.text("SELECT COUNT(*) FROM mf_approvals WHERE action = :a"), {"a": NEW_ACTION})
        .scalar_one()
    )
    if in_use:
        raise RuntimeError(
            f"Refusing to downgrade 0031_wsp_endpoint_register: {in_use} approval row(s) use "
            f"{NEW_ACTION}. Narrowing the action CHECK would fail on PostgreSQL and drop them on a "
            "SQLite rebuild; decide or delete them first."
        )
    _set_actions(from_actions=NEW_ACTIONS, to_actions=OLD_ACTIONS)
