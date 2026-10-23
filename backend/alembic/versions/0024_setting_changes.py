"""Settings: the append-only change history (ENH-003 section 3.4, gap 3; 013 G3).

Revision ID: 0024_setting_changes
Revises: 0023_purge_tombstones
Create Date: 2026-10-20

``mf_setting_changes`` - one row per ``PUT`` or ``DELETE`` on ``/api/v1/settings/{key}``: the
key, the value in force before and after (with where each came from), who made the change
(``operator`` or ``agent:<name>`` with the approval id) and an optional reason. Until this
revision ``mf_settings`` kept only the last value and its writer, so an operator's change left
no record of what it replaced.

**Append-only, enforced in the database**, as ``0023_purge_tombstones`` does for its
tombstones: a trigger refuses ``UPDATE`` and ``DELETE`` on PostgreSQL (one plpgsql function, one
row-level trigger) and on SQLite (two ``RAISE(ABORT)`` triggers). No foreign key: the key names
a setting, not a row, and a reset deletes the override row this history outlives.

The CHECK constraints are spelled out here rather than imported from ``models.setting``, so
this revision describes the schema as it was on this date. Nothing is backfilled: an override
saved before this revision has no history, and none is invented.
"""

from __future__ import annotations

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

revision: str = "0024_setting_changes"
down_revision: str | None = "0023_purge_tombstones"
branch_labels: str | None = None
depends_on: str | None = None

_JSON = sa.JSON().with_variant(postgresql.JSONB(), "postgresql")

#: The refusal text both engines raise; the history tests read it back.
APPEND_ONLY_MESSAGE = (
    "mf_setting_changes is append-only: a recorded settings change is never changed or removed"
)


def upgrade() -> None:
    op.create_table(
        "mf_setting_changes",
        sa.Column("id", sa.Integer(), autoincrement=True, nullable=False),
        sa.Column("key", sa.String(64), nullable=False),
        sa.Column("action", sa.String(8), nullable=False),
        sa.Column("previous", _JSON, nullable=True),
        sa.Column("previous_source", sa.String(8), nullable=False),
        sa.Column("value", _JSON, nullable=True),
        sa.Column("source", sa.String(8), nullable=False),
        sa.Column("actor", sa.String(64), nullable=False),
        sa.Column("approval_id", sa.String(40), nullable=True),
        sa.Column("reason", sa.String(2000), nullable=True),
        sa.Column("changed_at", sa.DateTime(timezone=True), nullable=False),
        sa.CheckConstraint(
            "action IN ('set', 'reset')", name=op.f("ck_mf_setting_changes_action_known")
        ),
        sa.CheckConstraint(
            "source IN ('env', 'override')", name=op.f("ck_mf_setting_changes_source_known")
        ),
        sa.CheckConstraint(
            "previous_source IN ('env', 'override')",
            name=op.f("ck_mf_setting_changes_previous_source_known"),
        ),
        sa.PrimaryKeyConstraint("id", name=op.f("pk_mf_setting_changes")),
    )
    op.create_index("ix_mf_setting_changes_changed_at", "mf_setting_changes", ["changed_at"])
    op.create_index(
        "ix_mf_setting_changes_key_changed_at", "mf_setting_changes", ["key", "changed_at"]
    )

    if op.get_bind().dialect.name == "postgresql":
        op.execute(
            "CREATE FUNCTION mf_setting_changes_append_only() RETURNS trigger AS $$ "
            f"BEGIN RAISE EXCEPTION '{APPEND_ONLY_MESSAGE}'; END; $$ LANGUAGE plpgsql"
        )
        op.execute(
            "CREATE TRIGGER mf_setting_changes_append_only BEFORE UPDATE OR DELETE "
            "ON mf_setting_changes FOR EACH ROW EXECUTE FUNCTION mf_setting_changes_append_only()"
        )
    else:
        for verb in ("UPDATE", "DELETE"):
            op.execute(
                f"CREATE TRIGGER mf_setting_changes_no_{verb.lower()} BEFORE {verb} "
                f"ON mf_setting_changes BEGIN SELECT RAISE(ABORT, '{APPEND_ONLY_MESSAGE}'); END"
            )


def downgrade() -> None:
    # Dropping the table drops its triggers on both engines; the function is PostgreSQL's own.
    op.drop_index("ix_mf_setting_changes_key_changed_at", table_name="mf_setting_changes")
    op.drop_index("ix_mf_setting_changes_changed_at", table_name="mf_setting_changes")
    op.drop_table("mf_setting_changes")
    if op.get_bind().dialect.name == "postgresql":
        op.execute("DROP FUNCTION IF EXISTS mf_setting_changes_append_only()")
