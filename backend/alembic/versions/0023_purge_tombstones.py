"""Purge: the append-only tombstone table for permanent deletes (operator 2026-10-20).

Revision ID: 0023_purge_tombstones
Revises: 0022_archive
Create Date: 2026-10-20

``mf_purges`` - one row per purge of an archived run or part: what was removed, by whom, why,
and the rows, files and bytes that went with it. **Append-only, enforced in the database**:
a trigger refuses ``UPDATE`` and ``DELETE`` on PostgreSQL (one plpgsql function, one
row-level trigger) and on SQLite (two ``RAISE(ABORT)`` triggers), so the record of a
permanent delete cannot itself be quietly deleted. No foreign key: the item it names is gone.

The CHECK constraints are spelled out here rather than imported from
``services.purge.tables``, so this revision describes the schema as it was on this date.
"""

from __future__ import annotations

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

revision: str = "0023_purge_tombstones"
down_revision: str | None = "0022_archive"
branch_labels: str | None = None
depends_on: str | None = None

_JSON = sa.JSON().with_variant(postgresql.JSONB(), "postgresql")

#: The refusal text both engines raise; the purge tests read it back.
APPEND_ONLY_MESSAGE = "mf_purges is append-only: a purge tombstone is never changed or removed"


def upgrade() -> None:
    op.create_table(
        "mf_purges",
        sa.Column("id", sa.String(40), nullable=False),
        sa.Column("kind", sa.String(8), nullable=False),
        sa.Column("item_id", sa.String(64), nullable=False),
        sa.Column("name", sa.Text(), nullable=False),
        sa.Column("purged_by", sa.String(64), nullable=False),
        sa.Column("reason", sa.String(2000), nullable=True),
        sa.Column("rows_removed", _JSON, nullable=False),
        sa.Column("files_removed", sa.Integer(), nullable=False),
        sa.Column("bytes_removed", sa.BigInteger(), nullable=False),
        sa.Column("detail", _JSON, nullable=False),
        sa.Column("purged_at", sa.DateTime(timezone=True), nullable=False),
        sa.CheckConstraint("kind IN ('run', 'part')", name=op.f("ck_mf_purges_kind_known")),
        sa.CheckConstraint(
            "files_removed >= 0 AND bytes_removed >= 0",
            name=op.f("ck_mf_purges_counts_non_negative"),
        ),
        sa.PrimaryKeyConstraint("id", name=op.f("pk_mf_purges")),
    )
    op.create_index("ix_mf_purges_purged_at", "mf_purges", ["purged_at"])
    op.create_index("ix_mf_purges_kind_item_id", "mf_purges", ["kind", "item_id"])

    if op.get_bind().dialect.name == "postgresql":
        op.execute(
            "CREATE FUNCTION mf_purges_append_only() RETURNS trigger AS $$ "
            f"BEGIN RAISE EXCEPTION '{APPEND_ONLY_MESSAGE}'; END; $$ LANGUAGE plpgsql"
        )
        op.execute(
            "CREATE TRIGGER mf_purges_append_only BEFORE UPDATE OR DELETE ON mf_purges "
            "FOR EACH ROW EXECUTE FUNCTION mf_purges_append_only()"
        )
    else:
        for verb in ("UPDATE", "DELETE"):
            op.execute(
                f"CREATE TRIGGER mf_purges_no_{verb.lower()} BEFORE {verb} ON mf_purges "
                f"BEGIN SELECT RAISE(ABORT, '{APPEND_ONLY_MESSAGE}'); END"
            )


def downgrade() -> None:
    # Dropping the table drops its triggers on both engines; the function is PostgreSQL's own.
    op.drop_index("ix_mf_purges_kind_item_id", table_name="mf_purges")
    op.drop_index("ix_mf_purges_purged_at", table_name="mf_purges")
    op.drop_table("mf_purges")
    if op.get_bind().dialect.name == "postgresql":
        op.execute("DROP FUNCTION IF EXISTS mf_purges_append_only()")
