"""base schema: the seven tables more than one feature reads  (foundation task 4.2)

ADR-003 (PostgreSQL, ``mf_`` prefix, Alembic) and ADR-004 (the filesystem is the source of
truth; this database is a rebuildable index).

**What is here and what is not.** This migration creates only the tables **more than one
feature reads** - ``mf_runs``, ``mf_steps``, ``mf_packs``, ``mf_pack_parts``,
``mf_checkpoints``, ``mf_adapters``, ``mf_leases``. A table only its own feature touches
ships with **that feature's** migration; that is the boundary rule the backlog states, and
without it this file becomes every feature's migration and no feature owns its own schema.

**The three-state ``domain`` is enforced here, not only in Python** (task 4.2a). Two CHECK
constraints on ``mf_checkpoints`` keep ``asserted_null`` and ``absent`` distinct, and two
more on ``mf_adapters`` make it impossible to store a domain on an adapter at all - so the
inference from an adapter's base, which 4.2a forbids, fails at the database rather than in
a review.

**``batch_alter_table`` for the indexes** so this migration is reversible on SQLite as well
as PostgreSQL: the unit suite runs on SQLite, and a migration that only reverses on one
engine is a migration whose downgrade is untested.

Revision ID: 0001
Revises:
"""

from __future__ import annotations

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op
from sqlalchemy import Text
from sqlalchemy.dialects import postgresql

revision: str = "0001"
down_revision: str | None = None
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None

#: ``JSONB`` on PostgreSQL, plain ``JSON`` elsewhere - the same type
#: :data:`db.base.JsonColumn` declares. JSON rather than shredded columns wherever a
#: manifest object is stored, because ``ScoreRecord.provider`` is nullable AND not
#: ``required``: only a JSON round trip preserves the difference between an omitted key and
#: an explicit null, which is the same distinction ``domain_state`` exists for.
JSON_COLUMN = sa.JSON().with_variant(postgresql.JSONB(astext_type=Text()), "postgresql")


def upgrade() -> None:
    op.create_table(
        "mf_adapters",
        sa.Column("name", sa.String(length=512), nullable=False),
        sa.Column("run_id", sa.String(length=64), nullable=False),
        sa.Column("step", sa.Integer(), nullable=False),
        sa.Column("domain", sa.String(length=128), nullable=True),
        sa.Column("domain_state", sa.String(length=16), nullable=False),
        sa.Column("base_name", sa.String(length=512), nullable=False),
        sa.Column("base_manifest_sha256", sa.String(length=64), nullable=True),
        sa.Column("weights_sha256", sa.String(length=64), nullable=False),
        sa.Column("manifest_sha256", sa.String(length=64), nullable=False),
        sa.Column("adapter_config", JSON_COLUMN, nullable=True),
        sa.Column("optimizer", JSON_COLUMN, nullable=True),
        sa.Column("parts", JSON_COLUMN, nullable=True),
        sa.Column("test", JSON_COLUMN, nullable=True),
        sa.Column("retention", JSON_COLUMN, nullable=True),
        sa.Column("path", sa.Text(), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column(
            "indexed_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("(CURRENT_TIMESTAMP)"),
            nullable=False,
        ),
        sa.CheckConstraint(
            "domain_state = 'not_in_kind'", name=op.f("ck_mf_adapters_domain_state_is_not_in_kind")
        ),
        sa.CheckConstraint(
            "domain IS NULL", name=op.f("ck_mf_adapters_domain_never_inferred_from_base")
        ),
        sa.CheckConstraint("step >= 0", name=op.f("ck_mf_adapters_step_non_negative")),
        sa.PrimaryKeyConstraint("name", name=op.f("pk_mf_adapters")),
    )
    with op.batch_alter_table("mf_adapters", schema=None) as batch_op:
        batch_op.create_index("ix_mf_adapters_base_name", ["base_name"], unique=False)
        batch_op.create_index("ix_mf_adapters_run_id_step", ["run_id", "step"], unique=False)

    op.create_table(
        "mf_checkpoints",
        sa.Column("name", sa.String(length=512), nullable=False),
        sa.Column("run_id", sa.String(length=64), nullable=False),
        sa.Column("step", sa.Integer(), nullable=False),
        sa.Column("domain", sa.String(length=128), nullable=True),
        sa.Column("domain_state", sa.String(length=16), nullable=False),
        sa.Column("signal_source", sa.String(length=32), nullable=False),
        sa.Column("parent", sa.String(length=512), nullable=True),
        sa.Column("base_model", sa.String(length=512), nullable=False),
        sa.Column("merged_adapter", JSON_COLUMN, nullable=True),
        sa.Column("weights_sha256", sa.String(length=64), nullable=False),
        sa.Column("manifest_sha256", sa.String(length=64), nullable=False),
        sa.Column("selector_hash", sa.String(length=64), nullable=True),
        sa.Column("grader_hash", sa.String(length=64), nullable=True),
        sa.Column("optimizer", JSON_COLUMN, nullable=True),
        sa.Column("test", JSON_COLUMN, nullable=True),
        sa.Column("retention", JSON_COLUMN, nullable=True),
        sa.Column("path", sa.Text(), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column(
            "indexed_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("(CURRENT_TIMESTAMP)"),
            nullable=False,
        ),
        sa.CheckConstraint(
            "(domain_state = 'recorded') = (domain IS NOT NULL)",
            name=op.f("ck_mf_checkpoints_domain_value_iff_recorded"),
        ),
        sa.CheckConstraint(
            "domain_state IN ('recorded','asserted_null','absent')",
            name=op.f("ck_mf_checkpoints_domain_state_is_a_checkpoint_state"),
        ),
        sa.CheckConstraint(
            "signal_source IN ('corpus','self-generation','preference-pairs','reward')",
            name=op.f("ck_mf_checkpoints_signal_source_known"),
        ),
        sa.CheckConstraint("step >= 0", name=op.f("ck_mf_checkpoints_step_non_negative")),
        sa.PrimaryKeyConstraint("name", name=op.f("pk_mf_checkpoints")),
    )
    with op.batch_alter_table("mf_checkpoints", schema=None) as batch_op:
        batch_op.create_index("ix_mf_checkpoints_domain_state", ["domain_state"], unique=False)
        batch_op.create_index("ix_mf_checkpoints_parent", ["parent"], unique=False)
        batch_op.create_index("ix_mf_checkpoints_run_id_step", ["run_id", "step"], unique=False)

    op.create_table(
        "mf_leases",
        sa.Column("id", sa.Integer(), autoincrement=True, nullable=False),
        sa.Column("resource", sa.String(length=64), nullable=False),
        sa.Column("holder_run_id", sa.String(length=64), nullable=False),
        sa.Column(
            "acquired_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("(CURRENT_TIMESTAMP)"),
            nullable=False,
        ),
        sa.Column("expires_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("released_at", sa.DateTime(timezone=True), nullable=True),
        sa.CheckConstraint(
            "expires_at > acquired_at", name=op.f("ck_mf_leases_expiry_after_acquisition")
        ),
        sa.PrimaryKeyConstraint("id", name=op.f("pk_mf_leases")),
    )
    with op.batch_alter_table("mf_leases", schema=None) as batch_op:
        batch_op.create_index("ix_mf_leases_holder_run_id", ["holder_run_id"], unique=False)
        batch_op.create_index(
            "uq_mf_leases_resource_unreleased",
            ["resource"],
            unique=True,
            sqlite_where=sa.text("released_at IS NULL"),
            postgresql_where=sa.text("released_at IS NULL"),
        )

    op.create_table(
        "mf_packs",
        sa.Column("id", sa.String(length=64), nullable=False),
        sa.Column("name", sa.String(length=128), nullable=False),
        sa.Column("version", sa.String(length=64), nullable=False),
        sa.Column("valid", sa.Boolean(), nullable=False),
        sa.Column("invalid_reason", sa.Text(), nullable=True),
        sa.Column("definition_sha256", sa.String(length=64), nullable=True),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("(CURRENT_TIMESTAMP)"),
            nullable=False,
        ),
        sa.CheckConstraint(
            "valid OR invalid_reason IS NOT NULL", name=op.f("ck_mf_packs_invalid_names_a_reason")
        ),
        sa.PrimaryKeyConstraint("id", name=op.f("pk_mf_packs")),
    )
    with op.batch_alter_table("mf_packs", schema=None) as batch_op:
        batch_op.create_index("uq_mf_packs_name_version", ["name", "version"], unique=True)

    op.create_table(
        "mf_pack_parts",
        sa.Column("id", sa.Integer(), autoincrement=True, nullable=False),
        sa.Column("pack_id", sa.String(length=64), nullable=False),
        sa.Column("part_kind", sa.String(length=32), nullable=False),
        sa.Column("source", sa.String(length=16), nullable=False),
        sa.Column("ref_type", sa.String(length=64), nullable=True),
        sa.Column("ref_name", sa.String(length=128), nullable=True),
        sa.Column("ref_version", sa.String(length=64), nullable=True),
        sa.Column("external_provenance", JSON_COLUMN, nullable=True),
        sa.CheckConstraint(
            "(source = 'midataworks' AND ref_type IS NOT NULL AND ref_name IS NOT NULL "
            "AND ref_version IS NOT NULL) OR "
            "(source = 'external' AND external_provenance IS NOT NULL)",
            name=op.f("ck_mf_pack_parts_part_is_pinned"),
        ),
        sa.CheckConstraint(
            "source IN ('midataworks','external')", name=op.f("ck_mf_pack_parts_source_known")
        ),
        sa.ForeignKeyConstraint(
            ["pack_id"],
            ["mf_packs.id"],
            name=op.f("fk_mf_pack_parts_pack_id_mf_packs"),
            ondelete="CASCADE",
        ),
        sa.PrimaryKeyConstraint("id", name=op.f("pk_mf_pack_parts")),
    )
    with op.batch_alter_table("mf_pack_parts", schema=None) as batch_op:
        batch_op.create_index(
            "uq_mf_pack_parts_pack_id_part_kind", ["pack_id", "part_kind"], unique=True
        )

    op.create_table(
        "mf_runs",
        sa.Column("id", sa.String(length=64), nullable=False),
        sa.Column("status", sa.String(length=16), nullable=False),
        sa.Column("run_type", sa.String(length=16), nullable=False),
        sa.Column("definition_submitted", JSON_COLUMN, nullable=True),
        sa.Column("definition_resolved", JSON_COLUMN, nullable=True),
        sa.Column("definition_sha256", sa.String(length=64), nullable=True),
        sa.Column("pack_id", sa.String(length=64), nullable=True),
        sa.Column("gpu_request", sa.String(length=64), nullable=True),
        sa.Column("gpu_uuid", sa.String(length=64), nullable=True),
        sa.Column("heartbeat_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("longest_heartbeat_gap_seconds", sa.Integer(), nullable=True),
        sa.Column("cancel_requested_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("last_step", sa.Integer(), nullable=True),
        sa.Column("failure_code", sa.String(length=64), nullable=True),
        sa.Column("failure_message", sa.Text(), nullable=True),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("(CURRENT_TIMESTAMP)"),
            nullable=False,
        ),
        sa.Column(
            "updated_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("(CURRENT_TIMESTAMP)"),
            nullable=False,
        ),
        sa.Column("finished_at", sa.DateTime(timezone=True), nullable=True),
        sa.CheckConstraint(
            "(status IN ('completed','failed','cancelled')) = (finished_at IS NOT NULL)",
            name=op.f("ck_mf_runs_terminal_iff_finished"),
        ),
        sa.CheckConstraint(
            "run_type IN ('create','full','adapter','merge')",
            name=op.f("ck_mf_runs_run_type_known"),
        ),
        sa.CheckConstraint(
            "status IN ('pending','running','completed','failed','cancelled')",
            name=op.f("ck_mf_runs_status_known"),
        ),
        sa.ForeignKeyConstraint(
            ["pack_id"],
            ["mf_packs.id"],
            name=op.f("fk_mf_runs_pack_id_mf_packs"),
            ondelete="RESTRICT",
        ),
        sa.PrimaryKeyConstraint("id", name=op.f("pk_mf_runs")),
    )
    with op.batch_alter_table("mf_runs", schema=None) as batch_op:
        batch_op.create_index(
            "ix_mf_runs_status_created_at", ["status", "created_at"], unique=False
        )

    op.create_table(
        "mf_steps",
        sa.Column("id", sa.Integer(), autoincrement=True, nullable=False),
        sa.Column("run_id", sa.String(length=64), nullable=False),
        sa.Column("step", sa.Integer(), nullable=False),
        sa.Column("loss", sa.Float(), nullable=True),
        sa.Column("grad_norm", sa.Float(), nullable=True),
        sa.Column("learning_rate", sa.Float(), nullable=True),
        sa.Column("wall_time_ms", sa.Integer(), nullable=True),
        sa.Column("batch_source_hashes", JSON_COLUMN, nullable=True),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("(CURRENT_TIMESTAMP)"),
            nullable=False,
        ),
        sa.CheckConstraint("step >= 0", name=op.f("ck_mf_steps_step_non_negative")),
        sa.ForeignKeyConstraint(
            ["run_id"], ["mf_runs.id"], name=op.f("fk_mf_steps_run_id_mf_runs"), ondelete="CASCADE"
        ),
        sa.PrimaryKeyConstraint("id", name=op.f("pk_mf_steps")),
    )
    with op.batch_alter_table("mf_steps", schema=None) as batch_op:
        batch_op.create_index("uq_mf_steps_run_id_step", ["run_id", "step"], unique=True)


def downgrade() -> None:
    with op.batch_alter_table("mf_steps", schema=None) as batch_op:
        batch_op.drop_index("uq_mf_steps_run_id_step")

    op.drop_table("mf_steps")
    with op.batch_alter_table("mf_runs", schema=None) as batch_op:
        batch_op.drop_index("ix_mf_runs_status_created_at")

    op.drop_table("mf_runs")
    with op.batch_alter_table("mf_pack_parts", schema=None) as batch_op:
        batch_op.drop_index("uq_mf_pack_parts_pack_id_part_kind")

    op.drop_table("mf_pack_parts")
    with op.batch_alter_table("mf_packs", schema=None) as batch_op:
        batch_op.drop_index("uq_mf_packs_name_version")

    op.drop_table("mf_packs")
    with op.batch_alter_table("mf_leases", schema=None) as batch_op:
        batch_op.drop_index(
            "uq_mf_leases_resource_unreleased",
            sqlite_where=sa.text("released_at IS NULL"),
            postgresql_where=sa.text("released_at IS NULL"),
        )
        batch_op.drop_index("ix_mf_leases_holder_run_id")

    op.drop_table("mf_leases")
    with op.batch_alter_table("mf_checkpoints", schema=None) as batch_op:
        batch_op.drop_index("ix_mf_checkpoints_run_id_step")
        batch_op.drop_index("ix_mf_checkpoints_parent")
        batch_op.drop_index("ix_mf_checkpoints_domain_state")

    op.drop_table("mf_checkpoints")
    with op.batch_alter_table("mf_adapters", schema=None) as batch_op:
        batch_op.drop_index("ix_mf_adapters_run_id_step")
        batch_op.drop_index("ix_mf_adapters_base_name")

    op.drop_table("mf_adapters")
