"""F10: the part index and the review binding  (F10 task 5.3; R-02.36, R-02.37)

Four tables, two groups, **two different rules** - and the split has to be explicit here
or a reindex test gets written against the wrong half.

**Group A, index, rebuildable from disk:** ``mf_contract_parts``, ``mf_part_splits``. The
authority is ``<part_dir>/midataworks-dataset-version.json`` (ADR-004), exactly as a
checkpoint's authority is its ``manifest.json``.
``services.parts.index.reindex_parts`` rebuilds both from manifests alone.

**Group B, durable run state, NOT rebuildable:** ``mf_review_bindings``,
``mf_review_candidate_refs``. No local file can reconstruct a miDataworks queue id or an
``external_id`` -> ``item_id`` mapping, because the authority for a decision is
**miDataworks**. Losing these rows loses the ability to read decisions back.

**These are not a review queue.** R-02.36 puts the queue, the decision history and the
review screen at miDataworks, and holding a queue id is not holding a queue - R-02.36's own
closing sentence *requires* this map, because the decisions' identifiers are recorded with
the batch they admitted.

**Four constraints carry requirements rather than hygiene:**

* ``external_part_has_no_contract_reference`` - R-02.40 at the database. miForge does not
  silently promote an external part to contract provenance, and a null column is the
  enforcement rather than a convention three call sites are trusted to honour.
* ``contract_reference_is_whole_or_absent`` - two of three fields names nothing resolvable,
  and the schema's own ``ContractReference`` requires all three.
* ``only_override_carries_a_label`` - miDataworks enforces the same rule on the way in
  (``validate()``: *"Only an override carries a label"*), so a label beside an ``accept``
  here would mean miForge recorded something miDataworks never said.
* ``decision_known`` - the column stores **miDataworks' vocabulary as read** (``accept``,
  ``override``, ``flag``, ``reject``) plus ``none`` for *no decision yet*, never miForge's
  verdict value. A pending candidate is a **stored value**, not an absence: an absence
  invites a default, and "how many is this run waiting on" becomes a ``COUNT`` over an
  index rather than a set difference.

**Every timestamp is ``timezone=True``.** A naive column silently shifts the ``since``
watermark, and a shifted watermark loses decisions permanently.

**The partial unique index is partial for a reason.** Idempotent re-import is a database
guarantee rather than a service convention - but external parts and conformant versions
carrying a **null** ``contract_reference`` both have null reference columns, and several of
them must be able to coexist.

**``batch_alter_table`` for the indexes** so this migration reverses on SQLite as well as
PostgreSQL: the unit suite runs on SQLite, and a migration that only reverses on one engine
has an untested downgrade.

Revision ID: 0010_f010
Revises: 0001
"""

from __future__ import annotations

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op
from sqlalchemy import Text
from sqlalchemy.dialects import postgresql

revision: str = "0010_f010"
down_revision: str | None = "0001"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None

#: ``JSONB`` on PostgreSQL, plain ``JSON`` elsewhere - the same type
#: :data:`db.base.JsonColumn` declares. JSON rather than shredded columns for ``caveats``
#: and ``licences``, because their shape is **miDataworks'**: normalising them into tables
#: would be miForge taking an opinion on someone else's vocabulary, and FR-010.17 requires
#: them carried, not understood.
JSON_COLUMN = sa.JSON().with_variant(postgresql.JSONB(astext_type=Text()), "postgresql")

_SOURCE_CONTRACT = "midataworks_contract"
_SOURCE_EXTERNAL = "external_files"


def upgrade() -> None:
    op.create_table(
        "mf_contract_parts",
        sa.Column("id", sa.String(length=64), nullable=False),
        sa.Column("part_kind", sa.String(length=32), nullable=False),
        sa.Column("source", sa.String(length=24), nullable=False),
        sa.Column("ref_type", sa.String(length=64), nullable=True),
        sa.Column("ref_name", sa.String(length=128), nullable=True),
        sa.Column("ref_version", sa.String(length=64), nullable=True),
        sa.Column("dataset", sa.String(length=128), nullable=True),
        sa.Column("version_id", sa.String(length=64), nullable=True),
        sa.Column("version_number", sa.Integer(), nullable=True),
        sa.Column("recipe_sha256", sa.String(length=64), nullable=True),
        sa.Column("content_kind", sa.String(length=24), nullable=False),
        sa.Column("config_sha256", sa.String(length=64), nullable=True),
        sa.Column("local_path", sa.Text(), nullable=False),
        sa.Column("manifest_sha256", sa.String(length=64), nullable=False),
        sa.Column("row_count", sa.Integer(), server_default="0", nullable=False),
        sa.Column("licence_classes", JSON_COLUMN, nullable=True),
        sa.Column("licences", JSON_COLUMN, nullable=True),
        sa.Column("caveats", JSON_COLUMN, nullable=True),
        sa.Column("external_provenance", JSON_COLUMN, nullable=True),
        sa.Column("imported_at", sa.DateTime(timezone=True), nullable=False),
        sa.CheckConstraint(
            f"source IN ('{_SOURCE_CONTRACT}', '{_SOURCE_EXTERNAL}')",
            name=op.f("ck_mf_contract_parts_source_known"),
        ),
        sa.CheckConstraint(
            f"source <> '{_SOURCE_EXTERNAL}' OR "
            "(ref_type IS NULL AND ref_name IS NULL AND ref_version IS NULL)",
            name=op.f("ck_mf_contract_parts_external_part_has_no_contract_reference"),
        ),
        sa.CheckConstraint(
            "(ref_type IS NULL AND ref_name IS NULL AND ref_version IS NULL) OR "
            "(ref_type IS NOT NULL AND ref_name IS NOT NULL AND ref_version IS NOT NULL)",
            name=op.f("ck_mf_contract_parts_contract_reference_is_whole_or_absent"),
        ),
        sa.PrimaryKeyConstraint("id", name=op.f("pk_mf_contract_parts")),
    )
    with op.batch_alter_table("mf_contract_parts", schema=None) as batch_op:
        batch_op.create_index(
            "uq_mf_contract_parts_reference",
            ["ref_type", "ref_name", "ref_version"],
            unique=True,
            sqlite_where=sa.text(f"source = '{_SOURCE_CONTRACT}'"),
            postgresql_where=sa.text(f"source = '{_SOURCE_CONTRACT}'"),
        )
        batch_op.create_index("ix_mf_contract_parts_part_kind", ["part_kind"], unique=False)

    op.create_table(
        "mf_part_splits",
        sa.Column("id", sa.Integer(), autoincrement=True, nullable=False),
        sa.Column("part_id", sa.String(length=64), nullable=False),
        sa.Column("name", sa.String(length=128), nullable=False),
        sa.Column("path", sa.Text(), nullable=False),
        sa.Column("rows", sa.Integer(), nullable=False),
        sa.Column("bytes", sa.Integer(), nullable=False),
        sa.Column("sha256", sa.String(length=64), nullable=False),
        sa.Column("logical_digest", sa.String(length=64), nullable=False),
        # Required booleans, so F2's and F4's refusals have something to read rather than a
        # field they have to default (FR-010.18). Both are required on the contract's
        # `Split`, which is why neither is nullable here.
        sa.Column("held_out", sa.Boolean(), nullable=False),
        sa.Column("evaluation_only", sa.Boolean(), nullable=False),
        sa.Column("label_counts", JSON_COLUMN, nullable=True),
        sa.ForeignKeyConstraint(
            ["part_id"],
            ["mf_contract_parts.id"],
            name=op.f("fk_mf_part_splits_part_id_mf_contract_parts"),
            ondelete="CASCADE",
        ),
        sa.PrimaryKeyConstraint("id", name=op.f("pk_mf_part_splits")),
    )
    with op.batch_alter_table("mf_part_splits", schema=None) as batch_op:
        batch_op.create_index("uq_mf_part_splits_part_id_name", ["part_id", "name"], unique=True)

    op.create_table(
        "mf_review_bindings",
        sa.Column("id", sa.String(length=64), nullable=False),
        # FK to mf_runs is deliberately NOT declared. `mf_runs` is the base schema's and a
        # review binding outlives nothing of its own - but more importantly, F10's tables
        # are on their own metadata (see services/parts/tables.py), and a cross-metadata FK
        # would be emitted here while being unresolvable in any `create_all`. The
        # uniqueness that matters - one binding per run - is enforced below.
        sa.Column("run_id", sa.String(length=64), nullable=False),
        sa.Column("queue_id", sa.String(length=64), nullable=False),
        sa.Column("question", sa.Text(), nullable=False),
        sa.Column("label_set", JSON_COLUMN, nullable=False),
        sa.Column("posted", sa.Integer(), server_default="0", nullable=False),
        sa.Column("decided", sa.Integer(), server_default="0", nullable=False),
        sa.Column("pending", sa.Integer(), server_default="0", nullable=False),
        sa.Column("read_back_cursor", sa.DateTime(timezone=True), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False),
        sa.CheckConstraint(
            "posted >= 0 AND decided >= 0 AND pending >= 0",
            name=op.f("ck_mf_review_bindings_counts_non_negative"),
        ),
        sa.PrimaryKeyConstraint("id", name=op.f("pk_mf_review_bindings")),
        sa.UniqueConstraint("run_id", name=op.f("uq_mf_review_bindings_run_id")),
    )

    op.create_table(
        "mf_review_candidate_refs",
        sa.Column("id", sa.Integer(), autoincrement=True, nullable=False),
        sa.Column("binding_id", sa.String(length=64), nullable=False),
        sa.Column("step", sa.Integer(), nullable=False),
        sa.Column("external_id", sa.String(length=256), nullable=False),
        sa.Column("item_id", sa.String(length=64), nullable=True),
        sa.Column("decision_id", sa.String(length=64), nullable=True),
        sa.Column("decision", sa.String(length=16), server_default="none", nullable=False),
        sa.Column("override_label", sa.Text(), nullable=True),
        sa.Column("reason", sa.Text(), nullable=True),
        sa.Column("decided_by", sa.String(length=128), nullable=True),
        sa.Column("decided_by_origin", sa.String(length=16), nullable=True),
        sa.Column("decided_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.CheckConstraint(
            "decision IN ('accept', 'override', 'flag', 'reject', 'none')",
            name=op.f("ck_mf_review_candidate_refs_decision_known"),
        ),
        sa.CheckConstraint(
            "override_label IS NULL OR decision = 'override'",
            name=op.f("ck_mf_review_candidate_refs_only_override_carries_a_label"),
        ),
        sa.ForeignKeyConstraint(
            ["binding_id"],
            ["mf_review_bindings.id"],
            name=op.f("fk_mf_review_candidate_refs_binding_id_mf_review_bindings"),
            ondelete="CASCADE",
        ),
        sa.PrimaryKeyConstraint("id", name=op.f("pk_mf_review_candidate_refs")),
    )
    with op.batch_alter_table("mf_review_candidate_refs", schema=None) as batch_op:
        batch_op.create_index(
            "uq_mf_review_candidate_refs_binding_id_external_id",
            ["binding_id", "external_id"],
            unique=True,
        )
        # "How many undecided" runs EVERY step; it is this feature's hot query.
        batch_op.create_index(
            "ix_mf_review_candidate_refs_binding_id_decision",
            ["binding_id", "decision"],
            unique=False,
        )
        # The watermark read.
        batch_op.create_index(
            "ix_mf_review_candidate_refs_binding_id_decided_at",
            ["binding_id", "decided_at"],
            unique=False,
        )


def downgrade() -> None:
    """Drop all four.

    Safe in both groups and for different reasons: Group A is rebuildable from the part
    directories, which a rollback does not touch - that is the point of the filesystem
    being authoritative - and Group B belongs to runs that would be re-submitted.
    """
    with op.batch_alter_table("mf_review_candidate_refs", schema=None) as batch_op:
        batch_op.drop_index("ix_mf_review_candidate_refs_binding_id_decided_at")
        batch_op.drop_index("ix_mf_review_candidate_refs_binding_id_decision")
        batch_op.drop_index("uq_mf_review_candidate_refs_binding_id_external_id")
    op.drop_table("mf_review_candidate_refs")

    op.drop_table("mf_review_bindings")

    with op.batch_alter_table("mf_part_splits", schema=None) as batch_op:
        batch_op.drop_index("uq_mf_part_splits_part_id_name")
    op.drop_table("mf_part_splits")

    with op.batch_alter_table("mf_contract_parts", schema=None) as batch_op:
        batch_op.drop_index("ix_mf_contract_parts_part_kind")
        batch_op.drop_index("uq_mf_contract_parts_reference")
    op.drop_table("mf_contract_parts")
