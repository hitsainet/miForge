"""mf_scores and mf_score_items - F4's score index.

Revision ID: 0013_scoring
Revises: 0012_save_req
Create Date: 2026-10-14

Generated from ``services/scoring/tables.py``'s own ``MetaData`` rather than hand-written,
so the DDL cannot drift from the model F4's code actually reads.

**Two constraints here are load-bearing rather than hygienic**, and deleting either breaks
a derivation elsewhere:

* ``ck_mf_scores_retention_has_a_floor`` - the run log's ``EvaluationBlock`` carries **no
  ``kind``** on a score entry, so ``reindex`` must *infer* test-vs-retention, and the rule
  is "a floor means retention". That inference is only sound because this constraint makes
  a floorless retention row impossible. The derivation is written once, in
  ``kind_of_entry()``; two derivations would be two answers and the symptom would be a
  rebuild that quietly disagrees with the live write.
* ``ck_mf_scores_breach_needs_a_floor`` - a breach without a floor is a claim with no
  threshold behind it.

**The unique index is on ``(run_id, step, kind, created_at)``, not ``(run_id, step, kind)``,
so a re-score appends rather than overwriting.** Scoring the same step twice is a real
operation - a provider changed, a set was corrected - and the second answer does not erase
the first; ``rescore_reason`` says why it happened and ``provider_discontinuity`` flags the
case where the two are not comparable.

**No cascade from ``mf_checkpoints``**: a score's truth is the run log on disk, so deleting
an index row must not take the score with it. The cascade that *does* exist is
``score_id -> mf_scores`` on the items, because an item has no meaning without its pass.

*Found by F4 and worth recording: the first draft of this DDL rendered ``BOOLEAN DEFAULT 0``
and ``CHECK (breached = 0)`` - valid SQLite, **rejected by PostgreSQL**. The suite runs on
SQLite by default, so every test passed and the migration would have failed on a clean
clone. Caught by compiling the DDL for the dialect rather than reading it, which is why
``sa.false()``/``sa.true()`` and ``NOT breached`` appear below.*

Chained off ``0012_save_req``, the then-head. The Alembic graph is serialised through the
coordinator while feature agents run concurrently.
"""

from __future__ import annotations

import sqlalchemy as sa
from alembic import op

revision: str = "0013_scoring"
down_revision: str | None = "0012_save_req"
branch_labels: str | None = None
depends_on: str | None = None


def upgrade() -> None:
    op.create_table(
        "mf_scores",
        sa.Column("id", sa.Integer(), autoincrement=True, nullable=False),
        sa.Column("run_id", sa.String(length=64), nullable=False),
        sa.Column("step", sa.Integer(), nullable=False),
        sa.Column("kind", sa.String(length=16), nullable=False),
        sa.Column("set_ref", sa.JSON(), nullable=True),
        sa.Column("set_rendered", sa.Text(), nullable=False),
        sa.Column("part_id", sa.String(length=64), nullable=True),
        sa.Column("score", sa.Float(), nullable=False),
        sa.Column("n", sa.Integer(), nullable=False),
        sa.Column("floor", sa.Float(), nullable=True),
        sa.Column("floor_layer", sa.String(length=16), server_default="none", nullable=False),
        sa.Column("breached", sa.Boolean(), server_default=sa.false(), nullable=False),
        sa.Column("provider", sa.Text(), nullable=False),
        sa.Column("provider_kind", sa.String(length=32), nullable=False),
        sa.Column("provider_model", sa.Text(), nullable=False),
        sa.Column("provider_base_url", sa.Text(), nullable=True),
        sa.Column("parameter_set_digest", sa.Text(), nullable=True),
        sa.Column("aggregation", sa.String(length=32), nullable=False),
        sa.Column("decode", sa.JSON(), nullable=False),
        sa.Column("subsample_seed", sa.Integer(), nullable=True),
        sa.Column("split_markers", sa.JSON(), nullable=True),
        sa.Column("selector_hash", sa.String(length=64), nullable=True),
        sa.Column("grader_hash", sa.String(length=64), nullable=True),
        sa.Column("reproducible", sa.Boolean(), server_default=sa.true(), nullable=False),
        sa.Column("not_reproducible_reason", sa.Text(), nullable=True),
        sa.Column("out_of_cadence", sa.Boolean(), server_default=sa.false(), nullable=False),
        sa.Column(
            "provider_discontinuity", sa.Boolean(), server_default=sa.false(), nullable=False
        ),
        sa.Column("rescore_reason", sa.Text(), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.PrimaryKeyConstraint("id", name=op.f("pk_mf_scores")),
        sa.CheckConstraint("kind IN ('test', 'retention')", name=op.f("ck_mf_scores_kind_known")),
        sa.CheckConstraint("step >= 0", name=op.f("ck_mf_scores_step_non_negative")),
        sa.CheckConstraint(
            "score >= 0.0 AND score <= 1.0", name=op.f("ck_mf_scores_score_in_unit_range")
        ),
        sa.CheckConstraint("n >= 1", name=op.f("ck_mf_scores_n_at_least_one")),
        sa.CheckConstraint(
            "floor IS NULL OR (floor >= 0.0 AND floor <= 1.0)",
            name=op.f("ck_mf_scores_floor_in_range"),
        ),
        sa.CheckConstraint(
            "kind <> 'retention' OR floor IS NOT NULL",
            name=op.f("ck_mf_scores_retention_has_a_floor"),
        ),
        sa.CheckConstraint(
            "NOT breached OR floor IS NOT NULL", name=op.f("ck_mf_scores_breach_needs_a_floor")
        ),
        sa.CheckConstraint(
            "reproducible OR not_reproducible_reason IS NOT NULL",
            name=op.f("ck_mf_scores_not_reproducible_names_its_reason"),
        ),
        sa.CheckConstraint(
            "floor_layer IN ('pack_slot', 'run_override', 'none')",
            name=op.f("ck_mf_scores_floor_layer_known"),
        ),
    )
    op.create_index(
        "uq_mf_scores_run_step_kind_at",
        "mf_scores",
        ["run_id", "step", "kind", "created_at"],
        unique=True,
    )
    op.create_index("ix_mf_scores_run_id_kind_step", "mf_scores", ["run_id", "kind", "step"])

    op.create_table(
        "mf_score_items",
        sa.Column("id", sa.Integer(), autoincrement=True, nullable=False),
        sa.Column("score_id", sa.Integer(), nullable=False),
        sa.Column("prompt_index", sa.Integer(), nullable=False),
        sa.Column("prompt_id", sa.Text(), nullable=True),
        sa.Column("verdict", sa.String(length=16), nullable=True),
        sa.Column("scalar", sa.Float(), nullable=True),
        sa.Column("reason_code", sa.Text(), nullable=True),
        sa.Column("completion_excerpt", sa.Text(), nullable=True),
        sa.PrimaryKeyConstraint("id", name=op.f("pk_mf_score_items")),
        sa.ForeignKeyConstraint(
            ["score_id"],
            ["mf_scores.id"],
            name=op.f("fk_mf_score_items_score_id_mf_scores"),
            ondelete="CASCADE",
        ),
    )
    op.create_index("ix_mf_score_items_score_id", "mf_score_items", ["score_id"])
    op.create_index(
        "uq_mf_score_items_score_id_prompt_index",
        "mf_score_items",
        ["score_id", "prompt_index"],
        unique=True,
    )


def downgrade() -> None:
    op.drop_index("uq_mf_score_items_score_id_prompt_index", table_name="mf_score_items")
    op.drop_index("ix_mf_score_items_score_id", table_name="mf_score_items")
    op.drop_table("mf_score_items")
    op.drop_index("ix_mf_scores_run_id_kind_step", table_name="mf_scores")
    op.drop_index("uq_mf_scores_run_step_kind_at", table_name="mf_scores")
    op.drop_table("mf_scores")
