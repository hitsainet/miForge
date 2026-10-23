"""F4's two tables, both **indexes**  (FR-004.32; ADR-003, ADR-004).

## Core ``Table`` on its own ``MetaData``, not ``db.base.Base``

Foundation task 4.2's ratchet asserts ``set(Base.metadata.tables)`` **equals** the seven
base-schema tables, with the message *"a table only one feature reads belongs in that
feature's own migration."* So a feature's tables are deliberately not on ``Base.metadata``;
F10's ``services/parts/tables.py`` is the worked example and this follows it exactly,
carrying the same :data:`db.base.NAMING_CONVENTION` - unnamed CHECK and UNIQUE constraints
cannot be dropped by name, which makes a migration reversible on PostgreSQL and irreversible
on the SQLite the unit tests run against - and the same ``mf_`` prefix.

**The Alembic revision is NOT created here.** The revision graph is shared state and three
features were being built concurrently; two revisions chaining off one parent make
``alembic upgrade head`` fail for everyone, on someone else's clean clone. The DDL is handed
to the coordinator, which owns the chain.

## Neither table is a store of record  (ADR-004)

The authoritative copy of a score is the **evaluation block in the on-disk run log**, exactly
as F1 settled for the step record: *"the authoritative copy is the run log on disk, and
``mf_steps`` is the index over it."* :mod:`services.scoring.reindex_scores` rebuilds both
tables from (a) saved steps' ``manifest.json``, for the five ``ScoreRecord`` fields, and
(b) the run log, for everything else and for **every** step. A score at a non-save step
exists only in the log, which is therefore part of the durability story and not telemetry -
stated here so nobody prunes it for disk space.

## Boolean defaults and predicates are written engine-portably, and that was a real bug

``server_default=sa.text("0")`` renders as the literal ``0``, which PostgreSQL **rejects**
for a boolean column - and ``CHECK (breached = 0)`` likewise. Both passed every test, because
the suite runs on SQLite by default, where ``0`` is a boolean. ``support/db.py`` names this
blind spot in terms: *"a PostgreSQL-only type or a JSONB operator would pass here and fail
there."* Caught by **compiling the DDL against the PostgreSQL dialect** rather than by reading
it, which is also what the coordinator needs handed over - so
``test_scoring_static_guards.py`` now compiles both tables for both dialects.

## A re-score appends; nothing updates a score row

The unique constraint is on ``(run_id, step, kind, created_at)`` rather than
``(run_id, step, kind)``, so re-scoring one step inserts a new row with its own reason. **A
mutated history is worse than a duplicated one**: the duplicate is visible and the mutation
is not, and the question an operator asks of a score series is *"did this number change, and
when"*. ``backend/tests/unit/test_scoring_tables.py`` asserts no ``UPDATE`` against either
table exists anywhere in ``backend/src``.
"""

from __future__ import annotations

import sqlalchemy as sa

from db.base import NAMING_CONVENTION, JsonColumn

#: F4's own metadata. See the module docstring for why this is not ``Base.metadata``.
metadata = sa.MetaData(naming_convention=NAMING_CONVENTION)

#: The two evaluation kinds. ``kind`` is F4's own axis and is **not** a field of either
#: vendored schema: ``ScoreRecord`` is reached through the manifest's ``test`` and
#: ``retention`` properties, and the run log's ``ScoreEntry`` distinguishes them by whether
#: ``floor`` is set. See :func:`services.scoring.evaluation_block.kind_of_entry`.
KIND_TEST = "test"
KIND_RETENTION = "retention"
KINDS: tuple[str, ...] = (KIND_TEST, KIND_RETENTION)

#: Where a resolved retention floor came from. Recorded because **FPRD OQ-2 / DEF-7 is
#: unsettled** - R-02.38 puts the floor on the pack's retention slot and the UI mockup shows
#: it editable per run - so the record says which layer was actually in force whichever way
#: the operator settles it. ``none`` is for a test set, which has no floor.
FLOOR_LAYER_PACK = "pack_slot"
FLOOR_LAYER_RUN = "run_override"
FLOOR_LAYER_NONE = "none"
FLOOR_LAYERS: tuple[str, ...] = (FLOOR_LAYER_PACK, FLOOR_LAYER_RUN, FLOOR_LAYER_NONE)


SCORES = sa.Table(
    "mf_scores",
    metadata,
    sa.Column("id", sa.Integer, primary_key=True, autoincrement=True),
    # FK to mf_runs by name rather than by object: that table is on `Base.metadata` and this
    # one is not, so the constraint is emitted by the migration and never resolved by a
    # `create_all` over this metadata alone. Same shape as F10's REVIEW_BINDINGS.
    sa.Column("run_id", sa.String(64), nullable=False),
    # THE CANONICAL STEP, as one integer (BRD-01 R-01.24). `step = 0` is a real row and not
    # a sentinel: criterion 12 names the step-0 test score as a creation run's baseline, and
    # without it a run's first data point is already post-update with nothing to compare to.
    sa.Column("step", sa.Integer, nullable=False),
    sa.Column("kind", sa.String(16), nullable=False),
    # The set reference, stored BOTH ways on purpose. The structured form is the identity -
    # a contract reference `{type, name, version}` or an external provenance - and the
    # rendered form is the display string, which is what `ScoreRecord.set` carries. One
    # renderer produces the second from the first (`render_reference`), because two
    # renderings of one identity is how the pack/contract rendering confusion started.
    sa.Column("set_ref", JsonColumn, nullable=True),
    sa.Column("set_rendered", sa.Text, nullable=False),
    sa.Column("part_id", sa.String(64), nullable=True),
    # The number, and the sample it was taken over. ADR-012: every number names its scale
    # and sample, which is why `n` is NOT NULL beside it.
    sa.Column("score", sa.Float, nullable=False),
    sa.Column("n", sa.Integer, nullable=False),
    sa.Column("floor", sa.Float, nullable=True),
    sa.Column("floor_layer", sa.String(16), nullable=False, server_default=FLOOR_LAYER_NONE),
    # A FACT ON THE PASS, not something derived later by comparing two numbers. ADR-004 has
    # `reindex` rebuild from the log alone, so the decision the loop took has to be
    # reconstructible from what was written at the time - and `score < floor` recomputed
    # later against a floor that has since been edited gives a different answer.
    sa.Column("breached", sa.Boolean, nullable=False, server_default=sa.false()),
    # The provider, as columns because every series query filters or groups by them. The
    # rendered string is what `ScoreRecord.provider` carries; the parts are what a reader
    # needs to tell "the in-process provider" from "a model name".
    sa.Column("provider", sa.Text, nullable=False),
    sa.Column("provider_kind", sa.String(32), nullable=False),
    sa.Column("provider_model", sa.Text, nullable=False),
    sa.Column("provider_base_url", sa.Text, nullable=True),
    sa.Column("parameter_set_digest", sa.Text, nullable=True),
    # The three values `ScoreRecord` still does NOT carry (DEF-2, which BRD-02 section 9
    # records as "still owed and deliberately not bundled"). An accept rate and a normalised
    # grader mean are both "0.84" and are not the same measurement, so the method rides with
    # the number here and in the run log.
    sa.Column("aggregation", sa.String(32), nullable=False),
    sa.Column("decode", JsonColumn, nullable=False),
    sa.Column("subsample_seed", sa.Integer, nullable=True),
    # miDataworks' split markers, carried not understood: the names that are TRUE, as an
    # array, which is the shape the run log's `EvaluationBlock.split` uses. The set of names
    # is derived from their `Split`'s boolean properties rather than retyped
    # (`services.scoring.sets.split_marker_names`), so a marker they add is carried the day
    # the schema is re-vendored instead of being silently dropped - a dropped provenance flag
    # is one of this project's three recorded defects of that class.
    # NULL means "this part declares no splits", which an external part legitimately does;
    # `[]` means the splits were read and none is marked. Two different facts.
    sa.Column("split_markers", JsonColumn, nullable=True),
    # F3's two published hashes (its FR-003.16), so a score names the configuration that
    # judged it. Nullable: a reference-match test set has a selector and no grader.
    sa.Column("selector_hash", sa.String(64), nullable=True),
    sa.Column("grader_hash", sa.String(64), nullable=True),
    # Reproducibility as a RECORDED MARKING WITH A REASON, never inferred by a reader
    # (FR-004.18). A sampled pass and a pass whose provider did not declare a parameter the
    # pass required are both irreproducible, for different reasons, and the reason is the
    # half an operator can act on.
    sa.Column("reproducible", sa.Boolean, nullable=False, server_default=sa.true()),
    sa.Column("not_reproducible_reason", sa.Text, nullable=True),
    # An ad-hoc check must never silently join the step series.
    sa.Column("out_of_cadence", sa.Boolean, nullable=False, server_default=sa.false()),
    # Set when the run's provider was re-pinned at this step. F9 BREAKS the curve here
    # rather than joining two instruments' numbers (FR-004.10).
    sa.Column("provider_discontinuity", sa.Boolean, nullable=False, server_default=sa.false()),
    sa.Column("rescore_reason", sa.Text, nullable=True),
    sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
    sa.CheckConstraint(f"kind IN ('{KIND_TEST}', '{KIND_RETENTION}')", name="kind_known"),
    sa.CheckConstraint("step >= 0", name="step_non_negative"),
    # `ScoreRecord.score` is bounded 0..1 in the vendored schema and `n` has minimum 1.
    # Mirrored as constraints because a row the manifest could not express is a row
    # `reindex` cannot round-trip - and because clamping to fit is forbidden (E-8), so the
    # database must refuse what the aggregator already refuses.
    sa.CheckConstraint("score >= 0.0 AND score <= 1.0", name="score_in_unit_range"),
    sa.CheckConstraint("n >= 1", name="n_at_least_one"),
    sa.CheckConstraint("floor IS NULL OR (floor >= 0.0 AND floor <= 1.0)", name="floor_in_range"),
    # A retention score with no floor cannot exist: FR-004.27 refuses a missing floor rather
    # than reading it as "nothing to breach", and a fabricated 0.0 would pass every run.
    sa.CheckConstraint(
        f"kind <> '{KIND_RETENTION}' OR floor IS NOT NULL", name="retention_has_a_floor"
    ),
    # A breach is only expressible where a floor is. Without this a `breached` row with a
    # null floor would be a claim nothing could explain.
    sa.CheckConstraint("NOT breached OR floor IS NOT NULL", name="breach_needs_a_floor"),
    sa.CheckConstraint(
        "reproducible OR not_reproducible_reason IS NOT NULL",
        name="not_reproducible_names_its_reason",
    ),
    sa.CheckConstraint(
        f"floor_layer IN ({', '.join(repr(layer) for layer in FLOOR_LAYERS)})",
        name="floor_layer_known",
    ),
    # A re-score APPENDS. See the module docstring.
    sa.Index("uq_mf_scores_run_step_kind_at", "run_id", "step", "kind", "created_at", unique=True),
    # The series query, which is F9's curve.
    sa.Index("ix_mf_scores_run_id_kind_step", "run_id", "kind", "step"),
)


SCORE_ITEMS = sa.Table(
    "mf_score_items",
    metadata,
    sa.Column("id", sa.Integer, primary_key=True, autoincrement=True),
    sa.Column(
        "score_id",
        sa.Integer,
        sa.ForeignKey("mf_scores.id", ondelete="CASCADE"),
        nullable=False,
    ),
    sa.Column("prompt_index", sa.Integer, nullable=False),
    sa.Column("prompt_id", sa.Text, nullable=True),
    # The verdict value as F3's contract spells it, or null for a pure grader pass. Never a
    # fourth vocabulary: `schemas.selector.VerdictValue` is the one enum.
    sa.Column("verdict", sa.String(16), nullable=True),
    sa.Column("scalar", sa.Float, nullable=True),
    sa.Column("reason_code", sa.Text, nullable=True),
    # Truncated completion text for a drill-down. Run-log and index only: it NEVER enters a
    # manifest, which travels with the artifact and is published beside a checkpoint
    # (ADR-020).
    sa.Column("completion_excerpt", sa.Text, nullable=True),
    sa.Index("ix_mf_score_items_score_id", "score_id"),
    sa.Index("uq_mf_score_items_score_id_prompt_index", "score_id", "prompt_index", unique=True),
)
