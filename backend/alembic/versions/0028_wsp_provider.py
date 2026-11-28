"""WSP provider: eleven tables and two new approval actions (feature 016; FTDD §4, FTID §4).

Revision ID: 0028_wsp_provider
Revises: 0027_run_hub_access
Create Date: 2026-11-16

**Numbering (operator decision 2026-11-16):** miForge 017's ``0027_run_hub_access`` pushed first,
so this revision is ``0028`` and descends from it. ``tests/integration/wsp/test_migration_chain.py``
pins the parent from the chain itself (one head, and the parent).

The tables are written out in full here (not reflected from the models), so this revision describes
the schema as it was on this date; ``test_db_migration.py`` compares it with ``wsp.models`` on both
engines. CHECK constraints are spelled literally, as ``0021`` spells its own.

- **At most one run in ``pushing`` or ``running``** (P49, P60): a unique partial index on a
  constant.
- **Append-only** (FR-016.6; the ``0024_setting_changes`` pattern): ``mf_wsp_candidates`` refuses
  UPDATE
  and ``mf_wsp_usage`` refuses UPDATE and DELETE, by a trigger on each engine (PostgreSQL: one
  plpgsql
  function and a row trigger per table; SQLite: ``RAISE(ABORT)`` triggers).
- **``mf_approvals``' action CHECK** widens to admit ``wsp_session`` and ``wsp_session_external``
  (FR-016.37), inside ``batch_alter_table`` so SQLite rebuilds the table (R1B-M15).

**The downgrade refuses while any approval uses a new action**, and names the count: narrowing the
CHECK
under such a row would fail half-way on PostgreSQL and silently drop it on a SQLite rebuild.
"""

from __future__ import annotations

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

revision: str = "0028_wsp_provider"
down_revision: str | None = "0027_run_hub_access"
branch_labels: str | None = None
depends_on: str | None = None

APPEND_ONLY_MESSAGE = "WSP append-only table: rows are never updated"
OLD_ACTIONS = "action IN ('gpu_start', 'hotupdate_attach', 'delete', 'config_write')"
NEW_ACTIONS = (
    "action IN ('gpu_start', 'hotupdate_attach', 'delete', 'config_write', "
    "'wsp_session', 'wsp_session_external')"
)
#: table -> the verbs its trigger refuses.
APPEND_ONLY = {"mf_wsp_candidates": ("UPDATE",), "mf_wsp_usage": ("UPDATE", "DELETE")}


def upgrade() -> None:
    op.create_table(
        "mf_wsp_bundles",
        sa.Column("bundle_sha256", sa.String(length=64), nullable=False),
        sa.Column("owner", sa.String(length=64), nullable=False),
        sa.Column("bundle_id", sa.String(length=64), nullable=False),
        sa.Column("bundle_kind", sa.String(length=16), nullable=False),
        sa.Column("artifact_kind", sa.String(length=16), nullable=False),
        sa.Column("artifact_id", sa.String(length=128), nullable=False),
        sa.Column("detector_digest", sa.String(length=64), nullable=True),
        sa.Column("model", sa.String(length=256), nullable=False),
        sa.Column("layer", sa.Integer(), nullable=False),
        sa.Column("sensitivity_value", sa.String(length=16), nullable=False),
        sa.Column(
            "manifest",
            sa.JSON().with_variant(postgresql.JSONB(astext_type=sa.Text()), "postgresql"),
            nullable=False,
        ),
        sa.Column("bytes", sa.BigInteger(), nullable=False),
        sa.Column("import_status", sa.String(length=16), nullable=False),
        sa.Column("received_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("received_by", sa.String(length=64), nullable=False),
        sa.Column(
            "import_failure",
            sa.JSON().with_variant(postgresql.JSONB(astext_type=sa.Text()), "postgresql"),
            nullable=True,
        ),
        sa.CheckConstraint(
            "bundle_kind IN ('linear_probe', 'sae')",
            name=op.f("ck_mf_wsp_bundles_bundle_kind_known"),
        ),
        sa.CheckConstraint(
            "import_status IN ('importing', 'imported', 'failed')",
            name=op.f("ck_mf_wsp_bundles_import_status_known"),
        ),
        sa.CheckConstraint(
            "sensitivity_value IN ('sensitive', 'not_sensitive', 'unknown')",
            name=op.f("ck_mf_wsp_bundles_sensitivity_known"),
        ),
        sa.PrimaryKeyConstraint("bundle_sha256", name=op.f("pk_mf_wsp_bundles")),
        sa.UniqueConstraint("owner", "bundle_id", name="uq_mf_wsp_bundles_owner_bundle_id"),
    )
    op.create_table(
        "mf_wsp_endpoints",
        sa.Column("id", sa.String(length=40), nullable=False),
        sa.Column("kind", sa.String(length=24), nullable=False),
        sa.Column("base_url", sa.String(length=512), nullable=False),
        sa.Column("model", sa.String(length=256), nullable=False),
        sa.Column("locality", sa.String(length=16), nullable=False),
        sa.Column("price_in_per_mtok", sa.Numeric(precision=12, scale=6), nullable=True),
        sa.Column("price_out_per_mtok", sa.Numeric(precision=12, scale=6), nullable=True),
        sa.Column("price_cache_write_per_mtok", sa.Numeric(precision=12, scale=6), nullable=True),
        sa.Column("price_cache_read_per_mtok", sa.Numeric(precision=12, scale=6), nullable=True),
        sa.Column("output_cap_field", sa.String(length=32), nullable=False),
        sa.Column("serving", sa.String(length=16), nullable=False),
        sa.Column("key_ciphertext", sa.LargeBinary(), nullable=True),
        sa.Column("key_fingerprint", sa.String(length=8), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("created_by", sa.String(length=64), nullable=False),
        sa.Column("removed_at", sa.DateTime(timezone=True), nullable=True),
        sa.CheckConstraint(
            "kind IN ('openai_compatible', 'anthropic')",
            name=op.f("ck_mf_wsp_endpoints_kind_known"),
        ),
        sa.CheckConstraint(
            "locality <> 'external' OR "
            "(price_in_per_mtok IS NOT NULL AND price_out_per_mtok IS NOT NULL)",
            name=op.f("ck_mf_wsp_endpoints_external_has_prices"),
        ),
        sa.CheckConstraint(
            "locality IN ('local', 'external')", name=op.f("ck_mf_wsp_endpoints_locality_known")
        ),
        sa.CheckConstraint(
            "output_cap_field IN ('max_tokens', 'max_completion_tokens')",
            name=op.f("ck_mf_wsp_endpoints_output_cap_field_known"),
        ),
        sa.CheckConstraint(
            "serving IN ('generic', 'miserver')", name=op.f("ck_mf_wsp_endpoints_serving_known")
        ),
        sa.PrimaryKeyConstraint("id", name=op.f("pk_mf_wsp_endpoints")),
    )
    op.create_table(
        "mf_wsp_heldout_uses",
        sa.Column("owner", sa.String(length=64), nullable=False),
        sa.Column("set_name", sa.String(length=64), nullable=False),
        sa.Column("row_ids_sha256", sa.String(length=64), nullable=False),
        sa.Column("heldout_uses", sa.Integer(), nullable=False),
        sa.Column("last_used_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("last_candidate_id", sa.String(length=17), nullable=True),
        sa.CheckConstraint(
            "heldout_uses >= 0", name=op.f("ck_mf_wsp_heldout_uses_heldout_uses_non_negative")
        ),
        sa.PrimaryKeyConstraint(
            "owner", "set_name", "row_ids_sha256", name=op.f("pk_mf_wsp_heldout_uses")
        ),
    )
    op.create_table(
        "mf_wsp_spend_days",
        sa.Column("utc_day", sa.Date(), nullable=False),
        sa.Column("external_settled_usd", sa.Numeric(precision=12, scale=6), nullable=False),
        sa.Column("external_open_usd", sa.Numeric(precision=12, scale=6), nullable=False),
        sa.PrimaryKeyConstraint("utc_day", name=op.f("pk_mf_wsp_spend_days")),
    )
    op.create_table(
        "mf_wsp_worker_boots",
        sa.Column("boot_id", sa.String(length=32), nullable=False),
        sa.Column("first_seen_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("run_id", sa.String(length=16), nullable=True),
        sa.PrimaryKeyConstraint("boot_id", name=op.f("pk_mf_wsp_worker_boots")),
    )
    op.create_table(
        "mf_wsp_runs",
        sa.Column("run_id", sa.String(length=16), nullable=False),
        sa.Column("operation", sa.String(length=16), nullable=False),
        sa.Column("style", sa.String(length=16), nullable=False),
        sa.Column("status", sa.String(length=16), nullable=False),
        sa.Column("stop_reason", sa.String(length=16), nullable=True),
        sa.Column("outcome", sa.String(length=16), nullable=True),
        sa.Column(
            "request",
            sa.JSON().with_variant(postgresql.JSONB(astext_type=sa.Text()), "postgresql"),
            nullable=False,
        ),
        sa.Column("request_sha256", sa.String(length=64), nullable=False),
        sa.Column("bundle_sha256", sa.String(length=64), nullable=False),
        sa.Column("endpoint_id", sa.String(length=40), nullable=True),
        sa.Column("endpoint_locality", sa.String(length=16), nullable=True),
        sa.Column("sensitivity_effective", sa.String(length=16), nullable=False),
        sa.Column("provider_id", sa.String(length=64), nullable=False),
        sa.Column("provider_version", sa.String(length=32), nullable=False),
        sa.Column("approval_id", sa.String(length=40), nullable=True),
        sa.Column("gateway_token_sha256", sa.String(length=64), nullable=True),
        sa.Column("token_expires_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("boot_id", sa.String(length=32), nullable=True),
        sa.Column("pushed_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("last_status_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("result_pulled_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("acked_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("endpoint_fingerprint", sa.String(length=64), nullable=True),
        sa.Column("caller", sa.String(length=64), nullable=False),
        sa.Column("caller_ref", sa.String(length=128), nullable=True),
        sa.Column("content_sha256", sa.String(length=64), nullable=True),
        sa.Column(
            "usage",
            sa.JSON().with_variant(postgresql.JSONB(astext_type=sa.Text()), "postgresql"),
            nullable=False,
        ),
        sa.Column("forward_passes_used", sa.Integer(), nullable=True),
        sa.Column("transcript_sha256", sa.String(length=64), nullable=True),
        sa.Column("result_sha256", sa.String(length=64), nullable=True),
        sa.Column("patch_sha256", sa.String(length=64), nullable=True),
        sa.Column("program_sha256", sa.String(length=64), nullable=True),
        sa.Column("replay_of", sa.String(length=16), nullable=True),
        sa.Column("replay_reproduced", sa.Boolean(), nullable=True),
        # The replay's own patch digest (it never becomes a candidate, so not patch_sha256).
        sa.Column("replay_patch_sha256", sa.String(length=64), nullable=True),
        sa.Column(
            "failure",
            sa.JSON(none_as_null=True).with_variant(
                postgresql.JSONB(none_as_null=True, astext_type=sa.Text()), "postgresql"
            ),
            nullable=True,
        ),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("started_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("finished_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("deleted_at", sa.DateTime(timezone=True), nullable=True),
        sa.CheckConstraint(
            "(outcome = 'candidate') = (patch_sha256 IS NOT NULL)",
            name=op.f("ck_mf_wsp_runs_candidate_iff_patch"),
        ),
        sa.CheckConstraint(
            "endpoint_locality IS NULL OR endpoint_locality IN ('local', 'external')",
            name=op.f("ck_mf_wsp_runs_endpoint_locality_known"),
        ),
        sa.CheckConstraint(
            "operation IN ('inspect', 'create', 'repair')",
            name=op.f("ck_mf_wsp_runs_operation_known"),
        ),
        sa.CheckConstraint(
            "outcome IS NULL OR outcome IN ('candidate', 'no_change', 'report', 'failed')",
            name=op.f("ck_mf_wsp_runs_outcome_known"),
        ),
        sa.CheckConstraint(
            "sensitivity_effective IN ('sensitive', 'not_sensitive')",
            name=op.f("ck_mf_wsp_runs_sensitivity_effective_known"),
        ),
        sa.CheckConstraint(
            "status <> 'completed' OR outcome IS NOT NULL",
            name=op.f("ck_mf_wsp_runs_completed_has_outcome"),
        ),
        sa.CheckConstraint(
            "status IN ('queued', 'pushing', 'running', 'collecting', 'completed', 'failed', "
            "'cancelled')",
            name=op.f("ck_mf_wsp_runs_status_known"),
        ),
        sa.CheckConstraint(
            "stop_reason IS NULL OR stop_reason IN ('submitted', 'answered', 'max_steps', "
            "'token_budget', 'cost_budget', 'time_budget', 'interrupted', 'error', "
            "'gateway_refused', 'worker_lost', 'cancelled')",
            name=op.f("ck_mf_wsp_runs_stop_reason_known"),
        ),
        sa.CheckConstraint(
            "style IN ('session', 'program', 'program_rerun', 'replay')",
            name=op.f("ck_mf_wsp_runs_style_known"),
        ),
        sa.CheckConstraint(
            "style NOT IN ('program_rerun', 'replay') OR "
            "(endpoint_id IS NULL AND gateway_token_sha256 IS NULL)",
            name=op.f("ck_mf_wsp_runs_rerun_replay_have_no_token"),
        ),
        sa.ForeignKeyConstraint(
            ["boot_id"],
            ["mf_wsp_worker_boots.boot_id"],
            name=op.f("fk_mf_wsp_runs_boot_id_mf_wsp_worker_boots"),
        ),
        sa.ForeignKeyConstraint(
            ["bundle_sha256"],
            ["mf_wsp_bundles.bundle_sha256"],
            name=op.f("fk_mf_wsp_runs_bundle_sha256_mf_wsp_bundles"),
        ),
        sa.ForeignKeyConstraint(
            ["endpoint_id"],
            ["mf_wsp_endpoints.id"],
            name=op.f("fk_mf_wsp_runs_endpoint_id_mf_wsp_endpoints"),
        ),
        sa.PrimaryKeyConstraint("run_id", name=op.f("pk_mf_wsp_runs")),
    )
    op.create_index(
        "ix_mf_wsp_runs_caller_status", "mf_wsp_runs", ["caller", "status"], unique=False
    )
    op.create_index(
        "ix_mf_wsp_runs_status_created_at", "mf_wsp_runs", ["status", "created_at"], unique=False
    )
    op.create_table(
        "mf_wsp_candidates",
        sa.Column("candidate_id", sa.String(length=17), nullable=False),
        sa.Column("run_id", sa.String(length=16), nullable=False),
        sa.Column("parent_kind", sa.String(length=16), nullable=False),
        sa.Column("parent_ref", sa.String(length=512), nullable=False),
        sa.Column("bundle_base_sha256", sa.String(length=64), nullable=False),
        sa.Column("patch_sha256", sa.String(length=64), nullable=False),
        sa.Column("changed_entries", sa.Integer(), nullable=False),
        sa.Column(
            "tensors",
            sa.JSON().with_variant(postgresql.JSONB(astext_type=sa.Text()), "postgresql"),
            nullable=False,
        ),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.CheckConstraint(
            "parent_kind IN ('bundle', 'bundle_zeroed', 'candidate', 'checkpoint')",
            name=op.f("ck_mf_wsp_candidates_parent_kind_known"),
        ),
        sa.CheckConstraint(
            "changed_entries > 0", name=op.f("ck_mf_wsp_candidates_changed_entries_positive")
        ),
        sa.ForeignKeyConstraint(
            ["run_id"], ["mf_wsp_runs.run_id"], name=op.f("fk_mf_wsp_candidates_run_id_mf_wsp_runs")
        ),
        sa.PrimaryKeyConstraint("candidate_id", name=op.f("pk_mf_wsp_candidates")),
        sa.UniqueConstraint("patch_sha256", name=op.f("uq_mf_wsp_candidates_patch_sha256")),
        sa.UniqueConstraint("run_id", name=op.f("uq_mf_wsp_candidates_run_id")),
    )
    op.create_table(
        "mf_wsp_programs",
        sa.Column("program_sha256", sa.String(length=64), nullable=False),
        sa.Column("first_run_id", sa.String(length=16), nullable=False),
        sa.Column("bytes", sa.Integer(), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.ForeignKeyConstraint(
            ["first_run_id"],
            ["mf_wsp_runs.run_id"],
            name=op.f("fk_mf_wsp_programs_first_run_id_mf_wsp_runs"),
        ),
        sa.PrimaryKeyConstraint("program_sha256", name=op.f("pk_mf_wsp_programs")),
    )
    op.create_table(
        "mf_wsp_reservations",
        sa.Column("id", sa.BigInteger().with_variant(sa.Integer(), "sqlite"), nullable=False),
        sa.Column("run_id", sa.String(length=16), nullable=False),
        sa.Column("endpoint_id", sa.String(length=40), nullable=False),
        sa.Column("external", sa.Boolean(), nullable=False),
        sa.Column("worst_case_usd", sa.Numeric(precision=12, scale=6), nullable=False),
        sa.Column("worst_case_tokens", sa.BigInteger(), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("settled_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("settled_cost_usd", sa.Numeric(precision=12, scale=6), nullable=True),
        sa.ForeignKeyConstraint(
            ["run_id"],
            ["mf_wsp_runs.run_id"],
            name=op.f("fk_mf_wsp_reservations_run_id_mf_wsp_runs"),
        ),
        sa.PrimaryKeyConstraint("id", name=op.f("pk_mf_wsp_reservations")),
    )
    op.create_index(
        "ix_mf_wsp_reservations_run_id", "mf_wsp_reservations", ["run_id"], unique=False
    )
    op.create_table(
        "mf_wsp_usage",
        sa.Column("id", sa.BigInteger().with_variant(sa.Integer(), "sqlite"), nullable=False),
        sa.Column("run_id", sa.String(length=16), nullable=False),
        sa.Column("seq", sa.Integer(), nullable=False),
        sa.Column("endpoint_id", sa.String(length=40), nullable=False),
        sa.Column("input_tokens", sa.Integer(), nullable=False),
        sa.Column("output_tokens", sa.Integer(), nullable=False),
        sa.Column("cost_usd", sa.Numeric(precision=12, scale=6), nullable=False),
        sa.Column("at", sa.DateTime(timezone=True), nullable=False),
        sa.ForeignKeyConstraint(
            ["run_id"], ["mf_wsp_runs.run_id"], name=op.f("fk_mf_wsp_usage_run_id_mf_wsp_runs")
        ),
        sa.PrimaryKeyConstraint("id", name=op.f("pk_mf_wsp_usage")),
        sa.UniqueConstraint("run_id", "seq", name="uq_mf_wsp_usage_run_id_seq"),
    )
    op.create_table(
        "mf_wsp_verdicts",
        sa.Column("verdict_id", sa.String(length=40), nullable=False),
        sa.Column("candidate_id", sa.String(length=17), nullable=False),
        sa.Column("request_sha256", sa.String(length=64), nullable=False),
        sa.Column("value", sa.String(length=16), nullable=False),
        sa.Column(
            "document",
            sa.JSON().with_variant(postgresql.JSONB(astext_type=sa.Text()), "postgresql"),
            nullable=False,
        ),
        sa.Column("snapshot", sa.LargeBinary(), nullable=False),
        sa.Column("run_sha256", sa.String(length=64), nullable=False),
        sa.Column("scorer_owner", sa.String(length=64), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.CheckConstraint(
            "value IN ('improved', 'not_significant', 'regressed', 'no_change', 'unmeasured')",
            name=op.f("ck_mf_wsp_verdicts_value_known"),
        ),
        sa.ForeignKeyConstraint(
            ["candidate_id"],
            ["mf_wsp_candidates.candidate_id"],
            name=op.f("fk_mf_wsp_verdicts_candidate_id_mf_wsp_candidates"),
        ),
        sa.PrimaryKeyConstraint("verdict_id", name=op.f("pk_mf_wsp_verdicts")),
        sa.UniqueConstraint(
            "candidate_id", "request_sha256", name="uq_mf_wsp_verdicts_candidate_request"
        ),
    )

    op.create_index(
        "uq_mf_wsp_runs_one_active",
        "mf_wsp_runs",
        [sa.text("(1)")],
        unique=True,
        postgresql_where=sa.text("status IN ('pushing', 'running')"),
        sqlite_where=sa.text("status IN ('pushing', 'running')"),
    )
    _create_append_only_triggers()
    _set_actions(
        from_name=f"{DOUBLE}action_known",
        from_actions=OLD_ACTIONS,
        to_name=f"{SINGLE}action_known",
        to_actions=NEW_ACTIONS,
    )


def _create_append_only_triggers() -> None:
    postgres = op.get_bind().dialect.name == "postgresql"
    if postgres:
        op.execute(
            "CREATE FUNCTION mf_wsp_refuse_update() RETURNS trigger AS $$ "
            f"BEGIN RAISE EXCEPTION '{APPEND_ONLY_MESSAGE}'; END; $$ LANGUAGE plpgsql"
        )
    for table, verbs in APPEND_ONLY.items():
        if postgres:
            op.execute(
                f"CREATE TRIGGER {table}_no_update BEFORE {' OR '.join(verbs)} "
                f"ON {table} FOR EACH ROW EXECUTE FUNCTION mf_wsp_refuse_update()"
            )
        else:
            for verb in verbs:
                op.execute(
                    f"CREATE TRIGGER {table}_no_{verb.lower()} BEFORE {verb} ON {table} "
                    f"BEGIN SELECT RAISE(ABORT, '{APPEND_ONLY_MESSAGE}'); END"
                )


def downgrade() -> None:
    bind = op.get_bind()
    in_use = bind.execute(
        sa.text(
            "SELECT COUNT(*) FROM mf_approvals "
            "WHERE action IN ('wsp_session', 'wsp_session_external')"
        )
    ).scalar_one()
    if in_use:
        raise RuntimeError(
            f"Refusing to downgrade 0028_wsp_provider: {in_use} approval row(s) use a WSP action "
            "(wsp_session, wsp_session_external). Narrowing the action CHECK would fail on "
            "PostgreSQL and drop them on a SQLite rebuild; decide or delete them first."
        )
    _set_actions(
        from_name=f"{SINGLE}action_known",
        from_actions=NEW_ACTIONS,
        to_name=f"{DOUBLE}action_known",
        to_actions=OLD_ACTIONS,
    )
    op.drop_index("uq_mf_wsp_runs_one_active", table_name="mf_wsp_runs")
    for table in (
        "mf_wsp_verdicts",
        "mf_wsp_usage",
        "mf_wsp_reservations",
        "mf_wsp_programs",
        "mf_wsp_candidates",
    ):
        op.drop_table(table)
    op.drop_index("ix_mf_wsp_runs_status_created_at", table_name="mf_wsp_runs")
    op.drop_index("ix_mf_wsp_runs_caller_status", table_name="mf_wsp_runs")
    for table in (
        "mf_wsp_runs",
        "mf_wsp_worker_boots",
        "mf_wsp_spend_days",
        "mf_wsp_heldout_uses",
        "mf_wsp_endpoints",
        "mf_wsp_bundles",
    ):
        op.drop_table(table)
    # Dropping a table drops its triggers on both engines; the function is PostgreSQL's own.
    if bind.dialect.name == "postgresql":
        op.execute("DROP FUNCTION IF EXISTS mf_wsp_refuse_update()")


_JSON = sa.JSON().with_variant(postgresql.JSONB(), "postgresql")


#: 0021 passed already-prefixed names to ``sa.CheckConstraint`` under the naming convention, so
#: the convention prefixed them again: every database at 0026 holds
#: ``ck_mf_approvals_ck_mf_approvals_*`` (verified on PostgreSQL), while the model names
#: ``ck_mf_approvals_action_known``. This revision
#: replaces the action CHECK under the model's name and leaves the other three as they are (a
#: pre-existing naming defect with no behavioural effect, recorded as debt).
DOUBLE = "ck_mf_approvals_ck_mf_approvals_"
SINGLE = "ck_mf_approvals_"


def _mf_approvals(action_check: str, actions: str) -> sa.Table:
    """``mf_approvals`` as it stands (0021 plus 0025's ``batch_id``), for ``copy_from``.

    **Not optional** (``0014``'s lesson): SQLAlchemy does not reflect SQLite CHECK constraints, so a
    ``batch_alter_table`` left to reflect rebuilds the table without its other three CHECKs and its
    index, while reporting success. ``test_migration_0028.py`` pins that the rebuild keeps them.
    """
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
        sa.CheckConstraint(actions, name=action_check),
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


def _set_actions(*, from_name: str, from_actions: str, to_name: str, to_actions: str) -> None:
    if op.get_bind().dialect.name == "postgresql":
        # PostgreSQL alters a CHECK in place; no rebuild, nothing else touched.
        op.drop_constraint(op.f(from_name), "mf_approvals", type_="check")
        op.create_check_constraint(op.f(to_name), "mf_approvals", to_actions)
        return
    with op.batch_alter_table(
        "mf_approvals", copy_from=_mf_approvals(from_name, from_actions)
    ) as batch:
        batch.drop_constraint(op.f(from_name), type_="check")
        batch.create_check_constraint(op.f(to_name), to_actions)
