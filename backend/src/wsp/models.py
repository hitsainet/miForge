"""WSP's eleven tables (FTDD §4; FTID §4), on WSP's own metadata.

**Not on ``Base.metadata``**: the base schema holds only tables more than one feature reads
(``test_models_base_schema.py``), and only WSP reads these. ``FEATURE_METADATA`` in
``test_db_migration.py`` names this metadata, so the migration ``0028_wsp_provider`` and these
declarations are compared on both engines.

**Append-only tables.** ``mf_wsp_candidates`` refuses UPDATE (FR-016.6: a candidate is an immutable
artifact; "rollback" is selecting the parent) and ``mf_wsp_usage`` refuses UPDATE and DELETE (the
metered
record of spend; the ``mf_setting_changes`` pattern). Both are enforced by **database triggers on
both
engines**, written in the migration as ``0024_setting_changes`` writes its own, so raw SQL is
refused as
well as the ORM (OVN-wsp-forge-9).

**One worker, one session (P49, P60):** a unique partial index on a constant admits at most one run
in
``pushing`` or ``running``.
"""

from __future__ import annotations

import datetime as dt
from decimal import Decimal
from typing import Any

from sqlalchemy import (
    BigInteger,
    Boolean,
    CheckConstraint,
    Date,
    DateTime,
    ForeignKey,
    Index,
    Integer,
    LargeBinary,
    MetaData,
    Numeric,
    String,
    UniqueConstraint,
    text,
)
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column

from db.base import NAMING_CONVENTION, JsonColumn
from models.approval import NullableJson

metadata = MetaData(naming_convention=NAMING_CONVENTION)


class WSPBase(DeclarativeBase):
    """Declarative base for WSP's tables, on WSP's own metadata."""

    metadata = metadata


def _in(column: str, values: tuple[str, ...]) -> str:
    return f"{column} IN ({', '.join(repr(v) for v in values)})"


BUNDLE_KINDS = ("linear_probe", "sae")
IMPORT_STATUSES = ("importing", "imported", "failed")
SENSITIVITY_VALUES = ("sensitive", "not_sensitive", "unknown")
ENDPOINT_KINDS = ("openai_compatible", "anthropic")
LOCALITIES = ("local", "external")
OUTPUT_CAP_FIELDS = ("max_tokens", "max_completion_tokens")
SERVINGS = ("generic", "miserver")
#: How the editor gives its one action per turn (feature 018, FR-018.8): OpenAI ``tools`` /
#: Anthropic ``tool_use``, or a ``response_format`` JSON schema (miServer, OpenAI kind, only).
ACTION_PROTOCOLS = ("tool_calls", "structured")
DEFAULT_ACTION_PROTOCOL = "tool_calls"
#: Whether ``miwsp_engine`` asks the editor to think (2026-11-21, migration 0032): ``off`` (the
#: default: on a miServer OpenAI-kind endpoint the engine sends ``enable_thinking: false``),
#: ``on`` (``enable_thinking: true`` and a per-turn reasoning allowance; a miServer OpenAI-kind
#: endpoint only) or ``default`` (the engine sends nothing and the server's template decides).
THINKING_VALUES = ("off", "on", "default")
DEFAULT_THINKING = "off"
#: Whether each editor request names the per-turn output cap (operator, 2026-11-21): ``on`` (the
#: default, ``miwsp_engine.context.editor_output_cap``) or ``off`` (no cap named, so the gateway's
#: window-aware default applies). Both providers read it.
OUTPUT_CAP_VALUES = ("on", "off")
DEFAULT_OUTPUT_CAP = "on"
#: Whether ``miwsp_engine`` pages bulky tool results out of the history and keeps its experiment
#: ledger (operator, 2026-11-21, migration 0034): ``on`` (the default) or ``off``. The engine's
#: only; a ``weightspace`` run is refused a ``paging`` of its own and freezes none.
PAGING_VALUES = ("on", "off")
DEFAULT_PAGING = "on"
OPERATIONS = ("inspect", "create", "repair")
STYLES = ("session", "program", "program_rerun", "replay")
RUN_STATUSES = ("queued", "pushing", "running", "collecting", "completed", "failed", "cancelled")
TERMINAL_STATUSES = ("completed", "failed", "cancelled")
STOP_REASONS = (
    "submitted",
    "answered",
    "max_steps",
    "token_budget",
    "cost_budget",
    "time_budget",
    "interrupted",
    "error",
    "gateway_refused",
    "worker_lost",
    "cancelled",
)
OUTCOMES = ("candidate", "no_change", "report", "failed")
#: The worker's own states (``wsp_worker/supervisor.py``), as the driver stores the last one it
#: read (live progress, 2026-11-20). The worker is untrusted: any other value is stored
#: ``unknown``, never as sent.
WORKER_STATES = ("idle", "receiving", "running", "finished", "exiting", "unknown")
#: What became of one editor call the gateway saw (``mf_wsp_usage.outcome``, live progress
#: 2026-11-20). ``served``: the upstream answered under 400 and the answer went back;
#: ``upstream_error``: it answered 400 or over and that answer went back; ``upstream_failed``: it
#: did not answer (settled at worst case); ``session_ended``: metered, then the gateway ended the
#: session (miServer's session errors, P47/PF-49); ``refused``: the gateway refused it before or at
#: the reservation, nothing was spent; ``closed_at_worst``: open when the run ended, closed at
#: worst case. A row from before migration 0030 has none (null).
CALL_OUTCOMES = (
    "served",
    "upstream_error",
    "upstream_failed",
    "session_ended",
    "refused",
    "closed_at_worst",
)
PARENT_KINDS = ("bundle", "bundle_zeroed", "candidate", "checkpoint")
VERDICT_VALUES = ("improved", "not_significant", "regressed", "no_change", "unmeasured")
EFFECTIVE_SENSITIVITY = ("sensitive", "not_sensitive")

APPEND_ONLY_MESSAGE = "WSP append-only table: rows are never updated"


class WspBundle(WSPBase):
    __tablename__ = "mf_wsp_bundles"
    __table_args__ = (
        UniqueConstraint("owner", "bundle_id", name="uq_mf_wsp_bundles_owner_bundle_id"),
        CheckConstraint(_in("bundle_kind", BUNDLE_KINDS), name="bundle_kind_known"),
        CheckConstraint(_in("sensitivity_value", SENSITIVITY_VALUES), name="sensitivity_known"),
        CheckConstraint(_in("import_status", IMPORT_STATUSES), name="import_status_known"),
    )

    bundle_sha256: Mapped[str] = mapped_column(String(64), primary_key=True)
    owner: Mapped[str] = mapped_column(String(64), nullable=False)
    bundle_id: Mapped[str] = mapped_column(String(64), nullable=False)
    bundle_kind: Mapped[str] = mapped_column(String(16), nullable=False)
    artifact_kind: Mapped[str] = mapped_column(String(16), nullable=False)
    artifact_id: Mapped[str] = mapped_column(String(128), nullable=False)
    detector_digest: Mapped[str | None] = mapped_column(String(64), nullable=True)
    model: Mapped[str] = mapped_column(String(256), nullable=False)
    layer: Mapped[int] = mapped_column(Integer, nullable=False)
    sensitivity_value: Mapped[str] = mapped_column(String(16), nullable=False)
    manifest: Mapped[dict[str, Any]] = mapped_column(JsonColumn, nullable=False)
    bytes: Mapped[int] = mapped_column(BigInteger, nullable=False)
    import_status: Mapped[str] = mapped_column(String(16), nullable=False)
    received_at: Mapped[dt.datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    received_by: Mapped[str] = mapped_column(String(64), nullable=False)
    #: ``{code, message}`` of a failed import (FTASKS 7.5), else null.
    import_failure: Mapped[dict[str, Any] | None] = mapped_column(JsonColumn, nullable=True)


class WspEndpoint(WSPBase):
    __tablename__ = "mf_wsp_endpoints"
    __table_args__ = (
        CheckConstraint(_in("kind", ENDPOINT_KINDS), name="kind_known"),
        CheckConstraint(_in("locality", LOCALITIES), name="locality_known"),
        CheckConstraint(_in("output_cap_field", OUTPUT_CAP_FIELDS), name="output_cap_field_known"),
        CheckConstraint(_in("serving", SERVINGS), name="serving_known"),
        # PF-24: an external endpoint is metered in USD, so it must carry both prices.
        CheckConstraint(
            "locality <> 'external' OR "
            "(price_in_per_mtok IS NOT NULL AND price_out_per_mtok IS NOT NULL)",
            name="external_has_prices",
        ),
        # Feature 018 (migration 0029): column-level CHECKs there, the same names here.
        CheckConstraint(_in("action_protocol", ACTION_PROTOCOLS), name="action_protocol_known"),
        CheckConstraint("context_window_tokens > 0", name="context_window_positive"),
        # 2026-11-21 (migration 0032): column-level CHECKs there, the same names here.
        CheckConstraint(_in("thinking", THINKING_VALUES), name="thinking_known"),
        CheckConstraint("thinking_tokens > 0", name="thinking_tokens_positive"),
        CheckConstraint(_in("output_cap", OUTPUT_CAP_VALUES), name="output_cap_known"),
        # 2026-11-21 (migration 0034): a column-level CHECK there, the same name here.
        CheckConstraint(_in("paging", PAGING_VALUES), name="paging_known"),
    )

    id: Mapped[str] = mapped_column(String(40), primary_key=True)
    kind: Mapped[str] = mapped_column(String(24), nullable=False)
    base_url: Mapped[str] = mapped_column(String(512), nullable=False)
    model: Mapped[str] = mapped_column(String(256), nullable=False)
    locality: Mapped[str] = mapped_column(String(16), nullable=False)
    price_in_per_mtok: Mapped[Decimal | None] = mapped_column(Numeric(12, 6), nullable=True)
    price_out_per_mtok: Mapped[Decimal | None] = mapped_column(Numeric(12, 6), nullable=True)
    price_cache_write_per_mtok: Mapped[Decimal | None] = mapped_column(
        Numeric(12, 6), nullable=True
    )
    price_cache_read_per_mtok: Mapped[Decimal | None] = mapped_column(Numeric(12, 6), nullable=True)
    output_cap_field: Mapped[str] = mapped_column(String(32), nullable=False)
    serving: Mapped[str] = mapped_column(String(16), nullable=False)
    key_ciphertext: Mapped[bytes | None] = mapped_column(LargeBinary, nullable=True)
    key_fingerprint: Mapped[str | None] = mapped_column(String(8), nullable=True)
    created_at: Mapped[dt.datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    created_by: Mapped[str] = mapped_column(String(64), nullable=False)
    removed_at: Mapped[dt.datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    #: Feature 018 (FR-018.8, FR-018.12). Neither is a sealed (bound) field, so the operator's
    #: PATCH never breaks a key; each joins the endpoint fingerprint only away from its default
    #: (F18-D8, ``wsp.runs.endpoint_fingerprint``).
    action_protocol: Mapped[str] = mapped_column(
        String(16), nullable=False, server_default=DEFAULT_ACTION_PROTOCOL
    )
    context_window_tokens: Mapped[int | None] = mapped_column(Integer, nullable=True)
    #: 2026-11-21 (migration 0032): whether the engine asks the editor to think, and the reasoning
    #: allowance per turn (null: the engine's default, ``context.DEFAULT_THINKING_TOKENS``). Not
    #: sealed; each joins the fingerprint only away from its default, like the two above.
    thinking: Mapped[str] = mapped_column(
        String(8), nullable=False, server_default=DEFAULT_THINKING
    )
    thinking_tokens: Mapped[int | None] = mapped_column(Integer, nullable=True)
    #: 2026-11-21 (migration 0032): whether each request names the per-turn output cap.
    output_cap: Mapped[str] = mapped_column(
        String(4), nullable=False, server_default=DEFAULT_OUTPUT_CAP
    )
    #: 2026-11-21 (migration 0034): whether the engine pages and keeps its ledger. Not sealed;
    #: joins the fingerprint only away from its default (``off``), like the fields above.
    paging: Mapped[str] = mapped_column(String(4), nullable=False, server_default=DEFAULT_PAGING)


class WspWorkerBoot(WSPBase):
    """The fresh-process ledger (P49): a ``boot_id`` seen here is never pushed to again.

    ``run_id`` is a plain column, not a foreign key: ``mf_wsp_runs.boot_id`` already references this
    table, and a foreign key each way is a cycle neither engine can create in one step
    (OVN-wsp-forge-10).
    """

    __tablename__ = "mf_wsp_worker_boots"

    boot_id: Mapped[str] = mapped_column(String(32), primary_key=True)
    first_seen_at: Mapped[dt.datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    run_id: Mapped[str | None] = mapped_column(String(16), nullable=True)


class WspRun(WSPBase):
    __tablename__ = "mf_wsp_runs"
    __table_args__ = (
        CheckConstraint(_in("operation", OPERATIONS), name="operation_known"),
        CheckConstraint(_in("style", STYLES), name="style_known"),
        CheckConstraint(_in("status", RUN_STATUSES), name="status_known"),
        CheckConstraint(
            f"stop_reason IS NULL OR {_in('stop_reason', STOP_REASONS)}", name="stop_reason_known"
        ),
        CheckConstraint(f"outcome IS NULL OR {_in('outcome', OUTCOMES)}", name="outcome_known"),
        CheckConstraint(
            _in("sensitivity_effective", EFFECTIVE_SENSITIVITY), name="sensitivity_effective_known"
        ),
        CheckConstraint(
            f"endpoint_locality IS NULL OR {_in('endpoint_locality', LOCALITIES)}",
            name="endpoint_locality_known",
        ),
        CheckConstraint(
            "status <> 'completed' OR outcome IS NOT NULL", name="completed_has_outcome"
        ),
        CheckConstraint(
            "(outcome = 'candidate') = (patch_sha256 IS NOT NULL)", name="candidate_iff_patch"
        ),
        # R3A-13: no token is ever minted for a rerun or a replay, and neither has an endpoint.
        CheckConstraint(
            "style NOT IN ('program_rerun', 'replay') OR "
            "(endpoint_id IS NULL AND gateway_token_sha256 IS NULL)",
            name="rerun_replay_have_no_token",
        ),
        # Feature 018 (migration 0029): the engine's counters, as column-level CHECKs there.
        CheckConstraint(
            f"action_protocol IS NULL OR {_in('action_protocol', ACTION_PROTOCOLS)}",
            name="action_protocol_known",
        ),
        CheckConstraint("handoffs >= 0", name="handoffs_non_negative"),
        CheckConstraint("refused_turns >= 0", name="refused_turns_non_negative"),
        CheckConstraint("kernel_restarts >= 0", name="kernel_restarts_non_negative"),
        CheckConstraint("reasoning_turns >= 0", name="reasoning_turns_non_negative"),
        CheckConstraint("repeated_turns >= 0", name="repeated_turns_non_negative"),
        CheckConstraint("served_context_tokens > 0", name="served_context_tokens_positive"),
        # Paging (migration 0034): column-level CHECKs there, the same names here.
        CheckConstraint("pages_out >= 0", name="pages_out_non_negative"),
        CheckConstraint("recalls >= 0", name="recalls_non_negative"),
        CheckConstraint("ledger_rows >= 0", name="ledger_rows_non_negative"),
        CheckConstraint("peak_prompt_tokens >= 0", name="peak_prompt_tokens_non_negative"),
        # Live progress (migration 0030): column-level CHECKs there, the same names here.
        CheckConstraint(
            f"worker_state IS NULL OR {_in('worker_state', WORKER_STATES)}",
            name="worker_state_known",
        ),
        CheckConstraint("worker_elapsed_s >= 0", name="worker_elapsed_non_negative"),
        Index("ix_mf_wsp_runs_status_created_at", "status", "created_at"),
        Index("ix_mf_wsp_runs_caller_status", "caller", "status"),
        Index(
            "uq_mf_wsp_runs_one_active",
            text("(1)"),
            unique=True,
            postgresql_where=text("status IN ('pushing', 'running')"),
            sqlite_where=text("status IN ('pushing', 'running')"),
        ),
    )

    run_id: Mapped[str] = mapped_column(String(16), primary_key=True)
    operation: Mapped[str] = mapped_column(String(16), nullable=False)
    style: Mapped[str] = mapped_column(String(16), nullable=False)
    status: Mapped[str] = mapped_column(String(16), nullable=False)
    stop_reason: Mapped[str | None] = mapped_column(String(16), nullable=True)
    outcome: Mapped[str | None] = mapped_column(String(16), nullable=True)
    request: Mapped[dict[str, Any]] = mapped_column(JsonColumn, nullable=False)
    request_sha256: Mapped[str] = mapped_column(String(64), nullable=False)
    bundle_sha256: Mapped[str] = mapped_column(
        String(64), ForeignKey("mf_wsp_bundles.bundle_sha256"), nullable=False
    )
    endpoint_id: Mapped[str | None] = mapped_column(
        String(40), ForeignKey("mf_wsp_endpoints.id"), nullable=True
    )
    endpoint_locality: Mapped[str | None] = mapped_column(String(16), nullable=True)
    sensitivity_effective: Mapped[str] = mapped_column(String(16), nullable=False)
    provider_id: Mapped[str] = mapped_column(String(64), nullable=False)
    provider_version: Mapped[str] = mapped_column(String(32), nullable=False)
    approval_id: Mapped[str | None] = mapped_column(String(40), nullable=True)
    gateway_token_sha256: Mapped[str | None] = mapped_column(String(64), nullable=True)
    token_expires_at: Mapped[dt.datetime | None] = mapped_column(
        DateTime(timezone=True), nullable=True
    )
    boot_id: Mapped[str | None] = mapped_column(
        String(32), ForeignKey("mf_wsp_worker_boots.boot_id"), nullable=True
    )
    pushed_at: Mapped[dt.datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    last_status_at: Mapped[dt.datetime | None] = mapped_column(
        DateTime(timezone=True), nullable=True
    )
    result_pulled_at: Mapped[dt.datetime | None] = mapped_column(
        DateTime(timezone=True), nullable=True
    )
    acked_at: Mapped[dt.datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    endpoint_fingerprint: Mapped[str | None] = mapped_column(String(64), nullable=True)
    caller: Mapped[str] = mapped_column(String(64), nullable=False)
    caller_ref: Mapped[str | None] = mapped_column(String(128), nullable=True)
    content_sha256: Mapped[str | None] = mapped_column(String(64), nullable=True)
    usage: Mapped[dict[str, Any]] = mapped_column(JsonColumn, nullable=False)
    forward_passes_used: Mapped[int | None] = mapped_column(Integer, nullable=True)
    transcript_sha256: Mapped[str | None] = mapped_column(String(64), nullable=True)
    result_sha256: Mapped[str | None] = mapped_column(String(64), nullable=True)
    patch_sha256: Mapped[str | None] = mapped_column(String(64), nullable=True)
    program_sha256: Mapped[str | None] = mapped_column(String(64), nullable=True)
    replay_of: Mapped[str | None] = mapped_column(String(16), nullable=True)
    replay_reproduced: Mapped[bool | None] = mapped_column(Boolean, nullable=True)
    replay_patch_sha256: Mapped[str | None] = mapped_column(String(64), nullable=True)
    failure: Mapped[dict[str, Any] | None] = mapped_column(NullableJson, nullable=True)
    created_at: Mapped[dt.datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    started_at: Mapped[dt.datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    finished_at: Mapped[dt.datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    deleted_at: Mapped[dt.datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    #: Feature 018 (FR-018.11a): ``result.json``'s ``engine`` block, stored by the intake; null
    #: for a provider that writes none. Served in the list rows, never in ``run_document``.
    action_protocol: Mapped[str | None] = mapped_column(String(16), nullable=True)
    handoffs: Mapped[int | None] = mapped_column(Integer, nullable=True)
    refused_turns: Mapped[int | None] = mapped_column(Integer, nullable=True)
    kernel_restarts: Mapped[int | None] = mapped_column(Integer, nullable=True)
    #: 2026-11-21 (migration 0032): the engine block's ``reasoning_turns``, the replies that
    #: carried reasoning; null for a provider or an engine that writes none.
    reasoning_turns: Mapped[int | None] = mapped_column(Integer, nullable=True)
    #: 2026-11-21 (migration 0032): the context window miServer reported for the endpoint's
    #: model when the run was pushed (``dispatch.served_window``), the same figure the push's
    #: ``spec.editor.served_context_tokens`` carries; the gateway sizes an uncapped request to it
    #: when the endpoint states no window. Null when not read (not miServer, or the endpoint
    #: states its own window, or the read failed).
    served_context_tokens: Mapped[int | None] = mapped_column(Integer, nullable=True)
    #: 2026-11-21 (migration 0032): the engine block's ``repeated_turns``, the actions its repeat
    #: guard did not run (the same tool and arguments on the same weights).
    repeated_turns: Mapped[int | None] = mapped_column(Integer, nullable=True)
    #: 2026-11-21 (migration 0034): the engine block's paging counters: results paged out,
    #: recalls that returned a page, ledger rows, and the largest prompt the engine estimated
    #: (tokens). Null for a provider or an engine that writes none.
    pages_out: Mapped[int | None] = mapped_column(Integer, nullable=True)
    recalls: Mapped[int | None] = mapped_column(Integer, nullable=True)
    ledger_rows: Mapped[int | None] = mapped_column(Integer, nullable=True)
    peak_prompt_tokens: Mapped[int | None] = mapped_column(Integer, nullable=True)
    #: Live progress (migration 0030, 2026-11-20): the worker's last state and its elapsed whole
    #: seconds as ``GET /status`` reported them, stored by the driver on every heartbeat it reads
    #: (``last_status_at`` is the heartbeat's time). Untrusted values, bounded on the way in
    #: (``dispatch.worker_heartbeat``); null before the first heartbeat.
    worker_state: Mapped[str | None] = mapped_column(String(16), nullable=True)
    worker_elapsed_s: Mapped[int | None] = mapped_column(Integer, nullable=True)


class WspReservation(WSPBase):
    __tablename__ = "mf_wsp_reservations"
    __table_args__ = (Index("ix_mf_wsp_reservations_run_id", "run_id"),)

    id: Mapped[int] = mapped_column(BigInteger().with_variant(Integer, "sqlite"), primary_key=True)
    run_id: Mapped[str] = mapped_column(
        String(16), ForeignKey("mf_wsp_runs.run_id"), nullable=False
    )
    endpoint_id: Mapped[str] = mapped_column(String(40), nullable=False)
    external: Mapped[bool] = mapped_column(Boolean, nullable=False)
    worst_case_usd: Mapped[Decimal] = mapped_column(Numeric(12, 6), nullable=False)
    worst_case_tokens: Mapped[int] = mapped_column(BigInteger, nullable=False)
    created_at: Mapped[dt.datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    settled_at: Mapped[dt.datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    settled_cost_usd: Mapped[Decimal | None] = mapped_column(Numeric(12, 6), nullable=True)


class WspSpendDay(WSPBase):
    """The per-UTC-day lock row the D19 daily cap takes ``FOR UPDATE`` on (R2A-25)."""

    __tablename__ = "mf_wsp_spend_days"

    utc_day: Mapped[dt.date] = mapped_column(Date, primary_key=True)
    external_settled_usd: Mapped[Decimal] = mapped_column(Numeric(12, 6), nullable=False)
    external_open_usd: Mapped[Decimal] = mapped_column(Numeric(12, 6), nullable=False)


class WspCandidate(WSPBase):
    __tablename__ = "mf_wsp_candidates"
    __table_args__ = (
        CheckConstraint(_in("parent_kind", PARENT_KINDS), name="parent_kind_known"),
        CheckConstraint("changed_entries > 0", name="changed_entries_positive"),
    )

    candidate_id: Mapped[str] = mapped_column(String(17), primary_key=True)
    run_id: Mapped[str] = mapped_column(
        String(16), ForeignKey("mf_wsp_runs.run_id"), nullable=False, unique=True
    )
    parent_kind: Mapped[str] = mapped_column(String(16), nullable=False)
    parent_ref: Mapped[str] = mapped_column(String(512), nullable=False)
    bundle_base_sha256: Mapped[str] = mapped_column(String(64), nullable=False)
    patch_sha256: Mapped[str] = mapped_column(String(64), nullable=False, unique=True)
    changed_entries: Mapped[int] = mapped_column(Integer, nullable=False)
    tensors: Mapped[dict[str, Any]] = mapped_column(JsonColumn, nullable=False)
    created_at: Mapped[dt.datetime] = mapped_column(DateTime(timezone=True), nullable=False)


class WspProgram(WSPBase):
    __tablename__ = "mf_wsp_programs"

    program_sha256: Mapped[str] = mapped_column(String(64), primary_key=True)
    first_run_id: Mapped[str] = mapped_column(
        String(16), ForeignKey("mf_wsp_runs.run_id"), nullable=False
    )
    bytes: Mapped[int] = mapped_column(Integer, nullable=False)
    created_at: Mapped[dt.datetime] = mapped_column(DateTime(timezone=True), nullable=False)


class WspVerdict(WSPBase):
    __tablename__ = "mf_wsp_verdicts"
    __table_args__ = (
        UniqueConstraint(
            "candidate_id", "request_sha256", name="uq_mf_wsp_verdicts_candidate_request"
        ),
        CheckConstraint(_in("value", VERDICT_VALUES), name="value_known"),
    )

    verdict_id: Mapped[str] = mapped_column(String(40), primary_key=True)
    candidate_id: Mapped[str] = mapped_column(
        String(17), ForeignKey("mf_wsp_candidates.candidate_id"), nullable=False
    )
    request_sha256: Mapped[str] = mapped_column(String(64), nullable=False)
    value: Mapped[str] = mapped_column(String(16), nullable=False)
    document: Mapped[dict[str, Any]] = mapped_column(JsonColumn, nullable=False)
    snapshot: Mapped[bytes] = mapped_column(LargeBinary, nullable=False)
    run_sha256: Mapped[str] = mapped_column(String(64), nullable=False)
    scorer_owner: Mapped[str] = mapped_column(String(64), nullable=False)
    created_at: Mapped[dt.datetime] = mapped_column(DateTime(timezone=True), nullable=False)


class WspHeldoutUse(WSPBase):
    __tablename__ = "mf_wsp_heldout_uses"
    __table_args__ = (CheckConstraint("heldout_uses >= 0", name="heldout_uses_non_negative"),)

    owner: Mapped[str] = mapped_column(String(64), primary_key=True)
    set_name: Mapped[str] = mapped_column(String(64), primary_key=True)
    row_ids_sha256: Mapped[str] = mapped_column(String(64), primary_key=True)
    heldout_uses: Mapped[int] = mapped_column(Integer, nullable=False)
    last_used_at: Mapped[dt.datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    last_candidate_id: Mapped[str | None] = mapped_column(String(17), nullable=True)


class WspUsage(WSPBase):
    """One editor call the gateway saw: what it spent and, since migration 0030, what became of it.

    **Counts and codes only, never text** (live progress, 2026-11-20): no column holds a prompt,
    an answer or an upstream message. ``finish_reason`` and ``refusal_code`` are short
    identifiers, checked on the way in (``gateway_ledger.call_columns``); anything else is stored
    ``other`` or nothing. ``test_wsp_calls.py`` pins this column set.
    """

    __tablename__ = "mf_wsp_usage"
    __table_args__ = (
        UniqueConstraint("run_id", "seq", name="uq_mf_wsp_usage_run_id_seq"),
        CheckConstraint(
            f"outcome IS NULL OR {_in('outcome', CALL_OUTCOMES)}", name="outcome_known"
        ),
        CheckConstraint("http_status BETWEEN 100 AND 599", name="http_status_range"),
        CheckConstraint("latency_ms >= 0", name="latency_non_negative"),
        # Migration 0033: column-level CHECKs there, the same names here.
        CheckConstraint("sent_bytes > 0", name="sent_bytes_positive"),
        CheckConstraint("prompt_messages >= 0", name="prompt_messages_non_negative"),
    )

    id: Mapped[int] = mapped_column(BigInteger().with_variant(Integer, "sqlite"), primary_key=True)
    run_id: Mapped[str] = mapped_column(
        String(16), ForeignKey("mf_wsp_runs.run_id"), nullable=False
    )
    seq: Mapped[int] = mapped_column(Integer, nullable=False)
    endpoint_id: Mapped[str] = mapped_column(String(40), nullable=False)
    input_tokens: Mapped[int] = mapped_column(Integer, nullable=False)
    output_tokens: Mapped[int] = mapped_column(Integer, nullable=False)
    cost_usd: Mapped[Decimal] = mapped_column(Numeric(12, 6), nullable=False)
    at: Mapped[dt.datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    outcome: Mapped[str | None] = mapped_column(String(16), nullable=True)
    http_status: Mapped[int | None] = mapped_column(Integer, nullable=True)
    latency_ms: Mapped[int | None] = mapped_column(Integer, nullable=True)
    finish_reason: Mapped[str | None] = mapped_column(String(32), nullable=True)
    refusal_code: Mapped[str | None] = mapped_column(String(64), nullable=True)
    #: 2026-11-21 (migration 0033): a served call's prompt evidence, so a restarted gateway
    #: recovers its per-session estimate (``wsp_gateway.meter.PromptMeter``): the bytes it
    #: forwarded, how many messages they held and the conversation's digest. Never text.
    sent_bytes: Mapped[int | None] = mapped_column(Integer, nullable=True)
    prompt_messages: Mapped[int | None] = mapped_column(Integer, nullable=True)
    conversation_sha256: Mapped[str | None] = mapped_column(String(64), nullable=True)
