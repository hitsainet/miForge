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

**A pool of workers (feature 019 P3, F19-D17):** at most one run in ``pushing`` or ``running`` per
scheduling key (``uq_mf_wsp_runs_one_active_key``), and at most one run holding each worker
(``uq_mf_wsp_runs_one_per_worker``); before 019 one index on a constant admitted one run in all.
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
#: Feature 020 (FR-020.4, F20-D6; migration 0039): the three effort levels. A hosted Anthropic
#: endpoint takes ``default`` or an effort only; a hosted OpenAI-compatible one may declare an
#: effort (``reasoning_effort``); a local endpoint takes none (``wsp.endpoints.check_thinking``).
#: Never above ``high`` (W3-D9).
EFFORTS = ("low", "medium", "high")
THINKING_VALUES = ("off", "on", "default", *EFFORTS)
DEFAULT_THINKING = "off"
#: Whether each editor request names the per-turn output cap (operator, 2026-11-21): ``on`` (the
#: default, ``miwsp_engine.context.editor_output_cap``) or ``off`` (no cap named, so the gateway's
#: window-aware default applies). Both providers read it.
OUTPUT_CAP_VALUES = ("on", "off")
DEFAULT_OUTPUT_CAP = "on"
#: Whether ``miwsp_engine`` pages bulky tool results out of the history and keeps its experiment
#: ledger (operator, 2026-11-21, migration 0034): ``on`` (the default) or ``off``. The engine's
#: only.
PAGING_VALUES = ("on", "off")
DEFAULT_PAGING = "on"
#: How ``miwsp_engine`` keeps a session's conversation (feature 020, FR-020.1, F20-D4; migration
#: 0039): ``paging`` (018's, which rewrites older results into stubs; ``paging`` above is its
#: switch) or ``append_only`` (nothing already sent ever changes). A hosted endpoint defaults to
#: ``append_only``, and a hosted Anthropic one cannot choose ``paging``
#: (``wsp.endpoints.check_history_mode``).
HISTORY_MODES = ("paging", "append_only")
DEFAULT_HISTORY_MODE = "paging"
#: Where a run's window came from (feature 019, FR-019.2, migration 0036): the endpoint's stated
#: ``context_window_tokens``, the server's report at push, or the 8,192 assumed when neither.
WINDOW_SOURCES = ("stated", "reported", "assumed")
#: A protocol test's result on its endpoint (feature 019 P2, migration 0037).
PROTOCOL_TEST_RESULTS = ("passed", "failed")
#: Feature 019 P3 (migration 0038): the statuses that hold a scheduling key (active) and those that
#: hold a worker (a ``collecting`` run's worker still serves its result).
ACTIVE_STATES_SQL = "status IN ('pushing', 'running')"
HOLDING_STATES_SQL = "status IN ('pushing', 'running', 'collecting')"
HOLDING_STATES = ("pushing", "running", "collecting")
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
        # Feature 020 (migration 0039): a column-level CHECK there, the same name here.
        CheckConstraint(_in("history_mode", HISTORY_MODES), name="history_mode_known"),
        # Feature 019 (migration 0036): column-level CHECKs there, the same names here.
        CheckConstraint("output_ceiling_tokens >= 256", name="output_ceiling_at_least_256"),
        CheckConstraint("registration_window_tokens > 0", name="registration_window_positive"),
        # Feature 019 P2 (migration 0037): column-level CHECKs there, the same names here.
        CheckConstraint(
            f"protocol_test_result IS NULL OR {_in('protocol_test_result', PROTOCOL_TEST_RESULTS)}",
            name="protocol_test_result_known",
        ),
        CheckConstraint(
            "protocol_tested_action_protocol IS NULL OR "
            f"{_in('protocol_tested_action_protocol', ACTION_PROTOCOLS)}",
            name="protocol_tested_action_protocol_known",
        ),
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
    #: Feature 020 (FR-020.1, migration 0039): ``paging`` or ``append_only``. Not sealed; joins
    #: the fingerprint only away from ``paging`` (F18-D8's rule), so a ``paging`` row hashes as
    #: before 020.
    history_mode: Mapped[str] = mapped_column(
        String(16), nullable=False, server_default=DEFAULT_HISTORY_MODE
    )
    #: Feature 019 (FR-019.14, migration 0036): the most output tokens one editor request may ask
    #: for (null: 32,000, ``wsp_worker.editor_limits``); at least 256, and at most the window when
    #: one is known. Not sealed; joins the fingerprint only when set (F18-D8's rule).
    output_ceiling_tokens: Mapped[int | None] = mapped_column(Integer, nullable=True)
    #: Feature 019 (FR-019.3, F19-D4): the window the server reported when the endpoint was
    #: registered (null: nothing reported), and when that read ran, whatever it returned. Shown
    #: beside the latest run's read; a run never uses it (F19-D11).
    registration_window_tokens: Mapped[int | None] = mapped_column(Integer, nullable=True)
    registration_window_read_at: Mapped[dt.datetime | None] = mapped_column(
        DateTime(timezone=True), nullable=True
    )
    #: Feature 019 P2 (FR-019.12, F19-D9, F19-D27; migration 0037): the LATEST protocol test's
    #: record, written when its run ends (``wsp.protocol_tests.record``): passed or failed, the
    #: run, when, and the base URL, model and action protocol the run was pushed with. Whether a
    #: pass is current is computed on read (``protocol_tests.pass_state``). Outside the
    #: fingerprint and the sealed fields: storing a result never fails a queued run.
    protocol_test_result: Mapped[str | None] = mapped_column(String(8), nullable=True)
    protocol_test_run_id: Mapped[str | None] = mapped_column(String(16), nullable=True)
    protocol_test_at: Mapped[dt.datetime | None] = mapped_column(
        DateTime(timezone=True), nullable=True
    )
    protocol_tested_base_url: Mapped[str | None] = mapped_column(String(512), nullable=True)
    protocol_tested_model: Mapped[str | None] = mapped_column(String(256), nullable=True)
    protocol_tested_action_protocol: Mapped[str | None] = mapped_column(String(16), nullable=True)


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
        # Feature 019 (migration 0036): column-level CHECKs there, the same names here.
        CheckConstraint("window_tokens > 0", name="window_tokens_positive"),
        CheckConstraint(
            f"window_source IS NULL OR {_in('window_source', WINDOW_SOURCES)}",
            name="window_source_known",
        ),
        CheckConstraint("output_ceiling_tokens > 0", name="output_ceiling_positive"),
        Index("ix_mf_wsp_runs_status_created_at", "status", "created_at"),
        Index("ix_mf_wsp_runs_caller_status", "caller", "status"),
        # Feature 019 P3 (migration 0038, F19-D17, TP-4): one active run per scheduling key, and
        # one run holding each worker; replacing 016's one-active index on a constant.
        Index(
            "uq_mf_wsp_runs_one_active_key",
            "schedule_key",
            unique=True,
            postgresql_where=text(f"{ACTIVE_STATES_SQL} AND schedule_key IS NOT NULL"),
            sqlite_where=text(f"{ACTIVE_STATES_SQL} AND schedule_key IS NOT NULL"),
        ),
        Index(
            "uq_mf_wsp_runs_one_per_worker",
            "worker_index",
            unique=True,
            postgresql_where=text(f"{HOLDING_STATES_SQL} AND worker_index IS NOT NULL"),
            sqlite_where=text(f"{HOLDING_STATES_SQL} AND worker_index IS NOT NULL"),
        ),
        CheckConstraint("worker_index >= 0", name="worker_index_non_negative"),
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
    #: ``action_protocol`` since 019 P2 (TID §3.6): the protocol the run was PUSHED with, written
    #: at the push; the intake refuses an engine block naming another (``wsp_result_invalid``),
    #: and a protocol test records it as the protocol tested.
    action_protocol: Mapped[str | None] = mapped_column(String(16), nullable=True)
    handoffs: Mapped[int | None] = mapped_column(Integer, nullable=True)
    refused_turns: Mapped[int | None] = mapped_column(Integer, nullable=True)
    kernel_restarts: Mapped[int | None] = mapped_column(Integer, nullable=True)
    #: 2026-11-21 (migration 0032): the engine block's ``reasoning_turns``, the replies that
    #: carried reasoning; null for a provider or an engine that writes none.
    reasoning_turns: Mapped[int | None] = mapped_column(Integer, nullable=True)
    #: The window the server reported at this push (every serving, 019; F19-D28), written even
    #: when the endpoint states one, so a stated window above it is found (F19-D10); the push's
    #: ``spec.editor.served_context_tokens`` carries the same figure. Null when nothing was read.
    #: Before 019 (migration 0032, 2026-11-21) it was miServer's only, read only when no window
    #: was stated: rows from then keep that meaning, nothing is backfilled.
    served_context_tokens: Mapped[int | None] = mapped_column(Integer, nullable=True)
    #: Feature 019 (FR-019.2, FR-019.15, migration 0036): the window this run used and its source
    #: (``WINDOW_SOURCES``), and the output ceiling it used, after F19-D12 lowered it to a smaller
    #: reported window (``output_ceiling_lowered``). Written at push (``wsp.window_read.resolve``),
    #: and again at intake when a cut-off reply lowered an uncapped session's ceiling
    #: (2026-11-28, ``miwsp_engine`` step 5d: the lowered figure, ``output_ceiling_lowered`` true);
    #: the worker's ``spec.editor`` and the gateway's session entry carry the same figures. Null
    #: on a run pushed before 019 ("not recorded"). miForge's own: never in ``run_document``.
    window_tokens: Mapped[int | None] = mapped_column(Integer, nullable=True)
    window_source: Mapped[str | None] = mapped_column(String(16), nullable=True)
    output_ceiling_tokens: Mapped[int | None] = mapped_column(Integer, nullable=True)
    output_ceiling_lowered: Mapped[bool | None] = mapped_column(Boolean, nullable=True)
    #: Feature 019 P2 (F19-D25; migration 0037): the endpoint this run is the protocol test of,
    #: set only by the protocol-test route (``runs.create_run`` with ``Prepared.protocol_test``);
    #: equal to ``endpoint_id`` (``create_run`` is its one writer; no CHECK, TID §4.2). A
    #: ``caller_ref`` of ``protocol_test:...`` marks nothing.
    protocol_test_endpoint_id: Mapped[str | None] = mapped_column(String(40), nullable=True)
    #: Feature 019 P3 (F19-D17, F19-D19; migration 0038): written at the claim. ``schedule_key``
    #: (``wsp.scheduling.key_for``; null for a replay or a rerun and on every queued row),
    #: ``worker_index`` (the worker the run holds, ``0..max-1``; null before 019, read as 0), and
    #: ``push_heartbeat_at`` (the push lease: the push task writes it as it streams; a ``pushing``
    #: run whose lease lapses is failed ``worker_push_failed``).
    schedule_key: Mapped[str | None] = mapped_column(String(600), nullable=True)
    worker_index: Mapped[int | None] = mapped_column(Integer, nullable=True)
    push_heartbeat_at: Mapped[dt.datetime | None] = mapped_column(
        DateTime(timezone=True), nullable=True
    )
    #: 2026-11-21 (migration 0032): the engine block's ``repeated_turns``, the actions its repeat
    #: guard did not run (the same tool and arguments on the same weights); and, in a run whose
    #: transcript header has ``edit_repeats: 1`` (2026-11-28), also the edits that recreated
    #: weights already measured and reverted (``miwsp_engine.session`` step 5c), which did run.
    repeated_turns: Mapped[int | None] = mapped_column(Integer, nullable=True)
    #: 2026-11-21 (migration 0034): the engine block's paging counters: results paged out,
    #: recalls that returned a page, ledger rows, and the largest prompt the engine estimated
    #: (tokens). Null for a provider or an engine that writes none.
    pages_out: Mapped[int | None] = mapped_column(Integer, nullable=True)
    recalls: Mapped[int | None] = mapped_column(Integer, nullable=True)
    ledger_rows: Mapped[int | None] = mapped_column(Integer, nullable=True)
    peak_prompt_tokens: Mapped[int | None] = mapped_column(Integer, nullable=True)
    #: 2026-11-23 (migration 0035): why a verified patch was kept as evidence and the run ended
    #: ``no_change`` (``wsp.runs.qualify``); null on every other run. miForge's own: served in the
    #: list rows and the progress read, never in ``run_document``.
    outcome_reason: Mapped[str | None] = mapped_column(String(300), nullable=True)
    #: 018 FU-2 (migration 0040, 2026-11-25): the claimed figures beside the measured ones, from
    #: the run's ``submitted_measured`` transcript line (``wsp.runs.submitted_check``), stored by
    #: the intake. ``run_document`` serves it as ``submitted_check`` when set; null on every run
    #: taken in before it, and on a run whose submitted weights were not measured.
    submitted_check: Mapped[dict[str, Any] | None] = mapped_column(NullableJson, nullable=True)
    #: Live progress (migration 0030, 2026-11-20): the worker's last state and its elapsed whole
    #: seconds as ``GET /status`` reported them, stored by the driver on every heartbeat it reads
    #: (``last_status_at`` is the heartbeat's time). Untrusted values, bounded on the way in
    #: (``dispatch.worker_heartbeat``); null before the first heartbeat.
    worker_state: Mapped[str | None] = mapped_column(String(16), nullable=True)
    worker_elapsed_s: Mapped[int | None] = mapped_column(Integer, nullable=True)
    #: 2026-11-27 (migration 0042, run ``wsp_4fbf5a18758e``): when the driver first found the
    #: run's miServer endpoint reporting no window (no model resident) in its current wait
    #: (``wsp.window_read.awaits_resident``). The claim skips the run while the read is empty and
    #: fails it ``editor_not_resident`` once ``wsp_editor_resident_wait_s`` has passed since this
    #: time; a push that puts the run back in the queue starts a new wait. Null on a run that
    #: never waited. miForge's own: never in ``run_document``.
    editor_wait_since: Mapped[dt.datetime | None] = mapped_column(
        DateTime(timezone=True), nullable=True
    )


#: Feature 019 P3 (migration 0038): what the driver last read from a worker; ``not_answering`` when
#: its ``/ready`` failed.
WORKER_ROW_STATES = (*WORKER_STATES, "not_answering")


class WspWorker(WSPBase):
    """Feature 019 P3 (F19-D32, migration 0038): one row per provisioned worker, written by every
    pass (``wsp.pool.record_ready``), so the pool route reads the database and never a worker
    (ADR-007). A row at or above the maximum is deleted by the pass."""

    __tablename__ = "mf_wsp_workers"
    __table_args__ = (
        CheckConstraint(_in("state", WORKER_ROW_STATES), name="state_known"),
        CheckConstraint("worker_index >= 0", name="index_non_negative"),
    )

    worker_index: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=False)
    address: Mapped[str] = mapped_column(String(256), nullable=False)
    state: Mapped[str] = mapped_column(String(16), nullable=False)
    boot_id: Mapped[str | None] = mapped_column(String(32), nullable=True)
    read_at: Mapped[dt.datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    answered_at: Mapped[dt.datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    #: 2026-11-26 (migration 0041, the 019 P4 disk gap): the free and total bytes of the
    #: filesystem under the worker's ``/work`` emptyDir, as its last answered ``/ready`` reported
    #: them (the worker is untrusted: bounded non-negative integers, else null). Null before the
    #: worker first reports them.
    work_free_bytes: Mapped[int | None] = mapped_column(BigInteger, nullable=True)
    work_total_bytes: Mapped[int | None] = mapped_column(BigInteger, nullable=True)


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
        # Feature 020 (migration 0039): column-level CHECKs there, the same names here.
        CheckConstraint("cache_read_tokens >= 0", name="cache_read_non_negative"),
        CheckConstraint("cache_write_tokens >= 0", name="cache_write_non_negative"),
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
    #: Feature 020 (FR-020.20, F20-D10; migration 0039): the call's input tokens read from the
    #: prompt cache and written to it, as the gateway metered them (Anthropic
    #: ``cache_read_input_tokens`` / ``cache_creation_input_tokens``; OpenAI-compatible
    #: ``prompt_tokens_details.cached_tokens`` and no writes). Null on a call from before 020
    #: ("not recorded", never zero).
    cache_read_tokens: Mapped[int | None] = mapped_column(Integer, nullable=True)
    cache_write_tokens: Mapped[int | None] = mapped_column(Integer, nullable=True)
