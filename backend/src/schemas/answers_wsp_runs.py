"""The answers of the WSP run, pool and program routes (feature 021; ``wsp_runs.py``).

Documentation only: the routes still return plain dicts; these models describe them (MF9).
One run row (``list_runs``) carries exactly ``tests/support/wsp_list_keys.py``'s
``RUN_ROW_KEYS``. ``GET /wsp/runs/{id}`` serves the published ``miforge.wsp-run/v1`` document,
declared as ``answers_documents.WspRunDocument`` (a ``$ref`` to its ``$id``, FR-021.9).
"""

from pydantic import BaseModel, Field

from schemas.answers_common import JsonValue, free_form


class WspRunStartRunAnswer(BaseModel):
    """``POST /wsp/runs`` (``api/v1/endpoints/wsp_runs.py:start_run``): the queued run."""

    run_id: str
    request_sha256: str
    status: str


class WspRunReplayRunAnswer(BaseModel):
    """``POST /wsp/runs/{id}/replay`` (``api/v1/endpoints/wsp_runs.py:replay_run``)."""

    run_id: str
    request_sha256: str
    status: str


class WspProgramRerunProgramAnswer(BaseModel):
    """``POST /wsp/programs/{sha}/rerun`` (``api/v1/endpoints/wsp_runs.py:rerun_program``)."""

    run_id: str
    request_sha256: str
    status: str


class WspRunCancelRunAnswer(BaseModel):
    """``POST /wsp/runs/{id}/cancel`` (``api/v1/endpoints/wsp_runs.py:cancel_run``)."""

    run_id: str
    status: str
    stop_reason: str | None


# ------------------------------------------------------------------------------------- the pool
class WspPoolWorker(BaseModel):
    """One ``workers`` item (``wsp/pool.py:pool_document``)."""

    index: int
    state: str
    worker_state: str
    run_id: str | None
    boot_id: str | None
    read_at: str


class WspPoolQueuedByEndpoint(BaseModel):
    """One ``queued_by_endpoint`` item (``wsp/pool.py:pool_document``)."""

    endpoint_id: str
    queued: int


class WspPoolSharedBaseUrl(BaseModel):
    """One ``shared_base_urls`` item (``wsp/pool.py:pool_document``): active runs on one server."""

    base_url: str
    endpoint_ids: list[str]
    run_ids: list[str]


class WspPoolGetPoolAnswer(BaseModel):
    """``GET /wsp/pool`` (``wsp/pool.py:pool_document``)."""

    pool_size: int
    maximum: int
    effective: int
    busy: int
    idle: int
    restarting: int
    not_answering: int
    workers: list[WspPoolWorker]
    queued_by_endpoint: list[WspPoolQueuedByEndpoint]
    queued_without_endpoint: int
    shared_base_urls: list[WspPoolSharedBaseUrl]


# ------------------------------------------------------------------------------- the runs list
class WspRunFailure(BaseModel):
    """A run's ``failure`` (``wsp/runs.py:intake``, ``wsp/dispatch.py``): its cause."""

    code: str
    message: str


class WspRunUsage(BaseModel):
    """A run row's ``usage`` (``api/v1/endpoints/wsp_runs.py:_usage``)."""

    requests: int
    input_tokens: int
    output_tokens: int
    cost_usd: float


class WspRunMetered(BaseModel):
    """A run row's ``metered`` (``wsp/runs.py:metered_totals``): cost and cache totals."""

    cost_usd: float
    cache_read_tokens: int | None
    cache_write_tokens: int | None


class WspRunCompactBudget(BaseModel):
    """One ``progress.budgets`` item of a run row (``wsp/progress.py:compact``)."""

    name: str
    spent: float | None
    cap: float | None
    fraction: float | None
    estimate: bool


class WspRunCompactProgress(BaseModel):
    """A live run row's ``progress`` (``wsp/progress.py:compact``)."""

    elapsed_s: float
    clock_from: str
    worker_state: str | None
    heartbeat_age_s: float | None
    calls: int
    refused: int
    tokens: int
    cost_usd: float
    last_call_at: str | None
    queue_position: int | None
    budgets: list[WspRunCompactBudget]


class WspRunRow(BaseModel):
    """One run row (``api/v1/endpoints/wsp_runs.py:list_runs``): miForge's own, not wsp-run/v1."""

    run_id: str
    status: str
    kind: str
    mode: str
    caller: str
    caller_ref: str | None
    endpoint_id: str | None
    stop_reason: str | None
    outcome: str | None
    outcome_reason: str | None
    created_at: str
    provider: str
    provider_version: str
    failure: WspRunFailure | None
    usage: WspRunUsage
    started_at: str | None
    finished_at: str | None
    action_protocol: str | None
    handoffs: int | None
    refused_turns: int | None
    kernel_restarts: int | None
    thinking: str | None
    thinking_tokens: int | None
    output_cap: str
    reasoning_turns: int | None
    repeated_turns: int | None
    paging: str | None
    pages_out: int | None
    recalls: int | None
    ledger_rows: int | None
    peak_prompt_tokens: int | None
    reasoning_returned: bool | None
    forward_passes_used: int | None
    window_tokens: int | None
    window_source: str | None
    window_reported_tokens: int | None
    output_ceiling_tokens: int | None
    output_ceiling_lowered: bool | None
    window_warnings: list[str]
    editor_wait_since: str | None = Field(
        description=(
            "When the run began waiting for its editor model (the driver's claim read no window: "
            "a miServer endpoint with no model resident, or a server that gave no answer); a "
            "queued run with it set is waiting now, and fails editor_not_resident after "
            "WSP_EDITOR_RESIDENT_WAIT_S. Kept after the wait ends; null on a run that never "
            "waited. Added in revision 5 (debt D3, 2026-11-28)."
        )
    )
    metered: WspRunMetered
    progress: WspRunCompactProgress | None


class WspRunListRunsAnswer(BaseModel):
    """``GET /wsp/runs`` (``api/v1/endpoints/wsp_runs.py:list_runs``), newest first."""

    runs: list[WspRunRow]


# --------------------------------------------------------------------------- live progress
class WspRunProgressWorker(BaseModel):
    """``worker`` of a run's progress (``wsp/progress.py:progress_of``): the last heartbeat."""

    state: str | None
    elapsed_s: int | None
    last_heartbeat_at: str | None
    heartbeat_age_s: float | None


class WspRunProgressMetered(BaseModel):
    """``metered`` of a run's progress (``wsp/progress.py:_metered``, ``run_progress``)."""

    calls: int
    refused: int
    input_tokens: int
    output_tokens: int
    cost_usd: float
    last_call_at: str | None
    tokens: int
    cache_read_tokens: int | None
    cache_write_tokens: int | None


class WspRunProgressInFlight(BaseModel):
    """``in_flight`` of a run's progress (``wsp/progress.py:_open``): the open reservation."""

    worst_case_tokens: int
    worst_case_usd: float
    since: str


class WspRunProgressBudget(BaseModel):
    """One ``budgets`` item of a run's progress (``wsp/progress.py:_budget``)."""

    name: str
    unit: str
    spent: float | None
    cap: float | None
    fraction: float | None
    estimate: bool


class WspRunProgressQueue(BaseModel):
    """``queue`` of a queued run's progress (``wsp/progress.py:progress_of``)."""

    position: int
    queued: int
    active_run_id: str | None


class WspRunGetProgressAnswer(BaseModel):
    """``GET /wsp/runs/{id}/progress`` (``wsp/progress.py:run_progress``): live or final."""

    run_id: str
    status: str
    live: bool
    now: str
    created_at: str
    started_at: str | None
    finished_at: str | None
    clock_from: str
    elapsed_s: float
    outcome_reason: str | None
    window_tokens: int | None
    window_source: str | None
    window_reported_tokens: int | None
    output_ceiling_tokens: int | None
    output_ceiling_lowered: bool | None
    window_warnings: list[str]
    editor_wait_since: str | None = Field(
        description=(
            "When the run began waiting for its editor model (null: it never waited); a queued "
            "run with it set is waiting now. As the list row's (revision 5, debt D3)."
        )
    )
    worker: WspRunProgressWorker | None
    metered: WspRunProgressMetered
    in_flight: WspRunProgressInFlight | None
    budgets: list[WspRunProgressBudget]
    queue: WspRunProgressQueue | None


# --------------------------------------------------------------------------- the call timeline
class WspRunCall(BaseModel):
    """One ``calls`` item (``wsp/progress.py:run_calls``): counts and codes only, never text."""

    seq: int
    at: str
    outcome: str | None
    http_status: int | None
    latency_ms: int | None
    finish_reason: str | None
    refusal_code: str | None
    input_tokens: int
    output_tokens: int
    cost_usd: float
    cache_read_tokens: int | None
    cache_write_tokens: int | None


class WspRunGetCallsAnswer(BaseModel):
    """``GET /wsp/runs/{id}/calls`` (``wsp/progress.py:run_calls``), oldest first."""

    run_id: str
    status: str
    live: bool
    calls: list[WspRunCall]
    more: bool


# ------------------------------------------------------------------------------ the transcript
class WspRunGetTranscriptAnswer(BaseModel):
    """``GET /wsp/runs/{id}/transcript`` (``api/v1/endpoints/wsp_runs.py:get_transcript``)."""

    run_id: str
    transcript_sha256: str | None
    entries: list[dict[str, JsonValue]] = free_form(
        "each stored transcript line, served as data: the engine's own JSON object, or "
        '{"raw": <text>} for a line that is not one'
    )
    partial: bool
    offset: int
    total: int
    next_offset: int | None


# -------------------------------------------------------------------------------- the programs
class WspProgramGetProgramAnswer(BaseModel):
    """``GET /wsp/programs/{sha}`` (``api/v1/endpoints/wsp_runs.py:get_program``): data only."""

    program_sha256: str
    first_run_id: str
    bytes: int
    created_at: str
    source: str
    reruns: list[str]
