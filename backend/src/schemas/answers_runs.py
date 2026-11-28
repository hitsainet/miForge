"""The answer models of the run routes (feature 021, P3): ``runs_index``, ``runs``, ``runs_review``,
``runs_acquisition``, ``runs_loop`` and ``scoring``.

Documentation only (MF9): the routes still return their dicts, and these models change what the
OpenAPI document (and so the published answer contract) says about them, nothing else. Each class
names the function that builds the shape it describes.
"""

from pydantic import BaseModel, Field

from .answers_common import FREE_FORM, JsonValue, NextStep, free_form, may_be_absent
from .answers_documents import BoundaryBlockPart, EvaluationBlockPart

# ---------------------------------------------------------------------------- runs_index


class RunRow(BaseModel):
    """One run in the list (``api/v1/endpoints/runs_index.py:run_row``)."""

    run_id: str
    status: str
    run_type: str
    created_at: str | None
    finished_at: str | None
    last_step: int | None
    stop_reason: str | None
    failure_code: str | None
    archived_at: str | None


class RunListRunsAnswer(BaseModel):
    """``GET /api/v1/runs`` (``api/v1/endpoints/runs_index.py:list_runs``)."""

    runs: list[RunRow]
    total: int
    limit: int
    offset: int
    include_archived: bool
    statuses: list[str]
    run_types: list[str]


# ---------------------------------------------------------------------------- shared blocks


class RunCadences(BaseModel):
    """The four cadences (``schemas/run_definition.py:Cadences``, dumped as JSON)."""

    test_every: int
    retention_every: int
    save_every: int | None
    refresh_every: int | None


class RunByteRange(BaseModel):
    """A low and a high byte bound (``services/footprint.py:Footprint.payload``)."""

    low: int
    high: int


class RunFootprint(BaseModel):
    """The projected artifact footprint (``services/footprint.py:Footprint.payload``)."""

    checkpoints: int
    bytes_per_checkpoint: int
    trainer_state_bytes_per_save: RunByteRange
    total_bytes: RunByteRange
    free_disk_bytes: int | None
    root: str | None
    exceeds_free_disk: bool | None
    note: str


# ---------------------------------------------------------------------------- runs (GET one)


class RunHeartbeat(BaseModel):
    """The liveness block (``workers/heartbeat.py:liveness_of``)."""

    heartbeat_at: str | None
    age_seconds: float | None
    longest_gap_seconds: int | None
    lapse_threshold_seconds: int
    lapsed: bool


class RunAttachment(BaseModel):
    """Where an attached run attached (``api/v1/endpoints/runs.py:_attachment_block``)."""

    serving_base_url: str
    resident_model: str | None
    weights_state: str | None
    holder: str | None
    stale: bool | None
    attached: bool | None
    observed_at: str | None


class RunGetRunAnswer(BaseModel):
    """``GET /api/v1/runs/{run_id}`` (``api/v1/endpoints/runs.py:run_payload``, ``_loop_block``,
    ``get_run``)."""

    run_id: str
    status: str
    run_type: str
    last_step: int | None
    cancel_requested_at: str | None
    failure_code: str | None
    failure_message: str | None
    heartbeat: RunHeartbeat
    attachment: RunAttachment | None
    acquisition_mode: str | None
    archived_at: str | None
    stop_reason: str | None
    stop_detail: str | None
    all_fired: list[str]
    round: int | None
    stop_conditions: dict[str, dict[str, JsonValue]] | None = free_form(
        "each enabled stop condition's parameters, by condition name, as the run's definition "
        "submitted them; each condition takes its own parameters"
    )
    cadences: RunCadences | None
    retention_exposure_steps: int | None
    footprint: RunFootprint | None
    seed: int | None
    domain: str | None
    next_steps: list[NextStep]


# ---------------------------------------------------------------------------- runs_review


class RunReviewCounts(BaseModel):
    """Posted, decided and pending (``services/review/binding.py:ReviewCounts.as_dict``)."""

    posted: int
    decided: int
    pending: int


class RunGetRunReviewAnswer(BaseModel):
    """``GET /api/v1/runs/{run_id}/review`` (``services/review/binding.py:review_payload``)."""

    run_id: str
    queue_id: str
    question: str
    label_set: list[str]
    counts: RunReviewCounts
    summary: str
    read_back_cursor: str | None
    queue_owner: str


# ---------------------------------------------------------------------------- runs_acquisition


class RunAcquisitionSource(BaseModel):
    """The model source (``services/acquisition.py:_source_summary``)."""

    path: str | None
    repo: str | None
    revision_requested: str | None
    revision_resolved: str | None
    architecture: str | None


class RunAcquisition(BaseModel):
    """How the run acquires its model (``services/acquisition.py:acquisition_summary``)."""

    mode: str
    source: RunAcquisitionSource
    mode_available: bool
    mode_unavailable_reason: str | None
    seed: int | None


class RunOptimizer(BaseModel):
    """The optimizer identity (``services/acquisition.py:acquisition_summary``)."""

    name: str
    state_precision: str
    config_hash: str


class RunConfigAnswer(BaseModel):
    """R-02.13's configuration (``training/run_config.py:RunConfig``, dumped as JSON)."""

    learning_rate: float
    warmup_steps: int
    batch_size: int
    gradient_accumulation: int
    optimizer_state_precision: str
    seed: int | None


class RunGpu(BaseModel):
    """The device asked for and the one resolved
    (``services/acquisition.py:acquisition_summary``)."""

    request: str
    uuid: str | None
    name: str | None


class RunSubmitRunAnswer(BaseModel):
    """``POST /api/v1/runs`` (``services/acquisition.py:acquisition_summary``)."""

    run_id: str
    acquisition: RunAcquisition
    optimizer: RunOptimizer
    run_config: RunConfigAnswer
    gpu: RunGpu


class RunCancelRunAnswer(BaseModel):
    """``POST /api/v1/runs/{run_id}/cancel``
    (``api/v1/endpoints/runs_acquisition.py:cancel_run``)."""

    run_id: str
    cancel_pending: bool
    cancel_requested_at: str | None
    collapsed: bool


class StepBatchSourceHashes(BaseModel):
    """A step's batch provenance (``training/run_log.py:_batch_hashes``)."""

    batch_sha256: str


class StepRow(BaseModel):
    """One ``mf_steps`` row (``api/v1/endpoints/runs_acquisition.py:_step_payload``)."""

    step: int
    loss: float | None
    grad_norm: float | None
    learning_rate: float | None
    wall_time_ms: int | None
    batch_source_hashes: StepBatchSourceHashes | None
    quiesce_hold_ms: float | None
    commit_duration_ms: float | None


class RunListStepsAnswer(BaseModel):
    """``GET /api/v1/runs/{run_id}/steps`` (``api/v1/endpoints/runs_acquisition.py:list_steps``)."""

    run_id: str
    steps: list[StepRow]
    count: int
    limit: int
    next_before_step: int | None


class RunAcquisitionModeRow(BaseModel):
    """One acquisition mode (``training/acquisition/__init__.py:modes``)."""

    mode: str
    available: bool
    default: bool
    reason: str | None
    message: str | None
    capabilities_unavailable: list[str]
    sources: list[str]


class RunModelSourcesAnswer(BaseModel):
    """``GET /api/v1/models/sources`` (``api/v1/endpoints/runs_acquisition.py:model_sources``)."""

    modes: list[RunAcquisitionModeRow]
    default: str
    model_roots: list[str]


# -------------------------------------------------------------------------- the resolved definition


class RunPackUnresolvedSlot(BaseModel):
    """A filled slot whose part is absent (``packs/validation.py:PackValidity.payload``)."""

    slot: str
    reference: str
    reason: str


class RunPackUnavailableSlot(BaseModel):
    """A filled slot whose service is absent (``packs/validation.py:PackValidity.payload``)."""

    slot: str
    reference: str
    service: str | None
    reason: str


class RunPackValidity(BaseModel):
    """A pack version's validity (``packs/validation.py:PackValidity.payload``)."""

    state: str
    missing_slots: list[str]
    unresolved: list[RunPackUnresolvedSlot]
    service_unavailable: list[RunPackUnavailableSlot]
    retention_floor_missing: bool


class RunPartContractReference(BaseModel):
    """A miDataworks part reference (``packs/references.py:ContractPartRef.as_contract_dict``,
    ``services/parts/contract_validator.py:ContractReference.as_dict``)."""

    type: str
    name: str
    version: str


class RunPartExternalReference(BaseModel):
    """An external part (``packs/references.py:ExternalPartRef.as_external_dict``)."""

    source: str
    content_sha256: str
    retrieved_at: str


class RunResolvedPart(BaseModel):
    """One part a run draws on: a pack's ``PartReference`` element
    (``packs/records.py:PartReferenceRecord.element``) or a loose part
    (``services/run_definition.py:_live_parts``)."""

    kind: str
    contract_reference: RunPartContractReference | None = may_be_absent(
        description="A pack's part: its miDataworks reference, or null for an external part."
    )
    external: RunPartExternalReference | None = may_be_absent(
        description="A pack's part: its external origin, or null for a miDataworks part."
    )
    part_id: str | None = may_be_absent(description="A loose part: its part id.")
    reference: RunPartContractReference | None = may_be_absent(
        description="A loose part: its contract reference, or null."
    )


class RunResolvedPack(BaseModel):
    """The pack a run was gated through (``packs/gate.py:ResolvedPack.payload``)."""

    pack_id: str
    version: str
    domain: str | None
    validity: RunPackValidity
    retention_floor: float | None
    parts: list[RunResolvedPart]


class RunResolvedParts(BaseModel):
    """What the run's parts resolved to (``services/run_definition.py:_live_parts``)."""

    source: str
    parts: list[RunResolvedPart]


class RunRefreshBlock(BaseModel):
    """The refresh rule, recorded (``services/run_definition.py:_validate_loop``)."""

    mode: str
    rule_sha256: str
    refresh_every: int | None


class RunResolvedLoop(BaseModel):
    """The loop's cadences, conditions and refresh
    (``services/run_definition.py:_validate_loop``)."""

    cadences: RunCadences
    stop_conditions: dict[str, dict[str, JsonValue]] = free_form(
        "each enabled stop condition's parameters, by condition name, as the definition "
        "submitted them; each condition takes its own parameters"
    )
    stop_conditions_sha256: str
    refresh: RunRefreshBlock | None
    retention_exposure_steps: int


class RunSubmissionGuards(BaseModel):
    """F4's submission guards (``services/run_definition.py:_check_submission_guards``)."""

    checked: list[str]
    retention_guard_deferred: str | None = may_be_absent(
        description="The deferred retention guard's code, sent only when it was deferred."
    )


class RunProviderIdentity(BaseModel):
    """A provider identity (``providers/base.py:ProviderIdentity``, dumped as JSON)."""

    kind: str
    model: str
    base_url: str | None
    engine_version: str | None


class RunJudgeCheck(BaseModel):
    """R-02.11's self-judging check (``services/run_definition.py:_check_judge_identity``)."""

    compared: list[str]
    run_target: RunProviderIdentity | None
    caveats: list[str]
    unchecked: list[str]


class RunTokenizerPlan(BaseModel):
    """Which tokenizer a create run ships (``services/run_definition.py:_tokenizer_plan``)."""

    source: str
    repo: str | None = may_be_absent(description="A Hub tokenizer: its repository.")
    revision: str | None = may_be_absent(description="A Hub tokenizer: the revision asked for.")
    commit: str | None = may_be_absent(description="A Hub tokenizer: the commit it resolved to.")
    path: str | None = may_be_absent(description="A directory tokenizer: its path.")
    vocab_size: int | None


class RunResolvedBudget(BaseModel):
    """The budget step's resolution (``services/run_definition.py:_live_budget``)."""

    mode: str
    param_count: int | None
    gpu_uuid: str | None
    gpu_name: str | None
    free_vram_bytes: int | None
    mode_available: bool
    optimizer_config_hash: str
    tokenizer: RunTokenizerPlan | None


class RunMidataworksStatus(BaseModel):
    """miDataworks' integration status (``clients/availability.py:integration_status``)."""

    configured: bool
    setting: str
    code: str | None
    agent: str
    reachable: bool | None
    capabilities: dict[str, str]
    capabilities_unavailable: list[str]


class RunResolvedServices(BaseModel):
    """The optional services a definition needs (``services/run_definition.py:_live_services``)."""

    needed: list[str]
    midataworks: RunMidataworksStatus


class RunResumedFrom(BaseModel):
    """The checkpoint a continue resumes from (``services/run_continue.py:submit_continue``)."""

    run_id: str
    step: int
    name: str
    weights_sha256: str
    manifest_file_sha256: str
    directory: str


class RunResumeOptimizer(BaseModel):
    """The trainer-state decision (``services/run_continue.py:submit_continue``)."""

    state: str
    reason: str


class RunSupersededSteps(BaseModel):
    """The steps a continue writes again (``services/run_continue.py:submit_continue``)."""

    first: int
    last: int


class RunPreviousAttempt(BaseModel):
    """The attempt a continue reopened (``training/run_state.py:reopen_for_continue``)."""

    status: str
    failure_code: str | None
    failure_message: str | None
    stop_reason: str | None
    finished_at: str | None
    last_step: int | None


class RunResumeRecord(BaseModel):
    """A continue's ``resume`` record (``services/run_continue.py:submit_continue``)."""

    resumed_from: RunResumedFrom
    optimizer: RunResumeOptimizer
    superseded_steps: RunSupersededSteps | None
    previous_attempt: RunPreviousAttempt


class RunResolvedDefinition(BaseModel):
    """The validator's resolved definition (``services/run_definition.py:validate``; on the
    create path also ``services/run_launch.py:launch``). A block is absent when its step failed
    or could not run."""

    pack: RunResolvedPack | None = may_be_absent(
        description="The pack gate's resolution; null for a loose-parts run."
    )
    parts: RunResolvedParts | None = may_be_absent()
    loop: RunResolvedLoop | None = may_be_absent()
    submission_guards: RunSubmissionGuards | None = may_be_absent()
    judge: RunJudgeCheck | None = may_be_absent(
        description="Sent only when the definition names a judge provider."
    )
    budget: RunResolvedBudget | None = may_be_absent()
    services: RunResolvedServices | None = may_be_absent()
    seed: int
    domain: str | None
    run_type: str
    submitted_sha256: str
    footprint: RunFootprint | None = may_be_absent(
        description="The create path's projection (services/run_launch.py:launch)."
    )
    scoring: dict[str, JsonValue] | None = may_be_absent(
        description="The create path's scoring block (services/run_launch.py:_scoring_block).",
        json_schema_extra={
            FREE_FORM: "the definition's own scoring block as submitted, plus the pack's "
            "retention floor; the scorer reads it and its keys follow the submitted configuration"
        },
    )
    resume: RunResumeRecord | None = may_be_absent(
        description="A submitted continue's record (services/run_continue.py:submit_continue)."
    )


# ---------------------------------------------------------------------------- runs_loop


class RunValidationFailure(BaseModel):
    """One collected failure (``services/run_definition.py:Failure.payload``)."""

    step: str
    code: str
    message: str
    details: dict[str, JsonValue] = free_form(
        "the owning feature's refusal details, unchanged; each refusal code carries its own facts"
    )


class RunValidationSkipped(BaseModel):
    """A step the validator could not run (``services/run_definition.py:Skipped.payload``)."""

    step: str
    status: str
    waited_on: list[str]
    message: str


class RunValidateDefinitionAnswer(BaseModel):
    """``POST /api/v1/runs/validate`` (``services/run_definition.py:Outcome.payload``,
    ``services/run_launch.py:validate_only``,
    ``api/v1/endpoints/runs_loop.py:validate_definition``).
    """

    ok: bool
    may_be_more_to_fix: bool
    steps_checked: list[str]
    failures: list[RunValidationFailure]
    skipped: list[RunValidationSkipped]
    resolved: RunResolvedDefinition
    footprint: RunFootprint
    next_steps: list[NextStep]


class RunDefineRunAnswer(BaseModel):
    """``POST /api/v1/runs/define`` (``services/run_launch.py:launch``)."""

    run_id: str
    status: str
    definition_sha256: str
    resolved: RunResolvedDefinition
    footprint: RunFootprint
    may_be_more_to_fix: bool


class RunReadRunDefinitionAnswer(BaseModel):
    """``GET /api/v1/runs/{run_id}/definition`` (``services/run_launch.py:read_definition``)."""

    run_id: str
    source: str
    submitted: dict[str, JsonValue] | None = free_form(
        "the stored request body (a RunDefinition) as the run log or the index recorded it"
    )
    resolved: dict[str, JsonValue] | None = free_form(
        "the resolved definition as the run log's definition block (step-record-v1, where "
        "`resolved` is free-form) or the index recorded it, by whichever release wrote it"
    )
    definition_sha256: str | None
    index_agrees: bool


class RunStopConditionValues(BaseModel):
    """A condition's structured figures (``Verdict.values`` of each condition in
    ``training/stop/``); each condition sends its own subset of these keys."""

    completed: int | None = may_be_absent(description="step_count")
    total: int | None = may_be_absent(description="step_count")
    headroom: float | None = may_be_absent(description="step_count, test_threshold")
    score: float | None = may_be_absent(description="test_threshold, retention_floor")
    threshold: float | None = may_be_absent(description="test_threshold")
    n: int | None = may_be_absent(description="test_threshold, retention_floor")
    provider: str | None = may_be_absent(description="test_threshold, retention_floor")
    aggregation: str | None = may_be_absent(description="test_threshold, retention_floor")
    set: str | None = may_be_absent(description="test_threshold, retention_floor")
    breached: bool | None = may_be_absent(description="retention_floor")
    floor: float | None = may_be_absent(description="retention_floor")
    floor_layer: str | None = may_be_absent(description="retention_floor")
    exposure_steps: int | None = may_be_absent(description="retention_floor")
    requested: bool | None = may_be_absent(description="operator_stop")
    requested_at: str | None = may_be_absent(description="operator_stop")
    metric: str | None = may_be_absent(description="loss_plateau")
    window: int | None = may_be_absent(description="loss_plateau")
    window_filled: int | None = may_be_absent(description="loss_plateau")
    min_steps: int | None = may_be_absent(description="loss_plateau")
    min_relative_improvement: float | None = may_be_absent(description="loss_plateau")
    improvement: float | None = may_be_absent(description="loss_plateau")
    round: int | None = may_be_absent(description="loss_plateau")
    health: dict[str, JsonValue] | None = may_be_absent(
        description="loss_plateau",
        json_schema_extra={
            FREE_FORM: "the selectors' and graders' health signals over the window, by signal "
            "name, as the loop observed them"
        },
    )


class RunStopConditionEvaluation(BaseModel):
    """One condition's verdict (``training/stop/base.py:Verdict.as_evaluation`` plus ``values``)."""

    condition: str
    triggered: bool
    observed_at_step: int
    detail: str | None
    values: RunStopConditionValues


class RunReadStopConditionsAnswer(BaseModel):
    """``GET /api/v1/runs/{run_id}/stop-conditions``
    (``api/v1/endpoints/runs_loop.py:read_stop_conditions``)."""

    run_id: str
    status: str
    step: int
    retention_exposure_steps: int | None
    conditions: list[RunStopConditionEvaluation]
    precedence: list[str]
    all_fired: list[str]


class RunListBoundariesAnswer(BaseModel):
    """``GET /api/v1/runs/{run_id}/boundaries``
    (``api/v1/endpoints/runs_loop.py:list_boundaries``)."""

    run_id: str
    boundaries: list[BoundaryBlockPart]
    count: int
    limit: int
    total: int


class RunStopConditionEntry(BaseModel):
    """One registered stop condition (``api/v1/endpoints/runs_loop.py:list_stop_conditions``)."""

    condition: str
    configured: bool
    required_parameters: list[str]
    precedence_rank: int
    summary: str
    metrics: list[str] | None = may_be_absent(description="Sent for loss_plateau only.")


class RunListStopConditionsAnswer(BaseModel):
    """``GET /api/v1/stop-conditions`` (``api/v1/endpoints/runs_loop.py:list_stop_conditions``)."""

    conditions: list[RunStopConditionEntry]
    count: int
    closed: bool
    precedence: list[str]
    minimum_enabled: int


class RunRefreshModeEntry(BaseModel):
    """One registered refresh mode (``api/v1/endpoints/runs_loop.py:list_refresh_modes``)."""

    mode: str
    applies_to: str
    summary: str


class RunListRefreshModesAnswer(BaseModel):
    """``GET /api/v1/refresh-modes`` (``api/v1/endpoints/runs_loop.py:list_refresh_modes``)."""

    modes: list[RunRefreshModeEntry]
    count: int
    closed: bool
    applies_to: str
    requires_cadence: str


# ---------------------------------------------------------------------------- scoring


class ScoreProviderRow(BaseModel):
    """One provider registry row (``providers/registry.py:ProviderRow.payload``)."""

    name: str
    kind: str
    model: str | None
    base_url: str | None
    capabilities: list[str]
    available: bool
    unavailable_reason: str | None
    judge_eligible: bool


class ScoreListProvidersAnswer(BaseModel):
    """``GET /api/v1/providers`` (``api/v1/endpoints/scoring.py:list_providers``)."""

    providers: list[ScoreProviderRow]
    judge_eligible: list[str]
    capability_vocabulary: list[str]


class ScoreProviderIdentity(BaseModel):
    """A provider identity (``providers/base.py:ProviderIdentity``, dumped as JSON)."""

    kind: str
    model: str
    base_url: str | None
    engine_version: str | None


class ScoreCheckProviderAnswer(BaseModel):
    """``POST /api/v1/providers/{name}/check`` (``api/v1/endpoints/scoring.py:check_provider``)."""

    name: str
    identity: ScoreProviderIdentity
    capabilities: list[str]
    available: bool
    completion_chars: int


class ScoreDecode(BaseModel):
    """The decode parameters sent
    (``services/scoring/evaluation_block.py:decode_params_payload``)."""

    temperature: float
    top_p: float
    max_tokens: int
    stop: list[str] | None = may_be_absent(description="Sent only when stop strings were set.")


class ScoreElement(BaseModel):
    """One ``mf_scores`` row (``api/v1/endpoints/scoring.py:_score_payload``)."""

    step: int
    kind: str
    set: str
    score: float
    n: int
    floor: float | None
    floor_layer: str
    breached: bool
    provider: str
    provider_kind: str
    provider_model: str
    aggregation: str
    decode: ScoreDecode
    subsample_seed: int | None
    split_markers: list[str] | None
    reproducible: bool
    not_reproducible_reason: str | None
    out_of_cadence: bool
    provider_discontinuity: bool
    message: str


class ScoreRunScoresAnswer(BaseModel):
    """``GET /api/v1/runs/{run_id}/scores`` (``api/v1/endpoints/scoring.py:run_scores``)."""

    run_id: str
    scores: list[ScoreElement]
    test_every: int
    retention_every: int
    exposure_window_steps: int
    provider_discontinuities: list[int]


class ScoreItem(BaseModel):
    """One retained per-prompt result (``api/v1/endpoints/scoring.py:run_scores_at_step``)."""

    prompt_index: int
    verdict: str | None
    scalar: float | None
    reason_code: str | None
    completion_excerpt: str | None


class ScoreStepElement(ScoreElement):
    """One score with its retained items (``api/v1/endpoints/scoring.py:run_scores_at_step``)."""

    items_retained: int
    items: list[ScoreItem]


class ScoreRunScoresAtStepAnswer(BaseModel):
    """``GET /api/v1/runs/{run_id}/scores/{step}``
    (``api/v1/endpoints/scoring.py:run_scores_at_step``)."""

    run_id: str
    step: int
    scores: list[ScoreStepElement]


class ScoreEvaluateNowAnswer(BaseModel):
    """``POST /api/v1/runs/{run_id}/evaluate`` (``api/v1/endpoints/scoring.py:evaluate_now``)."""

    run_id: str
    step: int
    out_of_cadence: bool
    evaluation: EvaluationBlockPart = Field(
        description="The evaluation block services/scoring/record.py:write_pass wrote."
    )
