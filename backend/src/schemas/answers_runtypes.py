"""Answer models for run types, the budget preview, adapters and merges (feature 021, P3).

Documentation only: each model describes what ``api/v1/endpoints/runtypes.py`` answers today
and is declared on the route's ``responses``; no route validates against it (MF9).
"""

from pydantic import BaseModel

from schemas.answers_common import JsonValue, NextStep, free_form, may_be_absent
from schemas.answers_documents import (
    AdapterManifestDocument,
    OptimizerIdentityPart,
    ScoreRecordPart,
)

# ================================================================== run types


class RunTypeEntry(BaseModel):
    """One registered run type (``services/runtypes_service.py:run_types``)."""

    name: str
    stored_as: str
    changes: str


class RunTypeListRunTypesAnswer(BaseModel):
    """``GET /run-types`` (``api/v1/endpoints/runtypes.py:list_run_types``)."""

    run_types: list[RunTypeEntry]
    count: int


# ============================================================= budget preview


class RunTypeBudgetEstimate(BaseModel):
    """``estimate``: the projection's record (``training/budget_guard.py:BudgetEstimate.as_record``
    for ``change``/``create``, with ``run_type`` added in ``runtypes_service.estimate_budget``;
    ``training/runtypes/adapter_budget.py:AdapterBudgetEstimate.as_record`` for ``adapter``)."""

    run_type: str
    basis: str
    #: Full-parameter runs only (``change``, ``create``).
    param_count: int | None = may_be_absent()
    #: Adapter runs only.
    base_param_count: int | None = may_be_absent()
    #: Adapter runs only.
    adapter_param_count: int | None = may_be_absent()
    #: Adapter runs only.
    adapter_param_basis: str | None = may_be_absent()
    weights_gib: float
    gradients_gib: float
    fp32_master_gib: float
    optimizer_states_gib: float
    activations_gib_low: float
    activations_gib_high: float
    total_gib_low: float
    total_gib_high: float


class RunTypeBudgetTerm(BaseModel):
    """One term of the estimate, in bytes (``services/runtypes_service.py:estimate_budget``)."""

    name: str
    bytes: int


class RunTypeBudgetSource(BaseModel):
    """Where the model's size was read (``services/budget_source.py:SizedSource.as_record``)."""

    kind: str
    param_count: int | None
    from_config: bool
    #: A ``create`` source with a config file (``size_of``).
    config_path: str | None = may_be_absent()
    #: A local directory or checkpoint source (``_directory``).
    path: str | None = may_be_absent()
    #: A Hub source (``_hub``).
    repo: str | None = may_be_absent()
    #: A Hub source: the revision asked for.
    revision: str | None = may_be_absent()
    #: A Hub source: the commit that revision resolved to.
    revision_resolved: str | None = may_be_absent()


class RunTypeBudgetCard(BaseModel):
    """One card the run could land on, with its own verdict
    (``services/runtypes_service.py:estimate_budget``)."""

    uuid: str
    name: str
    free_bytes: int | None
    verdict: str


class RunTypeEstimateBudgetAnswer(BaseModel):
    """``POST /budget/estimate`` (``api/v1/endpoints/runtypes.py:estimate_budget``, over
    ``_estimate`` and ``services/runtypes_service.py:estimate_budget``)."""

    verdict: str
    estimate: RunTypeBudgetEstimate
    estimate_bytes: int
    estimate_low_bytes: int
    terms: list[RunTypeBudgetTerm]
    source: RunTypeBudgetSource | None
    cards: list[RunTypeBudgetCard]
    free_bytes: int | None
    free_gib: float | None
    claimable_bytes: int | None
    claimable_gib: float | None
    safety_fraction: float
    gpu_uuid: str | None
    gpu_name: str | None
    #: Sent only when the verdict is ``fits``.
    next_steps: list[NextStep] | None = may_be_absent()


# ==================================================================== adapters


class AdapterRow(BaseModel):
    """One indexed adapter (``services/reindex.py:_adapter_payload``)."""

    name: str
    run_id: str
    step: int
    domain: str | None
    domain_state: str
    domain_display: str
    base_name: str
    base_manifest_sha256: str | None
    weights_sha256: str
    manifest_sha256: str
    adapter_config: dict[str, JsonValue] | None = free_form(
        "the adapter manifest's `adapter` object, stored opaquely so an unrecognised scaling "
        "rule reaches its refusal: deliberately not checked against the manifest's AdapterConfig"
    )
    optimizer: OptimizerIdentityPart | None
    parts: list[dict[str, JsonValue]] | None = free_form(
        "the adapter manifest's `parts` list as indexed; the manifest itself declares each item "
        "only as an object (a contract reference or an external part's provenance)"
    )
    test: ScoreRecordPart | None
    retention: ScoreRecordPart | None
    path: str
    created_at: str


class AdapterListAdaptersAnswer(BaseModel):
    """``GET /adapters`` (``api/v1/endpoints/runtypes.py:list_adapters``)."""

    adapters: list[AdapterRow]
    count: int


class AdapterTextDisagreement(BaseModel):
    """A text column that differs from the manifest
    (``services/runtypes_service.py:_index_disagreements``)."""

    manifest: str
    index: str


class AdapterStepDisagreement(BaseModel):
    """The step column, when it differs from the manifest
    (``services/runtypes_service.py:_index_disagreements``)."""

    manifest: int
    index: int


class AdapterDisagreements(BaseModel):
    """The indexed columns that differ from the manifest; each key is present only when that
    column differs, and the map is empty when the row is absent or agrees
    (``services/runtypes_service.py:_index_disagreements``)."""

    run_id: AdapterTextDisagreement | None = may_be_absent()
    step: AdapterStepDisagreement | None = may_be_absent()
    base_name: AdapterTextDisagreement | None = may_be_absent()
    weights_sha256: AdapterTextDisagreement | None = may_be_absent()


class AdapterReadAdapterAnswer(BaseModel):
    """``GET /adapters/{name}`` (``services/runtypes_service.py:read_adapter``)."""

    manifest: AdapterManifestDocument
    path: str
    indexed: bool
    index_disagrees: bool
    disagreements: AdapterDisagreements


class AdapterMergeAdmitted(BaseModel):
    """What the merge preflight admitted (``services/runtypes_service.py:preflight_merge``)."""

    name: str
    step: int
    path: str
    scaling: str | None
    base_checkpoint: str
    base_checkpoint_path: str


class AdapterMergeAdapterAnswer(BaseModel):
    """``POST /adapters/{name}/merge``, once admitted
    (``api/v1/endpoints/runtypes.py:merge_adapter``)."""

    status: str
    adapter: AdapterMergeAdmitted
    run_id: str
    step: int
    note: str


class AdapterMergeRecord(BaseModel):
    """One ``mf_merges`` row (``services/merge_record.py:listing``, the table in
    ``training/runtypes/tables.py:MERGES``)."""

    id: int
    adapter_name: str
    adapter_step: int
    adapter_manifest_sha256: str
    base_checkpoint_name: str
    result_checkpoint_name: str | None
    outcome: str
    refusal_code: str | None
    created_at: str


class AdapterListMergesAnswer(BaseModel):
    """``GET /merges`` (``api/v1/endpoints/runtypes.py:list_merges``)."""

    merges: list[AdapterMergeRecord]
    count: int
