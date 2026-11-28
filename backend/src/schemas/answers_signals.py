"""Answer models for signal sources, a run's batches and the miDataworks integration (021, P3).

Documentation only: each model describes what ``api/v1/endpoints/signal_sources.py``,
``signal_batches.py`` and ``midataworks.py`` answer today and is declared on the route's
``responses``; no route validates against it (MF9).
"""

from pydantic import BaseModel

from schemas.answers_common import JsonValue, free_form
from schemas.answers_documents import BatchRecordDocument, ExternalProvenancePart

# ============================================================= signal sources


class SignalSourceRequires(BaseModel):
    """What a source needs (``services/signal_source_service.py:listing``)."""

    part_kind: str | None
    selector: bool
    grader: bool
    generation_provider: bool


class SignalSourceRow(BaseModel):
    """One registered source (``services/signal_source_service.py:listing``)."""

    name: str
    part_kind: str | None
    objective: str
    requires: SignalSourceRequires
    available: bool
    unavailable_reason: str | None
    summary: str


class SignalListSignalSourcesAnswer(BaseModel):
    """``GET /signal-sources`` (``api/v1/endpoints/signal_sources.py:list_signal_sources``)."""

    sources: list[SignalSourceRow]
    count: int
    available: int


class SignalContractReference(BaseModel):
    """A part's miDataworks contract reference (``services/parts/index.py:part_payload``)."""

    type: str
    name: str
    version: str


class SignalCaveat(BaseModel):
    """A part's caveat, verbatim (``services/parts/contract_validator.py:Caveat.as_dict``)."""

    code: str
    severity: str
    message: str
    detail: dict[str, JsonValue] = free_form(
        "the caveat's own facts as miDataworks published them; each caveat code carries its own"
    )


class SignalPreflightPart(BaseModel):
    """One resolved part (``services/signal_source_service.py:_part_summary``)."""

    part_id: str
    kind: str
    contract_reference: SignalContractReference | None
    external: bool
    splits: list[str]
    rows: int
    #: ``role -> declared semantic``; the roles are the source's own
    #: (``training/signals/base.py:ColumnBinding.as_record``).
    binding: dict[str, str]
    caveats: list[SignalCaveat]
    licence_classes: list[str]


class SignalConfigSummary(BaseModel):
    """A resolved selector or grader configuration
    (``services/signal_source_service.py:_config_summary``)."""

    kind: str
    name: str
    config_sha256: str
    provenance: str
    caveats: list[str]


class SignalPreflightSignalSourceAnswer(BaseModel):
    """``POST /signal-sources/preflight`` (``services/signal_source_service.py:preflight``)."""

    source: str
    objective: str
    parts: list[SignalPreflightPart]
    selector: SignalConfigSummary | None
    grader: SignalConfigSummary | None
    batch_size: int
    group_size: int | None
    groups_per_batch: int | None


# ===================================================================== batches


class BatchContractReference(BaseModel):
    """A batch part's contract reference (``training/signals/parts_reader.py:part_reference``)."""

    type: str
    name: str
    version: str


class BatchPartReference(BaseModel):
    """One part a batch drew from: the gate contract's ``PartReference``
    (``training/signals/parts_reader.py:part_reference``, copied to ``mf_batches.part_refs``)."""

    kind: str
    contract_reference: BatchContractReference | None
    external: ExternalProvenancePart | None


class BatchVerdicts(BaseModel):
    """The three verdict counts (``api/v1/endpoints/signal_batches.py:_row_payload``)."""

    accept: int
    reject: int
    undecided: int


class BatchSelectionHealth(BaseModel):
    """The selection health figures (``api/v1/endpoints/signal_batches.py:_row_payload``)."""

    candidates_generated: int
    candidates_admitted: int
    acceptance_rate: float
    exhausted: bool


class BatchRow(BaseModel):
    """One ``mf_batches`` row on the wire (``api/v1/endpoints/signal_batches.py:_row_payload``)."""

    batch_sha256: str
    step: int | None
    source: str
    resolved_model: str | None
    parts: list[BatchPartReference]
    prompt_ids: list[str]
    n_prompts: int
    verdicts: BatchVerdicts
    selection_health: BatchSelectionHealth
    selector_config_sha256: str | None
    grader_config_sha256: str | None
    created_at: str


class BatchListBatchesAnswer(BaseModel):
    """``GET /runs/{run_id}/batches`` (``api/v1/endpoints/signal_batches.py:list_batches``)."""

    run_id: str
    batches: list[BatchRow]
    count: int
    limit: int
    next_before: str | None


class BatchGetBatchAnswer(BaseModel):
    """``GET /runs/{run_id}/batches/{batch_sha256}``
    (``api/v1/endpoints/signal_batches.py:get_batch``)."""

    run_id: str
    step: int | None
    ts: str
    batch: BatchRecordDocument


# ================================================================= miDataworks


class DataworksCapabilities(BaseModel):
    """Each capability the integration gates, described
    (``clients/availability.py:integration_status`` over ``clients/capabilities.py:CAPABILITIES``).
    """

    part_import_http: str
    available_parts_listing: str
    human_review_selector: str


class DataworksGetStatusAnswer(BaseModel):
    """``GET /midataworks/status`` (``clients/availability.py:integration_status``)."""

    configured: bool
    setting: str
    code: str | None
    agent: str
    reachable: bool | None
    capabilities: DataworksCapabilities
    capabilities_unavailable: list[str]


class DataworksOfferedPart(BaseModel):
    """One miForge-set export miDataworks offers; each value is read with ``.get`` from
    miDataworks' export record (``api/v1/endpoints/midataworks.py:available_parts``)."""

    export_id: str | None
    target: str | None
    version_id: str | None
    status: str | None
    miforge_set_kind: str | None
    manifest_sha256: str | None
    created_at: str | None


class DataworksAvailablePartsAnswer(BaseModel):
    """``GET /midataworks/available-parts`` (``endpoints/midataworks.py:available_parts``)."""

    items: list[DataworksOfferedPart]
    total: int
    source: str
    cached: bool
