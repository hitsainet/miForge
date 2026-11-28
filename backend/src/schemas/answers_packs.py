"""The answer models of the pack, pack-draft, pack-part-reference and part routes (feature 021).

Documentation only (MF9): the routes still return their dicts, and these models describe exactly
what those dicts hold. Each class names the function that builds the shape it describes.
"""

from pydantic import BaseModel, Field

from .answers_common import JsonValue, NextStep, free_form, may_be_absent

# ------------------------------------------------------------------------ references and slots


class PartContractReference(BaseModel):
    """miDataworks' ``{type, name, version}`` (``packs/references.py:as_contract_dict``).

    The same three keys come from ``services/parts/contract_validator.py:ContractReference.as_dict``
    and ``services/parts/index.py:part_payload``.
    """

    type: str
    name: str
    version: str


class PackExternalProvenance(BaseModel):
    """An external part's provenance (``packs/references.py:ExternalPartRef.as_external_dict``)."""

    source: str = Field(description="Where the part came from, as the operator gave it.")
    content_sha256: str
    retrieved_at: str = Field(description="ISO 8601 time the part was taken in.")


class PackSlotDocument(BaseModel):
    """One filled slot as the pack document carries it (``packs/definition.py:slot_document``)."""

    contract_reference: PartContractReference | None
    external: PackExternalProvenance | None
    floor: float | None


class PartRefContract(BaseModel):
    """A miDataworks part's pinnable reference (``packs/references.py:ContractPartRef``)."""

    source: str = Field(description="midataworks")
    type: str
    name: str
    version: str


class PartRefExternal(BaseModel):
    """An external part's pinnable reference (``packs/references.py:ExternalPartRef``)."""

    source: str = Field(description="external")
    origin: str
    content_sha256: str
    taken_in_at: str = Field(description="ISO 8601 time the part was taken in.")


# ------------------------------------------------------------------------------- validity


class PackUnresolvedSlot(BaseModel):
    """One unresolved slot (``packs/validation.py:PackValidity.payload``, ``unresolved``)."""

    slot: str
    reference: str
    reason: str


class PackUnavailableSlot(BaseModel):
    """One slot an absent service left unresolved (``packs/validation.py:PackValidity.payload``)."""

    slot: str
    reference: str
    service: str | None
    reason: str


class PackValidity(BaseModel):
    """A pack's validity verdict (``packs/validation.py:PackValidity.payload``)."""

    state: str = Field(description="valid | incomplete")
    missing_slots: list[str]
    unresolved: list[PackUnresolvedSlot]
    service_unavailable: list[PackUnavailableSlot]
    retention_floor_missing: bool


class PackGetValidityAnswer(PackValidity):
    """The validity answer (``api/v1/endpoints/packs.py:get_validity``)."""

    next_steps: list[NextStep] | None = may_be_absent(description="Sent only when state is valid.")


# ------------------------------------------------------------------------------- registry


class PackRegistryRow(BaseModel):
    """One pack version's registry line (``packs/registry.py:RegistryRow.payload``).

    The answer of ``get_pack_version``, ``create_pack``, ``import_pack`` and ``clone_pack``, which
    all return ``pack_service.get`` unchanged.
    """

    pack_id: str
    version: str
    name: str
    domain: str | None
    parts_filled: int
    parts_total: int
    prompt_count: int | None = Field(description="null when unknown, never 0.")
    runs_using: list[str]
    validity: PackValidity
    derived_from: str | None
    is_example: bool | None
    definition_sha256: str
    created_at: str


class PackLineageSlot(BaseModel):
    """One slot's lineage (``packs/registry.py:lineage``); an empty slot sends only three keys."""

    slot: str
    source: str | None = Field(description="midataworks | external; null for an empty slot.")
    state: str
    reference: PartContractReference | PackExternalProvenance | None = may_be_absent(
        description="Absent for an empty slot."
    )
    inner_form: str | None = may_be_absent(description="Absent for an empty slot.")
    display_form: str | None = may_be_absent(description="Absent for an empty slot.")
    addressed_through: str | None = may_be_absent(description="Absent for an empty slot.")
    floor: float | None = may_be_absent(description="Absent for an empty slot.")
    rows: int | None = may_be_absent(description="Absent for an empty slot.")
    service: str | None = may_be_absent(description="Absent for an empty slot.")


class PackGetLineageAnswer(BaseModel):
    """The lineage answer (``api/v1/endpoints/packs.py:get_lineage``)."""

    slots: list[PackLineageSlot]


# ------------------------------------------------------------------------- the dry-run


class PackProviderIdentity(BaseModel):
    """Who answered or judged (``providers/base.py:ProviderIdentity``, model_dump)."""

    kind: str
    model: str
    base_url: str | None
    engine_version: str | None


class PackGraderStats(BaseModel):
    """A batch's score distribution (``services/selectors/types.py:GraderStats``)."""

    n: int
    min: float
    max: float
    mean: float
    variance: float


class PackSelectionSignal(BaseModel):
    """One degenerate-selection signal (``services/selectors/types.py:SelectionSignal``)."""

    name: str
    message: str
    details: dict[str, float] = Field(
        description="The signal's counts and figures, by name (services/selectors/health.py)."
    )


class PackSelectionHealth(BaseModel):
    """F3's selection health record (``services/selectors/types.py:SelectionHealth``)."""

    n: int
    accepted: int
    rejected: int
    undecided: int
    acceptance_rate: float
    reason_histogram: dict[str, int]
    grader_stats: PackGraderStats | None
    signals: list[PackSelectionSignal]


class DraftDryRunVerdict(BaseModel):
    """One dry-run verdict (``packs/dryrun.py:DryRunResult.payload``, ``verdicts``)."""

    prompt: str
    completion: str
    verdict: str = Field(description="accept | reject | undecided")
    reason_code: str
    reason: str | None
    score: float | None
    admitted: bool


class DraftDryRunAnswer(BaseModel):
    """``POST /pack-drafts/{draft_id}/dry-run`` (``packs/dryrun.py:DryRunResult.payload``)."""

    provider: PackProviderIdentity
    judge: PackProviderIdentity | None
    config_sha256: str
    config_kind: str
    hash_provenance: str
    health: PackSelectionHealth
    verdicts: list[DraftDryRunVerdict]
    asynchronous_note: str | None
    advisory: bool


# ---------------------------------------------------------------------------------- drafts


class DraftSlots(BaseModel):
    """A draft's seven slots (``packs/authoring.py:Draft.payload``, ``slots``; ``SLOT_ORDER``)."""

    prompt_set: PackSlotDocument | None
    corpus: PackSlotDocument | None
    preference_pairs: PackSlotDocument | None
    selector_config: PackSlotDocument | None
    grader_config: PackSlotDocument | None
    test_set: PackSlotDocument | None
    retention_set: PackSlotDocument | None


class DraftRecord(BaseModel):
    """A draft (``packs/authoring.py:Draft.payload``; the listing serves the stored copy of it)."""

    draft_id: str
    name: str | None
    version: str
    domain: str | None
    current_step: int
    step_count: int
    step_states: dict[str, str] = Field(description="Step number -> todo | current | done.")
    slots: DraftSlots
    created_at: str
    updated_at: str
    last_dry_run: DraftDryRunAnswer | None = Field(
        description="The last dry-run, for display only (packs/authoring.py:record_dry_run)."
    )


class DraftCreateDraftAnswer(DraftRecord):
    """``POST /pack-drafts`` (``api/v1/endpoints/packs_drafts.py:create_draft``)."""

    next_steps: list[NextStep]


class DraftListDraftsAnswer(BaseModel):
    """``GET /pack-drafts`` (``api/v1/endpoints/packs_drafts.py:list_drafts``)."""

    drafts: list[DraftRecord]


class DraftGetDraftAnswer(DraftRecord):
    """``GET /pack-drafts/{draft_id}`` (``api/v1/endpoints/packs_drafts.py:get_draft``)."""

    subtitle: str


class DraftCommitStepAnswer(DraftGetDraftAnswer):
    """A committed step (``api/v1/endpoints/packs_drafts.py:commit_step``)."""

    next_steps: list[NextStep]


class DraftReviewDraftAnswer(BaseModel):
    """Step 7's review (``api/v1/endpoints/packs_drafts.py:review_draft``)."""

    draft: DraftRecord
    validity: PackValidity
    next_steps: list[NextStep]


class DraftCommitDraftAnswer(PackRegistryRow):
    """A committed draft (``api/v1/endpoints/packs_drafts.py:commit_draft``)."""

    next_steps: list[NextStep]


class DraftSamplePrompt(BaseModel):
    """One sampled row (``packs/samples.py:SlotSample.payload``, ``prompts``)."""

    prompt: str
    reference: str | None


class DraftSampleSlotAnswer(BaseModel):
    """``POST /pack-drafts/{draft_id}/samples/{slot}`` (``packs/samples.py:SlotSample.payload``)."""

    slot: str
    set: str = Field(description="The sampled set's inner rendering.")
    n: int
    population: int
    seed: int
    prompts: list[DraftSamplePrompt]


class DraftScoreSampleAnswer(DraftSampleSlotAnswer):
    """A scored sample (``packs/samples.py:ScoredSample.payload``)."""

    score: float
    provider: str
    aggregation: str
    floor: float | None
    floor_breached: bool | None = Field(description="null when no floor was proposed.")


# ----------------------------------------------------------------------------------- parts


class PartPinnableReferenceAnswer(BaseModel):
    """A pinnable reference (``api/v1/endpoints/packs_parts.py:pinnable_reference``)."""

    part_id: str
    ref: PartRefContract | PartRefExternal
    source: str = Field(description="midataworks | external")
    inner_form: str
    slot_document: PackSlotDocument


class PartCaveat(BaseModel):
    """A caveat, carried verbatim (``services/parts/contract_validator.py:Caveat.as_dict``)."""

    code: str
    severity: str
    message: str
    detail: dict[str, JsonValue] = free_form(
        "miDataworks' caveat detail, open in their dataset-version contract and carried verbatim"
    )


class PartLicence(BaseModel):
    """A licence block (``services/parts/contract_validator.py:DatasetVersion.licences``)."""

    displayed: str
    raw: str | list[str] | None
    origin: str
    licence_class: str
    table_version: int


class PartSplit(BaseModel):
    """One split of a part (``services/parts/index.py:part_payload``, ``splits``)."""

    name: str
    path: str
    rows: int
    bytes: int
    sha256: str
    logical_digest: str
    held_out: bool
    evaluation_only: bool
    label_counts: dict[str, int]
    training_eligible: bool


class PartRecord(BaseModel):
    """One part (``services/parts/index.py:part_payload`` plus the routes' ``archived_at``).

    The answer of ``get_part`` and each item of ``list_parts``.
    """

    id: str
    part_kind: str
    source: str
    contract_reference: PartContractReference | None
    rendered: str | None
    dataset: str | None
    version_id: str | None
    version_number: int | None
    recipe_sha256: str | None
    content_kind: str
    config_sha256: str | None
    manifest_sha256: str
    row_count: int
    licence_classes: list[str]
    licences: list[PartLicence]
    caveats: list[PartCaveat]
    external_provenance: dict[str, JsonValue] | None = free_form(
        "the external_provenance object the importer supplied, stored and served verbatim "
        "(services/parts/ingest.py checks only that it is non-empty)"
    )
    local_path: str
    imported_at: str | None
    splits: list[PartSplit]
    archived_at: str | None


class PartListPartsAnswer(BaseModel):
    """``GET /parts`` (``api/v1/endpoints/parts.py:list_parts``)."""

    items: list[PartRecord]
    total: int
    include_archived: bool


class PartImportPartAnswer(BaseModel):
    """``POST /parts/import`` with ``wait`` (``api/v1/endpoints/parts.py:import_part``, 200)."""

    part_id: str
    created: bool
    part_kind: str
    manifest_sha256: str
    splits: list[str]
    caveats: list[PartCaveat]
    licence_classes: list[str]


class PartImportPartQueuedAnswer(BaseModel):
    """``POST /parts/import`` without ``wait`` (``api/v1/endpoints/parts.py:import_part``, 202)."""

    job_id: str
    source: str
    locator: str
    poll: str


class PartValidateManifestAnswer(BaseModel):
    """``POST /parts/validate`` (``api/v1/endpoints/parts.py:validate_manifest``)."""

    valid: bool
    part_kind: str
    target_kind: str
    content_kind: str
    contract_reference: PartContractReference | None
    rendered: str | None
    splits: list[str]
    caveats: list[PartCaveat]
    licence_classes: list[str]


class PackListingDoc(BaseModel):
    """``api/v1/endpoints/packs.py:list_packs``: the documentation twin of ``PackListing``
    (021 FT2), whose rows are ``packs/registry.py:payloads`` (``row.payload()``, as
    ``pack_service.get``)."""

    packs: list[PackRegistryRow]
