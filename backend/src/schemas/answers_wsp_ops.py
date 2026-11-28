"""The answers of the WSP provider, validation, weight-operation and patch routes (feature 021, P3):
``api/v1/endpoints/wsp_ops.py``.

Documentation only (MF9): the routes still return their dicts (inspect its canonical bytes); these
models describe them. The frozen request is the run contract's ``FrozenRequest``
(``docs/schemas/miforge-wsp-run-v1.json``), checked by ``wsp.freeze.validate_frozen_request``.
"""

from pydantic import BaseModel, Field

from .answers_common import may_be_absent
from .answers_documents import WSP_PATCH
from .contract_tools import EXTERNAL

# ----------------------------------------------------------------------------- providers


class WspOpsProvider(BaseModel):
    """One provider (``wsp/providers/registry.py:ProviderDescriptor.public``)."""

    id: str
    version: str
    operations: list[str] = Field(description="Sorted: create, inspect, repair.")
    styles: list[str] = Field(description="Sorted: program, session.")
    available: bool = Field(description="Named in WSP_PROVIDERS_AVAILABLE.")


class WspOpsListProvidersAnswer(BaseModel):
    """``GET /wsp/providers`` (``wsp_ops.py:list_providers``)."""

    providers: list[WspOpsProvider]


# ----------------------------------------------------------------------------- run validation


class WspValidateBudgets(BaseModel):
    """The frozen ``budgets`` (``wsp/freeze.py:freeze``; the run contract's ``Budgets``)."""

    forward_passes: int
    tokens: int
    max_steps: int
    time_s: int
    cost_usd: float | None = Field(description="Null for a local endpoint unless the body set one.")


class WspValidateBundleRef(BaseModel):
    """The frozen ``bundle`` (the start body's, checked against the imported bundle)."""

    owner: str
    bundle_id: str
    bundle_sha256: str


class WspValidateParent(BaseModel):
    """The frozen ``parent`` (``wsp/freeze.py:freeze``; only a bundle today)."""

    kind: str
    ref: str


class WspValidateProvider(BaseModel):
    """The frozen ``provider`` (``wsp/freeze.py:freeze``)."""

    id: str
    version: str


class WspValidateEndpoint(BaseModel):
    """The frozen ``endpoint`` (``wsp/freeze.py:freeze``): the stored row's id, locality, model."""

    id: str
    locality: str = Field(description="local or external.")
    model: str


class WspValidateChangeLimits(BaseModel):
    """The frozen ``change_limits`` (``wsp/freeze.py:freeze``, defaults then the body's)."""

    max_changed_entries: int | None
    allowed_tensors: list[str] | None
    frozen_tensors: list[str]


class WspValidateObservationTool(BaseModel):
    """One frozen ``observation_tools[]`` entry (``wsp/freeze.py:_observation_tool``)."""

    name: str
    version: str
    cost: int


class WspValidateSupportSet(BaseModel):
    """The run contract's ``SupportSet`` (``freeze`` always sends null at a start)."""

    kind: str
    digest: str


class WspValidateFrozenRequest(BaseModel):
    """The frozen start request (``wsp/freeze.py:freeze``); the run contract's
    ``FrozenRequest``."""

    kind: str = Field(description="inspect, create or repair.")
    goal: str | None
    mode: str = Field(description="session or program at a start.")
    task: str
    bundle: WspValidateBundleRef
    sensitivity_declared: str
    parent: WspValidateParent
    chain: bool
    keep_weights: bool
    provider: WspValidateProvider
    endpoint: WspValidateEndpoint
    thinking: str | None = Field(description="The thinking in effect; null for the default.")
    budgets: WspValidateBudgets
    change_limits: WspValidateChangeLimits
    observation_tools: list[WspValidateObservationTool] = Field(description="Sorted by name.")
    primary_sets: list[str]
    program_sha256: str | None = Field(description="Null at a start.")
    support_set: WspValidateSupportSet | None = Field(description="Null at a start.")
    of_run_id: str | None = Field(description="Null at a start.")
    output_cap: str | None = may_be_absent(description="off; present only when off.")
    paging: str | None = may_be_absent(description="off; present only when off.")
    python: str | None = may_be_absent(description="off; present only when off.")
    thinking_tokens: int | None = may_be_absent(
        description="The per-turn reasoning allowance; present only with thinking on."
    )


class WspValidateSensitivity(BaseModel):
    """``sensitivity`` (``wsp_ops.py:validate_run``)."""

    value: str = Field(description="The bundle's mark.")
    effective: str = Field(description="sensitive or not_sensitive.")


class WspOpsValidateRunAnswer(BaseModel):
    """``POST /wsp/runs/validate`` (``wsp_ops.py:validate_run``)."""

    request: WspValidateFrozenRequest
    request_sha256: str = Field(description="RFC 8785 sha256 of request.")
    approval_action: str | None = Field(description="Null for a local inspect (no approval).")
    locality: str
    sensitivity: WspValidateSensitivity
    budgets: WspValidateBudgets = Field(description="request.budgets, repeated.")
    caller_ref: str | None


# ----------------------------------------------------------------------------- inspect


class WspInspectQuantiles(BaseModel):
    """``values.quantiles`` (``wsp/weight_ops/inspect.py:_summary``): keyed by the quantile;
    null for a non-finite value."""

    q0: float | None = Field(alias="0")
    q0_01: float | None = Field(alias="0.01")
    q0_05: float | None = Field(alias="0.05")
    q0_25: float | None = Field(alias="0.25")
    q0_5: float | None = Field(alias="0.5")
    q0_75: float | None = Field(alias="0.75")
    q0_95: float | None = Field(alias="0.95")
    q0_99: float | None = Field(alias="0.99")
    q1: float | None = Field(alias="1")


class WspInspectSummary(BaseModel):
    """A value summary (``wsp/weight_ops/inspect.py:_summary``) over the finite entries."""

    count: int
    finite: int
    quantiles: WspInspectQuantiles | None = Field(description="Null when no entry is finite.")
    mean: float | None
    std: float | None


class WspInspectTensor(BaseModel):
    """One ``tensors[]`` entry (``wsp/weight_ops/inspect.py:tensor_stats``), sorted by name."""

    name: str
    shape: list[int]
    dtype: str
    sha256: str
    values: WspInspectSummary
    l2_norm: float | None = Field(description="Null when an entry is not finite.")
    row_norms: list[float | None] | WspInspectSummary | None = may_be_absent(
        description="2-D finite tensors only: listed up to 4096, else summarised."
    )
    column_norms: list[float | None] | WspInspectSummary | None = may_be_absent(
        description="2-D finite tensors only: listed up to 4096, else summarised."
    )
    singular_values: list[float | None] | None = may_be_absent(
        description="2-D finite tensors only: the top 32."
    )


class WspInspectProbe(BaseModel):
    """``probe`` (``wsp/weight_ops/inspect.py:inspect_tensors``): the effective direction."""

    effective_norm: float | None
    cosine_with: dict[str, float | None] = Field(
        description="Keyed by each requested tensor name; null when a norm is zero."
    )


class WspOpsInspectWeightsAnswer(BaseModel):
    """``POST /wsp/weight-ops/inspect`` (``wsp_ops.py:inspect_weights``, the JSON of
    ``inspect_ops.inspect_response``; its sha256 is in ``X-Content-SHA256``)."""

    tensors: list[WspInspectTensor]
    probe: WspInspectProbe | None = may_be_absent(
        description="A linear_probe bundle with a weight tensor only."
    )


# ----------------------------------------------------------------------------- patches


class WspPatchTensorSummary(BaseModel):
    """One ``tensors[]`` entry (``wsp_ops.py:_summary``)."""

    name: str
    edited_sha256: str = Field(description="The rebuilt tensor's sha256.")
    changed_entries: int


class WspOpsVerifyPatchAnswer(BaseModel):
    """``POST /wsp/patches/verify`` (``wsp_ops.py:verify_patch``)."""

    verified: bool = Field(description="Always true; a failure is a refusal.")
    patch_sha256: str
    content_sha256: str
    changed_entries_total: int
    tensors: list[WspPatchTensorSummary]


class WspPatchFiles(BaseModel):
    """A patch as it travels (``wsp_ops.py:stack_patches``): its two files' contents."""

    patch_json: str = Field(
        description="The miforge.wsp-patch/v1 document's text (patch.json).",
        json_schema_extra={
            "contentMediaType": "application/json",
            "contentSchema": {EXTERNAL: WSP_PATCH},
        },
    )
    edit_npz_base64: str = Field(description="edit.npz, base64.")


class WspOpsStackPatchesAnswer(BaseModel):
    """``POST /wsp/patches/stack`` (``wsp_ops.py:stack_patches``)."""

    no_change: bool
    patch: WspPatchFiles | None = Field(description="Null when the stack changes nothing.")
    patch_sha256: str | None = may_be_absent(description="Absent when no_change.")
    content_sha256: str | None = may_be_absent(description="Absent when no_change.")
