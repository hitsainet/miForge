"""The answers of the WSP candidate routes (feature 021, P3): ``wsp_candidates.py``.

Documentation only (MF9): the routes still return their dicts; these models describe them. The
verdict document is the run contract's ``Verdict`` (``docs/schemas/miforge-wsp-run-v1.json``),
checked by ``wsp.schemas.validate_run`` before it is stored.
"""

from pydantic import BaseModel, Field

from .answers_common import may_be_absent


class WspCandidateParent(BaseModel):
    """A candidate's ``parent`` (``wsp_candidates.py:_summary``, ``get_lineage``)."""

    kind: str = Field(description="bundle, bundle_zeroed, candidate or checkpoint.")
    ref: str = Field(description="bundle_sha256, candidate_id or checkpoint name.")


class WspCandidateTensorAppend(BaseModel):
    """A patch tensor's ``append`` (``miforge-wsp-patch-v1.json`` ``Append``)."""

    axis: int
    count: int


class WspCandidateTensor(BaseModel):
    """One ``tensors[]`` entry: the patch manifest's tensor, stored whole at intake
    (``wsp/runs.py:record_candidate``; ``miforge-wsp-patch-v1.json`` ``PatchTensor``)."""

    index: int
    name: str
    dtype: str
    form: str = Field(description="sparse, lowrank or dense.")
    shape: list[int] = Field(description="The base shape.")
    edited_shape: list[int] = Field(description="Equal to shape unless append.")
    base_sha256: str
    edited_sha256: str
    changed_entries: int
    l2_delta: float
    max_abs_delta: float
    rank: int | None = Field(description="lowrank only.")
    append: WspCandidateTensorAppend | None = Field(description="SAE features only.")


class WspVerdictReading(BaseModel):
    """A verdict as the candidate routes show it (``wsp_candidates.py:_verdict``,
    ``wsp/standing.py:reading``)."""

    verdict_id: str
    value: str = Field(description="The stored verdict value.")
    standing: str = Field(
        description="improved only when confirmed; improved_unconfirmed; else the value."
    )
    confirmed_by: list[str] = Field(description="The primary sets that confirm an improved value.")


class WspCandidateSummary(BaseModel):
    """One candidate (``wsp_candidates.py:_summary``), as list and get show it."""

    candidate_id: str
    run_id: str
    parent: WspCandidateParent
    bundle_base_sha256: str
    patch_sha256: str
    changed_entries: int
    tensors: list[WspCandidateTensor]
    created_at: str = Field(description="RFC 3339 UTC.")
    verdict: WspVerdictReading | None = Field(description="Null until the candidate is evaluated.")


class WspCandidateListCandidatesAnswer(BaseModel):
    """``GET /wsp/candidates`` (``wsp_candidates.py:list_candidates``)."""

    candidates: list[WspCandidateSummary] = Field(description="Newest first.")


class WspCandidateGetCandidateAnswer(WspCandidateSummary):
    """``GET /wsp/candidates/{candidate_id}`` (``wsp_candidates.py:get_candidate``)."""


class WspCandidateLineageStep(BaseModel):
    """One ``steps[]`` entry (``wsp_candidates.py:get_lineage``), bundle first."""

    candidate_id: str
    run_id: str
    parent: WspCandidateParent
    patch_sha256: str
    provider: str | None = Field(description="Null when the step's run row is gone.")
    provider_version: str | None = Field(description="Null when the step's run row is gone.")
    transcript_sha256: str | None
    verdict: WspVerdictReading | None


class WspCandidateGetLineageAnswer(BaseModel):
    """``GET /wsp/candidates/{candidate_id}/lineage`` (``wsp_candidates.py:get_lineage``)."""

    candidate_id: str
    bundle_sha256: str | None = Field(description="The first step's run's bundle.")
    steps: list[WspCandidateLineageStep]


# ----------------------------------------------------------------------------- the verdict


class WspVerdictDeltaFigures(BaseModel):
    """``control_figures.candidate`` / ``.control_post_cast`` (``wsp/evaluate.py:_figures``)."""

    changed_entries: int
    l2_delta: float


class WspVerdictControlFigures(BaseModel):
    """``control_figures`` (``wsp/evaluate.py:material``)."""

    candidate: WspVerdictDeltaFigures
    control_post_cast: WspVerdictDeltaFigures


class WspVerdictScorer(BaseModel):
    """``scorer`` (``wsp/evaluate.py:_base_document``)."""

    owner: str
    contract: str = Field(description="miworkbench-wsp-scoring-v1.")
    call_id: str = Field(description="Empty when the scorer was never called.")
    metrics_agree: bool


class WspVerdictCompleteness(BaseModel):
    """``completeness`` (``wsp/verdict.py:decide``)."""

    ok: bool
    missing: list[str] = Field(
        description="heldout_set_missing:<n>, heldout_set_too_small:<n>, baseline_missing:<k>."
    )


class WspVerdictBar(BaseModel):
    """A head's ``bar`` (``wsp/evaluate.py:_set_verdict``), from the scorer's answer."""

    threshold: float | None
    source: str
    target_fpr: float
    realised_fpr: float


class WspVerdictHead(BaseModel):
    """One ``per_set[].heads[]`` entry (``wsp/evaluate.py:_set_verdict``)."""

    label: str = Field(description="parent, candidate, control, control:<n> or baseline:<kind>.")
    auroc: float | None
    recall_at_bar: float | None
    false_alarms_at_bar: int | None
    false_alarm_rate: float | None = Field(description="false_alarms_at_bar / negatives.")
    bar: WspVerdictBar | None = Field(
        description="Always an object from this code; the run contract allows null."
    )
    fvu: float | None = Field(description="SAE only; null for a probe.")
    l0: float | None = Field(description="SAE only; null for a probe.")
    dead_fraction: float | None = Field(description="SAE only; null for a probe.")
    scores_sha256: str
    identical_to_parent: bool | None = may_be_absent(
        description=(
            "A baseline:<kind> head only (absent on every other head, and on a verdict stored "
            "before 2026-11-28): true when its scores_sha256 equals the parent head's on the same "
            "set, so the baseline ranks every row exactly as the parent does. It stays among the "
            "baselines improved must beat: beating it is beating the parent, which is already "
            "required."
        ),
    )


class WspVerdictDelta(BaseModel):
    """One ``per_set[].deltas[]`` entry (``wsp/verdict.py:decide``)."""

    metric: str
    against: str = Field(description="A head label.")
    estimate: float
    lower: float
    upper: float
    worse_if: str = Field(description="lower or higher.")


class WspVerdictSet(BaseModel):
    """One ``per_set[]`` entry (``wsp/evaluate.py:_set_verdict``)."""

    name: str
    role: str
    primary: bool
    rows: int
    positives: int
    negatives: int
    excluded_truncated: int
    truncated_fired: dict[str, int] = Field(
        description="Keyed by head label: the set's truncated rows that fire at that head's bar."
    )
    heldout_uses: int = Field(description="The set's count after this evaluation.")
    heads: list[WspVerdictHead]
    deltas: list[WspVerdictDelta]
    value: str


class WspVerdictDocument(BaseModel):
    """The stored verdict document (``wsp/evaluate.py:_base_document``, ``judge``, ``store``);
    the run contract's ``Verdict``."""

    verdict_id: str
    value: str = Field(description="improved, not_significant, regressed, no_change or unmeasured.")
    reasons: list[str]
    request_sha256: str
    statistic: str = Field(description="paired_bootstrap.")
    B: int = Field(description="Bootstrap resamples.")
    seed: int
    ci: float
    control_seed: int
    control_figures: WspVerdictControlFigures
    primary_metric: str
    scorer: WspVerdictScorer
    guarded_comparisons: int
    completeness: WspVerdictCompleteness
    per_set: list[WspVerdictSet] = Field(description="Empty when the scorer gave no answer.")
    evaluated_at: str = Field(description="RFC 3339 UTC.")


class WspCandidateEvaluateCandidateAnswer(WspVerdictDocument):
    """``POST /wsp/candidates/{candidate_id}/evaluate`` (``wsp/evaluate.py:evaluate``)."""


class WspCandidateGetVerdictAnswer(WspVerdictDocument):
    """``GET /wsp/candidates/{candidate_id}/verdict`` (``wsp_candidates.py:get_verdict``)."""


# ----------------------------------------------------------------------------- held-out uses


class WspHeldoutUse(BaseModel):
    """One ``heldout_uses[]`` row (``wsp_candidates.py:list_heldout_uses``)."""

    owner: str
    set_name: str
    row_ids_sha256: str
    heldout_uses: int
    last_used_at: str = Field(description="RFC 3339 UTC.")
    last_candidate_id: str | None


class WspHeldoutListHeldoutUsesAnswer(BaseModel):
    """``GET /wsp/heldout-uses`` (``wsp_candidates.py:list_heldout_uses``)."""

    heldout_uses: list[WspHeldoutUse]
