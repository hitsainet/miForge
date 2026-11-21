"""F3's own in-memory shapes  (FR-003.3, FR-003.10, FR-003.14; R-02.9, R-02.10).

**None of this is a contract and none of it redefines one.** The verdict contract lives in
:mod:`schemas.selector` (foundation 3.1/3.1a) and the provider contract in
:mod:`providers.base` (foundation 3.8). This module holds the shapes F3 needs *around*
them: what a resolved configuration is, where a verdict's provenance rides, and what a
batch's selection health looks like.

## Why a ``VerdictRecord`` exists beside ``Verdict``

:class:`schemas.selector.Verdict` is ``frozen`` with ``extra="forbid"`` - deliberately, and
F3 does not widen it (FR-003.2: *this feature adds no field to that schema and no method to
that protocol*). But three things still have to be recorded per verdict:

* **the machine reason code** - which rides on ``Verdict.label``, whose own contract comment
  says it carries *"a free-text label from a human decision, **or a rule's reason code**"*;
* **who judged** (FR-003.18, R-02.11) - the resolved judge :class:`ProviderIdentity`,
  because BRD-02 acceptance criterion 3 is *"a self-generation run **with the judge-model
  selector**"* and that is not auditable afterwards without the identity on the record;
* **which mode or which queue item** answered - a reference-match verdict is not comparable
  across modes, and an ``undecided`` human-review verdict is useless without its item id.

So :class:`VerdictRecord` pairs the contract ``Verdict`` with that provenance. The
``Verdict`` is what every consumer reasons about and :func:`schemas.selector.is_admitted` is
still the only thing that decides admission; the record is what gets written down.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

from providers.base import ProviderIdentity
from schemas.selector import SelectorKind, Verdict

# --------------------------------------------------------------------------------------
# Capabilities: what a type offers and what it needs
# --------------------------------------------------------------------------------------


class Requirement(StrEnum):
    """What a selector type needs before it can answer.

    Three values, not four: **embedding similarity is not here** because
    ``selector-config-v1.json`` cut it. R-02.9's amendment offered it, R-02.49's provider
    abstraction is scoped to generation, and *"embedding"* occurs exactly once in the whole
    of BRD-02 - in that amendment. A requirement for a capability with no mechanism is the
    missing-load-call shape, so the contract refuses the mode and this enum has no row for
    it.
    """

    #: Pure and in-process. Guaranteed available on the standalone path (AC20).
    NONE = "none"
    #: An OpenAI-compatible generation endpoint, reached through :mod:`providers.base`.
    #: **Never the in-process provider** - R-02.11 makes a model grading itself not a
    #: measurement, so the judge type reports itself unavailable by name rather than
    #: falling back to the model being trained.
    GENERATION_PROVIDER = "generation_provider"
    #: miDataworks' review queue (R-02.36, BRD-03 R-03.41). Reached through F10's single
    #: outbound call site, never from here (ADR-010).
    MIDATAWORKS = "midataworks"


class SelectorForm(StrEnum):
    """Whether a type admits (selector) or scores (grader). Some offer both."""

    SELECTOR = "selector"
    GRADER = "grader"


@dataclass(frozen=True)
class Capability:
    """A registered type's declaration of itself. Static; availability is not."""

    kind: SelectorKind
    forms: frozenset[SelectorForm]
    requires: Requirement
    #: One line for ``GET /api/v1/selectors`` and F9's screen - what this type decides on.
    summary: str


class CapabilityView(BaseModel):
    """One row of ``GET /api/v1/selectors``: the declaration plus live availability.

    ``available`` is computed **per call from injected settings**, never cached at import.
    A module global would freeze the answer, so an operator who configures miDataworks
    would be told the type is still unavailable until something restarted (FTID I-3).
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    name: SelectorKind
    forms: tuple[str, ...]
    requires: Requirement
    summary: str
    available: bool
    #: The refusal code a call would raise, naming the missing service (ADR-010, ADR-013).
    #: ``None`` exactly when ``available``. A disabled control with no explanation is what
    #: this field exists to prevent.
    unavailable_reason: str | None = None


# --------------------------------------------------------------------------------------
# Configuration identity (R-02.10)
# --------------------------------------------------------------------------------------

HashProvenance = Literal["contract", "local"]


class ContractReference(BaseModel):
    """miDataworks' ``{type, name, version}`` citation form (BRD-03 R-03.47).

    This is the **contract** rendering. ``<pack>/<type>/<name>@<version>`` is F7 prefixing
    its own pack identity for display - two layers, and conflating them is a recorded
    mistake in this project.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    type: str
    name: str
    version: str | int


class ResolvedHash(BaseModel):
    """A configuration's identity, with **where the identity came from**.

    ``provenance`` has **no default**, here and in every caller. ADR-018: a field that
    defaults to ``"contract"`` is a fabricated provenance claim the moment someone forgets
    to set it, and R-02.40 forbids miForge claiming miDataworks provenance for anything
    miDataworks did not publish. Mutation control 5 breaks exactly this.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    sha256: str
    provenance: HashProvenance
    #: Present only when ``provenance == "contract"``. A local hash is **never** rendered
    #: as a contract reference.
    contract_reference: ContractReference | None = None
    #: Where a local configuration came from, for the honesty rule. ``None`` for a
    #: contract hash, whose provenance is the reference above.
    source: str | None = None


class PublishedHashes(BaseModel):
    """What F2's batch record and F5's manifest write get from F3 (FR-003.16).

    **One scalar per kind.** ``checkpoint-manifest-v1.json`` holds ``selector_hash`` and
    ``grader_hash`` as single nullable strings - verified against the vendored file, not
    assumed - so a composite publishes **its own** hash and its children are enumerated in
    the batch record instead. A list is not expressible in the manifest.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    selector_hash: str | None = None
    grader_hash: str | None = None
    selector_hash_provenance: HashProvenance | None = None
    grader_hash_provenance: HashProvenance | None = None
    selector_contract_reference: ContractReference | None = None
    grader_contract_reference: ContractReference | None = None


# --------------------------------------------------------------------------------------
# A verdict's provenance
# --------------------------------------------------------------------------------------


class VerdictRecord(BaseModel):
    """One :class:`Verdict` plus everything needed to audit it afterwards.

    See the module docstring for why this is not extra fields on ``Verdict``.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    verdict: Verdict
    #: The stable machine code a UI maps on. Mirrors ``verdict.label``; carried explicitly
    #: so a reader of this record does not have to know that ``label`` doubles as the code.
    reason_code: str
    #: The configuration that produced it, by hash - so a verdict can be tied back to the
    #: exact selector body that admitted or dropped the completion.
    config_sha256: str
    config_kind: SelectorKind
    #: **Who judged** (FR-003.18). Present on every judge-model verdict and absent
    #: otherwise; absence is the distinction, not a missing value.
    judge: ProviderIdentity | None = None
    #: ``judge_shares_lineage`` when the judge is a different parameter set of the same
    #: lineage (E-7, OQ-3). Returned and **recorded**, never only logged - a log line is
    #: not a record.
    caveats: tuple[str, ...] = ()
    #: ``exact`` | ``fuzzy`` for reference-match. A verdict is not comparable across modes,
    #: so the mode in force is part of the record.
    match_mode: str | None = None
    #: The metric that produced a fuzzy score. The metric is **part of the verdict**:
    #: swapping it silently rescopes every reference-match decision.
    match_metric: str | None = None
    #: miDataworks' queue item, for an ``undecided`` human-review verdict and for the
    #: decision identifiers R-02.36's last sentence requires on the batch record.
    review_item_id: str | None = None
    review_decision_id: str | None = None


# --------------------------------------------------------------------------------------
# Selection health (FR-003.10)
# --------------------------------------------------------------------------------------

#: The three degenerate outcomes, by name. Named rather than described because F6 stops on
#: them (R-02.25) and F7/F9 render them - a UI keyed on prose breaks on a wording change.
SIGNAL_ACCEPTS_EVERYTHING = "selection_accepts_everything"
SIGNAL_REJECTS_EVERYTHING = "selection_rejects_everything"
SIGNAL_NO_SPREAD = "grader_scores_have_no_spread"


class GraderStats(BaseModel):
    """A batch's score distribution. ``variance`` is the one that matters.

    A group-relative policy objective (R-02.7, GRPO) computes an advantage from the spread
    of rewards within a group. A constant reward gives **zero advantage for every member**,
    so the update is numerically nothing: the run burns GPU hours, records a loss, and
    learns nothing. That is why zero variance is a named signal and not a curiosity.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    n: int
    min: float
    max: float
    mean: float
    variance: float


class SelectionSignal(BaseModel):
    """One named degenerate outcome, with the counts and the operator sentence.

    The sentence is supplied **here, once**, so F7 and F9 render one sentence rather than
    inventing two. A signal shown under a sentence that does not describe it is worse than
    no sentence: the operator reads the wrong failure.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    name: str
    message: str
    details: dict[str, Any] = Field(default_factory=dict)


class SelectionHealth(BaseModel):
    """What a batch's selection looked like. **Data, not a log line.**

    ``acceptance_rate`` never travels without ``n`` (ADR-012: every number names its scale
    and sample). ``Accepted 3 of 5`` - never a bare ``60%``.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    n: int
    accepted: int
    rejected: int
    undecided: int
    acceptance_rate: float
    reason_histogram: dict[str, int]
    grader_stats: GraderStats | None = None
    signals: tuple[SelectionSignal, ...] = ()

    @property
    def signal_names(self) -> tuple[str, ...]:
        return tuple(signal.name for signal in self.signals)


# --------------------------------------------------------------------------------------
# Evaluation context
# --------------------------------------------------------------------------------------


@dataclass
class EvaluationContext:
    """Everything an evaluation needs that is not the configuration or the completions.

    A dataclass rather than ``**kwargs`` so MyPy strict catches a misspelled field instead
    of it arriving as a silently-ignored extra.
    """

    #: Bound into every log line for the run's lifetime (ADR-021). ``None`` for F7's
    #: dry-run, which happens before a run exists.
    run_id: str | None = None
    #: Identifies the batch, and seeds the deterministic ``external_id`` of a review
    #: candidate so a retried post is idempotent.
    batch_id: str = "batch"
    #: Per-completion extras the plugins read: ``reference`` for reference-match,
    #: ``objective`` for F2. Opaque here.
    items: tuple[dict[str, Any], ...] = ()
    #: Checked at each **completion boundary** (ADR-006, ADR-007). Cooperative: a
    #: completion is evaluated and recorded, or not started - never half.
    #: ``revoke(terminate=True)`` is never used; a ``--pool=solo -c 1`` worker has no pool
    #: child to signal.
    is_cancelled: Callable[[], bool] | None = None
    #: Where the health signals go. Injected so a test can assert the payload **and the
    #: call count** - "was called" passes against a call sending the wrong arguments.
    emit_signal: Callable[[str, dict[str, Any]], None] | None = None
    #: Extra fields merged into the context handed to each plugin.
    extra: dict[str, Any] = field(default_factory=dict)


class EvaluationResult(BaseModel):
    """What :func:`services.selectors.service.evaluate` returns.

    Verdicts are a **list in input order**, never a mapping keyed on completion text: two
    identical completions are two candidates, and a dict would silently merge them.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    records: tuple[VerdictRecord, ...]
    health: SelectionHealth
    hashes: PublishedHashes

    @property
    def verdicts(self) -> tuple[Verdict, ...]:
        return tuple(record.verdict for record in self.records)
