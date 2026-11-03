"""What a signal source is, and what it hands the step  (FR-002.2; R-02.4 - R-02.8).

Four sources answer one question - *what is the next batch?* - behind one Protocol, so
F1's step never learns which one it is talking to (``training/step.py``'s ``BatchSource``).

## Why ``preflight`` is separate from ``next``

FR-002.4 requires a run that cannot proceed to be refused **before it starts**, and a
source that can only report its own impossibility by failing to produce a batch cannot
satisfy that. ``preflight`` is also the **no-GPU half**: parts, kinds, splits, columns,
selector resolution and group size are all decided without a card, which is what makes
every refusal path in this feature testable on a laptop (ADR-023) - and the refusals are
where this feature's defect risk lives.

## THE OBJECTIVE TRAVELS ON THIS OBJECT, NOT ON THE RECORD

``batch-record-v1.json`` carries ``source`` and **no objective**, while
``step-record-v1.json``'s step block *requires* ``objective``. Neither schema is wrong:
the provenance record says which source filled the batch, and the step record says which
loss consumed it. So the objective rides on :class:`TrainingBatch` in process, F1 reads it
through ``resolve_objective``, and F1's step writer is where it is written down. The
asymmetry is stated here because a reader who expects ``record["objective"]`` will add it,
and the record's objects are closed - so the addition would be a *validation* failure
rather than a silent one, which is the design working.

## The same applies to the health signals

The record's ``SelectionHealth`` is four fields - generated, admitted, rate, exhausted -
and has no room for F3's reason histogram, grader statistics or its three **named**
degenerate signals. F6 stops on those names (R-02.25), so F3's full health record rides on
this object as :attr:`TrainingBatch.health` and the reduced form goes to the record.
Recomputing either from the other would produce a second number that can disagree.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable

from pydantic import BaseModel, ConfigDict, Field, model_validator

from providers.base import GenerationParams, ProviderIdentity
from services.selectors.types import PublishedHashes, SelectionHealth

# --------------------------------------------------------------------------------------
# The specification: what a run asks a source for
# --------------------------------------------------------------------------------------


class PartSpec(BaseModel):
    """One part a source should read, named either by contract reference or by part id.

    **Exactly one of the two**, refused rather than resolved by precedence: picking one
    would read a part the operator did not ask for and report success, which is ADR-018's
    fabricated value at the scale of a whole run. The two forms exist because F10's index
    is keyed both ways - a contract part by miDataworks' ``{type, name, version}``, an
    external part by the id of the row carrying its provenance (R-02.40).
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    #: The slot this part fills, in the gate contract's ``PartReference.kind`` vocabulary.
    kind: str
    reference: Mapping[str, str] | None = None
    part_id: str | None = None
    #: Specific splits by name. Empty means *every training-eligible split*; a named split
    #: that is held out is **refused by name** by F10's one rule, never filtered out.
    splits: tuple[str, ...] = ()
    #: ``role -> column name``, for a part declaring more than one usable candidate for a
    #: role. Without it, ambiguity is refused and the candidates are listed (FR-002.7).
    columns: Mapping[str, str] = Field(default_factory=dict)

    @model_validator(mode="after")
    def _exactly_one_target(self) -> PartSpec:
        if (self.reference is None) == (self.part_id is None):
            raise ValueError(
                "a part is named by a contract reference OR by a part id, never both and "
                "never neither: resolving by precedence would read a part nobody asked for"
            )
        return self


class SourceSpec(BaseModel):
    """What a run tells a source. Closed, so a misspelled key is a refusal.

    Deliberately flat and small. Everything a source needs that is *not* configuration -
    the session, the provider, the tokenizer, the cancellation check - is on
    :class:`RunContext`, because those are per-process and this is per-run and the two have
    different lifetimes and different serialisability.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    #: One of the registry's four canonical names.
    source: str
    parts: tuple[PartSpec, ...] = ()
    #: A selector configuration body, as ``selector-config-v1.json`` defines it. F2
    #: resolves it **through F3** and writes none of its own (FR-002.12).
    selector: Mapping[str, Any] | None = None
    #: A grader configuration body. Required by ``reward``; optional for the
    #: from-generations pair path, where a scalar is what orders two completions.
    grader: Mapping[str, Any] | None = None
    #: Where a local configuration came from (R-02.40). Required when a configuration is
    #: supplied without a miDataworks handoff; F3 refuses a hash with no provenance.
    config_source: str | None = None
    #: Examples per batch. One batch is one step's material (R-02.12).
    batch_size: int = Field(default=8, ge=1)
    #: ``reward`` only: completions per prompt. **Below two is refused** (E-16) - a single
    #: completion has no relative advantage by construction.
    group_size: int | None = Field(default=None, ge=1)
    #: ``reward`` only: prompts per batch.
    groups_per_batch: int | None = Field(default=None, ge=1)
    #: Decode parameters for generation. Part of what a batch *is*: candidates drawn at
    #: ``temperature=0`` are not the candidates drawn at ``0.8``.
    params: GenerationParams = GenerationParams()


# --------------------------------------------------------------------------------------
# The process context
# --------------------------------------------------------------------------------------


@runtime_checkable
class Emitter(Protocol):
    """Publishes a run event. The foundation's ``api.sockets.emit_run_event``, injected."""

    def __call__(self, run_id: str, event: str, payload: dict[str, Any]) -> Any: ...


@dataclass
class RunContext:
    """Everything a source needs that is not its configuration.

    Injected rather than imported at the call sites so a test can assert the **payload and
    the call count** of each side effect - "was called" passes against a call sending the
    wrong arguments (ADR-016 #1) - and so a source runs with no database, no socket and no
    provider when the path under test needs none.
    """

    run_id: str = "run"
    #: An open session, for F10's part index and this feature's batch index. ``None`` in a
    #: pre-flight that resolves nothing from the database.
    session: Any | None = None
    #: Where run logs live: ``<log_root>/<run_id>/run.jsonl``. The authoritative record.
    log_root: Any | None = None
    #: The provider generation reaches the model through (ADR-009). **Never a sibling by
    #: name**: miLLM is one row beside vLLM, Ollama and the OpenAI API.
    provider: Any | None = None
    #: Returns the run's current step, for the in-process provider's resolved identity -
    #: so no completion is ever attributed to the wrong version of the model (PPRD §7).
    current_step: Callable[[], int] | None = None
    #: The tokenizer the objectives render with. On the batch rather than fetched by the
    #: loss, because *two runs rendered by different chat templates are not the same run*
    #: and the template's identity belongs on the record.
    tokenizer: Any | None = None
    #: The run's resolved GPU, by **UUID** (ADR-008). Carried, never resolved here: F2
    #: adds no ``nvidia-smi`` call, no index and no bus address.
    gpu_uuid: str | None = None
    #: Cooperative cancellation, checked at a candidate boundary (ADR-006, ADR-007).
    is_cancelled: Callable[[], bool] | None = None
    emit: Emitter | None = None
    heartbeat: Any | None = None


# --------------------------------------------------------------------------------------
# What pre-flight produced
# --------------------------------------------------------------------------------------


@dataclass(frozen=True)
class ColumnBinding:
    """Which declared semantic filled which role, and under which column name.

    Recorded on the batch because *a corpus trained from ``messages`` and one trained from
    ``text`` are not the same run*, and because the binding is the thing a reader needs in
    order to believe the batch was built from the field the operator meant.
    """

    #: ``role -> (column name, declared semantic)``. Roles are a source's own:
    #: ``text``/``prompt``/``completion``/``chosen``/``rejected``.
    roles: Mapping[str, tuple[str, str]]

    def column(self, role: str) -> str:
        return self.roles[role][0]

    def as_record(self) -> dict[str, str]:
        """``role -> semantic``, for a log line. Never the column's *name* alone: a name
        says nothing about what the contract declared the column to mean."""
        return {role: semantic for role, (_name, semantic) in self.roles.items()}


@dataclass(frozen=True)
class ResolvedPart:
    """One part, resolved: its row, its usable splits and its column binding.

    Built only by :mod:`training.signals.parts_reader`, which refuses at each of the three
    layers rather than guessing. Nothing here re-validates the manifest - that is F10's,
    and a second validator is a second opinion about conformance (FR-002.6).
    """

    #: F10's row, as ``services.parts.index.part_payload`` serves it.
    payload: Mapping[str, Any]
    kind: str
    splits: tuple[Mapping[str, Any], ...]
    binding: ColumnBinding
    #: The gate contract's ``PartReference`` object for this part, **copied** onto every
    #: batch record rather than joined - a batch keeps its meaning if the part row is
    #: removed, and parts are immutable once imported (F10), so the copy is truthful.
    reference: Mapping[str, Any]


@dataclass(frozen=True)
class ResolvedSource:
    """What a source's pre-flight produced. A source either returns this or refuses.

    **Never half-resolved.** A partially resolved source is one whose failure surfaces at
    step 37, by which point the operator is reading a transport error instead of a
    configuration error.
    """

    source: str
    spec: SourceSpec
    parts: tuple[ResolvedPart, ...] = ()
    #: F3's resolved selector, for the two sources that select. ``None`` otherwise.
    selector: Any | None = None
    #: F3's resolved grader configuration, for ``reward`` and for the ordered-pair path.
    grader: Any | None = None
    #: The objective name that will be written on every batch this source yields.
    objective: str = ""
    #: Set at pre-flight for the sources that generate: what a verdict's R-02.11 identity
    #: guard compares a judge against (FR-002.14, F3 FR-003.17).
    run_target: ProviderIdentity | None = None


@dataclass(frozen=True)
class SourceCapability:
    """What a source needs before it can fill a batch, for ``GET /signal-sources``.

    Declared statically; **availability is computed per call from injected settings** and
    is not cached at import - an operator who configures a provider must see the source
    become available without a restart (F3's ``GET /selectors`` does the same).
    """

    name: str
    #: The ``PartReference.kind`` this source reads, or ``None`` when it reads none.
    part_kind: str | None
    needs_selector: bool
    needs_grader: bool
    needs_generation: bool
    objective: str
    summary: str


# --------------------------------------------------------------------------------------
# The batch
# --------------------------------------------------------------------------------------


@dataclass(frozen=True)
class BatchCounts:
    """Every outcome a batch can have, counted. **None of these is an error.**

    ``pending`` is the one to get right: an ``undecided`` verdict is counted here and
    **nowhere else** - never in ``rejected``, never in the admitted set. Four sites that
    each decide what pending means is four chances for one of them to call it a reject
    (FR-002.13), which is why there is one partition function and one of these.
    """

    generated: int = 0
    admitted: int = 0
    rejected: int = 0
    pending: int = 0
    #: Pairs dropped as malformed - identical or empty sides (E-10).
    dropped: int = 0
    #: Pairs not produced because two scalars were equal or absent (E-11). **A tie is
    #: never broken**, so this is a normal outcome with a count and not a failure.
    ties: int = 0
    #: True when the material ran out before the batch was filled.
    exhausted: bool = False
    #: How many examples were asked for, so :attr:`shortfall` is a measurement rather
    #: than an inference. ADR-012: every number names its scale and sample.
    requested: int = 0

    @property
    def shortfall(self) -> int:
        """How many examples short of what was asked for. **A count, never a licence.**

        OQ-1 leaves *refuse the batch* or *take a short one* to configuration; neither
        value admits a pending completion, and this number is what the short-batch
        setting records. There is no path from a shortfall to an accept (FR-002.13).
        """
        return max(0, self.requested - self.admitted)


@dataclass(frozen=True)
class SupervisedExample:
    """One next-token cross-entropy example (R-02.4, R-02.5).

    ``prompt`` is empty for plain text. The **masking policy decides which tokens the loss
    is taken over**, and it is applied in ``objectives/sft.py`` from the value on the
    batch - so the record and the computation cannot disagree.
    """

    prompt_id: str
    prompt: str
    completion: str
    #: Set when the example was rendered through a tokenizer chat template, so the
    #: template's identity can be recorded. Two runs rendered differently are not one run.
    messages: tuple[Mapping[str, Any], ...] | None = None


@dataclass(frozen=True)
class PreferenceExample:
    """One preference pair (R-02.6), with **what ordered it** when it was derived."""

    prompt_id: str
    prompt: str
    chosen: str
    rejected: str
    #: ``file`` for a pairs part; ``grader_score`` or ``verdict_score`` for a pair built
    #: from two generations. A pair ordered by a grader and one ordered by a verdict's
    #: score are not the same evidence, so the record says which.
    ordered_by: str = "file"
    #: The scalar margin that ordered it. ``None`` for a pairs part, where nothing was
    #: compared. **Zero is never here**: an equal pair is a ``NoPair`` (E-11).
    margin: float | None = None
    #: Reference log-probabilities for the ``precomputed`` DPO reference source (FPRD
    #: OQ-3), cached **before the run** from the parameter set the run started with. Fields
    #: rather than an attribute set at a call site, so MyPy sees them and a pair that was
    #: never given them refuses by name instead of raising an AttributeError at the loss.
    reference_chosen_logp: float | None = None
    reference_rejected_logp: float | None = None


@dataclass(frozen=True)
class RewardGroup:
    """One prompt's completions and their scores, with the statistics on the record.

    A group of fewer than two is refused (E-16). The statistics travel **with the grader's
    declared range** so two runs' rewards are never compared as though they shared a
    scale (ADR-012: every number names its scale and sample).
    """

    prompt_id: str
    prompt: str
    completions: tuple[str, ...]
    scores: tuple[float, ...]
    declared_range: tuple[float, float]
    minimum: float
    maximum: float
    mean: float
    variance: float
    #: True when every reward in the group is equal. The step then runs in full and
    #: applies **numerically nothing**: a group-relative advantage over a constant reward
    #: is zero for every member. Recorded rather than skipped - see ``reward.py``.
    zero_spread: bool


@dataclass(frozen=True)
class TrainingBatch:
    """One batch, as F1's step consumes it (``training.step.Batch``).

    ``objective`` and ``record`` satisfy F1's Protocol. Everything else is the material
    the objective needs and the provenance the record has no field for - see the module
    docstring for why that asymmetry is deliberate.
    """

    objective: str
    record: Mapping[str, Any]
    #: The material, whose element type is the objective's: ``SupervisedExample`` for
    #: ``sft``, ``PreferenceExample`` for ``dpo``, ``RewardGroup`` for ``grpo``. One field
    #: rather than three, because the step hands the whole batch to one loss and a second
    #: field would let a batch carry material for an objective it does not name.
    examples: tuple[Any, ...]
    counts: BatchCounts
    #: F3's **full** health record, with its named signals. F6 reads the names from here.
    health: SelectionHealth | None = None
    hashes: PublishedHashes | None = None
    #: Which tokens the cross-entropy is taken over. ``None`` for an objective that does
    #: not mask. **Never defaulted inside a loss** (FR-002.8, E-5).
    masking_policy: str | None = None
    #: Where DPO's reference log-probabilities come from (FPRD OQ-3). Recorded, so a loss
    #: figure can be read against another.
    reference_source: str | None = None
    tokenizer: Any | None = None
    #: The chat template's identity, when one rendered this batch.
    chat_template: str | None = None
    #: Who generated the candidates, for the sources that generate.
    provider: ProviderIdentity | None = None
    binding: Mapping[str, str] = field(default_factory=dict)
    #: Review item/decision identifiers for the candidates a human adjudicated
    #: (R-02.36's closing sentence). Identifiers only - never the reviewer's text.
    review_ids: tuple[str, ...] = ()

    @property
    def size(self) -> int:
        return len(self.examples)

    @property
    def batch_sha256(self) -> str:
        """The content hash the step record cites (R-02.14's batch source hash)."""
        value = self.record.get("batch_sha256")
        if not isinstance(value, str):  # pragma: no cover - the writer guarantees it
            raise AssertionError("a batch record always carries its own hash")
        return value


@runtime_checkable
class SignalSource(Protocol):
    """One signal source. Four implementations, resolved from one registry (FR-002.1)."""

    def capability(self) -> SourceCapability:
        """What this source needs. Answers with no run and no spec in hand."""
        ...

    def preflight(self, spec: SourceSpec) -> ResolvedSource:
        """Validate everything decidable without a GPU and without a step.

        Returns a :class:`ResolvedSource` or raises a named ``Refusal``. **Never partially
        succeeds.**
        """
        ...

    def next(self) -> TrainingBatch:
        """One batch, recorded before it is returned.

        Raises:
            Refusal: any assembly refusal. **The whole batch is refused** - no partial
                batch reaches a step and no completion is substituted (E-6, E-17).
        """
        ...

    def close(self) -> None:
        """Release readers and provider sessions. Idempotent."""
        ...


#: What the registry holds: a callable taking the run's context and returning a source.
SourceFactory = Callable[[RunContext], SignalSource]


def example_prompt_ids(examples: Sequence[Any]) -> tuple[str, ...]:
    """The prompt identifiers of a batch's material, in order, de-duplicated.

    **Identifiers, never the prompts** (R-02.8's own words, ADR-020). One helper so the
    four sources cannot each take a different view of what "the prompt IDs" means - which
    would make two batch records incomparable while both validated.
    """
    seen: list[str] = []
    for example in examples:
        prompt_id = getattr(example, "prompt_id", None)
        if isinstance(prompt_id, str) and prompt_id and prompt_id not in seen:
            seen.append(prompt_id)
    return tuple(seen)
