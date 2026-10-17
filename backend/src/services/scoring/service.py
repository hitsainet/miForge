"""One entry point for an evaluation pass  (FR-004.7, 14a, 20, 21, 24; E-18, E-19; M-6, M-9).

## Callable three ways, implemented once  (FR-004.14a)

* **directly, synchronously, in the ``train`` worker** - the in-cadence pass F6's loop
  awaits. ADR-009's in-process provider generates *"from the parameters miForge holds"*, and
  those exist **only** in that worker's process: a ``generate`` worker would need a second
  copy of the parameter set on the leased card, against ADR-008's one-run-per-card lease and
  BRD-02 §6's budget, and a cross-queue ``.get()`` from a ``--pool=solo -c 1`` task blocks
  the only worker thread on a queue it cannot drain (ADR-006);
* from ``miforge.generate.evaluate`` on the ``generate`` queue, for an **HTTP-provider**
  pass, where no VRAM of miForge's is involved at all;
* from ``POST /api/v1/runs/{run_id}/evaluate``, out of cadence, which the loop does not wait
  on.

**The Celery task is a thin wrapper over this function, not a second implementation.** A test
asserts all three callers reach one implementation by **call count**, not by "something was
called".

## The paired measurement is one call, not two

FR-004.24: test and retention at one step share the step, the provider and the decode
parameters. :func:`evaluate` produces **both** from one resolution of the provider and one
``GenerationParams``, so they cannot diverge. Two calls scheduled to coincide will eventually
be configured, or retried, differently, and R-02.17's *"scored the same way"* is the
requirement the whole comparison rests on.

## No partial pass, ever  (M-6)

A provider failure propagates and the pass records **no set score**. There is no
``except Exception: return partial`` here, and there must never be: that is the
plausible-looking "robustness" change that silently reports a confident number over an
unknown subset. A refusal is recoverable; a silently wrong number is not.

## No provider fallback  (M-9)

The provider is **pinned** at run start and resolved by name from the one registry. An
unreachable endpoint refuses; it never becomes the in-process provider, which would rewrite
the meaning of the run's curve halfway through it with nothing saying so.

## Cancellation at a prompt boundary

A prompt is scored and recorded, or not started (E-18). Checked **between** chunks, never
inside a generate. Cancellation is read from the database because every worker is
``--pool=solo -c 1``: ``revoke(terminate=True)`` signals a pool child that does not exist, and
a busy solo worker never reads the control queue.

## This module writes no parameter, no optimizer state and no checkpoint

FR-004.20, asserted statically. An evaluation is read-only on the model, and the cheapest way
to corrupt a run is to measure it carelessly.
"""

from __future__ import annotations

import logging
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from core.errors import Refusal, refuse
from providers import registry
from providers.base import Completion, GenerationParams, InferenceProvider
from services.scoring import aggregate as aggregation
from services.scoring import expectations, record, retention
from services.scoring.sets import ResolvedSet
from services.scoring.settings import ScoringSettings, get_scoring_settings
from services.scoring.tables import KIND_RETENTION, KIND_TEST

log = logging.getLogger("miforge.scoring")

#: Transport-shaped refusal codes F3's HTTP provider raises. Re-raised under this feature's
#: own code **with the original preserved in ``details.provider_code``**: the noun matters -
#: a scoring pass is not a judge, so ``judge_provider_unavailable`` sends an operator to the
#: wrong screen - and *a re-coded refusal loses its owner*, which is why the owner's code
#: travels with it rather than being discarded.
_TRANSPORT_CODES = frozenset(
    {
        "judge_provider_unavailable",
        "judge_provider_not_configured",
        "inference_provider_unreachable",
    }
)
_PARSE_CODES = frozenset({"judge_reply_unparseable", "inference_provider_reply_unparseable"})


@dataclass(frozen=True, slots=True)
class EvaluationSpec:
    """A run's resolved scoring configuration. **F4-internal; not a contract.**

    *Divergence from this feature's documents, reported not hidden:* they spell the entry
    point ``evaluate(run, step, kind)`` with ``run.slot_for(kind)``,
    ``run.pinned_provider_name``, ``run.decode``, ``run.batch_bound`` and
    ``run.aggregation_method``. :class:`models.run.Run` has **none** of those and ``mf_runs``
    has no column for them - R-02.53's run definition lives in ``definition_resolved``, and
    assembling it is F6's. So the resolved configuration arrives as this value object, which
    F6 builds from the run definition and a test builds directly.
    """

    run_id: str
    provider_name: str
    aggregation: str
    decode: GenerationParams
    test_set: ResolvedSet | None = None
    retention_set: ResolvedSet | None = None
    test_expectation: expectations.ResolvedExpectation | None = None
    retention_expectation: expectations.ResolvedExpectation | None = None
    floor: retention.ResolvedFloor = field(
        default_factory=lambda: retention.ResolvedFloor(value=None, layer="none")
    )
    batch_bound: int = 8
    #: Set by the caller when the provider was re-pinned at this step (FR-004.10). F9 breaks
    #: the curve here rather than joining two instruments' numbers.
    provider_discontinuity: bool = False
    #: Parameters the pass **depends on the provider honouring** (FR-004.8). *"OpenAI
    #: compatible" is a shape, not a guarantee*: miStudio records in code that miLLM ignores
    #: the OpenAI ``response_format`` parameter - verified, identical output with and
    #: without - so a pass that needs one declares it and the resolution refuses a provider
    #: that does not. Nothing is sent hopefully.
    requires: frozenset[str] = frozenset()


@dataclass(frozen=True, slots=True)
class EvaluationOutcome:
    """What one pass produced: the records, the breach if any, and the block written."""

    records: tuple[record.ScoreRecordData, ...]
    breach: retention.BreachSignal | None
    block: dict[str, Any]

    def by_kind(self, kind: str) -> record.ScoreRecordData | None:
        for item in self.records:
            if item.kind == kind:
                return item
        return None


def evaluate(
    spec: EvaluationSpec,
    step: int,
    *,
    session: Any,
    log_path: Path,
    kinds: Sequence[str] = (KIND_TEST, KIND_RETENTION),
    out_of_cadence: bool = False,
    emit: Callable[[str, str, dict[str, Any]], Any] | None = None,
    heartbeat: Any = None,
    is_cancelled: Callable[[], bool] | None = None,
    provider: InferenceProvider | None = None,
    settings: ScoringSettings | None = None,
    now: datetime | None = None,
) -> EvaluationOutcome:
    """Run one evaluation pass. **The one implementation; three callers.**

    Args:
        spec: The run's resolved scoring configuration.
        step: The canonical step as one integer. ``0`` is a real evaluation and not a
            sentinel - criterion 12 names the step-0 test score as a creation run's baseline.
        session: An open session. The caller commits.
        log_path: The run's ``run.jsonl``.
        kinds: Which sets to score. Both by default, which is FR-004.24's pairing.
        out_of_cadence: Records ``out_of_cadence: true`` so an ad-hoc check never silently
            joins the step series.
        emit: ``api.sockets.emit_run_event``. ``None`` completes the pass without it - the
            socket is a convenience, never the record.
        heartbeat: The foundation's throttled database heartbeat.
        is_cancelled: Read from the database, checked **between** chunks.
        provider: An already-resolved provider, for a caller that resolved and pinned it.
            ``None`` resolves ``spec.provider_name`` from the one registry.
        settings: Injected for tests; the process settings otherwise.
        now: The pass's timestamp, shared by both records so they are one pass.

    Raises:
        Refusal: any of this feature's codes. A failure refuses the **evaluation** - not the
            prompt - and records **no** score (FR-004.7).
    """
    settings = settings or get_scoring_settings()
    resolved_provider = provider if provider is not None else registry.get(spec.provider_name)
    _assert_capabilities(resolved_provider, spec)
    timestamp = now or datetime.now(UTC)

    reproducible, reason = _reproducibility(spec)

    records: list[record.ScoreRecordData] = []
    breach: retention.BreachSignal | None = None

    for kind in kinds:
        resolved_set, expectation = _for_kind(spec, kind)
        if resolved_set is None or expectation is None:
            continue
        completions = _generate(
            resolved_provider, resolved_set, spec, is_cancelled=is_cancelled, step=step
        )
        scored = expectations.score_completions(
            expectation,
            resolved_set.texts,
            [completion.text for completion in completions],
            references=[prompt.reference for prompt in resolved_set.prompts],
            run_id=spec.run_id,
            batch_id=f"{spec.run_id}:{step}:{kind}",
            is_cancelled=is_cancelled,
        )
        score = aggregation.aggregate(spec.aggregation, scored.values)

        floor = spec.floor if kind == KIND_RETENTION else retention.ResolvedFloor(None, "none")
        signal = None
        if kind == KIND_RETENTION:
            signal = retention.evaluate_floor(
                score,
                floor,
                run_id=spec.run_id,
                step=step,
                n=resolved_set.n,
                provider=resolved_provider.identity,
                set_ref=resolved_set.reference.rendered,
            )
            breach = signal

        records.append(
            record.ScoreRecordData(
                run_id=spec.run_id,
                step=step,
                kind=kind,
                set_rendered=resolved_set.reference.rendered,
                set_ref=resolved_set.reference.structured,
                score=score,
                n=resolved_set.n,
                floor=floor.value,
                floor_layer=floor.layer,
                breached=signal is not None,
                provider=resolved_provider.identity,
                parameter_set_digest=getattr(resolved_provider, "parameter_set_digest", None),
                aggregation=spec.aggregation,
                decode=spec.decode,
                subsample_seed=resolved_set.subsample_seed,
                split_markers=resolved_set.markers,
                selector_hash=scored.selector_hash,
                grader_hash=scored.grader_hash,
                reproducible=reproducible,
                not_reproducible_reason=reason,
                out_of_cadence=out_of_cadence,
                provider_discontinuity=spec.provider_discontinuity,
                part_id=resolved_set.reference.part_id,
                outcomes=scored.outcomes,
                completions=tuple(completion.text for completion in completions),
                created_at=timestamp,
            )
        )

    if not records:
        raise refuse(
            "evaluation_set_unresolvable",
            f"No evaluation set is configured for kinds {list(kinds)} on run "
            f"{spec.run_id!r}. Refusing rather than recording an empty pass: an empty pass "
            f"reads as 'this ran and found nothing', which is a different and false claim.",
            run_id=spec.run_id,
            kinds=list(kinds),
        )

    block = record.write_pass(
        records,
        session=session,
        log_path=log_path,
        breach=breach,
        emit=emit,
        heartbeat=heartbeat,
    )
    return EvaluationOutcome(records=tuple(records), breach=breach, block=block)


def _for_kind(
    spec: EvaluationSpec, kind: str
) -> tuple[ResolvedSet | None, expectations.ResolvedExpectation | None]:
    if kind == KIND_TEST:
        return spec.test_set, spec.test_expectation
    if kind == KIND_RETENTION:
        return spec.retention_set, spec.retention_expectation
    raise refuse(
        "evaluation_kind_unknown",
        f"{kind!r} is not an evaluation kind; it is {KIND_TEST!r} or {KIND_RETENTION!r}.",
        kind=kind,
    )


def _assert_capabilities(provider: InferenceProvider, spec: EvaluationSpec) -> None:
    """Refuse a provider that does not declare what this pass needs (FR-004.6, FR-004.8).

    Two checks, and the second is the one with a verified precedent behind it. Generation is
    required of every provider. Beyond that, *"OpenAI-compatible" is a shape, not a
    guarantee*: miStudio's labelling service records in code that **miLLM ignores the OpenAI
    ``response_format`` parameter (verified: identical output with and without)** and moves
    the constraint into the prompt instead. Sending a parameter the server silently ignores
    produces a number whose meaning depends on which server answered - the exact failure the
    provider recording exists to prevent - so a pass declares what it depends on and this
    refuses rather than sending it hopefully.
    """
    declared = provider.capabilities() if hasattr(provider, "capabilities") else frozenset()
    if "generation" not in declared:
        raise refuse(
            "inference_provider_capability_undeclared",
            f"Provider {spec.provider_name!r} does not declare the 'generation' capability, "
            f"which every scoring pass needs. Refusing rather than trying it: a capability "
            f"inferred from a successful call is a capability nobody declared.",
            provider=spec.provider_name,
            declared=sorted(declared),
            needed="generation",
        )
    missing = sorted(spec.requires - declared)
    if missing:
        raise refuse(
            "inference_provider_capability_undeclared",
            f"This pass depends on {missing} being honoured and provider "
            f"{spec.provider_name!r} declares {sorted(declared)}. Refusing rather than "
            f"sending them hopefully: a parameter the server silently ignores produces a "
            f"number whose meaning depends on which server answered.",
            provider=spec.provider_name,
            declared=sorted(declared),
            needed=missing,
        )


def _reproducibility(spec: EvaluationSpec) -> tuple[bool, str | None]:
    """Mark the pass reproducible, **or not reproducible with a reason** (FR-004.18).

    Criterion 2 is written against exactly one case - *"with deterministic decoding and the
    same provider, exactly"* - and names every other case as a gap no requirement fills. So a
    sampled pass is marked, with the reason, rather than being quietly compared anyway; and a
    pass whose provider did not declare a parameter it needed never gets here, because
    :func:`_assert_capabilities` refuses it first.
    """
    if spec.decode.temperature > 0.0:
        return False, (
            f"decoding sampled at temperature {spec.decode.temperature}; criterion 2's "
            f"comparison is exact only under deterministic decoding, and no requirement "
            f"defines a tolerance for the sampled case"
        )
    if spec.decode.top_p < 1.0:
        return False, (f"decoding sampled with top_p {spec.decode.top_p}; see the temperature case")
    return True, None


def _generate(
    provider: InferenceProvider,
    resolved_set: ResolvedSet,
    spec: EvaluationSpec,
    *,
    is_cancelled: Callable[[], bool] | None,
    step: int,
) -> list[Completion]:
    """Generate every completion for one set, cancellable **between** chunks.

    There is deliberately no ``try``/``except`` around the generation that returns what was
    produced so far. That is mutation control **M-6**, and it is the change most likely to be
    proposed as robustness: a set score over the prompts that happened to answer is a
    confident number about an unknown subset, and it is indistinguishable from a real score
    on every surface this feature produces.
    """
    prompts = resolved_set.texts
    completions: list[Completion] = []
    for start in range(0, len(prompts), max(1, spec.batch_bound)):
        # THE BOUNDARY. Before a chunk is started, never inside a generate: a prompt is
        # scored and recorded, or not started (E-18).
        if is_cancelled is not None and is_cancelled():
            raise refuse(
                "evaluation_cancelled",
                f"The evaluation of {resolved_set.kind} at step {step} was cancelled after "
                f"{len(completions)} of {len(prompts)} prompts. **No set score is "
                f"recorded**: a score over an unknown subset is a fabricated number.",
                run_id=spec.run_id,
                step=step,
                kind=resolved_set.kind,
                completed=len(completions),
                total=len(prompts),
            )
        chunk = prompts[start : start + max(1, spec.batch_bound)]
        try:
            completions.extend(provider.generate(list(chunk), spec.decode))
        except Refusal as refusal:
            raise _as_evaluation_refusal(refusal, spec, step) from refusal

    if len(completions) != len(prompts):
        raise refuse(
            "inference_provider_reply_unparseable",
            f"The provider returned {len(completions)} completions for {len(prompts)} "
            f"prompts. Refusing rather than scoring what came back: results are joined to "
            f"prompts positionally, so a short list misattributes every score after the gap.",
            provider=spec.provider_name,
            asked=len(prompts),
            received=len(completions),
        )
    return completions


def _as_evaluation_refusal(refusal: Refusal, spec: EvaluationSpec, step: int) -> Refusal:
    """Re-raise a provider's refusal under this feature's code, **keeping the owner's**.

    F3's HTTP provider is shared with the judge path and names its transport failures
    ``judge_provider_unavailable``. A scoring pass is not a judge, and an operator who reads
    that code goes to the selector screen; FR-004.7's own codes are
    ``inference_provider_unreachable`` and ``inference_provider_reply_unparseable``. So the
    noun is corrected **and the original code travels in ``details.provider_code``**, because
    a re-coded refusal that discards its origin loses its owner.
    """
    if refusal.code in _TRANSPORT_CODES:
        code = "inference_provider_unreachable"
    elif refusal.code in _PARSE_CODES:
        code = "inference_provider_reply_unparseable"
    else:
        # Not a transport or parse failure - a budget refusal, a misconfiguration, an
        # unknown option. Those are already this feature's codes or are the provider's own
        # and meaningful as they stand, so they propagate unchanged.
        return refusal
    return refuse(
        code,
        f"The evaluation at step {step} failed at the provider: {refusal.message} "
        f"**No partial score is recorded** - a score over the prompts that happened to "
        f"answer is a confident number about an unknown subset.",
        run_id=spec.run_id,
        step=step,
        provider=spec.provider_name,
        provider_code=refusal.code,
        provider_details=refusal.details,
    )
