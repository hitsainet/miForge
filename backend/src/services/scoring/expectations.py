"""R-02.16's three expectation forms, onto **F3's** registry  (FR-004.13; E-9, E-10, E-11).

R-02.16's expectation is *"a selector or grader applied to the answer, or a reference answer
for exact or fuzzy match"*. All three resolve from F3's one live registry:

* a **selector** verdict - ``accept`` counts toward the score;
* a **grader** scalar - already in 0..1 by F3's contract;
* a **reference answer** - F3's ``reference_match`` type in ``exact`` or ``fuzzy`` mode.

**F4 implements no selector, no grader and no matcher.** The fuzzy metric is part of the
verdict and is F3's named, pinned, configuration-recorded choice; two implementations would
be two answers to one question.

## Two refusals before any generation

Both at **run submission**, not at the first prompt - R-02.48's rule that *"a run that cannot
proceed says so before it starts, not partway through"*. A test set's expectation has to
answer *at a step*, which rules out two configurations:

* **``human_review``** (E-10) - it *"writes candidates to a queue and reads decisions back"*,
  necessarily asynchronous, and returns ``undecided`` until a person answers. F3's verdict
  being **three-valued** is what makes this detectable rather than a hang. Refused by name as
  ``expectation_not_synchronous``, including as a **child of a composite**, because a
  composite containing a human child cannot answer at a step either.
* **embedding-mode reference matching** (E-9) - refused by the **contract**, at validation,
  before any F4 code runs. ``selector-config-v1.json``'s ``ReferenceMatchSelector`` declares
  ``mode`` as ``exact | fuzzy`` only and says why: *"Embedding similarity was offered by
  R-02.9's amendment and is DELIBERATELY ABSENT ... no provider and no source exists for it
  ... a selector that silently fell back to fuzzy matching when asked for embeddings would be
  the fabricated value ADR-018 forbids."* F4 **surfaces that refusal unchanged** and adds no
  embeddings capability (DEF-3).

  *Divergence, reported not resolved:* F4's four documents say this surfaces F3's
  ``embeddings_unsupported`` code. **There is no such code** - F3's ``reference_match``
  module says in terms *"There is no ``embeddings_unsupported`` code here"* because the
  schema refuses the mode first. The refusal an operator receives is
  ``selector_config_invalid`` naming the ``/mode`` pointer, which is strictly more useful,
  and nothing falls back to fuzzy.

## ``undecided`` is never counted as a reject

E-11: an ``undecided`` from any cause refuses the **evaluation** with the counts. Counting it
as a reject to make the arithmetic close is a fabricated number with extra steps, and F3's
contract forbids the inverse too - ``is_admitted`` is *"the only place admission is
decided"* and nothing but ``accept`` admits.

## F3's refusals propagate unchanged

``grader_score_out_of_range`` and ``selector_config_invalid`` are re-raised as they are.
**A re-coded refusal loses its owner**, and an operator who greps a code wants to land in the
module that produced it.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from core.errors import refuse
from schemas.selector import VerdictValue, is_admitted
from services.selectors import ensure_registered
from services.selectors.resolve import ResolvedSelector
from services.selectors.resolve import resolve as resolve_selector
from services.selectors.service import evaluate as evaluate_selectors
from services.selectors.types import EvaluationContext, SelectorForm

#: The selector kind that cannot answer at a step. F3's spelling, with an underscore -
#: ``schemas.selector.SelectorKind`` is the one vocabulary and F4's documents' hyphenated
#: ``human-review`` is a prose spelling, not a value.
ASYNCHRONOUS_KIND = "human_review"

#: Which form of F3's plugin an expectation reads. Not a new vocabulary:
#: ``services.selectors.types.SelectorForm`` is the enum and these are its two members as a
#: caller names them.
FORM_SELECTOR = "selector"
FORM_GRADER = "grader"


@dataclass(frozen=True, slots=True)
class ResolvedExpectation:
    """A resolved configuration plus which form of it this set reads."""

    resolved: ResolvedSelector
    form: str

    @property
    def kind(self) -> str:
        return str(self.resolved.kind)


@dataclass(frozen=True, slots=True)
class PromptOutcome:
    """One prompt's contribution to the score, with the provenance a drill-down needs."""

    index: int
    outcome: float
    verdict: str | None
    scalar: float | None
    reason_code: str | None


@dataclass(frozen=True, slots=True)
class ScoredCompletions:
    """Per-prompt outcomes and the two configuration hashes F3 published."""

    outcomes: tuple[PromptOutcome, ...]
    selector_hash: str | None
    grader_hash: str | None

    @property
    def values(self) -> list[float]:
        return [outcome.outcome for outcome in self.outcomes]


def resolve_expectation(
    config: Mapping[str, Any],
    *,
    form: str = FORM_SELECTOR,
    catalog: Mapping[str, Mapping[str, Any]] | None = None,
    source: str | None = None,
    run_target: Any = None,
) -> ResolvedExpectation:
    """Resolve a test-set expectation, refusing the two unusable shapes up front.

    Args:
        config: The selector or grader configuration body, as F10 imported it or as a pack
            slot cites it by hash. F4 authors none of these.
        form: ``selector`` or ``grader`` - which of F3's two forms this set reads.
        catalog: Other configured selectors by name, for a composite's children.
        source: Where a local configuration came from (R-02.40). F3 requires it without a
            handoff; ``"pack"`` for a pack-sourced configuration.
        run_target: The run's own inference target, for F3's R-02.11 guard. Passing it is
            what makes a self-judging configuration refuse **before the run's first step**.

    Raises:
        Refusal: ``expectation_not_synchronous`` (E-10) for ``human_review``, including as a
            composite child; ``expectation_form_unsupported`` for a form the resolved type
            does not offer; and anything F3's ``resolve`` refuses, **unchanged** - including
            ``selector_config_invalid`` for embedding mode (E-9) and
            ``judge_is_resident_model`` for a self-judging judge.
    """
    if form not in (FORM_SELECTOR, FORM_GRADER):
        raise refuse(
            "expectation_form_unsupported",
            f"{form!r} is not an expectation form; it is {FORM_SELECTOR!r} or " f"{FORM_GRADER!r}.",
            form=form,
        )
    _refuse_asynchronous(dict(config), catalog or {}, path=())

    ensure_registered()
    resolved = resolve_selector(
        dict(config),
        catalog={name: dict(body) for name, body in (catalog or {}).items()},
        source=source,
        run_target=run_target,
    )
    # A second check after resolution, against the RESOLVED tree rather than the raw bodies.
    # The pre-check reads configuration; this reads what the registry actually built, and the
    # two can differ if a catalog entry resolves differently than its body suggested. Two
    # cheap checks beat one that is trusted.
    _refuse_asynchronous_resolved(resolved)

    wanted = SelectorForm.SELECTOR if form == FORM_SELECTOR else SelectorForm.GRADER
    if wanted not in resolved.capability.forms:
        raise refuse(
            "expectation_form_unsupported",
            f"Selector type {resolved.kind!r} offers "
            f"{sorted(f.value for f in resolved.capability.forms)} and this test set reads "
            f"it as a {form}. Refusing rather than coercing: a rule's 1.0/0.0 admission "
            f"flag read as a graded scalar is a different measurement wearing the same "
            f"number.",
            kind=resolved.kind,
            form=form,
            offers=sorted(f.value for f in resolved.capability.forms),
        )
    return ResolvedExpectation(resolved=resolved, form=form)


def _refuse_asynchronous(
    config: Mapping[str, Any], catalog: Mapping[str, Mapping[str, Any]], *, path: tuple[str, ...]
) -> None:
    """Refuse ``human_review`` anywhere in a configuration tree, naming where it was found."""
    kind = str(config.get("kind") or "")
    name = str(config.get("name") or "")
    if kind == ASYNCHRONOUS_KIND:
        where = " -> ".join((*path, name)) if path else name
        raise refuse(
            "expectation_not_synchronous",
            f"A test-set expectation must answer at a step, and the {ASYNCHRONOUS_KIND!r} "
            f"selector cannot: it writes candidates to miDataworks' review queue and reads "
            f"decisions back, returning 'undecided' until a person answers. Use it as a "
            f"signal-source selector rather than as a test-set expectation. Found "
            f"at {where!r}.",
            type=ASYNCHRONOUS_KIND,
            at=where or name,
        )
    if kind == "composite":
        for child in config.get("children") or []:
            body = catalog.get(str(child))
            if body is None:
                # Unknown children are F3's `composite_child_unknown`, raised by its own
                # resolution. Not re-coded here.
                continue
            _refuse_asynchronous(body, catalog, path=(*path, name))


def _refuse_asynchronous_resolved(resolved: ResolvedSelector) -> None:
    """The same rule over the resolved tree, which is what will actually be evaluated."""
    if str(resolved.kind) == ASYNCHRONOUS_KIND:
        raise refuse(
            "expectation_not_synchronous",
            f"The resolved expectation is a {ASYNCHRONOUS_KIND!r} selector, which cannot "
            f"answer at a step.",
            type=ASYNCHRONOUS_KIND,
            at=resolved.name,
        )
    for child in resolved.children:
        _refuse_asynchronous_resolved(child)


def score_completions(
    expectation: ResolvedExpectation,
    prompts: Sequence[str],
    completions: Sequence[str],
    *,
    references: Sequence[str | None] | None = None,
    run_id: str | None = None,
    batch_id: str = "evaluation",
    is_cancelled: Any = None,
) -> ScoredCompletions:
    """Score completions through F3, and turn verdicts into per-prompt outcomes.

    Args:
        expectation: From :func:`resolve_expectation`.
        prompts: The prompts, positionally joined to ``completions``.
        completions: What the provider produced, in prompt order.
        references: Reference answers for a ``reference_match`` expectation, positionally.
            Passed through F3's per-item context under the key its plugin reads.
        run_id: Bound into F3's evaluation context for the log.
        batch_id: Identifies this evaluation to F3.
        is_cancelled: Cooperative cancellation, checked by F3 at each **completion**
            boundary as well as by this feature at each prompt chunk.

    Raises:
        Refusal: ``evaluation_incomplete`` (E-11) when any verdict is ``undecided``, with
            the counts; ``evaluation_grader_missing_score`` when a grader form returned no
            scalar; and anything F3 refuses, **unchanged** - a transport failure, an
            unparseable judge reply, ``grader_score_out_of_range``.
    """
    items = tuple(
        {"reference": reference} for reference in (references or [None] * len(completions))
    )
    result = evaluate_selectors(
        list(completions),
        expectation.resolved,
        EvaluationContext(
            run_id=run_id,
            batch_id=batch_id,
            items=items,
            is_cancelled=is_cancelled,
        ),
        prompts=list(prompts),
    )

    undecided = [
        index
        for index, record in enumerate(result.records)
        if record.verdict.value is VerdictValue.UNDECIDED
    ]
    if undecided:
        raise refuse(
            "evaluation_incomplete",
            f"{len(undecided)} of {len(result.records)} completions came back 'undecided', "
            f"so this set has no score. Refusing rather than counting an undecided as a "
            f"reject to make the arithmetic close: that is a confident number over a "
            f"half-answered set, and nothing may construct a path from 'undecided' to a "
            f"verdict it did not give.",
            undecided=len(undecided),
            total=len(result.records),
            prompt_indexes=undecided[:20],
        )

    outcomes: list[PromptOutcome] = []
    for index, record in enumerate(result.records):
        verdict = record.verdict
        if expectation.form == FORM_GRADER:
            if verdict.score is None:
                raise refuse(
                    "evaluation_grader_missing_score",
                    f"The grader form of {expectation.kind!r} returned no scalar for prompt "
                    f"{index}. Refusing rather than substituting the admission flag: an "
                    f"accept rate and a grader mean are different measurements, and "
                    f"recording one under the other's method name would make the number "
                    f"uninterpretable.",
                    kind=expectation.kind,
                    prompt_index=index,
                )
            outcome = float(verdict.score)
        else:
            # `is_admitted` is F3's single decision point, called rather than re-derived:
            # "no path from undecided to admitted" is enforced in one place that can be
            # tested, instead of being a rule four features are each trusted to honour.
            outcome = 1.0 if is_admitted(verdict) else 0.0
        outcomes.append(
            PromptOutcome(
                index=index,
                outcome=outcome,
                verdict=verdict.value.value,
                scalar=verdict.score,
                reason_code=record.reason_code,
            )
        )

    return ScoredCompletions(
        outcomes=tuple(outcomes),
        selector_hash=result.hashes.selector_hash,
        grader_hash=result.hashes.grader_hash,
    )
