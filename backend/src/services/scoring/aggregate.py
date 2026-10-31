"""Per-prompt outcomes into one number, by a **named** method  (FR-004.15; E-8).

## The method name is the contract's, not this module's

``step-record-v1.json``'s ``EvaluationBlock.aggregation`` is an **enum of exactly five
values** - ``mean``, ``median``, ``pass_at_k``, ``exact_match_rate``, ``weighted_mean`` - and
its own description says why it exists: *"'mean' and 'pass@k' are not the same claim."* So
this module's method table is that enum and nothing else. F4's own documents named the
methods ``accept_rate`` and ``mean_normalised``; **the contract wins and those spellings are
reported as a divergence**, because the run log is where a score's aggregation method
actually has to be recorded and an undeclared value would be refused by the validator.

An accept rate over accept/reject verdicts **is** ``mean`` over the admitted indicator - the
arithmetic is identical and the contract's name for it is ``mean``.

## Nothing is clamped, ever

``ScoreRecord.score`` is bounded 0..1 in the vendored manifest (parsed, not assumed).
Clamping a 1.4 to 1.0 silently changes the number the manifest claims and hides the true
fault, which is that the aggregation or the grader's declared range is wrong. So an
out-of-range aggregate **refuses with the value and the range** (E-8) - the shape ADR-018
requires: *refuse with a reason rather than guess.*

## ``undecided`` is never counted

A completion whose verdict is ``undecided`` is not a reject with extra steps. Counting it as
one makes a half-answered set into a confident number, so this module is handed only decided
outcomes and :mod:`services.scoring.expectations` refuses the pass when any are not (E-11).
"""

from __future__ import annotations

from collections.abc import Sequence
from statistics import median as _median

from core.errors import refuse

#: The five values ``EvaluationBlock.aggregation`` declares. **Read as a constant and
#: asserted against the schema file by test** rather than retyped and hoped for: that
#: assertion is what will move this list when the contract moves it, instead of letting a
#: stale name reach the validator.
AGGREGATION_MEAN = "mean"
AGGREGATION_MEDIAN = "median"
AGGREGATION_PASS_AT_K = "pass_at_k"
AGGREGATION_EXACT_MATCH_RATE = "exact_match_rate"
AGGREGATION_WEIGHTED_MEAN = "weighted_mean"

DECLARED_METHODS: tuple[str, ...] = (
    AGGREGATION_MEAN,
    AGGREGATION_MEDIAN,
    AGGREGATION_PASS_AT_K,
    AGGREGATION_EXACT_MATCH_RATE,
    AGGREGATION_WEIGHTED_MEAN,
)

#: The methods this release implements. ``pass_at_k`` is declared by the contract and
#: **refused by name** here: it needs k independent samples per prompt, and no requirement in
#: R-02.16 - R-02.19 asks for repeated sampling. Refusing is the honest answer; implementing
#: it as "mean over one sample" would be a number that claims pass@k and is not.
SUPPORTED_METHODS: frozenset[str] = frozenset(
    {
        AGGREGATION_MEAN,
        AGGREGATION_MEDIAN,
        AGGREGATION_EXACT_MATCH_RATE,
        AGGREGATION_WEIGHTED_MEAN,
    }
)


def aggregate(
    method: str,
    outcomes: Sequence[float],
    *,
    weights: Sequence[float] | None = None,
) -> float:
    """One score in 0..1 from per-prompt outcomes.

    Args:
        method: One of :data:`DECLARED_METHODS`. An unknown name **refuses**; it is not
            defaulted to ``mean``, because the method is recorded beside the number and two
            scores aggregated differently are not the same measurement.
        outcomes: Per-prompt outcomes, each already in 0..1 - an admitted indicator for a
            selector, a grader's scalar for a grader.
        weights: Per-prompt weights, required by ``weighted_mean`` and refused for anything
            else. Refused rather than ignored: a caller passing weights to ``mean`` believes
            they are in force.

    Raises:
        Refusal: ``aggregation_method_unknown`` for a name the contract does not declare;
            ``aggregation_method_unsupported`` for ``pass_at_k``;
            ``evaluation_set_empty`` for no outcomes (there is no ``n: 0`` score);
            ``score_out_of_range`` when an **input** or the **result** falls outside 0..1.
            Never clamped.
    """
    if method not in DECLARED_METHODS:
        raise refuse(
            "aggregation_method_unknown",
            f"{method!r} is not one of step-record-v1.json's aggregation methods "
            f"{list(DECLARED_METHODS)}. Refusing rather than defaulting to 'mean': the "
            f"method is recorded beside the number, and an accept rate and a normalised "
            f"grader mean are both '0.84' and are not the same measurement.",
            method=method,
            declared=list(DECLARED_METHODS),
        )
    if method not in SUPPORTED_METHODS:
        raise refuse(
            "aggregation_method_unsupported",
            f"{method!r} is declared by the run-log contract but not implemented: it needs "
            f"k independent samples per prompt, and miForge takes one "
            f"sample per prompt. Computing it over one sample would be a number "
            f"that claims pass@k and is not.",
            method=method,
            supported=sorted(SUPPORTED_METHODS),
        )
    if not outcomes:
        raise refuse(
            "evaluation_set_empty",
            "There are no per-prompt outcomes to aggregate. ScoreRecord.n has minimum 1, "
            "so no score exists over zero prompts.",
            method=method,
        )

    for index, value in enumerate(outcomes):
        _assert_in_range(value, where=f"outcome {index}", method=method)

    if method == AGGREGATION_WEIGHTED_MEAN:
        score = _weighted_mean(outcomes, weights, method=method)
    else:
        if weights is not None:
            raise refuse(
                "aggregation_weights_not_applicable",
                f"Weights were supplied for aggregation method {method!r}, which does not "
                f"use them. Refusing rather than ignoring them: a caller that passed "
                f"weights believes they are in force, and the number would not be the one "
                f"they asked for.",
                method=method,
            )
        if method == AGGREGATION_MEAN:
            score = sum(outcomes) / len(outcomes)
        elif method == AGGREGATION_MEDIAN:
            score = float(_median(outcomes))
        else:
            # exact_match_rate: the proportion of prompts that matched exactly. A reference
            # match in `exact` mode produces 1.0 or 0.0, so this and `mean` agree there -
            # and they are still different claims over a grader's continuous scalar, which
            # is why both names exist in the contract.
            score = sum(1.0 for value in outcomes if value >= 1.0) / len(outcomes)

    _assert_in_range(score, where="the aggregate", method=method)
    return float(score)


def _weighted_mean(
    outcomes: Sequence[float], weights: Sequence[float] | None, *, method: str
) -> float:
    """A weighted mean, refusing a missing, mis-sized or zero-sum weight vector."""
    if weights is None:
        raise refuse(
            "aggregation_weights_missing",
            f"Aggregation method {method!r} needs one weight per prompt and none were "
            f"supplied. Refusing rather than weighting everything equally: that is 'mean' "
            f"under another name, and the recorded method would then describe a "
            f"computation that did not happen.",
            method=method,
        )
    if len(weights) != len(outcomes):
        raise refuse(
            "aggregation_weights_mismatch",
            f"{len(weights)} weights were supplied for {len(outcomes)} outcomes. Refusing "
            f"rather than padding: weights are joined to prompts positionally, so a "
            f"mis-sized vector misweights every prompt after the gap.",
            method=method,
            weights=len(weights),
            outcomes=len(outcomes),
        )
    if any(weight < 0.0 for weight in weights):
        raise refuse(
            "aggregation_weights_invalid",
            "A negative weight was supplied. A negative weight can push the aggregate "
            "outside 0..1, which the manifest cannot express.",
            method=method,
        )
    total = sum(weights)
    if total <= 0.0:
        raise refuse(
            "aggregation_weights_invalid",
            "The supplied weights sum to zero, so there is no weighted mean. Refusing "
            "rather than falling back to an unweighted mean, which would record a method "
            "that was not used.",
            method=method,
        )
    return sum(value * weight for value, weight in zip(outcomes, weights, strict=True)) / total


def _assert_in_range(value: float, *, where: str, method: str) -> None:
    """Refuse a value outside 0..1. **Never clamp** (E-8).

    ``ScoreRecord.score`` is ``minimum: 0, maximum: 1`` in the vendored manifest. A clamp
    would make the manifest's number differ from the computation and hide the real fault -
    an aggregation that is wrong, or a grader whose declared range is wrong.
    """
    if value != value or value in (float("inf"), float("-inf")):
        raise refuse(
            "score_out_of_range",
            f"{where} is not a finite number ({value}) under aggregation {method!r}. A "
            f"non-finite score cannot be written as JSON - no parser is required to accept "
            f"the literal NaN - so it is refused at the point it is computed.",
            method=method,
            value=str(value),
            range=[0.0, 1.0],
        )
    if not 0.0 <= value <= 1.0:
        raise refuse(
            "score_out_of_range",
            f"{where} is {value} under aggregation {method!r}, outside the 0..1 range "
            f"ScoreRecord.score declares. Refusing rather than clamping: a clamped number "
            f"differs from the computation the manifest claims, and the real fault is the "
            f"aggregation or the grader's declared range, not the bound.",
            method=method,
            value=value,
            range=[0.0, 1.0],
        )
