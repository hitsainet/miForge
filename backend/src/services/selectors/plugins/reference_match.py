"""The reference-match selector  (FR-003.6; R-02.9, mechanism shared with R-02.16).

Compares a completion to a reference answer. **Two modes, ``exact`` and ``fuzzy``** - and
the third mode the FPRD describes is deliberately absent.

## Embedding similarity is CUT, and must not be reintroduced

``selector-config-v1.json``'s ``mode`` enum is ``["exact", "fuzzy"]``, and its own
description says why: R-02.9's amendment offered embedding similarity, **R-02.49's provider
abstraction is scoped to generation**, and the word *"embedding"* occurs **exactly once in
the whole of BRD-02** - in that amendment. No provider, no source, no mechanism. It is the
same shape as the missing load call: a capability named in prose that nothing implements.

So a configuration asking for ``embedding`` is refused **by the schema**, at validation,
before this module sees it. There is no ``embeddings_unsupported`` code here and no
embeddings capability on the provider port, because a refusal code for a mode the contract
cannot express would be dead code pretending to be a guard.

**And there is no fuzzy fallback, in any form.** A silent mode change would alter every
verdict while the R-02.10 configuration hash stayed identical - the worst available outcome
for a feature whose whole job is that selection is observable.

## The reference comes from the item, not the configuration

The configuration is ``{kind, name, config_sha256, mode, threshold}`` - there is no
``reference`` field, and correctly so: the reference answer belongs to the **test set item**
(R-02.16), not to the selector. So it arrives in the per-completion context. A missing
reference is a named refusal, never a reject: a reject would silently discard signal and
look like a model failing a test it was never given.

## The metric is part of the verdict

Swapping the string metric rescopes every fuzzy decision, so the metric is **named and
recorded on the record**. This release pins :class:`difflib.SequenceMatcher`'s ``ratio``,
from the standard library. *(The TDD named ``rapidfuzz``'s token-set ratio; it is not a
declared dependency and ``pyproject.toml`` is not this feature's file, so the stdlib metric
is used and recorded by name. Reported, not silently substituted - the point of recording
the metric is exactly that a reader can see which one answered.)*
"""

from __future__ import annotations

from difflib import SequenceMatcher
from typing import Any

from core.errors import refuse
from schemas.selector import SelectorKind, Verdict, VerdictValue
from services.selectors.resolve import ResolvedSelector

#: The pinned fuzzy metric, by name, recorded on every fuzzy verdict.
FUZZY_METRIC = "difflib.SequenceMatcher.ratio"

#: Keys the per-completion context may carry a reference under. Two spellings accepted
#: because F4's test sets and F7's dry-run arrived at different ones; both are explicit and
#: neither is a guess.
_REFERENCE_KEYS = ("reference", "reference_answer")


def references_from(context: dict[str, Any]) -> tuple[str, ...]:
    """Every reference answer in ``context``, as a tuple. Empty when there is none.

    A list of references is accepted because a test item may have several acceptable
    answers; matching **any** of them is a match, and the reason names the best score.
    """
    for key in _REFERENCE_KEYS:
        value = context.get(key)
        if value is None:
            continue
        if isinstance(value, str):
            return (value,)
        if isinstance(value, (list, tuple)):
            return tuple(str(item) for item in value)
    plural = context.get("references")
    if isinstance(plural, (list, tuple)):
        return tuple(str(item) for item in plural)
    if isinstance(plural, str):
        return (plural,)
    return ()


class ReferenceMatchSelector:
    """Satisfies :class:`schemas.selector.Selector` and offers a grader form."""

    kind: SelectorKind = "reference_match"

    def __init__(self, resolved: ResolvedSelector) -> None:
        self._resolved = resolved

    @property
    def name(self) -> str:
        return self._resolved.name

    @property
    def mode(self) -> str:
        return str(self._resolved.body["mode"])

    @property
    def threshold(self) -> float | None:
        value = self._resolved.body.get("threshold")
        return None if value is None else float(value)

    def prepare(self, resolved: ResolvedSelector, *, run_target: Any = None) -> None:
        """Refuse a configuration whose threshold and mode disagree. Needs no model."""
        del run_target
        mode = str(resolved.body["mode"])
        threshold = resolved.body.get("threshold")
        if mode == "fuzzy" and threshold is None:
            raise refuse(
                "reference_match_threshold_missing",
                "A fuzzy reference-match selector needs an explicit threshold: the "
                "similarity score means nothing without the line it is compared against, "
                "and miForge will not choose one. Set 'threshold' in the configuration.",
                mode=mode,
                metric=FUZZY_METRIC,
            )
        if mode == "exact" and threshold is not None:
            # Refused rather than ignored. An operator who set a threshold expects
            # tolerance; silently giving them byte-equality would mean every near-miss is
            # rejected while the configuration says otherwise.
            raise refuse(
                "reference_match_threshold_not_applicable",
                "An exact reference-match selector compares for equality, so a threshold "
                "has no meaning and would mislead: nothing between 0 and 1 can change the "
                "verdict. Remove the threshold, or set mode to 'fuzzy'.",
                mode=mode,
                threshold=threshold,
            )
        resolved.prepared = {"mode": mode, "threshold": threshold, "metric": FUZZY_METRIC}

    def evaluate(self, *, prompt: str, completion: str, context: dict[str, Any]) -> Verdict:
        """One verdict, with the mode and the score in the reason."""
        del prompt
        references = self._require_references(context)
        score, best = self.score_against(completion, references)
        mode = self.mode

        if mode == "exact":
            matched = score == 1.0
            reason = (
                "exact match to reference"
                if matched
                else f"no exact match among {len(references)} reference(s)"
            )
            return self._verdict(matched, reason, score, mode)

        threshold = self.threshold
        assert threshold is not None  # prepare() refused a fuzzy config without one
        matched = score >= threshold
        reason = (
            f"similarity {score:.2f} vs threshold {threshold:.2f} ({mode}, {FUZZY_METRIC})"
            if matched
            else f"no reference within {threshold:.2f} (best {score:.2f}, {mode}, {FUZZY_METRIC})"
        )
        del best
        return self._verdict(matched, reason, score, mode)

    def grade(self, *, prompt: str, completion: str, context: dict[str, Any]) -> float:
        """The similarity, as a score in 0..1 (FR-003.9).

        The same number the selector thresholds, so a grader and a selector over one
        configuration can never disagree about what the completion scored.
        """
        del prompt
        references = self._require_references(context)
        score, _ = self.score_against(completion, references)
        return score

    def score_against(self, completion: str, references: tuple[str, ...]) -> tuple[float, str]:
        """``(best score, best reference)``. Exact mode scores 1.0 or 0.0, never between.

        No normalisation is applied in ``exact`` mode - not case-folding, not whitespace
        trimming. The contract gives this type no normalisation field, and *"exact after
        case-folding is a different verdict from exact"*: choosing one here would be a
        verdict policy chosen by whoever wrote the line.
        """
        mode = self.mode
        best_score = 0.0
        best_reference = ""
        for reference in references:
            if mode == "exact":
                score = 1.0 if completion == reference else 0.0
            else:
                score = SequenceMatcher(None, completion, reference).ratio()
            if score >= best_score:
                best_score, best_reference = score, reference
        return best_score, best_reference

    def _require_references(self, context: dict[str, Any]) -> tuple[str, ...]:
        references = references_from(context)
        if not references:
            raise refuse(
                "reference_missing",
                "A reference-match selector was given a completion with no reference "
                "answer to compare it against. Refusing rather than rejecting: a reject "
                "would discard the completion and look like the model failing a test it "
                "was never given. Supply 'reference' on the item (R-02.16).",
                selector=self.name,
                looked_for=list(_REFERENCE_KEYS),
            )
        return references

    def _verdict(self, matched: bool, reason: str, score: float, mode: str) -> Verdict:
        return Verdict(
            value=VerdictValue.ACCEPT if matched else VerdictValue.REJECT,
            selector=self.name,
            label=f"reference_match_{mode}_{'hit' if matched else 'miss'}",
            reason=reason,
            score=round(score, 6),
        )


def factory(resolved: ResolvedSelector) -> ReferenceMatchSelector:
    return ReferenceMatchSelector(resolved)
