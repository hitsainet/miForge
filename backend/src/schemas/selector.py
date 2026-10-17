"""The selector/grader contract  (foundation task 3.1 and 3.1a; R-02.9, R-02.10, R-02.11).

**This is a gate contract. Consumed by F3, F4, F7 and F10. Do not re-derive it.**

A selector decides whether a completion is admitted into training. A grader scores one.
Both are pure functions of a completion plus their configuration - which is why F3 needs
nothing from F1: scoring a model does not require training one.

## The verdict has THREE values, and the reason is structural

``accept``, ``reject``, ``undecided``. Two values cannot express one of R-02.9's own five
selector types: **human review** writes candidates to a queue and reads decisions back,
which is necessarily asynchronous, so at the moment the selector returns there is a third
outcome - *no decision yet*. Three features reached that conclusion independently and
none of them could be built against two values:

* **F3** (FR-003.3, OQ-1) - its own evaluate path.
* **F10** (FR-010.24, AC-010.9) - miDataworks' review queue is asynchronous and F10's own
  requirement **forbids defaulting**, so without a third value it would have to invent one.
* **F2** - which wrote the handling as a prohibition: **no path from pending to admitted.**

They are named here so the type is not reopened as one feature's preference.

**The spelling is ``undecided``, and that is part of what this contract settles.** The gate
task once offered ``undecided``/``pending`` as alternatives and the ambiguity propagated:
one feature's TID had specified a *fourth* vocabulary of its own, a four-member enum
including ``label`` and ``pending``, which its own FPRD and FTDD forbade. So: *"pending" is
the **condition of the candidate**, may be said in prose and UI copy, and is **not** a value
of this type.* No consumer defines a verdict type of its own.

**No consumer may invent a route from ``undecided`` to admitted.** That is the invariant the
third value exists to protect, and :func:`is_admitted` is the only place it is decided.
"""

from __future__ import annotations

from enum import StrEnum
from typing import Any, Literal, Protocol, runtime_checkable

from pydantic import BaseModel, ConfigDict, Field, model_validator


class VerdictValue(StrEnum):
    """The three-valued verdict. A ``StrEnum`` so it serialises as its own name."""

    ACCEPT = "accept"
    REJECT = "reject"
    #: No decision is available *yet* - the human-review queue has not answered. It is
    #: **not** a soft reject and **not** a soft accept: a run that treated it as either
    #: would be training on an answer nobody gave.
    UNDECIDED = "undecided"


class Verdict(BaseModel):
    """One selector's answer about one completion.

    ``label`` rides **beside** the value, never as a fourth enum member. That is how a
    human decision of kind ``override`` is represented: the value stays faithful to what
    is known and the label carries what was said.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    value: VerdictValue
    #: The selector that produced this, for the batch record's provenance.
    selector: str
    #: A free-text label from a human decision, or a rule's reason code. Present on any
    #: value. miDataworks' ``label_set`` is 2-64 free strings, so this is NOT an enum.
    label: str | None = None
    #: Why, in one line, for the run log and the UI. Never a score on its own: ADR-012
    #: requires every number to name its scale and sample.
    reason: str | None = None
    #: Scores in 0..1 when the selector computes one (reference-match similarity, a judge
    #: model's confidence). ``None`` when the type does not produce one.
    score: float | None = Field(default=None, ge=0.0, le=1.0)

    @model_validator(mode="after")
    def _undecided_carries_no_score(self) -> Verdict:
        """An ``undecided`` verdict with a score invites exactly the forbidden inference.

        If a pending candidate carried ``score=0.9``, some consumer would eventually
        threshold it and admit the completion - reconstructing the route from undecided to
        admitted that this contract exists to forbid. So the combination is refused at the
        type, not left to each consumer's discipline.
        """
        if self.value is VerdictValue.UNDECIDED and self.score is not None:
            raise ValueError(
                "an undecided verdict must not carry a score: a score on a pending "
                "candidate is how a consumer reconstructs the forbidden path from "
                "undecided to admitted (foundation task 3.1a)"
            )
        return self


def is_admitted(verdict: Verdict) -> bool:
    """**The only place admission is decided.** ``accept`` admits; nothing else does.

    A one-line function on purpose. Every consumer calls this rather than comparing to a
    value itself, so that "no path from undecided to admitted" is enforced in one place
    that can be tested, instead of being a rule four features are each trusted to honour.
    """
    return verdict.value is VerdictValue.ACCEPT


#: The five selector types R-02.9 names. ``reference_match`` and ``composite`` were added
#: on 2026-10-13: the UI mockup assumed both, and reference-match was already implied by
#: R-02.16's test sets.
SelectorKind = Literal["rule", "judge_model", "reference_match", "human_review", "composite"]


class SelectorConfigBase(BaseModel):
    """Common head of every selector configuration.

    The configuration itself is **authored and versioned in miDataworks** and cited here
    by hash (R-02.7, R-02.9 as amended); miForge runs it. ``config_sha256`` is that
    citation, and it is what a checkpoint manifest's ``selector_hash`` records.
    """

    model_config = ConfigDict(extra="forbid")

    kind: SelectorKind
    name: str
    #: Hash of the miDataworks-authored configuration body, by the one canonical function
    #: (ADR-005). Null only for a selector configured inline during a pack dry-run.
    config_sha256: str | None = None


@runtime_checkable
class Selector(Protocol):
    """What every selector implementation provides.

    Deliberately narrow. A selector is a pure function of a completion and its own
    configuration: it does not see the model, the step, or the run. That narrowness is
    what makes F3 independent of F1 and what makes a dry-run (R-02.39) possible before a
    run exists.

    **The asynchronous types still satisfy this signature.** ``human_review`` returns
    ``undecided`` immediately, having enqueued the candidate; the decision arrives later
    through F10 and is reconciled by the caller. The Protocol does not grow an async
    variant for it, because a second shape is a second contract.
    """

    @property
    def name(self) -> str:
        """Stable identity, recorded on every verdict."""
        ...

    @property
    def kind(self) -> SelectorKind: ...

    def evaluate(self, *, prompt: str, completion: str, context: dict[str, Any]) -> Verdict:
        """Judge one completion. **Refuses rather than guessing** (ADR-018).

        A selector that cannot reach what it needs - an unconfigured judge provider, an
        unreachable review queue - raises a named ``Refusal``. It must never fall back to
        a default verdict: a fabricated accept trains on an answer nobody gave, and a
        fabricated reject silently discards signal. Both are invisible.
        """
        ...


@runtime_checkable
class Grader(Protocol):
    """A grader scores a completion rather than admitting it (R-02.11).

    Separate from :class:`Selector` because the two answer different questions and R-02.11
    makes the in-process provider unusable as a judge - so a type that conflated them
    would make one of the two unimplementable on the default path.
    """

    @property
    def name(self) -> str: ...

    def grade(self, *, prompt: str, completion: str, context: dict[str, Any]) -> float:
        """A score in 0..1. Refuses rather than returning a sentinel."""
        ...


# --------------------------------------------------------------------------------------
# The human-decision mapping - OPERATOR DECISION B, recorded here as the single statement
# --------------------------------------------------------------------------------------

#: How a miDataworks review decision becomes a verdict value.
#:
#: **``override`` maps to ``undecided``, always, carrying the label.** This was the open
#: question: only one layer of one feature had ever named a value for ``override``, and it
#: said ``accept``, while that feature's own FPRD, its FTDD and both of F3's statements
#: named a label and no value at all.
#:
#: ``undecided`` is the answer because miDataworks' ``label_set`` is **2-64 free strings**
#: (verified in that repo's model: ``Field(min_length=2, max_length=64)``). For a label set
#: that is not accept/reject-shaped, reading "override" as an admission is precisely the
#: silent default F10's own FR-010.24 forbids - it would train on an answer nobody gave.
#: A human looked and said something the machine cannot interpret, so the candidate stays
#: out of the admitted count and the label is preserved for a person to read.
#:
#: If the binary case later proves common enough to matter, the refinement is
#: "``accept`` when the label set is binary, ``undecided`` otherwise" - but that needs a
#: definition of "binary" that nothing currently has, and guessing one here would be the
#: same defect in a new place.
#: **Corrected 2026-10-14, and the error was mine.** This map held ``"skip"`` and omitted
#: ``"flag"``. miDataworks' shipped vocabulary is
#: ``DECISIONS = ("accept", "override", "flag", "reject")``
#: (``services/review/decision_rules.py``), and **the string ``skip`` occurs nowhere in
#: their review code** - I invented a member and asserted it as theirs, which is the
#: twelfth instance of this project's signature defect and the second of mine today.
#:
#: It was not a harmless extra key. ``AGENT_DECISIONS = {"accept", "flag"}``, so for an
#: *agent* reviewer ``flag`` is **the** non-accept outcome - and it arrived here as a
#: ``review_decision_unrecognised`` refusal. The refusal was loud and safe, which is why
#: nothing broke; it was also wrong about the commonest decision on that surface.
#:
#: ``flag`` maps to ``UNDECIDED`` on the merits, not merely to fill the gap: their own rule
#: is that *any flag leaves the row flagged until an operator decides again*, and
#: ``NEEDS_REASON = {"override", "flag", "reject"}`` means it always carries a reason - so
#: it is precisely "a human looked and the answer is not yet accept or reject", which is
#: what the third value is for. ``skip`` is **removed rather than kept as an alias**: an
#: invented key retained for safety is how the fiction survives the correction.
#:
#: Found by F10 checking this map against that repository's source instead of against this
#: comment. The rule at the top of AGENT.md, again: verify that a cited source says what
#: you claim before building on it.
DECISION_TO_VERDICT: dict[str, VerdictValue] = {
    "accept": VerdictValue.ACCEPT,
    "reject": VerdictValue.REJECT,
    "override": VerdictValue.UNDECIDED,
    "flag": VerdictValue.UNDECIDED,
}


def map_decision(decision: str, *, label: str | None, selector: str) -> Verdict:
    """Turn a human decision into a :class:`Verdict`. **Refuses an unknown kind.**

    An unrecognised decision kind is a refusal, not a default. miDataworks may add kinds;
    if one arrives that this map does not know, the honest outcome is a named failure that
    an operator can see, not a verdict invented on its behalf.
    """
    from core.errors import refuse

    value = DECISION_TO_VERDICT.get(decision)
    if value is None:
        raise refuse(
            "review_decision_unrecognised",
            f"miDataworks returned decision {decision!r}, which this release does not map "
            f"to a verdict. Refusing rather than defaulting: a guessed verdict trains on "
            f"an answer nobody gave. Known kinds: {sorted(DECISION_TO_VERDICT)}.",
            decision=decision,
            known=sorted(DECISION_TO_VERDICT),
        )
    return Verdict(value=value, selector=selector, label=label)
