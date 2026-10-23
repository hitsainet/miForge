"""A miDataworks decision becomes a verdict  (F10 task 7.2; FR-010.24, R-02.36).

**Pure.** Values in, a :class:`schemas.selector.Verdict` out. No I/O, no ORM, no ``httpx``.
It is the module most likely to be got subtly wrong - five cases, one of which is an
*absence* - and a pure function is the one that can be exhaustively tested without a server.
If this module ever imports :mod:`services.review.binding`, the dependency has gone the
wrong way.

**F10 declares no verdict type.** The verdict is the gate contract's three-valued
``VerdictValue`` (``accept``/``reject``/``undecided``) and the mapping is the gate
contract's :data:`schemas.selector.DECISION_TO_VERDICT`, consumed here and **not
re-derived**. Two mappings of one decision set is how ``flag`` ends up meaning accept in
one of them. An earlier draft of this feature's FTID specified a *fourth* vocabulary - a
local ``accept|reject|label|pending`` enum - which its own FPRD and FTDD both forbade; it
is not coming back.

**``override`` maps to ``undecided``, carrying the label.** Settled in the gate contract:
miDataworks' ``label_set`` is 2-64 free strings, so for a label set that is not
accept/reject-shaped, reading "override" as an admission is precisely the silent default
FR-010.24 forbids. A human looked and said something the machine cannot interpret, so the
candidate stays out of the admitted count and the label is preserved for a person to read.

## The case that is an absence in the data and a value in the model

No decision yet is **``undecided``**, and it is the first case to write a test for. A
candidate with no decision is simply absent from the read-back - so the absence has to be
turned into a value *here*, by :func:`verdict_for_absent`, rather than being modelled as
"not in the accepted list". An absence invites a default; a value does not.

Concretely: ``accepted = [c for c, v in zip(cs, vs) if is_admitted(v)]`` is correct, while
``accepted = [c for c in cs if c.external_id not in rejected]`` is the bug.

## OPEN - one decision kind the gate contract does not map, and it is a real one

``DECISION_TO_VERDICT`` holds ``accept``, ``reject``, ``override`` and **``skip``**.
miDataworks' shipped vocabulary is ``accept``, ``override``, **``flag``** and ``reject`` -
``DECISIONS = ("accept", "override", "flag", "reject")`` in its
``services/review/decision_rules.py``, with ``AGENT_DECISIONS = {"accept", "flag"}`` and
``NEEDS_REASON = {"override", "flag", "reject"}``; its ``DecisionIn`` body type is
``Literal["accept", "override", "flag", "reject"]``. **The string ``skip`` does not occur
anywhere in that repository's review code.**

So today a ``flag`` - *"leave it flagged until an operator decides again"*, the one decision
that most obviously means `undecided` - arrives here and
:func:`schemas.selector.map_decision` **refuses** it as unrecognised. That refusal is loud,
named and recoverable, which is why this module propagates it rather than papering over it
with a local mapping: a second mapping is the defect this module's whole docstring is about.
But it is wrong, and the fix is one line in the gate contract -
``"flag": VerdictValue.UNDECIDED`` - with ``"skip"`` removed or kept as a harmless alias.
The gate contract is final and not F10's to edit, so this is reported rather than patched,
and ``test_review_queue_verdicts.py`` carries a strict ``xfail`` that turns red the moment
it is fixed.
"""

from __future__ import annotations

import logging
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

from schemas.selector import DECISION_TO_VERDICT, Verdict, VerdictValue, map_decision

log = logging.getLogger("miforge.review.verdicts")

#: The selector name recorded on every verdict this module produces. One string, so a
#: batch record's provenance cannot disagree with the registry's name for the type.
SELECTOR_NAME = "human_review"

#: miDataworks' decision vocabulary, as read from its ``decision_rules.DECISIONS``. Stated
#: here so a test can assert the gate contract's map against the **real** vocabulary rather
#: than against itself - which is the check that found the ``skip``/``flag`` discrepancy in
#: the module docstring.
MIDATAWORKS_DECISIONS: tuple[str, ...] = ("accept", "override", "flag", "reject")


@dataclass(frozen=True, slots=True)
class Anomaly:
    """Something read back that miForge records and does not act on.

    Recorded rather than silently dropped (E-16, E-17). Neither of these admits a
    candidate: an anomaly is a thing a person needs to see, not a thing a run decides on.
    """

    code: str
    external_id: str
    detail: str

    def as_dict(self) -> dict[str, Any]:
        return {"code": self.code, "external_id": self.external_id, "detail": self.detail}


def verdict_for_absent() -> Verdict:
    """The verdict for a candidate with **no decision yet**.

    Separated into its own named function because it is the one case that exists only in
    the model: there is no decision record to map, so something has to turn the absence
    into ``undecided`` deliberately. A caller that reached for a default here would have
    reached for ``accept``.
    """
    return Verdict(value=VerdictValue.UNDECIDED, selector=SELECTOR_NAME)


def verdict_for(
    decision: str,
    *,
    label: str | None = None,
    label_set: Sequence[str] | None = None,
    external_id: str = "",
) -> tuple[Verdict, Anomaly | None]:
    """Map one decision, returning the verdict and any anomaly worth recording.

    Args:
        decision: miDataworks' decision kind, exactly as read back.
        label: ``override_label``, when there is one.
        label_set: The queue's label set, to check an override's label against. Optional
            because the check is the queue's own rule and miDataworks enforces it on the
            way in; miForge records a mismatch as an anomaly rather than silently trusting
            it.
        external_id: For the anomaly record.

    Returns:
        ``(verdict, anomaly or None)``.

    Raises:
        Refusal: ``review_decision_unrecognised`` from the gate contract, for a decision
            kind the shared map does not know. **Propagated, not swallowed** - see the
            module docstring's OPEN note. Logged first, so an operator reading the log
            sees which kind arrived even if the refusal is caught upstream.
    """
    anomaly: Anomaly | None = None

    if label is not None and label_set is not None and label not in label_set:
        # Not admitted and not dropped. An override whose label falls outside the queue's
        # own set means miForge and miDataworks disagree about the question being asked.
        anomaly = Anomaly(
            code="override_label_outside_label_set",
            external_id=external_id,
            detail=f"label {label!r} is not in the queue's label set {list(label_set)}",
        )
        log.warning(
            "review_anomaly",
            extra={
                "code": "override_label_outside_label_set",
                "external_id": external_id,
                "label": label,
            },
        )

    if decision not in DECISION_TO_VERDICT:
        log.error(
            "review_decision_unrecognised",
            extra={
                "code": "review_decision_unrecognised",
                "decision": decision,
                "external_id": external_id,
                "known": sorted(DECISION_TO_VERDICT),
            },
        )

    # No `else: ACCEPT`, no `.get(..., ACCEPT)`. The gate contract's function refuses an
    # unknown kind, and guessing here would change what the model trains on.
    return map_decision(decision, label=label, selector=SELECTOR_NAME), anomaly


def admits(verdict: Verdict) -> bool:
    """Whether this verdict admits its candidate.

    A one-line delegation to :func:`schemas.selector.is_admitted` on purpose: *the only
    place admission is decided* lives in the gate contract, and every consumer calling it
    is what makes "no path from undecided to admitted" enforceable in one place rather
    than a rule four features are each trusted to honour.
    """
    from schemas.selector import is_admitted

    return is_admitted(verdict)
