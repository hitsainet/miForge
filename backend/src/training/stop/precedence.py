"""Which reason is recorded when two conditions fire at once  (FR-006.27; derived, DEF-5).

**R-02.25 defines no precedence.** Verified against the BRD: it lists five conditions and
says nothing about simultaneity. Without a fixed order the recorded ``stop_reason`` depends
on iteration order, so the same run stopped by the same two conditions can report different
reasons on two machines - and the reason is what an operator acts on. Reported as **DEF-5**;
this module is the fixed order and it is one tuple and one lookup.

## The order, and why it is this order

``operator_stop`` -> ``retention_floor`` -> ``test_threshold`` -> ``loss_plateau`` ->
``step_count``.

The operator's explicit act comes first because it is the one reason a human asserted. The
correctness floor comes second because *"the run stopped because general capability fell
below the floor"* is a different and far more important sentence than *"the run reached its
target"* - and a run that hit both wants the floor named, since that is the finding. The
three optimistic reasons follow, with the step budget last: reaching the step count is the
least informative thing that can end a run, and anything else that fired at the same
boundary says more.

**Every condition that fired is recorded** on the boundary block regardless (FR-006.16),
so nothing is lost by the choice - only the single-valued ``stop_reason`` is decided here.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence

from core.errors import refuse
from training.stop.base import STOP_REASONS, StopReason, Verdict

#: FR-006.27's fixed order, most significant first. A module-level tuple so there is one
#: declaration of it and no second ordering anywhere.
PRECEDENCE: tuple[str, ...] = (
    StopReason.OPERATOR_STOP.value,
    StopReason.RETENTION_FLOOR.value,
    StopReason.TEST_THRESHOLD.value,
    StopReason.LOSS_PLATEAU.value,
    StopReason.STEP_COUNT.value,
)

# A LIVE ASSERTION, not a comment. If a condition is added to the registry and not to this
# tuple, the stop reason for a boundary where it fires alongside another would fall through
# to "whatever came first" - which is the defect this module exists to remove. Checked at
# import so the failure is a red suite at collection rather than a wrong reason in a run log.
assert set(PRECEDENCE) == set(STOP_REASONS), (
    "every stop reason must have a precedence rank: "
    f"missing {sorted(set(STOP_REASONS) - set(PRECEDENCE))}, "
    f"unknown {sorted(set(PRECEDENCE) - set(STOP_REASONS))}"
)

_RANK: dict[str, int] = {name: index for index, name in enumerate(PRECEDENCE)}


def rank(condition: str) -> int:
    """This condition's precedence rank, lower being more significant.

    Raises:
        Refusal: ``stop_condition_unknown`` for a name outside the five. Refused rather
            than ranked last, because a silent last place is how a sixth condition becomes
            the least important thing in the system without anyone deciding.
    """
    if condition not in _RANK:
        raise refuse(
            "stop_condition_unknown",
            f"{condition!r} has no precedence rank. The set is closed at five: "
            f"{list(PRECEDENCE)}.",
            condition=condition,
            known=list(PRECEDENCE),
        )
    return _RANK[condition]


def fired(verdicts: Iterable[Verdict]) -> tuple[str, ...]:
    """Every condition that fired, in precedence order. The boundary record's ``all_fired``."""
    return tuple(
        verdict.condition
        for verdict in sorted((v for v in verdicts if v.triggered), key=lambda v: rank(v.condition))
    )


def first_fired(verdicts: Sequence[Verdict]) -> Verdict | None:
    """The verdict whose condition becomes the run's ``stop_reason``, or ``None``.

    ``None`` rather than a sentinel: "nothing fired" is the normal case at almost every
    boundary, and a caller that had to compare against a sentinel string could mistake it
    for a reason.
    """
    triggered = [verdict for verdict in verdicts if verdict.triggered]
    if not triggered:
        return None
    return min(triggered, key=lambda verdict: rank(verdict.condition))


__all__ = ["PRECEDENCE", "fired", "first_fired", "rank"]
