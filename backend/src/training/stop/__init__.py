"""The stop-condition registry - **exactly five, explicitly registered**  (FR-006.15, .16).

## Five explicit registrations, and no second lookup path

No entry-point scan, no ``importlib`` over this directory, no ``__subclasses__`` hook. The
reason is the one F2's signal registry and F3's selector registry both give, one directory
over: a scan makes *"is it registered?"* **untestable** and an unregistered condition
**silent**. That is the failure this project's whole review discipline was written after - a
sibling shipped sixteen fully-implemented, unit-tested, documented MCP tools that were never
registered with its server, so the suite was green and the docs said done while the feature
did not exist for any caller.

Here the cost is worse than a missing feature. **A stop condition that is configured and
never evaluated is indistinguishable from a run that is going well**: the loss curve falls,
the test score rises, and the run carries on past its floor. :data:`_REGISTRATIONS` is the
single declaration of the surface, and **mutation M-1** removes ``retention_floor`` from it.

## ``evaluate_all`` does not short-circuit, and that is a requirement

FR-006.16: *every* enabled condition is evaluated at *every* boundary and **every**
evaluation is recorded. So this function collects all five verdicts even once one has fired.
Short-circuiting would be a defensible optimisation and an indefensible record: the boundary
block would then say nothing about the conditions that did not get asked, and an operator
reading it could not tell "did not fire" from "was not evaluated". It is also what makes
FR-006.27's precedence meaningful - you cannot pick the most significant of two reasons if
you stopped looking at the first.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from core.errors import refuse
from training.stop import (
    loss_plateau,
    operator_stop,
    precedence,
    retention_floor,
    step_count,
    test_threshold,
)
from training.stop.base import (
    STOP_REASONS,
    StopCondition,
    StopContext,
    StopReason,
    Verdict,
)

#: **THE WIRING.** Five entries, one per condition R-02.25 names. Deleting one must make
#: ``backend/tests/unit/test_stop_registry.py`` red - and for ``retention_floor`` it must
#: make ``integration/test_retention_floor_stops_the_run.py`` red too, which is M-1 and the
#: sharpest control in this feature.
_REGISTRATIONS: tuple[tuple[str, StopCondition], ...] = (
    (step_count.NAME, step_count.CONDITION),
    (test_threshold.NAME, test_threshold.CONDITION),
    (retention_floor.NAME, retention_floor.CONDITION),
    (loss_plateau.NAME, loss_plateau.CONDITION),
    (operator_stop.NAME, operator_stop.CONDITION),
)

_CONDITIONS: dict[str, StopCondition] = dict(_REGISTRATIONS)

# The registry's key set IS R-02.25's list. Asserted at import against the enum rather than
# against a retyped literal, so a condition registered under a misspelled name is a red
# suite at collection instead of a condition nothing can enable.
assert set(_CONDITIONS) == set(STOP_REASONS), (
    "the registry must hold exactly R-02.25's five: "
    f"missing {sorted(set(STOP_REASONS) - set(_CONDITIONS))}, "
    f"unknown {sorted(set(_CONDITIONS) - set(STOP_REASONS))}"
)

#: Conditions that take **no** configuration at all. Enumerated because the validator must
#: not demand parameters for them, and because a ``floor`` on ``retention_floor`` would be a
#: third floor layer nobody asked for (F4 FR-004.23 already needed a field to tell two
#: apart).
UNCONFIGURED: frozenset[str] = frozenset({retention_floor.NAME, operator_stop.NAME})


def names() -> tuple[str, ...]:
    """The five registered names, in R-02.25's own order."""
    return tuple(name for name, _condition in _REGISTRATIONS)


def resolve(name: str) -> StopCondition:
    """The condition registered as ``name``.

    Raises:
        Refusal: ``stop_condition_unknown``, **listing the five** (E-2). Listed rather than
            merely named, because R-02.25's set is closed and the useful next step is to
            pick one of the five rather than to go looking for a sixth.
    """
    condition = _CONDITIONS.get(name)
    if condition is None:
        raise refuse(
            "stop_condition_unknown",
            f"No stop condition named {name!r}. R-02.25 names five and the set is closed: "
            f"{list(names())}. Choose one of them.",
            condition=name,
            known=list(names()),
        )
    return condition


def evaluate_all(
    enabled: Mapping[str, Mapping[str, Any]],
    ctx: StopContext,
) -> list[Verdict]:
    """Evaluate **every** enabled condition and return **every** verdict (FR-006.16).

    Args:
        enabled: ``condition name -> its configuration``. An unconfigured condition maps to
            an empty mapping, which is a legitimate enablement and not an omission.
        ctx: the boundary's context.

    Returns:
        One verdict per enabled condition, in R-02.25's registry order so the boundary
        record's array is stable across runs and machines. **No short-circuit**: a
        non-firing condition still appears, which is what lets a reader tell "did not fire"
        from "was not evaluated" (M-1's test reads exactly that).
    """
    verdicts: list[Verdict] = []
    for name in names():
        if name not in enabled:
            continue
        verdicts.append(resolve(name).evaluate(ctx, enabled[name]))
    return verdicts


def assert_known(conditions: Sequence[str]) -> None:
    """Refuse the first unknown name among ``conditions`` (E-2). Used by the validator."""
    for name in conditions:
        resolve(name)


__all__ = [
    "STOP_REASONS",
    "UNCONFIGURED",
    "StopContext",
    "StopReason",
    "Verdict",
    "assert_known",
    "evaluate_all",
    "names",
    "precedence",
    "resolve",
]
