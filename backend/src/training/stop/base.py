"""What a stop condition is  (R-02.25; F6 FR-006.15, FR-006.16).

Five conditions, one Protocol, one registry. The shapes here exist so that the loop asks
every enabled condition the same question at every boundary and records every answer -
because **a condition that is configured and never evaluated is the defect this feature
exists to prevent**, and it is invisible on every other surface: the loss curve falls, the
test score rises, and the run simply carries on.

## ``Verdict`` is the contract's ``StopConditionEvaluation``, not a shape of F6's

``step-record-v1.json``'s ``BoundaryBlock.stop_conditions`` is an array of
``{condition, triggered, observed_at_step, detail?}`` with ``additionalProperties: false``,
verified by parsing the file. :meth:`Verdict.as_evaluation` produces exactly that object, so
there is one shape and the boundary writer cannot invent a second.

**``observed_at_step`` is required by the contract and it is the field that carries the
honesty.** A test threshold observed forty steps ago is a different statement from one
observed last step (FR-006.19, US-8), and a verdict that reported only ``triggered`` would
let the run screen render a stale number as a live one.

## ``FAILED`` is not here, deliberately

A run that raises terminates ``failed`` with an error code (FR-006.8, FR-006.13). It is a
terminal *cause*, not a stop condition, and putting it in :class:`StopReason` would make
FR-006.15's closed set of five into a set of six - which is exactly the drift the explicit
registry exists to prevent. The same applies to a failed refresh (FR-006.38): *"a failed
refresh is a failure, not a sixth stop condition."*
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from datetime import datetime
from enum import StrEnum
from pathlib import Path
from typing import Any, Protocol, runtime_checkable

from core.errors import refuse


class StopReason(StrEnum):
    """R-02.25's five, by name. **The set is closed.**

    The values are the strings written to the boundary record and returned on the run
    response, so they are the vocabulary F9 renders and nothing translates them. A sixth
    member here would be a sixth condition arriving without anyone deciding it should.
    """

    STEP_COUNT = "step_count"
    TEST_THRESHOLD = "test_threshold"
    RETENTION_FLOOR = "retention_floor"
    LOSS_PLATEAU = "loss_plateau"
    OPERATOR_STOP = "operator_stop"


#: The five names as the registry and every refusal list them. Derived from the enum rather
#: than retyped: two spellings of one vocabulary is this project's signature defect.
STOP_REASONS: tuple[str, ...] = tuple(reason.value for reason in StopReason)


@dataclass(frozen=True, slots=True)
class Verdict:
    """One condition's answer at one boundary.

    Args:
        condition: the condition's registered name, one of :data:`STOP_REASONS`.
        triggered: whether this condition says the run should stop **now**.
        observed_at_step: the step the observed value was taken at. Equal to the boundary's
            step for ``step_count`` and ``operator_stop``; **earlier** for a threshold or a
            floor, whose numbers only exist at an evaluation boundary (FR-006.19, .20).
        detail: one sentence naming the number, its scale and its sample (ADR-012). Carried
            as a string because the contract types it as ``string | null``.
        values: the same figures structurally, for the API's stop-conditions view. **Not
            written to the run log** - ``StopConditionEvaluation`` is closed to unknown
            keys - which is stated here so nobody tries.
    """

    condition: str
    triggered: bool
    observed_at_step: int
    detail: str | None = None
    values: Mapping[str, Any] = field(default_factory=dict)

    def as_evaluation(self) -> dict[str, Any]:
        """The contract's ``StopConditionEvaluation`` object, and nothing else.

        Four keys, because ``additionalProperties`` is ``false`` on that subschema. A
        fifth would make the whole boundary block fail validation at the writer - which is
        the right place for it to fail, and the reason this conversion is a method here
        rather than a dict literal at the writer.
        """
        return {
            "condition": self.condition,
            "triggered": self.triggered,
            "observed_at_step": self.observed_at_step,
            "detail": self.detail,
        }


@dataclass
class StopContext:
    """Everything a condition may read, injected rather than imported.

    Injected so every condition is testable with no database, no GPU and no run, and so a
    test can assert the **payload and the call count** of each read (ADR-016 #1).

    ``session`` is the loop's session. :mod:`training.stop.operator_stop` deliberately does
    **not** use it - see that module - which is the one place in this package where the
    difference between "through the database" and "freshly from the database" is the whole
    mechanism.
    """

    run_id: str
    step: int
    round: int
    #: The loop's session. ``None`` in a unit test of a condition that needs none.
    session: Any = None
    #: A callable returning a context manager yielding a **new** session, for the one
    #: condition that must defeat the loop's own transaction snapshot.
    session_scope: Any = None
    #: The authoritative run log, for conditions that read what was recorded rather than
    #: what is in memory.
    log_path: Path | None = None
    #: The losses of this round, oldest first, as the loop observed them. Cleared on a
    #: refresh (FR-006.22) - the loop owns the clearing and the plateau condition reads the
    #: window, so a window can never span a refresh by construction rather than by policy.
    window_losses: tuple[float, ...] = ()
    #: The step each entry of ``window_losses`` was taken at, same order and length.
    window_steps: tuple[int, ...] = ()
    #: F3's health signals observed over the window, attached to a plateau verdict
    #: (FR-006.23). **Attached, never acted on** - F6 stops on none of them.
    window_health: Mapping[str, Any] = field(default_factory=dict)
    now: datetime | None = None


@runtime_checkable
class StopCondition(Protocol):
    """One of R-02.25's five. Resolved from one registry, never by module path."""

    #: One plain sentence for the New run form. Not the docstring: docstrings carry
    #: requirement ids and history, which the operator's screen must not.
    SUMMARY: str
    name: str

    def evaluate(self, ctx: StopContext, config: Mapping[str, Any]) -> Verdict:
        """Answer for this boundary. **Never raises for "not triggered".**

        Returns a :class:`Verdict` with ``triggered=False`` rather than ``None``, so a
        caller cannot publish a non-stop by forgetting to read a boolean and so every
        enabled condition appears on the boundary record whether it fired or not
        (FR-006.16).
        """
        ...


def config_value(
    config: Mapping[str, Any],
    key: str,
    *,
    condition: str,
    constraint: str,
) -> Any:
    """Read a required stop parameter, or refuse **naming it** (E-3, FR-006.21).

    There is no default and no fallback, and this helper exists so the refusal is spelled
    once. ADR-018: *"a `getattr` default that always fires is a fabricated value."* For a
    cadence that would be visible in the record; for a parameter that decides **when a run
    ends** it is invisible, which is why these have none.

    Raises:
        Refusal: ``stop_condition_config_incomplete``, naming the condition, the parameter
            and the constraint it must satisfy.
    """
    if key not in config or config[key] is None:
        raise refuse(
            "stop_condition_config_incomplete",
            f"The {condition!r} stop condition needs {key!r} and the run definition does "
            f"not give it one. {constraint} Supply it on the condition's configuration - "
            f"miForge will not choose a value that decides when your run ends.",
            condition=condition,
            parameter=key,
            constraint=constraint,
        )
    return config[key]


__all__ = [
    "STOP_REASONS",
    "StopCondition",
    "StopContext",
    "StopReason",
    "Verdict",
    "config_value",
]
