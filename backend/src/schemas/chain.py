"""Chaining a curriculum  (R-02.27; FR-006.39 - .46).

R-02.27: *"A run can be **chained**: the stop condition of one run triggers the start of
another with a different signal source or domain, from the last saved checkpoint. This is
how a model is biased through a curriculum: corpus first, then self-generation, then
preference pairs, for instance."*

## What is here and what is not

**The types and the advance filter are here. The table-backed orchestration is not**, and
that is a reported boundary rather than an omission: ``mf_chains`` and ``mf_chain_stages``
need an Alembic revision, and the Alembic graph is serialised through the orchestrator
because two revisions off one parent make ``alembic upgrade head`` fail on a clean clone.
The exact DDL is in F6's feature report.

What that leaves is the half that is safety-critical and needs no table: **three stop
reasons can never advance a chain**, and FR-006.43 wants that kept in three layers - the
``Literal`` type, an insert-time check, and :func:`advance_filter`. Two of the three are
expressible now and both are here; the third is the insert, and it refuses the same set.

## Why ``retention_floor`` can never advance a chain  (FR-006.43, DEF-7, **M-8**)

**R-02.27 names no filter at all.** *"The stop condition of one run triggers the start of
another"* reads as though **any** stop advances it - verified against the requirement, and
reported as DEF-7 so the operator sees that this is derived.

Advancing past a floor breach would start the next stage **from the checkpoint whose
general capability had just collapsed**, and push it further in the same direction -
compounding the one failure in the loop whose symptom is invisible on every curve. The
PPRD names that risk as a project risk and names the floor as its only mitigation; a chain
that advanced through it would undo the mitigation one stage at a time, and the suite would
be green for every stage.

Advancing past an **operator stop** would restart work the operator had just stopped.
Advancing past a **failed** stage would start the next from a checkpoint nothing verified.

**Mutation M-8** lets ``retention_floor`` into the advance set.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from core.errors import refuse
from training.stop.base import STOP_REASONS, StopReason

#: The three reasons that can **never** advance a chain. Derived from R-02.27 (DEF-7) and
#: enumerated once, so the type, the insert-time check and the runtime filter share one set
#: rather than three copies of it.
NEVER_ADVANCES: frozenset[str] = frozenset(
    {
        StopReason.RETENTION_FLOOR.value,
        StopReason.OPERATOR_STOP.value,
        # Not a stop reason at all - a `failed` stage has an error code, not one of
        # R-02.25's five - and named here because the filter must refuse it too.
        "failed",
    }
)

#: The reasons a stage **may** declare. ``Literal`` rather than ``str``, so a configuration
#: naming ``retention_floor`` is refused by **the type** before any code runs - the first of
#: FR-006.43's three layers, and the one a later edit cannot forget.
AdvanceOn = Literal["step_count", "test_threshold", "loss_plateau"]

# A LIVE ASSERTION that the type and the set agree. If a stop reason is added to R-02.25
# and to neither of these, the `Literal` would silently exclude it from every chain
# configuration - which is the safe direction, but silently.
assert set(STOP_REASONS) - NEVER_ADVANCES == {
    "step_count",
    "test_threshold",
    "loss_plateau",
}, "AdvanceOn and NEVER_ADVANCES disagree about R-02.25's five"


class StageSpec(BaseModel):
    """One stage of a curriculum: a run definition, plus what advances past it.

    The run definition itself is **not redefined here** - it is R-02.53's object, validated
    by the one validator, so a stage cannot be configured in a way a single run could not
    be. That is also what makes FR-006.46's deletability property true: a chain does
    nothing a sequence of manual branches could not do.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    #: R-02.53's definition for this stage, as submitted. Opaque here and validated by
    #: :func:`services.run_definition.validate` at **chain creation** - so a curriculum's
    #: fourth stage cannot be discovered invalid three hours in (FR-006.39).
    definition: dict[str, object]
    #: Which stop reasons advance the chain past this stage. **Non-empty**: a stage that
    #: advances on nothing is a chain that silently stops at it.
    advance_on: tuple[AdvanceOn, ...] = Field(min_length=1)


class ChainSpec(BaseModel):
    """An ordered list of stages. Each stage is a run (FR-006.39, FR-006.41)."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    name: str = Field(min_length=1, max_length=120)
    #: At least two: a "chain" of one is a run, and accepting one would make the chain
    #: machinery the path a single run took, which is the opposite of FR-006.46.
    stages: tuple[StageSpec, ...] = Field(min_length=2)


def assert_advanceable(reasons: Iterable[str]) -> None:
    """Refuse a configuration naming a reason that can never advance (FR-006.43).

    FR-006.43's **second** layer - the insert-time check. The type already excludes the
    three, and this catches a configuration that arrived as plain strings (from JSON, from
    a migration, from a later edit that widened the type).

    Raises:
        Refusal: ``chain_stage_invalid``, naming the reason **and why**.
    """
    offenders = sorted(set(reasons) & NEVER_ADVANCES)
    if not offenders:
        return
    raise refuse(
        "chain_stage_invalid",
        f"A chain stage may not advance on {offenders}. A floor breach would start the "
        f"next stage from the checkpoint whose general capability had just collapsed and "
        f"push it further in the same direction; an operator stop would restart work "
        f"someone had just stopped; a failed stage would start the next from a checkpoint "
        f"nothing verified. Advance on a step count, a test threshold or a loss plateau.",
        offenders=offenders,
        never_advances=sorted(NEVER_ADVANCES),
        advanceable=["step_count", "test_threshold", "loss_plateau"],
    )


def advance_filter(stop_reason: str | None, advance_on: Sequence[str]) -> bool:
    """Whether a finished stage's ``stop_reason`` advances the chain.

    FR-006.43's **third** layer, after the type and the insert-time check - **M-8's
    target.** Three layers rather than one because each catches a different way the rule
    could be lost: the type catches a configuration, the insert catches data, and this
    catches a stop reason arriving at runtime from a row that predates either.

    Args:
        stop_reason: the finished stage's reason. ``None`` for a stage that ended without
            one, which cannot advance anything.
        advance_on: what the stage's configuration declared.

    Returns:
        ``True`` only when the reason is declared **and** is not in
        :data:`NEVER_ADVANCES`. The second half is not redundant: it is the check that
        holds **in every configuration**, including one that got past the other two.
    """
    if stop_reason is None:
        return False
    # THE LINE M-8 BREAKS. Not `stop_reason in advance_on` alone: that would advance on a
    # floor breach the moment any configuration declared one, which is exactly what a
    # widened type or a hand-edited row would do.
    if stop_reason in NEVER_ADVANCES:
        return False
    return stop_reason in set(advance_on)


def halt_reason(stop_reason: str | None) -> str:
    """The code a chain halts with, given the reason its stage stopped on.

    Named rather than generic, because *"the chain stopped"* and *"the chain stopped
    because general capability fell below the floor"* are different sentences and only one
    of them is actionable.
    """
    if stop_reason == StopReason.RETENTION_FLOOR.value:
        return "chain_halted_on_floor_breach"
    if stop_reason == StopReason.OPERATOR_STOP.value:
        return "chain_halted_on_operator_stop"
    if stop_reason is None or stop_reason == "failed":
        return "chain_halted_on_failed_stage"
    return "chain_not_advanceable"


__all__ = [
    "NEVER_ADVANCES",
    "AdvanceOn",
    "ChainSpec",
    "StageSpec",
    "advance_filter",
    "assert_advanceable",
    "halt_reason",
]
