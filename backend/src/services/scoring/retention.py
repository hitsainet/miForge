"""The retention floor: four load-bearing lines  (FR-004.23, 25, 26, 27; E-12, E-13, E-14).

**The most load-bearing lines in this feature, and the reason it exists.** The retention set
answers one question: *is the domain costing general capability?* The floor is the answer's
threshold, and miForge owns it - miDataworks owns the set, this project owns what counts as an
unacceptable regression on it.

## This module imports nothing that can stop, cancel or transition a run

R-02.25 is **F6's**. F4 detects, names and publishes; F6 stops. The division is the one F3
uses for its health signals - *"F3 measures and names; it does not stop a run"* - and here it
is kept by **what this file cannot reach** rather than by discipline:
``backend/tests/unit/test_retention_boundary.py`` asserts this module imports no cancellation,
no run-state writer and no Celery control path. A constraint made unreachable beats one
merely stated.

## The comparison is strict, and the strictness is stated here so it is not left to the line

Equality is **not** a breach (E-13): a run sitting exactly on its floor has not fallen below
it, and treating it as a breach stops a healthy run. See the comment above the comparison
itself for the other direction, which is the expensive one.

## A missing floor is a refusal, never "nothing to breach"

``floor or 0.0`` here is mutation control **M-3** shipped on purpose: a fabricated floor of
0.0 passes every run forever, and ADR-018 names the shape - *a ``getattr`` default that
always fires is a fabricated value, worse than a refusal because it is silent.* A retention
set without a floor is a configuration error caught at **submission** (E-12, R-02.48's *"a
run that cannot proceed says so before it starts, not partway through"*); reaching
:func:`evaluate_floor` with ``None`` means the submission guard was bypassed, which is itself
worth a loud refusal.

## RE-RUN MUTATION CONTROLS M-1 AND M-3 AFTER ANY EDIT TO THIS FILE

M-1 inverts the comparison below; M-3 defaults the missing floor. Both must make a test red.
Recorded here rather than only in the task list so the next session finds it without reading
one.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from core.errors import refuse
from providers.base import ProviderIdentity
from services.scoring.tables import (
    FLOOR_LAYER_NONE,
    FLOOR_LAYER_PACK,
    FLOOR_LAYER_RUN,
    KIND_RETENTION,
)

#: The event name a breach is published under, on the score record, the run row and the run
#: room. One spelling, because F6's stop condition matches on it and a second spelling is a
#: stop condition that never fires.
BREACH_EVENT = "retention_floor_breached"


@dataclass(frozen=True, slots=True)
class ResolvedFloor:
    """A floor and **which layer supplied it** (FR-004.23, AC-004.14).

    The layer is recorded because FPRD OQ-2 / DEF-7 is unsettled above this document:
    R-02.38 puts the floor on the pack's retention slot, the UI mockup shows it editable on
    the New-run form, and F7's OQ-8 asked the same question. Recording the layer makes the
    record honest whichever way the operator settles it - *"a pack version names a
    reproducible run input"* means something different under each answer, and a score that
    said only "floor 0.72" could not tell them apart.
    """

    value: float | None
    layer: str

    def payload(self) -> dict[str, Any]:
        return {"floor": self.value, "floor_layer": self.layer}


@dataclass(frozen=True, slots=True)
class BreachSignal:
    """What a breach says. Published durably **before** any socket emission (ADR-007).

    Carries ``n`` and the provider as well as the two numbers, because ADR-012's rule applies
    hardest here: *every number names its scale and sample.* "Retention fell below the floor"
    with no sample size is a sentence an operator cannot act on, and with no provider it is
    not comparable with the step before it.
    """

    run_id: str
    step: int
    kind: str
    score: float
    floor: float
    n: int
    provider: str
    set_ref: str
    floor_layer: str

    def payload(self) -> dict[str, Any]:
        """The published body. Asserted field by field, with the call count (ADR-016 #1)."""
        return {
            "event": BREACH_EVENT,
            "run_id": self.run_id,
            "step": self.step,
            "kind": self.kind,
            "score": self.score,
            "floor": self.floor,
            "n": self.n,
            "provider": self.provider,
            "set": self.set_ref,
            "floor_layer": self.floor_layer,
        }

    def message(self) -> str:
        """One sentence, saying **what happens next** rather than how to feel (ADR-021).

        Names the set, the score, the floor, the sample and the step, and states the
        consequence - that the run's stop condition will act - because an operator reading a
        breach needs to know whether something is already happening.
        """
        return (
            f"Retention on {self.set_ref} is {self.score:.3f} over {self.n} prompts, below "
            f"the floor of {self.floor:.3f} in force at step {self.step} "
            f"(from the {self.floor_layer.replace('_', ' ')}). The run's stop condition "
            f"reads this signal; if stopping on a floor breach is enabled, the run will "
            f"stop at the next step boundary. Continue from the last checkpoint above the "
            f"floor, or lower the floor deliberately."
        )


def resolve_floor(
    *,
    kind: str,
    pack_slot_floor: float | None,
    run_override_floor: float | None,
) -> ResolvedFloor:
    """The floor in force, through **one** function that records its layer.

    One function rather than two call sites reading two values, because the layer has to be
    recorded and a second resolution path is a second answer to *"what floor was in force"*.

    Raises:
        Refusal: ``retention_floor_missing`` when a retention set has no floor from either
            layer (E-12); ``retention_floor_out_of_range`` for a floor outside 0..1, which
            ``ScoreRecord.floor`` cannot express.
    """
    if run_override_floor is not None:
        floor = ResolvedFloor(value=_checked(run_override_floor), layer=FLOOR_LAYER_RUN)
    elif pack_slot_floor is not None:
        floor = ResolvedFloor(value=_checked(pack_slot_floor), layer=FLOOR_LAYER_PACK)
    else:
        floor = ResolvedFloor(value=None, layer=FLOOR_LAYER_NONE)

    if kind == KIND_RETENTION and floor.value is None:
        raise refuse(
            "retention_floor_missing",
            "This run's retention set has no floor, from the pack slot or from a run "
            "override. R-02.38 makes the retention slot with its floor a required pack "
            "slot, and miForge will not choose one: a fabricated floor of 0.0 passes every "
            "run forever while the loss curve falls and nothing says a thing. Set the floor "
            "on the pack's retention slot, or on the run.",
            kind=kind,
        )
    return floor


def _checked(value: float) -> float:
    """A floor inside 0..1, or a refusal naming the value and the range."""
    if not 0.0 <= value <= 1.0:
        raise refuse(
            "retention_floor_out_of_range",
            f"A retention floor of {value} is outside the 0..1 range ScoreRecord.floor "
            f"declares, so it could not be written to a checkpoint manifest. Refusing "
            f"rather than clamping: a clamped floor is a threshold the operator did not set.",
            floor=value,
            range=[0.0, 1.0],
        )
    return float(value)


def evaluate_floor(
    score: float,
    floor: ResolvedFloor,
    *,
    run_id: str,
    step: int,
    n: int,
    provider: ProviderIdentity,
    set_ref: str,
) -> BreachSignal | None:
    """Compare a retention score with its floor. Returns a signal **or ``None``**.

    Returns ``None`` for "no breach" rather than a signal with a False flag, so a caller
    cannot publish a non-breach by forgetting to read a boolean.

    Raises:
        Refusal: ``retention_floor_missing`` when the resolved floor is ``None``. Reaching
            here with ``None`` means the submission guard (E-12) was bypassed.
    """
    if floor.value is None:
        # A RETENTION SET WITHOUT A FLOOR IS A CONFIGURATION ERROR, caught at submission
        # (E-12). `floor or 0.0` here is mutation control M-3 shipped on purpose.
        raise refuse(
            "retention_floor_missing",
            f"A retention score of {score:.3f} was computed with no floor in force at step "
            f"{step}. This should have been refused at run submission; refusing here rather "
            f"than reading a missing floor as 'nothing to breach', which would pass every "
            f"run for the rest of its life.",
            run_id=run_id,
            step=step,
            score=score,
        )

    # Invert this comparison and every run passes its floor forever: the loss curve still
    # falls, the test score still rises, the suite stays green, and general capability is
    # spent invisibly. That is the project risk the PPRD names with this feature as its only
    # mitigation. Mutation control M-1 breaks this line deliberately and a test must go red.
    # STRICT: equality is NOT a breach (E-13) - a run sitting exactly on its floor has not
    # fallen below it. If you change this line, re-run M-1.
    if score < floor.value:
        return BreachSignal(
            run_id=run_id,
            step=step,
            kind=KIND_RETENTION,
            score=float(score),
            floor=float(floor.value),
            n=n,
            provider=provider.record(),
            set_ref=set_ref,
            floor_layer=floor.layer,
        )
    return None
