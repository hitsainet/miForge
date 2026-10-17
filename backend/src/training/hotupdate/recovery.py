"""Recovery after an interrupted commit - three outcomes  (R-02.30, R-02.31; FR-011.38-45).

**COMPLETE | ROLLBACK | STOP.** The third is the one a careless implementation omits, and
it is the one that keeps the feature honest: *the run does not continue against a parameter
set whose version it cannot state.*

## An interrupted step is never retried

The optimizer update is **not idempotent**. Re-applying it to tensors that already received
it while others did not **compounds the damage rather than correcting it** (R-02.31). So an
interrupted step is **abandoned** and recorded as a discontinuity, and the loss curve
carries the gap explicitly rather than appearing continuous across a step that never
completed (BRD AC22). **MUT-3 flips that guard and must go red.**

**The one retryable case** is a quiesce that timed out *before* ``commit-begin`` (BRD AC7):
nothing was written, the weights are unchanged, and the step is retried with the timeout
recorded. It is detectable because ``commit-begin`` never returned successfully - which is
the only honest test, since "did a tensor move" is not observable from this process.

## Path (a) is unreachable after a process death, and R-02.30 does not say so

``COMPLETE`` needs **all three**: the same miForge process, the computed update still in
memory, and the attachment id. After a crash the id is gone - miLLM returns it once and no
read path returns it again (miLLM Feature 031 FR-31.1.6) - and the update is gone with the
process. So path (a) is permanently unavailable after a restart (FPRD OQ-4/AMB-N), and this
module is built on that assumption rather than on a hope that the id can be re-fetched.

**The post-crash ordering lives here and nowhere else:** ``load`` -> **an operator clears
the stale attachment** -> ``attach`` -> verify -> resume. The load needs no attachment;
resuming does. R-02.30 does not state the ordering and R-02.33's stale row implies it, so it
is written down once, in the module that executes it.

## ROLLBACK reads; it re-specifies nothing

The checkpoint is loaded through miLLM's own ``load`` (BRD-01 R-01.25, which verifies the
hash before touching a resident tensor and returns the set to ``clean``). The optimizer
state is restored through **F1's single trainer-state reader**: F11 is that layout's third
reader - after F1's writer and F5's reader - and writes down **none** of its file names,
tensor keys or padding. *That layout is the sixth shared contract PPRD §12's gate does not
list*, and three readers is three chances to spell it differently.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any

from core.errors import refuse

log = logging.getLogger(__name__)


class Recovery(StrEnum):
    """What recovery was possible, and therefore what happened."""

    #: R-02.30(a). The same process completes its own commit.
    COMPLETE = "complete"
    #: R-02.30(b). Load the last checkpoint, restore optimizer state, resume.
    ROLLBACK = "rollback"
    #: Neither. **A first-class outcome with a named reason.**
    STOP = "stop"


@dataclass(frozen=True)
class RecoveryPlan:
    """The decision, and **why the other path was not taken**.

    ``unavailable_conditions`` is the field that makes a run log readable afterwards: a log
    that says *"rolled back"* without saying which of (a)'s three conditions failed cannot
    tell a crashed worker from a lost attachment id.
    """

    outcome: Recovery
    reason: str
    unavailable_conditions: tuple[str, ...] = ()
    resume_from_step: int | None = None
    #: Set on STOP and on the post-crash path: the action an operator must take, named,
    #: with the fact that miForge cannot take it.
    operator_action: str | None = None
    ordering: tuple[str, ...] = ()


@dataclass
class Interruption:
    """What is known at the moment of an interruption. Facts, not inferences."""

    #: Whether this is the process that opened the bracket. ``False`` after any restart.
    same_process: bool
    #: Whether the computed update is still in memory. Worker memory only, by design
    #: (TDD §7): *its loss is what makes R-02.30(a) unavailable after a restart.*
    holds_update: bool
    #: The attachment id, or ``None``. Returned once; no read path returns it again.
    attachment_id: str | None
    #: Whether ``commit-begin`` returned successfully. **The only honest test for "was
    #: anything written"** from this process.
    commit_begun: bool
    #: The last checkpoint step this run saved, or ``None`` when it has saved none.
    last_checkpoint_step: int | None = None
    #: Whether the attachment is reported stale, which blocks a re-attach until an
    #: **operator** clears it (BRD-01 R-01.29).
    attachment_stale: bool = False
    notes: list[str] = field(default_factory=list)


POST_CRASH_ORDERING: tuple[str, ...] = (
    "load the last checkpoint (BRD-01 R-01.25; needs no attachment)",
    "an OPERATOR clears the stale attachment (BRD-01 R-01.29; miForge cannot)",
    "attach (BRD-01 R-01.1)",
    "verify the reconstruction (R-02.2)",
    "resume the run from the loaded step",
)


def plan(interruption: Interruption) -> RecoveryPlan:
    """Choose the recovery. **Rollback is always available; completion is an optimisation.**

    Raises nothing: an undecidable situation is :attr:`Recovery.STOP` with a reason, not an
    exception, because the caller has a run to fail cleanly and a record to write.
    """
    missing: list[str] = []
    if not interruption.same_process:
        missing.append("same_process")
    if not interruption.holds_update:
        missing.append("holds_the_computed_update")
    if interruption.attachment_id is None:
        missing.append("holds_the_attachment_id")

    if not missing:
        return RecoveryPlan(
            outcome=Recovery.COMPLETE,
            reason=(
                "this process still holds the computed update and the attachment id, so "
                "miLLM accepts a late commit-end from the same attachment (BRD-01 "
                "R-01.23a) and the overrun is recorded on both sides"
            ),
        )

    if interruption.last_checkpoint_step is None:
        return RecoveryPlan(
            outcome=Recovery.STOP,
            reason=(
                "the commit cannot be completed and this run has saved no checkpoint, so "
                "there is nothing to roll back to. The run stops rather than continuing "
                "against a parameter set whose version it cannot state."
            ),
            unavailable_conditions=tuple(missing),
            operator_action=(
                "load a known checkpoint into the serving process yourself, or restart the "
                "run; miForge will not step against a set it cannot name the version of"
            ),
        )

    ordering = POST_CRASH_ORDERING if not interruption.same_process else ()
    return RecoveryPlan(
        outcome=Recovery.ROLLBACK,
        reason=(
            "the commit cannot be completed, so the last checkpoint is loaded into the "
            "serving process (BRD-01 R-01.25 returns the set to clean), optimizer state is "
            "restored for that step through F1's trainer-state reader (R-02.22), the "
            "reconstruction is re-verified, and the run resumes from there"
        ),
        unavailable_conditions=tuple(missing),
        resume_from_step=interruption.last_checkpoint_step,
        operator_action=(
            (
                "clear the stale attachment (BRD-01 R-01.29 is OPERATOR-ONLY; miForge holds "
                "'trainer' and does not contain the call) before this run can attach again"
            )
            if interruption.attachment_stale or not interruption.same_process
            else None
        ),
        ordering=ordering,
    )


def is_retryable(interruption: Interruption) -> bool:
    """Whether the **step** may be retried. True in exactly one case (BRD AC7, R-02.31).

    A quiesce that timed out **before** ``commit-begin``: nothing was written, the weights
    are unchanged, the step is retried and the timeout recorded. Every other interruption is
    an abandonment - *re-applying a non-idempotent AdamW update to a half-written set
    compounds the damage.* **MUT-3 inverts this and the never-retry test must go red.**
    """
    return not interruption.commit_begun


def abandonment_record(
    *, run_id: str, step: int, plan_taken: RecoveryPlan, cause: str
) -> dict[str, Any]:
    """The run-log note for an abandoned step (FR-011.42, FR-011.45; BRD AC22).

    Recorded as a **discontinuity**, so the loss curve carries the gap rather than appearing
    continuous across a step that never completed. It names which recovery path was taken,
    which of (a)'s conditions failed, and the step the run resumed from - *a run that cannot
    say which step it resumed from cannot have its loss curve read.*
    """
    return {
        "run_id": run_id,
        "step": step,
        "event": "step_abandoned_commit_interrupted",
        "cause": cause,
        "retried": False,
        "discontinuity": True,
        "recovery_path": plan_taken.outcome.value,
        "recovery_reason": plan_taken.reason,
        "completion_unavailable_because": list(plan_taken.unavailable_conditions),
        "resumed_from_step": plan_taken.resume_from_step,
        "operator_action_required": plan_taken.operator_action,
        "ordering": list(plan_taken.ordering),
    }


def complete_the_commit(client: Any, timings: Any = None) -> dict[str, Any]:
    """R-02.30(a): send the late ``commit-end``, then release serving.

    Raises:
        Refusal: ``weights_recovery_failed`` when miLLM will not accept it. Not retried, and
            the caller falls through to :attr:`Recovery.ROLLBACK` rather than trying again.
    """
    from core.errors import Refusal

    try:
        body = client.call("commit-end")
    except Refusal as exc:
        raise refuse(
            "weights_recovery_failed",
            "miLLM did not accept the late commit-end, so recovery by completion is not "
            "available and the parameter set stays torn. Roll back to the last checkpoint "
            "(BRD-01 R-01.25) instead; the step is NOT retried either way.",
            upstream_refusal=exc.code,
            retried=False,
        ) from exc
    client.call("resume")
    return dict(body) if isinstance(body, dict) else {}


def restore_trainer_state(run_id: str, step: int, *, root: Any = None) -> dict[str, Any]:
    """Optimizer state for ``step``, **through F1's reader** (R-02.22; FR-011.39).

    One line of delegation on purpose. F11 is this layout's third reader and re-specifies
    nothing about it: no file names, no tensor keys, no zero-padding. Any refusal is F1's,
    with F1's wording.
    """
    from training import trainer_state

    state: dict[str, Any] = trainer_state.read(run_id, step, root=root)
    return state
