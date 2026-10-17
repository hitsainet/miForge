"""``retention_floor`` - **F6 stops; F4 compares**  (R-02.25, R-02.17; F4 FR-004.25).

The sharpest file in this feature, and the shortest on purpose.

## There is no comparison in this module, and no configuration either

This condition fires on **F4's durable breach record**: the ``mf_scores`` row F4 wrote with
``breached = true``. It does not read a floor, does not take a floor, and does not evaluate
``score < floor``. F4's FR-004.25 owns that strict comparison, declares equality **not** a
breach (E-6), and calls that line *"the most load-bearing in the feature"*. A second
comparison here would be a second answer to a correctness question - the argument ADR-005
makes about canonicalisers, with more force.

``backend/tests/unit/test_stop_no_floor_comparison.py`` asserts the absence structurally:
no module under ``training/stop/`` or ``training/loop.py`` contains a floor comparison. *The
division is kept by what the code cannot contain, not by discipline* - the same way F4's TID
keeps its half.

## The exposure window, as a number

A breach is observable only at a retention evaluation, so with a cadence of ``N`` up to
``N - 1`` further training steps occur between the breach happening and the loop stopping.
**At F4's default of ``retention_every = 1`` the number is 0** - the breach is seen at the
boundary it occurred on and the loop stops before the next step. The verdict's
``observed_at_step`` is what makes that window visible on the run screen rather than
implicit, and :func:`exposure_steps` gives it a name so FR-006.20's figure is computed in
one place.

**Mutation M-1** removes this condition from the registry, or makes it always answer
not-fired. If no test goes red, F4's entire feature is a log line: it detects and names the
breach, and nothing stops the run.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from training.stop import scores
from training.stop.base import StopContext, StopReason, Verdict

NAME = StopReason.RETENTION_FLOOR.value


def exposure_steps(retention_every: int) -> int:
    """How many training steps may follow an unobserved breach (FR-006.20).

    ``retention_every - 1``, and **0** at a cadence of 1. One function so the number
    rendered on a screen, the number in a refusal and the number in the run log are the
    same number.
    """
    return max(0, int(retention_every) - 1)


class RetentionFloor:
    """R-02.25's third condition. Reads a fact; computes nothing.

    Takes **no configuration**: the floor lives in exactly one place and F4's
    ``ResolvedFloor`` records which layer supplied it (FR-004.23). A ``floor`` parameter
    here would be a third layer nobody asked for, and the two that exist already needed a
    dedicated field to tell them apart.
    """

    name = NAME

    def evaluate(self, ctx: StopContext, config: Mapping[str, Any]) -> Verdict:
        del config  # this condition has none, and accepting one would invite a third floor
        row = None if ctx.session is None else scores.latest_breach(ctx.session, ctx.run_id)
        if row is None:
            return Verdict(
                condition=NAME,
                triggered=False,
                observed_at_step=ctx.step,
                detail="no retention floor breach recorded",
                values={"breached": False},
            )

        observed_at = int(row["step"])
        floor = None if row["floor"] is None else float(row["floor"])
        return Verdict(
            condition=NAME,
            triggered=True,
            # The step F4 observed the breach at, which with a coarse retention cadence is
            # EARLIER than this boundary. That difference is the exposure window.
            observed_at_step=observed_at,
            detail=(
                f"retention {scores.sample_phrase(row)} at step {observed_at} is below the "
                f"floor of {floor} (from the {str(row['floor_layer']).replace('_', ' ')})"
            ),
            values={
                "breached": True,
                "score": float(row["score"]),
                "floor": floor,
                "floor_layer": row["floor_layer"],
                "n": int(row["n"]),
                "provider": row["provider"],
                "aggregation": row["aggregation"],
                "set": row["set_rendered"],
                "exposure_steps": max(0, ctx.step - observed_at),
            },
        )


CONDITION = RetentionFloor()

__all__ = ["CONDITION", "NAME", "RetentionFloor", "exposure_steps"]
