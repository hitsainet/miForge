"""``step_count``  (R-02.25; FR-006.18).

Stop when the count of **completed** steps reaches the configured total.

The comparison is against :attr:`training.stop.base.StopContext.step`, which is the counter
F6 maintains (FR-006.4) - **not** a ``SELECT count(*) FROM mf_steps``. That distinction is
load-bearing: ``mf_steps`` is an index that ``reindex`` rebuilds from the run log (ADR-004),
and a row count could be short by one for a step whose log line is durable and whose index
row has not been written yet. A run that stopped at a different step after a reindex would be
a run whose stopping point depended on the state of a cache.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from core.errors import refuse
from training.stop.base import StopContext, StopReason, Verdict, config_value

NAME = StopReason.STEP_COUNT.value


class StepCount:
    """R-02.25's first condition."""

    #: The operator-facing line the New run form shows; the docstring is for developers.
    SUMMARY = "Stops when the run has completed a set number of steps."
    name = NAME

    def evaluate(self, ctx: StopContext, config: Mapping[str, Any]) -> Verdict:
        total = config_value(
            config,
            "total",
            condition=NAME,
            constraint="It is the number of completed steps after which the run stops.",
        )
        if not isinstance(total, int) or isinstance(total, bool) or total < 1:
            raise refuse(
                "stop_condition_config_incomplete",
                f"{NAME!r} needs an integer 'total' of at least 1; got {total!r}. A run "
                f"bounded at zero steps would terminate before its step-0 baseline, which "
                f"is the one measurement a creation run has nothing to compare without.",
                condition=NAME,
                parameter="total",
                value=repr(total),
            )

        triggered = ctx.step >= total
        return Verdict(
            condition=NAME,
            triggered=triggered,
            # The counter is read at this boundary, so the observation is this step.
            observed_at_step=ctx.step,
            detail=f"{ctx.step} of {total} steps completed",
            values={"completed": ctx.step, "total": total, "headroom": max(0, total - ctx.step)},
        )


CONDITION = StepCount()

__all__ = ["CONDITION", "NAME", "StepCount"]
