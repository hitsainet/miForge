"""``test_threshold``  (R-02.25; FR-006.19, R-02.16).

Stop when the most recent recorded test-set score is **at or above** the configured
threshold - ``>=``, because the operator's statement is *"stop when it is this good"* and a
score landing exactly on the target has met it.

## The score is read as F4 recorded it, with its scale and its sample

The verdict carries F4's ``n``, its ``provider`` and its ``aggregation`` method. ADR-012's
rule - *every number names its scale and sample* - applies hardest to the number that ended
a run: "0.84" from a mean over 320 prompts and "0.84" from an accept rate over 12 are
different claims, and a stop record that said only "0.84" could not tell them apart.

## The exposure window

A test score exists only at an evaluation boundary, so the threshold can be met up to
``test_every - 1`` steps before it is observed, and the verdict's ``observed_at_step`` is
what says so. It is the same arithmetic FR-006.20 states for the floor, in the harmless
direction: a run that trains a few extra steps past its target has lost a few steps, while
a run that trains past its floor has lost general capability.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from core.errors import refuse
from training.stop import scores
from training.stop.base import StopContext, StopReason, Verdict, config_value

NAME = StopReason.TEST_THRESHOLD.value


class TestThreshold:
    """R-02.25's second condition."""

    #: The operator-facing line the New run form shows; the docstring is for developers.
    SUMMARY = "Stops when the latest test score reaches the threshold or higher."

    name = NAME

    def evaluate(self, ctx: StopContext, config: Mapping[str, Any]) -> Verdict:
        threshold = config_value(
            config,
            "threshold",
            condition=NAME,
            constraint=(
                "It is the test-set score at or above which the run stops, on "
                "ScoreRecord's 0..1 scale."
            ),
        )
        try:
            bound = float(threshold)
        except (TypeError, ValueError) as exc:
            raise refuse(
                "stop_condition_config_incomplete",
                f"{NAME!r} needs a numeric 'threshold'; got {threshold!r}.",
                condition=NAME,
                parameter="threshold",
                value=repr(threshold),
            ) from exc
        if not 0.0 <= bound <= 1.0:
            raise refuse(
                "stop_condition_config_incomplete",
                f"A test threshold of {bound} is outside the 0..1 range ScoreRecord.score "
                f"declares, so no recorded score would ever meet it. Refusing rather "
                f"than clamping: a clamped threshold is a target the operator did not set.",
                condition=NAME,
                parameter="threshold",
                value=bound,
                range=[0.0, 1.0],
            )

        row = None if ctx.session is None else scores.latest_test_score(ctx.session, ctx.run_id)
        if row is None:
            # No cadenced test score yet. NOT a stop, and not a refusal either: before the
            # first evaluation there is simply nothing to compare, and `observed_at_step`
            # is this boundary so the run screen renders "not yet observed" rather than a
            # stale zero.
            return Verdict(
                condition=NAME,
                triggered=False,
                observed_at_step=ctx.step,
                detail=f"no test score recorded yet; target {bound:.4f}",
                values={"score": None, "threshold": bound, "headroom": None},
            )

        score = float(row["score"])
        observed_at = int(row["step"])
        return Verdict(
            condition=NAME,
            triggered=score >= bound,
            # THE STEP THE SCORE WAS TAKEN AT, not this boundary. A threshold observed
            # forty steps ago is a different statement from one observed last step (US-8).
            observed_at_step=observed_at,
            detail=(
                f"test {scores.sample_phrase(row)} at step {observed_at}; " f"target {bound:.4f}"
            ),
            values={
                "score": score,
                "threshold": bound,
                "headroom": round(bound - score, 6),
                "n": int(row["n"]),
                "provider": row["provider"],
                "aggregation": row["aggregation"],
                "set": row["set_rendered"],
            },
        )


CONDITION = TestThreshold()

__all__ = ["CONDITION", "NAME", "TestThreshold"]
