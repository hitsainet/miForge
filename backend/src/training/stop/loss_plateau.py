"""``loss_plateau`` - four required parameters and no defaults  (R-02.25; FR-006.21-.23).

**R-02.25 names "loss plateau" and defines none of it.** Verified against the BRD rather
than assumed: the word ``plateau`` occurs twice in BRD-02 - in R-02.25's list and in
R-02.39's rationale - and neither parametrises it. No metric, no window, no threshold, no
minimum run length. Reported as **DEF-4**; what this module does is require all four and
default none, which is the only honest answer to a condition whose parameters decide when a
run ends:

================================  ====================================================
Parameter                         Why it cannot be defaulted
================================  ====================================================
``metric``                        A raw per-step loss and a 50-step mean plateau at
                                  different steps.
``window``                        The window **is** the definition of plateau.
``min_relative_improvement``      This is the threshold; a default decides when runs end.
``min_steps``                     Without it a run can plateau-stop during warmup,
                                  before the learning rate has arrived.
================================  ====================================================

## The window never spans a refresh  (FR-006.22)

A refresh changes the batch distribution, so the losses either side of it are not
comparable: a detector spanning the boundary measures the refresh rather than the model, in
**either** direction - a harder refreshed set raises the loss and resets a plateau that had
genuinely arrived, and an easier one lowers it and hides one.

The window this condition reads is :attr:`~training.stop.base.StopContext.window_losses`,
which **the loop clears at a refresh**. So the window cannot span a refresh by construction
rather than by this module remembering to check - and :func:`training.loop` logs the reset
with the round number. **Mutation M-10** lets the window span a refresh.

## A plateau stop records the window's selection and grader health  (FR-006.23)

A flat loss has more than one cause and two of them are already named by other features.
F3 publishes ``selection_accepts_everything``, ``selection_rejects_everything`` and
``grader_scores_have_no_spread``; F2 records that a reward group whose rewards are all equal
yields a zero advantage for every member, so *"the step runs in full and applies numerically
nothing"* while *"reporting a healthy-looking loss"*. A plateau on a ``reward`` run can
therefore mean the model converged **or** that nothing has been learned for a hundred steps,
with the same curve.

So the health signals are attached to the verdict's detail. **F6 stops on none of them.** F3
FR-003.10 states the division - *"F3 measures and names; it does not stop a run"* - and no
requirement makes a degenerate selector a stop condition. R-02.39's rationale even relies on
their not stopping it: *"one that rejects everything stalls the run."*
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from core.errors import refuse
from training.stop.base import StopContext, StopReason, Verdict, config_value

NAME = StopReason.LOSS_PLATEAU.value

#: The four parameters, in the order a refusal should mention them. Enumerated so the
#: "no defaults" test asserts over a list rather than over whatever the reader remembers.
REQUIRED_PARAMETERS: tuple[str, ...] = (
    "metric",
    "window",
    "min_relative_improvement",
    "min_steps",
)

#: The metrics this release can read. ``raw`` is the per-step loss F1 records;
#: ``window_mean`` is the mean over the window. **Closed**, and an unknown metric is refused
#: rather than treated as ``raw``: *"a raw per-step loss and a 50-step mean plateau at
#: different steps"*, so guessing changes when the run ends.
METRICS: tuple[str, ...] = ("raw", "window_mean")


def improvement_ratio(losses: tuple[float, ...]) -> float | None:
    """Relative improvement across the window, or ``None`` for a window of under two.

    ``(first - last) / abs(first)``. **``abs(first)``**, not ``first``: a negative first
    loss - which a DPO or a reward objective can legitimately produce - would otherwise
    invert the sign of the whole ratio, so a run that was improving would read as a plateau
    and stop. That one ``abs`` is the difference between a detector and a coin toss on any
    objective whose loss can go below zero.

    A ``first`` of exactly zero has no relative scale, so the ratio is ``None`` and the
    condition does not fire: refusing to answer beats dividing by zero and beats inventing
    an epsilon that decides when runs end.
    """
    if len(losses) < 2:
        return None
    first, last = losses[0], losses[-1]
    if first == 0.0:
        return None
    return (first - last) / abs(first)


class LossPlateau:
    """R-02.25's fourth condition."""

    name = NAME

    def evaluate(self, ctx: StopContext, config: Mapping[str, Any]) -> Verdict:
        metric = config_value(
            config,
            "metric",
            condition=NAME,
            constraint=f"Which loss to watch, one of {list(METRICS)}.",
        )
        if metric not in METRICS:
            raise refuse(
                "stop_condition_config_incomplete",
                f"{metric!r} is not a loss metric this release can read. A raw per-step "
                f"loss and a windowed mean plateau at different steps, so the metric is "
                f"refused rather than guessed. Known: {list(METRICS)}.",
                condition=NAME,
                parameter="metric",
                value=metric,
                known=list(METRICS),
            )
        window = _positive_int(config, "window", minimum=2)
        min_steps = _positive_int(config, "min_steps", minimum=1)
        threshold = _ratio(config, "min_relative_improvement")

        if min_steps < window:
            raise refuse(
                "stop_condition_config_incomplete",
                f"{NAME!r} has min_steps={min_steps} below window={window}, so the "
                f"condition could never see a full window before it was allowed to fire. "
                f"Set min_steps to at least the window.",
                condition=NAME,
                parameter="min_steps",
                window=window,
                min_steps=min_steps,
            )

        losses = ctx.window_losses[-window:]
        full = len(losses) == window
        eligible = ctx.step >= min_steps
        ratio = improvement_ratio(_series(losses, metric))

        triggered = bool(full and eligible and ratio is not None and ratio < threshold)
        detail = (
            f"{metric} improvement over the last {len(losses)} of {window} steps: "
            f"{'n/a' if ratio is None else format(ratio, '.5f')} "
            f"(fires below {threshold:.5f}, from step {min_steps})"
        )
        if triggered and ctx.window_health:
            # FR-006.23. Attached so the stop reason is interpretable; acted on by nothing.
            flags = sorted(name for name, value in ctx.window_health.items() if value)
            if flags:
                detail += f"; selection/grader health over the window: {', '.join(flags)}"

        return Verdict(
            condition=NAME,
            triggered=triggered,
            # The window's last observation, which with a full window is this boundary.
            observed_at_step=ctx.window_steps[-1] if ctx.window_steps else ctx.step,
            detail=detail,
            values={
                "metric": metric,
                "window": window,
                "window_filled": len(losses),
                "min_steps": min_steps,
                "min_relative_improvement": threshold,
                "improvement": ratio,
                "round": ctx.round,
                "health": dict(ctx.window_health),
            },
        )


def _series(losses: tuple[float, ...], metric: str) -> tuple[float, ...]:
    """The series the ratio is taken over, per ``metric``.

    ``window_mean`` compares the mean of the first half with the mean of the second, by
    presenting a two-element series: that is a *smoothed* comparison, which is what the
    parameter promises, and it is computed here rather than by the caller so the two
    metrics cannot be read off two different sources.
    """
    if metric == "raw" or len(losses) < 2:
        return losses
    half = len(losses) // 2
    first = sum(losses[:half]) / half
    second = sum(losses[half:]) / (len(losses) - half)
    return (first, second)


def _positive_int(config: Mapping[str, Any], key: str, *, minimum: int) -> int:
    value = config_value(
        config,
        key,
        condition=NAME,
        constraint=f"An integer of at least {minimum}.",
    )
    if not isinstance(value, int) or isinstance(value, bool) or value < minimum:
        raise refuse(
            "stop_condition_config_incomplete",
            f"{NAME!r} needs an integer {key!r} of at least {minimum}; got {value!r}.",
            condition=NAME,
            parameter=key,
            value=repr(value),
            minimum=minimum,
        )
    return value


def _ratio(config: Mapping[str, Any], key: str) -> float:
    value = config_value(
        config,
        key,
        condition=NAME,
        constraint="The relative improvement over the window below which the run stops.",
    )
    try:
        ratio = float(value)
    except (TypeError, ValueError) as exc:
        raise refuse(
            "stop_condition_config_incomplete",
            f"{NAME!r} needs a numeric {key!r}; got {value!r}.",
            condition=NAME,
            parameter=key,
            value=repr(value),
        ) from exc
    if ratio < 0.0:
        raise refuse(
            "stop_condition_config_incomplete",
            f"A negative {key} of {ratio} would fire only on a run whose loss is getting "
            f"worse by more than that much, which is a divergence detector and not a "
            f"plateau detector. Refusing rather than treating the sign as a typo.",
            condition=NAME,
            parameter=key,
            value=ratio,
        )
    return ratio


CONDITION = LossPlateau()

__all__ = [
    "CONDITION",
    "METRICS",
    "NAME",
    "REQUIRED_PARAMETERS",
    "LossPlateau",
    "improvement_ratio",
]
