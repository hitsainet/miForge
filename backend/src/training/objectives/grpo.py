"""A group-relative policy gradient, **with no value network**  (R-02.7).

*"A policy-gradient objective (GRPO or equivalent, no value network."* The "no value
network" clause is **requirement text, so it is a testable property** and not a stylistic
note: this module creates no module, no parameter and no optimizer group of its own, and
AC-002.11 asserts it by inspecting the real optimizer's parameter groups before and after a
step rather than by reading this file.

## The advantage is the group's own spread

Each completion's advantage is its reward minus the group's mean. That is the whole reason
a value network is unnecessary - the group supplies the baseline - and it is also why a
**zero-spread group applies numerically nothing**: subtract the mean from a constant and
every advantage is zero, so the step runs in full and the update is zero. The group is
**recorded with its statistics and its named signal** rather than skipped
(:mod:`training.signals.reward`), because skipping it would hide the one failure in this
feature that costs a full GPU step and reports a healthy loss.
"""

from __future__ import annotations

from typing import Any

from training.objectives.base import require_examples, torch_module
from training.objectives.dpo import sequence_logprob
from training.signals.base import RewardGroup

NAME = "grpo"


def advantages(scores: tuple[float, ...]) -> list[float]:
    """Each member's reward minus the group's mean. **Pure, and tested as such.**

    Not divided by the standard deviation: a zero-spread group would then be a division by
    zero, and the common guard - adding an epsilon - turns a group that learned nothing into
    a group with enormous normalised advantages for whichever member rounded up. The raw
    centred advantage is zero for every member, which is the truth, and the signal beside it
    is what says so.
    """
    mean = sum(scores) / len(scores)
    return [score - mean for score in scores]


class GroupRelativePolicyGradient:
    """``grpo``: a group-relative policy gradient. **No second trainable module.**"""

    name = NAME

    def loss(self, model: Any, batch: Any) -> Any:
        """The scalar policy-gradient loss. ``backward()`` is the step's."""
        groups = require_examples(batch, RewardGroup, NAME)
        tokenizer = getattr(batch, "tokenizer", None)
        torch = torch_module()

        terms = []
        for group in groups:
            for advantage, completion in zip(
                advantages(group.scores), group.completions, strict=True
            ):
                logprob = sequence_logprob(model, tokenizer, group.prompt, completion)
                # The advantage is a constant with respect to the parameters - it comes
                # from the grader, not from the model - so it is a plain float and carries
                # no gradient. Making it a tensor with `requires_grad` would backpropagate
                # into the reward, which is not what a policy gradient is.
                terms.append(-float(advantage) * logprob)
        return torch.stack(terms).mean()


OBJECTIVE = GroupRelativePolicyGradient()
