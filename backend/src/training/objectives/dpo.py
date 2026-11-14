"""A direct preference objective  (R-02.6: *"DPO or equivalent"*).

The loss is ``-log sigmoid(beta * ((policy margin) - (reference margin)))`` over a batch of
chosen/rejected pairs, where a *margin* is the sequence log-probability of the chosen
continuation minus the rejected one.

## The reference source is configuration, is recorded, and is refused when absent

BRD-02 §9 leaves **how** open - *"keep a frozen copy in CPU RAM (128 GB is plenty) and
compute reference logprobs there, or cache reference logprobs per batch before the run?"* -
and leaving a choice open is not leaving it absent. Both options sit behind
:class:`ReferenceLogprobs` so the spike's answer is a configuration value rather than a
rewrite, and the value in force is on **every batch**: a DPO loss figure is comparable with
another only when both say what they were measured against.

**Without a reference this is not a preference objective at all** - it degenerates to
preferring the chosen continuation unconditionally - and it would still produce a falling
loss. Hence the refusals rather than a default.

**The frozen-copy option's cost must be measured on this node**: both PCIe links here are
**x4** (Gen4 ≈ 7.9 GB/s, Gen3 ≈ 3.9), so an estimate taken from an x16 figure is wrong by
three to six times (ADR-024).
"""

from __future__ import annotations

from typing import Any, Protocol, runtime_checkable

from core.errors import refuse
from training.objectives.base import require_examples, torch_module
from training.objectives.tokenize import IGNORE_INDEX, encode_example
from training.signals.base import PreferenceExample

NAME = "dpo"


@runtime_checkable
class ReferenceLogprobs(Protocol):
    """Where a pair's reference log-probabilities come from. **One small interface.**"""

    @property
    def source(self) -> str:
        """``frozen_copy`` or ``precomputed`` - the value recorded on the batch."""
        ...

    def for_pair(self, example: PreferenceExample, tokenizer: Any) -> tuple[Any, Any]:
        """``(chosen logprob, rejected logprob)`` for one pair, under the reference."""
        ...


class PrecomputedReference:
    """Reference log-probabilities cached per batch **before the run** started.

    Invalid the moment the reference changes - which is why the batch records that this is
    what it used. A pair arriving without them is **refused**: computing them from the
    policy instead would compare the policy with itself.
    """

    source = "precomputed"

    def for_pair(self, example: PreferenceExample, tokenizer: Any) -> tuple[Any, Any]:
        del tokenizer
        chosen = getattr(example, "reference_chosen_logp", None)
        rejected = getattr(example, "reference_rejected_logp", None)
        if chosen is None or rejected is None:
            raise refuse(
                "dpo_reference_logprobs_missing",
                "SIGNAL_DPO_REFERENCE_SOURCE is 'precomputed' and this pair carries no "
                "cached reference log-probabilities. Refusing rather than computing them "
                "from the policy: that would compare the policy with itself, which removes "
                "the reference term entirely while the loss keeps falling.",
                prompt_id=example.prompt_id,
            )
        return chosen, rejected


class FrozenCopyReference:
    """A frozen parameter set held beside the policy, usually in host memory.

    The copy is made **once**, by whoever holds the parameters, and handed here. This class
    does not create it: a reference built lazily from the *current* policy would be a
    reference to the model as it is now, which is no reference at all.
    """

    source = "frozen_copy"

    def __init__(self, model: Any) -> None:
        if model is None:
            raise refuse(
                "dpo_reference_model_unavailable",
                "SIGNAL_DPO_REFERENCE_SOURCE is 'frozen_copy' and no frozen parameter set "
                "was supplied. The copy is taken from the model as it was before the run "
                "and held beside it; taking it from the model as it is now would compare "
                "the policy with itself.",
            )
        self._model = model

    def for_pair(self, example: PreferenceExample, tokenizer: Any) -> tuple[Any, Any]:
        torch = torch_module()
        with torch.no_grad():
            chosen = sequence_logprob(self._model, tokenizer, example.prompt, example.chosen)
            rejected = sequence_logprob(self._model, tokenizer, example.prompt, example.rejected)
        return chosen, rejected


def resolve_reference(source: str, *, frozen_model: Any = None) -> ReferenceLogprobs:
    """The reference implementation for a configured source. **Refuses an unknown one.**"""
    if source == "precomputed":
        return PrecomputedReference()
    if source == "frozen_copy":
        return FrozenCopyReference(frozen_model)
    raise refuse(
        "dpo_reference_source_unknown",
        f"Reference source {source!r} is neither 'frozen_copy' nor 'precomputed'. Refusing "
        f"rather than choosing one: the two have different costs and different failure "
        f"modes, and the batch records which was used.",
        source=source,
        options=["frozen_copy", "precomputed"],
    )


def sequence_logprob(model: Any, tokenizer: Any, prompt: str, completion: str) -> Any:
    """The total log-probability of ``completion`` given ``prompt``, under ``model``.

    Masked to the completion's positions **always**, whatever the SFT masking policy says:
    a preference is about the continuation, and including the prompt's log-probability
    would make two pairs with the same continuations but different prompts incomparable.
    """
    torch = torch_module()
    input_ids, labels = encode_example(tokenizer, prompt, completion, policy="completion_only")
    ids = torch.tensor([input_ids], dtype=torch.long)
    label_tensor = torch.tensor([labels], dtype=torch.long)
    outputs = model(input_ids=ids, attention_mask=torch.ones_like(ids))
    logits = getattr(outputs, "logits", outputs)
    log_probs = torch.nn.functional.log_softmax(logits[:, :-1, :], dim=-1)
    targets = label_tensor[:, 1:]
    keep = targets != IGNORE_INDEX
    gathered = log_probs.gather(-1, targets.clamp(min=0).unsqueeze(-1)).squeeze(-1)
    return (gathered * keep).sum()


class DirectPreference:
    """``dpo``: the direct preference objective over a batch of pairs."""

    name = NAME

    def loss(self, model: Any, batch: Any) -> Any:
        """The scalar preference loss. ``backward()`` is the step's."""
        examples = require_examples(batch, PreferenceExample, NAME)
        source = getattr(batch, "reference_source", None)
        if not isinstance(source, str) or not source:
            raise refuse(
                "dpo_reference_source_absent",
                "This batch names no reference-logprob source, so there is nothing for the "
                "preference term to be measured against. It is recorded on every batch "
                "precisely so this cannot be decided here: without a reference the "
                "objective degenerates to preferring the chosen continuation "
                "unconditionally, and the loss still falls.",
                objective=NAME,
            )
        reference = getattr(batch, "reference", None) or resolve_reference(
            source, frozen_model=getattr(batch, "frozen_model", None)
        )
        tokenizer = getattr(batch, "tokenizer", None)
        beta = float(getattr(batch, "beta", 0.1))

        torch = torch_module()
        margins = []
        for example in examples:
            policy_chosen = sequence_logprob(model, tokenizer, example.prompt, example.chosen)
            policy_rejected = sequence_logprob(model, tokenizer, example.prompt, example.rejected)
            reference_chosen, reference_rejected = reference.for_pair(example, tokenizer)
            margins.append(
                (policy_chosen - policy_rejected) - (reference_chosen - reference_rejected)
            )
        stacked = torch.stack([torch.as_tensor(margin) for margin in margins])
        # -log sigmoid(beta * margin), in the numerically stable form torch provides. The
        # naive `-log(sigmoid(x))` underflows to inf for a confidently-wrong pair, which
        # poisons the whole batch's mean with no indication of which pair did it.
        return -torch.nn.functional.logsigmoid(beta * stacked).mean()


OBJECTIVE = DirectPreference()
