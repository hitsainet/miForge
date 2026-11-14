"""Next-token cross-entropy, masked as the batch says  (R-02.4 **and** R-02.5).

**One implementation, two materials.** R-02.5's accepted completions are *"supervised
examples"*, which is R-02.4's objective applied to different material - so ``corpus`` and
``self-generation`` share this loss and the **batch** says which material it was. Two
cross-entropy implementations would drift apart, and the symptom would be two runs whose
losses are not comparable while both say ``sft``.

The masking policy comes **from the batch**, never from a module default (FR-002.8, E-5).
The tokenisation and the masking are in :mod:`training.objectives.tokenize`, which is pure
Python, so the policy is asserted exhaustively with no ``torch`` and no card.
"""

from __future__ import annotations

from typing import Any

from core.errors import refuse
from training.objectives.base import require_examples, torch_module
from training.objectives.tokenize import IGNORE_INDEX, encode_example, pad, pad_token_id
from training.signals.base import SupervisedExample

NAME = "sft"


class SupervisedFineTuning:
    """``sft``: next-token cross-entropy over the tokens the policy admits."""

    name = NAME

    def loss(self, model: Any, batch: Any) -> Any:
        """The scalar cross-entropy for ``batch``. ``backward()`` is the step's.

        Raises:
            Refusal: ``sft_masking_policy_absent`` when the batch names no policy. The
                batch carries it because pre-flight refused a run without one; a batch that
                reached here without it was assembled by something that skipped that check,
                and guessing here would undo the refusal.
        """
        examples = require_examples(batch, SupervisedExample, NAME)
        policy = getattr(batch, "masking_policy", None)
        if not isinstance(policy, str) or not policy:
            raise refuse(
                "sft_masking_policy_absent",
                "This batch names no masking policy, so there is no answer to which tokens "
                "the cross-entropy is taken over. It is recorded on every batch precisely "
                "so this cannot be decided here: whole-sequence training on prompt/"
                "completion data teaches the model to generate prompts, and no loss curve "
                "says so.",
                objective=NAME,
            )

        tokenizer = getattr(batch, "tokenizer", None)
        encoded = [
            encode_example(tokenizer, example.prompt, example.completion, policy=policy)
            for example in examples
        ]
        pad_id = pad_token_id(tokenizer)
        input_ids, attention = pad([ids for ids, _ in encoded], value=pad_id)
        # Labels are padded with IGNORE_INDEX, not with the pad id: a padded position must
        # contribute nothing to the loss, and padding labels with a real token id would
        # train the model to emit padding at the end of every example.
        labels, _ = pad([label for _, label in encoded], value=IGNORE_INDEX)

        torch = torch_module()
        device = _device_of(model)
        ids_tensor = torch.tensor(input_ids, dtype=torch.long, device=device)
        mask_tensor = torch.tensor(attention, dtype=torch.long, device=device)
        label_tensor = torch.tensor(labels, dtype=torch.long, device=device)

        logits = _logits(model, ids_tensor, mask_tensor)
        # Shift by one, HERE and nowhere else: position t predicts token t+1. Shifting in
        # the tokeniser as well is the commonest way to train a model one position off, and
        # the loss still falls.
        shifted_logits = logits[:, :-1, :]
        shifted_labels = label_tensor[:, 1:]
        return torch.nn.functional.cross_entropy(
            shifted_logits.reshape(-1, shifted_logits.size(-1)),
            shifted_labels.reshape(-1),
            ignore_index=IGNORE_INDEX,
        )


def _logits(model: Any, input_ids: Any, attention_mask: Any) -> Any:
    """The model's logits for one forward pass.

    ``outputs.logits`` when the model returns a Hugging Face output object, else the return
    value itself - and **nothing else is tried**. A ``getattr`` chain over plausible
    attribute names is the default-that-always-fires in another costume.
    """
    outputs = model(input_ids=input_ids, attention_mask=attention_mask)
    logits = getattr(outputs, "logits", None)
    if logits is None:
        logits = outputs
    return logits


def _device_of(model: Any) -> Any:
    """The device the model's parameters are on, or ``None`` to let ``torch`` decide.

    Read from the model rather than resolved here: F2 adds **no GPU-resolution logic, no
    index and no bus address** (ADR-008), and the run's card was pinned by UUID at submit.
    """
    try:
        # `iter(...)` because `parameters()` is a generator on a real model and may be a
        # plain sequence elsewhere; `next()` on a list raises TypeError, which would turn a
        # device lookup into a step failure.
        return next(iter(model.parameters())).device
    except (AttributeError, TypeError, StopIteration):
        return None


OBJECTIVE = SupervisedFineTuning()
