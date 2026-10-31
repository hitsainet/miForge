"""Turning examples into token ids and labels  (FR-002.8; ADR-018).

**The masking policy is applied here, from the value on the batch** - never from a module
default - so the record and the computation cannot disagree. R-02.4 says *"next-token
cross-entropy (supervised fine-tuning)"* and **says nothing about masking**, while the
choice changes the objective: a prompt/completion or chat example trained over the whole
sequence teaches the model to produce **prompts** as well as answers.

Everything in this module is **pure Python over a tokenizer** - no tensors - which is what
lets the masking policy be asserted exhaustively with no GPU and no ``torch`` (ADR-023).

``IGNORE_INDEX`` is -100 because that is ``torch.nn.functional.cross_entropy``'s own
default ``ignore_index``; stating it as a constant rather than relying on the default keeps
the masked positions visible in the labels a test reads.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from core.canonical import sha256_of
from core.errors import refuse

IGNORE_INDEX = -100


def encode_text(tokenizer: Any, text: str) -> list[int]:
    """``text`` as token ids, through the tokenizer's own ``encode``.

    Special tokens are **off**: an example is assembled from a prompt and a completion, and
    letting the tokenizer add a BOS to each half would put one in the middle of the
    sequence - which trains the model to emit it there.
    """
    if tokenizer is None:
        raise refuse(
            "tokenizer_unavailable",
            "This batch carries no tokenizer, so its examples cannot be encoded. The "
            "tokenizer travels with the batch because the masking policy is applied where "
            "the loss is computed and because a chat template's identity belongs on the "
            "record.",
        )
    encode = getattr(tokenizer, "encode", None)
    if encode is None:
        raise refuse(
            "tokenizer_unsupported",
            f"{type(tokenizer).__name__} offers no encode(), so this batch cannot be "
            f"tokenised. Refusing rather than falling back to splitting on whitespace: "
            f"a different tokenisation is a different training example.",
            tokenizer=type(tokenizer).__name__,
        )
    ids = encode(text, add_special_tokens=False)
    return [int(value) for value in ids]


def render_messages(tokenizer: Any, messages: Sequence[Any]) -> tuple[str, str | None]:
    """Chat messages as text, plus **the template's identity**.

    Two runs rendered by different chat templates are not the same run, so the template is
    hashed and recorded. Refused rather than concatenated by hand when the tokenizer has no
    template: hand-rolling a chat format produces text the model was never trained to read,
    and the loss falls anyway.
    """
    apply = getattr(tokenizer, "apply_chat_template", None)
    template = getattr(tokenizer, "chat_template", None)
    # BOTH conditions. Every real `transformers` tokenizer HAS `apply_chat_template` - it is
    # a base-class method - and raises a bare `ValueError` when no template is set. Checking
    # the method alone made this refusal unreachable for every real tokenizer (4.x and 5.x
    # alike; found 2026-10-18 during the transformers 5 upgrade), so a template-less
    # tokenizer failed with a library message and no refusal code.
    if apply is None or (hasattr(tokenizer, "chat_template") and not template):
        raise refuse(
            "chat_template_unavailable",
            "This part declares chat-formatted messages and the tokenizer has no chat "
            "template, so there is no faithful way to render them. Refusing rather than "
            "concatenating the turns: an invented chat format is text the model was never "
            "trained to read, and the loss falls regardless.",
        )
    text = apply(list(messages), tokenize=False, add_generation_prompt=False)
    identity = None if not isinstance(template, str) else sha256_of(template)
    return str(text), identity


def encode_example(
    tokenizer: Any, prompt: str, completion: str, *, policy: str
) -> tuple[list[int], list[int]]:
    """``(input_ids, labels)`` for one supervised example under ``policy``.

    ``completion_only`` masks every prompt position with :data:`IGNORE_INDEX`;
    ``whole_sequence`` labels every position. **An unknown policy is refused** rather than
    treated as one of the two - a silently chosen policy changes the objective.

    The labels are **not** shifted here. ``cross_entropy`` is applied to the logits shifted
    by one in :mod:`training.objectives.sft`, which is where the model's output shape is
    known; shifting in both places is the commonest way to train a model one position off
    and never notice, because the loss still falls.
    """
    prompt_ids = encode_text(tokenizer, prompt) if prompt else []
    completion_ids = encode_text(tokenizer, completion)
    if not completion_ids:
        raise refuse(
            "example_empty_completion",
            "An example's completion encodes to no tokens, so it carries no training "
            "signal at all. Refusing rather than training on a zero-length target: it "
            "contributes nothing and dilutes the batch it is counted in.",
        )
    input_ids = [*prompt_ids, *completion_ids]
    if policy == "completion_only":
        labels = [IGNORE_INDEX] * len(prompt_ids) + list(completion_ids)
    elif policy == "whole_sequence":
        labels = list(input_ids)
    else:
        raise refuse(
            "sft_masking_policy_unknown",
            f"Masking policy {policy!r} is not one of 'completion_only' or "
            f"'whole_sequence'. Refusing rather than picking one: the choice decides "
            f"whether the model is trained to produce prompts as well as answers, and no "
            f"loss curve says which happened.",
            policy=policy,
            options=["completion_only", "whole_sequence"],
        )
    return input_ids, labels


def pad(
    sequences: Sequence[Sequence[int]], *, value: int
) -> tuple[list[list[int]], list[list[int]]]:
    """Right-pad to the longest sequence, returning ``(padded, attention mask)``.

    Right rather than left because these are training examples, not generation prefixes,
    and the mask is returned beside the padding rather than derived later - a model given
    padded inputs and no mask attends to the padding and learns from it.
    """
    width = max((len(sequence) for sequence in sequences), default=0)
    padded = [[*sequence, *([value] * (width - len(sequence)))] for sequence in sequences]
    mask = [[1] * len(sequence) + [0] * (width - len(sequence)) for sequence in sequences]
    return padded, mask


def pad_token_id(tokenizer: Any) -> int:
    """A padding id, refusing when the tokenizer has none rather than inventing 0.

    Token 0 is a real token in most vocabularies. Padding with it and masking it out works
    - until something drops the mask, and then the model trains on whatever token 0 is.
    """
    for attribute in ("pad_token_id", "eos_token_id"):
        value = getattr(tokenizer, attribute, None)
        if isinstance(value, int):
            return value
    raise refuse(
        "tokenizer_has_no_pad_token",
        "This tokenizer declares neither a pad token nor an end-of-sequence token, so a "
        "batch of unequal lengths cannot be padded. Refusing rather than padding with id 0: "
        "token 0 is a real token in most vocabularies.",
    )
