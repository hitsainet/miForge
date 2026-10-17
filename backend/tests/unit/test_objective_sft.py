"""The masking policy, which decides what the objective actually is  (FR-002.8, E-5).

R-02.4 says *"next-token cross-entropy (supervised fine-tuning)"* and **says nothing about
masking**, while the choice changes the objective: a prompt/completion or chat example
trained over the whole sequence teaches the model to produce **prompts** as well as answers,
and no loss curve says which happened.

So: the policy is explicit configuration, it is recorded on every batch, an absent one is
refused, and the labels are asserted position by position - including through the loss,
where ``cross_entropy`` is checked for **the payload and the call count**.

``torch`` is not installed on a machine without a card (ADR-023), so the one import seam -
``training.objectives.base.torch_module`` - is substituted by a recording double. Everything
above it is pure Python and runs anywhere.
"""

from __future__ import annotations

from typing import Any

import pytest

from core.errors import Refusal
from training.objectives import base as objective_base
from training.objectives import sft
from training.objectives.tokenize import IGNORE_INDEX, encode_example, pad, pad_token_id
from training.signals.base import SupervisedExample, TrainingBatch


class Tokenizer:
    """One token per character, so a label position maps to a character a test can read."""

    pad_token_id = 0
    chat_template = "{% for m in messages %}{{m.content}}{% endfor %}"

    def encode(self, text: str, add_special_tokens: bool = False) -> list[int]:
        assert add_special_tokens is False, "special tokens must be off between the halves"
        return [ord(character) for character in text]

    def apply_chat_template(self, messages: list[Any], **kwargs: Any) -> str:
        return "|".join(str(message["content"]) for message in messages)


class TestTheMaskingPolicy:
    def test_completion_only_masks_every_prompt_position(self) -> None:
        ids, labels = encode_example(Tokenizer(), "ab", "cd", policy="completion_only")
        assert ids == [97, 98, 99, 100]
        assert labels == [IGNORE_INDEX, IGNORE_INDEX, 99, 100]

    def test_whole_sequence_labels_every_position(self) -> None:
        _ids, labels = encode_example(Tokenizer(), "ab", "cd", policy="whole_sequence")
        assert labels == [97, 98, 99, 100]

    def test_an_unknown_policy_is_refused_rather_than_treated_as_one_of_the_two(self) -> None:
        with pytest.raises(Refusal) as caught:
            encode_example(Tokenizer(), "a", "b", policy="completion")
        assert caught.value.code == "sft_masking_policy_unknown"
        assert caught.value.details["options"] == ["completion_only", "whole_sequence"]

    def test_an_empty_completion_is_refused_rather_than_trained_on(self) -> None:
        """E-9: a zero-length target contributes nothing and dilutes the batch it is
        counted in."""
        with pytest.raises(Refusal) as caught:
            encode_example(Tokenizer(), "a", "", policy="completion_only")
        assert caught.value.code == "example_empty_completion"

    def test_a_batch_naming_no_policy_is_refused_at_the_loss_too(self) -> None:
        """E-5 at the second layer. Pre-flight refuses a run without a policy; this refuses
        a batch that reached the loss without one, rather than guessing and undoing that."""
        batch = TrainingBatch(
            objective="sft",
            record={},
            examples=(SupervisedExample(prompt_id="p", prompt="a", completion="b"),),
            counts=_counts(),
            tokenizer=Tokenizer(),
            masking_policy=None,
        )
        with pytest.raises(Refusal) as caught:
            sft.OBJECTIVE.loss(_Model(), batch)
        assert caught.value.code == "sft_masking_policy_absent"


class TestPadding:
    def test_labels_are_padded_with_the_ignore_index_not_the_pad_token(self) -> None:
        """Padding labels with a real token id trains the model to emit padding."""
        padded, mask = pad([[1, 2, 3], [4]], value=IGNORE_INDEX)
        assert padded == [[1, 2, 3], [4, IGNORE_INDEX, IGNORE_INDEX]]
        assert mask == [[1, 1, 1], [1, 0, 0]]

    def test_a_tokenizer_with_no_pad_token_is_refused_rather_than_padded_with_zero(self) -> None:
        """Token 0 is a real token in most vocabularies."""

        class NoPad:
            def encode(self, text: str, add_special_tokens: bool = False) -> list[int]:
                return [1]

        with pytest.raises(Refusal) as caught:
            pad_token_id(NoPad())
        assert caught.value.code == "tokenizer_has_no_pad_token"

    def test_an_eos_token_serves_as_padding_when_there_is_no_pad_token(self) -> None:
        class Eos:
            eos_token_id = 7

        assert pad_token_id(Eos()) == 7


class TestTheLossItself:
    def test_cross_entropy_is_called_once_with_the_masked_labels(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """**The payload and the call count.** The labels reaching the loss carry
        ``-100`` at exactly the prompt positions, shifted by one - and the shift happens
        **here and nowhere else**, because shifting in the tokeniser as well trains the
        model one position off while the loss still falls.
        """
        torch = _FakeTorch()
        monkeypatch.setattr(objective_base, "torch_module", lambda: torch)
        monkeypatch.setattr(sft, "torch_module", lambda: torch)
        batch = TrainingBatch(
            objective="sft",
            record={},
            examples=(SupervisedExample(prompt_id="p", prompt="ab", completion="cd"),),
            counts=_counts(),
            tokenizer=Tokenizer(),
            masking_policy="completion_only",
        )
        sft.OBJECTIVE.loss(_Model(), batch)
        assert len(torch.cross_entropy_calls) == 1
        logits, labels, kwargs = torch.cross_entropy_calls[0]
        # Labels: [-100, -100, 99, 100] shifted left by one -> [-100, 99, 100]
        assert labels == [IGNORE_INDEX, 99, 100]
        assert kwargs["ignore_index"] == IGNORE_INDEX
        # Logits: one row per predicted position, the last position dropped.
        assert len(logits) == 3

    def test_whole_sequence_sends_no_ignored_position(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        torch = _FakeTorch()
        monkeypatch.setattr(sft, "torch_module", lambda: torch)
        batch = TrainingBatch(
            objective="sft",
            record={},
            examples=(SupervisedExample(prompt_id="p", prompt="ab", completion="cd"),),
            counts=_counts(),
            tokenizer=Tokenizer(),
            masking_policy="whole_sequence",
        )
        sft.OBJECTIVE.loss(_Model(), batch)
        _logits, labels, _kwargs = torch.cross_entropy_calls[0]
        assert IGNORE_INDEX not in labels

    def test_material_for_another_objective_is_refused(self) -> None:
        """A batch names the objective that consumes it; a mismatch means one of the two is
        wrong - **and the loss would still produce a number**."""
        from training.signals.base import PreferenceExample

        batch = TrainingBatch(
            objective="sft",
            record={},
            examples=(PreferenceExample(prompt_id="p", prompt="q", chosen="a", rejected="b"),),
            counts=_counts(),
            tokenizer=Tokenizer(),
            masking_policy="completion_only",
        )
        with pytest.raises(Refusal) as caught:
            sft.OBJECTIVE.loss(_Model(), batch)
        assert caught.value.code == "batch_material_mismatch"

    def test_an_empty_batch_is_refused(self) -> None:
        batch = TrainingBatch(
            objective="sft",
            record={},
            examples=(),
            counts=_counts(),
            masking_policy="whole_sequence",
        )
        with pytest.raises(Refusal) as caught:
            sft.OBJECTIVE.loss(_Model(), batch)
        assert caught.value.code == "batch_empty"


def _counts() -> Any:
    from training.signals.base import BatchCounts

    return BatchCounts(admitted=1)


class _Model:
    """A model that returns logits shaped like a real one, without any tensor library."""

    def __call__(self, *, input_ids: Any, attention_mask: Any) -> Any:
        rows = input_ids.data
        vocab = 128
        return _FakeTensor([[[0.0] * vocab for _ in row] for row in rows])

    def parameters(self) -> list[Any]:
        return []


class _FakeTensor:
    """Enough tensor semantics to carry data through the slicing the loss does.

    A *recording* double rather than a mock: the labels really are sliced, so the assertion
    is about what the loss would hand ``cross_entropy``, not about what the test arranged.
    """

    def __init__(self, data: list[Any]) -> None:
        self.data = data

    def __getitem__(self, key: Any) -> _FakeTensor:
        if isinstance(key, tuple):
            rows = self.data[key[0]]
            return _FakeTensor([row[key[1]] for row in rows])
        return _FakeTensor(self.data[key])

    def size(self, dim: int) -> int:
        node: Any = self.data
        for _ in range(dim % 3):
            node = node[0]
        return len(node[0]) if isinstance(node[0], list) else len(node)

    def reshape(self, *shape: int) -> _FakeTensor:
        flat: list[Any] = []
        for row in self.data:
            flat.extend(row)
        return _FakeTensor(flat)


class _FakeTorch:
    """The one import seam, recorded. ``nn.functional.cross_entropy`` captures its payload."""

    long = "long"

    def __init__(self) -> None:
        self.cross_entropy_calls: list[tuple[Any, Any, dict[str, Any]]] = []
        outer = self

        class _Functional:
            @staticmethod
            def cross_entropy(logits: Any, labels: Any, **kwargs: Any) -> Any:
                outer.cross_entropy_calls.append((logits.data, labels.data, kwargs))
                return 0.5

        class _Nn:
            functional = _Functional()

        self.nn = _Nn()

    def tensor(self, data: list[Any], dtype: Any = None, device: Any = None) -> _FakeTensor:
        return _FakeTensor(data)
