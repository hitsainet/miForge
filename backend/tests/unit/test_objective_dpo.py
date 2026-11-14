"""The DPO reference source: explicit, recorded, and refused when absent  (FR-002.21, E-13).

BRD-02 §9 leaves **how** open - *"keep a frozen copy in CPU RAM ... or cache reference
logprobs per batch before the run?"* - and leaving a choice open is not leaving it absent.
**Without a reference this is not a preference objective at all**: it degenerates to
preferring the chosen continuation unconditionally, and it would still produce a falling
loss. That is why every path here refuses rather than defaulting.
"""

from __future__ import annotations

from dataclasses import replace
from typing import Any

import pytest

from core.errors import Refusal
from training.objectives import dpo
from training.signals.base import PreferenceExample, TrainingBatch
from training.signals.settings import SignalSettings, require_reference_source


def pair(**overrides: Any) -> PreferenceExample:
    example = PreferenceExample(prompt_id="p-1", prompt="q", chosen="good", rejected="bad")
    return replace(example, **overrides) if overrides else example


class TestTheConfiguredSource:
    def test_an_unset_source_refuses_naming_both_options(self) -> None:
        with pytest.raises(Refusal) as caught:
            require_reference_source(SignalSettings(dpo_reference_source=None))
        assert caught.value.code == "dpo_reference_source_not_configured"
        assert caught.value.details["options"] == ["frozen_copy", "precomputed"]

    @pytest.mark.parametrize("source", ["frozen_copy", "precomputed"])
    def test_a_configured_source_is_returned_unchanged(self, source: str) -> None:
        assert require_reference_source(SignalSettings(dpo_reference_source=source)) == source

    def test_an_unknown_source_is_refused_rather_than_chosen_between(self) -> None:
        with pytest.raises(Refusal) as caught:
            dpo.resolve_reference("whatever")
        assert caught.value.code == "dpo_reference_source_unknown"

    def test_frozen_copy_without_a_frozen_model_refuses(self) -> None:
        """A reference built from the *current* policy would compare the policy with
        itself, which removes the reference term while the loss keeps falling."""
        with pytest.raises(Refusal) as caught:
            dpo.resolve_reference("frozen_copy", frozen_model=None)
        assert caught.value.code == "dpo_reference_model_unavailable"

    def test_precomputed_without_cached_values_refuses_per_pair(self) -> None:
        reference = dpo.resolve_reference("precomputed")
        with pytest.raises(Refusal) as caught:
            reference.for_pair(pair(), tokenizer=None)
        assert caught.value.code == "dpo_reference_logprobs_missing"

    def test_precomputed_returns_the_cached_values_when_they_are_there(self) -> None:
        example = pair(reference_chosen_logp=-1.0, reference_rejected_logp=-2.0)
        reference = dpo.resolve_reference("precomputed")
        assert reference.for_pair(example, tokenizer=None) == (-1.0, -2.0)

    def test_each_implementation_names_the_source_recorded_on_the_batch(self) -> None:
        """A DPO loss figure is comparable with another only when both say what they were
        measured against."""
        assert dpo.resolve_reference("precomputed").source == "precomputed"
        assert dpo.resolve_reference("frozen_copy", frozen_model=object()).source == "frozen_copy"


class TestTheLoss:
    def test_a_batch_naming_no_reference_source_is_refused(self) -> None:
        batch = TrainingBatch(
            objective="dpo",
            record={},
            examples=(pair(),),
            counts=_counts(),
            reference_source=None,
        )
        with pytest.raises(Refusal) as caught:
            dpo.OBJECTIVE.loss(_Model(), batch)
        assert caught.value.code == "dpo_reference_source_absent"

    def test_material_for_another_objective_is_refused(self) -> None:
        from training.signals.base import SupervisedExample

        batch = TrainingBatch(
            objective="dpo",
            record={},
            examples=(SupervisedExample(prompt_id="p", prompt="", completion="x"),),
            counts=_counts(),
            reference_source="precomputed",
        )
        with pytest.raises(Refusal) as caught:
            dpo.OBJECTIVE.loss(_Model(), batch)
        assert caught.value.code == "batch_material_mismatch"

    def test_the_sequence_logprob_is_masked_to_the_continuation(self) -> None:
        """A preference is about the continuation: including the prompt's log-probability
        would make two pairs with the same continuations but different prompts
        incomparable. Asserted on the mask the encoder produces, which is the input the
        gather uses."""
        from training.objectives.tokenize import IGNORE_INDEX, encode_example

        _ids, labels = encode_example(_Tokenizer(), "qq", "aa", policy="completion_only")
        assert labels[:2] == [IGNORE_INDEX, IGNORE_INDEX]
        assert labels[2:] != [IGNORE_INDEX, IGNORE_INDEX]


def _counts() -> Any:
    from training.signals.base import BatchCounts

    return BatchCounts(admitted=1)


class _Tokenizer:
    def encode(self, text: str, add_special_tokens: bool = False) -> list[int]:
        return [ord(c) for c in text]


class _Model:
    def parameters(self) -> list[Any]:
        return []
