"""No value network, asserted as a property  (R-02.7, AC-002.11).

*"A policy-gradient objective (GRPO or equivalent, no value network."* The clause is
**requirement text**, so it is a testable property rather than a stylistic note - and it is
tested by inspecting the **real optimizer's parameter groups before and after a step**,
never by reading the file. A second trainable module would also break the budget R-02.42
estimated the run against (F8's), which is the second reason it cannot be left to good
intentions.
"""

from __future__ import annotations

from typing import Any

import pytest

from core.errors import Refusal
from training.objectives import grpo
from training.signals.base import RewardGroup, TrainingBatch
from training.signals.reward import group


class TestNoValueNetwork:
    def test_a_reward_step_adds_no_parameter_group(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """AC-002.11, on a real optimizer object, before and after.

        The optimizer is a recording implementation that really holds parameter groups, so
        a module created by the loss would appear in them. A mocked optimizer would make
        the assertion agree with itself (ADR-016).
        """
        torch = _FakeTorch()
        monkeypatch.setattr(grpo, "torch_module", lambda: torch)
        monkeypatch.setattr("training.objectives.dpo.torch_module", lambda: torch)
        model = _Model()
        optimizer = _Optimizer(model.parameter_list)
        before = [list(g["params"]) for g in optimizer.param_groups]

        grpo.OBJECTIVE.loss(model, _batch())

        after = [list(g["params"]) for g in optimizer.param_groups]
        assert before == after
        assert len(optimizer.param_groups) == 1

    def test_the_objective_holds_no_state_of_its_own(self) -> None:
        """A value head kept on the objective would survive between steps and be invisible
        to the optimizer, which is the version of this defect a parameter-group check
        alone would miss."""
        assert vars(grpo.OBJECTIVE) == {}

    def test_the_advantage_carries_no_gradient(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """The advantage comes from the grader, not from the model: making it a tensor with
        a gradient would backpropagate into the reward, which is not a policy gradient."""
        torch = _FakeTorch()
        monkeypatch.setattr(grpo, "torch_module", lambda: torch)
        monkeypatch.setattr("training.objectives.dpo.torch_module", lambda: torch)
        grpo.OBJECTIVE.loss(_Model(), _batch())
        assert all(isinstance(term, float) for term in torch.stacked[0])


class TestRefusals:
    def test_material_for_another_objective_is_refused(self) -> None:
        from training.signals.base import SupervisedExample

        batch = TrainingBatch(
            objective="grpo",
            record={},
            examples=(SupervisedExample(prompt_id="p", prompt="", completion="x"),),
            counts=_counts(),
        )
        with pytest.raises(Refusal) as caught:
            grpo.OBJECTIVE.loss(_Model(), batch)
        assert caught.value.code == "batch_material_mismatch"

    def test_an_empty_batch_is_refused(self) -> None:
        with pytest.raises(Refusal) as caught:
            grpo.OBJECTIVE.loss(
                _Model(), TrainingBatch(objective="grpo", record={}, examples=(), counts=_counts())
            )
        assert caught.value.code == "batch_empty"


def _batch() -> TrainingBatch:
    groups: tuple[RewardGroup, ...] = (group("p-1", "q", ["a", "b"], [0.1, 0.9]),)
    return TrainingBatch(
        objective="grpo", record={}, examples=groups, counts=_counts(), tokenizer=_Tokenizer()
    )


def _counts() -> Any:
    from training.signals.base import BatchCounts

    return BatchCounts(admitted=1)


class _Tokenizer:
    def encode(self, text: str, add_special_tokens: bool = False) -> list[int]:
        return [ord(c) for c in text]


class _Model:
    """A model that really holds parameters, so a new one would be visible."""

    def __init__(self) -> None:
        self.parameter_list = [_Parameter("w1"), _Parameter("w2")]

    def parameters(self) -> list[Any]:
        return self.parameter_list

    def __call__(self, *, input_ids: Any, attention_mask: Any) -> Any:
        return _FakeTensor([[[0.0] * 128 for _ in input_ids.data[0]]])


class _Parameter:
    def __init__(self, name: str) -> None:
        self.name = name


class _Optimizer:
    def __init__(self, params: list[Any]) -> None:
        self.param_groups = [{"params": list(params)}]


class _FakeTensor:
    def __init__(self, data: Any) -> None:
        self.data = data

    def __getitem__(self, key: Any) -> _FakeTensor:
        if isinstance(key, tuple):
            rows = self.data[key[0]]
            return _FakeTensor([row[key[1]] for row in rows])
        return _FakeTensor(self.data[key])

    def gather(self, dim: int, index: Any) -> _FakeTensor:
        return _FakeTensor([[0.0 for _ in row] for row in self.data])

    def squeeze(self, dim: int) -> _FakeTensor:
        return self

    def unsqueeze(self, dim: int) -> _FakeTensor:
        return self

    def clamp(self, min: int = 0) -> _FakeTensor:
        return self

    def sum(self) -> float:
        return -1.0

    def __ne__(self, other: Any) -> _FakeTensor:  # type: ignore[override]
        return _FakeTensor([[1 for _ in row] for row in self.data])

    def __mul__(self, other: Any) -> _FakeTensor:
        return self


class _FakeTorch:
    long = "long"

    def __init__(self) -> None:
        self.stacked: list[list[Any]] = []

        class _Functional:
            @staticmethod
            def log_softmax(tensor: Any, dim: int = -1) -> Any:
                return tensor

            @staticmethod
            def logsigmoid(value: Any) -> Any:
                return value

        class _Nn:
            functional = _Functional()

        self.nn = _Nn()

    def tensor(self, data: Any, dtype: Any = None, device: Any = None) -> _FakeTensor:
        return _FakeTensor(data)

    def ones_like(self, tensor: Any) -> _FakeTensor:
        return tensor

    def stack(self, items: list[Any]) -> Any:
        self.stacked.append(list(items))
        return _Mean()


class _Mean:
    def mean(self) -> float:
        return 0.25
