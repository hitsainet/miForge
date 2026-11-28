"""The objective registry, and F1's seam through it  (FR-002.3; ADR-016 #1).

**The negative control for this file:** delete one ``register(...)`` from
``training/objectives/__init__.py``'s ``ensure_registered`` and the first test goes red.

The seam test matters more than it looks. R-02.12 ends *"Nothing else is in the step"*, and
a hard-coded cross-entropy there would have become the surviving definition of F2's
contract: a DPO or GRPO batch would train with the wrong loss while the loss still **falls**
and the batch record still says ``dpo``. So the dispatch is asserted through F1's **real**
``run_one`` with the real registry, on the payload and the call count.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import pytest

from core.errors import Refusal
from training import objectives
from training.step import default_objective_registry, resolve_objective, run_one


class TestTheRegistry:
    def test_the_three_objectives_resolve_from_the_live_registry(self) -> None:
        assert objectives.names() == ("dpo", "grpo", "sft")
        for name in objectives.names():
            assert objectives.get(name).name == name

    def test_an_unknown_objective_raises_rather_than_defaulting(self) -> None:
        """An ``Optional`` return invites ``get(name) or default``, and the only default
        available is the loss nobody chose."""
        with pytest.raises(Refusal) as caught:
            objectives.get("cross-entropy")
        assert caught.value.code == "objective_unknown"
        assert caught.value.details["registered"] == ["dpo", "grpo", "sft"]

    def test_registering_a_name_twice_raises(self) -> None:
        with pytest.raises(Refusal) as caught:
            objectives.register(objectives.get("sft"))
        assert caught.value.code == "objective_already_registered"

    def test_every_source_names_a_registered_objective(self) -> None:
        """Four sources, three objectives: ``sft`` serves ``corpus`` **and**
        ``self-generation``, because R-02.5's accepted completions are supervised
        examples - R-02.4's objective on different material."""
        from training.signals import ensure_registered, registry

        ensure_registered()
        assert set(objectives.SOURCE_OBJECTIVE) == set(registry.names())
        for source, objective in objectives.SOURCE_OBJECTIVE.items():
            assert objective in objectives.names()
            assert registry.capability(source).objective == objective
        assert (
            objectives.SOURCE_OBJECTIVE["corpus"] == objectives.SOURCE_OBJECTIVE["self-generation"]
        )


class TestF1sSeam:
    def test_the_package_satisfies_f1s_registry_protocol(self) -> None:
        """F1 imports this package lazily and casts it to ``ObjectiveRegistry``."""
        registry = default_objective_registry()
        assert registry.get("sft").name == "sft"

    def test_the_step_dispatches_on_the_batch_and_calls_the_loss_once_with_it(self) -> None:
        """**The payload and the call count.** "Was called" passes against a call sending
        the wrong batch - which is precisely the failure this seam exists to prevent."""
        batch = _Batch(objective="dpo")
        recorder = _Recorder("dpo")
        outcome = run_one(
            _Model(), _Optimizer(), _Source(batch), objectives=_Registry({"dpo": recorder})
        )
        assert recorder.calls == [batch]
        assert outcome.objective == "dpo"
        assert outcome.batch_sha256 == "f" * 64

    def test_a_batch_naming_no_objective_refuses(self) -> None:
        with pytest.raises(Refusal) as caught:
            resolve_objective(_Batch(objective=""), _Registry({}))
        assert caught.value.code == "batch_objective_missing"

    def test_an_unknown_objective_from_a_batch_reaches_the_registrys_refusal(self) -> None:
        with pytest.raises(Refusal) as caught:
            resolve_objective(_Batch(objective="sft2"), objectives)
        assert caught.value.code == "objective_unknown"


# --------------------------------------------------------------------------- doubles
# Recording implementations, not mocks: they really hold a gradient and really record the
# order they were touched in, because a mocked optimizer makes the three stages agree by
# construction (ADR-016).


@dataclass
class _Batch:
    objective: str
    record: dict[str, Any] = field(default_factory=lambda: {"batch_sha256": "f" * 64})


class _Source:
    def __init__(self, batch: _Batch) -> None:
        self._batch = batch

    def next(self) -> _Batch:
        return self._batch


class _Scalar:
    def __init__(self) -> None:
        self.backward_calls = 0

    def backward(self) -> None:
        self.backward_calls += 1

    def detach(self) -> _Scalar:
        return self

    def item(self) -> float:
        return 0.5


class _Recorder:
    def __init__(self, name: str) -> None:
        self.name = name
        self.calls: list[Any] = []

    def loss(self, model: Any, batch: Any) -> _Scalar:
        self.calls.append(batch)
        return _Scalar()


class _Registry:
    def __init__(self, table: dict[str, Any]) -> None:
        self._table = table

    def get(self, name: str) -> Any:
        if name not in self._table:
            from core.errors import refuse

            raise refuse("objective_unknown", f"no {name}", registered=sorted(self._table))
        return self._table[name]


class _Model:
    def parameters(self) -> list[Any]:
        return []


class _Optimizer:
    def __init__(self) -> None:
        self.steps = 0

    def step(self) -> None:
        self.steps += 1

    def zero_grad(self, set_to_none: bool = False) -> None:
        return None
