"""Cancellation lands **outside** the bracket  (ADR-007, R-02.31; TDD §7).

ADR-007: a step either completes and is recorded, or has not started. **A cancellation
honoured *inside* a commit bracket is a half-applied step and a ``torn`` set** - the one
thing both ADR-007 and R-02.31 forbid - and the serving process would be left refusing every
completion because a user pressed stop.

F1's step already checks the token before stage 1 and never between stages; F11 adds four
protocol calls around stage 3 and must not add a check inside them. So this file asserts the
*absence* two ways: behaviourally, by cancelling a run mid-flight and showing the bracket
either never opened or closed completely; and structurally, over the bracket's own syntax
tree, because an absence is easiest to reintroduce by being helpful.
"""

from __future__ import annotations

from typing import Any

import pytest

from tests.support.fixtures_hotupdate import Harness, RecordingOptimizer
from training.hotupdate import barrier
from workers.cancellation import RunCancelled


class CancelAtTheBoundary:
    """F1's ``Cancellation`` shape: ``check()`` raising at a boundary."""

    def __init__(self, *, cancelled: bool) -> None:
        self.cancelled = cancelled
        self.checks = 0

    def check(self) -> None:
        self.checks += 1
        if self.cancelled:
            raise RunCancelled("hot-run")


class Scalar:
    def backward(self) -> None:
        return None

    def detach(self) -> Scalar:
        return self

    def item(self) -> float:
        return 1.0


class Objective:
    name = "sft"

    def loss(self, model: Any, batch: Any) -> Scalar:
        return Scalar()


class Registry:
    def get(self, name: str) -> Objective:
        return Objective()


class Batch:
    objective = "sft"

    @property
    def record(self) -> dict[str, Any]:
        return {}


class Source:
    def next(self) -> Batch:
        return Batch()


class Parameter:
    """One parameter with no gradient. Enough to be iterated and filtered out.

    F1's gradient-norm step does
    ``[p.grad... for p in model.parameters() if getattr(p, "grad", None) is not None]``, so a
    parameter whose ``grad`` is ``None`` exercises the real iteration and the real filter
    without needing a tensor.
    """

    grad = None


class Model:
    """A module double that can express a COMPLETED step, not only a cancelled one.

    **Both call sites passed ``object()``, and that only worked on the short-circuit path.** A
    cancellation *before* the step never dereferences the model, so the cancelled test
    passed; the uncancelled test runs the whole step, reaches ``model.parameters()``, and
    raised ``AttributeError: 'object' object has no attribute 'parameters'``.

    It was green locally only because ``torch`` was missing from the venv, so the step
    refused earlier than that line. **With torch installed — as CI installs it and as
    ``pyproject.toml`` declares it — the real path runs and ``object()`` cannot stand in for
    a module.** Same trap F11 found in its own MUT-19a: *a double that cannot express the
    defect makes its assertion vacuous.*

    Deliberately **not** a ``torch.nn.Module``: this test is about the cancellation bracket,
    and a double that needed torch would put the test back under the environment's control,
    which is exactly what went wrong.
    """

    def parameters(self) -> list[Parameter]:
        return [Parameter()]


class TestACancelledRunNeverLeavesAnOpenBracket:
    def test_a_cancellation_before_the_step_opens_no_bracket_at_all(self, harness: Harness) -> None:
        harness.attach()
        harness.fake.calls.clear()
        optimizer = RecordingOptimizer([])
        with pytest.raises(RunCancelled):
            barrier.hot_update_step(
                Model(),
                optimizer,
                Source(),
                client=harness.client,
                timings=harness.timings,
                cancel=CancelAtTheBoundary(cancelled=True),
                objectives=Registry(),
            )
        assert harness.fake.call_count("quiesce") == 0
        assert harness.fake.call_count("commit-begin") == 0
        assert optimizer.steps == 0
        assert harness.fake.weights_state == "clean"
        # And serving was never held: a cancelled run must not cost a serving pause.
        assert harness.fake.quiesced is False

    def test_an_uncancelled_step_completes_the_bracket_and_checks_once(
        self, harness: Harness
    ) -> None:
        """The check happens **once, at the boundary** - F1's rule, unchanged by F11."""
        harness.attach()
        harness.fake.calls.clear()
        token = CancelAtTheBoundary(cancelled=False)
        barrier.hot_update_step(
            Model(),
            RecordingOptimizer([]),
            Source(),
            client=harness.client,
            timings=harness.timings,
            cancel=token,
            objectives=Registry(),
        )
        assert token.checks == 1
        assert harness.fake.ops()[-4:] == list(barrier.STAGE_ORDER)
        assert harness.fake.weights_state == "clean"

    def test_a_cancellation_raised_from_inside_the_write_leaves_the_commit_open(
        self, harness: Harness
    ) -> None:
        """Belt and braces for the case that must not happen: if something *did* cancel mid
        write, the write failed partway - so (operator decision 2026-10-17) commit-end is
        withheld and miLLM's watchdog declares the set torn. Nothing is served from it."""
        harness.attach()
        harness.fake.calls.clear()
        optimizer = barrier.BracketedOptimizer(
            RecordingOptimizer([], fail=RunCancelled("hot-run")),
            harness.client,
            harness.timings,
        )
        with pytest.raises(RunCancelled):
            optimizer.step()
        assert harness.fake.ops() == ["quiesce", "commit-begin"]
        assert harness.fake.weights_state == "writing"
        assert harness.fake.served_while_not_clean == []


class TestTheAbsenceIsStructural:
    def test_the_bracket_calls_no_cancellation_check(self) -> None:
        import ast
        import inspect

        tree = ast.parse(inspect.getsource(barrier))
        bracket = next(
            node
            for node in ast.walk(tree)
            if isinstance(node, ast.FunctionDef) and node.name == "commit_bracket"
        )
        names = {
            node.func.attr
            for node in ast.walk(bracket)
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
        } | {
            node.func.id
            for node in ast.walk(bracket)
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
        }
        assert "check" not in names
        assert "is_cancelled" not in names

    def test_the_step_checks_before_the_batch_not_between_the_stages(self) -> None:
        """F1's ``run_one`` owns that ordering and F11 does not re-implement it - the
        bracketed step calls F1's step with a wrapped optimizer. Asserted so a future
        'optimisation' that copies the three stages here is red."""
        import ast
        import inspect

        tree = ast.parse(inspect.getsource(barrier))
        step = next(
            node
            for node in ast.walk(tree)
            if isinstance(node, ast.FunctionDef) and node.name == "hot_update_step"
        )
        called = {
            node.func.id
            for node in ast.walk(step)
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
        }
        assert "run_one" in called, "F11 must call F1's step rather than re-implementing it"
