"""The step: three stages, in order, each once  (R-02.12; FR-001.17 - FR-001.20, AC-001.5).

**The optimizer and the model are not mocked.** A mocked optimizer makes the three stages
agree by construction, which is precisely how a suite stays green over a real bug (ADR-016),
so the fakes below are *recording* implementations that really hold parameters, really
accumulate gradients and really record the order in which they were touched. Only the batch
source is a stub, because F2 owns the sources.

**Assertions are on the payload and the call count**, never on "was called". A reachability
assertion that only checks invocation passes against a call sending the wrong arguments -
which is the whole reason ADR-016 words its first rule the way it does.

## The one this file exists for

Stage 2 is **one registry lookup on the batch's declared objective**. A hard-coded
cross-entropy here would become the surviving definition of F2's contract, and it would be
worse than a missing feature: a DPO or GRPO batch would train with the wrong loss while the
loss still *falls* and the batch record still says ``dpo``. The record and the computation
would disagree, silently.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, ClassVar

import pytest

from core.errors import Refusal
from training.step import StepOutcome, resolve_objective, run_one


class Scalar:
    """A loss value that records whether ``backward()`` was called, and when.

    Not a mock: it is the real shape the step consumes - something with ``backward()``,
    ``detach()`` and ``item()`` - and it writes into the shared trace, which is what lets the
    order be asserted rather than inferred.
    """

    def __init__(self, value: float, trace: list[str]) -> None:
        self.value = value
        self.trace = trace
        self.backward_calls = 0

    def backward(self) -> None:
        self.backward_calls += 1
        self.trace.append("backward")

    def detach(self) -> Scalar:
        return self

    def item(self) -> float:
        return self.value


@dataclass
class Param:
    """A parameter that holds a gradient, so ``zero_grad`` has something to clear."""

    grad: Any = None


@dataclass
class Model:
    """A parameter holder. Real enough that the step's three stages cannot agree by accident."""

    trace: list[str]
    params: list[Param] = field(default_factory=lambda: [Param(), Param()])

    def parameters(self) -> list[Param]:
        return self.params


@dataclass
class Optimizer:
    """A recording optimizer. ``step`` and ``zero_grad`` both land in the trace."""

    trace: list[str]
    steps: int = 0
    zeroed: int = 0
    zero_grad_kwargs: list[dict[str, Any]] = field(default_factory=list)

    def step(self) -> None:
        self.steps += 1
        self.trace.append("optimizer.step")

    def zero_grad(self, **kwargs: Any) -> None:
        self.zeroed += 1
        self.zero_grad_kwargs.append(kwargs)
        self.trace.append("optimizer.zero_grad")


@dataclass
class Batch:
    """What F2 will hand the step. ``objective`` and a provenance ``record``."""

    objective: str = "sft"
    record: dict[str, Any] = field(default_factory=lambda: {"batch_sha256": "f" * 64})


@dataclass
class Source:
    """One batch per call, counted. The only stub here - F2 owns the sources."""

    trace: list[str]
    batch: Batch
    calls: int = 0

    def next(self) -> Batch:
        self.calls += 1
        self.trace.append("batch_source.next")
        return self.batch


@dataclass
class RecordingObjective:
    """An objective that records what it was handed. **F2 owns every real one.**"""

    name: str
    trace: list[str]
    value: float = 1.25
    seen: list[tuple[Any, Any]] = field(default_factory=list)

    def loss(self, model: Any, batch: Any) -> Scalar:
        self.seen.append((model, batch))
        self.trace.append(f"loss:{self.name}")
        return Scalar(self.value, self.trace)


@dataclass
class Registry:
    """F2's registry shape: ``get(name)`` raising on an unknown name."""

    objectives: dict[str, RecordingObjective]
    lookups: list[str] = field(default_factory=list)

    def get(self, name: str) -> RecordingObjective:
        self.lookups.append(name)
        if name not in self.objectives:
            raise KeyError(name)
        return self.objectives[name]


@dataclass
class Cancel:
    """A cancellation token that counts checks and can raise."""

    trace: list[str]
    raises: bool = False
    checks: int = 0

    def check(self) -> None:
        self.checks += 1
        self.trace.append("cancel.check")
        if self.raises:
            raise _Cancelled()


class _Cancelled(BaseException):
    """A ``BaseException``, as the foundation's ``RunCancelled`` is."""


@pytest.fixture
def rig():
    trace: list[str] = []
    objective = RecordingObjective(name="sft", trace=trace)
    return {
        "trace": trace,
        "model": Model(trace=trace),
        "optimizer": Optimizer(trace=trace),
        "source": Source(trace=trace, batch=Batch()),
        "registry": Registry({"sft": objective}),
        "objective": objective,
    }


class TestThreeStagesInOrderEachOnce:
    def test_the_observed_sequence_is_exactly_the_three_stages(self, rig) -> None:
        """**The order is the requirement.** Applying the update before ``backward()``
        silently trains on stale gradients and *the loss still moves* - so only the observed
        sequence catches it, never the outcome."""
        run_one(rig["model"], rig["optimizer"], rig["source"], objectives=rig["registry"])
        assert rig["trace"] == [
            "batch_source.next",
            "loss:sft",
            "backward",
            "optimizer.step",
            "optimizer.zero_grad",
        ]

    def test_each_stage_runs_exactly_once(self, rig) -> None:
        """**The call count, not merely "was called".** A step that drew two batches, or
        applied two updates, would satisfy an invocation check and break R-02.14's
        per-step record."""
        run_one(rig["model"], rig["optimizer"], rig["source"], objectives=rig["registry"])
        assert rig["source"].calls == 1
        assert len(rig["objective"].seen) == 1
        assert rig["optimizer"].steps == 1
        assert rig["optimizer"].zeroed == 1

    def test_the_objective_is_handed_THE_MODEL_AND_THE_BATCH(self, rig) -> None:
        """The payload. An objective handed the wrong model would train the wrong thing and
        report a perfectly plausible loss."""
        run_one(rig["model"], rig["optimizer"], rig["source"], objectives=rig["registry"])
        seen_model, seen_batch = rig["objective"].seen[0]
        assert seen_model is rig["model"]
        assert seen_batch is rig["source"].batch

    def test_gradients_are_released_rather_than_zeroed_in_place(self, rig) -> None:
        """Zeroed gradient tensors stay allocated, and on a card budgeted to the megabyte
        that is the difference between fitting and not."""
        run_one(rig["model"], rig["optimizer"], rig["source"], objectives=rig["registry"])
        assert rig["optimizer"].zero_grad_kwargs == [{"set_to_none": True}]

    def test_an_optimizer_without_set_to_none_still_works(self, rig) -> None:
        """Older optimizers take no keyword; the step must not refuse one over a kwarg."""

        class Old(Optimizer):
            def zero_grad(self, **kwargs: Any) -> None:  # type: ignore[override]
                if kwargs:
                    raise TypeError("zero_grad() takes no keyword arguments")
                self.zeroed += 1
                self.trace.append("optimizer.zero_grad")

        optimizer = Old(trace=rig["trace"])
        run_one(rig["model"], optimizer, rig["source"], objectives=rig["registry"])
        assert optimizer.zeroed == 1


class TestTheLossSeam:
    """**The defect two internally-consistent document chains disagreed about.**"""

    def test_stage_2_dispatches_on_the_batchs_declared_objective(self, rig) -> None:
        rig["source"].batch = Batch(objective="sft")
        run_one(rig["model"], rig["optimizer"], rig["source"], objectives=rig["registry"])
        assert rig["registry"].lookups == ["sft"]

    def test_a_dpo_batch_resolves_the_dpo_objective_not_a_fixed_one(self, rig) -> None:
        """The mutation this guards: *"replace the lookup with a fixed cross-entropy"*. The
        loss would still fall and the batch record would still say ``dpo``."""
        dpo = RecordingObjective(name="dpo", trace=rig["trace"])
        registry = Registry({"sft": rig["objective"], "dpo": dpo})
        rig["source"].batch = Batch(objective="dpo")

        outcome = run_one(rig["model"], rig["optimizer"], rig["source"], objectives=registry)

        assert registry.lookups == ["dpo"]
        assert len(dpo.seen) == 1
        assert rig["objective"].seen == [], "the sft objective must not have been consulted"
        # The RESOLVED objective's identity is recorded, not the name the batch claimed. A
        # test asserting the batch's own string would pass against a registry that returned
        # the wrong objective.
        assert outcome.objective == "dpo"

    def test_the_recorded_objective_is_the_RESOLVED_ones_identity(self, rig) -> None:
        """A registry that resolved ``dpo`` to the SFT objective is a real failure mode -
        a typo in one registration block - and only this assertion sees it."""
        registry = Registry({"dpo": RecordingObjective(name="sft", trace=rig["trace"])})
        rig["source"].batch = Batch(objective="dpo")
        outcome = run_one(rig["model"], rig["optimizer"], rig["source"], objectives=registry)
        assert outcome.objective == "sft", (
            "the step records what the registry actually returned, so a mis-registration "
            "shows up in the run log rather than being hidden by the batch's own claim"
        )

    def test_a_batch_declaring_no_objective_REFUSES(self) -> None:
        """ADR-018: ``getattr(batch, "objective", "sft")`` is a default that always fires,
        and it would train every batch with one loss the moment an attribute name changed."""

        class Objectiveless:
            record: ClassVar[dict[str, Any]] = {}

        with pytest.raises(Refusal) as caught:
            resolve_objective(Objectiveless(), Registry({}))
        assert caught.value.code == "batch_objective_missing"
        assert "guessing" in caught.value.message

    def test_an_empty_objective_name_refuses(self) -> None:
        with pytest.raises(Refusal) as caught:
            resolve_objective(Batch(objective=""), Registry({}))
        assert caught.value.code == "batch_objective_missing"

    def test_an_absent_registry_refuses_BY_NAME_rather_than_defaulting(
        self, rig, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Until F2 lands there is nothing to resolve against, and *refusing* is the only
        honest answer - a substituted loss is the fabricated value ADR-018 forbids, applied
        to the whole purpose of the step."""
        import builtins

        real_import = builtins.__import__

        def no_objectives(name, *args, **kwargs):
            if name == "training" and args and "objectives" in (args[2] or ()):
                raise ImportError("no module named training.objectives")
            if name.startswith("training.objectives"):
                raise ImportError("no module named training.objectives")
            return real_import(name, *args, **kwargs)

        monkeypatch.setattr(builtins, "__import__", no_objectives)
        with pytest.raises(Refusal) as caught:
            run_one(rig["model"], rig["optimizer"], rig["source"])
        assert caught.value.code == "objective_registry_unavailable"
        assert "will not substitute a cross-entropy" in caught.value.message

    def test_no_module_in_the_training_tree_defines_a_loss(self) -> None:
        """**The structural half of the seam.** F1 defines no objective name and implements
        no loss; the losses are F2's (``training/objectives/``). A ``cross_entropy`` written
        here as a placeholder would be the one that survived.
        """
        import ast
        from pathlib import Path

        src = Path(__file__).resolve().parents[3] / "backend" / "src" / "training"
        offenders: list[str] = []
        for path in src.rglob("*.py"):
            # `"objectives" in path.parts`, not `path.parts[-2:] == ("training",
            # "objectives")`: for `training/objectives/sft.py` the last two parts are
            # `("objectives", "sft.py")`, so the original exclusion could only ever match a
            # FILE named `objectives`. It excluded nothing, and was green only because
            # `training/objectives/` did not exist yet - the construction-green failure this
            # project keeps finding. Corrected when F2 landed the package this guard's own
            # docstring points at. The guard itself is unchanged and still bites: F1 must
            # define no loss, and the three attribute names are still the ones searched for.
            if "hotupdate" in path.parts or "objectives" in path.parts:
                continue
            for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
                if isinstance(node, ast.Attribute) and node.attr in {
                    "cross_entropy",
                    "nll_loss",
                    "binary_cross_entropy",
                }:
                    offenders.append(f"{path.relative_to(src)}:{node.lineno}")
        assert not offenders, (
            f"F1 must compute no loss of its own; found {offenders}. The objectives are "
            f"F2's and a placeholder here becomes the surviving definition of its contract."
        )


class TestTheCancellationBoundary:
    def test_the_check_is_before_stage_1(self, rig) -> None:
        cancel = Cancel(trace=rig["trace"])
        run_one(
            rig["model"],
            rig["optimizer"],
            rig["source"],
            cancel=cancel,
            objectives=rig["registry"],
        )
        assert rig["trace"][0] == "cancel.check"
        assert cancel.checks == 1, "once, at the boundary - not between stages"

    def test_a_cancelled_step_does_not_start(self, rig) -> None:
        """ADR-007: a step either completes and is recorded, or has not started. A check
        between the forward and the update would produce exactly the half-applied step
        R-02.31 forbids - re-applying a non-idempotent AdamW update to a half-written
        parameter set compounds the damage rather than repairing it."""
        cancel = Cancel(trace=rig["trace"], raises=True)
        with pytest.raises(_Cancelled):
            run_one(
                rig["model"],
                rig["optimizer"],
                rig["source"],
                cancel=cancel,
                objectives=rig["registry"],
            )
        assert rig["source"].calls == 0
        assert rig["optimizer"].steps == 0
        assert rig["trace"] == ["cancel.check"]

    def test_the_check_appears_exactly_once_in_the_step_source(self) -> None:
        """A second ``cancel.check()`` inside the body is the mutation; it is a one-line
        edit and its only visible consequence is a half-applied step after a crash."""
        import ast
        from pathlib import Path

        source = (
            Path(__file__).resolve().parents[3] / "backend" / "src" / "training" / "step.py"
        ).read_text(encoding="utf-8")
        tree = ast.parse(source)
        run_one_def = next(
            node
            for node in ast.walk(tree)
            if isinstance(node, ast.FunctionDef) and node.name == "run_one"
        )
        checks = [
            node
            for node in ast.walk(run_one_def)
            if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr == "check"
        ]
        assert len(checks) == 1, f"expected one cancel.check() in run_one, found {len(checks)}"


class TestTheOutcome:
    def test_the_loss_value_and_the_wall_time_are_recorded(self, rig) -> None:
        outcome = run_one(rig["model"], rig["optimizer"], rig["source"], objectives=rig["registry"])
        assert isinstance(outcome, StepOutcome)
        assert outcome.loss == pytest.approx(1.25)
        assert outcome.wall_time_ms >= 0.0
        assert outcome.finite is True

    def test_the_batch_hash_is_READ_from_the_record_not_recomputed(self, rig) -> None:
        """Recomputing would be a second definition of what a batch's content hash covers,
        and ADR-005's rule applies: two canonicalisers are two answers."""
        rig["source"].batch = Batch(record={"batch_sha256": "a" * 64})
        outcome = run_one(rig["model"], rig["optimizer"], rig["source"], objectives=rig["registry"])
        assert outcome.batch_sha256 == "a" * 64

    def test_a_batch_with_no_hash_records_None_rather_than_a_placeholder(self, rig) -> None:
        rig["source"].batch = Batch(record={})
        outcome = run_one(rig["model"], rig["optimizer"], rig["source"], objectives=rig["registry"])
        assert outcome.batch_sha256 is None

    def test_a_non_finite_loss_is_KEPT_and_the_step_is_marked_failed(self, rig) -> None:
        """E-12: *a missing number is indistinguishable from a healthy one.* So the value
        stays and ``finite`` carries the verdict - two separate facts."""
        rig["objective"].value = float("nan")
        outcome = run_one(rig["model"], rig["optimizer"], rig["source"], objectives=rig["registry"])
        assert outcome.loss != outcome.loss, "the NaN must survive into the outcome"
        assert outcome.finite is False

    def test_an_infinite_loss_is_also_marked_non_finite(self, rig) -> None:
        rig["objective"].value = float("inf")
        outcome = run_one(rig["model"], rig["optimizer"], rig["source"], objectives=rig["registry"])
        assert outcome.finite is False

    def test_a_gradient_norm_that_cannot_be_taken_is_None_not_zero(self, rig) -> None:
        """Zero is a measurement - a model whose gradients all vanished - and absence is
        not. Reporting absence as zero makes broken instrumentation look like a real signal.
        """
        outcome = run_one(rig["model"], rig["optimizer"], rig["source"], objectives=rig["registry"])
        assert outcome.grad_norm is None


class TestTheStepHasNoBarrierAndNoSiblingReference:
    """AC-001.5, FR-001.18 - asserted over the source, not by reading it."""

    def test_the_step_module_names_no_barrier_concept(self) -> None:
        """R-02.52, R-02.30 and R-02.31 are §5.10's and F11's. The *mechanism* must be
        absent from this path, not merely unused, because §5.10 grew to thirteen required
        rows once without anyone deciding it should."""
        import ast
        from pathlib import Path

        path = Path(__file__).resolve().parents[3] / "backend" / "src" / "training" / "step.py"
        tree = ast.parse(path.read_text(encoding="utf-8"))
        names: set[str] = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Name):
                names.add(node.id)
            elif isinstance(node, ast.Attribute):
                names.add(node.attr)
            elif isinstance(node, (ast.Import, ast.ImportFrom)):
                names.update(alias.name for alias in node.names)
                if isinstance(node, ast.ImportFrom) and node.module:
                    names.add(node.module)
        forbidden = {"quiesce", "commit_begin", "commit_end", "resume", "watchdog", "torn"}
        assert not (names & forbidden), f"the step references §5.10 machinery: {names & forbidden}"

    def test_the_step_makes_no_outbound_call(self, rig, monkeypatch: pytest.MonkeyPatch) -> None:
        """No HTTP client is reachable from the step. Enforced by breaking the one this
        project would use, so an added call fails rather than merely being noticed."""
        import httpx

        def explode(*_args, **_kwargs):  # pragma: no cover - the point is it is not hit
            raise AssertionError("the step must make no network call")

        monkeypatch.setattr(httpx, "request", explode, raising=False)
        monkeypatch.setattr(httpx.Client, "request", explode, raising=False)
        run_one(rig["model"], rig["optimizer"], rig["source"], objectives=rig["registry"])
