"""ADR-009's DEFAULT provider exists in the worker that holds the parameters.

*A capability is not shipped until a test FAILS when its wiring is removed* (ADR-016 #1).

``providers.registry.bind_model_handle`` and
:class:`providers.model_handle.AcquiredModelHandle` were called in fourteen test files and
**nowhere in ``backend/src/``**. So ``registry.get("in_process")`` refused
``inference_provider_unavailable`` in every process - and its refusal said *"It is
available in the training worker once the model is acquired (R-02.46)"*, **which nothing
made true.** ``scoring_provider`` defaults to ``in_process``, so F4's test and retention
scoring (R-02.16, R-02.17) refused in the one worker that could have answered, on the
standalone path acceptance criterion 20 pins.

## Nothing here constructs a handle

Every assertion goes through :func:`providers.registry.get` and
:func:`providers.registry.list_rows` - the paths F4's ``entrypoint.evaluate_run`` and
``GET /api/v1/providers`` actually resolve from - and the only thing that ever binds one is
``training.tasks.drive_run`` itself. A test that bound its own handle would be the
fourteenth file that passes while production has nothing.

## THE TWO CONTROLS

1. **Reachability.** Delete ``provider_registry.bind_acquired_model(acquired,
   step=start_step)`` in :func:`training.tasks.drive_run` and
   :class:`TestTheProviderIsReachableInsideARun` must go red with
   ``inference_provider_unavailable``.
2. **The payload, which is the one that matters here.** Binding once at acquisition with
   ``step=None`` would make control 1 green and stamp **every** score of the run
   ``<hash>@as-acquired`` - the *"ours is not an identity"* failure
   ``parameter_set_digest``'s own docstring exists to prevent, and strictly worse than the
   refusal because it is silent. So :class:`TestTheIdentityNamesTheStepsParameterSet`
   asserts the **exact sequence** of digests an evaluation sees across a run, which pins
   the payload and the rebind count together: one bind at ``start_step`` plus one per
   completed step, in order, and the string ``as-acquired`` nowhere.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pytest

from core.config import Settings, reset_settings
from core.errors import Refusal
from providers import registry as provider_registry
from tests.support.jobs import insert_run, memory_engine, session_for
from training import tasks, trainer_state
from training.acquisition.base import (
    AcquiredModel,
    AcquisitionMode,
    AcquisitionRecord,
    AcquisitionSpec,
    SourceKind,
)
from training.optimizer import OptimizerIdentity
from training.run_config import RunConfig

RUN = "run-in-process-wiring"
CFG = RunConfig(learning_rate=1e-5, warmup_steps=2, batch_size=2, gradient_accumulation=2)
STARTING_HASH = "a" * 64
BASE_MODEL = "local:base@abcdef0123456789"


# --------------------------------------------------------------------------------------
# The narrowest tensor fakes, copied from `test_step_loop.py`'s rig. Nothing above them is
# faked: the registry, the handle adapter and the provider are the production ones.
# --------------------------------------------------------------------------------------


class Loss:
    def __init__(self, value: float) -> None:
        self.value = value

    def backward(self) -> None:
        pass

    def detach(self) -> Loss:
        return self

    def item(self) -> float:
        return self.value


@dataclass
class Objective:
    name: str = "sft"
    losses: list[float] = field(default_factory=lambda: [0.9, 0.7, 0.5, 0.4, 0.35, 0.3])
    index: int = 0

    def loss(self, model: Any, batch: Any) -> Loss:
        value = self.losses[min(self.index, len(self.losses) - 1)]
        self.index += 1
        return Loss(value)


@dataclass
class Registry:
    objective: Objective

    def get(self, name: str) -> Objective:
        return self.objective


@dataclass
class Batch:
    objective: str = "sft"
    record: dict[str, Any] = field(default_factory=lambda: {"batch_sha256": "b" * 64})


@dataclass
class Source:
    def next(self) -> Batch:
        return Batch()


@dataclass
class Optimizer:
    steps: int = 0

    def step(self) -> None:
        self.steps += 1

    def zero_grad(self, **_kwargs: Any) -> None:
        pass

    def state_dict(self) -> dict[str, Any]:
        return {"steps": self.steps}


class Model:
    def __init__(self) -> None:
        self.config = type("C", (), {"use_cache": True})()

    def parameters(self) -> list[Any]:
        return []

    def gradient_checkpointing_enable(self) -> None:
        pass


class FakeTorch:
    class cuda:
        @staticmethod
        def is_available() -> bool:
            return False

    @staticmethod
    def get_rng_state() -> str:
        return "rng"

    @staticmethod
    def save(obj: Any, path: Path) -> None:
        Path(path).write_text(json.dumps(obj, default=str))

    @staticmethod
    def load(path: Path, **_kwargs: Any) -> Any:
        return json.loads(Path(path).read_text())


RECORD = AcquisitionRecord(
    mode=AcquisitionMode.LOAD,
    source_kind=SourceKind.LOCAL_DIR,
    source_path="/data/checkpoints/base",
    starting_weights_sha256=STARTING_HASH,
    base_model=BASE_MODEL,
    parent=None,
    param_count=1000,
)


@pytest.fixture(autouse=True)
def unbound(monkeypatch: pytest.MonkeyPatch):
    """Start and finish with nothing bound, and with no HTTP row configured.

    The unbind at both ends is what makes the standalone world assertable: a file that
    could not bind a provider *away* could not express acceptance criterion 20's case.
    """
    provider_registry.bind_model_handle(None)
    for key in list(provider_registry.http_provider_rows.__globals__["os"].environ):
        if key.startswith("PROVIDER_"):
            monkeypatch.delenv(key, raising=False)
    yield
    provider_registry.bind_model_handle(None)


@pytest.fixture
def rig(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    states = tmp_path / "trainer-state"
    checkpoints = tmp_path / "checkpoints"
    runs = tmp_path / "runs"
    for directory in (states, checkpoints, runs):
        directory.mkdir()
    monkeypatch.setattr(
        "core.config._settings",
        Settings(  # type: ignore[call-arg]
            _env_file=None, trainer_state_root=states, checkpoint_root=checkpoints
        ),
    )

    optimizer = Optimizer()
    monkeypatch.setattr(trainer_state, "_torch", lambda: FakeTorch)
    monkeypatch.setattr(tasks, "acquire_for_run", lambda spec: AcquiredModel(Model(), None, RECORD))
    monkeypatch.setattr(
        tasks.optimizer_module,
        "build",
        lambda module, cfg, state="fresh": (
            optimizer,
            OptimizerIdentity(
                name=cfg.optimizer_name(), state=state, config_hash=cfg.config_hash()
            ),
        ),
    )

    engine = memory_engine()
    with session_for(engine) as session:
        insert_run(session, RUN, status="running")

    yield {"engine": engine, "log_root": runs, "states": states}
    reset_settings()


def drive(rig, *, observer=None, **overrides):
    """Drive a run, with ``observer(step)`` called at every evaluation boundary."""
    kwargs: dict[str, Any] = {
        "run_id": RUN,
        "spec": AcquisitionSpec(path=Path("/data/checkpoints/base")),
        "cfg": CFG,
        "batch_source": Source(),
        "objectives": Registry(Objective()),
        "log_root": rig["log_root"].parent,
        "trainer_state_root": rig["states"],
        "max_steps": 1,
        # Production's two loop-owned behaviours, as `run_driver` sets them: the step-0
        # baseline is where the first score of a run is taken, so a test that skipped it
        # would not see the digest the baseline is recorded with.
        "evaluate_step_zero": True,
        "evaluate": observer,
    }
    kwargs.update(overrides)
    with session_for(rig["engine"]) as session:
        kwargs.setdefault("session", session)
        return tasks.drive_run(**kwargs)


def resolve_identity(step: int, seen: list[tuple[int, Any]]) -> None:
    """What F4 does at an evaluation boundary: pin the configured provider, by name.

    Through :func:`providers.registry.pin`, which is the call
    ``services.scoring.entrypoint.evaluate_run`` makes - not through a handle this test
    holds, because the handle is exactly the thing under test.
    """
    seen.append((step, provider_registry.pin(provider_registry.IN_PROCESS_NAME).identity))


# --------------------------------------------------------------------------------------
# CONTROL 1: reachability
# --------------------------------------------------------------------------------------


class TestTheProviderIsReachableInsideARun:
    """Remove the bind in ``drive_run`` and every test in this class goes red."""

    def test_the_default_provider_resolves_at_an_evaluation_boundary(self, rig) -> None:
        """The whole user-visible defect, in one assertion.

        ``scoring_provider`` defaults to ``in_process`` and F4 pins it at every evaluation.
        Before the wiring this raised ``inference_provider_unavailable`` **inside the
        training worker**, so a run with a test set scored nothing and said the provider
        was unavailable in the process that held the parameters.
        """
        seen: list[tuple[int, Any]] = []
        drive(rig, max_steps=3, observer=lambda step: resolve_identity(step, seen))
        assert [step for step, _ in seen] == [0, 1, 2, 3]
        assert all(identity.kind == "in_process" for _, identity in seen)

    def test_the_in_process_row_is_listed_available_during_a_run(self, rig) -> None:
        """``GET /api/v1/providers`` in the training worker. The row is **always** listed;
        before the wiring it was always listed *unavailable*, everywhere."""
        rows: list[Any] = []

        def observe(step: int) -> None:
            del step
            rows.append(
                next(
                    row
                    for row in provider_registry.list_rows()
                    if row.name == provider_registry.IN_PROCESS_NAME
                )
            )

        drive(rig, max_steps=1, observer=observe)
        assert [row.available for row in rows] == [True, True]
        assert [row.unavailable_reason for row in rows] == [None, None]
        assert [row.model for row in rows] == [BASE_MODEL, BASE_MODEL]
        # R-02.11 is structural and the binding does not loosen it: there is no
        # configuration under which the model being trained becomes judge-eligible.
        assert [row.judge_eligible for row in rows] == [False, False]

    def test_outside_a_run_it_still_refuses_by_name(self) -> None:
        """The API process holds no parameters, and that must stay a **named** refusal
        rather than a substituted provider (M-9) - ADR-013: an option missing from a
        response is indistinguishable from one that is broken."""
        assert provider_registry.current_model_handle() is None
        with pytest.raises(Refusal) as caught:
            provider_registry.get(provider_registry.IN_PROCESS_NAME)
        assert caught.value.code == "inference_provider_unavailable"


# --------------------------------------------------------------------------------------
# CONTROL 2: the payload - which parameter set the score names
# --------------------------------------------------------------------------------------


class TestTheIdentityNamesTheStepsParameterSet:
    """The control that a once-at-acquisition binding would **not** satisfy.

    *"Was bound" passes against a binding that names the wrong parameters.* So the
    assertion is the exact sequence of digests, which pins the payload and the rebind
    count in one: ``@step0`` at the baseline and ``@stepN`` at every boundary after step
    ``N``.
    """

    def test_every_boundary_names_its_own_step(self, rig) -> None:
        seen: list[tuple[int, Any]] = []
        drive(rig, max_steps=4, observer=lambda step: resolve_identity(step, seen))
        assert [(step, identity.engine_version) for step, identity in seen] == [
            (0, f"{STARTING_HASH}@step0"),
            (1, f"{STARTING_HASH}@step1"),
            (2, f"{STARTING_HASH}@step2"),
            (3, f"{STARTING_HASH}@step3"),
            (4, f"{STARTING_HASH}@step4"),
        ]

    def test_no_score_is_ever_stamped_as_acquired(self, rig) -> None:
        """``AcquiredModelHandle``'s ``step=None`` spelling must not reach a run.

        It is a legitimate value for a handle nobody steps, and it is the wrong value for
        every score after the first update - which is why
        :func:`providers.registry.bind_acquired_model` gives ``step`` no default and this
        asserts the consequence rather than the signature.
        """
        seen: list[tuple[int, Any]] = []
        drive(rig, max_steps=2, observer=lambda step: resolve_identity(step, seen))
        assert seen
        assert not any("as-acquired" in identity.engine_version for _, identity in seen)

    def test_a_continue_names_the_resumed_step_not_zero(self, rig) -> None:
        """``start_step`` is what is loaded, so the baseline digest is the resumed step's.

        A continue from step 500 that reported ``@step0`` would make its first score
        incomparable with the curve it is continuing - the same defect as ``as-acquired``,
        one step along.
        """
        seen: list[tuple[int, Any]] = []
        drive(
            rig,
            max_steps=502,
            start_step=500,
            observer=lambda step: resolve_identity(step, seen),
            evaluate_step_zero=False,
            test_every=1,
        )
        assert [(step, identity.engine_version) for step, identity in seen] == [
            (501, f"{STARTING_HASH}@step501"),
            (502, f"{STARTING_HASH}@step502"),
        ]

    def test_the_digest_moves_before_the_step_record_is_written(self, rig) -> None:
        """The ordering inside ``take_step``, asserted where it can fail.

        ``step._apply_update`` calls ``optimizer.step()`` on **every** path, so the
        parameters have already moved by the time ``run_one`` returns - including when the
        loss is non-finite, which is returned rather than raised. A rebind placed after
        ``StepWriter.write`` would therefore leave the provider naming step ``N-1`` for a
        diverged step, attributing a number to weights that no longer exist.

        The observable version: ``writer.write`` is where the step record is produced, and
        the digest read from inside it already names the completed step.
        """
        digests: list[str] = []
        real_write = tasks.StepWriter.write

        def spy(self, **kwargs):  # type: ignore[no-untyped-def]
            handle = provider_registry.current_model_handle()
            assert handle is not None
            digests.append(handle.parameter_set_digest())
            return real_write(self, **kwargs)

        with pytest.MonkeyPatch.context() as patch:
            patch.setattr(tasks.StepWriter, "write", spy)
            drive(rig, max_steps=2)
        assert digests == [f"{STARTING_HASH}@step1", f"{STARTING_HASH}@step2"]


# --------------------------------------------------------------------------------------
# The handle does not outlive the run
# --------------------------------------------------------------------------------------


class TestTheHandleDoesNotOutliveTheRun:
    """A solo worker (ADR-006) runs one run after another in **one process**.

    So a leaked handle is not a tidiness problem: the next run's step-0 baseline, taken
    before its own bind, would be attributed to the previous run's parameters by name -
    and the previous run's digest is a perfectly well-formed value, so nothing would say
    so.
    """

    def test_it_is_unbound_after_a_completed_run(self, rig) -> None:
        drive(rig, max_steps=2)
        assert provider_registry.current_model_handle() is None

    def test_it_is_unbound_after_a_failed_run(self, rig) -> None:
        """A refusal leaves the worker alive, so the ``finally`` is the only thing that
        clears it. ``step_diverged`` is the refusal F1 raises from inside the loop."""

        def boom(step: int) -> None:
            raise Refusal("evaluation_exploded", "deliberate", {})

        with pytest.raises(Refusal):
            drive(rig, max_steps=2, observer=boom)
        assert provider_registry.current_model_handle() is None
