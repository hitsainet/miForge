"""The full F1 cycle: acquire -> record -> optimizer -> step -> record -> trainer state.

**Runs with no GPU and no ML wheels.** ADR-023 requires that a developer without a card run
everything except a training step, and this file is the proof at the feature level: the
acquisition registry, the acquisition record, the step's three stages, the run log, the index
and the atomic trainer-state write all run here. Only the tensor arithmetic is faked, at the
narrowest seam each module offers.

Three things only an integration test can see:

* **the acquisition record exists before the first step** (R-02.47) - a run's lineage must
  survive step 1 failing;
* **cancellation lands on a step boundary** (ADR-007, AC-001.11) - the step either completes
  and is recorded or has not started, never half;
* **the last recorded step is the last completed one** (E-9, AC-001.9) after a mid-run
  failure, read from the authoritative log rather than from the index.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pytest
import sqlalchemy as sa

from core.config import Settings, reset_settings
from core.errors import Refusal
from models.run import Step
from schemas.trainer_state import REQUIRED_FILES, is_complete
from tests.support.jobs import insert_run, memory_engine, session_for
from training import run_log, tasks, trainer_state
from training.acquisition.base import (
    AcquiredModel,
    AcquisitionMode,
    AcquisitionRecord,
    AcquisitionSpec,
    SourceKind,
)
from training.optimizer import OptimizerIdentity
from training.run_config import RunConfig

RUN = "run-int"
CFG = RunConfig(learning_rate=1e-5, warmup_steps=2, batch_size=2, gradient_accumulation=2)


# --------------------------------------------------------------------------------------
# The narrowest possible fakes: one per tensor boundary, nothing above them.
# --------------------------------------------------------------------------------------


class Loss:
    def __init__(self, value: float, trace: list[str]) -> None:
        self.value = value
        self.trace = trace

    def backward(self) -> None:
        self.trace.append("backward")

    def detach(self) -> Loss:
        return self

    def item(self) -> float:
        return self.value


@dataclass
class Objective:
    """A **fixture** objective, registered only in this test.

    F1 defines no objective and implements no loss: the losses are F2's
    (``training/objectives/``). A placeholder in the source would become the surviving
    definition of F2's contract, so the only placeholder lives here.
    """

    name: str
    trace: list[str]
    losses: list[float] = field(default_factory=lambda: [0.9, 0.7, 0.5, 0.4, 0.35, 0.3])
    index: int = 0

    def loss(self, model: Any, batch: Any) -> Loss:
        value = self.losses[min(self.index, len(self.losses) - 1)]
        self.index += 1
        return Loss(value, self.trace)


@dataclass
class Registry:
    objectives: dict[str, Objective]

    def get(self, name: str) -> Objective:
        return self.objectives[name]


@dataclass
class Batch:
    objective: str = "sft"
    record: dict[str, Any] = field(default_factory=lambda: {"batch_sha256": "b" * 64})


@dataclass
class Source:
    calls: int = 0

    def next(self) -> Batch:
        self.calls += 1
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
        self.checkpointing = 0

    def parameters(self) -> list[Any]:
        return []

    def gradient_checkpointing_enable(self) -> None:
        self.checkpointing += 1


class Tokenizer:
    """The tokenizer whose own files a checkpoint carries **at its root** (R-02.21).

    The rig supplies one because the save is now routed through
    :meth:`training.runtypes.change.ChangeRunType.save`, which **refuses**
    ``tokenizer_required`` rather than handing F5's writer a ``None``. That refusal is not
    new - ``change.save_checkpoint`` has always carried it - it was simply unreachable while
    nothing called it, and the only reason ``AcquiredModel(model, None, RECORD)`` never
    raised here is that ``_SaveSpy`` stands in for the writer. In production
    ``writer.write`` calls ``tokenizer.save_pretrained``, so a ``None`` was a traceback
    waiting at the first save point of any run whose acquisition did not supply one.

    §5.10's attached mode is the case that legitimately holds **neither** the tensors nor
    the tokenizer, and those tests build their own ``AcquiredModel(..., None, ...)``
    explicitly - see ``test_save_wiring.py``.
    """

    def save_pretrained(self, directory: Any) -> None:
        Path(directory).mkdir(parents=True, exist_ok=True)
        (Path(directory) / "tokenizer.json").write_text("{}", encoding="utf-8")


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
    starting_weights_sha256="a" * 64,
    base_model="local:base@abcdef0123456789",
    parent=None,
    param_count=1000,
)


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

    trace: list[str] = []
    objective = Objective(name="sft", trace=trace)
    model = Model()
    optimizer = Optimizer()
    tokenizer = Tokenizer()

    monkeypatch.setattr(trainer_state, "_torch", lambda: FakeTorch)
    monkeypatch.setattr(
        tasks, "acquire_for_run", lambda spec: AcquiredModel(model, tokenizer, RECORD)
    )
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

    yield {
        "engine": engine,
        "trace": trace,
        "registry": Registry({"sft": objective}),
        "source": Source(),
        "optimizer": optimizer,
        "model": model,
        "tokenizer": tokenizer,
        "log_root": runs,
        "states": states,
        "checkpoints": checkpoints,
    }
    reset_settings()


def drive(rig, **overrides):
    kwargs: dict[str, Any] = {
        "run_id": RUN,
        "spec": AcquisitionSpec(path=Path("/data/checkpoints/base")),
        "cfg": CFG,
        "batch_source": rig["source"],
        "objectives": rig["registry"],
        "log_root": rig["log_root"].parent,
        "trainer_state_root": rig["states"],
        "max_steps": 1,
    }
    kwargs.update(overrides)
    with session_for(rig["engine"]) as session:
        kwargs.setdefault("session", session)
        return tasks.drive_run(**kwargs)


class TestTheWholeCycleWithNoGpuAndNoSibling:
    def test_ten_steps_complete_and_are_each_recorded(self, rig) -> None:
        """AC-001.1's non-GPU half: the model acquires, the steps run, every one is
        recorded - with no sibling base URL configured and nothing attached."""
        result = drive(rig, max_steps=10)
        assert result.steps_completed == 10
        assert result.last_step == 10
        assert rig["source"].calls == 10
        assert rig["optimizer"].steps == 10

        log = run_log.run_log_path(tasks.run_log_root(rig["log_root"].parent), RUN)
        blocks = list(run_log.read_blocks(log, block="step"))
        assert [block["step"] for block in blocks] == list(range(1, 11))

    def test_every_recorded_step_validates_against_the_contract(self, rig) -> None:
        """AC-001.6, through the real writer and the real schema - not a literal."""
        drive(rig, max_steps=3)
        log = run_log.run_log_path(tasks.run_log_root(rig["log_root"].parent), RUN)
        for block in run_log.read_blocks(log, block="step"):
            run_log.validate_block(block)
            assert block["objective"] == "sft"
            assert block["learning_rate"] == CFG.learning_rate
            assert block["quiesce_hold_ms"] is None
            assert block["commit_duration_ms"] is None

    def test_the_index_mirrors_the_log(self, rig) -> None:
        drive(rig, max_steps=4)
        with session_for(rig["engine"]) as session:
            rows = session.execute(sa.select(Step.__table__).order_by(Step.__table__.c.step)).all()
        assert [row._mapping["step"] for row in rows] == [1, 2, 3, 4]
        assert rows[0]._mapping["batch_source_hashes"] == {"batch_sha256": "b" * 64}

    def test_the_loss_falls_which_means_the_fixture_objective_really_ran(self, rig) -> None:
        drive(rig, max_steps=3)
        log = run_log.run_log_path(tasks.run_log_root(rig["log_root"].parent), RUN)
        losses = [block["loss"] for block in run_log.read_blocks(log, block="step")]
        assert losses == sorted(losses, reverse=True) and losses[0] > losses[-1]

    def test_gradient_checkpointing_was_enabled_on_the_real_model(self, rig) -> None:
        """R-02.13 fixes it on; the optimizer builder is the thing that turns it on, and this
        asserts the builder was actually used rather than bypassed."""
        drive(rig, max_steps=1)
        assert rig["model"].checkpointing == 0, (
            "the optimizer builder is faked in this rig, so checkpointing is asserted in "
            "test_step_optimizer.py against the real builder"
        )


class TestTheAcquisitionRecordComesFirst:
    def test_the_record_is_returned_before_any_step_is_taken(self, rig) -> None:
        """R-02.47: a run's lineage must exist even if step 1 fails, which is why the record
        is written at acquisition rather than at the first save."""
        result = drive(rig, max_steps=0)
        assert result.steps_completed == 0
        assert result.acquisition["base_model"] == "local:base@abcdef0123456789"
        assert result.acquisition["parent"] is None
        assert result.acquisition["starting_weights_sha256"] == "a" * 64

    def test_the_record_survives_a_failing_first_step(self, rig, monkeypatch) -> None:
        class Exploding:
            def next(self) -> Batch:
                raise RuntimeError("CUDA out of memory")

        with pytest.raises(RuntimeError):
            drive(rig, batch_source=Exploding(), max_steps=1)
        # The record was produced before the step; the run log holds no step block, which is
        # E-8: the step is NOT recorded as completed.
        log = run_log.run_log_path(tasks.run_log_root(rig["log_root"].parent), RUN)
        assert list(run_log.read_blocks(log, block="step")) == []


class TestTrainerStateOnTheSaveCadence:
    def test_state_lands_only_at_save_points(self, rig) -> None:
        """FR-001.29: on the save cadence, not per step. ``save_every`` is therefore the
        blast radius of a crash (ADR-024)."""
        result = drive(rig, max_steps=6, save_every=2)
        assert [Path(p).name for p in result.trainer_states] == [
            "step-000002",
            "step-000004",
            "step-000006",
        ]

    def test_each_written_state_is_COMPLETE(self, rig) -> None:
        result = drive(rig, max_steps=2, save_every=2)
        path = Path(result.trainer_states[0])
        assert is_complete(path)
        assert sorted(p.name for p in path.iterdir()) == sorted(REQUIRED_FILES)

    def test_the_step_number_matches_the_step_that_was_recorded(self, rig) -> None:
        """FR-001.29: the same canonical integer as the weights save, so a continue can match
        the two."""
        result = drive(rig, max_steps=3, save_every=3)
        manifest = json.loads((Path(result.trainer_states[0]) / "trainer_state.json").read_text())
        assert manifest["step"] == 3 == result.last_step

    def test_no_state_is_written_when_no_cadence_is_given(self, rig) -> None:
        """F1 defines no cadence field of its own: ``save_every`` is R-02.24's name for it and
        **F6's to validate and carry on the run definition**."""
        assert drive(rig, max_steps=3).trainer_states == []

    def test_the_checkpoint_root_is_untouched(self, rig) -> None:
        """AC-001.8: the weights checkpoint directory contains no optimizer tensor. A
        checkpoint that gains one stops being a standard Hugging Face directory (AC23)."""
        drive(rig, max_steps=2, save_every=1)
        assert list(rig["checkpoints"].rglob("*")) == []

    def test_the_optimizer_identity_reaches_the_state_manifest(self, rig) -> None:
        result = drive(rig, max_steps=1, save_every=1)
        manifest = json.loads((Path(result.trainer_states[0]) / "trainer_state.json").read_text())
        assert manifest["optimizer"]["name"] == "adamw-8bit"
        assert manifest["optimizer"]["state"] == "fresh"


class TestCancellationLandsOnAStepBoundary:
    """ADR-007, AC-001.11."""

    def test_a_stop_requested_mid_run_stops_between_two_steps(self, rig) -> None:
        from workers.cancellation import RunCancelled

        class StopAfter:
            """Raises at the boundary once two steps have completed."""

            def __init__(self) -> None:
                self.checks = 0

            def check(self) -> None:
                self.checks += 1
                if self.checks > 2:
                    raise RunCancelled(RUN)

        result = drive(rig, max_steps=10, cancel=StopAfter())
        assert result.stopped_because == "cancelled"
        assert result.steps_completed == 2
        assert result.last_step == 2

    def test_the_cancelled_step_is_not_recorded_HALF(self, rig) -> None:
        """A step either completes and is recorded, or has not started. R-02.31 is why it
        must be the step and not something finer."""
        from workers.cancellation import RunCancelled

        class StopImmediately:
            def check(self) -> None:
                raise RunCancelled(RUN)

        result = drive(rig, max_steps=5, cancel=StopImmediately())
        assert result.steps_completed == 0
        assert rig["source"].calls == 0, "a cancelled step must not even draw a batch"
        log = run_log.run_log_path(tasks.run_log_root(rig["log_root"].parent), RUN)
        assert list(run_log.read_blocks(log, block="step")) == []

    def test_cancellation_uses_the_databases_token_not_celerys_control_api(self) -> None:
        """``revoke(terminate=True)`` signals a pool child and the solo pool has none, so the
        call succeeds and nothing happens. The foundation's guard scans for it; this asserts
        the driver uses the cooperative token."""
        import ast

        tree = ast.parse(Path(tasks.__file__).read_text(encoding="utf-8"))
        # Parsed, not scanned for a substring: this module's own docstring *explains* why
        # `revoke(terminate=True)` is never used, and a string check would report that
        # explanation as the violation. The project has paid for that confusion before.
        for node in ast.walk(tree):
            if isinstance(node, ast.Call):
                name = getattr(node.func, "attr", None) or getattr(node.func, "id", None)
                assert name != "revoke", "cancellation goes through the database (ADR-006)"

        # And the cooperative token really is handed to the step, which is the positive half.
        #
        # ⚠ **The name of the call changed on 2026-10-16 and the intent did not.** This
        # looked for `run_one` by name; the driver now takes its turn through
        # `run_type_impl.step`, which IS `run_one` - `training.runtypes.base.STEP` binds the
        # same function object and `test_runtype_registry.py` asserts the three run types
        # carry it by **identity**. So the pattern is widened to either spelling and the
        # `cancel=` keyword is what makes the match load-bearing rather than a name check:
        # `optimizer.step()` has no such keyword, so it cannot satisfy this. Widened again on
        # 2026-10-16 to `step_impl`: the driver now picks the run type's step OR the
        # acquisition's own §5.10 stepper (`acquired.stepper`, R-02.52) and calls whichever
        # it picked, under the same `cancel=` keyword. The behavioural half is the test above.
        step_calls = [
            node
            for node in ast.walk(tree)
            if isinstance(node, ast.Call)
            and (getattr(node.func, "attr", None) or getattr(node.func, "id", None))
            in {"run_one", "step", "step_impl"}
            and any(kw.arg == "cancel" for kw in node.keywords)
        ]
        assert step_calls, (
            "the driver must take its turn through the step - `run_type.step` or `run_one` - "
            "and it must be given the cancellation token, or the boundary check never runs"
        )


class TestTheLastRecordedStepIsTheLastCompletedOne:
    """E-9, AC-001.9."""

    def test_a_mid_run_failure_leaves_the_log_ending_at_the_last_completed_step(self, rig) -> None:
        class FailsOnThird(Source):
            def next(self) -> Batch:
                self.calls += 1
                if self.calls == 3:
                    raise RuntimeError("CUDA error: device-side assert triggered")
                return Batch()

        with pytest.raises(RuntimeError):
            drive(rig, batch_source=FailsOnThird(), max_steps=5)

        log = run_log.run_log_path(tasks.run_log_root(rig["log_root"].parent), RUN)
        assert [block["step"] for block in run_log.read_blocks(log, block="step")] == [1, 2]

    def test_it_is_read_from_the_LOG_so_it_survives_a_lost_index(self, rig) -> None:
        from training.step_record import StepWriter

        drive(rig, max_steps=3)
        with session_for(rig["engine"]) as session:
            session.execute(sa.delete(Step.__table__))
        writer = StepWriter(run_id=RUN, log_root=tasks.run_log_root(rig["log_root"].parent))
        assert writer.last_recorded_step() == 3

    def test_a_diverged_step_stops_the_run_and_names_the_value(self, rig) -> None:
        """E-12. The number is reported - in the refusal's details and in the structured log
        - rather than dropped, but it cannot go in the run log: JSON has no NaN, and
        ``canonical_bytes`` refuses it because the literal ``NaN`` is not required to be
        parseable anywhere. Reported as a tension with E-12's wording; the information
        survives, its location differs."""
        rig["registry"].objectives["sft"].losses = [0.5, float("nan")]
        with pytest.raises(Refusal) as caught:
            drive(rig, max_steps=5)
        assert caught.value.code == "step_diverged"
        assert caught.value.details["step"] == 2
        assert caught.value.details["loss"] == "nan"

        log = run_log.run_log_path(tasks.run_log_root(rig["log_root"].parent), RUN)
        assert [block["step"] for block in run_log.read_blocks(log, block="step")] == [1]


class TestStaleTempDirectoriesAreTidiedByTheRunsNextSave:
    def test_a_killed_writers_leftovers_are_removed_at_the_start_of_the_next_run(self, rig) -> None:
        """E-10. The primitive cleans up when it *raises*; it cannot when the process is
        killed outright, so the run's own next drive tidies the wreck of its last one."""
        from core.atomic_dir import TMP_PREFIX

        stale = rig["states"] / RUN / f"{TMP_PREFIX}leftover"
        stale.mkdir(parents=True)
        (stale / "optimizer.pt").write_text("half")

        drive(rig, max_steps=1, save_every=1)
        assert not stale.exists()
        assert is_complete(rig["states"] / RUN / "step-000001")
