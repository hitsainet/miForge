"""The trainer-state writer  (R-02.15; FR-001.28 - FR-001.31, AC-001.8, AC-001.10, E-10).

F1 is the **single writer** of foundation task 3.7's layout; F5 reads it to continue a run
(R-02.22) and F11 reads it to recover after an interrupted commit (R-02.30(b)). This file
pins the four properties those two readers depend on:

* the path is ``trainer-state/<run_id>/<NNNNNN>/``, rendered from the **canonical integer**
  through ``schemas/trainer_state.py`` - one implementation of the padding rule;
* the write goes through **``core/atomic_dir.py``**, the backend's only directory
  rename-into-place, with ``latest_link=False`` because this layout has no ``latest``;
* the guarantee is **absent or complete**, and a reader checks completeness anyway rather
  than trusting that the writer was atomic;
* it is **never under the weights checkpoint root** - a checkpoint that gains optimizer
  tensors stops being a standard Hugging Face directory, which breaks R-02.51 and acceptance
  criterion 23.

``torch`` is not installed on this machine (ADR-023), so the saver is faked at the one seam
that needs it. **Everything the assertions are about runs through the real code**: the path
renderer, the primitive, the manifest writer and the completeness check.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from core.atomic_dir import TMP_PREFIX
from core.config import Settings, reset_settings
from core.errors import Refusal
from schemas.trainer_state import (
    OPTIMIZER_STATE,
    REQUIRED_FILES,
    RNG_STATE,
    SCHEDULER_STATE,
    STATE_MANIFEST,
    is_complete,
    latest_complete,
    require_complete,
)
from training import trainer_state
from training.optimizer import OptimizerIdentity

RUN = "run-7"


class FakeOptimizer:
    def state_dict(self) -> dict[str, Any]:
        return {"state": {"exp_avg": [0.1, 0.2]}, "param_groups": [{"lr": 1e-5}]}


class FakeScheduler:
    def state_dict(self) -> dict[str, Any]:
        return {"last_epoch": 3}


class FakeTorch:
    """The two functions this module uses, and nothing else.

    ``save`` writes real bytes - a fake that wrote nothing would make every file-existence
    assertion below pass for the wrong reason.
    """

    class cuda:
        @staticmethod
        def is_available() -> bool:
            return False

    @staticmethod
    def get_rng_state() -> str:
        return "torch-rng"

    @staticmethod
    def save(obj: Any, path: Path) -> None:
        Path(path).write_text(json.dumps(obj, default=str))

    @staticmethod
    def load(path: Path, **kwargs: Any) -> Any:
        return json.loads(Path(path).read_text())


@pytest.fixture
def roots(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """Separate trainer-state and checkpoint roots, as the deployment provisions them."""
    states = tmp_path / "trainer-state"
    checkpoints = tmp_path / "checkpoints"
    states.mkdir()
    checkpoints.mkdir()
    monkeypatch.setattr(
        "core.config._settings",
        Settings(  # type: ignore[call-arg]
            _env_file=None, trainer_state_root=states, checkpoint_root=checkpoints
        ),
    )
    monkeypatch.setattr(trainer_state, "_torch", lambda: FakeTorch)
    yield states, checkpoints
    reset_settings()


IDENTITY = OptimizerIdentity(name="adamw-8bit", state="fresh", config_hash="d" * 64)


class TestThePathRendering:
    def test_the_step_is_zero_padded_to_six_digits(self, roots) -> None:
        """BRD-01 R-01.24: one canonical integer, paths pad it to six digits. **Dropping the
        padding breaks the lexicographic ordering every reader relies on**, and is a mutation
        target."""
        states, _ = roots
        assert trainer_state.step_path(RUN, 42) == states / RUN / "step-000042"
        assert trainer_state.step_path(RUN, 0) == states / RUN / "step-000000"
        # Beyond six digits the integer simply renders longer - the padding is a minimum,
        # not a truncation, so a step past 999999 is not silently mangled.
        assert trainer_state.step_path(RUN, 1_234_567) == states / RUN / "step-1234567"

    def test_the_path_is_rendered_through_the_shared_layout_module(self, roots) -> None:
        """One implementation of the rule. ``step-<N>`` appears in no contract and must not
        be reintroduced - the foundation's layout module says so in terms."""
        from schemas.trainer_state import TrainerStateDir

        states, _ = roots
        assert trainer_state.step_path(RUN, 5) == TrainerStateDir(root=states / RUN, step=5).path
        # BRD-01 R-01.24: the path form IS `step-` plus six digits, for BOTH the weights
        # checkpoint and the trainer state, and R-02.15 requires them to agree. This
        # line asserted the opposite because the gate contract I built against said
        # `step-<N>` appeared in no contract - a misreading of R-01.24, which removed
        # only the UNPADDED form and only from paths. Corrected 2026-10-14.
        assert str(trainer_state.step_path(RUN, 5)).endswith("/step-000005")

    def test_a_negative_step_refuses(self, roots) -> None:
        with pytest.raises(Refusal) as caught:
            trainer_state.step_path(RUN, -1)
        assert caught.value.code == "trainer_state_step_negative"

    def test_an_unnamed_run_refuses(self, roots) -> None:
        with pytest.raises(Refusal) as caught:
            trainer_state.run_root("")
        assert caught.value.code == "trainer_state_run_id_required"
        assert "matched to the checkpoint" in caught.value.message


class TestNeverUnderTheWeightsCheckpointRoot:
    """FR-001.30 - asserted, not requested by a comment."""

    def test_a_path_inside_the_checkpoint_root_is_refused(self, roots) -> None:
        _, checkpoints = roots
        with pytest.raises(Refusal) as caught:
            trainer_state.assert_outside_checkpoint_root(checkpoints / RUN / "step-000001")
        assert caught.value.code == "trainer_state_inside_checkpoint_root"
        assert "standard Hugging Face directory" in caught.value.message

    def test_the_normal_path_is_accepted(self, roots) -> None:
        trainer_state.assert_outside_checkpoint_root(trainer_state.step_path(RUN, 1))

    def test_a_misconfigured_root_is_caught_at_the_WRITE(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The check is on the write path rather than only at startup, so a root changed
        underneath a running process cannot slip a tensor into a checkpoint."""
        shared = tmp_path / "shared"
        shared.mkdir()
        monkeypatch.setattr(
            "core.config._settings",
            Settings(_env_file=None, trainer_state_root=shared, checkpoint_root=shared),  # type: ignore[call-arg]
        )
        monkeypatch.setattr(trainer_state, "_torch", lambda: FakeTorch)
        with pytest.raises(Refusal) as caught:
            trainer_state.write(run_id=RUN, step=1, optimizer=FakeOptimizer())
        assert caught.value.code == "trainer_state_inside_checkpoint_root"
        reset_settings()

    def test_the_weights_checkpoint_directory_gains_no_optimizer_tensor(self, roots) -> None:
        """AC-001.8's second half, stated as the observable fact: after a trainer-state
        write, the checkpoint root is untouched."""
        states, checkpoints = roots
        trainer_state.write(run_id=RUN, step=1, optimizer=FakeOptimizer())
        assert list(checkpoints.rglob("*")) == []
        assert (states / RUN / "step-000001" / OPTIMIZER_STATE).is_file()


class TestTheLayoutIsTheFoundationS:
    def test_all_four_required_files_are_written(self, roots) -> None:
        """``is_complete`` requires all four, so an omitted scheduler file would make every
        state this writer produces read as incomplete - and the reader could not tell
        "no scheduler" from "the write was cut short", which is the whole point of the check.
        """
        path = trainer_state.write(run_id=RUN, step=3, optimizer=FakeOptimizer())
        assert sorted(p.name for p in path.iterdir()) == sorted(REQUIRED_FILES)
        assert is_complete(path)

    def test_a_run_with_no_scheduler_still_writes_the_scheduler_file(self, roots) -> None:
        path = trainer_state.write(run_id=RUN, step=1, optimizer=FakeOptimizer())
        assert (path / SCHEDULER_STATE).is_file()
        assert json.loads((path / STATE_MANIFEST).read_text())["has_scheduler"] is False

    def test_a_scheduler_is_saved_SEPARATELY_from_the_optimizer(self, roots) -> None:
        """A continue may legally resume the optimizer while rebuilding the schedule - a
        curriculum stage boundary does exactly that, and one blob would make it impossible."""
        path = trainer_state.write(
            run_id=RUN, step=1, optimizer=FakeOptimizer(), scheduler=FakeScheduler()
        )
        assert json.loads((path / SCHEDULER_STATE).read_text()) == {"last_epoch": 3}
        assert "last_epoch" not in (path / OPTIMIZER_STATE).read_text()

    def test_the_rng_state_is_saved_because_otherwise_a_continue_is_not_one(self, roots) -> None:
        """A resumed run that reseeds draws a different sample order from the same data and
        produces a loss discontinuity indistinguishable from a real regression."""
        path = trainer_state.write(run_id=RUN, step=1, optimizer=FakeOptimizer())
        rng = json.loads((path / RNG_STATE).read_text())
        assert rng["torch"] == "torch-rng"
        # CUDA absence is recorded as a fact rather than left implicit, so a state captured
        # on the GPU node and restored on a laptop says what it is missing.
        assert "cuda" in rng and rng["cuda"] is None

    def test_the_manifest_is_written_LAST_which_is_what_makes_is_complete_meaningful(
        self, roots
    ) -> None:
        """A directory that lost power mid-write must read as incomplete rather than as a
        valid state missing its RNG."""
        import ast

        assert (
            REQUIRED_FILES[-1] == STATE_MANIFEST
        ), "the shared layout lists the manifest last; this writer's order follows it"
        tree = ast.parse(Path(trainer_state.__file__).read_text(encoding="utf-8"))
        write_def = next(
            node
            for node in ast.walk(tree)
            if isinstance(node, ast.FunctionDef) and node.name == "write"
        )
        # Where each layout constant is *referenced* inside `write`, by line. The manifest's
        # reference must come after all three state files', because that ordering is the only
        # thing that makes `is_complete` able to tell a finished directory from a torn one.
        lines = {
            node.id: node.lineno
            for node in ast.walk(write_def)
            if isinstance(node, ast.Name)
            and node.id in {"OPTIMIZER_STATE", "SCHEDULER_STATE", "RNG_STATE", "STATE_MANIFEST"}
        }
        assert set(lines) == {
            "OPTIMIZER_STATE",
            "SCHEDULER_STATE",
            "RNG_STATE",
            "STATE_MANIFEST",
        }, f"write() does not reference every required file: {sorted(lines)}"
        assert lines["STATE_MANIFEST"] == max(
            lines.values()
        ), f"the manifest must be written last; references are at {lines}"

    def test_the_manifest_records_the_optimizer_identity(self, roots) -> None:
        """So a reader can tell which optimizer wrote these states *before* loading them."""
        path = trainer_state.write(run_id=RUN, step=1, optimizer=FakeOptimizer(), identity=IDENTITY)
        manifest = json.loads((path / STATE_MANIFEST).read_text())
        assert manifest["optimizer"] == {
            "name": "adamw-8bit",
            "state": "fresh",
            "config_hash": "d" * 64,
        }
        assert manifest["step"] == 1 and manifest["run_id"] == RUN


class TestAtomicity:
    """R-02.15 as amended 2026-10-13, ADR-015. Guarantee: **absent or complete**."""

    def test_the_write_goes_through_the_shared_primitive(self, roots) -> None:
        """**One definition of this guarantee.** Two implementations are two answers
        (ADR-005), and the primitive is the backend's only directory rename-into-place -
        F5's checkpoint writer and F8's adapter writer call the same one."""
        import ast

        tree = ast.parse(Path(trainer_state.__file__).read_text(encoding="utf-8"))
        calls = [
            node
            for node in ast.walk(tree)
            if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id == "atomic_dir"
        ]
        assert len(calls) == 1, "exactly one call to the primitive, and no second writer"

    def test_latest_link_is_passed_as_FALSE_explicitly(self, roots) -> None:
        """The trainer-state layout carries **no ``latest`` pointer**; checkpoints do. A
        "behaviour-preserving" refactor that left the default on would publish a symlink F1
        never wrote - which is why the parameter is explicit at the call site."""
        import ast

        tree = ast.parse(Path(trainer_state.__file__).read_text(encoding="utf-8"))
        call = next(
            node
            for node in ast.walk(tree)
            if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id == "atomic_dir"
        )
        keywords = {kw.arg: kw.value for kw in call.keywords}
        assert "latest_link" in keywords, "latest_link must be passed, not inherited"
        assert isinstance(keywords["latest_link"], ast.Constant)
        assert keywords["latest_link"].value is False

    def test_no_latest_symlink_is_published(self, roots) -> None:
        """The observable half of the assertion above."""
        states, _ = roots
        trainer_state.write(run_id=RUN, step=1, optimizer=FakeOptimizer())
        assert not (states / RUN / "latest").exists()

    def test_a_failed_write_leaves_the_directory_ABSENT(self, roots) -> None:
        """AC-001.10: absent or complete, never in between. **The failure this prevents is
        the expensive kind** - a half-written directory restores as a *valid* optimizer state
        and produces exactly the unexplained loss discontinuity R-02.22 exists to prevent,
        failing acceptance criterion 4. It does not crash."""
        states, _ = roots

        class Exploding:
            def state_dict(self) -> dict[str, Any]:
                raise OSError("disk full")

        with pytest.raises(OSError):
            trainer_state.write(run_id=RUN, step=4, optimizer=Exploding())

        assert not (states / RUN / "step-000004").exists()
        assert [
            p.name for p in (states / RUN).iterdir()
        ] == [], "no temp directory may be left behind by a write that raised"

    def test_overwriting_a_published_step_is_refused(self, roots) -> None:
        """The step is part of the identity, so an overwrite means two different optimizer
        states claiming one name. The primitive refuses it."""
        trainer_state.write(run_id=RUN, step=1, optimizer=FakeOptimizer())
        with pytest.raises(Refusal) as caught:
            trainer_state.write(run_id=RUN, step=1, optimizer=FakeOptimizer())
        assert caught.value.code == "artifact_already_exists"


class TestTheReadSide:
    def test_a_complete_state_reads_back(self, roots) -> None:
        trainer_state.write(
            run_id=RUN, step=2, optimizer=FakeOptimizer(), scheduler=FakeScheduler()
        )
        loaded = trainer_state.read(RUN, 2)
        assert loaded["optimizer"]["param_groups"] == [{"lr": 1e-5}]
        assert loaded["scheduler"] == {"last_epoch": 3}
        assert loaded["rng"]["torch"] == "torch-rng"

    def test_an_absent_state_refuses_by_name(self, roots) -> None:
        with pytest.raises(Refusal) as caught:
            trainer_state.read(RUN, 9)
        assert caught.value.code == "trainer_state_absent"
        assert "unexplained loss discontinuity" in caught.value.message

    def test_an_INCOMPLETE_state_refuses_rather_than_restoring_half_an_optimizer(
        self, roots
    ) -> None:
        """**The reader does not trust that the writer was atomic**, because the writer may
        have been killed before its own cleanup ran - which is precisely when a reader runs.
        """
        path = trainer_state.write(run_id=RUN, step=2, optimizer=FakeOptimizer())
        (path / RNG_STATE).unlink()
        with pytest.raises(Refusal) as caught:
            trainer_state.read(RUN, 2)
        assert caught.value.code == "trainer_state_incomplete"
        assert RNG_STATE in caught.value.details["missing"]

    def test_require_complete_is_the_foundations_not_a_second_copy(self, roots) -> None:
        path = trainer_state.write(run_id=RUN, step=1, optimizer=FakeOptimizer())
        assert require_complete(path) == path
        assert trainer_state.require_complete is require_complete

    def test_latest_complete_skips_a_torn_newest_directory(self, roots) -> None:
        """A crash leaves the newest directory torn, and the useful answer is the newest
        *usable* one."""
        trainer_state.write(run_id=RUN, step=1, optimizer=FakeOptimizer())
        torn = trainer_state.write(run_id=RUN, step=2, optimizer=FakeOptimizer())
        (torn / STATE_MANIFEST).unlink()
        found = latest_complete(trainer_state.run_root(RUN))
        assert found is not None and found.step == 1


class TestStaleTempDirectories:
    """E-10: the temp directory is cleaned on the next write for that run."""

    def test_a_killed_writers_temp_directory_is_removed_by_the_next_write(self, roots) -> None:
        """The primitive cleans up when it *raises*; it cannot when the process is killed
        outright. So the run's own next save tidies the wreck of its last one, and nothing
        sweeps a directory another process is currently writing into."""
        states, _ = roots
        stale = states / RUN / f"{TMP_PREFIX}deadbeef"
        stale.mkdir(parents=True)
        (stale / OPTIMIZER_STATE).write_text("half")

        removed = trainer_state.clean_stale_temp_dirs(RUN)
        assert removed == [stale]
        assert not stale.exists()

    def test_a_published_directory_is_never_swept(self, roots) -> None:
        path = trainer_state.write(run_id=RUN, step=1, optimizer=FakeOptimizer())
        assert trainer_state.clean_stale_temp_dirs(RUN) == []
        assert is_complete(path)

    def test_sweeping_a_run_with_no_directory_is_a_no_op(self, roots) -> None:
        assert trainer_state.clean_stale_temp_dirs("never-ran") == []
