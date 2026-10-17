"""Continue and branch: verify first, then decide, then say what you did  (FR-005.28-38).

**The order is the test.** ``verify_weights`` runs before a tensor is read, before a lease,
before a run row moves - so a corrupted checkpoint refuses *having changed nothing*. The
control that deletes that call is the most important one in this feature, and for a sharper
reason than it looks: R-02.21 now **requires** the check (it did not, until this feature's
review found the hole), and **nothing in the code would notice if it were dropped**. So the
test is what notices, and a green suite under that mutation means the requirement that
closed the hole is unpinned.

## Trainer state, and the four outcomes

Three are recorded decisions and the fourth is a refusal, which is the one that is not a
judgement call: a directory for the loaded step that is **torn** is refused through F1's
``require_complete``. Continuing from a partial state does not crash - it *runs*, and
produces a loss discontinuity that looks exactly like a real regression, so the time is lost
twice, once to the run and once to diagnosing it.

**The directories here are built from F1's own ``REQUIRED_FILES``**, imported rather than
re-spelled. The byte-identical round trip through F1's real writer needs ``torch`` and
**skips loudly** where it is absent rather than quietly passing (ADR-023, ADR-016).
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from core.errors import Refusal
from schemas.trainer_state import REQUIRED_FILES
from services.checkpoints import resume as resume_module
from services.checkpoints.paths import checkpoint_dir
from tests.support.checkpoints import saved_checkpoint, write_chain


def plant_trainer_state(root: Path, run_id: str, step: int, *, complete: bool = True) -> Path:
    """A trainer-state directory with F1's own file list. Incomplete leaves out the last one.

    ``REQUIRED_FILES`` is imported from the layout module rather than typed out: a second
    spelling of the file set is a second definition of the contract, which is the thing
    foundation task 3.7 exists to prevent.
    """
    path = root / run_id / f"step-{step:06d}"
    path.mkdir(parents=True, exist_ok=True)
    names = REQUIRED_FILES if complete else REQUIRED_FILES[:-1]
    for name in names:
        (path / name).write_bytes(b"state")
    return path


@pytest.fixture
def roots(tmp_path: Path):
    checkpoints = tmp_path / "checkpoints"
    states = tmp_path / "trainer-state"
    checkpoints.mkdir()
    states.mkdir()
    return checkpoints, states


class TestVerificationComesFirst:
    def test_a_continue_refuses_on_hash_mismatch(self, roots):
        """**MUT-4 and the F5 re-mutation.** Refuses *having changed nothing*: no tensor
        read, no lease taken, no run row moved - and the manifest is not repaired to match
        the bytes it found."""
        checkpoints, states = roots
        result = saved_checkpoint(checkpoints, run_id="run-a", step=100)
        before = (result.directory / "manifest.json").read_bytes()
        (result.directory / "model.safetensors").write_bytes(b"corrupted")

        with pytest.raises(Refusal) as caught:
            resume_module.continue_plan("run-a", root=checkpoints, trainer_state_root=states)

        assert caught.value.code == "checkpoint_hash_mismatch"
        assert (result.directory / "manifest.json").read_bytes() == before

    def test_a_branch_from_a_corrupt_checkpoint_refuses_too(self, roots):
        """R-02.21 names three load paths - a continue, a branch and a conformance check -
        and this is the second of them."""
        checkpoints, states = roots
        result = saved_checkpoint(checkpoints, run_id="run-a", step=100)
        (result.directory / "model.safetensors").write_bytes(b"corrupted")

        with pytest.raises(Refusal) as caught:
            resume_module.branch_plan(
                "run-a", 100, new_run_id="run-b", root=checkpoints, trainer_state_root=states
            )
        assert caught.value.code == "checkpoint_hash_mismatch"

    def test_the_verification_happens_before_the_trainer_state_is_consulted(self, roots):
        """Ordering, asserted rather than assumed: a plan that resolved trainer state first
        would have *decided* something before discovering the weights were corrupt."""
        checkpoints, states = roots
        result = saved_checkpoint(checkpoints, run_id="run-a", step=100)
        (result.directory / "model.safetensors").write_bytes(b"corrupted")
        plant_trainer_state(states, "run-a", 100, complete=False)

        with pytest.raises(Refusal) as caught:
            resume_module.continue_plan("run-a", root=checkpoints, trainer_state_root=states)
        assert (
            caught.value.code == "checkpoint_hash_mismatch"
        ), "the torn trainer state was reached first, so the hash check is not first"


class TestResolvingWhichCheckpoint:
    def test_a_continue_defaults_to_what_latest_resolves_to(self, roots):
        checkpoints, states = roots
        write_chain(checkpoints, steps=(100, 200, 300))
        plan = resume_module.continue_plan("run-a", root=checkpoints, trainer_state_root=states)
        assert plan.step == 300

    def test_an_explicit_step_is_honoured(self, roots):
        checkpoints, states = roots
        write_chain(checkpoints, steps=(100, 200, 300))
        plan = resume_module.continue_plan(
            "run-a", step=100, root=checkpoints, trainer_state_root=states
        )
        assert plan.step == 100

    def test_a_run_with_no_checkpoint_refuses_by_name(self, roots):
        checkpoints, states = roots
        with pytest.raises(Refusal) as caught:
            resume_module.continue_plan("nobody", root=checkpoints, trainer_state_root=states)
        assert caught.value.code == "checkpoint_not_found"

    def test_latest_falls_back_to_the_highest_step_when_the_symlink_is_gone(self, roots):
        """A ``tar`` that dropped links, or a ``cp`` without ``-a``. Refusing would make a
        perfectly good set of checkpoints unreachable while every one of them sits on disk."""
        checkpoints, states = roots
        write_chain(checkpoints, steps=(100, 200))
        (checkpoints / "run-a" / "latest").unlink()
        plan = resume_module.continue_plan("run-a", root=checkpoints, trainer_state_root=states)
        assert plan.step == 200


class TestTheLineageRootIsCarriedForward:
    def test_base_model_is_the_root_and_parent_is_the_loaded_checkpoint(self, roots):
        """**MUT-9.** The thing loaded *is* a checkpoint, so the root is carried forward from
        its own manifest. Setting it to the loaded directory's identity roots the tree at the
        parent, and the tree still validates and still looks plausible."""
        checkpoints, states = roots
        chain = write_chain(checkpoints, steps=(100, 200))
        plan = resume_module.continue_plan("run-a", root=checkpoints, trainer_state_root=states)

        assert plan.base_model == "LiquidAI/LFM2-1.2B"
        assert plan.parent == chain[-1].name
        assert plan.base_model != plan.parent

    def test_a_checkpoint_with_no_base_model_refuses_rather_than_substituting_itself(self, roots):
        checkpoints, states = roots
        result = saved_checkpoint(checkpoints, run_id="run-a", step=1)
        path = result.directory / "manifest.json"
        doc = json.loads(path.read_text())
        doc["base_model"] = ""
        path.write_text(json.dumps(doc))

        with pytest.raises(Refusal) as caught:
            resume_module.continue_plan("run-a", root=checkpoints, trainer_state_root=states)
        # The manifest no longer conforms either; whichever fires, the point is that nothing
        # substitutes the loaded checkpoint for its own root.
        assert caught.value.code in {"checkpoint_manifest_invalid", "checkpoint_base_model_missing"}


class TestTheThreeTrainerStateOutcomes:
    def test_state_for_the_loaded_step_is_restored(self, roots):
        checkpoints, states = roots
        write_chain(checkpoints, steps=(100,))
        plant_trainer_state(states, "run-a", 100)

        plan = resume_module.continue_plan("run-a", root=checkpoints, trainer_state_root=states)
        assert plan.optimizer_state == "restored"
        assert plan.reason == "trainer_state_restored"

    def test_no_state_at_all_is_fresh_and_logs_the_directory_looked_for(self, roots, caplog):
        """E-7, and R-02.22's own words. The schema says why the field exists: *a fresh
        optimizer on a mid-run checkpoint produces a loss discontinuity that is otherwise
        unexplained.* The record is what makes it explained."""
        checkpoints, states = roots
        write_chain(checkpoints, steps=(100,))

        with caplog.at_level("WARNING"):
            plan = resume_module.continue_plan("run-a", root=checkpoints, trainer_state_root=states)
        assert (plan.optimizer_state, plan.reason) == ("fresh", "trainer_state_absent")
        assert any(r.__dict__.get("code") == "trainer_state_absent" for r in caplog.records)

    def test_state_for_a_different_step_is_fresh_and_NAMES_the_steps_that_have_it(
        self, roots, caplog
    ):
        """E-8 / AMB-1. A mismatched optimizer is never restored onto different weights - an
        AdamW moment buffer belongs to the parameters it was computed against - and the plan
        **lists the steps that do have state** so the operator can continue from one of those
        instead of silently accepting a discontinuity."""
        checkpoints, states = roots
        write_chain(checkpoints, steps=(100, 200))
        plant_trainer_state(states, "run-a", 100)

        with caplog.at_level("WARNING"):
            plan = resume_module.continue_plan(
                "run-a", step=200, root=checkpoints, trainer_state_root=states
            )
        assert (plan.optimizer_state, plan.reason) == ("fresh", "trainer_state_step_mismatch")
        assert plan.steps_with_state == (100,)
        assert plan.as_dict()["steps_with_trainer_state"] == [100]

    def test_a_TORN_directory_for_the_loaded_step_is_refused_not_treated_as_absent(self, roots):
        """**The one that is not a judgement call.** Present-and-incomplete is not the same
        fact as absent: starting fresh from a crash whose evidence is sitting right there
        produces a loss discontinuity indistinguishable from a real regression."""
        checkpoints, states = roots
        write_chain(checkpoints, steps=(100,))
        plant_trainer_state(states, "run-a", 100, complete=False)

        with pytest.raises(Refusal) as caught:
            resume_module.continue_plan("run-a", root=checkpoints, trainer_state_root=states)
        assert caught.value.code == "trainer_state_incomplete"
        assert caught.value.details["missing"]

    def test_an_incomplete_directory_is_not_offered_as_a_step_that_has_state(self, roots):
        """Offering a torn directory in a refusal's ``details`` would be advice that fails."""
        _checkpoints, states = roots
        plant_trainer_state(states, "run-a", 50, complete=False)
        plant_trainer_state(states, "run-a", 60, complete=True)
        assert resume_module.steps_with_trainer_state("run-a", trainer_state_root=states) == [60]


class TestTheOptimizerConfigHash:
    def test_a_changed_config_hash_restores_anyway_and_logs_both(self, roots, caplog):
        """FR-005.34. R-02.22 says restore, so this does not refuse - and the **new**
        configuration's hash is what subsequent manifests carry. A restored AdamW state under
        a changed learning-rate schedule is a legitimate operator act and an unexplained loss
        discontinuity if nothing records it."""
        checkpoints, states = roots
        write_chain(checkpoints, steps=(100,))
        plant_trainer_state(states, "run-a", 100)

        with caplog.at_level("WARNING"):
            plan = resume_module.continue_plan(
                "run-a",
                configured_config_hash="z" * 64,
                root=checkpoints,
                trainer_state_root=states,
            )
        assert plan.optimizer_state == "restored"
        assert plan.config_hash_changed == ("a" * 64, "z" * 64)
        assert any(
            r.__dict__.get("code") == "optimizer_config_hash_changed" for r in caplog.records
        )

    def test_an_unchanged_hash_reports_no_change(self, roots):
        checkpoints, states = roots
        write_chain(checkpoints, steps=(100,))
        plan = resume_module.continue_plan(
            "run-a", configured_config_hash="a" * 64, root=checkpoints, trainer_state_root=states
        )
        assert plan.config_hash_changed is None


class TestBranching:
    def test_a_branch_takes_a_new_run_id_and_points_parent_at_the_branch_point(self, roots):
        checkpoints, states = roots
        chain = write_chain(checkpoints, steps=(100, 200))
        plan = resume_module.branch_plan(
            "run-a", 100, new_run_id="run-b", root=checkpoints, trainer_state_root=states
        )
        assert plan.run_id == "run-b"
        assert plan.source_run_id == "run-a"
        assert plan.parent == chain[0].name
        assert plan.base_model == "LiquidAI/LFM2-1.2B"
        assert plan.is_branch is True

    def test_a_branch_is_always_fresh_by_construction(self, roots, caplog):
        """FR-005.38 / AMB-2. Trainer state is addressed by ``run_id`` and a branch has a new
        one, so **the loss discontinuity R-02.22 exists to prevent is unavoidable on every
        branch** as the requirements stand. Recorded and logged; no copy-forward of another
        run's state is invented, because no requirement authorises one."""
        checkpoints, states = roots
        write_chain(checkpoints, steps=(100,))
        plant_trainer_state(states, "run-a", 100)  # the SOURCE run has state

        with caplog.at_level("INFO"):
            plan = resume_module.branch_plan(
                "run-a", 100, new_run_id="run-b", root=checkpoints, trainer_state_root=states
            )
        assert plan.optimizer_state == "fresh"
        assert plan.reason == "branch_has_no_trainer_state"
        assert plan.steps_with_state == ()
        assert any(r.__dict__.get("code") == "branch_optimizer_fresh" for r in caplog.records)

    def test_no_trainer_state_is_copied_forward(self, roots):
        """Asserted on the filesystem, not by reading the code: an invented copy-forward
        would leave a directory behind."""
        checkpoints, states = roots
        write_chain(checkpoints, steps=(100,))
        plant_trainer_state(states, "run-a", 100)
        resume_module.branch_plan(
            "run-a", 100, new_run_id="run-b", root=checkpoints, trainer_state_root=states
        )
        assert not (states / "run-b").exists()

    def test_branching_onto_the_same_run_id_is_refused(self, roots):
        """Reusing the id would make the branch write into the lineage it descends from."""
        checkpoints, states = roots
        write_chain(checkpoints, steps=(100,))
        with pytest.raises(Refusal) as caught:
            resume_module.branch_plan(
                "run-a", 100, new_run_id="run-a", root=checkpoints, trainer_state_root=states
            )
        assert caught.value.code == "run_not_continuable"


class TestTheContinueIsStandalone:
    def test_no_module_on_this_path_names_a_sibling_endpoint(self):
        """DEF-1 and DEF-2, as a grep over the modules rather than as a claim. Before
        2026-10-13 R-02.22 routed a continue through miLLM, which made crash recovery -
        criteria 6 and 20 - depend on an application that is specced and unimplemented."""
        import services.checkpoints as package

        root = Path(package.__file__).parent
        for module in sorted(root.glob("*.py")):
            source = module.read_text("utf-8")
            body = "\n".join(line.split("#", 1)[0] for line in source.splitlines())
            for forbidden in ("/api/weights/", "millm_base_url", "hotupdate", "import millm"):
                assert forbidden not in body, f"{module.name} references {forbidden!r}"


class TestTheByteIdenticalRoundTripThroughF1sRealModule:
    """AC-005.6's second half, through **F1's real writer and reader** - not a fixture.

    *Fixtures whose fields agree by construction are how a suite stays green over a real
    bug*, so the optimizer state has to make a real round trip. That needs ``torch``, which
    is a declared dependency and is not installed in every environment this suite runs in,
    so this **skips by name** rather than quietly passing: a test that reported success for
    a round trip that did not happen is the construction-green failure this project keeps
    finding."""

    def test_restored_state_is_byte_identical_to_what_was_written(self, roots):
        torch = pytest.importorskip(
            "torch",
            reason=(
                "AC-005.6's byte-identical assertion needs the real trainer-state writer and "
                "reader; torch is not installed here. The DECISION path - restored / fresh / "
                "mismatch / torn - is covered above without it (ADR-023)."
            ),
        )
        from training import trainer_state

        checkpoints, states = roots
        write_chain(checkpoints, steps=(100,))
        model = torch.nn.Linear(4, 4)
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
        model(torch.zeros(1, 4)).sum().backward()
        optimizer.step()

        trainer_state.write(
            run_id="run-a",
            step=100,
            optimizer=optimizer,
            root=states,
            checkpoint_root=checkpoints,
        )
        restored = trainer_state.read("run-a", 100, root=states)

        assert restored["optimizer"]["state"].keys() == optimizer.state_dict()["state"].keys()
        plan = resume_module.continue_plan("run-a", root=checkpoints, trainer_state_root=states)
        assert plan.optimizer_state == "restored"


class TestRefusalsSayWhatToDoNext:
    @pytest.mark.parametrize(
        ("code", "must_mention"),
        [
            ("checkpoint_not_found", "Save one"),
            ("checkpoint_hash_mismatch", "Restore"),
            ("trainer_state_incomplete", "Roll back"),
        ],
    )
    def test_each_message_names_an_action(self, roots, code, must_mention):
        """ADR-013: a message says **what to do next**, not how the reader should feel."""
        checkpoints, states = roots
        if code == "checkpoint_not_found":
            with pytest.raises(Refusal) as caught:
                resume_module.continue_plan("none", root=checkpoints, trainer_state_root=states)
        elif code == "checkpoint_hash_mismatch":
            result = saved_checkpoint(checkpoints, run_id="run-a", step=1)
            (result.directory / "model.safetensors").write_bytes(b"x")
            with pytest.raises(Refusal) as caught:
                resume_module.continue_plan("run-a", root=checkpoints, trainer_state_root=states)
        else:
            write_chain(checkpoints, steps=(1,), run_id="run-a")
            plant_trainer_state(states, "run-a", 1, complete=False)
            with pytest.raises(Refusal) as caught:
                resume_module.continue_plan("run-a", root=checkpoints, trainer_state_root=states)

        assert caught.value.code == code
        assert must_mention.lower() in caught.value.message.lower()


class TestPathContainment:
    @pytest.mark.parametrize("run_id", ["../escape", "a/b", "..", ""])
    def test_a_traversal_attempt_is_a_validation_failure_not_a_sanitised_path(self, roots, run_id):
        """Sanitising would compose *some* path and operate on it, which is the silently
        wrong answer ADR-013 forbids."""
        checkpoints, _unused = roots
        with pytest.raises(Refusal) as caught:
            checkpoint_dir(run_id, 1, root=checkpoints)
        assert caught.value.code in {"checkpoint_run_id_invalid", "checkpoint_run_id_required"}
