"""Recovery: three outcomes, and the never-retry rule  (R-02.30, R-02.31; MUT-3).

``STOP`` is asserted as hard as the other two, because it is the one a careless
implementation omits - and omitting it means falling through into a step against a parameter
set whose version the run cannot state.

**MUT-3** makes an interrupted step retryable. Its symptom is that *the loss curve still
moves*: a non-idempotent AdamW update re-applied to tensors that already received it while
others did not compounds the damage, and nothing raises.
"""

from __future__ import annotations

import pytest

from core.errors import Refusal
from tests.support.fixtures_hotupdate import Harness
from training.hotupdate import recovery
from training.hotupdate.recovery import Interruption, Recovery


def interruption(**overrides: object) -> Interruption:
    base: dict[str, object] = {
        "same_process": True,
        "holds_update": True,
        "attachment_id": "attachment-0001",
        "commit_begun": True,
        "last_checkpoint_step": 100,
    }
    base.update(overrides)
    return Interruption(**base)  # type: ignore[arg-type]


class TestTheThreeOutcomes:
    def test_complete_needs_all_three_conditions(self) -> None:
        assert recovery.plan(interruption()).outcome is Recovery.COMPLETE

    @pytest.mark.parametrize(
        ("override", "condition"),
        [
            ({"same_process": False}, "same_process"),
            ({"holds_update": False}, "holds_the_computed_update"),
            ({"attachment_id": None}, "holds_the_attachment_id"),
        ],
    )
    def test_any_one_missing_condition_rules_completion_out_and_SAYS_WHICH(
        self, override: dict[str, object], condition: str
    ) -> None:
        """*"and logs which one was false"* - so the run log explains **why** the other path
        was taken rather than merely that it was."""
        plan = recovery.plan(interruption(**override))
        assert plan.outcome is Recovery.ROLLBACK
        assert condition in plan.unavailable_conditions

    def test_rollback_names_the_step_it_resumes_from(self) -> None:
        plan = recovery.plan(interruption(holds_update=False))
        assert plan.resume_from_step == 100

    def test_stop_when_there_is_nothing_to_roll_back_to(self) -> None:
        """**The honest outcome.** No checkpoint means no known version of the parameter
        set, and the run stops rather than continuing against one it cannot name."""
        plan = recovery.plan(interruption(holds_update=False, last_checkpoint_step=None))
        assert plan.outcome is Recovery.STOP
        assert "nothing to roll back to" in plan.reason
        assert plan.operator_action is not None

    def test_rollback_is_always_available_and_completion_is_the_optimisation(self) -> None:
        """FR-011.40, stated as a property: with a checkpoint, the answer is never STOP."""
        for override in ({"same_process": False}, {"holds_update": False}, {"attachment_id": None}):
            assert recovery.plan(interruption(**override)).outcome is Recovery.ROLLBACK


class TestThePostCrashOrdering:
    def test_after_a_process_death_the_operator_clear_is_named_as_the_blocker(self) -> None:
        """FR-011.41, FPRD AMB-N. The id is gone, so path (a) is permanently unavailable;
        the rollback ``load`` needs no attachment, but **resuming** does - and attaching
        needs a stale attachment cleared by an **operator**."""
        plan = recovery.plan(
            interruption(same_process=False, holds_update=False, attachment_id=None)
        )
        assert plan.outcome is Recovery.ROLLBACK
        assert plan.operator_action is not None
        assert "OPERATOR-ONLY" in plan.operator_action
        assert "does not contain the call" in plan.operator_action

    def test_the_ordering_is_load_then_clear_then_attach_then_verify_then_resume(self) -> None:
        plan = recovery.plan(
            interruption(same_process=False, attachment_id=None, holds_update=False)
        )
        assert [step.split(" ")[0] for step in plan.ordering] == [
            "load",
            "an",
            "attach",
            "verify",
            "resume",
        ]
        assert plan.ordering == recovery.POST_CRASH_ORDERING

    def test_the_same_process_needs_no_ordering(self) -> None:
        assert recovery.plan(interruption(holds_update=False)).ordering == ()


class TestNeverRetry:
    def test_an_interrupted_commit_is_never_retryable(self) -> None:
        """**MUT-3.** R-02.31: the update is not idempotent."""
        assert recovery.is_retryable(interruption(commit_begun=True)) is False

    def test_a_quiesce_timeout_BEFORE_commit_begin_is_the_one_retryable_case(self) -> None:
        """BRD AC7. Nothing was written and the weights are unchanged, so the step is
        retried and the timeout recorded - detectable because ``commit-begin`` never
        returned successfully, which is the only honest test from this process."""
        assert recovery.is_retryable(interruption(commit_begun=False)) is True

    def test_the_abandonment_record_carries_the_discontinuity_and_the_resumed_step(self) -> None:
        """FR-011.45, BRD AC22: *the loss curve carries the gap explicitly rather than
        appearing continuous across a step that never completed.*"""
        plan = recovery.plan(interruption(holds_update=False))
        record = recovery.abandonment_record(
            run_id="hot-run", step=101, plan_taken=plan, cause="commit watchdog expired"
        )
        assert record["discontinuity"] is True
        assert record["retried"] is False
        assert record["recovery_path"] == "rollback"
        assert record["resumed_from_step"] == 100
        assert record["completion_unavailable_because"] == ["holds_the_computed_update"]


class TestCompletingTheCommit:
    def test_a_late_commit_end_is_accepted_and_serving_is_released(self, harness: Harness) -> None:
        """R-01.23a: *"miLLM accepts a late commit-end from the same attachment, returns to
        clean, and records that the commit overran."*"""
        harness.attach()
        harness.client.call("quiesce")
        harness.client.call("commit-begin")
        harness.fake.trip_commit_watchdog()
        assert harness.fake.weights_state == "torn"
        harness.fake.calls.clear()
        recovery.complete_the_commit(harness.client)
        assert harness.fake.ops() == ["commit-end", "resume"]
        assert harness.fake.weights_state == "clean"
        assert harness.fake.commit_overran is True

    def test_a_refused_late_commit_end_is_reported_and_not_retried(self, harness: Harness) -> None:
        harness.attach()
        harness.fake.refuse_always["commit-end"] = (409, "WEIGHTS_NOT_CLEAN")
        harness.fake.calls.clear()
        with pytest.raises(Refusal) as caught:
            recovery.complete_the_commit(harness.client)
        assert caught.value.code == "weights_recovery_failed"
        assert caught.value.details["retried"] is False
        assert harness.fake.call_count("commit-end") == 1


class TestTheTrainerStateReadGoesThroughF1:
    def test_it_delegates_rather_than_respelling_the_layout(self) -> None:
        """FR-011.39, AMB-O: F11 is the layout's **third** reader and re-specifies nothing -
        no file names, no tensor keys, no zero-padding. Asserted on the source, because the
        property is an absence."""
        import ast
        import inspect

        source = inspect.getsource(recovery)
        tree = ast.parse(source)
        function = next(
            node
            for node in ast.walk(tree)
            if isinstance(node, ast.FunctionDef) and node.name == "restore_trainer_state"
        )
        literals = [
            node.value
            for node in ast.walk(function)
            if isinstance(node, ast.Constant) and isinstance(node.value, str)
        ]
        spellings = [text for text in literals if ".pt" in text or "step-" in text or "/" in text]
        assert spellings == [], f"F11 re-spelled the trainer-state layout: {spellings}"
