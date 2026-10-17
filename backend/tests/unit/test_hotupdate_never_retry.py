"""R-02.31 end to end: an interrupted step is abandoned, and the one case that is not.

Separate from ``test_hotupdate_recovery_selection.py`` on purpose: that file asserts the
*decision*, this one asserts the **behaviour** - that an interrupted step produces no second
write, and that a pre-``commit-begin`` timeout produces exactly one retry from unchanged
weights.

**MUT-3 must turn this file red.** Its symptom is the dangerous kind: a re-applied AdamW
update makes the loss curve move, so the run looks healthy while the parameter set has been
updated twice in some tensors and once in others.
"""

from __future__ import annotations

import pytest

from core.errors import Refusal
from tests.support.fixtures_hotupdate import Harness, RecordingOptimizer
from training.hotupdate import barrier, recovery
from training.hotupdate.recovery import Interruption


class TestAnInterruptedStepIsNotRetried:
    def test_a_failed_commit_produces_exactly_one_write_attempt(self, harness: Harness) -> None:
        harness.attach()
        harness.fake.refuse_always["commit-end"] = (409, "WEIGHTS_TORN")
        events: list[str] = []
        optimizer = barrier.BracketedOptimizer(
            RecordingOptimizer(events), harness.client, harness.timings
        )
        with pytest.raises(Refusal):
            optimizer.step()
        assert events.count("write") == 1
        assert optimizer.bracketed_steps == 1
        assert harness.fake.call_count("commit-begin") == 1

    def test_the_bracket_itself_contains_no_retry(self) -> None:
        """A bracket that could retry is a bracket that can compound a non-idempotent
        update, which is why the decision lives in ``recovery.py``. Asserted on the source."""
        import ast
        import inspect

        tree = ast.parse(inspect.getsource(barrier))
        bracket = next(
            node
            for node in ast.walk(tree)
            if isinstance(node, ast.FunctionDef) and node.name == "commit_bracket"
        )
        loops = [node for node in ast.walk(bracket) if isinstance(node, (ast.For, ast.While))]
        assert loops == [], "a loop in the bracket is a retry waiting to be written"

    def test_the_abandonment_is_recorded_as_a_discontinuity(self) -> None:
        plan = recovery.plan(
            Interruption(
                same_process=True,
                holds_update=False,
                attachment_id="a",
                commit_begun=True,
                last_checkpoint_step=50,
            )
        )
        record = recovery.abandonment_record(run_id="r", step=51, plan_taken=plan, cause="watchdog")
        assert record["event"] == "step_abandoned_commit_interrupted"
        assert record["discontinuity"] is True
        assert record["retried"] is False


class TestTheOneRetryableCase:
    def test_a_quiesce_timeout_before_commit_begin_leaves_the_weights_unchanged(
        self, harness: Harness
    ) -> None:
        """BRD AC7 / AC-011.4. The fake stalls the ``quiesce`` itself, so ``commit-begin``
        is never reached and no tensor was written."""
        harness.attach()
        harness.fake.stall_next("quiesce")
        events: list[str] = []
        optimizer = barrier.BracketedOptimizer(
            RecordingOptimizer(events), harness.client, harness.timings
        )
        with pytest.raises(Refusal) as caught:
            optimizer.step()
        assert caught.value.code == "weights_surface_unreachable"
        assert events == [], "nothing may be written when the quiesce did not land"
        assert harness.fake.call_count("commit-begin") == 0
        assert harness.fake.weights_state == "clean"
        assert recovery.is_retryable(
            Interruption(
                same_process=True,
                holds_update=True,
                attachment_id="a",
                commit_begun=False,
                last_checkpoint_step=None,
            )
        )

    def test_the_retry_then_succeeds_and_produces_one_write(self, harness: Harness) -> None:
        """The retried step runs against the **same** unchanged weights, which is what makes
        it a legitimate retry rather than the forbidden one."""
        harness.attach()
        harness.fake.stall_next("quiesce")
        events: list[str] = []
        optimizer = barrier.BracketedOptimizer(
            RecordingOptimizer(events), harness.client, harness.timings
        )
        with pytest.raises(Refusal):
            optimizer.step()
        optimizer.step()
        assert events.count("write") == 1
        assert harness.fake.weights_state == "clean"
        assert harness.fake.steps_since_load == 1
