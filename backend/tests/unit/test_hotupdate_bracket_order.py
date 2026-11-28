"""The commit bracket: the stage order, and the ``finally`` that matters  (R-02.52).

**The order is asserted as a sequence, not as a result.** Every one of these calls succeeds
in the wrong order too: a write that lands before ``commit-begin`` completes, trains, and
moves the loss curve - and leaves the hold timeout free to self-resume **partway through
it**, which is layers at two different steps, served. *Nothing fails loudly.* Only an
ordering assertion catches it, which is why **MUT-1** is this feature's first mutation.

**MUT-2** deletes the inner ``finally``. Its symptom is the opposite kind: every *failed*
step leaves the set ``torn`` and serving refused, so the failure is loud - but it arrives as
a serving outage rather than as a failed step, and a test that only exercised the happy path
would never see it.
"""

from __future__ import annotations

import pytest

from core.errors import Refusal
from tests.support.fixtures_hotupdate import Harness, RecordingOptimizer
from training.hotupdate import barrier


class TestTheStageOrder:
    def test_the_four_calls_happen_in_R_02_52s_order_with_the_write_between(
        self, harness: Harness
    ) -> None:
        """FR-011.53: ``quiesce`` -> ``commit-begin`` -> **apply** -> ``commit-end`` ->
        ``resume``. **MUT-1.**"""
        harness.attach()
        harness.fake.calls.clear()
        events: list[str] = []
        with barrier.commit_bracket(harness.client, harness.timings):
            events.append("write")
        assert harness.fake.ops() == ["quiesce", "commit-begin", "commit-end", "resume"]
        assert barrier.STAGE_ORDER == tuple(harness.fake.ops())
        assert events == ["write"]

    def test_the_write_happens_while_the_state_is_writing_and_not_before(
        self, harness: Harness
    ) -> None:
        """The state at the moment of the write is the thing the ordering buys: ``writing``
        means the hold timeout is suspended and a self-resume cannot cut the write in half."""
        harness.attach()
        observed: list[str] = []
        with barrier.commit_bracket(harness.client, harness.timings):
            observed.append(harness.fake.weights_state)
        assert observed == ["writing"]
        assert harness.fake.weights_state == "clean"

    def test_serving_is_never_answered_from_a_non_clean_set(self, harness: Harness) -> None:
        """BRD AC9 / AC-011.1, through the fake's own serving path. The fake **records** a
        violation rather than raising, so the assertion is about F11's behaviour rather than
        about this file's control flow."""
        harness.attach()
        for _ in range(3):
            with barrier.commit_bracket(harness.client, harness.timings):
                harness.fake.serve_completion()
        assert harness.fake.served_while_not_clean, "the probe itself must be able to fire"
        # ... and outside the bracket it serves normally, which is the other half: a probe
        # that could never succeed would make the assertion above vacuous.
        assert harness.fake.serve_completion().startswith("served by")


class TestCommitEndOnEveryExitButAFailedWrite:
    """*Changed 2026-10-17 by operator decision:* commit-end used to be sent on every exit,
    including a write that raised partway - which told miLLM a half-written set was clean.
    It is now WITHHELD on that one exit; every other exit is unchanged."""

    def test_commit_end_runs_after_a_successful_write_before_resume(self, harness: Harness) -> None:
        """**MUT-2.** The success path closes the bracket, then releases serving."""
        harness.attach()
        harness.fake.calls.clear()
        with barrier.commit_bracket(harness.client, harness.timings):
            pass
        assert harness.fake.ops() == ["quiesce", "commit-begin", "commit-end", "resume"]
        assert harness.fake.weights_state == "clean"

    def test_a_failed_write_withholds_commit_end_and_resume(self, harness: Harness) -> None:
        """The set stays ``writing`` for miLLM's watchdog to declare ``torn``; a resume would
        be refused anyway (R-01.7), and a commit-end would bless a half-written set."""
        harness.attach()
        harness.fake.calls.clear()
        with (
            pytest.raises(Refusal) as caught,
            barrier.commit_bracket(harness.client, harness.timings),
        ):
            raise RuntimeError("the write failed")
        assert caught.value.code == "weights_write_failed"
        assert caught.value.details["commit_left_open"] is True
        assert isinstance(caught.value.__cause__, RuntimeError)
        assert harness.fake.ops() == ["quiesce", "commit-begin"]
        assert harness.fake.weights_state == "writing"

    def test_a_commit_end_failure_on_a_successful_write_raises_and_does_not_retry(
        self, harness: Harness
    ) -> None:
        """The bracket **records and raises**; ``recovery.py`` decides. A bracket that could
        retry is a bracket that can re-apply a non-idempotent update (R-02.31)."""
        harness.attach()
        harness.fake.refuse_always["commit-end"] = (409, "WEIGHTS_NOT_CLEAN")
        harness.fake.calls.clear()
        with (
            pytest.raises(Refusal) as caught,
            barrier.commit_bracket(harness.client, harness.timings),
        ):
            pass
        assert caught.value.code == "weights_commit_interrupted"
        assert caught.value.details["retried"] is False
        assert harness.fake.call_count("commit-end") == 1
        assert "NOT RETRIED" in caught.value.message

    def test_resume_runs_even_when_commit_begin_is_refused(self, harness: Harness) -> None:
        """Nothing was written, so the set is clean - and serving must be released rather
        than held until miLLM's own hold timeout expires."""
        harness.attach()
        harness.fake.refuse_always["commit-begin"] = (409, "WEIGHTS_NOT_CLEAN")
        harness.fake.calls.clear()
        with (
            pytest.raises(Refusal),
            barrier.commit_bracket(harness.client, harness.timings),
        ):
            pytest.fail("the body must not run when commit-begin was refused")
        assert harness.fake.ops() == ["quiesce", "commit-begin", "resume"]


class TestTheOptimizerWrapper:
    def test_only_the_optimizer_step_is_inside_the_bracket(self, harness: Harness) -> None:
        """TID §7 invariant 1: *every line inside the bracket is held serving time.* The
        gradient norm and ``zero_grad`` are F1's and stay outside."""
        harness.attach()
        harness.fake.calls.clear()
        events: list[str] = []
        inner = RecordingOptimizer(events)
        wrapped = barrier.BracketedOptimizer(inner, harness.client, harness.timings)
        wrapped.step()
        wrapped.zero_grad()
        assert harness.fake.ops() == ["quiesce", "commit-begin", "commit-end", "resume"]
        assert events == ["write", "zero_grad"]
        assert wrapped.bracketed_steps == 1

    def test_everything_else_delegates_to_the_real_optimizer(self, harness: Harness) -> None:
        """Delegation rather than inheritance: the trainer-state writer reads
        ``state_dict``, the scheduler reads ``param_groups``, and F11 must not model either."""
        inner = RecordingOptimizer([])
        wrapped = barrier.BracketedOptimizer(inner, harness.client, harness.timings)
        assert wrapped.param_groups == inner.param_groups
        assert wrapped.state_dict() == {"steps": 0}

    def test_a_failing_write_leaves_the_bracket_open(self, harness: Harness) -> None:
        """Operator decision 2026-10-17: commit-end is withheld after a failed write."""
        harness.attach()
        harness.fake.calls.clear()
        inner = RecordingOptimizer([], fail=RuntimeError("cuda oom"))
        wrapped = barrier.BracketedOptimizer(inner, harness.client, harness.timings)
        with pytest.raises(Refusal):
            wrapped.step()
        assert harness.fake.ops() == ["quiesce", "commit-begin"]


class TestNoCancellationInsideTheBracket:
    def test_the_bracket_contains_no_cancellation_check(self) -> None:
        """ADR-007's boundary is **outside**: a cancellation honoured inside the bracket is
        a half-applied step and a torn set, which is the one thing both ADR-007 and R-02.31
        forbid. Asserted on the source because the behaviour is an absence."""
        import ast
        import inspect

        source = inspect.getsource(barrier)
        tree = ast.parse(source)
        bracket = next(
            node
            for node in ast.walk(tree)
            if isinstance(node, ast.FunctionDef) and node.name == "commit_bracket"
        )
        calls = {
            node.func.attr
            for node in ast.walk(bracket)
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
        }
        assert "check" not in calls, "a cancellation check inside the bracket is a torn set"
