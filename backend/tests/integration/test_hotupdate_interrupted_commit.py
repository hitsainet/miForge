"""An interrupted commit: ``torn``, no retry, each recovery path, and the run log.

**AC-011.3 (BRD AC22), AC-011.11, and MUT-2 / MUT-3's integration half.**

The sequence this file reproduces is the one the whole commit bracket exists for: the write
begins, the watchdog expires, miLLM marks the set ``torn`` and **refuses to serve** rather
than answering from a parameter set whose layers sit at two different steps. *A visible 503
is recoverable; a silently wrong answer is not.*

Both recovery paths are exercised, and so is the third outcome - because ``STOP`` is the one
a careless implementation omits, and omitting it means stepping against a parameter set
whose version the run cannot state.
"""

from __future__ import annotations

from typing import Any

import pytest

from core.errors import Refusal
from tests.support.fixtures_hotupdate import Harness, RecordingOptimizer
from training.hotupdate import barrier, recovery, state
from training.hotupdate.recovery import Interruption, Recovery


def interrupt(harness: Harness) -> None:
    """Open a bracket, write, and let the watchdog expire before ``commit-end``."""
    harness.client.call("quiesce")
    harness.client.call("commit-begin")
    harness.fake.trip_commit_watchdog()


class _Model:
    """A module double, because ``object()`` made this test red for the WRONG REASON.

    Both arguments were ``object()``. Under **MUT-6** — ``require_clean`` accepting any
    state — the step no longer refuses at the precondition, so it proceeds, reaches
    ``model.parameters()`` and dies with ``AttributeError: 'object' object has no attribute
    'parameters'``. The test went red, which looked like a working control, but it never
    reached its own ``optimizer.steps == 0`` assertion — **the assertion it exists for was
    unreachable under precisely the defect it guards.**

    Commit ``c40b082`` fixed exactly this double in ``test_hotupdate_cancellation.py`` and
    **did not fix this one**: a half-applied fix, and the three sibling documents of this
    project's own rule say to grep the *words* you removed across the whole chain rather
    than trusting that the fix landed everywhere.

    Deliberately **not** a ``torch.nn.Module``, for the reason that file records: a double
    needing torch puts the test back under the environment's control, which is what went
    wrong in the first place.
    """

    def parameters(self) -> list[Any]:
        return [_Parameter()]


class _Parameter:
    """Something an optimizer can hold. Never read; it only has to exist."""

    requires_grad = True
    grad = None


class _Batch:
    """F2's batch shape: a named objective and a record. Not a dict.

    A dict was the first attempt and it refused `batch_objective_missing`, which is the
    step reading `batch.objective` as an attribute — so the double was still unable to
    express a completed step, just one refusal later. Matched to
    `test_hotupdate_cancellation.py`'s `Batch`, which is the shape F1's step actually reads.
    """

    objective = "sft"

    @property
    def record(self) -> dict[str, Any]:
        return {}


class _Source:
    def next(self) -> _Batch:
        return _Batch()


class _Objective:
    """F2's objective: a `name` and `loss(model, batch)`.

    Read from `test_hotupdate_cancellation.py`'s `Objective` rather than guessed — a first
    attempt gave it `__call__` and MUT-6 went red with
    `AttributeError: '_Objective' object has no attribute 'loss'`, which is the same
    wrong-reason red this whole double exists to remove. Copying the interface from the
    file that already models it is the cheap answer; inventing one is how a double ends up
    unable to express the behaviour under test.
    """

    name = "sft"

    def loss(self, model: Any, batch: Any) -> _Scalar:
        return _Scalar()


class _Scalar:
    """A loss that can be backwarded and read, without torch."""

    def backward(self) -> None:
        return None

    def item(self) -> float:
        return 0.5

    def detach(self) -> _Scalar:
        return self

    def float(self) -> _Scalar:
        return self


class _Registry:
    def get(self, name: str) -> _Objective:
        return _Objective()


class TestTheSetGoesTornAndServingIsRefused:
    def test_a_commit_that_overruns_leaves_the_set_torn_and_nothing_is_served(
        self, harness: Harness
    ) -> None:
        harness.attach()
        interrupt(harness)
        assert harness.fake.weights_state == "torn"
        assert harness.fake.serve_completion() == "", "miLLM never serves a torn set"
        assert harness.fake.served_while_not_clean == ["torn"]

    def test_no_step_begins_against_it(self, harness: Harness) -> None:
        """**MUT-8's integration half.** The precondition is the step's, not the caller's.

        **The call-count assertions are what make MUT-6 bite, and they were learned the
        hard way.** This test passed `object()` for the model and the batch source, so
        under MUT-6 — `require_clean` accepting any state — it died with
        `AttributeError: 'object' object has no attribute 'parameters'` and never reached
        its own assertions: red, and red for the wrong reason, which reads exactly like a
        working control.

        Giving it real doubles then made MUT-6 **survive**, which was worse. The reason is
        the point: with a working double the step proceeds, calls `commit-begin`, and **the
        fake refuses it** with `409 WEIGHTS_NOT_CLEAN` — so a `Refusal` with code
        `weights_not_clean` is raised either way and the two cases are indistinguishable by
        the code alone. The client-side precondition and the server-side refusal are
        different guarantees: R-02.30 requires miForge to refuse *before* it asks, because
        a trainer that relies on miLLM to say no has no precondition at all, only a remote
        one it cannot depend on when the surface is unreachable.

        So the discriminating assertion is that **no call was made**. ADR-016 #1 in its own
        words: assert the payload *and the call count* — "was called" passes against a call
        sending the wrong arguments, and here "it refused" passes against a refusal that
        came from the wrong side of the wire.
        """
        harness.attach()
        interrupt(harness)
        harness.fake.calls.clear()
        optimizer = RecordingOptimizer([])
        with pytest.raises(Refusal) as caught:
            barrier.hot_update_step(
                _Model(),
                optimizer,
                _Source(),
                client=harness.client,
                timings=harness.timings,
                objectives=_Registry(),
            )
        assert caught.value.code == "weights_not_clean"
        assert optimizer.steps == 0, "a step ran against a torn parameter set"
        # The refusal came from HERE, not from miLLM saying no.
        assert (
            harness.fake.call_count("quiesce") == 0
        ), "the step quiesced a serving process before checking its own precondition"
        assert harness.fake.call_count("commit-begin") == 0, (
            "the step opened a commit against a non-clean set and relied on miLLM to "
            "refuse it. R-02.30 requires the refusal on this side: a precondition that "
            "only the other end enforces is not a precondition."
        )

    def test_the_state_survives_a_detach(self, harness: Harness) -> None:
        """BRD-01 R-01.23: *"the state survives detach and re-attach, so the next process to
        attach sees what it inherited."* A fake that cleared it on detach would make every
        recovery test agree with a rule miLLM does not have."""
        harness.attach()
        interrupt(harness)
        harness.client.call("detach")
        assert harness.fake.weights_state == "torn"


class TestRecoveryByCompletingTheCommit:
    def test_the_same_process_sends_a_late_commit_end_and_the_overrun_is_recorded(
        self, harness: Harness
    ) -> None:
        """R-02.30(a), BRD-01 R-01.23a. All three conditions hold, so completion is taken."""
        harness.attach()
        interrupt(harness)
        plan = recovery.plan(
            Interruption(
                same_process=True,
                holds_update=True,
                attachment_id=harness.client.attachment_id,
                commit_begun=True,
                last_checkpoint_step=100,
            )
        )
        assert plan.outcome is Recovery.COMPLETE
        harness.fake.calls.clear()
        recovery.complete_the_commit(harness.client)
        assert harness.fake.weights_state == "clean"
        assert harness.fake.commit_overran is True
        assert harness.fake.ops() == ["commit-end", "resume"]
        # And serving resumes, which is the point of recovering at all.
        assert harness.fake.serve_completion().startswith("served by")

    def test_the_step_is_still_not_retried(self, harness: Harness) -> None:
        """Recovering the *parameter set* and retrying the *step* are different things.
        R-02.31 forbids the second even when the first succeeds."""
        harness.attach()
        interrupt(harness)
        assert (
            recovery.is_retryable(
                Interruption(
                    same_process=True,
                    holds_update=True,
                    attachment_id="a",
                    commit_begun=True,
                    last_checkpoint_step=1,
                )
            )
            is False
        )


class TestRecoveryByRollingBack:
    def test_a_load_returns_the_set_to_clean_and_the_run_resumes_from_that_step(
        self, harness: Harness
    ) -> None:
        """R-02.30(b) through BRD-01 R-01.25, which *"verifies ``weights_sha256`` before
        touching a resident tensor"* and, for a matching architecture, writes **in place** -
        so the attachment's handles stay valid. That is the only reason the verification is
        re-run rather than the attachment being rebuilt."""
        harness.attach()
        interrupt(harness)
        plan = recovery.plan(
            Interruption(
                same_process=True,
                holds_update=False,
                attachment_id=harness.client.attachment_id,
                commit_begun=True,
                last_checkpoint_step=100,
            )
        )
        assert plan.outcome is Recovery.ROLLBACK
        assert plan.resume_from_step == 100
        harness.fake.calls.clear()
        harness.client.call("load", {"checkpoint": "step-000100", "manifest_sha256": "d" * 64})
        assert harness.fake.weights_state == "clean"
        assert harness.fake.loads[-1]["checkpoint"] == "step-000100"
        # The identity moved, which is exactly what the re-verification exists to notice.
        assert state.read(harness.client)[1].manifest_sha256 == "d" * 64

    def test_a_load_is_permitted_while_torn_which_is_its_recovery_role(
        self, harness: Harness
    ) -> None:
        """R-01.25 / R-01.23b. **This is the only recovery that works when the trainer is
        dead**, which is why the missing load call was the defect it was."""
        harness.attach()
        interrupt(harness)
        harness.client.call("detach")
        assert harness.fake.weights_state == "torn"
        harness.client.call("load", {"checkpoint": "step-000100"})
        assert harness.fake.weights_state == "clean"

    def test_a_load_is_refused_while_ANOTHER_process_holds_an_open_commit(
        self, harness: Harness
    ) -> None:
        """*Two writers to one parameter set is what the barrier exists to prevent.*

        "Another process" is the whole of R-01.25's condition, so the fake is put into the
        state where a **different** attachment holds the open commit. The holder recovering
        its *own* commit is the case above, and conflating the two would refuse R-02.30(b)'s
        primary recovery path - which is the only one that works when the trainer is dead.
        """
        harness.attach()
        harness.client.call("quiesce")
        harness.client.call("commit-begin")
        # Someone else is now the holder, and our header no longer matches.
        harness.fake.attachment_id = "another-trainers-attachment"
        with pytest.raises(Refusal) as caught:
            harness.client.call("load", {"checkpoint": "x"})
        assert caught.value.code == "weights_not_clean"

    def test_the_holder_may_load_while_its_own_commit_is_open(self, harness: Harness) -> None:
        """The other side of that condition, asserted so the distinction cannot be lost:
        R-01.31 is what makes the holder recognisable on a call that merely *reads* the
        header."""
        harness.attach()
        harness.client.call("quiesce")
        harness.client.call("commit-begin")
        harness.fake.trip_commit_watchdog()
        assert harness.client.call("load", {"checkpoint": "step-000100"})["in_place"] is True


class TestTheThirdOutcome:
    def test_with_no_checkpoint_the_run_stops_and_names_the_action(self, harness: Harness) -> None:
        harness.attach()
        interrupt(harness)
        plan = recovery.plan(
            Interruption(
                same_process=False,
                holds_update=False,
                attachment_id=None,
                commit_begun=True,
                last_checkpoint_step=None,
            )
        )
        assert plan.outcome is Recovery.STOP
        assert plan.operator_action is not None
        assert "cannot state" in plan.reason


class TestTheRunLogRecordsWhichPathAndWhichStep:
    def test_the_record_names_the_path_the_failed_condition_and_the_resumed_step(self) -> None:
        """FR-011.42: *a run that cannot say which step it resumed from cannot have its loss
        curve read*, and FR-011.45: the gap is explicit rather than the curve looking
        continuous across a step that never completed."""
        plan = recovery.plan(
            Interruption(
                same_process=True,
                holds_update=False,
                attachment_id="a",
                commit_begun=True,
                last_checkpoint_step=100,
            )
        )
        record = recovery.abandonment_record(
            run_id="hot-run", step=101, plan_taken=plan, cause="commit watchdog expired"
        )
        assert record == {
            "run_id": "hot-run",
            "step": 101,
            "event": "step_abandoned_commit_interrupted",
            "cause": "commit watchdog expired",
            "retried": False,
            "discontinuity": True,
            "recovery_path": "rollback",
            "recovery_reason": plan.reason,
            "completion_unavailable_because": ["holds_the_computed_update"],
            "resumed_from_step": 100,
            "operator_action_required": None,
            "ordering": [],
        }
