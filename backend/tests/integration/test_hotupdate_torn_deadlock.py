"""A contract gap, reproduced: both of BRD-01's recovery paths can be unavailable at once.

**This is a finding about another repository's requirements, pinned here rather than fixed
there** (that would be a commit in that repo). It is reported to the operator, and this file
is what stops the finding going stale silently: if miLLM's implementation or BRD-01's text
closes the gap, the behaviour asserted here changes and the test says so.

## The sequence

1. a trainer quiesces, calls ``commit-begin``, starts writing, and **dies**;
2. the commit watchdog expires, so the set is ``torn`` and ``/v1/*`` is refused (R-01.22).
   miLLM *"never self-resumes out of it"*;
3. the heartbeat stops arriving, so the attachment is marked **stale** (R-01.4);
4. **R-01.23 names exactly two recoveries.** Path (a) is *"the attached process completes
   its write and calls ``commit-end``"* - it needs a live trainer **and** the attachment id,
   which is returned once and by no read path again, so after a process death it is gone.
   Path (b) is *"an operator or trainer loads a checkpoint"* - and **R-01.25 refuses a load
   "while another process holds an open commit"**. The dead trainer's commit is still open.
5. **R-01.29's ``clear-stale`` clears the attachment and says nothing about its open
   commit.** So whether step 4's refusal can be lifted at all is unstated.

**Both paths can therefore be closed simultaneously**, and the set stays ``torn`` with
serving refused. This is the same shape as the missing load call that BRD-01 itself records
as a lesson - *"a section title promising a capability is not a requirement defining one"* -
one level further in: a recovery path **is** defined, and in this case nothing can reach it.

## What F11 does about it, and what it does not

It **reports** it, by name, with the two refusals it actually received, and stops. It does
**not** loop, does not retry, and does not try to force anything: R-02.34 forbids working
around a refusal on miForge's own initiative, and an optimizer that clears someone else's
state to get started is how two trainers end up writing one parameter set. The honest
outcome here is ``STOP`` with an operator action named - which is exactly why ``STOP`` is a
first-class recovery outcome rather than a fall-through.
"""

from __future__ import annotations

import pytest

from core.errors import Refusal
from tests.support.fixtures_hotupdate import Harness
from training.hotupdate import recovery
from training.hotupdate.recovery import Interruption, Recovery


def kill_the_trainer_mid_commit(harness: Harness) -> None:
    """Steps 1 to 3: an open commit, an expired watchdog, and a stale attachment."""
    harness.attach()
    harness.client.call("quiesce")
    harness.client.call("commit-begin")
    harness.fake.trip_commit_watchdog()
    # The trainer is gone: its heartbeat stops and miLLM marks the attachment stale. The
    # commit it opened stays open - nothing in BRD-01 closes it on staleness.
    harness.fake.stale = True
    harness.fake.foreign_holder = "trainer:the-dead-worker"


class TestBothRecoveriesCanBeClosed:
    def test_the_set_is_torn_and_serving_is_refused(self, harness: Harness) -> None:
        kill_the_trainer_mid_commit(harness)
        assert harness.fake.weights_state == "torn"
        assert harness.fake.serve_completion() == ""

    def test_path_a_is_unavailable_because_the_id_died_with_the_process(self) -> None:
        plan = recovery.plan(
            Interruption(
                same_process=False,
                holds_update=False,
                attachment_id=None,
                commit_begun=True,
                last_checkpoint_step=100,
                attachment_stale=True,
            )
        )
        assert plan.outcome is Recovery.ROLLBACK, "the plan correctly reaches for path (b)"
        assert "holds_the_attachment_id" in plan.unavailable_conditions

    def test_path_b_is_then_REFUSED_for_the_dead_trainers_open_commit(
        self, harness: Harness
    ) -> None:
        """**The gap.** A fresh process - no attachment, no id - takes path (b) and miLLM
        refuses it, because from its side an open commit belongs to *another* process."""
        kill_the_trainer_mid_commit(harness)
        from training.hotupdate.client import WeightsClient

        fresh = WeightsClient(settings=harness.client.settings, transport=harness.fake.transport())
        try:
            with pytest.raises(Refusal) as caught:
                fresh.call("load", {"checkpoint": "step-000100"})
            assert caught.value.code == "weights_not_clean"
            assert caught.value.details["upstream_code"] == "WEIGHTS_NOT_CLEAN"
        finally:
            fresh.close()

    def test_clearing_the_stale_attachment_does_not_state_that_it_closes_the_commit(
        self, harness: Harness
    ) -> None:
        """R-01.29 defines what ``clear-stale`` clears: *an attachment marked stale*. Whether
        it also closes that attachment's open commit bracket is **not stated**, so the fake
        models exactly what the requirement says and no more - *a fake that invented the
        convenient behaviour would make this gap untestable and would make F11 agree with a
        rule miLLM does not have.*

        And miForge cannot perform the clear in any case: R-01.29 is operator-only and the
        code does not contain the call (R-02.34).
        """
        kill_the_trainer_mid_commit(harness)
        assert harness.fake.commit_open is True
        # An operator clears it. (miForge cannot: the call is operator-only and absent here.)
        harness.fake.role = "operator"
        import httpx

        http = httpx.Client(base_url="http://millm.test", transport=harness.fake.transport())
        response = http.post(
            "/api/weights/clear-stale", headers={"authorization": "Bearer trainer-token"}
        )
        assert response.status_code == 200
        assert harness.fake.commit_open is True, (
            "the fake models R-01.29 as written: the attachment is cleared and the open "
            "commit is not mentioned. If miLLM's implementation closes it, this assertion "
            "is the tripwire that says the gap is closed and this file can go."
        )

    def test_miforge_stops_and_names_the_action_rather_than_looping(self, harness: Harness) -> None:
        """What F11 does: **reports, stops, and never forces.** The run does not continue
        against a parameter set whose version it cannot state, and the message names the
        action an operator must take and that miForge cannot take it."""
        kill_the_trainer_mid_commit(harness)
        plan = recovery.plan(
            Interruption(
                same_process=False,
                holds_update=False,
                attachment_id=None,
                commit_begun=True,
                last_checkpoint_step=None,
                attachment_stale=True,
            )
        )
        assert plan.outcome is Recovery.STOP
        assert plan.operator_action is not None
        assert (
            recovery.is_retryable(
                Interruption(
                    same_process=False,
                    holds_update=False,
                    attachment_id=None,
                    commit_begun=True,
                    last_checkpoint_step=None,
                )
            )
            is False
        )
        assert harness.fake.call_count("clear-stale") == 0, "R-02.34: miForge never clears"
