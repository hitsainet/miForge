"""A worker restart mid-run, and a serving-process restart - the two different restarts.

**AC-011.8, AC-011.12, FR-011.41; MUT-12 and MUT-17.**

They are not symmetrical, and the asymmetry is the point:

* **miForge restarts.** The computed update is gone with the process and the attachment id
  is gone unless it was persisted, so **R-02.30(a) is permanently unavailable** - miLLM
  returns the id once and no read path returns it again. Re-attaching additionally needs an
  **operator** to clear the stale attachment. R-02.30 does not state that ordering;
  R-02.33's stale row implies it (FPRD AMB-N).
* **the serving process restarts.** Its barrier state lives in process memory and a restart
  begins ``clean``, while the resident identity goes back to the last **load**. So it
  reports ``clean`` while holding a parameter set this run never wrote. State alone cannot
  see that; the identity read can (FPRD AMB-J).

The second is the dangerous one: nothing refuses, nothing times out, and the loss curve
looks healthy.
"""

from __future__ import annotations

import pytest

from core.errors import Refusal
from tests.support.fixtures_hotupdate import Harness
from training.hotupdate import recovery, state
from training.hotupdate.client import WeightsClient
from training.hotupdate.recovery import Interruption, Recovery


class TestWhenMiForgeRestarts:
    def test_a_fresh_client_with_the_recalled_id_can_still_commit(self, harness: Harness) -> None:
        """**MUT-12.** The id is run state (``run_state.py``, and the two columns it needs
        are owed DDL). Here the round trip is through a value that outlives the client, which
        is the property that matters."""
        attachment_id = str(harness.attach()["attachment_id"])
        restarted = WeightsClient(
            settings=harness.client.settings, transport=harness.fake.transport()
        )
        restarted.remember_attachment(attachment_id)
        assert restarted.call("quiesce")["weights_state"] == "clean"
        restarted.call("resume")
        restarted.close()

    def test_without_the_id_every_holder_only_call_is_refused_locally(
        self, harness: Harness
    ) -> None:
        """The symptom MUT-12 produces - made immediate here instead of a step later."""
        harness.attach()
        amnesiac = WeightsClient(
            settings=harness.client.settings, transport=harness.fake.transport()
        )
        for op in ("quiesce", "commit-begin", "commit-end", "save", "heartbeat", "detach"):
            with pytest.raises(Refusal) as caught:
                amnesiac.call(op, {})
            assert caught.value.code == "attachment_id_missing"
        assert harness.fake.call_count("quiesce") == 0
        amnesiac.close()

    def test_path_a_is_unavailable_and_the_operator_clear_is_named(self) -> None:
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
        assert plan.outcome is Recovery.ROLLBACK
        assert "holds_the_computed_update" in plan.unavailable_conditions
        assert "holds_the_attachment_id" in plan.unavailable_conditions
        assert plan.operator_action is not None and "OPERATOR-ONLY" in plan.operator_action
        assert plan.ordering == recovery.POST_CRASH_ORDERING

    def test_re_attaching_after_the_crash_is_refused_until_an_operator_clears_it(
        self, harness: Harness
    ) -> None:
        """And miForge **cannot** clear it: the role gate is miLLM's (R-01.29), and the code
        does not contain the call (R-02.34)."""
        harness.fake.make_stale("trainer:the-dead-worker")
        with pytest.raises(Refusal) as caught:
            harness.client.call("attach")
        # STALE, not held-by-other: the fake now states the holder the way miLLM does
        # (`details.state == "stale"`). Until 2026-10-17 it sent no `state`, so this test
        # asserted `attachment_held_by_other` - whose advice ("wait for it to detach") never
        # clears a dead worker's attachment.
        assert caught.value.code == "attachment_stale_operator_action_required"
        assert caught.value.details["holder"] == "trainer:the-dead-worker"
        with pytest.raises(Refusal) as forbidden:
            harness.client.call("clear-stale")
        assert forbidden.value.code == "weights_op_forbidden"
        assert harness.fake.call_count("clear-stale") == 0


class TestWhenTheServingProcessRestarts:
    def test_clean_with_a_stale_identity_is_caught(self, harness: Harness) -> None:
        """**AC-011.12, MUT-17.** This is the whole reason ``require_clean`` reads an
        identity rather than returning a boolean."""
        harness.attach()
        attached_at = state.require_clean(harness.client)
        # The run steps three times, so miLLM's counter moves.
        for _ in range(3):
            harness.client.call("quiesce")
            harness.client.call("commit-begin")
            harness.client.call("commit-end")
            harness.client.call("resume")
        believed = state.ResidentIdentity(
            model=attached_at.model,
            manifest_sha256=attached_at.manifest_sha256,
            steps_since_load=harness.fake.steps_since_load,
        )
        assert believed.steps_since_load == 3

        harness.fake.restart()
        assert harness.fake.weights_state == "clean", "the premise: a restart begins clean"
        with pytest.raises(Refusal) as caught:
            state.require_clean(harness.client, believed)
        assert caught.value.code == "weights_identity_changed"
        assert "steps_since_load" in caught.value.details["changed"]

    def test_the_handles_are_dead_too_and_the_holder_only_calls_say_so(
        self, harness: Harness
    ) -> None:
        """E-12's other half: a restart loses the attachment, so the id this run holds is no
        longer anybody's. miLLM answers ``409 NOT_ATTACHED`` - *"the attachment was lost or
        detached - most likely the serving process restarted"* - which is a different
        remedy from a lost id and must not be collapsed into one."""
        harness.attach()
        harness.fake.restart()
        with pytest.raises(Refusal) as caught:
            harness.client.call("quiesce")
        assert caught.value.code == "weights_not_attached"
        assert "restarted" in caught.value.message

    def test_the_state_read_alone_would_have_said_everything_was_fine(
        self, harness: Harness
    ) -> None:
        """**MUT-17 stated as an assertion rather than as a comment.** The state half passes
        against a restarted serving process; only the identity half does not."""
        harness.attach()
        harness.fake.steps_since_load = 9
        harness.fake.restart()
        weights_state, identity = state.read(harness.client)
        assert weights_state == "clean", "the state check alone is satisfied"
        assert identity.steps_since_load == 0, "and the identity is what disagrees"
