"""R-02.33's refusals, by name - and the one the map does not cover  (FR-011.27 - FR-011.35).

Each case asserts **the code and the message naming the reason**, never merely that an
exception was raised. The fifth is the load-bearing one: an unmapped code is reported
**verbatim** and the run stops, because *a wrong diagnosis is worse than an unhelpful one -
it ends the investigation* (ADR-018).

**MUT-16's target** is the already-attached row: forcing a detach there would interrupt
another trainer's commit from outside, which is how two writers end up on one parameter set.
"""

from __future__ import annotations

import pytest

from core.errors import Refusal
from tests.support.fake_weights_surface import FakeWeightsSurface
from tests.support.fixtures_hotupdate import Harness
from training.hotupdate import refusals
from training.hotupdate.client import WeightsClient


class TestTheFourDefinedRows:
    def test_already_attached_names_the_holder_and_never_forces_a_detach(
        self, harness: Harness
    ) -> None:
        """R-02.33 row 1 (BRD-01 R-01.2). **MUT-16.**

        A refusal that says "someone else is attached" without saying who cannot be acted
        on - and the holder may be mid-commit, which is why no detach is forced.
        """
        harness.fake.make_stale("trainer:nightly-forge")
        with pytest.raises(Refusal) as caught:
            harness.client.call("attach")
        assert caught.value.code == "attachment_held_by_other"
        assert caught.value.details["holder"] == "trainer:nightly-forge"
        assert "NOT force a detach" in caught.value.message
        assert harness.fake.call_count("detach") == 0, "forcing a detach is R-02.34's defect"
        assert harness.fake.call_count("attach") == 1, "no second attach after this refusal"

    def test_not_attachable_states_the_requirement(self, harness: Harness) -> None:
        """R-02.33 row 2 (BRD-01 R-01.5): the message says FP16 or BF16, so the operator
        learns which file to replace rather than that something failed."""
        harness.fake.dtype = "int8"
        with pytest.raises(Refusal) as caught:
            harness.client.call("attach")
        assert caught.value.code == "resident_model_not_attachable"
        assert "FP16" in caught.value.message and "BF16" in caught.value.message

    def test_stale_names_the_operator_action_and_that_miforge_cannot_perform_it(self) -> None:
        """R-02.33 row 3 (BRD-01 R-01.4, R-01.29 **operator-only**).

        **This test used to PIN THE DEFECT rather than prevent it.** It drove
        ``{"code": "ATTACHMENT_STALE"}`` — a code **miLLM emits nowhere**, verified by
        grepping their whole tree. So it exercised a dead `_MAP` row and reported the
        correct message as reachable, while the path a real stale attachment takes produced
        `attachment_held_by_other`, whose message says *"Stop the other trainer, or wait for
        it to detach"* — advice that can never work, because the holder is gone and only an
        operator's `clear-stale` clears it.

        What miLLM actually sends is `409 ALREADY_ATTACHED` with `details.state` of
        ``"stale"`` or ``"active"`` (their `_already_attached` computes exactly that), so
        that is what this drives now.
        """
        refusal = refusals.interpret(
            409,
            {
                "error": {
                    "code": "ALREADY_ATTACHED",
                    "message": "already attached",
                    "details": {"state": "stale", "label": "trainer@host", "role": "trainer"},
                }
            },
            op="attach",
        )
        assert refusal.code == "attachment_stale_operator_action_required"
        assert refusal.details["miforge_can_perform_it"] is False
        assert "clear-stale" in refusal.details["operator_action"]
        assert "AN OPERATOR MUST CLEAR IT" in refusal.message

    def test_an_ACTIVE_holder_is_still_the_wait_for_it_case(self) -> None:
        """The positive control. Without it, "stale" could be returned unconditionally and
        a live holder would be reported as needing an operator — the opposite wrong advice.
        """
        refusal = refusals.interpret(
            409,
            {
                "error": {
                    "code": "ALREADY_ATTACHED",
                    "message": "already attached",
                    "details": {"state": "active", "label": "trainer@host"},
                }
            },
            op="attach",
        )
        assert refusal.code == "attachment_held_by_other"
        assert "wait for it to detach" in refusal.message

    def test_a_missing_state_stays_the_held_by_other_case(self) -> None:
        """An older miLLM that does not state it. Unknown is not "stale"."""
        refusal = refusals.interpret(
            409, {"error": {"code": "ALREADY_ATTACHED", "message": "already attached"}}, op="attach"
        )
        assert refusal.code == "attachment_held_by_other"

    def test_the_dead_code_is_no_longer_mapped(self) -> None:
        """`ATTACHMENT_STALE` must fall to the unmapped branch, not to a map row.

        A row for a code nobody sends is a second source of truth about the contract, and
        its presence is what hid the real path for as long as it did.
        """
        assert "ATTACHMENT_STALE" not in refusals._MAP
        refusal = refusals.interpret(
            409, {"error": {"code": "ATTACHMENT_STALE", "message": "stale"}}, op="attach"
        )
        assert refusal.code == "weights_refusal_unmapped"

    @pytest.mark.parametrize("upstream", ["WEIGHTS_NOT_CLEAN", "WEIGHTS_TORN"])
    def test_not_clean_and_torn_both_hand_over_to_recovery(self, upstream: str) -> None:
        """R-02.33 row 4 (BRD-01 R-01.22). ``refusals`` decides nothing here - R-02.30 does."""
        refusal = refusals.interpret(
            409, {"error": {"code": upstream, "message": "x"}}, op="quiesce"
        )
        assert refusal.code == "weights_not_clean"
        assert refusal.details["retried"] is False


class TestTheUnmappedCase:
    def test_an_unknown_code_is_reported_verbatim_and_the_run_stops(self) -> None:
        """E-15, FR-011.32. **Never mapped onto the nearest familiar case.**"""
        refusal = refusals.interpret(
            418,
            {"error": {"code": "WEIGHTS_ON_FIRE", "message": "the card is unhappy"}},
            op="commit-begin",
        )
        assert refusal.code == "weights_refusal_unmapped"
        assert refusal.details["upstream_code"] == "WEIGHTS_ON_FIRE"
        assert refusal.details["upstream_message"] == "the card is unhappy"
        assert refusal.details["status"] == 418
        assert "WEIGHTS_ON_FIRE" in refusal.message

    def test_a_refusal_with_no_code_at_all_does_not_invent_one(self) -> None:
        refusal = refusals.interpret(500, {"oops": True}, op="save")
        assert refusal.code == "weights_refusal_unmapped"
        assert refusal.details["upstream_code"] is None

    def test_a_body_that_is_not_a_mapping_is_still_reported(self) -> None:
        """miForge must not refuse to *understand* a refusal because the wrapper differed -
        and must not invent a code either."""
        refusal = refusals.interpret(502, "<html>bad gateway</html>", op="status")
        assert refusal.code == "weights_refusal_unmapped"

    def test_ATTACHMENT_NOT_STALE_is_unmapped_on_purpose(self) -> None:
        """BRD-01 R-01.29's refusal belongs to a call **miForge never makes**. Mapping it
        would be miForge claiming to understand the outcome of an operator-only action it
        cannot take."""
        refusal = refusals.interpret(
            409, {"error": {"code": "ATTACHMENT_NOT_STALE"}}, op="clear-stale"
        )
        assert refusal.code == "weights_refusal_unmapped"


class TestNoUpstreamCodeIsReExportedAsMiForges:
    def test_every_mapped_code_is_in_miforges_own_namespace(self) -> None:
        """Re-exporting one would make miForge's API depend on another repository's string
        table (TID §2.3)."""
        for miforge_code in refusals._MAP.values():
            assert miforge_code.islower(), miforge_code
            assert miforge_code.startswith(("attachment_", "weights_", "resident_")), miforge_code

    def test_the_mapped_codes_all_have_a_message(self) -> None:
        """A code with no message is a refusal that names nothing - and ``Refusal`` itself
        refuses to be built without one, so this would be a 500 at the worst moment."""
        for miforge_code in set(refusals._MAP.values()) - {"weights_refusal_unmapped"}:
            assert miforge_code in refusals._MESSAGES


class TestNoRefusalIsRetried:
    def test_a_refused_call_is_made_exactly_once(
        self, hotupdate_env: None, surface: FakeWeightsSurface
    ) -> None:
        from training.hotupdate import settings as settings_module

        surface.refuse_always["quiesce"] = (409, "WEIGHTS_NOT_CLEAN")
        client = WeightsClient(settings=settings_module.load(), transport=surface.transport())
        client.remember_attachment("x")
        surface.attachment_id = "x"
        with pytest.raises(Refusal):
            client.call("quiesce")
        assert surface.call_count("quiesce") == 1
        client.close()

    def test_the_holder_read_for_an_already_attached_refusal_is_the_only_extra_call(
        self, harness: Harness
    ) -> None:
        """One extra ``status`` read, and **only** on the row that must name the holder. A
        status read on every refusal would make a zero-call-count assertion fail for a
        reason that has nothing to do with what it tests (AC-011.18)."""
        harness.fake.refuse_always["save"] = (409, "WEIGHTS_NOT_CLEAN")
        harness.attach()
        harness.fake.calls.clear()
        with pytest.raises(Refusal):
            harness.client.call("save", {})
        assert harness.fake.call_count("status") == 0


class TestMUT16SurvivedAndThisIsWhatItSurvived:
    """**MUT-16's regression test, written because MUT-16 SURVIVED on 2026-10-14.**

    TDD §10.4's MUT-16 is *"``refusals`` - force a detach on ``ALREADY_ATTACHED``"*, and
    ``test_already_attached_names_the_holder_and_never_forces_a_detach`` above asserts
    ``call_count("detach") == 0`` for it. Applied as a forced
    ``self.call("detach")`` on the refusal path in ``client._interpret``, the F11 suite
    stayed at **430 passed**.

    **Why it survived, which is the transferable part.** That test attaches *nothing* - it
    goes straight to a refused ``attach`` from a cold client. So the client holds no
    attachment id, and ``detach`` is **holder-only** (BRD-01 R-01.31), which means
    ``identity.headers_for`` refuses it *locally* before a byte leaves the process. The zero
    in the assertion is therefore produced by **the scenario**, not by the behaviour: it
    would read zero against a build that forces a detach on every refusal it ever sees.
    *A fixture that cannot express the defect makes its assertion vacuous* - here through
    the client's own state rather than through a double.

    And the scenario the defect actually lives in is the common one. A run attaches, the
    serving process is restarted or the attachment goes stale, miForge re-attaches - and
    *now* it is the holder, so a forced detach lands on the wire and takes down whatever
    attachment miLLM currently has. Which is R-02.34's whole prohibition: the holder may be
    mid-commit.

    So this class pins the same prohibition **from a client that is the holder**, which is
    the only state in which the forced call is reachable.
    """

    def test_an_already_attached_refusal_forces_no_detach_EVEN_WHEN_WE_ARE_THE_HOLDER(
        self, harness: Harness
    ) -> None:
        """R-02.33 row 1 / R-02.34. **The call count is the assertion**, and it is taken
        from a client that *could* have sent the call."""
        harness.attach()
        assert harness.client.attachment_id, "the premise: this client IS the holder"
        harness.fake.make_stale("trainer:someone-else")
        harness.fake.calls.clear()
        with pytest.raises(Refusal) as caught:
            harness.client.call("attach")
        assert caught.value.code in {
            "attachment_held_by_other",
            "attachment_stale_operator_action_required",
        }
        assert harness.fake.call_count("detach") == 0, (
            "forcing a detach is R-02.34's defect: the holder may be mid-commit, and "
            "miForge never clears another process's attachment on its own initiative"
        )
        assert harness.fake.call_count("attach") == 1, "and no second attach after it"

    def test_the_premise_is_real_a_detach_from_this_client_WOULD_reach_the_wire(
        self, harness: Harness
    ) -> None:
        """**The positive control for the test above, and the reason it exists.**

        Without this, the new assertion could be zero for the same reason the old one was -
        a locally refused call - and the regression test would be as vacuous as the thing it
        replaces. This proves the client in this state *can* send a ``detach``, so the zero
        above is the behaviour's and not the scenario's.
        """
        harness.attach()
        harness.fake.calls.clear()
        harness.client.call("detach")
        assert harness.fake.call_count("detach") == 1

    def test_a_cold_client_cannot_send_one_at_all_which_is_why_MUT_16_hid(
        self, hotupdate_env: None, surface: FakeWeightsSurface
    ) -> None:
        """The mechanism, asserted rather than described: from a client holding no
        attachment id a ``detach`` never reaches the fake, so a count of zero proves
        nothing about whether anything tried."""
        from training.hotupdate import settings as settings_module

        client = WeightsClient(settings=settings_module.load(), transport=surface.transport())
        try:
            assert client.attachment_id is None
            with pytest.raises(Refusal) as caught:
                client.call("detach")
            assert caught.value.code == "attachment_id_missing"
            assert surface.call_count("detach") == 0, (
                "refused locally, never sent - which is exactly how a zero call count can "
                "be true of a build that forces a detach"
            )
        finally:
            client.close()
