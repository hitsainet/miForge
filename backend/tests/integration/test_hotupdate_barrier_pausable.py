"""An unpausable barrier is refused at pre-flight, not discovered on a refused call.

miLLM reports `pause_unavailable: "cbm" | "concurrency" | null` on
`GET /api/weights/status` -- no token, no slot -- computed by the same function its
`409 BARRIER_UNAVAILABLE` refusal raises from, with a test on their side asserting status
and the refusal's `details.reason` agree in every case.

**Why pre-flight is the right place.** The condition comes from deployment settings on the
serving side (continuous batching enabled, or more than one request slot), so **a retry
cannot help** and no amount of waiting changes it. Learning it from a refused `quiesce`
means learning it mid-run, after a lease and after the allocation -- the same argument the
neighbouring `weights_not_clean` refusal already makes for itself.

**An absent field is UNKNOWN, never "pausable".** The field is additive and currently lives
on a miLLM branch that is not pushed, so against the published contract it is simply not
there. Treating absent as pausable would be the `getattr` default that always fires
(ADR-018): it would turn an older miLLM into a confident wrong answer. So the three cases
are distinct and each is asserted below -- non-null refuses, null proceeds, absent proceeds
-- and the two "proceeds" are **not** the same fact even though they look alike from
outside.
"""

from __future__ import annotations

import pytest

from core.errors import Refusal
from tests.support.fixtures_hotupdate import Harness
from training.acquisition.base import AcquisitionMode, AcquisitionSpec
from training.hotupdate import state


class TestTheFieldIsReadAsThreeCases:
    def test_a_stated_reason_is_returned(self) -> None:
        for reason in ("cbm", "concurrency"):
            assert state.pause_unavailable({"pause_unavailable": reason}) == reason

    def test_an_explicit_null_is_None(self) -> None:
        assert state.pause_unavailable({"pause_unavailable": None}) is None

    def test_an_ABSENT_field_is_None_and_not_an_error(self) -> None:
        """An older miLLM. Unknown, which this reads as 'no information'."""
        assert state.pause_unavailable({"weights_state": "clean"}) is None

    def test_a_non_dict_body_does_not_raise(self) -> None:
        for body in (None, "nope", [1], 3):
            assert state.pause_unavailable(body) is None

    def test_an_empty_string_is_not_a_reason(self) -> None:
        """`""` would otherwise read as a reason and refuse with an unnameable cause."""
        assert state.pause_unavailable({"pause_unavailable": ""}) is None


class TestPreflightRefusesIt:
    def _preflight(self, harness: Harness):
        from training.hotupdate.attach import AttachAcquirer

        acquirer = AttachAcquirer(transport=harness.fake.transport())
        return acquirer.preflight(AcquisitionSpec(mode=AcquisitionMode.ATTACH))

    def test_it_refuses_BEFORE_any_quiesce(self, harness: Harness) -> None:
        harness.fake.pause_unavailable = "cbm"
        harness.fake.calls.clear()
        with pytest.raises(Refusal) as caught:
            self._preflight(harness)
        assert caught.value.code == "weights_barrier_unavailable"
        assert caught.value.details["reason"] == "cbm"
        # THE CALL COUNT: refusing after a quiesce would also raise, and would have paused a
        # serving process to find out something status already said.
        assert harness.fake.call_count("quiesce") == 0, (
            "pre-flight quiesced the serving process to discover a fact the status read "
            "already carried"
        )
        assert harness.fake.call_count("attach") == 0, "it attached before refusing"

    def test_the_message_says_a_retry_will_not_help(self, harness: Harness) -> None:
        """The operator action is a deployment change, so the message must not read as
        transient -- an operator who retries a deployment setting waits forever."""
        harness.fake.pause_unavailable = "concurrency"
        with pytest.raises(Refusal) as caught:
            self._preflight(harness)
        assert "NOT HELP" in caught.value.message.upper()

    def test_a_null_field_does_NOT_refuse(self, harness: Harness) -> None:
        """The positive control: without this, 'refuses' could be unconditional."""
        harness.fake.pause_unavailable = None
        self._preflight(harness)  # must not raise

    def test_an_ABSENT_field_does_not_refuse_either(self, harness: Harness) -> None:
        """Today's published miLLM. The mode stays usable against it."""
        harness.fake.omit_pause_unavailable = True
        self._preflight(harness)  # must not raise
