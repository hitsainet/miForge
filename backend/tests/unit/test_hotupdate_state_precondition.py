"""``require_clean``: a precondition, and the identity read that makes it honest.

**FR-011.36, FR-011.37; MUT-8 and MUT-17.**

MUT-8 deletes the state read: a step then runs against a ``torn`` set and reports a loss for
a model that never existed at any step. MUT-17 is the subtler one - it keeps the state check
and drops the **identity** comparison, which passes against a serving process that restarted
mid-commit: miLLM's barrier state lives in process memory and a restart begins ``clean``,
so it comes back reporting clean while holding its last *load* rather than this run's last
step. *The failure is silent and arrives as a plausible loss curve.*
"""

from __future__ import annotations

import pytest

from core.errors import Refusal
from tests.support.fixtures_hotupdate import Harness
from training.hotupdate import state


class TestTheStateGate:
    def test_clean_proceeds_and_returns_the_identity(self, harness: Harness) -> None:
        identity = state.require_clean(harness.client)
        assert identity.model == "LFM2-1.2B"
        assert identity.manifest_sha256 == "a" * 64

    @pytest.mark.parametrize("weights_state", ["writing", "torn"])
    def test_a_non_clean_set_refuses_and_carries_the_state(
        self, harness: Harness, weights_state: str
    ) -> None:
        """**MUT-8.** E-4 and E-5: no step begins, and no recovery is decided here."""
        harness.fake.weights_state = weights_state
        with pytest.raises(Refusal) as caught:
            state.require_clean(harness.client)
        assert caught.value.code == "weights_not_clean"
        assert caught.value.details["weights_state"] == weights_state
        assert caught.value.details["resident_model"] == "LFM2-1.2B"

    def test_state_and_identity_come_from_ONE_call(self, harness: Harness) -> None:
        """R-01.20 carries both, and **two calls can straddle a change** - a state read
        saying clean and an identity read taken after a restart describe two moments."""
        harness.fake.calls.clear()
        state.require_clean(harness.client)
        assert harness.fake.ops() == ["status"]

    def test_the_answer_is_never_cached(self, harness: Harness) -> None:
        """*A cached answer about another process's memory is a guess with a timestamp*, and
        a cached ``clean`` is the premise of every wrong thing this mode could do."""
        harness.fake.calls.clear()
        state.require_clean(harness.client)
        state.require_clean(harness.client)
        assert harness.fake.call_count("status") == 2

    def test_a_status_without_a_weights_state_is_refused_rather_than_assumed_clean(
        self, harness: Harness
    ) -> None:
        import httpx

        from training.hotupdate.client import WeightsClient

        transport = httpx.MockTransport(
            lambda request: httpx.Response(200, json={"attached": True})
        )
        client = WeightsClient(settings=harness.client.settings, transport=transport)
        with pytest.raises(Refusal) as caught:
            state.require_clean(client)
        assert caught.value.code == "weights_surface_unreachable"
        assert "assuming it is the premise" in caught.value.message
        client.close()


class TestTheIdentityCrossCheck:
    def test_a_restarted_serving_process_reports_clean_and_is_caught(
        self, harness: Harness
    ) -> None:
        """**AC-011.12, MUT-17, FPRD AMB-J.** The fake's restart is the only way to produce
        this: ``clean`` plus an identity that went back to the last load."""
        expected = state.require_clean(harness.client)
        # ... the run steps, so miLLM's counter moves ...
        harness.fake.steps_since_load = 7
        expected = state.ResidentIdentity(
            model=expected.model, manifest_sha256=expected.manifest_sha256, steps_since_load=7
        )
        harness.fake.restart()
        assert harness.fake.weights_state == "clean", "the premise of this test"
        with pytest.raises(Refusal) as caught:
            state.require_clean(harness.client, expected)
        assert caught.value.code == "weights_identity_changed"
        assert "steps_since_load" in caught.value.details["changed"]

    def test_a_different_resident_model_is_caught(self, harness: Harness) -> None:
        expected = state.require_clean(harness.client)
        harness.fake.resident_model = "something-else"
        with pytest.raises(Refusal) as caught:
            state.require_clean(harness.client, expected)
        assert caught.value.details["changed"] == ["model"]

    def test_a_different_manifest_hash_is_caught(self, harness: Harness) -> None:
        expected = state.require_clean(harness.client)
        harness.fake.manifest_sha256 = "c" * 64
        with pytest.raises(Refusal) as caught:
            state.require_clean(harness.client, expected)
        assert caught.value.details["changed"] == ["manifest_sha256"]

    def test_a_null_manifest_hash_is_a_value_and_not_a_wildcard(self) -> None:
        """R-01.1 returns ``null`` for an **instantiated** resident model: *"not loaded from
        a checkpoint manifest"*, never "unknown". Flattening the distinction is how a
        comparison silently passes - two unknowns compare equal."""
        instantiated = state.ResidentIdentity(model="m", manifest_sha256=None, steps_since_load=0)
        loaded = state.ResidentIdentity(model="m", manifest_sha256="a" * 64, steps_since_load=0)
        assert instantiated.differs_from(loaded) == ["manifest_sha256"]
        assert loaded.differs_from(instantiated) == ["manifest_sha256"]
        assert instantiated.differs_from(instantiated) == []

    def test_a_counter_that_moved_FORWARD_is_not_a_change(self, harness: Harness) -> None:
        """The counter increments on every completed commit, so a run that has stepped
        legitimately disagrees with the value it attached at. Only a *decrease* means a
        different load - and treating any difference as a change would refuse every second
        step, which is the false-positive direction of the failure this exists to catch."""
        attached_at = state.ResidentIdentity(
            model="LFM2-1.2B", manifest_sha256="a" * 64, steps_since_load=0
        )
        harness.fake.steps_since_load = 12
        assert state.require_clean(harness.client, attached_at).steps_since_load == 12
