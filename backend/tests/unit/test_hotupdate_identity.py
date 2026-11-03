"""``401``/``403``: the run stops, the identity is named, **nothing is retried**  (R-02.32).

**Why the heartbeat gets its own case.** A heartbeat refused for a missing token is
*indistinguishable from a dead trainer*: miLLM marks the attachment stale (BRD-01 R-01.4)
and the run dies one interval later looking like a crash - the right failure arriving as
the wrong diagnosis. BRD-01 R-01.17 gates it for exactly that reason, and R-02.32's own list
omitted it until miLLM's Feature 031 review found it. **MUT-5 reverts that fix**, which is
why it is this feature's negative control for the previous review round's own work: if
reverting it does not fail loudly, that round produced an unpinned fix.

**"No retry" is asserted as a call count of exactly one.** A test that only checks the run
stopped passes against a stop after three attempts - and three rejected heartbeats inside
one staleness window is the difference between a loud identity failure and a stale
attachment.
"""

from __future__ import annotations

import pytest

from core.errors import Refusal
from tests.support.fixtures_hotupdate import Harness
from training.hotupdate import identity
from training.hotupdate.client import OPS
from training.hotupdate.heartbeat import Heartbeat

GATED_OPS = sorted(set(OPS) & identity.GATED)


class TestEveryGatedCallStopsTheRunOnARejectedIdentity:
    @pytest.mark.parametrize("op", GATED_OPS)
    def test_a_401_is_terminal_and_is_not_retried(self, harness: Harness, op: str) -> None:
        if op in identity.HOLDER_ONLY:
            harness.attach()
            harness.fake.calls.clear()
        harness.fake.valid_token = "a-different-token"
        with pytest.raises(Refusal) as caught:
            harness.client.call(op, {})
        assert caught.value.code == "weights_identity_rejected"
        assert caught.value.details["retried"] is False
        assert harness.fake.call_count(op) == 1, "a rejected identity must not be retried"

    @pytest.mark.parametrize("op", GATED_OPS)
    def test_the_message_names_the_identity_and_the_remedy(self, harness: Harness, op: str) -> None:
        """ADR-021: messages say **what to do next**. Not "authentication failed", which
        reads as transient and invites exactly the retry this forbids."""
        if op in identity.HOLDER_ONLY:
            harness.attach()
        harness.fake.valid_token = "wrong"
        with pytest.raises(Refusal) as caught:
            harness.client.call(op, {})
        message = caught.value.message
        assert "MILLM_TRAINER_TOKEN" in message
        assert "NOT retried" in message
        assert "credential" in message


class TestTheHeartbeatSpecifically:
    def test_a_rejected_heartbeat_stops_the_run_rather_than_going_quiet(
        self, harness: Harness
    ) -> None:
        """AC-011.7. The failure must arrive as an identity failure **now**, not as a stale
        attachment one interval later."""
        harness.attach()
        stopped: list[Refusal] = []
        beat = Heartbeat(
            client=harness.client,
            interval_seconds=1.0,
            on_identity_rejected=stopped.append,
            sleep=lambda _seconds: None,
        )
        harness.fake.valid_token = "rotated-away"
        assert beat.beat_once() is None
        assert [exc.code for exc in stopped] == ["weights_identity_rejected"]
        assert beat.stopped is True, "the heartbeat must not keep beating into a dead run"
        assert beat.beats == 0

    def test_it_is_not_retried_inside_one_interval(self, harness: Harness) -> None:
        harness.attach()
        harness.fake.calls.clear()
        harness.fake.valid_token = "rotated-away"
        beat = Heartbeat(client=harness.client, interval_seconds=1.0, sleep=lambda _s: None)
        beat.beat_once()
        assert harness.fake.call_count("heartbeat") == 1

    def test_the_heartbeat_carries_both_headers_like_any_other_holder_only_call(
        self, harness: Harness
    ) -> None:
        """*"Through ``client.call``, with no special path"* (TID §3.10 hint 3)."""
        attachment_id = str(harness.attach()["attachment_id"])
        Heartbeat(client=harness.client, interval_seconds=1.0).beat_once()
        headers = harness.fake.headers_of("heartbeat")[-1]
        assert headers["authorization"] == "Bearer trainer-token"
        assert headers["x-millm-attachment"] == attachment_id


class TestTheHolderRefusalIsNotACredentialRefusal:
    def test_a_403_NOT_ATTACHMENT_HOLDER_is_reported_as_a_holder_problem(
        self, harness: Harness
    ) -> None:
        """R-01.31 uses ``403`` for *a valid token that is not the holder*. Mapping it to
        "fix your credential" would send the operator to the wrong file - and the token is
        fine, which they would discover after changing it."""
        harness.attach()
        harness.client.attachment_id = "someone-elses-id"
        with pytest.raises(Refusal) as caught:
            harness.client.call("quiesce")
        assert caught.value.code == "attachment_not_holder"
        assert caught.value.details["upstream_code"] == "NOT_ATTACHMENT_HOLDER"


class TestTheSecretsNeverAppearInARefusal:
    def test_the_token_is_absent_from_the_identity_refusal(self, harness: Harness) -> None:
        harness.fake.valid_token = "wrong"
        with pytest.raises(Refusal) as caught:
            harness.client.call("attach")
        rendered = f"{caught.value.message} {caught.value.details}"
        assert "trainer-token" not in rendered
        assert "miforge-nightly" not in rendered, "the audit label is one of the three"
        # A digest reference, which is how a post-mortem distinguishes two runs' labels
        # without either label appearing anywhere.
        assert caught.value.details["audit_label_ref"] == identity.digest_ref("miforge-nightly")

    def test_digest_ref_of_absent_is_absent(self) -> None:
        """A reference to a secret that is not set would read as a secret that is."""
        assert identity.digest_ref(None) is None
