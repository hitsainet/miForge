"""The single call site: the two headers, **payload and call count**  (FR-011.21-23).

ADR-016: *"was called" passes against a call sending the wrong arguments.* So every
assertion here is on what the request actually carried and on how many requests there were -
never on "the client was used".

**The endpoint diff is in this file deliberately.** Enumerating every path F11 can send and
diffing it against the set BRD-01 §4 defines is the check that caught the missing load call
(*"a section title promising a capability is not a requirement defining one"*) and the
missing ``heartbeat``. Doing it in a test means a future op added without a citation is red.
"""

from __future__ import annotations

import contextlib

import httpx
import pytest

from core.errors import Refusal
from tests.support.fixtures_hotupdate import Harness
from training.hotupdate import identity
from training.hotupdate.client import OPS, WeightsClient

#: Every path BRD-01 §4 defines, read out of that document. **Fifteen**, and the fifteenth
#: is the heartbeat (R-01.30, added upstream 2026-10-13).
BRD_01_ENDPOINTS: frozenset[str] = frozenset(
    {
        "/api/weights/attach",
        "/api/weights/instantiate",
        "/api/weights/detach",
        "/api/weights/quiesce",
        "/api/weights/resume",
        "/api/weights/commit-begin",
        "/api/weights/commit-end",
        "/api/weights/save",
        "/api/weights/load",
        "/api/weights/lineage",
        "/api/weights/adapter/load",
        "/api/weights/adapter/unload",
        "/api/weights/clear-stale",
        "/api/weights/status",
        "/api/weights/heartbeat",
    }
)


class TestTheEndpointSetIsBRD01s:
    def test_every_op_this_client_can_send_is_an_endpoint_BRD_01_defines(self) -> None:
        """No invented path. A guessed endpoint is a request against a contract nobody wrote."""
        unknown = {op: path for op, (_, path, _) in OPS.items() if path not in BRD_01_ENDPOINTS}
        assert unknown == {}, f"these ops are not in BRD-01 §4: {unknown}"

    def test_the_endpoints_F11_does_not_call_are_the_ones_it_says_it_does_not(self) -> None:
        """The diff in the other direction, which is the half that finds a *missing* call.

        BRD-01 defines fifteen; F11 sends eleven and declines four **by name and with a
        reason**. An endpoint in neither set would be one nobody decided about.
        """
        from training.hotupdate.client import DELIBERATELY_ABSENT

        sent = {path for _, path, _ in OPS.values()}
        declined = {f"/api/weights/{op}" for op in DELIBERATELY_ABSENT}
        assert (
            sent | declined == BRD_01_ENDPOINTS
        ), f"unaccounted for: {BRD_01_ENDPOINTS - (sent | declined)}"

    def test_every_op_cites_a_requirement(self) -> None:
        for op, (_, _, requirement) in OPS.items():
            assert requirement.startswith("R-01."), f"{op} cites {requirement!r}"

    def test_the_gated_set_is_R_01_17s_and_includes_the_heartbeat(self) -> None:
        """**MUT-5's target.** R-02.32's list omitted ``heartbeat`` until miLLM's Feature
        031 review found it, and R-01.17 was amended the same day to gate it."""
        assert "heartbeat" in identity.GATED
        assert identity.GATED == frozenset(
            {
                "attach",
                "detach",
                "heartbeat",
                "quiesce",
                "commit-begin",
                "commit-end",
                "resume",
                "save",
                "load",
                "instantiate",
                "adapter/load",
                "adapter/unload",
                "clear-stale",
            }
        )

    def test_the_holder_only_set_is_R_01_31s_seven(self) -> None:
        """Seven, and the complement is recorded too (``load``/``instantiate`` read it when
        present; ``adapter/*``, ``clear-stale`` and ``/v1/*`` ignore it)."""
        assert identity.HOLDER_ONLY == frozenset(
            {"detach", "heartbeat", "quiesce", "resume", "commit-begin", "commit-end", "save"}
        )
        assert len(identity.HOLDER_ONLY) == 7
        assert not identity.HOLDER_ONLY & identity.READS_HEADER_WHEN_PRESENT
        assert not identity.HOLDER_ONLY & identity.IGNORES_HEADER


class TestTheCredentialAndTheHeaderAreOnEveryCallThatNeedsThem:
    @pytest.mark.parametrize("op", sorted(set(OPS) & identity.GATED))
    def test_every_gated_op_carries_the_bearer_credential(self, harness: Harness, op: str) -> None:
        harness.attach()
        before = harness.fake.call_count()
        with contextlib.suppress(Refusal):
            harness.client.call(op, {} if op != "load" else {"checkpoint": "x"})
        sent = harness.fake.headers_of(op)
        assert sent, f"{op} did not reach the surface"
        assert sent[-1]["authorization"] == "Bearer trainer-token"
        # The call count, not merely that it happened: a header test that counted nothing
        # would pass against a client that sent the call twice with one header right.
        assert harness.fake.call_count() == before + 1

    @pytest.mark.parametrize("op", sorted(identity.HOLDER_ONLY & set(OPS)))
    def test_every_holder_only_op_carries_the_attachment_header(
        self, harness: Harness, op: str
    ) -> None:
        """**MUT-4's target.** Dropping one entry loses that call and, with it, the commit."""
        attachment_id = str(harness.attach()["attachment_id"])
        with contextlib.suppress(Refusal):
            harness.client.call(op, {})
        sent = harness.fake.headers_of(op)
        assert sent[-1]["x-millm-attachment"] == attachment_id

    def test_a_non_holder_op_does_not_carry_the_attachment_header(self, harness: Harness) -> None:
        """R-01.31's complement. ``status`` ignores it, and sending it there is not what is
        asserted - the point is that the table decides, so a test can see the difference."""
        harness.attach()
        harness.client.call("status")
        assert "x-millm-attachment" not in harness.fake.headers_of("status")[-1]

    def test_no_audit_label_header_is_invented(self, harness: Harness) -> None:
        """The label travels **inside the token** (R-02.32 as amended); BRD-01's wire detail
        names exactly two headers. A third of miForge's own invention would be a field no
        requirement defines - the restatement that produced three of this project's
        cross-document defects."""
        harness.attach()
        headers = harness.fake.headers_of("attach")[-1]
        invented = [key for key in headers if "audit" in key or "label" in key]
        assert invented == [], f"invented header(s): {invented}"


class TestRefusingLocallyBeforeTheCall:
    def test_a_holder_only_op_with_no_attachment_id_is_refused_before_any_http_call(
        self, harness: Harness
    ) -> None:
        """FR-011.22, E-8. Refused **locally**: a round trip wasted and a vaguer cause."""
        with pytest.raises(Refusal) as caught:
            harness.client.call("quiesce")
        assert caught.value.code == "attachment_id_missing"
        assert harness.fake.call_count() == 0, "a refused-locally call must not be sent"

    def test_a_gated_op_with_no_token_is_refused_before_any_http_call(
        self, monkeypatch: pytest.MonkeyPatch, surface: object
    ) -> None:
        from tests.support.fixtures_hotupdate import configure
        from training.hotupdate import settings as settings_module

        configure(monkeypatch, MILLM_TRAINER_TOKEN="")
        client = WeightsClient(
            settings=settings_module.load(), transport=surface.transport()  # type: ignore[attr-defined]
        )
        with pytest.raises(Refusal) as caught:
            client.call("attach")
        assert caught.value.code == "weights_identity_incomplete"
        assert caught.value.details["missing"] == "millm_trainer_token"
        assert surface.call_count() == 0  # type: ignore[attr-defined]

    def test_a_gated_op_with_no_audit_label_is_refused(
        self, monkeypatch: pytest.MonkeyPatch, surface: object
    ) -> None:
        """*"Not optional decoration"* (R-02.32): a token without a label is a weights change
        nobody can attribute."""
        from tests.support.fixtures_hotupdate import configure
        from training.hotupdate import settings as settings_module

        configure(monkeypatch, MILLM_AUDIT_LABEL="")
        client = WeightsClient(
            settings=settings_module.load(), transport=surface.transport()  # type: ignore[attr-defined]
        )
        with pytest.raises(Refusal) as caught:
            client.call("attach")
        assert caught.value.details["missing"] == "millm_audit_label"


class TestClearStaleIsUnreachableThroughThisClient:
    def test_calling_it_is_a_programming_error_and_nothing_is_sent(self, harness: Harness) -> None:
        """FR-011.34's innermost layer; **MUT-7** adds such a call elsewhere and the source
        scan in ``test_no_clear_stale_call.py`` is what catches that one."""
        with pytest.raises(Refusal) as caught:
            harness.client.call("clear-stale")
        assert caught.value.code == "weights_op_forbidden"
        assert harness.fake.call_count() == 0

    def test_an_unknown_op_is_refused_rather_than_guessed_at(self, harness: Harness) -> None:
        with pytest.raises(Refusal) as caught:
            harness.client.call("quiesce-all")
        assert caught.value.code == "weights_op_unknown"
        assert harness.fake.call_count() == 0


class TestTheBoundaryIsValidated:
    def test_a_body_that_is_not_an_object_is_a_named_refusal(self, harness: Harness) -> None:
        """ADR-018: no half-read dict, no ``getattr`` default."""
        transport = httpx.MockTransport(lambda request: httpx.Response(200, text="not json"))
        client = WeightsClient(settings=harness.client.settings, transport=transport)
        with pytest.raises(Refusal) as caught:
            client.call("status")
        assert caught.value.code == "weights_surface_unreachable"
        client.close()

    def test_a_transport_failure_is_named_and_not_retried(self, harness: Harness) -> None:
        """A blind retry of a weights call can double a write, so the transport failure is
        reported rather than retried: at that point it is unknown whether miLLM accepted it."""
        harness.fake.stall_next("status")
        with pytest.raises(Refusal) as caught:
            harness.client.call("status")
        assert caught.value.code == "weights_surface_unreachable"
        assert harness.fake.call_count("status") == 1, "a refusal must not be retried"

    def test_commit_end_gets_the_longest_timeout(self, harness: Harness) -> None:
        """A late ``commit-end`` is accepted upstream (R-01.23a) and beats abandoning the
        bracket, which leaves the set torn and serving refused."""
        cfg = harness.client.settings
        assert cfg.commit_end_timeout_seconds > cfg.request_timeout_seconds

    def test_one_connection_serves_the_whole_attachment(self, harness: Harness) -> None:
        """Four calls per step at a fresh connection each is latency added to *held serving
        time* for no reason (TID §13)."""
        first = harness.client._client
        harness.attach()
        harness.client.call("status")
        assert harness.client._client is first
