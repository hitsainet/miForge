"""One test per row of the client's status table  (F10 task 2.6; FR-010.6, AC-010.15).

miDataworks documents its own outbound status mapping as a table and records that *"each row
is a test in ``test_endpoint_caller.py``"*. This is the mirror of that, and the reason is the
same: **a mapping expressed only as a chain of ``if``s cannot be audited against the design.**
A reader comparing this file with the table in ``clients/midataworks.py``'s docstring can see
in one pass whether the two agree.

The two rows that are policy rather than mechanism:

* **401/403 is never retried.** A rejected credential does not become valid by repetition
  (ADR-020). Asserted on the **call count**, because "it raised" is also true of a mapping
  that retried three times first.
* **``unreachable`` is not ``not_configured``.** Different audience, different status,
  different retry policy. Collapsing them is the single most tempting simplification in this
  feature and the one that hides a misconfiguration behind a network message.

``sleep`` is injected throughout. A test that really sleeps through a backoff is a test
nobody runs.
"""

from __future__ import annotations

import httpx
import pytest

from clients.midataworks import AGENT_HEADER, FORBIDDEN_HEADERS, MiDataworksClient
from clients.midataworks_errors import (
    MiDataworksRefused,
    MiDataworksUnauthorized,
    MiDataworksUnreachable,
)
from core.config import reset_settings
from core.errors import Refusal


@pytest.fixture(autouse=True)
def configured(monkeypatch):
    monkeypatch.setenv("MIDATAWORKS_BASE_URL", "http://dw.invalid:8000")
    reset_settings()
    yield
    reset_settings()


class Recorder:
    """A transport that answers from a script and records what it was sent."""

    def __init__(self, *responses: httpx.Response | Exception) -> None:
        self.script = list(responses)
        self.requests: list[httpx.Request] = []

    def transport(self) -> httpx.MockTransport:
        def handle(request: httpx.Request) -> httpx.Response:
            self.requests.append(request)
            answer = self.script.pop(0) if self.script else httpx.Response(200, json={})
            if isinstance(answer, Exception):
                raise answer
            return answer

        return httpx.MockTransport(handle)


def client(recorder: Recorder, **kwargs) -> MiDataworksClient:
    waits: list[float] = kwargs.pop("waits", [])
    return MiDataworksClient(
        action="testing the status mapping",
        capability="part_import_http",
        transport=recorder.transport(),
        sleep=waits.append,
        **kwargs,
    )


class TestTheTableRowByRow:
    def test_2xx_returns_the_parsed_body(self):
        recorder = Recorder(httpx.Response(200, json={"items": [1, 2]}))
        with client(recorder) as api:
            assert api.list_exports() == []  # no mappings in `items`
        assert len(recorder.requests) == 1

    def test_a_5xx_is_retried_then_refused_as_unreachable(self):
        waits: list[float] = []
        recorder = Recorder(*(httpx.Response(503, json={}) for _ in range(4)))
        with (
            pytest.raises(MiDataworksUnreachable) as caught,
            client(recorder, waits=waits, retry_max=3) as api,
        ):
            api.get_queue("rq1")
        assert caught.value.code == "midataworks_unreachable"
        assert len(recorder.requests) == 4, "one attempt plus three retries"
        assert waits == [1.0, 2.0, 4.0], "exponential backoff, not a flat sleep"

    def test_a_transport_error_is_retried_then_refused_as_unreachable(self):
        waits: list[float] = []
        recorder = Recorder(*(httpx.ConnectError("refused") for _ in range(3)))
        with (
            pytest.raises(MiDataworksUnreachable),
            client(recorder, waits=waits, retry_max=2) as api,
        ):
            api.get_queue("rq1")
        assert len(recorder.requests) == 3
        assert waits == [1.0, 2.0]

    def test_retry_after_is_honoured_but_capped(self):
        """A server asking for an hour must not park a step loop for an hour."""
        waits: list[float] = []
        recorder = Recorder(
            httpx.Response(503, json={}, headers={"Retry-After": "3600"}),
            httpx.Response(200, json={"items": []}),
        )
        with client(recorder, waits=waits, retry_max=1) as api:
            api.list_exports()
        assert waits == [30.0], "capped at MAX_BACKOFF_SECONDS"

    def test_a_403_naming_no_code_is_a_credential_rejection(self):
        """A bare 403 - a reverse proxy, a gateway, a gate miDataworks has just grown - has
        no better reading than "the credential was rejected"."""
        recorder = Recorder(httpx.Response(403, text="Forbidden"))
        with pytest.raises(MiDataworksUnauthorized), client(recorder) as api:
            api.get_queue("rq1")

    def test_a_403_naming_its_own_code_is_a_declined_action(self):
        """``ORIGIN_APP_MISMATCH``: miForge posted to a queue another application created.
        **Not a credential problem**, and reporting it as one sends the reader looking for a
        token that does not exist on this surface (FR-010.8)."""
        recorder = Recorder(
            httpx.Response(
                403,
                json={
                    "error": {
                        "code": "ORIGIN_APP_MISMATCH",
                        "message": "This queue belongs to someone-else.",
                        "details": {},
                    }
                },
            )
        )
        with pytest.raises(MiDataworksRefused) as caught, client(recorder) as api:
            api.get_queue("rq1")
        assert caught.value.details["midataworks_code"] == "ORIGIN_APP_MISMATCH"
        assert len(recorder.requests) == 1, "a 4xx must not be retried"

    @pytest.mark.parametrize("status", [401, 403])
    def test_401_and_403_are_never_retried(self, status: int):
        """**The row that is a policy.** Asserted on the call count: "it raised" is also
        true of a mapping that retried three times first."""
        waits: list[float] = []
        recorder = Recorder(httpx.Response(status, text="nope"))
        with (
            pytest.raises(MiDataworksUnauthorized) as caught,
            client(recorder, waits=waits, retry_max=3) as api,
        ):
            api.get_queue("rq1")
        assert caught.value.code == "midataworks_unauthorized"
        assert len(recorder.requests) == 1, "a rejected credential was retried"
        assert waits == [], "a non-retryable failure must not sleep"
        assert caught.value.details["retried"] is False
        assert caught.value.details["agent"] == "agent:miforge"

    def test_another_4xx_carries_midataworks_own_code(self):
        """Their code is **carried, not translated**: a translation table would be a second
        description of someone else's error vocabulary."""
        # A 422, not a 403: 403 belongs to the unauthorized row of the table, and reusing
        # it here would test that row twice and this one not at all.
        recorder = Recorder(
            httpx.Response(
                422,
                json={
                    "error": {
                        "code": "QUEUE_KIND_INVALID",
                        "message": "Candidates go to an external queue only.",
                        "details": {"kind": "audit"},
                    }
                },
            )
        )
        with pytest.raises(MiDataworksRefused) as caught, client(recorder) as api:
            api.get_queue("rq1")
        assert caught.value.code == "midataworks_refused"
        assert caught.value.details["midataworks_code"] == "QUEUE_KIND_INVALID"
        assert caught.value.details["midataworks_details"] == {"kind": "audit"}
        assert len(recorder.requests) == 1, "a 4xx must not be retried"

    def test_a_4xx_with_a_body_that_is_not_json_still_refuses_usefully(self):
        """A refusal that raised while reporting a refusal is the worst outcome available."""
        recorder = Recorder(httpx.Response(422, text="<html>nginx</html>"))
        with pytest.raises(MiDataworksRefused) as caught, client(recorder) as api:
            api.get_queue("rq1")
        assert caught.value.details["midataworks_code"] is None
        assert "nginx" in caught.value.details["midataworks_message"]


class TestUnreachableIsNotUnconfigured:
    def test_the_two_codes_are_distinct(self, monkeypatch):
        """AC-010.15. The mutation that collapses them must make this red."""
        recorder = Recorder(httpx.Response(503, json={}))
        with (
            pytest.raises(MiDataworksUnreachable) as unreachable,
            client(recorder, retry_max=0) as api,
        ):
            api.get_queue("rq1")

        monkeypatch.delenv("MIDATAWORKS_BASE_URL", raising=False)
        reset_settings()
        with pytest.raises(Refusal) as not_configured:
            MiDataworksClient(action="x", capability="part_import_http")

        assert unreachable.value.code == "midataworks_unreachable"
        assert not_configured.value.code == "midataworks_not_configured"
        assert unreachable.value.code != not_configured.value.code

    def test_the_statuses_differ_too(self):
        """The app's ``status_for`` must not render both the same way, or a client that
        branches on the status cannot tell a decision from a fault."""
        from api.app import status_for

        assert status_for("midataworks_unreachable") != status_for("midataworks_not_configured")


class TestHeaders:
    def test_every_request_carries_the_agent_header(self):
        recorder = Recorder(httpx.Response(200, json={"items": []}))
        with client(recorder) as api:
            api.list_exports()
        assert recorder.requests[0].headers[AGENT_HEADER] == "agent:miforge"

    def test_no_authorization_header_is_ever_sent(self):
        """**Three surfaces, three credentials, none shared.** The credential that would be
        reached for is ``MILLM_TRAINER_TOKEN``, which belongs to a different service -
        sending a sibling's credential to the wrong sibling is a leak, not a typo."""
        recorder = Recorder(httpx.Response(200, json={"items": []}))
        with client(recorder) as api:
            api.list_exports()
        sent = {name.lower() for name in recorder.requests[0].headers}
        assert not (sent & FORBIDDEN_HEADERS), sorted(sent & FORBIDDEN_HEADERS)

    def test_read_back_carries_it_too_although_it_is_not_required(self):
        """One call site means one set of headers and no per-route special cases."""
        recorder = Recorder(httpx.Response(200, json={"items": []}))
        with client(recorder) as api:
            api.read_decisions("rq1")
        assert recorder.requests[0].headers[AGENT_HEADER] == "agent:miforge"


class TestCooperativeWaiting:
    def test_on_wait_is_called_before_each_backoff(self):
        seen: list[tuple[float, str]] = []
        recorder = Recorder(httpx.Response(503, json={}), httpx.Response(200, json={"items": []}))
        with MiDataworksClient(
            action="x",
            capability="part_import_http",
            transport=recorder.transport(),
            sleep=lambda _s: None,
            retry_max=1,
            on_wait=lambda seconds, reason: seen.append((seconds, reason)),
        ) as api:
            api.list_exports()
        assert seen == [(1.0, "HTTP 503")]

    def test_a_raising_on_wait_abandons_the_call(self):
        """**This is how a cancelled run interrupts a backoff** under the solo pool: there
        is no pool child to signal, so something the run controls has to be consulted."""

        class Cancelled(BaseException):
            pass

        def refuse_to_wait(_seconds: float, _reason: str) -> None:
            raise Cancelled()

        recorder = Recorder(httpx.Response(503, json={}), httpx.Response(200, json={}))
        with (
            pytest.raises(Cancelled),
            MiDataworksClient(
                action="x",
                capability="part_import_http",
                transport=recorder.transport(),
                sleep=lambda _s: None,
                retry_max=3,
                on_wait=refuse_to_wait,
            ) as api,
        ):
            api.list_exports()
        assert len(recorder.requests) == 1, "the second attempt was made despite cancellation"


class TestPingReportsRatherThanRaises:
    def test_an_unreachable_sibling_does_not_fail_the_probe(self):
        """A probe that threw would let an optional sibling fail the health route itself."""
        recorder = Recorder(httpx.ConnectError("refused"))
        with client(recorder, retry_max=0) as api:
            assert api.ping() is False

    def test_a_reachable_sibling_pings_true(self):
        recorder = Recorder(httpx.Response(200, json={"status": "ok"}))
        with client(recorder) as api:
            assert api.ping() is True

    def test_it_probes_the_versionless_health_path(self):
        """``/api/health``, not ``/api/v1/health``: their own health module's docstring says
        *"Mounted at ``/api``, not ``/api/v1``"*, for the same reason miForge pins it there."""
        recorder = Recorder(httpx.Response(200, json={"status": "ok"}))
        with client(recorder) as api:
            api.ping()
        assert recorder.requests[0].url.path == "/api/health"


class TestBatchSizeIsRefusedLocally:
    def test_an_oversized_batch_is_not_sent(self):
        """A 413 round trip costs the whole payload's upload for an answer miForge had."""
        recorder = Recorder(httpx.Response(201, json={"items": []}))
        with pytest.raises(Refusal) as caught, client(recorder) as api:
            api.post_candidates("rq1", [{"external_id": str(i)} for i in range(501)])
        assert caught.value.code == "review_batch_too_large"
        assert recorder.requests == [], "the oversized batch was sent to be refused"

    def test_an_empty_batch_sends_nothing_at_all(self):
        """Zero requests, not one empty request."""
        recorder = Recorder(httpx.Response(201, json={"items": []}))
        with client(recorder) as api:
            assert api.post_candidates("rq1", []) == []
        assert recorder.requests == []
