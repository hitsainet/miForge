"""F10 8.6's write measurement, against miDataworks' real D2 close semantics (the fake).

The live run writes to a real miDataworks, so its one safety property - **every queue it
opens is closed, and the close is verified by reading the queue back** - is proven here. The
fake implements D2 exactly as miDataworks ships it on ``e2e/outstanding`` ``989038f``:
``POST /api/v1/review-queues/{id}/close``, idempotent, ``403 QUEUE_NOT_YOURS`` for an agent
closing a queue it did not create, ``409 AUDIT_IN_PROGRESS``, ``409 QUEUE_CLOSED`` for new
candidates after a close, decisions still readable.
"""

from __future__ import annotations

from typing import Any

import httpx
import pytest

from clients.midataworks import MiDataworksClient
from clients.midataworks_errors import MiDataworksRefused
from tests.support import round_trip_8_6 as rt
from tests.support.fake_midataworks import FakeMiDataworks

BASE = "http://midataworks.test:8000"


@pytest.fixture
def configured(monkeypatch: pytest.MonkeyPatch):  # type: ignore[no-untyped-def]
    from core.config import reset_settings

    monkeypatch.setenv("MIDATAWORKS_BASE_URL", BASE)
    reset_settings()
    yield
    reset_settings()


def _client(fake: Any, **kw: Any) -> MiDataworksClient:
    return MiDataworksClient(
        action="testing the 8.6 measurement",
        capability="human_review_selector",
        transport=fake.transport(),
        **kw,
    )


def _open_queue(client: MiDataworksClient) -> str:
    return str(
        client.create_external_queue(
            question="q", label_set=["good", "bad"], external_ref={"t": 1}
        )["id"]
    )


class TestTheRealCloseRoute:
    def test_close_answers_the_queue_closed_with_its_details(self, configured: None) -> None:
        fake = FakeMiDataworks()
        with _client(fake) as client:
            qid = _open_queue(client)
            closed = client.close_queue(qid, reason="done")
        assert closed["state"] == "closed"
        assert closed["closed_by"] == "agent:miforge"
        assert closed["closed_by_origin"] == "agent"
        assert closed["close_reason"] == "done"
        assert closed["closed_at"]
        assert fake.calls[-1] == ("POST", f"/api/v1/review-queues/{qid}/close")

    def test_close_is_idempotent_and_keeps_the_first_details(self, configured: None) -> None:
        fake = FakeMiDataworks()
        with _client(fake) as client:
            qid = _open_queue(client)
            first = client.close_queue(qid, reason="first")
            again = client.close_queue(qid, reason="second")
        assert again["close_reason"] == "first"
        assert again["closed_at"] == first["closed_at"]

    def test_a_closed_queue_refuses_candidates_and_keeps_decisions_readable(
        self, configured: None
    ) -> None:
        fake = FakeMiDataworks()
        with _client(fake) as client:
            qid = _open_queue(client)
            client.post_candidates(qid, rt.candidates(2, 0))
            fake.decide(qid, "f10-86-r00-0000", "accept", label="good")
            client.close_queue(qid)
            with pytest.raises(MiDataworksRefused) as caught:
                client.post_candidates(qid, rt.candidates(1, 9))
            assert caught.value.details["midataworks_code"] == "QUEUE_CLOSED"
            assert len(client.read_decisions(qid)) == 1
        with pytest.raises(ValueError, match="QUEUE_CLOSED"):
            fake.decide(qid, "f10-86-r00-0001", "accept")

    def test_a_queue_miForge_did_not_create_is_not_yours(self, configured: None) -> None:
        fake = FakeMiDataworks()
        with _client(fake) as client:
            qid = _open_queue(client)
        fake.queues[qid].created_by = "agent:someone-else"
        with _client(fake) as client, pytest.raises(MiDataworksRefused) as caught:
            client.close_queue(qid)
        assert caught.value.details["midataworks_code"] == "QUEUE_NOT_YOURS"

    def test_an_audit_in_progress_cannot_close(self, configured: None) -> None:
        fake = FakeMiDataworks()
        with _client(fake) as client:
            qid = _open_queue(client)
            fake.queues[qid].audit_in_progress = True
            with pytest.raises(MiDataworksRefused) as caught:
                client.close_queue(qid)
        assert caught.value.details["midataworks_code"] == "AUDIT_IN_PROGRESS"

    def test_an_unknown_queue_is_not_found(self, configured: None) -> None:
        with _client(FakeMiDataworks()) as client, pytest.raises(MiDataworksRefused) as caught:
            client.close_queue("rq9999")
        assert caught.value.details["midataworks_code"] == "QUEUE_NOT_FOUND"

    def test_the_list_filters_by_state(self, configured: None) -> None:
        fake = FakeMiDataworks()
        with _client(fake) as client:
            a = _open_queue(client)
            _open_queue(client)
            client.close_queue(a)
        with httpx.Client(base_url=BASE, transport=fake.transport()) as http:
            closed = http.get("/api/v1/review-queues", params={"state": "closed"}).json()
            opened = http.get("/api/v1/review-queues", params={"state": "open"}).json()
        assert [q["id"] for q in closed["items"]] == [a]
        assert len(opened["items"]) == 1


class TestEveryQueueOpenedIsClosedAndVerified:
    def test_the_happy_path_closes_each_queue_once(self, configured: None) -> None:
        fake = FakeMiDataworks()
        with _client(fake) as client:
            report = rt.measure(client, batch=10, reps=3)
        assert len(report.opened) == 3
        assert report.closed == report.opened
        assert all(fake.queues[q].state == "closed" for q in report.opened)
        assert all(fake.queues[q].close_reason == rt.CLOSE_REASON for q in report.opened)
        assert all(leg.n == 3 for leg in report.legs.values())
        text = "\n".join(report.lines())
        assert "batch 10" in text and "NO human waiting" in text and "(n=3)" in text
        assert "do not label" in fake.queues[report.opened[0]].question

    def test_a_failing_leg_still_closes_every_queue_already_opened(
        self, configured: None, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        fake = FakeMiDataworks()
        client = _client(fake)
        real_post = client.post_candidates
        calls = {"n": 0}

        def flaky(queue_id: str, batch: Any) -> Any:
            calls["n"] += 1
            if calls["n"] == 2:
                raise RuntimeError("leg failed mid-measurement")
            return real_post(queue_id, batch)

        monkeypatch.setattr(client, "post_candidates", flaky)
        with client, pytest.raises(RuntimeError, match="mid-measurement"):
            rt.measure(client, batch=5, reps=4)
        assert len(fake.queues) == 2
        assert all(q.state == "closed" for q in fake.queues.values())

    def test_a_queue_that_will_not_close_is_named_loudly(
        self, configured: None, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A 409 AUDIT_IN_PROGRESS on one close: that id is named; the rest still close."""
        fake = FakeMiDataworks()
        created: list[str] = []
        original = fake._create_queue

        def creating(request, agent):  # type: ignore[no-untyped-def]
            response = original(request, agent)
            created.append(response.json()["id"])
            if len(created) == 2:
                fake.queues[created[-1]].audit_in_progress = True
            return response

        monkeypatch.setattr(fake, "_create_queue", creating)
        with _client(fake) as client, pytest.raises(AssertionError) as caught:
            rt.measure(client, batch=3, reps=3)
        assert created[1] in str(caught.value)
        assert "close these by hand" in str(caught.value)
        assert [fake.queues[q].state for q in created] == ["closed", "open", "closed"]

    def test_a_200_that_did_not_close_is_caught_by_reading_back(
        self, configured: None, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        fake = FakeMiDataworks()
        monkeypatch.setattr(
            fake,
            "_close",
            lambda request, queue, agent: httpx.Response(200, json={"state": "open"}),
        )
        with _client(fake) as client, pytest.raises(AssertionError, match="not 'closed'"):
            rt.measure(client, batch=2, reps=1)
