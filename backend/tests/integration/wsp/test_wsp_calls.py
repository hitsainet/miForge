"""A WSP run's editor-call timeline (live progress, 2026-11-20): what the gateway records per call
and what ``GET /wsp/runs/{id}/calls`` serves. **Counts and codes only, never message text.**

Every call here goes through the real gateway (``wsp_gateway.app.Gateway``) against the real
internal application and a recording fake upstream (``test_gateway.py``'s rig), and is read back
through the production app's route, so the gateway's ledger write, the internal route, the
backend's checks and the served route are each on the path.
"""

from __future__ import annotations

import json
from typing import Any

import pytest
import sqlalchemy as sa
from fastapi.testclient import TestClient

from tests.integration.wsp.test_dispatch import world as dispatch_world  # noqa: F401
from tests.integration.wsp.test_gateway import Rig, _running, anth, code, oai, post
from tests.integration.wsp.test_gateway import rig as rig
from tests.integration.wsp.test_internal_app_routes import inner as inner
from tests.integration.wsp.test_run_cause import _client
from wsp import gateway_ledger, progress

#: Written into every prompt and every answer: it must appear in no stored row and no response.
SENTINEL = "SENTINEL-prompt-and-answer-text-7f3a"

#: ``mf_wsp_usage``'s columns, exactly. A text column added here would be a place for content.
USAGE_COLUMNS = {
    "id",
    "run_id",
    "seq",
    "endpoint_id",
    "input_tokens",
    "output_tokens",
    "cost_usd",
    "at",
    "outcome",
    "http_status",
    "latency_ms",
    "finish_reason",
    "refusal_code",
    # Migration 0033: a served call's prompt evidence, two counts and a digest (never text).
    "sent_bytes",
    "prompt_messages",
    "conversation_sha256",
}


@pytest.fixture()
def api(rig: Rig, monkeypatch: pytest.MonkeyPatch) -> TestClient:
    return _client(rig.world, monkeypatch)


def _calls(api: TestClient, run_id: str, **query: Any) -> dict[str, Any]:
    answer = api.get(f"/api/v1/wsp/runs/{run_id}/calls", params=query)
    assert answer.status_code == 200, answer.text
    body: dict[str, Any] = answer.json()
    return body


def _stored(rig: Rig, run_id: str) -> list[dict[str, Any]]:
    with rig.world.engine.connect() as conn:
        rows = conn.execute(
            sa.text("SELECT * FROM mf_wsp_usage WHERE run_id = :r ORDER BY seq"), {"r": run_id}
        ).mappings()
        return [dict(r) for r in rows]


def _anth_with(text: str) -> dict[str, Any]:
    return anth(messages=[{"role": "user", "content": [{"type": "text", "text": text}]}])


# ----------------------------------------------------------------------------- the outcomes


def test_a_served_call_records_its_status_latency_finish_reason_and_tokens(
    rig: Rig, api: TestClient
) -> None:
    run_id, token = _running(rig.world, "ep-anth", cost_usd=1, tokens=10_000_000)
    assert post(rig, token, anth()).status_code == 200
    (call,) = _calls(api, run_id)["calls"]
    assert call["seq"] == 0 and call["outcome"] == "served" and call["http_status"] == 200
    assert isinstance(call["latency_ms"], int) and call["latency_ms"] >= 0
    assert call["finish_reason"] == "end_turn" and call["refusal_code"] is None
    assert call["input_tokens"] > 0 and call["output_tokens"] > 0 and call["cost_usd"] > 0
    assert call["at"] is not None


def test_an_openai_finish_reason_is_read_from_its_first_choice(rig: Rig, api: TestClient) -> None:
    run_id, token = _running(rig.world, "ep-local", tokens=50_000)
    rig.upstream.finish = "length"
    assert post(rig, token, oai(), "/v1/chat/completions").status_code == 200
    (call,) = _calls(api, run_id)["calls"]
    assert (call["outcome"], call["finish_reason"]) == ("served", "length")


def test_a_refused_call_is_recorded_with_its_code_and_is_not_a_request(
    rig: Rig, api: TestClient
) -> None:
    """The budget cannot pay for the floor: refused before the reservation, nothing spent."""
    run_id, token = _running(rig.world, "ep-local", tokens=100)
    r = post(rig, token, oai(max_tokens=300), "/v1/chat/completions")
    assert code(r) == "wsp_budget_exhausted"
    assert rig.upstream.calls == []
    (call,) = _calls(api, run_id)["calls"]
    assert call["outcome"] == "refused" and call["refusal_code"] == "wsp_budget_exhausted"
    assert (call["input_tokens"], call["output_tokens"], call["cost_usd"]) == (0, 0, 0.0)
    assert call["http_status"] is None and call["latency_ms"] is None
    with rig.world.factory() as session:
        from wsp import runs

        assert runs.metered_usage(session, run_id)["requests"] == 0


def test_a_refusal_at_the_reservation_is_recorded_too(rig: Rig, api: TestClient) -> None:
    """The backend refuses the reserve (``wsp_request_in_flight``: an open reservation)."""
    run_id, token = _running(rig.world, "ep-local", tokens=50_000)
    with rig.world.factory() as session:
        gateway_ledger.reserve(
            session,
            run_id=run_id,
            token_sha256=_token_sha(token),
            worst_case_usd=gateway_ledger.ZERO,
            worst_case_tokens=1,
            external=False,
            settings=rig.world.settings,
        )
        session.commit()
    r = post(rig, token, oai(), "/v1/chat/completions")
    assert code(r) == "wsp_request_in_flight"
    (call,) = _calls(api, run_id)["calls"]
    assert (call["outcome"], call["refusal_code"]) == ("refused", "wsp_request_in_flight")


def test_an_upstream_that_does_not_answer_is_recorded_at_worst_case(
    rig: Rig, api: TestClient
) -> None:
    run_id, token = _running(rig.world, "ep-anth", cost_usd=1, tokens=10_000_000)
    # A request that was sent and never answered (fix 2026-11-21: a connection never made is
    # not this case, and settles at zero; ``test_gateway.py``).
    rig.upstream.fail = "read_timeout"
    assert code(post(rig, token, anth())) == "wsp_gateway_upstream_failed"
    (call,) = _calls(api, run_id)["calls"]  # one row: the settle's, not a second refusal row
    assert call["outcome"] == "upstream_failed"
    assert call["refusal_code"] == "wsp_gateway_upstream_failed"
    assert call["http_status"] is None and call["latency_ms"] >= 0
    assert call["output_tokens"] > 0 and call["cost_usd"] > 0  # the worst case, never zero


def test_a_refused_upstream_call_is_recorded_at_zero(rig: Rig, api: TestClient) -> None:
    """Live run wsp_c3e954d3f21f: ``/calls`` showed ``upstream_error, 400, output_tokens: 9421``
    for a call that generated nothing. It reads zero now, as the run's usage does."""
    run_id, token = _running(rig.world, "ep-anth", cost_usd=1, tokens=10_000_000)
    rig.upstream.error = (400, "invalid_request_error")
    assert post(rig, token, anth()).status_code == 400
    (call,) = _calls(api, run_id)["calls"]
    assert (call["outcome"], call["http_status"]) == ("upstream_error", 400)
    assert (call["input_tokens"], call["output_tokens"], call["cost_usd"]) == (0, 0, 0)


def test_an_upstream_error_answer_is_recorded_with_its_status(rig: Rig, api: TestClient) -> None:
    run_id, token = _running(rig.world, "ep-anth", cost_usd=1, tokens=10_000_000)
    rig.upstream.error = (400, "invalid_request_error")
    assert post(rig, token, anth()).status_code == 400
    (call,) = _calls(api, run_id)["calls"]
    assert (call["outcome"], call["http_status"]) == ("upstream_error", 400)


def test_a_miserver_answer_that_ends_the_session_is_recorded_once(
    rig: Rig, api: TestClient
) -> None:
    run_id, token = _running(rig.world, "ep-local", tokens=50_000)
    rig.upstream.steering = "profile:abc"
    assert code(post(rig, token, oai(), "/v1/chat/completions")) == "editor_steered"
    (call,) = _calls(api, run_id)["calls"]
    assert (call["outcome"], call["refusal_code"]) == ("session_ended", "editor_steered")
    assert call["output_tokens"] > 0  # metered: the call was served and billed


def test_a_refusal_for_an_ended_session_is_never_recorded(rig: Rig, api: TestClient) -> None:
    run_id, token = _running(rig.world, "ep-local", tokens=50_000)
    rig.upstream.steering = "profile:abc"
    post(rig, token, oai(), "/v1/chat/completions")
    assert code(post(rig, token, oai(), "/v1/chat/completions")) == "wsp_session_not_active"
    assert len(_calls(api, run_id)["calls"]) == 1


def test_a_failed_refusal_record_never_changes_the_answer(rig: Rig) -> None:
    from wsp_gateway.meter import LedgerRefused

    run_id, token = _running(rig.world, "ep-local", tokens=100)

    def down(**_kw: Any) -> None:
        raise LedgerRefused(None, "down")

    rig.gateway.ledger.refused = down  # type: ignore[method-assign]
    r = post(rig, token, oai(max_tokens=300), "/v1/chat/completions")
    assert r.status_code == 429 and code(r) == "wsp_budget_exhausted"
    assert rig.usage_rows(run_id, refused=True) == 0


# ------------------------------------------------------------------------ the page and the route


def test_after_seq_reads_only_what_is_new_and_pages_in_order(rig: Rig, api: TestClient) -> None:
    run_id, token = _running(rig.world, "ep-local", tokens=500_000)
    for _ in range(3):
        assert post(rig, token, oai(max_tokens=300), "/v1/chat/completions").status_code == 200
    assert [c["seq"] for c in _calls(api, run_id)["calls"]] == [0, 1, 2]
    newer = _calls(api, run_id, after_seq=0)
    assert [c["seq"] for c in newer["calls"]] == [1, 2] and newer["more"] is False
    page = _calls(api, run_id, limit=2)
    assert [c["seq"] for c in page["calls"]] == [0, 1] and page["more"] is True
    assert page["live"] is True and page["status"] == "running"


def test_an_unknown_runs_calls_are_404(api: TestClient) -> None:
    answer = api.get("/api/v1/wsp/runs/wsp_ffffffffffff/calls")
    assert answer.status_code == 404 and answer.json()["error"]["code"] == "wsp_run_not_found"


# ---------------------------------------------------------------------- the no-content guarantee


def test_no_message_text_reaches_the_table_or_the_route(rig: Rig, api: TestClient) -> None:
    """The prompt and the answer carry the sentinel; so does a hostile finish reason. None of it
    is stored or served: the table's columns are counts and codes, the route's fields exactly
    :data:`progress.CALL_FIELDS`, and a finish reason that is not an identifier is ``other``."""
    run_id, token = _running(rig.world, "ep-anth", cost_usd=1, tokens=10_000_000)
    rig.upstream.text = f"answer {SENTINEL}"
    assert post(rig, token, _anth_with(f"prompt {SENTINEL}")).status_code == 200
    rig.upstream.finish = f"stop because {SENTINEL}"
    assert post(rig, token, _anth_with(f"prompt {SENTINEL}")).status_code == 200
    rig.upstream.fail = True
    post(rig, token, _anth_with(f"prompt {SENTINEL}"))
    assert SENTINEL in json.dumps(rig.upstream.calls[0].body)  # the control: it was sent

    stored = _stored(rig, run_id)
    assert len(stored) == 3
    assert set(stored[0]) == USAGE_COLUMNS
    for row in stored:
        for value in row.values():
            assert SENTINEL not in str(value), row
    served = api.get(f"/api/v1/wsp/runs/{run_id}/calls")
    assert SENTINEL not in served.text
    calls = served.json()["calls"]
    assert all(set(c) == set(progress.CALL_FIELDS) for c in calls)
    assert [c["finish_reason"] for c in calls] == ["end_turn", "other", None]


def test_the_usage_model_holds_no_text_column() -> None:
    """The model's columns equal the table the migration makes (``test_db_migration`` compares
    the two); here, that set is pinned and every string column is a short identifier."""
    from wsp.models import WspUsage

    columns = {c.name: c for c in WspUsage.__table__.columns}
    assert set(columns) == USAGE_COLUMNS
    strings = {n: c.type.length for n, c in columns.items() if isinstance(c.type, sa.String)}
    assert strings == {
        "run_id": 16,
        "endpoint_id": 40,
        "outcome": 16,
        "finish_reason": 32,
        "refusal_code": 64,
        "conversation_sha256": 64,
    }


@pytest.mark.parametrize(
    ("call", "expected"),
    [
        ({"finish_reason": "stop"}, "stop"),
        ({"finish_reason": "Stop now, here is the answer"}, "other"),
        ({"finish_reason": "x" * 40}, "other"),
        ({"finish_reason": 7}, "other"),
        ({}, None),
    ],
)
def test_call_columns_keeps_only_identifiers(call: dict[str, Any], expected: Any) -> None:
    assert gateway_ledger.call_columns(call)["finish_reason"] == expected


def test_call_columns_drops_what_fails_its_check() -> None:
    assert gateway_ledger.call_columns(
        {
            "outcome": "made_up",
            "http_status": 42,
            "latency_ms": -1,
            "refusal_code": "Not A Code!",
            "prompt": SENTINEL,
            "sent_bytes": 0,
            "prompt_messages": -1,
            "conversation_sha256": SENTINEL,
        }
    ) == {
        "outcome": None,
        "http_status": None,
        "latency_ms": None,
        "finish_reason": None,
        "refusal_code": None,
        "sent_bytes": None,
        "prompt_messages": None,
        "conversation_sha256": None,
    }


def test_call_columns_keeps_a_served_calls_prompt_evidence() -> None:
    """Migration 0033: two counts and a digest; a digest that is not 64 lower-case hex is
    dropped, so no text can ride in it."""
    kept = gateway_ledger.call_columns(
        {"sent_bytes": 33_100, "prompt_messages": 30, "conversation_sha256": "ab" * 32}
    )
    assert (kept["sent_bytes"], kept["prompt_messages"], kept["conversation_sha256"]) == (
        33_100,
        30,
        "ab" * 32,
    )
    assert (
        gateway_ledger.call_columns({"conversation_sha256": "AB" * 32})["conversation_sha256"]
        is None
    )


def _token_sha(token: str) -> str:
    import hashlib

    return hashlib.sha256(token.encode()).hexdigest()
