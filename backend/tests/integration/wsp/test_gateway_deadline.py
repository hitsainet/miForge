"""Each editor call's deadline (fix 2026-11-22, D128 round 7).

Production, 2026-11-21: one runaway reply (32,000 tokens, 46 minutes on miServer) failed three
runs. The gateway now sends every miServer call ``max_time_s`` = the smaller of its upstream
timeout less a minute and the session's time left (to the supervisor's kill) less a margin; a
session with under 30 s left is refused through the budget path; a client cannot send its own;
a miServer that refuses the field gets the call once more without it. Through the real gateway,
the real internal application and a recording fake upstream (``test_gateway.py``'s rig): what is
asserted is what the upstream RECEIVED, call by call, and what the call timeline stored.
"""

from __future__ import annotations

import datetime as dt
from typing import Any

import pytest
import sqlalchemy as sa
from pydantic import ValidationError

from tests.integration.wsp.test_dispatch import world as dispatch_world  # noqa: F401
from tests.integration.wsp.test_gateway import Rig, _running, anth, code, oai, post
from tests.integration.wsp.test_gateway import rig as rig
from tests.integration.wsp.test_internal_app_routes import inner as inner
from tests.support.fake_upstream import Call
from wsp import gateway_ledger, runs
from wsp.settings import get_wsp_settings
from wsp_gateway import app as gateway_app
from wsp_gateway.settings import GatewaySettings
from wsp_worker import call_time

CHAT = "/v1/chat/completions"


def _expires_in(rig: Rig, run_id: str, seconds: float) -> None:
    """Re-date the run's token: the session then has ``seconds`` until its expiry."""
    with rig.world.engine.begin() as conn:
        conn.execute(
            sa.text("UPDATE mf_wsp_runs SET token_expires_at = :e WHERE run_id = :r"),
            {"e": gateway_ledger.utcnow() + dt.timedelta(seconds=seconds), "r": run_id},
        )


def _grace() -> float:
    return float(get_wsp_settings().wsp_session_grace_s)


def _timeline(rig: Rig, run_id: str) -> list[dict[str, Any]]:
    with rig.world.engine.connect() as conn:
        rows = conn.execute(
            sa.text(
                "SELECT outcome, http_status, finish_reason, refusal_code, input_tokens, "
                "output_tokens FROM mf_wsp_usage WHERE run_id = :r ORDER BY seq"
            ),
            {"r": run_id},
        ).mappings()
        return [dict(r) for r in rows]


# --------------------------------------------------------------------------- the deadline sent


@pytest.mark.parametrize("capped", [True, False], ids=["output_cap_on", "output_cap_off"])
def test_every_miserver_call_carries_the_deadline_inside_the_upstream_timeout(
    rig: Rig, capped: bool
) -> None:
    """An hour left: the upstream timeout's term wins (900 - 60 = 840 s), with the cap named
    by the request (output_cap on) and with none (off: the gateway sizes it). One call each."""
    _run, token = _running(rig.world, "ep-local", tokens=10_000_000)
    body = oai(max_tokens=500) if capped else oai()
    assert post(rig, token, body, CHAT).status_code == 200
    (call,) = rig.upstream.calls
    expected = int(call_time.GATEWAY_UPSTREAM_TIMEOUT_S - call_time.DEADLINE_HEADROOM_S)
    assert call.body[gateway_app.DEADLINE_FIELD] == expected == 840
    assert (call.body["max_tokens"] == 500) is capped


def test_a_lowered_upstream_timeout_lowers_the_deadline(rig: Rig) -> None:
    rig.overrides = {"wsp_gateway_upstream_timeout_s": 300.0}
    rig.restart()
    _run, token = _running(rig.world, "ep-local", tokens=10_000_000)
    assert post(rig, token, oai(), CHAT).status_code == 200
    assert rig.upstream.calls[-1].body["max_time_s"] == 240


def test_the_deadline_is_bounded_by_the_sessions_time_left(rig: Rig) -> None:
    """300 s to the token's expiry, which carries the backend's grace: the supervisor kills at
    300 - grace, and the deadline keeps the session margin before that."""
    run_id, token = _running(rig.world, "ep-local", tokens=10_000_000)
    _expires_in(rig, run_id, 300 + _grace())
    assert post(rig, token, oai(), CHAT).status_code == 200
    sent = rig.upstream.calls[-1].body["max_time_s"]
    # Seconds pass between the re-dating and the call: the floor of 300 - 10, at most.
    assert 285 <= sent <= 300 - call_time.SESSION_MARGIN_S


def test_the_session_entry_names_the_grace_the_deadline_takes_off(rig: Rig) -> None:
    """The internal application's session entry carries the grace (the wiring of the bound
    above): a gateway reading no grace would bound the deadline by the expiry, 120 s late."""
    run_id, token = _running(rig.world, "ep-local", tokens=10_000_000)
    assert post(rig, token, oai(), CHAT).status_code == 200
    (session,) = [s for s in rig.gateway.registry._sessions.values() if s.run_id == run_id]
    entry = rig.internal.get(f"/wsp-internal/v1/gateway/sessions/by-token/{session.token_sha256}")
    assert entry.json()["grace_s"] == _grace() > 0
    assert session.grace_s == _grace()


def test_under_the_floor_the_call_is_refused_as_the_time_budget_never_sent(rig: Rig) -> None:
    """35 s to the kill leaves 25 s for a call, under the 30 s floor: refused through the budget
    path (429 ``wsp_budget_exhausted``), recorded on the call timeline, nothing sent or reserved.
    The run's intake labels such a stop ``time_budget``, so the run completes as at its time
    budget (``wsp.runs.BUDGET_STOPS``)."""
    run_id, token = _running(rig.world, "ep-local", tokens=10_000_000)
    _expires_in(rig, run_id, 35 + _grace())
    r = post(rig, token, oai(), CHAT)
    assert r.status_code == 429 and code(r) == "wsp_budget_exhausted"
    assert "time budget" in r.json()["error"]["message"]
    assert rig.upstream.calls == []
    assert rig.usage_rows(run_id) == 0 and rig.usage_rows(run_id, refused=True) == 1
    # The engine's report of this answer ends in the time budget's stop. (Until 018-P4 so did
    # weightspace's ``error`` text; the intake no longer reads that field.)
    engine = {"failure": {"code": "wsp_budget_exhausted", "message": r.json()["error"]["message"]}}
    cause = runs.reported_failure(engine, "error")
    assert runs.labelled_stop("error", cause) == "time_budget"


def test_just_over_the_floor_the_call_is_sent_with_what_is_left(rig: Rig) -> None:
    run_id, token = _running(rig.world, "ep-local", tokens=10_000_000)
    _expires_in(rig, run_id, 45 + _grace())
    assert post(rig, token, oai(), CHAT).status_code == 200
    assert 30 <= rig.upstream.calls[-1].body["max_time_s"] <= 35


@pytest.mark.parametrize(
    "endpoint, route, body",
    [("ep-ext", CHAT, oai), ("ep-anth", "/v1/messages", anth)],
    ids=["generic_openai", "anthropic"],
)
def test_a_non_miserver_endpoint_gets_no_deadline(
    rig: Rig, endpoint: str, route: str, body: Any
) -> None:
    _run, token = _running(rig.world, endpoint, cost_usd=1, tokens=10_000_000)
    assert post(rig, token, body(), route).status_code == 200
    (call,) = rig.upstream.calls
    assert "max_time_s" not in call.body


@pytest.mark.parametrize("asked", [99_999, 5])
@pytest.mark.parametrize("endpoint", ["ep-local", "ep-ext"])
def test_a_client_supplied_deadline_is_refused_so_only_the_gateways_is_sent(
    rig: Rig, asked: int, endpoint: str
) -> None:
    """The gateway's value is the only one: a client's ``max_time_s``, larger or smaller, for a
    miServer endpoint or not, is refused before any upstream call."""
    _run, token = _running(rig.world, endpoint, cost_usd=1, tokens=10_000_000)
    r = post(rig, token, oai(max_time_s=asked), CHAT)
    assert r.status_code == 422 and code(r) == "wsp_request_field_refused"
    assert "max_time_s" in r.json()["error"]["message"]
    assert rig.upstream.calls == []


def test_the_deadline_settings_are_bounded_so_no_client_can_hang_up_first() -> None:
    """The upstream timeout may be lowered, never raised past the constant the worker's client
    timeouts derive from, nor lowered under the headroom plus the shortest call."""
    assert GatewaySettings().wsp_gateway_upstream_timeout_s == call_time.GATEWAY_UPSTREAM_TIMEOUT_S
    for value in (call_time.CLIENT_TIMEOUT_S, call_time.GATEWAY_UPSTREAM_TIMEOUT_S + 1, 89.0):
        with pytest.raises(ValidationError):
            GatewaySettings(wsp_gateway_upstream_timeout_s=value)
    assert GatewaySettings(wsp_gateway_upstream_timeout_s=90.0).wsp_gateway_upstream_timeout_s == 90


def test_the_deadline_rule() -> None:
    assert call_time.call_deadline(900.0, None) == 840
    assert call_time.call_deadline(900.0, 3_600.0) == 840
    assert call_time.call_deadline(900.0, 200.5) == 190
    assert call_time.call_deadline(900.0, 40.0) == 30
    assert call_time.call_deadline(900.0, 39.9) is None
    assert call_time.call_deadline(5_000.0, None) == 3_600  # miServer's own bound


# ------------------------------------------------------ a miServer that does not take the field


def test_a_miserver_that_refuses_the_field_gets_the_call_once_more_without_it(rig: Rig) -> None:
    """One older than the field, or an engine that refuses it: the call is resent ONCE without
    it (the refusal settled at zero, on the timeline as ``editor_deadline_refused``), and the
    endpoint is sent none until the retry interval has passed."""
    run_id, token = _running(rig.world, "ep-local", tokens=10_000_000)
    rig.upstream.refuse_fields = frozenset({"max_time_s"})
    r = post(rig, token, oai(max_tokens=400), CHAT)
    assert r.status_code == 200 and r.json()["choices"][0]["message"]["content"] == "ok"
    first, second = rig.upstream.calls
    assert first.body["max_time_s"] == 840 and "max_time_s" not in second.body
    assert {k: v for k, v in first.body.items() if k != "max_time_s"} == second.body
    timeline = _timeline(rig, run_id)
    assert [(c["outcome"], c["refusal_code"]) for c in timeline] == [
        ("upstream_error", "editor_deadline_refused"),
        ("served", None),
    ]
    assert (timeline[0]["input_tokens"], timeline[0]["output_tokens"]) == (0, 0)
    # The next call is sent without it at once (one upstream call, not two).
    assert post(rig, token, oai(max_tokens=400), CHAT).status_code == 200
    assert len(rig.upstream.calls) == 3 and "max_time_s" not in rig.upstream.calls[2].body
    # After the retry interval the endpoint is tried again.
    rig.upstream.refuse_fields = frozenset()
    for endpoint in list(rig.gateway._deadline_refused):
        rig.gateway._deadline_refused[endpoint] -= gateway_app.DEADLINE_RETRY_S
    assert post(rig, token, oai(max_tokens=400), CHAT).status_code == 200
    assert len(rig.upstream.calls) == 4 and rig.upstream.calls[3].body["max_time_s"] == 840


def test_a_miserver_that_ignores_the_field_is_served_and_never_marked_a_deadline_stop(
    rig: Rig,
) -> None:
    """An older miServer reports the field unused (non-strict; the gateway never sends
    ``X-miLLM-Strict``): the answer is served as before, and its length stop is not called the
    deadline's, since that miServer had none."""
    run_id, token = _running(rig.world, "ep-local", tokens=10_000_000)
    rig.upstream.ignored_fields = frozenset({"max_time_s"})
    _deadline_length_stop(rig)
    assert post(rig, token, oai(max_tokens=400), CHAT).status_code == 200
    (call,) = rig.upstream.calls
    assert call.body["max_time_s"] == 840
    ((outcome, finish, code_),) = [
        (c["outcome"], c["finish_reason"], c["refusal_code"]) for c in _timeline(rig, run_id)
    ]
    assert (outcome, finish, code_) == ("served", "length", None)


# --------------------------------------------------- the deadline's length stop, on the timeline


class Clock:
    def __init__(self) -> None:
        self.now = 5_000.0

    def monotonic(self) -> float:
        return self.now


def _deadline_length_stop(rig: Rig, *, took_s: float | None = None, output: int = 50) -> None:
    """The fake miServer answers ``length`` with ``output`` tokens, after ``took_s`` on the
    gateway's clock (default: the deadline it was sent)."""
    clock = Clock()
    rig.gateway._monotonic = clock.monotonic

    def answer(call: Call) -> None:
        clock.now += float(call.body.get("max_time_s", 0)) if took_s is None else took_s

    rig.upstream.on_call = answer
    rig.upstream.finish = "length"
    rig.upstream.usage = {"prompt_tokens": 10, "completion_tokens": output}


@pytest.mark.parametrize(
    "took_s, output, recorded",
    [
        (None, 50, gateway_app.DEADLINE_STOP_CODE),  # the whole deadline, short of the cap
        (839.5, 50, gateway_app.DEADLINE_STOP_CODE),  # within the slack
        (500.0, 50, None),  # stopped early: not the deadline's
        (None, 400, None),  # stopped AT its cap: the cap's, not the deadline's
    ],
    ids=["deadline", "deadline_within_slack", "early", "at_cap"],
)
def test_a_length_stop_at_the_deadline_is_recorded_as_the_deadlines(
    rig: Rig, took_s: float | None, output: int, recorded: str | None
) -> None:
    run_id, token = _running(rig.world, "ep-local", tokens=10_000_000)
    _deadline_length_stop(rig, took_s=took_s, output=output)
    r = post(rig, token, oai(max_tokens=400), CHAT)
    assert r.status_code == 200
    ((outcome, finish, code_),) = [
        (c["outcome"], c["finish_reason"], c["refusal_code"]) for c in _timeline(rig, run_id)
    ]
    assert (outcome, finish, code_) == ("served", "length", recorded)
