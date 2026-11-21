"""The WSP gateway against the real internal application and a recording fake upstream (feature 016;
FTASKS 6.3, 6.3b, 6.4, 6.7, 6.7b in part; FR-016.15, FR-016.16; P28, P47, P50, P61, R1B-A2, R1B-A5,
R1B-M6, R1B-M7, R1B-M8, R1B-M22, R2A-5, R2A-11, R2A-23, R3A-6, R3A-17).

The gateway is ``wsp_gateway.app.Gateway`` itself, served by ``create_gateway_app``; its internal
client is a ``TestClient`` of ``create_internal_app()`` bound to a migrated database (the same
reserve and settle the pod uses); its upstream is ``tests/support/fake_upstream.FakeUpstream``.
"""

from __future__ import annotations

import base64
import datetime as dt
import json
import logging
import math
import threading
from collections.abc import Iterator
from decimal import Decimal
from typing import Any
from unittest.mock import patch

import pytest
import sqlalchemy as sa
from fastapi.testclient import TestClient

from api.app import REFUSAL_STATUS
from tests.integration.wsp.test_dispatch import World
from tests.integration.wsp.test_dispatch import world as dispatch_world  # noqa: F401
from tests.integration.wsp.test_internal_app_routes import (
    AUTH,
    SERVICE,
    _long_sqlite_busy_timeout,
)
from tests.integration.wsp.test_internal_app_routes import inner as inner
from tests.integration.wsp.wsp_db import NOW, insert
from tests.support.fake_upstream import FakeUpstream
from wsp import gateway_ledger
from wsp_gateway import sealing
from wsp_gateway.app import STATUS, Gateway, create_gateway_app
from wsp_gateway.settings import GatewaySettings

KEY = "sk-ant-SENTINEL-KEY-0123456789"
MISERVER_IP = "10.152.183.10"


def _running(world: World, endpoint: str = "ep-local", **budgets: Any) -> tuple[str, str]:
    """A queued run made live for the gateway as the driver leaves it after a push (``running``,
    a minted token's sha256, an expiry), without the one-worker driver, so a test can hold two
    live sessions at once. The push itself is test_dispatch.py's."""
    from wsp import tokens

    world.body = {**world.body, "budgets": {**world.body["budgets"], **budgets}}
    run_id = world.queue(endpoint)
    plaintext, sha = tokens.mint()
    with world.engine.begin() as conn:
        conn.execute(
            sa.text(
                "UPDATE mf_wsp_runs SET status = 'running', gateway_token_sha256 = :sha, "
                "token_expires_at = :exp WHERE run_id = :r"
            ),
            {"sha": sha, "exp": gateway_ledger.utcnow() + dt.timedelta(hours=1), "r": run_id},
        )
    return run_id, plaintext


def _finish(world: World, run_id: str) -> None:
    """End a run (one session is active at a time: ``uq_mf_wsp_runs_one_active``)."""
    from wsp import runs

    with world.factory() as session:
        runs.transition(session, run_id, "running", "cancelled", stop_reason="cancelled")
        session.commit()


def resolver(host: str) -> list[str]:
    return {"miserver-backend.miserver.svc": [MISERVER_IP]}.get(host, ["93.184.216.34"])


class Rig:
    """One world, its internal app, a fake upstream and the gateway under test."""

    def __init__(self, world: World, internal: TestClient, keys: tuple[str, str]) -> None:
        self.world, self.internal, self.private, self.public = world, internal, *keys
        self.upstream = FakeUpstream()
        self.overrides: dict[str, Any] = {}
        self.gateway, self.client = self.start()

    def start(self) -> tuple[Gateway, TestClient]:
        settings = GatewaySettings(
            wsp_internal_url="http://testserver",
            wsp_gateway_service_token=SERVICE,
            wsp_gateway_private_key=self.private,
            **self.overrides,
        )
        gateway = Gateway(
            settings,
            internal=self.internal,
            upstream_client=self.upstream.client(),
            resolver=resolver,
        )
        gateway.registry.rebuild()
        return gateway, TestClient(create_gateway_app(gateway), raise_server_exceptions=False)

    def restart(self) -> None:
        self.gateway, self.client = self.start()

    def seal(self, endpoint_id: str, key: str = KEY) -> None:
        from api.wsp_internal_app import endpoint_document
        from wsp import runs

        with self.world.factory() as session:
            row = runs.live_endpoint(session, endpoint_id)
            assert row is not None
            row.key_ciphertext = sealing.seal(
                self.public, key.encode(), sealing.aad_digest(endpoint_document(row))
            )
            row.key_fingerprint = sealing.fingerprint(key)
            session.commit()

    def usage_rows(self, run_id: str, *, refused: bool = False) -> int:
        """The run's metered calls; with ``refused``, the calls the gateway refused before any
        spend instead (``outcome: refused``, recorded for the call timeline since 2026-11-20)."""
        test = "outcome = 'refused'" if refused else "(outcome IS NULL OR outcome <> 'refused')"
        with self.world.engine.connect() as conn:
            return int(
                conn.execute(
                    sa.text(f"SELECT COUNT(*) FROM mf_wsp_usage WHERE run_id = :r AND {test}"),
                    {"r": run_id},
                ).scalar_one()
            )


@pytest.fixture()
def rig(inner: Any) -> Iterator[Rig]:
    world, internal = inner
    internal.headers.update(AUTH)
    with world.engine.begin() as conn:
        insert(
            conn,
            "mf_wsp_endpoints",
            id="ep-anth",
            kind="anthropic",
            base_url="https://api.anthropic.com",
            model="claude-editor",
            locality="external",
            price_in_per_mtok=3,
            price_out_per_mtok=15,
            price_cache_write_per_mtok=3.75,
            price_cache_read_per_mtok=0.3,
            output_cap_field="max_tokens",
            serving="generic",
            created_at=NOW,
            created_by="operator",
        )
    r = Rig(world, internal, sealing.generate_keypair())
    r.seal("ep-anth")
    r.seal("ep-ext")
    yield r


def anth(**extra: Any) -> dict[str, Any]:
    return {
        "model": "anything",
        "max_tokens": 1000,
        "messages": [{"role": "user", "content": [{"type": "text", "text": "hi"}]}],
        **extra,
    }


def oai(**extra: Any) -> dict[str, Any]:
    return {"model": "anything", "messages": [{"role": "user", "content": "hi"}], **extra}


def post(rig: Rig, token: str, body: Any, route: str = "/v1/messages", **headers: str) -> Any:
    return rig.client.post(route, content=json.dumps(body), headers={"x-api-key": token, **headers})


def code(response: Any) -> str:
    return str(response.json()["error"]["code"])


# ------------------------------------------------------------------- the money and the tokens


def _window(rig: Rig, endpoint_id: str, tokens: int | None) -> None:
    with rig.world.engine.begin() as conn:
        conn.execute(
            sa.text("UPDATE mf_wsp_endpoints SET context_window_tokens = :w WHERE id = :e"),
            {"w": tokens, "e": endpoint_id},
        )


def _sent_bytes(rig: Rig) -> int:
    return len(json.dumps(rig.upstream.calls[-1].body, separators=(",", ":")).encode())


def _first_call_bounds(rig: Rig, window: int) -> tuple[int, int]:
    """The cap a session's FIRST uncapped call may get (fix 2026-11-21): the window less the
    margined estimate at 0.75 tokens per byte, over the bytes the gateway sized (the forwarded
    body, within 100 bytes of slack for the cap's digits and the 64-byte allowance)."""
    import math

    from wsp_gateway import meter

    sent = _sent_bytes(rig)
    low = window - meter.margined(math.ceil((sent + 100) * meter.FIRST_TOKENS_PER_BYTE))
    high = window - meter.margined(math.ceil(sent * meter.FIRST_TOKENS_PER_BYTE))
    return low, high


@pytest.mark.parametrize("window", [32_768, None])
def test_a_request_with_no_cap_fits_the_endpoints_window(rig: Rig, window: int | None) -> None:
    """Production 2026-11-20: a no-cap request on a 32,768-token miServer window was sent with
    32,000 output tokens; miServer answered ``context_length_exceeded``, which ends the session.
    Now the forwarded cap plus the estimated prompt (a first call: 0.75 tokens per byte, plus
    10% and the template's 512) fits the endpoint's window, or 8,192 when it states none; the
    budget is too large to bind."""
    from wsp_gateway import meter

    _window(rig, "ep-local", window)
    _run, token = _running(rig.world, "ep-local", tokens=10_000_000)
    big = oai(messages=[{"role": "user", "content": "x" * 6_000}])
    assert post(rig, token, big, "/v1/chat/completions").status_code == 200
    cap = rig.upstream.calls[-1].body["max_tokens"]
    limit = window or meter.DEFAULT_WINDOW_TOKENS
    assert meter.MIN_OUTPUT_TOKENS <= cap < meter.MAX_OUTPUT_TOKENS
    low, high = _first_call_bounds(rig, limit)
    assert low <= cap <= high


def test_a_request_naming_its_cap_keeps_it(rig: Rig) -> None:
    """The window rule is only for a request with no cap: the providers' own caps pass as sent
    (the engine sizes its own against the same window)."""
    _window(rig, "ep-local", 4_096)
    _run, token = _running(rig.world, "ep-local", tokens=10_000_000)
    assert post(rig, token, oai(max_tokens=2_048), "/v1/chat/completions").status_code == 200
    assert rig.upstream.calls[-1].body["max_tokens"] == 2_048


def test_a_spent_budget_says_which_budget(rig: Rig) -> None:
    """The intake labels the stop from this message (``wsp.runs.budget_stop``): the token
    budget on an unpriced endpoint, the cost cap on a priced one, from the clamp and from the
    backend's reservation alike."""
    _run, token = _running(rig.world, "ep-local", tokens=300)
    r = post(rig, token, oai(), "/v1/chat/completions")
    assert r.status_code == 429 and code(r) == "wsp_budget_exhausted"
    assert "token budget" in r.json()["error"]["message"]
    _finish(rig.world, _run)
    run2, token2 = _running(rig.world, "ep-anth", cost_usd=0.0001, tokens=10_000_000)
    r = post(rig, token2, anth(max_tokens=300))
    assert r.status_code == 429 and code(r) == "wsp_budget_exhausted"
    assert "cost cap" in r.json()["error"]["message"]
    _finish(rig.world, run2)
    # The reservation's refusal carries the backend's own message, not a generic one: a second
    # gateway whose spend snapshot is stale passes its own clamp, and the backend refuses.
    _run3, token3 = _running(rig.world, "ep-local", tokens=1_000)
    first = rig.client
    rig.restart()
    for _ in range(2):
        sent = first.post(
            "/v1/chat/completions",
            content=json.dumps(oai(max_tokens=300)),
            headers={"x-api-key": token3},
        )
        assert sent.status_code == 200, sent.text
    r = post(rig, token3, oai(max_tokens=300), "/v1/chat/completions")
    assert r.status_code == 429 and code(r) == "wsp_budget_exhausted", r.text
    assert r.json()["error"]["message"] == (
        "The session's token budget would be exceeded. (code: wsp_budget_exhausted)"
    )


def test_eight_concurrent_requests_never_take_spend_past_the_usd_cap(rig: Rig) -> None:
    """R1B-A2: maximal usage from eight threads, one token, an external endpoint."""
    run_id, token = _running(rig.world, "ep-anth", cost_usd=0.2, tokens=10_000_000)
    _long_sqlite_busy_timeout(rig.world.engine)
    barrier, statuses = threading.Barrier(8), []

    def one() -> None:
        barrier.wait()
        for _ in range(6):
            statuses.append(post(rig, token, anth(max_tokens=4000)).status_code)

    threads = [threading.Thread(target=one) for _ in range(8)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    with rig.world.factory() as session:
        usd, _tokens = gateway_ledger.spent(session, run_id)
    assert 200 in statuses and set(statuses) <= {200, 409, 429}, sorted(set(statuses))
    assert 0 < usd <= 0.2
    assert rig.usage_rows(run_id) == len(rig.upstream.calls)  # one usage record per upstream call
    # Every refusal is on the call timeline too, and none of them is a request.
    assert rig.usage_rows(run_id, refused=True) == sum(1 for s in statuses if s != 200)


def test_a_local_endpoint_never_passes_the_token_budget(rig: Rig) -> None:
    """R2A-11: prices are 0 locally, so only the token reservation bounds the session."""
    run_id, token = _running(rig.world, "ep-local", tokens=3000)
    for _ in range(10):
        post(rig, token, oai(), "/v1/chat/completions")
    with rig.world.factory() as session:
        _usd, used = gateway_ledger.spent(session, run_id)
    assert len(rig.upstream.calls) >= 1 and 0 < used <= 3000
    last = post(rig, token, oai(), "/v1/chat/completions")
    assert last.status_code == 429 and code(last) == "wsp_budget_exhausted"


def test_two_gateways_on_one_local_session_stay_within_the_token_budget(rig: Rig) -> None:
    """R2A-11: each gateway's own spend snapshot goes stale when another serves the same token
    (a restart overlap); only the backend's TOKEN reservation then bounds a local, unpriced
    session (control C15o: a USD-only reservation lets the total pass the budget)."""
    run_id, token = _running(rig.world, "ep-local", tokens=4000)
    first = rig.client
    rig.restart()
    second = rig.client
    for _ in range(6):
        for client in (first, second):
            client.post(
                "/v1/chat/completions",
                content=json.dumps(oai(max_tokens=900)),
                headers={"x-api-key": token},
            )
    with rig.world.factory() as session:
        _usd, used = gateway_ledger.spent(session, run_id)
    assert len(rig.upstream.calls) >= 2 and 0 < used <= 4000


def test_restarting_the_gateway_mid_session_keeps_the_total_within_the_cap(rig: Rig) -> None:
    """R1B-M7: spend is rebuilt from the backend, never the gateway's memory."""
    run_id, token = _running(rig.world, "ep-anth", cost_usd=0.1, tokens=10_000_000)
    for _ in range(3):
        post(rig, token, anth(max_tokens=2000))
    rig.restart()
    assert len(rig.gateway.registry) == 1  # rebuilt from GET /sessions, with spend
    # The rebuilt spend sizes the cap to what is LEFT, so the next request fits and is served
    # (a registry rebuilt without spend asks for a cap the backend's reserve then refuses).
    (session,) = list(rig.gateway.registry._sessions.values())
    assert session.spend_usd > 0
    nocap = anth()
    del nocap["max_tokens"]
    assert post(rig, token, nocap).status_code == 200
    for _ in range(10):
        post(rig, token, anth(max_tokens=2000))
    with rig.world.factory() as session:
        usd, _tokens = gateway_ledger.spent(session, run_id)
    assert usd <= 0.1


def test_the_output_cap_is_inserted_and_the_thinking_budget_clamped(rig: Rig) -> None:
    """R1B-A5: no max_tokens in the request; budget_tokens above the cap is clamped below it."""
    run, token = _running(rig.world, "ep-anth", cost_usd=0.02, tokens=10_000_000)
    body = anth(thinking={"type": "enabled", "budget_tokens": 30_000})
    del body["max_tokens"]
    assert post(rig, token, body).status_code == 200
    sent = rig.upstream.calls[-1].body
    assert 256 <= sent["max_tokens"] < 30_000
    assert sent["thinking"]["budget_tokens"] < sent["max_tokens"]
    _finish(rig.world, run)
    _run2, token2 = _running(rig.world, "ep-local", tokens=50_000)
    assert post(rig, token2, oai(), "/v1/chat/completions").status_code == 200
    assert "max_tokens" in rig.upstream.calls[-1].body  # the endpoint's field, though none was sent


# --------------------------------------------------------------------------------- the allow-list


@pytest.mark.parametrize(
    "body",
    [
        pytest.param(
            anth(tools=[{"type": "web_fetch_20250910", "name": "web_fetch"}]), id="web_fetch"
        ),
        pytest.param(
            anth(
                tools=[
                    {
                        "type": "web_search_20250305",
                        "name": "web_search",
                        "input_schema": {"type": "object"},
                    }
                ]
            ),
            id="server-tool-shaped-like-a-custom-one",
        ),
        pytest.param(anth(mcp_servers=[{"type": "url", "url": "https://x"}]), id="mcp_servers"),
        pytest.param(anth(service_tier="priority"), id="service_tier"),
        pytest.param(
            anth(
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {"type": "image", "source": {"type": "url", "url": "https://x/i.png"}}
                        ],
                    }
                ]
            ),
            id="nested-image",
        ),
        pytest.param(
            anth(
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "tool_result",
                                "tool_use_id": "t1",
                                "content": [
                                    {"type": "image", "source": {"type": "url", "url": "https://x"}}
                                ],
                            }
                        ],
                    }
                ]
            ),
            id="image-in-tool_result",
        ),
        pytest.param(
            anth(
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": "x", "cache_control": {"type": "persistent"}}
                        ],
                    }
                ]
            ),
            id="cache_control-shape",
        ),
        pytest.param(anth(output_config={"effort": "max"}), id="effort-above-high"),
    ],
)
def test_a_refused_field_never_reaches_the_upstream(rig: Rig, body: dict[str, Any]) -> None:
    _run, token = _running(rig.world, "ep-anth", cost_usd=1, tokens=10_000_000)
    r = post(rig, token, body)
    assert r.status_code == 422 and code(r) == "wsp_request_field_refused", r.text
    assert rig.upstream.calls == []


def test_an_anthropic_beta_header_is_refused(rig: Rig) -> None:
    _run, token = _running(rig.world, "ep-anth", cost_usd=1, tokens=10_000_000)
    r = post(rig, token, anth(), **{"anthropic-beta": "web-fetch-2025-09-10"})
    assert r.status_code == 422 and code(r) == "wsp_request_field_refused"
    assert rig.upstream.calls == []


def test_a_replayed_thinking_turn_with_cache_control_on_a_tool_result_passes(rig: Rig) -> None:
    """R2A-5, R3A-6: what weightspace sends on its second turn."""
    _run, token = _running(rig.world, "ep-anth", cost_usd=1, tokens=10_000_000)
    body = anth(
        system="sys",
        tools=[{"name": "run", "description": "d", "input_schema": {"type": "object"}}],
        messages=[
            {"role": "user", "content": [{"type": "text", "text": "task"}]},
            {
                "role": "assistant",
                "content": [
                    {"type": "thinking", "thinking": "hmm", "signature": "sig"},
                    {"type": "text", "text": "calling"},
                    {"type": "tool_use", "id": "t1", "name": "run", "input": {"code": "1"}},
                ],
            },
            {
                "role": "user",
                "content": [
                    {
                        "type": "tool_result",
                        "tool_use_id": "t1",
                        "content": "out",
                        "cache_control": {"type": "ephemeral"},
                    }
                ],
            },
        ],
        thinking={"type": "adaptive"},
        output_config={"effort": "high"},
    )
    r = post(rig, token, body)
    assert r.status_code == 200, r.text
    sent = rig.upstream.calls[-1].body
    assert sent["messages"] == body["messages"] and sent["model"] == "claude-editor"


def test_an_openai_content_array_is_refused(rig: Rig) -> None:
    """R3A-17."""
    _run, token = _running(rig.world, "ep-local", tokens=50_000)
    body = oai(
        messages=[
            {
                "role": "user",
                "content": [{"type": "image_url", "image_url": {"url": "https://x/i.png"}}],
            }
        ]
    )
    r = post(rig, token, body, "/v1/chat/completions")
    assert r.status_code == 422 and code(r) == "wsp_request_field_refused"
    assert rig.upstream.calls == []


def test_reasoning_effort_passes_only_for_a_miserver_endpoint(rig: Rig) -> None:
    r1, local = _running(rig.world, "ep-local", tokens=50_000)  # serving: miserver
    assert post(rig, local, oai(reasoning_effort="low"), "/v1/chat/completions").status_code == 200
    _finish(rig.world, r1)
    _r2, ext = _running(rig.world, "ep-ext", cost_usd=1, tokens=10_000_000)  # serving: generic
    r = post(rig, ext, oai(reasoning_effort="low"), "/v1/chat/completions")
    assert r.status_code == 422 and code(r) == "wsp_request_field_refused"


# ------------------------------------------ feature 018: the miServer-only fields (FR-018.9)

ONE_ACTION = {
    "type": "json_schema",
    "json_schema": {
        "name": "one_action",
        "schema": {"type": "object", "properties": {"tool": {"const": "submit"}}},
    },
}


def test_response_format_and_parallel_tool_calls_pass_only_for_a_miserver_endpoint(
    rig: Rig,
) -> None:
    """FR-018.9's AC, after the template above: both fields reach miServer unchanged, and a
    generic endpoint refuses each one ``wsp_request_field_refused`` before any upstream call."""
    r1, local = _running(rig.world, "ep-local", tokens=50_000)  # serving: miserver
    for extra in ({"parallel_tool_calls": False}, {"response_format": ONE_ACTION}):
        r = post(rig, local, oai(**extra), "/v1/chat/completions")
        assert r.status_code == 200, r.text
        ((name, value),) = extra.items()
        assert rig.upstream.calls[-1].body[name] == value
    _finish(rig.world, r1)
    _r2, ext = _running(rig.world, "ep-ext", cost_usd=1, tokens=10_000_000)  # serving: generic
    sent = len(rig.upstream.calls)
    for extra in ({"parallel_tool_calls": False}, {"response_format": ONE_ACTION}):
        r = post(rig, ext, oai(**extra), "/v1/chat/completions")
        assert r.status_code == 422 and code(r) == "wsp_request_field_refused", r.text
        assert next(iter(extra)) in r.json()["error"]["message"]
    assert len(rig.upstream.calls) == sent


@pytest.mark.parametrize(
    "extra",
    [
        {"parallel_tool_calls": True},
        {"parallel_tool_calls": None},
        {"response_format": {"type": "json_object"}},
        {"response_format": {"type": "text"}},
        # C9.3d's survivor: a well-formed json_schema object under another type.
        {"response_format": {"type": "json_object", "json_schema": ONE_ACTION["json_schema"]}},
        {"response_format": {**ONE_ACTION, "extra": 1}},
        {"response_format": {"type": "json_schema", "json_schema": {"name": "x"}}},
        {
            "response_format": {
                "type": "json_schema",
                "json_schema": {**ONE_ACTION["json_schema"], "other": 1},
            }
        },
        {
            "response_format": {
                "type": "json_schema",
                "json_schema": {**ONE_ACTION["json_schema"], "strict": "yes"},
            }
        },
        {"response_format": ONE_ACTION, "tools": []},
        {"response_format": ONE_ACTION, "tool_choice": "none"},
        {"response_format": ONE_ACTION, "stop": "x"},
        {"response_format": "json_schema"},
    ],
    ids=[
        "parallel-true",
        "parallel-null",
        "json-object",
        "text",
        "other-type-with-schema",
        "format-extra-key",
        "no-schema",
        "schema-extra-key",
        "strict-not-bool",
        "with-tools",
        "with-tool-choice",
        "with-stop",
        "not-an-object",
    ],
)
def test_any_other_value_is_refused_for_miserver_too(rig: Rig, extra: dict[str, Any]) -> None:
    """``parallel_tool_calls`` is ``false`` only; ``response_format`` is miServer's ``json_schema``
    shape only, and never beside ``tools`` or ``stop`` (miServer refuses them together)."""
    _run, local = _running(rig.world, "ep-local", tokens=50_000)
    r = post(rig, local, oai(**extra), "/v1/chat/completions")
    assert r.status_code == 422 and code(r) == "wsp_request_field_refused", r.text
    assert rig.upstream.calls == []


class _Relay:
    """The engine's gateway address: a local HTTP server that hands each request, bytes and
    headers as sent, to the rig's REAL gateway app and returns its answer as is."""

    def __init__(self, client: TestClient) -> None:
        from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *_args: Any) -> None:
                return

            def do_POST(self) -> None:
                raw = self.rfile.read(int(self.headers.get("content-length") or 0))
                headers = {
                    k: v
                    for k, v in self.headers.items()
                    if k.lower() not in ("host", "content-length")
                }
                answer = client.post(self.path, content=raw, headers=headers)
                self.send_response(answer.status_code)
                self.send_header("content-type", "application/json")
                self.send_header("content-length", str(len(answer.content)))
                self.end_headers()
                self.wfile.write(answer.content)

        self.server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.port = self.server.server_address[1]
        threading.Thread(target=self.server.serve_forever, daemon=True).start()

    def close(self) -> None:
        self.server.shutdown()
        self.server.server_close()


@pytest.mark.parametrize("protocol", ["structured", "tool_calls"])
def test_an_engine_session_on_miserver_runs_through_the_real_gateway(
    rig: Rig, tmp_path: Any, protocol: str
) -> None:
    """FR-018.7 / FR-018.9, the structured-mode conformance test through the gateway rig: a real
    ``miwsp_engine`` session (the worker's interpreter, the child preamble) talks to the REAL
    gateway for a miServer endpoint. The engine's own bodies pass the allow-list, reach miServer
    with the protocol's field, and the session submits with no refused turn. Without the 9.3
    admission the gateway refuses the first request and the session ends ``error``."""
    from tests.support.wsp_engine import line_for, run_case
    from tests.support.wsp_fixtures import make_bundle

    _run, token = _running(rig.world, "ep-local", tokens=200_000)
    submit = {"answer": "a one-line description", "confidence": 0.5}
    if protocol == "structured":
        rig.upstream.messages = [
            {"role": "assistant", "content": json.dumps({"tool": "submit", "arguments": submit})}
        ]
    else:
        rig.upstream.messages = [
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {
                        "id": "c0",
                        "type": "function",
                        "function": {"name": "submit", "arguments": json.dumps(submit)},
                    }
                ],
            }
        ]
    fx = make_bundle(tmp_path / "bundle", rule="mean")
    relay = _Relay(rig.client)
    try:
        gateway = {"ip": "127.0.0.1", "port": relay.port, "token": token}
        line = line_for(
            tmp_path / "run",
            fx,
            "inspect",
            gateway,
            editor={"action_protocol": protocol},
            tokens=200_000,
        )
        seen = run_case(tmp_path / "run", line)
    finally:
        relay.close()
    assert seen["exit"] == 0, seen.get("stderr", "")[-2000:]
    result = json.loads((tmp_path / "run" / "out" / "result.json").read_text())
    assert (result["stop_reason"], result["failure"]) == ("submitted", None), result
    assert result["engine"]["action_protocol"] == protocol
    assert result["engine"]["refused_turns"] == 0
    (call,) = rig.upstream.calls
    if protocol == "structured":
        assert call.body["response_format"]["type"] == "json_schema"
        assert "tools" not in call.body and "stop" not in call.body
    else:
        assert call.body["parallel_tool_calls"] is False and call.body["tools"]


@pytest.mark.parametrize("protocol", ["structured", "tool_calls"])
def test_a_thinking_engine_session_on_miserver_runs_through_the_real_gateway(
    rig: Rig, tmp_path: Any, protocol: str
) -> None:
    """2026-11-21, the thinking switch through the REAL gateway: the engine's body with
    ``chat_template_kwargs.enable_thinking`` and the reasoning allowance passes the allow-list and
    reaches miServer as sent (the allowance under the inserted cap), and the reasoning miServer
    answers comes back to the engine, which records it and still takes the action only from the
    content or the tool call. Without the gateway's admission of the allowance field the first
    request is refused and the session ends ``error``."""
    from tests.support.wsp_engine import line_for, run_case
    from tests.support.wsp_fixtures import make_bundle

    _run, token = _running(rig.world, "ep-local", tokens=200_000)
    submit = {"answer": "a one-line description", "confidence": 0.5}
    reasoning = "The weight is a mean direction; I can submit."
    if protocol == "structured":
        message: dict[str, Any] = {
            "role": "assistant",
            "content": json.dumps({"tool": "submit", "arguments": submit}),
        }
    else:
        message = {
            "role": "assistant",
            "content": None,
            "tool_calls": [
                {
                    "id": "c0",
                    "type": "function",
                    "function": {"name": "submit", "arguments": json.dumps(submit)},
                }
            ],
        }
    rig.upstream.messages = [{**message, "reasoning_content": reasoning}]
    fx = make_bundle(tmp_path / "bundle", rule="mean")
    relay = _Relay(rig.client)
    try:
        gateway = {"ip": "127.0.0.1", "port": relay.port, "token": token}
        line = line_for(
            tmp_path / "run",
            fx,
            "inspect",
            gateway,
            editor={"action_protocol": protocol, "context_window_tokens": 32768},
            tokens=200_000,
            thinking="on",
            thinking_tokens=3000,
        )
        seen = run_case(tmp_path / "run", line)
    finally:
        relay.close()
    assert seen["exit"] == 0, seen.get("stderr", "")[-2000:]
    result = json.loads((tmp_path / "run" / "out" / "result.json").read_text())
    assert (result["stop_reason"], result["failure"]) == ("submitted", None), result
    assert result["engine"]["reasoning_turns"] == 1 and result["engine"]["refused_turns"] == 0
    (call,) = rig.upstream.calls
    assert call.body["chat_template_kwargs"] == {"enable_thinking": True}
    # miServer's allowance field, beside response_format only.
    assert call.body.get("max_reasoning_tokens") == (3000 if protocol == "structured" else None)
    assert call.body["max_tokens"] == 2048 + 3000  # the endpoint's cap field
    assert reasoning not in seen["stdout"]  # the child log carries named fields only


@pytest.mark.parametrize("name", ["bad\nname", "x" * 100])
def test_an_unsafe_refused_field_name_is_echoed_as_other(rig: Rig, name: str) -> None:
    """R2A-23."""
    _run, token = _running(rig.world, "ep-anth", cost_usd=1, tokens=10_000_000)
    r = post(rig, token, {**anth(), name: 1})
    assert code(r) == "wsp_request_field_refused"
    assert "<other>" in r.json()["error"]["message"] and name not in r.text


def test_streaming_is_refused(rig: Rig) -> None:
    _run, token = _running(rig.world, "ep-anth", cost_usd=1, tokens=10_000_000)
    r = post(rig, token, anth(stream=True))
    assert r.status_code == 422 and code(r) == "wsp_streaming_unsupported"
    assert rig.upstream.calls == []


def test_the_forwarded_header_set_is_exact_and_the_model_overridden(rig: Rig) -> None:
    """R1B-M6: headers built from scratch; the session token never passes upstream."""
    _run, token = _running(rig.world, "ep-anth", cost_usd=1, tokens=10_000_000)
    assert (
        post(rig, token, anth(model="claude-other"), **{"x-stainless-lang": "python"}).status_code
        == 200
    )
    call = rig.upstream.calls[-1]
    assert set(call.headers) == {"content-type", "anthropic-version", "x-api-key"} | HTTPX_OWN
    assert call.headers["x-api-key"] == KEY and token not in json.dumps(call.headers)
    assert SERVICE not in json.dumps(call.headers)
    assert call.body["model"] == "claude-editor"
    assert call.url == "https://api.anthropic.com/v1/messages"


#: What httpx itself adds to every request it sends (never copied from the incoming request).
HTTPX_OWN = {"host", "accept", "accept-encoding", "connection", "user-agent", "content-length"}


# -------------------------------------------------------------------------- sessions and endpoints


def test_token_a_cannot_reach_endpoint_b(rig: Rig) -> None:
    """The endpoint is the session's, never the request's; an ended session's token is dead."""
    a, token_a = _running(rig.world, "ep-local", tokens=50_000)
    r = post(rig, token_a, oai(model="claude-editor"), "/v1/chat/completions")
    assert r.status_code == 200
    assert rig.upstream.calls[-1].url.startswith(f"http://{MISERVER_IP}:8000/")
    assert rig.upstream.calls[-1].body["model"] == "m"
    assert post(rig, token_a, anth()).status_code == 422  # A's endpoint does not serve /messages
    _finish(rig.world, a)
    _b, token_b = _running(rig.world, "ep-anth", cost_usd=1, tokens=10_000_000)
    rig.restart()  # a fresh registry: A must be resolved again, and is not live
    refused = post(rig, token_a, oai(), "/v1/chat/completions")
    assert refused.status_code == 401 and code(refused) == "wsp_gateway_token_invalid"
    assert post(rig, token_b, anth()).status_code == 200
    assert rig.upstream.calls[-1].url == "https://api.anthropic.com/v1/messages"


def test_an_unknown_token_is_refused(rig: Rig) -> None:
    _running(rig.world, "ep-anth", cost_usd=1, tokens=10_000_000)
    r = post(rig, "not-a-token", anth())
    assert r.status_code == 401 and code(r) == "wsp_gateway_token_invalid"


def test_a_sensitive_session_is_refused_even_with_a_forged_registry(rig: Rig) -> None:
    """The locality is re-derived from the host, so a row or registry saying ``local`` is not
    believed; and a session the registry marks sensitive never reaches an external endpoint."""
    _run, token = _running(rig.world, "ep-anth", cost_usd=1, tokens=10_000_000)
    assert post(rig, token, anth()).status_code == 200
    (session,) = list(rig.gateway.registry._sessions.values())
    session.sensitivity = "sensitive"
    real = rig.gateway.registry.endpoint
    rig.gateway.registry.endpoint = lambda eid: {**real(eid), "locality": "local"}  # type: ignore[method-assign]
    calls = len(rig.upstream.calls)
    r = post(rig, token, anth())
    assert r.status_code == 403 and code(r) == "wsp_sensitive_data_external_endpoint"
    assert len(rig.upstream.calls) == calls


def test_an_external_host_outside_the_allowlist_is_refused(rig: Rig) -> None:
    _run, token = _running(rig.world, "ep-anth", cost_usd=1, tokens=10_000_000)
    rig.overrides = {"wsp_external_host_allowlist": "api.example.org"}
    rig.restart()
    r = post(rig, token, anth())
    assert r.status_code == 422 and code(r) == "wsp_endpoint_url_refused"
    assert rig.upstream.calls == []


def test_an_external_endpoint_edited_to_plain_http_is_refused_in_the_gateway(rig: Rig) -> None:
    """Review OR2-016-6: registration refuses ``http`` for an external endpoint, and the gateway
    re-checks it at call time, before the key is opened (the sealed key's AAD binds the URL, a
    second layer). Control ``S2-gw-ext-http`` (the gateway's ``https`` check removed) turns this
    red."""
    _run, token = _running(rig.world, "ep-anth", cost_usd=1, tokens=10_000_000)
    with rig.world.engine.begin() as conn:
        conn.execute(
            sa.text(
                "UPDATE mf_wsp_endpoints SET base_url = 'http://api.anthropic.com' "
                "WHERE id = 'ep-anth'"
            )
        )
    r = post(rig, token, anth())
    assert r.status_code == 422 and code(r) == "wsp_endpoint_url_refused", r.text
    assert rig.upstream.calls == []


def test_a_row_saying_external_is_external_even_with_a_local_url(rig: Rig) -> None:
    """Review OR2-016-6: the locality is ``external`` if EITHER the stored column or the host
    says so. A local URL whose row says ``external`` is held to the external rules (https and
    the external allow-list), so it is refused here. Control ``S3-gw-locality-either`` (the
    gateway trusting ``derive_locality`` alone) turns this red."""
    _run, token = _running(rig.world, "ep-local", tokens=50_000)
    with rig.world.engine.begin() as conn:
        conn.execute(
            sa.text(
                "UPDATE mf_wsp_endpoints SET locality = 'external', price_in_per_mtok = 1, "
                "price_out_per_mtok = 1 WHERE id = 'ep-local'"
            )
        )
    r = post(rig, token, oai(), "/v1/chat/completions")
    assert r.status_code == 422 and code(r) == "wsp_endpoint_url_refused", r.text
    assert rig.upstream.calls == []


def test_a_stored_base_url_path_is_refused_in_the_gateway(rig: Rig) -> None:
    """P61: the path rule is re-checked at call time, not only at registration (C16e)."""
    _run, token = _running(rig.world, "ep-local", tokens=50_000)
    with rig.world.engine.begin() as conn:
        conn.execute(
            sa.text(
                "UPDATE mf_wsp_endpoints SET base_url = "
                "'http://miserver-backend.miserver.svc:8000/api/v1/admin' WHERE id = 'ep-local'"
            )
        )
    r = post(rig, token, oai(), "/v1/chat/completions")
    assert r.status_code == 422 and code(r) == "wsp_endpoint_url_refused"
    assert rig.upstream.calls == []


def test_a_local_host_resolving_outside_private_ranges_is_refused(rig: Rig) -> None:
    """P61, R2A-10: the connect-time check, at the resolved address."""
    _run, token = _running(rig.world, "ep-local", tokens=50_000)
    rig.gateway.resolver = lambda host: ["93.184.216.34"]
    r = post(rig, token, oai(), "/v1/chat/completions")
    assert r.status_code == 422 and code(r) == "wsp_endpoint_url_refused"
    assert rig.upstream.calls == []


@pytest.mark.parametrize(
    "answer",
    [
        ["10.0.0.5", "93.184.216.34"],
        ["93.184.216.34", "10.0.0.5"],
        ["169.254.169.254"],
        ["10.0.0.5", "169.254.169.254"],
        ["0.0.0.0"],
        ["::ffff:93.184.216.34"],
        ["fe80::1"],
        ["not-an-address"],
    ],
    ids=[
        "private-first",
        "public-first",
        "metadata",
        "private-and-metadata",
        "this-network",
        "v4-mapped-public",
        "v6-link-local",
        "garbage",
    ],
)
def test_a_local_host_with_any_address_outside_the_ranges_is_refused(
    rig: Rig, answer: list[str]
) -> None:
    """Review OR2-016-2 (P61, R2A-10): EVERY resolved address must be in the cluster or RFC 1918
    ranges. A mixed answer is refused in both orders (with ``any`` the private-first answer
    passed and the gateway called the public address with the session's body). Link-local (a
    cloud metadata service), ``0.0.0.0`` and an IPv4-mapped public address are not local either.
    Controls ``S1-gw-dns-any`` and ``C-OR2-2-is-private`` turn this red."""
    _run, token = _running(rig.world, "ep-local", tokens=50_000)
    rig.gateway.resolver = lambda host: list(answer)
    r = post(rig, token, oai(), "/v1/chat/completions")
    assert r.status_code == 422 and code(r) == "wsp_endpoint_url_refused", r.text
    assert rig.upstream.calls == []


@pytest.mark.parametrize(
    "answer",
    [["10.0.0.5"], ["172.16.4.4", "192.168.1.9"], ["127.0.0.1"], ["fd00::7"], ["::ffff:10.0.0.5"]],
    ids=["rfc1918", "two-rfc1918", "loopback", "v6-unique-local", "v4-mapped-private"],
)
def test_a_local_host_inside_the_ranges_is_called(rig: Rig, answer: list[str]) -> None:
    """The positive control for the refusal above: the same request, every address local."""
    _run, token = _running(rig.world, "ep-local", tokens=50_000)
    rig.gateway.resolver = lambda host: list(answer)
    r = post(rig, token, oai(), "/v1/chat/completions")
    assert r.status_code == 200, r.text
    assert len(rig.upstream.calls) == 1


def test_the_call_goes_to_the_checked_address_without_a_second_resolution(rig: Rig) -> None:
    """DNS rebinding (review OR2-016-2): the resolver answers a private address once and a public
    one after that. The gateway resolves once per request and calls the address it checked, by
    IP, with the name in the Host header, so the later answer is never used. Controls
    ``C-OR2-2-reresolve`` (a second resolution for the call) and ``C-OR2-2-by-name`` (the call
    made to the name) turn this red."""
    _run, token = _running(rig.world, "ep-local", tokens=50_000)
    answers = [["10.0.0.5"]]
    asked: list[str] = []

    def rebinding(host: str) -> list[str]:
        asked.append(host)
        return answers.pop(0) if answers else ["93.184.216.34"]

    rig.gateway.resolver = rebinding
    r = post(rig, token, oai(), "/v1/chat/completions")
    assert r.status_code == 200, r.text
    assert asked == ["miserver-backend.miserver.svc"]
    (call,) = rig.upstream.calls
    assert call.url == "http://10.0.0.5:8000/v1/chat/completions"
    assert call.headers["host"] == "miserver-backend.miserver.svc:8000"


@pytest.mark.parametrize(
    "column, value",
    [
        ("base_url", "https://api.anthropic.com/v1"),
        ("model", "claude-cheaper"),
        ("price_out_per_mtok", 0),
        ("price_cache_read_per_mtok", 0.1),
        ("serving", "miserver"),
    ],
)
def test_a_changed_row_no_longer_opens_its_key(rig: Rig, column: str, value: Any) -> None:
    """R1B-M8, R2A-9: every routing and price field is bound into the associated data."""
    _run, token = _running(rig.world, "ep-anth", cost_usd=1, tokens=10_000_000)
    with rig.world.engine.begin() as conn:
        conn.execute(
            sa.text(f"UPDATE mf_wsp_endpoints SET {column} = :v WHERE id = 'ep-anth'"), {"v": value}
        )
    r = post(rig, token, anth())
    assert r.status_code == 409 and code(r) == "wsp_endpoint_key_binding_failed"
    assert rig.upstream.calls == []


def test_an_expired_session_is_refused(rig: Rig) -> None:
    _run_id, token = _running(rig.world, "ep-anth", cost_usd=1, tokens=10_000_000)
    rig.gateway.clock = lambda: gateway_ledger.utcnow().replace(year=2099)
    r = post(rig, token, anth())
    assert r.status_code == 401 and code(r) == "wsp_session_token_invalid"


def test_a_token_redated_after_the_gateway_cached_it_is_served(rig: Rig) -> None:
    """Fix 2026-11-21: the driver re-dates the token when the push completes, which may be after
    the gateway resolved and cached it. An expiry the gateway holds as passed is read again from
    the backend once; the token is refused only if the backend's expiry has passed too."""
    run_id, token = _running(rig.world, "ep-anth", cost_usd=1, tokens=10_000_000)
    assert post(rig, token, anth()).status_code == 200  # resolved and cached: now + 1 h
    later = gateway_ledger.utcnow() + dt.timedelta(hours=2)
    with rig.world.engine.begin() as conn:
        conn.execute(
            sa.text("UPDATE mf_wsp_runs SET token_expires_at = :exp WHERE run_id = :r"),
            {"exp": later + dt.timedelta(hours=1), "r": run_id},
        )
    rig.gateway.clock = lambda: later
    with patch.object(gateway_ledger, "utcnow", lambda: later):
        assert post(rig, token, anth()).status_code == 200
    rig.gateway.clock = lambda: later + dt.timedelta(hours=2)
    with patch.object(gateway_ledger, "utcnow", lambda: later + dt.timedelta(hours=2)):
        r = post(rig, token, anth())
    assert r.status_code == 401 and code(r) == "wsp_session_token_invalid"


def _settled(rig: Rig, run_id: str) -> list[tuple[Any, Any]]:
    """``(worst_case_usd, cost_usd)`` for a run with exactly one reservation and one usage row."""
    with rig.world.engine.connect() as conn:
        worst = (
            conn.execute(
                sa.text("SELECT worst_case_usd FROM mf_wsp_reservations WHERE run_id = :r"),
                {"r": run_id},
            )
            .scalars()
            .all()
        )
        cost = (
            conn.execute(
                sa.text("SELECT cost_usd FROM mf_wsp_usage WHERE run_id = :r"), {"r": run_id}
            )
            .scalars()
            .all()
        )
    return list(zip(worst, cost, strict=True))


@pytest.mark.parametrize(
    "usage",
    ["absent", "not an object", 'raw:{"input_tokens": 1e999, "output_tokens": 5}'],
    ids=["absent", "not-an-object", "non-finite"],
)
def test_an_answer_without_usable_usage_settles_at_the_worst_case(rig: Rig, usage: Any) -> None:
    """Review OR1-016-8: a 2xx answer that may have been billed is never metered at zero."""
    run_id, token = _running(rig.world, "ep-anth", cost_usd=1, tokens=10_000_000)
    rig.upstream.usage = usage
    r = post(rig, token, anth())
    assert r.status_code == 200, r.text
    [(worst, cost)] = _settled(rig, run_id)
    assert Decimal(str(cost)) == Decimal(str(worst)) > 0


def test_negative_usage_counts_do_not_refund_spend(rig: Rig) -> None:
    run_id, token = _running(rig.world, "ep-anth", cost_usd=1, tokens=10_000_000)
    rig.upstream.usage = {"input_tokens": -1_000_000, "output_tokens": 10}
    assert post(rig, token, anth()).status_code == 200
    [(_worst, cost)] = _settled(rig, run_id)
    assert Decimal(str(cost)) == Decimal(10 * 15) / Decimal(1_000_000)


# ------------------------------------------------------------------------ withheld and redacted


def test_a_failed_settle_withholds_the_response_and_ends_the_session(rig: Rig) -> None:
    _run_id, token = _running(rig.world, "ep-anth", cost_usd=1, tokens=10_000_000)
    real = rig.gateway.ledger.settle

    def broken(reservation: int, usage: Any, call: Any = None) -> None:
        from wsp_gateway.meter import LedgerRefused

        raise LedgerRefused(None, "down")

    rig.gateway.ledger.settle = broken  # type: ignore[method-assign]
    r = post(rig, token, anth())
    assert r.status_code == 503 and code(r) == "wsp_gateway_ledger_unavailable"
    assert '"ok"' not in r.text  # the upstream answer is withheld
    rig.gateway.ledger.settle = real  # type: ignore[method-assign]
    again = post(rig, token, anth())
    assert again.status_code == 409 and code(again) == "wsp_session_not_active"
    assert len(rig.upstream.calls) == 1


def test_secrets_appear_in_no_answer_and_no_log_record(
    rig: Rig, caplog: pytest.LogCaptureFixture
) -> None:
    """R1B-M22: header-shaped fields (``x-api-key``, ``authorization``) on every error path."""
    caplog.set_level(logging.DEBUG)
    _run, token = _running(rig.world, "ep-anth", cost_usd=1, tokens=10_000_000)
    answers = [
        post(rig, token, anth()),  # success: the key is opened and cached
        post(rig, token, anth(service_tier="x")),  # a refusal
        rig.client.post(
            "/v1/messages", content=json.dumps(anth()), headers={"authorization": f"Bearer {token}"}
        ),
        post(rig, KEY, anth()),  # the key presented as a token
    ]
    rig.upstream.fail = True
    answers.append(post(rig, token, anth()))  # the upstream fails
    rig.upstream.fail = False
    answers.append(post(rig, token, anth(), **{"x-millm-strict": KEY}))
    text = (
        caplog.text
        + "".join(a.text for a in answers)
        + json.dumps([dict(a.headers) for a in answers])
    )
    for secret in (KEY, token, SERVICE):
        assert secret not in text
    assert any(r.name == "wsp_gateway" for r in caplog.records)  # the scan read the gateway's log
    # The value filter itself: a gateway log line that DID carry a secret comes out scrubbed
    # (control C15-valuefilter; a redaction by key name would miss the header-shaped forms).
    caplog.clear()
    logging.getLogger("wsp_gateway").warning("x-api-key: %s authorization: Bearer %s", KEY, token)
    assert caplog.records and KEY not in caplog.text and token not in caplog.text


# ---------------------------------------------------------------------------- miServer (P28, P47)


def test_a_miserver_call_is_unsteered_and_carries_the_p28_headers(rig: Rig) -> None:
    _run, token = _running(rig.world, "ep-local", tokens=50_000)
    r = post(rig, token, oai(), "/v1/chat/completions")
    assert r.status_code == 200
    call = rig.upstream.calls[-1]
    assert call.headers["x-millm-origin"] == "judge"
    assert call.headers["x-millm-load-policy"] == "refuse"
    assert "x-millm-strict" not in call.headers and "logit_bias" not in call.body
    assert call.body["steering"] == {"features": []}
    assert "profile" not in call.body and "steering_intensity" not in call.body
    assert call.headers["host"] == "miserver-backend.miserver.svc:8000"
    (session,) = list(rig.gateway.registry._sessions.values())
    assert session.served_models == [rig.upstream.served_model] and session.steering == ["none"]


@pytest.mark.parametrize(
    "sent", [{"logit_bias": {"1": 5}}, {"steering_intensity": 0}, {"profile": "p"}]
)
def test_a_miserver_steering_or_bias_field_is_refused(rig: Rig, sent: dict[str, Any]) -> None:
    _run, token = _running(rig.world, "ep-local", tokens=50_000)
    r = post(rig, token, oai(**sent), "/v1/chat/completions")
    assert code(r) == "wsp_request_field_refused" and rig.upstream.calls == []


@pytest.mark.parametrize(
    "error, failure",
    [
        ((409, "model_not_resident"), "editor_model_not_resident"),
        ((400, "context_length_exceeded"), "editor_context_exceeded"),
        ((503, "model_loading"), "editor_model_loading"),
    ],
)
def test_a_miserver_session_error_ends_the_session_after_one_call(
    rig: Rig, error: tuple[int, str], failure: str
) -> None:
    run_id, token = _running(rig.world, "ep-local", tokens=50_000)
    rig.upstream.error = error
    r = post(rig, token, oai(), "/v1/chat/completions")
    assert code(r) == failure
    # The message names the code too: an SDK's error string keeps only the message.
    assert r.json()["error"]["message"].endswith(f"(code: {failure})")
    assert code(post(rig, token, oai(), "/v1/chat/completions")) == "wsp_session_not_active"
    # A context refusal is sent once more at a smaller cap first (fix 2026-11-21); the second
    # refusal ends the session. The others end it after one call.
    calls = 2 if failure == "editor_context_exceeded" else 1
    assert len(rig.upstream.calls) == calls and rig.usage_rows(run_id) == calls


def test_a_steered_answer_is_withheld_and_ends_the_session(rig: Rig) -> None:
    _run, token = _running(rig.world, "ep-local", tokens=50_000)
    rig.upstream.steering = "profile:abc"
    r = post(rig, token, oai(), "/v1/chat/completions")
    assert r.status_code == 409 and code(r) == "editor_steered" and '"ok"' not in r.text


def test_the_edit_target_is_refused_before_any_call(rig: Rig) -> None:
    _run, token = _running(rig.world, "ep-local", tokens=50_000)
    assert post(rig, token, oai(), "/v1/chat/completions").status_code == 200
    (session,) = list(rig.gateway.registry._sessions.values())
    session.edit_target_model = "m"  # set for checkpoint bundles only (PF-34)
    calls = len(rig.upstream.calls)
    r = post(rig, token, oai(), "/v1/chat/completions")
    assert code(r) == "wsp_editor_is_edit_target" and len(rig.upstream.calls) == calls


# ------------------------------------------------------------------------------------ the codes


def test_shared_codes_carry_the_backends_status() -> None:
    shared = set(STATUS) & set(REFUSAL_STATUS)
    assert {"wsp_request_field_refused", "wsp_budget_exhausted", "wsp_session_not_active"} <= shared
    assert {c: STATUS[c] for c in shared} == {c: REFUSAL_STATUS[c] for c in shared}


def test_the_gateway_serves_three_routes_only() -> None:
    import httpx

    gateway = Gateway(
        GatewaySettings(), internal=httpx.Client(), upstream_client=FakeUpstream().client()
    )
    routes = {
        (method, route.path)
        for route in create_gateway_app(gateway).routes
        for method in getattr(route, "methods", ())
    }
    assert routes == {
        ("GET", "/health"),
        ("POST", "/v1/messages"),
        ("POST", "/v1/chat/completions"),
    }


def test_base64_ciphertext_is_all_the_internal_app_serves(rig: Rig) -> None:
    doc = rig.internal.get("/wsp-internal/v1/gateway/endpoints/ep-anth").json()
    blob = base64.b64decode(doc["key_ciphertext"])
    assert KEY.encode() not in blob and KEY not in json.dumps(doc)
    assert sealing.open_sealed(rig.private, blob, sealing.aad_digest(doc)) == KEY.encode()


# --------------------------------------------- R3 review fix (OR3-016-4, -6), 2026-11-16


def test_a_local_base_url_edited_to_carry_credentials_is_refused_in_the_gateway(rig: Rig) -> None:
    """OR3-016-6 (``SW3-up-userinfo``): the gateway re-checks a stored row's ``base_url`` for
    userinfo at call time. A local endpoint registered without a key has no sealed key, so nothing
    else binds its URL."""
    _run, token = _running(rig.world, "ep-local", tokens=50_000)
    with rig.world.engine.begin() as conn:
        conn.execute(
            sa.text(
                "UPDATE mf_wsp_endpoints SET base_url = "
                "'http://u:p@miserver-backend.miserver.svc:8000' WHERE id = 'ep-local'"
            )
        )
    r = post(rig, token, oai(), "/v1/chat/completions")
    assert r.status_code == 422 and code(r) == "wsp_endpoint_url_refused", r.text
    assert rig.upstream.calls == []


def test_a_local_host_edited_outside_the_allowlist_is_refused_in_the_gateway(rig: Rig) -> None:
    """OR3-016-6 (``SW5-up-local-allowlist``): a local host must be in WSP_LOCAL_HOST_ALLOWLIST at
    call time too, so an edited row cannot reach another in-cluster service. The host resolves
    to an in-cluster address here, so only the allow-list can refuse it (control ``SW5``)."""
    _run, token = _running(rig.world, "ep-local", tokens=50_000)
    rig.gateway.resolver = lambda host: [MISERVER_IP]
    with rig.world.engine.begin() as conn:
        conn.execute(
            sa.text(
                "UPDATE mf_wsp_endpoints SET base_url = "
                "'http://miforge-backend.miforge.svc:8000/v1' WHERE id = 'ep-local'"
            )
        )
    r = post(rig, token, oai(), "/v1/chat/completions")
    assert r.status_code == 422 and code(r) == "wsp_endpoint_url_refused", r.text
    assert rig.upstream.calls == []


class _Redirector:
    """A local server that answers 307 to ``/v1/messages`` and ``/x``, and records each hit."""

    def __init__(self) -> None:
        from http.server import BaseHTTPRequestHandler, HTTPServer

        hits = self.hits = []

        class Handler(BaseHTTPRequestHandler):
            def _answer(self) -> None:
                hits.append(self.path)
                self.rfile.read(int(self.headers.get("content-length", 0)))
                self.send_response(307 if self.path in ("/v1/messages", "/x") else 200)
                self.send_header("Location", "/elsewhere")
                self.send_header("content-length", "0")
                self.end_headers()

            do_GET = do_POST = _answer  # noqa: N815

            def log_message(self, *args: Any) -> None:
                pass

        self.server = HTTPServer(("127.0.0.1", 0), Handler)
        self.port = self.server.server_port
        threading.Thread(target=self.server.serve_forever, daemon=True).start()

    def close(self) -> None:
        self.server.shutdown()
        self.server.server_close()


def test_the_production_clients_follow_no_redirect_and_ignore_proxy_env(monkeypatch) -> None:
    """OR3-016-4 (``SW1``/``SW2``): ``python -m wsp_gateway``'s own ``build()`` (the production
    wiring, not a client the test builds) never follows a redirect (a 307 would re-send the
    session's body to the ``Location``) and never takes a proxy from the environment, for the
    upstream client and the internal one. ``HTTP_PROXY`` points at a dead port: a client that
    honoured it would fail instead of reaching the local server."""
    from wsp_gateway.__main__ import build

    server = _Redirector()
    monkeypatch.setenv("HTTP_PROXY", "http://127.0.0.1:9")
    monkeypatch.setenv("http_proxy", "http://127.0.0.1:9")
    monkeypatch.delenv("NO_PROXY", raising=False)
    monkeypatch.delenv("no_proxy", raising=False)
    try:
        gateway = build(GatewaySettings(wsp_internal_url=f"http://127.0.0.1:{server.port}"))
        up = gateway.upstream.post(f"http://127.0.0.1:{server.port}/v1/messages", content=b"b")
        internal = gateway.registry._internal.get("/x")
    finally:
        server.close()
    assert up.status_code == 307 and internal.status_code == 307
    assert server.hits == ["/v1/messages", "/x"]


# -------------------------------- fix 2026-11-21: a refused call is not charged (wsp_c3e954d3f21f)


def _usage_row(rig: Rig, run_id: str) -> dict[str, Any]:
    with rig.world.engine.connect() as conn:
        rows = conn.execute(
            sa.text(
                "SELECT input_tokens, output_tokens, cost_usd FROM mf_wsp_usage WHERE run_id = :r"
            ),
            {"r": run_id},
        ).mappings()
        (row,) = [dict(r) for r in rows]
    return row


@pytest.mark.parametrize(
    "endpoint, path, body",
    [("ep-anth", "/v1/messages", "anth"), ("ep-ext", "/v1/chat/completions", "oai")],
    ids=["anthropic", "openai"],
)
@pytest.mark.parametrize("status", [400, 404, 422, 429])
def test_a_refused_call_with_no_usage_settles_at_zero(
    rig: Rig, endpoint: str, path: str, body: str, status: int
) -> None:
    """Live run wsp_c3e954d3f21f: an upstream 400 that generated nothing was settled at the
    reservation (9,421 output tokens) and the run's usage read as if it had been served."""
    run_id, token = _running(rig.world, endpoint, cost_usd=1, tokens=10_000_000)
    rig.upstream.error = (status, "invalid_request_error")
    r = post(rig, token, anth() if body == "anth" else oai(), path)
    assert r.status_code == status
    row = _usage_row(rig, run_id)
    assert (row["input_tokens"], row["output_tokens"]) == (0, 0)
    assert Decimal(str(row["cost_usd"])) == 0
    [(worst, _cost)] = _settled(rig, run_id)
    assert Decimal(str(worst)) > 0  # it WAS reserved at the worst case, then released


def test_a_refused_call_that_reports_usage_settles_at_it(rig: Rig) -> None:
    run_id, token = _running(rig.world, "ep-anth", cost_usd=1, tokens=10_000_000)
    rig.upstream.error = (400, "invalid_request_error")
    rig.upstream.error_usage = {"input_tokens": 100, "output_tokens": 0}
    assert post(rig, token, anth()).status_code == 400
    row = _usage_row(rig, run_id)
    assert (row["input_tokens"], row["output_tokens"]) == (100, 0)


@pytest.mark.parametrize("status", [500, 502, 503])
def test_a_server_error_with_no_usage_keeps_the_worst_case(rig: Rig, status: int) -> None:
    """The control: a 5xx may have been generated and billed, so it stays at the worst case."""
    run_id, token = _running(rig.world, "ep-anth", cost_usd=1, tokens=10_000_000)
    rig.upstream.error = (status, "overloaded_error")
    post(rig, token, anth())
    [(worst, cost)] = _settled(rig, run_id)
    assert Decimal(str(cost)) == Decimal(str(worst)) > 0


@pytest.mark.parametrize(
    "failure, charged",
    [("connect", False), ("connect_timeout", False), ("read_timeout", True), ("dropped", True)],
)
def test_only_a_request_that_was_sent_and_lost_keeps_the_worst_case(
    rig: Rig, failure: str, charged: bool
) -> None:
    """A connection never made carried no request (zero); a timeout or a dropped connection
    after the request was sent is an unknown outcome (the worst case)."""
    run_id, token = _running(rig.world, "ep-anth", cost_usd=1, tokens=10_000_000)
    rig.upstream.fail = failure
    r = post(rig, token, anth())
    assert code(r) == "wsp_gateway_upstream_failed"
    [(worst, cost)] = _settled(rig, run_id)
    assert (Decimal(str(cost)) == Decimal(str(worst))) is charged
    assert (Decimal(str(cost)) == 0) is not charged


# ------------------------------------- 2026-11-21: the served window reaches the uncapped default


@pytest.mark.parametrize(
    ("declared", "served", "window"),
    [(None, 32_768, 32_768), (16_384, 32_768, 16_384), (None, None, 8_192)],
    ids=["served", "endpoint-first", "default"],
)
def test_an_uncapped_request_is_sized_to_the_window_miserver_serves(
    rig: Rig, declared: int | None, served: int | None, window: int
) -> None:
    """Endpoint ``miserver-qwen36-27b`` states no window and miServer serves 32,768: with the
    output cap off, a request naming no cap was sized to the 8,192 fallback, so a 19k prompt
    could get no reply. The window read at push (``mf_wsp_runs.served_context_tokens``) now
    reaches the gateway's session, after the endpoint's own window."""
    run_id, token = _running(rig.world, "ep-local", tokens=200_000)
    with rig.world.engine.begin() as conn:
        conn.execute(
            sa.text("UPDATE mf_wsp_runs SET served_context_tokens = :w WHERE run_id = :r"),
            {"w": served, "r": run_id},
        )
        conn.execute(
            sa.text("UPDATE mf_wsp_endpoints SET context_window_tokens = :w WHERE id = 'ep-local'"),
            {"w": declared},
        )
    from wsp_gateway import meter

    big = oai(messages=[{"role": "user", "content": "x" * 6_000}])
    assert post(rig, token, big, "/v1/chat/completions").status_code == 200
    cap = rig.upstream.calls[-1].body["max_tokens"]
    low, high = _first_call_bounds(rig, window)
    assert meter.MIN_OUTPUT_TOKENS <= low <= cap <= high


# ------------------- 2026-11-21: the uncapped cap from the upstream's own counts, and the retry


def _served(rig: Rig, run_id: str, window: int) -> None:
    with rig.world.engine.begin() as conn:
        conn.execute(
            sa.text("UPDATE mf_wsp_runs SET served_context_tokens = :w WHERE run_id = :r"),
            {"w": window, "r": run_id},
        )


def _timeline(rig: Rig, run_id: str) -> list[tuple[Any, ...]]:
    with rig.world.engine.connect() as conn:
        return [
            tuple(row)
            for row in conn.execute(
                sa.text(
                    "SELECT outcome, http_status, refusal_code, input_tokens, output_tokens "
                    "FROM mf_wsp_usage WHERE run_id = :r ORDER BY seq"
                ),
                {"r": run_id},
            )
        ]


def _grow(*contents: str) -> dict[str, Any]:
    """One conversation, growing by appended messages."""
    messages: list[dict[str, Any]] = []
    for n, text in enumerate(contents):
        messages.append({"role": "user" if n % 2 == 0 else "assistant", "content": text})
    return oai(messages=messages)


def test_a_long_session_is_sized_from_the_upstreams_counts_not_its_bytes(rig: Rig) -> None:
    """Run ``wsp_79660d1613a1`` through the real gateway: miServer serves 32,768 tokens and its
    tokenizer counts about 0.6 tokens per body byte. A session at about 31,000 bytes, grown to
    about 33,000, got the 256-token floor when every byte counted as a token (its reply was cut
    at 192 tokens, no tool call, and the session ended); sized from the counts it gets about
    9,500, which the true prompt leaves room for."""
    from wsp_gateway import meter

    run_id, token = _running(rig.world, "ep-local", tokens=10_000_000)
    _served(rig, run_id, 32_768)
    rig.upstream.tokens_per_byte, rig.upstream.window = 0.6, 32_768
    assert post(rig, token, _grow("x" * 30_900), "/v1/chat/completions").status_code == 200
    first = rig.upstream.calls[-1]
    grown = _grow("x" * 30_900, "ok", "y" * 1_900)
    assert post(rig, token, grown, "/v1/chat/completions").status_code == 200
    second = rig.upstream.calls[-1]
    cap, true_prompt = second.body["max_tokens"], rig.upstream.prompt_tokens(second.raw)
    counted = rig.upstream.prompt_tokens(first.raw)
    # The last count plus the new bytes at a token each, with the margin and the template.
    low = 32_768 - meter.margined(counted + len(second.raw) + 100 - len(first.raw))
    high = 32_768 - meter.margined(counted + len(second.raw) - len(first.raw))
    assert low <= cap <= high and cap > 9_000
    assert true_prompt + cap <= 32_768 and _timeline(rig, run_id)[-1][0] == "served"
    # Every byte a token: the floor.
    assert 32_768 - len(second.raw) - meter.TEMPLATE_TOKENS < meter.MIN_OUTPUT_TOKENS


def test_the_reservation_still_counts_every_byte_as_a_token(rig: Rig) -> None:
    """Only the window's room uses the estimate: the cost and token reservation of an uncapped
    request stays the worst case, every forwarded byte a token plus the whole cap."""
    run_id, token = _running(rig.world, "ep-local", tokens=10_000_000)
    _served(rig, run_id, 32_768)
    rig.upstream.tokens_per_byte = 0.6
    assert post(rig, token, _grow("x" * 20_000), "/v1/chat/completions").status_code == 200
    call = rig.upstream.calls[-1]
    with rig.world.engine.connect() as conn:
        worst = conn.execute(
            sa.text("SELECT worst_case_tokens FROM mf_wsp_reservations WHERE run_id = :r"),
            {"r": run_id},
        ).scalar_one()
    assert worst == len(call.raw) + call.body["max_tokens"]


def test_a_context_overflow_is_retried_once_at_the_room_the_refusal_names(rig: Rig) -> None:
    """Fix 2026-11-21: the first call's estimate (0.75 tokens per byte) is short of a tokenizer
    that counts a token per byte, so miServer's transformers engine refuses prompt + cap past its
    32,768 tokens, naming the prompt. The gateway sends it once more at the room between them,
    the editor gets that answer, and the session goes on; the refused attempt is on the call
    timeline (``editor_context_retried``), settled at zero."""
    run_id, token = _running(rig.world, "ep-local", tokens=10_000_000)
    _served(rig, run_id, 32_768)
    rig.upstream.window = 32_768
    r = post(rig, token, _grow("x" * 20_000), "/v1/chat/completions")
    assert r.status_code == 200, r.text
    refused, retried = rig.upstream.calls
    assert retried.body["max_tokens"] == 32_768 - rig.upstream.prompt_tokens(refused.raw)
    assert retried.body["max_tokens"] < refused.body["max_tokens"]
    assert _timeline(rig, run_id) == [
        ("upstream_error", 400, "editor_context_retried", 0, 0),
        ("served", 200, None, len(retried.raw), retried.body["max_tokens"]),
    ]
    # Not ended: the next call is served.
    assert post(rig, token, _grow("x" * 20_000, "ok", "z"), "/v1/chat/completions").status_code
    assert _timeline(rig, run_id)[-1][0] == "served"


def test_a_second_overflow_ends_the_session(rig: Rig) -> None:
    """miServer's llama.cpp engine refuses a prompt that fills its window whatever the cap: the
    retry at half the cap is refused too, and the session ends ``editor_context_exceeded`` (the
    budget-like stop) after exactly two calls."""
    run_id, token = _running(rig.world, "ep-local", tokens=10_000_000)
    _served(rig, run_id, 8_192)
    rig.upstream.window, rig.upstream.engine = 8_192, "llamacpp"
    r = post(rig, token, _grow("x" * 9_000), "/v1/chat/completions")
    assert code(r) == "editor_context_exceeded"
    first, second = rig.upstream.calls
    assert second.body["max_tokens"] == first.body["max_tokens"] // 2
    assert [row[2] for row in _timeline(rig, run_id)] == [
        "editor_context_retried",
        "editor_context_exceeded",
    ]
    assert code(post(rig, token, _grow("x"), "/v1/chat/completions")) == "wsp_session_not_active"
    assert len(rig.upstream.calls) == 2


def test_llamacpp_clamps_an_oversized_cap_and_the_next_call_learns(rig: Rig) -> None:
    """miServer's GGUF path does not refuse prompt + cap past its window: llama-cpp-python 0.3.36
    clamps ``max_tokens`` to what the window leaves and stops there (``finish_reason: length``).
    The gateway passes that answer on and meters what was generated; the next call is sized
    from that count and fits."""
    run_id, token = _running(rig.world, "ep-local", tokens=10_000_000)
    _served(rig, run_id, 32_768)
    rig.upstream.window, rig.upstream.engine = 32_768, "llamacpp"
    r = post(rig, token, _grow("x" * 20_000), "/v1/chat/completions")
    assert r.status_code == 200 and r.json()["choices"][0]["finish_reason"] == "length"
    first = rig.upstream.calls[-1]
    room = 32_768 - rig.upstream.prompt_tokens(first.raw)
    assert first.body["max_tokens"] > room
    assert _timeline(rig, run_id)[-1] == ("served", 200, None, len(first.raw), room)
    r = post(rig, token, _grow("x" * 20_000, "ok", "z" * 500), "/v1/chat/completions")
    assert r.status_code == 200 and r.json()["choices"][0]["finish_reason"] == "stop"
    second = rig.upstream.calls[-1]
    assert rig.upstream.prompt_tokens(second.raw) + second.body["max_tokens"] <= 32_768
    assert len(rig.upstream.calls) == 2


def test_a_reasoning_allowance_leaves_the_action_room_at_the_new_sizes(rig: Rig) -> None:
    """The engine's thinking turn with the cap off sends ``max_reasoning_tokens`` (4,096 unless
    set): the gateway keeps it at most half of the cap it sizes and never closer to the cap than
    the action's 256, at a roomy window and at the 1,024 floor alike."""
    from wsp_gateway import allowlist, meter

    run_id, token = _running(rig.world, "ep-local", tokens=10_000_000)
    _served(rig, run_id, 8_192)
    # Roomy: the window leaves several thousand tokens.
    assert (
        post(
            rig, token, _grow("x" * 500) | {"max_reasoning_tokens": 4_096}, "/v1/chat/completions"
        ).status_code
        == 200
    )
    body = rig.upstream.calls[-1].body
    assert body["max_reasoning_tokens"] == min(4_096, body["max_tokens"] // 2)
    # Tight: the margined room is under the floor, so the cap is the 1,024 floor.
    tight = _grow("x" * 500, "ok", "y" * 6_200) | {"max_reasoning_tokens": 4_096}
    assert post(rig, token, tight, "/v1/chat/completions").status_code == 200
    body = rig.upstream.calls[-1].body
    assert body["max_tokens"] == meter.UNCAPPED_FLOOR_TOKENS
    assert body["max_reasoning_tokens"] <= body["max_tokens"] - allowlist.ACTION_FLOOR_TOKENS


# ------------------- 2026-11-21: a restarted gateway recovers the counts; no allowance of 1


def _two_calls(rig: Rig, *, restart: bool) -> tuple[int, int]:
    """A session at about 31,000 bytes, then grown to about 33,000 (0.6 tokens per byte, a
    32,768-token served window): the second call's cap, and the session's run id's usage rows."""
    run_id, token = _running(rig.world, "ep-local", tokens=10_000_000)
    _served(rig, run_id, 32_768)
    rig.upstream.tokens_per_byte, rig.upstream.window = 0.6, 32_768
    assert post(rig, token, _grow("x" * 30_900), "/v1/chat/completions").status_code == 200
    if restart:
        rig.restart()
    grown = _grow("x" * 30_900, "ok", "y" * 1_900)
    assert post(rig, token, grown, "/v1/chat/completions").status_code == 200
    cap = int(rig.upstream.calls[-1].body["max_tokens"])
    _finish(rig.world, run_id)
    return cap, rig.usage_rows(run_id)


def test_a_restarted_gateway_sizes_the_next_call_as_if_it_had_not_restarted(rig: Rig) -> None:
    """The served calls' prompt evidence (sent bytes, messages, the conversation's digest) is
    persisted with each settle (migration 0033) and served with the session; the restarted
    gateway recovers it, so the grown conversation is still its last count plus its new bytes,
    not 0.75 tokens per byte of the whole body."""
    from wsp_gateway import meter

    kept, _ = _two_calls(rig, restart=False)
    recovered, _ = _two_calls(rig, restart=True)
    assert recovered == kept
    sent = len(rig.upstream.calls[-1].raw)
    fresh = meter.uncapped_default(32_768, prompt_tokens=math.ceil((sent + 64) * 0.75))
    assert recovered > fresh + 4_000  # what a gateway with no evidence would have sized


def test_the_settle_persists_the_evidence_and_the_session_serves_it(rig: Rig) -> None:
    run_id, token = _running(rig.world, "ep-local", tokens=10_000_000)
    rig.upstream.tokens_per_byte = 0.6
    assert post(rig, token, _grow("x" * 2_000), "/v1/chat/completions").status_code == 200
    call = rig.upstream.calls[-1]
    with rig.world.engine.connect() as conn:
        row = conn.execute(
            sa.text(
                "SELECT sent_bytes, prompt_messages, conversation_sha256, input_tokens "
                "FROM mf_wsp_usage WHERE run_id = :r"
            ),
            {"r": run_id},
        ).one()
    from wsp_gateway import meter

    assert (
        row[0] == len(call.raw) and row[1] == 1 and row[3] == rig.upstream.prompt_tokens(call.raw)
    )
    assert row[2] == meter.conversation_key(
        {k: v for k, v in call.body.items() if k != "model"} | {"model": call.body["model"]}
    )
    (session,) = [s for s in rig.gateway.registry._sessions.values() if s.run_id == run_id]
    entry = rig.internal.get(f"/wsp-internal/v1/gateway/sessions/by-token/{session.token_sha256}")
    assert entry.json()["prompt_evidence"] == [
        {
            "sent_bytes": row[0],
            "prompt_tokens": row[3],
            "messages": 1,
            "conversation_sha256": row[2],
        }
    ]


def test_an_allowance_the_cap_leaves_under_256_is_sent_as_thinking_off(rig: Rig) -> None:
    """Run wsp_855a74f820ff: a clamped cap left max_reasoning_tokens at 1. Now the turn goes with
    thinking off (no allowance, enable_thinking false) and the answer says so, for the engine's
    transcript; an allowance that still fits is untouched."""
    from wsp_gateway.app import THINKING_SKIPPED_HEADER

    _run, token = _running(rig.world, "ep-local", tokens=10_000_000)
    asked = oai(
        max_tokens=400,
        max_reasoning_tokens=4_096,
        chat_template_kwargs={"enable_thinking": True},
    )
    r = post(rig, token, asked, "/v1/chat/completions")
    assert r.status_code == 200 and r.headers[THINKING_SKIPPED_HEADER] == "room"
    sent = rig.upstream.calls[-1].body
    assert "max_reasoning_tokens" not in sent
    assert sent["chat_template_kwargs"] == {"enable_thinking": False}
    roomy = oai(
        max_tokens=8_000, max_reasoning_tokens=4_096, chat_template_kwargs={"enable_thinking": True}
    )
    r = post(rig, token, roomy, "/v1/chat/completions")
    assert THINKING_SKIPPED_HEADER not in r.headers
    assert rig.upstream.calls[-1].body["max_reasoning_tokens"] == 4_096
    assert rig.upstream.calls[-1].body["chat_template_kwargs"] == {"enable_thinking": True}
