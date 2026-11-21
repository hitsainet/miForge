"""miForge's scorer client (feature 016; FTASKS 13.5; FR-016.26; P29, P31, P64, R3B-9).

Against ``httpx.MockTransport``: one call to the configured base URL's ``/api/v1/wsp/score``, no
``X-Miforge-Agent``, the request validated against the vendored contract BEFORE it is sent, and
refusals dispatched on the body's code. A full valid answer is the fake scorer's (13.3), not built.
"""

from __future__ import annotations

import json
from typing import Any

import httpx
import pytest

from clients.miworkbench import ScoreRefused, bundle_client, score

REQUEST: dict[str, Any] = {
    "kind": "miworkbench.wsp-score-request/v1",
    "bundle_id": "wsb_0123456789ab",
    "bundle_sha256": "b" * 64,
    "run_ref": {"miforge_run_id": "wsp_0123456789ab"},
    "heldout_ref": {
        "owner": "miworkbench",
        "sets": [
            {
                "name": "eval",
                "view_id": "view-eval",
                "role": "eval",
                "rows": 400,
                "row_ids_sha256": "c" * 64,
            }
        ],
    },
    "heads": [{"label": "parent"}],
}


def _client(handler: Any) -> httpx.Client:
    return bundle_client("http://miworkbench.test:8000/", transport=httpx.MockTransport(handler))


def test_the_request_fixture_is_valid_against_the_vendored_contract() -> None:
    from clients.miworkbench import _shape_errors

    assert _shape_errors("request", REQUEST) == []


def test_one_call_to_the_configured_route_with_no_agent_header() -> None:
    seen: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        return httpx.Response(200, json={"not": "an answer"})

    with pytest.raises(ScoreRefused) as caught:
        score(_client(handler), REQUEST)
    assert caught.value.code == "scorer_contract_invalid"  # a 200 that breaks the contract
    assert len(seen) == 1
    assert str(seen[0].url) == "http://miworkbench.test:8000/api/v1/wsp/score"
    assert seen[0].method == "POST"
    assert not any(h.lower().startswith("x-miforge") for h in seen[0].headers)
    assert json.loads(seen[0].content) == REQUEST


def test_an_invalid_request_is_refused_before_any_call() -> None:
    calls: list[Any] = []
    bad = {**REQUEST, "heads": [{"label": "candidate", "tensors": [{"name": "w"}]}]}
    with pytest.raises(ScoreRefused) as caught:
        score(_client(lambda r: calls.append(r) or httpx.Response(200)), bad)
    assert caught.value.code == "scorer_contract_invalid" and calls == []


@pytest.mark.parametrize(
    ("status", "body", "code"),
    [
        (429, {"error": {"code": "scorer_busy"}}, "scorer_busy"),
        (409, {"error": {"code": "scorer_busy"}}, "scorer_busy"),  # the body decides (P64)
        (
            409,
            {
                "error": {
                    "code": "heldout_set_head_budget_exhausted",
                    "set": "eval",
                    "rows_exhausted": 3,
                }
            },
            "heldout_set_head_budget_exhausted",
        ),
        (403, {"code": "operator_or_service_only"}, "operator_or_service_only"),
        # miWorkbench's REAL shape: FastAPI's HTTPException with a dict detail (found 016-P2).
        (429, {"detail": {"code": "scorer_busy", "message": "m"}}, "scorer_busy"),
        (409, {"detail": {"code": "scorer_busy", "message": "m"}}, "scorer_busy"),
        (
            409,
            {
                "detail": {
                    "code": "heldout_set_head_budget_exhausted",
                    "message": "m",
                    "set": "eval",
                    "row_ids_sha256": "c" * 64,
                    "rows_exhausted": 3,
                }
            },
            "heldout_set_head_budget_exhausted",
        ),
        (502, None, "scorer_unreachable"),
    ],
)
def test_a_refusal_is_dispatched_on_the_bodys_code(status: int, body: Any, code: str) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(status, json=body) if body is not None else httpx.Response(status)

    with pytest.raises(ScoreRefused) as caught:
        score(_client(handler), REQUEST)
    assert caught.value.code == code and caught.value.status == status
    assert caught.value.body == body


def test_a_transport_failure_is_scorer_unreachable() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("refused", request=request)

    with pytest.raises(ScoreRefused) as caught:
        score(_client(handler), REQUEST)
    assert caught.value.code == "scorer_unreachable"


# ------------------------------------------------- miWorkbench's clients, off-box guards (r016p2)
#
# Review 016-P2 (18.4, 2026-11-19): the tests above build ``bundle_client``, so nothing exercised
# ``scorer_client``'s two off-box guards, and no test anywhere exercised ``bundle_client``'s or the
# sensitivity reader's. With ``follow_redirects=True`` or ``trust_env=True`` in any of the three,
# the whole WSP tree stayed green (controls L5, L6, L12, L13, L14). For the scorer that sends the
# candidate's and control's tensors somewhere other than miForge's configured miWorkbench: a 307
# re-POSTs the body to whatever the answer names, and ``HTTP_PROXY`` routes it through a proxy
# taken from the environment.


def _scorer_call(base_url: str, transport: Any = None) -> None:
    from clients.miworkbench import scorer_client

    with scorer_client(base_url, transport=transport) as client:
        score(client, REQUEST)


def _manifest_call(base_url: str, transport: Any = None) -> None:
    from clients.miworkbench import fetch_bundle_manifest

    with bundle_client(base_url, transport=transport) as client:
        fetch_bundle_manifest(client, "wsb_0123456789ab")


@pytest.mark.parametrize("call", [_scorer_call, _manifest_call], ids=["scorer", "bundle"])
def test_no_client_follows_a_redirect(call: Any) -> None:
    """A redirect to another host is a refusal; nothing is sent to the host it names."""
    from clients.miworkbench import FetchFailed

    seen: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        if request.url.host == "miworkbench.test":
            return httpx.Response(307, headers={"Location": "http://elsewhere.test/x"})
        return httpx.Response(200, json={"not": "an answer"})

    with pytest.raises((ScoreRefused, FetchFailed)):
        call("http://miworkbench.test:8000/", httpx.MockTransport(handler))
    assert [r.url.host for r in seen] == ["miworkbench.test"]


def _sensitivity_call(base_url: str) -> None:
    from clients.miworkbench import read_bundle_sensitivity

    assert read_bundle_sensitivity("wsb_0123456789ab", base_url=base_url) == "not_sensitive"


@pytest.mark.parametrize(
    "call",
    [_scorer_call, _manifest_call, _sensitivity_call],
    ids=["scorer", "bundle", "sensitivity"],
)
def test_no_client_takes_a_proxy_from_the_environment(
    call: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    """``HTTP_PROXY`` set: the call still goes straight to the configured miWorkbench, and the
    proxy sees nothing (real sockets: httpx reads the environment only without a transport)."""
    import threading
    from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

    hits: dict[str, list[str]] = {"proxy": [], "miworkbench": []}

    def server(name: str) -> ThreadingHTTPServer:
        class Handler(BaseHTTPRequestHandler):
            def _answer(self) -> None:
                length = int(self.headers.get("Content-Length") or 0)
                self.rfile.read(length)
                hits[name].append(self.command)
                document = {"value": "not_sensitive"} if self.command == "GET" else {"not": 1}
                body = json.dumps(document).encode()
                self.send_response(200 if name == "miworkbench" else 502)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

            do_GET = do_POST = _answer  # noqa: N815 - http.server's names

            def log_message(self, *args: Any) -> None:
                return None

        httpd = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        threading.Thread(target=httpd.serve_forever, daemon=True).start()
        return httpd

    proxy, target = server("proxy"), server("miworkbench")
    try:
        proxy_url = f"http://127.0.0.1:{proxy.server_address[1]}"
        for name in ("HTTP_PROXY", "http_proxy", "ALL_PROXY", "all_proxy"):
            monkeypatch.setenv(name, proxy_url)
        for name in ("NO_PROXY", "no_proxy"):
            monkeypatch.delenv(name, raising=False)
        try:
            call(f"http://127.0.0.1:{target.server_address[1]}")
        except ScoreRefused as refused:  # the scorer's 200 breaks the contract: expected
            assert refused.code == "scorer_contract_invalid"
        assert hits["proxy"] == [] and len(hits["miworkbench"]) == 1
    finally:
        proxy.shutdown()
        target.shutdown()
