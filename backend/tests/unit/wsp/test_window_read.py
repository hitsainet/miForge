"""``wsp.window_read``: the three readers, the dispatcher and the resolution (feature 019; W-1,
FTASKS 2.4, 2.6, 2.8; FR-019.1, FR-019.2, FR-019.4, FR-019.15; F19-D10, F19-D11, F19-D12).

The recorded answers are ``tests.support.window_answers``, hand-written from each project's
source at its pinned tag (``0xcc/reviews/019_window_readers.md``); the scripted servers answer
them over real HTTP on 127.0.0.1.
"""

from __future__ import annotations

import copy
import json
import logging
import threading
from collections.abc import Iterator
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any

import pytest

from tests.support import window_answers as wa
from wsp import window_read
from wsp.models import WspEndpoint
from wsp_worker import editor_limits

# ------------------------------------------------------------------ the parsers over the answers


@pytest.mark.parametrize(
    ("body", "model", "window"),
    [
        (wa.VLLM_MODELS, wa.VLLM_MODEL, wa.VLLM_WINDOW),
        (wa.SGLANG_MODELS, wa.SGLANG_MODEL, wa.SGLANG_WINDOW),
    ],
    ids=["vllm", "sglang"],
)
def test_an_openai_models_answer_gives_its_window(body: Any, model: str, window: int) -> None:
    assert window_read.parse_openai_models(body, model) == window


def test_llamacpp_props_gives_the_per_slot_window() -> None:
    assert window_read.parse_llamacpp_props(wa.LLAMACPP_PROPS) == wa.LLAMACPP_SLOT_WINDOW


@pytest.mark.parametrize(
    ("body", "model"),
    [
        (wa.VLLM_MODELS, "probe-editor-lora"),  # a LoRA card: max_model_len null
        (wa.VLLM_MODELS, "not-listed"),
        (wa.LLAMACPP_MODELS, wa.LLAMACPP_MODEL),  # no max_model_len; meta.n_ctx is not read
        (wa.LLAMACPP_ROUTER_MODELS, wa.LLAMACPP_MODEL),  # context_length is the training length
        ({"object": "list"}, wa.VLLM_MODEL),
        ({"data": {"id": wa.VLLM_MODEL}}, wa.VLLM_MODEL),
        ([], wa.VLLM_MODEL),
        (None, wa.VLLM_MODEL),
        ("text", wa.VLLM_MODEL),
    ],
    ids=[
        "lora",
        "unlisted",
        "llamacpp",
        "router",
        "no-data",
        "data-not-list",
        "list",
        "none",
        "str",
    ],
)
def test_an_openai_models_answer_without_a_window_gives_none(body: Any, model: str) -> None:
    assert window_read.parse_openai_models(body, model) is None


MALFORMED = [True, False, 0, -1, 32768.0, "32768", None, [32768], {"n": 1}]
IDS = ["true", "false", "zero", "negative", "float", "string", "null", "list", "dict"]


@pytest.mark.parametrize("value", MALFORMED, ids=IDS)
def test_a_malformed_max_model_len_gives_none(value: Any) -> None:
    body = copy.deepcopy(wa.VLLM_MODELS)
    body["data"][0]["max_model_len"] = value
    assert window_read.parse_openai_models(body, wa.VLLM_MODEL) is None


@pytest.mark.parametrize("value", MALFORMED, ids=IDS)
def test_a_malformed_n_ctx_gives_none(value: Any) -> None:
    body = copy.deepcopy(wa.LLAMACPP_PROPS)
    body["default_generation_settings"]["n_ctx"] = value
    assert window_read.parse_llamacpp_props(body) is None


@pytest.mark.parametrize(
    "body",
    [
        wa.LLAMACPP_ROUTER_PROPS,  # router mode's placeholder n_ctx: 0
        {"n_ctx": 8192},  # a top-level total is never used
        {"default_generation_settings": 8192},
        {},
        None,
        "props",
    ],
    ids=["router", "top-level", "settings-not-dict", "empty", "none", "str"],
)
def test_props_without_a_slot_window_gives_none(body: Any) -> None:
    assert window_read.parse_llamacpp_props(body) is None


# ------------------------------------------------------------------ read_window over real HTTP


class Server:
    """Answers each path from ``answers``; records the paths and the headers it was sent."""

    def __init__(self, answers: dict[str, Any], status: int = 200) -> None:
        self.answers = answers
        self.status = status
        self.paths: list[str] = []
        self.delay = 0.0
        outer = self

        class Handler(BaseHTTPRequestHandler):
            def do_GET(self) -> None:
                outer.paths.append(self.path)
                if outer.delay:
                    threading.Event().wait(outer.delay)
                # The full path first (a router's ``/props?model=``), else the path alone.
                body = outer.answers.get(self.path, outer.answers.get(self.path.split("?")[0]))
                status = outer.status if body is not None else 404
                data = json.dumps(body).encode()
                try:
                    self.send_response(status)
                    if status in (301, 302):
                        self.send_header("Location", "http://127.0.0.1:9/elsewhere")
                    self.send_header("Content-Type", "application/json")
                    self.send_header("Content-Length", str(len(data)))
                    self.end_headers()
                    self.wfile.write(data)
                except (BrokenPipeError, ConnectionResetError):
                    return  # the reader timed out and closed its side (the slow-server case)

            def log_message(self, *args: Any) -> None:
                return

        self.httpd = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.port = self.httpd.server_address[1]
        threading.Thread(target=self.httpd.serve_forever, daemon=True).start()

    def endpoint(
        self, model: str, *, serving: str = "generic", path: str = "/v1", host: str = "127.0.0.1"
    ) -> WspEndpoint:
        return WspEndpoint(
            id="ep-read",
            kind="openai_compatible",
            base_url=f"http://{host}:{self.port}{path}",
            model=model,
            locality="local",
            output_cap_field="max_completion_tokens",
            serving=serving,
        )

    def close(self) -> None:
        self.httpd.shutdown()
        self.httpd.server_close()


@pytest.fixture()
def serve() -> Iterator[Any]:
    made: list[Server] = []

    def make(answers: dict[str, Any], status: int = 200) -> Server:
        made.append(Server(answers, status))
        return made[-1]

    yield make
    for server in made:
        server.close()


@pytest.mark.parametrize("path", ["/v1", "", "/v1/"])
def test_a_vllm_server_is_read_at_its_origins_models_list(serve: Any, path: str) -> None:
    server = serve({"/v1/models": wa.VLLM_MODELS})
    assert window_read.read_window(server.endpoint(wa.VLLM_MODEL, path=path)) == wa.VLLM_WINDOW
    assert server.paths == ["/v1/models"]


def test_an_sglang_server_needs_no_reader_of_its_own(serve: Any) -> None:
    server = serve({"/v1/models": wa.SGLANG_MODELS})
    assert window_read.read_window(server.endpoint(wa.SGLANG_MODEL)) == wa.SGLANG_WINDOW


def test_a_llamacpp_server_is_read_at_props_after_the_models_list(serve: Any) -> None:
    server = serve({"/v1/models": wa.LLAMACPP_MODELS, "/props": wa.LLAMACPP_PROPS})
    assert window_read.read_window(server.endpoint(wa.LLAMACPP_MODEL)) == wa.LLAMACPP_SLOT_WINDOW
    assert server.paths == ["/v1/models", f"/props?model={wa.LLAMACPP_MODEL}"]


def test_a_llamacpp_router_is_read_for_the_named_model(serve: Any) -> None:
    """Feature 019 P2 (12.4): a router answers its placeholder ``n_ctx: 0`` to a bare ``/props``
    and the named model's slot figure to ``/props?model=<name>``; the reader asks the latter."""
    server = serve(
        {
            "/v1/models": wa.LLAMACPP_ROUTER_MODELS,
            "/props": wa.LLAMACPP_ROUTER_PROPS,
            f"/props?model={wa.LLAMACPP_MODEL}": wa.LLAMACPP_PROPS,
        }
    )
    assert window_read.read_window(server.endpoint(wa.LLAMACPP_MODEL)) == wa.LLAMACPP_SLOT_WINDOW
    assert server.paths == ["/v1/models", f"/props?model={wa.LLAMACPP_MODEL}"]


def test_a_router_without_the_named_model_reads_as_nothing(serve: Any) -> None:
    server = serve({"/v1/models": wa.LLAMACPP_ROUTER_MODELS, "/props": wa.LLAMACPP_ROUTER_PROPS})
    assert window_read.read_window(server.endpoint("not-a-model-it-serves")) is None


def test_the_model_query_is_url_encoded(serve: Any) -> None:
    server = serve({"/v1/models": {"data": []}, "/props": wa.LLAMACPP_PROPS})
    assert window_read.read_window(server.endpoint("org/model name")) == wa.LLAMACPP_SLOT_WINDOW
    assert server.paths[-1] == "/props?model=org%2Fmodel+name"


def test_a_miserver_endpoint_is_read_through_the_018_reader(serve: Any) -> None:
    """FR-019.22: miServer's read is ``wsp.tasks.read_served_window``, unchanged."""
    server = serve({"/api/health/inference": {"context_length": 32768}})
    assert window_read.read_window(server.endpoint("qwen", serving="miserver")) == 32768
    assert server.paths == ["/api/health/inference"]


@pytest.mark.parametrize("status", [500, 401, 302], ids=["500", "401-keyed", "redirect"])
def test_a_refusing_server_gives_none(serve: Any, status: int) -> None:
    server = serve({"/v1/models": wa.VLLM_MODELS, "/props": wa.LLAMACPP_PROPS}, status)
    assert window_read.read_window(server.endpoint(wa.VLLM_MODEL)) is None
    # Both readers asked once each; a redirect's Location (port 9) is never followed.
    assert server.paths == ["/v1/models", f"/props?model={wa.VLLM_MODEL.replace('/', '%2F')}"]


def test_an_unreachable_server_gives_none_with_one_log_line(caplog: Any) -> None:
    endpoint = WspEndpoint(
        id="ep-down",
        kind="openai_compatible",
        base_url="http://127.0.0.1:9/v1",
        model="m",
        locality="local",
        output_cap_field="max_completion_tokens",
        serving="generic",
    )
    with caplog.at_level(logging.WARNING, logger="wsp.window_read"):
        assert window_read.read_window(endpoint) is None
    lines = [r for r in caplog.records if getattr(r, "code", None) == "wsp_window_unread"]
    assert len(lines) == 1 and lines[0].endpoint_id == "ep-down"  # type: ignore[attr-defined]


def test_a_slow_server_times_out_to_none(
    serve: Any, monkeypatch: pytest.MonkeyPatch, caplog: Any
) -> None:
    monkeypatch.setattr(window_read, "READ_TIMEOUT_S", 0.2)
    server = serve({"/v1/models": wa.VLLM_MODELS})
    server.delay = 1.0
    with caplog.at_level(logging.WARNING, logger="wsp.window_read"):
        assert window_read.read_window(server.endpoint(wa.VLLM_MODEL)) is None
    assert [getattr(r, "code", None) for r in caplog.records].count("wsp_window_unread") == 1


def test_a_hosted_endpoint_is_never_read(serve: Any) -> None:
    """FR-019.4: an external endpoint makes no request at all."""
    server = serve({"/v1/models": wa.VLLM_MODELS})
    endpoint = server.endpoint(wa.VLLM_MODEL)
    endpoint.locality = "external"
    assert window_read.read_window(endpoint) is None
    assert server.paths == []


def test_the_read_ignores_the_proxy_environment(
    serve: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    """``trust_env=False``: a proxy in the environment never sees the read (it would answer
    nothing here, so the read would give ``None``)."""
    monkeypatch.setenv("HTTP_PROXY", "http://127.0.0.1:9")
    monkeypatch.setenv("http_proxy", "http://127.0.0.1:9")
    monkeypatch.delenv("NO_PROXY", raising=False)
    monkeypatch.delenv("no_proxy", raising=False)
    server = serve({"/v1/models": wa.VLLM_MODELS})
    assert window_read.read_window(server.endpoint(wa.VLLM_MODEL)) == wa.VLLM_WINDOW


# ------------------------------------------------------------------ resolve


def _endpoint(stated: int | None = None, ceiling: int | None = None) -> WspEndpoint:
    return WspEndpoint(
        id="ep",
        kind="openai_compatible",
        base_url="http://127.0.0.1:8000/v1",
        model="m",
        locality="local",
        output_cap_field="max_completion_tokens",
        serving="generic",
        context_window_tokens=stated,
        output_ceiling_tokens=ceiling,
    )


def test_a_stated_window_wins_and_a_larger_one_than_reported_is_warned() -> None:
    """F19-D10: stated 65,536 over a reported 32,768 is used, with the mismatch warning."""
    r = window_read.resolve(_endpoint(stated=65_536), 32_768, style="session")
    assert (r.window_tokens, r.window_source, r.reported) == (65_536, "stated", 32_768)
    assert r.warnings == ("window_stated_above_reported",)


def test_a_stated_window_below_the_report_has_no_warning() -> None:
    r = window_read.resolve(_endpoint(stated=16_384), 32_768, style="session")
    assert (r.window_tokens, r.window_source, r.warnings) == (16_384, "stated", ())


def test_a_reported_window_is_used_when_none_is_stated() -> None:
    r = window_read.resolve(_endpoint(), 32_768, style="session")
    assert (r.window_tokens, r.window_source, r.reported, r.warnings) == (
        32_768,
        "reported",
        32_768,
        (),
    )


@pytest.mark.parametrize("reported", [None, 0, -1, True], ids=["none", "zero", "neg", "bool"])
def test_nothing_read_is_assumed_8192_with_its_warning(reported: Any) -> None:
    """F19-D11: a failed read gives ``assumed`` 8,192, whatever any earlier read was."""
    r = window_read.resolve(_endpoint(), reported, style="session")
    assert (r.window_tokens, r.window_source, r.reported) == (8_192, "assumed", None)
    assert r.window_tokens is editor_limits.DEFAULT_WINDOW_TOKENS
    assert r.warnings == ("window_assumed",)


def test_a_ceiling_above_the_reported_window_is_lowered() -> None:
    """F19-D12: 32,768 with a reported 16,384 is lowered to 16,384."""
    r = window_read.resolve(_endpoint(ceiling=32_768), 16_384, style="session")
    assert (r.ceiling, r.ceiling_lowered) == (16_384, True)


def test_a_ceiling_below_the_reported_window_is_kept() -> None:
    r = window_read.resolve(_endpoint(ceiling=32_768), 65_536, style="session")
    assert (r.ceiling, r.ceiling_lowered) == (32_768, False)


def test_nothing_read_lowers_nothing() -> None:
    r = window_read.resolve(_endpoint(ceiling=32_768), None, style="session")
    assert (r.ceiling, r.ceiling_lowered) == (32_768, False)


def test_the_default_ceiling_is_lowered_against_a_small_report_only() -> None:
    assert window_read.resolve(_endpoint(), 16_384, style="session").ceiling == 16_384
    default = window_read.resolve(_endpoint(), 32_768, style="session")
    assert (default.ceiling, default.ceiling_lowered) == (32_000, False)


def test_a_stated_window_never_lowers_the_ceiling() -> None:
    """The ceiling is lowered against the REPORTED window only (TID §3.2)."""
    r = window_read.resolve(_endpoint(stated=8_192, ceiling=8_192), None, style="session")
    assert (r.ceiling, r.ceiling_lowered, r.window_tokens) == (8_192, False, 8_192)
    r = window_read.resolve(_endpoint(stated=16_384), None, style="session")
    assert (r.ceiling, r.ceiling_lowered) == (32_000, False)


@pytest.mark.parametrize(
    ("source", "window", "reported", "expected"),
    [
        ("assumed", 8_192, None, ["window_assumed"]),
        ("stated", 65_536, 32_768, ["window_stated_above_reported"]),
        ("stated", 16_384, 32_768, []),
        ("stated", 16_384, None, []),
        ("reported", 32_768, 32_768, []),
        (None, None, 32_768, []),  # a run from before 019
    ],
)
def test_a_stored_runs_warnings_follow_resolves_rule(
    source: str | None, window: int | None, reported: int | None, expected: list[str]
) -> None:
    assert window_read.warnings_of(source, window, reported) == expected
