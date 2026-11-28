"""``wsp.tasks.read_served_window`` against a real HTTP server (018 FTASKS 9.1, 9.2; FR-018.12).

miServer's ``GET /api/health/inference`` answers ``context_length`` as a TOP-LEVEL integer and no
model key (read live by the suite agent, 2026-11-18: ``backend``, ``description``,
``capabilities``, ``queue``, ``limitations``, ``context_length``; 32,768 with
Qwen2.5-7B-Instruct resident). The fake below answers that shape, on 127.0.0.1, and records the
path it was asked for: the reader must strip the endpoint's ``/v1`` and never follow a redirect.
"""

from __future__ import annotations

import json
import threading
from collections.abc import Iterator
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any

import pytest

from wsp import tasks
from wsp.models import WspEndpoint

LIVE_SHAPE = {
    "backend": "transformers",
    "description": "x",
    "capabilities": {},
    "queue": {},
    "limitations": [],
    "context_length": 32768,
}


class Server:
    def __init__(self) -> None:
        self.status = 200
        self.body: Any = LIVE_SHAPE
        self.paths: list[str] = []
        outer = self

        class Handler(BaseHTTPRequestHandler):
            def do_GET(self) -> None:
                outer.paths.append(self.path)
                data = json.dumps(outer.body).encode()
                self.send_response(outer.status)
                if outer.status in (301, 302):
                    self.send_header("Location", "http://127.0.0.1:9/elsewhere")
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(data)))
                self.end_headers()
                self.wfile.write(data)

            def log_message(self, *args: Any) -> None:
                return

        self.httpd = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.port = self.httpd.server_address[1]
        threading.Thread(target=self.httpd.serve_forever, daemon=True).start()

    def endpoint(self, path: str = "/v1") -> WspEndpoint:
        return WspEndpoint(
            id="ep-local",
            kind="openai_compatible",
            base_url=f"http://127.0.0.1:{self.port}{path}",
            model="qwen",
            locality="local",
            output_cap_field="max_tokens",
            serving="miserver",
        )


@pytest.fixture()
def server() -> Iterator[Server]:
    s = Server()
    yield s
    s.httpd.shutdown()
    s.httpd.server_close()


@pytest.mark.parametrize("path", ["/v1", "", "/v1/"])
def test_the_live_shape_gives_its_context_length(server: Server, path: str) -> None:
    assert tasks.read_served_window(server.endpoint(path)) == 32768
    assert server.paths == ["/api/health/inference"]


@pytest.mark.parametrize(
    "body",
    [
        {**LIVE_SHAPE, "context_length": 0},  # llama.cpp's "unknown"
        {k: v for k, v in LIVE_SHAPE.items() if k != "context_length"},  # no model resident
        {**LIVE_SHAPE, "context_length": "32768"},
        {**LIVE_SHAPE, "context_length": True},
        [LIVE_SHAPE],
    ],
    ids=["zero", "absent", "string", "bool", "list"],
)
def test_anything_but_a_positive_integer_is_none(server: Server, body: Any) -> None:
    server.body = body
    assert tasks.read_served_window(server.endpoint()) is None


@pytest.mark.parametrize("status", [302, 404, 503])
def test_a_non_200_answer_is_none_and_a_redirect_is_not_followed(
    server: Server, status: int
) -> None:
    server.status = status
    assert tasks.read_served_window(server.endpoint()) is None
    assert server.paths == ["/api/health/inference"]
