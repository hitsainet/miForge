"""The push-time sensitivity re-read on its production path (FPRD P41; review OR1-016-1).

The dispatch tests inject their own reader, so until this file nothing exercised
``clients.miworkbench.read_bundle_sensitivity`` or the driver's wiring to it: returning
``"not_sensitive"`` from either survived the whole WSP suite, and that mutant sends a sensitive
bundle's data to an external provider the day miWorkbench answers 404. These tests talk to a
real HTTP server on loopback (no transport double), so what is asserted is the request the
production client sends, how many it sends, and that every answer other than a well-formed
``200 {"value": ...}`` reads as ``unknown`` (which freeze and dispatch treat as sensitive).
"""

from __future__ import annotations

import json
import socket
import threading
from collections.abc import Iterator
from dataclasses import dataclass, field
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from types import SimpleNamespace
from typing import Any

import pytest

from clients.miworkbench import read_bundle_sensitivity
from wsp.settings import get_wsp_settings

BUNDLE_ID = "bundle/one two"  # quoted in the path: a slash and a space
PATH = "/api/v1/wsp/bundles/bundle%2Fone%20two/sensitivity"


@dataclass
class Sibling:
    """A stand-in miWorkbench: one scripted answer, every request recorded."""

    url: str
    status: int = 200
    body: bytes = b'{"value": "not_sensitive"}'
    headers: dict[str, str] = field(default_factory=lambda: {"Content-Type": "application/json"})
    requests: list[tuple[str, str, dict[str, str]]] = field(default_factory=list)


@pytest.fixture()
def sibling() -> Iterator[Sibling]:
    state = Sibling(url="")

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self) -> None:
            state.requests.append(("GET", self.path, dict(self.headers.items())))
            self.send_response(state.status)
            for name, value in state.headers.items():
                self.send_header(name, value)
            self.send_header("Content-Length", str(len(state.body)))
            self.end_headers()
            self.wfile.write(state.body)

        def log_message(self, *args: Any) -> None:
            return

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    state.url = f"http://127.0.0.1:{server.server_address[1]}"
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield state
    finally:
        server.shutdown()
        server.server_close()


def _closed_port_url() -> str:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        port = s.getsockname()[1]
    return f"http://127.0.0.1:{port}"


class TestTheClient:
    @pytest.mark.parametrize("value", ["sensitive", "not_sensitive", "unknown"])
    def test_a_well_formed_answer_is_read_as_given(self, sibling: Sibling, value: str) -> None:
        sibling.body = json.dumps({"value": value}).encode()
        assert read_bundle_sensitivity(BUNDLE_ID, base_url=sibling.url) == value
        assert [(m, p) for m, p, _ in sibling.requests] == [("GET", PATH)]
        # P31: the bundle reads carry no agent header.
        assert not any(k.lower() == "x-miforge-agent" for k in sibling.requests[0][2])

    @pytest.mark.parametrize(
        ("status", "body"),
        [
            (404, b'{"value": "not_sensitive"}'),
            (500, b'{"value": "not_sensitive"}'),
            (503, b""),
            (204, b""),
            (302, b'{"value": "not_sensitive"}'),
            (200, b"not json"),
            (200, b'["not_sensitive"]'),
            (200, b'"not_sensitive"'),
            (200, b'{"value": "bogus"}'),
            (200, b'{"value": null}'),
            (200, b"{}"),
            (200, b'{"sensitivity": "not_sensitive"}'),
        ],
    )
    def test_every_other_answer_reads_as_unknown(
        self, sibling: Sibling, status: int, body: bytes
    ) -> None:
        sibling.status, sibling.body = status, body
        if status == 302:
            sibling.headers["Location"] = "/elsewhere"
        assert read_bundle_sensitivity(BUNDLE_ID, base_url=sibling.url) == "unknown"
        assert len(sibling.requests) == 1  # no retry, and a redirect is not followed

    def test_an_unreachable_sibling_reads_as_unknown(self) -> None:
        assert read_bundle_sensitivity(BUNDLE_ID, base_url=_closed_port_url()) == "unknown"

    def test_an_unset_url_reads_as_unknown_without_a_request(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.delenv("MIWORKBENCH_API_URL", raising=False)
        get_wsp_settings.cache_clear()
        try:
            assert read_bundle_sensitivity(BUNDLE_ID) == "unknown"
        finally:
            get_wsp_settings.cache_clear()


class TestTheCauseIsNamed:
    """R3 review fix (OR3-016-5), 2026-11-16: every ``unknown`` logs its cause, so an outage is
    told apart from a sensitive bundle. The value stays ``unknown`` (fail-closed)."""

    @staticmethod
    def _causes(caplog: pytest.LogCaptureFixture) -> list[str]:
        return [
            str(r.cause)
            for r in caplog.records
            if getattr(r, "code", None) == "wsp_sensitivity_unread"
        ]

    @pytest.mark.parametrize(
        ("status", "body", "cause"),
        [
            (404, b"{}", "miWorkbench answered HTTP 404"),
            (500, b"{}", "miWorkbench answered HTTP 500"),
            (200, b"not json", "miWorkbench's answer is not JSON"),
            (200, b'{"value": "bogus"}', "miWorkbench's answer has no known value"),
        ],
    )
    def test_an_answer_names_its_cause(
        self, sibling: Sibling, caplog: pytest.LogCaptureFixture, status, body, cause
    ) -> None:
        sibling.status, sibling.body = status, body
        with caplog.at_level("WARNING", logger="clients.miworkbench"):
            assert read_bundle_sensitivity(BUNDLE_ID, base_url=sibling.url) == "unknown"
        assert self._causes(caplog) == [cause]

    def test_unreachable_and_unset_name_their_cause(
        self, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
    ) -> None:
        monkeypatch.delenv("MIWORKBENCH_API_URL", raising=False)
        get_wsp_settings.cache_clear()
        try:
            with caplog.at_level("WARNING", logger="clients.miworkbench"):
                read_bundle_sensitivity(BUNDLE_ID, base_url=_closed_port_url())
                read_bundle_sensitivity(BUNDLE_ID)
        finally:
            get_wsp_settings.cache_clear()
        causes = self._causes(caplog)
        assert causes[0].startswith("miWorkbench is unreachable (")
        assert causes[1] == "MIWORKBENCH_API_URL is unset"

    def test_a_well_formed_answer_logs_nothing(
        self, sibling: Sibling, caplog: pytest.LogCaptureFixture
    ) -> None:
        sibling.body = b'{"value": "sensitive"}'
        with caplog.at_level("WARNING", logger="clients.miworkbench"):
            assert read_bundle_sensitivity(BUNDLE_ID, base_url=sibling.url) == "sensitive"
        assert self._causes(caplog) == []


class TestTheDriverWiring:
    """``wsp.tasks`` hands ``dispatch.drive`` the production reader, and that reader asks the
    configured miWorkbench for the run's bundle."""

    def test_the_tasks_reader_asks_the_configured_sibling(
        self, sibling: Sibling, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from wsp import tasks

        monkeypatch.setenv("MIWORKBENCH_API_URL", sibling.url)
        get_wsp_settings.cache_clear()
        run = SimpleNamespace(run_id="r1")
        bundle = SimpleNamespace(bundle_id=BUNDLE_ID)
        try:
            sibling.body = b'{"value": "not_sensitive"}'
            assert tasks.read_sensitivity(run, bundle) == "not_sensitive"
            sibling.status = 404
            assert tasks.read_sensitivity(run, bundle) == "unknown"
            sibling.status, sibling.body = 200, b'{"value": "sensitive"}'
            assert tasks.read_sensitivity(run, bundle) == "sensitive"
        finally:
            get_wsp_settings.cache_clear()
        assert [p for _, p, _ in sibling.requests] == [PATH] * 3

    def test_the_tasks_reader_with_no_sibling_configured_is_unknown(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from wsp import tasks

        monkeypatch.delenv("MIWORKBENCH_API_URL", raising=False)
        get_wsp_settings.cache_clear()
        try:
            got = tasks.read_sensitivity(SimpleNamespace(), SimpleNamespace(bundle_id=BUNDLE_ID))
        finally:
            get_wsp_settings.cache_clear()
        assert got == "unknown"

    def test_the_drive_task_passes_the_production_reader(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from wsp import dispatch, tasks, worker_client

        calls: list[dict[str, Any]] = []

        class Worker:
            def __init__(self, url: str) -> None:
                self.url = url

            def close(self) -> None:
                return

        def recorder(factory: Any, workers: Any, settings: Any, **kwargs: Any) -> Any:
            # 019 P3: the pass takes one slot per provisioned worker.
            calls.append({"worker_url": [w.client.url for w in workers], **kwargs})
            return SimpleNamespace(actions=[])

        monkeypatch.setenv("WSP_WORKER_URL", "http://worker.invalid:9000")
        monkeypatch.setenv("WSP_GATEWAY_URL", "http://gateway.invalid:8080")
        monkeypatch.setattr(worker_client, "HttpWorker", Worker)
        monkeypatch.setattr(dispatch, "drive", recorder)
        monkeypatch.setattr("db.session.get_session_factory", lambda: "factory")
        get_wsp_settings.cache_clear()
        try:
            assert tasks.drive.run() == {"actions": []}
        finally:
            get_wsp_settings.cache_clear()
        assert len(calls) == 1
        assert calls[0]["worker_url"] == ["http://worker.invalid:9000"]
        assert calls[0]["read_sensitivity"] is tasks.read_sensitivity
        assert calls[0]["files_for"] is tasks.run_files
