"""019 P3 (F19-D19, the push lease): ``HttpWorker.push`` reports progress while it streams, after
every chunk of every file and after each part, so a long push keeps its lease. Against a real
HTTP server that reads the whole body, as the supervisor's ``POST /session`` does."""

from __future__ import annotations

import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

from wsp import worker_client
from wsp.worker_client import HttpWorker


class _Supervisor:
    def __init__(self) -> None:
        self.received: list[int] = []
        outer = self

        class Handler(BaseHTTPRequestHandler):
            def do_POST(self) -> None:
                length = int(self.headers["Content-Length"])
                outer.received.append(len(self.rfile.read(length)))
                answer = json.dumps({"state": "running"}).encode()
                self.send_response(202)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(answer)))
                self.end_headers()
                self.wfile.write(answer)

            def log_message(self, *args: Any) -> None:
                return

        self.server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        threading.Thread(target=self.server.serve_forever, daemon=True).start()
        self.url = f"http://127.0.0.1:{self.server.server_address[1]}"

    def close(self) -> None:
        self.server.shutdown()
        self.server.server_close()


def test_a_push_reports_progress_per_chunk_and_per_part(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(worker_client, "_CHUNK", 1024)
    big, small = tmp_path / "big.bin", tmp_path / "small.bin"
    big.write_bytes(b"x" * (1024 * 5 + 10))  # six chunks
    small.write_bytes(b"y" * 100)  # one chunk
    supervisor = _Supervisor()
    client = HttpWorker(supervisor.url)
    beats: list[int] = []
    try:
        answer = client.push(
            {"run_id": "wsp_0123456789ab", "boot_id": "b" * 32},
            [("big.bin", big), ("small.bin", small)],
            on_progress=lambda: beats.append(1),
        )
    finally:
        client.close()
        supervisor.close()
    assert answer == {"state": "running"}
    assert len(beats) == (6 + 1) + 2  # every chunk, then each part
    assert supervisor.received and supervisor.received[0] > 1024 * 5


def test_a_push_without_a_progress_callback_still_streams(tmp_path: Path) -> None:
    data = tmp_path / "a.bin"
    data.write_bytes(b"z" * 10)
    supervisor = _Supervisor()
    client = HttpWorker(supervisor.url)
    try:
        assert client.push({"run_id": "r", "boot_id": "b"}, [("a.bin", data)]) == {
            "state": "running"
        }
    finally:
        client.close()
        supervisor.close()
