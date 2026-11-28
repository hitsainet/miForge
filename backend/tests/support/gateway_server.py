"""Serve an ASGI app (the real WSP gateway) on a loopback port in a thread (feature 020; FTID §3.7).

The rehearsal (``tests/integration/wsp/test_hosted_rehearsal.py``) needs the gateway reachable
over real HTTP, because the worker's child process connects to it exactly as it connects in the
pod. ``uvicorn`` is already a backend dependency. :func:`serve` binds ``127.0.0.1:0``, waits for
the server to start, and reads the port it bound; ``stop()`` asks it to exit and joins the thread.
"""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass
from typing import Any

import uvicorn


@dataclass
class Served:
    host: str
    port: int
    server: uvicorn.Server
    thread: threading.Thread

    @property
    def url(self) -> str:
        return f"http://{self.host}:{self.port}"

    def stop(self) -> None:
        self.server.should_exit = True
        self.thread.join(timeout=30)
        assert not self.thread.is_alive(), "the served app did not stop"


def serve(app: Any, *, timeout_s: float = 30.0) -> Served:
    """``app`` served on ``127.0.0.1`` at a free port, in a daemon thread."""
    server = uvicorn.Server(
        uvicorn.Config(app, host="127.0.0.1", port=0, log_level="warning", lifespan="off")
    )
    thread = threading.Thread(target=server.run, daemon=True, name="gateway-server")
    thread.start()
    deadline = time.monotonic() + timeout_s
    while not server.started:
        if not thread.is_alive():
            raise RuntimeError("the served app's thread ended before it started")
        if time.monotonic() > deadline:
            server.should_exit = True
            raise TimeoutError("the served app did not start")
        time.sleep(0.02)
    port = int(server.servers[0].sockets[0].getsockname()[1])
    return Served("127.0.0.1", port, server, thread)
