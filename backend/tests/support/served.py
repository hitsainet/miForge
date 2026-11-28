"""An ASGI app under a REAL uvicorn server on a free loopback port, in a thread (feature 016).

For what ``TestClient`` cannot show: it runs the app to completion before handing back the
response, so a dependency that commits AFTER the response is sent looks committed to it. A real
server sends the response first, as production does.
"""

from __future__ import annotations

import contextlib
import socket
import threading
import time
from collections.abc import Iterator
from typing import Any


@contextlib.contextmanager
def served(app: Any) -> Iterator[str]:
    """``app`` under uvicorn on ``127.0.0.1``; yields its base URL."""
    import uvicorn

    sock = socket.socket()
    sock.bind(("127.0.0.1", 0))
    port = sock.getsockname()[1]
    sock.close()
    server = uvicorn.Server(
        uvicorn.Config(app, host="127.0.0.1", port=port, log_level="warning", access_log=False)
    )
    thread = threading.Thread(target=server.run, daemon=True)
    thread.start()
    deadline = time.monotonic() + 10
    while not server.started:
        assert time.monotonic() < deadline, "uvicorn did not start"
        time.sleep(0.01)
    try:
        yield f"http://127.0.0.1:{port}"
    finally:
        server.should_exit = True
        thread.join(timeout=10)
