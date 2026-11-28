"""The driver's HTTP client for the supervisor's API (feature 016; FTDD §5.4a, §7).

Every call: a 5 s connect and a 60 s read timeout; a push streams its files with a 300 s per-chunk
timeout. Any transport failure, and any answer that is not the expected status, raises
``WorkerUnreachableError``: the driver never interprets a half-answer.
"""

from __future__ import annotations

import json
import secrets
from collections.abc import Callable, Iterator
from pathlib import Path
from typing import Any

import httpx

from wsp.dispatch import WorkerUnreachableError
from wsp.result_archive import RESULT_MAX_BYTES

CONNECT_S, READ_S, CHUNK_S = 5.0, 60.0, 300.0
_CHUNK = 1 << 20


class HttpWorker:
    def __init__(self, base_url: str, transport: httpx.BaseTransport | None = None) -> None:
        self._client = httpx.Client(
            base_url=base_url.rstrip("/"),
            timeout=httpx.Timeout(READ_S, connect=CONNECT_S),
            transport=transport,
            trust_env=False,
            follow_redirects=False,
        )

    def close(self) -> None:
        self._client.close()

    def _call(self, method: str, path: str, expect: int, **kwargs: Any) -> httpx.Response:
        try:
            response = self._client.request(method, path, **kwargs)
        except httpx.HTTPError as exc:
            raise WorkerUnreachableError(f"{method} {path}: {type(exc).__name__}") from exc
        if response.status_code != expect:
            raise WorkerUnreachableError(f"{method} {path}: {response.status_code}")
        return response

    def ready(self) -> dict[str, Any]:
        return dict(self._call("GET", "/ready", 200).json())

    def status(self) -> dict[str, Any]:
        return dict(self._call("GET", "/status", 200).json())

    def result(self) -> bytes:
        try:
            with self._client.stream("GET", "/result") as response:
                if response.status_code != 200:
                    raise WorkerUnreachableError(f"GET /result: {response.status_code}")
                data = bytearray()
                for chunk in response.iter_bytes():
                    data += chunk
                    if len(data) > RESULT_MAX_BYTES:
                        raise WorkerUnreachableError("GET /result: larger than its bound")
                return bytes(data)
        except httpx.HTTPError as exc:
            raise WorkerUnreachableError(f"GET /result: {type(exc).__name__}") from exc

    def ack(self, boot_id: str, run_id: str, result_sha256: str) -> None:
        body = {"boot_id": boot_id, "run_id": run_id, "result_sha256": result_sha256}
        self._call("POST", "/ack", 202, json=body)

    def recycle(self, boot_id: str) -> None:
        self._call("POST", "/recycle", 202, json={"boot_id": boot_id})

    def push(
        self,
        document: dict[str, Any],
        files: list[tuple[str, Path]],
        *,
        on_progress: Callable[[], None] | None = None,
    ) -> dict[str, Any]:
        """Stream the session and its files. ``on_progress`` (feature 019 P3, the push lease) is
        called after every chunk of a file and after each part; it throttles itself."""
        boundary = "wsp" + secrets.token_hex(12)
        head = (
            f'--{boundary}\r\nContent-Disposition: form-data; name="session"\r\n'
            "Content-Type: application/json\r\n\r\n".encode()
            + json.dumps(document).encode()
            + b"\r\n"
        )
        parts = [
            (
                f'--{boundary}\r\nContent-Disposition: form-data; name="file"; '
                f'filename="{name}"\r\nContent-Type: application/octet-stream\r\n\r\n'.encode(),
                path,
            )
            for name, path in files
        ]
        tail = f"--{boundary}--\r\n".encode()
        length = len(head) + len(tail) + sum(len(h) + p.stat().st_size + 2 for h, p in parts)

        def progress() -> None:
            if on_progress is not None:
                on_progress()

        def body() -> Iterator[bytes]:
            yield head
            for header, path in parts:
                yield header
                with path.open("rb") as handle:
                    for chunk in iter(lambda h=handle: h.read(_CHUNK), b""):  # type: ignore[misc]
                        yield chunk
                        progress()
                yield b"\r\n"
                progress()
            yield tail

        response = self._call(
            "POST",
            "/session",
            202,
            content=body(),
            headers={
                "Content-Type": f"multipart/form-data; boundary={boundary}",
                "Content-Length": str(length),
            },
            timeout=httpx.Timeout(CHUNK_S, connect=CONNECT_S),
        )
        return dict(response.json())
