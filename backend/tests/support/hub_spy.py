"""A transport spy for the REAL ``huggingface_hub`` client (017 FR-017.26 - .28, .33, .34; A28).

Installed with ``huggingface_hub.set_client_factory``, so every request the installed library
makes - ``repo_info``, the file tree, ``hf_hub_download``'s HEAD and GET, the safetensors header
read, ``snapshot_download`` - goes through :meth:`HubSpy.handle`, which records the method, the
path, the ``Authorization`` header and the ``Range`` header, and answers from a canned repository.

The assertions are on the **header the wire carries**, never on the ``token`` argument miForge
passed, so the spy cannot agree with the code by construction (R1A-3). The fixture sets
``HF_TOKEN=hf_spy_env``: an operator's request must show ``Bearer hf_spy_env`` (or the agent case
proves nothing), and an agent's must show no ``Authorization`` header at all.
"""

from __future__ import annotations

import json
import struct
from collections.abc import Iterator
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pytest

SPY_TOKEN = "hf_spy_env"
BEARER = f"Bearer {SPY_TOKEN}"
COMMIT = "c0ffee" + "0" * 34

#: Weight files: a request for one of these is a weight fetch.
WEIGHT_SUFFIXES = (".safetensors", ".bin", ".pth", ".pt", ".gguf")


def safetensors_bytes(dtype: str = "BF16", shape: tuple[int, ...] = (8, 8)) -> bytes:
    """A real safetensors file: the 8-byte header length, the JSON header, then the data."""
    width = {"BF16": 2, "F16": 2, "F32": 4, "I8": 1, "U8": 1}[dtype]
    count = 1
    for dim in shape:
        count *= dim
    header = json.dumps(
        {"w": {"dtype": dtype, "shape": list(shape), "data_offsets": [0, count * width]}}
    ).encode()
    return struct.pack("<Q", len(header)) + header + b"\0" * (count * width)


@dataclass
class Seen:
    method: str
    path: str
    authorization: str | None
    range: str | None

    @property
    def file(self) -> str | None:
        """The repository file a ``resolve`` request names, else ``None``."""
        if "/resolve/" not in self.path:
            return None
        return self.path.split("/resolve/", 1)[1].split("/", 1)[1]


@dataclass
class HubSpy:
    repos: dict[str, dict[str, bytes]] = field(default_factory=dict)
    seen: list[Seen] = field(default_factory=list)

    def add_repo(self, repo: str, files: dict[str, bytes]) -> None:
        self.repos[repo] = dict(files)

    def clear(self) -> None:
        self.seen.clear()

    # ------------------------------------------------------------------ what was sent

    @property
    def hub_requests(self) -> list[Seen]:
        """Requests to a repository (the library's own telemetry calls are not counted)."""
        return [s for s in self.seen if any(f"/{repo}/" in s.path for repo in self.repos)]

    def authorizations(self) -> set[str | None]:
        return {s.authorization for s in self.hub_requests}

    def whole_weight_fetches(self) -> list[Seen]:
        """GETs of a weight file without a ``Range`` header: a weight file fetched whole."""
        return [
            s
            for s in self.hub_requests
            if s.method == "GET" and (s.file or "").endswith(WEIGHT_SUFFIXES) and not s.range
        ]

    def fetched_files(self) -> set[str]:
        return {s.file for s in self.hub_requests if s.method == "GET" and s.file}

    # ------------------------------------------------------------------ the fake Hub

    def handle(self, request: Any) -> Any:
        import httpx2

        path = request.url.path
        self.seen.append(
            Seen(
                request.method,
                path,
                request.headers.get("authorization"),
                request.headers.get("range"),
            )
        )
        for repo, files in self.repos.items():
            if path.startswith(f"/api/models/{repo}/revision/"):
                return httpx2.Response(
                    200,
                    json={
                        "id": repo,
                        "sha": COMMIT,
                        "siblings": [{"rfilename": name} for name in files],
                    },
                )
            if path.startswith(f"/api/models/{repo}/tree/"):
                return httpx2.Response(
                    200,
                    json=[
                        {"type": "file", "path": name, "size": len(body), "oid": "0" * 40}
                        for name, body in files.items()
                    ],
                )
            if path.startswith(f"/{repo}/resolve/"):
                name = path.split("/resolve/", 1)[1].split("/", 1)[1]
                body = files.get(name)
                if body is None:
                    return httpx2.Response(404, headers={"x-error-code": "EntryNotFound"})
                headers = {
                    "x-repo-commit": COMMIT,
                    "etag": f'"{len(body)}-{abs(hash(name))}"',
                    "content-length": str(len(body)),
                }
                if request.method == "HEAD":
                    return httpx2.Response(200, headers=headers)
                wanted = request.headers.get("range")
                if wanted:
                    start, end = wanted.split("=", 1)[1].split("-")
                    first, last = int(start), min(int(end), len(body) - 1)
                    part = body[first : last + 1]
                    headers["content-length"] = str(len(part))
                    headers["content-range"] = f"bytes {first}-{last}/{len(body)}"
                    return httpx2.Response(206, headers=headers, content=part)
                return httpx2.Response(200, headers=headers, content=body)
        return httpx2.Response(404)


@pytest.fixture
def hub_spy(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Iterator[HubSpy]:
    """The real ``huggingface_hub`` over the spy, with ``HF_TOKEN`` set and a fresh cache."""
    import httpx2
    import huggingface_hub
    from huggingface_hub import constants
    from huggingface_hub.utils import _http

    monkeypatch.setenv("HF_TOKEN", SPY_TOKEN)
    monkeypatch.delenv("HF_HUB_OFFLINE", raising=False)
    monkeypatch.delenv("HF_HUB_DISABLE_IMPLICIT_TOKEN", raising=False)
    cache = tmp_path / "hf-cache"
    monkeypatch.setattr(constants, "HF_HUB_CACHE", str(cache))
    monkeypatch.setattr(constants, "HF_TOKEN_PATH", str(tmp_path / "hf-home" / "token"))
    monkeypatch.setattr(constants, "HF_HUB_OFFLINE", False)
    monkeypatch.setattr(constants, "HF_HUB_DISABLE_IMPLICIT_TOKEN", False)

    spy = HubSpy()
    huggingface_hub.set_client_factory(
        lambda: httpx2.Client(transport=httpx2.MockTransport(spy.handle), follow_redirects=True)
    )
    try:
        yield spy
    finally:
        huggingface_hub.set_client_factory(_http.default_client_factory)
        _http.close_session()
