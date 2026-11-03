"""``ForgeClient``: the agent header on every request, no way to drop it, errors verbatim
(FR-012.22, FR-012.4; FTID section 3.4)."""

from __future__ import annotations

import ast
import asyncio
import inspect
from pathlib import Path
from typing import Any

import httpx
import pytest

from mcp_server import client as client_module
from mcp_server.client import AGENT_HEADER, ForgeClient, ForgeError
from mcp_server.config import AGENT_IDENTITY_PATTERN
from tests.support.mcp_harness import RecordingClient

SRC = Path(__file__).resolve().parents[3] / "src"


def _constant(path: Path, name: str) -> str:
    for node in ast.walk(ast.parse(path.read_text())):
        if isinstance(node, ast.Assign) and any(
            isinstance(t, ast.Name) and t.id == name for t in node.targets
        ):
            assert isinstance(node.value, ast.Constant)
            return str(node.value.value)
    raise AssertionError(f"{name} not found in {path}")


def test_the_header_and_pattern_equal_the_rest_layers() -> None:
    """Compared by AST: this package must not import the backend to read them."""
    rest = SRC / "api" / "agent_origin.py"
    assert AGENT_HEADER == _constant(rest, "AGENT_HEADER") == "X-Miforge-Agent"
    assert AGENT_IDENTITY_PATTERN == _constant(rest, "AGENT_PATTERN")


def test_request_takes_no_headers_argument() -> None:
    assert "headers" not in inspect.signature(ForgeClient.request).parameters
    for name in ("get", "post", "put", "delete"):
        assert "headers" not in inspect.signature(getattr(ForgeClient, name)).parameters


@pytest.mark.parametrize("name", ["get", "post", "put", "delete"])
def test_the_recording_client_is_no_more_forgiving_than_the_real_one(name: str) -> None:
    assert inspect.signature(getattr(RecordingClient, name)) == inspect.signature(
        getattr(ForgeClient, name)
    )


def _client_over(handler: Any, identity: str = "agent:forge-mcp") -> ForgeClient:
    client = ForgeClient("http://backend:8000", identity)
    client._http = httpx.AsyncClient(
        base_url="http://backend:8000/api/v1",
        transport=httpx.MockTransport(handler),
        # Deliberately WITHOUT the default header: the per-request assertion must carry it.
    )
    return client


def test_every_request_carries_the_header_even_without_the_client_default() -> None:
    seen: list[tuple[str, str, str | None]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append((request.method, request.url.path, request.headers.get(AGENT_HEADER)))
        return httpx.Response(200, json={"ok": True})

    client = _client_over(handler)

    async def calls() -> None:
        await client.get("/runs")
        await client.post("/runs/define", json_body={"x": 1})
        await client.put("/settings/log_level", json_body={"value": "DEBUG"})
        await client.delete("/pack-drafts/d1")
        await client.get("/api/health")

    asyncio.run(calls())
    assert [identity for _, _, identity in seen] == ["agent:forge-mcp"] * 5
    assert [path for _, path, _ in seen][-1] == "/api/health"


def test_an_envelope_is_raised_verbatim_as_a_tool_error() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            409,
            json={"error": {"code": "gpu_lease_held", "message": "held", "details": {"run": "r"}}},
        )

    with pytest.raises(ForgeError) as caught:
        asyncio.run(_client_over(handler).get("/runs/r"))
    assert caught.value.code == "gpu_lease_held" and caught.value.status == 409
    assert '"run": "r"' in str(caught.value)


def test_a_2xx_that_is_not_json_is_an_error() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, text="<html>proxy</html>", headers={"content-type": "text/html"})

    with pytest.raises(ForgeError) as caught:
        asyncio.run(_client_over(handler).get("/runs"))
    assert caught.value.code == "NON_JSON_RESPONSE"


def test_a_read_timeout_warns_the_request_may_have_been_applied() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ReadTimeout("slow", request=request)

    with pytest.raises(ForgeError) as caught:
        asyncio.run(_client_over(handler).post("/runs/define", json_body={}))
    assert "may already have been applied" in str(caught.value)


def test_the_module_imports_nothing_from_the_backend() -> None:
    assert "from api" not in inspect.getsource(client_module).replace("from api.", "")
