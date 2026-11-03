"""The MCP transport's bearer token (D6, 2026-10-18): required on every request but /health,
and the server refuses to start over HTTP without one."""

from __future__ import annotations

import pytest
from starlette.testclient import TestClient

from mcp_server.config import MCPSettings
from mcp_server.server import build_http_app, build_server

TOKEN = "t" * 40


def app() -> TestClient:
    settings = MCPSettings(auth_token=TOKEN)
    mcp, _ = build_server(settings)
    return TestClient(build_http_app(mcp, settings))


def test_http_start_without_a_token_is_refused() -> None:
    with pytest.raises(SystemExit, match="MCP_AUTH_TOKEN is required"):
        build_server(MCPSettings(auth_token=""))


def test_anonymous_is_honoured_on_stdio_only() -> None:
    with pytest.raises(SystemExit, match="stdio transport only"):
        build_server(MCPSettings(auth_token=TOKEN, allow_anonymous=True))
    build_server(MCPSettings(auth_token="", allow_anonymous=True), stdio=True)


@pytest.mark.parametrize(
    "header",
    [None, "Bearer wrong", f"Basic {TOKEN}", "Bearer ", "Bearer tökén"],
)
def test_a_missing_or_wrong_token_is_401(header: str | None) -> None:
    # Bytes, so a non-ASCII token reaches the server (which must answer 401, not 500).
    headers = {} if header is None else {"Authorization": header.encode("utf-8")}
    response = app().post("/mcp", headers=headers, content=b"{}")
    assert response.status_code == 401


def test_health_is_open_and_names_the_categories() -> None:
    response = app().get("/health")
    assert response.status_code == 200
    body = response.json()
    assert body["service"] == "miforge-mcp" and len(body["categories"]) == 10
