"""Errors that teach: code first, lower case, then fix, next and the original (017 FR-017.9 - .11).

Through ``build_server`` and ``mcp.call_tool`` (what the transport calls), so building with the
SDK's plain ``MCPServer`` turns these red. Line 1 must validate as the VENDORED contract's
``error_code``; the four parts come in order; argument values never appear.
"""

from __future__ import annotations

import ast
import asyncio
import json
import re
from pathlib import Path
from typing import Any

import httpx
import pytest

from mcp_server.config import MCPSettings
from mcp_server.health_gate import HealthGate
from mcp_server.server import build_server
from mcp_server.teaching import line_one_code
from tests.support.result_language import error_code_pattern, validate

LINE_ONE = re.compile(r"^[a-z][a-z0-9_]*(\.[a-z0-9_]+)?: \S.{0,199}$")
SRC = Path(__file__).resolve().parents[3] / "src"


@pytest.fixture(autouse=True)
def _open_gate(monkeypatch: pytest.MonkeyPatch) -> None:
    async def unavailable(self: Any, *_: str) -> None:
        return None

    monkeypatch.setattr(HealthGate, "unavailable", unavailable)


def _server(response: httpx.Response | Exception) -> Any:
    mcp, client = build_server(MCPSettings(auth_token="x" * 32))

    def handler(request: httpx.Request) -> httpx.Response:
        if isinstance(response, Exception):
            raise response
        return response

    client._http = httpx.AsyncClient(
        base_url="http://backend/api/v1", transport=httpx.MockTransport(handler)
    )
    return mcp


def _parts(result: Any) -> tuple[str, str, list[dict[str, Any]], str]:
    assert result.is_error is True
    (block,) = result.content
    lines = block.text.split("\n")
    assert len(lines) >= 4, block.text
    assert LINE_ONE.match(lines[0]), lines[0]
    assert not lines[0].startswith("Error executing tool")
    code = lines[0].split(":", 1)[0]
    validate("error_code", code)
    assert lines[1].startswith("fix: "), lines[1]
    assert lines[2].startswith("next: "), lines[2]
    steps = json.loads(lines[2].removeprefix("next: "))
    for step in steps:
        validate("next_step", step)
    return code, lines[1], steps, "\n".join(lines[3:])


def _call(mcp: Any, name: str, args: dict[str, Any]) -> Any:
    return asyncio.run(mcp.call_tool(name, args))


def test_a_backend_refusal_keeps_its_code_fix_next_and_original() -> None:
    envelope = {
        "error": {
            "code": "run_not_archivable",
            "message": "Run 'r' is running. Only a finished run can be archived.",
            "details": {
                "run_id": "r",
                "next_steps": [
                    {"tool": "forge_get_run", "args": {"run_id": "r"}, "why": "see its status"}
                ],
            },
        }
    }
    mcp = _server(httpx.Response(409, json=envelope))
    code, fix, steps, original = _parts(_call(mcp, "forge_archive_run", {"run_id": "r"}))
    assert code == "run_not_archivable"
    assert fix == "fix: see its status"
    assert steps == envelope["error"]["details"]["next_steps"]
    assert original.startswith("run_not_archivable (409): Run 'r' is running.")


@pytest.mark.parametrize(
    ("raised", "code"),
    # BACKEND_UNREACHABLE never surfaces as an error: the health gate answers it as the
    # `unavailable` result shape (`health_gate.gated_call`), which is not an error.
    [(httpx.ReadTimeout("slow"), "backend_timeout")],
)
def test_a_transport_failure_is_lower_case_on_line_one_and_points_to_health(
    raised: Exception, code: str
) -> None:
    mcp = _server(raised)
    got, _fix, steps, original = _parts(_call(mcp, "forge_list_runs", {}))
    assert got == code
    assert steps[0]["tool"] == "forge_health"
    if code == "backend_timeout":
        assert [s["tool"] for s in steps] == ["forge_health", "forge_list_runs"]
    assert original.startswith(code.upper())  # the original keeps the code as raised


def test_a_non_json_error_is_lower_case_on_line_one() -> None:
    mcp = _server(httpx.Response(502, text="<html>bad gateway</html>"))
    got, _fix, steps, original = _parts(_call(mcp, "forge_list_runs", {}))
    assert got == "non_json_error"
    assert original.startswith("NON_JSON_ERROR (502)")
    assert steps[0]["tool"] == "forge_health"


def test_an_unknown_argument_is_named_and_its_value_never_echoed() -> None:
    mcp = _server(httpx.Response(200, json={}))
    result = _call(mcp, "forge_list_runs", {"bogus_name": "s3cret-value"})
    code, _fix, steps, _original = _parts(result)
    assert code == "unknown_arguments"
    assert "bogus_name" in result.content[0].text
    assert "s3cret-value" not in result.content[0].text
    assert steps == []


def test_an_invalid_argument_names_the_field_not_the_value() -> None:
    mcp = _server(httpx.Response(200, json={}))
    result = _call(mcp, "forge_list_runs", {"limit": "s3cret-value"})
    code, fix, _steps, _original = _parts(result)
    assert code == "invalid_arguments"
    assert "limit" in fix
    assert "s3cret-value" not in result.content[0].text


def test_an_unknown_tool_points_to_howto() -> None:
    mcp = _server(httpx.Response(200, json={}))
    code, _fix, steps, _original = _parts(_call(mcp, "forge_no_such_tool", {}))
    assert code == "unknown_tool"
    assert [s["tool"] for s in steps] == ["forge_howto"]


def test_an_unexpected_exception_is_internal_error_and_hides_its_text() -> None:
    mcp = _server(httpx.Response(200, json={}))

    @mcp.tool()
    async def forge_crash_stub() -> Any:
        raise RuntimeError("private detail")

    result = _call(mcp, "forge_crash_stub", {})
    code, _fix, _steps, _original = _parts(result)
    assert code == "internal_error"
    assert "private detail" not in result.content[0].text


def test_every_code_miforge_can_raise_maps_to_a_valid_error_code_without_the_fallback() -> None:
    """An AST pass over every ``refuse(`` code literal and the MCP client's own codes."""
    pattern = re.compile(error_code_pattern())
    codes: set[str] = set()
    for path in SRC.rglob("*.py"):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if (
                isinstance(node, ast.Call)
                and getattr(node.func, "id", getattr(node.func, "attr", None)) == "refuse"
                and node.args
                and isinstance(node.args[0], ast.Constant)
                and isinstance(node.args[0].value, str)
            ):
                codes.add(node.args[0].value)
            if (
                path.name == "client.py"
                and "mcp_server" in path.parts
                and isinstance(node, ast.Call)
                and getattr(node.func, "id", None) == "ForgeError"
                and len(node.args) >= 2
                and isinstance(node.args[1], ast.Constant)
            ):
                codes.add(str(node.args[1].value))
    codes |= {"unknown_arguments", "invalid_arguments", "unknown_tool", "internal_error"}
    assert len(codes) >= 100, len(codes)
    assert {"BACKEND_TIMEOUT", "NON_JSON_RESPONSE"} <= codes
    bad = {
        code: line_one_code(code) for code in codes if not pattern.fullmatch(line_one_code(code))
    }
    assert not bad, bad
    fell_back = sorted(code for code in codes if line_one_code(code) == "error")
    assert not fell_back, fell_back
    assert line_one_code("BACKEND_TIMEOUT") == "backend_timeout"
