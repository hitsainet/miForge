"""Output schemas where a typed shape exists, and structuredContent to match (FR-017.19 - .21).

``backend/scripts/generate_output_schemas.py`` writes ``mcp_server/output_schemas.json`` from the
live OpenAPI document; ``build_server`` publishes them and ``shape_results`` answers
``structuredContent`` for those tools. Every schema is permissive (A30): the contract's pending
and unavailable bodies validate against it, and none declares ``next``.
"""

from __future__ import annotations

import asyncio
import json
import re
import subprocess
import sys
from pathlib import Path
from typing import Any

import httpx
import jsonschema
import pytest

from mcp_server.config import MCPSettings
from mcp_server.health_gate import HealthGate
from mcp_server.server import OUTPUT_SCHEMAS_PATH, _outcome, build_server, structured

REPO = Path(__file__).resolve().parents[4]
PENDING = {
    "approval_id": "apr_1",
    "status": "pending",
    "action": "gpu_start",
    "summary": {},
    "request_digest": "sha256:0",
    "expires_at": "2026-11-17T00:00:00+00:00",
    "hint": "x",
    "approval": {
        "id": "apr_1",
        "status": "pending",
        "decide_in": "miforge",
        "then": "server_runs",
        "summary": "Start GPU work.",
    },
    "next_steps": [{"tool": "forge_get_approval_status", "args": {}, "why": "poll"}],
}
UNAVAILABLE = {"unavailable": "backend", "reason": "down"}


@pytest.fixture(autouse=True)
def _open_gate(monkeypatch: pytest.MonkeyPatch) -> None:
    async def unavailable(self: Any, *_: str) -> None:
        return None

    monkeypatch.setattr(HealthGate, "unavailable", unavailable)


def _server(answer: Any) -> Any:
    mcp, client = build_server(MCPSettings(auth_token="x" * 32))
    client._http = httpx.AsyncClient(
        base_url="http://backend/api/v1",
        transport=httpx.MockTransport(lambda request: httpx.Response(200, json=answer)),
    )
    return mcp


def _published() -> dict[str, dict[str, Any]]:
    mcp = _server({})
    return {t.name: t.output_schema for t in asyncio.run(mcp.list_tools()) if t.output_schema}


def test_the_generated_file_is_current() -> None:
    result = subprocess.run(
        [sys.executable, "backend/scripts/generate_output_schemas.py", "--check"],
        cwd=REPO,
        env={"PYTHONPATH": f"{REPO}/backend/src:{REPO}/backend:{REPO}", "PATH": "/usr/bin:/bin"},
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr[-2000:]


def test_the_live_registry_publishes_exactly_the_files_typed_tools() -> None:
    document = json.loads(OUTPUT_SCHEMAS_PATH.read_text(encoding="utf-8"))
    published = _published()
    assert published == document["typed"]
    # Coverage, 2026-11-16: the typed routes and the approval view.
    assert set(published) == {
        "forge_evaluate_completions",
        "forge_get_approval_status",
        "forge_list_approvals",
        "forge_list_packs",
        "forge_list_selectors",
    }
    assert len(document["untyped"]) + len(published) >= 90
    assert all(reason for reason in document["untyped"].values())


@pytest.mark.parametrize("tool", sorted(_published()))
def test_every_schema_is_permissive(tool: str) -> None:
    schema = _published()[tool]
    assert schema["type"] == "object"
    assert "required" not in schema
    assert "next" not in schema["properties"]
    for body in (PENDING, UNAVAILABLE):
        jsonschema.Draft202012Validator(schema).validate(body)


def test_a_list_answers_structured_content_under_result() -> None:
    mcp = _server([])
    result = asyncio.run(mcp.call_tool("forge_list_selectors", {}))
    assert result.structured_content == {"result": []}
    assert [c.text for c in result.content] == ["[]"]  # still one block
    jsonschema.Draft202012Validator(_published()["forge_list_selectors"]).validate(
        result.structured_content
    )


def test_a_dict_answers_itself_and_validates() -> None:
    body = {"approvals": [], "count": 0, "pending": 0, "limit": 200}
    mcp = _server(body)
    result = asyncio.run(mcp.call_tool("forge_list_approvals", {}))
    assert result.structured_content == body
    jsonschema.Draft202012Validator(_published()["forge_list_approvals"]).validate(body)


@pytest.mark.parametrize("value", ["done", 3, None, ["a"]])
def test_a_non_dict_value_is_wrapped_under_result(value: Any) -> None:
    """A30: a scalar, None or a list, never only lists."""
    assert structured(value) == {"result": value}
    assert structured({"a": 1}) == {"a": 1}


def test_an_untyped_tool_answers_no_structured_content() -> None:
    mcp = _server({"runs": []})
    result = asyncio.run(mcp.call_tool("forge_list_runs", {}))
    assert result.structured_content is None


def test_the_audit_line_classifies_a_typed_tools_answer() -> None:
    from mcp_types import CallToolResult, TextContent

    content = [TextContent(type="text", text="{}")]
    assert _outcome(CallToolResult(content=content, structured_content=UNAVAILABLE)) == (
        "unavailable"
    )
    assert _outcome(CallToolResult(content=content, structured_content=PENDING)) == "pending"


def test_a_missing_schema_file_stops_the_server_at_start(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Not a warning and an untyped server (R2 review fix OR2-017-1, 2026-11-16): the start
    itself - ``build_server`` - refuses, naming the file it could not find."""
    from mcp_server import server

    missing = tmp_path / "output_schemas.json"
    with pytest.raises(SystemExit, match=re.escape(str(missing))):
        server.load_output_schemas(missing)
    monkeypatch.setattr(server, "OUTPUT_SCHEMAS_PATH", missing)
    with pytest.raises(SystemExit, match=re.escape(str(missing))):
        build_server(MCPSettings(auth_token="x" * 32))


def _shipped() -> dict[str, Any]:
    document: dict[str, Any] = json.loads(OUTPUT_SCHEMAS_PATH.read_text(encoding="utf-8"))
    return document


@pytest.mark.parametrize(
    ("label", "content"),
    [
        ("truncated", lambda: OUTPUT_SCHEMAS_PATH.read_text(encoding="utf-8")[:4000]),
        ("not an object", lambda: "[]"),
        ("no untyped map", lambda: json.dumps({"typed": _shipped()["typed"]})),
        ("empty typed map", lambda: json.dumps({"typed": {}, "untyped": _shipped()["untyped"]})),
        (
            "a typed entry that is not a schema",
            lambda: json.dumps({**_shipped(), "typed": {**_shipped()["typed"], "forge_x": 1}}),
        ),
    ],
)
def test_an_unreadable_or_partial_schema_file_stops_the_server_too(
    tmp_path: Path, label: str, content: Any
) -> None:
    """R3 review fix OR3-017-5, 2026-11-16: a truncated file died with a bare JSONDecodeError and
    ``{"typed": {}, "untyped": {}}`` started a server with every tool untyped and no log line."""
    from mcp_server import server

    path = tmp_path / "output_schemas.json"
    path.write_text(content(), encoding="utf-8")
    with pytest.raises(SystemExit, match=re.escape(str(path))):
        server.load_output_schemas(path)


def test_a_file_that_does_not_name_every_registered_tool_stops_the_server(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Every registered tool is typed or untyped in the file; a stale or partial file is not
    (OR3-017-5). The start names the tool."""
    from mcp_server import server

    document = _shipped()
    dropped = sorted(document["untyped"])[0]
    del document["untyped"][dropped]
    path = tmp_path / "output_schemas.json"
    path.write_text(json.dumps(document), encoding="utf-8")
    monkeypatch.setattr(server, "OUTPUT_SCHEMAS_PATH", path)
    with pytest.raises(SystemExit, match=re.escape(dropped)):
        build_server(MCPSettings(auth_token="x" * 32))
