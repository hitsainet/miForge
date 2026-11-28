"""Output schemas from the answer contract, and structuredContent to match (021 FR-021.17, .18,
.22; it replaced 017's permissive wrapping, FR-017.19 - .21).

``backend/scripts/generate_output_schemas.py`` projects each live tool's ledger route out of
``docs/schemas/miforge-answers-v1.json`` (FTDD §6.6): the route's answers at the root's ``anyOf``,
``Unavailable`` always, the pending approval on a gated route only, no ``next``, the referenced
documents embedded, no ``title`` and (operator, 2026-11-26, FT7 option (b)) no ``description``
(the contract keeps them). ``build_server`` publishes them and ``shape_results`` answers
``structuredContent`` for every tool.
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
from tests.support.mcp_ledger import EXPECTED_CALLS, GATED_CLASSES

REPO = Path(__file__).resolve().parents[4]


def _pending_body() -> dict[str, Any]:
    """The gate's real pending body (``approval_service.pending_body``) for a gpu_start request."""
    from datetime import timedelta

    from models.approval import Approval
    from services import approval_service

    now = approval_service.now()
    row = Approval(
        id="apr_0123456789abcdef",
        action="gpu_start",
        executor_key="api.v1.endpoints.runs_acquisition.submit_run",
        summary={"route": "POST /api/v1/runs"},
        payload={"body": {}},
        request_digest="0" * 64,
        requested_by="agent:forge-mcp",
        status="pending",
        created_at=now,
        expires_at=now + timedelta(hours=23),
    )
    return approval_service.pending_body(row)


PENDING = _pending_body()
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


_PUBLISHED: dict[str, dict[str, Any]] = {}


def _published() -> dict[str, dict[str, Any]]:
    """The live registry's output schemas, built once per process (a server build per case cost
    about a second each across ~230 cases; the schemas are read, never changed)."""
    if not _PUBLISHED:
        mcp = _server({})
        _PUBLISHED.update(
            {t.name: t.output_schema for t in asyncio.run(mcp.list_tools()) if t.output_schema}
        )
    return _PUBLISHED


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
    # 021 P5: every tool is typed (FR-021.16: an empty `untyped` map, no generic tool).
    assert document["untyped"] == {}
    tools = {t.name for t in asyncio.run(_server({}).list_tools())}
    assert set(published) == tools


def _gated(tool: str) -> bool:
    return EXPECTED_CALLS.get(tool, ("", "", {}, {}, "read"))[4] in GATED_CLASSES


@pytest.mark.parametrize("tool", sorted(_published()))
def test_every_schema_is_the_contract_projection(tool: str) -> None:
    """FTDD §6.6, steps 3 to 6, and the gated alternative's rule."""
    schema = _published()[tool]
    assert schema["type"] == "object" and "required" not in schema
    branches = schema["anyOf"]
    refs = [b.get("$ref") for b in branches]
    assert "#/$defs/Unavailable" in refs
    assert ("#/$defs/ApprovalPending" in refs) == _gated(tool), refs
    validator = jsonschema.Draft202012Validator(schema)
    validator.validate(UNAVAILABLE)
    if _gated(tool):
        validator.validate(PENDING)
    for branch in branches:
        target = branch
        while "$ref" in target and target["$ref"].count("/") == 2:
            target = schema["$defs"][target["$ref"].rsplit("/", 1)[-1]]
        assert "next" not in (target.get("properties") or {}), (tool, branch)
    text = json.dumps(schema)
    assert '"https://' not in text, "an MCP client cannot resolve a published document's $id"
    assert '"title"' not in text.replace('"title": {', "")


def test_the_pending_branch_is_on_exactly_the_gated_tools() -> None:
    published = _published()
    with_pending = {
        t for t, s in published.items() if {"$ref": "#/$defs/ApprovalPending"} in s["anyOf"]
    }
    assert with_pending == {t for t in published if _gated(t)}


def test_a_route_answer_missing_a_required_field_fails_its_schema() -> None:
    """The projection keeps `required` (always sent): the published schema is not permissive."""
    validator = jsonschema.Draft202012Validator(_published()["forge_list_approvals"])
    assert not validator.is_valid({"approvals": [], "count": 0, "pending": 0})
    assert validator.is_valid({"approvals": [], "count": 0, "pending": 0, "limit": 200})


def test_a_tool_with_no_route_and_no_overlay_fails_the_generator() -> None:
    sys.path.insert(0, str(REPO / "backend" / "scripts"))
    import generate_output_schemas as generator
    from output_overlays import Overlay

    contract = json.loads((REPO / "docs" / "schemas" / "miforge-answers-v1.json").read_text())
    language = json.loads(generator.RESULT_LANGUAGE.read_text())
    with pytest.raises(ValueError, match="no ledger route and no hand schema"):
        generator.output_transform(contract, {}, {"forge_stub": Overlay()}, language, set())


def test_the_delete_tools_answer_the_clients_status_code() -> None:
    for tool in ("forge_delete_pack", "forge_delete_draft"):
        validator = jsonschema.Draft202012Validator(_published()[tool])
        assert validator.is_valid({"status_code": 204})
        assert validator.is_valid(PENDING)


def test_a_list_answers_structured_content_under_result() -> None:
    mcp = _server([])
    result = asyncio.run(mcp.call_tool("forge_list_selectors", {}))
    assert result.structured_content == {"result": []}
    assert [c.text for c in result.content] == ["[]"]  # still one block
    schema = _published()["forge_list_selectors"]
    jsonschema.Draft202012Validator(schema).validate(result.structured_content)
    # The generator marks the wrapper, so the generic rules judge what it wraps (review round 4).
    (wrapper,) = [b for b in schema["anyOf"] if "properties" in b]
    assert wrapper["x-mcp-wrapped"] is True and set(wrapper["properties"]) == {"result"}


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


def test_a_delete_tool_answers_structured_content_too() -> None:
    # 021 P5: no tool is untyped; a 204 delete answers the client's {status_code}.
    mcp = _server({"status_code": 204})
    result = asyncio.run(mcp.call_tool("forge_delete_draft", {"draft_id": "draft-1"}))
    assert result.structured_content == {"status_code": 204}
    jsonschema.Draft202012Validator(_published()["forge_delete_draft"]).validate(
        result.structured_content
    )


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
    dropped = sorted(document["typed"])[0]  # 021 P5: every tool is typed
    del document["typed"][dropped]
    path = tmp_path / "output_schemas.json"
    path.write_text(json.dumps(document), encoding="utf-8")
    monkeypatch.setattr(server, "OUTPUT_SCHEMAS_PATH", path)
    with pytest.raises(SystemExit, match=re.escape(dropped)):
        build_server(MCPSettings(auth_token="x" * 32))


def _refs(node: Any) -> set[str]:
    found: set[str] = set()
    stack = [node]
    while stack:
        current = stack.pop()
        if isinstance(current, dict):
            ref = current.get("$ref")
            if isinstance(ref, str) and ref.startswith("#/$defs/"):
                found.add(ref[len("#/$defs/") :].split("/", 1)[0])
            stack.extend(current.values())
        elif isinstance(current, list):
            stack.extend(current)
    return found


@pytest.mark.parametrize("tool", sorted(_published()))
def test_each_schema_keeps_exactly_the_definitions_it_reaches(tool: str) -> None:
    """FR-021.22: no unused definition and no dangling reference in a tool's schema."""
    schema = _published()[tool]
    defs = schema.get("$defs", {})
    reached: set[str] = set()
    pending = _refs({k: v for k, v in schema.items() if k != "$defs"})
    while pending:
        name = pending.pop()
        if name in reached:
            continue
        assert name in defs, f"{tool}: #/$defs/{name} is referenced and not defined"
        reached.add(name)
        pending |= _refs(defs[name])
    assert reached == set(defs), sorted(set(defs) - reached)


def test_the_file_states_its_coverage_and_it_recounts() -> None:
    """FR-021.16: tools, typed, generic, and an empty untyped map, each equal to a recount."""
    document = json.loads(OUTPUT_SCHEMAS_PATH.read_text(encoding="utf-8"))
    tools = {t.name for t in asyncio.run(_server({}).list_tools())}

    from tests.support.output_schema_rules import generic

    assert document["coverage"] == {
        "tools": len(tools),
        "typed": len(document["typed"]),
        "generic": sum(1 for s in document["typed"].values() if generic(s)),
        "untyped": len(document["untyped"]),
    }
    assert document["coverage"]["generic"] == 0, "FR-021.16: no tool is generic"
    assert document["coverage"]["typed"] == len(tools) and document["untyped"] == {}


def test_the_write_switch_goes_through_the_contracts_writer(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """``generate_output_schemas.py`` without ``--check`` writes the contract only through
    ``answers_contract.write`` (which refuses a breaking change, FR-021.19), once, before the
    output schemas."""
    sys.path.insert(0, str(REPO / "backend" / "scripts"))
    import generate_output_schemas as generator

    from schemas import answers_contract

    committed = json.loads(answers_contract.CONTRACT_PATH.read_text(encoding="utf-8"))
    calls: list[str] = []

    def refusing_write(*_a: Any, **_k: Any) -> Any:
        calls.append("write")
        raise answers_contract.BreakingAnswerChange([])

    monkeypatch.setattr(answers_contract, "write", refusing_write)
    monkeypatch.setattr(generator, "OUT", tmp_path / "output_schemas.json")
    with pytest.raises(answers_contract.BreakingAnswerChange):
        generator.main([])
    assert calls == ["write"] and not (tmp_path / "output_schemas.json").exists()
    assert json.loads(answers_contract.CONTRACT_PATH.read_text(encoding="utf-8")) == committed


def _root(*answers: Any) -> dict[str, Any]:
    return {"type": "object", "anyOf": [*answers, {"$ref": "#/$defs/Unavailable"}], "$defs": {}}


def _wrapped(answer: Any) -> dict[str, Any]:
    return {
        "type": "object",
        "properties": {"result": answer},
        "required": ["result"],
        "x-mcp-wrapped": True,
    }


TYPED_ITEM = {"type": "object", "properties": {"id": {"type": "string"}}}


@pytest.mark.parametrize(
    ("schema", "is_generic"),
    [
        (_root(_wrapped({})), True),
        (_root(_wrapped({"type": "array", "items": {"type": "object"}})), True),
        (_root({"type": "object"}), True),
        (_root(_wrapped({"type": "array", "items": TYPED_ITEM})), False),
        (_root(TYPED_ITEM), False),
        # A route's own object whose only property is `result` is typed (it is not a wrapper).
        (_root({"type": "object", "properties": {"result": {}}}), False),
        # Review round 5's disagreements, now one rule (FR-021.4's is_typed) on both sides.
        (_root({"type": "object", "properties": {}}), True),
        (_root({"anyOf": [{"type": "object"}, {"type": "object"}]}), True),
        (_root(_wrapped({"type": "string", "enum": ["a"]})), True),
        (
            _root(_wrapped({"type": "array", "items": {"type": "array", "items": TYPED_ITEM}})),
            False,
        ),
        # Through a reference, as in the real file.
        (
            {
                "type": "object",
                "anyOf": [{"$ref": "#/$defs/Item"}, {"$ref": "#/$defs/Unavailable"}],
                "$defs": {"Item": TYPED_ITEM, "Unavailable": {"type": "object"}},
            },
            False,
        ),
        (
            {
                "type": "object",
                "anyOf": [{"$ref": "#/$defs/Open"}, {"$ref": "#/$defs/Unavailable"}],
                "$defs": {"Open": {"type": "object"}, "Unavailable": {"type": "object"}},
            },
            True,
        ),
    ],
)
def test_both_generic_rules_agree_on_planted_schemas(
    schema: dict[str, Any], is_generic: bool
) -> None:
    """Review rounds 4 and 5: the generator's rule and the recount's copy, against hand-written
    verdicts: a wrapped untyped answer, a wrapped array of untyped objects, a bare object, typed
    controls, a route object whose only property is ``result``, round 5's former disagreements (an
    empty ``properties``, an ``anyOf`` of open objects, a wrapped enum, a wrapped array of arrays),
    and answers through a ``$ref``."""
    sys.path.insert(0, str(REPO / "backend" / "scripts"))
    import generate_output_schemas as generator

    from tests.support.output_schema_rules import generic

    assert generator._is_generic(schema) is is_generic
    assert generic(schema) is is_generic
