"""Generate ``backend/src/mcp_server/output_schemas.json`` (017 FR-017.19, .20; 017-D7).

For each live MCP tool, its ledger route (``tests/support/mcp_ledger.py``), then:

1. the route's OpenAPI 2xx schema when the route is typed (a ``response_model``);
2. else a miForge-owned published document the route returns (``DOCUMENT_SCHEMAS``, each proved
   against a real answer before it is listed);
3. else ``UNTYPED``, with the reason. Typing the untyped routes is recorded debt (BRD §6).

Every schema is wrapped permissively (A30): an object root, no ``required``, the contract's
optional ``approval``, ``next_steps``, ``unavailable`` and ``reason`` declared, and never a
``next`` property. A non-object answer (a list) is declared under ``result``, which is where
``shape_results`` puts it in ``structuredContent``.

The MCP pod imports no backend module, so this runs in tests and by hand; the server reads only
the generated JSON. ``--check`` regenerates and fails on any difference.

    PYTHONPATH=backend/src:backend python backend/scripts/generate_output_schemas.py [--check]
"""

from __future__ import annotations

import asyncio
import json
import sys
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parents[2]
OUT = REPO / "backend" / "src" / "mcp_server" / "output_schemas.json"

#: tool -> a miForge-owned document schema in docs/schemas/ the route returns as its body.
#: Empty on 2026-11-16: each candidate was checked and none returns the bare document.
DOCUMENT_SCHEMAS: dict[str, str] = {}

#: Candidates considered for DOCUMENT_SCHEMAS and why each is not listed (FR-017.19).
NOT_A_DOCUMENT: dict[str, str] = {
    "forge_get_checkpoint": "answers {manifest, ...}: the manifest wrapped",
    "forge_get_adapter": "answers a registry row with the manifest inside, not adapter-manifest-v1",
    "forge_get_batch": "answers a route-built body around batch-record-v1",
    "forge_list_steps": "answers {steps, ...}: a page object, not a list of step-record-v1",
}

RESULT_KEYS: dict[str, Any] = {
    "approval": {"type": "object"},
    "next_steps": {"type": "array", "items": {"type": "object"}},
    "unavailable": {"type": "string"},
    "reason": {"type": ["string", "null"]},
}


def _rewrite_refs(node: Any) -> Any:
    if isinstance(node, dict):
        return {
            key: (
                value.replace("#/components/schemas/", "#/$defs/")
                if key == "$ref" and isinstance(value, str)
                else _rewrite_refs(value)
            )
            for key, value in node.items()
        }
    if isinstance(node, list):
        return [_rewrite_refs(item) for item in node]
    return node


def _is_typed(schema: dict[str, Any]) -> bool:
    """A ``dict[str, Any]`` route answers ``{type: object, additionalProperties: true}``."""
    if "$ref" in schema or schema.get("type") == "array":
        return True
    return bool(schema.get("properties"))


def permissive(schema: dict[str, Any], components: dict[str, Any]) -> dict[str, Any]:
    body = _rewrite_refs(schema)
    if body.get("type") == "array":
        root: dict[str, Any] = {
            "type": "object",
            "properties": {"result": body, **RESULT_KEYS},
        }
    elif "$ref" in body:
        name = body["$ref"].rsplit("/", 1)[-1]
        target = _rewrite_refs(components[name])
        root = {
            "type": "object",
            "properties": {**target.get("properties", {}), **RESULT_KEYS},
        }
    else:
        root = {"type": "object", "properties": {**body.get("properties", {}), **RESULT_KEYS}}
    root["properties"].pop("next", None)
    root["additionalProperties"] = True
    needed = _reachable(root, components)
    if needed:
        root["$defs"] = {name: _rewrite_refs(components[name]) for name in sorted(needed)}
    return root


def _reachable(node: Any, components: dict[str, Any]) -> set[str]:
    """The component names ``node`` refers to, transitively."""
    found: set[str] = set()
    stack = [node]
    while stack:
        current = stack.pop()
        if isinstance(current, dict):
            ref = current.get("$ref")
            if isinstance(ref, str) and ref.startswith("#/$defs/"):
                name = ref.rsplit("/", 1)[-1]
                if name not in found and name in components:
                    found.add(name)
                    stack.append(_rewrite_refs(components[name]))
            stack.extend(current.values())
        elif isinstance(current, list):
            stack.extend(current)
    return found


def generate() -> dict[str, Any]:
    from tests.support.mcp_ledger import EXPECTED_CALLS, full, template_matches

    from api.app import create_app
    from mcp_server.config import MCPSettings
    from mcp_server.server import build_server

    openapi = create_app().openapi()
    components = openapi.get("components", {}).get("schemas", {})
    mcp, _ = build_server(MCPSettings(auth_token="schema-generation"))
    tools = sorted(t.name for t in asyncio.run(mcp.list_tools()))
    typed: dict[str, Any] = {}
    untyped: dict[str, str] = {}
    for tool in tools:
        if tool not in EXPECTED_CALLS:
            untyped[tool] = "makes no backend call"
            continue
        method, path, *_ = EXPECTED_CALLS[tool]
        operation = None
        for template, item in openapi["paths"].items():
            if template_matches(template, full(path)) and method.lower() in item:
                operation = item[method.lower()]
                break
        if operation is None:
            untyped[tool] = "route not in the OpenAPI document"
            continue
        schema = None
        for code, response in operation.get("responses", {}).items():
            if str(code).startswith("2"):
                schema = response.get("content", {}).get("application/json", {}).get("schema")
                break
        if schema and _is_typed(schema):
            typed[tool] = permissive(schema, components)
        elif tool in DOCUMENT_SCHEMAS:
            document = json.loads((REPO / "docs" / "schemas" / DOCUMENT_SCHEMAS[tool]).read_text())
            typed[tool] = permissive(document, {})
        else:
            untyped[tool] = NOT_A_DOCUMENT.get(tool, "the route answers an untyped dict")
    return {"typed": typed, "untyped": untyped}


def render() -> str:
    return json.dumps(generate(), indent=1, sort_keys=True) + "\n"


def main(argv: list[str]) -> int:
    text = render()
    if "--check" in argv:
        if OUT.read_text(encoding="utf-8") != text:
            print(f"{OUT} is stale: regenerate it", file=sys.stderr)
            return 1
        return 0
    OUT.write_text(text, encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
