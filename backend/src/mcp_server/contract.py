# Origin: miDataworks (Onegaishimas/miDataworks) backend/src/mcp_server/contract.py @ a055a38
# Mode: adapt (docs/REUSE.md). Kept: the contract is DERIVED from the live registry and a test
# regenerates and diffs it; endpoints come from the tool bodies' AST. Changed: miForge's REST
# contract section; the "waits for" column from GATED_TOOLS (itself pinned to the live gate).
"""Generate ``docs/mcp-contract.md`` from the LIVE tool registry (FR-012.20).

``tests/contract/test_mcp_contract_generated.py`` regenerates and fails on any difference.

Regenerate::

    cd backend/src && python -c "from mcp_server.contract import write_contract; write_contract()"
"""

from __future__ import annotations

import ast
import asyncio
import inspect
from pathlib import Path
from typing import Any

from .config import MCPSettings
from .context import ESCALATES_TO_ATTACH, GATED_TOOLS
from .server import build_server
from .tools import CATEGORY_MODULES

CONTRACT_PATH = Path(__file__).resolve().parents[3] / "docs" / "mcp-contract.md"

HEADER = """<!-- GENERATED FILE: DO NOT EDIT BY HAND.
Regenerate from backend/src:
  python -c "from mcp_server.contract import write_contract; write_contract()"
backend/tests/contract/test_mcp_contract_generated.py fails when this differs from the registry. -->

# miForge MCP contract

Every tool the miForge MCP server (feature 012) registers, its category, the backend route it
calls and, when an agent calls it, the approval it waits for. Derived from the live registry by
`backend/src/mcp_server/contract.py`.

Agents: call `forge_howto` first; it carries the workflow and the result shapes.
"""

#: The REST contract a proxy relies on: (item, value) rows of the generated table.
REST_ROWS: tuple[tuple[str, str], ...] = (
    ("Base", "Backend under `/api/`, REST under `/api/v1`"),
    (
        "Health",
        '`GET /api/health`: `{"status", "dependencies": {name: {"status", "detail"}}, '
        '"integrations"}`; 200, or 503 with the same body when a dependency is degraded',
    ),
    ("Success", "Plain JSON resources"),
    ("Error", '`{"error": {"code", "message", "details"}}`'),
    (
        "Header",
        "`X-Miforge-Agent: agent:<name>` (lower case, digits, hyphens; at most 48 characters "
        "after `agent:`). This server sends `agent:forge-mcp`; a miStudio proxy would send "
        "`agent:mistudio-mcp`. A malformed or empty header is refused on every route with "
        "`400 invalid_agent_identity`. A request without it is the operator's",
    ),
    (
        "Pending",
        '`202` with `{"approval_id", "status": "pending", "action", "summary", '
        '"request_digest", "expires_at", "hint"}`',
    ),
    (
        "Status",
        "`GET /api/v1/approvals/{id}`: `pending`, `executing`, `executed` (with `result`), "
        "`failed` (with `error`), `rejected` (with `reason`), `expired`",
    ),
    (
        "Refusals",
        "An agent approve or reject is `403 agent_cannot_decide`; a second approve is "
        "`409 approval_not_pending`",
    ),
    ("Approval actions", "`gpu_start`, `hotupdate_attach`, `delete`, `config_write`"),
    (
        "Class by request",
        "A run, define or chain whose (stage-0) `model_source.mode` is `attach` waits for "
        "`hotupdate_attach`; `continue` waits only with `submit: true`",
    ),
    (
        "Validation first",
        "A gated request is validated with the route's own code before it is stored; an "
        "invalid one is refused at once and never queued",
    ),
    (
        "Digest",
        "An approval binds to the exact stored request; it executes once, in-process, with "
        "that payload",
    ),
    ("Expiry", "A pending approval expires after 24 hours"),
    (
        "Transport",
        "Streamable HTTP on port 8765 at `/mcp`; `Authorization: Bearer <MCP_AUTH_TOKEN>` on "
        "every request but `/health`",
    ),
    (
        "OpenAPI",
        "`GET /api/openapi.json`; gated operations carry `x-approval-action` (and "
        "`x-approval-escalation`)",
    ),
)

REST_CONTRACT = (
    "## REST contract for proxies\n\n"
    "What this server - and a future miStudio proxy - relies on. Additive changes only.\n\n"
    "| Item | Value |\n|---|---|\n"
    + "".join(f"| {item} | {value} |\n" for item, value in REST_ROWS)
)


def _endpoints(func: ast.AST) -> list[str]:
    """``METHOD /path`` for every client call in a tool body."""
    verbs = {"get": "GET", "post": "POST", "put": "PUT", "delete": "DELETE"}
    found: list[str] = []
    for node in ast.walk(func):
        if not (isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)):
            continue
        if node.func.attr not in verbs or not node.args:
            continue
        if not (isinstance(node.func.value, ast.Name) and node.func.value.id == "client"):
            continue
        arg = node.args[0]
        path = None
        if isinstance(arg, ast.Constant) and isinstance(arg.value, str):
            path = arg.value
        elif isinstance(arg, ast.JoinedStr):
            path = "".join(
                str(v.value) if isinstance(v, ast.Constant) else "{" + ast.unparse(v.value) + "}"
                for v in arg.values
                if isinstance(v, ast.Constant | ast.FormattedValue)
            )
        if path and path.startswith("/"):
            full = path if path.startswith("/api/") else "/api/v1" + path
            found.append(f"{verbs[node.func.attr]} {full}")
    return sorted(set(found))


def collect() -> dict[str, list[dict[str, Any]]]:
    """``{category: [{name, summary, endpoints, gated}]}``: the registry decides membership."""
    mcp, _ = build_server(MCPSettings(auth_token="contract-generation"))
    tools = {t.name: t for t in asyncio.run(mcp.list_tools())}
    out: dict[str, list[dict[str, Any]]] = {}
    for category, modules in CATEGORY_MODULES.items():
        rows: list[dict[str, Any]] = []
        for module in modules:
            tree = ast.parse(inspect.getsource(module))
            for node in ast.walk(tree):
                if isinstance(node, ast.AsyncFunctionDef) and node.name in tools:
                    description = (tools[node.name].description or "").strip()
                    gated = GATED_TOOLS.get(node.name)
                    if gated and node.name in ESCALATES_TO_ATTACH:
                        gated = f"{gated} / hotupdate_attach"
                    rows.append(
                        {
                            "name": node.name,
                            "summary": " ".join(description.split()).split(". ")[0].rstrip(".")
                            + ".",
                            "endpoints": _endpoints(node),
                            "gated": gated,
                        }
                    )
        out[category] = rows
    return out


def generate_contract() -> str:
    data = collect()
    total = sum(len(rows) for rows in data.values())
    gated = sum(1 for rows in data.values() for row in rows if row["gated"])
    lines = [
        HEADER,
        f"**{total} tools in {len(data)} categories; {gated} wait for the operator.**\n",
    ]
    for category, rows in data.items():
        lines.append(f"## `{category}` ({len(rows)})\n")
        lines.append("| Tool | Calls | Waits for | What it does |")
        lines.append("|---|---|---|---|")
        for row in rows:
            calls = (
                "<br>".join(f"`{e}`" for e in row["endpoints"]) or "none (served by the MCP server)"
            )
            waits = f"`{row['gated']}`" if row["gated"] else ""
            summary = row["summary"].replace("|", "\\|")
            lines.append(f"| `{row['name']}` | {calls} | {waits} | {summary} |")
        lines.append("")
    lines.append(REST_CONTRACT)
    return "\n".join(lines).rstrip() + "\n"


def write_contract(path: Path = CONTRACT_PATH) -> None:
    path.write_text(generate_contract(), encoding="utf-8")
