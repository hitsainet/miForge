# Origin: miDataworks (Onegaishimas/miDataworks) backend/src/mcp_server/server.py @ a055a38
# Mode: adapt (docs/REUSE.md). Kept: BearerAuthMiddleware (UTF-8 constant-time compare, /health
# exempt), both startup refusals, the stateless JSON HTTP app with `host` passed through, the
# non-blocking /health, the audit wrapper, registry-derived instruction counts.
# Changed: miForge's client, identity and categories; no upload size cap (no file tools).
"""miForge MCP server assembly: MCPServer (mcp SDK 2.x), bearer auth, categories, audit.

Auth (D6, 2026-10-18): the streamable-HTTP transport requires ``Authorization: Bearer
<MCP_AUTH_TOKEN>`` on every request but ``/health``. Start is refused without a token, and
``MCP_ALLOW_ANONYMOUS`` is honoured on stdio only. The token is a machine credential for the
transport, not a user login (there is none, suite decision 2026-10-06).
"""

from __future__ import annotations

import ast
import hashlib
import hmac
import inspect
import json
import logging
import time
from collections.abc import Awaitable, Callable
from pathlib import Path
from typing import Any

import pydantic_core
from mcp.server.mcpserver import MCPServer
from mcp.server.mcpserver.utilities.func_metadata import ArgModelBase
from pydantic import ConfigDict, model_validator
from starlette.applications import Starlette
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import JSONResponse, Response

from .annotations import apply_annotations
from .client import ForgeClient
from .config import MCPSettings
from .context import ToolContext
from .health_gate import DEPENDENCIES, HealthGate
from .teaching import ForgeMCPServer
from .tools import CATEGORY_MODULES

logger = logging.getLogger(__name__)

SERVER_INSTRUCTIONS = """\
miForge MCP server: make weights - create a model, change one in place, or train an adapter -
through runs you define, validate and watch.

**Call `forge_howto` first.** {tool_count} tools across {category_count} categories; that tool
holds the workflow and the three result shapes.

Every tool returns one of three things:
1. the backend's resource;
2. a pending approval, {"approval_id": ..., "status": "pending", ...}: the request waits for the
   operator in miForge's Approvals view. Poll `forge_get_approval_status`. You cannot approve it;
3. {"unavailable": "<dependency>", "reason": ...}: the backend or one of its dependencies is
   down. Report the reason; do not retry in a loop.

A pending approval also carries `approval` {id, status, decide_in, then, summary}, the suite's
shared shape; `then: server_runs` means miForge runs the request itself once the operator
approves it, so poll and never call again. Hints for what to call next are in `next_steps`
[{tool, args, why}]. A failed call's text starts `<code>: <message>`, then `fix:`, then
`next:` (a JSON list), then the original error.

Starting GPU work, attaching to miServer, deleting and changing settings wait for the operator;
reads, validation, plans and dry-runs do not. A tool refuses an argument name it does not declare
and lists the names it accepts; nothing runs.
"""

REDACTED = "<redacted>"

#: Tool arguments that carry a credential. **Empty by design**: no miForge tool takes a secret
#: (D5 keeps the trainer token off the settings surface). Kept so the audit line's redaction is
#: in place the day one is added; ``test_audit.py`` pins that redaction runs before the digest.
SECRET_PARAMS: dict[str, frozenset[str]] = {}


class BearerAuthMiddleware(BaseHTTPMiddleware):
    """Constant-time bearer-token check on every HTTP request except ``/health``."""

    def __init__(self, app: Any, token: str) -> None:
        super().__init__(app)
        self._token = token

    async def dispatch(
        self, request: Request, call_next: Callable[[Request], Awaitable[Response]]
    ) -> Response:
        if request.url.path == "/health":
            return await call_next(request)
        header = request.headers.get("authorization", "")
        provided = header[7:] if header.lower().startswith("bearer ") else ""
        # Bytes, not str: `hmac.compare_digest` raises TypeError on a non-ASCII str, which
        # turned a wrong token into a 500 on an unauthenticated port (miStudio MIS-E2E-117).
        if not provided or not hmac.compare_digest(
            provided.encode("utf-8"), self._token.encode("utf-8")
        ):
            return JSONResponse({"detail": "Invalid or missing bearer token"}, status_code=401)
        return await call_next(request)


def redact(tool: str, args: dict[str, Any]) -> dict[str, Any]:
    """The arguments with every ``SECRET_PARAMS`` value replaced by ``"<redacted>"``."""
    secret = SECRET_PARAMS.get(tool, frozenset())
    return {k: (REDACTED if k in secret and v is not None else v) for k, v in args.items()}


class AuditToolLogger:
    """One structured line per tool call, carrying the caller identity (FR-012.23)."""

    @staticmethod
    def digest(tool: str, args: dict[str, Any]) -> str:
        canonical = json.dumps(redact(tool, args), sort_keys=True, default=str)
        return hashlib.sha256(canonical.encode()).hexdigest()[:12]

    @staticmethod
    def log(
        tool: str, identity: str, args: dict[str, Any], status: str, duration_ms: float
    ) -> None:
        logger.info(
            "mcp_tool_call tool=%s identity=%s args_digest=%s status=%s duration_ms=%.0f",
            tool,
            identity,
            AuditToolLogger.digest(tool, args),
            status,
            duration_ms,
        )


def tool_names_declared(module: Any) -> list[str]:
    """Tool names a module declares, read from its AST (``@mcp.tool(...)`` decorators).

    Not obtained by registering the module, which would ask the registration path to vouch
    for itself; the reachability test compares this with the live registry.
    """
    tree = ast.parse(inspect.getsource(module))
    names: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.AsyncFunctionDef | ast.FunctionDef):
            for dec in node.decorator_list:
                target = dec.func if isinstance(dec, ast.Call) else dec
                if isinstance(target, ast.Attribute) and target.attr == "tool":
                    names.append(node.name)
                    break
    return names


def _server_instructions() -> str:
    """The instructions with counts DERIVED from the tool modules (AST: building a server
    here would recurse)."""
    tool_count = sum(
        len(tool_names_declared(module))
        for modules in CATEGORY_MODULES.values()
        for module in modules
    )
    return SERVER_INSTRUCTIONS.replace("{tool_count}", str(tool_count)).replace(
        "{category_count}", str(len(CATEGORY_MODULES))
    )


def build_server(settings: MCPSettings, stdio: bool = False) -> tuple[MCPServer, ForgeClient]:
    """Create the MCPServer with only the enabled categories registered."""
    if not settings.auth_token and not stdio:
        raise SystemExit(
            "MCP_AUTH_TOKEN is required: the MCP port is reachable on the LAN and through the "
            "tunnel. Set a token, or use the stdio transport with MCP_ALLOW_ANONYMOUS=true for "
            "local development only."
        )
    if settings.allow_anonymous and not stdio:
        raise SystemExit(
            "MCP_ALLOW_ANONYMOUS is honoured on the stdio transport only. Over HTTP the port is "
            "reachable and can request GPU work, deletes and setting changes, so anonymous "
            "access is refused. Set MCP_AUTH_TOKEN instead."
        )
    if stdio and not settings.auth_token and not settings.allow_anonymous:
        raise SystemExit(
            "Set MCP_ALLOW_ANONYMOUS=true to run the stdio transport without MCP_AUTH_TOKEN."
        )

    categories = settings.enabled_categories()
    # ForgeMCPServer: every tool failure answers in the suite's error layout (017 FR-017.9).
    mcp = ForgeMCPServer("miforge", instructions=_server_instructions())
    client = ForgeClient(settings.api_url, settings.agent_identity)
    gate = HealthGate(settings.api_url)
    ctx = ToolContext(settings=settings, gate=gate)

    registered: list[str] = []
    for category, modules in CATEGORY_MODULES.items():
        if category not in categories:
            continue
        for module in modules:
            module.register(mcp, client, ctx)
        registered.append(category)
    logger.info("MCP tool categories enabled: %s", sorted(registered))
    refuse_unknown_arguments(mcp)
    apply_annotations(mcp)
    schemas = attach_output_schemas(mcp)
    shape_results(mcp, schemas)

    @mcp.custom_route("/health", methods=["GET"])  # type: ignore[untyped-decorator]
    async def health(request: Request) -> JSONResponse:
        """Never blocks on a probe (snapshots only); coarse reasons on an unauthenticated
        route, because detailed reasons carry internal URLs."""
        dependencies = {}
        for dep in DEPENDENCIES:
            available, reason = gate.snapshot(dep)
            dependencies[dep] = {"available": available, "reason": gate.public_reason(reason)}
        body: dict[str, Any] = {
            "status": "ok",
            "service": "miforge-mcp",
            "categories": sorted(registered),
            "dependencies": dependencies,
        }
        unknown = sorted(settings.unknown_categories())
        if unknown:
            body["unknown_categories"] = unknown
        return JSONResponse(body)

    async def _close_backend_clients() -> None:
        for close in (client.close, gate.aclose):
            try:
                await close()
            except Exception:
                logger.debug("Closing an MCP backend client failed", exc_info=True)

    mcp.close_backend_clients = _close_backend_clients  # type: ignore[attr-defined]
    mcp.forge_identity = settings.agent_identity  # type: ignore[attr-defined]
    return mcp, client


def _strict_arguments(tool_name: str, base: type[ArgModelBase]) -> type[ArgModelBase]:
    """``base`` (the SDK's argument model for one tool) refusing any name it does not declare.

    The refusal names the unknown arguments and every accepted one, so a caller can correct the
    call. ``hide_input_in_errors`` keeps the submitted values out of every validation error, so
    the day a tool takes a secret (``SECRET_PARAMS``) its value is not echoed back.
    """
    accepted = sorted({field.alias or name for name, field in base.model_fields.items()})
    known = set(accepted) | set(base.model_fields)

    class Strict(base):  # type: ignore[valid-type, misc]
        model_config = ConfigDict(hide_input_in_errors=True, title=f"{tool_name} arguments")

        @model_validator(mode="before")
        @classmethod
        def _refuse_unknown(cls, data: Any) -> Any:
            if isinstance(data, dict):
                unknown = sorted(str(key) for key in data if key not in known)
                if unknown:
                    raise ValueError(
                        f"unknown argument(s) {', '.join(unknown)}; {tool_name} accepts "
                        f"{', '.join(accepted) if accepted else 'no arguments'}. Nothing was run."
                    )
            return data

    return Strict


def refuse_unknown_arguments(mcp: MCPServer) -> int:
    """Make EVERY registered tool refuse an argument name it does not declare. Returns the count.

    The SDK's argument models ignore extra keys, so a misplaced or misspelt name was dropped and
    the call ran on defaults: miStudio's server changed GPU work that way (2026-11-10), and
    ``forge_list_runs`` with an unknown name here listed every run. Server-wide, applied once after
    registration, so no tool can opt out by omission; the published input schema says the same
    (``additionalProperties: false``).
    """
    manager = mcp._tool_manager
    count = 0
    for tool in manager.list_tools():
        tool.fn_metadata.arg_model = _strict_arguments(tool.name, tool.fn_metadata.arg_model)
        tool.parameters = {**tool.parameters, "additionalProperties": False}
        count += 1
    return count


# Origin: miDataworks (Onegaishimas/miDataworks) backend/src/mcp_server/server.py @ b858ed3,
# `as_one_block` and `wrap_list_results` - copied (docs/REUSE.md), the pass renamed for 017.
def as_one_block(result: Any) -> Any:
    """A list result as ONE JSON text block (FR-017.21; miDataworks' PII run L5, 2026-10-20).

    The SDK renders a returned list as one content block PER ITEM, so an empty list arrived as
    ``content: []``: indistinguishable from a dropped response, and a client that read the first
    block crashed on it. A ``None`` result is the same zero blocks. Every list (and ``None``) is
    therefore serialised here, so ``[]`` is a block that says ``[]`` and a non-empty list is one
    JSON array rather than N fragments.
    """
    if result is None:
        return "null"
    if isinstance(result, list | tuple):
        return pydantic_core.to_json(list(result), fallback=str, indent=2).decode()
    return result


#: Generated by ``backend/scripts/generate_output_schemas.py`` (017 FR-017.19); read at start.
OUTPUT_SCHEMAS_PATH = Path(__file__).with_name("output_schemas.json")


def _refuse(path: Path, why: str) -> SystemExit:
    return SystemExit(
        f"MCP output schemas at {path} {why}. The server does not start with its tools "
        "untyped. Regenerate the file: python backend/scripts/generate_output_schemas.py "
        "(and the package must ship it: [tool.setuptools.package-data] in pyproject.toml)."
    )


def read_output_schemas(
    path: Path | None = None,
) -> tuple[dict[str, dict[str, Any]], dict[str, Any]]:
    """``(typed, untyped)`` from the generated file, or ``SystemExit`` naming the path.

    The server does not start untyped: a missing file (R2 review fix OR2-017-1), and also an
    unreadable, truncated or partial one (R3 review fix OR3-017-5, 2026-11-16) - invalid JSON,
    a missing ``typed`` or ``untyped`` map, an empty ``typed`` map, or a typed entry that is
    not a schema object. ``attach_output_schemas`` then requires every registered tool to be
    named in one of the two maps.
    """
    path = OUTPUT_SCHEMAS_PATH if path is None else path
    if not path.is_file():
        raise _refuse(path, "are missing: the installed miforge package lacks the file")
    try:
        document = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, ValueError) as exc:
        raise _refuse(path, f"are unreadable ({type(exc).__name__}: {exc})") from exc
    typed = document.get("typed") if isinstance(document, dict) else None
    untyped = document.get("untyped") if isinstance(document, dict) else None
    if not isinstance(typed, dict) or not isinstance(untyped, dict):
        raise _refuse(path, "lack the 'typed' and 'untyped' maps")
    if not typed:
        raise _refuse(path, "type no tool at all")
    bad = sorted(name for name, schema in typed.items() if not isinstance(schema, dict))
    if bad:
        raise _refuse(path, f"carry a typed entry that is not a schema object: {', '.join(bad)}")
    return typed, untyped


def load_output_schemas(path: Path | None = None) -> dict[str, dict[str, Any]]:
    """``{tool: outputSchema}`` for the typed tools (see :func:`read_output_schemas`)."""
    return read_output_schemas(path)[0]


def attach_output_schemas(mcp: MCPServer) -> dict[str, dict[str, Any]]:
    """Publish each typed tool's ``outputSchema`` (017 FR-017.19). Returns the attached map.

    ``Tool.output_schema`` is a cached property the SDK reads at ``list_tools()``, so it is
    set here, before the first listing (FPRD §6). Every registered tool must be named in the
    file, typed or untyped: a tool the file does not know means a stale or partial file, and
    the server refuses to start (OR3-017-5).
    """
    schemas, untyped = read_output_schemas()
    tools = mcp._tool_manager.list_tools()
    unknown = sorted(t.name for t in tools if t.name not in schemas and t.name not in untyped)
    if unknown:
        raise _refuse(
            OUTPUT_SCHEMAS_PATH, f"do not name these registered tools: {', '.join(unknown)}"
        )
    attached: dict[str, dict[str, Any]] = {}
    for tool in tools:
        schema = schemas.get(tool.name)
        if schema is not None:
            tool.__dict__["output_schema"] = schema
            attached[tool.name] = schema
    return attached


def structured(value: Any) -> dict[str, Any]:
    """``structuredContent`` for a schema'd tool: a dict as is, anything else under ``result``."""
    return value if isinstance(value, dict) else {"result": value}


def shape_results(mcp: MCPServer, schemas: dict[str, dict[str, Any]] | None = None) -> int:
    """Route EVERY registered tool's result through one pass. Returns the count.

    A list (or ``None``) becomes one JSON block (:func:`as_one_block`); a tool with an output
    schema also answers ``structuredContent`` (017 FR-017.21; A30: a non-dict value - a list, a
    scalar or ``None`` - under ``result``). Server-wide, applied once after registration, so no
    tool opts out by omission.
    """
    from mcp.server.mcpserver.utilities.func_metadata import _convert_to_content
    from mcp_types import CallToolResult

    typed = set(schemas or {})
    manager = mcp._tool_manager
    wrapped = 0
    for tool in manager.list_tools():

        def make(
            fn: Callable[..., Awaitable[Any]], has_schema: bool
        ) -> Callable[..., Awaitable[Any]]:
            async def one_block(*args: Any, **kwargs: Any) -> Any:
                value = await fn(*args, **kwargs)
                text = as_one_block(value)
                if not has_schema:
                    return text
                return CallToolResult(
                    content=_convert_to_content(text), structured_content=structured(value)
                )

            return one_block

        tool.fn = make(tool.fn, tool.name in typed)
        wrapped += 1
    return wrapped


def build_http_app(mcp: MCPServer, settings: MCPSettings) -> Starlette:
    """The streamable-HTTP ASGI app ``__main__`` serves: stateless JSON, bearer-gated.

    ``host`` is passed on purpose: the app factory defaults to ``127.0.0.1``, which turns on
    DNS-rebinding protection with a loopback-only ``Host`` allow-list, so every request
    arriving as ``miforge-mcp:8765`` or through the ingress would be ``421``.
    """
    app = mcp.streamable_http_app(json_response=True, stateless_http=True, host=settings.host)
    app.add_middleware(BearerAuthMiddleware, token=settings.auth_token)
    return app


def _outcome(result: Any) -> str:
    # A typed tool answers a CallToolResult (017 FR-017.21): classify its structured value.
    structured_content = getattr(result, "structured_content", None)
    if structured_content is not None:
        result = structured_content
    if isinstance(result, dict) and result.get("status") == "pending" and "approval_id" in result:
        return "pending"
    if isinstance(result, dict) and "unavailable" in result:
        return "unavailable"
    return "ok"


def wrap_tool_with_audit(mcp: MCPServer) -> int:
    """Wrap every registered tool's function with the audit logger. Returns the count."""
    identity = getattr(mcp, "forge_identity", "agent:unknown")
    manager = mcp._tool_manager
    wrapped = 0
    for tool in manager.list_tools():

        def make_wrapper(
            fn: Callable[..., Awaitable[Any]], tool_name: str
        ) -> Callable[..., Awaitable[Any]]:
            async def wrapper(*args: Any, **kwargs: Any) -> Any:
                start = time.monotonic()
                try:
                    result = await fn(*args, **kwargs)
                except Exception as exc:
                    AuditToolLogger.log(
                        tool_name,
                        identity,
                        kwargs,
                        f"error:{type(exc).__name__}",
                        (time.monotonic() - start) * 1000,
                    )
                    raise
                AuditToolLogger.log(
                    tool_name, identity, kwargs, _outcome(result), (time.monotonic() - start) * 1000
                )
                return result

            return wrapper

        tool.fn = make_wrapper(tool.fn, tool.name)
        wrapped += 1
    return wrapped
