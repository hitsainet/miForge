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
from typing import Any

from mcp.server.mcpserver import MCPServer
from starlette.applications import Starlette
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import JSONResponse, Response

from .client import ForgeClient
from .config import MCPSettings
from .context import ToolContext
from .health_gate import DEPENDENCIES, HealthGate
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

Starting GPU work, attaching to miLLM, deleting and changing settings wait for the operator;
reads, validation, plans and dry-runs do not.
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
    mcp = MCPServer("miforge", instructions=_server_instructions())
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
