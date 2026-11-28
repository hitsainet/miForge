# Origin (design): miDataworks 022 FTDD section 7 (suite ENH-012 CHAIN_PLAN A13, A30), written for
# miForge (017-D11, 017-D15): miDataworks 022 was not built when this file was written.
"""Errors that teach (017 FR-017.9 - .11, .32; ADR-028).

``ForgeMCPServer`` overrides ``call_tool`` so EVERY tool failure - a backend refusal, an unknown
or invalid argument, an unknown tool, a crash - reaches the client as an ``isError`` result in the
suite's error layout (``mistudio-result-language-v1.json`` ``x-error-text-format``)::

    <code>: <one-line message>
    fix: <what to do>
    next: <JSON list of next_step>        ("[]" when there is none)
    <the original text, unchanged>

The line-1 code is lower case and valid as the contract's ``error_code`` (A30): miForge's own
refusal codes already are; the MCP layer's transport codes (``BACKEND_TIMEOUT`` ...) are lowered
on line 1 only and keep their spelling in the original text. Argument VALUES never enter the
text: the strict argument models set ``hide_input_in_errors`` and only field names and messages
are read here.

**Converting keeps the log line** (FR-017.32). The SDK's handler logs each failure before it
answers (``mcp/server/mcpserver/server.py`` ``_handle_call_tool``): a crash with its traceback, an
argument error by field names, anything else by its text. An override that returns a result
instead of raising would silence all three, so :func:`log_tool_failure` repeats those three
branches first, on the SDK's own logger.
"""

from __future__ import annotations

import json
import logging
import re
from typing import Any

from mcp.server.mcpserver import MCPServer
from mcp.server.mcpserver.exceptions import ToolError, UnexpectedToolError
from mcp_types import CallToolResult, TextContent
from pydantic import ValidationError

from .client import ForgeError

#: The SDK's handler logs here; the converted failures keep logging to the same place.
sdk_logger = logging.getLogger("mcp.server.mcpserver.server")

MESSAGE_LIMIT = 200
_PREFIX = re.compile(r"^Error executing tool [^:]*: ?")

#: The MCP layer's transport codes (``client.py``) and the server's own, as raised.
TRANSPORT_CODES = frozenset(
    {"BACKEND_TIMEOUT", "BACKEND_UNREACHABLE", "NON_JSON_ERROR", "NON_JSON_RESPONSE"}
)


def line_one_code(code: str) -> str:
    """A code as the contract's ``error_code``: lower case, ``[^a-z0-9_.]`` -> ``_``."""
    lowered = re.sub(r"[^a-z0-9_.]", "_", str(code).lower()).strip("_.")
    if not lowered or not lowered[0].isalpha():
        return "error"
    return lowered[:64]


def _one_line(text: str) -> str:
    first = (text or "").strip().split("\n", 1)[0].strip()
    if len(first) > MESSAGE_LIMIT:
        first = first[: MESSAGE_LIMIT - 3] + "..."
    return first or "the call failed"


def _step(tool: str, why: str, args: dict[str, Any] | None = None) -> dict[str, Any]:
    return {"tool": tool, "args": dict(args or {}), "why": why}


def log_tool_failure(name: str, exc: Exception) -> None:
    """The SDK handler's three log branches, verbatim in effect (FR-017.32)."""
    if isinstance(exc, ToolError) and not isinstance(exc, UnexpectedToolError):
        if isinstance(exc.__cause__, ValidationError):
            # Field names only: the rejected values are the caller's data.
            fields = sorted(
                {".".join(str(part) for part in err["loc"]) for err in exc.__cause__.errors()}
            )
            sdk_logger.info("Tool %r rejected arguments: %r", name, fields)
        else:
            sdk_logger.info("Tool %r failed: %r", name, str(exc))
    else:
        sdk_logger.exception("Tool %r raised an unexpected exception", name)


def _forge_error(exc: BaseException) -> ForgeError | None:
    seen: BaseException | None = exc
    while seen is not None:
        if isinstance(seen, ForgeError):
            return seen
        seen = seen.__cause__
    return None


def teach(name: str, exc: Exception) -> CallToolResult:
    """The four-part error text for one failed call."""
    original = _PREFIX.sub("", str(exc)).strip()
    fix = ""
    steps: list[dict[str, Any]] = []
    forge = _forge_error(exc)
    if isinstance(exc, UnexpectedToolError):
        code, message = "internal_error", f"{name} failed unexpectedly; the server logged it"
        fix = (
            "report it; the server log names the failure. Retry only if the call is safe to repeat"
        )
        original = original or f"Error executing tool {name}"
    elif forge is not None:
        code, message = forge.code, forge.message or str(forge)
        original = str(forge)
        details = forge.details or {}
        listed = details.get("next_steps")
        if isinstance(listed, list):
            steps = [s for s in listed if isinstance(s, dict) and "tool" in s]
        if isinstance(details.get("fix"), str):
            fix = details["fix"]
        elif steps and isinstance(steps[0].get("why"), str):
            fix = steps[0]["why"]
        if forge.code in TRANSPORT_CODES:
            steps = steps or [_step("forge_health", "check the backend and its dependencies")]
            if forge.code == "BACKEND_TIMEOUT":
                steps.append(
                    _step("forge_list_runs", "see whether the request was already applied")
                )
            fix = fix or "check miForge's health before retrying"
        fix = fix or "read the message; correct the request and call again"
    elif isinstance(exc.__cause__, ValidationError):
        errors = exc.__cause__.errors()
        unknown = [e for e in errors if "unknown argument" in str(e.get("msg", ""))]
        if unknown:
            code = "unknown_arguments"
            message = str(unknown[0]["msg"]).removeprefix("Value error, ")
            fix = "call again with only the accepted argument names listed above"
        else:
            code = "invalid_arguments"
            fields = sorted({".".join(str(p) for p in e["loc"]) for e in errors})
            message = f"{name}: invalid value for {', '.join(fields) or 'the arguments'}"
            fix = "correct " + (", ".join(fields) or "the arguments") + " and call again"
    elif original.startswith("Unknown tool"):
        code, message = "unknown_tool", original
        fix = "call forge_howto for the tools and the workflow"
        steps = [_step("forge_howto", "it lists the workflow and the tools")]
    else:
        code, message = "error", original
        fix = "read the message; correct the request and call again"
    text = "\n".join(
        [
            f"{line_one_code(code)}: {_one_line(message)}",
            f"fix: {_one_line(fix)}",
            f"next: {json.dumps(steps, separators=(',', ':'), default=str)}",
            original,
        ]
    )
    return CallToolResult(content=[TextContent(type="text", text=text)], is_error=True)


class ForgeMCPServer(MCPServer):
    """``MCPServer`` whose tool failures answer in the suite's error layout (FR-017.9)."""

    async def call_tool(self, name: str, arguments: dict[str, Any], context: Any = None) -> Any:
        try:
            return await super().call_tool(name, arguments, context)
        except ToolError as exc:  # UnexpectedToolError included
            log_tool_failure(name, exc)
            return teach(name, exc)


__all__ = ["ForgeMCPServer", "line_one_code", "log_tool_failure", "teach"]
