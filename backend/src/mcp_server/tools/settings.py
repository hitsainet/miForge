"""Category ``settings``: read every setting and the change history; change or clear an
allow-listed one (``config_write``). Secrets are never readable and never writable through this
surface (D5). The history and the reasons are ENH-003's (013 G3)."""

from __future__ import annotations

from typing import Annotated, Any

from mcp.server.mcpserver import MCPServer
from pydantic import Field

from ..client import ForgeClient
from ..context import KEY_PATTERN, READ, WAITS, ToolContext
from ..health_gate import gated_call

Key = Annotated[str, Field(pattern=KEY_PATTERN, description="A setting name.")]
Reason = Annotated[
    str | None,
    Field(default=None, max_length=2000, description="Why, in a sentence. Kept in the history."),
]


def register(mcp: MCPServer, client: ForgeClient, ctx: ToolContext) -> None:
    @mcp.tool()
    async def forge_get_settings() -> Any:
        """Every setting: effective value, source (env or override) and whether it is
        runtime-writable. The trainer token and URL passwords are redacted."""
        return await gated_call(ctx.gate, READ, lambda: client.get("/settings"))

    @mcp.tool()
    async def forge_get_settings_history(
        key: Annotated[
            str | None,
            Field(default=None, pattern=KEY_PATTERN, description="Only this setting's changes."),
        ] = None,
        limit: Annotated[
            int, Field(default=50, ge=1, le=500, description="At most this many.")
        ] = 50,
    ) -> Any:
        """The settings change history, newest first: each change's old and new value, who
        made it (the operator, or an agent with the approval that let it run) and why."""
        return await gated_call(
            ctx.gate, READ, lambda: client.get("/settings/history", key=key, limit=limit)
        )

    @mcp.tool(
        description=(
            "Set a runtime override for an allow-listed setting (checkpoint pruning, miLLM and "
            "miDataworks URLs, log_level, hot-update timeouts and tolerances). Any other key is "
            "refused. Enabling pruning deletes checkpoints." + WAITS + " Action config_write."
        )
    )
    async def forge_set_setting(
        key: Key,
        value: Annotated[Any, Field(description="The new value, typed as the setting.")],
        reason: Reason = None,
    ) -> Any:
        """Set a runtime override."""
        body: dict[str, Any] = {"value": value}
        if reason is not None:
            body["reason"] = reason
        return await gated_call(
            ctx.gate, READ, lambda: client.put(f"/settings/{key}", json_body=body)
        )

    @mcp.tool(
        description=(
            "Clear a runtime override; the environment value applies again."
            + WAITS
            + " Action config_write."
        )
    )
    async def forge_reset_setting(key: Key, reason: Reason = None) -> Any:
        """Clear a runtime override."""
        return await gated_call(
            ctx.gate, READ, lambda: client.delete(f"/settings/{key}", reason=reason)
        )
