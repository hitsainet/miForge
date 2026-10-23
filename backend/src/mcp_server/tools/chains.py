"""Category ``chains``: curricula of runs. Creating one waits (``gpu_start``); stopping does not."""

from __future__ import annotations

from typing import Annotated, Any

from mcp.server.mcpserver import MCPServer
from pydantic import Field

from ..client import ForgeClient
from ..context import ID_PATTERN, JOB, READ, WAITS, ToolContext
from ..health_gate import gated_call

ChainId = Annotated[str, Field(pattern=ID_PATTERN, description="The chain id.")]


def register(mcp: MCPServer, client: ForgeClient, ctx: ToolContext) -> None:
    @mcp.tool()
    async def forge_list_chains(
        limit: Annotated[int | None, Field(ge=1, le=500, description="Page size.")] = None,
    ) -> Any:
        """Every chain, newest first, with its position, status and halt code."""
        return await gated_call(ctx.gate, READ, lambda: client.get("/chains", limit=limit))

    @mcp.tool()
    async def forge_get_chain(chain_id: ChainId) -> Any:
        """One chain: each stage's outcome and the advance decision for each finished stage."""
        return await gated_call(ctx.gate, READ, lambda: client.get(f"/chains/{chain_id}"))

    @mcp.tool(
        description=(
            "Create a chain (R-02.27): every stage validated now, the card leased for the chain, "
            "stage 0 started once approved."
            + WAITS
            + " Action gpu_start, or hotupdate_attach when stage 0 attaches to miLLM."
        )
    )
    async def forge_create_chain(
        spec: Annotated[
            dict[str, Any],
            Field(description="{name, stages: [{definition, advance_on}] (two or more)}"),
        ],
    ) -> Any:
        """Create a chain."""
        return await gated_call(ctx.gate, JOB, lambda: client.post("/chains", json_body=spec))

    @mcp.tool()
    async def forge_stop_chain(
        chain_id: ChainId,
        requested_by: Annotated[
            str | None, Field(max_length=120, description="An audit label.")
        ] = None,
    ) -> Any:
        """Halt a chain and stop its running stage. Runs at once (stopping is never gated)."""
        body = None if requested_by is None else {"requested_by": requested_by}
        return await gated_call(
            ctx.gate, READ, lambda: client.post(f"/chains/{chain_id}/stop", json_body=body)
        )
