"""Category ``adapters``: adapters, the merge record, and merging (``gpu_start``)."""

from __future__ import annotations

from typing import Annotated, Any

from mcp.server.mcpserver import MCPServer
from pydantic import Field

from ..client import ForgeClient
from ..context import ADAPTER_PATTERN, ID_PATTERN, JOB, READ, WAITS, RunId, Step, ToolContext
from ..health_gate import gated_call

AdapterName = Annotated[
    str,
    Field(pattern=ADAPTER_PATTERN, description="<base_model>:<run_id>:adapter-<step>."),
]


def register(mcp: MCPServer, client: ForgeClient, ctx: ToolContext) -> None:
    @mcp.tool()
    async def forge_list_adapters(
        run_id: Annotated[str | None, Field(pattern=ID_PATTERN, description="Filter.")] = None,
    ) -> Any:
        """Every adapter, with its base, rank and scaling."""
        return await gated_call(ctx.gate, READ, lambda: client.get("/adapters", run_id=run_id))

    @mcp.tool()
    async def forge_get_adapter(name: AdapterName) -> Any:
        """One adapter's manifest."""
        return await gated_call(ctx.gate, READ, lambda: client.get(f"/adapters/{name}"))

    @mcp.tool()
    async def forge_list_merges(
        adapter_name: Annotated[
            str | None, Field(pattern=ADAPTER_PATTERN, description="Filter.")
        ] = None,
        limit: Annotated[int | None, Field(ge=1, le=500, description="Page size.")] = None,
    ) -> Any:
        """The merge record: written, interrupted or refused, each with its reason."""
        return await gated_call(
            ctx.gate, READ, lambda: client.get("/merges", adapter_name=adapter_name, limit=limit)
        )

    @mcp.tool(
        description=(
            "Merge an adapter into its base, writing a merged checkpoint of the named run at the "
            "named step. Refused before anything is queued when no unmerged-base checkpoint exists."
            + WAITS
            + " Action gpu_start."
        )
    )
    async def forge_merge_adapter(name: AdapterName, run_id: RunId, step: Step) -> Any:
        """Merge an adapter into its base."""
        return await gated_call(
            ctx.gate,
            JOB,
            lambda: client.post(f"/adapters/{name}/merge", run_id=run_id, step=step),
        )
