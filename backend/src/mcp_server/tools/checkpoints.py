"""Category ``checkpoints``: the checkpoint index, the artifact forest, saves and branches.

``forge_request_save`` adds an artifact (D9) and ``forge_plan_branch`` returns a plan; both
run at once. ``forge_verify_checkpoint`` recomputes a hash and changes nothing.
"""

from __future__ import annotations

from typing import Annotated, Any

from mcp.server.mcpserver import MCPServer
from pydantic import Field

from ..client import ForgeClient
from ..context import ID_PATTERN, READ, RunId, Step, ToolContext
from ..health_gate import gated_call


def register(mcp: MCPServer, client: ForgeClient, ctx: ToolContext) -> None:
    @mcp.tool()
    async def forge_list_all_checkpoints(
        run_id: Annotated[str | None, Field(pattern=ID_PATTERN, description="Filter.")] = None,
        domain: Annotated[str | None, Field(max_length=200, description="Filter.")] = None,
        base_model: Annotated[str | None, Field(max_length=200, description="Filter.")] = None,
        limit: Annotated[int | None, Field(ge=1, le=1000, description="Page size.")] = None,
    ) -> Any:
        """Every indexed checkpoint across runs."""
        return await gated_call(
            ctx.gate,
            READ,
            lambda: client.get(
                "/checkpoints", run_id=run_id, domain=domain, base_model=base_model, limit=limit
            ),
        )

    @mcp.tool()
    async def forge_list_checkpoints(run_id: RunId) -> Any:
        """One run's checkpoints, by step."""
        return await gated_call(ctx.gate, READ, lambda: client.get(f"/runs/{run_id}/checkpoints"))

    @mcp.tool()
    async def forge_get_checkpoint(run_id: RunId, step: Step) -> Any:
        """One checkpoint's manifest: parent, base model, weights hash, scores."""
        return await gated_call(
            ctx.gate, READ, lambda: client.get(f"/runs/{run_id}/checkpoints/{step}")
        )

    @mcp.tool()
    async def forge_verify_checkpoint(run_id: RunId, step: Step) -> Any:
        """Recompute a checkpoint's weights hash against its manifest. Changes nothing."""
        return await gated_call(
            ctx.gate, READ, lambda: client.post(f"/runs/{run_id}/checkpoints/{step}/verify")
        )

    @mcp.tool()
    async def forge_get_checkpoint_lineage(
        run_id: RunId,
        step: Step,
        limit: Annotated[int | None, Field(ge=1, le=4096, description="Hops.")] = None,
    ) -> Any:
        """The parent chain, walked through manifests to the root."""
        return await gated_call(
            ctx.gate,
            READ,
            lambda: client.get(f"/runs/{run_id}/checkpoints/{step}/lineage", limit=limit),
        )

    @mcp.tool()
    async def forge_get_tree(
        run_id: Annotated[str | None, Field(pattern=ID_PATTERN, description="Filter.")] = None,
        domain: Annotated[str | None, Field(max_length=200, description="Filter.")] = None,
        kind: Annotated[str | None, Field(max_length=32, description="Node kind filter.")] = None,
        limit: Annotated[int | None, Field(ge=1, le=2000, description="Page size.")] = None,
        offset: Annotated[int | None, Field(ge=0, description="Items to skip.")] = None,
    ) -> Any:
        """The artifact forest: checkpoints joined on parent, adapters on their base."""
        return await gated_call(
            ctx.gate,
            READ,
            lambda: client.get(
                "/tree", run_id=run_id, domain=domain, kind=kind, limit=limit, offset=offset
            ),
        )

    @mcp.tool()
    async def forge_request_save(run_id: RunId) -> Any:
        """Ask a running run to save at its next boundary. Adds a checkpoint; runs at once."""
        return await gated_call(ctx.gate, READ, lambda: client.post(f"/runs/{run_id}/checkpoints"))

    @mcp.tool()
    async def forge_plan_branch(
        run_id: RunId,
        step: Step,
        new_run_id: Annotated[str, Field(pattern=ID_PATTERN, description="The branch's run id.")],
        optimizer_config_hash: Annotated[
            str | None, Field(max_length=128, description="The new run's optimizer config hash.")
        ] = None,
    ) -> Any:
        """Plan a branch from any checkpoint: a new run id, parent = this checkpoint. Returns
        the plan; submits nothing."""
        return await gated_call(
            ctx.gate,
            READ,
            lambda: client.post(
                f"/runs/{run_id}/checkpoints/{step}/branch",
                json_body={"run_id": new_run_id, "optimizer_config_hash": optimizer_config_hash},
            ),
        )
