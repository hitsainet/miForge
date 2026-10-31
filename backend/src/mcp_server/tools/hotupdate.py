"""Category ``hotupdate``: continuing a run, and miLLM's live attachment state.

Two tools call ``POST /runs/{run_id}/continue``. ``forge_plan_continue`` always sends
``submit: false`` and gets the plan back at once; ``forge_continue_run`` always sends
``submit: true`` and waits for the operator (``hotupdate_attach``: D52's continue is for an
attached run, which trains miLLM's served weights). The split is by payload because the REST
gate's class is decided by the payload (FR-012.15).
"""

from __future__ import annotations

from typing import Annotated, Any

from mcp.server.mcpserver import MCPServer
from pydantic import Field

from ..client import ForgeClient
from ..context import CORE, JOB, READ, WAITS, RunId, ToolContext
from ..health_gate import gated_call

FromStep = Annotated[int | None, Field(ge=0, description="The checkpoint step; omit for latest.")]
ConfigHash = Annotated[
    str | None, Field(max_length=128, description="The optimizer config hash in force.")
]


def register(mcp: MCPServer, client: ForgeClient, ctx: ToolContext) -> None:
    @mcp.tool()
    async def forge_plan_continue(
        run_id: RunId, step: FromStep = None, optimizer_config_hash: ConfigHash = None
    ) -> Any:
        """Plan continuing a run from a checkpoint: which checkpoint, whether its trainer state
        is restored or fresh, and why. Submits nothing; runs at once."""
        body = {"step": step, "optimizer_config_hash": optimizer_config_hash, "submit": False}
        return await gated_call(
            ctx.gate, READ, lambda: client.post(f"/runs/{run_id}/continue", json_body=body)
        )

    @mcp.tool(
        description=(
            "Continue an attached run from the checkpoint miLLM saved for it (D52): verified, "
            "trainer state restored, a new attempt of the same run id. miLLM must already serve "
            "that checkpoint." + WAITS + " Action hotupdate_attach."
        )
    )
    async def forge_continue_run(
        run_id: RunId, step: FromStep = None, optimizer_config_hash: ConfigHash = None
    ) -> Any:
        """Continue an attached run from its last save."""
        body = {"step": step, "optimizer_config_hash": optimizer_config_hash, "submit": True}
        return await gated_call(
            ctx.gate, JOB, lambda: client.post(f"/runs/{run_id}/continue", json_body=body)
        )

    @mcp.tool()
    async def forge_attachment_status() -> Any:
        """miLLM's attachment state now, read live: configured, reachable, attached, holder,
        resident model, weights state."""
        return await gated_call(ctx.gate, CORE, lambda: client.get("/attachment"))
