"""Category ``drafts``: the seven-step pack authoring flow, its panels and dry-run.

Authoring writes a draft and destroys nothing (D9); the panels and the dry-run compute and
persist nothing (D8). ``forge_delete_draft`` waits for the operator (``delete``).
"""

from __future__ import annotations

from typing import Annotated, Any, Literal

from mcp.server.mcpserver import MCPServer
from pydantic import Field

from ..client import ForgeClient
from ..context import ID_PATTERN, READ, WAITS, ToolContext
from ..health_gate import gated_call

DraftId = Annotated[str, Field(pattern=ID_PATTERN, description="The draft id.")]
Slot = Literal[
    "prompt_set",
    "corpus",
    "preference_pairs",
    "selector_config",
    "grader_config",
    "test_set",
    "retention_set",
]


def register(mcp: MCPServer, client: ForgeClient, ctx: ToolContext) -> None:
    @mcp.tool()
    async def forge_list_drafts() -> Any:
        """Every pack draft in progress, with its current step."""
        return await gated_call(ctx.gate, READ, lambda: client.get("/pack-drafts"))

    @mcp.tool()
    async def forge_get_draft(draft_id: DraftId) -> Any:
        """One draft: its answers so far and each step's state."""
        return await gated_call(ctx.gate, READ, lambda: client.get(f"/pack-drafts/{draft_id}"))

    @mcp.tool()
    async def forge_review_draft(draft_id: DraftId) -> Any:
        """Step 7: what is still missing, by name."""
        return await gated_call(
            ctx.gate, READ, lambda: client.get(f"/pack-drafts/{draft_id}/validity")
        )

    @mcp.tool()
    async def forge_create_draft(
        name: Annotated[str | None, Field(pattern=ID_PATTERN, description="The pack name.")] = None,
        version: Annotated[str, Field(pattern=ID_PATTERN, description="The version.")] = "1",
        domain: Annotated[str | None, Field(max_length=200, description="The domain.")] = None,
    ) -> Any:
        """Start a pack draft. Writes a draft; runs at once."""
        return await gated_call(
            ctx.gate,
            READ,
            lambda: client.post(
                "/pack-drafts", json_body={"name": name, "version": version, "domain": domain}
            ),
        )

    @mcp.tool()
    async def forge_commit_draft_step(
        draft_id: DraftId,
        number: Annotated[int, Field(ge=1, le=7, description="The step, 1 to 7.")],
        answers: Annotated[
            dict[str, Any],
            Field(description="{name?, version?, domain?, domain_given?, slots?, skip?}"),
        ],
    ) -> Any:
        """Commit one step's answers. Idempotent; runs at once."""
        return await gated_call(
            ctx.gate,
            READ,
            lambda: client.put(f"/pack-drafts/{draft_id}/steps/{number}", json_body=answers),
        )

    @mcp.tool()
    async def forge_sample_slot(draft_id: DraftId, slot: Slot) -> Any:
        """A seeded sample of a slot's part, for the authoring panel. Persists nothing."""
        return await gated_call(
            ctx.gate, READ, lambda: client.post(f"/pack-drafts/{draft_id}/samples/{slot}")
        )

    @mcp.tool()
    async def forge_score_sample(
        draft_id: DraftId,
        slot: Slot,
        request: Annotated[
            dict[str, Any],
            Field(description="{config, catalog?, form?, provider?, n?, floor?}"),
        ],
    ) -> Any:
        """Score a slot's sample with a selector or grader through a provider. Persists
        nothing."""
        return await gated_call(
            ctx.gate,
            READ,
            lambda: client.post(f"/pack-drafts/{draft_id}/samples/{slot}/score", json_body=request),
        )

    @mcp.tool()
    async def forge_dry_run_draft(
        draft_id: DraftId,
        request: Annotated[dict[str, Any], Field(description="{config, catalog?, provider?, n?}")],
    ) -> Any:
        """Dry-run the draft's selector over sampled completions. Persists nothing (D8)."""
        return await gated_call(
            ctx.gate,
            READ,
            lambda: client.post(f"/pack-drafts/{draft_id}/dry-run", json_body=request),
        )

    @mcp.tool()
    async def forge_commit_draft(draft_id: DraftId) -> Any:
        """Commit a complete draft as a pack version. Refused, with what is missing, otherwise."""
        return await gated_call(
            ctx.gate, READ, lambda: client.post(f"/pack-drafts/{draft_id}/commit")
        )

    @mcp.tool(description="Delete a pack draft." + WAITS + " Action delete.")
    async def forge_delete_draft(draft_id: DraftId) -> Any:
        """Delete a pack draft."""
        return await gated_call(ctx.gate, READ, lambda: client.delete(f"/pack-drafts/{draft_id}"))
