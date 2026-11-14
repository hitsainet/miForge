"""Category ``inputs``: signal sources, the part index, and miDataworks as an optional source.

``forge_import_part`` adds a part and deletes nothing, so it runs at once (D9); so do
``forge_archive_part`` and ``forge_unarchive_part``, which only hide and show one.
"""

from __future__ import annotations

from typing import Annotated, Any

from mcp.server.mcpserver import MCPServer
from pydantic import Field

from ..client import ForgeClient
from ..context import JOB, READ, Id, ToolContext
from ..health_gate import gated_call


def register(mcp: MCPServer, client: ForgeClient, ctx: ToolContext) -> None:
    @mcp.tool()
    async def forge_list_signal_sources() -> Any:
        """The four signal sources (corpus, self-generation, preference pairs, scored
        completions) and what each needs."""
        return await gated_call(ctx.gate, READ, lambda: client.get("/signal-sources"))

    @mcp.tool()
    async def forge_list_parts(
        part_kind: Annotated[str | None, Field(max_length=32, description="Filter.")] = None,
        include_archived: Annotated[
            bool, Field(description="Also list archived parts (hidden by default).")
        ] = False,
    ) -> Any:
        """The indexed dataset parts (prompt sets, corpora, test and retention sets...).
        Archived parts are left out unless include_archived."""
        return await gated_call(
            ctx.gate,
            READ,
            lambda: client.get("/parts", part_kind=part_kind, include_archived=include_archived),
        )

    @mcp.tool()
    async def forge_get_part(part_id: Id) -> Any:
        """One part: its provenance, splits and content hash."""
        return await gated_call(ctx.gate, READ, lambda: client.get(f"/parts/{part_id}"))

    @mcp.tool()
    async def forge_validate_part(
        manifest: Annotated[dict[str, Any], Field(description="A dataset-version manifest.")],
    ) -> Any:
        """Validate a part manifest against the vendored contract. Stores nothing."""
        return await gated_call(
            ctx.gate,
            READ,
            lambda: client.post("/parts/validate", json_body={"manifest": manifest}),
        )

    @mcp.tool()
    async def forge_import_part(
        request: Annotated[
            dict[str, Any],
            Field(
                description="{source, locator, recorded_source?, external_provenance?, "
                "wait?} as POST /parts/import."
            ),
        ],
    ) -> Any:
        """Import a dataset part (from a directory or miDataworks). Adds a part; runs at once."""
        return await gated_call(
            ctx.gate, JOB, lambda: client.post("/parts/import", json_body=request)
        )

    @mcp.tool()
    async def forge_midataworks_status(
        probe: Annotated[bool | None, Field(description="Probe the service now.")] = None,
    ) -> Any:
        """Whether miDataworks is configured and reachable, and what is unavailable without it."""
        return await gated_call(
            ctx.gate, READ, lambda: client.get("/midataworks/status", probe=probe)
        )

    @mcp.tool()
    async def forge_midataworks_available_parts(
        kind: Annotated[str | None, Field(max_length=32, description="Filter.")] = None,
    ) -> Any:
        """Dataset versions miDataworks offers as parts; refuses by name when not configured."""
        return await gated_call(
            ctx.gate, READ, lambda: client.get("/midataworks/available-parts", kind=kind)
        )

    @mcp.tool()
    async def forge_archive_part(part_id: Id) -> Any:
        """Hide a part from the default part lists and pickers. Deletes nothing: forge_get_part
        still reads it and a pack already pinning it is untouched. Runs at once (a
        non-deleting write); forge_unarchive_part reverses it."""
        return await gated_call(ctx.gate, READ, lambda: client.post(f"/parts/{part_id}/archive"))

    @mcp.tool()
    async def forge_unarchive_part(part_id: Id) -> Any:
        """Return an archived part to the default part lists. Runs at once."""
        return await gated_call(ctx.gate, READ, lambda: client.post(f"/parts/{part_id}/unarchive"))
