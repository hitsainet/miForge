"""Category ``packs``: the pack registry, export and import, and the one delete (``delete``).

Creating, importing and cloning a pack add an immutable version and destroy nothing, so they
run at once (D9). ``forge_delete_pack`` waits for the operator.
"""

from __future__ import annotations

from typing import Annotated, Any

from mcp.server.mcpserver import MCPServer
from pydantic import Field

from ..client import ForgeClient
from ..context import ID_PATTERN, READ, WAITS, Id, ToolContext
from ..health_gate import gated_call

PackId = Annotated[str, Field(pattern=ID_PATTERN, description="The pack's name.")]
Version = Annotated[str, Field(pattern=ID_PATTERN, description="The pack version.")]


def register(mcp: MCPServer, client: ForgeClient, ctx: ToolContext) -> None:
    @mcp.tool()
    async def forge_list_packs() -> Any:
        """Every pack version, with its domain, validity and the runs that used it."""
        return await gated_call(ctx.gate, READ, lambda: client.get("/packs"))

    @mcp.tool()
    async def forge_get_pack(pack_id: PackId, version: Version) -> Any:
        """One pack version: its slots and where each part comes from."""
        return await gated_call(
            ctx.gate, READ, lambda: client.get(f"/packs/{pack_id}/versions/{version}")
        )

    @mcp.tool()
    async def forge_get_pack_validity(pack_id: PackId, version: Version) -> Any:
        """Whether a pack version can define a run, and every missing or unresolved slot."""
        return await gated_call(
            ctx.gate, READ, lambda: client.get(f"/packs/{pack_id}/versions/{version}/validity")
        )

    @mcp.tool()
    async def forge_get_pack_lineage(pack_id: PackId, version: Version) -> Any:
        """Where a pack version came from (cloned from, imported) and what derives from it."""
        return await gated_call(
            ctx.gate, READ, lambda: client.get(f"/packs/{pack_id}/versions/{version}/lineage")
        )

    @mcp.tool()
    async def forge_export_pack(pack_id: PackId, version: Version) -> Any:
        """The portable pack document, for another installation's forge_import_pack."""
        return await gated_call(
            ctx.gate, READ, lambda: client.get(f"/packs/{pack_id}/versions/{version}/export")
        )

    @mcp.tool()
    async def forge_get_pack_part_ref(part_id: Id) -> Any:
        """An indexed part as the reference a pack slot pins."""
        return await gated_call(ctx.gate, READ, lambda: client.get(f"/pack-part-refs/{part_id}"))

    @mcp.tool()
    async def forge_create_pack(
        request: Annotated[
            dict[str, Any],
            Field(description="{name, version?, domain?, slots, addressed_through?}"),
        ],
    ) -> Any:
        """Create a pack version from slot documents. Immutable once written; runs at once."""
        return await gated_call(ctx.gate, READ, lambda: client.post("/packs", json_body=request))

    @mcp.tool()
    async def forge_import_pack(
        document: Annotated[dict[str, Any], Field(description="An exported pack document.")],
    ) -> Any:
        """Import an exported pack document; an unresolvable one is stored and reported."""
        return await gated_call(
            ctx.gate, READ, lambda: client.post("/packs/import", json_body=document)
        )

    @mcp.tool()
    async def forge_clone_pack(
        pack_id: PackId,
        version: Version,
        into: Annotated[str, Field(pattern=ID_PATTERN, description="The new pack's name.")],
        new_version: Annotated[
            str, Field(pattern=ID_PATTERN, description="The new pack's version.")
        ] = "1",
    ) -> Any:
        """Clone a pack version under a new name; the derivation is recorded."""
        return await gated_call(
            ctx.gate,
            READ,
            lambda: client.post(
                f"/packs/{pack_id}/versions/{version}/clone",
                json_body={"into": into, "version": new_version},
            ),
        )

    @mcp.tool(
        description=(
            "Delete a pack version. Refused while a run cites it; deletes no part."
            + WAITS
            + " Action delete."
        )
    )
    async def forge_delete_pack(pack_id: PackId, version: Version) -> Any:
        """Delete a pack version."""
        return await gated_call(
            ctx.gate, READ, lambda: client.delete(f"/packs/{pack_id}/versions/{version}")
        )
