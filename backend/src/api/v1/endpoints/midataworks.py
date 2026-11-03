"""``/api/v1/midataworks`` - the integration's status and what it offers  (FR-010.4, .19).

**One registry, two readers.** This route and ``GET /api/health`` both build their
miDataworks entry from :func:`clients.availability.integration_status`, which loops
:data:`clients.capabilities.CAPABILITIES`. Writing the list twice would let health report a
capability the guard does not gate, which is worse than no report at all.

**An unconfigured integration is a neutral statement, not a fault.** The payload says
``configured: false`` with the setting that configures it and ``reachable: null`` - *not*
``false``, because "not configured" and "configured and down" are different states with
different fixes (FR-010.6). A UI renders this in indigo, miDataworks' state colour, with no
red and no warning triangle: absent is normal (ADR-012, R-02.50).
"""

from __future__ import annotations

from typing import Annotated, Any

from fastapi import APIRouter, Query

from clients import availability
from clients.midataworks import MiDataworksClient
from core.errors import refuse
from services.parts.kind_map import MIFORGE_SET_KINDS, PART_KINDS

router = APIRouter(prefix="/midataworks", tags=["midataworks"])


@router.get("/status", summary="Whether miDataworks is configured, reachable, and what it enables")
def get_status(
    probe: Annotated[bool, Query(description="Also check reachability (short timeout)")] = False,
) -> dict[str, Any]:
    """The integration's state.

    ``probe`` is off by default and the reason is health's rule applied here: a probe makes
    this route as slow as the sibling is, and a status page that hangs because an optional
    service is down is a status page nobody trusts. With ``probe=false`` the answer is a
    configuration fact and is instant.
    """
    if not probe or not availability.is_configured():
        return availability.integration_status()

    # Never raises; `ping` reports. A probe that threw would let an unreachable optional
    # sibling fail the status route itself, which is the one thing it must not do.
    with MiDataworksClient(
        action="checking whether miDataworks is reachable", capability="part_import_http"
    ) as client:
        reachable = client.ping()
    return availability.integration_status(reachable=reachable)


@router.get("/available-parts", summary="The versions miDataworks offers, by set kind")
def available_parts(
    kind: Annotated[str | None, Query(description=f"one of {sorted(MIFORGE_SET_KINDS)}")] = None,
) -> dict[str, Any]:
    """List what miDataworks has exported, so an operator need not copy identifiers by hand.

    **Read-only, and never cached as truth** (FR-010.19). The authoritative record of a part
    miForge *has* is its own imported copy and the manifest beside it; a cached catalogue
    would be a second answer to "what parts exist", and the stale one is the one a UI shows.

    Raises:
        Refusal: ``midataworks_not_configured`` (409/503 by the app's suffix rule) when
            unset, naming the setting and what to do instead; the client's transport
            refusals when configured and failing.
    """
    if kind is not None and kind not in MIFORGE_SET_KINDS and kind not in PART_KINDS:
        raise refuse(
            "part_kind_unknown",
            f"{kind!r} is not a miForge set kind. The contract's set kinds are "
            f"{sorted(MIFORGE_SET_KINDS)}; selector and grader configurations are their "
            f"own target kinds rather than set kinds, which is why both lists exist.",
            given=kind,
            miforge_set_kinds=sorted(MIFORGE_SET_KINDS),
            part_kinds=sorted(PART_KINDS),
        )

    # The guard is in the client's constructor, so an unconfigured install refuses here
    # before any work and with the capability named.
    with MiDataworksClient(
        action="listing the versions miDataworks offers", capability="available_parts_listing"
    ) as client:
        exports = client.list_exports()

    offered = [
        {
            "export_id": record.get("id"),
            "target": record.get("target"),
            "version_id": record.get("version_id"),
            "status": record.get("status"),
            "miforge_set_kind": (record.get("params") or {}).get("miforge_set_kind"),
            "manifest_sha256": record.get("manifest_sha256"),
            "created_at": record.get("created_at"),
        }
        for record in exports
        # Only miForge's own export target. A `trl` or `reward_bundle` export is a real
        # artifact for another consumer and offering it here would invite an import that
        # the kind map refuses - a refusal is correct but late, and this listing exists
        # precisely so the operator does not have to discover that.
        if record.get("target") == "miforge_set"
    ]
    if kind is not None:
        offered = [row for row in offered if row["miforge_set_kind"] == kind]

    return {
        "items": offered,
        "total": len(offered),
        # Named so a reader knows this is a view of a sibling rather than of miForge's own
        # index - two lists that would otherwise look alike.
        "source": "miDataworks",
        "cached": False,
    }
