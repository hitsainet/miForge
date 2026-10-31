"""``GET /api/v1/host/resources`` - what the shell's resource strip shows on every screen.

CPU percent, memory used of total, and each visible GPU **by UUID** with its utilisation,
memory used of total and temperature. A source that could not be read is listed in
``unavailable`` with a short reason and its reading is ``null`` - always a 200, because
"this process cannot see a GPU" is a true answer to the question the strip asks.

Read-only; one ``nvidia-smi`` query and two psutil reads per request. The collector is a
dependency so a test can replace it without patching psutil or the driver.
"""

from __future__ import annotations

from typing import Annotated, Any

from fastapi import APIRouter, Depends

from core.host_metrics import HostCollector, get_collector

router = APIRouter(prefix="/host", tags=["host"])


@router.get("/resources", summary="CPU, memory and per-GPU readings for the resource strip")
def host_resources(
    collector: Annotated[HostCollector, Depends(get_collector)],
) -> dict[str, Any]:
    """The collector's reading, unchanged."""
    return collector.read()
