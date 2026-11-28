"""``GET /api/v1/host/resources`` - what the shell's resource strip shows on every screen.

CPU percent, memory used of total, and each visible GPU **by UUID** with its utilisation,
memory used of total and temperature. A source that could not be read is listed in
``unavailable`` with a short reason and its reading is ``null`` - always a 200, because
"this process cannot see a GPU" is a true answer to the question the strip asks.

**Disk** (2026-11-26, the 019 P4 MCP gap): ``disk.volumes`` is the collector's ``statvfs`` of the
checkpoint and WSP roots this pod mounts; ``disk.wsp_workers`` is each WSP worker's ``/work``
filesystem as its last ``/ready`` reported it (``mf_wsp_workers``, written by the driver's pass,
so this route never calls a worker, ADR-007), beside ``disk.wsp_work_limit_bytes``, the ``/work``
emptyDir's size limit (``WSP_WORKER_WORK_BYTES``). When the database cannot be read the worker
list is empty and ``unavailable`` names it; the rest of the reading is still served.

Read-only; one ``nvidia-smi`` query, two psutil reads, one ``statvfs`` per volume and one query
per request. The collector is a dependency so a test can replace it without patching psutil or
the driver.
"""

from __future__ import annotations

from typing import Annotated, Any

from fastapi import APIRouter, Depends
from sqlalchemy.exc import SQLAlchemyError

from api.deps import SessionDep
from core.host_metrics import HostCollector, get_collector
from schemas.answers_host import HostHostResourcesAnswer
from wsp import pool
from wsp.settings import get_wsp_settings

router = APIRouter(prefix="/host", tags=["host"])

#: The ``unavailable`` code when the workers' disk figures cannot be read from the database.
WORKER_DISK_UNREADABLE = "wsp_worker_disk_unreadable"


@router.get(
    "/resources",
    summary="CPU, memory, per-GPU and disk readings for the resource strip",
    responses={200: {"model": HostHostResourcesAnswer}},
)
def host_resources(
    collector: Annotated[HostCollector, Depends(get_collector)],
    session: SessionDep,
) -> dict[str, Any]:
    """The collector's reading, with the WSP workers' disk figures added under ``disk``."""
    body = dict(collector.read())
    wsp = get_wsp_settings()
    disk = dict(body.get("disk") or {})
    disk.setdefault("volumes", [])
    disk["wsp_work_limit_bytes"] = int(wsp.wsp_worker_work_bytes)
    try:
        disk["wsp_workers"] = pool.worker_disk(session, wsp)
    except SQLAlchemyError:
        session.rollback()
        disk["wsp_workers"] = []
        body["unavailable"] = [
            *body.get("unavailable", []),
            {
                "metric": "disk",
                "code": WORKER_DISK_UNREADABLE,
                "message": "WSP worker disk figures not readable (database)",
            },
        ]
    body["disk"] = disk
    return body
