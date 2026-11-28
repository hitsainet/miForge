"""WSP bundles: import (pull), list, get (feature 016; FR-016.1; FTDD §5.2; P29, R1B-X7).

``POST /wsp/bundles/import`` takes ``{owner, bundle_id, bundle_sha256}`` and answers 202 with the
import's row; the pull itself runs as ``miforge.wsp.import_bundle`` on the default queue. No request
carries a URL: miForge pulls from its own ``MIWORKBENCH_API_URL`` (``miworkbench_not_configured``
when unset). Importing an imported bundle again is a no-op; a failed import is retried.
"""

from __future__ import annotations

from typing import Annotated, Any

import sqlalchemy as sa
from fastapi import APIRouter, Body, Query, Request
from fastapi.responses import JSONResponse

from api.agent_origin import get_actor
from api.deps import SessionDep
from core.errors import refuse
from schemas.answers_wsp_bundles import (
    WspBundleGetBundleAnswer,
    WspBundleImportBundleAnswer,
    WspBundleListBundlesAnswer,
)
from wsp import bundles
from wsp.models import WspBundle
from wsp.settings import get_wsp_settings

router = APIRouter(prefix="/wsp", tags=["wsp"])
MAX_PAGE = 200


def document(row: WspBundle) -> dict[str, Any]:
    return {
        "bundle_sha256": row.bundle_sha256,
        "owner": row.owner,
        "bundle_id": row.bundle_id,
        "bundle_kind": row.bundle_kind,
        "artifact_kind": row.artifact_kind,
        "artifact_id": row.artifact_id,
        "model": row.model,
        "layer": row.layer,
        "sensitivity": row.sensitivity_value,
        "bytes": row.bytes,
        "import_status": row.import_status,
        "import_failure": row.import_failure,
        "received_at": row.received_at.isoformat(),
        "received_by": row.received_by,
    }


@router.post(
    "/bundles/import",
    status_code=202,
    summary="Import a bundle from miWorkbench (pull)",
    responses={202: {"model": WspBundleImportBundleAnswer}},
)
def import_bundle(
    request: Request, body: Annotated[dict[str, Any], Body()], session: SessionDep
) -> JSONResponse:
    actor = get_actor(request)
    row, queue_it = bundles.request_import(
        session, body, received_by=actor.agent or "operator", settings=get_wsp_settings()
    )
    out = document(row)
    session.commit()  # the task must see the row
    if queue_it:
        from wsp.tasks import import_bundle_task

        import_bundle_task.delay(row.bundle_sha256)
    return JSONResponse(status_code=202, content={**out, "queued": queue_it})


@router.get(
    "/bundles",
    summary="List imported bundles",
    responses={200: {"model": WspBundleListBundlesAnswer}},
)
def list_bundles(
    session: SessionDep, limit: Annotated[int, Query(ge=1, le=MAX_PAGE)] = 50
) -> dict[str, Any]:
    rows = session.scalars(
        sa.select(WspBundle).order_by(WspBundle.received_at.desc()).limit(limit)
    ).all()
    return {"bundles": [document(r) for r in rows]}


@router.get(
    "/bundles/{bundle_sha256}",
    summary="One bundle and its import status",
    responses={200: {"model": WspBundleGetBundleAnswer}},
)
def get_bundle(bundle_sha256: str, session: SessionDep) -> dict[str, Any]:
    row = session.get(WspBundle, bundle_sha256)
    if row is None:
        raise refuse("wsp_bundle_not_found", "No such bundle; import it first.")
    return {**document(row), "manifest": row.manifest if row.import_status == "imported" else None}
