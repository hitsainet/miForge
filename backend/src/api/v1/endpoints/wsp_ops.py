"""WSP providers, run validation, weight operations and patches (FR-016.3-.5, .8, .22; FTDD §5.2).

``GET /wsp/providers`` lists the registered providers with their availability; ``POST
/wsp/runs/validate`` freezes a start body exactly as ``POST /wsp/runs`` will (``wsp.freeze``) and
writes nothing. The weight operations need no LLM (D5): ``POST /wsp/weight-ops/inspect`` answers
the canonical (RFC 8785) report of an imported bundle's tensors with its sha256 in
``X-Content-SHA256``; ``POST /wsp/patches/verify`` applies a ``miforge.wsp-patch/v1`` to the
bundle's tensors and reports what it rebuilds; ``POST /wsp/patches/stack`` composes chained patches
into one. Patches travel as ``{patch_json, edit_npz_base64}``; each is bounded before it is decoded.
"""

from __future__ import annotations

import base64
import binascii
import datetime as dt
import tempfile
from pathlib import Path
from typing import Annotated, Any

from fastapi import APIRouter, Body, Request, Response
from sqlalchemy.orm import Session

from api.agent_origin import AGENT_HEADER
from api.deps import SessionDep
from core.errors import refuse
from schemas.answers_wsp_ops import (
    WspOpsInspectWeightsAnswer,
    WspOpsListProvidersAnswer,
    WspOpsStackPatchesAnswer,
    WspOpsVerifyPatchAnswer,
)
from schemas.wsp_validate import WspRunValidation
from wsp import runs
from wsp.models import WspBundle
from wsp.providers import registry
from wsp.settings import get_wsp_settings
from wsp.weight_ops import inspect as inspect_ops
from wsp.weight_ops import patch as patch_ops
from wsp.weight_ops import stack as stack_ops
from wsp.weight_ops.st_io import read_safetensors
from wsp.weight_ops.tensor import Tensor

router = APIRouter(prefix="/wsp", tags=["wsp"])


def caller_of(request: Request) -> str:
    """The queue identity (R1B-M23): the agent header's value, or ``operator``."""
    return request.headers.get(AGENT_HEADER) or "operator"


@router.get("/providers", responses={200: {"model": WspOpsListProvidersAnswer}})
def list_providers() -> dict[str, Any]:
    return {"providers": registry.listing(get_wsp_settings())}


#: Documentation only (as feature 021's answers, MF9): the route keeps the serializer it landed
#: with (train F3, ``response_model=None``: ``jsonable_encoder``), and the model types
#: ``forge_wsp_validate_run``'s output schema.
@router.post("/runs/validate", response_model=None, responses={200: {"model": WspRunValidation}})
def validate_run(
    request: Request,
    body: Annotated[dict[str, Any], Body()],
    session: SessionDep,
) -> dict[str, Any]:
    prepared = runs.prepare_start(session, body, caller_of(request))
    frozen = prepared.frozen
    return {
        "request": frozen.request,
        "request_sha256": frozen.request_sha256,
        "approval_action": frozen.approval_action,
        "locality": frozen.locality,
        "sensitivity": {
            "value": frozen.sensitivity_value,
            "effective": frozen.sensitivity_effective,
        },
        "budgets": frozen.request["budgets"],
        "caller_ref": frozen.caller_ref,
    }


# --------------------------------------------------------------------------- weight operations

#: A patch document or archive larger than this is refused before it is decoded.
MAX_PATCH_JSON_BYTES = 4 << 20
MAX_PATCH_ARCHIVE_BYTES = 256 << 20
MAX_STACK = 32


def bundle_tensors(session: Session, bundle_sha256: Any) -> tuple[WspBundle, dict[str, Tensor]]:
    """An IMPORTED bundle's stored heads (``wsp_bundle_not_imported`` otherwise)."""
    if not isinstance(bundle_sha256, str):
        raise refuse("wsp_bundle_invalid", "bundle_sha256 is required.")
    row = session.get(WspBundle, bundle_sha256)
    if row is None:
        raise refuse("wsp_bundle_not_found", "No such bundle; import it first.")
    if row.import_status != "imported":
        raise refuse("wsp_bundle_not_imported", "The bundle's import has not completed.")
    heads = next(f["name"] for f in row.manifest["files"] if f["role"] == "heads")
    root = Path(get_wsp_settings().wsp_bundle_root) / "bundles" / row.bundle_sha256
    return row, read_safetensors(root / heads)


@router.post(
    "/weight-ops/inspect",
    summary="Per-tensor statistics of a bundle (no LLM)",
    responses={200: {"model": WspOpsInspectWeightsAnswer}},
)
def inspect_weights(body: Annotated[dict[str, Any], Body()], session: SessionDep) -> Response:
    row, tensors = bundle_tensors(session, body.get("bundle_sha256"))
    cosine_with = body.get("cosine_with") or []
    if not isinstance(cosine_with, list) or not all(isinstance(n, str) for n in cosine_with):
        raise refuse("validation_failed", "cosine_with is a list of tensor names.")
    data, digest = inspect_ops.inspect_response(
        tensors, bundle_kind=row.bundle_kind, cosine_with=cosine_with
    )
    return Response(
        content=data, media_type="application/json", headers={"X-Content-SHA256": digest}
    )


def _patch_dir(document: Any, into: Path) -> Path:
    """Write one ``{patch_json, edit_npz_base64}`` to a fresh directory, bounded first."""
    if not isinstance(document, dict) or set(document) != {"patch_json", "edit_npz_base64"}:
        raise refuse("wsp_patch_malformed", "A patch is exactly {patch_json, edit_npz_base64}.")
    text, encoded = document["patch_json"], document["edit_npz_base64"]
    if not isinstance(text, str) or len(text) > MAX_PATCH_JSON_BYTES:
        raise refuse("wsp_patch_malformed", "patch_json is a JSON text of at most 4 MiB.")
    if not isinstance(encoded, str) or len(encoded) > MAX_PATCH_ARCHIVE_BYTES * 4 // 3 + 4:
        raise refuse("wsp_patch_malformed", "edit_npz_base64 is base64 of at most 256 MiB.")
    try:
        data = base64.b64decode(encoded, validate=True)
    except (binascii.Error, ValueError) as exc:
        raise refuse("wsp_patch_malformed", "edit_npz_base64 is not base64.") from exc
    into.mkdir()
    (into / patch_ops.PATCH_JSON).write_text(text, encoding="utf-8")
    (into / patch_ops.ARRAYS_FILE).write_bytes(data)
    return into


def _summary(manifest: dict[str, Any], rebuilt: dict[str, Tensor]) -> dict[str, Any]:
    return {
        "patch_sha256": manifest["patch_sha256"],
        "content_sha256": manifest["content_sha256"],
        "changed_entries_total": manifest["changed_entries_total"],
        "tensors": [
            {
                "name": t["name"],
                "edited_sha256": rebuilt[t["name"]].sha256,
                "changed_entries": t["changed_entries"],
            }
            for t in manifest["tensors"]
        ],
    }


@router.post(
    "/patches/verify",
    summary="Apply a patch to its bundle and report (writes nothing)",
    responses={200: {"model": WspOpsVerifyPatchAnswer}},
)
def verify_patch(body: Annotated[dict[str, Any], Body()], session: SessionDep) -> dict[str, Any]:
    row, tensors = bundle_tensors(session, body.get("bundle_sha256"))
    with tempfile.TemporaryDirectory(prefix="wsp-verify-") as tmp:
        directory = _patch_dir(body.get("patch"), Path(tmp) / "p")
        manifest, rebuilt = patch_ops.apply_patch(directory, tensors)
    if manifest["bundle_base_sha256"] != body["bundle_sha256"]:
        raise refuse("wsp_patch_base_mismatch", "The patch names another bundle.")
    runs.refuse_non_editable(row.manifest, manifest)
    return {"verified": True, **_summary(manifest, rebuilt)}


@router.post(
    "/patches/stack",
    summary="Compose chained patches into one equal patch",
    responses={200: {"model": WspOpsStackPatchesAnswer}},
)
def stack_patches(body: Annotated[dict[str, Any], Body()], session: SessionDep) -> dict[str, Any]:
    row, tensors = bundle_tensors(session, body.get("bundle_sha256"))
    patches = body.get("patches")
    if not isinstance(patches, list) or not 0 < len(patches) <= MAX_STACK:
        raise refuse("wsp_patch_malformed", f"patches is a list of 1 to {MAX_STACK} patches.")
    with tempfile.TemporaryDirectory(prefix="wsp-stack-") as tmp:
        dirs = [_patch_dir(p, Path(tmp) / f"p{i}") for i, p in enumerate(patches)]
        built = stack_ops.stack(
            tensors,
            dirs,
            # The stack keeps the first patch's provenance (the run that started the chain).
            created_by=patch_ops.load_manifest(dirs[0])["created_by"],
            created_at=dt.datetime.now(dt.UTC).isoformat().replace("+00:00", "Z"),
        )
    if built is None:
        return {"no_change": True, "patch": None}
    runs.refuse_non_editable(row.manifest, built.manifest)
    return {
        "no_change": False,
        "patch": {
            "patch_json": patch_ops.manifest_text(built.manifest),
            "edit_npz_base64": base64.b64encode(built.arrays).decode(),
        },
        "patch_sha256": built.manifest["patch_sha256"],
        "content_sha256": built.manifest["content_sha256"],
    }
