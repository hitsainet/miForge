"""Bundle import: pull, check every hash, check parity, store (feature 016; FR-016.1, FR-016.2;
P29, P30, P54; FTASKS 7.5).

``request_import`` records the request (``importing``) and the caller queues
``miforge.wsp.import_bundle``; ``import_bundle`` is that task's body. miForge pulls from its OWN
``MIWORKBENCH_API_URL`` (no request carries a URL), and:

- validates the manifest against ``miforge-wsp-bundle-v1.json`` (every object closed) and recomputes
  ``bundle_sha256`` by RFC 8785 (the ``rfc8785`` package, never ``canonical.py``);
- streams each ``files[]`` entry, whose ``X-Content-SHA256`` must equal the manifest's digest AND
the
  digest of the bytes (``wsp_bundle_fetch_failed``, naming the file);
- refuses any tensor file that is not safetensors, any practice file that is not raw f16/f32 or a
  non-pickled ``.npy`` (``wsp_bundle_format_refused``);
- recomputes every tensor sha256 (little-endian, C-contiguous) and the practice ``row_ids_sha256``
  (``wsp_bundle_hash_mismatch``), and reproduces ``function.parity`` with miForge's own evaluator
  (``wsp_bundle_parity_failed``);
- checks ``practice.supplements`` when present (revision 2026-11-25, FU-7a C2;
  ``check_supplements``): the five ``supplement_*`` files against every entry's rows, tokens,
  labels and truncation, each entry's own row-id digest and its own parity, and that no supplement
  is the practice view or a held-out set;
- stores the files content-addressed under ``WSP_BUNDLE_ROOT/bundles/<bundle_sha256>/`` through
  ``core/atomic_dir.py``: nothing is stored unless every check passed.

The manifest lives in the database row only, never in the store directory: the driver pushes
every file of that directory to the worker (``wsp.tasks.bundle_files``).
"""

from __future__ import annotations

import datetime as dt
import logging
import os
import re
import shutil
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import httpx
import numpy as np
import sqlalchemy as sa
from sqlalchemy.orm import Session

from core.errors import Refusal, refuse
from wsp import schemas
from wsp.models import WspBundle
from wsp.settings import WSPSettings
from wsp.weight_ops.evaluator import parity, require_parity
from wsp.weight_ops.hashing import jcs_sha256, tensor_sha256
from wsp.weight_ops.st_io import SafetensorsError, read_safetensors

log = logging.getLogger(__name__)

HEX64 = re.compile(r"[0-9a-f]{64}\Z")
BUNDLE_ID = re.compile(r"wsb_[0-9a-z]{6,40}\Z")
#: Pickle-capable or framework-specific formats, refused by name before any byte is parsed.
REFUSED_SUFFIXES = (".pt", ".pth", ".bin", ".ckpt", ".pkl", ".pickle")
RAW_PRACTICE = {"float16": (".f16", np.float16), "float32": (".f32", np.float32)}


@dataclass(frozen=True)
class ImportRequest:
    owner: str
    bundle_id: str
    bundle_sha256: str


def parse_request(body: dict[str, Any]) -> ImportRequest:
    owner, bundle_id, sha = body.get("owner"), body.get("bundle_id"), body.get("bundle_sha256")
    if owner != "miworkbench":
        raise refuse("wsp_bundle_invalid", "owner must be miworkbench.")
    if not isinstance(bundle_id, str) or not BUNDLE_ID.match(bundle_id):
        raise refuse("wsp_bundle_invalid", "bundle_id must match ^wsb_[0-9a-z]{6,40}$.")
    if not isinstance(sha, str) or not HEX64.match(sha):
        raise refuse("wsp_bundle_invalid", "bundle_sha256 must be 64 lowercase hex characters.")
    extra = set(body) - {"owner", "bundle_id", "bundle_sha256"}
    if extra:
        raise refuse("wsp_bundle_invalid", "The import takes owner, bundle_id and bundle_sha256.")
    return ImportRequest(owner, bundle_id, sha)


def store_dir(settings: WSPSettings, bundle_sha256: str) -> Path:
    return Path(settings.wsp_bundle_root) / "bundles" / bundle_sha256


def request_import(
    session: Session, body: dict[str, Any], *, received_by: str, settings: WSPSettings
) -> tuple[WspBundle, bool]:
    """``(row, queue_it)``: a new or failed import is (re)queued; an imported one is a no-op."""
    if not settings.miworkbench_api_url:
        raise refuse(
            "miworkbench_not_configured",
            "Set MIWORKBENCH_API_URL so miForge can pull bundles from miWorkbench.",
        )
    req = parse_request(body)
    row = session.get(WspBundle, req.bundle_sha256)
    if row is not None:
        if row.bundle_id != req.bundle_id:
            raise refuse(
                "wsp_bundle_hash_mismatch", "This bundle_sha256 belongs to another bundle_id."
            )
        if row.import_status == "failed":
            row.import_status, row.import_failure = "importing", None
            row.received_at, row.received_by = dt.datetime.now(dt.UTC), received_by
            return row, True
        return row, False
    other = session.scalar(
        sa.select(WspBundle).where(
            WspBundle.owner == req.owner, WspBundle.bundle_id == req.bundle_id
        )
    )
    if other is not None:
        raise refuse(
            "wsp_bundle_hash_mismatch",
            "This bundle_id was imported with another bundle_sha256; a bundle never changes.",
        )
    # Placeholders until the manifest is checked: the row says `importing`, and a start on it is
    # refused `wsp_bundle_not_imported` (R1B-X7).
    row = WspBundle(
        bundle_sha256=req.bundle_sha256,
        owner=req.owner,
        bundle_id=req.bundle_id,
        bundle_kind="linear_probe",
        artifact_kind="probe",
        artifact_id="",
        detector_digest=None,
        model="",
        layer=0,
        sensitivity_value="unknown",
        manifest={},
        bytes=0,
        import_status="importing",
        received_at=dt.datetime.now(dt.UTC),
        received_by=received_by,
    )
    session.add(row)
    session.flush()
    return row, True


# ----------------------------------------------------------------------------------- the checks


def _check_manifest(manifest: dict[str, Any], req: ImportRequest, settings: WSPSettings) -> None:
    schemas.validate_bundle(manifest)
    if manifest["bundle_id"] != req.bundle_id or manifest["identity"]["owner"] != req.owner:
        raise refuse("wsp_bundle_invalid", "The manifest names another bundle or owner.")
    body = {k: v for k, v in manifest.items() if k != "bundle_sha256"}
    if (
        jcs_sha256(body) != manifest["bundle_sha256"]
        or manifest["bundle_sha256"] != req.bundle_sha256
    ):
        raise refuse(
            "wsp_bundle_hash_mismatch",
            "bundle_sha256 is not the RFC 8785 digest of the manifest, or not the one requested.",
        )
    if manifest["bundle_kind"] == "sae" and not settings.wsp_sae_enabled:
        raise refuse("wsp_phase_not_enabled", "SAE bundles open in phase 4 (WSP_SAE_ENABLED).")
    total = sum(int(f["bytes"]) for f in manifest["files"])
    if total > settings.wsp_bundle_max_bytes:
        raise refuse(
            "wsp_bundle_too_large",
            f"The bundle is {total} bytes, above WSP_BUNDLE_MAX_BYTES "
            f"({settings.wsp_bundle_max_bytes}).",
        )
    for entry in manifest["files"]:
        if entry["name"].lower().endswith(REFUSED_SUFFIXES):
            raise refuse(
                "wsp_bundle_format_refused",
                f"{entry['name']}: pickle-capable formats are refused; tensors are safetensors.",
                name=entry["name"],
            )


def _by_role(manifest: dict[str, Any], directory: Path) -> dict[str, Path]:
    return {e["role"]: directory / e["name"] for e in manifest["files"]}


def _load_npy(path: Path) -> np.ndarray:
    if path.suffix != ".npy":
        raise refuse(
            "wsp_bundle_format_refused", f"{path.name}: a practice array is .npy.", name=path.name
        )
    try:
        array = np.load(path, allow_pickle=False)
    except (ValueError, OSError, EOFError) as exc:  # an object array needs pickle: refused
        raise refuse(
            "wsp_bundle_format_refused",
            f"{path.name}: not a plain .npy (object arrays are refused).",
            name=path.name,
        ) from exc
    if not isinstance(array, np.ndarray):
        # An .npz stored under a .npy name loads as an NpzFile (review OR1-016-7).
        close = getattr(array, "close", None)
        if close is not None:
            close()
        raise refuse(
            "wsp_bundle_format_refused", f"{path.name}: not a single .npy array.", name=path.name
        )
    if array.dtype == object:
        raise refuse("wsp_bundle_format_refused", f"{path.name}: an object array.", name=path.name)
    return np.asarray(array)


def check_files(manifest: dict[str, Any], directory: Path) -> None:
    """Formats, tensor hashes, the row-ids digest and parity, on the downloaded files."""
    paths = _by_role(manifest, directory)
    heads = paths.get("heads")
    if heads is None or heads.suffix != ".safetensors":
        raise refuse("wsp_bundle_format_refused", "The heads file must be safetensors.")
    try:
        tensors = read_safetensors(heads)
    except (SafetensorsError, ValueError) as exc:
        raise refuse("wsp_bundle_format_refused", f"{heads.name}: not safetensors.") from exc
    declared = {t["name"]: t for t in manifest["tensors"]}
    if set(declared) != set(tensors):
        raise refuse(
            "wsp_bundle_hash_mismatch", "The heads file holds other tensors than the manifest."
        )
    for name, entry in declared.items():
        t = tensors[name]
        if list(t.shape) != list(entry["shape"]) or t.dtype != entry["dtype"]:
            raise refuse("wsp_bundle_hash_mismatch", f"Tensor {name}: shape or dtype differs.")
        if tensor_sha256(t.bits) != entry["sha256"]:
            raise refuse("wsp_bundle_hash_mismatch", f"Tensor {name}: sha256 differs.", name=name)
    practice = manifest["practice"]
    suffix, np_dtype = RAW_PRACTICE[practice["dtype"]]
    activations_path = paths.get("practice_activations")
    if activations_path is None or activations_path.suffix != suffix:
        raise refuse(
            "wsp_bundle_format_refused",
            f"The practice activations are raw {practice['dtype']} ({suffix}).",
        )
    expected = int(practice["tokens"]) * int(practice["d_model"]) * np.dtype(np_dtype).itemsize
    if activations_path.stat().st_size != expected:
        raise refuse("wsp_bundle_format_refused", "The practice activations have the wrong size.")
    arrays = {
        role: _load_npy(p)
        for role, p in paths.items()
        if role.startswith("practice_") and role != "practice_activations"
    }
    positions = arrays.get("practice_positions")
    offsets = arrays.get("practice_offsets")
    if positions is None or offsets is None:
        raise refuse("wsp_bundle_invalid", "The practice offsets and positions are required.")
    if positions.ndim != 1 or offsets.ndim != 1 or positions.dtype.kind not in "iu":
        raise refuse(
            "wsp_bundle_invalid", "The practice positions and offsets are 1-D integer arrays."
        )
    digest = jcs_sha256(
        {
            "algorithm": "miworkbench.row-ids/1",
            "view_id": practice["view_id"],
            "positions": [int(p) for p in positions.tolist()],
        }
    )
    if digest != practice["row_ids_sha256"]:
        raise refuse("wsp_bundle_hash_mismatch", "practice.row_ids_sha256 differs.")
    if (
        offsets.ndim != 1
        or offsets.size != int(practice["rows"]) + 1
        or int(offsets[0]) != 0
        or int(offsets[-1]) != int(practice["tokens"])
        or bool(np.any(np.diff(offsets) < 0))
    ):
        raise refuse("wsp_bundle_invalid", "The practice offsets do not describe the rows.")
    activations = np.memmap(
        activations_path,
        dtype=np.dtype(np_dtype).newbyteorder("<"),
        mode="r",
        shape=(int(practice["tokens"]), int(practice["d_model"])),
    )
    require_parity(parity(manifest["function"], tensors, np.asarray(activations), offsets))
    check_supplements(manifest, paths, tensors)


def check_supplements(manifest: dict[str, Any], paths: dict[str, Path], tensors: Any) -> None:
    """``practice.supplements`` (revision 2026-11-25, FU-7a C2): the files and counts
    (``load_supplements``), each supplement's row-id digest over its own positions, and its own
    parity with miForge's evaluator (rows counted from its ``first_row``). Nothing when the bundle
    has none."""
    from wsp.weight_ops.practice import load_supplements

    data = load_supplements(paths, manifest)
    if data is None:
        return
    activations = np.asarray(data.activations)
    for index, entry in enumerate(data.entries):
        rows = data.rows_of(index)
        digest = jcs_sha256(
            {
                "algorithm": "miworkbench.row-ids/1",
                "view_id": entry["view_id"],
                "positions": [int(p) for p in data.positions[rows].tolist()],
            }
        )
        if digest != entry["row_ids_sha256"]:
            raise refuse(
                "wsp_bundle_hash_mismatch",
                f"practice.supplements[{index}].row_ids_sha256 differs.",
                view_id=entry["view_id"],
            )
        first = int(entry["first_row"])
        window = data.offsets[first : first + int(entry["rows"]) + 1]
        if any(int(r) >= int(entry["rows"]) for r in entry["parity"]["rows"]):
            raise refuse(
                "wsp_bundle_invalid",
                f"practice.supplements[{index}].parity names a row past the supplement's rows.",
            )
        function = {**manifest["function"], "parity": entry["parity"]}
        report = parity(function, tensors, activations, window)
        if not report.ok:
            raise refuse(
                "wsp_bundle_parity_failed",
                f"miForge's evaluator does not reproduce supplement {entry['view_id']}'s parity "
                f"scores: its row {report.worst_row} differs by {report.worst_abs_diff:.3g} "
                f"(tolerance {report.tolerance:.3g}).",
                view_id=entry["view_id"],
                worst_row=report.worst_row,
            )


# ------------------------------------------------------------------------------------ the task


def import_bundle(
    session: Session,
    bundle_sha256: str,
    *,
    settings: WSPSettings,
    transport: httpx.BaseTransport | None = None,
) -> str:
    """Pull, check and store one bundle; the row ends ``imported`` or ``failed`` (with its code)."""
    from clients.miworkbench import FetchFailed, bundle_client, fetch_bundle_manifest

    row = session.get(WspBundle, bundle_sha256)
    if row is None or row.import_status != "importing":
        return "skipped"
    req = ImportRequest(row.owner, row.bundle_id, row.bundle_sha256)
    dest = store_dir(settings, bundle_sha256)
    try:
        if not settings.miworkbench_api_url:
            raise refuse("miworkbench_not_configured", "Set MIWORKBENCH_API_URL.")
        with bundle_client(settings.miworkbench_api_url, transport) as client:
            try:
                manifest = fetch_bundle_manifest(client, req.bundle_id)
            except FetchFailed as exc:
                raise refuse("wsp_bundle_fetch_failed", str(exc)) from exc
            _check_manifest(manifest, req, settings)
            if not dest.exists():
                _download_and_store(client, manifest, req, dest)
            else:  # a stored directory is content-addressed: re-check it instead of refetching
                check_files(manifest, dest)
    except Refusal as refusal:
        row.import_status = "failed"
        row.import_failure = {"code": refusal.code, "message": str(refusal.message)[:1000]}
        return f"failed:{refusal.code}"
    except Exception as exc:  # a hostile file must not leave the row importing
        # Review OR1-016-7: anything else a malformed file provokes ends the import ``failed``
        # instead of rolling back and leaving the row ``importing`` (a re-request is a no-op then).
        log.warning(
            "wsp bundle import raised", extra={"code": "wsp_bundle_invalid", "error": repr(exc)}
        )
        row.import_status = "failed"
        row.import_failure = {
            "code": "wsp_bundle_invalid",
            "message": f"The bundle could not be checked ({type(exc).__name__}).",
        }
        return "failed:wsp_bundle_invalid"
    ident = manifest["identity"]
    row.bundle_kind = manifest["bundle_kind"]
    row.artifact_kind = ident["artifact_kind"]
    row.artifact_id = ident["artifact_id"]
    row.detector_digest = ident["detector_digest"]
    row.model = ident["model"]
    row.layer = int(ident["layer"])
    row.sensitivity_value = manifest["sensitivity"]["value"]
    row.manifest = manifest
    row.bytes = sum(int(f["bytes"]) for f in manifest["files"])
    row.import_status, row.import_failure = "imported", None
    return "imported"


def _download_and_store(
    client: httpx.Client, manifest: dict[str, Any], req: ImportRequest, dest: Path
) -> None:
    from clients.miworkbench import FetchFailed, fetch_bundle_file
    from core.atomic_dir import atomic_dir

    staging = Path(tempfile.mkdtemp(prefix=".import-", dir=_parent(dest)))
    try:
        for entry in manifest["files"]:
            try:
                fetch_bundle_file(
                    client,
                    req.bundle_id,
                    entry["name"],
                    expected_sha256=entry["sha256"],
                    expected_bytes=int(entry["bytes"]),
                    dest=staging / entry["name"],
                )
            except FetchFailed as exc:
                raise refuse("wsp_bundle_fetch_failed", str(exc), name=exc.name) from exc
        check_files(manifest, staging)
        with atomic_dir(dest, latest_link=False) as tmp:
            for entry in manifest["files"]:
                os.replace(staging / entry["name"], tmp / entry["name"])
    finally:
        shutil.rmtree(staging, ignore_errors=True)


def _parent(dest: Path) -> Path:
    dest.parent.mkdir(parents=True, exist_ok=True)
    return dest.parent
