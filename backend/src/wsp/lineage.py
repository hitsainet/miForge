"""A candidate's tensors and its lineage (feature 016; FR-016.33, FR-016.35; FTDD §4).

**The bytes are the truth** (ADR-004): a candidate's tensors are never stored, they are REBUILT by
applying its stored patch to its parent's tensors, and ``apply_patch`` re-checks every base hash,
the archive's digest, every rebuilt tensor's ``edited_sha256`` and every changed-entry count. A
parent that is itself a candidate is rebuilt the same way, down to the bundle.

``bundle_zeroed`` (FR-016.31): a ``create`` run without ``keep_weights`` zeroes the editable
tensors before the editor writes. Its patch is still made against the bundle's own bits (the
worker diffs the live store against the bundle), so its tensors rebuild from the bundle; the
LINEAGE names the zeroed tensors as the parent, ``ref`` being :func:`zeroed_sha256`.
"""

from __future__ import annotations

import tempfile
from pathlib import Path
from typing import Any

import numpy as np
from sqlalchemy.orm import Session

from core.errors import refuse
from wsp.models import WspBundle, WspCandidate, WspRun
from wsp.settings import WSPSettings
from wsp.weight_ops.hashing import jcs_sha256
from wsp.weight_ops.tensor import Tensor, from_values

#: The stored names of a candidate's patch in its run directory (``runs.record_candidate``).
STORED_PATCH_JSON = "patch_patch.json"
STORED_PATCH_ARRAYS = "patch_edit.npz"
#: How deep a chain may go before the walk refuses (a cycle in stored rows would never end).
MAX_DEPTH = 64


def bundle_heads(settings: WSPSettings, bundle: WspBundle) -> dict[str, Tensor]:
    """The bundle's stored ``heads`` tensors (``WSP_BUNDLE_ROOT/bundles/<sha>/``)."""
    from wsp.bundles import store_dir
    from wsp.weight_ops.st_io import read_safetensors

    name = next(f["name"] for f in bundle.manifest["files"] if f["role"] == "heads")
    return read_safetensors(store_dir(settings, bundle.bundle_sha256) / name)


def zeroed(manifest: dict[str, Any], tensors: dict[str, Tensor]) -> dict[str, Tensor]:
    """The bundle's tensors with every EDITABLE tensor zeroed, each in its own dtype."""
    editable = {t["name"] for t in manifest.get("tensors", []) if t.get("editable") is True}
    return {
        name: (from_values(t.dtype, np.zeros(t.shape)) if name in editable else t)
        for name, t in tensors.items()
    }


def zeroed_sha256(manifest: dict[str, Any], tensors: dict[str, Tensor]) -> str:
    """``bundle_zeroed``'s ``ref``: RFC 8785 sha256 of ``{name: tensor sha256}`` of the zeroed
    tensors (every tensor, so the frozen ones are named too)."""
    return jcs_sha256({name: t.sha256 for name, t in sorted(zeroed(manifest, tensors).items())})


def starts_zeroed(request: dict[str, Any]) -> bool:
    """A ``create`` SESSION without ``keep_weights`` edits from zeroed tensors (the adapter zeroes
    them); a program (and its rerun) gets the bundle's own tensors as ``base``."""
    return (
        request.get("kind") == "create"
        and not request.get("keep_weights")
        and request.get("mode") == "session"
    )


def lineage_parent(
    request: dict[str, Any], bundle_manifest: dict[str, Any], tensors: dict[str, Tensor]
) -> dict[str, str]:
    """The lineage parent a run's candidate records (FR-016.31)."""
    parent = request.get("parent") or {"kind": "bundle", "ref": bundle_manifest["bundle_sha256"]}
    if starts_zeroed(request):
        return {"kind": "bundle_zeroed", "ref": zeroed_sha256(bundle_manifest, tensors)}
    return {"kind": str(parent["kind"]), "ref": str(parent["ref"])}


def patch_dir(settings: WSPSettings, run_id: str) -> Path:
    from wsp.runs import run_dir

    return run_dir(settings, run_id)


def stored_patch(settings: WSPSettings, candidate: WspCandidate) -> tuple[bytes, bytes]:
    """``(patch.json bytes, edit.npz bytes)`` as the intake stored them."""
    directory = patch_dir(settings, candidate.run_id)
    try:
        return (
            (directory / STORED_PATCH_JSON).read_bytes(),
            (directory / STORED_PATCH_ARRAYS).read_bytes(),
        )
    except OSError as exc:
        raise refuse(
            "wsp_result_invalid",
            "The candidate's stored patch cannot be read under WSP_BUNDLE_ROOT/runs; restore it "
            "from a backup of WSP_BUNDLE_ROOT.",
            candidate_id=candidate.candidate_id,
        ) from exc


def parent_candidate(session: Session, candidate: WspCandidate) -> WspCandidate:
    """The candidate a chained candidate's ``parent_ref`` names (its id, or its patch sha256)."""
    import sqlalchemy as sa

    ref = candidate.parent_ref
    row = session.get(WspCandidate, ref) or session.scalar(
        sa.select(WspCandidate).where(WspCandidate.patch_sha256 == ref)
    )
    if row is None:
        raise refuse(
            "wsp_candidate_not_found",
            f"Candidate {candidate.candidate_id} names a parent candidate {ref!r} that is not "
            "stored.",
            candidate_id=candidate.candidate_id,
        )
    return row


def rebuild(
    session: Session, settings: WSPSettings, candidate: WspCandidate, _depth: int = 0
) -> tuple[dict[str, Any], dict[str, Tensor], dict[str, Tensor]]:
    """``(patch manifest, base tensors, edited tensors)`` for ``candidate``, verified."""
    from wsp.weight_ops.patch import ARRAYS_FILE, PATCH_JSON, apply_patch

    if _depth > MAX_DEPTH:
        raise refuse("wsp_patch_not_chained", "The candidate's lineage does not end at a bundle.")
    if candidate.parent_kind == "candidate":
        _, _, base = rebuild(session, settings, parent_candidate(session, candidate), _depth + 1)
    elif candidate.parent_kind in ("bundle", "bundle_zeroed"):
        run = session.get(WspRun, candidate.run_id)
        bundle = None if run is None else session.get(WspBundle, run.bundle_sha256)
        if bundle is None or bundle.import_status != "imported":
            raise refuse("wsp_bundle_not_found", "The candidate's bundle is not stored.")
        base = bundle_heads(settings, bundle)
    else:
        raise refuse(
            "wsp_phase_not_enabled",
            f"A {candidate.parent_kind} parent is rebuilt from phase 5 (checkpoint edits).",
        )
    patch_json, arrays = stored_patch(settings, candidate)
    with tempfile.TemporaryDirectory(prefix="wsp-candidate-") as tmp:
        directory = Path(tmp)
        (directory / PATCH_JSON).write_bytes(patch_json)
        (directory / ARRAYS_FILE).write_bytes(arrays)
        manifest, edited = apply_patch(directory, base)
    if manifest["patch_sha256"] != candidate.patch_sha256:
        raise refuse(
            "wsp_result_invalid",
            "The stored patch is not the candidate's (its patch_sha256 differs); it is not used.",
            candidate_id=candidate.candidate_id,
        )
    return manifest, base, edited


def walk(session: Session, candidate: WspCandidate) -> list[WspCandidate]:
    """The candidate and its ancestors, nearest first, ending at the one whose parent is not a
    candidate (a bundle, or the bundle's zeroed tensors)."""
    chain = [candidate]
    while chain[-1].parent_kind == "candidate":
        if len(chain) > MAX_DEPTH:
            raise refuse(
                "wsp_patch_not_chained", "The candidate's lineage does not end at a bundle."
            )
        chain.append(parent_candidate(session, chain[-1]))
    return chain


def chain_shas(session: Session, candidate: WspCandidate) -> list[str]:
    """``Lineage.chain``: the patch sha256s from the bundle to this candidate."""
    return [c.patch_sha256 for c in reversed(walk(session, candidate))]
