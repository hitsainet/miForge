"""WSP candidates: list, get, lineage, tensors, patch, run document, evaluate, verdict, scores; and
the held-out use counts (feature 016 phase 2; FTDD §5.2; FR-016.26, .29, .33, .33a, .34, .35).

``evaluate`` is an ungated, idempotent write (PF-12): a candidate is scored once per run digest,
and a re-call returns the stored verdict. miForge never evaluates on its own (R1B-X8): the caller
reads ``candidate_id`` from ``GET /wsp/runs/{run_id}`` and calls ``evaluate`` itself.

The three binary routes (``tensors``, ``patch/{name}``, ``verdict/scores``) answer with an explicit
media type, ``X-Content-SHA256`` over exactly the bytes served, and ``nosniff`` (FTID §5). Every
served byte is checked against its stored digest first: a file changed under ``WSP_BUNDLE_ROOT``
is refused ``wsp_result_invalid``, never served.
"""

from __future__ import annotations

import hashlib
import json
from typing import Annotated, Any

import sqlalchemy as sa
from fastapi import APIRouter, Body, Query
from fastapi.responses import Response
from sqlalchemy.orm import Session

from api.deps import SessionDep
from core.errors import refuse
from schemas.answers_common import binary_answer, media_alternatives
from schemas.answers_documents import WspPatchDocument, WspRunDocumentOrSnapshot
from schemas.answers_wsp_candidates import (
    WspCandidateEvaluateCandidateAnswer,
    WspCandidateGetCandidateAnswer,
    WspCandidateGetLineageAnswer,
    WspCandidateGetVerdictAnswer,
    WspCandidateListCandidatesAnswer,
    WspHeldoutListHeldoutUsesAnswer,
)
from wsp import evaluate, lineage, runs, standing
from wsp.models import WspCandidate, WspHeldoutUse, WspRun, WspVerdict
from wsp.settings import get_wsp_settings

router = APIRouter(prefix="/wsp", tags=["wsp"])

MAX_PAGE = 200
PATCH_FILES = {
    "patch.json": (lineage.STORED_PATCH_JSON, "application/json"),
    "edit.npz": (lineage.STORED_PATCH_ARRAYS, "application/octet-stream"),
}


def _binary(data: bytes, media_type: str, digest: str | None = None) -> Response:
    return Response(
        content=data,
        media_type=media_type,
        headers={
            "X-Content-SHA256": digest or hashlib.sha256(data).hexdigest(),
            "X-Content-Type-Options": "nosniff",
        },
    )


def _verdict(row: WspVerdict | None) -> dict[str, Any] | None:
    """A verdict as the candidate routes show it: its id and value, and (FU-7b) its standing:
    ``improved`` only when a primary held-out set scored for the first time confirms it, else
    ``improved_unconfirmed`` (``wsp.standing``), with the sets that confirm it."""
    if row is None:
        return None
    return {
        "verdict_id": row.verdict_id,
        "value": row.value,
        **standing.reading(dict(row.document or {})),
    }


def _summary(row: WspCandidate, verdict_row: WspVerdict | None) -> dict[str, Any]:
    return {
        "candidate_id": row.candidate_id,
        "run_id": row.run_id,
        "parent": {"kind": row.parent_kind, "ref": row.parent_ref},
        "bundle_base_sha256": row.bundle_base_sha256,
        "patch_sha256": row.patch_sha256,
        "changed_entries": row.changed_entries,
        "tensors": list((row.tensors or {}).get("tensors", [])),
        "created_at": runs._iso(row.created_at),
        "verdict": _verdict(verdict_row),
    }


@router.get(
    "/candidates",
    summary="WSP candidates, newest first",
    responses={200: {"model": WspCandidateListCandidatesAnswer}},
)
def list_candidates(
    session: SessionDep,
    limit: Annotated[int, Query(ge=1, le=MAX_PAGE)] = 50,
) -> dict[str, Any]:
    rows = session.scalars(
        sa.select(WspCandidate)
        .join(WspRun, WspRun.run_id == WspCandidate.run_id)
        .where(WspRun.deleted_at.is_(None))
        .order_by(WspCandidate.created_at.desc(), WspCandidate.candidate_id)
        .limit(limit)
    ).all()
    return {
        "candidates": [_summary(r, evaluate.stored_verdict(session, r.candidate_id)) for r in rows]
    }


def _candidate(session: Session, candidate_id: str) -> WspCandidate:
    row = evaluate.get_candidate(session, candidate_id)
    runs.get_run(session, row.run_id)  # a deleted run's candidate is not served
    return row


@router.get(
    "/candidates/{candidate_id}",
    summary="One WSP candidate",
    responses={200: {"model": WspCandidateGetCandidateAnswer}},
)
def get_candidate(candidate_id: str, session: SessionDep) -> dict[str, Any]:
    row = _candidate(session, candidate_id)
    return _summary(row, evaluate.stored_verdict(session, candidate_id))


@router.get(
    "/candidates/{candidate_id}/lineage",
    summary="A candidate's parents, to the bundle",
    responses={200: {"model": WspCandidateGetLineageAnswer}},
)
def get_lineage(candidate_id: str, session: SessionDep) -> dict[str, Any]:
    """FR-016.35: each step's patch sha256, provider and version, transcript sha256 and verdict,
    from the bundle to this candidate; the first step's parent is the bundle (or its zeroed
    tensors)."""
    chain = lineage.walk(session, _candidate(session, candidate_id))
    steps = []
    for row in reversed(chain):
        run = session.get(WspRun, row.run_id)
        stored = evaluate.stored_verdict(session, row.candidate_id)
        steps.append(
            {
                "candidate_id": row.candidate_id,
                "run_id": row.run_id,
                "parent": {"kind": row.parent_kind, "ref": row.parent_ref},
                "patch_sha256": row.patch_sha256,
                "provider": None if run is None else run.provider_id,
                "provider_version": None if run is None else run.provider_version,
                "transcript_sha256": None if run is None else run.transcript_sha256,
                "verdict": _verdict(stored),
            }
        )
    first = session.get(WspRun, chain[-1].run_id)
    return {
        "candidate_id": candidate_id,
        "bundle_sha256": None if first is None else first.bundle_sha256,
        "steps": steps,
    }


@router.get(
    "/candidates/{candidate_id}/tensors",
    summary="A candidate's tensors (safetensors)",
    response_class=Response,
    responses=binary_answer(
        "application/octet-stream",
        "the candidate's edited tensors, rebuilt from the stored patch, as safetensors bytes",
    ),
)
def get_tensors(candidate_id: str, session: SessionDep) -> Response:
    """The edited tensors, REBUILT from the stored patch (each hashes to its ``edited_sha256``)."""
    from wsp.weight_ops.st_io import dump_safetensors

    row = _candidate(session, candidate_id)
    _, _, edited = lineage.rebuild(session, get_wsp_settings(), row)
    return _binary(dump_safetensors(edited), "application/octet-stream")


@router.get(
    "/candidates/{candidate_id}/patch/{name}",
    summary="A candidate's patch file (patch.json or edit.npz)",
    responses={
        200: {
            "model": WspPatchDocument,
            "description": "patch.json as JSON; edit.npz as bytes.",
            **media_alternatives(**{"application/octet-stream": "name = edit.npz"}),
        }
    },
)
def get_patch(candidate_id: str, name: str, session: SessionDep) -> Response:
    """miWorkbench's ``get_patch`` (R1B-X8): the stored file, checked against the candidate's
    ``patch_sha256`` (``patch.json``) or the manifest's ``arrays_sha256`` (``edit.npz``)."""
    from wsp.weight_ops.patch import patch_sha256

    if name not in PATCH_FILES:
        raise refuse("validation_failed", f"name is one of {sorted(PATCH_FILES)}.", name=name)
    row = _candidate(session, candidate_id)
    patch_json, arrays = lineage.stored_patch(get_wsp_settings(), row)
    try:
        manifest = json.loads(patch_json)
    except ValueError:
        manifest = None
    intact = (
        isinstance(manifest, dict)
        and patch_sha256(manifest) == row.patch_sha256
        and hashlib.sha256(arrays).hexdigest() == manifest.get("arrays_sha256")
    )
    if not intact:
        raise refuse(
            "wsp_result_invalid",
            "The stored patch no longer matches the candidate's patch_sha256: a file under "
            "WSP_BUNDLE_ROOT/runs was changed after the intake stored it, so it is not served. "
            "Restore it from a backup of WSP_BUNDLE_ROOT.",
            candidate_id=candidate_id,
        )
    data = patch_json if name == "patch.json" else arrays
    return _binary(data, PATCH_FILES[name][1])


@router.get(
    "/candidates/{candidate_id}/run-document",
    summary="The candidate's run as miforge.wsp-run/v1 (?verdict_id= for its verdict snapshot)",
    responses={200: {"model": WspRunDocumentOrSnapshot}},
)
def get_run_document(
    candidate_id: str,
    session: SessionDep,
    verdict_id: Annotated[str | None, Query()] = None,
) -> Any:
    """FR-016.33a: with ``verdict_id``, EXACTLY the bytes stored when that verdict was stored
    (RFC 8785, without ``run_sha256`` and ``program_reruns``), with ``X-Content-SHA256`` equal to
    its ``run_sha256``; without it, the live document."""
    row = _candidate(session, candidate_id)
    run = runs.get_run(session, row.run_id)
    if verdict_id is None:
        return evaluate.document_of(session, run)
    stored = session.get(WspVerdict, verdict_id)
    if stored is None or stored.candidate_id != candidate_id:
        raise refuse(
            "wsp_verdict_not_found",
            f"No verdict {verdict_id!r} of candidate {candidate_id!r}.",
            candidate_id=candidate_id,
            verdict_id=verdict_id,
        )
    snapshot = bytes(stored.snapshot)
    if hashlib.sha256(snapshot).hexdigest() != stored.run_sha256:
        raise refuse(
            "wsp_result_invalid",
            "The stored verdict snapshot no longer hashes to its run_sha256; it is not served.",
            verdict_id=verdict_id,
        )
    return _binary(snapshot, "application/json", stored.run_sha256)


@router.post(
    "/candidates/{candidate_id}/evaluate",
    summary="Score the candidate on the held-out sets and decide its verdict (idempotent)",
    responses={200: {"model": WspCandidateEvaluateCandidateAnswer}},
)
def evaluate_candidate(
    candidate_id: str,
    # SessionDep (scope="function"): the verdict COMMITS before the answer is sent, so a caller
    # that fetches the snapshot by the verdict_id it was just given finds it (FastAPI's default
    # scope ends the session after the response; the gateway's settle lost that race, 016-P2).
    session: SessionDep,
    body: Annotated[dict[str, Any], Body()],
) -> dict[str, Any]:
    """Body ``{request_sha256}`` only: no head, tensor or URL is ever taken from a caller (P46)."""
    if not isinstance(body, dict) or set(body) != {"request_sha256"}:
        raise refuse(
            "validation_failed",
            "The body is {request_sha256} and nothing else: miForge builds every head itself.",
            accepted=["request_sha256"],
        )
    return evaluate.evaluate(session, candidate_id, body["request_sha256"])


@router.get(
    "/candidates/{candidate_id}/verdict",
    summary="The candidate's stored verdict",
    responses={200: {"model": WspCandidateGetVerdictAnswer}},
)
def get_verdict(candidate_id: str, session: SessionDep) -> dict[str, Any]:
    _candidate(session, candidate_id)
    stored = evaluate.stored_verdict(session, candidate_id)
    if stored is None:
        raise refuse(
            "wsp_verdict_not_found",
            f"Candidate {candidate_id!r} has no verdict; call evaluate first.",
            candidate_id=candidate_id,
        )
    return dict(stored.document)


@router.get(
    "/candidates/{candidate_id}/verdict/scores",
    summary="The candidate head's float32 scores on one held-out set (?set=)",
    response_class=Response,
    responses=binary_answer(
        "application/octet-stream",
        "the candidate head's float32 little-endian scores on one held-out set, in the "
        "scorer's row order",
    ),
)
def get_verdict_scores(
    candidate_id: str,
    session: SessionDep,
    set_name: Annotated[str | None, Query(alias="set")] = None,
) -> Response:
    """FR-016.34: the candidate head's scores on one set, float32 little-endian in the scorer's
    row order; ``X-Content-SHA256`` is the verdict's ``scores_sha256`` for that set."""
    _candidate(session, candidate_id)
    stored = evaluate.stored_verdict(session, candidate_id)
    if stored is None:
        raise refuse(
            "wsp_verdict_not_found",
            f"Candidate {candidate_id!r} has no verdict.",
            candidate_id=candidate_id,
        )
    names = [s["name"] for s in stored.document.get("per_set", [])]
    if set_name not in names:
        raise refuse("validation_failed", f"set is one of {names}.", sets=names)
    index = names.index(set_name)
    entry = stored.document["per_set"][index]
    digest = next(h["scores_sha256"] for h in entry["heads"] if h["label"] == "candidate")
    path = evaluate.scores_dir(
        get_wsp_settings(), stored_run_id(session, candidate_id), stored.verdict_id
    )
    try:
        data = (path / f"{index}.f32").read_bytes()
    except OSError as exc:
        raise refuse("wsp_result_invalid", "The stored scores cannot be read.") from exc
    if hashlib.sha256(data).hexdigest() != digest:
        raise refuse(
            "wsp_result_invalid",
            "The stored scores no longer hash to the verdict's scores_sha256; they are not served.",
        )
    return _binary(data, "application/octet-stream", digest)


def stored_run_id(session: Session, candidate_id: str) -> str:
    return evaluate.get_candidate(session, candidate_id).run_id


@router.get(
    "/heldout-uses",
    summary="How many times each held-out set has been scored",
    responses={200: {"model": WspHeldoutListHeldoutUsesAnswer}},
)
def list_heldout_uses(
    session: SessionDep,
) -> dict[str, Any]:
    """FR-016.29: one row per (owner, set name, row_ids_sha256), counted inside each verdict's
    transaction, so a failed scorer call never counts."""
    rows = session.scalars(
        sa.select(WspHeldoutUse).order_by(WspHeldoutUse.owner, WspHeldoutUse.set_name)
    ).all()
    return {
        "heldout_uses": [
            {
                "owner": r.owner,
                "set_name": r.set_name,
                "row_ids_sha256": r.row_ids_sha256,
                "heldout_uses": r.heldout_uses,
                "last_used_at": runs._iso(r.last_used_at),
                "last_candidate_id": r.last_candidate_id,
            }
            for r in rows
        ]
    }
