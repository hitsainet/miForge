"""``evaluate``: a candidate's held-out verdict (feature 016; FR-016.26, .29, .33a, .34; FTDD §3.4,
§6.3; FTID §3.5; P7, P23, P24, P29, P46, P53, P56, P64, P65, P70, P71).

miWorkbench scores the heads miForge proposes on held-out rows miForge never sees; miForge checks
the answer and decides. In order:

1. The candidate and its run; a ``request_sha256`` that is not the run's is refused
   (``wsp_request_sha256_mismatch``). A verdict already stored for ``(candidate, request_sha256)``
   from a scorer ANSWER is returned as it is (idempotent, PF-12): no call, no count.
2. The heads (P46): ``candidate`` is REBUILT here as ``apply(patch, parent)`` from the stored
   patch, after the run's change limits are checked again (``wsp_change_limit_exceeded``, nothing
   sent); ``control`` is FR-016.27's seeded permutation of the same patch; ``parent`` and one
   ``baseline:<kind>`` per ``baselines_available`` entry go by LABEL only. No route takes a
   caller's tensors.
3. ``MIWORKBENCH_API_URL`` unset: ``unmeasured`` (``miworkbench_not_configured``), no HTTP call.
4. The call cap (PF-46): a held-out set already used ``WSP_SCORER_MAX_CALLS_PER_HELDOUT`` times
   refuses ``wsp_heldout_call_cap_reached`` with nothing sent.
5. ONE call (``clients.miworkbench.score``). Refusals by the BODY's code (P64): ``scorer_busy``
   refuses ``wsp_scorer_busy`` with nothing stored or counted; the budget's 409 is ``unmeasured``
   (``heldout_set_head_budget_exhausted:<set>``); anything else is ``scorer_unreachable``,
   ``scorer_contract_invalid`` or ``scorer_refused:<code>``.
6. The contract checks (``scorer_contract_invalid``): ``heldout_ref_sha256`` is the RFC 8785
   digest of the ``heldout_ref`` SENT (R3B-13), the sets and their ``row_ids_sha256`` are the
   bundle's, every head is answered, each head's bar is the same in every set, and
   ``excluded.truncated_fired`` holds one count per head sent, each from 0 to the set's
   ``excluded.truncated`` (F-2: the verdict records every head's count).
7. The metric guard (P29): every non-calibration set's metrics recomputed from exactly the
   returned float32 scores, labels and bars; counts exact, AUROC within 1e-12, else
   ``scorer_metric_disagreement``.
8. ``verdict.decide`` (completeness, the paired bootstrap, the decision).
9. ONE savepoint: the use counts (only for a call that came back with scores, FR-016.29), the
   verdict row with its run-document snapshot and ``run_sha256`` (FR-016.33a). A verdict from no
   answer (steps 3 and 5) is stored too, so it is addressable, but a later call REPLACES it: an
   outage never freezes a candidate as ``unmeasured``.
"""

from __future__ import annotations

import base64
import datetime as dt
import hashlib
import secrets
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import httpx
import numpy as np
import sqlalchemy as sa
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from core.errors import refuse
from wsp import lineage, runs, verdict
from wsp.models import WspBundle, WspCandidate, WspHeldoutUse, WspRun, WspVerdict
from wsp.settings import WSPSettings, get_wsp_settings
from wsp.weight_ops.hashing import jcs_bytes, jcs_sha256
from wsp.weight_ops.tensor import Tensor

OWNER = "miworkbench"
CONTRACT = "miworkbench-wsp-scoring-v1"
REQUEST_KIND = "miworkbench.wsp-score-request/v1"
AUROC_TOLERANCE = 1e-12


def utcnow() -> dt.datetime:
    return dt.datetime.now(dt.UTC)


def new_verdict_id() -> str:
    return f"wspv_{secrets.token_hex(6)}"


# ----------------------------------------------------------------------------- the request


def head_tensors(tensors: dict[str, Tensor]) -> list[dict[str, Any]]:
    """A head's tensors as the contract carries them: little-endian bytes, base64, and the
    tensor's own sha256 (the [wl] rule miWorkbench re-checks)."""
    out = []
    for name in sorted(tensors):
        t = tensors[name]
        raw = np.ascontiguousarray(t.bits).astype(t.bits.dtype.newbyteorder("<")).tobytes()
        out.append(
            {
                "name": name,
                "dtype": t.dtype,
                "shape": list(t.shape),
                "data_b64": base64.b64encode(raw).decode("ascii"),
                "sha256": t.sha256,
            }
        )
    return out


def build_request(
    run: WspRun,
    candidate: WspCandidate,
    manifest: dict[str, Any],
    edited: dict[str, Tensor],
    control: dict[str, Tensor],
) -> dict[str, Any]:
    """The score request (FTDD §3.4): tensors only on ``candidate`` and ``control``."""
    limits = run.request.get("change_limits") or {}
    heads: list[dict[str, Any]] = [
        {"label": "parent"},
        {"label": "candidate", "tensors": head_tensors(edited)},
        {"label": "control", "tensors": head_tensors(control)},
    ]
    heads += [
        {"label": f"baseline:{kind}"}
        for kind in manifest.get("baselines_available") or []
        if kind in verdict.BASELINES
    ]
    return {
        "kind": REQUEST_KIND,
        "bundle_id": manifest["bundle_id"],
        "bundle_sha256": manifest["bundle_sha256"],
        "run_ref": {
            "miforge_run_id": run.run_id,
            "patch_sha256": candidate.patch_sha256,
            "max_changed_entries": limits.get("max_changed_entries"),
        },
        "heldout_ref": manifest["heldout_ref"],
        "heads": heads,
        "primary_sets": list(run.request.get("primary_sets") or []),
    }


@dataclass(frozen=True)
class Material:
    """What ``evaluate`` sends, rebuilt from the stored patch (P46)."""

    run: WspRun
    candidate: WspCandidate
    bundle: WspBundle
    patch: dict[str, Any]
    edited: dict[str, Tensor]
    control_seed: int
    request: dict[str, Any]
    #: FR-016.27 (R1B-A8; review 016-P2, R16P2-4): the candidate's figures and the cast control's
    #: actual ones, which the verdict records beside each other.
    control_figures: dict[str, dict[str, Any]]


def material(session: Session, settings: WSPSettings, candidate: WspCandidate) -> Material:
    """Rebuild the candidate from its stored patch, re-check the run's limits, build the control
    and the request. Refuses before anything is sent."""
    from wsp.weight_ops.control import random_control
    from wsp.weight_ops.limits import check_limits

    run = runs.get_run(session, candidate.run_id)
    bundle = session.get(WspBundle, run.bundle_sha256)
    if bundle is None or bundle.import_status != "imported":
        raise refuse("wsp_bundle_not_found", "The candidate's bundle is not stored.")
    patch, base, edited = lineage.rebuild(session, settings, candidate)
    runs.refuse_non_editable(bundle.manifest, patch)
    violations = check_limits(patch, run.request.get("change_limits") or {})
    if violations:
        raise refuse(
            "wsp_change_limit_exceeded",
            "The candidate's patch breaks its run's change limits, so it is not scored: "
            + "; ".join(v.detail for v in violations),
            candidate_id=candidate.candidate_id,
        )
    patched = [t["name"] for t in patch["tensors"]]
    control = random_control(base, edited, patched, patch["content_sha256"], 0)
    return Material(
        run=run,
        candidate=candidate,
        bundle=bundle,
        patch=patch,
        edited=edited,
        control_seed=control.seed,
        request=build_request(run, candidate, bundle.manifest, edited, {**base, **control.tensors}),
        control_figures={
            "candidate": _figures(control.candidate),
            "control_post_cast": _figures(control.post_cast),
        },
    )


def _figures(figures: Any) -> dict[str, Any]:
    return {"changed_entries": int(figures.changed_entries), "l2_delta": float(figures.l2_delta)}


# ----------------------------------------------------------------------------- the checks


def contract_problem(result: dict[str, Any], request: dict[str, Any]) -> str | None:
    """``None``, or what makes the answer not one this request can be judged on (P65)."""
    if result.get("bundle_sha256") != request["bundle_sha256"]:
        return "the answer is for another bundle"
    if result.get("heldout_ref_sha256") != jcs_sha256(request["heldout_ref"]):
        return "heldout_ref_sha256 is not the RFC 8785 digest of the heldout_ref sent"
    sent = {s["name"]: s for s in request["heldout_ref"]["sets"]}
    names = [s["name"] for s in result["sets"]]
    if len(set(names)) != len(names) or not set(names) <= set(sent):
        return "the answer names a set that was not sent, or one twice"
    labels = {h["label"] for h in request["heads"]}
    bars: dict[str, Any] = {}
    for one in result["sets"]:
        ref = sent[one["name"]]
        if one["row_ids_sha256"] != ref["row_ids_sha256"] or one["role"] != ref["role"]:
            return f"set {one['name']}: its row ids or role are not the bundle's"
        if set(one["bars"]) != labels:
            return f"set {one['name']}: bars are not one per head sent"
        for label, bar in one["bars"].items():
            if bars.setdefault(label, bar) != bar:
                return f"head {label}: its bar differs between sets"
        excluded = one["excluded"]
        if set(excluded["truncated_fired"]) != labels:
            return f"set {one['name']}: truncated_fired is not one count per head sent"
        if any(
            type(n) is not int or not 0 <= n <= excluded["truncated"]
            for n in excluded["truncated_fired"].values()
        ):
            return f"set {one['name']}: a truncated_fired count is outside 0..truncated"
        if one["role"] == "calibration":
            continue  # bars only (the contract's if/then makes scores, labels, metrics null)
        scores, metrics, ys = one.get("scores"), one.get("metrics"), one.get("labels")
        if scores is None or metrics is None or ys is None:
            return f"set {one['name']}: no scores, labels or metrics"
        if set(scores) != labels or set(metrics) != labels:
            return f"set {one['name']}: scores or metrics are not one per head sent"
        if any(len(v) != len(ys) for v in scores.values()):
            return f"set {one['name']}: a head has another number of scores than labels"
    return None


def metrics_disagree(result: dict[str, Any]) -> str | None:
    """The metric guard (P29, R1B-X3): ``None``, or the first set and head whose returned metrics
    are not what its own float32 scores, labels and bar give."""
    for one in result["sets"]:
        if one["role"] == "calibration":
            continue
        ys = np.asarray(one["labels"], dtype=np.int64)
        positives, negatives = int((ys == 1).sum()), int((ys == 0).sum())
        for label, values in one["scores"].items():
            scores = np.asarray(values, dtype=np.float32)
            threshold = one["bars"][label]["threshold"]
            fires = (
                np.zeros(scores.size, dtype=bool)
                if threshold is None
                else scores >= np.float32(threshold)
            )
            got = one["metrics"][label]
            false_alarms = int(fires[ys == 0].sum())
            recall = float(fires[ys == 1].sum()) / positives if positives else None
            expected_auroc = verdict.auroc(scores, ys)
            if (
                got["positives"] != positives
                or got["negatives"] != negatives
                or type(got["false_alarms_at_bar"]) is not int  # a COUNT, never a ratio
                or got["false_alarms_at_bar"] != false_alarms
                or got["recall_at_bar"] != recall
                or (got["auroc"] is None) != (expected_auroc is None)
                or (
                    expected_auroc is not None
                    and abs(float(got["auroc"]) - expected_auroc) > AUROC_TOLERANCE
                )
            ):
                return f"{one['name']}:{label}"
    return None


# ----------------------------------------------------------------------------- the document


def scores_sha256(values: Any) -> str:
    """P23: sha256 of the float32 little-endian scores in the scorer's row order."""
    return hashlib.sha256(np.asarray(values, dtype="<f4").tobytes()).hexdigest()


@dataclass
class Judged:
    """A verdict's document and the candidate's per-set scores to store beside it."""

    document: dict[str, Any]
    answered: bool
    candidate_scores: list[tuple[str, list[float]]] = field(default_factory=list)


def _base_document(
    value: str,
    reasons: list[str],
    *,
    request_sha256: str,
    control_seed: int,
    control_figures: dict[str, dict[str, Any]],
    settings: WSPSettings,
    call_id: str,
    metrics_agree: bool,
    now: dt.datetime,
) -> dict[str, Any]:
    return {
        "verdict_id": "",
        "value": value,
        "reasons": [r[:500] for r in reasons],
        "request_sha256": request_sha256,
        "statistic": "paired_bootstrap",
        "B": int(settings.wsp_bootstrap_b),
        "seed": bootstrap_seed(request_sha256),
        "ci": 0.95,
        "control_seed": control_seed,
        "control_figures": control_figures,
        "primary_metric": "auroc",
        "scorer": {
            "owner": OWNER,
            "contract": CONTRACT,
            "call_id": call_id,
            "metrics_agree": metrics_agree,
        },
        "guarded_comparisons": 0,
        "completeness": {"ok": False, "missing": []},
        "per_set": [],
        "evaluated_at": runs._iso(now),
    }


def bootstrap_seed(request_sha256: str) -> int:
    """The bootstrap's seed, from the run's request digest, so a re-evaluation reproduces it."""
    return int(request_sha256[:8], 16)


def no_answer(reason: str, m: Material, settings: WSPSettings, now: dt.datetime) -> Judged:
    document = _base_document(
        "unmeasured",
        [reason],
        request_sha256=m.run.request_sha256,
        control_seed=m.control_seed,
        control_figures=m.control_figures,
        settings=settings,
        call_id="",
        metrics_agree=False,
        now=now,
    )
    return Judged(document, answered=False)


def judge(result: dict[str, Any], m: Material, settings: WSPSettings, now: dt.datetime) -> Judged:
    """The verdict of one scorer answer (steps 6 to 8)."""

    def doc(value: str, reasons: list[str], agree: bool) -> dict[str, Any]:
        return _base_document(
            value,
            reasons,
            request_sha256=m.run.request_sha256,
            control_seed=m.control_seed,
            control_figures=m.control_figures,
            settings=settings,
            call_id=str(result["call_id"]),
            metrics_agree=agree,
            now=now,
        )

    problem = contract_problem(result, m.request)
    if problem is not None:
        document = doc("unmeasured", ["scorer_contract_invalid", problem], False)
        return Judged(document, answered=True)
    disagreement = metrics_disagree(result)
    if disagreement is not None:
        document = doc(
            "unmeasured", ["scorer_metric_disagreement", f"first at {disagreement}"], False
        )
        return Judged(document, answered=True)
    primary = list(m.request["primary_sets"])
    scored = [s for s in result["sets"] if s["role"] != "calibration"]
    sets = [
        verdict.HeldoutSet(
            name=s["name"],
            role=s["role"],
            labels=np.asarray(s["labels"], dtype=np.int64),
            heads={
                label: verdict.Head(
                    label=label,
                    scores=np.asarray(values, dtype=np.float32),
                    threshold=s["bars"][label]["threshold"],
                    metrics=s["metrics"][label],
                )
                for label, values in s["scores"].items()
            },
        )
        for s in scored
    ]
    outcome = verdict.decide(
        sets,
        primary,
        changed_entries=int(m.candidate.changed_entries),
        seed=bootstrap_seed(m.run.request_sha256),
        b=int(settings.wsp_bootstrap_b),
        min_per_class=int(settings.wsp_min_rows_per_class),
        min_false_alarm_rows=int(settings.wsp_min_false_alarm_rows),
    )
    document = doc(outcome.value, outcome.reasons, True)
    document["guarded_comparisons"] = outcome.guarded_comparisons
    document["completeness"] = outcome.completeness
    deltas = {p["name"]: p["deltas"] for p in outcome.per_set}
    document["per_set"] = [
        _set_verdict(s, s["name"] in primary, deltas.get(s["name"], []), outcome) for s in scored
    ]
    return Judged(
        document,
        answered=True,
        candidate_scores=[(s["name"], list(s["scores"]["candidate"])) for s in scored],
    )


def _set_value(name: str, outcome: verdict.Outcome, compared: bool) -> str:
    """One set's value. A set reads ``improved`` only when it was compared and nothing blocked
    it: review 016-P2 (R16P2-2) found a ``regressed`` verdict whose primary set read
    ``improved`` though the candidate had not beaten the control there, because a regressed
    outcome's reasons name only the regressions."""
    if outcome.value in ("unmeasured", "no_change"):
        return outcome.value
    prefix = f"{name}:"
    if outcome.value == "regressed" and any(r.startswith(prefix) for r in outcome.reasons):
        return "regressed"
    if any(r.startswith(prefix) for r in outcome.blocked) or not compared:
        return "not_significant"
    return "improved"


def _set_verdict(
    one: dict[str, Any], primary: bool, deltas: list[dict[str, Any]], outcome: verdict.Outcome
) -> dict[str, Any]:
    candidate = one["metrics"]["candidate"]
    negatives = int(candidate["negatives"])
    fired = one["excluded"]["truncated_fired"]
    heads = []
    for label in sorted(one["scores"]):
        metrics, bar = one["metrics"][label], one["bars"][label]
        false_alarms = metrics["false_alarms_at_bar"]
        heads.append(
            {
                "label": label,
                "auroc": metrics["auroc"],
                "recall_at_bar": metrics["recall_at_bar"],
                "false_alarms_at_bar": false_alarms,
                "false_alarm_rate": (
                    None if false_alarms is None or not negatives else false_alarms / negatives
                ),
                "bar": {
                    "threshold": bar["threshold"],
                    "source": bar["source"],
                    "target_fpr": bar["target_fpr"],
                    "realised_fpr": bar["realised_fpr"],
                },
                "fvu": None,
                "l0": None,
                "dead_fraction": None,
                "scores_sha256": scores_sha256(one["scores"][label]),
            }
        )
    return {
        "name": one["name"],
        "role": one["role"],
        "primary": primary,
        "rows": int(one["rows"]),
        "positives": int(candidate["positives"]),
        "negatives": negatives,
        "excluded_truncated": int(one["excluded"]["truncated"]),
        # Every head's count, keyed by its label (information only, G5; F-2, 2026-11-19: the
        # schema had one literal key `label`, so only the candidate's count was kept).
        "truncated_fired": {label: int(fired[label]) for label in sorted(fired)},
        "heldout_uses": 1,  # filled in from the stored count when the verdict is stored
        "heads": heads,
        "deltas": deltas,
        "value": _set_value(one["name"], outcome, bool(deltas)),
    }


# ----------------------------------------------------------------------------- storing


def stored_verdict(session: Session, candidate_id: str) -> WspVerdict | None:
    """The candidate's verdict (one per candidate: ``evaluate`` takes only its run's digest)."""
    return session.scalar(
        sa.select(WspVerdict)
        .where(WspVerdict.candidate_id == candidate_id)
        .order_by(WspVerdict.created_at.desc())
        .limit(1)
    )


def answered(row: WspVerdict) -> bool:
    return bool(((row.document or {}).get("scorer") or {}).get("call_id"))


def _use_keys(manifest: dict[str, Any]) -> list[tuple[str, str, str]]:
    ref = manifest["heldout_ref"]
    return [(ref["owner"], s["name"], s["row_ids_sha256"]) for s in ref["sets"]]


def check_call_cap(session: Session, manifest: dict[str, Any], settings: WSPSettings) -> None:
    """PF-46: past ``WSP_SCORER_MAX_CALLS_PER_HELDOUT`` uses of a held-out set, nothing is sent."""
    cap = int(settings.wsp_scorer_max_calls_per_heldout)
    for owner, name, digest in _use_keys(manifest):
        row = session.get(WspHeldoutUse, (owner, name, digest))
        if row is not None and row.heldout_uses >= cap:
            raise refuse(
                "wsp_heldout_call_cap_reached",
                f"Held-out set {name!r} has been scored {row.heldout_uses} times, the cap "
                f"(WSP_SCORER_MAX_CALLS_PER_HELDOUT = {cap}); nothing was sent. The way forward is "
                "a bundle over new held-out rows.",
                set=name,
                row_ids_sha256=digest,
                heldout_uses=row.heldout_uses,
            )


def count_uses(
    session: Session, manifest: dict[str, Any], candidate_id: str, now: dt.datetime
) -> dict[str, int]:
    """FR-016.29: one more use of every held-out set, INSIDE the verdict's transaction."""
    counts: dict[str, int] = {}
    for owner, name, digest in _use_keys(manifest):
        row = session.get(WspHeldoutUse, (owner, name, digest), with_for_update=True)
        if row is None:
            row = WspHeldoutUse(
                owner=owner,
                set_name=name,
                row_ids_sha256=digest,
                heldout_uses=0,
                last_used_at=now,
            )
            session.add(row)
        row.heldout_uses += 1
        row.last_used_at = now
        row.last_candidate_id = candidate_id
        counts[name] = row.heldout_uses
    session.flush()
    return counts


def document_of(
    session: Session,
    run: WspRun,
    *,
    verdict_row: WspVerdict | None = None,
    verdict_document: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """The live ``miforge.wsp-run/v1`` document: the candidate, its lineage and its verdict."""
    from wsp.models import WspCandidate as Candidate

    bundle = session.get(WspBundle, run.bundle_sha256)
    candidate = session.scalar(sa.select(Candidate).where(Candidate.run_id == run.run_id))
    row = verdict_row
    if row is None and candidate is not None and verdict_document is None:
        row = stored_verdict(session, candidate.candidate_id)
    return runs.run_document(
        run,
        None if bundle is None else bundle.manifest,
        reruns=runs.program_reruns(session, run.program_sha256),
        candidate=candidate,
        chain=[] if candidate is None else lineage.chain_shas(session, candidate),
        verdict=(
            verdict_document
            if verdict_document is not None
            else (None if row is None else dict(row.document))
        ),
        run_sha256=None if row is None else row.run_sha256,
    )


def snapshot_bytes(document: dict[str, Any]) -> bytes:
    """FR-016.33a: RFC 8785 of the document WITHOUT ``run_sha256`` and ``program_reruns``."""
    return jcs_bytes(
        {k: v for k, v in document.items() if k not in ("run_sha256", "program_reruns")}
    )


def scores_dir(settings: WSPSettings, run_id: str, verdict_id: str) -> Path:
    return runs.run_dir(settings, run_id) / "verdicts" / verdict_id


def store(
    session: Session,
    m: Material,
    judged: Judged,
    settings: WSPSettings,
    now: dt.datetime,
    replace: WspVerdict | None,
) -> dict[str, Any]:
    """Step 9, in one savepoint. A duplicate ``(candidate, request_sha256)`` (a concurrent call)
    rolls the whole savepoint back - counts included - and the stored verdict is returned."""
    from wsp.schemas import validate_run

    document = judged.document
    document["verdict_id"] = new_verdict_id()
    try:
        with session.begin_nested():
            if replace is not None:
                session.delete(replace)
                session.flush()
            if judged.answered:
                counts = count_uses(session, m.bundle.manifest, m.candidate.candidate_id, now)
                for entry in document["per_set"]:
                    entry["heldout_uses"] = counts[entry["name"]]
            full = document_of(session, m.run, verdict_document=document)
            raw = snapshot_bytes(full)
            run_sha256 = hashlib.sha256(raw).hexdigest()
            validate_run({**full, "run_sha256": run_sha256})
            session.add(
                WspVerdict(
                    verdict_id=document["verdict_id"],
                    candidate_id=m.candidate.candidate_id,
                    request_sha256=m.run.request_sha256,
                    value=document["value"],
                    document=document,
                    snapshot=raw,
                    run_sha256=run_sha256,
                    scorer_owner=OWNER,
                    created_at=now,
                )
            )
            session.flush()
    except IntegrityError:
        existing = stored_verdict(session, m.candidate.candidate_id)
        if existing is None:  # pragma: no cover - the unique key names this very row
            raise
        return dict(existing.document)
    _write_scores(settings, m.run.run_id, document["verdict_id"], judged.candidate_scores)
    return document


def _write_scores(
    settings: WSPSettings, run_id: str, verdict_id: str, sets: list[tuple[str, list[float]]]
) -> None:
    """The candidate head's per-set scores (FR-016.34), named by their position in ``per_set``."""
    if not sets:
        return
    directory = scores_dir(settings, run_id, verdict_id)
    directory.mkdir(parents=True, exist_ok=True)
    for index, (_name, values) in enumerate(sets):
        (directory / f"{index}.f32").write_bytes(np.asarray(values, dtype="<f4").tobytes())


# ----------------------------------------------------------------------------- the entry point


def evaluate(
    session: Session,
    candidate_id: str,
    request_sha256: Any,
    *,
    settings: WSPSettings | None = None,
    transport: httpx.BaseTransport | None = None,
    now: dt.datetime | None = None,
) -> dict[str, Any]:
    """``POST /wsp/candidates/{id}/evaluate``: the stored (or new) verdict document."""
    from clients.miworkbench import ScoreRefused, refusal_of, score, scorer_client

    settings = settings or get_wsp_settings()
    now = now or utcnow()
    candidate = get_candidate(session, candidate_id)
    run = runs.get_run(session, candidate.run_id)
    if request_sha256 != run.request_sha256:
        raise refuse(
            "wsp_request_sha256_mismatch",
            "request_sha256 is not this candidate's run's; read it from GET /wsp/runs/{run_id}.",
            candidate_id=candidate_id,
        )
    existing = stored_verdict(session, candidate_id)
    if existing is not None and answered(existing):
        return dict(existing.document)
    m = material(session, settings, candidate)
    if not settings.miworkbench_api_url:
        judged = no_answer("miworkbench_not_configured", m, settings, now)
        return store(session, m, judged, settings, now, existing)
    check_call_cap(session, m.bundle.manifest, settings)
    try:
        with scorer_client(settings.miworkbench_api_url, transport) as client:
            result = score(client, m.request)
    except ScoreRefused as refused:
        if refused.code == "scorer_busy":
            raise refuse(
                "wsp_scorer_busy",
                "miWorkbench's scorer is busy; nothing was stored or counted. Call evaluate "
                "again shortly.",
            ) from refused
        if refused.code == "heldout_set_head_budget_exhausted":
            body = refusal_of(refused.body) or {}
            reason = f"heldout_set_head_budget_exhausted:{body.get('set', '')}"
        elif refused.code in ("scorer_unreachable", "scorer_contract_invalid"):
            reason = refused.code
        else:
            reason = f"scorer_refused:{refused.code}"
        return store(session, m, no_answer(reason, m, settings, now), settings, now, existing)
    return store(session, m, judge(result, m, settings, now), settings, now, existing)


def get_candidate(session: Session, candidate_id: str) -> WspCandidate:
    row = session.get(WspCandidate, candidate_id)
    if row is None:
        raise refuse(
            "wsp_candidate_not_found",
            f"No WSP candidate with id {candidate_id!r}.",
            candidate_id=candidate_id,
        )
    return row
