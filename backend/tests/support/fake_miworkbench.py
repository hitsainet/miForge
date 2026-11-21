"""A recording fake of miWorkbench's bundle routes (feature 016; FTASKS 7.4), as an httpx transport.

It serves ``GET /api/v1/wsp/bundles/{id}`` (the manifest) and ``GET …/files/{name}`` (the bytes,
with ``X-Content-SHA256``) for one fixture bundle, the shapes miWorkbench 044 publishes. Tests can
make a header disagree with the manifest, or the bytes disagree with the header.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from typing import Any

import httpx

from tests.support.wsp_fixtures import FixtureBundle

BASE = "http://miworkbench.test"


@dataclass
class FakeMiWorkbench:
    bundle: FixtureBundle
    manifest: dict[str, Any] | None = None
    #: name -> the X-Content-SHA256 to answer instead of the true digest.
    header_override: dict[str, str] = field(default_factory=dict)
    #: name -> the bytes to answer instead of the file's (the header still names the file's digest).
    body_override: dict[str, bytes] = field(default_factory=dict)
    requests: list[httpx.Request] = field(default_factory=list)

    def transport(self) -> httpx.MockTransport:
        return httpx.MockTransport(self)

    def __call__(self, request: httpx.Request) -> httpx.Response:
        self.requests.append(request)
        manifest = self.manifest if self.manifest is not None else self.bundle.manifest
        prefix = f"/api/v1/wsp/bundles/{manifest['bundle_id']}"
        path = request.url.path
        if path == prefix:
            return httpx.Response(200, content=json.dumps(manifest).encode())
        if path.startswith(prefix + "/files/"):
            name = path[len(prefix) + len("/files/") :]
            source = self.bundle.root / name
            if "/" in name or not source.is_file():
                return httpx.Response(404, json={"error": {"code": "not_found"}})
            data = source.read_bytes()
            digest = self.header_override.get(name, hashlib.sha256(data).hexdigest())
            body = self.body_override.get(name, data)
            return httpx.Response(200, content=body, headers={"X-Content-SHA256": digest})
        return httpx.Response(404, json={"error": {"code": "not_found"}})


# ----------------------------------------------------------------------------- the fake scorer
#
# FTASKS 13.3. ``POST /api/v1/wsp/score`` as miWorkbench 044 answers it, for a test's own score
# PLAN (per set: labels, and per head label its float32 scores and bar threshold). Its requests
# AND its answers are validated against the VENDORED contract (``docs/schemas/
# miworkbench-wsp-scoring-v1.json``), so it cannot drift from the shape miForge reads; its
# metrics come from an independent pure-Python reference (pairwise AUROC, counts by loops),
# never from ``wsp.evaluate``'s guard, so the guard is not checked against itself. Refusals use
# miWorkbench's REAL wire shape, ``{"detail": {code, message, ...}}`` (FastAPI's HTTPException),
# at the status the contract's refusal table gives.

SCORE_PATH = "/api/v1/wsp/score"


@dataclass
class SetPlan:
    """One held-out set as the scorer will answer it."""

    labels: list[int]
    #: head label -> float32 scores, one per row (every head the request names must be here,
    #: except on a calibration set, which carries bars only).
    scores: dict[str, Any]
    #: head label -> the bar's threshold (default 0.5); a head absent here gets the default.
    thresholds: dict[str, float | None] = field(default_factory=dict)
    truncated: int = 0


def _vendored() -> dict[str, Any]:
    from wsp.schemas import schema_dir

    document: dict[str, Any] = json.loads(
        (schema_dir() / "miworkbench-wsp-scoring-v1.json").read_text(encoding="utf-8")
    )
    return document


def contract_errors(part: str, document: Any) -> list[str]:
    from jsonschema import Draft202012Validator

    validator = Draft202012Validator(_vendored()[part])
    return sorted(e.message for e in validator.iter_errors(document))


def reference_auroc(scores: list[float], labels: list[int]) -> float | None:
    """Pairwise Mann-Whitney (a tie counts half), in plain Python: no shared code with miForge."""
    pos = [s for s, y in zip(scores, labels, strict=True) if y == 1]
    neg = [s for s, y in zip(scores, labels, strict=True) if y == 0]
    if not pos or not neg:
        return None
    neg_sorted = sorted(neg)
    import bisect

    wins = 0.0
    for p in pos:
        below = bisect.bisect_left(neg_sorted, p)
        ties = bisect.bisect_right(neg_sorted, p) - below
        wins += below + 0.5 * ties
    return wins / (len(pos) * len(neg))


def reference_metrics(
    scores: list[float], labels: list[int], threshold: float | None
) -> dict[str, Any]:
    import numpy as np

    f32 = [float(np.float32(s)) for s in scores]
    bar = None if threshold is None else float(np.float32(threshold))
    fires = [bar is not None and s >= bar for s in f32]
    positives = sum(1 for y in labels if y == 1)
    negatives = len(labels) - positives
    hit = sum(1 for f, y in zip(fires, labels, strict=True) if f and y == 1)
    return {
        "auroc": reference_auroc(f32, labels),
        "recall_at_bar": (float(hit) / positives) if positives else None,
        "false_alarms_at_bar": sum(1 for f, y in zip(fires, labels, strict=True) if f and y == 0),
        "negatives": negatives,
        "positives": positives,
    }


@dataclass
class FakeScorer:
    """A recording fake of miWorkbench's scorer, usable as an httpx transport (``transport()``)
    or as a REAL local HTTP server (``serve()``), so a route test sends a real request."""

    heldout_ref: dict[str, Any]
    plans: dict[str, SetPlan]
    #: (status-less code, extra detail): answer this refusal instead of scoring.
    refusal: tuple[str, dict[str, Any]] | None = None
    #: called with the answer before it is sent: a test breaks one field here.
    tamper: Any = None
    validate_answer: bool = True
    requests: list[dict[str, Any]] = field(default_factory=list)
    headers: list[dict[str, str]] = field(default_factory=list)
    invalid_requests: list[list[str]] = field(default_factory=list)
    calls: int = 0

    def transport(self) -> httpx.MockTransport:
        def handler(request: httpx.Request) -> httpx.Response:
            status, body = self.answer(request.url.path, dict(request.headers), request.content)
            return httpx.Response(status, json=body)

        return httpx.MockTransport(handler)

    def refusal_status(self, code: str) -> int:
        return int(_vendored()["refusals"][code]["status"])

    def answer(self, path: str, headers: dict[str, str], raw: bytes) -> tuple[int, Any]:
        self.calls += 1
        self.headers.append({k.lower(): v for k, v in headers.items()})
        if path != SCORE_PATH:
            return 404, {"detail": "Not Found"}
        body = json.loads(raw)
        self.requests.append(body)
        errors = contract_errors("request", body)
        if errors:
            self.invalid_requests.append(errors)
            return 422, {"detail": [{"msg": e} for e in errors]}
        if self.refusal is not None:
            code, extra = self.refusal
            return self.refusal_status(code), {"detail": {"code": code, "message": "m", **extra}}
        if body["heldout_ref"] != self.heldout_ref:
            return 422, {"detail": {"code": "heldout_ref_mismatch", "message": "not the bundle's"}}
        answer = self.score(body)
        if self.tamper is not None:
            self.tamper(answer)
        if self.validate_answer:
            problems = contract_errors("response", answer)
            assert not problems, f"the fake scorer's own answer breaks the contract: {problems}"
        return 200, answer

    def score(self, request: dict[str, Any]) -> dict[str, Any]:
        import secrets

        from wsp.weight_ops.hashing import jcs_sha256

        labels_sent = [h["label"] for h in request["heads"]]
        sets = []
        for entry in request["heldout_ref"]["sets"]:
            plan = self.plans.get(entry["name"])
            if plan is None:
                continue  # a set the scorer did not answer (completeness names it)
            bars = {
                label: {
                    "threshold": plan.thresholds.get(label, 0.5),
                    "source": "calibration",
                    "n_negatives": 1000,
                    "realised_fpr": 0.01,
                    "target_fpr": 0.01,
                }
                for label in labels_sent
            }
            one: dict[str, Any] = {
                "name": entry["name"],
                "view_id": entry["view_id"],
                "role": entry["role"],
                "row_ids_sha256": entry["row_ids_sha256"],
                "scored_row_ids_sha256": entry["row_ids_sha256"],
                "rows": len(plan.labels) + plan.truncated,
                "excluded": {
                    "truncated": plan.truncated,
                    "truncated_fired": dict.fromkeys(labels_sent, 0),
                },
                "bars": bars,
            }
            if entry["role"] != "calibration":
                scores = {label: [float(v) for v in plan.scores[label]] for label in labels_sent}
                one["labels"] = list(plan.labels)
                one["scores"] = scores
                one["metrics"] = {
                    label: reference_metrics(
                        scores[label], list(plan.labels), bars[label]["threshold"]
                    )
                    for label in labels_sent
                }
            sets.append(one)
        return {
            "kind": "miworkbench.wsp-score-result/v1",
            "bundle_sha256": request["bundle_sha256"],
            "heldout_ref_sha256": jcs_sha256(request["heldout_ref"]),
            "function": {
                "rule": "mean",
                "params": {},
                "layer": 12,
                "scope": "all",
                "skip_template_prefix": False,
            },
            "sets": sets,
            "scored_at": "2026-11-19T05:00:00Z",
            "call_id": f"wsc_{secrets.token_hex(6)}",
        }

    def serve(self) -> Any:
        """A context manager: this fake on a real ``127.0.0.1`` port; yields its base URL."""
        import contextlib
        import threading
        from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

        fake = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *_args: Any) -> None:
                return

            def do_POST(self) -> None:
                raw = self.rfile.read(int(self.headers.get("content-length") or 0))
                status, body = fake.answer(self.path, dict(self.headers.items()), raw)
                data = json.dumps(body).encode()
                self.send_response(status)
                self.send_header("content-type", "application/json")
                self.send_header("content-length", str(len(data)))
                self.end_headers()
                self.wfile.write(data)

        @contextlib.contextmanager
        def running() -> Any:
            server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
            thread = threading.Thread(target=server.serve_forever, daemon=True)
            thread.start()
            try:
                yield f"http://127.0.0.1:{server.server_address[1]}"
            finally:
                server.shutdown()
                server.server_close()

        return running()


# ----------------------------------------------------------------------------- known truth

#: The fixture's held-out sets for evaluate (a primary eval set, the two false-alarm sets of
#: rung 2, and a calibration set that carries bars only).
EVAL_ROWS, EVERYDAY_ROWS, NONSENSE_ROWS, CALIBRATION_ROWS = 400, 2000, 1200, 300
BASELINE_HEADS = (
    "baseline:logistic_regression",
    "baseline:lda",
    "baseline:diff_of_means",
    "baseline:threshold_recut",
    "baseline:affine_recalibration",
)


def evaluate_heldout_sets() -> list[dict[str, Any]]:
    from tests.support.wsp_fixtures import row_ids_sha256

    return [
        {
            "name": name,
            "view_id": f"view-{name}",
            "role": role,
            "rows": rows,
            "row_ids_sha256": row_ids_sha256(f"view-{name}", list(range(rows))),
        }
        for name, role, rows in (
            ("eval", "eval", EVAL_ROWS),
            ("everyday", "everyday_text", EVERYDAY_ROWS),
            ("nonsense", "nonsense_text", NONSENSE_ROWS),
            ("calibration", "calibration", CALIBRATION_ROWS),
        )
    ]


def known_truth(
    candidate_gap: float, *, seed: int = 7, recut_gap: float = 1.0, control_gap: float = 1.0
) -> dict[str, SetPlan]:
    """Score plans with a known answer. On ``eval`` (200 positives, 200 negatives) each head's
    scores are N(0, 1) plus ``gap`` on the positives: the parent and the control at 1.0, the three
    refits at 1.2, the two operating-point baselines the parent's scores (``recut_gap`` moves
    ``threshold_recut``), the candidate at ``candidate_gap``. On the two false-alarm sets every
    head shares ONE draw of negatives, so every false-alarm delta is exactly zero. Bars sit at
    2.33 (about 1% of N(0, 1))."""
    import numpy as np

    rng = np.random.default_rng(seed)
    labels = np.array([1] * (EVAL_ROWS // 2) + [0] * (EVAL_ROWS // 2), dtype=np.int64)

    def head(gap: float) -> Any:
        return (rng.normal(size=labels.size) + gap * labels).astype(np.float32)

    parent = head(1.0)
    eval_scores = {
        "parent": parent,
        "candidate": head(candidate_gap),
        "control": head(control_gap),
        "baseline:logistic_regression": head(1.2),
        "baseline:lda": head(1.2),
        "baseline:diff_of_means": head(1.2),
        "baseline:threshold_recut": parent if recut_gap == 1.0 else head(recut_gap),
        "baseline:affine_recalibration": parent,
    }
    labels_all = ["parent", "candidate", "control", *BASELINE_HEADS]
    thresholds = dict.fromkeys(labels_all, 2.33)
    plans = {"eval": SetPlan([int(y) for y in labels], eval_scores, dict(thresholds))}
    for name, rows in (("everyday", EVERYDAY_ROWS), ("nonsense", NONSENSE_ROWS)):
        shared = rng.normal(size=rows).astype(np.float32)
        plans[name] = SetPlan([0] * rows, dict.fromkeys(labels_all, shared), dict(thresholds))
    plans["calibration"] = SetPlan([], {}, dict(thresholds))
    return plans
