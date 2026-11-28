"""A generated ``miforge.wsp-bundle/v1`` linear-probe bundle (feature 018, FTASKS 9.4; moved here
from ``tests/support/wsp_fixtures.py``, which re-exports it, so the tests and the protocol test's
built-in bundle share ONE generator; FTDD §5.3).

``make_bundle`` writes ``heads.safetensors``, the practice token file and the five practice ``.npy``
files to a directory, with a manifest whose every hash is real.

**The parity scores are computed by an independent reference** (``reference_row_score``): plain
Python loops over ``math``, written from miWorkbench's definition, sharing no code with
``wsp.weight_ops.evaluator`` (``test_wsp_fences.py`` pins that this module never imports it). A
fixture whose expected values came from the implementation under test would agree with it by
construction (global CLAUDE.md: "fixtures whose fields agree by construction"). The import's
parity check (``bundles.check_files``) then compares the evaluator against this reference.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from wsp.weight_ops.hashing import file_sha256, jcs_sha256, tensor_sha256
from wsp.weight_ops.st_io import write_safetensors
from wsp.weight_ops.tensor import Tensor, from_array

RULES = ("mean", "max", "last", "softmax", "attention", "rolling_mean_max")
EPS = 1e-6

PRACTICE_FILES = {
    "practice_offsets": "practice_offsets.npy",
    "practice_positions": "practice_positions.npy",
    "practice_labels": "practice_labels.npy",
    "practice_validation_mask": "practice_validation_mask.npy",
    "practice_truncated_mask": "practice_truncated_mask.npy",
}


# --------------------------------------------------------------------------------------
# The independent reference
# --------------------------------------------------------------------------------------


def _standardise(z: list[float], mean: list[float] | None, std: list[float] | None) -> list[float]:
    out = []
    for i, v in enumerate(z):
        x = v - (mean[i] if mean is not None else 0.0)
        if std is not None:
            x = 0.0 if abs(std[i]) <= EPS else x / std[i]
        out.append(x)
    return out


def _dot(a: list[float], b: list[float]) -> float:
    return math.fsum(x * y for x, y in zip(a, b, strict=True))


def _softmax_weighted(logits: list[float], values: list[float]) -> float:
    top = max(logits)
    exps = [math.exp(v - top) for v in logits]
    total = math.fsum(exps)
    return math.fsum(e / total * v for e, v in zip(exps, values, strict=True))


def reference_row_score(
    rule: str,
    tokens: list[list[float]],
    weight: list[float],
    bias: float,
    mean: list[float] | None,
    std: list[float] | None,
    query: list[float] | None,
    tau: float,
    window: int,
) -> float:
    standard = [_standardise(z, mean, std) for z in tokens]
    scores = [_dot(s, weight) + bias for s in standard]
    if rule == "mean":
        return math.fsum(scores) / len(scores)
    if rule == "max":
        return max(scores)
    if rule == "last":
        return scores[-1]
    if rule == "softmax":
        return _softmax_weighted([s / tau for s in scores], scores)
    if rule == "attention":
        assert query is not None
        return _softmax_weighted([_dot(s, query) for s in standard], scores)
    if rule == "rolling_mean_max":
        if len(scores) <= window:
            return math.fsum(scores) / len(scores)
        return max(
            math.fsum(scores[i : i + window]) / window for i in range(len(scores) - window + 1)
        )
    raise ValueError(rule)


# --------------------------------------------------------------------------------------
# The bundle
# --------------------------------------------------------------------------------------


@dataclass
class FixtureBundle:
    root: Path
    manifest: dict[str, Any]
    tensors: dict[str, Tensor]
    activations: np.ndarray
    offsets: np.ndarray
    labels: np.ndarray
    paths_by_role: dict[str, Path] = field(default_factory=dict)

    def write_manifest(self) -> Path:
        path = self.root / "manifest.json"
        path.write_text(json.dumps(self.manifest, indent=2, sort_keys=True), encoding="utf-8")
        return path


def row_ids_sha256(view_id: str, positions: list[int]) -> str:
    return jcs_sha256(
        {"algorithm": "miworkbench.row-ids/1", "view_id": view_id, "positions": positions}
    )


def seal(manifest: dict[str, Any]) -> dict[str, Any]:
    """Recompute ``bundle_sha256`` (RFC 8785 over the manifest without it)."""
    body = {k: v for k, v in manifest.items() if k != "bundle_sha256"}
    manifest["bundle_sha256"] = jcs_sha256(body)
    return manifest


def make_bundle(
    root: Path,
    *,
    rule: str = "mean",
    d_model: int = 8,
    rows: int = 24,
    seed: int = 0,
    practice_dtype: str = "float32",
    weight_dtype: str = "float32",
    tau: float | None = None,
    window: int | None = None,
    parity_rows: list[int] | None = None,
    degenerate_channel: bool = True,
    bundle_id: str | None = None,
    basis_source: str = "operator_mark",
    description: str | None = None,
    heldout_sets: list[dict[str, Any]] | None = None,
    labels_from_scores: bool = False,
    target_fpr: float = 0.01,
    supplements: list[int] | None = None,
    supplement_positives: int = 0,
) -> FixtureBundle:
    """A small linear-probe bundle for ``rule``, with real hashes and reference parity scores.

    ``target_fpr`` (``operating_point.target_fpr``, default 0.01) lets a test pin a target the
    worker's fallback cannot fake (fix 2026-11-22).

    ``bundle_id``, ``basis_source`` and ``description`` default to the test fixture's own values,
    so every manifest the tests generated before 018 is byte-identical; the protocol test passes
    its own (``wsp.protocol_test``).

    ``labels_from_scores`` labels each practice row by the parent's own reference score (above
    the median is positive), so the parent's practice AUROC is 1.0 on any rows holding both
    classes: what FR-016.30's qualification tests need (a harmful edit is then provably worse).
    ``bundle_id`` and ``heldout_sets`` override the manifest's (the scorer's contract wants a
    ``wsb_`` + 12-hex id; evaluate's tests want all four held-out roles).

    ``supplements`` (revision 2026-11-25, FU-7a C2) adds one validation-only supplement per entry,
    of that many rows (each at least 8), drawn from a SHIFTED distribution so a test can tell its
    rows from practice's; the first ``supplement_positives`` rows of each are positive, the rest
    negative, and each one's fifth row is truncated. ``None`` (the default) writes no supplement
    file, so every earlier manifest is byte-identical."""
    rng = np.random.default_rng(seed)
    root.mkdir(parents=True, exist_ok=True)
    np_dtype = {"float16": np.float16, "float32": np.float32}[practice_dtype]
    # Ragged rows, some longer than the rolling window, so every rule sees its interesting case.
    lengths = rng.integers(3, 40, size=rows)
    offsets = np.concatenate([[0], np.cumsum(lengths)]).astype(np.int64)
    tokens = int(offsets[-1])
    # A non-unit scale per channel, so ignoring std changes every score (FR-016.2's control).
    scale = rng.uniform(0.5, 4.0, size=d_model)
    activations = (rng.normal(size=(tokens, d_model)) * scale + rng.normal(size=d_model)).astype(
        np_dtype
    )
    mean = activations.astype(np.float64).mean(axis=0)
    std = activations.astype(np.float64).std(axis=0)
    if degenerate_channel:
        std[-1] = 0.0  # a dead channel: zeroed, never divided by a floor
    weight = rng.normal(size=d_model).astype(
        np.float32 if weight_dtype == "float32" else np.float64
    )
    bias = np.array([rng.normal() * 0.1], dtype=np.float64)
    query = rng.normal(size=d_model).astype(np.float32) if rule == "attention" else None
    labels = (rng.random(rows) < 0.5).astype(np.int8)
    positions = np.arange(rows, dtype=np.int64) * 3 + 7
    validation = np.zeros(rows, dtype=bool)
    validation[: max(1, rows // 4)] = True
    truncated = np.zeros(rows, dtype=bool)

    tensors: dict[str, Tensor] = {
        "weight": from_array(weight),
        "bias": from_array(bias),
        "mean": from_array(mean.astype(np.float32)),
        "std": from_array(std.astype(np.float32)),
    }
    if query is not None:
        tensors["attention_query"] = from_array(query)

    heads = root / "heads.safetensors"
    write_safetensors(heads, tensors)
    token_file = root / f"tokens_layer12.{'f16' if practice_dtype == 'float16' else 'f32'}"
    token_file.write_bytes(
        np.ascontiguousarray(activations).astype(np.dtype(np_dtype).newbyteorder("<")).tobytes()
    )
    np.save(root / PRACTICE_FILES["practice_offsets"], offsets)
    np.save(root / PRACTICE_FILES["practice_positions"], positions)
    np.save(root / PRACTICE_FILES["practice_labels"], labels)
    np.save(root / PRACTICE_FILES["practice_validation_mask"], validation)
    np.save(root / PRACTICE_FILES["practice_truncated_mask"], truncated)

    tau_v = 1.0 if tau is None else tau
    window_v = 16 if window is None else window
    parity_rows = parity_rows if parity_rows is not None else list(range(8))
    # The reference reads the tensors back as the evaluator will (float32 mean/std), in pure Python.
    w_l = [float(v) for v in tensors["weight"].values64()]
    m_l = [float(v) for v in tensors["mean"].values64()]
    s_l = [float(v) for v in tensors["std"].values64()]
    q_l = [float(v) for v in tensors["attention_query"].values64()] if query is not None else None
    parity_scores = []
    for r in parity_rows:
        toks = [
            [float(x) for x in row]
            for row in activations[offsets[r] : offsets[r + 1]].astype(np.float64)
        ]
        parity_scores.append(
            float(
                np.float32(
                    reference_row_score(
                        rule, toks, w_l, float(bias[0]), m_l, s_l, q_l, tau_v, window_v
                    )
                )
            )
        )

    if labels_from_scores:
        every = [
            reference_row_score(
                rule,
                [[float(x) for x in row] for row in activations[offsets[r] : offsets[r + 1]]],
                w_l,
                float(bias[0]),
                m_l,
                s_l,
                q_l,
                tau_v,
                window_v,
            )
            for r in range(rows)
        ]
        labels = (np.asarray(every) > np.median(every)).astype(np.int8)
        np.save(root / PRACTICE_FILES["practice_labels"], labels)

    files: list[dict[str, Any]] = [
        {
            "name": heads.name,
            "role": "heads",
            "sha256": file_sha256(heads),
            "bytes": heads.stat().st_size,
        }
    ]
    files.append(
        {
            "name": token_file.name,
            "role": "practice_activations",
            "sha256": file_sha256(token_file),
            "bytes": token_file.stat().st_size,
        }
    )
    for role, name in PRACTICE_FILES.items():
        p = root / name
        files.append(
            {"name": name, "role": role, "sha256": file_sha256(p), "bytes": p.stat().st_size}
        )

    supplement_entries = (
        _write_supplements(
            root,
            files,
            rng,
            supplements,
            supplement_positives,
            d_model=d_model,
            np_dtype=np_dtype,
            practice_dtype=practice_dtype,
            scale=scale,
            score=lambda toks: float(
                np.float32(
                    reference_row_score(
                        rule, toks, w_l, float(bias[0]), m_l, s_l, q_l, tau_v, window_v
                    )
                )
            ),
        )
        if supplements
        else None
    )

    manifest: dict[str, Any] = {
        "kind": "miforge.wsp-bundle/v1",
        "bundle_kind": "linear_probe",
        "bundle_id": bundle_id or f"wsb_fixture{seed:04d}{rule.replace('_', '')[:8]}",
        "created_at": "2026-11-16T05:00:00Z",
        "identity": {
            "owner": "miworkbench",
            "artifact_kind": "probe",
            "artifact_id": f"probe-{rule}",
            "detector_digest": None,
            "model": "fixture/model",
            "layer": 12,
            "run_id": "ptr-fixture",
        },
        "files": files,
        "tensors": [
            {
                "name": name,
                "shape": list(t.shape),
                "dtype": t.dtype,
                "sha256": tensor_sha256(t.bits),
                "editable": name in ("weight", "bias", "attention_query"),
            }
            for name, t in tensors.items()
        ],
        "function": {
            "kind": "linear_probe/v1",
            "description": description or f"A {rule}-pooled linear probe over layer 12 (fixture).",
            "linear_probe": {
                "rule": rule,
                "params": {"tau": tau, "window": window},
                "layer": 12,
                "scope": "all",
                "skip_template_prefix": False,
            },
            "sae_fn": None,
            "parity": {"rows": parity_rows, "scores": parity_scores, "tolerance": 1e-5},
        },
        "practice": {
            "format": "token_activations",
            "dtype": practice_dtype,
            "rows": rows,
            "tokens": tokens,
            "d_model": d_model,
            "view_id": "view-train",
            "row_ids_sha256": row_ids_sha256("view-train", [int(p) for p in positions]),
            "label_counts": {"positive": int(labels.sum()), "negative": int(rows - labels.sum())},
            "validation_rows": int(validation.sum()),
            "truncated_rows": int(truncated.sum()),
        },
        "heldout_ref": {
            "owner": "miworkbench",
            "sets": heldout_sets
            or [
                {
                    "name": "eval",
                    "view_id": "view-eval",
                    "role": "eval",
                    "rows": 400,
                    "row_ids_sha256": row_ids_sha256("view-eval", list(range(400))),
                },
                {
                    "name": "everyday",
                    "view_id": "view-everyday",
                    "role": "everyday_text",
                    "rows": 2000,
                    "row_ids_sha256": row_ids_sha256("view-everyday", list(range(2000))),
                },
            ],
        },
        "baselines_available": [
            "logistic_regression",
            "lda",
            "diff_of_means",
            "threshold_recut",
            "affine_recalibration",
        ],
        "operating_point": {
            "target_fpr": target_fpr,
            "threshold": 0.0,
            "threshold_source": "calibration",
        },
        "sensitivity": {
            "value": "not_sensitive",
            "basis": [{"source": basis_source, "ref": None, "value": "not_sensitive"}],
        },
        "sae": None,
        "bundle_sha256": "",
    }
    if supplement_entries is not None:
        manifest["practice"]["supplements"] = supplement_entries
    seal(manifest)
    paths = {e["role"]: root / e["name"] for e in files}
    return FixtureBundle(root, manifest, tensors, activations, offsets, labels, paths)


SUPPLEMENT_FILES = {
    "supplement_offsets": "supplement_offsets.npy",
    "supplement_positions": "supplement_positions.npy",
    "supplement_labels": "supplement_labels.npy",
    "supplement_truncated_mask": "supplement_truncated_mask.npy",
}


def _write_supplements(
    root: Path,
    files: list[dict[str, Any]],
    rng: np.random.Generator,
    sizes: list[int],
    positives: int,
    *,
    d_model: int,
    np_dtype: Any,
    practice_dtype: str,
    scale: np.ndarray,
    score: Any,
) -> list[dict[str, Any]]:
    """The five supplement files (appended to ``files``) and their manifest entries."""
    lengths = [rng.integers(3, 40, size=n) for n in sizes]
    all_lengths = np.concatenate(lengths)
    offsets = np.concatenate([[0], np.cumsum(all_lengths)]).astype(np.int64)
    tokens = int(offsets[-1])
    shift = rng.normal(size=d_model) + 2.0  # another generator: a shifted distribution
    acts = (rng.normal(size=(tokens, d_model)) * scale + shift).astype(np_dtype)
    labels = np.concatenate([(np.arange(n) < positives).astype(np.int8) for n in sizes])
    positions = np.concatenate([np.arange(n, dtype=np.int64) * 5 + 11 for n in sizes])
    # One truncated row per supplement (its fifth), so a reader that forgets the mask is seen.
    truncated = np.concatenate([(np.arange(n) == 4).astype(np.int8) for n in sizes])
    suffix = "f16" if practice_dtype == "float16" else "f32"
    token_file = root / f"supplement_activations.{suffix}"
    token_file.write_bytes(
        np.ascontiguousarray(acts).astype(np.dtype(np_dtype).newbyteorder("<")).tobytes()
    )
    for role, values in (
        ("supplement_offsets", offsets),
        ("supplement_positions", positions),
        ("supplement_labels", labels),
        ("supplement_truncated_mask", truncated),
    ):
        np.save(root / SUPPLEMENT_FILES[role], values)
    files.append(
        {
            "name": token_file.name,
            "role": "supplement_activations",
            "sha256": file_sha256(token_file),
            "bytes": token_file.stat().st_size,
        }
    )
    for role, name in SUPPLEMENT_FILES.items():
        p = root / name
        files.append(
            {"name": name, "role": role, "sha256": file_sha256(p), "bytes": p.stat().st_size}
        )
    entries: list[dict[str, Any]] = []
    first = 0
    for index, n in enumerate(sizes):
        view = f"view-supplement-{index}"
        rows = list(range(first, first + n))
        parity_rows = list(range(min(8, n)))
        parity_scores = [
            score(
                [
                    [float(x) for x in row]
                    for row in acts[offsets[first + r] : offsets[first + r + 1]].astype(np.float64)
                ]
            )
            for r in parity_rows
        ]
        entries.append(
            {
                "view_id": view,
                "first_row": first,
                "rows": n,
                "tokens": int(offsets[first + n] - offsets[first]),
                "label_counts": {
                    "positive": int(labels[rows].sum()),
                    "negative": int(n - labels[rows].sum()),
                },
                "truncated_rows": int(truncated[rows].sum()),
                "row_ids_sha256": row_ids_sha256(view, [int(p) for p in positions[rows]]),
                "validation_only": True,
                "parity": {"rows": parity_rows, "scores": parity_scores, "tolerance": 1e-5},
            }
        )
        first += n
    return entries
