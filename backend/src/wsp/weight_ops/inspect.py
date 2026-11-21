"""Deterministic per-tensor statistics (FR-016.3, WR-4).

The response is canonical JSON (RFC 8785) with its sha256, so two calls on the same bundle return
byte-identical bodies. Quantiles use numpy's default (linear) method; singular values come from
``numpy.linalg.svd(compute_uv=False)`` in float64. For a probe it adds ``|w/std|`` (the effective
direction in raw activation space, the degenerate channels zeroed as the evaluator zeroes them) and
its
cosine with each 1-D tensor the request names.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from core.errors import refuse
from wsp.weight_ops.evaluator import EPS
from wsp.weight_ops.hashing import jcs_bytes, jcs_sha256
from wsp.weight_ops.tensor import Tensor

QUANTILES = (0.0, 0.01, 0.05, 0.25, 0.5, 0.75, 0.95, 0.99, 1.0)
TOP_SINGULAR = 32
#: Above this many entries the row and column norms are summarised by quantiles, not listed.
NORM_LIST_MAX = 4096


def _f(x: float) -> float | None:
    """A JSON-safe float: non-finite values are reported as null, never as NaN."""
    return float(x) if np.isfinite(x) else None


def _summary(values: np.ndarray) -> dict[str, Any]:
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return {
            "count": int(values.size),
            "finite": 0,
            "quantiles": None,
            "mean": None,
            "std": None,
        }
    qs = np.quantile(finite, QUANTILES)
    return {
        "count": int(values.size),
        "finite": int(finite.size),
        "quantiles": {f"{q:g}": _f(v) for q, v in zip(QUANTILES, qs, strict=True)},
        "mean": _f(finite.mean()),
        "std": _f(finite.std()),
    }


def _norms(values: np.ndarray) -> list[float | None] | dict[str, Any]:
    if values.size <= NORM_LIST_MAX:
        return [_f(v) for v in values]
    return _summary(values)


def tensor_stats(name: str, t: Tensor) -> dict[str, Any]:
    v = t.values64()
    out: dict[str, Any] = {
        "name": name,
        "shape": list(t.shape),
        "dtype": t.dtype,
        "sha256": t.sha256,
        "values": _summary(v.reshape(-1)),
        "l2_norm": _f(float(np.linalg.norm(v.reshape(-1)))) if np.isfinite(v).all() else None,
    }
    if v.ndim == 2 and np.isfinite(v).all():
        out["row_norms"] = _norms(np.linalg.norm(v, axis=1))
        out["column_norms"] = _norms(np.linalg.norm(v, axis=0))
        singular = np.linalg.svd(v, compute_uv=False)
        out["singular_values"] = [_f(s) for s in singular[:TOP_SINGULAR]]
    return out


def inspect_tensors(
    tensors: dict[str, Tensor],
    *,
    bundle_kind: str = "linear_probe",
    cosine_with: list[str] | None = None,
) -> dict[str, Any]:
    """The inspect report: one entry per tensor, sorted by name, and a probe block for a probe."""
    report: dict[str, Any] = {
        "tensors": [tensor_stats(name, tensors[name]) for name in sorted(tensors)],
    }
    if bundle_kind == "linear_probe" and "weight" in tensors:
        w = tensors["weight"].values64().reshape(-1)
        effective = w.copy()
        if "std" in tensors:
            std = tensors["std"].values64().reshape(-1)
            degenerate = np.abs(std) <= EPS
            effective = np.where(degenerate, 0.0, w / np.where(degenerate, 1.0, std))
        norm = float(np.linalg.norm(effective))
        cosines: dict[str, float | None] = {}
        for other in cosine_with or []:
            if other not in tensors:
                raise refuse("wsp_bundle_invalid", f"Tensor {other!r} is not in the bundle.")
            o = tensors[other].values64().reshape(-1)
            if o.shape != effective.shape:
                raise refuse(
                    "wsp_bundle_invalid",
                    f"Tensor {other!r} has {o.size} entries; the effective direction has "
                    f"{effective.size}.",
                )
            denominator = norm * float(np.linalg.norm(o))
            cosines[other] = _f(float(effective @ o) / denominator) if denominator > 0 else None
        report["probe"] = {"effective_norm": _f(norm), "cosine_with": cosines}
    return report


def inspect_response(tensors: dict[str, Tensor], **kwargs: Any) -> tuple[bytes, str]:
    """``(canonical body, its sha256)``. The bytes are what the route returns, unchanged."""
    report = inspect_tensors(tensors, **kwargs)
    return jcs_bytes(report), jcs_sha256(report)
