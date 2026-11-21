"""The function evaluator: ``linear_probe/v1`` (FR-016.2; FTDD §6.2).

miForge re-implements the bundle's function so that it can **check** the bundle rather than trust
it:
a bundle is accepted only if this evaluator reproduces the owner's ``function.parity`` scores within
the
tolerance (the pilot's B1 discipline). The definition is miWorkbench's
(``backend/src/ml/probe_monitor_model.py``), restated in numpy:

- ``score_t = w · standardise(z_t) + b``, where ``standardise`` subtracts ``mean`` and divides by
  ``std``,
  and a **degenerate channel** (``|std| <= eps``, eps = 1e-6) is **zeroed**, never divided by a
  floor
  (miWorkbench's ``ProbeHead.standardise``: clamping amplified a dead channel 1000x);
- one of the six rules combines the per-token scores of a row: ``mean``, ``max``, ``last``,
  ``softmax``
  (``tau``), ``attention`` (weights ``softmax(q · standardise(z_t))``, values ``score_t``), and
  ``rolling_mean_max`` (``window``; a row no longer than the window is its mean).

The practice rows hold exactly the tokens the owner scored (scope and template prefix are applied at
capture, 044 FTDD §4.5), so a row's tokens are ``activations[offsets[r]:offsets[r+1]]``.

Arithmetic is float64 and the row score is rounded to float32, the owner's score dtype. ``sae/v1``
is
phase 4 and refused here until then (PF-19).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from core.errors import refuse
from wsp.weight_ops.tensor import Tensor

RULES = ("mean", "max", "last", "softmax", "attention", "rolling_mean_max")
EPS = 1e-6
DEFAULT_TAU = 1.0
DEFAULT_WINDOW = 16


@dataclass(frozen=True)
class LinearHead:
    weight: np.ndarray
    bias: float
    mean: np.ndarray | None
    std: np.ndarray | None
    attention_query: np.ndarray | None

    @classmethod
    def from_tensors(cls, tensors: dict[str, Tensor]) -> LinearHead:
        def vec(name: str) -> np.ndarray | None:
            t = tensors.get(name)
            return None if t is None else t.values64().reshape(-1)

        weight = vec("weight")
        bias = vec("bias")
        if weight is None or bias is None or bias.size != 1:
            raise refuse(
                "wsp_bundle_invalid",
                "A linear_probe bundle needs a 1-D 'weight' and a [1] 'bias' tensor.",
            )
        head = cls(weight, float(bias[0]), vec("mean"), vec("std"), vec("attention_query"))
        for name in ("mean", "std", "attention_query"):
            v = getattr(head, name)
            if v is not None and v.shape != weight.shape:
                raise refuse(
                    "wsp_bundle_invalid",
                    f"Tensor '{name}' has {v.size} entries but 'weight' has {weight.size}.",
                )
        return head

    def standardise(self, z: np.ndarray) -> np.ndarray:
        out = z
        if self.mean is not None:
            out = out - self.mean
        if self.std is not None:
            degenerate = np.abs(self.std) <= EPS
            divisor = np.where(degenerate, 1.0, self.std)
            out = np.where(degenerate, 0.0, out / divisor)
        return out


def combine(
    rule: str,
    scores: np.ndarray,
    *,
    attention_logits: np.ndarray | None = None,
    tau: float = DEFAULT_TAU,
    window: int = DEFAULT_WINDOW,
) -> float:
    """Combine one row's per-token scores (float64, length >= 1) into the row's score."""
    if scores.size == 0:
        raise refuse("wsp_bundle_invalid", "A practice row has no tokens, so it has no score.")
    if rule == "mean":
        return float(scores.mean())
    if rule == "max":
        return float(scores.max())
    if rule == "last":
        return float(scores[-1])
    if rule == "softmax":
        if tau <= 0:
            raise refuse("wsp_bundle_invalid", f"softmax tau must be > 0, got {tau}.")
        return float((_softmax(scores / tau) * scores).sum())
    if rule == "attention":
        if attention_logits is None:
            raise refuse(
                "wsp_bundle_invalid",
                "The attention rule needs an 'attention_query' tensor; using the scores instead "
                "would make it softmax at tau=1 under another name.",
            )
        return float((_softmax(attention_logits) * scores).sum())
    if rule == "rolling_mean_max":
        if window < 1:
            raise refuse("wsp_bundle_invalid", f"window must be >= 1, got {window}.")
        n = scores.size
        if n <= window:
            return float(scores.mean())
        cumulative = np.concatenate([[0.0], np.cumsum(scores)])
        return float(((cumulative[window:] - cumulative[:-window]) / window).max())
    raise refuse("wsp_bundle_invalid", f"Unknown rule {rule!r}; known: {', '.join(RULES)}.")


def _softmax(logits: np.ndarray) -> np.ndarray:
    shifted = logits - logits.max()
    weights: np.ndarray = np.exp(shifted)
    out: np.ndarray = weights / weights.sum()
    return out


def score_rows(
    function: dict[str, Any],
    tensors: dict[str, Tensor],
    activations: np.ndarray,
    offsets: np.ndarray,
    rows: list[int] | np.ndarray,
) -> np.ndarray:
    """Scores for ``rows`` of the practice set, as float32."""
    if function.get("kind") != "linear_probe/v1":
        raise refuse(
            "wsp_phase_not_enabled",
            f"Function kind {function.get('kind')!r} is not evaluated in this phase; "
            "sae/v1 opens with WSP_SAE_ENABLED in phase 4.",
        )
    spec = function["linear_probe"]
    rule = spec["rule"]
    params = spec.get("params") or {}
    tau = DEFAULT_TAU if params.get("tau") is None else float(params["tau"])
    window = DEFAULT_WINDOW if params.get("window") is None else int(params["window"])
    head = LinearHead.from_tensors(tensors)
    if activations.shape[-1] != head.weight.size:
        raise refuse(
            "wsp_bundle_invalid",
            f"Practice activations are d_model={activations.shape[-1]} but the head is "
            f"d_model={head.weight.size}.",
        )
    n_rows = offsets.size - 1
    out = np.empty(len(rows), dtype=np.float32)
    for i, r in enumerate(rows):
        r = int(r)
        if not 0 <= r < n_rows:
            raise refuse(
                "wsp_bundle_invalid", f"Parity row {r} is outside the {n_rows} practice rows."
            )
        z = np.asarray(activations[int(offsets[r]) : int(offsets[r + 1])], dtype=np.float64)
        standard = head.standardise(z)
        token_scores = standard @ head.weight + head.bias
        logits = None
        if rule == "attention" and head.attention_query is not None:
            logits = standard @ head.attention_query
        out[i] = np.float32(
            combine(rule, token_scores, attention_logits=logits, tau=tau, window=window)
        )
    return out


@dataclass(frozen=True)
class ParityReport:
    ok: bool
    worst_row: int
    worst_abs_diff: float
    tolerance: float
    scores: list[float]


def parity(
    function: dict[str, Any],
    tensors: dict[str, Tensor],
    activations: np.ndarray,
    offsets: np.ndarray,
) -> ParityReport:
    """Reproduce ``function.parity``; report the worst row."""
    spec = function["parity"]
    rows = list(spec["rows"])
    expected = np.asarray(spec["scores"], dtype=np.float64)
    if expected.size != len(rows) or not rows:
        raise refuse(
            "wsp_bundle_invalid", "function.parity has rows and scores of different lengths."
        )
    got = score_rows(function, tensors, activations, offsets, rows).astype(np.float64)
    diff = np.abs(got - expected)
    diff = np.where(np.isfinite(diff), diff, np.inf)
    worst = int(np.argmax(diff))
    tolerance = float(spec["tolerance"])
    return ParityReport(
        ok=bool(diff.max() <= tolerance),
        worst_row=int(rows[worst]),
        worst_abs_diff=float(diff[worst]),
        tolerance=tolerance,
        scores=[float(s) for s in got],
    )


def require_parity(report: ParityReport) -> None:
    if not report.ok:
        raise refuse(
            "wsp_bundle_parity_failed",
            f"miForge's evaluator does not reproduce the bundle's parity scores: practice row "
            f"{report.worst_row} differs by {report.worst_abs_diff:.3g} (tolerance "
            f"{report.tolerance:.3g}). The bundle's function, tensors or practice files disagree.",
            worst_row=report.worst_row,
            worst_abs_diff=report.worst_abs_diff if np.isfinite(report.worst_abs_diff) else None,
        )
