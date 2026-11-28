"""Whether a head can change the held-out verdict at all (fix 2026-11-22, D128 round 7).

The held-out verdict (``wsp.verdict.decide``) judges a candidate by AUROC and by recall and false
alarms at the candidate's OWN bar, which miWorkbench re-cuts for every head to the probe's target
false-alarm rate. Each of those figures depends only on how the head ranks rows. So a head whose
row scores are a positive affine map of the parent's, ``s' = k * s + c`` with ``k > 0``, gets the
parent's verdict exactly. In D128 round 7 all three candidates changed only the bias, and all
three verdicts equalled the parent's bit for bit.

For a ``linear_probe/v1`` head the token score is ``z . w' + offset`` after folding in the
standardisation (``wsp_worker.scoring.fold``). The pooled row score is then a positive affine map
of the parent's when the folded weight is ``k`` times the parent's (``k > 0``) and the attention
query, if any, is unchanged, for every rule but ``softmax``: there the softmax weights see
``k * s / tau``, so only ``k == 1`` (a bias shift) keeps the ranking.

Standard library and numpy only (the worker's fence): used by the observation tools, by both
providers' ``submit`` and by ``submit_program``.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

import numpy as np

from wsp_worker.scoring import HEAD_TENSORS, ScoringError, fold

#: Relative tolerance on the folded weight (a float32 vector times a constant, rounded to float32
#: again, is off by at most about 6e-8 of its norm).
RTOL = 1e-6
BIAS_SHIFT = "bias_shift"
POSITIVE_RESCALE = "positive_rescale"


@dataclass(frozen=True)
class Equivalence:
    #: ``bias_shift`` (k == 1) or ``positive_rescale`` (k > 0, k != 1).
    kind: str
    #: ``k``: the candidate's row scores are ``k * parent + shift``.
    scale: float
    shift: float


def _same(a: Any, b: Any) -> bool:
    if a is None or b is None:
        return a is None and b is None
    x, y = np.asarray(a, dtype=np.float64), np.asarray(b, dtype=np.float64)
    return x.shape == y.shape and bool(np.array_equal(x, y))


def _close(a: np.ndarray, b: np.ndarray) -> bool:
    scale = max(float(np.linalg.norm(a)), float(np.linalg.norm(b)), 1e-300)
    return float(np.linalg.norm(a - b)) <= RTOL * scale


def affine_to_parent(
    rule: str, parent: Mapping[str, Any], current: Mapping[str, Any]
) -> Equivalence | None:
    """The equivalence when ``current``'s row scores are a positive affine map of ``parent``'s
    under ``rule``; ``None`` when they are not, when the two heads are identical (no change), or
    when either is not a linear probe head."""
    if all(_same(parent.get(n), current.get(n)) for n in HEAD_TENSORS):
        return None
    try:
        before, after = fold(parent), fold(current)
    except ScoringError:
        return None
    if before.matrix.shape != after.matrix.shape:
        return None
    if before.has_query and not _close(after.matrix[:, 1], before.matrix[:, 1]):
        return None
    w0, w1 = before.matrix[:, 0], after.matrix[:, 0]
    norm0 = float(w0 @ w0)
    if norm0 == 0.0:
        if float(np.linalg.norm(w1)) != 0.0:
            return None
        k = 1.0
    else:
        k = float(w1 @ w0) / norm0
        if not k > 0.0 or not _close(w1, k * w0):
            return None
    bias_shift = abs(k - 1.0) <= RTOL
    if rule == "softmax" and not bias_shift:
        return None
    shift = float(after.offset[0]) - k * float(before.offset[0])
    return Equivalence(
        BIAS_SHIFT if bias_shift else POSITIVE_RESCALE, 1.0 if bias_shift else k, shift
    )


def for_bundle(
    manifest: Mapping[str, Any], parent: Mapping[str, Any], current: Mapping[str, Any]
) -> Equivalence | None:
    """:func:`affine_to_parent` at the bundle's own rule, or ``None`` for a bundle whose function
    is not a linear probe (an SAE bundle is not judged this way)."""
    function = manifest.get("function") or {}
    if function.get("kind") != "linear_probe/v1":
        return None
    return affine_to_parent(str(function["linear_probe"]["rule"]), parent, current)


def describe(eq: Equivalence) -> str:
    """The change in words, for the submit warning."""
    if eq.kind == BIAS_SHIFT:
        return "a uniform shift of the bias"
    return f"a positive rescale of the scores (x{eq.scale:.4g}) and a uniform shift"
