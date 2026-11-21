"""The worker scorer as it stood before the 2026-11-21 vectorisation (``wsp_worker/scoring.py`` at
``7100c3b``), frozen verbatim as the golden reference for ``test_worker_scoring.py`` and the
benchmark ``backend/scripts/bench_wsp_scorer.py``.

One row at a time, every token copied to float64, standardised per row: the path that made each
observation tool take ~3 minutes on a 4.26M-token practice set (run ``wsp_cb092cedad97``). Do not
"fix" this file: its value is that it is the old arithmetic.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np

from wsp_worker.scoring import DEFAULT_TAU, DEFAULT_WINDOW, EPS, ScoringError, combine


def _vec(tensors: Mapping[str, np.ndarray], name: str) -> np.ndarray | None:
    value = tensors.get(name)
    return None if value is None else np.asarray(value, dtype=np.float64).reshape(-1)


class OldScorer:
    """Row scores for one function over the practice activations, for any set of head tensors."""

    def __init__(
        self, function: Mapping[str, Any], activations: np.ndarray, offsets: np.ndarray
    ) -> None:
        if function.get("kind") != "linear_probe/v1":
            raise ScoringError(f"function kind {function.get('kind')!r} is not scored in phase 1")
        spec = function["linear_probe"]
        params = spec.get("params") or {}
        self.rule = str(spec["rule"])
        self.tau = DEFAULT_TAU if params.get("tau") is None else float(params["tau"])
        self.window = DEFAULT_WINDOW if params.get("window") is None else int(params["window"])
        self.activations = activations
        self.offsets = offsets

    @property
    def rows(self) -> int:
        return int(self.offsets.size) - 1

    def tokens(self, row: int) -> np.ndarray:
        """One row's token activations, as float64 (a copy: the caller cannot reach the memmap)."""
        if not 0 <= row < self.rows:
            raise ScoringError(f"row {row} is outside the {self.rows} practice rows")
        start, end = int(self.offsets[row]), int(self.offsets[row + 1])
        return np.array(self.activations[start:end], dtype=np.float64)

    def token_scores(self, tensors: Mapping[str, np.ndarray], row: int) -> np.ndarray:
        weight, standard = self._standard(tensors, row)
        bias = _vec(tensors, "bias")
        assert bias is not None
        scores: np.ndarray = standard @ weight + float(bias[0])
        return scores

    def _standard(self, tensors: Mapping[str, np.ndarray], row: int) -> tuple[np.ndarray, Any]:
        weight, bias = _vec(tensors, "weight"), _vec(tensors, "bias")
        if weight is None or bias is None or bias.size != 1:
            raise ScoringError("a linear probe needs a 1-D 'weight' and a [1] 'bias' tensor")
        z = self.tokens(row)
        if z.shape[-1] != weight.size:
            raise ScoringError(f"activations are d_model={z.shape[-1]}, the head {weight.size}")
        mean, std = _vec(tensors, "mean"), _vec(tensors, "std")
        out = z
        if mean is not None:
            out = out - mean
        if std is not None:
            degenerate = np.abs(std) <= EPS
            out = np.where(degenerate, 0.0, out / np.where(degenerate, 1.0, std))
        return weight, out

    def score(
        self, tensors: Mapping[str, np.ndarray], rows: Sequence[int] | np.ndarray
    ) -> np.ndarray:
        """float32 row scores for ``rows``."""
        out = np.empty(len(rows), dtype=np.float32)
        query = _vec(tensors, "attention_query")
        bias = _vec(tensors, "bias")
        for i, r in enumerate(rows):
            weight, standard = self._standard(tensors, int(r))
            assert bias is not None
            token_scores = standard @ weight + float(bias[0])
            logits = standard @ query if self.rule == "attention" and query is not None else None
            out[i] = np.float32(
                combine(
                    self.rule,
                    token_scores,
                    attention_logits=logits,
                    tau=self.tau,
                    window=self.window,
                )
            )
        return out
