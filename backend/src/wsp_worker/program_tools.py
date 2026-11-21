"""Program style's two session tools, provider-neutral (feature 018; FR-018.15; FTDD §6.6;
TD-2). Moved out of ``adapters/weightspace_adapter.py`` so both providers use one copy.

- ``test_program(code)``: run a program once on the practice TRAIN split; metrics on the
  VALIDATION split. It is an observation tool: it costs :attr:`ProgramTools.COST` forward passes,
  which **the provider** charges (``weightspace`` through its session's forward budget,
  ``miwsp_engine`` through its own), so this module never sees a budget.
- ``submit_program(code)``: store the program, at most once, after two identical runs and a
  change-limit check.

Nothing here imports a WSP library or an LLM SDK (the fence in ``test_wsp_fences.py``); the
patch's ``created_by`` is the provider it is given, never a hard-coded one.
"""

from __future__ import annotations

import datetime as dt
from collections.abc import Mapping
from typing import Any

import numpy as np

from wsp_worker import limits as limits_mod
from wsp_worker import program
from wsp_worker.bundle import Bundle
from wsp_worker.scoring import Scorer, auroc, bar_at, rates

#: The tools' JSON schemas and descriptions, shared by both providers' wrappers.
CODE_PARAMETERS: dict[str, Any] = {
    "type": "object",
    "properties": {"code": {"type": "string"}},
    "required": ["code"],
}
TEST_DESCRIPTION = (
    "Run a program (Python source defining edit(base, support)) on the practice train split; "
    "metrics on the validation split."
)
SUBMIT_DESCRIPTION = (
    "Submit the program as this session's result (once). It is run twice and must give the same "
    "edit both times."
)


def now() -> str:
    return dt.datetime.now(dt.UTC).isoformat().replace("+00:00", "Z")


class ProgramTools:
    """``test_program`` (an observation tool at its cost) and ``submit_program`` (once)."""

    COST = 1

    def __init__(
        self,
        bundle: Bundle,
        request: Mapping[str, Any],
        run_id: str,
        *,
        created_by: Mapping[str, str],
    ) -> None:
        self.bundle, self.request, self.run_id = bundle, request, run_id
        self.created_by = dict(created_by)
        self.submitted: str | None = None
        self.train = program.SupportSet("practice_train", bundle)
        arrays = bundle.practice.arrays
        self._scorer = Scorer(
            bundle.manifest["function"], bundle.practice.activations, arrays["practice_offsets"]
        )
        self._labels = np.asarray(arrays["practice_labels"])
        validation = np.asarray(arrays["practice_validation_mask"], dtype=bool)
        self._validation_rows = np.arange(self._scorer.rows, dtype=np.int64)[validation]
        self._target_fpr = float(
            (bundle.manifest.get("operating_point") or {}).get("target_fpr") or 0.01
        )

    def test_program(self, code: str) -> dict[str, Any]:
        """Run the program on the practice TRAIN split; metrics on the VALIDATION split."""
        try:
            edited = program.run_once(code, self.bundle, self.train, self.request)
        except program.ProgramError as exc:
            return {"refused": exc.code, "message": str(exc)[:500]}
        tensors = {n: t.values().astype(np.float64) for n, t in edited.items()}
        train = self.train.rows
        bar = bar_at(self._scorer.score(tensors, train), self._labels[train], self._target_fpr)
        rows = self._validation_rows
        scores = self._scorer.score(tensors, rows)
        return {
            "split": "validation",
            "rows": int(rows.size),
            "auroc": auroc(scores, self._labels[rows]),
            "bar": bar,
            **rates(scores, self._labels[rows], bar),
        }

    def submit_program(self, code: str) -> str:
        """Store the program (at most once), after two identical runs."""
        if self.submitted is not None:
            return "refused: wsp_program_already_submitted: a program was already submitted."
        try:
            done = program.run_checked(
                code,
                self.bundle,
                self.train,
                self.request,
                self.run_id,
                now(),
                created_by=self.created_by,
            )
        except program.ProgramError as exc:
            return f"refused: {exc.code}: {exc}"
        if done.manifest is not None:
            broken = limits_mod.check_limits(done.manifest, self.request["change_limits"])
            if broken:
                details = "; ".join(v.detail for v in broken)
                return f"refused: wsp_change_limit_exceeded: {details}"
        self.submitted = code
        return "program stored; call submit to finish."

    def candidate(self) -> program.Outcome | None:
        """After the session: the stored program run once more in a fresh namespace, or None."""
        if self.submitted is None:
            return None
        edited = program.run_once(self.submitted, self.bundle, self.train, self.request)
        return program.outcome(
            edited, self.bundle, self.request, self.run_id, now(), created_by=self.created_by
        )
