"""Program style's two session tools, provider-neutral (feature 018; FR-018.15; FTDD §6.6;
TD-2). Moved out of ``adapters/weightspace_adapter.py`` so both providers used one copy; the
engine is its one user since that adapter was deleted (018-P4).

- ``test_program(code)``: run a program once on the practice TRAIN split; metrics on the
  VALIDATION split. It is an observation tool: it costs :attr:`ProgramTools.COST` forward passes,
  which **the provider** charges (``miwsp_engine`` through its own forward budget), so this
  module never sees a budget.
- ``submit_program(code)``: store the program, at most once, after two identical runs and a
  change-limit check.

A run frozen with the version-4 observation tools (``judging.shows_uncertainty``; plan A1) gets
``test_program``'s AUROC difference from the parent with its interval, as ``compare_to_parent`` 4
gives it: in D128 round 9 the regressed edit was chosen on 51 ``test_program`` calls.

Nothing here imports a WSP library or an LLM SDK (the fence in ``test_wsp_fences.py``); the
patch's ``created_by`` is the provider it is given, never a hard-coded one.

**Judging on** (``judging=True``; fix 2026-11-22, D128 round 7, whose candidates all came from
programs): ``test_program`` answers as observation tools version 3 do (AUROC, and the head and the
parent each at its own bar re-cut to the target false-alarm rate on the practice validation
negatives that were not truncated), and the FIRST ``submit_program`` whose edit is a positive
affine map of the parent (:mod:`wsp_worker.equivalence`) is not stored: the editor is told once
(:data:`wsp_worker.judging.AFFINE_SUBMIT`) and may submit it again. Off, both answer exactly as
before, so a run recorded before replays as it ran; the provider decides (the engine from its
transcript header, the adapter from the original transcript).
"""

from __future__ import annotations

import datetime as dt
from collections.abc import Callable, Mapping
from typing import Any

import numpy as np

from wsp_worker import equivalence, observation_tools, program
from wsp_worker import judging as judging_mod
from wsp_worker import limits as limits_mod
from wsp_worker.bundle import Bundle
from wsp_worker.scoring import Scorer, auroc, bar_at, cut_bar, rates, rates_at_or_above

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
        judging: bool = False,
        on_affine: Callable[[equivalence.Equivalence], None] | None = None,
        data_free_errors: bool = True,
    ) -> None:
        self.bundle, self.request, self.run_id = bundle, request, run_id
        self.created_by = dict(created_by)
        self.submitted: str | None = None
        self.judging = judging
        self._on_affine = on_affine
        #: Whether the editor has been told that a submitted program cannot change the verdict.
        self.affine_warned = False
        self._parent = {n: t.values().astype(np.float64) for n, t in bundle.tensors.items()}
        #: Fix 2026-11-28 (E7 §5.3 gap 2): no refusal quotes the program's own text; a replay
        #: passes its header's ``program_errors`` (``program`` module docstring).
        self.data_free = data_free_errors
        self.train = program.SupportSet("practice_train", bundle, data_free=data_free_errors)
        arrays = bundle.practice.arrays
        self._scorer = Scorer(
            bundle.manifest["function"], bundle.practice.activations, arrays["practice_offsets"]
        )
        self._labels = np.asarray(arrays["practice_labels"])
        validation = np.asarray(arrays["practice_validation_mask"], dtype=bool)
        self._validation_rows = np.arange(self._scorer.rows, dtype=np.int64)[validation]
        truncated = np.asarray(arrays["practice_truncated_mask"], dtype=bool)
        #: The scorer's practice basis: validation negatives not truncated (``judging`` on).
        self._basis_rows = np.flatnonzero(
            validation & ~truncated & ~self._labels.astype(bool)
        ).astype(np.int64)
        self._target_fpr = observation_tools.target_fpr_of(bundle.manifest)
        self._uncertainty = judging_mod.shows_uncertainty(request)
        self._seed = program.seed_for(bundle.manifest["bundle_sha256"])

    def test_program(self, code: str) -> dict[str, Any]:
        """Run the program on the practice TRAIN split; metrics on the VALIDATION split."""
        try:
            edited = program.run_once(
                code, self.bundle, self.train, self.request, data_free=self.data_free
            )
        except program.ProgramError as exc:
            return {"refused": exc.code, "message": str(exc)[:500]}
        tensors = {n: t.values().astype(np.float64) for n, t in edited.items()}
        if self.judging:
            return self._judged(tensors)
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

    def _at_own_bar(self, tensors: Mapping[str, np.ndarray]) -> dict[str, Any]:
        rows = self._validation_rows
        scores = self._scorer.score(tensors, rows)
        bar = cut_bar(self._scorer.score(tensors, self._basis_rows), self._target_fpr)
        return {
            "auroc": auroc(scores, self._labels[rows]),
            "at_own_bar": {
                "bar": bar,
                "basis": "practice_validation_negatives",
                "basis_negatives": int(self._basis_rows.size),
                "target_false_alarm_rate": self._target_fpr,
                "fires": "at_or_above",
                **rates_at_or_above(scores, self._labels[rows], bar),
            },
        }

    def _judged(self, tensors: Mapping[str, np.ndarray]) -> dict[str, Any]:
        """``test_program`` with judging on: the program's head and the parent, each at its own
        re-cut bar, on the VALIDATION split."""
        rule = str(self._scorer.rule)
        same = equivalence.affine_to_parent(rule, self._parent, tensors) is not None
        out: dict[str, Any] = {
            "split": "validation",
            "rows": int(self._validation_rows.size),
            "primary_metric": judging_mod.PRIMARY_METRIC,
            **self._at_own_bar(tensors),
            "parent": self._at_own_bar(self._parent),
            "affine_equivalent_to_parent": same,
            "note": judging_mod.invariance_note(rule),
        }
        if same:
            out["same_verdict_note"] = judging_mod.SAME_VERDICT
        if self._uncertainty:
            rows = self._validation_rows
            after = self._scorer.score(tensors, rows)
            before = self._scorer.score(self._parent, rows)
            out.update(observation_tools.uncertainty(after, before, self._labels[rows], self._seed))
        return out

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
                data_free=self.data_free,
            )
        except program.ProgramError as exc:
            return f"refused: {exc.code}: {exc}"
        if done.manifest is not None:
            broken = limits_mod.check_limits(done.manifest, self.request["change_limits"])
            if broken:
                details = "; ".join(v.detail for v in broken)
                return f"refused: wsp_change_limit_exceeded: {details}"
        if self.judging and not self.affine_warned:
            tensors = {n: t.values().astype(np.float64) for n, t in done.tensors.items()}
            same = equivalence.for_bundle(self.bundle.manifest, self._parent, tensors)
            if same is not None:
                self.affine_warned = True
                if self._on_affine is not None:
                    self._on_affine(same)
                return judging_mod.AFFINE_SUBMIT.format(
                    what=equivalence.describe(same), tool="submit_program"
                )
        self.submitted = code
        return "program stored; call submit to finish."

    def candidate(self) -> program.Outcome | None:
        """After the session: the stored program run once more in a fresh namespace, or None."""
        if self.submitted is None:
            return None
        edited = program.run_once(
            self.submitted, self.bundle, self.train, self.request, data_free=self.data_free
        )
        return program.outcome(
            edited, self.bundle, self.request, self.run_id, now(), created_by=self.created_by
        )
