"""The F4/F6 boundary, kept by what a file **cannot reach**  (FR-004.25; R-02.25 is F6's).

F4 detects, names and publishes a breach; **F6 stops the run.** If F4 stopped runs it would
duplicate R-02.25 and the two would eventually disagree - and the division is the same one F3
uses for its health signals: *"F3 measures and names; it does not stop a run."*

A convention is a rule somebody remembers. This is the structural version: ``retention.py``
imports nothing that can stop, cancel or transition a run, so a future edit that wanted to
would have to add the import first - and that is what this file watches.
"""

from __future__ import annotations

import ast
from pathlib import Path
from typing import ClassVar

import pytest

SRC = Path(__file__).resolve().parents[2] / "src"
RETENTION = SRC / "services" / "scoring" / "retention.py"

#: Modules that can stop, cancel or transition a run. Named rather than pattern-matched, so
#: a reader can see exactly what the boundary is.
FORBIDDEN_MODULES = (
    "workers.cancellation",
    "workers.celery_app",
    "workers.janitor",
    "workers.run_record",
    "models.run",
    "db.session",
)

#: Call names that would end a run from inside this module.
FORBIDDEN_CALLS = ("request_stop", "revoke", "RunCancelled", "check")


def _called(func: ast.expr) -> str:
    """The callee's readable name, for a failure message that names the line."""
    return func.attr if isinstance(func, ast.Attribute) else ast.unparse(func)


def _imports(path: Path) -> list[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    names: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            names.extend(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            names.append(node.module)
    return names


class TestRetentionCannotStopARun:
    def test_it_imports_nothing_that_can(self) -> None:
        offenders = [name for name in _imports(RETENTION) if name in FORBIDDEN_MODULES]
        assert offenders == [], (
            f"retention.py imports {offenders}. R-02.25 is F6's; F4 publishes a signal and "
            f"F6 acts on it. Keeping that boundary structural rather than conventional is "
            f"the point of this file."
        )

    def test_it_names_no_stop_transition(self) -> None:
        source = RETENTION.read_text(encoding="utf-8")
        tree = ast.parse(source)
        hits = [
            f"{node.lineno}:{_called(node.func)}"
            for node in ast.walk(tree)
            if isinstance(node, ast.Call)
            and (
                (isinstance(node.func, ast.Attribute) and node.func.attr in FORBIDDEN_CALLS)
                or (isinstance(node.func, ast.Name) and node.func.id in FORBIDDEN_CALLS)
            )
        ]
        assert hits == [], f"retention.py reaches a run transition at {hits}"

    def test_it_carries_the_rerun_instruction_for_M1_and_M3(self) -> None:
        """The instruction lives **in the file**, not only in a task list.

        A task list is not read during a refactor; a module docstring is. M-1 and M-3 must be
        re-run after any edit to these lines, and the next session has to find that without
        being told where to look.
        """
        source = RETENTION.read_text(encoding="utf-8")
        assert "RE-RUN MUTATION CONTROLS M-1 AND M-3 AFTER ANY EDIT TO THIS FILE" in source

    def test_the_strictness_comment_sits_above_the_comparison(self) -> None:
        """The comment is load-bearing and must survive a refactor (FTID §14).

        It is the only place that says what inverting the line costs, and it is addressed to
        whoever is about to change it.
        """
        lines = RETENTION.read_text(encoding="utf-8").splitlines()
        index = next(i for i, line in enumerate(lines) if "if score < floor.value:" in line)
        preceding = "\n".join(lines[max(0, index - 12) : index])
        assert "Invert this comparison" in preceding
        assert "STRICT" in preceding
        assert "re-run M-1" in preceding


class TestTheScoringPathWritesNoParameters:
    """FR-004.20: an evaluation is **read-only on the model**.

    *The cheapest way to corrupt a run is to measure it carelessly*, so the scoring package
    and the in-process provider are asserted not to reach an optimizer, a backward pass or a
    save.
    """

    MODULES: ClassVar[list[Path]] = [
        *sorted((SRC / "services" / "scoring").glob("*.py")),
        SRC / "providers" / "in_process.py",
    ]

    FORBIDDEN = (
        "training.optimizer",
        "training.step",
        "training.trainer_state",
        "services.checkpoints.writer",
        "services.checkpoints.service",
    )

    @pytest.mark.parametrize("path", MODULES, ids=lambda p: p.name)
    def test_no_optimizer_or_save_import(self, path: Path) -> None:
        offenders = [name for name in _imports(path) if name in self.FORBIDDEN]
        assert offenders == [], f"{path.name} reaches {offenders}; scoring writes nothing"

    @pytest.mark.parametrize("path", MODULES, ids=lambda p: p.name)
    def test_no_backward_or_step_call(self, path: Path) -> None:
        tree = ast.parse(path.read_text(encoding="utf-8"))
        hits = [
            node.lineno
            for node in ast.walk(tree)
            if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr in ("backward", "zero_grad", "save_pretrained")
        ]
        assert hits == [], f"{path.name} performs a training write at lines {hits}"


class TestNoUpdateOnAScoreRow:
    """A re-score **appends**; a mutated history is worse than a duplicated one.

    One exception, and it is not history: :mod:`services.scoring.reindex_scores` corrects the
    **index** toward the manifest, which ADR-004 requires - *"where they disagree, the
    manifest wins and the disagreement is logged"* - and F10's ``resolve_part`` does exactly
    the same. Correcting a cache is not mutating a record, and the exclusion is named here
    rather than being a silent gap in the pattern.
    """

    def test_only_the_reindex_reconciliation_updates(self) -> None:
        offenders: list[str] = []
        for path in sorted((SRC / "services" / "scoring").glob("*.py")):
            if path.name == "reindex_scores.py":
                continue
            tree = ast.parse(path.read_text(encoding="utf-8"))
            for node in ast.walk(tree):
                if (
                    isinstance(node, ast.Call)
                    and isinstance(node.func, ast.Attribute)
                    and node.func.attr == "update"
                    and isinstance(node.func.value, ast.Name)
                    and node.func.value.id == "sa"
                ):
                    offenders.append(f"{path.name}:{node.lineno}")
        assert offenders == [], f"a score row is mutated at {offenders}; a re-score appends"
