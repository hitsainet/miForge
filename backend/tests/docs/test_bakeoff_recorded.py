"""The bake-off is recorded before the provider interface is called stable, and the bake-off's
libraries stay out of every dependency list (feature 016; FTASKS 10.1, 10.4; FR-016.24; P3, D9).

- **The docs check:** the FTASKS may tick 10.3 ("the provider interface is stable") only when
  ``0xcc/reviews/016_bakeoff.md`` exists and records both libraries (control C24: tick 10.3 with
  no record, and this goes red).
- **The dependency scan:** ``wsplab`` and ``dspy`` (``dspy-wsp``) are comparators only; neither
  may appear in ``pyproject.toml``, ``constraints.txt`` or either image's requirements.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

from tests.support.mirror_snapshot import mirror_snapshot

ROOT = Path(__file__).resolve().parents[3]
#: The FTASKS and the bake-off record live in `0xcc/`, which the public mirror's sync strips:
#: the two docs checks skip on the mirror's snapshot ONLY (`mirror_snapshot()`; the private CI
#: is the gate), and the dependency scan below runs everywhere. Read lazily: a module-level
#: `next()` over an empty glob raised StopIteration at collection and errored the mirror's whole
#: file (016-P1b, 2026-11-17).
TASKS_DIR = ROOT / "0xcc" / "tasks"
ON_MIRROR = mirror_snapshot(ROOT)
needs_ftasks = pytest.mark.skipif(
    ON_MIRROR, reason="the public mirror strips 0xcc/; checked in the private repository"
)


def _ftasks() -> Path:
    found = sorted(TASKS_DIR.glob("016_FTASKS*.md"))
    assert len(found) == 1, f"expected one 016 FTASKS in {TASKS_DIR}, found {len(found)}"
    return found[0]


RECORD = ROOT / "0xcc" / "reviews" / "016_bakeoff.md"
DEPENDENCY_FILES = [
    ROOT / "pyproject.toml",
    ROOT / "constraints.txt",
    *sorted((ROOT / "backend").glob("requirements*.txt")),
]
COMPARATORS = re.compile(r"(?im)^\s*[\"']?(wsplab|dspy(-wsp)?)\b")


def task_box(text: str, task: str) -> str:
    found = re.search(rf"^\s*- \[(.)\] {re.escape(task)} ", text, re.MULTILINE)
    assert found, f"task {task} not found in the 016 FTASKS"
    return found.group(1)


@needs_ftasks
def test_the_interface_is_stable_only_after_the_bakeoff_is_recorded() -> None:
    text = _ftasks().read_text(encoding="utf-8")
    if task_box(text, "10.3") != "x":
        return  # not claimed stable: nothing to check yet
    assert RECORD.is_file(), "10.3 is ticked but 0xcc/reviews/016_bakeoff.md does not exist"
    record = RECORD.read_text(encoding="utf-8")
    for library in ("wsplab", "dspy-wsp"):
        assert library in record, f"the bake-off record does not cover {library}"
    assert re.search(
        r"(?i)provider interface.*\b(stable|changes?)\b", record
    ), "the record does not state whether the provider interface is stable"


@needs_ftasks
def test_the_docs_check_reads_the_real_task_line() -> None:
    """The check above is not vacuous: 10.3 is in the FTASKS, with a box it can read."""
    assert task_box(_ftasks().read_text(encoding="utf-8"), "10.3") in (" ", "x", "?")


@pytest.mark.parametrize("path", DEPENDENCY_FILES, ids=lambda p: p.name)
def test_no_bakeoff_library_is_a_dependency(path: Path) -> None:
    assert path.is_file(), f"{path} is missing"
    hits = [m.group(0).strip() for m in COMPARATORS.finditer(path.read_text(encoding="utf-8"))]
    assert not hits, f"{path.name} names a bake-off comparator: {hits}"


def test_the_scan_sees_a_planted_line(tmp_path: Path) -> None:
    """The scan's own negative control: the pattern matches the lines it must refuse."""
    for line in ("wsplab==0.2.0", "dspy-wsp==0.2.0", '  "dspy>=2",', "DSPy==3"):
        assert COMPARATORS.search(line), line
    assert not COMPARATORS.search("weightspace==0.3.0")
