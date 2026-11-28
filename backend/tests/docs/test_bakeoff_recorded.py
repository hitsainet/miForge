"""The bake-off is recorded before the provider interface is called stable, and the bake-off's
libraries stay out of every dependency list (feature 016; FTASKS 10.1, 10.4; FR-016.24; P3, D9).

- **The docs check:** the FTASKS may tick 10.3 ("the provider interface is stable") only when
  ``0xcc/reviews/016_bakeoff.md`` exists and records both libraries (control C24: tick 10.3 with
  no record, and this goes red).
- **The dependency scan:** ``wsplab`` and ``dspy`` (``dspy-wsp``) are comparators only, and since
  018-P4 so is ``weightspace`` (retired, FR-018.18; it was the worker's provider until then); none
  may appear in ``pyproject.toml``, ``constraints.txt`` or any requirements file in the tree. The
  bake-off harness (``scripts/wsp_bakeoff/``, which the mirror strips) is the one place a
  comparator is installed, and it is not scanned.
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
#: Where a comparator may be installed: the throwaway bake-off harness only.
HARNESS = ROOT / "scripts" / "wsp_bakeoff"
SKIPPED_DIRS = ("node_modules", ".git", ".venv", "venv")


def requirements_files(root: Path = ROOT) -> list[Path]:
    """Every ``requirements*.txt`` in the tree, outside the bake-off harness (and vendored or
    virtual-environment directories): a file added anywhere is scanned without an edit here."""
    return sorted(
        path
        for path in root.rglob("requirements*.txt")
        if not path.is_relative_to(root / "scripts" / "wsp_bakeoff")
        and not any(part in SKIPPED_DIRS for part in path.relative_to(root).parts)
    )


DEPENDENCY_FILES = [ROOT / "pyproject.toml", ROOT / "constraints.txt", *requirements_files()]
#: A dependency line naming a comparator (a comment naming one is not a dependency).
COMPARATORS = re.compile(r"(?im)^\s*[\"']?(wsplab|dspy(-wsp)?|weightspace)\b")


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
    """The scan's own negative control: the pattern matches the lines it must refuse, the retired
    provider's included (until 018-P4 this asserted ``weightspace==0.3.0`` did NOT match: it was
    the worker's own dependency then), and passes a comment that only names one."""
    for line in (
        "wsplab==0.2.0",
        "dspy-wsp==0.2.0",
        '  "dspy>=2",',
        "DSPy==3",
        "weightspace==0.3.0",
        "weightspace[openai,anthropic]==0.3.0",
        '  "weightspace>=0.3",',
    ):
        assert COMPARATORS.search(line), line
    assert not COMPARATORS.search("# weightspace 0.3.0 was the provider until 018-P4")


def test_the_scan_covers_every_requirements_file_in_the_tree() -> None:
    """The scan reads the two worker locks (the files weightspace lived in) and would read a new
    requirements file wherever it is added; the harness alone is left out."""
    names = {p.relative_to(ROOT).as_posix() for p in DEPENDENCY_FILES}
    assert {
        "backend/requirements-wsp-worker.txt",
        "backend/requirements-wsp-worker-test.txt",
        "pyproject.toml",
        "constraints.txt",
    } <= names, names
    assert not any(n.startswith("scripts/wsp_bakeoff/") for n in names), names


def test_the_scan_finds_a_planted_requirements_file(tmp_path: Path) -> None:
    """A requirements file planted in a new directory is found and refused; the same file in the
    bake-off harness is not scanned."""
    planted = tmp_path / "backend" / "new" / "requirements-x.txt"
    planted.parent.mkdir(parents=True)
    planted.write_text("numpy==2.5.4\nweightspace[openai,anthropic]==0.3.0\n", encoding="utf-8")
    harness = tmp_path / "scripts" / "wsp_bakeoff" / "requirements-harness.txt"
    harness.parent.mkdir(parents=True)
    harness.write_text("weightspace==0.3.0\n", encoding="utf-8")
    assert requirements_files(tmp_path) == [planted]
    assert [m.group(1) for m in COMPARATORS.finditer(planted.read_text("utf-8"))] == ["weightspace"]
