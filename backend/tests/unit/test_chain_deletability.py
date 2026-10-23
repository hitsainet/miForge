"""FR-006.46: **nothing outside the chain modules depends on a chain existing**  (AMB-5).

BRD §9 asks whether chained runs are *"a first-release feature or a script on top of single
runs"* and does not answer. R-02.27 is in F6's requirement set, so F6 builds it - as a thin
orchestrator with no privileged path, so that **if the operator decides it is a script,
deleting these modules removes a feature rather than unpicking the loop.** That property is
the requirement; the decision is not F6's.

This file is the assertion rather than the intention, and it is a *static* scan because the
property is about imports: a dynamic test would only notice a dependency the test happened
to execute.

## Two deliberate non-members of the deletable set, and why neither breaks the property

* ``mf_runs.chain_id`` / ``stage_index`` - **columns**, added by migration ``0016``, both
  nullable with no default. ``NULL`` is what a standalone run carries, and every run row
  already works with them null; dropping chaining leaves two unused nullable columns, not a
  broken loop. The reindex deliberately registers no chain rebuilder (task 9.4).
* The ``TASK_MODULES`` / ``beat_schedule`` entries and the router mount - **three
  registration lines** in two shared files. They are what makes the capability reachable at
  all (ADR-016 #1), so they are counted as part of the deletion, not as a dependency on it.

What the property forbids is the *loop* knowing about chains: a step, a boundary, a stop
condition or a save that behaves differently because a run is a chain stage would be the
privileged path FR-006.46 rules out, and it is what would have to be unpicked.
"""

from __future__ import annotations

import re
from pathlib import Path

SRC = Path(__file__).resolve().parents[2] / "src"

#: The modules that may know about chains. Deleting exactly these four paths - plus the
#: three registration lines named in the module docstring - removes the feature.
CHAIN_MODULES: tuple[str, ...] = (
    # **Six files, not the four the task list names.** ``schemas/chain.py`` shipped ahead of
    # the orchestration with F6's safety-critical half (``AdvanceOn``, ``assert_advanceable``,
    # ``advance_filter``, ``halt_reason``) and is imported by nothing outside this set -
    # verified, not assumed - so it belongs to the deletion too. ``tables.py`` is the other
    # one the four-module framing leaves out.
    "schemas/chain.py",
    "services/chains/__init__.py",
    "services/chains/service.py",
    "services/chains/tables.py",
    "workers/chain_tasks.py",
    "api/v1/endpoints/chains.py",
)

#: Shared files that carry a *registration* line and nothing more. Each is part of the
#: deletion, so a reference from one of them is expected - but it must be **only** the
#: registration, which the test below checks by counting what it matches.
REGISTRATION_FILES: tuple[str, ...] = (
    "workers/celery_app.py",
    "api/v1/router.py",
)

#: Every way a module could reach the chain code. A scanner that cannot see one form is a
#: guard that is not there, so the self-test below proves it sees each of them.
PATTERNS: tuple[re.Pattern[str], ...] = (
    re.compile(r"\bimport\s+services\.chains\b"),
    re.compile(r"\bfrom\s+services\.chains\b"),
    re.compile(r"\bfrom\s+services\s+import\s+chains\b"),
    re.compile(r"\bimport\s+workers\.chain_tasks\b"),
    re.compile(r"\bfrom\s+workers\.chain_tasks\b"),
    re.compile(r"\bfrom\s+workers\s+import\s+chain_tasks\b"),
    re.compile(r"\bfrom\s+api\.v1\.endpoints\.chains\b"),
    re.compile(r"\bfrom\s+api\.v1\.endpoints\s+import\s+\([^)]*\bchains\b"),
    re.compile(r"\bfrom\s+\.+chains\b"),
    re.compile(r"\bmf_chains\b"),
    re.compile(r"\bmf_chain_stages\b"),
    re.compile(r"\bchain_service\b"),
    re.compile(r"\bchain_sweep\b"),
    # The registration forms, which are STRINGS rather than imports: `TASK_MODULES` names
    # the module and the beat entry names the task. A scanner blind to them would report
    # `celery_app.py` as carrying no reference at all, which would make the
    # "only a registration" assertion below vacuous.
    re.compile(r"[\"']workers\.chain_tasks[\"']"),
    re.compile(r"[\"']services\.chains[\w.]*[\"']"),
)


#: Triple-quoted strings and whole-line comments, stripped before the search.
#:
#: **A citation is not an operation**, and this project has already paid for that retune
#: once: F4's migration guard flagged a *comment* citing a precedent. ``schemas/chain.py``'s
#: own docstring names ``mf_chains``, and ``services/reindex.py`` explains in prose why it
#: registers no chain rebuilder - which is the opposite of a dependency.
_DOCSTRING = re.compile(r'("""|\'\'\')(?:.|\n)*?\1')
_COMMENT = re.compile(r"^\s*#.*$", re.MULTILINE)


def _code_of(text: str) -> str:
    return _COMMENT.sub("", _DOCSTRING.sub("", text))


def _hits(text: str) -> list[str]:
    """Every chain reference in ``text``'s **code**. Searched whole, not line by line.

    Whole-text rather than per-line because ``from api.v1.endpoints import (\n    chains,``
    is one import across two lines, and a scanner that cannot see one form is a guard that
    is not there - which is what the self-test below exists to prove.
    """
    code = _code_of(text)
    return [match.group(0) for pattern in PATTERNS for match in pattern.finditer(code)]


def _modules() -> list[Path]:
    return sorted(path for path in SRC.rglob("*.py"))


class TestTheScannerSeesEachForm:
    """The self-test. Without it this file is a guard that may not be there."""

    def test_every_pattern_matches_the_form_it_is_for(self) -> None:
        samples = [
            "import services.chains",
            "from services.chains import service",
            "from services import chains",
            "import workers.chain_tasks",
            "from workers.chain_tasks import chain_sweep",
            "from workers import chain_tasks",
            "from api.v1.endpoints.chains import router",
            "from api.v1.endpoints import (\n    chains,",
            "from .chains import service",
            "from ..chains import service",
            "    table = sa.table('mf_chains')",
            "    rows = select(mf_chain_stages)",
            "    chain_service.advance_one(session, cid)",
            "    chain_sweep.delay()",
        ]
        for sample in samples:
            assert _hits(sample), f"the scanner cannot see: {sample!r}"

    def test_a_comment_or_a_docstring_is_not_a_dependency(self) -> None:
        """A retune this project has already paid for once: F4's migration guard flagged a
        *comment* citing a precedent, and a citation is not an operation."""
        assert not _hits("# chain rows are deliberately excluded (task 9.4)")
        assert not _hits("    # see services.chains.service for why")
        assert not _hits('"""mf_chains is not rebuildable from disk (task 9.4)."""')

    def test_stripping_prose_does_not_blind_it_to_code_beside_the_prose(self) -> None:
        """The negative control for the stripper itself. A regex that ate too much would
        make every file read as clean - *a measurement that did not happen reported as a
        result.*"""
        assert _hits('"""A docstring."""\nimport services.chains\n')


class TestNothingOutsideTheChainModulesDependsOnAChain:
    def test_no_module_outside_the_set_reaches_the_chain_code(self) -> None:
        offenders: dict[str, list[str]] = {}
        for path in _modules():
            relative = path.relative_to(SRC).as_posix()
            if relative in CHAIN_MODULES or relative in REGISTRATION_FILES:
                continue
            hits = _hits(path.read_text(encoding="utf-8"))
            if hits:
                offenders[relative] = hits
        assert not offenders, (
            "FR-006.46 requires that deleting the chain modules removes a feature rather "
            f"than unpicking the loop. These reach into it: {offenders}"
        )

    def test_the_registration_files_carry_only_a_registration(self) -> None:
        """Each shared file may name the chain **once or twice**, as the wiring - not as
        logic.

        A ``if run.chain_id is not None`` in either would be a privileged path, and the
        count is what notices: a third match means something beyond the registration arrived.
        """
        for relative in REGISTRATION_FILES:
            path = SRC / relative
            if not path.is_file():  # pragma: no cover - both exist
                continue
            hits = _hits(path.read_text(encoding="utf-8"))
            assert len(hits) <= 4, f"{relative} carries more than its registration: {hits}"

    def test_the_loop_itself_knows_nothing_about_chains(self) -> None:
        """Named explicitly as well as covered by the sweep above.

        These are the files a chain would have to reach into to become a privileged path,
        and they are the ones that would have to be *unpicked* rather than deleted. Listed
        by name so a future reader sees which files the property is actually about.
        """
        for relative in (
            "training/loop.py",
            "training/boundary.py",
            "training/tasks.py",
            "training/step.py",
            "services/run_launch.py",
            "services/run_definition.py",
            "services/stop_service.py",
            "services/checkpoints/service.py",
            "services/reindex.py",
            "workers/janitor.py",
        ):
            path = SRC / relative
            assert path.is_file(), f"{relative} moved; this list needs updating"
            assert not _hits(path.read_text(encoding="utf-8")), relative

    def test_the_finishing_task_does_not_enqueue_its_successor(self) -> None:
        """FR-006.45, and the static form is the one that holds.

        *"The finishing stage's own task does not enqueue its successor: a worker that died
        between the terminal transition and the enqueue would stall the chain forever."* The
        previous test already proves ``training/tasks.py`` cannot reach the chain code at
        all; this one says **why** that particular absence matters, and names the second
        file the rule is about.
        """
        for relative in ("training/tasks.py", "workers/run_record.py"):
            text = (SRC / relative).read_text(encoding="utf-8")
            assert "chain" not in text.replace("chain_id", "").replace(
                "chaining", ""
            ), f"{relative} mentions a chain; the finishing task must not know there is one"
