"""Origin headers name a current repository and have a ``docs/REUSE.md`` row (FR-015.28).

Copied code says where it came from (ADR-001, ``docs/REUSE.md``): repository, path, commit.
miStudio becomes miWorkbench and miLLM becomes miServer, so a header's ``owner/repo`` must name
a current repository, or an old one only while its census ``repo`` record is ``transition``
(CF-13; never a list in this file). The three private-organisation miStudio headers were
rewritten in P4, and since then this tree's own headers must each name a CURRENT repository.
Until the export's ``closed`` reaches P4 the repository rule REPORTS; from then it is enforced
in the private repository and reports on the public mirror (suite R23, ``brand_scan.mode``).
The index rule is enforced now, because the index was filled in P0 (it had said "Nothing yet"
over sixteen headers, a pre-existing defect).
"""

from __future__ import annotations

import re
import subprocess
from dataclasses import dataclass
from pathlib import Path

import pytest

from tests.support import brand_scan as bs
from tests.support import legacy_identifiers as li
from tests.support.mirror_snapshot import mirror_snapshot

REPO_ROOT = li.REPO_ROOT
REUSE = Path("docs/REUSE.md")
CURRENT = frozenset({"miWorkbench", "miServer", "miDataworks"})
HEADER = re.compile(r"^\W*Origin:\s|Copied from\s")
REPO = re.compile(rf"\b({li.PRIVATE}|hitsainet)/(mi[A-Za-z]+)")
#: Files whose header is configuration prose, not a copy (REUSE.md: workflows and manifests
#: need no row), and the index document itself, whose example header is still checked.
NO_ROW = re.compile(r"^(\.github/workflows/|gitops/|docs/REUSE\.md$)")
#: The measured count (2026-11-08): 16 code headers (8 backend from miDataworks, 3 from
#: miStudio, 5 frontend from miDataworks), the workflow line and the REUSE.md example. FTID
#: §3.11 lists the same three groups but totals them 15, hence its "17".
EXPECTED = 18


@dataclass(frozen=True)
class Header:
    path: str
    line: int
    text: str


def headers(root: Path = REPO_ROOT) -> list[Header]:
    out = subprocess.run(
        [
            "git",
            "ls-files",
            "-z",
            "--",
            ".",
            ":(exclude)AGENT.md",
            ":(exclude,glob)0xcc/**",
            *bs.EXPORT_EXCLUSIONS,
        ],
        cwd=root,
        capture_output=True,
        check=True,
    ).stdout.decode()
    found: list[Header] = []
    for rel in sorted(p for p in out.split("\0") if p):
        path = root / rel
        try:
            lines = path.read_text("utf-8").splitlines()
        except (UnicodeDecodeError, FileNotFoundError, IsADirectoryError):
            continue
        window = lines if rel == REUSE.as_posix() else lines[:12]
        for number, line in enumerate(window, 1):
            if HEADER.search(line):
                # A header may break its `owner/repo` onto the next line.
                joined = " ".join(window[number - 1 : number + 1])
                found.append(Header(rel, number, joined))
                break
    return found


def repo_findings(
    found: list[Header], export: li.Export, *, allow_transition: bool = True
) -> list[str]:
    # By the whole `owner/repo`: the private and public halves of a pair are separate records
    # (control N31 found a name-only check letting a retired one through on its twin's status).
    # ``allow_transition=False`` is the rule for THIS tree since the P4 rewrite (FR-015.28,
    # CF-13: an old name was acceptable "before the P4 rewrite" only): every header names a
    # current repository, whatever the export's `repo` records still say.
    transition = (
        {r["old"] for r in export.repo if r["status"] == "transition"}
        if allow_transition
        else set()
    )
    problems = []
    for header in found:
        names = REPO.findall(header.text)
        if not names:
            problems.append(f"{header.path}:{header.line}: names no owner/repo")
        for owner, repo in names:
            if repo not in CURRENT and f"{owner}/{repo}" not in transition:
                problems.append(f"{header.path}:{header.line}: {repo} is not a current repository")
    return problems


def index_section(reuse_text: str) -> str:
    """The ``## Index`` section alone: a file named in a per-feature table further down is
    not thereby indexed (control N64 found the check satisfied by those tables)."""
    match = re.search(r"^## Index\n(.*?)(?=^## |\Z)", reuse_text, re.S | re.M)
    assert match, "docs/REUSE.md has no '## Index' section"
    return match.group(1)


def index_findings(found: list[Header], reuse_text: str) -> list[str]:
    reuse_text = index_section(reuse_text)
    return [
        f"{h.path}: no row in docs/REUSE.md's index"
        for h in found
        if not NO_ROW.match(h.path) and f"| `{h.path}` |" not in reuse_text
    ]


def test_discovers_every_header() -> None:
    found = headers()
    listing = "\n".join(f"{h.path}:{h.line}" for h in found)
    # docs/REUSE.md (and its example header) is stripped from the public mirror: one fewer
    # there, and only on the mirror snapshot; every code and workflow header must still be found.
    on_mirror = mirror_snapshot(REPO_ROOT)
    assert [h.path for h in found].count(REUSE.as_posix()) == (0 if on_mirror else 1), listing
    assert len(found) == EXPECTED - (1 if on_mirror else 0), listing


def test_every_header_names_a_current_repo() -> None:
    # Strict since the P4 rewrite: no header of this tree may name an old repository, even one
    # whose record is still `transition` (control N21: reverting a header must be red).
    problems = repo_findings(headers(), li.load(), allow_transition=False)
    print(f"\n[origin headers] {len(problems)} finding(s) ({bs.mode()} mode)")
    for line in problems:
        print(f"  {line}")
    if bs.mode() == "enforced":
        assert not problems, "\n".join(problems)


def test_every_header_has_an_index_row() -> None:
    reuse = REPO_ROOT / REUSE
    if not reuse.is_file() and mirror_snapshot(REPO_ROOT):
        pytest.skip("MIRROR: docs/REUSE.md is private and stripped from the public mirror")
    assert reuse.is_file(), "docs/REUSE.md is missing from a tree that is not the mirror snapshot"
    problems = index_findings(headers(), reuse.read_text("utf-8"))
    assert not problems, "\n".join(problems)


def test_old_repo_only_while_transition() -> None:
    """The repository rule, enforced on planted headers against the committed export."""
    export = li.load()
    stand_in = li.withheld_key("repo", f"{li.PRIVATE}/miStudio")
    if export.withholds(stand_in):
        pytest.skip(f"{stand_in} withheld from the public variant")
    planted = [Header("x.py", 1, f"# Origin: miStudio ({li.PRIVATE}/miStudio) a.py @ 1")]

    def with_status(status: str) -> li.Export:
        return li.Export(
            path=export.path,
            header=export.header,
            entries=export.entries,
            env=export.env,
            host=export.host,
            history_tools=export.history_tools,
            repo=[{**r, "status": status} for r in export.repo],
            guards=export.guards,
        )

    transition, retired = with_status("transition"), with_status("retired")
    assert not repo_findings(planted, transition), "an old repo name is allowed while transition"
    assert repo_findings(planted, retired), "a retired old repo name must be a finding"
    status = {r["old"]: r["status"] for r in export.repo}[f"{li.PRIVATE}/miStudio"]
    assert bool(repo_findings(planted, export)) == (
        status == "retired"
    ), f"the committed export says {status}, and the header rule disagrees"
    assert repo_findings(
        planted, transition, allow_transition=False
    ), "after the P4 rewrite an old repo name in a header is a finding even while transition"
    assert repo_findings([Header("y.py", 1, "# Copied from miLLM's workflow")], export)
    assert not repo_findings(
        [Header("z.py", 1, f"# Origin: miDataworks ({li.PRIVATE}/miDataworks) b.py @ 2")],
        export,
    )


def test_the_repository_rule_fails_when_the_tree_enforces(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """Wiring: a header naming an old repository FAILS in an enforced tree and is only
    printed in a reporting one (a rule that printed in every mode would make P4's flip a
    no-op here)."""
    import sys

    module = sys.modules[__name__]
    old = Header("x.py", 1, f"# Origin: miStudio ({li.PRIVATE}/miStudio) a.py @ 1")
    monkeypatch.setattr(module, "headers", lambda *a, **k: [old])
    monkeypatch.setattr(bs, "mode", lambda *a, **k: "enforced")
    with pytest.raises(AssertionError, match="miStudio is not a current repository"):
        test_every_header_names_a_current_repo()
    monkeypatch.setattr(bs, "mode", lambda *a, **k: "report")
    test_every_header_names_a_current_repo()
    assert "[origin headers] 1 finding(s) (report mode)" in capsys.readouterr().out
