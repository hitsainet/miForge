"""What ``manual/`` publishes: only allowed files, and nothing internal or secret in them.

``manual/`` is NOT stripped by ``sync-to-clean.yml``: it goes to the public mirror as it is, and
the mirror builds it into the public GitHub Pages site. So everything under it is published
twice over, and a host name, a private address, a token or a reference to the internal planning
tree that reaches it is public the moment ``main`` syncs.

The rules live in ONE place, ``manual/publish-rules.json``, read here (for the sources) and by
``manual/scripts/check-published.mjs`` (for the built site, after every build). A pattern list
copied into two languages would be the second source of truth that drifts.

## Why the file list is read from git, not from the disk

Published means committed. The listing is every tracked file plus every untracked file git does
not ignore - that is, everything a ``git add -A`` would publish - so a new file is caught before
it is committed, while ``node_modules/`` and ``build/`` (ignored) never count. Outside a git
checkout the walk falls back to the filesystem minus the build products.

## The table-pipe rule

A ``|`` inside a code span in a table row splits the cell in GitHub-flavoured Markdown. The page
then BUILDS SUCCESSFULLY into a broken table, so the Docusaurus build cannot catch it (measured:
``| a | `x | y` |`` builds with exit 0). It is checked here, where it can be.
"""

from __future__ import annotations

import json
import re
import subprocess
from pathlib import Path
from typing import Any

import pytest

from core.config import RUNTIME_KEYS

REPO_ROOT = Path(__file__).resolve().parents[3]
MANUAL = REPO_ROOT / "manual"
RULES_PATH = MANUAL / "publish-rules.json"
CONTRACT = REPO_ROOT / "docs" / "mcp-contract.md"

#: Never tracked (manual/.gitignore) and never part of the published source.
BUILD_PRODUCTS = ("node_modules", "build", ".docusaurus", ".cache-loader")
#: Generated at build time from docs/mcp-contract.md, and git-ignored.
GENERATED = "manual/docs/reference/mcp-tools.md"

TEXT_SUFFIXES = {".md", ".mdx", ".ts", ".mjs", ".css", ".svg", ".json"}


def _rules() -> dict[str, Any]:
    loaded: dict[str, Any] = json.loads(RULES_PATH.read_text("utf-8"))
    return loaded


def _compiled(category: str | None = None) -> list[tuple[str, re.Pattern[str], str]]:
    out = []
    for rule in _rules()["forbidden"]:
        if category is not None and rule["category"] != category:
            continue
        flags = re.IGNORECASE if "i" in rule.get("flags", "") else 0
        out.append((rule["id"], re.compile(rule["pattern"], flags), rule["why"]))
    return out


def published_files() -> list[str]:
    """Repo-relative paths under manual/ that a sync would publish."""
    try:
        listed = subprocess.run(
            ["git", "ls-files", "-z", "--cached", "--others", "--exclude-standard", "manual"],
            cwd=REPO_ROOT,
            capture_output=True,
            check=True,
        ).stdout.decode()
        paths = [p for p in listed.split("\0") if p]
    except (OSError, subprocess.CalledProcessError):
        paths = [
            p.relative_to(REPO_ROOT).as_posix()
            for p in MANUAL.rglob("*")
            if p.is_file() and not any(part in BUILD_PRODUCTS for part in p.parts)
        ]
    return sorted(p for p in paths if p != GENERATED)


def test_the_file_list_is_not_vacuous() -> None:
    files = published_files()
    # The scan below passes over nothing if the listing came back empty.
    for required in (
        "manual/docs/intro.md",
        "manual/docusaurus.config.ts",
        "manual/publish-rules.json",
    ):
        assert required in files, f"{required} missing from the published listing: {files[:5]}"


def allowed(path: str) -> bool:
    """Whether ``path`` matches an ``allowed_files`` glob, where ``*`` never crosses a ``/``.

    ``fnmatch``'s ``*`` DOES cross ``/``, so ``manual/docs/*.md`` would also admit any depth of
    nesting and the second-level glob would be decoration (found by mutation: deleting it left
    the suite green). Each glob therefore means exactly the depth it spells.
    """
    for glob in _rules()["allowed_files"]:
        pattern = "".join("[^/]*" if ch == "*" else re.escape(ch) for ch in glob)
        if re.fullmatch(pattern, path):
            return True
    return False


def test_the_allow_list_admits_what_it_names_and_nothing_else() -> None:
    assert allowed("manual/docs/intro.md")
    assert allowed("manual/docs/guides/runs.md")
    assert not allowed("manual/docs/guides/deep/nested.md")
    assert not allowed("manual/notes/internal-review.md")
    assert not allowed("manual/docs/guides/runs.txt")
    assert not allowed("manual/.env")


def test_every_published_file_is_on_the_allow_list() -> None:
    stray = [p for p in published_files() if not allowed(p)]
    assert not stray, (
        f"files under manual/ that publish-rules.json does not allow: {stray}. Everything here "
        f"is published to the public mirror and GitHub Pages; add a glob only for a kind of file "
        f"that is meant to be public."
    )


def _published_text() -> list[tuple[str, str]]:
    return [
        (p, (REPO_ROOT / p).read_text("utf-8"))
        for p in published_files()
        if Path(p).suffix in TEXT_SUFFIXES and p != "manual/publish-rules.json"
    ]


def scan(texts: list[tuple[str, str]], rules: list[tuple[str, re.Pattern[str], str]]) -> list[str]:
    found = []
    for path, text in texts:
        for rule_id, pattern, why in rules:
            match = pattern.search(text)
            if match:
                found.append(f"{path}: [{rule_id}] {match.group(0)!r} - {why}")
    return found


def test_no_published_source_carries_internal_or_secret_content() -> None:
    texts = _published_text()
    assert len(texts) >= 10, f"only {len(texts)} text files scanned"
    assert not scan(texts, _compiled()), "\n".join(scan(texts, _compiled()))


def test_the_mcp_contract_the_site_copies_carries_no_secret() -> None:
    """``docs/mcp-contract.md`` is published by the mirror and copied into the site. Its
    internal requirement numbers are rewritten by the build (and the built page is checked),
    so only the ``secret`` category applies to the source file."""
    found = scan([("docs/mcp-contract.md", CONTRACT.read_text("utf-8"))], _compiled("secret"))
    assert not found, "\n".join(found)


@pytest.mark.parametrize(
    ("rule_id", "planted"),
    [
        ("internal-host", "reach it at build.hitsai.local"),
        ("internal-host", "the tunnel mcp.hitsai.net"),
        ("private-ipv4", "ssh to 192.168.1.10"),
        ("private-ipv4", "the pod at 10.1.2.3"),
        ("gpu-uuid", "pin GPU-00000000-1111-2222-3333-444444444444"),
        # Built by concatenation so no literal shaped like a credential sits in the repository.
        ("github-token", "token " + "gh" + "p_" + "x" * 30),
        ("hf-token", "HF_TOKEN=" + "h" + "f_" + "x" * 30),
        ("planning-tree", "see 0xcc/prds for details"),
        ("planning-tree", "recorded in AGENT.md"),
        ("requirement-id", "as R-02.53 requires"),
        ("decision-id", "per the operator (D52)"),
        ("private-org", "github.com/" + "Onegai" + "shimas/miForge"),
        ("home-path", "under /home/someone/app"),
    ],
)
def test_each_rule_catches_what_it_names(rule_id: str, planted: str) -> None:
    """Negative control for the scan itself: a rule that matches nothing passes everything."""
    rules = [r for r in _compiled() if r[0] == rule_id]
    assert rules, f"no rule {rule_id!r}"
    assert scan([("planted.md", planted)], rules), f"{rule_id} did not catch {planted!r}"


@pytest.mark.parametrize(
    "allowed",
    [
        "the public site https://hitsai.net and https://hitsainet.github.io/miForge/",
        "on this machine at http://127.0.0.1:8010 and 0.0.0.0",
        "an example endpoint http://vllm.example.com:8000/v1",
        "a card's identifier, GPU-…",
        "BF16 and FP16",
    ],
)
def test_the_rules_leave_ordinary_manual_text_alone(allowed: str) -> None:
    assert not scan([("ok.md", allowed)], _compiled())


# ── the table-pipe rule ───────────────────────────────────────────────────────────────────

_CODE_SPAN = re.compile(r"`[^`\n]*`")


def table_pipes_in_code(text: str) -> list[str]:
    """Table rows whose code spans contain an unescaped ``|``."""
    bad = []
    in_fence = False
    for line in text.splitlines():
        if line.lstrip().startswith("```"):
            in_fence = not in_fence
            continue
        if in_fence or not line.lstrip().startswith("|"):
            continue
        # Pair the backticks as a table cell parser would NOT: across the whole row. A `|`
        # between a pair of backticks is a cell separator to GFM, whatever the author meant.
        for span in _CODE_SPAN.findall(line):
            if re.search(r"(?<!\\)\|", span):
                bad.append(line.strip())
                break
    return bad


def test_no_table_row_has_a_pipe_inside_a_code_span() -> None:
    docs = [p for p in published_files() if p.endswith((".md", ".mdx"))]
    assert docs
    found = {p: table_pipes_in_code((REPO_ROOT / p).read_text("utf-8")) for p in docs}
    found = {p: rows for p, rows in found.items() if rows}
    assert not found, f"a `|` inside a code span splits the table cell; escape it as \\|: {found}"


def test_the_table_pipe_check_sees_a_planted_row() -> None:
    assert table_pipes_in_code("| a | `x | y` |\n")
    assert not table_pipes_in_code("| a | `x \\| y` |\n")
    assert not table_pipes_in_code("```\n| a | `x | y` |\n```\n")


# ── the settings page names exactly the runtime-writable keys ─────────────────────────────


def test_the_settings_page_lists_exactly_the_runtime_writable_keys() -> None:
    """The manual's runtime table is the operator's answer to "what can I change live"; a key
    added to (or removed from) ``RUNTIME_KEYS`` without the page is a wrong answer in public."""
    text = (MANUAL / "docs" / "operators" / "settings.md").read_text("utf-8")
    section = text.split("Only these keys are runtime-writable:", 1)[1].split("Anything else", 1)[0]
    listed = set(re.findall(r"^\| `([a-z0-9_]+)` \|", section, flags=re.MULTILINE))
    assert listed == set(RUNTIME_KEYS), (
        f"missing from the page: {sorted(set(RUNTIME_KEYS) - listed)}; "
        f"on the page but not runtime-writable: {sorted(listed - set(RUNTIME_KEYS))}"
    )
