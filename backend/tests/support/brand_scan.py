"""The brand guard's scanner: pure functions over strings (feature 015; FTID §3.8, FR-015.10).

miLLM becomes miServer, miStudio becomes miWorkbench, and "miSuite" becomes "the miStudio suite"
(suite R1-R22). Every surface a person or an agent reads is scanned for the old names. What may
legitimately keep an old name is never a list in this file - it comes from the census export
(``docs/legacy-identifiers.md``) and from ``core/legacy_names.py``:

a. each non-retired register entry's ``pattern`` (applied to the source segment, quotes
   included, when there is one);
b. each entry's ``literals`` as substrings (a literal that is a bare old word masks only an
   item that IS that word);
c. the old environment names that still work (``LEGACY_ENV_ALIASES``) AND whose ``env`` record
   is ``transition`` - both authorities must agree, as the Vitest half reads the export alone
   (P7: a fallback reinstated in code against a ``retired`` record allows nothing);
d. retired environment names, only inside ``core/legacy_names.py``;
e. old repository names whose ``repo`` record is ``transition`` (CF-13);
f. the export's active ``allow`` guards (CF-15, CF-22).

Masked spans become spaces; then the hand-kept token and every active ``deny`` guard run over
what is left. A finding names every rule that hit (CF-23). The guard reports until the export
header's ``closed`` reaches P4 (CF-21: ``brand-guard.json``); then it enforces in the private
repository and reports on the public mirror (suite R23; see :func:`mode`).
"""

from __future__ import annotations

import ast
import json
import os
import re
import subprocess
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

from tests.support import legacy_identifiers as li
from tests.support.mirror_snapshot import mirror_snapshot

REPO_ROOT = li.REPO_ROOT
#: The hand-kept token (PD-9: miForge denies "miSuite" from P0, not only from P5).
OLD = re.compile(r"(?i)millm|mistudio|misuite")
#: CF-20: the export names every pattern by design and never counts against the guard.
#: BOTH forms, measured on git 2.43 (2026-11-08): with a positive pathspec that shares the
#: ``docs/`` prefix (``docs/**``), ``:(exclude,glob)**/docs/...`` does NOT exclude the ROOT-level
#: ``docs/legacy-identifiers.md``, while it does with no positive pathspec or several. The
#: root-anchored form covers that case; ``test_export_never_counts`` runs both shapes.
EXPORT_EXCLUSIONS = (
    ":(exclude,glob)**/docs/legacy-identifiers*.md",
    ":(exclude,glob)docs/legacy-identifiers*.md",
)
LEGACY_MODULE = "backend/src/core/legacy_names.py"
CONFIG = Path("brand-guard.json")
#: Set to "1" to enforce regardless of ``brand-guard.json`` (IPD-8; both halves honour it).
ENFORCE_ENV = "MIFORGE_BRAND_GUARD_ENFORCE"


@dataclass(frozen=True)
class Item:
    surface: str
    where: str
    text: str
    segment: str | None = None


@dataclass(frozen=True)
class Finding:
    item: Item
    token: str
    rules: tuple[str, ...]

    def line(self) -> str:
        return f"{self.item.where}: {self.token} [{', '.join(self.rules)}]"


# ------------------------------------------------------------------------------- discovery


def discover(pathspecs: list[str], root: Path = REPO_ROOT) -> list[str]:
    """Tracked files matching ``pathspecs``, minus the census export (CF-20)."""
    out = subprocess.run(
        ["git", "ls-files", "-z", "--", *pathspecs, *EXPORT_EXCLUSIONS],
        cwd=root,
        capture_output=True,
        check=True,
    ).stdout.decode()
    return sorted(p for p in out.split("\0") if p)


def _docstring_ids(tree: ast.AST) -> set[int]:
    ids: set[int] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Module | ast.ClassDef | ast.FunctionDef | ast.AsyncFunctionDef):
            body = node.body
            if body and isinstance(body[0], ast.Expr) and isinstance(body[0].value, ast.Constant):
                ids.add(id(body[0].value))
        if isinstance(node, ast.Expr) and isinstance(node.value, ast.Constant):
            ids.add(id(node.value))  # a bare string statement is a comment, not a surface
    return ids


def python_items(source: str, where: str) -> list[Item]:
    """Every non-docstring string constant, f-string parts included, with its source segment."""
    tree = ast.parse(source)
    skip = _docstring_ids(tree)
    items = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Constant) and isinstance(node.value, str) and id(node) not in skip:
            segment = ast.get_source_segment(source, node)
            items.append(Item("python", f"{where}:{node.lineno}", node.value, segment))
    return items


def text_items(
    text: str, where: str, surface: str, *, strip_html_comments: bool = False
) -> list[Item]:
    if strip_html_comments:
        text = re.sub(r"<!--.*?-->", lambda m: "\n" * m.group(0).count("\n"), text, flags=re.S)
    return [
        Item(surface, f"{where}:{n}", line) for n, line in enumerate(text.splitlines(), 1) if line
    ]


def yaml_items(data: Any, where: str) -> list[Item]:
    """Keys and scalar values of a YAML document (multi-line scalars and patch strings too)."""
    items: list[Item] = []

    def visit(node: Any, trail: str) -> None:
        if isinstance(node, dict):
            for key, value in node.items():
                here = f"{trail}.{key}" if trail else str(key)
                items.append(Item("yaml", f"{where} {here} (key)", str(key)))
                visit(value, here)
        elif isinstance(node, list):
            for index, value in enumerate(node):
                visit(value, f"{trail}[{index}]")
        elif isinstance(node, str):
            items.append(Item("yaml", f"{where} {trail}", node))

    visit(data, "")
    return items


def registry_items(mcp: Any) -> tuple[list[Item], int]:
    """The live MCP registry: server instructions, each tool's description, and every
    ``description`` in each tool's input schema. Returns the items and the tool count."""
    import asyncio

    tools = asyncio.run(mcp.list_tools())
    items = [Item("mcp", "mcp instructions", str(getattr(mcp, "instructions", "") or ""))]

    def walk(node: Any, trail: str) -> None:
        if isinstance(node, dict):
            if isinstance(node.get("description"), str):
                items.append(Item("mcp", trail, node["description"]))
            for key, value in node.items():
                walk(value, f"{trail}.{key}")
        elif isinstance(node, list):
            for index, value in enumerate(node):
                walk(value, f"{trail}[{index}]")

    for tool in tools:
        items.append(Item("mcp", f"mcp tool {tool.name}", tool.description or ""))
        walk(tool.input_schema, f"mcp tool {tool.name} schema")
    return items, len(tools)


def build_registry() -> tuple[list[Item], int]:
    """The real server (``build_server``, not the test harness: only it has instructions)."""
    from mcp_server.config import MCPSettings
    from mcp_server.server import build_server

    mcp, _ = build_server(MCPSettings(auth_token="x" * 32))
    return registry_items(mcp)


# ---------------------------------------------------------------------------------- masking


@dataclass(frozen=True)
class Rules:
    masks: tuple[re.Pattern[str], ...]
    literals: tuple[str, ...]
    bare_literals: frozenset[str]
    substrings: tuple[str, ...]
    retired: tuple[str, ...]
    deny: tuple[tuple[str, re.Pattern[str]], ...]


def rules_for(
    export: li.Export,
    *,
    legacy_aliases: frozenset[str],
    retired: frozenset[str],
    phase: str | None = None,
) -> Rules:
    entries = [e for e in export.entries if e["status"] != "retired"]
    literals = [lit for e in entries for lit in e["literals"]]
    bare = frozenset(lit for lit in literals if OLD.fullmatch(lit))
    repo_names = [r["old"] for r in export.repo if r["status"] == "transition"]
    transition_env = {r["old"] for r in export.env if r["status"] == "transition"}
    aliases = sorted(set(legacy_aliases) & transition_env)
    active = li.active_guards(export, phase)
    masks = [re.compile(e["pattern"]) for e in entries]
    masks += [re.compile(g["pattern"]) for g in active if g["effect"] == "allow"]
    deny = tuple((g["id"], re.compile(g["pattern"])) for g in active if g["effect"] == "deny")
    return Rules(
        masks=tuple(masks),
        literals=tuple(lit for lit in literals if lit not in bare),
        bare_literals=bare,
        substrings=tuple(aliases + repo_names),
        retired=tuple(sorted(retired)),
        deny=deny,
    )


def _blank(text: str, start: int, end: int) -> str:
    return text[:start] + " " * (end - start) + text[end:]


def masked(item: Item, rules: Rules) -> str:
    text = item.segment if item.segment is not None else item.text
    if item.text in rules.bare_literals:
        return " " * len(text)
    for pattern in rules.masks:
        for match in list(pattern.finditer(text)):
            text = _blank(text, match.start(), match.end())
    names = list(rules.literals) + list(rules.substrings)
    if item.where.startswith(LEGACY_MODULE):
        names += list(rules.retired)
    for name in names:
        start = text.find(name)
        while start != -1:
            text = _blank(text, start, start + len(name))
            start = text.find(name, start + len(name))
    return text


def scan(items: list[Item], rules: Rules) -> list[Finding]:
    findings: list[Finding] = []
    for item in items:
        text = masked(item, rules)
        deny_hits = [(gid, m.span()) for gid, pat in rules.deny for m in pat.finditer(text)]
        covered: set[int] = set()
        for match in OLD.finditer(text):
            ids = [gid for gid, (s, e) in deny_hits if s < match.end() and match.start() < e]
            for index, (_, span) in enumerate(deny_hits):
                if span[0] < match.end() and match.start() < span[1]:
                    covered.add(index)
            findings.append(
                Finding(item, match.group(0), (f"token:{match.group(0).lower()}", *ids))
            )
        for index, (gid, (s, e)) in enumerate(deny_hits):
            if index not in covered:
                findings.append(Finding(item, text[s:e], (gid,)))
    return findings


# ------------------------------------------------------------------------------------- mode


def expected_enforced(export: li.Export) -> bool:
    """CF-21: enforcement keys to the last CLOSED phase, never to the rules in force."""
    return li.phase_index(export.closed) >= li.phase_index("P4")


def configured_enforced(root: Path = REPO_ROOT) -> bool:
    return bool(json.loads((root / CONFIG).read_text("utf-8"))["enforced"])


def mode(
    root: Path = REPO_ROOT, environ: Mapping[str, str] | None = None
) -> Literal["report", "enforced"]:
    """Whether a finding FAILS (``enforced``) or is printed (``report``).

    - ``MIFORGE_BRAND_GUARD_ENFORCE=1`` enforces whatever the file says: the local proof of
      zero findings before the flip (015 FTID §9.5, IPD-8).
    - ``brand-guard.json`` ``{"enforced": false}`` reports (until P4 closes, CF-21).
    - Once it is ``true``: **enforced in the private repository, report on the public mirror**
      (suite R23, operator, 2026-11-11). The mirror carries only the public export, which
      withholds the private repository names and the old host names, so a finding there cannot
      be judged; the private CI is the enforcement point, because a commit is published only
      after it passes. "The mirror" is the sync's commit shape (``mirror_snapshot``), never a
      missing file: a private tree that lost its full export enforces against the public one
      and fails, which is the safe direction.
    """
    env = os.environ if environ is None else environ
    if env.get(ENFORCE_ENV) == "1":
        return "enforced"
    if not configured_enforced(root):
        return "report"
    if mirror_snapshot(root):
        return "report"
    return "enforced"


def default_rules(export: li.Export | None = None, phase: str | None = None) -> Rules:
    from core import legacy_names

    return rules_for(
        export or li.load(),
        legacy_aliases=legacy_names.old_names(),
        retired=legacy_names.retired_names(),
        phase=phase,
    )
