"""The manual links to miWorkbench's manual, not miStudio's old GitHub Pages site (FR-015.13).

miStudio's manual moves to ``docs-miworkbench.hitsai.net`` (suite BR-040). After this move the
old address survives only in history, which is what lets the register mark DR-29 history-only
for miForge (census ``FLAGS[("miForge", "DR-29")]``).
"""

from __future__ import annotations

import subprocess
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
#: Built by concatenation: the whole old address in this file would itself be a live DR-29
#: hit, which test_frozen_identifiers.py refuses once DR-29 is history-only.
OLD = "hitsainet.github.io/" + "miStudio"
NEW = "docs-miworkbench.hitsai.net"


def _manual_files() -> list[str]:
    out = subprocess.run(
        ["git", "ls-files", "-z", "--", "manual"], cwd=REPO_ROOT, capture_output=True, check=True
    ).stdout.decode()
    return [p for p in out.split("\0") if p]


def test_no_old_manual_link() -> None:
    files = _manual_files()
    assert len(files) > 10, "git ls-files found no manual; the scan would be vacuous"
    hits = []
    for rel in files:
        try:
            text = (REPO_ROOT / rel).read_text("utf-8")
        except (UnicodeDecodeError, FileNotFoundError):
            continue
        hits += [f"{rel}:{n}" for n, line in enumerate(text.splitlines(), 1) if OLD in line]
    assert not hits, f"links to the old miStudio manual remain: {hits}"
    config = (REPO_ROOT / "manual" / "docusaurus.config.ts").read_text("utf-8")
    assert f"https://{NEW}/" in config, "the footer no longer links miWorkbench's manual"
