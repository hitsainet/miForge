"""Every Node version the repository names agrees, and it is the suite's Node 22 LTS (D34).

Node 20 reached end-of-life on 2026-04-30; the suite moved to Node 22 on 2026-10-17. CI
(``actions/setup-node``), the image's builder stage (``FROM node:``) and the dev compose
service (``image: node:``) must use the same major, or the published image is built by a
toolchain CI never tested. The references are found by scanning the files, not by a hand-kept
list, so a new workflow, Dockerfile or compose service is covered the day it lands.
"""

from __future__ import annotations

import re
import subprocess
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
EXPECTED_MAJOR = "22"

NODE_VERSION = re.compile(r"""^\s*node-version:\s*['"]?([^'"\s#]+)""", re.MULTILINE)
SETUP_NODE = re.compile(r"uses:\s*actions/setup-node@")
FROM_NODE = re.compile(r"^\s*FROM\s+(?:--\S+\s+)*node:(\S+)", re.MULTILINE | re.IGNORECASE)
IMAGE_NODE = re.compile(r"""^\s*image:\s*['"]?node:([^'"\s#]+)""", re.MULTILINE)


def _tracked(*patterns: str) -> list[Path]:
    out = subprocess.run(
        ["git", "-C", str(REPO_ROOT), "ls-files", "--", *patterns],
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    return [REPO_ROOT / line for line in out.splitlines() if line]


def _major(spec: str) -> str:
    match = re.match(r"v?(\d+)", spec)
    assert match, f"unparseable Node version {spec!r}"
    return match.group(1)


def node_references() -> list[tuple[str, str]]:
    """(file, version spec) for every Node version named in workflows, Dockerfiles, compose
    files and ``.nvmrc``/``.node-version``."""
    refs: list[tuple[str, str]] = []
    for wf in _tracked(".github/workflows/*.yml", ".github/workflows/*.yaml"):
        text = wf.read_text()
        versions = NODE_VERSION.findall(text)
        # A setup-node step without node-version silently takes the runner's default Node.
        assert len(versions) >= len(
            SETUP_NODE.findall(text)
        ), f"{wf.relative_to(REPO_ROOT)}: a setup-node step has no node-version"
        refs += [(str(wf.relative_to(REPO_ROOT)), v) for v in versions]
    for df in _tracked("*Dockerfile*"):
        refs += [(str(df.relative_to(REPO_ROOT)), v) for v in FROM_NODE.findall(df.read_text())]
    for cf in _tracked("*compose*.yml", "*compose*.yaml"):
        refs += [(str(cf.relative_to(REPO_ROOT)), v) for v in IMAGE_NODE.findall(cf.read_text())]
    for rc in _tracked("*.nvmrc", "*.node-version"):
        refs.append((str(rc.relative_to(REPO_ROOT)), rc.read_text().strip()))
    return refs


def test_node_references_are_found() -> None:
    # CI, the image and the dev compose service must all be seen, or the next test is vacuous.
    where = {w for w, _ in node_references()}
    assert any(w.startswith(".github/workflows/") for w in where), where
    assert "frontend/Dockerfile" in where, where
    assert "docker-compose.yml" in where, where


def test_every_node_reference_is_node_22() -> None:
    refs = node_references()
    wrong = [(w, v) for w, v in refs if _major(v) != EXPECTED_MAJOR]
    assert not wrong, f"Node references not on {EXPECTED_MAJOR}: {wrong} (all: {refs})"
