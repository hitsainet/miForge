"""uid 10002 is the WSP worker's alone, across the suite (FR-016.10's AC; R3A-3, P60; FTASKS 4.4b).

The child's ``RLIMIT_NPROC`` is counted by the kernel **per real uid across the node**, so it caps
the worker only while no other pod runs as 10002: a second app on that uid would share (and
could exhaust or be exhausted by) the worker's process budget. This test reads every Kubernetes
and compose manifest of miForge and of each sibling (found by ``tests/support/siblings.py``) and
requires that only ``gitops/pilot/wsp-worker.yaml`` sets ``runAsUser`` or ``runAsGroup`` 10002.

An absent sibling skips by name; under ``MIFORGE_REQUIRE_CROSS_REPO_CHECKS=1`` it fails.
"""

from __future__ import annotations

import os
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest
import yaml

from tests.support import siblings

UID = 10002
OWNER = Path("gitops/pilot/wsp-worker.yaml")
SKIP_DIRS = {".git", "node_modules", ".venv", "venv", "__pycache__", ".mypy_cache", "dist", "build"}
KEYS = ("runAsUser", "runAsGroup", "fsGroup")


def _manifests(repo: Path) -> Iterator[Path]:
    for dirpath, dirnames, filenames in os.walk(repo):
        dirnames[:] = [d for d in dirnames if d not in SKIP_DIRS]
        for name in filenames:
            if name.endswith((".yaml", ".yml")):
                yield Path(dirpath, name)


def _hits(node: Any) -> Iterator[str]:
    if isinstance(node, dict):
        for key, value in node.items():
            if key in KEYS and str(value).strip('"') == str(UID):
                yield f"{key}: {value}"
            elif key == "user" and str(value).split(":")[0].strip('"') == str(UID):
                yield f"user: {value}"  # docker compose
            else:
                yield from _hits(value)
    elif isinstance(node, list):
        for item in node:
            yield from _hits(item)


def users_of_10002(repo: Path) -> list[str]:
    found: list[str] = []
    for path in _manifests(repo):
        rel = path.relative_to(repo)
        text = path.read_text(encoding="utf-8", errors="replace")
        if str(UID) not in text:
            continue
        try:
            docs = list(yaml.safe_load_all(text))
        except yaml.YAMLError:
            # A templated file yaml cannot read: fall back to the text, so it cannot hide a uid.
            found += [
                f"{rel}: {ln.strip()}"
                for ln in text.splitlines()
                if str(UID) in ln and any(k in ln for k in (*KEYS, "user:"))
            ]
            continue
        found += [f"{rel}: {hit}" for doc in docs for hit in _hits(doc)]
    return found


def test_only_the_wsp_worker_manifest_uses_uid_10002_in_miforge() -> None:
    hits = users_of_10002(siblings.REPO_ROOT)
    assert hits, "the worker manifest itself must set uid 10002 (the scan found nothing)"
    assert all(h.startswith(f"{OWNER}:") for h in hits), hits


@pytest.mark.parametrize(
    "sibling", [siblings.MISERVER, siblings.MIDATAWORKS, siblings.MIWORKBENCH], ids=lambda s: s.key
)
def test_no_sibling_manifest_uses_uid_10002(sibling: siblings.Sibling) -> None:
    located = siblings.locate(sibling)
    if not located.path.is_dir():
        siblings.absent(
            siblings.not_checked_out(sibling, located, "uid 10002's uniqueness is unchecked there")
        )
    assert users_of_10002(located.path) == []


def test_the_scan_finds_a_planted_user_in_yaml_compose_and_templated_text(tmp_path: Path) -> None:
    """The scanner cannot fail open: each form a manifest can take is found."""
    (tmp_path / "k8s").mkdir()
    (tmp_path / "k8s" / "a.yaml").write_text(
        "spec:\n  template:\n    spec:\n      securityContext:\n        runAsGroup: 10002\n"
    )
    (tmp_path / "compose.yml").write_text('services:\n  x:\n    user: "10002:10002"\n')
    (tmp_path / "chart.yaml").write_text("{{ if .x }}\nrunAsUser: 10002\n{{ end }}\n: [\n")
    (tmp_path / "node_modules").mkdir()
    (tmp_path / "node_modules" / "ignored.yaml").write_text("runAsUser: 10002\n")
    hits = users_of_10002(tmp_path)
    assert sorted(h.split(":")[0] for h in hits) == ["chart.yaml", "compose.yml", "k8s/a.yaml"]
