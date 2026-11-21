"""FR-016.14 (D14): no Kubernetes permission anywhere in miForge (FTDD §8.3; FTASKS 4.10).

- no ``Role``, ``ClusterRole``, ``RoleBinding``, ``ClusterRoleBinding`` or ``ServiceAccount`` object
  in
  any manifest under ``gitops/`` (the rendered set and everything applied by hand or by overlays);
- ``automountServiceAccountToken: false`` on the worker, the gateway, the MCP pod and the backend;
- no module under ``backend/src`` imports ``kubernetes``.
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest
import yaml

from tests.support.wsp_manifests import GITOPS, one, rendered

REPO = Path(__file__).resolve().parents[4]
SRC = REPO / "backend" / "src"
FORBIDDEN_KINDS = {"Role", "ClusterRole", "RoleBinding", "ClusterRoleBinding", "ServiceAccount"}
#: FTDD §8.3's pods. The gateway joins when its manifest lands (FTASKS 6.5).
NO_TOKEN = ("miforge-wsp-worker", "miforge-mcp", "miforge-backend")


def every_gitops_document() -> list[tuple[str, dict]]:
    found = []
    for path in sorted((REPO / "gitops").rglob("*.y*ml")):
        for doc in yaml.safe_load_all(path.read_text()):
            if isinstance(doc, dict):
                found.append((str(path.relative_to(REPO)), doc))
    assert len(found) > 10, "the walk read almost nothing"
    return found


def test_no_rbac_or_service_account_object() -> None:
    bad = [
        (p, d.get("kind")) for p, d in every_gitops_document() if d.get("kind") in FORBIDDEN_KINDS
    ]
    assert not bad, bad


@pytest.mark.parametrize("name", NO_TOKEN)
def test_no_service_account_token(name: str) -> None:
    spec = one("Deployment", name)["spec"]["template"]["spec"]
    assert spec.get("automountServiceAccountToken") is False, name
    assert "serviceAccountName" not in spec


def test_no_module_imports_kubernetes() -> None:
    files = sorted(SRC.rglob("*.py"))
    assert len(files) > 100
    bad = []
    for path in files:
        for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
            names = []
            if isinstance(node, ast.Import):
                names = [a.name for a in node.names]
            elif isinstance(node, ast.ImportFrom) and node.module and not node.level:
                names = [node.module]
            bad.extend(
                f"{path.relative_to(SRC)}:{node.lineno}"
                for n in names
                if n.split(".")[0] == "kubernetes"
            )
    assert not bad, bad


def test_the_walk_covers_the_rendered_set() -> None:
    assert GITOPS.is_dir() and len(rendered()) > 5
