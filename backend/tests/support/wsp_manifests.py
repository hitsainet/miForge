"""Read the WSP manifests the way ArgoCD applies them: every document of every file that
``gitops/pilot/kustomization.yaml`` lists under ``resources`` (no kustomize binary here; the
kustomization uses no patches, so the listed documents ARE the rendered set, image tags aside).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

GITOPS = Path(__file__).resolve().parents[3] / "gitops" / "pilot"


def kustomization() -> dict[str, Any]:
    loaded: dict[str, Any] = yaml.safe_load((GITOPS / "kustomization.yaml").read_text())
    return loaded


def rendered() -> list[dict[str, Any]]:
    docs: list[dict[str, Any]] = []
    for name in kustomization()["resources"]:
        docs.extend(d for d in yaml.safe_load_all((GITOPS / name).read_text()) if d)
    assert docs, "the kustomization renders nothing"
    return docs


def documents(filename: str) -> list[dict[str, Any]]:
    return [d for d in yaml.safe_load_all((GITOPS / filename).read_text()) if d]


def one(kind: str, name: str, docs: list[dict[str, Any]] | None = None) -> dict[str, Any]:
    found = [
        d
        for d in (docs if docs is not None else rendered())
        if d["kind"] == kind and d["metadata"]["name"] == name
    ]
    assert len(found) == 1, f"expected one {kind} {name}, found {len(found)}"
    return found[0]


def pod_specs(docs: list[dict[str, Any]]) -> list[tuple[str, dict[str, Any]]]:
    """``(name, pod spec)`` for every workload in ``docs``."""
    out = []
    for d in docs:
        if d["kind"] in ("Deployment", "StatefulSet", "DaemonSet", "Job"):
            out.append((d["metadata"]["name"], d["spec"]["template"]["spec"]))
        elif d["kind"] == "CronJob":
            out.append(
                (d["metadata"]["name"], d["spec"]["jobTemplate"]["spec"]["template"]["spec"])
            )
        elif d["kind"] == "Pod":
            out.append((d["metadata"]["name"], d["spec"]))
    return out
