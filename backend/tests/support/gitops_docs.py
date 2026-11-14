"""Every YAML document under ``gitops/``, including the ones embedded in kustomize patches.

Feature 015 (FTID §3.9). ``storage.yaml`` is deliberately outside the kustomization (it is
applied by hand), and the hostIPC component carries its pod changes as ``patches[].patch``
strings, so a walker that read only what kustomize builds, or only top-level documents, would
miss both. Each :class:`Doc` says where it came from.
"""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

REPO_ROOT = Path(__file__).resolve().parents[3]
GITOPS = REPO_ROOT / "gitops"


@dataclass(frozen=True)
class Doc:
    path: Path
    index: int
    data: Any
    #: ``None`` for a top-level document; else the Kustomization/Component it is a patch of.
    embedded_in: str | None = None


def walk(root: Path = GITOPS) -> Iterator[Doc]:
    files = sorted({*root.rglob("*.yaml"), *root.rglob("*.yml")})
    for path in files:
        for index, data in enumerate(yaml.safe_load_all(path.read_text("utf-8"))):
            if data is None:
                continue
            yield Doc(path, index, data)
            if isinstance(data, dict) and data.get("kind") in ("Kustomization", "Component"):
                for number, patch in enumerate(data.get("patches") or []):
                    text = patch.get("patch") if isinstance(patch, dict) else None
                    if isinstance(text, str):
                        for sub, body in enumerate(yaml.safe_load_all(text)):
                            if body is not None:
                                yield Doc(
                                    path, number * 100 + sub, body, f"{path.name}#patch{number}"
                                )


def find(root: Path = GITOPS, *, kind: str, name: str) -> Doc:
    """The one top-level document of ``kind`` named ``name``; fails loudly otherwise."""
    found = [
        d
        for d in walk(root)
        if d.embedded_in is None
        and isinstance(d.data, dict)
        and d.data.get("kind") == kind
        and (d.data.get("metadata") or {}).get("name") == name
    ]
    if len(found) != 1:
        raise LookupError(f"expected one {kind} {name} under {root}, found {len(found)}")
    return found[0]


def host_paths(data: Any, trail: str = "") -> Iterator[tuple[str, dict[str, Any]]]:
    """Every mapping under a ``hostPath`` key, with a readable trail to it."""
    if isinstance(data, dict):
        for key, value in data.items():
            here = f"{trail}.{key}" if trail else str(key)
            if key == "hostPath" and isinstance(value, dict):
                yield here, value
            else:
                yield from host_paths(value, here)
    elif isinstance(data, list):
        for index, item in enumerate(data):
            yield from host_paths(item, f"{trail}[{index}]")
