"""Every image the kustomization deploys is tracked by Image Updater (feature 016, 2026-11-17).

The WSP worker shipped in 016-P1b with a kustomize `images:` entry but no Image Updater entry, so a
new worker image would never be pinned and the Deployment would keep whatever `:latest` it first
pulled (`imagePullPolicy: IfNotPresent`). The two lists are read from the manifests themselves.
"""

from __future__ import annotations

from pathlib import Path

import yaml

PILOT = Path(__file__).resolve().parents[3] / "gitops" / "pilot"
PREFIX = "argocd-image-updater.argoproj.io/"


def kustomize_images() -> set[str]:
    data = yaml.safe_load((PILOT / "kustomization.yaml").read_text(encoding="utf-8"))
    return {entry["name"] for entry in data["images"]}


def app_annotations() -> dict[str, str]:
    docs = yaml.safe_load_all((PILOT / "argocd" / "miforge-app.yaml").read_text(encoding="utf-8"))
    apps = [d for d in docs if d and d.get("kind") == "Application"]
    assert len(apps) == 1
    return apps[0]["metadata"]["annotations"]


def tracked() -> dict[str, str]:
    """alias -> kustomize image name, for every alias in the image list."""
    notes = app_annotations()
    aliases = [item.split("=", 1)[0].strip() for item in notes[PREFIX + "image-list"].split(",")]
    return {alias: notes.get(f"{PREFIX}{alias}.kustomize.image-name", "") for alias in aliases}


def test_every_deployed_image_is_tracked_by_digest() -> None:
    images = kustomize_images()
    assert {"hitsai/miforge-backend", "hitsai/miforge-wsp-worker"} <= images, "precondition"
    names = tracked()
    assert set(names.values()) == images, f"kustomize images {images} vs tracked {names}"
    notes = app_annotations()
    for alias in names:
        assert notes.get(f"{PREFIX}{alias}.update-strategy") == "digest", alias
