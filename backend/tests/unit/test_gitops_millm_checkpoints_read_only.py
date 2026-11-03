"""D47: miLLM's checkpoint directory is mounted READ-ONLY and named as a read root.

Found live in F11 14.5 (2026-10-18): miLLM writes an attached run's checkpoints under its
own directory, miForge searched only its own root, and ``continue`` answered
``checkpoint_not_found`` for a run with two saves. The volume points at ANOTHER
APPLICATION'S DATA, so each property below is a safety property, not a style one: read-only
everywhere, never in the init container that chowns volume roots, Retain, and never
``DirectoryOrCreate``.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

yaml = pytest.importorskip("yaml")

PILOT = Path(__file__).resolve().parents[3] / "gitops" / "pilot"
MOUNT = "/data/millm-checkpoints"
HOST_DIR = "/data/millm/data/checkpoints"


def _docs(name: str) -> list[dict[str, Any]]:
    return [d for d in yaml.safe_load_all((PILOT / name).read_text("utf-8")) if d]


def _backend_pod() -> dict[str, Any]:
    deployment = next(d for d in _docs("backend.yaml") if d.get("kind") == "Deployment")
    return deployment["spec"]["template"]["spec"]


def test_every_container_that_reads_checkpoints_mounts_millms_read_only() -> None:
    pod = _backend_pod()
    readers = [
        c
        for c in pod["containers"]
        if any(m["mountPath"] == "/data/checkpoints" for m in c.get("volumeMounts", []))
    ]
    assert len(readers) >= 5, [c["name"] for c in readers]
    for container in readers:
        mounts = {m["name"]: m for m in container["volumeMounts"]}
        mount = mounts.get("millm-checkpoints")
        assert mount is not None, f"{container['name']} cannot read miLLM's checkpoints"
        assert mount["mountPath"] == MOUNT
        assert mount.get("readOnly") is True, f"{container['name']} mounts it writable"


def test_the_init_container_that_chowns_volume_roots_never_sees_it() -> None:
    for init in _backend_pod().get("initContainers", []):
        names = {m["name"] for m in init.get("volumeMounts", [])}
        assert "millm-checkpoints" not in names, init["name"]


def test_the_volume_is_a_read_only_claim_on_a_retained_existing_directory() -> None:
    volume = next(v for v in _backend_pod()["volumes"] if v["name"] == "millm-checkpoints")
    claim_name = volume["persistentVolumeClaim"]["claimName"]
    assert volume["persistentVolumeClaim"].get("readOnly") is True
    storage = _docs("storage.yaml")
    claim = next(
        d
        for d in storage
        if d.get("kind") == "PersistentVolumeClaim" and d["metadata"]["name"] == claim_name
    )
    pv = next(
        d
        for d in storage
        if d.get("kind") == "PersistentVolume"
        and d["metadata"]["name"] == claim["spec"]["volumeName"]
    )
    assert pv["spec"]["hostPath"] == {"path": HOST_DIR, "type": "Directory"}
    assert pv["spec"]["persistentVolumeReclaimPolicy"] == "Retain"
    assert pv["spec"]["accessModes"] == ["ReadOnlyMany"]


def test_the_mount_is_named_as_a_checkpoint_read_root() -> None:
    from core.config import Settings

    config = next(d for d in _docs("config.yaml") if d.get("kind") == "ConfigMap")
    value = config["data"]["CHECKPOINT_READ_ROOTS"]
    parsed = Settings(_env_file=None, checkpoint_read_roots=value).checkpoint_read_root_paths  # type: ignore[call-arg]
    assert parsed == [Path(MOUNT)]
