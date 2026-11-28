"""D47: miServer's checkpoint directory is mounted READ-ONLY and named as a read root.

Found live in F11 14.5 (2026-10-18): miLLM (now miServer) writes an attached run's checkpoints
under its own directory, miForge searched only its own root, and ``continue`` answered
``checkpoint_not_found`` for a run with two saves. The volume points at ANOTHER
APPLICATION'S DATA, so each property below is a safety property, not a style one: read-only
everywhere, never in the init container that chowns volume roots, Retain, and never
``DirectoryOrCreate``.

ENH-009 P2 (015 FTID §9.4): miServer's window moves ``/data/millm`` to ``/data/miserver``, so the
whole chain is renamed in one change - host path, PV, claim, volume, mount and read root.
Every name is asserted, not only the path: a half-renamed chain (new path, old claim) would
leave the old PV bound to a folder that no longer exists.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

yaml = pytest.importorskip("yaml")

PILOT = Path(__file__).resolve().parents[3] / "gitops" / "pilot"
MOUNT = "/data/miserver-checkpoints"
HOST_DIR = "/data/miserver/data/checkpoints"
VOLUME = "miserver-checkpoints"
CLAIM = "miforge-miserver-checkpoints"
PV = "miforge-miserver-checkpoints-pv"


def _docs(name: str) -> list[dict[str, Any]]:
    return [d for d in yaml.safe_load_all((PILOT / name).read_text("utf-8")) if d]


def _backend_pod() -> dict[str, Any]:
    deployment = next(d for d in _docs("backend.yaml") if d.get("kind") == "Deployment")
    return deployment["spec"]["template"]["spec"]


def test_every_container_that_reads_checkpoints_mounts_miservers_read_only() -> None:
    pod = _backend_pod()
    readers = [
        c
        for c in pod["containers"]
        if any(m["mountPath"] == "/data/checkpoints" for m in c.get("volumeMounts", []))
    ]
    assert len(readers) >= 5, [c["name"] for c in readers]
    for container in readers:
        mounts = {m["name"]: m for m in container["volumeMounts"]}
        mount = mounts.get(VOLUME)
        assert mount is not None, f"{container['name']} cannot read miServer's checkpoints"
        assert mount["mountPath"] == MOUNT
        assert mount.get("readOnly") is True, f"{container['name']} mounts it writable"


def test_the_init_container_that_chowns_volume_roots_never_sees_it() -> None:
    for init in _backend_pod().get("initContainers", []):
        names = {m["name"] for m in init.get("volumeMounts", [])}
        assert VOLUME not in names, init["name"]


def test_the_volume_is_a_read_only_claim_on_a_retained_existing_directory() -> None:
    volume = next(v for v in _backend_pod()["volumes"] if v["name"] == VOLUME)
    claim_name = volume["persistentVolumeClaim"]["claimName"]
    assert claim_name == CLAIM
    assert volume["persistentVolumeClaim"].get("readOnly") is True
    storage = _docs("storage.yaml")
    claim = next(
        d
        for d in storage
        if d.get("kind") == "PersistentVolumeClaim" and d["metadata"]["name"] == claim_name
    )
    assert claim["spec"]["volumeName"] == PV
    assert claim["spec"]["accessModes"] == ["ReadOnlyMany"]
    pv = next(
        d
        for d in storage
        if d.get("kind") == "PersistentVolume"
        and d["metadata"]["name"] == claim["spec"]["volumeName"]
    )
    assert pv["spec"]["hostPath"] == {"path": HOST_DIR, "type": "Directory"}
    assert pv["spec"]["persistentVolumeReclaimPolicy"] == "Retain"
    assert pv["spec"]["accessModes"] == ["ReadOnlyMany"]
    assert pv["metadata"]["labels"]["app.kubernetes.io/part-of"] == "miforge"


def test_the_old_chain_is_gone() -> None:
    """No PV, claim, volume or mount of the pre-P2 chain survives the rename: an old PV left
    in ``storage.yaml`` would be re-applied onto ``/data/millm/data/checkpoints``, which no
    longer exists after miServer's move (``type: Directory`` keeps the pod Pending)."""
    storage = {
        (d.get("kind"), d["metadata"]["name"]) for d in _docs("storage.yaml") if "metadata" in d
    }
    assert ("PersistentVolume", "miforge-millm-checkpoints-pv") not in storage
    assert ("PersistentVolumeClaim", "miforge-millm-checkpoints") not in storage
    pod = _backend_pod()
    assert "millm-checkpoints" not in {v["name"] for v in pod["volumes"]}
    for container in [*pod["containers"], *pod.get("initContainers", [])]:
        paths = {m["mountPath"] for m in container.get("volumeMounts", [])}
        assert "/data/millm-checkpoints" not in paths, container["name"]


def test_the_mount_is_named_as_a_checkpoint_read_root() -> None:
    from core.config import Settings

    config = next(d for d in _docs("config.yaml") if d.get("kind") == "ConfigMap")
    value = config["data"]["CHECKPOINT_READ_ROOTS"]
    parsed = Settings(_env_file=None, checkpoint_read_roots=value).checkpoint_read_root_paths  # type: ignore[call-arg]
    assert parsed == [Path(MOUNT)]
