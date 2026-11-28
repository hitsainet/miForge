"""The worker's ``/work`` is provisioned once and agreed everywhere (production defect 2026-11-20).

A 35 GB bundle was pushed into an 8 GiB ``/work``: the kubelet evicted the worker mid-push. Three
numbers must agree, read from the real files: the ``work`` emptyDir's ``sizeLimit`` in
``gitops/pilot/wsp-worker.yaml``, ``WSP_WORKER_WORK_BYTES`` in ``gitops/pilot/config.yaml`` (what
the backend refuses and sets the disk watchdog against) and ``WSPSettings``' default. The
container's ``ephemeral-storage`` limit must not bind before the volume does.
"""

from __future__ import annotations

import re
from typing import Any

import pytest

from tests.support.wsp_manifests import documents, one
from wsp import capacity
from wsp.settings import WSPSettings

NAME = "miforge-wsp-worker"
#: The production bundle that failed on 2026-11-20 (its import record's ``bytes``).
PHASE2_BUNDLE_BYTES = 34_881_633_698

_UNITS = {
    "": 1,
    "k": 1000,
    "M": 1000**2,
    "G": 1000**3,
    "T": 1000**4,
    "Ki": 1024,
    "Mi": 1024**2,
    "Gi": 1024**3,
    "Ti": 1024**4,
}


def quantity(text: str) -> int:
    """A Kubernetes quantity (``64Gi``, ``72Gi``, ``512Mi``, a plain integer) in bytes."""
    match = re.fullmatch(r"(\d+)(Ki|Mi|Gi|Ti|k|M|G|T)?", str(text).strip())
    assert match, f"not a quantity this test reads: {text!r}"
    return int(match.group(1)) * _UNITS[match.group(2) or ""]


def _pod() -> dict[str, Any]:
    spec: dict[str, Any] = one("StatefulSet", NAME)["spec"]["template"]["spec"]
    return spec


def work_size_limit() -> int:
    (work,) = [v for v in _pod()["volumes"] if v["name"] == "work"]
    return quantity(work["emptyDir"]["sizeLimit"])


def config_value() -> int:
    (config,) = [d for d in documents("config.yaml") if d.get("kind") == "ConfigMap"]
    value = config["data"]["WSP_WORKER_WORK_BYTES"]
    assert isinstance(value, str) and value.isdigit(), f"bytes as a quoted integer, got {value!r}"
    return int(value)


def test_quantity_reads_what_the_manifests_use() -> None:
    assert quantity("64Gi") == 64 * 1024**3 and quantity("512Mi") == 512 * 1024**2
    assert quantity("8G") == 8 * 1000**3 and quantity("68719476736") == 68719476736
    with pytest.raises(AssertionError):
        quantity("64 gibibytes")


def test_the_config_map_equals_the_work_volume() -> None:
    assert config_value() == work_size_limit(), (
        "WSP_WORKER_WORK_BYTES (config.yaml) and the worker's `work` sizeLimit (wsp-worker.yaml) "
        "disagree: raise both together."
    )


def test_the_setting_default_equals_the_work_volume() -> None:
    default = WSPSettings.model_fields["wsp_worker_work_bytes"].default
    assert default == work_size_limit()


def test_the_container_limit_does_not_bind_before_the_volume() -> None:
    (container,) = _pod()["containers"]
    limit = quantity(container["resources"]["limits"]["ephemeral-storage"])
    assert limit > work_size_limit()


def test_the_phase2_bundle_fits_the_provisioned_volume() -> None:
    """The bundle that was evicted on 2026-11-20 fits now, with the stated headroom."""
    settings = WSPSettings(wsp_worker_work_bytes=work_size_limit())
    assert capacity.refusal(PHASE2_BUNDLE_BYTES, settings) is None
    assert capacity.watchdog_bytes(settings) > PHASE2_BUNDLE_BYTES + capacity.EDITOR_HEADROOM_BYTES
