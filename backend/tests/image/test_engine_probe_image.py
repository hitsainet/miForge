"""The FR-018.14 probe INSIDE the worker image (feature 018; FTASKS 10.2; FTDD §8.1).

``tests/unit/wsp_worker/miwsp_engine/test_engine_security.py`` proves the grandchild's isolation
on the developer's host. This runs the same probe in the shipped ``test`` image, with the pod's
flags (``harness.py``: read-only root, uid 10002, every capability dropped, no-new-privileges,
the pod's volumes and its network namespace), where the engine is the image's own
``/opt/wsp/wsp_worker``. The host side is pytest and the standard library only (CI's harness): the
bundle, the token and the needles are made inside the container.
``wsp_children/engine_probe_image.py`` is the container's own process (PID 1) and starts the
REAL child entry point twice: as a child of PID 1, as under the pod's supervisor, and behind an
intermediate parent. Both times the editor's ``python`` code, the
probe, runs in the kernel grandchild and must find the session token in no form (raw, base64,
JSON-escaped; environment, argv, objects, memory), be refused the parent's ``/proc`` entries, find
the practice files gone, and be refused every socket.

Runs in the private-CI job ``wsp-worker-image``; anywhere else it skips by name (and fails under
``MIFORGE_REQUIRE_WSP_IMAGE=1``).
"""

from __future__ import annotations

import json
import re
from collections.abc import Iterator
from typing import Any

import pytest

from tests.image import harness as h

#: The practice files the in-image bundle writer makes (``image_bundle.write_bundle``): the
#: token file and the five practice arrays.
PRACTICE_FILES = 6


@pytest.fixture(scope="module")
def test_image() -> str:
    return h.require_image()


@pytest.fixture
def pod(test_image: str) -> Iterator[h.Pod]:
    with h.pod(test_image) as p:
        yield p


def _results(logs: str) -> dict[str, tuple[int, dict[str, Any]]]:
    out: dict[str, tuple[int, dict[str, Any]]] = {}
    for line in logs.splitlines():
        if line.startswith("PROBE-RESULT "):
            _tag, placement, parent, found = line.split(" ", 3)
            out[placement] = (int(parent), json.loads(found))
    return out


def test_the_probe_in_the_grandchild_finds_nothing_in_the_image(pod: h.Pod) -> None:
    w = pod.start(
        script=None,
        entrypoint="python",
        command=("-I", f"{h.MOUNT_ROOT}/wsp_children/engine_probe_image.py"),
        restart="no",
        wait_ready=False,
    )
    code = w.wait_exit(timeout=300)
    logs = w.logs()
    assert code == 0, logs
    assert "PROBE-FAILED" not in logs, logs
    assert re.search(r"^TOKEN-SHA256 [0-9a-f]{64}$", logs, flags=re.M), logs
    results = _results(logs)
    assert set(results) == {"pid1_child", "non_pid1_child"}, logs
    assert results["pid1_child"][0] == 1, results  # the script is PID 1: the session's parent
    for placement, (_parent, found) in results.items():
        for check in ("environ", "cmdline", "objects", "memory"):
            assert found[check] == "absent", (placement, check, found)
        for check in ("parent_environ", "parent_mem", "parent_fd", "parent_map_files"):
            assert found[check].startswith("refused:"), (placement, check, found)
        assert "parent_fd_token" not in found, (placement, found)
        assert found["practice_count"] == str(PRACTICE_FILES), (placement, found)
        practice = [found[f"practice_{i}"] for i in range(PRACTICE_FILES)]
        assert practice == ["missing"] * PRACTICE_FILES, (placement, found)
        for name in ("socket_inet", "socket_unix", "socketpair"):
            assert found[name] == "refused:PermissionError", (placement, name, found)
