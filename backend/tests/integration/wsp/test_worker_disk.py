"""Each WSP worker's ``/work`` disk, end to end (2026-11-26, the 019 P4 MCP gap: FTASKS 21.1 was
read with ``df`` on the node because ``forge_host_resources`` carried no disk reading).

The worker's ``/ready`` reports its filesystem (``supervisor.work_disk``); a real driver pass
records it (``pool.record_ready``, migration 0041); ``GET /api/v1/host/resources`` serves it under
``disk.wsp_workers``, beside ``disk.wsp_work_limit_bytes``. The worker is untrusted, so a figure
that is not a bounded non-negative integer is stored null.
"""

from __future__ import annotations

import os
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pytest
from fastapi.testclient import TestClient

from core import host_metrics
from tests.integration.wsp.test_dispatch import FakeWorker, World, build_world
from tests.integration.wsp.test_run_cause import _client
from wsp import pool
from wsp.settings import get_wsp_settings
from wsp_worker import supervisor

FREE, TOTAL = 157_212_762_112, 844_362_645_504


@dataclass
class DiskWorker(FakeWorker):
    """A worker whose ``/ready`` carries the disk figures, as the supervisor's does."""

    free: Any = FREE
    total: Any = TOTAL

    def ready(self) -> dict[str, Any]:
        return {**super().ready(), "work_free_bytes": self.free, "work_total_bytes": self.total}


@pytest.fixture()
def world(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Iterator[World]:
    monkeypatch.setenv("WSP_WORKER_URL", "http://wsp-worker-{index}.test:9000")
    monkeypatch.setenv("WSP_WORKER_POOL_MAX", "2")
    monkeypatch.setenv("WSP_WORKER_WORK_BYTES", str(72 * 1024**3))
    built = build_world(tmp_path, monkeypatch)
    yield built
    built.engine.dispose()
    get_wsp_settings.cache_clear()


@pytest.fixture()
def client(world: World, monkeypatch: pytest.MonkeyPatch) -> TestClient:
    test_client = _client(world, monkeypatch)
    collector = host_metrics.PsutilNvidiaCollector(lambda: [])
    test_client.app.dependency_overrides[host_metrics.get_collector] = lambda: collector  # type: ignore[attr-defined]
    return test_client


def _disk(client: TestClient) -> dict[str, Any]:
    answer = client.get("/api/v1/host/resources")
    assert answer.status_code == 200, answer.text
    return dict(answer.json()["disk"])


def test_each_workers_reported_disk_reaches_host_resources(
    world: World, client: TestClient
) -> None:
    first, second = DiskWorker(), DiskWorker(free=1, total=2)
    world.drive([first, second])
    disk = _disk(client)
    assert disk["wsp_work_limit_bytes"] == 72 * 1024**3
    workers = disk["wsp_workers"]
    assert [(w["index"], w["boot_id"], w["free_bytes"], w["total_bytes"]) for w in workers] == [
        (0, first.boot, FREE, TOTAL),
        (1, second.boot, 1, 2),
    ]
    assert all(w["reported_at"] for w in workers)
    assert set(workers[0]) == {"index", "boot_id", "free_bytes", "total_bytes", "reported_at"}


@pytest.mark.parametrize("hostile", ["157212762112", -1, True, 1.5, 2**63, None, [1]])
def test_a_figure_that_is_not_a_bounded_count_is_stored_null(
    world: World, client: TestClient, hostile: Any
) -> None:
    world.drive([DiskWorker(free=hostile, total=TOTAL), DiskWorker()])
    first, second = _disk(client)["wsp_workers"]
    assert (first["free_bytes"], first["total_bytes"]) == (None, TOTAL)
    assert (second["free_bytes"], second["total_bytes"]) == (FREE, TOTAL)


def test_a_worker_that_stops_answering_keeps_its_last_figures_and_their_time(
    world: World, client: TestClient
) -> None:
    worker = DiskWorker()
    world.drive([worker, DiskWorker()])
    before = _disk(client)["wsp_workers"][0]
    worker.unreachable = True
    world.drive([worker, DiskWorker()])
    after = _disk(client)["wsp_workers"][0]
    assert (after["free_bytes"], after["total_bytes"]) == (FREE, TOTAL)
    assert after["reported_at"] == before["reported_at"]


def test_a_worker_from_before_the_figures_reads_null_not_zero(
    world: World, client: TestClient
) -> None:
    world.drive([FakeWorker(), FakeWorker()])  # a /ready without the two keys
    assert [(w["free_bytes"], w["total_bytes"]) for w in _disk(client)["wsp_workers"]] == [
        (None, None),
        (None, None),
    ]


def test_the_supervisors_figures_are_its_work_roots_filesystem(tmp_path: Path) -> None:
    st = os.statvfs(tmp_path)
    figures = supervisor.work_disk(tmp_path)
    assert set(figures) == {"work_free_bytes", "work_total_bytes"}
    assert figures["work_total_bytes"] == st.f_blocks * st.f_frsize > 0
    assert isinstance(figures["work_free_bytes"], int) and figures["work_free_bytes"] >= 0
    assert pool.disk_bytes(figures["work_free_bytes"]) == figures["work_free_bytes"]


def test_an_unreadable_work_root_reports_null(tmp_path: Path) -> None:
    assert supervisor.work_disk(tmp_path / "absent") == {
        "work_free_bytes": None,
        "work_total_bytes": None,
    }
