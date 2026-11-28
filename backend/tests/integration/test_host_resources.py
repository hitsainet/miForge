"""``GET /api/v1/host/resources`` - the resource strip's data, through the live app.

Every request goes through ``create_app()`` and the real router, so deleting the
``include_router(host.router)`` line turns this file red (404s), not just the reachability
manifest. The collector is replaced through FastAPI's dependency seam, and the stub counts
its reads: the route must call it exactly once per request and serve its answer unchanged.
The production collector is then exercised for real over a stubbed driver and a stubbed
psutil, so the parsing and the "named, never zeroed" rule are tested in the code that ships.
"""

from __future__ import annotations

import os
import sys
import types
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest
from fastapi.testclient import TestClient

from api.app import create_app
from api.route_table import mounted_routes
from core import gpu, host_metrics
from core.config import reset_settings
from wsp.settings import get_wsp_settings

UUID_3090 = "GPU-247aa582-0d1b-e161-8156-983ed1fefc57"
UUID_3080TI = "GPU-f47ba814-49a2-603f-3595-275284140251"

TELEMETRY = (
    f"{UUID_3090}, NVIDIA GeForce RTX 3090, 97, 16149, 24576, 66\n"
    f"{UUID_3080TI}, NVIDIA GeForce RTX 3080 Ti, [N/A], 512, 12288, [N/A]\n"
)


class StubCollector:
    def __init__(self, body: dict[str, Any]) -> None:
        self.body = body
        self.reads = 0

    def read(self) -> dict[str, Any]:
        self.reads += 1
        return self.body


@pytest.fixture
def app(tmp_path: Path) -> Iterator[Any]:
    """The live app over a migrated database (the route reads the WSP workers' disk rows)."""
    from api.deps import session_dependency
    from tests.support.db import migrated_engine
    from tests.support.jobs import session_for

    reset_settings()
    engine = migrated_engine(tmp_path)
    application = create_app()

    def one_session() -> Iterator[Any]:
        with session_for(engine) as session:
            yield session

    application.dependency_overrides[session_dependency] = one_session
    try:
        yield application
    finally:
        application.dependency_overrides.clear()
        engine.dispose()
        reset_settings()


def no_volumes() -> list[tuple[str, Path]]:
    return []


def fake_psutil(percent: float = 37.5, total: int = 64 * 2**30, available: int = 40 * 2**30) -> Any:
    module = types.SimpleNamespace()
    module.calls = []

    def cpu_percent(interval: float | None = None) -> float:
        module.calls.append(interval)
        return percent

    module.cpu_percent = cpu_percent
    module.cpu_count = lambda logical=True: 32
    module.virtual_memory = lambda: types.SimpleNamespace(total=total, available=available)
    return module


class TestTheRouteServesTheCollector:
    def test_the_route_is_in_the_live_registry(self, app: Any) -> None:
        paths = {
            (method, route.path)
            for route in mounted_routes(app)
            for method in getattr(route, "methods", ())
        }
        assert ("GET", "/api/v1/host/resources") in paths

    def test_it_serves_the_collectors_reading_once_per_request(self, app: Any) -> None:
        body = {
            "observed_at": "2026-10-19T12:00:00+00:00",
            "cpu": {"percent": 12.0, "logical_cores": 32},
            "memory": {"used_bytes": 10, "total_bytes": 100},
            "gpus": [
                {
                    "uuid": UUID_3090,
                    "name": "NVIDIA GeForce RTX 3090",
                    "utilization_percent": 97.0,
                    "memory_used_bytes": 1,
                    "memory_total_bytes": 2,
                    "temperature_c": 66.0,
                }
            ],
            "unavailable": [],
        }
        stub = StubCollector(body)
        app.dependency_overrides[host_metrics.get_collector] = lambda: stub
        response = TestClient(app).get("/api/v1/host/resources")
        assert response.status_code == 200
        # The collector's reading unchanged, plus the WSP workers' disk rows (none here) and the
        # /work limit (2026-11-26).
        assert response.json() == {
            **body,
            "disk": {
                "volumes": [],
                "wsp_work_limit_bytes": get_wsp_settings().wsp_worker_work_bytes,
                "wsp_workers": [],
            },
        }
        assert stub.reads == 1


class TestTheProductionCollector:
    def test_it_reports_real_cpu_memory_and_each_gpu_by_uuid(
        self, app: Any, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        psutil = fake_psutil()
        monkeypatch.setitem(sys.modules, "psutil", psutil)
        monkeypatch.setattr(gpu, "_run", lambda args: TELEMETRY)
        collector = host_metrics.PsutilNvidiaCollector(no_volumes)
        app.dependency_overrides[host_metrics.get_collector] = lambda: collector

        body = TestClient(app).get("/api/v1/host/resources").json()

        assert body["cpu"] == {"percent": 37.5, "logical_cores": 32}
        assert body["memory"] == {"used_bytes": 24 * 2**30, "total_bytes": 64 * 2**30}
        assert body["gpus"] == [
            {
                "uuid": UUID_3090,
                "name": "NVIDIA GeForce RTX 3090",
                "utilization_percent": 97.0,
                "memory_used_bytes": 16149 * 1024 * 1024,
                "memory_total_bytes": 24576 * 1024 * 1024,
                "temperature_c": 66.0,
            },
            {
                # `[N/A]` is unknown, never zero.
                "uuid": UUID_3080TI,
                "name": "NVIDIA GeForce RTX 3080 Ti",
                "utilization_percent": None,
                "memory_used_bytes": 512 * 1024 * 1024,
                "memory_total_bytes": 12288 * 1024 * 1024,
                "temperature_c": None,
            },
        ]
        assert body["unavailable"] == []
        assert "index" not in str(body["gpus"])

    def test_the_first_cpu_read_samples_and_later_reads_span_the_interval(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """psutil's first ``interval=None`` read is a meaningless 0.0, so it is never served."""
        psutil = fake_psutil()
        monkeypatch.setitem(sys.modules, "psutil", psutil)
        monkeypatch.setattr(gpu, "_run", lambda args: TELEMETRY)
        collector = host_metrics.PsutilNvidiaCollector(no_volumes)
        collector.read()
        collector.read()
        assert psutil.calls == [host_metrics.FIRST_CPU_SAMPLE_SECONDS, None]

    def test_no_gpu_visible_is_named_not_zeroed(
        self, app: Any, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setitem(sys.modules, "psutil", fake_psutil())
        monkeypatch.setattr(gpu.shutil, "which", lambda name: None)
        collector = host_metrics.PsutilNvidiaCollector(no_volumes)
        app.dependency_overrides[host_metrics.get_collector] = lambda: collector

        response = TestClient(app).get("/api/v1/host/resources")

        assert response.status_code == 200
        body = response.json()
        assert body["gpus"] == []
        assert body["unavailable"] == [
            {
                "metric": "gpu",
                "code": "nvidia_smi_unavailable",
                "message": "no GPU visible (nvidia-smi not found)",
            }
        ]
        assert body["cpu"]["percent"] == 37.5

    def test_a_driver_listing_no_card_is_named(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setitem(sys.modules, "psutil", fake_psutil())
        monkeypatch.setattr(gpu, "_run", lambda args: "")
        body = host_metrics.PsutilNvidiaCollector(no_volumes).read()
        assert [entry["code"] for entry in body["unavailable"]] == ["no_cuda_device_visible"]

    def test_without_psutil_cpu_and_memory_are_named_unavailable(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setitem(sys.modules, "psutil", None)  # makes `import psutil` raise
        monkeypatch.setattr(gpu, "_run", lambda args: TELEMETRY)
        body = host_metrics.PsutilNvidiaCollector(no_volumes).read()
        assert body["cpu"] is None and body["memory"] is None
        assert {entry["metric"] for entry in body["unavailable"]} == {"cpu", "memory"}
        assert len(body["gpus"]) == 2


class TestTheDiskReading:
    """2026-11-26, the 019 P4 MCP gap: the checkpoint and WSP roots' free and total bytes, read
    with ``statvfs`` in this process (as psutil reads memory), and named when unreadable."""

    def test_the_volumes_are_the_checkpoint_and_wsp_roots_from_the_settings(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        checkpoints, wsp = tmp_path / "ckpt", tmp_path / "wsp"
        monkeypatch.setenv("CHECKPOINT_ROOT", str(checkpoints))
        monkeypatch.setenv("WSP_BUNDLE_ROOT", str(wsp))
        reset_settings()
        get_wsp_settings.cache_clear()
        try:
            assert host_metrics.disk_volumes() == [("checkpoints", checkpoints), ("wsp", wsp)]
        finally:
            reset_settings()
            get_wsp_settings.cache_clear()

    def test_each_volume_reads_statvfs_free_and_total_and_its_device(
        self, app: Any, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setitem(sys.modules, "psutil", fake_psutil())
        monkeypatch.setattr(gpu, "_run", lambda args: TELEMETRY)
        a, b = tmp_path / "a", tmp_path / "b"
        a.mkdir()
        b.mkdir()
        st = types.SimpleNamespace(f_bavail=7, f_frsize=4096, f_blocks=11)
        seen: list[str] = []

        def statvfs(path: Any) -> Any:
            seen.append(str(path))
            return st

        monkeypatch.setattr(host_metrics.os, "statvfs", statvfs)
        collector = host_metrics.PsutilNvidiaCollector(lambda: [("checkpoints", a), ("wsp", b)])
        app.dependency_overrides[host_metrics.get_collector] = lambda: collector

        body = TestClient(app).get("/api/v1/host/resources").json()

        assert seen == [str(a), str(b)]
        assert body["disk"]["volumes"] == [
            {
                "name": "checkpoints",
                "path": str(a),
                "free_bytes": 7 * 4096,
                "total_bytes": 11 * 4096,
                "device": os.stat(a).st_dev,
            },
            {
                "name": "wsp",
                "path": str(b),
                "free_bytes": 7 * 4096,
                "total_bytes": 11 * 4096,
                "device": os.stat(b).st_dev,
            },
        ]
        assert body["unavailable"] == []

    def test_a_real_volume_reads_the_filesystems_own_figures(self, tmp_path: Path) -> None:
        reading, missing = host_metrics.read_volume("wsp", tmp_path)
        st = os.statvfs(tmp_path)
        assert missing is None
        assert reading["total_bytes"] == st.f_blocks * st.f_frsize > 0
        assert isinstance(reading["free_bytes"], int) and reading["free_bytes"] >= 0
        assert reading["device"] == os.stat(tmp_path).st_dev

    def test_an_unreadable_volume_is_named_never_zeroed(self, tmp_path: Path) -> None:
        missing_path = tmp_path / "not-mounted"
        collector = host_metrics.PsutilNvidiaCollector(lambda: [("checkpoints", missing_path)])
        body = collector.read()
        assert body["disk"]["volumes"] == [
            {
                "name": "checkpoints",
                "path": str(missing_path),
                "free_bytes": None,
                "total_bytes": None,
                "device": None,
            }
        ]
        assert [e for e in body["unavailable"] if e["metric"] == "disk"] == [
            {
                "metric": "disk",
                "code": "disk_unreadable",
                "message": f"volume checkpoints ({missing_path}) cannot be read: "
                "FileNotFoundError",
            }
        ]

    def test_the_process_collector_reads_the_settings_volumes(self) -> None:
        """The production collector's default is :func:`disk_volumes`: removing the default
        (or the read loop) leaves the strip without disk again."""
        host_metrics._COLLECTOR = None
        try:
            collector = host_metrics.get_collector()
            assert collector._volumes is host_metrics.disk_volumes  # type: ignore[attr-defined]
        finally:
            host_metrics._COLLECTOR = None
