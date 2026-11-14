"""``GET /api/v1/host/resources`` - the resource strip's data, through the live app.

Every request goes through ``create_app()`` and the real router, so deleting the
``include_router(host.router)`` line turns this file red (404s), not just the reachability
manifest. The collector is replaced through FastAPI's dependency seam, and the stub counts
its reads: the route must call it exactly once per request and serve its answer unchanged.
The production collector is then exercised for real over a stubbed driver and a stubbed
psutil, so the parsing and the "named, never zeroed" rule are tested in the code that ships.
"""

from __future__ import annotations

import sys
import types
from collections.abc import Iterator
from typing import Any

import pytest
from fastapi.testclient import TestClient

from api.app import create_app
from api.route_table import mounted_routes
from core import gpu, host_metrics
from core.config import reset_settings

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
def app() -> Iterator[Any]:
    reset_settings()
    application = create_app()
    try:
        yield application
    finally:
        application.dependency_overrides.clear()
        reset_settings()


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
        assert response.json() == body
        assert stub.reads == 1


class TestTheProductionCollector:
    def test_it_reports_real_cpu_memory_and_each_gpu_by_uuid(
        self, app: Any, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        psutil = fake_psutil()
        monkeypatch.setitem(sys.modules, "psutil", psutil)
        monkeypatch.setattr(gpu, "_run", lambda args: TELEMETRY)
        collector = host_metrics.PsutilNvidiaCollector()
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
        collector = host_metrics.PsutilNvidiaCollector()
        collector.read()
        collector.read()
        assert psutil.calls == [host_metrics.FIRST_CPU_SAMPLE_SECONDS, None]

    def test_no_gpu_visible_is_named_not_zeroed(
        self, app: Any, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setitem(sys.modules, "psutil", fake_psutil())
        monkeypatch.setattr(gpu.shutil, "which", lambda name: None)
        collector = host_metrics.PsutilNvidiaCollector()
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
        body = host_metrics.PsutilNvidiaCollector().read()
        assert [entry["code"] for entry in body["unavailable"]] == ["no_cuda_device_visible"]

    def test_without_psutil_cpu_and_memory_are_named_unavailable(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setitem(sys.modules, "psutil", None)  # makes `import psutil` raise
        monkeypatch.setattr(gpu, "_run", lambda args: TELEMETRY)
        body = host_metrics.PsutilNvidiaCollector().read()
        assert body["cpu"] is None and body["memory"] is None
        assert {entry["metric"] for entry in body["unavailable"]} == {"cpu", "memory"}
        assert len(body["gpus"]) == 2
