"""Host readings for the resource strip: CPU, memory, and each visible GPU by UUID.

The shell's strip shows CPU, RAM, GPU, VRAM and temperature on every screen. Until
2026-10-19 nothing served them, so every reading rendered "not reported" in production while
the API pod could see the 3090 the whole time (``NVIDIA_VISIBLE_DEVICES=all``). This module
is the collector; ``api/v1/endpoints/host.py`` serves it.

**What each figure is, so the strip can name its scale.**

* **CPU** - ``psutil.cpu_percent`` over the interval since the previous read, across all
  logical cores. ``/proc/stat`` in a container is the node's, so this is the node's load,
  which is what the operator is asking when a run is slow. The first read in a process has
  no previous sample, so it takes a short blocking one rather than reporting psutil's
  meaningless ``0.0``.
* **Memory** - ``total - available`` over ``total``, the node's figure for the same reason.
* **GPUs** - one ``nvidia-smi --query-gpu`` row per card, keyed by UUID (ADR-008: never an
  index), through ``core.gpu`` so every refusal is the one the rest of the backend uses.

**An unreadable source is named, not zeroed.** Each source that could not be read adds an
entry to ``unavailable`` with a stable code and a short reason, and its reading is ``null``.
The strip prints that reason ("no GPU visible") instead of a row of "not reported".
"""

from __future__ import annotations

import threading
from dataclasses import asdict
from datetime import UTC, datetime
from typing import Any, Protocol

from core import gpu
from core.errors import Refusal

#: The first CPU read's blocking sample, in seconds. Short: it delays one request per process.
FIRST_CPU_SAMPLE_SECONDS = 0.2

#: Short, operator-facing reasons per refusal code. The strip shows these verbatim.
GPU_REASONS: dict[str, str] = {
    "nvidia_smi_unavailable": "no GPU visible (nvidia-smi not found)",
    "nvidia_smi_timeout": "GPU driver did not answer",
    "nvidia_smi_failed": "GPU driver error",
    "no_cuda_device_visible": "no GPU visible",
}


class HostCollector(Protocol):
    """What the route reads. A protocol so a test can hand the route a stub."""

    def read(self) -> dict[str, Any]: ...


class PsutilNvidiaCollector:
    """The production collector: psutil for CPU and memory, ``nvidia-smi`` for the cards."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._cpu_primed = False

    def _cpu(self, psutil: Any) -> dict[str, Any]:
        with self._lock:
            if not self._cpu_primed:
                percent = float(psutil.cpu_percent(interval=FIRST_CPU_SAMPLE_SECONDS))
                self._cpu_primed = True
            else:
                percent = float(psutil.cpu_percent(interval=None))
        return {"percent": percent, "logical_cores": psutil.cpu_count(logical=True)}

    def read(self) -> dict[str, Any]:
        unavailable: list[dict[str, str]] = []
        cpu: dict[str, Any] | None = None
        memory: dict[str, Any] | None = None
        try:
            import psutil
        except ImportError:
            for metric in ("cpu", "memory"):
                unavailable.append(
                    {
                        "metric": metric,
                        "code": "psutil_unavailable",
                        "message": "host metrics library not installed",
                    }
                )
        else:
            cpu = self._cpu(psutil)
            vm = psutil.virtual_memory()
            memory = {"used_bytes": int(vm.total - vm.available), "total_bytes": int(vm.total)}

        gpus: list[dict[str, Any]] = []
        try:
            gpus = [asdict(reading) for reading in gpu.telemetry_by_uuid()]
        except Refusal as refusal:
            unavailable.append(
                {
                    "metric": "gpu",
                    "code": refusal.code,
                    "message": GPU_REASONS.get(refusal.code, "GPU not readable"),
                }
            )
        else:
            if not gpus:
                unavailable.append(
                    {
                        "metric": "gpu",
                        "code": "no_cuda_device_visible",
                        "message": GPU_REASONS["no_cuda_device_visible"],
                    }
                )

        return {
            "observed_at": datetime.now(UTC).isoformat(),
            "cpu": cpu,
            "memory": memory,
            "gpus": gpus,
            "unavailable": unavailable,
        }


_COLLECTOR: HostCollector | None = None


def get_collector() -> HostCollector:
    """The process's collector, built once so the CPU sample spans successive reads."""
    global _COLLECTOR
    if _COLLECTOR is None:
        _COLLECTOR = PsutilNvidiaCollector()
    return _COLLECTOR
