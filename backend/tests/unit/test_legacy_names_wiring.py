"""Every process records its old names: the five call sites, by removal (FR-015.6; FTID §3.4).

The API at startup and on each health call, each worker before each task, and each worker and
beat process at start, with a 60 s heartbeat so an IDLE worker's old name still shows (a worker
that runs no task for a week would otherwise read as "nothing in use" and P7 would remove a name
it still reads). Each test asserts the PAYLOAD and the CALL COUNT - the process's own resolution
object and its reporter - because "was called" passes against a call sending the wrong
arguments (global CLAUDE.md). Controls N48a-e and N28/N28b/N28c delete each line in turn.
"""

from __future__ import annotations

import socket
import weakref
from collections.abc import Iterator
from dataclasses import dataclass
from typing import Any

import pytest

from api import health as health_module
from core.config import get_settings, reset_settings
from core.legacy_names import LegacyResolution, LegacyUse
from services.legacy_names import store
from services.legacy_names.store import StoredUses
from workers import celery_app


@dataclass
class Spy:
    calls: list[tuple[tuple[Any, ...], dict[str, Any]]]

    def __call__(self, *args: Any, **kwargs: Any) -> int:
        self.calls.append((args, kwargs))
        return 0


@pytest.fixture
def spies(monkeypatch: pytest.MonkeyPatch) -> Iterator[dict[str, Spy]]:
    reset_settings()
    record, heartbeat = Spy([]), Spy([])
    monkeypatch.setattr(store, "record", record)
    monkeypatch.setattr(store, "start_heartbeat", heartbeat)
    monkeypatch.setattr("core.runtime_settings.refresh_from_database", lambda: None)
    monkeypatch.setattr(celery_app, "_WORKER_REPORTER", None)
    # A resolution that is not empty, so a call carrying a DIFFERENT object is visible.
    get_settings()._legacy_names = LegacyResolution(
        (LegacyUse("env", "MILLM_BASE_URL", "MISERVER_BASE_URL", "in_use", "environment"),)
    )
    yield {"record": record, "heartbeat": heartbeat}
    reset_settings()


def _one_record(spy: Spy, reporter: str) -> None:
    assert len(spy.calls) == 1, f"expected exactly one record call, got {spy.calls}"
    args, kwargs = spy.calls[0]
    assert (
        args[0] is get_settings()._legacy_names
    ), "recorded something other than the process's own resolution"
    assert kwargs["reporter"] == reporter


def test_api_startup_records_once(spies: dict[str, Spy]) -> None:
    from api.app import load_runtime_settings

    load_runtime_settings()
    _one_record(spies["record"], f"api@{socket.gethostname()}")


def test_health_records_once(spies: dict[str, Spy], monkeypatch: pytest.MonkeyPatch) -> None:
    for probe in ("postgres", "redis"):
        monkeypatch.setitem(
            health_module.DEPENDENCY_PROBES, probe, lambda: {"status": "ok", "detail": None}
        )
    monkeypatch.setattr(store, "read_recent", lambda **_: StoredUses())
    health_module.health_payload()
    _one_record(spies["record"], f"api@{socket.gethostname()}")


def test_task_prerun_records_once(spies: dict[str, Spy]) -> None:
    celery_app.refresh_runtime_settings()
    _one_record(spies["record"], f"worker@{socket.gethostname()}")


@dataclass
class Sender:
    hostname: str


def test_worker_ready_records_and_starts_heartbeat(spies: dict[str, Spy]) -> None:
    celery_app.legacy_names_worker_ready(sender=Sender("train@pod-1"))
    _one_record(spies["record"], "train@pod-1")
    assert len(spies["heartbeat"].calls) == 1, "the idle worker's heartbeat was not started"
    args, kwargs = spies["heartbeat"].calls[0]
    assert args[0] is get_settings()._legacy_names and kwargs["reporter"] == "train@pod-1"
    # The worker's later tasks report under the same Celery node name.
    spies["record"].calls.clear()
    celery_app.refresh_runtime_settings()
    _one_record(spies["record"], "train@pod-1")


def test_beat_init_records_and_starts_heartbeat(spies: dict[str, Spy]) -> None:
    celery_app.legacy_names_beat_init(sender=object())
    _one_record(spies["record"], f"beat@{socket.gethostname()}")
    assert len(spies["heartbeat"].calls) == 1, "beat's heartbeat was not started"
    args, kwargs = spies["heartbeat"].calls[0]
    assert args[0] is get_settings()._legacy_names
    assert kwargs["reporter"] == f"beat@{socket.gethostname()}"


def _live(signal: Any) -> list[Any]:
    return [ref() if isinstance(ref, weakref.ReferenceType) else ref for _, ref in signal.receivers]


def test_signal_receivers_are_live() -> None:
    from celery.signals import beat_init, task_prerun, worker_ready

    assert celery_app.refresh_runtime_settings in _live(task_prerun)
    assert celery_app.legacy_names_worker_ready in _live(worker_ready)
    assert celery_app.legacy_names_beat_init in _live(beat_init)


def test_idle_worker_heartbeat_three_minutes() -> None:
    """Three ticks with no task: three records of the SAME resolution object."""
    resolution = LegacyResolution(
        (LegacyUse("env", "MILLM_AUDIT_LABEL", "MISERVER_AUDIT_LABEL", "in_use", "environment"),)
    )
    recorded: list[tuple[Any, str]] = []
    waits: list[float] = []

    def record_fn(res: Any, *, reporter: str) -> int:
        recorded.append((res, reporter))
        return 1

    def sleep(interval: float) -> bool:
        waits.append(interval)
        return len(waits) >= 3  # stop after the third minute

    store.stop_heartbeat()
    try:
        beat = store.start_heartbeat(
            resolution, reporter="default@pod", sleep=sleep, record_fn=record_fn
        )
        assert beat.thread is not None
        beat.thread.join(5)
        assert not beat.thread.is_alive()
    finally:
        store.stop_heartbeat()
    assert len(recorded) == 3, f"expected three ticks in three minutes, got {len(recorded)}"
    assert all(res is resolution and rep == "default@pod" for res, rep in recorded)
    assert waits == [store.THROTTLE_SECONDS] * 3


@pytest.mark.parametrize("start", ["api", "worker", "beat"])
def test_process_start_warns_for_its_own_resolution(
    spies: dict[str, Spy],
    caplog: pytest.LogCaptureFixture,
    monkeypatch: pytest.MonkeyPatch,
    start: str,
) -> None:
    """Each process start logs the WARNING for ITS resolution, once (P7: a retired name still
    set is otherwise visible only in health). Found by control C10 (015 P7): deleting the API's
    ``log_in_use`` call left the whole tree green."""
    import logging

    from api.app import load_runtime_settings
    from core import legacy_names

    get_settings()._legacy_names = LegacyResolution(
        (
            LegacyUse(
                "env",
                "MILLM_TRAINER_TOKEN",
                "MISERVER_TRAINER_TOKEN",
                "retired_but_set",
                "environment",
            ),
        )
    )
    # The API start re-applies the log configuration, which would drop caplog's handler.
    monkeypatch.setattr("core.logging.configure_logging", lambda *a, **k: None)
    legacy_names.reset_logged()
    try:
        with caplog.at_level(logging.WARNING, logger="core.legacy_names"):
            if start == "api":
                load_runtime_settings()
            elif start == "worker":
                celery_app.legacy_names_worker_ready(sender=Sender("train@pod-1"))
            else:
                celery_app.legacy_names_beat_init(sender=object())
        warned = [
            (r.__dict__["code"], r.__dict__["old"])
            for r in caplog.records
            if r.__dict__.get("code", "").startswith("legacy_env_name_")
        ]
        assert warned == [("legacy_env_name_retired", "MILLM_TRAINER_TOKEN")]
    finally:
        legacy_names.reset_logged()
