"""The WSP driver is a live Celery task, on its own queue, with its beat entry (FTASKS 7.12;
P60, PF-53, PF-57, R3A-7).

The registry is read in a FRESH interpreter that imports only what a worker started with
``-A workers.celery_app`` imports, so a test module that happened to import ``wsp.tasks`` earlier
cannot make this pass.
"""

from __future__ import annotations

import json
import subprocess
import sys
import types
from pathlib import Path

import pytest

SRC = Path(__file__).resolve().parents[3] / "src"

PROBE = """
import json
from workers.celery_app import celery_app, queue_for
celery_app.loader.import_default_modules()
beat = celery_app.conf.beat_schedule.get("wsp-drive")
print(json.dumps({
    "registered": "miforge.wsp.drive" in celery_app.tasks,
    "import_registered": "miforge.wsp.import_bundle" in celery_app.tasks,
    "push_registered": "miforge.wsp.push" in celery_app.tasks,
    "push_queue": queue_for("miforge.wsp.push"),
    "drive_queue": queue_for("miforge.wsp.drive"),
    "import_queue": queue_for("miforge.wsp.import_bundle"),
    "beat": beat and {"task": beat["task"], "schedule": float(beat["schedule"]),
                      "options": beat["options"]},
    "queues": [q.name for q in (celery_app.conf.task_queues or [])],
}))
"""


def _probe() -> dict:
    out = subprocess.run(
        [sys.executable, "-c", PROBE],
        cwd=SRC,
        env={"PATH": "/usr/bin:/bin", "PYTHONPATH": str(SRC)},
        capture_output=True,
        text=True,
        timeout=120,
        check=True,
    )
    return json.loads(out.stdout.strip().splitlines()[-1])


def test_the_driver_is_live_routed_and_scheduled() -> None:
    seen = _probe()
    assert seen["registered"] is True, "a worker would receive beat's message and have no task"
    assert seen["drive_queue"] == "wsp"
    assert seen["import_queue"] == "default"
    assert seen["import_registered"] is True, "POST /wsp/bundles/import would queue into nothing"
    # 019 P3 (D-12; F19-D19, D35): the push task, on its own solo queue.
    assert seen["push_registered"] is True, "a claimed run would be pushed by nothing"
    assert seen["push_queue"] == "wsp_push"
    assert "wsp_push" in seen["queues"]
    assert seen["beat"] == {
        "task": "miforge.wsp.drive",
        "schedule": 5.0,
        "options": {"queue": "wsp", "expires": 4.0},
    }


def test_the_pass_skips_loudly_when_wsp_is_not_configured(monkeypatch) -> None:
    from workers.celery_app import celery_app
    from wsp.settings import get_wsp_settings

    monkeypatch.delenv("WSP_WORKER_URL", raising=False)
    get_wsp_settings.cache_clear()
    celery_app.loader.import_default_modules()
    try:
        assert celery_app.tasks["miforge.wsp.drive"].run() == {"skipped": "wsp_not_configured"}
    finally:
        get_wsp_settings.cache_clear()


def test_the_live_pass_drives_every_worker_with_the_production_wiring(monkeypatch) -> None:
    """FR-016.21, .23 and 019 P3 (FR-019.6, .7; D-12): the beat's pass hands the driver one slot per
    provisioned worker (the URL template expanded), the pool size the runtime settings hold NOW,
    ``run_files``, the live sensitivity reader and the push task's enqueue, exactly once per pass;
    and every worker client is closed."""
    from core import runtime_settings
    from workers.celery_app import celery_app
    from wsp import dispatch, tasks, window_read, worker_client
    from wsp.settings import get_wsp_settings

    calls: list[dict] = []
    closed: list[str] = []

    class Worker:
        def __init__(self, url: str) -> None:
            self.url = url

        def close(self) -> None:
            closed.append(self.url)

    def fake_drive(factory, workers, settings, **kwargs):
        calls.append(
            {"factory": factory, "slots": [(w.index, w.address, w.client.url) for w in workers]}
            | kwargs
        )
        return dispatch.Pass()

    monkeypatch.setenv("WSP_WORKER_URL", "http://w-{index}.pool:9000")
    monkeypatch.setenv("WSP_WORKER_POOL_MAX", "3")
    monkeypatch.setenv("WSP_GATEWAY_URL", "http://127.0.0.1:8080")
    monkeypatch.setattr(dispatch, "drive", fake_drive)
    monkeypatch.setattr(worker_client, "HttpWorker", Worker)
    monkeypatch.setattr("db.session.get_session_factory", lambda: "factory")
    # A pool size no default could produce: the pass must read the setting, not a constant.
    monkeypatch.setattr(
        runtime_settings, "effective_settings", lambda: types.SimpleNamespace(wsp_pool_size=2)
    )
    get_wsp_settings.cache_clear()
    celery_app.loader.import_default_modules()
    try:
        assert celery_app.tasks["miforge.wsp.drive"].run() == {"actions": []}
    finally:
        get_wsp_settings.cache_clear()
    assert len(calls) == 1
    urls = [f"http://w-{i}.pool:9000" for i in range(3)]
    assert calls[0]["factory"] == "factory"
    assert calls[0]["slots"] == [(i, url, url) for i, url in enumerate(urls)]
    assert calls[0]["pool_size"] == 2
    assert calls[0]["files_for"] is tasks.run_files
    assert calls[0]["read_sensitivity"] is tasks.read_sensitivity
    assert calls[0]["enqueue_push"] is tasks.enqueue_push
    # 2026-11-27 (run wsp_4fbf5a18758e): the pass reads the window too, with the production
    # reader, so a run on a miServer endpoint that states none waits for its model; the push task
    # (below) still reads it again and resolves it.
    assert calls[0]["read_window"] is window_read.read_window_answer
    # Debt D5 (2026-11-28): the claim's hold on a server that gave no answer, kept across passes.
    assert calls[0]["backoff"] is tasks.CLAIM_BACKOFF
    assert isinstance(tasks.CLAIM_BACKOFF, window_read.NoAnswerBackoff)
    assert "read_served_window" not in calls[0]
    assert sorted(closed) == urls


def test_the_push_task_pushes_to_the_worker_the_run_holds(monkeypatch) -> None:
    """019 P3 (D-12; F19-D19): ``miforge.wsp.push`` reads the run's worker index, pushes to THAT
    worker's address with the real window reader and file list, once, and closes the client."""
    from workers.celery_app import celery_app
    from wsp import dispatch, tasks, window_read, worker_client
    from wsp.settings import get_wsp_settings

    calls: list[dict] = []
    closed: list[str] = []

    class Worker:
        def __init__(self, url: str) -> None:
            self.url = url

        def close(self) -> None:
            closed.append(self.url)

    class Session:
        def __enter__(self):
            return self

        def __exit__(self, *exc):
            return False

        def scalar(self, statement):
            return 1

    def fake_push(factory, worker, settings, run_id, **kwargs):
        calls.append({"url": worker.url, "run_id": run_id} | kwargs)
        return "pushed"

    monkeypatch.setenv("WSP_WORKER_URL", "http://w-{index}.pool:9000")
    monkeypatch.setenv("WSP_WORKER_POOL_MAX", "2")
    monkeypatch.setenv("WSP_GATEWAY_URL", "http://127.0.0.1:8080")
    monkeypatch.setattr(dispatch, "push_claimed", fake_push)
    monkeypatch.setattr(worker_client, "HttpWorker", Worker)
    monkeypatch.setattr("db.session.get_session_factory", lambda: Session)
    get_wsp_settings.cache_clear()
    celery_app.loader.import_default_modules()
    try:
        got = celery_app.tasks["miforge.wsp.push"].run("wsp-run-1")
    finally:
        get_wsp_settings.cache_clear()
    assert got == {"run_id": "wsp-run-1", "outcome": "pushed"}
    assert calls == [
        {
            "url": "http://w-1.pool:9000",
            "run_id": "wsp-run-1",
            "read_window": window_read.read_window_answer,
            "files_for": tasks.run_files,
        }
    ]
    assert closed == ["http://w-1.pool:9000"]


def test_the_claim_enqueues_the_run_id_alone_on_the_push_queue(monkeypatch) -> None:
    """019 P3 (D-11; TP-3): the broker message is the run id and nothing else - no token."""
    from wsp import tasks

    sent: list[tuple] = []
    monkeypatch.setattr(
        tasks.push, "apply_async", lambda *a, **k: sent.append((a, k)), raising=False
    )
    tasks.enqueue_push("wsp-run-7")
    assert sent == [((), {"args": ("wsp-run-7",), "queue": "wsp_push"})]


@pytest.mark.parametrize("unset", ["WSP_WORKER_URL", "WSP_GATEWAY_URL"])
@pytest.mark.parametrize("providers", ["", "miwsp_engine"])
def test_a_skipped_pass_says_why_only_when_a_provider_is_enabled(
    monkeypatch, caplog, unset, providers
) -> None:
    """R3 review fix (OR3-016-2), 2026-11-16: EITHER URL unset skips the pass (the push needs both),
    silently in a standalone install, and with a warning naming the setting when the operator has
    enabled a provider; the warning repeats at most every NOT_CONFIGURED_WARN_EVERY_S."""
    import logging

    from workers.celery_app import celery_app
    from wsp import dispatch, tasks
    from wsp.settings import get_wsp_settings

    monkeypatch.setenv("WSP_WORKER_URL", "http://127.0.0.1:9")
    monkeypatch.setenv("WSP_GATEWAY_URL", "http://127.0.0.1:8080")
    monkeypatch.setenv(unset, "")
    monkeypatch.setenv("WSP_PROVIDERS_AVAILABLE", providers)
    monkeypatch.setattr(dispatch, "drive", lambda *a, **k: pytest.fail("the pass must not run"))
    monkeypatch.setattr(tasks, "_last_not_configured_warning", [])
    get_wsp_settings.cache_clear()
    celery_app.loader.import_default_modules()
    task = celery_app.tasks["miforge.wsp.drive"]
    try:
        with caplog.at_level(logging.WARNING, logger="wsp.tasks"):
            assert task.run() == {"skipped": "wsp_not_configured"}
            assert task.run() == {"skipped": "wsp_not_configured"}
    finally:
        get_wsp_settings.cache_clear()
    warned = [r for r in caplog.records if getattr(r, "code", None) == "wsp_not_configured"]
    if providers:
        assert len(warned) == 1 and unset in warned[0].getMessage(), caplog.records
    else:
        assert warned == []
