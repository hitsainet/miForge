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


def test_the_live_pass_pushes_with_the_production_file_list(monkeypatch) -> None:
    """FR-016.21, .23: the beat's pass hands the driver ``run_files`` (bundle files, plus the
    replay's original transcript or the rerun's stored program), exactly once per pass."""
    from workers.celery_app import celery_app
    from wsp import dispatch, tasks
    from wsp.settings import get_wsp_settings

    calls: list[dict] = []

    def fake_drive(factory, worker, settings, **kwargs):
        calls.append(kwargs)
        return dispatch.Pass()

    monkeypatch.setenv("WSP_WORKER_URL", "http://127.0.0.1:9")
    monkeypatch.setenv("WSP_GATEWAY_URL", "http://127.0.0.1:8080")
    monkeypatch.setattr(dispatch, "drive", fake_drive)
    monkeypatch.setattr("db.session.get_session_factory", lambda: None)
    get_wsp_settings.cache_clear()
    celery_app.loader.import_default_modules()
    try:
        assert celery_app.tasks["miforge.wsp.drive"].run() == {"actions": []}
    finally:
        get_wsp_settings.cache_clear()
    assert len(calls) == 1
    assert calls[0]["files_for"] is tasks.run_files
    assert calls[0]["read_sensitivity"] is tasks.read_sensitivity
    # Feature 018 (FR-018.12): the pass reads miServer's served window with the real reader.
    assert calls[0]["read_served_window"] is tasks.read_served_window


@pytest.mark.parametrize("unset", ["WSP_WORKER_URL", "WSP_GATEWAY_URL"])
@pytest.mark.parametrize("providers", ["", "weightspace"])
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
