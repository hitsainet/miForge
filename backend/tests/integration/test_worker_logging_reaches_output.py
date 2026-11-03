"""A task's log lines reach the WORKER PROCESS's output - a real Celery worker  (D43).

Found live 2026-10-17 (F11 window 2): inside a Celery worker ``sys.stdout`` is Celery's
LoggingProxy, which logs every write back into the logging system. Each task calls
``configure_logging``, which bound its handler to that proxy: the handler fed itself, Celery's
recursion guard dropped the line, and every app log line from a task - and Celery's own
task-succeeded/failed lines, which propagate to the same root handler - vanished from
``kubectl logs``. A failed F11 run said nothing about why.

So this starts a REAL worker in a subprocess: the production app (``workers.celery_app``) with
its own configuration, only the broker swapped for kombu's filesystem transport (no Redis), and
the deployed worker's flags (solo pool, ``-Q train``, INFO). The worker sends itself a task on
``worker_ready``; the task calls the real ``configure_logging`` and logs, as ``run_step_loop``
does. The assertion is on the process's output.
"""

from __future__ import annotations

import os
import subprocess
import sys
import textwrap
import time
from pathlib import Path

import pytest

pytest.importorskip("celery")

BACKEND = Path(__file__).resolve().parents[2]
MARKER = "worker_logging_probe_5c1e"


def _probe_app(tmp_path: Path) -> Path:
    data = tmp_path / "broker"
    for name in ("in", "processed"):
        (data / name).mkdir(parents=True)
    done = tmp_path / "done"
    module = tmp_path / "probe_app.py"
    module.write_text(
        textwrap.dedent(f"""
            import logging
            from pathlib import Path

            from celery.signals import worker_ready

            from core.logging import configure_logging
            from workers.celery_app import celery_app as app

            app.conf.update(
                broker_url="filesystem://",
                broker_transport_options={{
                    "data_folder_in": "{data / 'in'}",
                    "data_folder_out": "{data / 'in'}",
                    "processed_folder": "{data / 'processed'}",
                }},
                result_backend=None,
                beat_schedule={{}},
            )


            @app.task(name="miforge.train.logging_probe", queue="train")
            def logging_probe():
                configure_logging("INFO")  # exactly what run_step_loop does first
                logging.getLogger("training.tasks").error(
                    "run failed", extra={{"code": "{MARKER}"}}
                )
                Path("{done}").write_text("ok")
                return "ok"


            @worker_ready.connect
            def _send(**_kwargs):
                logging_probe.apply_async(queue="train")
            """),
        "utf-8",
    )
    return done


def test_a_tasks_log_lines_and_celerys_own_reach_the_worker_output(tmp_path: Path) -> None:
    done = _probe_app(tmp_path)
    env = {
        **os.environ,
        "PYTHONPATH": os.pathsep.join([str(tmp_path), str(BACKEND / "src"), str(BACKEND)]),
        "MILLM_BASE_URL": "",
        "MIDATAWORKS_BASE_URL": "",
    }
    worker = subprocess.Popen(
        [
            sys.executable,
            "-m",
            "celery",
            "-A",
            "probe_app:app",
            "worker",
            "--pool=solo",
            "-c",
            "1",
            "-Q",
            "train",
            "-n",
            "probe@%h",
            "--loglevel=INFO",
            "--without-gossip",
            "--without-mingle",
            "--without-heartbeat",
        ],
        cwd=tmp_path,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )
    try:
        deadline = time.monotonic() + 120
        while not done.exists() and time.monotonic() < deadline and worker.poll() is None:
            time.sleep(0.5)
        time.sleep(3)  # Celery logs "succeeded" just after the task returns
    finally:
        worker.terminate()
        output, _ = worker.communicate(timeout=60)
    assert done.exists(), f"the probe task never ran:\n{output[-3000:]}"
    assert MARKER in output, f"the task's own log line never reached the output:\n{output[-3000:]}"
    assert (
        "logging_probe" in output and "succeeded" in output
    ), f"Celery's task-succeeded line never reached the output:\n{output[-3000:]}"
