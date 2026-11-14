"""App log lines reach the REAL stdout even when `sys.stdout` is redirected  (F11 window 2).

Inside a Celery worker `sys.stdout` is Celery's LoggingProxy, which logs each write back into
the logging system; a handler bound to it feeds itself and Celery's recursion guard drops the
line - so every task's log lines vanished from the container log (2026-10-17). The handler
must write to `sys.__stdout__`.
"""

from __future__ import annotations

import io
import logging
import sys

import pytest

from core.logging import configure_logging


def test_the_handler_writes_to_the_original_stream_not_the_redirect(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    real, redirect = io.StringIO(), io.StringIO()
    monkeypatch.setattr(sys, "__stdout__", real)
    monkeypatch.setattr(sys, "stdout", redirect)
    root = logging.getLogger()
    saved = (list(root.handlers), root.level)
    try:
        configure_logging("INFO")
        logging.getLogger("training.tasks").info("run failed", extra={"code": "run_failed"})
        for handler in root.handlers:
            handler.flush()
    finally:
        root.handlers[:] = saved[0]
        root.setLevel(saved[1])
    assert '"run_failed"' in real.getvalue()
    assert redirect.getvalue() == ""
