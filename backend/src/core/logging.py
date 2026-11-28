"""Structured JSON logs, with ``run_id`` bound for a run's lifetime  (task 2.4; ADR-021).

One line per event, machine-readable, and **``run_id`` on every line in a run's path** -
because the question asked of these logs is almost always "what happened to run X", and
grepping a run out of interleaved worker output is only possible if every line carries it.

``run_id`` is bound through a :class:`contextvars.ContextVar`, so it follows the code
rather than being threaded through every signature. A Celery task binds it once at entry.

**No secret reaches a log line** (ADR-020): the §5.10 trainer token and any bearer header
are redacted by key name, and the redaction is a test, not a convention.
"""

from __future__ import annotations

import json
import logging
import sys
from collections.abc import Iterator
from contextlib import contextmanager
from contextvars import ContextVar
from datetime import UTC, datetime
from typing import Any

_run_id: ContextVar[str | None] = ContextVar("run_id", default=None)

#: Keys whose value is never logged, matched case-insensitively on a substring so that
#: `authorization`, `millm_trainer_token` and `x_dataworks_agent_token` are all caught.
#: A denylist is the wrong shape for most problems, but here the universe of secrets is
#: small, known and named in ADR-020.
_REDACT_SUBSTRINGS = ("token", "secret", "password", "authorization", "bearer", "api_key")

REDACTED = "***"

#: Attributes `logging` puts on every record. Anything NOT here is caller-supplied and is
#: emitted as a field - which is what makes `log.info("saved", step=1200)` work.
_STANDARD = frozenset(
    [
        "args",
        "asctime",
        "created",
        "exc_info",
        "exc_text",
        "filename",
        "funcName",
        "levelname",
        "levelno",
        "lineno",
        "module",
        "msecs",
        "message",
        "msg",
        "name",
        "pathname",
        "process",
        "processName",
        "relativeCreated",
        "stack_info",
        "thread",
        "threadName",
        "taskName",
    ]
)


def bind_run_id(run_id: str | None) -> None:
    """Bind ``run_id`` for everything that follows on this task/request."""
    _run_id.set(run_id)


def current_run_id() -> str | None:
    return _run_id.get()


@contextmanager
def run_context(run_id: str) -> Iterator[None]:
    """Bind ``run_id`` for the duration of the block, restoring the previous value."""
    token = _run_id.set(run_id)
    try:
        yield
    finally:
        _run_id.reset(token)


def _redact(key: str, value: Any) -> Any:
    lowered = key.lower()
    if any(marker in lowered for marker in _REDACT_SUBSTRINGS):
        return REDACTED if value is not None else None
    if isinstance(value, dict):
        return {k: _redact(k, v) for k, v in value.items()}
    return value


class JsonFormatter(logging.Formatter):
    """One JSON object per line.

    Uses :mod:`json` directly rather than the canonical serialiser: a log line is not a
    hashed artifact, and sorting keys would move ``message`` away from the front where a
    human reading ``kubectl logs`` wants it.
    """

    def format(self, record: logging.LogRecord) -> str:
        payload: dict[str, Any] = {
            "ts": datetime.fromtimestamp(record.created, tz=UTC).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }
        run_id = _run_id.get()
        if run_id is not None:
            payload["run_id"] = run_id

        for key, value in record.__dict__.items():
            if key not in _STANDARD and not key.startswith("_"):
                payload[key] = _redact(key, value)

        if record.exc_info:
            payload["exception"] = self.formatException(record.exc_info)

        return json.dumps(payload, default=str, ensure_ascii=False)


def configure_logging(level: str = "INFO") -> None:
    """Install the JSON formatter on the root logger. Idempotent."""
    # The REAL stdout. Inside a Celery worker `sys.stdout` is Celery's LoggingProxy, which logs
    # every write back into the logging system - into this very handler - and Celery's
    # recursion guard drops the line. So every app log line from a task vanished from the
    # container log, including why a run failed (F11 window 2, 2026-10-17).
    handler = logging.StreamHandler(sys.__stdout__ or sys.stdout)
    handler.setFormatter(JsonFormatter())
    root = logging.getLogger()
    root.handlers.clear()
    root.addHandler(handler)
    root.setLevel(level)
    # uvicorn installs its own handlers; make them propagate so every line is JSON.
    for name in ("uvicorn", "uvicorn.error", "uvicorn.access"):
        logger = logging.getLogger(name)
        logger.handlers.clear()
        logger.propagate = True
