"""Worker-side run events reach Redis - every one, with no transport error  (found live).

F11 window 2, 2026-10-18: a Celery worker has no event loop, and the emit bridges ran
``asyncio.run(emit_run_event(...))`` per event. ``sio``'s ``AsyncRedisManager`` keeps the
connection it opened on the FIRST loop, which ``asyncio.run`` then closes, so every later
emit failed with "Event loop is closed", logged ``Cannot publish to redis... retrying`` at
ERROR, reconnected and only then delivered: 364 ERROR pairs in 51 s of one run.

A fake cannot show that - it is a property of the real client and a real connection - so
this subscribes to the real Redis the suite is configured with (CI runs a Redis service) on
python-socketio's channel, emits through the production bridge with no loop running, and
asserts every message arrived **and** that socketio logged no publish error. It skips,
loudly, when no Redis is reachable.
"""

from __future__ import annotations

import json
import logging
import time
from typing import Any

import pytest

redis = pytest.importorskip("redis")


@pytest.fixture
def subscriber() -> Any:
    from core.config import get_settings

    url = get_settings().redis_url
    client = redis.Redis.from_url(url, socket_connect_timeout=1)
    try:
        client.ping()
    except Exception as exc:  # pragma: no cover - environment
        pytest.skip(f"no Redis reachable at the configured REDIS_URL ({type(exc).__name__})")
    pubsub = client.pubsub()
    pubsub.subscribe("socketio")
    deadline = time.monotonic() + 5
    while time.monotonic() < deadline:  # drain the subscribe confirmation
        if pubsub.get_message(timeout=0.2):
            break
    yield pubsub
    pubsub.close()
    client.close()


def test_every_worker_emit_arrives_and_none_logs_a_publish_error(
    subscriber: Any, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    from api import sockets
    from training import run_events, tasks

    monkeypatch.setattr(sockets, "_WORKER_PUBLISHER", None)  # built fresh for this Redis
    emitter = tasks._emitter()
    with caplog.at_level(logging.WARNING):
        for step in range(1, 6):
            emitter("run-redis", run_events.EVENT_BOUNDARY, {"step": step})

    received: list[dict[str, Any]] = []
    deadline = time.monotonic() + 10
    while len(received) < 5 and time.monotonic() < deadline:
        message = subscriber.get_message(timeout=0.5)
        if message and message.get("type") == "message":
            received.append(json.loads(message["data"]))

    errors = [r.getMessage() for r in caplog.records if r.levelno >= logging.WARNING]
    assert errors == [], f"the worker emit path logged transport errors: {errors[:4]}"
    steps = [m["data"][0]["step"] for m in received if m.get("method") == "emit"]
    assert steps == [1, 2, 3, 4, 5], f"published {steps} of five worker events"
    assert {m["room"] for m in received} == {sockets.room_for("run-redis")}
