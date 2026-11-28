"""019 P3 (D-11; F19-D19, TP-3): the push task's broker message is the run id and nothing else.

The session token is minted INSIDE the push task (``dispatch.push_claimed``), so neither its
plaintext nor its base64 can reach the broker. This publishes through the real Celery producer
into an in-memory broker and reads the message as Redis would hold it: body (base64 JSON, per
the memory note "leak tests must look for the encoded form") and every header.
"""

from __future__ import annotations

import base64
import json

import pytest

from wsp import tasks, tokens


@pytest.fixture()
def memory_broker(monkeypatch: pytest.MonkeyPatch):
    from workers.celery_app import celery_app

    def fresh_pools() -> None:
        # The app caches its connection and producer pools; drop both so the next publish
        # opens a connection to the broker URL in force now.
        celery_app._pool = None
        celery_app.amqp._producer_pool = None

    previous = celery_app.conf.broker_url
    celery_app.conf.broker_url = "memory://"
    fresh_pools()
    try:
        yield celery_app
    finally:
        celery_app.conf.broker_url = previous
        fresh_pools()


def _drain(app, queue: str) -> list:
    with app.connection_for_read() as conn:
        simple = conn.SimpleQueue(queue)
        out = []
        while True:
            try:
                message = simple.get(block=False)
            except simple.Empty:
                break
            out.append(message)
            message.ack()
        simple.close()
        return out


def test_the_push_message_is_the_run_id_alone_and_no_token(memory_broker) -> None:
    plaintext, _ = tokens.mint()
    tasks.enqueue_push("wsp_0123456789ab")
    (message,) = _drain(memory_broker, "wsp_push")
    args, kwargs, _embed = message.payload
    assert args == ["wsp_0123456789ab"] and kwargs == {}
    assert message.headers["task"] == "miforge.wsp.push"
    raw = json.dumps(
        {
            "body": message.body if isinstance(message.body, str) else message.body.decode(),
            "headers": message.headers,
            "properties": message.properties,
        },
        default=str,
    )
    # A token minted in this process appears in no form anywhere in the message.
    for form in (plaintext, base64.b64encode(plaintext.encode()).decode()):
        assert form not in raw
    assert "token" not in json.dumps(message.payload).lower()
