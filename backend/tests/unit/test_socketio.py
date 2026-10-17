"""The Socket.IO server and the run room  (foundation 6.6; ADR-007, ADR-013).

**The socket is a convenience; the database is the truth.** So the tests that earn their
place here are not "does an event arrive" - they are the three properties whose failure is
silent:

* the room name is built in **one** place, because a run whose events go to
  ``miforge/runs/<id>`` is a screen that shows nothing and reports no error;
* the emitter sends to **that** room, with **that** payload, **once** - ADR-016's rule that
  "was called" passes against a call sending the wrong arguments;
* the server is actually **mounted** beside the REST routes, rather than merely constructed.
"""

from __future__ import annotations

import pytest

from api import sockets
from api.app import create_app


class TestTheRoomName:
    def test_the_room_is_the_adr_spelling(self):
        """ADR-007 fixes ``miforge/run/<id>``; F9's store joins by it. A single character
        wrong here is a run view that is permanently empty and silent about why."""
        assert sockets.room_for("abc123") == "miforge/run/abc123"

    def test_it_is_built_from_the_template_rather_than_a_second_literal(self):
        assert sockets.ROOM_TEMPLATE == "miforge/run/{run_id}"
        assert sockets.room_for("x") == sockets.ROOM_TEMPLATE.format(run_id="x")

    def test_an_empty_run_id_is_refused(self):
        """A room with no run in its name collects every run's events into one stream."""
        with pytest.raises(ValueError):
            sockets.room_for("")


class TestTheEmitter:
    async def test_it_emits_to_the_run_room_with_the_payload_exactly_once(self, monkeypatch):
        """**The room, the payload and the call count.** A test asserting only that
        ``emit`` was called passes against an emit to the wrong room - which is the exact
        failure that cannot be seen from either side at runtime: the worker reports
        success and the browser simply never hears anything."""
        calls: list[tuple] = []

        async def fake_emit(event, payload, room=None):
            calls.append((event, payload, room))

        monkeypatch.setattr(sockets.sio, "emit", fake_emit)
        await sockets.emit_run_event("run-7", "step", {"step": 412, "loss": 1.23})

        assert len(calls) == 1
        event, payload, room = calls[0]
        assert event == "step"
        assert payload == {"step": 412, "loss": 1.23}
        assert room == "miforge/run/run-7"

    async def test_subscribe_returns_the_room_it_joined(self, monkeypatch):
        """Returned rather than silent, so a client can assert it is in the room it meant
        to join. A silent join is how a screen waits for events being delivered correctly
        somewhere else."""
        joined: list[tuple] = []

        async def fake_enter(sid, room):
            joined.append((sid, room))

        monkeypatch.setattr(sockets.sio, "enter_room", fake_enter)
        result = await sockets.subscribe_run("sid-1", {"run_id": "run-7"})

        assert result == {"room": "miforge/run/run-7"}
        assert joined == [("sid-1", "miforge/run/run-7")]

    async def test_subscribing_without_a_run_id_refuses_in_the_envelope(self):
        """Over the socket as over HTTP: one error shape (ADR-013)."""
        result = await sockets.subscribe_run("sid-1", {})
        assert result["error"]["code"] == "run_id_required"


class TestItIsMounted:
    """Constructed is not mounted. ADR-016 #1 applies to a transport as much as to a route."""

    def test_the_server_is_in_asgi_mode(self):
        assert sockets.sio.async_mode == "asgi"

    def test_the_rest_app_is_wrapped_rather_than_mounted(self):
        """``socketio.ASGIApp`` handles engine.io's HTTP polling handshake, which a
        Starlette ``Mount`` does not pass through intact - so a socket that works in dev
        over websockets fails behind a proxy that forces polling."""
        rest = create_app()
        wrapped = sockets.asgi_app(rest)
        assert wrapped.other_asgi_app is rest
        # python-socketio normalises the path it serves on.
        assert wrapped.engineio_path == "/socket.io/"

    def test_the_connect_and_disconnect_handlers_are_registered(self):
        """In the live handler table, not merely defined as functions."""
        handlers = sockets.sio.handlers["/"]
        assert "connect" in handlers
        assert "disconnect" in handlers
        assert "subscribe_run" in handlers

    def test_redis_is_the_message_queue(self):
        """Without it, an event emitted by a Celery worker reaches nobody: the API process
        and the worker keep separate in-memory room tables. The failure looks like *"the
        socket works in dev and not in the cluster"*, because in dev they are often the
        same process."""
        import socketio

        assert isinstance(sockets.sio.manager, socketio.AsyncRedisManager)
