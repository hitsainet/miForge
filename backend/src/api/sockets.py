"""The Socket.IO server and the run room  (task 6.6; ADR-007, ADR-013).

**The socket is a convenience; the database is the truth.** ADR-013 says stores fall back
to polling the run record when the socket drops, and ADR-007 says why in one sentence:
*"progress reported only over the socket is invisible to the janitor,"* which is how
miStudio reaped a live 5.8-hour job as "worker lost."

So the contract for every emitter in this codebase is: **write the record first, emit
second.** :func:`emit_run_event` cannot enforce that from where it sits - it does not know
what the caller wrote - but it is the single place a run event is published, which is what
makes the rule checkable by reading one call site per feature instead of searching for
``sio.emit``.

One room per run, ``miforge/run/<id>``, named by :func:`room_for`. Never built by string
interpolation at a call site: a run whose events go to ``miforge/runs/<id>`` is a screen
that shows nothing and reports no error, and that typo is invisible in review.
"""

from __future__ import annotations

import logging
from typing import Any

import socketio

from core.config import get_settings

log = logging.getLogger(__name__)

#: The room name template. ADR-007 fixes the spelling; F9's store joins by it.
ROOM_TEMPLATE = "miforge/run/{run_id}"


def room_for(run_id: str) -> str:
    """The Socket.IO room carrying one run's events.

    The only place this string is built. See the module docstring for why.
    """
    if not run_id:
        raise ValueError("a run room must name a run")
    return ROOM_TEMPLATE.format(run_id=run_id)


def _build_server() -> socketio.AsyncServer:
    settings = get_settings()
    return socketio.AsyncServer(
        async_mode="asgi",
        # Redis as the message queue so an event emitted by a Celery worker reaches a
        # browser connected to the API process. Without it the two processes have separate
        # in-memory room tables and a worker's emit reaches nobody - the failure looks like
        # "the socket works in dev and not in the cluster", because in dev they are often
        # the same process.
        client_manager=socketio.AsyncRedisManager(settings.redis_url),
        # The frontend is served from a different origin in dev (Vite) and the same origin
        # in the cluster. Permissive in dev only; the cluster value is set by deployment.
        cors_allowed_origins="*" if settings.environment == "dev" else [],
        logger=False,
        engineio_logger=False,
    )


sio: socketio.AsyncServer = _build_server()


# python-socketio's decorators are untyped; the ignore sits on the decorator so each
# handler's own signature is still checked under MyPy strict.
@sio.event  # type: ignore[untyped-decorator]
async def connect(sid: str, environ: dict[str, Any], auth: Any = None) -> None:
    log.debug("socket connected", extra={"sid": sid})


@sio.event  # type: ignore[untyped-decorator]
async def disconnect(sid: str) -> None:
    log.debug("socket disconnected", extra={"sid": sid})


@sio.on("subscribe_run")  # type: ignore[untyped-decorator]
async def subscribe_run(sid: str, data: dict[str, Any]) -> dict[str, Any]:
    """Join a run's room.

    Returns the room it joined rather than nothing, so a client can assert it is in the
    room it meant to join. A silent join is how a screen ends up subscribed to a room that
    does not exist, waiting for events that are being delivered correctly elsewhere.
    """
    run_id = (data or {}).get("run_id")
    if not run_id:
        return {"error": {"code": "run_id_required", "message": "Name a run_id.", "details": {}}}
    room = room_for(str(run_id))
    await sio.enter_room(sid, room)
    return {"room": room}


async def emit_run_event(run_id: str, event: str, payload: dict[str, Any]) -> None:
    """Publish one event into a run's room. **The single emit site for run progress.**

    The caller must already have written whatever the database needs to carry
    (ADR-007). This function deliberately does not take a session and cannot write one:
    making it look as though emitting were sufficient is precisely the mistake.
    """
    await sio.emit(event, payload, room=room_for(run_id))


#: The worker-side publisher, built on first use. See :func:`publish_from_worker`.
_WORKER_PUBLISHER: Any = None


def _worker_publisher() -> Any:
    global _WORKER_PUBLISHER
    if _WORKER_PUBLISHER is None:
        _WORKER_PUBLISHER = socketio.RedisManager(get_settings().redis_url, write_only=True)
    return _WORKER_PUBLISHER


def publish_from_worker(run_id: str, event: str, payload: dict[str, Any]) -> None:
    """Publish one run event from a process with **no event loop** - a Celery worker.

    python-socketio's documented way for an external process to emit: a synchronous,
    write-only ``RedisManager`` on the same channel the API server's
    ``AsyncRedisManager`` listens to. Found live 2026-10-18 (F11 window 2): the workers
    used ``asyncio.run(emit_run_event(...))`` per event, which builds and CLOSES a loop each
    time, while ``sio``'s ``AsyncRedisManager`` keeps the connection it opened on the first
    one. Every later emit hit "Event loop is closed", logged ``Cannot publish to redis...
    retrying`` at ERROR, reconnected, and only then delivered: 364 ERROR pairs in 51 s of
    one run, a fresh Redis connection per event, and real errors drowned in the noise.

    Raises whatever the transport raises; the bridges log it (the record is durable).
    """
    _worker_publisher().emit(event, payload, namespace="/", room=room_for(run_id))


def asgi_app(other_asgi_app: Any) -> Any:
    """Wrap the FastAPI app so Socket.IO serves ``/socket.io`` beside the REST routes.

    Wrapped rather than mounted: ``socketio.ASGIApp`` handles the engine.io handshake's
    own HTTP polling transport, which a Starlette ``Mount`` does not pass through intact.
    """
    return socketio.ASGIApp(sio, other_asgi_app=other_asgi_app, socketio_path="socket.io")
