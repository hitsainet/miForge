"""``GET /api/health`` - each dependency separately, each integration named (task 5.2).

ADR-013. Two rules, and the second is the one that is easy to get wrong:

1. **Each dependency is reported separately.** Postgres and Redis are probed
   independently and one failing never hides the other. A single ``"healthy": false``
   tells an operator to go looking; two named probes tell them where.
2. **Every optional integration is reported by name, configured or not.** ADR-010 makes an
   unset base URL mean *absent and normal*, so "not configured" is a fact a reader needs -
   and **an integration simply missing from the response is indistinguishable from one
   that is broken.** That is why the unconfigured ones are listed explicitly, under their
   ``<service>_not_configured`` code, rather than omitted.

**The route is ``/api/health``, outside ``/api/v1``.** ADR-013 puts it there: a liveness
probe that moves when the API version moves is a probe that breaks a cluster on a day
nobody was changing the cluster.

**Status code.** 503 when a *required* dependency is unavailable - fail closed where
correctness is at stake, and a probe that answers 200 while Postgres is down is a probe
that keeps a broken pod in a Service. **The body shape is identical either way**, so a
caller reads the body and never has to branch on the code. Foundation task 1.7's standalone
harness asserts against the body for exactly that reason: on a machine with no Postgres the
integrations are still reported, by name, with the code at 503.
"""

from __future__ import annotations

import logging
import threading
from collections.abc import Callable
from typing import Any

from fastapi import APIRouter, Response

from core.config import get_settings
from core.errors import not_configured
from core.runtime_settings import effective_settings

log = logging.getLogger(__name__)

router = APIRouter(tags=["health"])

STATUS_OK = "ok"
STATUS_UNAVAILABLE = "unavailable"


def _probe_postgres() -> dict[str, Any]:
    """``SELECT 1`` through the configured engine.

    Imported at call time: foundation 4.1 owns ``db/session.py``, and a module-level
    import would make the health route - and therefore the reachability test - fail for a
    reason unrelated to the API.
    """
    try:
        import sqlalchemy as sa

        from db.session import get_probe_engine

        # The bounded engine (connect and statement time-outs): this runs every 10 s.
        with get_probe_engine().connect() as connection:
            connection.execute(sa.text("SELECT 1"))
    except Exception as exc:
        return {"status": STATUS_UNAVAILABLE, "detail": f"{type(exc).__name__}: {exc}"[:300]}
    return {"status": STATUS_OK, "detail": None}


def _probe_redis() -> dict[str, Any]:
    """``PING`` the broker.

    Redis is the Celery broker and the Socket.IO message queue. It is **never a store of
    record** (ADR-003), so its absence loses live progress and queued work, not data - but
    it does mean no run can be submitted, which is why it is required rather than optional.
    """
    try:
        import redis

        client = redis.Redis.from_url(get_settings().redis_url, socket_connect_timeout=2)
        try:
            client.ping()
        finally:
            client.close()
    except Exception as exc:
        return {"status": STATUS_UNAVAILABLE, "detail": f"{type(exc).__name__}: {exc}"[:300]}
    return {"status": STATUS_OK, "detail": None}


#: What each optional sibling is needed *for*. Carried into the refusal so the reader
#: learns which capabilities are unavailable, not merely that something is unset
#: (ADR-010, ADR-018).
INTEGRATION_CAPABILITIES: dict[str, tuple[str, ...]] = {
    "millm": ("the optional in-place hot-update mode",),
    "midataworks": (
        "contract dataset parts",
        "the human-review selector queue",
    ),
    "activation_source": ("activation statistics for the retention check",),
}

#: The probes, as a table, so the reachability test can assert the route calls **each one
#: exactly once** rather than merely that it called something. "Was called" passes against
#: a call sending the wrong arguments (ADR-016).
DEPENDENCY_PROBES: dict[str, Any] = {
    "postgres": _probe_postgres,
    "redis": _probe_redis,
}

#: A client-side deadline for every external call on the readiness path (015 review round 2).
#: The probe engine's server-side and TCP time-outs (``db.session.get_probe_engine``) bound a
#: dead or black-holed database; they cannot bound a peer that keeps acknowledging and never
#: answers, because libpq has no read time-out. So each call runs in a worker thread, and the
#: answer is given after at most this many seconds whatever the thread is doing. SINGLE FLIGHT:
#: while one call of a kind is still stuck, the next answers at once instead of starting
#: another, so a hung database leaves at most one stuck thread per kind (four kinds) rather than
#: one per probe every 10 s. What stays unbounded is how long such a thread itself lives: until
#: the peer or the OS closes its socket.
PROBE_DEADLINE_SECONDS = 5.0

_in_flight: dict[str, threading.Event] = {}
_in_flight_lock = threading.Lock()


class ProbeTimeoutError(TimeoutError):
    """A readiness-path call gave no answer within :data:`PROBE_DEADLINE_SECONDS`."""


class ProbeStillRunningError(TimeoutError):
    """The previous call of the same kind has not returned; no second one is started."""


def call_with_deadline[T](name: str, call: Callable[[], T], deadline: float | None = None) -> T:
    """``call()`` in a daemon thread, waited on for at most ``deadline`` seconds (single
    flight per ``name``). Raises :class:`ProbeTimeoutError` or
    :class:`ProbeStillRunningError`, or the call's own exception."""
    with _in_flight_lock:
        pending = _in_flight.get(name)
        if pending is not None and not pending.is_set():
            raise ProbeStillRunningError(f"the previous {name} call has not returned yet")
        done = threading.Event()
        _in_flight[name] = done
    box: dict[str, Any] = {}

    def run() -> None:
        try:
            box["value"] = call()
        except BaseException as exc:  # handed to the caller below
            box["error"] = exc
        finally:
            done.set()

    threading.Thread(target=run, name=f"health-{name}", daemon=True).start()
    limit = PROBE_DEADLINE_SECONDS if deadline is None else deadline
    if not done.wait(limit):
        raise ProbeTimeoutError(f"{name} gave no answer within {limit:g} s")
    if "error" in box:
        raise box["error"]
    return box["value"]  # type: ignore[no-any-return]


def reset_in_flight() -> None:
    """Forget every in-flight record (tests; a process starts with none). A thread still
    running keeps its own event, so it can never be mistaken for a finished call."""
    with _in_flight_lock:
        _in_flight.clear()


def _probe_within_deadline(name: str, probe: Callable[[], dict[str, Any]]) -> dict[str, Any]:
    try:
        return call_with_deadline(f"probe:{name}", probe)
    except TimeoutError as exc:
        return {"status": STATUS_UNAVAILABLE, "detail": str(exc)}


def health_payload() -> dict[str, Any]:
    """The health body. Separated from the route so it is testable without a transport."""
    # The integrations block reads the EFFECTIVE settings (F12): a sibling URL set or cleared
    # at runtime is reported as the readers actually see it.
    settings = effective_settings()

    dependencies = {
        name: _probe_within_deadline(name, probe) for name, probe in DEPENDENCY_PROBES.items()
    }
    configured = settings.configured_integrations()

    integrations: dict[str, Any] = {}
    for service, is_configured in configured.items():
        if is_configured:
            integrations[service] = {
                "configured": True,
                "code": None,
                "capabilities_unavailable": [],
            }
        else:
            # Built from the SAME helper the refusal path uses, so the code spelling on
            # this response cannot drift from the code a caller actually receives.
            refusal = not_configured(service, *INTEGRATION_CAPABILITIES.get(service, ()))
            integrations[service] = {
                "configured": False,
                "code": refusal.code,
                "capabilities_unavailable": refusal.details["capabilities_unavailable"],
            }

    degraded = [n for n, d in dependencies.items() if d["status"] != STATUS_OK]
    legacy_names, legacy_names_meta = _legacy_names_blocks()

    return {
        "status": STATUS_OK if not degraded else "degraded",
        "environment": settings.environment,
        "dependencies": dependencies,
        "integrations": integrations,
        # The unconfigured siblings as a flat, named list. Redundant with `integrations`
        # on purpose: foundation task 1.7 asserts each absent sibling is named, and a flat
        # list makes that assertion exact rather than a walk over a nested shape that
        # could silently become empty.
        "unconfigured_integrations": sorted(n for n, c in configured.items() if not c),
        "degraded_dependencies": sorted(degraded),
        # Feature 015 (FR-015.6, CF-4, CF-24): which old MILLM_* names are in use, for the
        # suite's P7 reading. Beside the other keys, never inside them; a failed store read is
        # `legacy_names_meta.error` and never changes the status code.
        "legacy_names": legacy_names,
        "legacy_names_meta": legacy_names_meta,
    }


def _legacy_names_blocks() -> tuple[dict[str, Any], dict[str, Any]]:
    """Record this process's old names (throttled), then serve live uses plus stored ones.

    The record is ``get_settings()``'s: the process's own resolution. Runtime overrides change
    values, not which environment variables are set.
    """
    from datetime import UTC, datetime

    from db.session import probe_session_scope
    from services.legacy_names import store

    live = get_settings()._legacy_names
    now = datetime.now(UTC)

    # Both store calls use the bounded probe session AND the client-side deadline: this
    # answers the readiness probe.
    def write() -> int:
        return store.record(live, reporter=store.api_reporter(), scope=probe_session_scope)

    def read() -> store.StoredUses:
        stored = store.read_recent(now=now, scope=probe_session_scope)
        return stored

    try:
        call_with_deadline("legacy_names_write", write)
    except TimeoutError as exc:
        log.warning(
            "legacy-name store write gave no answer",
            extra={"code": "legacy_names_store_unavailable", "detail": str(exc)},
        )
    read_error: BaseException | None = None
    stored: store.StoredUses | None = None
    try:
        stored = call_with_deadline("legacy_names_read", read)
    except Exception as exc:
        read_error = exc
        log.warning(
            "legacy-name store unreadable",
            extra={"code": "legacy_names_unreadable", "detail": type(exc).__name__},
        )
    return store.block(live, stored, now=now), store.meta(read_error=read_error, stored=stored)


@router.get("/health", summary="Each dependency and each optional integration, by name")
def get_health(response: Response) -> dict[str, Any]:
    """Report Postgres, Redis and every optional integration separately."""
    payload = health_payload()
    if payload["degraded_dependencies"]:
        response.status_code = 503
        log.warning(
            "health degraded",
            extra={
                "code": "health_degraded",
                "degraded_dependencies": payload["degraded_dependencies"],
            },
        )
    return payload
