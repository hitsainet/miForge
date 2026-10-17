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
from typing import Any

from fastapi import APIRouter, Response

from core.config import get_settings
from core.errors import not_configured

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

        from db.session import get_engine

        with get_engine().connect() as connection:
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
    "millm": ("the optional in-place hot-update mode (BRD-02 section 5.10)",),
    "midataworks": (
        "contract dataset parts",
        "the human-review selector queue",
    ),
    "activation_source": ("activation statistics for the retention check (R-02.19)",),
}

#: The probes, as a table, so the reachability test can assert the route calls **each one
#: exactly once** rather than merely that it called something. "Was called" passes against
#: a call sending the wrong arguments (ADR-016).
DEPENDENCY_PROBES: dict[str, Any] = {
    "postgres": _probe_postgres,
    "redis": _probe_redis,
}


def health_payload() -> dict[str, Any]:
    """The health body. Separated from the route so it is testable without a transport."""
    settings = get_settings()

    dependencies = {name: probe() for name, probe in DEPENDENCY_PROBES.items()}
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
    }


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
