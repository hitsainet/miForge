"""The FastAPI application factory  (tasks 5.1, 5.3; ADR-013, ADR-017, ADR-021).

**Every failure leaves this process as ``{"error": {"code", "message", "details"}}``.**
Success bodies stay plain JSON - no uniform wrapper, because a wrapper makes every client
unwrap twice and tells it nothing.

That envelope is installed as **exception handlers on the application**, not applied route
by route. The difference matters: a route that forgets to catch its own
:class:`core.errors.Refusal` would otherwise return a 500 with a stack trace, and a
refusal is *half this system's behaviour* (ADR-018) rather than an error path. Handlers on
the app mean a refusal raised anywhere - a route, a dependency, a service three calls down
- arrives at the client in one shape.

The reachability test asserts the handlers are present in the **live** handler registry and
that the envelope actually comes out of the transport, not that the functions exist.
"""

from __future__ import annotations

import logging
from typing import Any

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from starlette.exceptions import HTTPException as StarletteHTTPException

from api.health import router as health_router
from api.v1.router import api_v1_router
from core import atomic_dir
from core.config import get_settings
from core.errors import Refusal
from core.logging import configure_logging

log = logging.getLogger(__name__)

#: Refusal code -> HTTP status. **An explicit table with a documented default**, rather
#: than a status carried on each Refusal: the code is the thing a client matches on and it
#: is stable across repos, whereas the status is this transport's rendering of it. Keeping
#: the mapping here means a code used in both an HTTP route and a worker does not have to
#: carry an HTTP concept into the worker.
#:
#: The default is **400**, not 500: a Refusal is a deliberate, named "no", and reporting it
#: as a server error would send the reader looking for a bug in miForge instead of reading
#: the message, which says what to do next.
REFUSAL_STATUS: dict[str, int] = {
    "run_not_found": 404,
    "gpu_lease_held": 409,
    "gpu_lease_lost": 409,
    "database_unavailable": 503,
    "millm_not_configured": 503,
    "midataworks_not_configured": 503,
    "nvidia_smi_unavailable": 503,
    "nvidia_smi_timeout": 503,
    "nvidia_smi_failed": 503,
    "no_cuda_device_visible": 503,
    # ---- F5, checkpoints (FTDD 005 section 5.4) ----
    # Mapped rather than left to the 400 default because "no such checkpoint" reported as
    # "your request was bad" sends the reader to the wrong place, exactly as the suffix rule
    # below exists to prevent for an unconfigured sibling. The three 409s are conflicts with
    # the state of an artifact on disk, not with the request.
    "checkpoint_not_found": 404,
    "checkpoint_manifest_invalid": 409,
    "checkpoint_hash_mismatch": 409,
    "checkpoint_step_exists": 409,
    "artifact_already_exists": 409,
    "trainer_state_incomplete": 409,
    "trainer_state_absent": 409,
    "run_not_continuable": 409,
    "save_refused_missing_field": 409,
    # A misconfigured filesystem, not a bad request: the temp directory and the destination
    # are on different filesystems, so `rename(2)` is not atomic and the whole guarantee
    # would be void while every test still passed.
    "checkpoint_root_not_atomic": 500,
    # ---- F7, domain packs (FTDD 007 section 5's taxonomy) ----
    # **The 422-vs-503 split is the load-bearing line here** and the FTDD calls it "the
    # single most likely thing to be implemented carelessly": `midataworks_not_configured`
    # (503, above and by the suffix rule) says *the service is absent*, while
    # `pack_incomplete` and `pack_part_unresolved` say *your pack is wrong*. A standalone
    # operator must never be told the second when the first is true.
    #
    # Mapped rather than left to the 400 default because a pack that is missing a slot is not
    # a bad request - the request was fine and the artifact is not - and 422 is what F9 codes
    # its validity pill against. The two 409s are conflicts with the state of a stored
    # artifact: a committed version cannot be edited, and a pack a run cites cannot be
    # deleted.
    "pack_incomplete": 422,
    "pack_part_unresolved": 422,
    "part_unpinnable": 422,
    "retention_floor_missing": 422,
    "pack_document_invalid": 422,
    "pack_version_committed": 409,
    "pack_in_use": 409,
    "pack_not_found": 404,
    "pack_draft_not_found": 404,
    # A dry-run with no provider is an absent capability, not a bad request - and the
    # authoring step still saves without it (F7 E7), so this must not read as a failure of
    # the request that would have saved it.
    "provider_not_available": 503,
    # ---- F9, the artifact forest (FTDD 009 section 5.3) ----
    # The index could not be read. 503 rather than the 400 default because the request was
    # fine and a dependency is not: the manifests on disk are the record (ADR-004) and the
    # recovery is `reindex`, so reporting "your request was bad" would send the reader
    # looking for a mistake they did not make. `invalid_filter` keeps the 400 default,
    # which is exactly right - the filter *was* bad - and `run_not_found` is already 404
    # above.
    "index_unavailable": 503,
    # ---- F6, chaining (FTDD 006 section 5) ----
    # Each of these would otherwise fall to the documented 400 default, which is wrong in
    # the way this table exists to prevent: "your request was bad" sends the reader looking
    # for a mistake they did not make.
    "chain_not_found": 404,
    "chain_stage_invalid": 422,  # a well-formed request whose stage plan does not validate
    "chain_not_stoppable": 409,  # a conflict with the chain's state, not a bad request
    # Beside `gpu_lease_lost`, which is already 409 above: the same event, named for the
    # chain that lost the card rather than the run.
    "chain_lease_lost": 409,
    # ---- F11, §5.10 ----
    # 409, matching the `409 BARRIER_UNAVAILABLE` miLLM raises for the same condition: a
    # conflict with the serving deployment's configuration, not a malformed request. The
    # 400 default would tell the operator their request was wrong when the request was fine
    # and continuous batching is on.
    "weights_barrier_unavailable": 409,
}
DEFAULT_REFUSAL_STATUS = 400


def status_for(code: str) -> int:
    """The HTTP status for a refusal code.

    ``<service>_not_configured`` is matched by **suffix** as well as by the table, because
    ADR-010 fixes that spelling for every optional sibling and a new one must not need an
    edit here to be reported as 503 rather than 400 - a missing entry would otherwise turn
    "this sibling is not configured" into "your request was bad", which sends the reader
    to the wrong place entirely.
    """
    if code in REFUSAL_STATUS:
        return REFUSAL_STATUS[code]
    if code.endswith("_not_configured"):
        return 503
    return DEFAULT_REFUSAL_STATUS


async def refusal_handler(_request: Request, exc: Exception) -> JSONResponse:
    """Render a :class:`Refusal` as the envelope."""
    assert isinstance(exc, Refusal)
    status = status_for(exc.code)
    log.info(
        "refused",
        extra={"code": exc.code, "status": status, "details": exc.details},
    )
    return JSONResponse(status_code=status, content=exc.envelope())


async def validation_handler(_request: Request, exc: Exception) -> JSONResponse:
    """Render a request-validation failure as the envelope.

    FastAPI's default body is a bare ``{"detail": [...]}``, which is a *second* error shape
    on the same API. One shape or none: a client that has to recognise two is a client that
    will recognise one.
    """
    assert isinstance(exc, RequestValidationError)
    return JSONResponse(
        status_code=422,
        content={
            "error": {
                "code": "validation_failed",
                "message": "The request body or parameters did not validate. See details "
                "for the offending fields.",
                # `errors()` carries exception objects in `ctx`; `str` them so the body is
                # JSON-serialisable rather than failing while reporting a failure.
                "details": {"fields": _jsonable(exc.errors())},
            }
        },
    )


async def http_exception_handler(_request: Request, exc: Exception) -> JSONResponse:
    """Render Starlette's ``HTTPException`` - including a 404 for an unknown path."""
    assert isinstance(exc, StarletteHTTPException)
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "error": {
                "code": f"http_{exc.status_code}",
                "message": str(exc.detail),
                "details": {},
            }
        },
    )


async def unhandled_handler(_request: Request, exc: Exception) -> JSONResponse:
    """Last resort: an unexpected exception still leaves as the envelope.

    The message deliberately carries **no** exception text. An unhandled exception's
    message is the one place a connection string or a token can reach a client body
    (ADR-020); it is logged with its traceback instead, where the ``run_id`` binding makes
    it findable.
    """
    log.exception("unhandled exception", extra={"code": "internal_error"})
    return JSONResponse(
        status_code=500,
        content={
            "error": {
                "code": "internal_error",
                "message": "An unexpected error occurred. Check the server logs for the "
                "correlated entry.",
                "details": {},
            }
        },
    )


def _jsonable(value: Any) -> Any:
    """Coerce pydantic's error structures to JSON-safe values."""
    if isinstance(value, dict):
        return {k: _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


def create_app() -> FastAPI:
    """Build the application: handlers first, then routers.

    Handlers before routers is not cosmetic - a router whose import raised would otherwise
    leave an app with no envelope at all, and the first symptom would be a 500 with a
    stack trace in production.
    """
    settings = get_settings()
    configure_logging(settings.log_level)

    # ⚠ FR-005.5, AT STARTUP RATHER THAN AT THE FIRST SAVE. `atomic_dir` already refuses
    # `checkpoint_root_not_atomic` when it publishes, which is correct and is not enough: a
    # save point arrives hours into a run, so a misconfigured root is discovered at the
    # moment the first checkpoint would have been written and the run has nothing to show
    # for the time. The refusal was reachable and the DISCOVERY was late.
    #
    # Logged rather than raised, and that is the whole design decision here. Refusing to
    # construct the app would make an unwritable artifact root take down `/api/health`,
    # `GET /packs` and every screen that touches no artifact at all — and the operator
    # would be diagnosing a dead API rather than a misconfigured mount. The capability that
    # is actually unavailable is *saving*, and `atomic_dir` still refuses that by name with
    # both paths. So this is the early WARNING, not a second gate (ADR-013: name what
    # failed; never substitute a different behaviour for a missing one).
    try:
        atomic_dir.verify_root_is_atomic(settings.checkpoint_root)
    except Refusal as refusal:
        log.error(
            "the artifact root cannot publish atomically",
            extra={
                "code": refusal.code,
                "root": str(settings.checkpoint_root),
                "detail": refusal.message,
                "consequence": (
                    "Saves will refuse by name when they are attempted. Every other "
                    "surface is unaffected, which is why this is reported and not raised."
                ),
            },
        )

    app = FastAPI(
        title="miForge",
        version="0.1.0",
        description=(
            "miForge makes weights - creating them from scratch, changing them in place, "
            "or adding adapters that compose with them."
        ),
        # OpenAPI at a stable, versionless path for the same reason health is versionless.
        openapi_url="/api/openapi.json",
        docs_url="/api/docs",
        redoc_url=None,
    )

    app.add_exception_handler(Refusal, refusal_handler)
    app.add_exception_handler(RequestValidationError, validation_handler)
    app.add_exception_handler(StarletteHTTPException, http_exception_handler)
    app.add_exception_handler(Exception, unhandled_handler)

    # Health sits at /api/health, outside the version prefix (ADR-013).
    app.include_router(health_router, prefix="/api")
    app.include_router(api_v1_router)

    return app


def create_asgi_app() -> Any:
    """The application as it is actually served: FastAPI wrapped by Socket.IO.

    This is the object ``uvicorn`` is pointed at. Wrapping rather than mounting, because
    engine.io's HTTP polling transport does not survive a Starlette ``Mount`` intact.
    Separated from :func:`create_app` so the test suite can exercise the REST surface with
    ``TestClient`` without a Redis message queue being required to construct it.
    """
    from api.sockets import asgi_app

    return asgi_app(create_app())
