"""The MOUNTED route table, read the way FastAPI itself reads it.

⚠ ``app.routes`` STOPPED BEING THE ROUTE TABLE IN FASTAPI 0.14x (deps sync 2026-10-21).
Until FastAPI 0.115 ``include_router`` copied every route of the included router into the
app, so ``for route in app.routes`` saw each one with its full path. From the 0.14x line an
included router stays ONE entry, a private ``_IncludedRouter``, whose routes are resolved
lazily into per-inclusion contexts. ``app.routes`` on miForge's app now holds five entries:
three documentation routes and the two included routers.

Nothing raised. Every walk over ``app.routes`` simply found nothing to do. In production that
meant ``mark_gated_routes`` marked **zero** routes, so no gated route published
``x-approval-action`` and the MCP server would have described every gated tool as one that runs
at once. The gate itself still held, because the decorator wraps the endpoint. 123 tests went
red on the upgrade. Every reachability guard asked ``app.routes`` and found the routes "not
mounted".

``fastapi.routing.iter_route_contexts`` is the walk FastAPI's own OpenAPI generator uses
(``fastapi/openapi/utils.py``). It is public, and it flattens nested inclusions. Each item
answers ``path`` (with every prefix applied), ``methods``, ``endpoint`` and ``name`` for the
route as mounted, and ``original_route`` for the ``APIRoute`` object that was declared.
"""

from __future__ import annotations

from typing import Any

from fastapi import FastAPI
from fastapi.routing import APIRoute, RouteContext, iter_route_contexts


def mounted_routes(app: FastAPI) -> list[RouteContext]:
    """Every route the app serves, nested inclusions flattened, prefixes applied."""
    return list(iter_route_contexts(app.routes))


def mounted_api_routes(app: FastAPI) -> list[RouteContext]:
    """The subset that are ``APIRoute``s (no documentation, static or Socket.IO routes)."""
    return [r for r in mounted_routes(app) if isinstance(r.original_route, APIRoute)]


def declared_route(route: RouteContext) -> APIRoute:
    """The ``APIRoute`` object as declared (where ``openapi_extra`` lives)."""
    original = route.original_route
    assert isinstance(original, APIRoute), original
    return original


def set_openapi_extra(route: RouteContext, extra: dict[str, Any]) -> None:
    """Set ``openapi_extra`` where the OpenAPI generator will READ it.

    An included route's effective context COPIES ``openapi_extra`` from the declared route
    when the context is built, and the context is cached. So setting it on the declared
    ``APIRoute`` alone is enough only if no context has been built yet, and walking the
    table builds them. Set both. Pinned by ``test_agent_origin.py``, which reads the
    generated OpenAPI document and not the route objects.
    """
    declared_route(route).openapi_extra = extra
    effective = getattr(route, "_route_context", None)
    if effective is not None:
        effective.openapi_extra = extra
