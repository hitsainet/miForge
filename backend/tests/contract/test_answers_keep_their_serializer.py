"""Typing a route in documentation never changes how FastAPI serialises its answer (021 MF9 as
amended 2026-11-26, operator, suite-wide for ENH-013).

Switching a route to ``response_model=None`` swaps FastAPI's serializer from the response field's
(Pydantic) to ``jsonable_encoder``: a UTC datetime ``…Z`` becomes ``…+00:00``, and other values can
change form. So every route keeps the response model it served before 021 (the inferred
``dict[str, Any]`` for an untyped one), and the declared answer goes in ``responses={…}`` only.

- Every operation's served response model and response class equal the ones ``origin/main``
  ``4bb6d56`` served (``tests/support/fixtures/answer_serializers_4bb6d56.json``, recorded then).
  A route added on ``main`` after it is recorded as ``main`` first served it, never from this
  branch: train F3's ``POST /wsp/runs/validate`` (``response_model=None``) and its two run reads
  (their enforced models), at ``2b31b2f``.
  The one exception is :data:`BYTES_ONLY` (P4, FTASKS 5.7): the two routes that answer only bytes
  declare ``response_class=Response`` so FastAPI stops documenting a JSON body beside their bytes.
  Their endpoints return a ``Response`` on every path, and FastAPI sends a returned ``Response``
  as it is, so the class is never used: a small app below shows the bytes and headers unchanged.
- On a small app, the pattern (an inferred ``dict[str, Any]`` plus a declared model) serves the same
  bytes as the bare route for datetimes, small and large floats, Decimals, timedeltas and NaN, while
  ``response_model=None`` would not, which is why the rule exists.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime, timedelta
from decimal import Decimal
from pathlib import Path
from typing import Any

from fastapi import FastAPI
from fastapi.routing import APIRoute
from fastapi.testclient import TestClient
from pydantic import BaseModel

from api.route_table import mounted_api_routes

PINNED = (
    Path(__file__).resolve().parents[1] / "support" / "fixtures" / "answer_serializers_4bb6d56.json"
)


def _served() -> dict[str, dict[str, str]]:
    from api.app import create_app

    out: dict[str, dict[str, str]] = {}
    for ctx in mounted_api_routes(create_app()):
        route = ctx.original_route
        if isinstance(route, APIRoute):
            cls = route.response_class
            for method in sorted((ctx.methods or set()) - {"HEAD", "OPTIONS"}):
                out[f"{method} {ctx.path_format}"] = {
                    "response_model": repr(route.response_model).replace("typing.", ""),
                    "response_class": getattr(cls, "__name__", None) or type(cls).__name__,
                }
    return out


#: Routes that answer only bytes and declare ``response_class=Response`` for their documentation.
BYTES_ONLY = {
    "GET /api/v1/wsp/candidates/{candidate_id}/tensors",
    "GET /api/v1/wsp/candidates/{candidate_id}/verdict/scores",
}


def _returns_a_response(key: str) -> bool:
    """The endpoint's return annotation is a ``Response``: FastAPI never applies its class."""
    import inspect

    from fastapi import Response

    from api.app import create_app

    for ctx in mounted_api_routes(create_app()):
        if any(f"{m} {ctx.path_format}" == key for m in ctx.methods or ()):
            hint = inspect.signature(ctx.endpoint, eval_str=True).return_annotation
            return isinstance(hint, type) and issubclass(hint, Response)
    raise AssertionError(f"{key} is not a route")


def test_every_route_serves_through_the_model_it_served_before_021() -> None:
    pinned: dict[str, dict[str, str]] = json.loads(PINNED.read_text(encoding="utf-8"))
    served = _served()
    added = sorted(set(served) - set(pinned))
    changed = {k: (pinned[k], served[k]) for k in pinned if k in served and pinned[k] != served[k]}
    documented_bytes = {
        k: v
        for k, v in changed.items()
        if k in BYTES_ONLY
        and v[0] == {"response_class": "DefaultPlaceholder", "response_model": "None"}
        and v[1] == {"response_class": "Response", "response_model": "None"}
        and _returns_a_response(k)
    }
    assert set(documented_bytes) == BYTES_ONLY, f"the bytes-only routes moved: {documented_bytes}"
    changed = {k: v for k, v in changed.items() if k not in documented_bytes}
    assert not changed, f"a route's serializer changed (MF9): {changed}"
    # A route added after 4bb6d56 is pinned when it lands; until then it is listed here.
    assert not added, f"routes not in the pin (record their serializer): {added}"


VALUES: dict[str, Any] = {
    "at": datetime(2026, 11, 26, 12, 0, 0, 123456, tzinfo=UTC),
    "naive": datetime(2026, 11, 26, 12, 0, 0),
    "tiny": 1e-20,
    "huge": 1e20,
    "exact": 0.1,
    "money": Decimal("1.10"),
    "span": timedelta(seconds=90, microseconds=5),
    "nan": float("nan"),
    "inf": float("inf"),
}


class _Doc(BaseModel):
    at: str


def _app() -> FastAPI:
    app = FastAPI()

    @app.get("/bare")
    def bare() -> dict[str, Any]:
        return dict(VALUES)

    @app.get("/documented", responses={200: {"model": _Doc}})
    def documented() -> dict[str, Any]:
        return dict(VALUES)

    @app.get("/unmodelled", response_model=None)
    def unmodelled() -> dict[str, Any]:
        return dict(VALUES)

    return app


def _answer(client: TestClient, path: str) -> tuple[int, bytes]:
    response = client.get(path)
    return response.status_code, response.content


def test_the_documented_pattern_serves_the_bare_routes_bytes() -> None:
    with TestClient(_app(), raise_server_exceptions=False) as client:
        bare = _answer(client, "/bare")
        documented = _answer(client, "/documented")
        unmodelled = _answer(client, "/unmodelled")
    assert documented == bare, (documented, bare)
    # The hazard the rule prevents: without the response field, the bytes (or the status) change.
    assert unmodelled != bare, "the cases no longer tell the two serializers apart"


def test_a_returned_response_ignores_the_declared_response_class() -> None:
    """Why :data:`BYTES_ONLY` may declare ``response_class=Response``: the bytes, status and
    headers of a returned ``Response`` are the same with the declaration and without it."""
    from fastapi import Response

    app = FastAPI()

    def answer() -> Response:
        return Response(b"\x00\x01raw", media_type="application/octet-stream", headers={"X-A": "1"})

    app.get("/default")(answer)
    app.get("/declared", response_class=Response)(answer)
    with TestClient(app) as client:
        default, declared = client.get("/default"), client.get("/declared")
    assert (default.status_code, default.content) == (declared.status_code, declared.content)
    assert dict(default.headers) == dict(declared.headers)
