"""A write COMMITS BEFORE its answer is sent, on every route of every app (2026-11-19).

**The defect.** ``api.deps.session_dependency`` is a ``yield`` dependency that commits in
``session_scope()``'s exit. FastAPI's default dependency scope (``"request"``) runs that exit
AFTER the response has been sent. So a client that wrote and at once read could miss its own
write, and a commit that FAILED still reached the client as a 2xx for a write that was rolled
back. A sibling branch met it as a real race (016-P2: a gateway settle arriving before its
reserve had committed, 1 request in 15-40 on PostgreSQL) and fixed two places; this module pins
the fix for every route: the session comes through ONE alias, ``api.deps.SessionDep``, declared
``Depends(session_dependency, scope="function")``.

What is pinned, each by its own test:

* **the wiring, read from the LIVE route tables** of every application this backend serves (the
  main API, the WSP internal app, the WSP gateway, the MCP HTTP app), recursing into mounts and
  every sub-dependency: each ``yield`` dependency is function-scoped, and at least
  :data:`MIN_SESSION_ROUTES` routes were seen taking the session, so an empty walk cannot pass;
* **a failed commit is not a 2xx**, over a real server (``TestClient`` runs the app to completion
  before it answers, so it cannot show what a client is sent first): a run archive, a WSP endpoint
  registration and the internal app's reserve, each answering the house ``internal_error``
  envelope with the row rolled back;
* **write, then read at once from another connection, sees the row** (the commit is slowed so
  the order is deterministic, not a race). CI's PostgreSQL shards run every test here against
  PostgreSQL 15; the default run uses SQLite.
"""

from __future__ import annotations

import contextlib
import inspect
import time
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import httpx
import pytest
import sqlalchemy as sa
from fastapi.routing import APIRoute, iter_route_contexts
from sqlalchemy.exc import OperationalError
from sqlalchemy.orm import Session, sessionmaker
from starlette.routing import Mount

from core.config import reset_settings
from tests.integration.wsp.test_dispatch import world as dispatch_world  # noqa: F401
from tests.integration.wsp.test_internal_app_routes import (
    AUTH,
    PREFIX,
    _reserve,
    _running,
    inner,  # noqa: F401
)
from tests.support import db as support
from tests.support.jobs import insert_run
from tests.support.served import served
from workers.run_record import RUNS
from wsp.settings import get_wsp_settings
from wsp_gateway import sealing

#: The main API serves 102 routes taking the session (2026-11-19: 129 API routes, of which the
#: rest take none, e.g. ``/host`` and the tokenizer reads; 92 of 119 before 016-P2 added the ten
#: WSP candidate routes). A floor close to the real count, so a walk that silently finds
#: nothing, or most of nothing, fails.
MIN_SESSION_ROUTES = 100
#: The WSP internal app: sessions, by-token, endpoint, reserve, settle, refused (2026-11-20).
INTERNAL_SESSION_ROUTES = 6


# ======================================================================================
# The wiring, from the live route tables
# ======================================================================================


def _apps() -> dict[str, Any]:
    """Every ASGI application this backend serves, built by its own factory."""
    from api.app import create_app
    from api.wsp_internal_app import create_internal_app
    from mcp_server.config import MCPSettings
    from mcp_server.server import build_http_app, build_server
    from wsp_gateway.app import Gateway, create_gateway_app
    from wsp_gateway.settings import GatewaySettings

    mcp_settings = MCPSettings(auth_token="t" * 40)
    mcp, _ = build_server(mcp_settings)
    gateway = Gateway(GatewaySettings(), internal=httpx.Client(), upstream_client=httpx.Client())
    return {
        "api": create_app(),
        "wsp-internal": create_internal_app(),
        "wsp-gateway": create_gateway_app(gateway),
        "mcp": build_http_app(mcp, mcp_settings),
    }


def _api_routes(routes: Any) -> Iterator[Any]:
    """Every API route as SERVED (included routers flattened, mounts entered)."""
    for route in iter_route_contexts(routes):
        original = route.original_route
        if isinstance(original, APIRoute):
            yield route
        elif isinstance(original, Mount):
            yield from _api_routes(original.routes)


def _dependencies(dependant: Any) -> Iterator[Any]:
    for dep in dependant.dependencies:
        yield dep
        yield from _dependencies(dep)


def _is_yield(call: Any) -> bool:
    return inspect.isgeneratorfunction(call) or inspect.isasyncgenfunction(call)


def test_every_yield_dependency_of_every_app_ends_before_the_answer() -> None:
    """The EFFECTIVE dependant of each served route (``route.dependant``, which carries a
    router's own dependencies too), every level down: a ``yield`` dependency is function-scoped,
    so its exit (``session_scope``'s commit) runs before the response is sent."""
    seen: dict[str, int] = {}
    late: list[str] = []
    for name, app in _apps().items():
        for route in _api_routes(app.routes):
            for dep in _dependencies(route.dependant):
                if dep.call is None or not _is_yield(dep.call):
                    continue
                key = f"{name}:{dep.call.__module__}.{dep.call.__qualname__}"
                seen[key] = seen.get(key, 0) + 1
                if dep.scope != "function":
                    late.append(f"{name} {sorted(route.methods)} {route.path}: {key} {dep.scope}")
    assert late == [], "a yield dependency still ends after the response:\n" + "\n".join(late)
    assert seen.get("api:api.deps.session_dependency", 0) >= MIN_SESSION_ROUTES, seen
    assert seen.get("wsp-internal:api.deps.session_dependency") == INTERNAL_SESSION_ROUTES, seen
    # Only the one session dependency exists: a second yield dependency is a new decision.
    assert {key.split(":", 1)[1] for key in seen} == {"api.deps.session_dependency"}, seen


# ======================================================================================
# Behaviour, over a real server
# ======================================================================================


class _CommitFails(Session):
    """A session whose commit raises, as a lost connection or a deferred constraint does."""

    def commit(self) -> None:
        raise OperationalError("COMMIT", {}, Exception("injected commit failure"))


def _bind(monkeypatch: pytest.MonkeyPatch, engine: sa.Engine, session_class: type) -> None:
    """Make the REAL ``session_dependency`` -> ``session_scope`` open sessions on ``engine``."""
    from db import session as db_session

    monkeypatch.setattr(db_session, "_engine", engine)
    monkeypatch.setattr(
        db_session,
        "_session_factory",
        sessionmaker(bind=engine, expire_on_commit=False, class_=session_class),
    )


@pytest.fixture()
def api_engine(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Iterator[sa.Engine]:
    for name in ("CHECKPOINT_ROOT", "PARTS_ROOT", "TRAINER_STATE_ROOT"):
        monkeypatch.setenv(name, str(tmp_path / name.lower()))
    monkeypatch.delenv("MIDATAWORKS_BASE_URL", raising=False)
    keys = sealing.generate_keypair()
    monkeypatch.setenv("WSP_GATEWAY_PUBLIC_KEY", keys[1])
    reset_settings()
    get_wsp_settings.cache_clear()
    engine = support.migrated_engine(tmp_path / "db")
    with support.session_for(engine) as session:
        insert_run(session, "finished", status="completed")
    try:
        yield engine
    finally:
        engine.dispose()
        reset_settings()
        get_wsp_settings.cache_clear()


def _archived_at(engine: sa.Engine, run_id: str = "finished") -> Any:
    with engine.connect() as conn:
        return conn.execute(sa.select(RUNS.c.archived_at).where(RUNS.c.id == run_id)).scalar_one()


def _is_internal_error(response: httpx.Response) -> bool:
    body = response.json()
    return response.status_code == 500 and body["error"]["code"] == "internal_error"


def test_a_failed_commit_is_not_answered_2xx_archive(
    api_engine: sa.Engine, monkeypatch: pytest.MonkeyPatch
) -> None:
    """``POST /runs/{id}/archive``: the commit fails, so the client is told so, and the run is
    still unarchived. Under the default scope the client got 200 for a write that was rolled
    back."""
    from api.app import create_app

    _bind(monkeypatch, api_engine, _CommitFails)
    with served(create_app()) as url, httpx.Client(base_url=url, timeout=30) as http:
        answer = http.post("/api/v1/runs/finished/archive")
    assert _is_internal_error(answer), (answer.status_code, answer.text)
    assert _archived_at(api_engine) is None


def test_a_failed_commit_is_not_answered_2xx_wsp_endpoint(
    api_engine: sa.Engine, monkeypatch: pytest.MonkeyPatch
) -> None:
    """``POST /wsp/endpoints`` (the main app's WSP surface): same, and no endpoint row."""
    from api.app import create_app
    from tests.integration.wsp.test_endpoints_api import ANTHROPIC

    _bind(monkeypatch, api_engine, _CommitFails)
    with served(create_app()) as url, httpx.Client(base_url=url, timeout=30) as http:
        answer = http.post("/api/v1/wsp/endpoints", json=ANTHROPIC)
    assert _is_internal_error(answer), (answer.status_code, answer.text)
    with api_engine.connect() as conn:
        assert conn.execute(sa.text("SELECT COUNT(*) FROM mf_wsp_endpoints")).scalar_one() == 0


def test_a_failed_commit_is_not_answered_2xx_internal_reserve(
    inner: Any,  # noqa: F811
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The WSP internal app's ``reserve``: a reservation the gateway is told exists must exist."""
    from api.wsp_internal_app import create_internal_app

    world, _client = inner
    run_id, token = _running(world)
    _bind(monkeypatch, world.engine, _CommitFails)
    with served(create_internal_app()) as url, httpx.Client(base_url=url, headers=AUTH) as http:
        answer = http.post(f"{PREFIX}/reserve", json=_reserve(run_id, token))
    assert _is_internal_error(answer), (answer.status_code, answer.text)
    with world.engine.connect() as conn:
        rows = conn.execute(sa.text("SELECT COUNT(*) FROM mf_wsp_reservations"))
        assert rows.scalar_one() == 0


def test_a_write_is_visible_to_another_connection_the_moment_it_is_answered(
    api_engine: sa.Engine, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Write, then read at once on a separate connection: the row is there. The commit is
    slowed by 0.3 s so the order is deterministic; under the default scope the read ran first."""
    from api.app import create_app
    from db import session as db_session

    _bind(monkeypatch, api_engine, Session)
    real_scope = db_session.session_scope

    @contextlib.contextmanager
    def slow_commit() -> Iterator[Session]:
        with real_scope() as session:
            yield session
            time.sleep(0.3)  # before session_scope's own commit

    monkeypatch.setattr(db_session, "session_scope", slow_commit)
    with served(create_app()) as url, httpx.Client(base_url=url, timeout=30) as http:
        answer = http.post("/api/v1/runs/finished/archive")
        assert answer.status_code == 200, answer.text
        seen = _archived_at(api_engine)
    assert seen is not None
    assert answer.json()["archived"] is True
