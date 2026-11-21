"""Ask the LIVE gate what it would decide for the call a tool makes  (G1, G5, G9).

A tool's approval class must not be a second hand-kept list. So for each tool the ledger's
recorded call is mapped onto the mounted app's route, turned into that route's value
parameters exactly as FastAPI would hand them over, and passed to the gate's own decision
function (``__mf_decide__``, the closure the decorator built). The answer - ``gpu_start``,
``hotupdate_attach``, ``delete``, ``config_write`` or ``None`` - is what an agent's call
would actually wait for.
"""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from functools import cache
from typing import Any

from fastapi.routing import APIRoute
from pydantic import TypeAdapter

from api.route_table import mounted_api_routes
from tests.support.mcp_ledger import EXPECTED_CALLS, full, template_matches


@cache
def app_routes() -> tuple[APIRoute, ...]:
    from api.app import create_app

    return tuple(route for route in mounted_api_routes(create_app()))


def template(route: APIRoute) -> str:
    """The route's path as the OpenAPI document spells it (the ``:path`` converter dropped)."""
    return route.path.replace(":path}", "}")


def route_for(method: str, path: str) -> APIRoute | None:
    """The route serving ``method path``; a literal segment beats a parameter."""
    concrete = full(path)
    matches = [
        route
        for route in app_routes()
        if method in route.methods and template_matches(template(route), concrete)
    ]
    return min(matches, key=lambda route: template(route).count("{"), default=None)


def values_for(route: APIRoute, path: str, recorded: dict[str, Any]) -> dict[str, Any]:
    """The route's value parameters for this call, as FastAPI would parse them."""
    from api.agent_origin import EXECUTORS

    executor = EXECUTORS[route.endpoint.__mf_executor_key__]  # type: ignore[attr-defined]
    match = route.path_regex.match(full(path))
    assert match is not None, (route.path, path)
    raw: dict[str, Any] = dict(match.groupdict())
    body = recorded.get("json_body")
    raw.update({k: v for k, v in recorded.items() if k != "json_body"})
    body_params = route.dependant.body_params
    if len(body_params) == 1 and not getattr(body_params[0].field_info, "embed", False):
        raw[body_params[0].name] = body
    else:
        for param in body_params:
            raw[param.name] = None if body is None else body.get(param.name)
    return {
        name: TypeAdapter(annotation).validate_python(raw.get(name))
        for name, annotation in executor.value_params.items()
        if name in raw or raw.get(name) is not None
    }


def decision(tool: str, *, recorded: dict[str, Any] | None = None) -> tuple[APIRoute, str | None]:
    """``(route, action)`` - what the live gate decides for the tool's documented call."""
    method, path, _args, ledger_recorded, _class = EXPECTED_CALLS[tool]
    route = route_for(method, path)
    assert route is not None, f"{tool}: no mounted route serves {method} {full(path)}"
    decide = getattr(route.endpoint, "__mf_decide__", None)
    if decide is None:
        return route, None
    values = values_for(route, path, ledger_recorded if recorded is None else recorded)
    if not path.startswith("/wsp/"):
        return route, decide(values)
    with wsp_endpoint_store():
        return route, decide(values)


#: The editor endpoint the WSP ledger rows name: a LOCAL one (miServer in-cluster).
WSP_LEDGER_ENDPOINT = {
    "id": "ep-local",
    "kind": "openai_compatible",
    "base_url": "http://miserver-backend.miserver.svc:8000/v1",
    "model": "qwen",
    "locality": "local",
    "output_cap_field": "max_tokens",
    "serving": "miserver",
    "created_by": "operator",
}


#: Feature 018: the EXTERNAL endpoint the protocol test's ledger row names (its call waits only
#: for an external endpoint). Priced, as the CHECK requires; no key is needed for a decision.
WSP_LEDGER_EXTERNAL_ENDPOINT = {
    "id": "ep-ext",
    "kind": "anthropic",
    "base_url": "https://api.anthropic.com",
    "model": "claude-editor",
    "locality": "external",
    "output_cap_field": "max_tokens",
    "serving": "generic",
    "created_by": "operator",
    "price_in_per_mtok": 3,
    "price_out_per_mtok": 15,
}


@contextmanager
def wsp_endpoint_store() -> Iterator[None]:
    """The WSP gate's predicates read the STORED endpoint through their own session (P52), so
    the probe gives them a migrated SQLite database holding the ledger's endpoint, bound as the
    process engine for the duration of the decision only. Without it the predicates would reach
    whatever ``DATABASE_URL`` names and fail closed - a decision about the environment."""
    import datetime as dt
    import tempfile
    from pathlib import Path

    import sqlalchemy as sa

    from db import session as db_session
    from tests.support.db import migrated_engine

    saved = (db_session._engine, db_session._session_factory)
    with tempfile.TemporaryDirectory(prefix="mf-wsp-probe-") as tmp:
        engine = migrated_engine(Path(tmp) / "db")
        with engine.begin() as conn:
            for row in (WSP_LEDGER_ENDPOINT, WSP_LEDGER_EXTERNAL_ENDPOINT):
                values = {**row, "created_at": dt.datetime.now(dt.UTC)}
                conn.execute(
                    sa.text(
                        f"INSERT INTO mf_wsp_endpoints ({', '.join(values)}) VALUES "
                        f"({', '.join(':' + k for k in values)})"
                    ),
                    values,
                )
        db_session._engine, db_session._session_factory = engine, None
        try:
            yield
        finally:
            db_session._engine, db_session._session_factory = saved
            engine.dispose()
