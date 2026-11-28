# Origin: miDataworks (Onegaishimas/miDataworks)
#   backend/tests/unit/contract/test_answers_contract_coverage.py @ 30efbb68
# Mode: adapt (docs/REUSE.md). Kept: set equality, the media-type rule, the strict expected failure,
# the recount on a grown app, reachability. Changed: miForge's route table
# (api.route_table.mounted_api_routes; FastAPI 0.159) and counts; no attachment exception (021 FT4:
# exactly four operations are excluded once P4 declares them, FTASKS 1.5).
"""The answer contract covers the live route table exactly (021 FR-021.3, .4, .16, .22; FTASKS 1.5).

- The keys of ``responses`` and ``excluded`` together EQUAL every in-schema operation of
  ``mounted_api_routes(create_app())``, and neither holds a key the other does.
- An ``excluded`` entry's media type is the route's own declaration, never ``application/json``.
- Every ``responses`` entry is typed (FR-021.4). Until FTASKS 4.0 ends this is a strict expected
  failure, so the day it passes the marker must come off (4.11).
- ``coverage`` equals a recount, made on the live app plus one route added here, so a hard-coded
  count cannot pass.
- ``$defs`` holds exactly what ``responses`` and ``errors`` reach.

The generated body is checked (a writer that drops a route goes red here), and so is the committed
file (what consumers vendor).
"""

from __future__ import annotations

import json
from typing import Any

import pytest
from fastapi.routing import APIRoute
from pydantic import BaseModel

from api.route_table import mounted_api_routes
from schemas import answers_contract
from schemas import contract_tools as tools

JSON = "application/json"
#: The served operations on 2026-11-26 (`4bb6d56`, FTDD §0.1); a walk that finds fewer lost some.
OPERATIONS_AT_LEAST = 134  # re-measured at the rebased head (train F3's two reads), 2026-11-26


@pytest.fixture(scope="module")
def generated() -> dict[str, Any]:
    return answers_contract.answers_contract()


@pytest.fixture(scope="module")
def committed() -> dict[str, Any]:
    loaded: dict[str, Any] = json.loads(answers_contract.CONTRACT_PATH.read_text(encoding="utf-8"))
    return loaded


def _live_operations() -> set[str]:
    from api.app import create_app

    return {
        f"{method} {ctx.path_format}"
        for ctx in mounted_api_routes(create_app())
        if isinstance(ctx.original_route, APIRoute) and ctx.original_route.include_in_schema
        for method in ctx.methods or ()
        if method not in ("HEAD", "OPTIONS")
    }


@pytest.mark.parametrize("which", ["generated", "committed"])
def test_responses_and_excluded_equal_the_live_table(which: str, request: Any) -> None:
    doc = request.getfixturevalue(which)
    live = _live_operations()
    listed = set(doc["responses"]) | set(doc["excluded"])
    assert not set(doc["responses"]) & set(doc["excluded"]), "a route is in both"
    missing, extra = sorted(live - listed), sorted(listed - live)
    assert not missing and not extra, f"missing from the contract: {missing}; not live: {extra}"
    assert len(live) >= OPERATIONS_AT_LEAST, "the walk found fewer operations than the app serves"


def test_each_exclusion_is_the_routes_own_media_type(generated: dict[str, Any]) -> None:
    from api.app import create_app

    paths = create_app().openapi()["paths"]
    for key, row in generated["excluded"].items():
        method, path = key.split(" ", 1)
        operation = paths[path][method.lower()]
        declared = {
            media
            for status, response in operation["responses"].items()
            if status.startswith("2")
            for media in (response.get("content") or {})
        }
        assert row["media_type"] != JSON, f"{key}: a JSON route in excluded"
        if row["media_type"] is None:
            assert row["reason"] == "no body" and not declared, f"{key}: {row} vs {declared}"
        else:
            assert row["media_type"] in declared, f"{key}: {row['media_type']} vs {declared}"
            assert JSON not in declared, f"{key} declares JSON yet is excluded"


def test_every_json_route_is_typed(committed: dict[str, Any]) -> None:
    untyped = answers_contract.untyped_routes(committed)
    assert not untyped, f"{len(untyped)} untyped JSON route(s): {untyped}"
    assert committed["coverage"]["untyped"] == 0


def test_untyped_routes_are_counted(committed: dict[str, Any]) -> None:
    """The untyped routes are the ones ``coverage.untyped`` counts (none since P4)."""
    assert len(answers_contract.untyped_routes(committed)) == committed["coverage"]["untyped"]


class _Probe(BaseModel):
    probe: int


def _recount(doc: dict[str, Any]) -> dict[str, int]:
    defs, ext = doc["$defs"], frozenset(doc["external"])
    entries = list(doc["responses"].values())
    bodies = [(s, b) for e in entries for s, b in e.items() if s[:1].isdigit()]
    typed = sum(
        all(tools.is_typed(b, defs, ext) for s, b in e.items() if s[:1].isdigit()) for e in entries
    )

    def external(entry: dict[str, Any]) -> bool:
        reached = tools.reachable_defs([entry], defs)
        return any(
            str(s.get("$ref", "")).startswith("https://") for s in tools.walk([entry, reached])
        )

    return {
        "typed": typed,
        "untyped": len(entries) - typed,
        "unions": sum(len(tools.members(b, defs)) > 1 for _, b in bodies),
        "external": sum(external(e) for e in entries),
        "excluded": len(doc["excluded"]),
    }


def test_coverage_equals_a_recount_on_a_grown_app() -> None:
    from api.app import create_app

    app = create_app()

    @app.get("/api/v1/__coverage_probe", response_model=_Probe)
    async def coverage_probe() -> dict[str, int]:  # pragma: no cover - never called
        return {"probe": 1}

    doc = answers_contract.build(app.openapi(), mounted_api_routes(app))
    assert "GET /api/v1/__coverage_probe" in doc["responses"]
    assert doc["coverage"] == _recount(doc)


def test_the_committed_coverage_equals_a_recount(committed: dict[str, Any]) -> None:
    assert committed["coverage"] == _recount(committed)


@pytest.mark.parametrize("which", ["generated", "committed"])
def test_only_reachable_definitions_are_kept(which: str, request: Any) -> None:
    doc = request.getfixturevalue(which)
    reached = tools.reachable_defs([doc["responses"], doc["errors"]], doc["$defs"])
    unused = sorted(set(doc["$defs"]) - set(reached))
    assert not unused, f"{len(unused)} unreferenced definition(s): {unused[:20]}"
