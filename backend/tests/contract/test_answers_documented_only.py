"""The routes are typed in documentation only, and the models say what they leave open (021 P3).

- FR-021.5 (MF9 as amended 2026-11-26): every route the contract types declares a 2xx model in
  ``responses={…}`` and keeps the response model it served before 021 (an untyped route keeps the
  inferred ``dict[str, Any]``; ``test_answers_keep_their_serializer.py`` pins every one). The 7
  typed and enforced before 021 keep theirs and gain a documentation twin (FT2). The routes that
  answer only bytes are named (P4).
- FR-021.6: a map accepts a key it has not seen; a model refuses one (the strict form).
- FR-021.4 and FR-021.6: a schema that types nothing (``{}``, an open object) is allowed only
  where a field says why it is free-form (``x-free-form``, ``answers_common.free_form``).
- The pending-approval models (FTASKS 3.1, 3.2): ``ApprovalObject`` agrees with the vendored result
  language, and ``gated_responses`` renders one schema per 202 (FT3).
- An approval answer carries no secret-named key anywhere (FTDD §8; the validator's rule).
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

import pytest
from fastapi import FastAPI
from fastapi.routing import APIRoute
from pydantic import BaseModel

from api.route_table import mounted_api_routes
from schemas import answers_contract
from schemas.answers_common import FREE_FORM, ApprovalObject, ApprovalPending, gated_responses
from tests.support import answer_validator

REPO = Path(__file__).resolve().parents[3]
#: Typed and enforced before 021: each keeps its response_model and gains a twin (FT2, A3).
ENFORCED = {
    "GET /api/v1/approvals": ("ApprovalList", "ApprovalListDoc"),
    "GET /api/v1/approvals/{approval_id}": ("ApprovalView", "ApprovalViewDoc"),
    "POST /api/v1/approvals/{approval_id}/approve": ("ApprovalView", "ApprovalViewDoc"),
    "POST /api/v1/approvals/{approval_id}/reject": ("ApprovalView", "ApprovalViewDoc"),
    "GET /api/v1/packs": ("PackListing", "PackListingDoc"),
    "POST /api/v1/selectors/evaluate": ("EvaluateResponse", "EvaluateResponseDoc"),
    "GET /api/v1/selectors": ("list", None),
    # Typed and enforced on main before 021 reached them (train F3, 2026-11-26).
    "GET /api/v1/wsp/runs/{run_id}/record": ("WspRunRecord", None),
    "GET /api/v1/wsp/runs/{run_id}/ledger": ("WspRunLedger", None),
}
#: The routes that answer only bytes, listed under ``excluded`` (P4, FTASKS 5.7).
EXCLUDED = {
    "GET /api/v1/wsp/candidates/{candidate_id}/tensors": "safetensors bytes",
    "GET /api/v1/wsp/candidates/{candidate_id}/verdict/scores": "float32 bytes",
}


#: Routes whose endpoint returns a raw ``Response`` (no response field before 021 either). The
#: document routes (P4, FTASKS 5.5) declare their document; the two 204s answer no body.
RAW = {
    # Train F3 landed it with response_model=None (2b31b2f); it keeps that serializer (MF9).
    "POST /api/v1/wsp/runs/validate",
    "POST /api/v1/wsp/weight-ops/inspect",
    "POST /api/v1/wsp/bundles/import",
    "GET /api/v1/packs/{pack_id}/versions/{version}/export",
    "GET /api/v1/wsp/candidates/{candidate_id}/patch/{name}",
    "DELETE /api/v1/packs/{pack_id}/versions/{version}",
    "DELETE /api/v1/pack-drafts/{draft_id}",
    *EXCLUDED,
}


def _routes() -> dict[str, APIRoute]:
    from api.app import create_app

    out: dict[str, APIRoute] = {}
    for ctx in mounted_api_routes(create_app()):
        route = ctx.original_route
        if isinstance(route, APIRoute) and route.include_in_schema:
            for method in ctx.methods or ():
                out[f"{method} {ctx.path_format}"] = route
    return out


def _declared_models(route: APIRoute) -> list[Any]:
    return [
        spec["model"]
        for status, spec in (route.responses or {}).items()
        if str(status).startswith("2") and isinstance(spec, dict) and "model" in spec
    ]


def test_every_route_is_documented_only_or_enforced_with_a_twin() -> None:
    routes = _routes()
    assert set(ENFORCED) <= set(routes) and RAW <= set(routes)
    for key, route in sorted(routes.items()):
        if key in EXCLUDED:
            assert not _declared_models(route), f"{key} answers bytes, yet declares a model"
            continue
        if key in ENFORCED:
            enforced, twin = ENFORCED[key]
            name = getattr(route.response_model, "__name__", repr(route.response_model))
            assert name == enforced, f"{key}: the enforced model changed to {name}"
            if twin:
                assert [m.__name__ for m in _declared_models(route)] == [twin], key
            continue
        assert route.response_model is not None or key in RAW, f"{key}: response_model=None (MF9)"
        assert _declared_models(route) or route.status_code == 204, f"{key} declares no 2xx model"


def test_only_the_byte_routes_are_excluded_and_nothing_is_untyped() -> None:
    """FR-021.3, FR-021.4 (FTASKS 4.11): since P4 every JSON route is typed, and only the routes
    that answer bytes are excluded."""
    contract = json.loads(answers_contract.CONTRACT_PATH.read_text(encoding="utf-8"))
    assert answers_contract.untyped_routes(contract) == []
    assert set(contract["excluded"]) == set(EXCLUDED)


def test_a_map_accepts_a_new_key_and_a_model_refuses_one(world: Any) -> None:
    answer = world.client.get("/api/v1/settings")
    assert answer.status_code == 200, answer.text
    body = answer.json()
    check = answer_validator.answer_check()
    key = "GET /api/v1/settings"
    assert check.check(key, 200, body) == []
    first = next(iter(body["settings"]))
    grown = {
        **body,
        "settings": {**body["settings"], "a_setting_added_later": body["settings"][first]},
    }
    assert check.check(key, 200, grown) == [], "a map must accept a key it has not seen"
    planted = check.check(key, 200, {**body, "planted": True})
    assert [f.message for f in planted] == [
        "undeclared key 'planted' (model SettingReadSettingsAnswer)"
    ], [f.message for f in planted]


def _open(schema: dict[str, Any]) -> bool:
    """A schema that types nothing: ``{}``, ``JsonValue``, or an object open to any key."""
    if not schema or schema.get("$ref") == "#/$defs/JsonValue":
        return True
    if schema.get("type") == "object" and not schema.get("properties"):
        extra = schema.get("additionalProperties")
        return extra is True or extra == {} or extra == {"$ref": "#/$defs/JsonValue"}
    return False


def _unexplained(node: Any, where: str, explained: bool) -> list[str]:
    out: list[str] = []
    if isinstance(node, dict):
        explained = explained or FREE_FORM in node
        if _open(node) and not explained:
            out.append(where)
        for name, value in node.items():
            out += _unexplained(value, f"{where}.{name}", explained)
    elif isinstance(node, list):
        for index, item in enumerate(node):
            out += _unexplained(item, f"{where}[{index}]", explained)
    return out


def test_every_open_schema_says_why_it_is_free_form() -> None:
    contract = json.loads(answers_contract.CONTRACT_PATH.read_text(encoding="utf-8"))
    found = _unexplained(contract["responses"], "responses", False)
    for name, definition in contract["$defs"].items():
        if name == "JsonValue":
            continue
        if definition.get("properties"):
            found += _unexplained(definition["properties"], f"$defs.{name}", False)
    assert not found, f"{len(found)} schema(s) type nothing and say not why: {found[:15]}"


def test_the_approval_object_agrees_with_the_result_language() -> None:
    vendored = json.loads(
        (REPO / "docs" / "schemas" / "mistudio-result-language-v1.json").read_text("utf-8")
    )["$defs"]["approval"]
    fields = set(ApprovalObject.model_fields)
    assert fields <= set(vendored["properties"]), sorted(fields - set(vendored["properties"]))
    assert set(vendored["required"]) <= fields, sorted(set(vendored["required"]) - fields)


class _Own(BaseModel):
    run_id: str


def test_gated_responses_renders_one_schema_per_202() -> None:
    app = FastAPI()

    @app.post("/own", status_code=202, response_model=None, responses=gated_responses(_Own))
    def own() -> dict[str, str]:  # pragma: no cover - never called
        return {}

    @app.post("/plain", response_model=None, responses={200: {"model": _Own}, **gated_responses()})
    def plain() -> dict[str, str]:  # pragma: no cover - never called
        return {}

    paths = app.openapi()["paths"]
    union = paths["/own"]["post"]["responses"]["202"]["content"]["application/json"]["schema"]
    pending = paths["/plain"]["post"]["responses"]["202"]["content"]["application/json"]["schema"]
    assert union == {"$ref": "#/components/schemas/PendingOr_Own"}
    assert pending == {"$ref": "#/components/schemas/ApprovalPending"}
    assert gated_responses(_Own)[202]["model"] is gated_responses(_Own)[202]["model"], "cached"
    assert "approval_id" not in _Own.model_fields and "approval_id" in ApprovalPending.model_fields


def test_a_secret_named_key_in_an_approval_answer_fails() -> None:
    check = answer_validator.answer_check()
    key = "GET /api/v1/approvals/{approval_id}"
    found = check.check(key, 200, {"payload": {"body": {"api_key": "x", "key_ciphertext": "y"}}})
    named = sorted(
        re.findall(r"'(\w+)'", " ".join(f.message for f in found if "secret" in f.message))
    )
    assert named == ["api_key", "key_ciphertext"], [f.message for f in found]
    empty = check.check(key, 200, {"payload": {"body": {"api_key": None}}})
    assert not [f for f in empty if "secret" in f.message], "a null key carries no secret"


@pytest.mark.parametrize("name", sorted(ENFORCED))
def test_each_enforced_route_is_typed_in_the_contract(name: str) -> None:
    contract = json.loads(answers_contract.CONTRACT_PATH.read_text(encoding="utf-8"))
    assert name not in answers_contract.untyped_routes(contract)
