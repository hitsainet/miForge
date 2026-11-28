# Origin: miDataworks (Onegaishimas/miDataworks)
#   backend/tests/unit/contract/test_contract_tools.py @ 6b50a8f4
# Mode: adapt (docs/REUSE.md). Kept: every transform and marker test. Changed: miForge's
# imports; no closed_enum or literal_of (021 publishes no enum); the external document is the
# vendored dataset-version schema.
"""The answer contract's pure transforms and markers (021 FTID §3.1, §3.3; FTASKS 1.1, 1.2, 1.9).

Each function of ``backend/src/schemas/contract_tools.py`` on synthetic schemas, then the markers
of ``backend/src/schemas/answers_common.py`` through ``answers_contract.build`` on a small app, so
what a marker publishes is read from a built contract, not from the marker's own return value.
"""

from __future__ import annotations

import ast
import sys
from pathlib import Path
from typing import Any

import pytest
from fastapi import APIRouter, FastAPI
from fastapi.routing import iter_route_contexts
from pydantic import BaseModel

from schemas import answers_common, answers_contract
from schemas import contract_tools as tools

DEFS = {
    "Item": {"type": "object", "properties": {"n": {"type": "integer"}}},
    "Map": {"type": "object", "additionalProperties": {"type": "integer"}},
    "Open": {"type": "object", "additionalProperties": True},
    "Null": {"type": "null"},
    "Chain": {"$ref": "#/$defs/Item"},
    # A model that IS a published document, or one part of one (021 P4: answers_documents).
    "Doc": {"$ref": "https://example.test/doc.json"},
    "DocPart": {"$ref": "https://example.test/doc.json#/$defs/Part"},
    "Elsewhere": {"$ref": "https://example.test/unknown.json#/$defs/Part"},
}


def test_contract_tools_imports_only_the_standard_library() -> None:
    source = Path(tools.__file__).read_text(encoding="utf-8")
    modules: set[str] = set()
    for node in ast.walk(ast.parse(source)):
        if isinstance(node, ast.Import):
            modules |= {alias.name.split(".")[0] for alias in node.names}
        elif isinstance(node, ast.ImportFrom):
            assert node.level == 0, f"a relative import: {ast.dump(node)}"
            modules.add((node.module or "").split(".")[0])
    outside = sorted(m for m in modules if m not in sys.stdlib_module_names and m != "__future__")
    assert modules and not outside, outside


def test_rewrite_refs_and_refs() -> None:
    node = {"a": {"$ref": "#/components/schemas/X"}, "b": [{"$ref": "#/components/schemas/Y"}]}
    rewritten = tools.rewrite_refs(node)
    assert rewritten == {"a": {"$ref": "#/$defs/X"}, "b": [{"$ref": "#/$defs/Y"}]}
    assert node["a"]["$ref"] == "#/components/schemas/X", "rewrite_refs must not edit its input"
    assert tools.refs(rewritten) == {"X", "Y"}
    assert tools.refs({"$ref": "https://example.test/x.json"}) == set()


def test_reachable_defs_keeps_only_what_is_reached_and_refuses_a_dangling_reference() -> None:
    kept = tools.reachable_defs([{"$ref": "#/$defs/Chain"}], DEFS)
    assert list(kept) == ["Chain", "Item"]
    with pytest.raises(KeyError):
        tools.reachable_defs([{"$ref": "#/$defs/Missing"}], DEFS)


def test_always_sent_required_honours_the_marker() -> None:
    node = {
        "type": "object",
        "properties": {"a": {"type": "string"}, "b": {"type": "string", tools.MAY_BE_ABSENT: True}},
    }
    tools.always_sent_required(node)
    assert node["required"] == ["a"]
    assert tools.MAY_BE_ABSENT not in node["properties"]["b"]
    only_absent = {"properties": {"b": {tools.MAY_BE_ABSENT: True}}, "required": ["b"]}
    tools.always_sent_required(only_absent)
    assert "required" not in only_absent


def test_open_objects_drops_only_false() -> None:
    node = {
        "properties": {"m": {"additionalProperties": {"type": "integer"}}},
        "additionalProperties": False,
    }
    tools.open_objects(node)
    assert node == {"properties": {"m": {"additionalProperties": {"type": "integer"}}}}


def test_strict_form_closes_objects_and_leaves_maps_and_views_open() -> None:
    schema = {
        "type": "object",
        "properties": {
            "inner": {"type": "object", "properties": {"x": {"type": "integer"}}},
            "map": {"type": "object", "additionalProperties": {"type": "integer"}},
            "declared": {"type": "object", "properties": {"y": {}}, "additionalProperties": True},
        },
        "allOf": [{"properties": {"z": {"type": "integer"}}}],
    }
    strict = tools.strict_form(schema)
    assert strict[tools.OPEN_KEY] is False
    assert strict["properties"]["inner"][tools.OPEN_KEY] is False
    assert strict["properties"]["map"][tools.OPEN_KEY] == {"type": "integer"}
    assert strict["properties"]["declared"][tools.OPEN_KEY] is True
    assert tools.OPEN_KEY not in strict["allOf"][0], "an allOf member is a view of one object"
    assert tools.OPEN_KEY not in schema, "strict_form must work on a copy"


@pytest.mark.parametrize(
    ("schema", "typed"),
    [
        ({"$ref": "#/$defs/Item"}, True),
        ({"$ref": "#/$defs/Chain"}, True),
        ({"type": "array", "items": {"$ref": "#/$defs/Item"}}, True),
        ({"anyOf": [{"$ref": "#/$defs/Item"}, {"type": "null"}]}, True),
        ({"oneOf": [{"$ref": "#/$defs/Item"}, {"$ref": "#/$defs/Null"}]}, True),
        ({"$ref": "https://example.test/doc.json"}, True),
        ({"$ref": "https://example.test/unknown.json"}, False),
        ({"$ref": "https://example.test/doc.json#/$defs/Part"}, True),  # a part of a document
        ({"$ref": "#/$defs/Doc"}, True),
        ({"$ref": "#/$defs/DocPart"}, True),
        ({"$ref": "#/$defs/Elsewhere"}, False),
        ({}, False),
        ({"type": "object", "additionalProperties": True}, False),  # dict[str, Any]
        ({"$ref": "#/$defs/Open"}, False),
        ({"$ref": "#/$defs/Map"}, False),  # a map at the top
        ({"type": "array", "items": {}}, False),
        ({"anyOf": [{"$ref": "#/$defs/Item"}, {}]}, False),
        ({"anyOf": [{"type": "null"}]}, False),
        ({"type": "string"}, False),
    ],
)
def test_is_typed(schema: dict[str, Any], typed: bool) -> None:
    assert tools.is_typed(schema, DEFS, {"https://example.test/doc.json"}) is typed


def test_members_flattens_unions_through_references() -> None:
    schema = {
        "anyOf": [{"$ref": "#/$defs/Item"}, {"oneOf": [{"$ref": "#/$defs/Map"}]}, {"type": "null"}]
    }
    assert tools.members(schema, DEFS) == [{"$ref": "#/$defs/Item"}, {"$ref": "#/$defs/Map"}]


def test_content_digest_ignores_the_two_writer_keys_and_render_is_canonical() -> None:
    doc = {"b": 1, "a": "é", "revision": 3, "content_sha256": "x"}
    assert tools.content_digest(doc) == tools.content_digest({"a": "é", "b": 1})
    assert tools.content_digest(doc) != tools.content_digest({"a": "é", "b": 2})
    assert tools.render({"b": 1, "a": "é"}) == '{\n  "a": "é",\n  "b": 1\n}\n'


# -- the markers, read from a built contract -----------------------------------------------------

#: A published document this repository carries under docs/schemas (vendored from miDataworks).
DOCUMENT_ID = (
    "https://raw.githubusercontent.com/hitsainet/miDataworks/main/docs/schemas/"
    "midataworks-dataset-version-v1.json"
)


class Marked(BaseModel):
    always: str
    sometimes: str | None = answers_common.may_be_absent()
    document: dict[str, Any] = answers_common.external_ref(DOCUMENT_ID, description="the document")


def _contract_of(router: APIRouter) -> dict[str, Any]:
    app = FastAPI()
    app.include_router(router)
    return answers_contract.build(app.openapi(), iter_route_contexts(app.routes))


def _marked_contract() -> dict[str, Any]:
    router = APIRouter()

    @router.get("/marked", response_model=Marked)
    async def marked() -> dict[str, Any]:  # pragma: no cover - never called
        return {}

    return _contract_of(router)


def test_markers_publish_what_they_say() -> None:
    doc = _marked_contract()
    model = doc["$defs"]["Marked"]
    assert model["required"] == ["always", "document"]
    assert model["properties"]["document"] == {"$ref": DOCUMENT_ID, "description": "the document"}
    file = doc["external"][DOCUMENT_ID]["file"]
    assert file == "docs/schemas/midataworks-dataset-version-v1.json"
    assert doc["coverage"] == {"typed": 1, "unions": 0, "external": 1, "excluded": 0, "untyped": 0}
    assert set(doc["errors"]) == {"app"} and "ErrorEnvelope" in doc["$defs"]


def test_an_unknown_external_reference_fails_closed() -> None:
    class Unknown(BaseModel):
        document: dict[str, Any] = answers_common.external_ref("https://example.test/none.json")

    router = APIRouter()

    @router.get("/unknown", response_model=Unknown)
    async def unknown() -> dict[str, Any]:  # pragma: no cover - never called
        return {}

    with pytest.raises(ValueError, match="unknown external"):
        _contract_of(router)


def test_a_route_that_omits_fields_by_flag_fails_closed() -> None:
    router = APIRouter()

    @router.get("/sparse", response_model=Marked, response_model_exclude_none=True)
    async def sparse() -> dict[str, Any]:  # pragma: no cover - never called
        return {}

    with pytest.raises(ValueError, match="always sent"):
        _contract_of(router)
