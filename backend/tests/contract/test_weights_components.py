"""The vendored weights contract after miServer's package rename (feature 015 P8; FR-015.25).

miServer GENERATES ``docs/schemas/millm-weights-api-v1.json`` from its registered weights routes,
and pydantic names a component after the module that defines it. At suite P8 (miServer feature
041) miServer's Python package became ``miserver``, so the five generated component keys inside
the file changed their package prefix. The file name and its ``$id`` are frozen (DR-05, DR-04):
only the keys inside move, and miForge re-vendors the file byte for byte, never editing it.

``test_schema_sync.py`` proves the copy equals miServer's; it cannot run in CI (no sibling is
checked out there), so these checks pin the shape of the copy miForge actually holds:

- every generated component key carries the ``miserver`` package prefix, and there are exactly
  the five FR-015.25 names;
- every ``$ref`` resolves inside the file, and each response schema is a valid draft 2020-12
  schema against the file's ``$defs`` (what the fake weights surface is validated through);
- the re-vendor created no ``miserver-*`` file and did not change the ownership table.

The pre-rename prefix is deliberately not spelled here: the census retires it at P8 (DR-21), and
"every generated key starts ``miserver__``" is the stronger statement anyway, since it refuses
any other package prefix too.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from tests.contract import test_schema_sync as schema_sync

REPO_ROOT = Path(__file__).resolve().parents[3]
SCHEMAS = REPO_ROOT / "docs" / "schemas"
WEIGHTS = SCHEMAS / "millm-weights-api-v1.json"
WEIGHTS_ID = "https://hitsai.net/schemas/millm-weights-api-v1.json"

#: What marks a component key as generated from a miServer module path.
GENERATED = "__api__schemas__"
PREFIX = "miserver" + GENERATED

#: FR-015.25: the five generated components of the weights contract, after the rename.
FIVE = frozenset(
    PREFIX + suffix
    for suffix in (
        "common__ApiResponse_AttachResponse___2",
        "common__ApiResponse_DetachResponse___2",
        "weights__AttachResponse",
        "weights__AttachmentStatus",
        "weights__DetachResponse",
    )
)

#: ``test_schema_sync.OWNERSHIP``'s keys as they stood before P8. The re-vendor keeps every
#: file name (DR-05), so this set must not move.
OWNERSHIP_KEYS = frozenset(
    {
        "checkpoint-manifest-v1.json",
        "adapter-manifest-v1.json",
        "midataworks-dataset-version-v1.json",
        "millm-weights-api-v1.json",
        "millm-scoring-api-v1.json",
        "batch-record-v1.json",
        "step-record-v1.json",
        "selector-config-v1.json",
        "pack-definition-v1.json",
    }
)


def _contract() -> dict[str, Any]:
    document = json.loads(WEIGHTS.read_text("utf-8"))
    assert isinstance(document, dict), f"{WEIGHTS} is not a JSON object"
    return document


def _refs(node: Any) -> set[str]:
    found: set[str] = set()
    if isinstance(node, dict):
        if isinstance(node.get("$ref"), str):
            found.add(node["$ref"])
        for value in node.values():
            found |= _refs(value)
    elif isinstance(node, list):
        for value in node:
            found |= _refs(value)
    return found


def test_no_millm_component_keys() -> None:
    """No generated component key keeps a package prefix other than ``miserver`` (DR-21)."""
    keys = set(_contract()["$defs"])
    generated = {k for k in keys if GENERATED in k}
    assert generated, f"{WEIGHTS.name} has no generated component key; the check is vacuous"
    stale = sorted(k for k in generated if not k.startswith(PREFIX))
    assert not stale, (
        f"{WEIGHTS.name} still carries generated keys outside the miserver package: {stale}. "
        "Re-vendor it from miServer's docs/schemas/ (cp, never edit)."
    )


def test_exactly_five_miserver_components() -> None:
    keys = {k for k in _contract()["$defs"] if k.startswith(PREFIX)}
    assert (
        keys == FIVE
    ), f"miserver components: missing {sorted(FIVE - keys)}, unexpected {sorted(keys - FIVE)}"


def test_every_ref_resolves() -> None:
    """Every ``$ref`` points into ``$defs`` and resolves; each response validates as a schema."""
    jsonschema = pytest.importorskip("jsonschema")
    contract = _contract()
    assert contract.get("$id") == WEIGHTS_ID, f"$id changed (DR-04): {contract.get('$id')!r}"
    refs = _refs(contract)
    assert refs, f"{WEIGHTS.name} has no $ref; the check is vacuous"
    unresolved = sorted(
        ref
        for ref in refs
        if not ref.startswith("#/$defs/") or ref.removeprefix("#/$defs/") not in contract["$defs"]
    )
    assert not unresolved, f"unresolved $ref in {WEIGHTS.name}: {unresolved}"
    referenced = {ref.removeprefix("#/$defs/") for ref in refs}
    assert FIVE <= referenced, f"generated components no $ref reaches: {sorted(FIVE - referenced)}"
    responses = contract["responses"]
    assert responses, f"{WEIGHTS.name} declares no responses"
    for route, schema in responses.items():
        document = {**schema, "$defs": contract["$defs"]}
        try:
            jsonschema.Draft202012Validator.check_schema(document)
        except jsonschema.SchemaError as exc:
            pytest.fail(f"{route}: not a valid draft 2020-12 schema: {exc.message}")


def test_no_miserver_named_file() -> None:
    """The re-vendor keeps the file names (DR-05): no ``miserver-*`` contract appears."""
    assert SCHEMAS.is_dir() and any(SCHEMAS.glob("*.json")), f"no contracts under {SCHEMAS}"
    renamed = sorted(p.name for p in SCHEMAS.iterdir() if p.name.lower().startswith("miserver"))
    assert not renamed, f"docs/schemas/ holds renamed contracts {renamed}; DR-05 freezes the names"
    assert WEIGHTS.is_file(), f"{WEIGHTS} is missing"


def test_ownership_keys_unchanged() -> None:
    assert set(schema_sync.OWNERSHIP) == OWNERSHIP_KEYS
    for name in ("millm-weights-api-v1.json", "millm-scoring-api-v1.json"):
        entry = schema_sync.OWNERSHIP[name]
        assert isinstance(entry, schema_sync.Vendored), f"{name} is no longer vendored"
        assert entry.source.repo == "miServer", f"{name} names {entry.source.repo} as owner"
