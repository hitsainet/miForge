"""FTASKS 2.11b: the golden patches' ``patch.json`` files are examples, never owned contracts.

``discovered_schemas()`` (``test_schema_sync.py``) and ``test_gate_contracts.py``'s parametrisations
glob
``docs/schemas/*.json`` non-recursively. This pins that they list nothing under ``examples/``: a
recursive
glob would make every example need an ``OWNERSHIP`` row, and would run the closed-object rule over
data.
"""

from __future__ import annotations

from pathlib import Path

from tests.contract import test_gate_contracts, test_schema_sync

SCHEMA_DIR = Path(__file__).resolve().parents[3] / "docs" / "schemas"
EXAMPLES = SCHEMA_DIR / "examples"


def test_the_examples_exist_so_this_test_can_fail() -> None:
    assert list(
        EXAMPLES.rglob("patch.json")
    ), "no example patch.json: the checks below would be vacuous"


def test_discovered_schemas_lists_no_example() -> None:
    names = test_schema_sync.discovered_schemas()
    assert names, "discovered_schemas() found nothing"
    assert not [n for n in names if "/" in n or n == "patch.json"], names


def _parametrised(function: object) -> list[str]:
    marks = getattr(function, "pytestmark", [])
    values: list[str] = []
    for mark in marks:
        if mark.name == "parametrize":
            values.extend(str(v) for v in mark.args[1])
    return values


def test_gate_contract_parametrisations_list_no_example() -> None:
    for fn in (
        test_gate_contracts.test_schema_is_itself_valid,
        test_gate_contracts.test_every_object_is_closed_to_unknown_keys,
    ):
        values = _parametrised(fn)
        assert values, f"{fn.__name__} has no parametrisation to check"
        assert not [
            v for v in values if "/" in v or v == "patch.json" or v == "expected.json"
        ], values


def test_the_gate_registry_holds_no_example() -> None:
    registry = test_gate_contracts._registry()
    uris = list(registry)  # the resource URIs: file names relative to docs/schemas/
    assert uris
    assert not [u for u in uris if "examples" in u or u.endswith("patch.json")], uris
