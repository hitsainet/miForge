"""FU-7b: a verdict's standing (``wsp.standing``), held to the one case table the run page's rule
is held to (``frontend/src/components/panels/wsp/standing.cases.json``)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from wsp import standing

REPO = Path(__file__).resolve().parents[4]
CASES = REPO / "frontend" / "src" / "components" / "panels" / "wsp" / "standing.cases.json"
TABLE: list[dict[str, Any]] = json.loads(CASES.read_text("utf-8"))["cases"]


def test_the_table_is_read_and_holds_every_kind_of_case() -> None:
    assert len(TABLE) >= 10
    assert {c["standing"] for c in TABLE} >= {
        "improved",
        "improved_unconfirmed",
        "not_significant",
        "regressed",
        "no_change",
        "unmeasured",
    }


@pytest.mark.parametrize("case", TABLE, ids=[c["case"] for c in TABLE])
def test_each_case(case: dict[str, Any]) -> None:
    assert standing.standing(case["document"]) == case["standing"]
    assert standing.confirmed_by(case["document"]) == case["confirmed_by"]
    assert standing.reading(case["document"]) == {
        "standing": case["standing"],
        "confirmed_by": case["confirmed_by"],
    }


def test_the_rule_reads_only_fields_the_run_contract_already_has() -> None:
    """FU-7b adds no field to ``miforge-wsp-run-v1.json``: every field the rule reads is there."""
    schema = json.loads((REPO / "docs" / "schemas" / "miforge-wsp-run-v1.json").read_text("utf-8"))
    assert {"primary", "value", "heldout_uses", "name"} <= set(
        schema["$defs"]["SetVerdict"]["properties"]
    )
    assert {"value", "per_set"} <= set(schema["$defs"]["Verdict"]["properties"])
    assert "standing" not in json.dumps(schema) and "confirmed_by" not in json.dumps(schema)
