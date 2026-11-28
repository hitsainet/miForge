# Origin: miDataworks (Onegaishimas/miDataworks)
#   backend/tests/unit/contract/test_answers_classifier.py @ 6b50a8f4
# Mode: adapt (docs/REUSE.md). Kept: every pair test and writer test. Changed: miForge's
# imports and the next major's file name.
"""The writer classifies every difference and refuses a breaking one (021 FR-021.19; FTASKS 1.7).

One synthetic before-and-after pair per row of the table (miServer FR-044.16's rows, kept identical,
MF8), from ``tests/support/fixtures/answer_classifier_pairs.json``, plus unlisted changes, which
must come out breaking: the classifier fails closed. Then the writer on a planted file: a removal
is refused and leaves the committed bytes as they were; an additive change raises ``revision`` by
exactly one.
"""

from __future__ import annotations

import copy
import hashlib
import json
import re
from pathlib import Path
from typing import Any

import pytest

from schemas import answers_contract
from schemas import contract_tools as tools
from tests.support import classifier_pairs

PAIRS = classifier_pairs.load()
REPORTED = classifier_pairs.run(tools.classify, PAIRS)


@pytest.mark.parametrize("pair", PAIRS["pairs"], ids=[p["id"] for p in PAIRS["pairs"]])
def test_each_pair_reports_its_row(pair: dict[str, Any]) -> None:
    row, is_breaking = pair["expect"]
    rows = REPORTED[pair["id"]]
    assert [row, is_breaking] in rows, rows
    assert any(brk for _, brk in rows) is pair["breaking"], rows


def test_every_row_of_the_table_has_a_pair() -> None:
    rows = {pair["expect"][0] for pair in PAIRS["pairs"]}
    table = set(tools.ADDITIVE_ROWS) | {
        "removed",
        "retyped",
        "made required",
        "enum widened",
        "sometimes absent",
        "moved between responses and excluded",
        tools.UNLISTED,
    }
    assert rows == table


def test_only_the_named_rows_are_additive() -> None:
    assert tools.ADDITIVE_ROWS == {
        "new route",
        "new status code",
        "new optional field",
        "new always-sent field",
        "descriptive",
        "vendored schema revised (additive)",
    }
    assert tools.Change("p", tools.UNLISTED).breaking
    assert tools.Change("p", "a row nobody named").breaking


def test_a_rename_is_a_removal_plus_an_addition() -> None:
    rename = next(p for p in PAIRS["pairs"] if p.get("note", "").startswith("a rename"))
    rows = REPORTED[rename["id"]]
    assert ["removed", True] in rows and ["new optional field", False] in rows


def test_equal_documents_have_no_change() -> None:
    assert tools.classify(copy.deepcopy(PAIRS["base"]), copy.deepcopy(PAIRS["base"])) == []


def test_a_change_two_references_down_is_seen_through_equal_text() -> None:
    """The route's own schema text is equal (``$ref`` Item); the change is inside ``Sub``."""
    new = copy.deepcopy(PAIRS["base"])
    del new["$defs"]["Sub"]["properties"]["z"]
    assert ("removed", True) in {(c.row, c.breaking) for c in tools.classify(PAIRS["base"], new)}


def test_a_changed_external_document_without_a_resolver_fails_closed() -> None:
    new = copy.deepcopy(PAIRS["base"])
    new["external"]["https://example.test/doc.json"]["sha256"] = "sha-add"
    assert {(c.row, c.breaking) for c in tools.classify(PAIRS["base"], new)} == {
        (tools.UNLISTED, True)
    }


# -- the writer -----------------------------------------------------------------------------------


def _body() -> dict[str, Any]:
    body = copy.deepcopy(PAIRS["base"])
    body["external"] = {}
    del body["$defs"]["Item"]["properties"]["doc"]
    body["$defs"]["Item"]["required"].remove("doc")
    return body


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_the_writer_writes_revision_1_then_refuses_a_removal(tmp_path: Path) -> None:
    path = tmp_path / answers_contract.CONTRACT_FILENAME
    first = answers_contract.write(_body(), path)
    assert first["revision"] == 1
    assert first["content_sha256"] == tools.content_digest(first)
    assert answers_contract.problems(json.loads(path.read_text(encoding="utf-8"))) == []
    before = _sha(path)
    smaller = _body()
    del smaller["$defs"]["Item"]["properties"]["opt"]
    with pytest.raises(
        answers_contract.BreakingAnswerChange, match=re.escape("miforge-answers-v2.json")
    ):
        answers_contract.write(smaller, path)
    assert _sha(path) == before


def test_the_refusal_names_each_change_and_its_row(tmp_path: Path) -> None:
    path = tmp_path / answers_contract.CONTRACT_FILENAME
    answers_contract.write(_body(), path)
    smaller = _body()
    del smaller["$defs"]["Item"]["properties"]["opt"]
    with pytest.raises(answers_contract.BreakingAnswerChange) as refused:
        answers_contract.write(smaller, path)
    assert "$defs.Item.properties.opt: removed" in str(refused.value)


def test_an_additive_change_raises_the_revision_by_exactly_one(tmp_path: Path) -> None:
    path = tmp_path / answers_contract.CONTRACT_FILENAME
    answers_contract.write(_body(), path)
    bigger = _body()
    bigger["$defs"]["Item"]["properties"]["a2"] = {"type": "integer"}
    bigger["$defs"]["Item"]["required"].append("a2")
    doc = answers_contract.write(bigger, path)
    assert doc["revision"] == 2
    assert json.loads(path.read_text(encoding="utf-8"))["revision"] == 2


def test_an_equal_body_writes_nothing(tmp_path: Path) -> None:
    path = tmp_path / answers_contract.CONTRACT_FILENAME
    answers_contract.write(_body(), path)
    before = _sha(path)
    assert answers_contract.write(_body(), path)["revision"] == 1
    assert _sha(path) == before
