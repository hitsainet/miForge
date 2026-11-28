"""The contract's references to miForge's own documents (021 FR-021.9; FTASKS 5.6).

Since the operator's choice (b) of 2026-11-26 each ``docs/schemas`` document has an absolute ``$id``
(``https://hitsai.net/schemas/<file>``), and the contract references documents by it:

- every external ``$ref`` (a document, or ``<$id>#<pointer>`` for a part of one) names a document in
  ``external``, whose ``file`` is the committed file with that ``$id`` and whose ``sha256`` is that
  file's; every pointer resolves inside its document;
- a change of one byte in a referenced document changes ``external``'s sha256 (so the sync test
  goes red until the contract is regenerated);
- a real answer of each document route validates against its document: the pack export, the
  candidate's ``patch.json``, the live run document and the stored verdict snapshot.
"""

from __future__ import annotations

import hashlib
import json
import shutil
from pathlib import Path
from typing import Any

import pytest

from schemas import answers_contract
from schemas import contract_tools as tools
from tests.integration.wsp.test_evaluate import ev as ev  # the WSP candidate world (a fixture)
from tests.support import answer_validator

REPO = Path(__file__).resolve().parents[3]


@pytest.fixture(scope="module")
def contract() -> dict[str, Any]:
    return dict(json.loads(answers_contract.CONTRACT_PATH.read_text(encoding="utf-8")))


def _external_refs(contract: dict[str, Any]) -> set[str]:
    return {
        s["$ref"]
        for s in tools.walk([contract["responses"], contract["$defs"]])
        if isinstance(s.get("$ref"), str) and s["$ref"].startswith("https://")
    }


def test_each_reference_names_a_listed_document_by_its_id(contract: dict[str, Any]) -> None:
    refs = _external_refs(contract)
    assert {tools.document_id(r) for r in refs} == set(contract["external"])
    assert len(contract["external"]) == 8, sorted(contract["external"])
    for sid, row in contract["external"].items():
        raw = (REPO / row["file"]).read_bytes()
        assert json.loads(raw)["$id"] == sid, row["file"]
        assert hashlib.sha256(raw).hexdigest() == row["sha256"], row["file"]
        assert sid == "https://hitsai.net/schemas/" + Path(row["file"]).name


def test_every_pointer_resolves_in_its_document(contract: dict[str, Any]) -> None:
    from referencing import Registry, Resource
    from referencing.jsonschema import DRAFT202012

    registry = Registry().with_resources(
        (
            sid,
            Resource.from_contents(
                json.loads((REPO / row["file"]).read_text()), default_specification=DRAFT202012
            ),
        )
        for sid, row in contract["external"].items()
    )
    resolver = registry.resolver()
    for ref in sorted(_external_refs(contract)):
        resolved = resolver.lookup(ref).contents
        assert isinstance(resolved, dict) and resolved, ref


def test_one_byte_of_a_document_changes_its_sha256(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, contract: dict[str, Any]
) -> None:
    """M15's shape, in a temporary tree: the contract follows the file's bytes."""
    copy = tmp_path / "schemas"
    shutil.copytree(REPO / "docs" / "schemas", copy)
    target = copy / "pack-definition-v1.json"
    target.write_bytes(target.read_bytes() + b"\n")
    monkeypatch.setenv("MIFORGE_SCHEMA_DIR", str(copy))
    sid = "https://hitsai.net/schemas/pack-definition-v1.json"
    moved = answers_contract._external_files()[sid][1]
    assert moved == hashlib.sha256(target.read_bytes()).hexdigest()
    assert moved != contract["external"][sid]["sha256"]


def _document_check(contract: dict[str, Any], key: str, body: Any) -> list[Any]:
    return answer_validator.answer_check().check(key, 200, body)


def test_the_pack_export_validates_against_pack_definition(
    world: Any, contract: dict[str, Any]
) -> None:
    from tests.support import pack_fixtures as fx

    with world.session() as session:
        fx.commit(session, fx.definition(slots=fx.valid_slots()))
    answer = world.client.get("/api/v1/packs/notary-md/versions/1/export")
    assert answer.status_code == 200, answer.text
    assert answer.headers["content-type"].startswith("application/json")
    key = "GET /api/v1/packs/{pack_id}/versions/{version}/export"
    assert contract["responses"][key]["200"] == {"$ref": "#/$defs/PackDefinitionDocument"}
    assert _document_check(contract, key, json.loads(answer.content)) == []
    assert _document_check(contract, key, {"kind": "not a pack"}), "a non-pack passed"


def test_the_candidate_documents_validate(ev: Any, contract: dict[str, Any]) -> None:
    from tests.integration.wsp.test_evaluate import _client, scorer

    cid, sha = ev.candidate()
    client, _ = _client(ev)
    patch = client.get(f"/api/v1/wsp/candidates/{cid}/patch/patch.json")
    assert patch.status_code == 200, patch.text
    key = "GET /api/v1/wsp/candidates/{candidate_id}/patch/{name}"
    assert _document_check(contract, key, json.loads(patch.content)) == []

    key = "GET /api/v1/wsp/candidates/{candidate_id}/run-document"
    live = client.get(f"/api/v1/wsp/candidates/{cid}/run-document")
    assert live.status_code == 200, live.text
    assert _document_check(contract, key, live.json()) == []
    run_id = client.get(f"/api/v1/wsp/candidates/{cid}").json()["run_id"]
    by_run = client.get(f"/api/v1/wsp/runs/{run_id}")
    assert _document_check(contract, "GET /api/v1/wsp/runs/{run_id}", by_run.json()) == []

    verdict = ev.evaluate(cid, sha, scorer(ev))
    snapshot = client.get(
        f"/api/v1/wsp/candidates/{cid}/run-document", params={"verdict_id": verdict["verdict_id"]}
    )
    assert snapshot.status_code == 200, snapshot.text
    stored = json.loads(snapshot.content)
    assert "run_sha256" not in stored and "program_reruns" not in stored
    assert _document_check(contract, key, stored) == []
    # The snapshot variant is the document's own properties: a key it lacks is refused.
    assert _document_check(contract, key, {**stored, "planted": 1}), "a planted key passed"
