"""Media types, health's 503 and the one error shape (021 FR-021.10, FR-021.11; FTASKS 5.7, 5.8).

- The two routes that answer only bytes are ``excluded`` with the media type they serve; the
  candidate's patch route is in ``responses`` with JSON for ``patch.json`` and the media
  alternative it serves for ``edit.npz``. Each is served as declared, and the validator's media
  check refuses a type nothing declares.
- Health declares its 503 with the same model as its 200, and a real degraded answer validates.
- A refusal (``refuse(...)``), an ``HTTPException`` (an unknown path) and a request that fails
  validation each leave the API as ``errors.app``; and the MCP client's teaching text (``<code>``,
  message, details) is built from exactly those fields of a real refusal.
"""

from __future__ import annotations

import asyncio
import json
from typing import Any

import httpx
import pytest

from schemas import answers_contract
from tests.integration.wsp.test_evaluate import ev as ev  # the WSP candidate world (a fixture)
from tests.support import answer_validator

TENSORS = "GET /api/v1/wsp/candidates/{candidate_id}/tensors"
SCORES = "GET /api/v1/wsp/candidates/{candidate_id}/verdict/scores"
PATCH = "GET /api/v1/wsp/candidates/{candidate_id}/patch/{name}"
HEALTH = "GET /api/health"


@pytest.fixture(scope="module")
def contract() -> dict[str, Any]:
    return dict(json.loads(answers_contract.CONTRACT_PATH.read_text(encoding="utf-8")))


def _media(answer: Any) -> str:
    return str(answer.headers.get("content-type", "")).split(";")[0].strip()


def test_the_byte_routes_and_the_media_alternative_are_declared(contract: dict[str, Any]) -> None:
    excluded = contract["excluded"]
    assert set(excluded) == {TENSORS, SCORES}
    assert {excluded[k]["media_type"] for k in excluded} == {"application/octet-stream"}
    assert contract["responses"][PATCH]["x-media-alternatives"] == {
        "application/octet-stream": "name = edit.npz"
    }
    assert contract["responses"][PATCH]["200"] == {"$ref": "#/$defs/WspPatchDocument"}


def test_each_is_served_as_declared(ev: Any, contract: dict[str, Any]) -> None:
    from tests.integration.wsp.test_evaluate import _client, scorer

    cid, sha = ev.candidate()
    ev.evaluate(cid, sha, scorer(ev))
    client, _ = _client(ev)
    check = answer_validator.answer_check()
    served = {
        TENSORS: client.get(f"/api/v1/wsp/candidates/{cid}/tensors"),
        SCORES: client.get(f"/api/v1/wsp/candidates/{cid}/verdict/scores", params={"set": "eval"}),
        PATCH: client.get(f"/api/v1/wsp/candidates/{cid}/patch/edit.npz"),
    }
    for key, answer in served.items():
        assert answer.status_code == 200, (key, answer.text[:200])
        assert check.check_media(key, 200, _media(answer)) == [], key
        assert check.check_media(key, 200, "text/html"), f"{key}: an undeclared type passed"
    assert _media(served[TENSORS]) == contract["excluded"][TENSORS]["media_type"]
    assert _media(served[PATCH]) in contract["responses"][PATCH]["x-media-alternatives"]
    document = client.get(f"/api/v1/wsp/candidates/{cid}/patch/patch.json")
    assert _media(document) == "application/json"
    assert check.check(PATCH, 200, json.loads(document.content)) == []


def test_health_declares_its_503_and_a_degraded_answer_validates(
    contract: dict[str, Any], monkeypatch: pytest.MonkeyPatch
) -> None:
    from fastapi.testclient import TestClient

    from api.app import create_app
    from core.config import reset_settings

    entry = contract["responses"][HEALTH]
    assert entry["503"] == entry["200"] == {"$ref": "#/$defs/HealthGetHealthAnswer"}
    monkeypatch.setenv("REDIS_URL", "redis://127.0.0.1:1/0")  # nothing listens on port 1
    reset_settings()
    try:
        with TestClient(create_app(), raise_server_exceptions=False) as client:
            answer = client.get("/api/health")
    finally:
        monkeypatch.undo()
        reset_settings()
    assert answer.status_code == 503, answer.text
    assert answer.json()["dependencies"]["redis"]["status"] != "ok"
    assert answer_validator.answer_check().check(HEALTH, 503, answer.json()) == []


def _error_check(contract: dict[str, Any], body: Any) -> list[str]:
    from jsonschema import Draft202012Validator
    from referencing import Registry, Resource
    from referencing.jsonschema import DRAFT202012

    from schemas.contract_tools import strict_form

    strict = strict_form(contract)
    registry = Registry().with_resource(
        contract["$id"], Resource.from_contents(strict, default_specification=DRAFT202012)
    )
    validator = Draft202012Validator({"$ref": contract["$id"] + "#/errors/app"}, registry=registry)
    return [e.message for e in validator.iter_errors(body)]


def test_each_error_leaves_as_the_one_envelope(world: Any, contract: dict[str, Any]) -> None:
    assert contract["errors"] == {"app": {"$ref": "#/$defs/ErrorEnvelope"}}
    refused = world.client.get("/api/v1/runs/no-such-run")
    unknown = world.client.get("/api/v1/no-such-route")
    invalid = world.client.post("/api/v1/runs/validate", json=["not", "a", "definition"])
    for answer in (refused, unknown, invalid):
        assert answer.status_code >= 400, answer.text
        assert _error_check(contract, answer.json()) == [], answer.text
    assert refused.json()["error"]["code"] == "run_not_found"
    assert unknown.json()["error"]["code"] == "http_404"
    assert invalid.status_code == 422 and invalid.json()["error"]["code"] == "validation_failed"
    assert _error_check(contract, {"detail": "a second error shape"}), "FastAPI's shape passed"


def test_the_teaching_text_is_built_from_the_envelope(world: Any) -> None:
    """The MCP client reads ``code``, ``message`` and ``details`` of the envelope and nothing
    else: a real refusal through the real client (an ASGI transport to the real app)."""
    from mcp_server.client import ForgeClient, ForgeError

    app = world.client.app
    expected = world.client.get("/api/v1/runs/no-such-run").json()["error"]

    async def call() -> ForgeError:
        client = ForgeClient("http://forge.test", "agent:forge-mcp")
        await client.close()
        client._http = httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app), base_url="http://forge.test/api/v1"
        )
        try:
            await client.get("/runs/no-such-run")
        except ForgeError as error:
            return error
        finally:
            await client.close()
        raise AssertionError("the refusal did not raise")

    error = asyncio.run(call())
    assert (error.status, error.code, error.message, error.details) == (
        404,
        expected["code"],
        expected["message"],
        expected["details"],
    )


def test_the_validator_checks_the_served_media_type(
    ev: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The ASGI wrapper sends a non-JSON 2xx through ``check_media`` (its wiring, not the
    method): with the contract's media type for ``tensors`` changed, a real answer is recorded."""
    from tests.integration.wsp.test_evaluate import _client

    cid, _ = ev.candidate()
    client, _ = _client(ev)
    check = answer_validator.answer_check()
    monkeypatch.setitem(check.contract["excluded"][TENSORS], "media_type", "text/plain")
    answer_validator.FAILURES.clear()
    answer = client.get(f"/api/v1/wsp/candidates/{cid}/tensors")
    assert answer.status_code == 200
    recorded = [f.message for f in answer_validator.FAILURES]
    answer_validator.FAILURES.clear()
    assert recorded == [
        "media type 'application/octet-stream' is not declared (declared: ['text/plain'])"
    ], recorded


def test_the_validator_checks_a_declared_non_2xx_answer(monkeypatch: pytest.MonkeyPatch) -> None:
    """The ASGI wrapper sends health's 503 through the check (a declared status outside 2xx; its
    wiring, not the method): with the contract's 503 asking for a key no answer has, a real
    degraded answer is recorded."""
    from fastapi.testclient import TestClient
    from jsonschema import Draft202012Validator

    from api.app import create_app
    from core.config import reset_settings

    check = answer_validator.answer_check()
    original = check.validator
    strict_503 = Draft202012Validator({"type": "object", "required": ["a_key_no_answer_has"]})

    def validator(key: str, status: str) -> Any:
        return strict_503 if (key, status) == (HEALTH, "503") else original(key, status)

    monkeypatch.setattr(check, "validator", validator)
    monkeypatch.setenv("REDIS_URL", "redis://127.0.0.1:1/0")
    reset_settings()
    answer_validator.FAILURES.clear()
    try:
        with TestClient(create_app(), raise_server_exceptions=False) as client:
            answer = client.get("/api/health")
    finally:
        recorded = [f.message for f in answer_validator.FAILURES]
        answer_validator.FAILURES.clear()
        monkeypatch.undo()
        reset_settings()
    assert answer.status_code == 503
    assert any("a_key_no_answer_has" in m for m in recorded), recorded
