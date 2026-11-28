"""Approval answers redact secret-named keys on the way out (operator 2026-11-26).

Filing refuses keys, so no stored request holds one today; these tests PLANT one, nested at
depth, in a real stored request (a settings change filed by an agent), its summary, its result
and its error, then read every approval answer:

- over HTTP: the list, one approval, approve, reject and the batch approve;
- the 202 an agent receives (``pending_body``) and the ``approval`` object's summary words;
- through the MCP approval tools (``forge_get_approval_status``, ``forge_list_approvals``), served
  by the real app over an ASGI transport;

and require no planted value anywhere in the answer, each secret-named key present with
``[redacted]``, every other key unchanged, the stored row unchanged, and ``request_digest`` the
digest of the real stored payload: the approval still executes, so the operator's decision is
unaffected.
"""

from __future__ import annotations

import asyncio
import copy
import json
from typing import Any

import httpx
import pytest

from core.canonical import sha256_of
from core.secret_keys import REDACTED
from tests.support.approval_world import AGENT, World

#: Planted at depth: a dict in a list in a dict, beside keys that must survive.
SECRETS = {
    "api_key": "PLANTED-API-KEY",
    "hub_token": "PLANTED-HUB-TOKEN",
    "key_fingerprint": "PLANTED-FINGERPRINT",
    "price_in_per_mtok": "PLANTED-PRICE",
    "Authorization": "PLANTED-BEARER",
}
KEEP = {"tokens": 4096, "thinking_tokens": 512, "has_api_key": True, "model": "m"}
PLANT = {"deep": [{"inner": {**SECRETS, **KEEP}}], "api_key": "PLANTED-TOP-KEY"}
PLANTED_VALUES = (*SECRETS.values(), "PLANTED-TOP-KEY")


def _ask_set(world: World, key: str, value: Any) -> str:
    response = world.client.put(f"/api/v1/settings/{key}", json={"value": value}, headers=AGENT)
    assert response.status_code == 202, response.text
    return str(response.json()["approval_id"])


def _plant(world: World, approval_id: str, **fields: Any) -> dict[str, Any]:
    """Add ``PLANT`` to the stored payload (a key the executor ignores) and to the given fields,
    and re-take the digest over the real stored payload, as filing does. Returns the payload."""
    from models.approval import Approval

    with world.session() as session:
        row = session.get(Approval, approval_id)
        payload = {**copy.deepcopy(row.payload), "planted": copy.deepcopy(PLANT)}
        row.payload = payload
        row.request_digest = sha256_of(payload)
        row.summary = {**dict(row.summary or {}), "planted": copy.deepcopy(PLANT)}
        for name, value in fields.items():
            setattr(row, name, value)
        session.commit()
    return payload


def _stored(world: World, approval_id: str) -> Any:
    from models.approval import Approval

    with world.session() as session:
        row = session.get(Approval, approval_id)
        return copy.deepcopy(row.payload), row.request_digest


def _assert_redacted(answer: Any) -> None:
    text = json.dumps(answer)
    for value in PLANTED_VALUES:
        assert value not in text, value
    found = _plants(answer)
    assert found, "no planted block reached the answer: the test would prove nothing"
    for block in found:
        assert block["api_key"] == REDACTED
        (inner,) = block["deep"]
        assert inner["inner"] == {**dict.fromkeys(SECRETS, REDACTED), **KEEP}


def _plants(node: Any) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    if isinstance(node, dict):
        if isinstance(node.get("planted"), dict):
            out.append(node["planted"])
        for value in node.values():
            out.extend(_plants(value))
    elif isinstance(node, list):
        for value in node:
            out.extend(_plants(value))
    return out


def _executed(world: World, key: str = "checkpoint_lifecycle_enabled", value: Any = True) -> str:
    """A settings change filed by an agent and approved, so its row may hold a result."""
    approval_id = _ask_set(world, key, value)
    approved = world.client.post(f"/api/v1/approvals/{approval_id}/approve")
    assert approved.status_code == 200 and approved.json()["status"] == "executed", approved.text
    return approval_id


def _planted_result(key: str = "checkpoint_lifecycle_enabled") -> dict[str, Any]:
    """A settings write's stored result (``settings.py:write_setting``) with the plant inside its
    free-form ``value``: a shape the route can store, so the strict answer validator (021) passes
    it and only redaction is under test."""
    return {
        "key": key,
        "value": {"planted": copy.deepcopy(PLANT)},
        "previous": None,
        "source": "override",
    }


def test_the_list_and_the_one_approval_read_redact_at_depth(world: World) -> None:
    from services import approval_service

    executed = _executed(world)
    payload = _plant(world, executed, result=_planted_result())
    failed = _ask_set(world, "checkpoint_lifecycle_keep_last_n", 3)
    _plant(
        world,
        failed,
        status="failed",
        decided_by="operator",
        decided_at=approval_service.now(),
        error={"code": "x", "message": "y", "details": {"planted": copy.deepcopy(PLANT)}},
    )
    listed = {
        r["approval_id"]: r for r in world.client.get("/api/v1/approvals").json()["approvals"]
    }
    for approval_id in (executed, failed):
        one = world.client.get(f"/api/v1/approvals/{approval_id}")
        assert one.status_code == 200, one.text
        body = one.json()
        _assert_redacted(body)
        _assert_redacted(listed[approval_id])
        # payload, summary, and the result (executed) or the error (failed) each carried it.
        assert len(_plants(body)) == len(_plants(listed[approval_id])) == 3
    # The stored row is unchanged: the real secret is still there, and the digest is its own.
    stored, digest = _stored(world, executed)
    assert stored == payload and digest == sha256_of(payload)
    assert listed[executed]["request_digest"] == "sha256:" + digest


def test_approve_redacts_and_the_digest_of_the_real_payload_still_lets_it_run(
    world: World,
) -> None:
    approval_id = _ask_set(world, "checkpoint_lifecycle_enabled", True)
    payload = _plant(world, approval_id)
    answer = world.client.post(f"/api/v1/approvals/{approval_id}/approve")
    assert answer.status_code == 200, answer.text
    body = answer.json()
    assert body["status"] == "executed", body  # the digest check passed on the stored payload
    _assert_redacted(body)
    stored, digest = _stored(world, approval_id)
    assert stored == payload
    assert body["request_digest"] == "sha256:" + digest == "sha256:" + sha256_of(payload)
    assert (
        world.client.get("/api/v1/settings").json()["settings"]["checkpoint_lifecycle_enabled"][
            "value"
        ]
        is True
    )


def test_a_redacted_answer_never_feeds_the_digest(world: World) -> None:
    """Had the digest been taken over the redacted view, it would differ from the stored one."""
    from core.secret_keys import redacted

    approval_id = _ask_set(world, "checkpoint_lifecycle_enabled", True)
    payload = _plant(world, approval_id)
    assert sha256_of(redacted(payload)) != sha256_of(payload)
    body = world.client.get(f"/api/v1/approvals/{approval_id}").json()
    assert body["request_digest"] == "sha256:" + sha256_of(payload)


def test_reject_redacts(world: World) -> None:
    approval_id = _ask_set(world, "checkpoint_lifecycle_enabled", True)
    _plant(world, approval_id)
    answer = world.client.post(f"/api/v1/approvals/{approval_id}/reject", json={"reason": "no"})
    assert answer.status_code == 200, answer.text
    assert answer.json()["status"] == "rejected"
    _assert_redacted(answer.json())


def test_the_batch_approve_redacts_every_result(world: World) -> None:
    first = _ask_set(world, "checkpoint_lifecycle_enabled", True)
    second = _ask_set(world, "checkpoint_lifecycle_keep_last_n", 3)
    _plant(world, first)
    _plant(world, second)
    answer = world.client.post(
        "/api/v1/approvals/approve-batch", json={"approval_ids": [first, second]}
    )
    assert answer.status_code == 200, answer.text
    results = answer.json()["results"]
    assert [r["ok"] for r in results] == [True, True]
    for result in results:
        _assert_redacted(result)


def test_the_agents_202_and_the_approval_object_redact_the_summary(world: World) -> None:
    from services import approval_service
    from services.result_language import summary_words

    approval_id = _ask_set(world, "checkpoint_lifecycle_enabled", True)
    _plant(world, approval_id)
    with world.session() as session:
        row = approval_service.get(session, approval_id)
        body = approval_service.pending_body(row)
        raw_words = summary_words(row.action, row.summary)
    _assert_redacted(body)
    words = body["approval"]["summary"]
    for value in PLANTED_VALUES:
        assert value not in words
    # The words differ from the raw summary's only where a secret was (if they render it at all).
    assert isinstance(raw_words, str) and isinstance(words, str)


@pytest.fixture()
def _open_gate(monkeypatch: pytest.MonkeyPatch) -> None:
    from mcp_server.health_gate import HealthGate

    async def unavailable(self: Any, *_: str) -> None:
        return None

    monkeypatch.setattr(HealthGate, "unavailable", unavailable)


@pytest.mark.usefixtures("_open_gate")
def test_the_mcp_approval_tools_answer_redacted_from_the_real_app(world: World) -> None:
    from mcp_server.config import MCPSettings
    from mcp_server.server import build_server

    approval_id = _executed(world)
    _plant(world, approval_id, result=_planted_result())
    mcp, client = build_server(MCPSettings(auth_token="x" * 32))
    client._http = httpx.AsyncClient(
        base_url="http://backend/api/v1",
        transport=httpx.ASGITransport(app=world.client.app),
        headers={"X-Miforge-Agent": client.identity},
    )

    async def both() -> tuple[Any, Any]:
        one = await mcp.call_tool("forge_get_approval_status", {"approval_id": approval_id})
        many = await mcp.call_tool("forge_list_approvals", {})
        return one, many

    one, many = asyncio.run(both())
    for result in (one, many):
        assert not result.is_error, result
        text = "".join(getattr(block, "text", "") for block in result.content)
        for value in PLANTED_VALUES:
            assert value not in text, value
        assert REDACTED in text
    status = json.loads(one.content[0].text)
    _assert_redacted(status)
