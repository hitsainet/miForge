"""An agent's editor endpoint proposal (2026-11-20; ``forge_wsp_register_endpoint``).

``POST /api/v1/wsp/endpoints`` from an agent: a LOCAL endpoint with NO key and NO prices waits for
the operator as ``wsp_endpoint_register`` and, once approved, is registered exactly once. A key,
any price, or an external base URL or locality is refused by name **before** any approval is
filed. The operator's call is unchanged. ``PATCH`` and ``DELETE`` stay the operator's.

Everything goes through ``create_app()``, the gate decorator and the approval service on a migrated
database, so each assertion is about the mounted route, not an imported function.
"""

from __future__ import annotations

import json
from collections.abc import Iterator
from typing import Any

import pytest
import sqlalchemy as sa
from fastapi.testclient import TestClient

from api.app import create_app
from api.deps import session_dependency
from tests.integration.wsp.wsp_db import engine_at_head
from tests.support.jobs import session_for
from wsp.settings import get_wsp_settings
from wsp_gateway import sealing

AGENT = {"X-Miforge-Agent": "agent:forge-mcp"}
SENTINEL = "sk-SENTINEL-proposal-5a7e"
LOCAL = "http://miserver-backend.miserver.svc:8000/v1"
#: A local, keyless, unpriced miServer endpoint: what an agent may propose.
PROPOSAL: dict[str, Any] = {
    "id": "miserver-gemma4-26b",
    "kind": "openai_compatible",
    "serving": "miserver",
    "base_url": LOCAL,
    "model": "gemma-4-26b",
    "action_protocol": "structured",
    "context_window_tokens": 32768,
}
#: The external endpoint the default allowlist admits, priced and keyed: the operator's only.
EXTERNAL: dict[str, Any] = {
    "id": "claude-ed",
    "kind": "anthropic",
    "base_url": "https://api.anthropic.com",
    "model": "claude-editor",
    "price_in_per_mtok": "3",
    "price_out_per_mtok": "15",
    # Feature 020 (FR-020.2, FR-020.3): a hosted Anthropic endpoint's window and cache prices.
    "price_cache_write_per_mtok": "3.75",
    "price_cache_read_per_mtok": "0.3",
    "context_window_tokens": 200_000,
    "api_key": SENTINEL,
}


def _j(value: Any) -> Any:
    return json.loads(value) if isinstance(value, str | bytes) else value


class Env:
    def __init__(self, engine: sa.Engine, client: TestClient) -> None:
        self.engine, self.client = engine, client

    def count(self, table: str) -> int:
        with self.engine.connect() as conn:
            return int(conn.execute(sa.text(f"SELECT COUNT(*) FROM {table}")).scalar_one())

    def rows(self, table: str) -> list[dict[str, Any]]:
        with self.engine.connect() as conn:
            return [dict(r._mapping) for r in conn.execute(sa.text(f"SELECT * FROM {table}"))]

    def propose(self, body: dict[str, Any]) -> Any:
        return self.client.post("/api/v1/wsp/endpoints", json=body, headers=AGENT)

    def approve(self, approval_id: str) -> Any:
        return self.client.post(f"/api/v1/approvals/{approval_id}/approve")


@pytest.fixture()
def env(tmp_path: Any, monkeypatch: pytest.MonkeyPatch) -> Iterator[Env]:
    keys = sealing.generate_keypair()
    monkeypatch.setenv("WSP_GATEWAY_PUBLIC_KEY", keys[1])
    get_wsp_settings.cache_clear()
    engine = engine_at_head(tmp_path)
    app = create_app()

    def one_session() -> Iterator[Any]:
        with session_for(engine) as session:
            yield session

    app.dependency_overrides[session_dependency] = one_session
    yield Env(engine, TestClient(app, raise_server_exceptions=False))
    engine.dispose()
    get_wsp_settings.cache_clear()


def _waits(response: Any) -> dict[str, Any]:
    assert response.status_code == 202, response.text
    body = response.json()
    assert body["status"] == "pending" and body["action"] == "wsp_endpoint_register", body
    return dict(body)


# --------------------------------------------------------------------- the gated proposal


def test_an_agents_local_keyless_proposal_waits_and_registers_once_approved(env: Env) -> None:
    pending = _waits(env.propose(PROPOSAL))
    # Nothing is registered while it waits; exactly one approval is filed.
    assert env.count("mf_wsp_endpoints") == 0
    assert env.count("mf_approvals") == 1
    assert env.client.get("/api/v1/wsp/endpoints").json() == {"endpoints": []}
    # The card shows exactly what will be registered.
    assert pending["summary"] == {
        "route": "POST /api/v1/wsp/endpoints",
        "endpoint_id": "miserver-gemma4-26b",
        "model": "gemma-4-26b",
        "base_url": LOCAL,
        "action_protocol": "structured",
        "kind": "openai_compatible",
        "serving": "miserver",
        "locality": "local",
        "context_window_tokens": 32768,
        "thinking": "off",
        "thinking_tokens": None,
        "output_cap": "on",
        "paging": "on",
        "output_cap_field": "max_completion_tokens",
        "key_and_prices": "none: an agent cannot send an API key or set prices",
    }
    # The stored request is the body the agent sent, nothing added.
    (approval,) = env.rows("mf_approvals")
    assert _j(approval["payload"]) == {"body": PROPOSAL}
    assert approval["requested_by"] == "agent:forge-mcp"

    decided = env.approve(pending["approval_id"])
    assert decided.status_code == 200, decided.text
    (approval,) = env.rows("mf_approvals")
    assert approval["status"] == "executed", approval
    (row,) = env.rows("mf_wsp_endpoints")
    assert (row["id"], row["model"], row["base_url"], row["locality"]) == (
        "miserver-gemma4-26b",
        "gemma-4-26b",
        LOCAL,
        "local",
    )
    assert (row["action_protocol"], row["context_window_tokens"]) == ("structured", 32768)
    assert row["key_ciphertext"] is None and row["key_fingerprint"] is None
    assert all(row[name] is None for name in ("price_in_per_mtok", "price_out_per_mtok"))
    # Who asked is recorded, not "operator".
    assert row["created_by"] == "agent:forge-mcp"
    result = _j(approval["result"])
    assert result["id"] == "miserver-gemma4-26b" and result["has_key"] is False
    (listed,) = env.client.get("/api/v1/wsp/endpoints").json()["endpoints"]
    assert listed["id"] == "miserver-gemma4-26b" and listed["has_key"] is False

    # Once: a second approve is refused and registers nothing more.
    again = env.approve(pending["approval_id"])
    assert again.status_code == 409, again.text
    assert env.count("mf_wsp_endpoints") == 1


def test_an_agent_cannot_approve_its_own_proposal(env: Env) -> None:
    pending = _waits(env.propose(PROPOSAL))
    r = env.client.post(f"/api/v1/approvals/{pending['approval_id']}/approve", headers=AGENT)
    assert r.status_code == 403 and r.json()["error"]["code"] == "agent_cannot_decide"
    assert env.count("mf_wsp_endpoints") == 0


@pytest.mark.parametrize(
    "change, code",
    [
        ({"api_key": SENTINEL}, "wsp_endpoint_agent_key_refused"),
        ({"api_key": ""}, "wsp_endpoint_agent_key_refused"),
        ({"price_in_per_mtok": "0"}, "wsp_endpoint_agent_prices_refused"),
        ({"price_out_per_mtok": "1"}, "wsp_endpoint_agent_prices_refused"),
        ({"price_cache_write_per_mtok": "1"}, "wsp_endpoint_agent_prices_refused"),
        ({"price_cache_read_per_mtok": "1"}, "wsp_endpoint_agent_prices_refused"),
        # A declared locality is ignored for the operator (PF-16); from an agent it is refused.
        ({"locality": "external"}, "wsp_endpoint_agent_external_refused"),
        # Allow-listed external host with no key or prices: refused as external, by name.
        (
            {
                "base_url": "https://api.anthropic.com",
                "kind": "anthropic",
                "serving": "generic",
                "action_protocol": "tool_calls",
            },
            "wsp_endpoint_agent_external_refused",
        ),
    ],
    ids=[
        "key",
        "empty-key",
        "price-in",
        "price-out",
        "price-cache-write",
        "price-cache-read",
        "declared-external",
        "external-url",
    ],
)
def test_a_key_prices_or_external_is_refused_before_any_approval_is_filed(
    env: Env, change: dict[str, Any], code: str
) -> None:
    r = env.propose({**PROPOSAL, **change})
    assert r.status_code == 403, r.text
    error = r.json()["error"]
    assert error["code"] == code, r.text
    # The refusal names the operator's screen, and never echoes a key.
    assert "#/wsp/endpoints" in error["message"]
    assert SENTINEL not in r.text
    assert env.count("mf_approvals") == 0
    assert env.count("mf_wsp_endpoints") == 0


def test_the_full_external_body_is_refused_for_its_key_and_stores_nothing(env: Env) -> None:
    r = env.propose(EXTERNAL)
    assert r.status_code == 403
    assert r.json()["error"]["code"] == "wsp_endpoint_agent_key_refused"
    assert SENTINEL not in r.text
    assert env.count("mf_approvals") == 0 and env.count("mf_wsp_endpoints") == 0


@pytest.mark.parametrize(
    "change, code",
    [
        ({"id": None}, "validation_failed"),
        ({"id": "Bad Id"}, "validation_failed"),
        ({"serving": "generic"}, "wsp_endpoint_url_refused"),  # a miServer host needs miserver
        ({"base_url": "http://10.9.9.9:8000/v1"}, "wsp_endpoint_url_refused"),  # not allow-listed
        ({"kind": "anthropic"}, "wsp_endpoint_protocol_unsupported"),  # structured needs openai
        ({"context_window_tokens": 0}, "validation_failed"),
    ],
    ids=["no-id", "bad-id", "miserver-generic", "unlisted-host", "structured-anthropic", "window"],
)
def test_a_proposal_the_registration_would_refuse_files_nothing(
    env: Env, change: dict[str, Any], code: str
) -> None:
    body = {k: v for k, v in {**PROPOSAL, **change}.items() if v is not None}
    r = env.propose(body)
    assert r.json()["error"]["code"] == code, r.text
    assert env.count("mf_approvals") == 0 and env.count("mf_wsp_endpoints") == 0


def test_an_id_taken_after_filing_fails_the_approval_and_registers_nothing_more(env: Env) -> None:
    pending = _waits(env.propose(PROPOSAL))
    # The operator registers the same id in the meantime.
    assert env.client.post("/api/v1/wsp/endpoints", json=PROPOSAL).status_code == 201
    env.approve(pending["approval_id"])
    (approval,) = env.rows("mf_approvals")
    assert approval["status"] == "failed"
    assert _j(approval["error"])["code"] == "validation_failed"
    (row,) = env.rows("mf_wsp_endpoints")
    assert row["created_by"] == "operator"


def test_the_route_applies_the_agent_rule_again_when_a_stored_proposal_executes(env: Env) -> None:
    """A stored request that carries a key (filed by any path but the gate's validator) is
    refused by the ROUTE when it executes: the rule is not only the filer's."""
    from api.agent_origin import EXECUTORS
    from services import approval_service

    key = next(k for k in EXECUTORS if k.endswith("wsp_endpoints.register_endpoint"))
    with session_for(env.engine) as session:
        approval = approval_service.create_pending(
            session,
            action="wsp_endpoint_register",
            executor_key=key,
            payload={"body": {**PROPOSAL, "api_key": SENTINEL}},
            requested_by="agent:forge-mcp",
            summary={},
        )
        approval_id = approval.id
    answer = env.approve(approval_id)
    (row,) = env.rows("mf_approvals")
    assert row["status"] == "failed"
    assert _j(row["error"])["code"] == "wsp_endpoint_agent_key_refused"
    # Operator 2026-11-26: the stored key is never echoed by an approval answer.
    assert SENTINEL not in answer.text
    assert answer.json()["payload"]["body"]["api_key"] == "[redacted]"
    for path in (f"/api/v1/approvals/{approval_id}", "/api/v1/approvals"):
        assert SENTINEL not in env.client.get(path).text
    assert SENTINEL in json.dumps(_j(row["payload"]))  # the stored row keeps it
    assert env.count("mf_wsp_endpoints") == 0


# ------------------------------------------------------------ the operator, unchanged


def test_the_operators_registration_is_unchanged_and_files_no_approval(env: Env) -> None:
    r = env.client.post("/api/v1/wsp/endpoints", json=EXTERNAL)
    assert r.status_code == 201, r.text
    doc = r.json()
    assert doc["has_key"] is True and doc["locality"] == "external"
    assert doc["price_in_per_mtok"] == "3.000000"
    assert SENTINEL not in r.text
    local = env.client.post("/api/v1/wsp/endpoints", json={**PROPOSAL, "price_in_per_mtok": "0"})
    assert local.status_code == 201, local.text
    assert env.count("mf_approvals") == 0
    assert {r["created_by"] for r in env.rows("mf_wsp_endpoints")} == {"operator"}


def test_patch_and_delete_stay_the_operators_and_name_the_screen(env: Env) -> None:
    assert env.client.post("/api/v1/wsp/endpoints", json=PROPOSAL).status_code == 201
    for r in (
        env.client.patch(
            "/api/v1/wsp/endpoints/miserver-gemma4-26b",
            json={"action_protocol": "tool_calls"},
            headers=AGENT,
        ),
        env.client.delete("/api/v1/wsp/endpoints/miserver-gemma4-26b", headers=AGENT),
    ):
        assert r.status_code == 403, r.text
        error = r.json()["error"]
        assert error["code"] == "wsp_endpoint_operator_only"
        assert "Editor endpoints screen" in error["message"]
        assert "#/wsp/endpoints" in error["message"]
        assert "forge_wsp_register_endpoint" in error["message"]
    assert env.count("mf_approvals") == 0
    (row,) = env.rows("mf_wsp_endpoints")
    assert row["action_protocol"] == "structured" and row["removed_at"] is None
