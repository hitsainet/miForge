"""Editor endpoints through the mounted app (feature 016; FTASKS 6.6, 6.7; FR-016.16; D15, PF-16,
PF-24, PF-29, P50, P61, R1B-M8, R2A-9).

The app is ``create_app()`` itself on a migrated database; the sealed key is opened by the
gateway's own ``sealing.open_sealed`` against the document the internal application serves, so
a key the backend sealed is proven to open in the gateway, and only for an unchanged row.
"""

from __future__ import annotations

import base64
import logging
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

SENTINEL = "sk-SENTINEL-c0ffee-9b1d"
AGENT = {"X-Miforge-Agent": "agent:test"}
ANTHROPIC = {
    "id": "claude-ed",
    "kind": "anthropic",
    "base_url": "https://api.anthropic.com",
    "model": "claude-editor",
    "price_in_per_mtok": "3",
    "price_out_per_mtok": "15",
    "api_key": SENTINEL,
}


class Env:
    def __init__(self, engine: sa.Engine, client: TestClient, keys: tuple[str, str]) -> None:
        self.engine, self.client, (self.private, self.public) = engine, client, keys

    def rows(self, table: str) -> list[dict[str, Any]]:
        with self.engine.connect() as conn:
            return [dict(r._mapping) for r in conn.execute(sa.text(f"SELECT * FROM {table}"))]


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
    yield Env(engine, TestClient(app, raise_server_exceptions=False), keys)
    engine.dispose()
    get_wsp_settings.cache_clear()


def test_an_agent_is_refused_the_writes_and_may_list(env: Env) -> None:
    # 2026-11-20: an agent's POST is a proposal now (test_endpoint_proposals.py); one with a key
    # is refused by name before anything is filed. DELETE and PATCH stay the operator's.
    r = env.client.post("/api/v1/wsp/endpoints", json=ANTHROPIC, headers=AGENT)
    assert r.status_code == 403 and r.json()["error"]["code"] == "wsp_endpoint_agent_key_refused"
    assert env.rows("mf_approvals") == [] and env.rows("mf_wsp_endpoints") == []
    assert env.client.post("/api/v1/wsp/endpoints", json=ANTHROPIC).status_code == 201
    d = env.client.delete("/api/v1/wsp/endpoints/claude-ed", headers=AGENT)
    assert d.status_code == 403 and d.json()["error"]["code"] == "wsp_endpoint_operator_only"
    assert env.client.get("/api/v1/wsp/endpoints", headers=AGENT).status_code == 200
    assert env.client.delete("/api/v1/wsp/endpoints/claude-ed").status_code == 200
    assert env.client.get("/api/v1/wsp/endpoints").json() == {"endpoints": []}


def test_the_key_is_sealed_never_shown_never_logged(
    env: Env, caplog: pytest.LogCaptureFixture
) -> None:
    caplog.set_level(logging.DEBUG)
    created = env.client.post("/api/v1/wsp/endpoints", json=ANTHROPIC)
    listed = env.client.get("/api/v1/wsp/endpoints")
    assert created.status_code == 201, created.text
    (doc,) = listed.json()["endpoints"]
    assert doc["has_key"] is True and doc["key_fingerprint"] == sealing.fingerprint(SENTINEL)
    assert "key_ciphertext" not in doc and "api_key" not in doc
    for text in (created.text, listed.text, caplog.text):
        assert SENTINEL not in text
    (row,) = env.rows("mf_wsp_endpoints")
    assert SENTINEL.encode() not in bytes(row["key_ciphertext"])
    assert all(SENTINEL not in str(r) for r in env.rows("mf_approvals"))
    # The gateway opens it, against what the internal application serves for this row.
    from api.wsp_internal_app import endpoint_document
    from tests.support.jobs import session_for as s_for
    from wsp import runs

    with s_for(env.engine) as session:
        served = endpoint_document(runs.live_endpoint(session, "claude-ed"))  # type: ignore[arg-type]
    blob = base64.b64decode(served["key_ciphertext"])
    assert sealing.open_sealed(env.private, blob, sealing.aad_digest(served)) == SENTINEL.encode()


@pytest.mark.parametrize(
    "column, value",
    [("base_url", "https://api.anthropic.com/v1"), ("model", "other"), ("price_out_per_mtok", 0)],
)
def test_a_changed_row_does_not_open(env: Env, column: str, value: Any) -> None:
    """R1B-M8: the gateway refuses a row whose routing or price field changed after sealing."""
    assert env.client.post("/api/v1/wsp/endpoints", json=ANTHROPIC).status_code == 201
    with env.engine.begin() as conn:
        conn.execute(sa.text(f"UPDATE mf_wsp_endpoints SET {column} = :v"), {"v": value})
    from api.wsp_internal_app import endpoint_document
    from wsp import runs

    with session_for(env.engine) as session:
        served = endpoint_document(runs.live_endpoint(session, "claude-ed"))  # type: ignore[arg-type]
    with pytest.raises(sealing.KeyBindingFailed):
        sealing.open_sealed(
            env.private, base64.b64decode(served["key_ciphertext"]), sealing.aad_digest(served)
        )


def test_a_body_locality_is_ignored(env: Env) -> None:
    """PF-16: the locality is derived from the host."""
    r = env.client.post("/api/v1/wsp/endpoints", json={**ANTHROPIC, "locality": "local"})
    assert r.status_code == 201 and r.json()["locality"] == "external"
    (row,) = env.rows("mf_wsp_endpoints")
    assert row["locality"] == "external"


@pytest.mark.parametrize(
    "change, code",
    [
        # The credential rule is providers.settings' own (FR-016.16 reuses it), with its code.
        ({"base_url": "https://user:pw@api.anthropic.com"}, "inference_provider_misconfigured"),
        ({"base_url": "https://api.anthropic.com/api/v1/admin"}, "wsp_endpoint_url_refused"),
        ({"base_url": "https://api.anthropic.com/v1?x=1"}, "wsp_endpoint_url_refused"),
        ({"base_url": "https://api.example.org"}, "wsp_endpoint_url_refused"),
        ({"base_url": "http://10.0.0.5:8000/v1"}, "wsp_endpoint_url_refused"),
        ({"price_in_per_mtok": None}, "wsp_endpoint_pricing_required"),
        ({"price_out_per_mtok": None}, "wsp_endpoint_pricing_required"),
    ],
)
def test_refused_registrations_store_nothing(env: Env, change: dict[str, Any], code: str) -> None:
    r = env.client.post("/api/v1/wsp/endpoints", json={**ANTHROPIC, **change})
    assert r.json()["error"]["code"] == code, r.text
    assert "pw@" not in r.text and SENTINEL not in r.text
    assert env.rows("mf_wsp_endpoints") == []


def test_a_local_miserver_endpoint_needs_no_key_or_prices(env: Env) -> None:
    body = {
        "id": "local-ed",
        "kind": "openai_compatible",
        "base_url": "http://miserver-backend.miserver.svc:8000/v1",
        "model": "qwen",
        "serving": "miserver",
    }
    r = env.client.post("/api/v1/wsp/endpoints", json=body)
    assert r.status_code == 201, r.text
    assert r.json()["locality"] == "local" and r.json()["has_key"] is False
    generic = env.client.post(
        "/api/v1/wsp/endpoints", json={**body, "id": "x", "serving": "generic"}
    )
    assert generic.json()["error"]["code"] == "wsp_endpoint_url_refused"


# ------------------------------------------------- R3 review fix (OR3-016-5), 2026-11-16


@pytest.mark.parametrize(
    "public",
    ["{key}\n", "-----BEGIN PUBLIC KEY-----\n{key}\n-----END PUBLIC KEY-----", "{key}AAAA", ""],
    ids=["trailing-newline", "pem", "wrong-length", "unset"],
)
def test_a_malformed_or_unset_gateway_key_is_refused_with_what_to_do(
    env: Env, monkeypatch: pytest.MonkeyPatch, public: str
) -> None:
    """A malformed WSP_GATEWAY_PUBLIC_KEY was a 500 (binascii.Error from the seal). It is now
    ``wsp_gateway_key_not_configured`` (503), saying what the value must look like, and nothing is
    stored."""
    monkeypatch.setenv("WSP_GATEWAY_PUBLIC_KEY", public.format(key=env.public))
    get_wsp_settings.cache_clear()
    r = env.client.post("/api/v1/wsp/endpoints", json=ANTHROPIC)
    assert r.status_code == 503, r.text
    error = r.json()["error"]
    assert error["code"] == "wsp_gateway_key_not_configured"
    assert "WSP_GATEWAY_PUBLIC_KEY" in error["message"]
    if public:
        assert "base64" in error["message"] and "register the endpoint again" in error["message"]
    assert env.rows("mf_wsp_endpoints") == []
    assert SENTINEL not in r.text


# ------------------------------------------- feature 018: the action protocol and the window

MISERVER = {
    "id": "local-ed",
    "kind": "openai_compatible",
    "base_url": "http://miserver-backend.miserver.svc:8000/v1",
    "model": "qwen",
    "serving": "miserver",
}


def test_registration_stores_the_protocol_and_window_and_lists_them(env: Env) -> None:
    """FR-018.8: both fields are accepted at registration and shown by ``GET``; a body that names
    neither gets ``tool_calls`` and no window."""
    r = env.client.post(
        "/api/v1/wsp/endpoints",
        json={**MISERVER, "action_protocol": "structured", "context_window_tokens": 32768},
    )
    assert r.status_code == 201, r.text
    assert (r.json()["action_protocol"], r.json()["context_window_tokens"]) == ("structured", 32768)
    plain = env.client.post("/api/v1/wsp/endpoints", json=ANTHROPIC)
    assert plain.status_code == 201, plain.text
    listed = {e["id"]: e for e in env.client.get("/api/v1/wsp/endpoints").json()["endpoints"]}
    assert (listed["local-ed"]["action_protocol"], listed["local-ed"]["context_window_tokens"]) == (
        "structured",
        32768,
    )
    assert (
        listed["claude-ed"]["action_protocol"],
        listed["claude-ed"]["context_window_tokens"],
    ) == (
        "tool_calls",
        None,
    )
    rows = {r["id"]: r for r in env.rows("mf_wsp_endpoints")}
    assert rows["local-ed"]["action_protocol"] == "structured"
    assert rows["local-ed"]["context_window_tokens"] == 32768
    assert rows["claude-ed"]["action_protocol"] == "tool_calls"


@pytest.mark.parametrize(
    "body",
    [
        {**ANTHROPIC, "action_protocol": "structured"},
        {**ANTHROPIC, "kind": "openai_compatible", "action_protocol": "structured"},
    ],
    ids=["anthropic", "openai-generic"],
)
def test_structured_is_refused_for_anything_but_a_miserver_openai_endpoint(
    env: Env, body: dict[str, Any]
) -> None:
    r = env.client.post("/api/v1/wsp/endpoints", json=body)
    assert r.status_code == 422, r.text
    assert r.json()["error"]["code"] == "wsp_endpoint_protocol_unsupported"
    assert env.rows("mf_wsp_endpoints") == []


@pytest.mark.parametrize(
    "change",
    [
        {"action_protocol": "free_text"},
        {"action_protocol": None},
        {"context_window_tokens": 0},
        {"context_window_tokens": -5},
        {"context_window_tokens": "8192"},
        {"context_window_tokens": True},
        {"context_window_tokens": 10_000_001},
    ],
)
def test_a_bad_protocol_or_window_is_refused_at_registration(
    env: Env, change: dict[str, Any]
) -> None:
    r = env.client.post("/api/v1/wsp/endpoints", json={**MISERVER, **change})
    assert r.status_code == 400 and r.json()["error"]["code"] == "validation_failed", r.text
    assert env.rows("mf_wsp_endpoints") == []


def _patch(env: Env, body: Any, endpoint_id: str = "local-ed", **kw: Any) -> Any:
    return env.client.patch(f"/api/v1/wsp/endpoints/{endpoint_id}", json=body, **kw)


def test_the_patch_is_the_operators_alone(env: Env) -> None:
    """FR-018.8 / FR-018.19: an agent gets 403 and nothing changes; the operator's PATCH lands and
    ``GET`` shows it."""
    assert env.client.post("/api/v1/wsp/endpoints", json=MISERVER).status_code == 201
    r = _patch(env, {"action_protocol": "structured"}, headers=AGENT)
    assert r.status_code == 403 and r.json()["error"]["code"] == "wsp_endpoint_operator_only"
    assert env.rows("mf_wsp_endpoints")[0]["action_protocol"] == "tool_calls"
    r = _patch(env, {"action_protocol": "structured", "context_window_tokens": 16384})
    assert r.status_code == 200, r.text
    assert (r.json()["action_protocol"], r.json()["context_window_tokens"]) == ("structured", 16384)
    (listed,) = env.client.get("/api/v1/wsp/endpoints").json()["endpoints"]
    assert (listed["action_protocol"], listed["context_window_tokens"]) == ("structured", 16384)
    # Each field alone, and null clears the window.
    r = _patch(env, {"context_window_tokens": None})
    assert (r.json()["action_protocol"], r.json()["context_window_tokens"]) == ("structured", None)
    r = _patch(env, {"action_protocol": "tool_calls"})
    assert (r.json()["action_protocol"], r.json()["context_window_tokens"]) == ("tool_calls", None)


@pytest.mark.parametrize(
    "body, code, names",
    [
        ({"model": "other"}, "validation_failed", "model"),
        (
            {"action_protocol": "structured", "base_url": "http://x"},
            "validation_failed",
            "base_url",
        ),
        ({"api_key": SENTINEL}, "validation_failed", "api_key"),
        ({}, "validation_failed", None),
        ({"context_window_tokens": 0}, "validation_failed", None),
        ({"action_protocol": None}, "validation_failed", None),
    ],
)
def test_the_patch_refuses_any_other_field_by_name_and_changes_nothing(
    env: Env, body: dict[str, Any], code: str, names: str | None
) -> None:
    assert env.client.post("/api/v1/wsp/endpoints", json=MISERVER).status_code == 201
    before = env.rows("mf_wsp_endpoints")
    r = _patch(env, body)
    assert r.status_code == 400 and r.json()["error"]["code"] == code, r.text
    if names is not None:
        assert names in r.json()["error"]["message"]
    assert SENTINEL not in r.text
    assert env.rows("mf_wsp_endpoints") == before


def test_the_patch_refuses_structured_for_an_endpoint_that_cannot_use_it(env: Env) -> None:
    assert env.client.post("/api/v1/wsp/endpoints", json=ANTHROPIC).status_code == 201
    r = _patch(env, {"action_protocol": "structured"}, endpoint_id="claude-ed")
    assert r.status_code == 422 and r.json()["error"]["code"] == "wsp_endpoint_protocol_unsupported"
    assert env.rows("mf_wsp_endpoints")[0]["action_protocol"] == "tool_calls"


def test_the_patch_on_an_unknown_or_removed_endpoint_is_refused(env: Env) -> None:
    r = _patch(env, {"context_window_tokens": 4096}, endpoint_id="nope")
    assert r.json()["error"]["code"] == "wsp_endpoint_unknown"
    assert env.client.post("/api/v1/wsp/endpoints", json=MISERVER).status_code == 201
    assert env.client.delete("/api/v1/wsp/endpoints/local-ed").status_code == 200
    r = _patch(env, {"context_window_tokens": 4096})
    assert r.json()["error"]["code"] == "wsp_endpoint_unknown"


def test_a_patched_endpoints_key_still_opens(env: Env) -> None:
    """The two fields are not sealed (FR-018.8): after a PATCH the gateway opens the key against
    the row the internal application serves."""
    assert env.client.post("/api/v1/wsp/endpoints", json=ANTHROPIC).status_code == 201
    assert (
        _patch(env, {"context_window_tokens": 200000}, endpoint_id="claude-ed").status_code == 200
    )
    from api.wsp_internal_app import endpoint_document
    from wsp import runs

    with session_for(env.engine) as session:
        served = endpoint_document(runs.live_endpoint(session, "claude-ed"))  # type: ignore[arg-type]
    blob = base64.b64decode(served["key_ciphertext"])
    assert sealing.open_sealed(env.private, blob, sealing.aad_digest(served)) == SENTINEL.encode()


# ------------------------------------------------------------------------- thinking (2026-11-21)


def test_registration_stores_thinking_and_lists_it_defaulting_to_off(env: Env) -> None:
    r = env.client.post(
        "/api/v1/wsp/endpoints", json={**MISERVER, "thinking": "on", "thinking_tokens": 6000}
    )
    assert r.status_code == 201, r.text
    assert (r.json()["thinking"], r.json()["thinking_tokens"]) == ("on", 6000)
    assert env.client.post("/api/v1/wsp/endpoints", json=ANTHROPIC).status_code == 201
    listed = {e["id"]: e for e in env.client.get("/api/v1/wsp/endpoints").json()["endpoints"]}
    assert (listed["local-ed"]["thinking"], listed["local-ed"]["thinking_tokens"]) == ("on", 6000)
    assert (listed["claude-ed"]["thinking"], listed["claude-ed"]["thinking_tokens"]) == (
        "off",
        None,
    )
    rows = {r["id"]: r for r in env.rows("mf_wsp_endpoints")}
    assert (rows["local-ed"]["thinking"], rows["claude-ed"]["thinking"]) == ("on", "off")


@pytest.mark.parametrize(
    "body",
    [
        {**ANTHROPIC, "thinking": "on"},
        {**ANTHROPIC, "kind": "openai_compatible", "thinking": "on"},
    ],
    ids=["anthropic", "openai-generic"],
)
def test_thinking_on_is_refused_for_anything_but_a_miserver_openai_endpoint(
    env: Env, body: dict[str, Any]
) -> None:
    r = env.client.post("/api/v1/wsp/endpoints", json=body)
    assert r.status_code == 422, r.text
    assert r.json()["error"]["code"] == "wsp_endpoint_thinking_unsupported"
    assert env.rows("mf_wsp_endpoints") == []
    ok = env.client.post("/api/v1/wsp/endpoints", json={**body, "thinking": "default"})
    assert ok.status_code == 201, ok.text


@pytest.mark.parametrize(
    "change",
    [
        {"thinking": "maybe"},
        {"thinking": None},
        {"thinking": "high"},
        {"thinking_tokens": 0},
        {"thinking_tokens": "4096"},
        {"thinking_tokens": 40000, "context_window_tokens": 32768},
    ],
)
def test_a_bad_thinking_or_allowance_is_refused_at_registration(
    env: Env, change: dict[str, Any]
) -> None:
    r = env.client.post("/api/v1/wsp/endpoints", json={**MISERVER, **change})
    assert r.status_code == 400 and r.json()["error"]["code"] == "validation_failed", r.text
    assert env.rows("mf_wsp_endpoints") == []


def test_the_thinking_patch_is_the_operators_alone(env: Env) -> None:
    assert env.client.post("/api/v1/wsp/endpoints", json=MISERVER).status_code == 201
    r = _patch(env, {"thinking": "on"}, headers=AGENT)
    assert r.status_code == 403 and r.json()["error"]["code"] == "wsp_endpoint_operator_only"
    assert env.rows("mf_wsp_endpoints")[0]["thinking"] == "off"
    r = _patch(env, {"thinking": "on", "thinking_tokens": 8192})
    assert r.status_code == 200, r.text
    assert (r.json()["thinking"], r.json()["thinking_tokens"]) == ("on", 8192)
    (listed,) = env.client.get("/api/v1/wsp/endpoints").json()["endpoints"]
    assert (listed["thinking"], listed["thinking_tokens"]) == ("on", 8192)
    # A window below the allowance strands it, so it is refused, and nothing changes.
    before = env.rows("mf_wsp_endpoints")
    r = _patch(env, {"context_window_tokens": 4096})
    assert r.status_code == 400 and r.json()["error"]["code"] == "validation_failed", r.text
    assert env.rows("mf_wsp_endpoints") == before
    r = _patch(env, {"thinking": "default", "thinking_tokens": None})
    assert (r.json()["thinking"], r.json()["thinking_tokens"]) == ("default", None)


def test_the_patch_refuses_thinking_on_for_an_endpoint_that_cannot_use_it(env: Env) -> None:
    assert env.client.post("/api/v1/wsp/endpoints", json=ANTHROPIC).status_code == 201
    r = _patch(env, {"thinking": "on"}, endpoint_id="claude-ed")
    assert r.status_code == 422 and r.json()["error"]["code"] == "wsp_endpoint_thinking_unsupported"
    assert env.rows("mf_wsp_endpoints")[0]["thinking"] == "off"


def test_the_output_cap_is_registered_listed_and_patched_by_the_operator(env: Env) -> None:
    """2026-11-21: ``output_cap`` (on by default), at registration and in the operator's PATCH."""
    r = env.client.post("/api/v1/wsp/endpoints", json={**MISERVER, "output_cap": "off"})
    assert r.status_code == 201 and r.json()["output_cap"] == "off", r.text
    assert env.client.post("/api/v1/wsp/endpoints", json=ANTHROPIC).json()["output_cap"] == "on"
    r = _patch(env, {"output_cap": "on"}, headers=AGENT)
    assert r.status_code == 403
    r = _patch(env, {"output_cap": "on"})
    assert r.status_code == 200 and r.json()["output_cap"] == "on"
    bad = _patch(env, {"output_cap": "sometimes"})
    assert bad.status_code == 400 and bad.json()["error"]["code"] == "validation_failed"


def test_paging_is_registered_listed_and_patched_by_the_operator(env: Env) -> None:
    """2026-11-21: ``paging`` (on by default), at registration and in the operator's PATCH; an
    agent's PATCH is refused and changes nothing."""
    r = env.client.post("/api/v1/wsp/endpoints", json={**MISERVER, "paging": "off"})
    assert r.status_code == 201 and r.json()["paging"] == "off", r.text
    assert env.client.post("/api/v1/wsp/endpoints", json=ANTHROPIC).json()["paging"] == "on"
    before = env.rows("mf_wsp_endpoints")
    assert _patch(env, {"paging": "on"}, headers=AGENT).status_code == 403
    assert env.rows("mf_wsp_endpoints") == before
    r = _patch(env, {"paging": "on"})
    assert r.status_code == 200 and r.json()["paging"] == "on"
    listed = {e["id"]: e for e in env.client.get("/api/v1/wsp/endpoints").json()["endpoints"]}
    assert (listed["local-ed"]["paging"], listed["claude-ed"]["paging"]) == ("on", "on")
    bad = _patch(env, {"paging": "sometimes"})
    assert bad.status_code == 400 and bad.json()["error"]["code"] == "validation_failed"
