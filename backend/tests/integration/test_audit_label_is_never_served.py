"""The audit label is served only as set or not set (operator decision Q-F1, 015 review).

``training/hotupdate/identity.py`` treats ``millm_audit_label`` as one of the never-logged values
and refers to it only by digest. Until 2026-11-08 ``GET /api/v1/settings`` - and therefore
``forge_get_settings`` - served it in clear. It is now in ``SECRET_FIELDS``. The search covers
the value and its base64 forms (a value can leave encoded, as a Celery body does).
"""

from __future__ import annotations

import base64
import json

import pytest

from api.v1.endpoints.settings import REDACTED
from core.config import reset_settings
from tests.support.approval_world import World
from tests.support.mcp_harness import build_harness, call_tool

LABEL = "audit-label-sentinel-3f9a1c"


def _forms(value: str) -> list[str]:
    raw = value.encode()
    encoded = base64.b64encode(raw).decode()
    return [value, encoded, encoded.rstrip("="), base64.urlsafe_b64encode(raw).decode()]


def test_the_label_value_never_leaves(world: World, monkeypatch: pytest.MonkeyPatch) -> None:
    for name in ("MILLM_AUDIT_LABEL", "MISERVER_AUDIT_LABEL"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("MISERVER_AUDIT_LABEL", LABEL)
    reset_settings()
    try:
        response = world.client.get("/api/v1/settings")
        assert response.status_code == 200, response.text
        body = response.json()
        entry = body["settings"]["millm_audit_label"]
        assert entry["value"] == REDACTED and entry["env_value"] == REDACTED, entry
        mcp, client = build_harness(response=body)
        answered = call_tool(mcp, "forge_get_settings", {})
        assert client.calls and client.calls[0][1] == "/settings"
        for where, text in {
            "GET /api/v1/settings": response.text,
            "forge_get_settings": json.dumps(answered, default=str),
        }.items():
            for form in _forms(LABEL):
                assert form not in text, f"{where} carries the audit label ({form!r})"
    finally:
        reset_settings()


def test_an_unset_label_reads_as_not_set(world: World, monkeypatch: pytest.MonkeyPatch) -> None:
    for name in ("MILLM_AUDIT_LABEL", "MISERVER_AUDIT_LABEL"):
        monkeypatch.delenv(name, raising=False)
    reset_settings()
    try:
        entry = world.client.get("/api/v1/settings").json()["settings"]["millm_audit_label"]
        assert entry["value"] is None and entry["env_value"] is None
    finally:
        reset_settings()
