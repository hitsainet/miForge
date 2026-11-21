"""Review round 2 of 015 P0: the audit label leaves through no route, refusal or log line, in
any encoding.

Round 1's leak test used ``audit-label-sentinel-3f9a1c``, whose URL-encoded and JSON-escaped
forms ARE the plain form, so only the base64 search could see an encoded leak. The real
label is shaped ``miforge@<host>`` (``gitops/pilot/postgres.yaml``), and ``@`` URL-encodes.
This sentinel carries characters every encoding changes, and the search covers every route
that serves settings or their history, a refused write of the label, and the health answer.
"""

from __future__ import annotations

import base64
import json
import logging
from urllib.parse import quote, quote_plus

import pytest

from core.config import reset_settings
from tests.support.approval_world import World

LABEL = 'miforge@rev2-host/"label" é+1'


def _forms(value: str) -> dict[str, str]:
    raw = value.encode()
    return {
        "plain": value,
        "base64": base64.b64encode(raw).decode().rstrip("="),
        "base64url": base64.urlsafe_b64encode(raw).decode().rstrip("="),
        "url": quote(value, safe=""),
        "url_plus": quote_plus(value),
        "json": json.dumps(value)[1:-1],
        "json_ascii": json.dumps(value, ensure_ascii=True)[1:-1],
        "core": "rev2-host",  # the distinctive part, whatever wraps it
    }


def test_the_label_leaves_nowhere(
    world: World, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    from api import health as health_module

    for name in ("MILLM_AUDIT_LABEL", "MISERVER_AUDIT_LABEL"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("MISERVER_AUDIT_LABEL", LABEL)
    # The OLD name too, retired since P7: the legacy-name path runs and records it
    # (`retired_but_set`), so its record travels where this test looks.
    monkeypatch.setenv("MILLM_AUDIT_LABEL", LABEL)
    monkeypatch.setitem(health_module.DEPENDENCY_PROBES, "redis", lambda: {"status": "ok"})
    reset_settings()
    caplog.set_level(logging.DEBUG)
    try:
        from core.config import get_settings

        assert get_settings().millm_audit_label == LABEL, "the fixture configured it"
        answers = {
            "settings": world.client.get("/api/v1/settings"),
            "history": world.client.get("/api/v1/settings/history"),
            "history_key": world.client.get(
                "/api/v1/settings/history", params={"key": "millm_audit_label"}
            ),
            "refused_write": world.client.put(
                "/api/v1/settings/millm_audit_label", json={"value": "x"}
            ),
            "refused_reset": world.client.delete("/api/v1/settings/millm_audit_label"),
            "health": world.client.get("/api/health"),
        }
    finally:
        reset_settings()
    assert answers["settings"].status_code == 200
    assert answers["refused_write"].status_code >= 400, "the label is not runtime-writable"
    texts = {where: r.text for where, r in answers.items()}
    texts["logs"] = "\n".join(
        f"{r.getMessage()} {json.dumps(r.__dict__, default=str)}" for r in caplog.records
    )
    found = [
        (where, form)
        for where, text in texts.items()
        for form, encoded in _forms(LABEL).items()
        if encoded in text
    ]
    assert not found, f"the audit label left as {found}"
