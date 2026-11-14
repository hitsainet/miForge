"""``GET /api/health`` serves ``legacy_names`` and ``legacy_names_meta`` (FR-015.6; CF-4, CF-18,
CF-24; FTID §5.1, §7.2).

The suite's P7 sampler reads these two keys from every app. ``legacy_names`` has EXACTLY four
lists; ``legacy_names_meta`` ALWAYS carries ``error`` (null when the reading is good) and
``in_use_window_hours``. A failed store read is an error in the meta, never an empty block that
would read as "nothing in use", and it never changes the status code.
"""

from __future__ import annotations

from collections.abc import Iterator
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import pytest
from fastapi.testclient import TestClient

from api import health as health_module
from api.app import create_app
from core.config import get_settings, reset_settings
from core.legacy_names import LegacyResolution, LegacyUse
from services.legacy_names import store
from services.legacy_names.store import StoredUse, StoredUses

OK = {"status": "ok", "detail": None}
NOW = datetime(2026, 11, 8, 12, 0, 0, tzinfo=UTC)
LISTS = {"in_use", "also_set", "conflicts", "retired_but_set"}
#: The real writer, kept before the fixture replaces ``store.record`` for the health cases.
REAL_RECORD = store.record


@pytest.fixture(autouse=True)
def _isolated(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> Iterator[None]:
    import os

    for name in list(os.environ):
        if name.upper().startswith(("MILLM_", "MISERVER_")):
            monkeypatch.delenv(name)
    monkeypatch.chdir(tmp_path)
    for probe in ("postgres", "redis"):
        monkeypatch.setitem(health_module.DEPENDENCY_PROBES, probe, lambda: OK)
    monkeypatch.setattr(store, "record", lambda *a, **k: 0)
    reset_settings()
    yield
    reset_settings()


def _stored(monkeypatch: pytest.MonkeyPatch, stored: StoredUses) -> None:
    monkeypatch.setattr(store, "read_recent", lambda **_: stored)


def _live(monkeypatch: pytest.MonkeyPatch, *uses: LegacyUse) -> None:
    get_settings()._legacy_names = LegacyResolution(uses)


def test_block_has_exactly_four_keys(monkeypatch: pytest.MonkeyPatch) -> None:
    _stored(monkeypatch, StoredUses())
    body = health_module.health_payload()
    assert set(body["legacy_names"]) == LISTS
    assert all(value == [] for value in body["legacy_names"].values())
    assert set(body["legacy_names_meta"]) == {"error", "in_use_window_hours"}


def test_meta_always_present_with_error_key(monkeypatch: pytest.MonkeyPatch) -> None:
    """Serialised through the real route: ``error`` is present and null, never omitted."""
    _stored(monkeypatch, StoredUses())
    client = TestClient(create_app(), raise_server_exceptions=False)
    response = client.get("/api/health")
    assert '"legacy_names_meta":{"error":null,"in_use_window_hours":24}' in response.text.replace(
        " ", ""
    )
    assert response.json()["legacy_names_meta"] == {"error": None, "in_use_window_hours": 24}


def test_union_of_live_and_stored(monkeypatch: pytest.MonkeyPatch) -> None:
    _live(
        monkeypatch,
        LegacyUse("env", "MILLM_AUDIT_LABEL", "MISERVER_AUDIT_LABEL", "in_use", "environment"),
    )
    _stored(
        monkeypatch,
        StoredUses(
            uses=(
                StoredUse(
                    "env", "MILLM_BASE_URL", "MISERVER_BASE_URL", "in_use", NOW - timedelta(hours=3)
                ),
            )
        ),
    )
    body = health_module.health_payload()
    assert [item["old"] for item in body["legacy_names"]["in_use"]] == [
        "MILLM_AUDIT_LABEL",
        "MILLM_BASE_URL",
    ]
    stored_item = body["legacy_names"]["in_use"][1]
    assert stored_item == {
        "kind": "env",
        "old": "MILLM_BASE_URL",
        "new": "MISERVER_BASE_URL",
        "last_seen": "2026-11-08T09:00:00Z",
    }


def test_window_23h_in_25h_out(tmp_path: Path) -> None:
    """The 24 h window is applied by the store read itself, on the migrated database."""
    from tests.support.db import migrated_engine, session_for

    engine = migrated_engine(tmp_path / "db")
    try:
        store.reset_throttle()
        for old, hours in (("MILLM_AUDIT_LABEL", 23), ("MILLM_BASE_URL", 25)):
            use = LegacyUse("env", old, old.replace("MILLM", "MISERVER"), "in_use", None)
            REAL_RECORD(
                LegacyResolution((use,)),
                reporter="api@test",
                now=NOW - timedelta(hours=hours),
                scope=lambda: session_for(engine),
            )
        with session_for(engine) as session:
            stored = store.recent(session, now=NOW)
    finally:
        store.reset_throttle()
        engine.dispose()
    assert [u.old for u in stored.uses] == ["MILLM_AUDIT_LABEL"]


def test_failed_store_read_sets_meta_error(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    def broken(**_: Any) -> StoredUses:
        raise ConnectionError("store down")

    monkeypatch.setattr(store, "read_recent", broken)
    _live(
        monkeypatch,
        LegacyUse("env", "MILLM_BASE_URL", "MISERVER_BASE_URL", "in_use", "environment"),
    )
    body = health_module.health_payload()
    assert body["legacy_names_meta"]["error"] == "ConnectionError"
    assert [i["old"] for i in body["legacy_names"]["in_use"]] == [
        "MILLM_BASE_URL"
    ], "the live uses are still served when the stored ones cannot be read"
    assert any(r.__dict__.get("code") == "legacy_names_unreadable" for r in caplog.records)


def test_sorted_by_old(monkeypatch: pytest.MonkeyPatch) -> None:
    names = ["MILLM_TRAINER_TOKEN", "MILLM_AUDIT_LABEL", "MILLM_BASE_URL"]
    _live(
        monkeypatch,
        *(
            LegacyUse("env", n, n.replace("MILLM", "MISERVER"), "also_set", "environment")
            for n in names
        ),
    )
    _stored(monkeypatch, StoredUses())
    body = health_module.health_payload()
    assert [i["old"] for i in body["legacy_names"]["also_set"]] == sorted(names)


def test_conflict_listed_in_conflicts_and_also_set(monkeypatch: pytest.MonkeyPatch) -> None:
    """Through the real settings source: both names set, different values, on the pre-P7
    tables (a live alias; the transition path a rollback re-enables, FTDD §11.8)."""
    from core import legacy_names

    entry = legacy_names.EnvName("millm_base_url", "MISERVER_BASE_URL", "MILLM_BASE_URL")
    monkeypatch.setattr(legacy_names, "ENV_NAMES", (entry,))
    monkeypatch.setattr(legacy_names, "LEGACY_ENV_ALIASES", {"MILLM_BASE_URL": "MISERVER_BASE_URL"})
    monkeypatch.setattr(legacy_names, "RETIRED_ENV_NAMES", {})
    monkeypatch.setenv("MILLM_BASE_URL", "http://old:8000")
    monkeypatch.setenv("MISERVER_BASE_URL", "http://new:8000")
    reset_settings()
    _stored(monkeypatch, StoredUses())
    body = health_module.health_payload()
    pair = {"old": "MILLM_BASE_URL", "new": "MISERVER_BASE_URL"}
    assert body["legacy_names"]["also_set"] == [pair]
    assert body["legacy_names"]["conflicts"] == [pair]
    assert body["legacy_names"]["in_use"] == []


def test_status_code_unchanged_by_store_failure(monkeypatch: pytest.MonkeyPatch) -> None:
    def broken(**_: Any) -> StoredUses:
        raise ConnectionError("store down")

    monkeypatch.setattr(store, "read_recent", broken)
    client = TestClient(create_app(), raise_server_exceptions=False)
    response = client.get("/api/health")
    assert response.status_code == 200
    assert response.json()["degraded_dependencies"] == []
    assert response.json()["legacy_names_meta"]["error"] == "ConnectionError"
