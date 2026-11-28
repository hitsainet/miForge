"""The old-name evidence store, on the migrated database (FR-015.6; FTID §3.3, §4; CF-4, CF-18,
CF-24, CF-25 m2).

Suite phase P7 removes the ``MILLM_*`` fallback only after seven days with no old name in use.
That reading comes from ``GET /api/health``, which is built from what each process recorded
here. So the store must keep ``last_seen`` across restarts, write at most once a minute per row
per process, and never let a FAILED write or read look like "nothing in use": a worker whose
write fails leaves an error row that the health answer reports (CF-25 m2).

Runs on SQLite by default and on PostgreSQL 15 when ``MIFORGE_TEST_DATABASE_URL`` is set (the CI
``postgres`` job), through the real migration.
"""

from __future__ import annotations

import base64
import json
import logging
from collections.abc import Iterator
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import pytest
import sqlalchemy as sa
from sqlalchemy import Engine

from api.v1.endpoints.settings import served_value
from core import legacy_names
from core.config import Settings, reset_settings
from core.legacy_names import LegacyResolution, LegacyUse
from db.session import reset_engine
from services.legacy_names import store
from services.legacy_names.tables import USES, WRITE_ERRORS
from tests.support.db import migrated_engine, session_for
from tests.support.mcp_harness import build_harness, call_tool

NOW = datetime(2026, 11, 8, 12, 0, 0, tzinfo=UTC)


class Clock:
    """A monotonic clock the test moves by hand."""

    def __init__(self) -> None:
        self.now = 1000.0

    def __call__(self) -> float:
        return self.now


@pytest.fixture
def engine(tmp_path: Path) -> Iterator[Engine]:
    eng = migrated_engine(tmp_path / "db")
    store.reset_throttle()
    yield eng
    store.reset_throttle()
    eng.dispose()


def scope_for(engine: Engine) -> Any:
    return lambda: session_for(engine)


def resolution(*uses: tuple[str, str, str]) -> LegacyResolution:
    return LegacyResolution(
        tuple(LegacyUse("env", old, new, state, "environment") for old, new, state in uses)
    )


IN_USE = resolution(("MILLM_BASE_URL", "MISERVER_BASE_URL", "in_use"))


def rows(engine: Engine, table: sa.Table) -> list[dict[str, Any]]:
    with engine.connect() as conn:
        return [dict(r._mapping) for r in conn.execute(sa.select(table))]


def test_upsert_on_both_engines(engine: Engine) -> None:
    clock = Clock()
    written = store.record(
        IN_USE, reporter="api@pod", now=NOW, clock=clock, scope=scope_for(engine)
    )
    assert written == 1
    clock.now += 61
    later = NOW + timedelta(minutes=2)
    assert (
        store.record(IN_USE, reporter="api@pod", now=later, clock=clock, scope=scope_for(engine))
        == 1
    )
    (row,) = rows(engine, USES)
    assert (row["kind"], row["old_name"], row["state"], row["new_name"]) == (
        "env",
        "MILLM_BASE_URL",
        "in_use",
        "MISERVER_BASE_URL",
    )
    assert store.as_utc(row["first_seen"]) == NOW, "first_seen is set on insert only"
    assert store.as_utc(row["last_seen"]) == later


def test_last_seen_survives_restart(engine: Engine) -> None:
    """A process that restarts (an empty live record) still reports a use seen 2 h ago."""
    store.record(IN_USE, reporter="train@pod", now=NOW, clock=Clock(), scope=scope_for(engine))
    store.reset_throttle()  # the restart
    later = NOW + timedelta(hours=2)
    with session_for(engine) as session:
        stored = store.recent(session, now=later)
    block = store.block(LegacyResolution(), stored, now=later)
    assert block["in_use"] == [
        {
            "kind": "env",
            "old": "MILLM_BASE_URL",
            "new": "MISERVER_BASE_URL",
            "last_seen": "2026-11-08T12:00:00Z",
        }
    ]


def test_throttle_30s_once_61s_twice(engine: Engine) -> None:
    clock = Clock()
    scope = scope_for(engine)
    assert store.record(IN_USE, reporter="api@pod", now=NOW, clock=clock, scope=scope) == 1
    clock.now += 30
    assert store.record(IN_USE, reporter="api@pod", now=NOW, clock=clock, scope=scope) == 0
    clock.now += 31  # 61 s after the first write
    assert store.record(IN_USE, reporter="api@pod", now=NOW, clock=clock, scope=scope) == 1


def test_nothing_to_record_opens_no_session(engine: Engine) -> None:
    def no_scope() -> Any:
        raise AssertionError("a process with no old name in use must not touch the database")

    assert store.record(LegacyResolution(), reporter="api@pod", now=NOW, scope=no_scope) == 0


def _failing_upsert(monkeypatch: pytest.MonkeyPatch) -> None:
    def boom(*_: Any, **__: Any) -> int:
        raise RuntimeError("database went away")

    monkeypatch.setattr(store, "_upsert_uses", boom)


def test_worker_write_failure_records_error_row(
    engine: Engine, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    _failing_upsert(monkeypatch)
    with caplog.at_level(logging.WARNING, logger="services.legacy_names.store"):
        written = store.record(
            IN_USE, reporter="train@pod", now=NOW, clock=Clock(), scope=scope_for(engine)
        )
    assert written == 0
    errors = rows(engine, WRITE_ERRORS)
    assert len(errors) == 1, f"exactly one error row per reporter, got {errors}"
    (error,) = errors
    assert (error["reporter"], error["error"], error["cleared_at"]) == (
        "train@pod",
        "RuntimeError",
        None,
    )
    assert store.as_utc(error["failed_at"]) == NOW
    warned = [
        r for r in caplog.records if r.__dict__.get("code") == "legacy_names_store_unavailable"
    ]
    assert len(warned) == 1 and warned[0].__dict__["reporter"] == "train@pod"
    # Same failure again: still one row (upserted per reporter), and the throttle did NOT
    # advance, so the next heartbeat retries at once.
    store.record(IN_USE, reporter="train@pod", now=NOW, clock=Clock(), scope=scope_for(engine))
    assert len(rows(engine, WRITE_ERRORS)) == 1


def test_health_reports_worker_write_failure(
    engine: Engine, monkeypatch: pytest.MonkeyPatch
) -> None:
    _failing_upsert(monkeypatch)
    store.record(IN_USE, reporter="train@pod", now=NOW, clock=Clock(), scope=scope_for(engine))
    with session_for(engine) as session:
        stored = store.recent(session, now=NOW + timedelta(minutes=1))
    meta = store.meta(read_error=None, stored=stored)
    assert meta == {
        "error": "write failed in train@pod: RuntimeError",
        "in_use_window_hours": 24,
    }


def test_next_success_clears_error(engine: Engine, monkeypatch: pytest.MonkeyPatch) -> None:
    with monkeypatch.context() as patched:
        _failing_upsert(patched)
        store.record(IN_USE, reporter="train@pod", now=NOW, clock=Clock(), scope=scope_for(engine))
    later = NOW + timedelta(minutes=1)
    assert (
        store.record(
            IN_USE, reporter="train@pod", now=later, clock=Clock(), scope=scope_for(engine)
        )
        == 1
    )
    (error,) = rows(engine, WRITE_ERRORS)
    assert store.as_utc(error["cleared_at"]) == later
    with session_for(engine) as session:
        stored = store.recent(session, now=later)
    assert store.meta(read_error=None, stored=stored)["error"] is None


# ------------------------------------------------------------------------------------- leak

SENTINEL_OLD = "sentinel-old-7c1f0e9a"
SENTINEL_NEW = "sentinel-new-52d4b8e3"


def _encoded(value: str) -> list[str]:
    raw = value.encode()
    return [
        value,
        base64.b64encode(raw).decode(),
        base64.urlsafe_b64encode(raw).decode(),
        base64.b64encode(raw).decode().rstrip("="),
    ]


def test_leak_no_value_anywhere(
    engine: Engine,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """No value of any variable reaches the health body, the MCP answer, a log, a dump or a row."""
    from api import health as health_module

    for name in ("MILLM_TRAINER_TOKEN", "MISERVER_TRAINER_TOKEN", "MILLM_AUDIT_LABEL"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("MILLM_TRAINER_TOKEN", SENTINEL_OLD)
    monkeypatch.setenv("MISERVER_TRAINER_TOKEN", SENTINEL_NEW)
    monkeypatch.setenv("MILLM_AUDIT_LABEL", SENTINEL_OLD + "-label")
    monkeypatch.setenv("DATABASE_URL", engine.url.render_as_string(hide_password=False))
    for probe in ("postgres", "redis"):
        monkeypatch.setitem(
            health_module.DEPENDENCY_PROBES, probe, lambda: {"status": "ok", "detail": None}
        )
    reset_settings()
    reset_engine()
    legacy_names.reset_logged()
    try:
        with caplog.at_level(logging.DEBUG):
            settings = Settings()
            legacy_names.log_in_use(settings._legacy_names)
            from core.config import get_settings

            body = health_module.health_payload()
            # Recorded through the real wiring, into the migrated database.
            assert rows(engine, USES), "health_payload recorded nothing; the leak test is vacuous"
            mcp, _client = build_harness(response=body)
            answered = call_tool(mcp, "forge_health", {})
            outputs = {
                "health": json.dumps(body, default=str),
                "forge_health": json.dumps(answered, default=str),
                "logs": "\n".join(
                    f"{r.getMessage()} {json.dumps(r.__dict__, default=str)}"
                    for r in caplog.records
                ),
                "redacted": json.dumps(
                    {
                        k: served_value(k, v)
                        for k, v in get_settings().model_dump(mode="json").items()
                    },
                    default=str,
                ),
                "dump_json": settings.model_dump_json(),
                "uses_rows": json.dumps(rows(engine, USES), default=str),
                "error_rows": json.dumps(rows(engine, WRITE_ERRORS), default=str),
            }
    finally:
        reset_settings()
        reset_engine()
        legacy_names.reset_logged()
    # The OLD token is shadowed by the new one, so it is applied nowhere: it must appear
    # nowhere at all. The new token and the label are field values: they may sit in the dump
    # of the settings object itself (the token exactly once), and nowhere else - not in the
    # served settings (both are SECRET_FIELDS since the 015 review, Q-F1) and not where the
    # legacy-name evidence travels.
    label = SENTINEL_OLD + "-label"
    evidence = ("health", "forge_health", "logs", "redacted", "uses_rows", "error_rows")
    for where, text in outputs.items():
        for form in _encoded(SENTINEL_OLD):
            assert form not in text.replace(label, ""), f"{where} carries the old token ({form!r})"
        for secret in (SENTINEL_NEW, label):
            for form in _encoded(secret):
                if where in evidence:
                    assert form not in text, f"{where} carries a variable's value ({form!r})"
    assert outputs["dump_json"].count(SENTINEL_NEW) == 1
    assert set(body["legacy_names"]) == {"in_use", "also_set", "conflicts", "retired_but_set"}
    assert set(body["legacy_names_meta"]) == {"error", "in_use_window_hours"}
    for listing in body["legacy_names"].values():
        for item in listing:
            assert set(item) <= {"kind", "old", "new", "last_seen"}, item
    assert {u.old for u in settings._legacy_names.uses} == {
        "MILLM_TRAINER_TOKEN",
        "MILLM_AUDIT_LABEL",
    }
