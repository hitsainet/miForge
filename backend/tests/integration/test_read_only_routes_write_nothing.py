"""The ``POST`` routes a read-only tool calls write nothing (FR-017.14; 017-D12).

Each route in ``READ_ONLY_ROUTES`` is driven over the transport, through the production app on a
MIGRATED database, with the payload the MCP tool sends (``EXPECTED_CALLS``), against a world in
which it succeeds: a run with saved checkpoints, a pack draft with filled slots, imported parts.
Before and after each call the test takes:

- the row count of EVERY table in the database (a superset of ``Base.metadata`` and the F12
  approvals metadata, both asserted present);
- a recursive listing (path, size, mtime) of every artifact root and of ``HF_HOME`` (ADR-004
  makes the filesystem the truth);
- every Celery task sent (``Task.apply_async`` and ``Celery.send_task`` recorded).

All three must be unchanged. A route that writes moves its tool to write, recorded.
"""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pytest
import sqlalchemy as sa
from fastapi.testclient import TestClient

from tests.support.mcp_ledger import (
    EXPECTED_CALLS,
    READ_ONLY_ROUTES,
    full,
    template_matches,
)

#: Payloads that make the route SUCCEED in this world where the ledger's sample would be
#: refused (the ledger records the tool's shape, not a resolvable input). Keyed by tool.
SUCCEEDING_BODY: dict[str, Any] = {}


@dataclass
class World:
    client: TestClient
    engine: Any
    roots: list[Path]
    ids: dict[str, str]
    sent: list[str]


def _listing(roots: list[Path]) -> dict[str, tuple[int, int]]:
    out: dict[str, tuple[int, int]] = {}
    for root in roots:
        if not root.exists():
            out[str(root)] = (-1, -1)
            continue
        for path in sorted(root.rglob("*")):
            stat = path.lstat()
            out[str(path)] = (stat.st_size if path.is_file() else 0, stat.st_mtime_ns)
    return out


def _row_counts(engine: Any) -> dict[str, int]:
    tables = sa.inspect(engine).get_table_names()
    with engine.connect() as conn:
        return {
            t: int(conn.execute(sa.text(f'SELECT COUNT(*) FROM "{t}"')).scalar_one())
            for t in tables
        }


@pytest.fixture
def world(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Iterator[World]:
    from celery import Celery
    from celery.app.task import Task

    from api.app import create_app
    from api.deps import session_dependency
    from core import runtime_settings
    from core.config import reset_settings
    from packs import authoring
    from packs.settings import reset_pack_settings
    from packs.slots import SlotKind
    from services.reindex import reindex
    from tests.support import pack_fixtures as fx
    from tests.support import signal_fixtures
    from tests.support.checkpoints import write_chain
    from tests.support.db import migrated_engine, session_for
    from tests.support.jobs import insert_run
    from training.signals.settings import reset_signal_settings

    roots = {
        "CHECKPOINT_ROOT": tmp_path / "checkpoints",
        "TRAINER_STATE_ROOT": tmp_path / "trainer-state",
        "PARTS_ROOT": tmp_path / "parts",
        "PACKS_ROOT": tmp_path / "packs",
        "HF_HOME": tmp_path / "hf-home",
    }
    for name, path in roots.items():
        path.mkdir(parents=True, exist_ok=True)
        monkeypatch.setenv(name, str(path))
    for name in ("MISERVER_BASE_URL", "MIDATAWORKS_BASE_URL", "MISERVER_TRAINER_TOKEN"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("SIGNAL_SFT_MASKING_POLICY", "completion_only")
    reset_settings()
    reset_pack_settings()
    reset_signal_settings()
    runtime_settings.reset()

    sent: list[str] = []

    def record_apply(self: Any, *args: Any, **kwargs: Any) -> None:
        sent.append(f"apply_async:{self.name}")

    def record_send(self: Any, name: str, *args: Any, **kwargs: Any) -> None:
        sent.append(f"send_task:{name}")

    monkeypatch.setattr(Task, "apply_async", record_apply)
    monkeypatch.setattr(Celery, "send_task", record_send)

    engine = migrated_engine(tmp_path / "db")
    with session_for(engine) as session:
        insert_run(session, "run-1", status="completed", last_step=100)
        part = signal_fixtures.corpus_part(session, tmp_path, texts=["one", "two"])
        slots: dict[str, Any] = {}
        for slot, origin, digest in (
            (SlotKind.PROMPT_SET, "/data/prompts.jsonl", "a"),
            (SlotKind.TEST_SET, "/data/test.jsonl", "b"),
        ):
            _part_id, provenance = fx.import_external_part(
                session,
                parts_root=tmp_path / "parts",
                export_root=tmp_path / f"export-{slot.value}",
                origin=origin,
                content_sha256=digest * 64,
                rows=5,
            )
            slots[slot.value] = fx.external_slot(
                origin=provenance["source"],
                content_sha256=provenance["content_sha256"],
                retrieved_at=provenance["retrieved_at"],
            )
    write_chain(roots["CHECKPOINT_ROOT"], run_id="run-1", steps=(40, 100))
    with session_for(engine) as session:
        reindex(session, root=roots["CHECKPOINT_ROOT"])
    created = authoring.create(name="legal", domain="legal")
    authoring.commit_step(
        created.draft_id, 2, slots={SlotKind.PROMPT_SET.value: slots[SlotKind.PROMPT_SET.value]}
    )
    authoring.commit_step(
        created.draft_id, 5, slots={SlotKind.TEST_SET.value: slots[SlotKind.TEST_SET.value]}
    )

    from tests.support import part_fixtures

    manifest, _ = part_fixtures.miforge_set_manifest()
    SUCCEEDING_BODY.clear()
    SUCCEEDING_BODY["forge_validate_part"] = {"manifest": manifest}
    SUCCEEDING_BODY["forge_preflight_signal_source"] = {
        "spec": {"source": "corpus", "parts": [{"kind": "corpus", "part_id": str(part["id"])}]}
    }
    # 2026-11-26, forge_wsp_validate_run: an imported bundle and the ledger's local endpoint,
    # with WSP enabled, so the start body freezes (the ledger's sample names no real bundle).
    SUCCEEDING_BODY["forge_wsp_validate_run"] = _wsp_world(engine, tmp_path, monkeypatch)

    app = create_app()

    def override() -> Iterator[Any]:
        with session_for(engine) as session:
            yield session

    app.dependency_overrides[session_dependency] = override
    client = TestClient(app, raise_server_exceptions=False)
    yield World(
        client=client,
        engine=engine,
        roots=list(roots.values()),
        ids={"draft-1": created.draft_id},
        sent=sent,
    )
    app.dependency_overrides.clear()
    engine.dispose()
    from wsp.settings import get_wsp_settings

    get_wsp_settings.cache_clear()
    runtime_settings.reset()
    reset_settings()
    reset_pack_settings()
    reset_signal_settings()


def _wsp_world(engine: Any, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> dict[str, Any]:
    """An imported bundle and a local miServer endpoint, WSP enabled; the start body for them."""
    import json

    from tests.integration.wsp.wsp_db import NOW, insert
    from tests.support.mcp_gate_probe import WSP_LEDGER_ENDPOINT
    from tests.support.wsp_fixtures import enable_wsp, make_bundle
    from wsp.settings import get_wsp_settings

    fx = make_bundle(tmp_path / "wsp-bundle", rule="mean")
    with engine.begin() as conn:
        insert(
            conn,
            "mf_wsp_bundles",
            bundle_sha256=fx.manifest["bundle_sha256"],
            owner="miworkbench",
            bundle_id=fx.manifest["bundle_id"],
            bundle_kind="linear_probe",
            artifact_kind="probe",
            artifact_id="p",
            detector_digest=None,
            model="m",
            layer=12,
            sensitivity_value="not_sensitive",
            manifest=json.dumps(fx.manifest),
            bytes=1,
            import_status="imported",
            received_at=NOW,
            received_by="operator",
        )
        insert(conn, "mf_wsp_endpoints", **WSP_LEDGER_ENDPOINT, created_at=NOW)
    enable_wsp(monkeypatch)
    get_wsp_settings.cache_clear()
    return {
        "bundle": {
            "owner": "miworkbench",
            "bundle_id": fx.manifest["bundle_id"],
            "bundle_sha256": fx.manifest["bundle_sha256"],
        },
        "kind": "repair",
        "mode": "session",
        "task": "fewer false alarms",
        "endpoint_id": WSP_LEDGER_ENDPOINT["id"],
        "sensitivity": "not_sensitive",
        "primary_sets": ["eval"],
    }


def _read_only_tools() -> set[str]:
    """The tools the LIVE registry marks read-only."""
    import asyncio

    from mcp_server.config import MCPSettings
    from mcp_server.server import build_server

    mcp, _ = build_server(MCPSettings(auth_token="x" * 32))
    return {
        t.name
        for t in asyncio.run(mcp.list_tools())
        if t.annotations is not None and t.annotations.read_only_hint
    }


def _calls_on_listed_routes() -> list[tuple[str, str, str, Any]]:
    """(tool, method, path, json body) for each READ-ONLY tool's call on a listed route.

    ``forge_continue_run`` calls ``/runs/{}/continue`` too, with ``submit: true``; it is
    destructive, so its call is not a read-only call and is not driven here.
    """
    read_only = _read_only_tools()
    out = []
    for tool, (method, path, _args, recorded, _cls) in EXPECTED_CALLS.items():
        if tool not in read_only:
            continue
        for m, template in READ_ONLY_ROUTES:
            if method == m and template_matches(template.replace("{}", "{x}"), full(path)):
                out.append((tool, method, full(path), recorded.get("json_body")))
    return out


def test_both_metadata_objects_are_counted(world: World) -> None:
    from db.base import Base
    from models.approval import F12Base

    counted = set(_row_counts(world.engine))
    assert set(Base.metadata.tables) <= counted
    assert set(F12Base.metadata.tables) <= counted


def test_every_listed_route_succeeds_and_writes_nothing(world: World) -> None:
    calls = _calls_on_listed_routes()
    assert {(m, p) for _, m, p, _ in calls}  # found the calls
    exercised: set[tuple[str, str]] = set()
    for tool, method, path, body in calls:
        for placeholder, real in world.ids.items():
            path = path.replace(placeholder, real)
        if tool in SUCCEEDING_BODY:
            body = SUCCEEDING_BODY[tool]
        rows_before = _row_counts(world.engine)
        files_before = _listing(world.roots)
        sent_before = list(world.sent)

        response = world.client.request(method, path, json=body)

        assert response.status_code < 300, (tool, path, response.status_code, response.text)
        assert _row_counts(world.engine) == rows_before, (tool, "rows changed")
        assert _listing(world.roots) == files_before, (tool, "files changed")
        assert world.sent == sent_before, (tool, "a Celery task was sent")
        for m, template in READ_ONLY_ROUTES:
            if method == m and template_matches(template.replace("{}", "{x}"), full(path)):
                exercised.add((m, template))
    assert exercised == set(READ_ONLY_ROUTES), set(READ_ONLY_ROUTES) - exercised


#: A non-read-only tool that calls a listed route with a DIFFERENT payload, and why that call
#: is not the read the route is listed for.
SAME_ROUTE_OTHER_PAYLOAD = {
    "forge_continue_run": "sends submit: true, which files a hotupdate_attach approval and "
    "starts a run once approved; forge_plan_continue's submit: false is the read",
}


def test_no_write_tool_shares_a_listed_route_unexplained() -> None:
    """Listing a route a write tool calls (``POST /pack-drafts``) would make a write look read."""
    read_only = _read_only_tools()
    shared = {
        tool
        for tool, (method, path, *_rest) in EXPECTED_CALLS.items()
        if tool not in read_only
        and any(
            method == m and template_matches(template.replace("{}", "{x}"), full(path))
            for m, template in READ_ONLY_ROUTES
        )
    }
    assert shared == set(SAME_ROUTE_OTHER_PAYLOAD), shared
