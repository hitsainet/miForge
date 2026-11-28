"""A run made by a retired provider keeps its whole record, and its replay is refused by name
(feature 018; FTASKS 15.1, 15.2; FR-018.18; F18-D3, Q3a).

Production holds completed ``weightspace`` runs from 016-P1 to 018-P3. The fixture makes one as it
was made then, through the real start path, driver and intake: ``weightspace`` registered and
enabled while the run is started and its result taken in, then the registration taken away, as
the 018-P4 deploy takes it away. Every read then goes through the production app's routes:

- ``POST /wsp/runs/{id}/replay`` answers 409 ``replay_unavailable_provider_retired`` naming the
  provider, and stores no run (``wsp.runs.create_replay``; control: the check removed, and the
  replay is queued);
- the run's document, list row, transcript, editor calls, candidate, lineage and candidate
  run-document all read in full, each still naming ``weightspace`` 0.3.0.
"""

from __future__ import annotations

import json
from collections.abc import Iterator
from decimal import Decimal
from pathlib import Path
from typing import Any

import pytest
import sqlalchemy as sa
from fastapi.testclient import TestClient

from core.errors import Refusal
from tests.integration.wsp.test_dispatch import World
from tests.integration.wsp.test_dispatch import world as world
from tests.integration.wsp.test_replay_and_rerun import (
    PROGRAM,
    TRANSCRIPT,
    _program_patch,
    _session,
)
from tests.integration.wsp.test_run_cause import _client
from tests.integration.wsp.test_runs_lifecycle import _store
from wsp import runs, schemas
from wsp.models import WspCandidate, WspUsage
from wsp.providers import registry
from wsp.settings import get_wsp_settings

RETIRED = {"id": "weightspace", "version": "0.3.0"}


@pytest.fixture()
def historic(world: World, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Iterator[str]:
    """A completed ``weightspace`` session with a candidate and one metered editor call, made while
    the provider was registered; the registration is gone when the test body runs."""
    with monkeypatch.context() as before:
        before.setitem(
            registry.PROVIDERS,
            "weightspace",
            registry.ProviderDescriptor(
                id="weightspace",
                version="0.3.0",
                operations=frozenset({"inspect", "create", "repair"}),
                styles=frozenset({"session", "program"}),
            ),
        )
        before.setenv("WSP_PROVIDERS_AVAILABLE", "weightspace,miwsp_engine")
        get_wsp_settings.cache_clear()
        world.settings = get_wsp_settings()
        world.body = {**world.body, "provider": {"id": "weightspace"}}
        manifest = _store(world, tmp_path)
        made_by = {"provider": "weightspace", "provider_version": "0.3.0"}
        run_id, _ = _session(
            world, tmp_path, members=_program_patch(world, manifest, PROGRAM, created_by=made_by)
        )
    get_wsp_settings.cache_clear()
    world.settings = get_wsp_settings()
    assert "weightspace" not in registry.PROVIDERS, "the fixture left the provider registered"
    with world.factory() as session:
        session.add(
            WspUsage(
                run_id=run_id,
                seq=0,
                endpoint_id="ep-local",
                input_tokens=1200,
                output_tokens=80,
                cost_usd=Decimal("0"),
                at=runs.utcnow(),
                outcome="served",
                http_status=200,
                latency_ms=900,
                finish_reason="tool_calls",
            )
        )
        session.commit()
    row = world.run(run_id)
    assert (row["status"], row["outcome"], row["provider_id"]) == (
        "completed",
        "candidate",
        "weightspace",
    ), row
    yield run_id


def _app(world: World, monkeypatch: pytest.MonkeyPatch) -> TestClient:
    return _client(world, monkeypatch)


def _count(world: World) -> int:
    with world.engine.connect() as conn:
        return int(conn.execute(sa.text("SELECT COUNT(*) FROM mf_wsp_runs")).scalar_one())


def test_a_replay_of_a_weightspace_run_is_refused_by_name(
    world: World, historic: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    before = _count(world)
    answer = _app(world, monkeypatch).post(f"/api/v1/wsp/runs/{historic}/replay")
    assert answer.status_code == 409, answer.text
    error = answer.json()["error"]
    assert error["code"] == "replay_unavailable_provider_retired", error
    assert "'weightspace'" in error["message"] and "retired" in error["message"], error
    assert _count(world) == before  # nothing queued in its place


def test_the_service_refuses_it_before_queueing_naming_the_provider(
    world: World, historic: str
) -> None:
    before = _count(world)
    with world.factory() as session, pytest.raises(Refusal) as refused:
        runs.create_replay(session, historic, "operator", world.settings)
    assert refused.value.code == "replay_unavailable_provider_retired"
    assert refused.value.details == {"run_id": historic, "provider": "weightspace"}
    assert _count(world) == before


def test_an_engine_run_still_replays(world: World, tmp_path: Path) -> None:
    """The control beside the refusal: the check names the retired provider, not every replay."""
    run_id, _ = _session(world, tmp_path)
    with world.factory() as session:
        row = runs.create_replay(session, run_id, "operator", world.settings)
        assert row.style == "replay" and row.provider_id == "miwsp_engine"


def test_the_weightspace_runs_record_reads_in_full(
    world: World, historic: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    client = _app(world, monkeypatch)
    row = world.run(historic)

    document = client.get(f"/api/v1/wsp/runs/{historic}")
    assert document.status_code == 200, document.text
    doc = document.json()
    schemas.validate_run(doc)
    assert doc["request"]["provider"] == RETIRED
    assert (doc["status"], doc["outcome"], doc["stop_reason"]) == (
        "completed",
        "candidate",
        "submitted",
    )
    assert doc["patch_sha256"] == row["patch_sha256"] and doc["patch_sha256"]
    assert doc["content_sha256"] == row["content_sha256"] and doc["content_sha256"]
    assert doc["candidate_id"]

    (listed,) = [
        r for r in client.get("/api/v1/wsp/runs").json()["runs"] if r["run_id"] == historic
    ]
    assert listed["provider"] == "weightspace" and listed["status"] == "completed"

    transcript = client.get(f"/api/v1/wsp/runs/{historic}/transcript")
    assert transcript.status_code == 200, transcript.text
    served = transcript.json()
    assert served["transcript_sha256"] == row["transcript_sha256"]
    assert served["entries"] == [json.loads(TRANSCRIPT)] and served["partial"] is False

    calls = client.get(f"/api/v1/wsp/runs/{historic}/calls")
    assert calls.status_code == 200, calls.text
    (call,) = calls.json()["calls"]
    assert (call["seq"], call["input_tokens"], call["output_tokens"]) == (0, 1200, 80)

    with world.factory() as session:
        cid = session.scalars(
            sa.select(WspCandidate.candidate_id).where(WspCandidate.run_id == historic)
        ).one()
    assert doc["candidate_id"] == cid
    candidate = client.get(f"/api/v1/wsp/candidates/{cid}")
    assert candidate.status_code == 200, candidate.text
    assert candidate.json()["patch_sha256"] == row["patch_sha256"]
    lineage = client.get(f"/api/v1/wsp/candidates/{cid}/lineage").json()
    (step,) = lineage["steps"]
    assert (step["provider"], step["provider_version"]) == ("weightspace", "0.3.0")
    assert step["transcript_sha256"] == row["transcript_sha256"]
    run_document = client.get(f"/api/v1/wsp/candidates/{cid}/run-document")
    assert run_document.status_code == 200, run_document.text
    schemas.validate_run(run_document.json())
    assert run_document.json()["request"]["provider"] == RETIRED


def test_a_new_run_naming_it_is_refused_while_its_old_run_reads(
    world: World, historic: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The two halves side by side, on one database: the old run reads; a new start naming the
    same provider is ``provider_unknown`` (422), saying it was retired, and stores nothing."""
    client = _app(world, monkeypatch)
    before = _count(world)
    body: dict[str, Any] = {
        **world.body,
        "endpoint_id": "ep-local",
        "kind": "repair",
        "provider": {"id": "weightspace"},
    }
    answer = client.post("/api/v1/wsp/runs", json=body)
    assert answer.status_code == 422, answer.text
    error = answer.json()["error"]
    assert error["code"] == "provider_unknown" and "retired" in error["message"], error
    assert _count(world) == before
    assert client.get(f"/api/v1/wsp/runs/{historic}").status_code == 200
