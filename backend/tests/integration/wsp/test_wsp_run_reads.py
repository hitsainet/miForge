"""The WSP run reads added for agents on 2026-11-26 (the 019 P4 MCP gaps), through the production
app over a run the real driver and intake finished:

- ``GET /wsp/runs/{id}/record``: the run's list row, equal to what ``GET /wsp/runs`` serves for it,
  and its run document, equal to ``GET /wsp/runs/{id}``;
- ``GET /wsp/runs/{id}/transcript`` pages: ``offset``, ``limit`` and ``max_bytes`` bound a page,
  ``total`` and ``next_offset`` walk it, and without them the whole transcript is served;
- ``GET /wsp/runs/{id}/ledger``: each reservation and usage row of the run, with its endpoint,
  tokens, cost and times, totals over every row, its own cursors, and no row of another run.
"""

from __future__ import annotations

import datetime as dt
import json
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest
from fastapi.testclient import TestClient

from tests.integration.wsp.test_dispatch import FakeWorker, World, _running, archive, build_world
from tests.integration.wsp.test_run_cause import _client
from tests.integration.wsp.wsp_db import NOW, insert
from wsp import schemas
from wsp.settings import get_wsp_settings

LINES = [{"turn": i, "role": "assistant", "content": "x" * (100 * (i + 1))} for i in range(5)]


@pytest.fixture()
def world(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Iterator[World]:
    built = build_world(tmp_path, monkeypatch)
    yield built
    built.engine.dispose()
    get_wsp_settings.cache_clear()


@pytest.fixture()
def client(world: World, monkeypatch: pytest.MonkeyPatch) -> TestClient:
    return _client(world, monkeypatch)


def _finished(world: World, transcript: bytes) -> str:
    run_id, worker = _running(world)
    worker.state = "finished"
    worker.data = archive(
        worker.boot,
        run_id,
        **{
            "result.json": json.dumps({"stop_reason": "submitted"}).encode(),
            "transcript.jsonl": transcript,
        },
    )
    actions = world.drive(worker)
    assert actions[0].startswith(f"collected:{run_id}:"), actions
    return run_id


def _transcript(lines: list[dict[str, Any]]) -> bytes:
    return "".join(json.dumps(line) + "\n" for line in lines).encode()


# ----------------------------------------------------------------------------- the record


def test_the_record_is_the_list_row_and_the_run_document(world: World, client: TestClient) -> None:
    run_id = _finished(world, _transcript(LINES))
    answer = client.get(f"/api/v1/wsp/runs/{run_id}/record")
    assert answer.status_code == 200, answer.text
    record = answer.json()
    assert set(record) == {"run_id", "row", "document"}
    assert record["run_id"] == run_id
    (listed,) = [r for r in client.get("/api/v1/wsp/runs").json()["runs"] if r["run_id"] == run_id]
    assert record["row"] == listed
    assert record["document"] == client.get(f"/api/v1/wsp/runs/{run_id}").json()
    schemas.validate_run(record["document"])
    text = json.dumps(record)
    assert "price_" not in text and "key_fingerprint" not in text and "api_key" not in text


def test_the_record_of_an_unknown_run_is_refused_by_name(client: TestClient) -> None:
    answer = client.get("/api/v1/wsp/runs/wsp_000000000000/record")
    assert answer.status_code == 404, answer.text
    assert answer.json()["error"]["code"] == "wsp_run_not_found"


# ----------------------------------------------------------------------------- the transcript


def test_without_page_bounds_the_whole_transcript_is_served(
    world: World, client: TestClient
) -> None:
    run_id = _finished(world, _transcript(LINES))
    body = client.get(f"/api/v1/wsp/runs/{run_id}/transcript").json()
    assert body["entries"] == LINES
    assert (body["offset"], body["total"], body["next_offset"], body["partial"]) == (
        0,
        5,
        None,
        False,
    )


def test_limit_and_offset_walk_the_transcript_in_pages(world: World, client: TestClient) -> None:
    run_id = _finished(world, _transcript(LINES))
    seen: list[dict[str, Any]] = []
    offset: int | None = 0
    pages = 0
    while offset is not None:
        body = client.get(
            f"/api/v1/wsp/runs/{run_id}/transcript", params={"offset": offset, "limit": 2}
        ).json()
        assert len(body["entries"]) <= 2 and body["total"] == 5 and body["offset"] == offset
        seen.extend(body["entries"])
        offset = body["next_offset"]
        pages += 1
    assert seen == LINES and pages == 3


def test_max_bytes_bounds_a_page_but_never_empties_it(world: World, client: TestClient) -> None:
    run_id = _finished(world, _transcript(LINES))
    sizes = [len(json.dumps(line, separators=(",", ":")).encode()) for line in LINES]
    budget = sizes[0] + sizes[1]  # exactly the first two entries
    body = client.get(
        f"/api/v1/wsp/runs/{run_id}/transcript", params={"max_bytes": max(budget, 1024)}
    ).json()
    taken = len(body["entries"])
    assert sum(sizes[:taken]) <= max(budget, 1024) < sum(sizes[: taken + 1])
    assert body["next_offset"] == taken
    # A page whose first entry alone exceeds the budget still carries it.
    big = [{"turn": 0, "content": "y" * 5000}, {"turn": 1}]
    run_big = _finished(world, _transcript(big))
    body = client.get(f"/api/v1/wsp/runs/{run_big}/transcript", params={"max_bytes": 1024}).json()
    assert body["entries"] == big[:1] and body["next_offset"] == 1


@pytest.mark.parametrize(
    "params", [{"limit": 0}, {"limit": 201}, {"max_bytes": 1023}, {"offset": -1}]
)
def test_an_unbounded_or_negative_page_is_refused(
    world: World, client: TestClient, params: dict[str, int]
) -> None:
    run_id = _finished(world, _transcript(LINES))
    answer = client.get(f"/api/v1/wsp/runs/{run_id}/transcript", params=params)
    assert answer.status_code == 422, answer.text


# ----------------------------------------------------------------------------- the ledger


def _ledger_rows(world: World, run_id: str, endpoint: str, n: int, base: int) -> None:
    with world.engine.begin() as conn:
        for i in range(n):
            at = NOW + dt.timedelta(seconds=base + i)
            insert(
                conn,
                "mf_wsp_reservations",
                run_id=run_id,
                endpoint_id=endpoint,
                external=False,
                worst_case_usd=0,
                worst_case_tokens=4096,
                created_at=at,
                settled_at=at + dt.timedelta(seconds=1),
                settled_cost_usd=0,
            )
            insert(
                conn,
                "mf_wsp_usage",
                run_id=run_id,
                seq=i,
                endpoint_id=endpoint,
                input_tokens=1000 + i,
                output_tokens=20 + i,
                cost_usd="0.000100",
                at=at,
                outcome="served",
                http_status=200,
                latency_ms=500,
            )


def test_the_ledger_serves_each_row_with_its_endpoint_and_totals(
    world: World, client: TestClient
) -> None:
    mine = _finished(world, _transcript(LINES))
    other = _finished(world, _transcript(LINES))
    _ledger_rows(world, mine, "ep-local", 3, 0)
    _ledger_rows(world, other, "ep-other", 2, 100)
    answer = client.get(f"/api/v1/wsp/runs/{mine}/ledger")
    assert answer.status_code == 200, answer.text
    body = answer.json()
    assert body["run_id"] == mine and body["status"] == "completed"
    assert [u["seq"] for u in body["usage"]] == [0, 1, 2]
    assert {u["endpoint_id"] for u in body["usage"]} == {"ep-local"}
    assert [(u["input_tokens"], u["output_tokens"]) for u in body["usage"]] == [
        (1000, 20),
        (1001, 21),
        (1002, 22),
    ]
    assert body["usage"][0]["cost_usd"] == pytest.approx(0.0001)
    assert body["usage"][0]["at"] and body["usage"][0]["outcome"] == "served"
    assert len(body["reservations"]) == 3
    assert all(r["endpoint_id"] == "ep-local" and r["settled_at"] for r in body["reservations"])
    assert body["reservations"][0]["worst_case_tokens"] == 4096
    assert body["totals"] == {
        "reservations": 3,
        "reservations_open": 0,
        "settled_cost_usd": 0.0,
        "usage_rows": 3,
        "input_tokens": 3003,
        "output_tokens": 63,
        "cost_usd": pytest.approx(0.0003),
        "endpoint_ids": ["ep-local"],
    }
    assert (body["more_usage"], body["more_reservations"]) == (False, False)
    text = json.dumps(body)
    assert other not in text and "ep-other" not in text
    assert "price_" not in text and "prompt" not in text


def test_the_ledger_pages_on_its_own_cursors(world: World, client: TestClient) -> None:
    run_id = _finished(world, _transcript(LINES))
    _ledger_rows(world, run_id, "ep-local", 5, 0)
    first = client.get(f"/api/v1/wsp/runs/{run_id}/ledger", params={"limit": 2}).json()
    assert [u["seq"] for u in first["usage"]] == [0, 1] and first["more_usage"] is True
    assert len(first["reservations"]) == 2 and first["more_reservations"] is True
    last_id = first["reservations"][-1]["id"]
    rest = client.get(
        f"/api/v1/wsp/runs/{run_id}/ledger",
        params={"after_seq": 1, "after_reservation_id": last_id, "limit": 200},
    ).json()
    assert [u["seq"] for u in rest["usage"]] == [2, 3, 4] and rest["more_usage"] is False
    assert len(rest["reservations"]) == 3 and all(r["id"] > last_id for r in rest["reservations"])
    assert rest["totals"]["usage_rows"] == 5  # totals count every row, not the page


def test_the_ledger_of_a_run_with_no_calls_is_empty_not_missing(
    world: World, client: TestClient
) -> None:
    run_id = _finished(world, _transcript(LINES))
    body = client.get(f"/api/v1/wsp/runs/{run_id}/ledger").json()
    assert body["usage"] == [] and body["reservations"] == []
    assert body["totals"]["usage_rows"] == 0 and body["totals"]["endpoint_ids"] == []


# ----------------------------------------------------------------------------- lineage (fix 6)


def _no_candidate(world: World, kind: str, **body: Any) -> str:
    """One run of ``kind`` through the real driver and intake that leaves no patch, so no
    candidate: its document's lineage can only come from the run document's own fallback."""
    world.body = {**world.body, **body}
    run_id = world.queue(kind=kind)
    worker = FakeWorker()
    world.drive(worker)
    worker.state = "finished"
    worker.data = archive(
        worker.boot,
        run_id,
        **{"result.json": json.dumps({"stop_reason": "submitted"}).encode()},
    )
    world.drive(worker)
    assert world.run(run_id)["outcome"] == "no_change", world.run(run_id)
    return run_id


def _manifest(world: World) -> dict[str, Any]:
    from wsp.models import WspBundle

    with world.factory() as session:
        (bundle,) = session.query(WspBundle).all()
        return dict(bundle.manifest)


def test_a_zero_start_run_with_no_candidate_names_the_zeroed_start(
    world: World, client: TestClient, tmp_path: Path
) -> None:
    """2026-11-28 (fix 6, run wsp_3417a18bc85d, phase-2 E7): a ``create`` session without
    ``keep_weights`` edits zeroed tensors, so its lineage parent is ``bundle_zeroed``, named by the
    zeroed tensors' digest, as its candidate would have recorded it (``lineage.lineage_parent``,
    which reads the heads file). E7 left no candidate and read ``bundle``, the request's parent."""
    from wsp.lineage import zeroed_sha256
    from wsp.weight_ops.st_io import read_safetensors

    run_id = _no_candidate(world, "create")
    document = client.get(f"/api/v1/wsp/runs/{run_id}").json()
    manifest = _manifest(world)
    heads = read_safetensors(tmp_path / "bundle" / "heads.safetensors")
    assert document["candidate_id"] is None and document["run_sha256"] is None
    assert document["lineage"]["parent"] == {
        "kind": "bundle_zeroed",
        "ref": zeroed_sha256(manifest, heads),
    }
    assert document["lineage"]["parent"]["ref"] != manifest["bundle_sha256"]
    schemas.validate_run(document)
    record = client.get(f"/api/v1/wsp/runs/{run_id}/record").json()
    assert record["document"]["lineage"]["parent"]["kind"] == "bundle_zeroed"


@pytest.mark.parametrize(
    ("kind", "body"),
    [("repair", {}), ("create", {"keep_weights": True})],
)
def test_a_run_that_starts_from_the_bundle_names_the_bundle(
    world: World, client: TestClient, kind: str, body: dict[str, Any]
) -> None:
    """The fallback changes only a zero start: a repair, and a create that keeps the weights,
    still name the bundle."""
    run_id = _no_candidate(world, kind, **body)
    document = client.get(f"/api/v1/wsp/runs/{run_id}").json()
    assert document["lineage"]["parent"] == {
        "kind": "bundle",
        "ref": _manifest(world)["bundle_sha256"],
    }


# ----------------------------------------------------------------------------- the ceiling (fix 7)


@pytest.mark.parametrize("lowered", [True, False])
def test_a_ceiling_a_cut_off_reply_lowered_reaches_the_run_row(
    world: World, client: TestClient, lowered: bool
) -> None:
    """2026-11-28 (fix 7, run wsp_3417a18bc85d): the engine block's ``output_ceiling_lowered_to``
    through the real intake to the run's row: its ``output_ceiling_tokens`` and
    ``output_ceiling_lowered``, as the list row and the record serve them. Absent, the push's
    figures stand."""
    run_id, worker = _running(world)
    pushed = world.run(run_id)
    engine = {
        "action_protocol": pushed["action_protocol"] or "tool_calls",
        "handoffs": 0,
        "refused_turns": 1,
        "kernel_restarts": 0,
        **({"output_ceiling_lowered_to": 4_096} if lowered else {}),
    }
    worker.state = "finished"
    worker.data = archive(
        worker.boot,
        run_id,
        **{"result.json": json.dumps({"stop_reason": "submitted", "engine": engine}).encode()},
    )
    world.drive(worker)
    row = client.get(f"/api/v1/wsp/runs/{run_id}/record").json()["row"]
    if lowered:
        assert (row["output_ceiling_tokens"], row["output_ceiling_lowered"]) == (4_096, True)
    else:
        assert (row["output_ceiling_tokens"], row["output_ceiling_lowered"]) == (
            pushed["output_ceiling_tokens"],
            pushed["output_ceiling_lowered"],
        )
        assert row["output_ceiling_tokens"] is not None
