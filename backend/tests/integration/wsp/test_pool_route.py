"""``GET /api/v1/wsp/pool`` (feature 019 P3; D-19, FR-019.18; TDD §5.3), through the production app.

The route reads only the database: the worker rows a real driver pass wrote
(``pool.record_ready``) and the runs. Busy is a run holding the worker, not answering is a
worker whose last read failed (never idle), and ``shared_base_urls`` groups active runs on one
server.
"""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

import pytest
import sqlalchemy as sa
from fastapi.testclient import TestClient

from tests.integration.wsp.test_dispatch import FakeWorker, World, build_world
from tests.integration.wsp.test_run_cause import _client
from tests.integration.wsp.wsp_db import NOW, insert, run_row
from wsp.settings import get_wsp_settings


@pytest.fixture()
def world(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Iterator[World]:
    monkeypatch.setenv("WSP_WORKER_URL", "http://wsp-worker-{index}.test:9000")
    monkeypatch.setenv("WSP_WORKER_POOL_MAX", "2")
    built = build_world(tmp_path, monkeypatch)
    yield built
    built.engine.dispose()
    get_wsp_settings.cache_clear()


@pytest.fixture()
def client(world: World, monkeypatch: pytest.MonkeyPatch) -> TestClient:
    return _client(world, monkeypatch)


def test_one_busy_one_idle_and_the_queue_per_endpoint(world: World, client: TestClient) -> None:
    held = world.queue("ep-local")
    waiting_local = world.queue("ep-local")
    waiting_ext = world.queue("ep-ext")
    busy, idle = FakeWorker(), FakeWorker()
    actions = world.drive([busy, idle])
    assert actions == [f"pushed:{held}"], actions
    answer = client.get("/api/v1/wsp/pool")
    assert answer.status_code == 200, answer.text
    body = answer.json()
    assert {k: body[k] for k in ("pool_size", "maximum", "effective")} == {
        "pool_size": 1,
        "maximum": 2,
        "effective": 1,
    }
    assert (body["busy"], body["idle"], body["not_answering"]) == (1, 1, 0)
    assert [(w["index"], w["state"], w["run_id"]) for w in body["workers"]] == [
        (0, "busy", held),
        (1, "idle", None),
    ]
    assert body["workers"][0]["boot_id"] == busy.boot and body["workers"][0]["read_at"]
    assert body["queued_by_endpoint"] == [
        {"endpoint_id": "ep-ext", "queued": 1},
        {"endpoint_id": "ep-local", "queued": 1},
    ]
    assert body["queued_without_endpoint"] == 0
    assert body["shared_base_urls"] == []
    assert world.run(waiting_local)["status"] == world.run(waiting_ext)["status"] == "queued"


def test_a_worker_that_does_not_answer_is_not_answering_never_idle(
    world: World, client: TestClient
) -> None:
    silent = FakeWorker(unreachable=True)
    world.drive([FakeWorker(), silent])
    body = client.get("/api/v1/wsp/pool").json()
    assert [(w["index"], w["state"]) for w in body["workers"]] == [
        (0, "idle"),
        (1, "not_answering"),
    ]
    assert (body["busy"], body["idle"], body["not_answering"]) == (0, 1, 1)


def _generic(world: World, endpoint_id: str, base_url: str) -> None:
    with world.engine.begin() as conn:
        insert(
            conn,
            "mf_wsp_endpoints",
            id=endpoint_id,
            kind="openai_compatible",
            base_url=base_url,
            model="m",
            locality="local",
            output_cap_field="max_tokens",
            serving="generic",
            created_at=NOW,
            created_by="operator",
        )


def _active(world: World, n: int, endpoint_id: str) -> str:
    sha = world.body["bundle"]["bundle_sha256"]
    boot = f"{n:032x}"
    with world.engine.begin() as conn:
        insert(conn, "mf_wsp_worker_boots", boot_id=boot, first_seen_at=NOW, run_id=None)
        return run_row(
            conn,
            f"wsp_{n:012x}",
            bundle_sha256=sha,
            status="running",
            boot_id=boot,
            worker_index=n,
            endpoint_id=endpoint_id,
            schedule_key=f"endpoint:{endpoint_id}",
        )


def test_two_generic_rows_on_one_server_are_named_as_sharing_it(
    world: World, client: TestClient
) -> None:
    _generic(world, "gen-a", "http://vllm.lan:8000/v1")
    _generic(world, "gen-b", "http://VLLM.lan:8000")
    first, second = _active(world, 0, "gen-a"), _active(world, 1, "gen-b")
    body = client.get("/api/v1/wsp/pool").json()
    assert body["shared_base_urls"] == [
        {
            "base_url": "http://vllm.lan:8000",
            "endpoint_ids": ["gen-a", "gen-b"],
            "run_ids": [first, second],
        }
    ]


def test_different_servers_are_not_shared(world: World, client: TestClient) -> None:
    _generic(world, "gen-a", "http://vllm-a.lan:8000/v1")
    _generic(world, "gen-b", "http://vllm-b.lan:8000/v1")
    _active(world, 0, "gen-a")
    _active(world, 1, "gen-b")
    assert client.get("/api/v1/wsp/pool").json()["shared_base_urls"] == []


def test_a_worker_row_above_a_lowered_maximum_is_forgotten_at_the_next_pass(
    world: World, client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    """18.3: lowering the maximum leaves no ghost worker in the pool."""
    world.drive([FakeWorker(), FakeWorker()])
    assert [w["index"] for w in client.get("/api/v1/wsp/pool").json()["workers"]] == [0, 1]
    monkeypatch.setenv("WSP_WORKER_POOL_MAX", "1")
    get_wsp_settings.cache_clear()
    world.drive([FakeWorker()])
    with world.engine.connect() as conn:
        rows = conn.execute(sa.text("SELECT worker_index FROM mf_wsp_workers")).scalars().all()
    assert rows == [0]
    body = client.get("/api/v1/wsp/pool").json()
    assert body["maximum"] == 1 and [w["index"] for w in body["workers"]] == [0]


class _HostileReady(FakeWorker):
    """A worker is untrusted (P49): its ``/ready`` may say anything."""

    def ready(self) -> dict[str, object]:
        self.calls.append(("ready",))
        return {"boot_id": "c" * 40 + "<script>", "state": "<img src=x onerror=alert(1)>"}


def test_a_hostile_ready_is_stored_bounded_and_as_unknown(world: World, client: TestClient) -> None:
    world.drive([_HostileReady()])
    (worker,) = client.get("/api/v1/wsp/pool").json()["workers"]
    assert worker["worker_state"] == "unknown"
    assert worker["boot_id"] == "c" * 32
    assert worker["state"] == "idle"  # it answered; nothing holds it


def test_the_route_serves_the_pool_size_the_settings_hold_now(
    world: World, client: TestClient
) -> None:
    """The route reads ``wsp_pool_size`` from the runtime settings at each call: an operator's
    write through the live settings route shows at once (control P7: a constant pool size)."""
    from core import runtime_settings

    try:
        assert client.get("/api/v1/wsp/pool").json()["pool_size"] == 1
        put = client.put("/api/v1/settings/wsp_pool_size", json={"value": 2})
        assert put.status_code == 200, put.text
        body = client.get("/api/v1/wsp/pool").json()
        assert (body["pool_size"], body["maximum"], body["effective"]) == (2, 2, 2)
    finally:
        runtime_settings.reset()  # the override cache is process-wide
