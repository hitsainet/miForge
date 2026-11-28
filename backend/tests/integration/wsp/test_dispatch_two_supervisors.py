"""D-3 (feature 019 P3; FTASKS 16.3; FR-019.7): two REAL supervisors, one run each.

Two ``LocalProcessLauncher`` supervisors stand for two StatefulSet pods. The real driver claims each
run on a worker whose ``/ready`` answered idle, the push task's body pushes it there, and every
later call for that run (status, result, ack) goes to that worker and no other, by boot id on both
sides. Each client is wrapped only to COUNT its calls; every byte goes to the real supervisor.

Skips by name without the worker interpreter or seccomp; fails under
``MIFORGE_REQUIRE_WSP_WORKER=1``.
"""

from __future__ import annotations

import time
from collections.abc import Iterator
from contextlib import ExitStack
from pathlib import Path
from typing import Any

import pytest

from tests.integration.wsp.test_dispatch import World, build_world
from tests.integration.wsp.test_runs_lifecycle import _store
from tests.integration.wsp.wsp_db import NOW, insert
from tests.support import wsp_worker_env
from tests.support.wsp_drive import drive_inline
from tests.support.wsp_scripted_endpoint import OPENAI, ScriptedEndpoint
from wsp import tasks
from wsp.settings import get_wsp_settings

#: Enough plain submits for both sessions, whichever asks first (the editor's replies are the only
#: scripted part; a submit with no edit ends a session without a candidate).
STEPS = [{"calls": [("submit", {"answer": "no change", "confidence": 0.5})]} for _ in range(8)]


class Counting:
    """The real ``HttpWorker``, with each call recorded."""

    def __init__(self, inner: Any) -> None:
        self.inner = inner
        self.calls: list[tuple[Any, ...]] = []
        self.ready_boots: list[str] = []

    def ready(self) -> dict[str, Any]:
        answer: dict[str, Any] = self.inner.ready()
        self.calls.append(("ready",))
        self.ready_boots.append(str(answer.get("boot_id")))
        return answer

    def push(self, document: dict[str, Any], files: Any, **kwargs: Any) -> dict[str, Any]:
        self.calls.append(("push", document["run_id"], document["boot_id"]))
        answer: dict[str, Any] = self.inner.push(document, files, **kwargs)
        return answer

    def status(self) -> dict[str, Any]:
        answer: dict[str, Any] = self.inner.status()
        self.calls.append(("status", answer.get("run_id"), answer.get("boot_id")))
        return answer

    def result(self) -> bytes:
        self.calls.append(("result",))
        data: bytes = self.inner.result()
        return data

    def ack(self, boot_id: str, run_id: str, result_sha256: str) -> None:
        self.calls.append(("ack", run_id, boot_id))
        self.inner.ack(boot_id, run_id, result_sha256)

    def recycle(self, boot_id: str) -> None:
        self.calls.append(("recycle", boot_id))
        self.inner.recycle(boot_id)

    def of(self, name: str) -> list[tuple[Any, ...]]:
        return [c for c in self.calls if c[0] == name]


@pytest.fixture()
def pool_world(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Iterator[World]:
    wsp_worker_env.require_seccomp()
    world = build_world(tmp_path, monkeypatch, bundle_id="wsb_0123456789ab")
    _store(world, tmp_path)
    with world.engine.begin() as conn:
        insert(
            conn,
            "mf_wsp_endpoints",
            id="gen-b",
            kind="openai_compatible",
            base_url="http://gen-b.lan:8000/v1",
            model="m",
            locality="local",
            context_window_tokens=32_768,  # feature 020: a ``.lan`` host derives external
            output_cap_field="max_completion_tokens",
            serving="generic",
            created_at=NOW,
            created_by="operator",
        )
    yield world
    world.engine.dispose()
    get_wsp_settings.cache_clear()


def test_each_run_is_pushed_polled_pulled_and_acked_on_its_own_worker(
    pool_world: World, monkeypatch: pytest.MonkeyPatch
) -> None:
    from wsp.launcher import LocalProcessLauncher
    from wsp.worker_client import HttpWorker

    world = pool_world
    python = wsp_worker_env.worker_python()
    with ScriptedEndpoint(OPENAI, STEPS) as editor, ExitStack() as stack:
        monkeypatch.setenv("WSP_GATEWAY_URL", f"http://127.0.0.1:{editor.gateway['port']}")
        get_wsp_settings.cache_clear()
        world.settings = get_wsp_settings()
        world.body = {**world.body, "budgets": {"tokens": 100_000, "time_s": 120}}
        first = world.queue("ep-local")
        second = world.queue("gen-b")
        locals_ = [
            stack.enter_context(LocalProcessLauncher(environment="ci", python=python).session())
            for _ in range(2)
        ]
        clients = [Counting(HttpWorker(local.url)) for local in locals_]
        for client in clients:
            stack.callback(client.inner.close)
        actions: list[str] = []
        deadline = time.monotonic() + 180
        while not (f"acked:{first}" in actions and f"acked:{second}" in actions):
            assert time.monotonic() < deadline, actions
            actions += drive_inline(
                world.factory,
                clients,
                world.settings,
                read_sensitivity=world.read,
                files_for=tasks.run_files,
                pool_size=2,
            )
            time.sleep(0.2)
        for local in locals_:
            assert local.process.wait(timeout=30) == 0
    runs_by_worker = {}
    for index, client in enumerate(clients):
        (push,) = client.of("push")
        run_id, boot = push[1], push[2]
        runs_by_worker[index] = run_id
        row = world.run(run_id)
        # By boot id on both sides: the run holds the boot this worker's /ready answered.
        assert row["boot_id"] == boot and boot in client.ready_boots
        assert int(row["worker_index"]) == index
        # Every status this worker answered was for its own run, on its own boot.
        statuses = client.of("status")
        assert statuses and {(s[1], s[2]) for s in statuses} == {(run_id, boot)}
        assert len(client.of("result")) == 1
        assert client.of("ack") == [("ack", run_id, boot)]
        assert row["status"] in ("completed", "failed"), row["failure"]
        assert row["result_sha256"] is not None
    assert sorted(runs_by_worker.values()) == sorted([first, second])
    assert runs_by_worker[0] == first  # the oldest queued run takes the lowest free worker
