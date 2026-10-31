"""The three read-only listings the pickers read  (F9 task 14.0; AC-009.40, .41, .42).

Every request goes through the **mounted** app (``create_app()``) and its transport, never
an endpoint function imported directly - so deleting an ``include_router`` line in
``api/v1/router.py`` turns this file red. That is the removal control recorded in the 009
FTASKS, and it is why each assertion names the route rather than a function.

The database is the real schema in memory and rows go in through the production table; the
model roots are real directories on disk; ``nvidia-smi`` is replaced **at the process
boundary** (``core.gpu``'s ``_run``), so the parser that reads its output is the production
one.
"""

from __future__ import annotations

import json
import os
from collections.abc import Iterator
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import pytest
import sqlalchemy as sa
from fastapi.testclient import TestClient

from api.app import create_app
from api.deps import session_dependency
from core import gpu
from core.config import reset_settings
from tests.support.jobs import insert_run, memory_engine, session_for
from workers.run_record import RUNS

T0 = datetime(2026, 10, 1, 12, 0, tzinfo=UTC)


@pytest.fixture
def rig(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Iterator[tuple[TestClient, Any]]:
    monkeypatch.setenv("CHECKPOINT_ROOT", str(tmp_path / "checkpoints"))
    monkeypatch.setenv("PARTS_ROOT", str(tmp_path / "parts"))
    monkeypatch.setenv("TRAINER_STATE_ROOT", str(tmp_path / "trainer-state"))
    reset_settings()
    engine = memory_engine()
    app = create_app()

    def one_session():  # type: ignore[no-untyped-def]
        with session_for(engine) as session:
            yield session

    app.dependency_overrides[session_dependency] = one_session
    try:
        yield TestClient(app, raise_server_exceptions=False), engine
    finally:
        app.dependency_overrides.clear()
        reset_settings()


def seed_run(engine: Any, run_id: str, minutes: int, **values: Any) -> None:
    """A run row through the production table, created ``minutes`` after :data:`T0`."""
    extra = {key: values.pop(key) for key in ("stop_reason",) if key in values}
    with session_for(engine) as session:
        insert_run(session, run_id, **values)
        session.execute(
            sa.update(RUNS)
            .where(RUNS.c.id == run_id)
            .values(created_at=T0 + timedelta(minutes=minutes), **extra)
        )


# ======================================================================================
# GET /api/v1/runs
# ======================================================================================


class TestTheRunList:
    def test_it_lists_every_run_newest_first_with_the_row_a_picker_needs(self, rig) -> None:
        client, engine = rig
        seed_run(engine, "run-old", 1, status="completed", last_step=40, stop_reason="step_count")
        seed_run(engine, "run-new", 30, status="running", last_step=3)
        seed_run(engine, "run-mid", 10, status="failed", failure_code="tokenizer_unavailable")

        response = client.get("/api/v1/runs")

        assert response.status_code == 200, response.text
        body = response.json()
        assert [row["run_id"] for row in body["runs"]] == ["run-new", "run-mid", "run-old"]
        assert body["total"] == 3
        old = body["runs"][2]
        assert old == {
            "run_id": "run-old",
            "status": "completed",
            "run_type": "change",
            "created_at": (T0 + timedelta(minutes=1)).isoformat(),
            "finished_at": old["finished_at"],
            "last_step": 40,
            "stop_reason": "step_count",
            "failure_code": None,
            "archived_at": None,
        }
        assert old["finished_at"] is not None
        assert body["runs"][1]["failure_code"] == "tokenizer_unavailable"
        # The filter vocabularies travel with the answer, so a client lists rather than types.
        assert "running" in body["statuses"]
        assert set(body["run_types"]) == {"change", "create", "adapter"}

    def test_it_pages_without_repeating_or_dropping_a_row(self, rig) -> None:
        client, engine = rig
        for index in range(5):
            # Two runs per timestamp: the id is the tie-break that keeps pages disjoint.
            seed_run(engine, f"run-{index}", index // 2)

        first = client.get("/api/v1/runs", params={"limit": 2}).json()
        second = client.get("/api/v1/runs", params={"limit": 2, "offset": 2}).json()
        third = client.get("/api/v1/runs", params={"limit": 2, "offset": 4}).json()

        ids = [row["run_id"] for page in (first, second, third) for row in page["runs"]]
        assert ids == ["run-4", "run-3", "run-2", "run-1", "run-0"]
        assert first["total"] == second["total"] == third["total"] == 5
        assert (first["limit"], second["offset"]) == (2, 2)

    def test_it_filters_by_status_and_run_type(self, rig) -> None:
        client, engine = rig
        seed_run(engine, "a", 1, status="running", run_type="change")
        seed_run(engine, "b", 2, status="completed", run_type="adapter")
        seed_run(engine, "c", 3, status="running", run_type="adapter")

        running = client.get("/api/v1/runs", params={"status": "running"}).json()
        adapter_running = client.get(
            "/api/v1/runs", params={"status": "running", "run_type": "adapter"}
        ).json()

        assert [row["run_id"] for row in running["runs"]] == ["c", "a"]
        assert running["total"] == 2
        assert [row["run_id"] for row in adapter_running["runs"]] == ["c"]

    @pytest.mark.parametrize(
        ("params", "code"),
        [({"status": "done"}, "run_status_unknown"), ({"run_type": "full"}, "run_type_unknown")],
    )
    def test_an_unknown_filter_is_refused_by_name_not_answered_empty(
        self, rig, params: dict[str, str], code: str
    ) -> None:
        client, _ = rig
        response = client.get("/api/v1/runs", params=params)
        assert 400 <= response.status_code < 500
        error = response.json()["error"]
        assert error["code"] == code
        # The status refusal says `given`; F8's registry says `requested`. Either names it.
        named = error["details"].get("given", error["details"].get("requested"))
        assert named in params.values()

    def test_an_empty_database_is_an_empty_list_not_a_405(self, rig) -> None:
        client, _ = rig
        response = client.get("/api/v1/runs")
        assert response.status_code == 200
        assert response.json()["runs"] == []
        assert response.json()["total"] == 0

    def test_the_listing_does_not_shadow_the_run_view(self, rig) -> None:
        client, engine = rig
        seed_run(engine, "run-7", 1, status="running")
        assert client.get("/api/v1/runs/run-7").json()["run_id"] == "run-7"


# ======================================================================================
# GET /api/v1/models/local
# ======================================================================================


def model_dir(path: Path, *, weights: bool = True, tokenizer: bool = True, **extra: Any) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    (path / "config.json").write_text(json.dumps({"model_type": "llama", **extra}))
    if weights:
        (path / "model.safetensors").write_bytes(b"\0")
    if tokenizer:
        (path / "tokenizer.json").write_text("{}")
    return path


def touch(path: Path, minutes: int) -> None:
    stamp = (T0 + timedelta(minutes=minutes)).timestamp()
    os.utime(path / "config.json", (stamp, stamp))


class TestTheLocalModelList:
    def test_it_lists_the_model_directories_under_each_root(self, rig, tmp_path: Path) -> None:
        client, _ = rig
        checkpoints = tmp_path / "checkpoints"
        base = model_dir(checkpoints / "bases" / "tiny", architectures=["LlamaForCausalLM"])
        saved = model_dir(checkpoints / "run-a" / "step-40")
        (saved / "manifest.json").write_text("{}")
        bare = model_dir(checkpoints / "configs" / "arch", weights=False, tokenizer=False)
        touch(base, 3)
        touch(saved, 2)
        touch(bare, 1)
        (tmp_path / "parts").mkdir(exist_ok=True)

        body = client.get("/api/v1/models/local").json()

        assert [row["path"] for row in body["models"]] == [
            str(base.resolve()),
            str(saved.resolve()),
            str(bare.resolve()),
        ]
        first = body["models"][0]
        assert first == {
            "path": str(base.resolve()),
            "root": str(checkpoints.resolve()),
            "relative_path": "bases/tiny",
            "kind": "directory",
            "architecture": "LlamaForCausalLM",
            "has_weights": True,
            "has_tokenizer": True,
            "config_path": str(base.resolve() / "config.json"),
            "modified_at": (T0 + timedelta(minutes=3)).isoformat(),
        }
        assert body["models"][1]["kind"] == "checkpoint"
        assert body["models"][2]["has_weights"] is False
        assert body["models"][2]["architecture"] == "llama"
        assert body["total"] == 3
        assert body["roots"] == [str(checkpoints), str(tmp_path / "parts")]
        assert body["unreadable_roots"] == []
        assert body["max_depth"] == 4

    def test_has_weights_narrows_to_what_a_load_can_read(self, rig, tmp_path: Path) -> None:
        client, _ = rig
        model_dir(tmp_path / "checkpoints" / "loadable")
        model_dir(tmp_path / "checkpoints" / "config-only", weights=False)
        (tmp_path / "parts").mkdir(exist_ok=True)

        body = client.get("/api/v1/models/local", params={"has_weights": "true"}).json()
        assert [row["relative_path"] for row in body["models"]] == ["loadable"]
        assert body["total"] == 1

    def test_a_symlink_out_of_a_root_is_not_offered(self, rig, tmp_path: Path) -> None:
        """Confinement is checked on the RESOLVED path at preflight, so a linked directory
        leading out of the root is one the load would refuse - and must not be a choice."""
        client, _ = rig
        outside = model_dir(tmp_path / "elsewhere" / "secret-model")
        root = tmp_path / "checkpoints"
        root.mkdir(exist_ok=True)
        (root / "linked").symlink_to(outside, target_is_directory=True)
        (tmp_path / "parts").mkdir(exist_ok=True)

        body = client.get("/api/v1/models/local").json()
        assert body["models"] == []
        assert "secret-model" not in json.dumps(body)

    def test_the_walk_is_bounded_and_skips_hidden_directories(self, rig, tmp_path: Path) -> None:
        client, _ = rig
        root = tmp_path / "checkpoints"
        model_dir(root / "a" / "b" / "c" / "d" / "too-deep")
        model_dir(root / ".cache" / "hidden")
        model_dir(root / "a" / "b" / "c" / "deep-enough")
        (tmp_path / "parts").mkdir(exist_ok=True)

        body = client.get("/api/v1/models/local").json()
        assert [row["relative_path"] for row in body["models"]] == ["a/b/c/deep-enough"]

    def test_a_missing_root_is_named_not_silently_skipped(self, rig, tmp_path: Path) -> None:
        client, _ = rig
        model_dir(tmp_path / "checkpoints" / "m")

        body = client.get("/api/v1/models/local").json()
        assert body["unreadable_roots"] == [
            {
                "root": str(tmp_path / "parts"),
                "code": "model_root_missing",
                "reason": "not a directory on this host",
            }
        ]
        assert body["total"] == 1

    def test_an_unreadable_root_is_named_without_the_os_error_text(
        self, rig, tmp_path: Path, monkeypatch, caplog
    ) -> None:
        """CodeQL py/stack-trace-exposure (#28): the OSError's text - here a planted secret path -
        goes to the server log, never into the response; the client gets a stable code."""
        import logging

        from api.v1.endpoints import models_local

        client, _ = rig
        (tmp_path / "checkpoints").mkdir(exist_ok=True)
        (tmp_path / "parts").mkdir(exist_ok=True)
        planted = "/srv/private/host-detail-7f3a"

        def explode(root: Path) -> list[dict[str, object]]:
            raise PermissionError(13, "Permission denied", planted)

        monkeypatch.setattr(models_local, "scan_root", explode)
        with caplog.at_level(logging.WARNING, logger="miforge.api.models_local"):
            response = client.get("/api/v1/models/local")
        assert response.status_code == 200
        assert planted not in response.text
        assert "Permission denied" not in response.text
        rows = response.json()["unreadable_roots"]
        assert [(row["root"], row["code"]) for row in rows] == [
            (str(tmp_path / "checkpoints"), "model_root_unreadable"),
            (str(tmp_path / "parts"), "model_root_unreadable"),
        ]
        # ...and it is not lost: the server log carries it, once per root.
        logged = [r for r in caplog.records if getattr(r, "code", None) == "model_root_unreadable"]
        assert len(logged) == 2
        assert all(r.exc_info and planted in str(r.exc_info[1]) for r in logged)

    def test_it_pages(self, rig, tmp_path: Path) -> None:
        client, _ = rig
        for index in range(3):
            touch(model_dir(tmp_path / "checkpoints" / f"m{index}"), index)
        (tmp_path / "parts").mkdir(exist_ok=True)
        page = client.get("/api/v1/models/local", params={"limit": 1, "offset": 1}).json()
        assert [row["relative_path"] for row in page["models"]] == ["m1"]
        assert page["total"] == 3


# ======================================================================================
# GET /api/v1/gpus
# ======================================================================================

NVIDIA_L = (
    "GPU 0: NVIDIA GeForce RTX 3090 (UUID: GPU-247aa582-0d1b-e161-8156-983ed1fefc57)\n"
    "GPU 1: NVIDIA GeForce RTX 3080 Ti (UUID: GPU-f47ba814-49a2-603f-3595-275284140251)\n"
)
NVIDIA_FREE = (
    "GPU-247aa582-0d1b-e161-8156-983ed1fefc57, 20000\n"
    "GPU-f47ba814-49a2-603f-3595-275284140251, 9000\n"
)


def _keys(value: Any) -> set[str]:
    if isinstance(value, dict):
        return set(value) | {key for item in value.values() for key in _keys(item)}
    if isinstance(value, list):
        return {key for item in value for key in _keys(item)}
    return set()


class TestTheGpuList:
    def test_it_lists_each_card_by_uuid_with_its_free_memory(self, rig, monkeypatch) -> None:
        client, _ = rig
        monkeypatch.setattr(gpu, "_run", lambda args: NVIDIA_L if args[1] == "-L" else NVIDIA_FREE)

        body = client.get("/api/v1/gpus").json()

        assert body["available"] is True
        assert body["unavailable"] is None
        assert body["devices"] == [
            {
                "uuid": "GPU-247aa582-0d1b-e161-8156-983ed1fefc57",
                "name": "NVIDIA GeForce RTX 3090",
                "free_memory_bytes": 20000 * 1024 * 1024,
            },
            {
                "uuid": "GPU-f47ba814-49a2-603f-3595-275284140251",
                "name": "NVIDIA GeForce RTX 3080 Ti",
                "free_memory_bytes": 9000 * 1024 * 1024,
            },
        ]

    def test_no_index_appears_anywhere_in_the_body(self, rig, monkeypatch) -> None:
        """Indices inverted on this node on 2026-09-18. A response is where one is copied
        from, so none is in it - by key, and by the positional values it would carry."""
        client, _ = rig
        monkeypatch.setattr(gpu, "_run", lambda args: NVIDIA_L if args[1] == "-L" else NVIDIA_FREE)
        body = client.get("/api/v1/gpus").json()
        assert not {"index", "bus_id", "pci_bus_id", "minor_number"} & _keys(body)
        for device in body["devices"]:
            assert set(device) == {"uuid", "name", "free_memory_bytes"}

    def test_no_nvidia_smi_is_a_named_answer_not_an_error(self, rig, monkeypatch) -> None:
        client, _ = rig
        monkeypatch.setattr(gpu.shutil, "which", lambda name: None)

        response = client.get("/api/v1/gpus")

        assert response.status_code == 200
        body = response.json()
        assert body["available"] is False
        assert body["devices"] == []
        assert body["unavailable"]["code"] == "nvidia_smi_unavailable"
        assert body["auto"] is True

    def test_an_unreadable_free_figure_is_unknown_not_zero(self, rig, monkeypatch) -> None:
        client, _ = rig

        def run(args: list[str]) -> str:
            if args[1] == "-L":
                return NVIDIA_L
            raise gpu.refuse("nvidia_smi_failed", "query failed")

        monkeypatch.setattr(gpu, "_run", run)
        body = client.get("/api/v1/gpus").json()
        assert [device["free_memory_bytes"] for device in body["devices"]] == [None, None]
