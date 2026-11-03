"""A mounted miForge on a MIGRATED database, for the F12 approval tests.

Everything the gate and the approved request run is the production code: ``create_app``, the
gate decorator, the approval service, ``run_launch.launch``, the chains service, the routes'
own validators. What is stood in for, and why:

* the validator's four sibling- or card-bound steps (``_live_gate``, ``_live_parts``,
  ``_live_budget``, ``_live_services``) - they read packs, parts, free VRAM and sibling URLs,
  which other features' tests pin; the budget reports a card UUID so a lease WOULD be taken;
* F1's acquisition pre-flight and device resolution - they read a model directory and
  ``nvidia-smi``;
* the GPU lease's ``acquire`` - recorded, so "no lease was taken" is an observation, not an
  absence the fixture guarantees by construction;
* the two enqueues (``training.tasks.run_step_loop``, ``runtypes.enqueue_merge``) - recorded,
  so "nothing was enqueued" is counted.
"""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass, field
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
import sqlalchemy as sa
from fastapi.testclient import TestClient

AGENT = {"X-Miforge-Agent": "agent:forge-mcp"}
GPU = "GPU-00000000-0000-0000-0000-00000000f012"


class _PackDouble:
    class _Definition:
        domain = "legal"

    definition = _Definition()

    def payload(self) -> dict[str, Any]:
        return {"pack_id": "legal", "version": "1", "retention_floor": None, "parts": []}

    def part_records(self) -> list[Any]:
        return []


@dataclass
class World:
    engine: Any
    client: TestClient
    root: Path
    enqueued: list[str] = field(default_factory=list)
    merges: list[tuple[str, str, int]] = field(default_factory=list)
    leases: list[tuple[str, str]] = field(default_factory=list)

    def session(self) -> Any:
        from tests.support.db import session_for

        return session_for(self.engine)

    def count(self, table: str) -> int:
        with self.engine.connect() as conn:
            return int(conn.execute(sa.text(f"SELECT COUNT(*) FROM {table}")).scalar_one())

    def approvals(self) -> list[dict[str, Any]]:
        response = self.client.get("/api/v1/approvals")
        assert response.status_code == 200, response.text
        return list(response.json()["approvals"])


def definition(run_id: str | None = "run-f12", *, mode: str = "load") -> dict[str, Any]:
    """A run definition the stubbed validator accepts."""
    source: dict[str, Any] = {"mode": mode}
    if mode == "load":
        source["path"] = "/models/base"
    body: dict[str, Any] = {
        "model_source": source,
        "run_type": "change",
        "run_configuration": {
            "learning_rate": 1e-5,
            "warmup_steps": 0,
            "batch_size": 1,
            "gradient_accumulation": 1,
            "optimizer_state_precision": "32bit",
        },
        "signal_source": {"source": "corpus"},
        "parts": [{"part_id": "part-corpus", "kind": "corpus"}],
        "stop_conditions": [{"condition": "step_count", "config": {"total": 40}}],
        "domain": "legal",
        "seed": 7,
        "gpu_request": "auto",
    }
    if run_id is not None:
        body["run_id"] = run_id
    return body


def chain_spec(*, first_mode: str = "load") -> dict[str, Any]:
    first = definition(None, mode=first_mode)
    second = definition(None)
    return {
        "name": "f12-curriculum",
        "stages": [
            {"definition": first, "advance_on": ["step_count"]},
            {"definition": second, "advance_on": ["step_count"]},
        ],
    }


def submit_body(*, mode: str = "load") -> dict[str, Any]:
    body: dict[str, Any] = {
        "mode": mode,
        "learning_rate": 1e-5,
        "warmup_steps": 0,
        "batch_size": 1,
        "run_id": "run-f1-submit",
    }
    if mode == "load":
        body["path"] = "/models/base"
    return body


@pytest.fixture
def world(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Iterator[World]:
    from api.app import create_app
    from api.deps import session_dependency
    from api.v1.endpoints import runtypes as runtypes_endpoints
    from core import runtime_settings
    from core.config import reset_settings
    from packs.settings import reset_pack_settings
    from services import acquisition as acquisition_service
    from services import run_definition as validator
    from tests.support.db import migrated_engine, session_for
    from training import tasks
    from workers import lease as lease_module

    for name, sub in (
        ("CHECKPOINT_ROOT", "checkpoints"),
        ("TRAINER_STATE_ROOT", "state/trainer-state"),
        ("PARTS_ROOT", "parts"),
        ("PACKS_ROOT", "packs"),
    ):
        monkeypatch.setenv(name, str(tmp_path / sub))
    for name in ("MILLM_BASE_URL", "MIDATAWORKS_BASE_URL", "MILLM_TRAINER_TOKEN"):
        monkeypatch.delenv(name, raising=False)
    reset_settings()
    reset_pack_settings()
    runtime_settings.reset()

    engine = migrated_engine(tmp_path / "db")
    built = World(engine=engine, client=None, root=tmp_path)  # type: ignore[arg-type]

    monkeypatch.setattr(validator, "_live_gate", lambda *a, **k: _PackDouble())
    monkeypatch.setattr(validator, "_live_parts", lambda *a, **k: {"source": "parts", "parts": []})
    monkeypatch.setattr(validator, "_live_services", lambda *a, **k: {"needed": []})
    monkeypatch.setattr(
        validator,
        "_live_budget",
        lambda definition: {
            "mode": str(definition.model_source.get("mode") or "load"),
            "param_count": 10,
            "gpu_uuid": GPU,
            "gpu_name": "test card",
        },
    )
    monkeypatch.setattr(acquisition_service, "resolve_submit_device", lambda request: None)
    monkeypatch.setattr(
        acquisition_service,
        "preflight",
        lambda spec, cfg, **kw: (
            spec.model_copy(update={"param_count": 10}),
            SimpleNamespace(available=True, payload=lambda: {"available": True}),
        ),
    )

    def acquire(session: Any, run_id: str, gpu_request: str, *, device: Any) -> Any:
        built.leases.append((run_id, str(device.uuid)))
        return SimpleNamespace(gpu_uuid=str(device.uuid))

    monkeypatch.setattr(lease_module, "acquire", acquire)

    class _Task:
        @staticmethod
        def delay(run_id: str) -> None:
            built.enqueued.append(run_id)

    monkeypatch.setattr(tasks, "run_step_loop", _Task)
    monkeypatch.setattr(
        runtypes_endpoints,
        "enqueue_merge",
        lambda name, run_id, step: built.merges.append((name, run_id, step)),
    )

    app = create_app()

    def override() -> Iterator[Any]:
        with session_for(engine) as session:
            yield session

    app.dependency_overrides[session_dependency] = override
    built.client = TestClient(app, raise_server_exceptions=False)
    yield built
    engine.dispose()
    runtime_settings.reset()
    reset_settings()
    reset_pack_settings()
