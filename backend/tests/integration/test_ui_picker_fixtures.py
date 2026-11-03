"""The E2E pickers spec's backend, **generated from the real routes and pinned here** (task 14.8).

``frontend/e2e/pickers.spec.ts`` drives the New run form and the Runs screen against a
network fake - CI's frontend job runs no backend (ADR-016). A fake whose bodies were written by
hand would agree with the frontend's reader by construction, which is the defect family this
project keeps finding (the miLLM wire format, the fake weights surface). So the fake serves
**exactly these bodies**, and this test produces them by seeding a real database, real parts,
a real pack and a real model directory, and calling the mounted app's routes.

The committed file is compared with what the routes answer today. When a route's shape
changes this goes red, and the fixture is regenerated with
``MIFORGE_WRITE_UI_FIXTURES=1 pytest backend/tests/integration/test_ui_picker_fixtures.py`` -
which makes the change visible in the frontend's diff rather than silently absent from it.

Normalised, so the file is stable: the temporary root becomes ``/data``, and the two values a
seeded backend cannot hold still (an import time and the hash of the definition that carries
it) are fixed. Nothing else is touched.
"""

from __future__ import annotations

import json
import os
from datetime import UTC, datetime
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
import sqlalchemy as sa
from fastapi.testclient import TestClient

from api.app import create_app
from api.deps import session_dependency
from core import gpu
from core.config import reset_settings
from packs.settings import reset_pack_settings
from tests.support import pack_fixtures as fx
from tests.support.db import migrated_engine, session_for
from tests.support.jobs import insert_run
from workers.run_record import RUNS

REPO_ROOT = Path(__file__).resolve().parents[3]
FIXTURE = REPO_ROOT / "frontend" / "e2e" / "fixtures" / "backend.json"
#: The body the E2E spec asserts the New run form posts, choosing from the lists above.
DEFINE_BODY = REPO_ROOT / "frontend" / "e2e" / "fixtures" / "define-body.json"
WRITE = os.environ.get("MIFORGE_WRITE_UI_FIXTURES") == "1"

#: The routes the spec's fake serves, read from the mounted app. A GET with a query is keyed
#: with it, exactly as the browser sends it.
ROUTES: tuple[str, ...] = (
    "/api/v1/run-types",
    "/api/v1/packs",
    "/api/v1/selectors",
    "/api/v1/models/sources",
    "/api/v1/models/local?limit=500",
    # The create run's tokenizer picker: the tokenizer directories, and the Hub check a
    # "Check" press sends (the Hub client is a fake below; the route and its reader are real).
    "/api/v1/tokenizers/local?limit=500",
    "/api/v1/hub/revision?repo=org%2Ftiny-tok&revision=main",
    "/api/v1/signal-sources",
    "/api/v1/stop-conditions",
    "/api/v1/gpus",
    "/api/v1/parts?part_kind=prompt_set",
    "/api/v1/runs?limit=25&offset=0",
    "/api/v1/runs/run-seeded-1",
    "/api/v1/runs/run-seeded-1/scores",
    "/api/v1/runs/run-seeded-1/steps",
    "/api/v1/runs/run-seeded-1/checkpoints",
)

FIXED_TIME = "2026-10-17T12:00:00Z"
#: Keys whose values are the moment of seeding, not part of any shape.
VOLATILE = {"imported_at": FIXED_TIME, "taken_in_at": FIXED_TIME}


def _normalise(value: Any, root: str) -> Any:
    if isinstance(value, dict):
        return {
            key: (VOLATILE[key] if key in VOLATILE and item is not None else _normalise(item, root))
            for key, item in value.items()
        }
    if isinstance(value, list):
        return [_normalise(item, root) for item in value]
    if isinstance(value, str):
        return value.replace(root, "/data")
    return value


@pytest.fixture
def seeded(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> TestClient:
    monkeypatch.setenv("PACKS_ROOT", str(tmp_path / "packs"))
    monkeypatch.setenv("CHECKPOINT_ROOT", str(tmp_path / "checkpoints"))
    monkeypatch.setenv("PARTS_ROOT", str(tmp_path / "parts"))
    monkeypatch.setenv("TRAINER_STATE_ROOT", str(tmp_path / "trainer-state"))
    monkeypatch.delenv("MIDATAWORKS_BASE_URL", raising=False)
    monkeypatch.delenv("MILLM_BASE_URL", raising=False)
    # No card on a CI runner, and the developer's machine may have one: the fixture states
    # the no-card answer on both.
    monkeypatch.setattr(gpu.shutil, "which", lambda name: None)
    reset_pack_settings()
    reset_settings()

    engine = migrated_engine(tmp_path / "db")
    with session_for(engine) as session:
        for set_kind, reference in (
            ("prompt_set", ("prompt-set", "support-tickets", "3")),
            ("test_set", ("test-set", "notary-test", "1")),
            ("retention_set", ("retention-set", "general", "2")),
        ):
            fx.import_contract_part(
                session,
                parts_root=tmp_path / "parts",
                export_root=tmp_path / f"export-{set_kind}",
                set_kind=set_kind,
                reference=reference,
            )
        fx.commit(session, fx.definition(slots=fx.valid_slots()))
        insert_run(session, "run-seeded-1", status="completed", last_step=40)
        session.execute(
            sa.update(RUNS)
            .where(RUNS.c.id == "run-seeded-1")
            .values(
                created_at=datetime(2026, 10, 16, 9, 0, tzinfo=UTC),
                finished_at=datetime(2026, 10, 16, 10, 0, tzinfo=UTC),
                stop_reason="step_count",
            )
        )

    base = tmp_path / "checkpoints" / "bases" / "tiny"
    base.mkdir(parents=True)
    (base / "config.json").write_text(json.dumps({"architectures": ["LlamaForCausalLM"]}))
    (base / "model.safetensors").write_bytes(b"\0")
    (base / "tokenizer.json").write_text("{}")
    stamp = datetime(2026, 10, 15, tzinfo=UTC).timestamp()
    os.utime(base / "config.json", (stamp, stamp))
    os.utime(base / "tokenizer.json", (stamp, stamp))
    # A tokenizer-only directory: no config.json, so only the tokenizer list offers it.
    tokenizer = tmp_path / "checkpoints" / "tokenizers" / "tiny-tok"
    tokenizer.mkdir(parents=True)
    (tokenizer / "tokenizer.json").write_text("{}")
    (tokenizer / "tokenizer_config.json").write_text("{}")
    later = datetime(2026, 10, 16, tzinfo=UTC).timestamp()
    for name in ("tokenizer.json", "tokenizer_config.json"):
        os.utime(tokenizer / name, (later, later))

    from training.acquisition import hub as hub_module

    hub_fake = SimpleNamespace(
        repo_info=lambda repo_id, revision: SimpleNamespace(
            sha="0123456789abcdef" * 2 + "01234567"
        ),
        list_repo_files=lambda repo_id, revision: [
            "tokenizer.json",
            "tokenizer_config.json",
            "README.md",
        ],
    )
    monkeypatch.setattr(hub_module, "_hub", lambda: hub_fake)

    app = create_app()

    def override():  # type: ignore[no-untyped-def]
        with session_for(engine) as session:
            yield session

    app.dependency_overrides[session_dependency] = override
    client = TestClient(app, raise_server_exceptions=False)
    client.root = str(tmp_path)  # type: ignore[attr-defined]
    try:
        yield client
    finally:
        app.dependency_overrides.clear()
        engine.dispose()
        reset_pack_settings()
        reset_settings()


def _answers(client: TestClient) -> dict[str, Any]:
    root = client.root  # type: ignore[attr-defined]
    out: dict[str, Any] = {}
    for path in ROUTES:
        response = client.get(path)
        assert response.status_code == 200, f"{path}: {response.status_code} {response.text}"
        out[path] = _normalise(response.json(), root)
    return out


def test_the_e2e_fake_serves_exactly_what_the_routes_answer(seeded: TestClient) -> None:
    answers = _answers(seeded)
    # Not vacuous: the seeded things are in the answers the pickers read.
    assert [row["name"] for row in answers["/api/v1/packs"]["packs"]] == ["notary-md"]
    assert answers["/api/v1/packs"]["packs"][0]["validity"]["state"] == "valid"
    assert answers["/api/v1/models/local?limit=500"]["models"][0]["path"] == (
        "/data/checkpoints/bases/tiny"
    )
    assert [row["run_id"] for row in answers["/api/v1/runs?limit=25&offset=0"]["runs"]] == [
        "run-seeded-1"
    ]
    assert answers["/api/v1/gpus"]["unavailable"]["code"] == "nvidia_smi_unavailable"
    assert [
        row["relative_path"] for row in answers["/api/v1/tokenizers/local?limit=500"]["tokenizers"]
    ] == [
        "tokenizers/tiny-tok",
        "bases/tiny",
    ]

    document = json.dumps(answers, indent=2, sort_keys=True) + "\n"
    if WRITE:
        FIXTURE.parent.mkdir(parents=True, exist_ok=True)
        FIXTURE.write_text(document)
    assert (
        FIXTURE.is_file()
    ), f"{FIXTURE} is missing. Generate it: MIFORGE_WRITE_UI_FIXTURES=1 pytest {__file__}"
    committed = json.loads(FIXTURE.read_text())
    assert committed == answers, (
        "The E2E pickers fixture no longer matches what the routes answer. A route's shape "
        "changed: regenerate it with MIFORGE_WRITE_UI_FIXTURES=1 and review the frontend diff."
    )


def test_the_body_the_e2e_form_posts_is_one_the_backend_accepts() -> None:
    """``pickers.spec.ts`` asserts the form posts exactly ``define-body.json``. This asserts
    the backend's own models accept that document - so the two halves are about one body,
    and neither can drift without a red.

    Each part is handed to the model the definition hands it to (``RunDefinition`` keeps them
    as opaque dicts): ``AcquisitionSpec``, ``RunConfig``, ``SourceSpec``. And every
    identifier in it is one the fixture's lists actually offered: the path, the pack version
    and the stop condition with its parameter name.
    """
    from schemas.run_definition import RunDefinition
    from training.acquisition.base import AcquisitionSpec
    from training.run_config import RunConfig
    from training.signals.base import SourceSpec

    body = json.loads(DEFINE_BODY.read_text())
    definition = RunDefinition.model_validate(body)
    AcquisitionSpec(**definition.model_source)
    RunConfig(**definition.run_configuration)
    SourceSpec(**definition.signal_source)
    assert definition.enabled_conditions() == {"step_count": {"total": 40}}

    offered = json.loads(FIXTURE.read_text())
    paths = [row["path"] for row in offered["/api/v1/models/local?limit=500"]["models"]]
    assert body["model_source"]["path"] in paths
    packs = [(row["pack_id"], row["version"]) for row in offered["/api/v1/packs"]["packs"]]
    assert (body["pack"]["pack_id"], body["pack"]["version"]) in packs
    conditions = {
        row["condition"]: row["required_parameters"]
        for row in offered["/api/v1/stop-conditions"]["conditions"]
    }
    for stop in body["stop_conditions"]:
        assert sorted(stop["config"]) == sorted(conditions[stop["condition"]])
    sources = {
        row["name"]: row["available"] for row in offered["/api/v1/signal-sources"]["sources"]
    }
    assert sources[body["signal_source"]["source"]] is True
    run_types = [row["name"] for row in offered["/api/v1/run-types"]["run_types"]]
    assert body["run_type"] in run_types


DEFINE_CREATE_BODY = REPO_ROOT / "frontend" / "e2e" / "fixtures" / "define-body-create.json"


def test_the_create_body_the_e2e_form_posts_is_one_the_backend_accepts() -> None:
    """``pickers.spec.ts``'s create flow posts ``define-body-create.json``: a create run whose
    tokenizer was chosen from the tokenizer list. The backend's own models accept it, and its
    config and tokenizer are rows the fixture's lists actually offered."""
    from schemas.run_definition import RunDefinition
    from training.acquisition.base import AcquisitionMode, AcquisitionSpec

    body = json.loads(DEFINE_CREATE_BODY.read_text())
    definition = RunDefinition.model_validate(body)
    spec = AcquisitionSpec(**definition.model_source)
    assert spec.mode is AcquisitionMode.CREATE
    assert spec.tokenizer_path is not None

    offered = json.loads(FIXTURE.read_text())
    configs = [row["config_path"] for row in offered["/api/v1/models/local?limit=500"]["models"]]
    assert body["model_source"]["config_path"] in configs
    tokenizers = [
        row["path"] for row in offered["/api/v1/tokenizers/local?limit=500"]["tokenizers"]
    ]
    assert body["model_source"]["tokenizer_path"] in tokenizers
    run_types = [row["name"] for row in offered["/api/v1/run-types"]["run_types"]]
    assert body["run_type"] in run_types


TREE_FIXTURE = REPO_ROOT / "frontend" / "e2e" / "fixtures" / "tree.json"
#: Hashes of manifests that carry their write time; stable shape, unstable value.
TREE_VOLATILE = {"manifest_sha256", "base_manifest_sha256"}


def _normalise_tree(value: Any, root: str) -> Any:
    if isinstance(value, dict):
        return {
            key: (
                "h" * 64
                if key in TREE_VOLATILE and isinstance(item, str)
                else _normalise_tree(item, root)
            )
            for key, item in value.items()
        }
    if isinstance(value, list):
        return [_normalise_tree(item, root) for item in value]
    if isinstance(value, str):
        return value.replace(root, "/data")
    return value


def test_the_e2e_forest_is_what_the_tree_route_answers(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The Checkpoints screen reads ``GET /api/v1/tree`` (task 14.0's inventory, row T-2).

    The E2E specs build their forests from ``tree.json``'s nodes, so this pins those nodes
    to what the route answers for a forest written by the **real** checkpoint and adapter
    writers: a root, a child, and an adapter on the root.
    """
    from tests.support import tree as forest

    checkpoints = tmp_path / "checkpoints"
    checkpoints.mkdir()
    monkeypatch.setenv("CHECKPOINT_ROOT", str(checkpoints))
    monkeypatch.setenv("TRAINER_STATE_ROOT", str(tmp_path / "trainer-state"))
    reset_settings()
    engine = migrated_engine(tmp_path / "db")
    base = forest.root_checkpoint(checkpoints, run_id="run-a", step=100, domain="notary")
    forest.branch(checkpoints, parent=base.name, run_id="run-a", step=200)
    forest.adapter(checkpoints, base=base, run_id="run-b", step=50)
    forest.indexed(engine, checkpoints, run_ids=("run-a", "run-b"))

    app = create_app()

    def override():  # type: ignore[no-untyped-def]
        with session_for(engine) as session:
            yield session

    app.dependency_overrides[session_dependency] = override
    try:
        response = TestClient(app).get("/api/v1/tree", params={"limit": 2000})
    finally:
        app.dependency_overrides.clear()
        engine.dispose()
        reset_settings()
    assert response.status_code == 200, response.text
    answer = _normalise_tree(response.json(), str(tmp_path))
    kinds = sorted(node["kind"] for node in answer["nodes"])
    assert kinds == ["adapter", "checkpoint", "checkpoint"]

    if WRITE:
        TREE_FIXTURE.write_text(json.dumps(answer, indent=2, sort_keys=True) + "\n")
    assert json.loads(TREE_FIXTURE.read_text()) == answer, (
        "tree.json no longer matches GET /api/v1/tree: regenerate with "
        "MIFORGE_WRITE_UI_FIXTURES=1 and review the frontend diff."
    )
