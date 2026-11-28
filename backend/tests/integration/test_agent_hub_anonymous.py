"""An agent's Hub fetches never carry the operator's credential (017 FR-017.26, .28, .33, .34).

Suite CHAIN_PLAN A17, A28, A36, A42. Every case runs the production code path against the REAL
``huggingface_hub`` client behind a transport spy (``tests/support/hub_spy.py``) with
``HF_TOKEN=hf_spy_env`` set, and asserts the ``Authorization`` header each request carried:

- an agent's: none, on every request to the repository;
- the operator's: ``Bearer hf_spy_env`` - without it the agent case would prove nothing.

API process (before any approval): the revision check, run validation, the budget estimate,
and the approval gate's validators when an agent files ``POST /runs/define`` or ``POST /runs``.
For an agent those read metadata only: no weight file is fetched whole (FR-017.34).

Worker: an approved agent ``load`` run, an agent ``create`` run with a Hub tokenizer, and the
attach verification tokenizer fetched in two steps (only tokenizer files, A36).
"""

from __future__ import annotations

import json
from collections.abc import Iterator
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
from fastapi.testclient import TestClient

from tests.support.approval_world import services_answer
from tests.support.hub_spy import BEARER, COMMIT, HubSpy, hub_spy, safetensors_bytes
from tests.support.tiny_hf_model import LLAMA_TINY, write_tiny_tokenizer

__all__ = ["hub_spy"]  # the fixture, imported for pytest

AGENT = {"X-Miforge-Agent": "agent:forge-mcp"}
REPO = "org/model"
TOKENIZER_REPO = "org/tok"
CONFIG = {
    **LLAMA_TINY,
    "architectures": ["LlamaForCausalLM"],
    "torch_dtype": "bfloat16",
}


def _tokenizer_files(tmp_path: Path) -> dict[str, bytes]:
    directory = tmp_path / "tokenizer-src"
    directory.mkdir(exist_ok=True)
    write_tiny_tokenizer(directory)
    return {p.name: p.read_bytes() for p in directory.iterdir() if p.is_file()}


@pytest.fixture
def repos(hub_spy: HubSpy, tmp_path: Path) -> HubSpy:
    tokenizer = _tokenizer_files(tmp_path)
    hub_spy.add_repo(
        REPO,
        {
            "config.json": json.dumps(CONFIG).encode(),
            "model.safetensors": safetensors_bytes("BF16", (64, 64)),
            **tokenizer,
        },
    )
    hub_spy.add_repo(TOKENIZER_REPO, {**tokenizer, "model.safetensors": safetensors_bytes()})
    return hub_spy


def load_definition(run_id: str | None = None) -> dict[str, Any]:
    body: dict[str, Any] = {
        "model_source": {"mode": "load", "repo": REPO, "revision": "main"},
        "run_type": "change",
        "run_configuration": {"learning_rate": 1e-5, "warmup_steps": 0, "batch_size": 1},
        "signal_source": {"source": "corpus"},
        "parts": [{"part_id": "part-1", "kind": "corpus"}],
        "stop_conditions": [{"condition": "step_count", "config": {"total": 4}}],
    }
    if run_id is not None:
        body["run_id"] = run_id
    return body


@pytest.fixture
def client(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, repos: HubSpy) -> Iterator[TestClient]:
    from api.app import create_app
    from api.deps import session_dependency
    from core import runtime_settings
    from core.config import reset_settings
    from packs.settings import reset_pack_settings
    from services import acquisition as acquisition_service
    from services import run_definition as validator
    from tests.support.db import migrated_engine, session_for

    for name, sub in (
        ("CHECKPOINT_ROOT", "checkpoints"),
        ("TRAINER_STATE_ROOT", "trainer-state"),
        ("PARTS_ROOT", "parts"),
        ("PACKS_ROOT", "packs"),
    ):
        monkeypatch.setenv(name, str(tmp_path / sub))
    for name in ("MISERVER_BASE_URL", "MIDATAWORKS_BASE_URL", "MISERVER_TRAINER_TOKEN"):
        monkeypatch.delenv(name, raising=False)
    reset_settings()
    reset_pack_settings()
    runtime_settings.reset()

    # The validator's non-Hub steps read packs, parts and sibling URLs, which other features'
    # tests pin; the Hub step (`_live_budget` -> acquisition pre-flight) runs for real.
    monkeypatch.setattr(validator, "_live_parts", lambda *a, **k: {"source": "parts", "parts": []})
    monkeypatch.setattr(validator, "_live_services", services_answer)
    monkeypatch.setattr(acquisition_service, "resolve_submit_device", lambda request: None)

    engine = migrated_engine(tmp_path / "db")
    app = create_app()

    def override() -> Iterator[Any]:
        with session_for(engine) as session:
            yield session

    app.dependency_overrides[session_dependency] = override
    yield TestClient(app, raise_server_exceptions=False)
    app.dependency_overrides.clear()
    engine.dispose()
    runtime_settings.reset()
    reset_settings()
    reset_pack_settings()


def _budget(body: dict[str, Any]) -> dict[str, Any]:
    return dict((body.get("resolved") or {}).get("budget") or {})


# ====================================================================== API process


class TestTheRevisionCheck:
    def test_an_agents_check_sends_no_credential(self, client: TestClient, repos: HubSpy) -> None:
        response = client.get("/api/v1/hub/revision", params={"repo": REPO}, headers=AGENT)
        assert response.status_code == 200, response.text
        assert response.json()["commit"] == COMMIT
        assert repos.hub_requests
        assert repos.authorizations() == {None}

    def test_the_operators_check_sends_the_pods_credential(
        self, client: TestClient, repos: HubSpy
    ) -> None:
        response = client.get("/api/v1/hub/revision", params={"repo": REPO})
        assert response.status_code == 200, response.text
        assert repos.authorizations() == {BEARER}


class TestValidation:
    def test_an_agents_validate_is_anonymous_and_reads_metadata_only(
        self, client: TestClient, repos: HubSpy
    ) -> None:
        response = client.post("/api/v1/runs/validate", json=load_definition(), headers=AGENT)
        assert response.status_code == 200, response.text
        assert repos.hub_requests
        assert repos.authorizations() == {None}
        assert repos.whole_weight_fetches() == []

    def test_the_operators_validate_sends_the_credential_and_fetches_the_snapshot(
        self, client: TestClient, repos: HubSpy
    ) -> None:
        response = client.post("/api/v1/runs/validate", json=load_definition())
        assert response.status_code == 200, response.text
        assert repos.authorizations() == {BEARER}
        assert repos.whole_weight_fetches(), "the operator's validate fetches the snapshot"

    def test_both_modes_answer_the_same(self, client: TestClient, repos: HubSpy) -> None:
        """FR-017.34: metadata-only is the same answer, not a weaker one."""
        agent = client.post("/api/v1/runs/validate", json=load_definition(), headers=AGENT).json()
        operator = client.post("/api/v1/runs/validate", json=load_definition()).json()
        assert _budget(agent).get("param_count") == 64 * 64
        assert _budget(agent).get("param_count") == _budget(operator).get("param_count")
        assert agent.get("ok") == operator.get("ok")

    def test_a_quantized_model_is_refused_the_same_way_in_both_modes(
        self, client: TestClient, repos: HubSpy
    ) -> None:
        repos.repos[REPO]["model.safetensors"] = safetensors_bytes("I8", (64, 64))
        agent = client.post("/api/v1/runs/validate", json=load_definition(), headers=AGENT).json()
        operator = client.post("/api/v1/runs/validate", json=load_definition()).json()

        def codes(body: dict[str, Any]) -> list[str]:
            return sorted(str(f.get("code")) for f in body.get("failures") or [])

        assert codes(agent) == codes(operator)
        assert codes(agent), agent
        assert repos.whole_weight_fetches() and all(
            s.authorization == BEARER for s in repos.whole_weight_fetches()
        )


ESTIMATE_BODY: dict[str, Any] = {
    "run_type": "change",
    "config": {"learning_rate": 1e-5, "warmup_steps": 0, "batch_size": 1},
    "model_source": {"mode": "load", "repo": REPO, "revision": "main"},
}


class TestTheBudgetEstimate:

    def test_an_agents_estimate_sends_no_credential(
        self, client: TestClient, repos: HubSpy
    ) -> None:
        response = client.post("/api/v1/budget/estimate", json=ESTIMATE_BODY, headers=AGENT)
        assert response.status_code == 200, response.text
        assert repos.hub_requests
        assert repos.authorizations() == {None}

    def test_the_operators_estimate_sends_the_credential(
        self, client: TestClient, repos: HubSpy
    ) -> None:
        response = client.post("/api/v1/budget/estimate", json=ESTIMATE_BODY)
        assert response.status_code == 200, response.text
        assert repos.authorizations() == {BEARER}


class TestAnAgentsFiling:
    """The gate's validator runs BEFORE the 202, for agents only (FR-017.33, .34)."""

    def test_a_define_filing_is_anonymous_and_fetches_no_weights(
        self, client: TestClient, repos: HubSpy
    ) -> None:
        response = client.post("/api/v1/runs/define", json=load_definition(), headers=AGENT)
        assert response.status_code == 202, response.text
        assert response.json()["status"] == "pending"
        assert repos.hub_requests
        assert repos.authorizations() == {None}
        assert repos.whole_weight_fetches() == []

    def test_a_submit_filing_is_anonymous_and_fetches_no_weights(
        self, client: TestClient, repos: HubSpy
    ) -> None:
        body = {
            "mode": "load",
            "repo": REPO,
            "revision": "main",
            "learning_rate": 1e-5,
            "warmup_steps": 0,
            "batch_size": 1,
        }
        response = client.post("/api/v1/runs", json=body, headers=AGENT)
        assert response.status_code == 202, response.text
        assert repos.hub_requests
        assert repos.authorizations() == {None}
        assert repos.whole_weight_fetches() == []

    def test_an_approved_agent_submit_preflights_anonymously(
        self, client: TestClient, repos: HubSpy, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """AFTER the approval: the executor re-runs ``POST /runs``'s body as the agent, and its
        pre-flight must still send no credential (R2 review fix OR2-017-2, 2026-11-16).

        ``services/acquisition.submit`` reads ``hub_access.hub_token()``; the filing case above
        never reaches that line (the gate's validator answers the 202 first), so without this
        case ``token = None`` there was green while an approved agent run borrowed the pod's
        credential and fetched the whole snapshot.
        """
        from celery import Celery
        from celery.app.task import Task

        monkeypatch.setattr(Task, "apply_async", lambda self, *a, **k: None)
        monkeypatch.setattr(Celery, "send_task", lambda self, *a, **k: None)
        body = {
            "mode": "load",
            "repo": REPO,
            "revision": "main",
            "learning_rate": 1e-5,
            "warmup_steps": 0,
            "batch_size": 1,
        }
        filed = client.post("/api/v1/runs", json=body, headers=AGENT)
        assert filed.status_code == 202, filed.text
        before = len(repos.hub_requests)
        decided = client.post(f"/api/v1/approvals/{filed.json()['approval_id']}/approve")
        assert decided.status_code == 200, decided.text
        after = repos.hub_requests[before:]
        assert after, ("the approved submit made no Hub request", decided.json())
        assert {request.authorization for request in after} == {None}, decided.json()


# ====================================================================== worker


def _run_row(engine: Any, run_id: str, access: str | None) -> None:
    import sqlalchemy as sa

    from tests.support.db import session_for
    from tests.support.jobs import insert_run
    from workers.run_record import RUNS

    with session_for(engine) as session:
        insert_run(session, run_id, status="pending")
        session.execute(sa.update(RUNS).where(RUNS.c.id == run_id).values(hub_access=access))
        session.commit()


@pytest.fixture
def worker(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, repos: HubSpy) -> Any:
    from tests.support.db import migrated_engine

    engine = migrated_engine(tmp_path / "db")
    yield engine
    engine.dispose()


def _in_run_scope(engine: Any, run_id: str, fn: Any) -> Any:
    from tests.support.db import session_for
    from training import tasks

    with session_for(engine) as session, tasks._hub_scope(session, run_id):
        return fn()


class TestTheWorker:
    @pytest.mark.parametrize(("access", "expected"), [("anonymous", None), (None, BEARER)])
    def test_a_load_runs_fetches_follow_the_runs_hub_access(
        self,
        worker: Any,
        repos: HubSpy,
        monkeypatch: pytest.MonkeyPatch,
        access: str | None,
        expected: str | None,
    ) -> None:
        from training import tasks
        from training.acquisition import local_dir
        from training.acquisition.base import AcquisitionSpec

        loaded: list[Path] = []

        def load(directory: Path, dtype_name: str) -> Any:
            loaded.append(directory)
            raise RuntimeError("stop after the fetch: the load itself is not under test")

        monkeypatch.setattr(local_dir, "_load_pretrained", load)
        _run_row(worker, "run-w", access)
        spec = AcquisitionSpec(repo=REPO, revision="main")
        with pytest.raises(RuntimeError, match="stop after the fetch"):
            _in_run_scope(worker, "run-w", lambda: tasks.acquire_for_run(spec))
        assert loaded, "the snapshot reached the loader"
        assert repos.authorizations() == {expected}
        assert repos.whole_weight_fetches(), "the worker fetches the snapshot under its lease"

    @pytest.mark.parametrize(("access", "expected"), [("anonymous", None), (None, BEARER)])
    def test_a_create_runs_hub_tokenizer_follows_the_runs_hub_access(
        self,
        worker: Any,
        repos: HubSpy,
        access: str | None,
        expected: str | None,
    ) -> None:
        from training import acquisition
        from training.acquisition.base import AcquisitionSpec
        from training.acquisition.scratch import TOKENIZER_FILE_PATTERNS

        _run_row(worker, "run-c", access)
        spec = AcquisitionSpec(
            mode="create",
            config=dict(LLAMA_TINY),
            tokenizer_repo=TOKENIZER_REPO,
            tokenizer_revision="main",
        )
        from services import hub_access

        def preflight() -> Any:
            acquirer = acquisition.resolve(spec)
            return acquirer.preflight(spec, token=hub_access.worker_hub_token(), fetch="snapshot")

        enriched = _in_run_scope(worker, "run-c", preflight)
        assert enriched.tokenizer_vocab_size == LLAMA_TINY["vocab_size"]
        assert repos.authorizations() == {expected}
        assert repos.whole_weight_fetches() == []
        assert TOKENIZER_FILE_PATTERNS  # only tokenizer files were asked for

    @pytest.mark.parametrize(("token", "expected"), [(False, None), (None, BEARER)])
    def test_the_attach_verification_tokenizer_is_fetched_in_two_steps(
        self, repos: HubSpy, token: bool | None, expected: str | None
    ) -> None:
        """A36: never ``AutoTokenizer.from_pretrained(<hub name>)``, which lists the repo's
        files with the pod's token whatever ``token`` says."""
        from training.hotupdate.attach import AttachAcquirer

        acquirer = AttachAcquirer()
        acquirer._hub_token = token
        live = SimpleNamespace(verify_tokenizer=TOKENIZER_REPO)
        tokenizer = acquirer._tokenizer_loader("served-model", live)()
        assert len(tokenizer) == LLAMA_TINY["vocab_size"]
        assert repos.hub_requests
        assert repos.authorizations() == {expected}
        assert not any(name.endswith(".safetensors") for name in repos.fetched_files())

    def test_the_attach_acquirers_default_token_is_false(self, repos: HubSpy) -> None:
        """A loader built before any ``acquire`` fetches anonymously: the acquirer is the
        process-wide registry singleton, so its default is the fail-closed one (R2 review fix
        OR2-017-3, 2026-11-16)."""
        from training.hotupdate.attach import AttachAcquirer

        acquirer = AttachAcquirer()
        assert acquirer._hub_token is False
        live = SimpleNamespace(verify_tokenizer=TOKENIZER_REPO)
        acquirer._tokenizer_loader("served-model", live)()
        assert repos.hub_requests
        assert repos.authorizations() == {None}

    @pytest.mark.parametrize(("access", "expected"), [("anonymous", None), (None, BEARER)])
    def test_a_create_runs_acquire_through_acquire_for_run_follows_the_runs_hub_access(
        self,
        worker: Any,
        repos: HubSpy,
        monkeypatch: pytest.MonkeyPatch,
        access: str | None,
        expected: str | None,
    ) -> None:
        """Pre-flight AND acquire through the worker's one entry point (review R2, M07)."""
        from training import tasks
        from training.acquisition import scratch
        from training.acquisition.base import AcquisitionSpec

        def stop(*_a: Any, **_k: Any) -> Any:
            raise RuntimeError("stop after the tokenizer fetch")

        monkeypatch.setattr(scratch, "_instantiate", stop)
        _run_row(worker, "run-cc", access)
        spec = AcquisitionSpec(
            mode="create",
            config=dict(LLAMA_TINY),
            tokenizer_repo=TOKENIZER_REPO,
            tokenizer_revision="main",
        )
        before = len(repos.hub_requests)
        with pytest.raises(RuntimeError, match="stop after the tokenizer fetch"):
            _in_run_scope(worker, "run-cc", lambda: tasks.acquire_for_run(spec))
        assert len(repos.hub_requests) > before
        assert repos.authorizations() == {expected}

    @pytest.mark.parametrize("token", [False, None])
    def test_the_attach_acquirer_keeps_the_runs_token(self, token: bool | None) -> None:
        """``acquire`` records the token ``acquire_for_run`` handed it before anything else, so
        the verification tokenizer's fetch uses the run's choice (the wiring line)."""
        from training.acquisition.base import AcquisitionSpec
        from training.hotupdate.attach import AttachAcquirer

        acquirer = AttachAcquirer()
        acquirer._hub_token = "unset"  # type: ignore[assignment]
        with pytest.raises(Exception):  # noqa: B017 - unavailable on this machine: refused
            acquirer.acquire(AcquisitionSpec(mode="attach"), token=token)
        assert acquirer._hub_token is token


class TestTheWorkerTaskEntersTheRunsScope:
    """The wiring: ``run_step_loop`` - the Celery task - holds the run's scope around the run.

    The cases above call ``tasks._hub_scope`` themselves; this one goes through the task, so
    removing the scope from ``run_step_loop`` is red here (the fail-closed default would make
    the operator's run anonymous, and nothing would make an agent's run anonymous).
    """

    @pytest.mark.parametrize(("access", "expected"), [("anonymous", False), (None, None)])
    def test_the_task_drives_the_run_inside_its_scope(
        self, worker: Any, monkeypatch: pytest.MonkeyPatch, access: str | None, expected: Any
    ) -> None:
        import db.session as db_session
        from services import hub_access
        from tests.support.db import session_for
        from training import run_driver, tasks

        seen: list[Any] = []

        def drive(run_id: str, **_: Any) -> dict[str, Any]:
            seen.append(hub_access.worker_hub_token())
            return {}

        monkeypatch.setattr(run_driver, "drive_recorded_run", drive)
        monkeypatch.setattr(db_session, "session_scope", lambda: session_for(worker))
        monkeypatch.setattr(tasks, "release_device_memory", lambda run_id=None: None)
        _run_row(worker, "run-t", access)

        tasks.run_step_loop("run-t")

        assert seen == [expected]
        assert hub_access.worker_hub_token() is False  # the scope ended with the run
