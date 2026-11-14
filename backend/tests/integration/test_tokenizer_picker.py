"""The tokenizer a ``create`` run ships, chosen on the New run form  (operator request 2026-10-20).

A create run started from the form was refused ``tokenizer_required`` every time: the backend
accepted a tokenizer on the model source, and the form had nowhere to name one. These tests
cover the two read routes the form's tokenizer picker reads and the validator's answer, all
through the **mounted** app (``create_app()``), so deleting the ``include_router`` line in
``api/v1/router.py`` turns this file red.

* ``GET /api/v1/tokenizers/local`` - real directories on disk under the real model roots.
* ``GET /api/v1/hub/revision`` - the Hub client is replaced at the module boundary
  (``hub._hub``), and the fake's signatures are checked against the installed client's.
* ``POST /api/v1/runs/validate`` - a create definition with a tokenizer passes the budget step
  and the plan names the tokenizer; one without is refused ``tokenizer_required`` in plain
  words that name no field.
"""

from __future__ import annotations

import inspect
import re
from collections.abc import Iterator
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
from fastapi.testclient import TestClient

from api.app import create_app
from api.deps import session_dependency
from core.config import reset_settings
from tests.support.db import migrated_engine, session_for
from tests.support.tiny_hf_model import LLAMA_TINY, write_tiny_tokenizer
from tests.unit.test_served_text_carries_no_ids import IDS
from training.acquisition import hub as hub_module

COMMIT = "c" * 40


@pytest.fixture
def roots(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Iterator[Path]:
    """The model roots under ``tmp_path``. Returns the checkpoint root (a model root)."""
    checkpoints = tmp_path / "checkpoints"
    checkpoints.mkdir()
    monkeypatch.setenv("CHECKPOINT_ROOT", str(checkpoints))
    monkeypatch.setenv("PARTS_ROOT", str(tmp_path / "parts"))
    monkeypatch.setenv("TRAINER_STATE_ROOT", str(tmp_path / "trainer-state"))
    reset_settings()
    yield checkpoints
    reset_settings()


@pytest.fixture
def rig(roots: Path) -> tuple[TestClient, Path]:
    """The mounted app with no database: the two listings read the disk and the Hub only."""
    return TestClient(create_app(), raise_server_exceptions=False), roots


@pytest.fixture
def db_rig(roots: Path, tmp_path: Path) -> Iterator[tuple[TestClient, Path]]:
    """The mounted app on a migrated schema, for the validator (it resolves parts)."""
    engine = migrated_engine(tmp_path / "db")
    app = create_app()

    def one_session():  # type: ignore[no-untyped-def]
        with session_for(engine) as session:
            yield session

    app.dependency_overrides[session_dependency] = one_session
    try:
        yield TestClient(app, raise_server_exceptions=False), roots
    finally:
        app.dependency_overrides.clear()
        engine.dispose()


def touch(directory: Path, *names: str) -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    for name in names:
        (directory / name).write_text("{}", encoding="utf-8")
    return directory


# ======================================================================================
# GET /api/v1/tokenizers/local
# ======================================================================================


class TestTheTokenizerList:
    def test_it_lists_every_directory_with_tokenizer_files_at_its_root(self, rig) -> None:
        client, root = rig
        touch(root / "tok-only", "tokenizer.json", "tokenizer_config.json")
        touch(root / "org" / "model", "config.json", "tokenizer.json", "model.safetensors")
        touch(root / "run-1" / "step-10", "config.json", "manifest.json", "tokenizer.model")
        touch(root / "weights-only", "config.json", "model.safetensors")
        touch(root / ".cache" / "hidden", "tokenizer.json")

        response = client.get("/api/v1/tokenizers/local")

        assert response.status_code == 200, response.text
        body = response.json()
        rows = {row["relative_path"]: row for row in body["tokenizers"]}
        assert set(rows) == {"tok-only", "org/model", "run-1/step-10"}
        assert rows["tok-only"]["kind"] == "tokenizer"
        assert rows["tok-only"]["tokenizer_files"] == ["tokenizer.json", "tokenizer_config.json"]
        assert rows["org/model"]["kind"] == "model"
        assert rows["run-1/step-10"]["kind"] == "checkpoint"
        assert rows["tok-only"]["path"] == str((root / "tok-only").resolve())
        assert body["total"] == 3
        assert str(root) in body["roots"] or str(root.resolve()) in body["roots"]

    def test_a_tokenizer_subfolder_is_listed_only_when_its_parent_has_none(self, rig) -> None:
        """The create path refuses a parent whose tokenizer is one level down and accepts the
        subfolder itself, so the list offers the subfolder - and never both."""
        client, root = rig
        touch(root / "nested-model", "config.json")
        touch(root / "nested-model" / "tokenizer", "tokenizer.json")
        touch(root / "flat-model", "config.json", "tokenizer.json")
        touch(root / "flat-model" / "tokenizer", "tokenizer.json")

        paths = {
            row["relative_path"]
            for row in client.get("/api/v1/tokenizers/local").json()["tokenizers"]
        }

        assert paths == {"nested-model/tokenizer", "flat-model"}

    def test_a_root_that_is_not_there_is_named_not_skipped(
        self, rig, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        client, _root = rig
        monkeypatch.setenv("PARTS_ROOT", str(tmp_path / "absent-parts"))
        reset_settings()

        body = client.get("/api/v1/tokenizers/local").json()

        assert [row["code"] for row in body["unreadable_roots"]] == ["model_root_missing"]


# ======================================================================================
# GET /api/v1/hub/revision
# ======================================================================================


class FakeHub:
    """``huggingface_hub`` as ``training.acquisition.hub`` calls it. Records every call."""

    def __init__(self, files: list[str], *, missing_revision: bool = False) -> None:
        self.files = files
        self.missing_revision = missing_revision
        self.calls: list[tuple[str, str, str]] = []
        self.snapshot: Path | None = None
        #: A whole repository on disk; when set, a download copies only what matches.
        self.remote: Path | None = None
        self.allow_patterns: list[list[str] | None] = []

    def repo_info(self, repo_id: str, revision: str) -> Any:
        self.calls.append(("repo_info", repo_id, revision))
        if self.missing_revision:
            raise RevisionNotFoundError("no such revision")
        return SimpleNamespace(sha=COMMIT)

    def list_repo_files(self, repo_id: str, revision: str) -> list[str]:
        self.calls.append(("list_repo_files", repo_id, revision))
        return list(self.files)

    def snapshot_download(
        self, repo_id: str, revision: str, allow_patterns: list[str] | None = None
    ) -> str:
        self.calls.append(("snapshot_download", repo_id, revision))
        self.allow_patterns.append(allow_patterns)
        if self.remote is not None:
            # A repository with weights in it: materialise only what the patterns admit,
            # by the installed client's own matching rule, so the test measures what the
            # real fetch would have written to disk.
            from huggingface_hub.utils import filter_repo_objects

            names = [
                str(path.relative_to(self.remote))
                for path in sorted(self.remote.rglob("*"))
                if path.is_file()
            ]
            local = self.remote.parent / f"{self.remote.name}-fetched"
            for name in filter_repo_objects(names, allow_patterns=allow_patterns):
                (local / name).parent.mkdir(parents=True, exist_ok=True)
                (local / name).write_bytes((self.remote / name).read_bytes())
            return str(local)
        assert self.snapshot is not None
        return str(self.snapshot)


class RevisionNotFoundError(Exception):
    """Named as ``huggingface_hub.errors`` names it; the reader matches the name."""


def test_the_fake_hub_answers_the_way_the_real_client_is_called() -> None:
    import huggingface_hub

    for name in ("repo_info", "list_repo_files", "snapshot_download"):
        real = inspect.signature(getattr(huggingface_hub, name)).parameters
        for parameter in inspect.signature(getattr(FakeHub, name)).parameters:
            if parameter != "self":
                assert parameter in real, (name, parameter)


class TestTheHubCheck:
    def test_a_revision_resolves_to_its_commit_and_the_root_files_are_reported(
        self, rig, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        client, _root = rig
        fake = FakeHub(
            [
                "tokenizer.json",
                "tokenizer_config.json",
                "config.json",
                "model.safetensors",
                "nested/vocab.json",
            ]
        )
        monkeypatch.setattr(hub_module, "_hub", lambda: fake)

        response = client.get(
            "/api/v1/hub/revision", params={"repo": "org/tok", "revision": "main"}
        )

        assert response.status_code == 200, response.text
        assert response.json() == {
            "repo": "org/tok",
            "revision": "main",
            "commit": COMMIT,
            "is_commit": False,
            "tokenizer_files": ["tokenizer.json", "tokenizer_config.json"],
            "has_tokenizer": True,
            "has_config": True,
            "has_weights": True,
        }
        # Resolved FIRST, then listed AT THE COMMIT, once each - never at the moving branch.
        assert fake.calls == [
            ("repo_info", "org/tok", "main"),
            ("list_repo_files", "org/tok", COMMIT),
        ]

    def test_no_revision_means_main_and_a_subfolder_tokenizer_is_not_at_the_root(
        self, rig, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        client, _root = rig
        fake = FakeHub(["tokenizer/tokenizer.json", "onnx/model.safetensors", "README.md"])
        monkeypatch.setattr(hub_module, "_hub", lambda: fake)

        body = client.get("/api/v1/hub/revision", params={"repo": "org/tok"}).json()

        assert body["revision"] == "main"
        assert body["has_tokenizer"] is False
        assert body["tokenizer_files"] == []
        # Weights one level down are not weights a load reads: only the root counts.
        assert body["has_weights"] is False

    def test_an_unknown_revision_is_refused_by_name(
        self, rig, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        client, _root = rig
        monkeypatch.setattr(hub_module, "_hub", lambda: FakeHub([], missing_revision=True))

        response = client.get(
            "/api/v1/hub/revision", params={"repo": "org/tok", "revision": "nope"}
        )

        assert response.status_code == 400
        assert response.json()["error"]["code"] == "model_revision_not_found"

    def test_a_malformed_repository_id_never_reaches_the_hub(
        self, rig, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        client, _root = rig
        fake = FakeHub([])
        monkeypatch.setattr(hub_module, "_hub", lambda: fake)

        response = client.get("/api/v1/hub/revision", params={"repo": "../etc/passwd"})

        assert response.status_code == 422
        assert fake.calls == []


# ======================================================================================
# POST /api/v1/runs/validate - a create run's tokenizer
# ======================================================================================


def create_definition(**model_source: Any) -> dict[str, Any]:
    return {
        "run_id": "run-create-tok",
        "model_source": {"mode": "create", "config": dict(LLAMA_TINY), **model_source},
        "run_type": "create",
        "run_configuration": {
            "learning_rate": 1e-5,
            "warmup_steps": 2,
            "batch_size": 2,
            "gradient_accumulation": 1,
        },
        "signal_source": {"source": "corpus"},
        "parts": [{"part_id": "part-1", "kind": "corpus"}],
        "cadences": {"test_every": 2, "retention_every": 1, "save_every": 10},
        "stop_conditions": [{"condition": "step_count", "config": {"total": 100}}],
        "seed": 13,
    }


def tokenizer_failures(body: dict[str, Any]) -> list[dict[str, Any]]:
    return [row for row in body["failures"] if str(row["code"]).startswith("tokenizer")]


class TestACreateRunsTokenizerAtValidate:
    @pytest.fixture(autouse=True)
    def _needs_the_ml_runtime(self) -> None:
        pytest.importorskip("transformers")
        pytest.importorskip("tokenizers")

    def test_a_tokenizer_directory_passes_and_the_plan_names_it(self, db_rig) -> None:
        client, root = db_rig
        directory = root / "given-tokenizer"
        directory.mkdir()
        write_tiny_tokenizer(directory, vocab_size=LLAMA_TINY["vocab_size"])

        response = client.post(
            "/api/v1/runs/validate",
            json=create_definition(tokenizer_path=str(directory)),
        )

        assert response.status_code == 200, response.text
        body = response.json()
        assert tokenizer_failures(body) == []
        assert "budget" in body["steps_checked"], body
        assert body["resolved"]["budget"]["tokenizer"] == {
            "source": "directory",
            "path": str(directory.resolve()),
            "vocab_size": LLAMA_TINY["vocab_size"],
        }

    def test_a_hub_tokenizer_passes_and_the_plan_names_its_commit(
        self, db_rig, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        client, _root = db_rig
        snapshot = tmp_path / "snapshot"
        snapshot.mkdir()
        write_tiny_tokenizer(snapshot, vocab_size=LLAMA_TINY["vocab_size"])
        fake = FakeHub(["tokenizer.json"])
        fake.snapshot = snapshot
        monkeypatch.setattr(hub_module, "_hub", lambda: fake)

        response = client.post(
            "/api/v1/runs/validate",
            json=create_definition(tokenizer_repo="org/tok", tokenizer_revision="main"),
        )

        body = response.json()
        assert tokenizer_failures(body) == []
        assert body["resolved"]["budget"]["tokenizer"] == {
            "source": "hub",
            "repo": "org/tok",
            "revision": "main",
            "commit": COMMIT,
            "vocab_size": LLAMA_TINY["vocab_size"],
        }
        assert ("snapshot_download", "org/tok", COMMIT) in fake.calls

    def test_a_hub_tokenizer_fetches_ONLY_the_tokenizer_files_not_the_weights(
        self, db_rig, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        """Naming a model repository as the tokenizer source used to download the whole
        repository - every weight shard - to read one tokenizer.json, because the fetch
        had no file filter. The request now names the tokenizer's files, and the fake
        applies them with the installed client's own matcher, so a weight file reaching
        disk is a red here, not only a different argument."""
        from training.acquisition.scratch import TOKENIZER_FILE_PATTERNS

        client, _root = db_rig
        remote = tmp_path / "model-repo"
        remote.mkdir()
        write_tiny_tokenizer(remote, vocab_size=LLAMA_TINY["vocab_size"])
        (remote / "config.json").write_text("{}", encoding="utf-8")
        for weight in (
            "model.safetensors",
            "model-00001-of-00002.safetensors",
            "pytorch_model.bin",
            "original/consolidated.00.pth",
        ):
            (remote / weight).parent.mkdir(parents=True, exist_ok=True)
            (remote / weight).write_bytes(b"\0" * 64)
        fake = FakeHub(["tokenizer.json", "model.safetensors"])
        fake.remote = remote
        monkeypatch.setattr(hub_module, "_hub", lambda: fake)

        response = client.post(
            "/api/v1/runs/validate",
            json=create_definition(tokenizer_repo="org/model", tokenizer_revision="main"),
        )

        assert response.status_code == 200, response.text
        assert tokenizer_failures(response.json()) == []
        assert fake.allow_patterns == [
            list(TOKENIZER_FILE_PATTERNS)
        ], "one fetch, filtered to the tokenizer's files"
        fetched = {
            str(path.relative_to(tmp_path / "model-repo-fetched"))
            for path in (tmp_path / "model-repo-fetched").rglob("*")
            if path.is_file()
        }
        assert "tokenizer.json" in fetched
        assert not {name for name in fetched if name.endswith((".safetensors", ".bin", ".pth"))}

    def test_no_tokenizer_is_refused_in_plain_words(self, db_rig) -> None:
        client, _root = db_rig

        response = client.post("/api/v1/runs/validate", json=create_definition())

        assert response.status_code == 200, response.text
        failures = tokenizer_failures(response.json())
        assert [row["code"] for row in failures] == ["tokenizer_required"]
        message = failures[0]["message"]
        assert "tokenizer" in message.lower()
        # Plain words for the person at the form: no field names, no code spans, no ids.
        assert "`" not in message
        assert not re.search(r"tokenizer_(path|repo|revision)", message), message
        assert not [pattern.pattern for pattern in IDS if pattern.search(message)], message

    def test_a_load_run_names_no_tokenizer_in_its_plan(self, roots: Path) -> None:
        """Only a create has a tokenizer to choose; a load's comes from its own directory."""
        from services.run_definition import _tokenizer_plan
        from training.acquisition.base import AcquisitionSpec

        spec = AcquisitionSpec(mode="load", path=roots / "m")
        assert _tokenizer_plan(spec) is None
