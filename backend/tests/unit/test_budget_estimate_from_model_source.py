"""The budget preview reads the model's size from the model source  (found live, 2026-10-20).

The New run form sent ``run_type`` and the optimizer settings only - no parameter count and no
architecture config - so ``POST /api/v1/budget/estimate`` refused ``budget_spec_incomplete``
on every press of "Estimate GPU memory". The form now sends the model source it already
holds, and the server reads the size from it with the start path's own readers
(``services.budget_source``).

Each model-source kind is driven through the mounted route with a **real tiny model**: a
Hugging Face directory written by ``transformers`` itself. The expected count is not derived
from the same headers the code reads; it is ``sum(p.numel())`` over the model loaded back by
``transformers`` - an independent oracle, so a fixture cannot agree with the reader by
construction.
"""

from __future__ import annotations

import shutil
from collections.abc import Iterator
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
from fastapi.testclient import TestClient

from api.app import create_app
from core.config import Settings, reset_settings
from tests.support.tiny_hf_model import LLAMA_TINY, write_tiny_model
from training import acquisition
from training.acquisition import hub as hub_module
from training.acquisition.base import AcquisitionMode, Availability
from training.acquisition.dtype_gate import safetensors_header

pytest.importorskip("torch")
pytest.importorskip("transformers")

CFG = {"learning_rate": 1e-4, "warmup_steps": 0, "batch_size": 2}
COMMIT = "c" * 40
ROUTE = "/api/v1/budget/estimate"


@pytest.fixture(scope="module")
def roots(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """Model roots holding one real tiny model directory, written once per module."""
    base = tmp_path_factory.mktemp("budget-roots")
    (base / "checkpoints").mkdir()
    write_tiny_model(base / "parts" / "models" / "tiny")
    return base


@pytest.fixture(scope="module")
def oracle(roots: Path) -> int:
    """The tiny model's parameter count, from the model ``transformers`` loads back."""
    from transformers import AutoModelForCausalLM

    model = AutoModelForCausalLM.from_pretrained(roots / "parts" / "models" / "tiny")
    return int(sum(p.numel() for p in model.parameters()))


@pytest.fixture
def rooted(roots: Path, monkeypatch: pytest.MonkeyPatch) -> Iterator[Path]:
    monkeypatch.setattr(
        "core.config._settings",
        Settings(  # type: ignore[call-arg]
            _env_file=None, checkpoint_root=roots / "checkpoints", parts_root=roots / "parts"
        ),
    )
    yield roots / "parts" / "models" / "tiny"
    reset_settings()


@pytest.fixture
def client() -> TestClient:
    return TestClient(create_app())


def estimate(client: TestClient, **body: Any) -> Any:
    return client.post(ROUTE, json={"run_type": "change", "config": CFG, **body})


def assert_estimated(response: Any, *, param_count: int, kind: str) -> dict[str, Any]:
    assert response.status_code == 200, response.text
    body: dict[str, Any] = response.json()
    assert body["estimate"]["param_count"] == param_count
    assert body["source"]["kind"] == kind
    # The screen's figures: a total, and terms that add up to it.
    assert body["estimate_bytes"] > 0
    assert sum(term["bytes"] for term in body["terms"]) == body["estimate_bytes"]
    assert body["estimate_low_bytes"] < body["estimate_bytes"]
    return body


class TestEachModelSourceKindIsEstimated:
    def test_a_directory_on_this_host(self, client: TestClient, rooted: Path, oracle: int) -> None:
        body = assert_estimated(
            estimate(client, model_source={"mode": "load", "path": str(rooted)}),
            param_count=oracle,
            kind="directory",
        )
        assert body["estimate"]["basis"] == "weights"
        assert body["source"]["path"] == str(rooted.resolve())

    def test_a_miforge_checkpoint(
        self, client: TestClient, rooted: Path, oracle: int, tmp_path: Path
    ) -> None:
        from services.checkpoints import manifest as manifest_module

        checkpoint = rooted.parents[1] / "checkpoints" / f"run-{tmp_path.name}" / "step-10"
        shutil.copytree(rooted, checkpoint)
        (checkpoint / manifest_module.MANIFEST_NAME).write_text("{}", encoding="utf-8")
        assert_estimated(
            estimate(client, model_source={"mode": "load", "path": str(checkpoint)}),
            param_count=oracle,
            kind="checkpoint",
        )

    def test_a_hub_repository_is_read_without_downloading_it(
        self, client: TestClient, rooted: Path, oracle: int, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        fake = FakeHub(rooted)
        monkeypatch.setattr(hub_module, "_hub", lambda: fake)
        body = assert_estimated(
            estimate(client, model_source={"mode": "load", "repo": "org/tiny"}),
            param_count=oracle,
            kind="hub",
        )
        assert body["source"]["revision_resolved"] == COMMIT
        assert fake.calls == [
            ("repo_info", "org/tiny", "main"),
            ("hf_hub_download", "org/tiny", COMMIT),
            ("get_safetensors_metadata", "org/tiny", COMMIT),
        ], "pinned to the resolved commit, and nothing but the config file is downloaded"

    def test_an_inline_architecture_config(self, client: TestClient, oracle: int) -> None:
        body = assert_estimated(
            client.post(
                ROUTE,
                json={
                    "run_type": "create",
                    "config": CFG,
                    "model_source": {"mode": "create", "config": LLAMA_TINY},
                },
            ),
            param_count=oracle,
            kind="config",
        )
        assert body["estimate"]["basis"] == "config"

    def test_an_architecture_config_file(
        self, client: TestClient, rooted: Path, oracle: int
    ) -> None:
        assert_estimated(
            client.post(
                ROUTE,
                json={
                    "run_type": "create",
                    "config": CFG,
                    "model_source": {"mode": "create", "config_path": str(rooted / "config.json")},
                },
            ),
            param_count=oracle,
            kind="config",
        )

    def test_the_model_miLLM_is_serving(
        self, client: TestClient, harness: Any, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from training.hotupdate.attach import AttachAcquirer

        acquirer = AttachAcquirer(transport=harness.fake.transport())
        monkeypatch.setattr(AttachAcquirer, "available", lambda self: Availability(available=True))
        monkeypatch.setitem(acquisition._REGISTRY, AcquisitionMode.ATTACH, acquirer)
        assert_estimated(
            estimate(client, model_source={"mode": "attach"}),
            param_count=harness.fake.num_parameters,
            kind="attach",
        )
        assert harness.fake.call_count("models") == 1
        assert harness.fake.call_count("attach") == 0, "an estimate attaches nothing"

    def test_an_adapter_run_bounds_the_adapter_from_the_directorys_config(
        self, client: TestClient, rooted: Path, oracle: int
    ) -> None:
        response = client.post(
            ROUTE,
            json={
                "run_type": "adapter",
                "config": CFG,
                "model_source": {"mode": "load", "path": str(rooted)},
                "adapter": {"rank": 4, "alpha": 8.0, "target_modules": ["q_proj", "v_proj"]},
            },
        )
        assert response.status_code == 200, response.text
        body = response.json()
        assert body["estimate"]["base_param_count"] == oracle
        assert body["estimate"]["adapter_param_basis"] == "bound"
        assert body["source"]["from_config"] is True


class TestItStillRefusesWhenTheSizeCannotBeKnown:
    def test_nothing_to_read_a_size_from(self, client: TestClient) -> None:
        response = estimate(client)
        assert response.status_code == 400
        error = response.json()["error"]
        assert error["code"] == "budget_spec_incomplete"
        assert "Choose a model source" in error["message"]

    def test_a_model_source_and_a_size_together_is_ambiguous(
        self, client: TestClient, rooted: Path
    ) -> None:
        response = estimate(
            client, model_source={"mode": "load", "path": str(rooted)}, param_count=1000
        )
        assert response.json()["error"]["code"] == "budget_spec_ambiguous"

    def test_a_load_that_names_no_directory_or_repository(self, client: TestClient) -> None:
        response = estimate(client, model_source={"mode": "load"})
        assert response.status_code == 422, response.text

    def test_a_directory_outside_the_model_roots(
        self, client: TestClient, rooted: Path, tmp_path: Path
    ) -> None:
        response = estimate(client, model_source={"mode": "load", "path": str(tmp_path)})
        assert response.json()["error"]["code"] == "model_source_outside_root"

    def test_a_directory_with_no_weights(
        self, client: TestClient, rooted: Path, tmp_path: Path
    ) -> None:
        empty = rooted.parent / f"empty-{tmp_path.name}"
        empty.mkdir()
        shutil.copy(rooted / "config.json", empty / "config.json")
        response = estimate(client, model_source={"mode": "load", "path": str(empty)})
        assert response.json()["error"]["code"] == "model_source_incomplete"

    def test_a_quantized_directory_is_refused_not_estimated_from_packed_headers(
        self, client: TestClient, rooted: Path, tmp_path: Path
    ) -> None:
        """A quantized model's headers understate its size; the start refuses it, so the
        preview says so rather than showing a figure for a run that cannot start."""
        import json

        quantized = rooted.parent / f"quantized-{tmp_path.name}"
        shutil.copytree(rooted, quantized)
        config = json.loads((quantized / "config.json").read_text(encoding="utf-8"))
        config["quantization_config"] = {"quant_method": "bitsandbytes", "load_in_4bit": True}
        (quantized / "config.json").write_text(json.dumps(config), encoding="utf-8")
        response = estimate(client, model_source={"mode": "load", "path": str(quantized)})
        assert response.json()["error"]["code"] == "model_quantized"

    def test_a_hub_repository_without_safetensors(
        self, client: TestClient, rooted: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        fake = FakeHub(rooted, not_safetensors=True)
        monkeypatch.setattr(hub_module, "_hub", lambda: fake)
        response = estimate(client, model_source={"mode": "load", "repo": "org/pickled"})
        assert response.json()["error"]["code"] == "model_weights_format_unsupported"

    def test_a_served_model_that_reports_no_size(
        self, client: TestClient, harness: Any, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from training.hotupdate.attach import AttachAcquirer

        harness.fake.num_parameters = None
        acquirer = AttachAcquirer(transport=harness.fake.transport())
        monkeypatch.setattr(AttachAcquirer, "available", lambda self: Availability(available=True))
        monkeypatch.setitem(acquisition._REGISTRY, AcquisitionMode.ATTACH, acquirer)
        response = estimate(client, model_source={"mode": "attach"})
        assert response.json()["error"]["code"] == "attach_resident_size_unknown"

    def test_a_create_config_that_states_no_shape(self, client: TestClient) -> None:
        response = client.post(
            ROUTE,
            json={
                "run_type": "create",
                "config": CFG,
                "model_source": {"mode": "create", "config": {"model_type": "llama"}},
            },
        )
        assert response.json()["error"]["code"] == "architecture_config_invalid"


# ---------------------------------------------------------------------------------------- GPUs

UUIDS = (
    "GPU-247aa582-0d1b-e161-8156-983ed1fefc57",
    "GPU-f47ba814-49a2-603f-3595-275284140251",
    "GPU-00000000-1111-2222-3333-444444444444",
)


def fake_nvidia_smi(free_mib: list[int]) -> Any:
    """``core.gpu._run`` answering for ``len(free_mib)`` cards - the real parsers run."""

    def run(args: list[str]) -> str:
        if args[1] == "-L":
            return "\n".join(f"GPU {i}: Card {i} (UUID: {UUIDS[i]})" for i in range(len(free_mib)))
        return "\n".join(f"{UUIDS[i]}, {mib}" for i, mib in enumerate(free_mib))

    return run


class TestEveryVisibleCardGetsItsOwnVerdict:
    @pytest.mark.parametrize("count", [1, 2, 3])
    def test_auto_lists_every_card(
        self, client: TestClient, rooted: Path, monkeypatch: pytest.MonkeyPatch, count: int
    ) -> None:
        # The first card is nearly full, every other one has room.
        free = [1] + [24_000] * (count - 1)
        monkeypatch.setattr("core.gpu._run", fake_nvidia_smi(free))
        body = estimate(client, model_source={"mode": "load", "path": str(rooted)}).json()
        assert [card["uuid"] for card in body["cards"]] == list(UUIDS[:count])
        assert body["cards"][0]["verdict"] == "exceeds"
        assert all(card["verdict"] == "fits" for card in body["cards"][1:])
        assert body["cards"][0]["free_bytes"] == 1024 * 1024
        # `auto` resolves to the card with the most free memory.
        assert body["gpu_uuid"] == UUIDS[1 if count > 1 else 0]

    def test_a_named_card_is_the_only_one_judged(
        self, client: TestClient, rooted: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr("core.gpu._run", fake_nvidia_smi([24_000, 1]))
        body = estimate(
            client, model_source={"mode": "load", "path": str(rooted)}, gpu_request=UUIDS[1]
        ).json()
        assert [card["uuid"] for card in body["cards"]] == [UUIDS[1]]
        assert body["cards"][0]["verdict"] == "exceeds"
        assert body["verdict"] == "exceeds"

    def test_no_readable_driver_lists_no_card_and_says_unknown(
        self, client: TestClient, rooted: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from core.errors import refuse

        def absent(args: list[str]) -> str:
            raise refuse("nvidia_smi_unavailable", "absent")

        monkeypatch.setattr("core.gpu._run", absent)
        body = estimate(client, model_source={"mode": "load", "path": str(rooted)}).json()
        assert body["cards"] == []
        assert body["verdict"] == "unknown"


# ----------------------------------------------------------------------------------------- Hub


class FakeHub:
    """The ``huggingface_hub`` functions the preview uses, answering from a real directory.

    ``get_safetensors_metadata`` answers with the real shard headers, in the shape the Hub
    client returns them (files -> tensors -> ``shape`` / ``dtype``). ``snapshot_download`` is
    present so that a download would be *recorded*, not crash: the test asserts it never ran.
    """

    def __init__(self, directory: Path, *, not_safetensors: bool = False) -> None:
        self.directory = directory
        self.not_safetensors = not_safetensors
        self.calls: list[tuple[str, str, str]] = []

    def repo_info(self, repo_id: str, revision: str) -> Any:
        self.calls.append(("repo_info", repo_id, revision))
        return SimpleNamespace(sha=COMMIT)

    def hf_hub_download(self, repo_id: str, filename: str, revision: str) -> str:
        self.calls.append(("hf_hub_download", repo_id, revision))
        return str(self.directory / filename)

    def get_safetensors_metadata(self, repo_id: str, revision: str) -> Any:
        self.calls.append(("get_safetensors_metadata", repo_id, revision))
        if self.not_safetensors:
            raise NotASafetensorsRepoError("no safetensors")
        files = {}
        for shard in sorted(self.directory.glob("*.safetensors")):
            header = safetensors_header(shard)
            files[shard.name] = SimpleNamespace(
                tensors={
                    name: SimpleNamespace(dtype=entry["dtype"], shape=entry["shape"])
                    for name, entry in header.items()
                    if name != "__metadata__"
                }
            )
        return SimpleNamespace(files_metadata=files)

    def snapshot_download(self, repo_id: str, revision: str) -> str:
        self.calls.append(("snapshot_download", repo_id, revision))
        return str(self.directory)


class NotASafetensorsRepoError(Exception):
    """Named as ``huggingface_hub.errors`` names it; the reader matches the name."""


def test_the_fake_hub_answers_the_way_the_real_client_is_called() -> None:
    """The fake's signatures against the installed client's, so the fake cannot drift."""
    import inspect

    import huggingface_hub

    for name in ("repo_info", "hf_hub_download", "get_safetensors_metadata"):
        real = inspect.signature(getattr(huggingface_hub, name)).parameters
        for parameter in inspect.signature(getattr(FakeHub, name)).parameters:
            if parameter != "self":
                assert parameter in real, (name, parameter)


# ------------------------------------------------------------------------- the frontend's copy

FRONTEND_FIXTURE = (
    Path(__file__).resolve().parents[3]
    / "frontend"
    / "src"
    / "components"
    / "panels"
    / "newrun"
    / "fixtures"
    / "budget-estimate.json"
)


def shape(value: Any) -> Any:
    """Keys all the way down; a list by its first item. Values are not compared."""
    if isinstance(value, dict):
        return {key: shape(item) for key, item in sorted(value.items())}
    if isinstance(value, list):
        return [shape(value[0])] if value else []
    return type(value).__name__ if value is not None else "null"


def test_the_frontend_fixture_has_the_shape_the_route_serves(
    client: TestClient, rooted: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The New run form's tests render ``budget-estimate.json``. Until 2026-10-20 those
    tests rendered ``estimate_bytes`` and ``terms`` the route had never served - a fixture
    agreeing with the screen by construction. The fixture was written from this route
    (directory source, two cards); this pins that it still is."""
    import json

    monkeypatch.setattr("core.gpu._run", fake_nvidia_smi([1, 24_000]))
    live = estimate(client, model_source={"mode": "load", "path": str(rooted)}).json()
    fixture = json.loads(FRONTEND_FIXTURE.read_text(encoding="utf-8"))
    assert shape(fixture) == shape(live)


def test_the_counting_rule_skips_scalars_and_multiplies_every_dimension() -> None:
    """The one rule both a local directory and a Hub repository count through. A scalar
    buffer is not a parameter; a 3-D tensor is the product of all three dimensions."""
    from training.acquisition.local_dir import param_count_from_shapes

    assert param_count_from_shapes([[2, 3], [], [4], [2, 3, 5], None]) == 6 + 4 + 30
