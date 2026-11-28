"""``api.v1.endpoints.tokenizers``'s routes answer as the contract says (021 FR-021.13).

The Hub is replaced at ``training.acquisition.hub._hub`` (every outbound Hub call goes through
it), as ``test_ui_picker_fixtures.py`` does; the route and its reader are real.
"""

from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

from tests.support.answer_cases import answered, assert_covered, case
from tests.support.approval_world import World

ROUTER = "api.v1.endpoints.tokenizers"


@case(ROUTER, "GET /api/v1/tokenizers/local")
def test_list_local_tokenizers(world: World) -> None:
    """A tokenizer-only directory and a model directory, both offered."""
    root = world.root / "checkpoints"
    tokenizer = root / "tokenizers" / "tiny-tok"
    tokenizer.mkdir(parents=True)
    (tokenizer / "tokenizer.json").write_text("{}")
    (tokenizer / "tokenizer_config.json").write_text("{}")
    model = root / "bases" / "tiny"
    model.mkdir(parents=True)
    (model / "config.json").write_text(json.dumps({"model_type": "llama"}))
    (model / "tokenizer.json").write_text("{}")
    body = answered("GET /api/v1/tokenizers/local", world.client.get("/api/v1/tokenizers/local"))
    assert {row["kind"] for row in body["tokenizers"]} == {"tokenizer", "model"}


@case(ROUTER, "GET /api/v1/hub/revision")
def test_check_hub_revision(world: World, monkeypatch: pytest.MonkeyPatch) -> None:
    from training.acquisition import hub as hub_module

    hub_fake = SimpleNamespace(
        repo_info=lambda repo_id, revision, token=None: SimpleNamespace(
            sha="0123456789abcdef" * 2 + "01234567"
        ),
        list_repo_files=lambda repo_id, revision, token=None: [
            "config.json",
            "model.safetensors",
            "tokenizer.json",
            "tokenizer_config.json",
            "README.md",
        ],
    )
    monkeypatch.setattr(hub_module, "_hub", lambda: hub_fake)
    body = answered(
        "GET /api/v1/hub/revision",
        world.client.get("/api/v1/hub/revision", params={"repo": "org/tiny", "revision": "main"}),
    )
    assert body["has_tokenizer"] and body["has_config"] and body["has_weights"]


def test_every_route_of_the_router_has_a_case() -> None:
    assert_covered(ROUTER)
