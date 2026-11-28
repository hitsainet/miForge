"""``api.v1.endpoints.models_local``'s route answers as the contract says (021 FR-021.13).

A placed model directory (config, weights, tokenizer) and a checkpoint written by the real
writer sit under the checkpoint root; the parts root is absent, so ``unreadable_roots`` names it.
"""

from __future__ import annotations

import json

from tests.support.answer_cases import answered, assert_covered, case
from tests.support.approval_world import World

ROUTER = "api.v1.endpoints.models_local"


@case(ROUTER, "GET /api/v1/models/local")
def test_list_local_models(world: World) -> None:
    from tests.support.checkpoints import saved_checkpoint

    root = world.root / "checkpoints"
    base = root / "bases" / "tiny"
    base.mkdir(parents=True)
    (base / "config.json").write_text(json.dumps({"architectures": ["LlamaForCausalLM"]}))
    (base / "model.safetensors").write_bytes(b"\0")
    (base / "tokenizer.json").write_text("{}")
    saved_checkpoint(root, run_id="run-a", step=40)
    body = answered("GET /api/v1/models/local", world.client.get("/api/v1/models/local"))
    assert body["total"] == 2 and body["unreadable_roots"], body


def test_every_route_of_the_router_has_a_case() -> None:
    assert_covered(ROUTER)
