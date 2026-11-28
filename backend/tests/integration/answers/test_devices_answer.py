"""``api.v1.endpoints.devices``'s route answers as the contract says (021 FR-021.13).

``nvidia-smi`` is replaced at ``core.gpu._run`` with its own output, as
``test_picker_listings.py`` does, so the answer lists cards; the no-driver answer is driven
too (no ``nvidia-smi`` on the path).
"""

from __future__ import annotations

import pytest

from tests.support.answer_cases import answered, assert_covered, case
from tests.support.approval_world import World

ROUTER = "api.v1.endpoints.devices"
CARDS = ("GPU-00000000-0000-0000-0000-0000000000a1", "GPU-00000000-0000-0000-0000-0000000000b2")
NVIDIA_L = "".join(f"GPU {i}: Test Card {i} (UUID: {uuid})\n" for i, uuid in enumerate(CARDS))
NVIDIA_FREE = f"{CARDS[0]}, 20000\n{CARDS[1]}, 9000\n"


@case(ROUTER, "GET /api/v1/gpus")
def test_list_gpus(world: World, monkeypatch: pytest.MonkeyPatch) -> None:
    from core import gpu

    real_run = gpu._run
    monkeypatch.setattr(gpu, "_run", lambda args: NVIDIA_L if args[1] == "-L" else NVIDIA_FREE)
    body = answered("GET /api/v1/gpus", world.client.get("/api/v1/gpus"))
    assert [row["uuid"] for row in body["devices"]] == list(CARDS)
    # No driver: the real `_run` refuses by name, and the route answers it.
    monkeypatch.setattr(gpu, "_run", real_run)
    monkeypatch.setattr(gpu.shutil, "which", lambda name: None)
    body = answered("GET /api/v1/gpus", world.client.get("/api/v1/gpus"))
    assert body["available"] is False


def test_every_route_of_the_router_has_a_case() -> None:
    assert_covered(ROUTER)
