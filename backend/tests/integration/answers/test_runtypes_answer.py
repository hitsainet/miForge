"""``api.v1.endpoints.runtypes``'s routes answer as the contract says (021 FR-021.13).

Adapters are written through the real adapter save and indexed by the real ``reindex``; the
merge record through ``services.merge_record``. ``POST /adapters/{name}/merge`` is gated
(``gpu_start``): driven as an agent (pending) and as the operator (admitted; the enqueue is the
approval world's recorded one). The budget preview reads the card through ``core.gpu._run``,
stubbed with ``nvidia-smi``'s own output as ``test_picker_listings.py`` does, so the answer
carries a resolved card, its candidates and the ``fits`` branch's ``next_steps``.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from tests.support.answer_cases import answered, assert_covered, case
from tests.support.approval_world import AGENT, World

ROUTER = "api.v1.endpoints.runtypes"
CARD = "GPU-00000000-0000-0000-0000-0000000000a1"
NVIDIA_L = f"GPU 0: Test Card 24GB (UUID: {CARD})\n"
NVIDIA_FREE = f"{CARD}, 20000\n"


def _root(world: World) -> Path:
    root = world.root / "checkpoints"
    root.mkdir(exist_ok=True)
    return root


def _adapter(world: World, *, mergeable: bool = False) -> dict[str, Any]:
    """An adapter saved by the real writer; ``mergeable`` gives it a saved unmerged base."""
    from core.canonical import sha256_of
    from tests.support.checkpoints import saved_checkpoint
    from tests.support.test_adapter_doubles import (
        FakePeftModel,
        adapter_config,
        checkpoint_record,
        hub_record,
        optimizer_block,
    )
    from training.runtypes.adapters import manifest as adapter_manifest
    from training.runtypes.adapters import paths as adapter_paths
    from training.runtypes.adapters import save as adapter_save

    root = _root(world)
    if mergeable:
        base_checkpoint = saved_checkpoint(root, run_id="run-base", step=40)
        base = adapter_manifest.base_block(
            checkpoint_record(checkpoint_name=base_checkpoint.manifest["name"]),
            manifest_sha256=sha256_of(base_checkpoint.manifest),
        )
    else:
        base = adapter_manifest.base_block(hub_record("LiquidAI/LFM2.5-1.2B-Instruct"))
    doc = adapter_manifest.build(
        run_id="run-7",
        step=40,
        base=base,
        adapter=adapter_config(rank=8, alpha=16.0, target_modules=("q_proj",)),
        optimizer=optimizer_block(),
    )
    published: dict[str, Any] = adapter_save.save(
        FakePeftModel(rank=8, alpha=16.0, target_modules=("q_proj",)),
        doc,
        adapter_paths.adapter_dir("run-7", 40, root=root),
    )
    return published


def _reindex(world: World) -> None:
    from services.reindex import reindex

    with world.session() as session:
        reindex(session, root=_root(world))
        session.commit()


@case(ROUTER, "GET /api/v1/run-types")
def test_list_run_types(world: World) -> None:
    answered("GET /api/v1/run-types", world.client.get("/api/v1/run-types"))


@case(ROUTER, "POST /api/v1/budget/estimate")
def test_estimate_budget(world: World, monkeypatch: pytest.MonkeyPatch) -> None:
    """An adapter run on a card with room: the verdict ``fits``, its cards and next steps."""
    from core import gpu
    from tests.support.test_adapter_doubles import TINY_CONFIG

    monkeypatch.setattr(gpu, "_run", lambda args: NVIDIA_L if args[1] == "-L" else NVIDIA_FREE)
    body = answered(
        "POST /api/v1/budget/estimate",
        world.client.post(
            "/api/v1/budget/estimate",
            json={
                "run_type": "adapter",
                "config": {"learning_rate": 0.0001, "warmup_steps": 10, "batch_size": 1},
                "architecture_config": TINY_CONFIG,
                "adapter": {"rank": 8, "alpha": 16.0, "target_modules": ["q_proj"]},
            },
        ),
    )
    assert body["gpu_uuid"] == CARD and body["verdict"] == "fits", body


@case(ROUTER, "GET /api/v1/adapters")
def test_list_adapters(world: World) -> None:
    _adapter(world)
    _reindex(world)
    body = answered("GET /api/v1/adapters", world.client.get("/api/v1/adapters"))
    assert body["count"] == 1


@case(ROUTER, "GET /api/v1/adapters/{name}")
def test_read_adapter(world: World) -> None:
    published = _adapter(world)
    _reindex(world)
    body = answered(
        "GET /api/v1/adapters/{name}", world.client.get(f"/api/v1/adapters/{published['name']}")
    )
    assert body["indexed"] is True


@case(ROUTER, "GET /api/v1/merges")
def test_list_merges(world: World) -> None:
    """One merge of each outcome, through the real record writer."""
    from services import merge_record
    from training.runtypes.adapters.merge import OUTCOME_WRITTEN, MergeOutcome

    with world.session() as session:
        merge_record.record(
            session,
            MergeOutcome(
                outcome=OUTCOME_WRITTEN,
                adapter_name="LiquidAI/LFM2-1.2B:run-a:adapter-1",
                adapter_step=100,
                adapter_manifest_sha256="a" * 64,
                base_checkpoint_name="LiquidAI/LFM2-1.2B:run-a:step-100",
                result_checkpoint_name="LiquidAI/LFM2-1.2B:run-a:step-101",
                refusal_code=None,
            ),
        )
        merge_record.record(
            session,
            merge_record.refused(
                adapter_name="LiquidAI/LFM2-1.2B:run-b:adapter-1",
                adapter_step=100,
                adapter_manifest_sha256="b" * 64,
                base_checkpoint_name="LiquidAI/LFM2-1.2B:run-b:step-100",
                refusal_code="merge_base_not_checkpointed",
            ),
        )
        merge_record.record(
            session,
            merge_record.interrupted(
                adapter_name="LiquidAI/LFM2-1.2B:run-c:adapter-1",
                adapter_step=100,
                adapter_manifest_sha256="c" * 64,
                base_checkpoint_name="LiquidAI/LFM2-1.2B:run-c:step-100",
            ),
        )
        session.commit()
    body = answered("GET /api/v1/merges", world.client.get("/api/v1/merges"))
    assert body["count"] == 3


@case(ROUTER, "POST /api/v1/adapters/{name}/merge")
def test_merge_adapter(world: World) -> None:
    published = _adapter(world, mergeable=True)
    key = "POST /api/v1/adapters/{name}/merge"
    url = f"/api/v1/adapters/{published['name']}/merge"
    params = {"run_id": "run-merge", "step": 41}
    answered(key, world.client.post(url, params=params, headers=AGENT), pending=True)
    assert world.merges == []
    body = answered(key, world.client.post(url, params=params))
    assert body["status"] == "accepted" and world.merges == [(published["name"], "run-merge", 41)]


def test_every_route_of_the_router_has_a_case() -> None:
    assert_covered(ROUTER)
