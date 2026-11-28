"""``api.v1.endpoints.packs_drafts``'s routes answer as the contract says (021 FR-021.13).

The draft is authored through the routes themselves, its three required slots filled by
external parts imported through F10's real ingest path (five rows each, so the panels sample
real rows). The dry-run and scored panels need a provider: it is stubbed at the provider
interface, at the seam ``test_dry_run.py`` and ``test_pack_samples.py`` stub (``packs.dryrun``'s
``list_rows`` and ``get_provider``), with a real :class:`providers.registry.ProviderRow`.
"""

from __future__ import annotations

from typing import Any

import pytest

from packs import dryrun
from packs.slots import SlotKind
from providers.fakes import FakeProvider
from tests.support import pack_fixtures as fx
from tests.support.answer_cases import answered, assert_covered, case
from tests.support.approval_world import AGENT, World

ROUTER = "api.v1.endpoints.packs_drafts"

#: A rule selector; the stub's answers below give a mixture of verdicts against it.
RULE = {"kind": "rule", "name": "len", "config_sha256": None, "expression": '{"min_length": 4}'}


@pytest.fixture
def provider(monkeypatch: pytest.MonkeyPatch) -> FakeProvider:
    """Five answers for a five-prompt sample (``FakeModelHandle`` refuses a mis-sized one)."""
    from providers.registry import ProviderRow

    fake = FakeProvider(
        kind="openai_compatible",
        model="stub-1",
        base_url="http://stub.test",
        answers=["a long enough answer", "no", "another long answer", "x", "a fifth answer"],
    )
    row = ProviderRow(
        name="stub",
        kind="openai_compatible",
        model="stub-1",
        base_url="http://stub.test",
        capabilities=frozenset({"generate"}),
        available=True,
        unavailable_reason=None,
        judge_eligible=True,
    )
    monkeypatch.setattr(dryrun, "list_rows", lambda: [row])
    monkeypatch.setattr(dryrun, "get_provider", lambda name: fake)
    return fake


def _slots(world: World) -> dict[str, dict[str, Any]]:
    """The three required slot documents, each pinning a real imported external part."""
    slots: dict[str, dict[str, Any]] = {}
    with world.session() as session:
        for slot, origin, digest in (
            (SlotKind.PROMPT_SET, "/data/prompts.jsonl", "a"),
            (SlotKind.TEST_SET, "/data/test.jsonl", "b"),
            (SlotKind.RETENTION_SET, "/data/retain.jsonl", "c"),
        ):
            _part_id, provenance = fx.import_external_part(
                session,
                parts_root=world.root / "parts",
                export_root=world.root / f"export-{slot.value}",
                origin=origin,
                content_sha256=digest * 64,
                rows=5,
            )
            slots[slot.value] = fx.external_slot(
                origin=provenance["source"],
                content_sha256=provenance["content_sha256"],
                retrieved_at=provenance["retrieved_at"],
            )
    return slots


def _put(world: World, draft_id: str, number: int, body: dict[str, Any]) -> Any:
    return world.client.put(f"/api/v1/pack-drafts/{draft_id}/steps/{number}", json=body)


def _draft(world: World) -> str:
    """A complete draft (every step answered; 3 and 4 skipped), authored over the routes."""
    slots = _slots(world)
    created = world.client.post("/api/v1/pack-drafts", json={})
    assert created.status_code == 201, created.text
    draft_id = str(created.json()["draft_id"])
    retention = {**slots[SlotKind.RETENTION_SET.value], "floor": 0.6}
    for number, body in (
        (1, {"name": "notary-md", "domain": "maryland notary", "domain_given": True}),
        (2, {"slots": {SlotKind.PROMPT_SET.value: slots[SlotKind.PROMPT_SET.value]}}),
        (3, {"skip": True}),
        (4, {"skip": True}),
        (5, {"slots": {SlotKind.TEST_SET.value: slots[SlotKind.TEST_SET.value]}}),
        (6, {"slots": {SlotKind.RETENTION_SET.value: retention}}),
    ):
        response = _put(world, draft_id, number, body)
        assert response.status_code == 200, response.text
    return draft_id


@case(ROUTER, "POST /api/v1/pack-drafts")
def test_create_draft(world: World) -> None:
    response = world.client.post(
        "/api/v1/pack-drafts", json={"name": "notary-md", "domain": "maryland notary"}
    )
    answered("POST /api/v1/pack-drafts", response)


@case(ROUTER, "GET /api/v1/pack-drafts")
def test_list_drafts(world: World) -> None:
    _draft(world)
    body = answered("GET /api/v1/pack-drafts", world.client.get("/api/v1/pack-drafts"))
    assert body["drafts"]


@case(ROUTER, "GET /api/v1/pack-drafts/{draft_id}")
def test_get_draft(world: World, provider: FakeProvider) -> None:
    """Both shapes: before any dry-run, and with ``last_dry_run`` cached on the draft."""
    draft_id = _draft(world)
    key = "GET /api/v1/pack-drafts/{draft_id}"
    answered(key, world.client.get(f"/api/v1/pack-drafts/{draft_id}"))
    ran = world.client.post(f"/api/v1/pack-drafts/{draft_id}/dry-run", json={"config": RULE})
    assert ran.status_code == 200, ran.text
    body = answered(key, world.client.get(f"/api/v1/pack-drafts/{draft_id}"))
    assert body["last_dry_run"] is not None


@case(ROUTER, "PUT /api/v1/pack-drafts/{draft_id}/steps/{number}")
def test_commit_step(world: World) -> None:
    """A step that names, a step that fills a slot, and a skipped step."""
    slots = _slots(world)
    draft_id = str(world.client.post("/api/v1/pack-drafts", json={}).json()["draft_id"])
    key = "PUT /api/v1/pack-drafts/{draft_id}/steps/{number}"
    for number, body in (
        (1, {"name": "notary-md", "domain": "maryland notary", "domain_given": True}),
        (2, {"slots": {SlotKind.PROMPT_SET.value: slots[SlotKind.PROMPT_SET.value]}}),
        (3, {"skip": True}),
    ):
        answered(key, _put(world, draft_id, number, body))


@case(ROUTER, "POST /api/v1/pack-drafts/{draft_id}/samples/{slot}")
def test_sample_slot(world: World) -> None:
    draft_id = _draft(world)
    response = world.client.post(
        f"/api/v1/pack-drafts/{draft_id}/samples/{SlotKind.PROMPT_SET.value}"
    )
    body = answered("POST /api/v1/pack-drafts/{draft_id}/samples/{slot}", response)
    assert body["prompts"]


@case(ROUTER, "POST /api/v1/pack-drafts/{draft_id}/samples/{slot}/score")
def test_score_sample(world: World, provider: FakeProvider) -> None:
    """Step 6's panel, with a proposed floor, so ``floor_breached`` is answered."""
    draft_id = _draft(world)
    response = world.client.post(
        f"/api/v1/pack-drafts/{draft_id}/samples/{SlotKind.RETENTION_SET.value}/score",
        json={"config": RULE, "floor": 0.6},
    )
    body = answered("POST /api/v1/pack-drafts/{draft_id}/samples/{slot}/score", response)
    assert body["floor_breached"] is not None


@case(ROUTER, "POST /api/v1/pack-drafts/{draft_id}/dry-run")
def test_dry_run(world: World, provider: FakeProvider) -> None:
    draft_id = _draft(world)
    response = world.client.post(f"/api/v1/pack-drafts/{draft_id}/dry-run", json={"config": RULE})
    body = answered("POST /api/v1/pack-drafts/{draft_id}/dry-run", response)
    assert body["verdicts"]


@case(ROUTER, "GET /api/v1/pack-drafts/{draft_id}/validity")
def test_review_draft(world: World) -> None:
    draft_id = _draft(world)
    body = answered(
        "GET /api/v1/pack-drafts/{draft_id}/validity",
        world.client.get(f"/api/v1/pack-drafts/{draft_id}/validity"),
    )
    assert body["validity"]["state"] == "valid", body


@case(ROUTER, "POST /api/v1/pack-drafts/{draft_id}/commit")
def test_commit_draft(world: World) -> None:
    draft_id = _draft(world)
    answered(
        "POST /api/v1/pack-drafts/{draft_id}/commit",
        world.client.post(f"/api/v1/pack-drafts/{draft_id}/commit"),
    )


@case(ROUTER, "DELETE /api/v1/pack-drafts/{draft_id}")
def test_delete_draft(world: World) -> None:
    """Gated: the agent's call waits (202 pending); the operator's deletes (204, no body)."""
    draft_id = _draft(world)
    key = "DELETE /api/v1/pack-drafts/{draft_id}"
    path = f"/api/v1/pack-drafts/{draft_id}"
    answered(key, world.client.delete(path, headers=AGENT), pending=True)
    response = world.client.delete(path)
    assert response.status_code == 204, response.text
    answered(key, response)


def test_every_route_of_the_router_has_a_case() -> None:
    assert_covered(ROUTER)
