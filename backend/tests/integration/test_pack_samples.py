"""The try-before-committing panels  (FR-007.19, R-02.39; steps 2, 3, 5 and 6).

R-02.39: *"Each step that can be tried before it is committed is tried."* Step 4's panel is
the dry-run and has its own file; these are the others - a sample of the rows at steps 2 and
3, a **scored** sample at step 5, and the **current** score against the proposed floor at
step 6.

Two properties are asserted that a weaker test would miss:

* **The sample is seeded and stratified, through F4's own sampler.** Re-opening a step shows
  the same rows - a sample that changed on every view would read as the data changing - and a
  head slice would move when the file's order changed and nothing else did, which is why F4
  refuses to truncate.
* **Every number names its scale and sample** (ADR-012): ``n`` arrives with ``population``,
  so "5 of 15 prompts" is expressible and a bare "5" is not.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from api.app import create_app
from api.deps import session_dependency
from packs import authoring, dryrun
from packs.settings import reset_pack_settings
from packs.slots import SlotKind
from providers.fakes import FakeProvider
from tests.support import pack_fixtures as fx
from tests.support.db import migrated_engine, session_for

RULE = {"kind": "rule", "name": "len", "config_sha256": None, "expression": '{"min_length": 4}'}


@pytest.fixture
def packs_root(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("PACKS_ROOT", str(tmp_path / "packs"))
    monkeypatch.delenv("MIDATAWORKS_BASE_URL", raising=False)
    reset_pack_settings()
    yield tmp_path / "packs"
    reset_pack_settings()


@pytest.fixture
def engine(tmp_path: Path, packs_root: Path):
    engine = migrated_engine(tmp_path / "db")
    yield engine
    engine.dispose()


@pytest.fixture
def provider(monkeypatch: pytest.MonkeyPatch) -> FakeProvider:
    """A stub at the provider interface - no GPU, no endpoint (ADR-009, ADR-023)."""
    fake = FakeProvider(
        kind="openai_compatible",
        model="stub-1",
        base_url="http://stub.test",
        # FIVE answers for a five-prompt sample: `FakeModelHandle` refuses to wrap a
        # mis-sized fixture, which is right - "a wrapped answer makes a mis-sized fixture
        # look like a working pass."
        answers=[
            "a long enough answer",
            "another long answer",
            "a third long answer",
            "no",
            "a fifth long answer",
        ],
    )
    from providers.registry import ProviderRow

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


@pytest.fixture
def draft(engine, tmp_path: Path):
    """A draft with its prompt set, test set and retention set filled by real parts."""
    slots: dict[str, object] = {}
    with session_for(engine) as session:
        for slot, origin, digest in (
            (SlotKind.PROMPT_SET, "/data/prompts.jsonl", "a"),
            (SlotKind.TEST_SET, "/data/test.jsonl", "b"),
            (SlotKind.RETENTION_SET, "/data/retain.jsonl", "c"),
        ):
            _part_id, provenance = fx.import_external_part(
                session,
                parts_root=tmp_path / "parts",
                export_root=tmp_path / f"export-{slot.value}",
                origin=origin,
                content_sha256=digest * 64,
                rows=5,
            )
            slots[slot.value] = fx.external_slot(
                origin=provenance["source"],
                content_sha256=provenance["content_sha256"],
                retrieved_at=provenance["retrieved_at"],
            )

    created = authoring.create(name="notary-md", domain="maryland notary")
    authoring.commit_step(
        created.draft_id, 2, slots={SlotKind.PROMPT_SET.value: slots[SlotKind.PROMPT_SET.value]}
    )
    authoring.commit_step(
        created.draft_id, 5, slots={SlotKind.TEST_SET.value: slots[SlotKind.TEST_SET.value]}
    )
    retention = dict(slots[SlotKind.RETENTION_SET.value])  # type: ignore[arg-type]
    retention["floor"] = 0.6
    return authoring.commit_step(
        created.draft_id, 6, slots={SlotKind.RETENTION_SET.value: retention}
    )


@pytest.fixture
def client(engine):
    app = create_app()

    def override():
        with session_for(engine) as session:
            yield session

    app.dependency_overrides[session_dependency] = override
    with TestClient(app, raise_server_exceptions=False) as client:
        yield client


class TestTheRowSamplePanels:
    def test_a_slot_sample_names_its_scale_and_its_sample(self, client, draft) -> None:
        response = client.post(
            f"/api/v1/pack-drafts/{draft.draft_id}/samples/{SlotKind.PROMPT_SET.value}"
        )
        assert response.status_code == 200, response.text
        body = response.json()
        assert body["n"] == 5
        assert body["population"] >= body["n"]
        assert body["seed"] == 0
        assert len(body["prompts"]) == body["n"]
        assert all(item["prompt"] for item in body["prompts"])

    def test_the_sample_is_the_same_on_a_second_view(self, client, draft) -> None:
        """Seeded, so re-opening the step is not the data changing under the author."""
        path = f"/api/v1/pack-drafts/{draft.draft_id}/samples/{SlotKind.PROMPT_SET.value}"
        first = client.post(path).json()
        second = client.post(path).json()
        assert first["prompts"] == second["prompts"]

    def test_an_empty_slot_says_so_by_name_rather_than_showing_nothing(self, client, draft) -> None:
        """A panel that simply showed nothing is indistinguishable from an empty set."""
        response = client.post(
            f"/api/v1/pack-drafts/{draft.draft_id}/samples/{SlotKind.CORPUS.value}"
        )
        assert response.status_code == 422, response.text
        error = response.json()["error"]
        assert error["code"] == "pack_part_unresolved"
        assert error["details"]["slot"] == "corpus"

    def test_an_unknown_slot_is_refused_by_the_route(self, client, draft) -> None:
        response = client.post(f"/api/v1/pack-drafts/{draft.draft_id}/samples/not_a_slot")
        assert response.status_code == 422
        assert response.json()["error"]["code"] == "validation_failed"


class TestTheScoredPanels:
    def test_step_five_scores_a_sample_and_names_the_provider(
        self, client, draft, provider
    ) -> None:
        response = client.post(
            f"/api/v1/pack-drafts/{draft.draft_id}/samples/{SlotKind.TEST_SET.value}/score",
            json={"config": RULE},
        )
        assert response.status_code == 200, response.text
        body = response.json()
        assert 0.0 <= body["score"] <= 1.0
        # The provider and the aggregation method travel with the number: a score whose
        # instrument and method are unnamed is not comparable with any other score.
        assert body["provider"] == "openai_compatible:stub-1@http://stub.test"
        assert body["aggregation"] == "mean"
        assert body["n"] == 5
        assert body["population"] >= body["n"]
        assert body["floor"] is None
        assert body["floor_breached"] is None

    def test_step_six_says_whether_the_proposed_floor_is_already_breached(
        self, client, draft, provider
    ) -> None:
        """The one thing an author cannot guess, and the reason the panel exists."""
        high = client.post(
            f"/api/v1/pack-drafts/{draft.draft_id}/samples/{SlotKind.RETENTION_SET.value}/score",
            json={"config": RULE, "floor": 1.0},
        ).json()
        assert high["floor"] == 1.0
        assert high["floor_breached"] is (high["score"] < 1.0)

        low = client.post(
            f"/api/v1/pack-drafts/{draft.draft_id}/samples/{SlotKind.RETENTION_SET.value}/score",
            json={"config": RULE, "floor": 0.0},
        ).json()
        assert low["floor_breached"] is False

    def test_a_missing_floor_has_no_breach_answer_rather_than_false(
        self, client, draft, provider
    ) -> None:
        """``False`` would read as *"it is fine"*; there is no answer without a floor."""
        body = client.post(
            f"/api/v1/pack-drafts/{draft.draft_id}/samples/{SlotKind.TEST_SET.value}/score",
            json={"config": RULE},
        ).json()
        assert body["floor"] is None
        assert body["floor_breached"] is None

    def test_an_asynchronous_expectation_is_refused_by_f4_rather_than_scored(
        self, client, draft, provider
    ) -> None:
        """F4's rule, unchanged: an ``undecided`` verdict has **no** score.

        A human-review configuration cannot score a set - *"refusing rather than counting an
        undecided as a..."* - and F7 does not soften that into a partial number.
        """
        response = client.post(
            f"/api/v1/pack-drafts/{draft.draft_id}/samples/{SlotKind.TEST_SET.value}/score",
            json={
                "config": {"kind": "human_review", "name": "p", "config_sha256": None, "queue": "q"}
            },
        )
        assert response.status_code >= 400
        assert response.json()["error"]["code"] == "expectation_not_synchronous"

    def test_no_provider_refuses_by_name_and_the_step_still_saves(
        self, client, draft, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(dryrun, "list_rows", lambda: [])
        response = client.post(
            f"/api/v1/pack-drafts/{draft.draft_id}/samples/{SlotKind.TEST_SET.value}/score",
            json={"config": RULE},
        )
        assert response.status_code == 503
        assert response.json()["error"]["code"] == "provider_not_available"
        assert (
            client.put(
                f"/api/v1/pack-drafts/{draft.draft_id}/steps/5",
                json={"slots": {}},
            ).status_code
            == 200
        )
