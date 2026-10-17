"""The seven-step flow, over the transport  (FR-007.17 - FR-007.22, FAC-8, E13).

The flow's primary user is the **domain expert** - *"who has material but does not run
infrastructure"* - and this is the only screen in the product whose primary user is not the
operator. Two promises are therefore tested rather than described:

* **"progress is saved as you go"** - the subtitle claims it, so a claimed save must have
  happened. A draft survives a **backend restart**, which here means a brand-new application
  reading the same pack root (FAC-8).
* **A failed commit loses nothing.** Someone two hours into the flow must not lose their work
  to a validation failure, and re-entering a done step must not discard the later ones.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from api.app import create_app
from api.deps import session_dependency
from packs import authoring, store
from packs.settings import reset_pack_settings
from packs.slots import SlotKind
from tests.support import pack_fixtures as fx
from tests.support.db import migrated_engine, session_for


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


def make_client(engine) -> TestClient:
    """A **fresh application** over the same engine and the same pack root.

    Used twice in one test to model a backend restart: nothing in-process carries over, so a
    draft that survives is a draft that was persisted.
    """
    app = create_app()

    def override():
        with session_for(engine) as session:
            yield session

    app.dependency_overrides[session_dependency] = override
    return TestClient(app, raise_server_exceptions=False)


@pytest.fixture
def client(engine):
    with make_client(engine) as client:
        yield client


def external_slot_for(session, tmp_path: Path, origin: str, suffix: str) -> dict[str, object]:
    """An external slot whose part is really in the index, so the pack can be valid."""
    _part_id, provenance = fx.import_external_part(
        session,
        parts_root=tmp_path / "parts",
        export_root=tmp_path / f"export-{suffix}",
        origin=origin,
        content_sha256=f"{suffix[0]}" * 64,
    )
    return fx.external_slot(
        origin=provenance["source"],
        content_sha256=provenance["content_sha256"],
        retrieved_at=provenance["retrieved_at"],
    )


class TestTheSevenSteps:
    def test_the_flow_runs_end_to_end_and_commits(self, engine, client, tmp_path: Path) -> None:
        with session_for(engine) as session:
            prompts = external_slot_for(session, tmp_path, "/data/prompts.jsonl", "aprompt")
            tests = external_slot_for(session, tmp_path, "/data/test.jsonl", "btest")
            retention = external_slot_for(session, tmp_path, "/data/retain.jsonl", "cretain")

        draft_id = client.post("/api/v1/pack-drafts", json={}).json()["draft_id"]

        step1 = client.put(
            f"/api/v1/pack-drafts/{draft_id}/steps/1",
            json={"name": "notary-md", "domain": "maryland notary", "domain_given": True},
        ).json()
        assert step1["current_step"] == 2
        assert step1["subtitle"] == "step 2 of 7 · progress is saved as you go"

        client.put(
            f"/api/v1/pack-drafts/{draft_id}/steps/2",
            json={"slots": {SlotKind.PROMPT_SET.value: prompts}},
        )
        # Steps 3 and 4 are the skippable ones (FR-007.17): an operator with a corpus and no
        # selector is a real case, and R-02.4's plain corpus run needs no selection at all.
        for skipped in (3, 4):
            response = client.put(
                f"/api/v1/pack-drafts/{draft_id}/steps/{skipped}", json={"skip": True}
            )
            assert response.status_code == 200, response.text
        client.put(
            f"/api/v1/pack-drafts/{draft_id}/steps/5",
            json={"slots": {SlotKind.TEST_SET.value: tests}},
        )
        client.put(
            f"/api/v1/pack-drafts/{draft_id}/steps/6",
            json={"slots": {SlotKind.RETENTION_SET.value: {**retention, "floor": 0.72}}},
        )

        review = client.get(f"/api/v1/pack-drafts/{draft_id}/validity").json()
        assert review["validity"]["state"] == "valid", review

        committed = client.post(f"/api/v1/pack-drafts/{draft_id}/commit")
        assert committed.status_code == 201, committed.text
        assert committed.json()["validity"]["state"] == "valid"
        # The draft is gone and the pack is on disk.
        assert client.get(f"/api/v1/pack-drafts/{draft_id}").status_code == 404
        assert store.exists("notary-md", "1")

    def test_a_non_skippable_step_refuses_to_be_skipped(self, client) -> None:
        draft_id = client.post("/api/v1/pack-drafts", json={"name": "x"}).json()["draft_id"]
        response = client.put(f"/api/v1/pack-drafts/{draft_id}/steps/2", json={"skip": True})
        assert response.status_code == 400
        assert response.json()["error"]["code"] == "pack_draft_step_not_skippable"

    def test_a_slot_set_from_the_wrong_step_is_refused(self, client) -> None:
        """An author who believes step 3 set the test set has a different pack from the one
        they think they have - so this refuses rather than accepting it quietly."""
        draft_id = client.post("/api/v1/pack-drafts", json={"name": "x"}).json()["draft_id"]
        response = client.put(
            f"/api/v1/pack-drafts/{draft_id}/steps/2",
            json={"slots": {SlotKind.TEST_SET.value: fx.contract_slot()}},
        )
        assert response.status_code == 400
        assert response.json()["error"]["code"] == "pack_draft_slot_not_in_step"

    def test_an_unknown_step_is_refused_rather_than_clamped(self, client) -> None:
        draft_id = client.post("/api/v1/pack-drafts", json={"name": "x"}).json()["draft_id"]
        response = client.put(f"/api/v1/pack-drafts/{draft_id}/steps/9", json={})
        assert response.status_code == 400
        assert response.json()["error"]["code"] == "pack_draft_step_unknown"

    def test_the_step_table_marks_exactly_steps_three_and_four_skippable(self) -> None:
        skippable = [spec.number for spec in authoring.STEPS if spec.skippable]
        assert skippable == [3, 4]
        assert len(authoring.STEPS) == 7

    def test_steps_one_and_seven_have_no_panel(self) -> None:
        """*A step with nothing to sample has no panel* - R-02.39 read literally."""
        assert authoring.step(1).panel is None
        assert authoring.step(7).panel is None
        assert authoring.step(4).panel == "dry_run"


class TestADraftSurvivesARestart:
    def test_it_resumes_at_the_same_step_with_the_same_answers(
        self, engine, tmp_path: Path
    ) -> None:
        """FAC-8, and E13's resumable half. A **new application** reads the same store."""
        with make_client(engine) as first:
            draft_id = first.post("/api/v1/pack-drafts", json={}).json()["draft_id"]
            first.put(
                f"/api/v1/pack-drafts/{draft_id}/steps/1",
                json={"name": "notary-md", "domain": "maryland notary", "domain_given": True},
            )
            first.put(
                f"/api/v1/pack-drafts/{draft_id}/steps/2",
                json={"slots": {SlotKind.PROMPT_SET.value: fx.contract_slot()}},
            )

        with make_client(engine) as second:
            resumed = second.get(f"/api/v1/pack-drafts/{draft_id}").json()

        assert resumed["name"] == "notary-md"
        assert resumed["domain"] == "maryland notary"
        assert resumed["current_step"] == 3
        assert resumed["step_states"]["1"] == "done"
        assert resumed["step_states"]["2"] == "done"
        assert resumed["slots"][SlotKind.PROMPT_SET.value]["contract_reference"]["name"] == (
            "support-tickets"
        )

    def test_a_draft_is_deletable_at_any_time(self, client) -> None:
        draft_id = client.post("/api/v1/pack-drafts", json={}).json()["draft_id"]
        assert client.delete(f"/api/v1/pack-drafts/{draft_id}").status_code == 204
        assert client.get(f"/api/v1/pack-drafts/{draft_id}").status_code == 404


class TestAFailedCommitLosesNothing:
    def test_an_incomplete_commit_refuses_and_leaves_the_draft_intact(self, client) -> None:
        draft_id = client.post("/api/v1/pack-drafts", json={}).json()["draft_id"]
        client.put(
            f"/api/v1/pack-drafts/{draft_id}/steps/1",
            json={"name": "notary-md", "domain": None, "domain_given": True},
        )
        client.put(
            f"/api/v1/pack-drafts/{draft_id}/steps/2",
            json={"slots": {SlotKind.PROMPT_SET.value: fx.contract_slot()}},
        )
        response = client.post(f"/api/v1/pack-drafts/{draft_id}/commit")
        assert response.status_code == 422, response.text
        error = response.json()["error"]
        assert error["code"] == "pack_incomplete"
        assert error["details"]["missing_slots"] == ["test_set", "retention_set"]

        # Intact: the draft is still there, still at the step it was, with its answers.
        resumed = client.get(f"/api/v1/pack-drafts/{draft_id}").json()
        assert resumed["name"] == "notary-md"
        assert resumed["slots"][SlotKind.PROMPT_SET.value] is not None

    def test_an_unnamed_draft_refuses_with_a_readable_reason(self, client) -> None:
        """The slug pattern would otherwise report this as a regex failure."""
        draft_id = client.post("/api/v1/pack-drafts", json={}).json()["draft_id"]
        response = client.post(f"/api/v1/pack-drafts/{draft_id}/commit")
        assert response.json()["error"]["code"] == "pack_draft_unnamed"

    def test_re_entering_a_done_step_does_not_discard_the_later_ones(self, client) -> None:
        """FR-007.18: an author correcting step 2 must not lose step 6."""
        draft_id = client.post("/api/v1/pack-drafts", json={}).json()["draft_id"]
        client.put(
            f"/api/v1/pack-drafts/{draft_id}/steps/1",
            json={"name": "notary-md", "domain_given": True},
        )
        client.put(
            f"/api/v1/pack-drafts/{draft_id}/steps/2",
            json={"slots": {SlotKind.PROMPT_SET.value: fx.contract_slot()}},
        )
        client.put(
            f"/api/v1/pack-drafts/{draft_id}/steps/6",
            json={
                "slots": {
                    SlotKind.RETENTION_SET.value: fx.retention_slot(floor=0.6),
                }
            },
        )
        # Back to step 2 with a different prompt set.
        again = client.put(
            f"/api/v1/pack-drafts/{draft_id}/steps/2",
            json={
                "slots": {SlotKind.PROMPT_SET.value: fx.contract_slot("prompt-set", "other", "9")}
            },
        ).json()
        assert again["step_states"]["6"] == "done"
        assert again["slots"][SlotKind.RETENTION_SET.value]["floor"] == 0.6
        assert again["slots"][SlotKind.PROMPT_SET.value]["contract_reference"]["name"] == "other"


class TestStepFourSelectsAndDoesNotCompose:
    def test_no_route_accepts_an_authored_configuration_body_for_storage(self) -> None:
        """FR-007.22: rule terms and judge templates are miDataworks', cited by hash.

        Asserted over the **mounted** route table: a configuration arrives to be *run* (the
        dry-run and the scored panel), and there is no route that stores one. A write
        endpoint would be a second authoring surface and a second hash, which R-02.10 forbids.
        """
        app = create_app()
        paths = {
            (method, getattr(route, "path", ""))
            for route in app.routes
            for method in getattr(route, "methods", set())
        }
        for method, path in paths:
            if "selector" in path or "grader" in path or "config" in path:
                assert method in {"GET", "HEAD", "OPTIONS"} or path.endswith(
                    "/evaluate"
                ), f"{method} {path} looks like a configuration write surface"

    def test_the_draft_carries_no_authored_configuration_field(self, client) -> None:
        draft_id = client.post("/api/v1/pack-drafts", json={}).json()["draft_id"]
        payload = client.get(f"/api/v1/pack-drafts/{draft_id}").json()
        for forbidden in ("rule_terms", "required_terms", "forbidden_terms", "judge_template"):
            assert forbidden not in payload


class TestTheDomainIsNullableAndMeant:
    def test_a_null_domain_is_an_assertion_and_survives_the_commit(
        self, engine, client, tmp_path: Path
    ) -> None:
        """Null asserts *not domain-scoped*; an absent key is an omission.

        ``domain_given`` is what keeps the two apart over the wire: a single ``None``
        parameter could not say whether the field was supplied.
        """
        with session_for(engine) as session:
            slot = external_slot_for(session, tmp_path, "/data/general.jsonl", "dgeneral")

        draft_id = client.post("/api/v1/pack-drafts", json={}).json()["draft_id"]
        client.put(
            f"/api/v1/pack-drafts/{draft_id}/steps/1",
            json={"name": "general-pretrain", "domain": None, "domain_given": True},
        )
        for number, slot_kind in (
            (2, SlotKind.PROMPT_SET),
            (5, SlotKind.TEST_SET),
        ):
            client.put(
                f"/api/v1/pack-drafts/{draft_id}/steps/{number}",
                json={"slots": {slot_kind.value: slot}},
            )
        client.put(
            f"/api/v1/pack-drafts/{draft_id}/steps/6",
            json={"slots": {SlotKind.RETENTION_SET.value: {**slot, "floor": 0.5}}},
        )
        committed = client.post(f"/api/v1/pack-drafts/{draft_id}/commit")
        assert committed.status_code == 201, committed.text
        assert committed.json()["domain"] is None
        document = store.read("general-pretrain", "1").document()
        assert "domain" in document and document["domain"] is None
