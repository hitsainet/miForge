"""The authoring dry-run  (FR-007.20, FR-007.21, FAC-3 = BRD AC11, E6, E7).

AC11: *a selector dry-run during pack authoring answers a sample of the pack's own prompts
through the configured provider and shows a verdict with a reason for each, **before the step
is saved*** - with the accept rate as a count over the sample and the provider named.

Three things are asserted that a weaker test would miss:

* **Three counts, not two.** F3's verdict is three-valued, so the panel's record carries
  ``accept`` / ``reject`` / ``undecided`` with their denominator. *"Accepted 3 of 5"* is
  incomplete when anything is undecided, and a panel folding ``undecided`` into either other
  count is lying about the configuration being tried.
* **A human-review configuration dry-runs to ``undecided`` throughout, and that is the
  result.** Asserted on the miDataworks client's **call count being zero**, which is how F10
  asserts it from its own side: a panel full of ``undecided`` is where the queue is most
  tempting and where it is most clearly forbidden.
* **A degenerate selector is shown, not blocked** (E6). *"A selector that accepts everything
  is not selecting; one that rejects everything will stall the run"* - and discovering that
  here is the whole point of the requirement.

The provider is stubbed at ``providers/`` (ADR-009, ADR-023), so this file needs no GPU and
carries no ``gpu`` marker. *A GPU-needing test here would mean something was wired to the
training loop that should not be.*
"""

from __future__ import annotations

from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from api.app import create_app
from api.deps import session_dependency
from packs import dryrun
from packs.errors import PROVIDER_NOT_AVAILABLE, PackRefusal
from packs.settings import reset_pack_settings
from packs.slots import SlotKind
from providers.fakes import FakeProvider
from services.selectors.plugins import human_review
from tests.support import pack_fixtures as fx
from tests.support.db import migrated_engine, session_for
from tests.support.selector_stubs import StubReviewQueue

#: A rule selector that accepts anything with four characters, so a stubbed provider's
#: answers produce a **mixture** of verdicts rather than one value.
RULE = {"kind": "rule", "name": "len", "config_sha256": None, "expression": '{"min_length": 12}'}

#: The human-review type: it answers from miDataworks' queue, which this feature does not
#: have, so a dry-run of it is `undecided` throughout.
HUMAN_REVIEW = {
    "kind": "human_review",
    "name": "a-person",
    "config_sha256": None,
    # `queue` names miDataworks' queue. The field's presence here is the point: F7 does not
    # *have* a queue, so a dry-run of this configuration reaches nothing and answers
    # `undecided` - which is the correct result, not a failure.
    "queue": "q-1",
}


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
    """A stub at the **provider interface**, never at an HTTP client (ADR-009, ADR-016).

    Answers alternate long and short so a rule selector produces a mixture - a single-valued
    sample would make the three-count assertions pass for the wrong reason.
    """
    fake = FakeProvider(
        kind="openai_compatible",
        model="stub-1",
        base_url="http://stub.test",
        answers=["a long enough answer", "no", "another long answer", "x", "longer answer!!"],
    )
    monkeypatch.setattr(dryrun, "list_rows", lambda: [_row(fake)])
    monkeypatch.setattr(dryrun, "get_provider", lambda name: fake)
    return fake


def _row(fake: FakeProvider):
    from providers.registry import ProviderRow

    return ProviderRow(
        name="stub",
        kind=fake.identity.kind,
        model=fake.identity.model,
        base_url=fake.identity.base_url,
        capabilities=frozenset({"generate"}),
        available=True,
        unavailable_reason=None,
        judge_eligible=True,
    )


@pytest.fixture
def review_queue():
    """miDataworks' review queue, **stubbed at F3's port** rather than at an HTTP client.

    Needed for the all-``undecided`` case to be reachable at all: with miDataworks
    unconfigured F3's ``human_review`` type refuses ``midataworks_not_configured`` at
    resolution - it cannot post a candidate, so there is nothing to be undecided *about*.

    **That is a clarification to FR-007.20**, which says a human-review configuration
    "dry-runs to ``undecided`` for every candidate" without saying that it presupposes the
    service. Both behaviours are correct and E7 covers both: configured, the panel is all
    undecided with a note; absent, the dry-run reports the service by name - and either way
    the step saves.
    """
    stub = StubReviewQueue()
    human_review.bind_review_queue(stub)
    yield stub
    human_review.bind_review_queue(None)


@pytest.fixture
def draft(engine, tmp_path: Path):
    """A draft whose prompt set is a real imported part, so the sample is the pack's own."""
    with session_for(engine) as session:
        _part_id, provenance = fx.import_external_part(
            session,
            parts_root=tmp_path / "parts",
            export_root=tmp_path / "export",
            origin="/data/operator/prompts.jsonl",
            rows=5,
        )
    client_draft = fx.external_slot(
        origin=provenance["source"],
        content_sha256=provenance["content_sha256"],
        retrieved_at=provenance["retrieved_at"],
    )
    from packs import authoring

    created = authoring.create(name="notary-md", domain="maryland notary")
    return authoring.commit_step(
        created.draft_id, 2, slots={SlotKind.PROMPT_SET.value: client_draft}
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


class TestAcceptanceCriterionEleven:
    def test_every_prompt_gets_a_verdict_with_a_reason_before_the_step_is_saved(
        self, client, draft, provider
    ) -> None:
        response = client.post(
            f"/api/v1/pack-drafts/{draft.draft_id}/dry-run", json={"config": RULE}
        )
        assert response.status_code == 200, response.text
        body = response.json()
        assert len(body["verdicts"]) == 5
        for verdict in body["verdicts"]:
            assert verdict["prompt"]
            assert verdict["completion"]
            assert verdict["verdict"] in {"accept", "reject", "undecided"}
            assert verdict["reason"], "a verdict with no reason is half the requirement"
            assert verdict["reason_code"]
        # Before the step is saved: step 4 has not been committed, and the draft is still at
        # the step it was.
        assert client.get(f"/api/v1/pack-drafts/{draft.draft_id}").json()["current_step"] == 3

    def test_the_sample_is_the_packs_own_prompts(self, client, draft, provider, engine) -> None:
        """AC11 says *whose* prompts: a dry-run on arbitrary prompts measures nothing.

        Asserted against the prompt set's **own rows**, read back through the same loader the
        panel uses, rather than against a string the fixture happens to have written. A
        sample drawn from anywhere else would still look like five prompts.
        """
        from packs import samples
        from packs.authoring import definition_of, get

        body = client.post(
            f"/api/v1/pack-drafts/{draft.draft_id}/dry-run", json={"config": RULE}
        ).json()
        sampled = {verdict["prompt"] for verdict in body["verdicts"]}
        with session_for(engine) as session:
            available = set(
                samples.sample_prompts(session, definition_of(get(draft.draft_id)), n=1000).prompts
            )
        assert sampled <= available
        assert sampled

    def test_the_provider_that_answered_is_named(self, client, draft, provider) -> None:
        """ADR-009: every score records its provider; the panel names the instrument."""
        body = client.post(
            f"/api/v1/pack-drafts/{draft.draft_id}/dry-run", json={"config": RULE}
        ).json()
        assert body["provider"]["kind"] == "openai_compatible"
        assert body["provider"]["model"] == "stub-1"
        assert body["provider"]["base_url"] == "http://stub.test"

    def test_the_result_is_advisory_and_cached_on_the_draft_for_display(
        self, client, draft, provider
    ) -> None:
        client.post(f"/api/v1/pack-drafts/{draft.draft_id}/dry-run", json={"config": RULE})
        cached = client.get(f"/api/v1/pack-drafts/{draft.draft_id}").json()["last_dry_run"]
        assert cached["advisory"] is True

    def test_validity_is_unchanged_by_any_dry_run_outcome(self, client, draft, provider) -> None:
        """Task 10.4: the dry-run is **never** an input to validity."""
        before = client.get(f"/api/v1/pack-drafts/{draft.draft_id}/validity").json()["validity"]
        client.post(f"/api/v1/pack-drafts/{draft.draft_id}/dry-run", json={"config": RULE})
        after = client.get(f"/api/v1/pack-drafts/{draft.draft_id}/validity").json()["validity"]
        assert before == after


class TestThreeCountsNotTwo:
    def test_the_health_record_carries_all_three_counts_and_the_denominator(
        self, client, draft, provider
    ) -> None:
        body = client.post(
            f"/api/v1/pack-drafts/{draft.draft_id}/dry-run", json={"config": RULE}
        ).json()
        health = body["health"]
        for key in ("n", "accepted", "rejected", "undecided", "acceptance_rate"):
            assert key in health, key
        assert health["n"] == 5
        assert health["accepted"] + health["rejected"] + health["undecided"] == health["n"]

    def test_the_payload_holds_no_second_copy_of_the_counts(self, client, draft, provider) -> None:
        """*Two places computing an acceptance rate is two answers* - F3 built the record.

        The FTDD's own correction: the response *"carries no separate ``accepted``/``sample_n``
        pair"*, which an earlier two-valued draft of it had.
        """
        body = client.post(
            f"/api/v1/pack-drafts/{draft.draft_id}/dry-run", json={"config": RULE}
        ).json()
        assert "accepted" not in body
        assert "sample_n" not in body
        assert "accept_rate" not in body

    def test_admitted_comes_from_is_admitted_and_not_from_not_rejected(
        self, client, draft, provider, review_queue
    ) -> None:
        """Never ``verdict != "reject"`` as a stand-in - that is the mutation that admits
        an undecided candidate, and F3 warns the lie runs in both directions."""
        body = client.post(
            f"/api/v1/pack-drafts/{draft.draft_id}/dry-run", json={"config": HUMAN_REVIEW}
        ).json()
        assert all(verdict["verdict"] == "undecided" for verdict in body["verdicts"])
        assert all(verdict["admitted"] is False for verdict in body["verdicts"])
        assert body["health"]["accepted"] == 0
        assert body["health"]["undecided"] == body["health"]["n"]


class TestTheHumanReviewCaseIsNotAQueue:
    def test_with_midataworks_absent_the_dry_run_names_the_service_and_the_step_saves(
        self, client, draft, provider
    ) -> None:
        """No ``review_queue`` fixture here, deliberately: this is the standalone case.

        F3's type refuses ``midataworks_not_configured`` because it cannot post a candidate,
        and F7 surfaces that refusal unchanged. **Nothing is wrong with the pack**, nothing
        falls back to ``accept``, and the step still saves (E7).
        """
        response = client.post(
            f"/api/v1/pack-drafts/{draft.draft_id}/dry-run", json={"config": HUMAN_REVIEW}
        )
        assert response.status_code == 503, response.text
        assert response.json()["error"]["code"] == "midataworks_not_configured"
        saved = client.put(f"/api/v1/pack-drafts/{draft.draft_id}/steps/4", json={"skip": True})
        assert saved.status_code == 200

    def test_it_comes_back_all_undecided_with_a_note_naming_why(
        self, client, draft, provider, review_queue
    ) -> None:
        body = client.post(
            f"/api/v1/pack-drafts/{draft.draft_id}/dry-run", json={"config": HUMAN_REVIEW}
        ).json()
        assert body["asynchronous_note"]
        assert "review queue" in body["asynchronous_note"]
        assert "miDataworks" in body["asynchronous_note"]

    def test_the_step_still_saves_afterwards(self, client, draft, provider, review_queue) -> None:
        client.post(f"/api/v1/pack-drafts/{draft.draft_id}/dry-run", json={"config": HUMAN_REVIEW})
        saved = client.put(f"/api/v1/pack-drafts/{draft.draft_id}/steps/4", json={"skip": True})
        assert saved.status_code == 200, saved.text

    def test_it_never_reaches_midataworks_asserted_on_the_call_count(
        self, client, draft, provider, review_queue, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """**The call count is zero**, which is how F10 asserts the same thing from its side.

        A test that only checked "no queue row was written" would pass against a client call
        that failed; this counts the calls.
        """
        import clients.midataworks as midataworks_client

        calls: list[object] = []
        for name in dir(midataworks_client):
            attribute = getattr(midataworks_client, name)
            if callable(attribute) and not name.startswith("_"):
                monkeypatch.setattr(
                    midataworks_client,
                    name,
                    lambda *a, _n=name, **k: calls.append(_n),
                    raising=False,
                )
        client.post(f"/api/v1/pack-drafts/{draft.draft_id}/dry-run", json={"config": HUMAN_REVIEW})
        assert calls == []


class TestDegenerateSelectorsAreShownNotBlocked:
    @pytest.mark.parametrize("threshold", [0, 10_000])
    def test_accepting_everything_and_rejecting_everything_both_return(
        self, client, draft, provider, threshold: int
    ) -> None:
        """E6: *discovering it here is the point.* The result is advice, not a refusal."""
        config = {**RULE, "expression": f'{{"min_length": {threshold}}}'}
        response = client.post(
            f"/api/v1/pack-drafts/{draft.draft_id}/dry-run", json={"config": config}
        )
        assert response.status_code == 200, response.text
        health = response.json()["health"]
        assert health["accepted"] in (0, health["n"])

    def test_f3s_own_signal_is_carried_rather_than_re_detected(
        self, client, draft, provider
    ) -> None:
        """The signals are F3's to raise and name, not F7's to detect.

        A UI keyed on prose breaks on a wording change, so the panel renders the named signal
        and F3's fixed sentence.
        """
        config = {**RULE, "expression": '{"min_length": 0}'}
        body = client.post(
            f"/api/v1/pack-drafts/{draft.draft_id}/dry-run", json={"config": config}
        ).json()
        names = [signal["name"] for signal in body["health"]["signals"]]
        assert "selection_accepts_everything" in names
        assert all(signal["message"] for signal in body["health"]["signals"])


class TestNoProviderIsNotAWall:
    def test_it_refuses_by_name_and_the_step_still_saves(
        self, client, draft, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """E7. *A dry-run is advice; advice that becomes a wall in front of authoring is a
        worse failure than no advice.*"""
        monkeypatch.setattr(dryrun, "list_rows", lambda: [])
        response = client.post(
            f"/api/v1/pack-drafts/{draft.draft_id}/dry-run", json={"config": RULE}
        )
        assert response.status_code == 503, response.text
        assert response.json()["error"]["code"] == PROVIDER_NOT_AVAILABLE
        assert response.json()["error"]["details"]["capability"] == "dry_run"

        saved = client.put(f"/api/v1/pack-drafts/{draft.draft_id}/steps/4", json={"skip": True})
        assert saved.status_code == 200

    def test_the_refusal_lists_every_row_with_its_reason(self, monkeypatch) -> None:
        """An option missing from a response is indistinguishable from one that is broken."""
        from providers.registry import ProviderRow

        monkeypatch.setattr(
            dryrun,
            "list_rows",
            lambda: [
                ProviderRow(
                    name="in_process",
                    kind="in_process",
                    model=None,
                    base_url=None,
                    capabilities=frozenset(),
                    available=False,
                    unavailable_reason="this process holds no parameters",
                    judge_eligible=False,
                )
            ],
        )
        with pytest.raises(PackRefusal) as raised:
            dryrun.dry_run(["p"], RULE)
        assert raised.value.details["providers"][0]["name"] == "in_process"
        assert raised.value.details["providers"][0]["unavailable_reason"]


class TestTheDryRunIsSynchronous:
    def test_this_feature_registers_no_celery_task(self) -> None:
        """FR-007.21: *if this ever needs a worker, shrink the sample; do not add a queue.*"""
        from workers.celery_app import celery_app

        registered = [name for name in celery_app.tasks if name.startswith("miforge.packs")]
        assert registered == []

    def test_no_packs_module_imports_celery(self) -> None:
        import packs

        source_dir = Path(packs.__file__).resolve().parent
        for path in source_dir.glob("*.py"):
            text = path.read_text("utf-8")
            assert "import celery" not in text
            assert "shared_task" not in text
