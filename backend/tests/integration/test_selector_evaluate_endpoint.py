"""The evaluate endpoint, as F7's dry-run calls it  (AC-003.4, AC-003.14; FR-003.11, E-16).

**This is the one call F7's dry-run (R-02.39) makes**, so the shape asserted here is the
shape F7 builds against. It is exercised end to end over the **mounted** application, with a
composite of rule AND judge over a stub endpoint, because that is the combination the
authoring flow shows the operator: a verdict with a reason for each of a sample.

F7 owns the sample, the flow and the screen; F3 owns the verdicts and their reasons.
"""

from __future__ import annotations

import json

import pytest
from fastapi.testclient import TestClient

from api.app import create_app
from core.config import reset_settings
from services.selectors import provider_port
from services.selectors.plugins import human_review
from services.selectors.settings import reset_selector_settings
from tests.support.selector_stubs import (
    StubJudgeServer,
    StubProviderResolver,
    StubReviewQueue,
)

TEMPLATE = "Rate.\nQ: {prompt}\nA: {completion}\nReply 'SCORE: <0..1>'."

RULE = {
    "kind": "rule",
    "name": "no-guarantees",
    "config_sha256": None,
    "expression": json.dumps({"forbidden_terms": ["guarantee"], "min_length": 3}),
}
JUDGE = {
    "kind": "judge_model",
    "name": "quality",
    "config_sha256": None,
    "provider": "judge",
    "threshold": 0.6,
}
COMPOSITE = {
    "kind": "composite",
    "name": "house",
    "config_sha256": None,
    "children": ["no-guarantees", "quality"],
    "combine": "all",
}


@pytest.fixture(autouse=True)
def _clean(monkeypatch):
    monkeypatch.setenv("JUDGE_PROMPT_TEMPLATE", TEMPLATE)
    monkeypatch.setenv("JUDGE_PROVIDER_BASE_URL", "http://judge.test:8000")
    monkeypatch.setenv("JUDGE_PROVIDER_MODEL", "judge-7b")
    reset_settings()
    reset_selector_settings()
    yield
    provider_port.bind_provider_resolver(None)
    human_review.bind_review_queue(None)
    reset_selector_settings()
    reset_settings()


@pytest.fixture
def client():
    return TestClient(create_app(), raise_server_exceptions=False)


def post(client, body):
    return client.post("/api/v1/selectors/evaluate", json=body)


class TestTheListing:
    def test_it_serves_the_registry_with_live_availability(self, client) -> None:
        response = client.get("/api/v1/selectors")
        rows = {row["name"]: row for row in response.json()}
        assert set(rows) == {
            "rule",
            "judge_model",
            "reference_match",
            "human_review",
            "composite",
        }
        assert rows["rule"]["available"] is True
        assert rows["rule"]["unavailable_reason"] is None

    def test_an_unavailable_type_is_listed_naming_the_service(self, client) -> None:
        """Never omitted and never a mute disabled control: an option missing from a
        response is indistinguishable from one that is broken (ADR-013)."""
        rows = {row["name"]: row for row in client.get("/api/v1/selectors").json()}
        assert rows["human_review"]["available"] is False
        assert rows["human_review"]["unavailable_reason"] == "midataworks_not_configured"
        assert rows["judge_model"]["unavailable_reason"] == "judge_provider_not_configured"

    def test_availability_is_computed_per_call_rather_than_cached_at_import(self, client) -> None:
        """So an operator who configures a sibling is not told it is still unavailable."""
        before = {row["name"]: row for row in client.get("/api/v1/selectors").json()}
        assert before["human_review"]["available"] is False

        human_review.bind_review_queue(StubReviewQueue())
        after = {row["name"]: row for row in client.get("/api/v1/selectors").json()}
        assert after["human_review"]["available"] is True
        assert after["human_review"]["unavailable_reason"] is None

    def test_each_row_says_what_the_type_decides(self, client) -> None:
        """F9 renders this; a screen with no summary is a list of names."""
        for row in client.get("/api/v1/selectors").json():
            assert row["summary"]
            assert row["requires"] in {"none", "generation_provider", "midataworks"}


class TestTheDryRunCall:
    def test_a_sample_of_five_through_rule_and_judge_with_reasons(self, client) -> None:
        """What the authoring panel shows: a verdict with a reason for each of a sample."""
        server = StubJudgeServer(reply="SCORE: 0.80\nREASON: coherent")
        provider_port.bind_provider_resolver(StubProviderResolver(server.provider()))

        items = [
            {"prompt": f"Q{i}", "completion": f"A sufficiently long answer {i}"} for i in range(4)
        ] + [{"prompt": "Q4", "completion": "we guarantee a refund"}]

        response = post(client, {"config": COMPOSITE, "catalog": [RULE, JUDGE], "items": items})
        assert response.status_code == 200, response.text
        payload = response.json()

        assert [row["value"] for row in payload["verdicts"]] == [
            "accept",
            "accept",
            "accept",
            "accept",
            "reject",
        ]
        for row in payload["verdicts"]:
            assert row["reason"], "every verdict carries a reason (AC-003.2)"
            assert row["reason_code"], "and a stable machine code for the UI to map on"
        assert "guarantee" in payload["verdicts"][4]["reason"]

        # Cheapest-first ordering: the rule settles the fifth, so the judge saw only four.
        assert server.call_count == 4

    def test_the_health_record_comes_back_with_the_rate_and_its_n(self, client) -> None:
        server = StubJudgeServer(reply="SCORE: 0.80")
        provider_port.bind_provider_resolver(StubProviderResolver(server.provider()))
        items = [{"prompt": "Q", "completion": "a long enough answer"} for _ in range(5)]
        payload = post(
            client, {"config": COMPOSITE, "catalog": [RULE, JUDGE], "items": items}
        ).json()
        health = payload["health"]
        assert health["n"] == 5
        assert health["accepted"] == 5
        assert health["acceptance_rate"] == 1.0
        assert "selection_accepts_everything" in [s["name"] for s in health["signals"]]

    def test_the_degenerate_signal_carries_the_fixed_sentence(self, client) -> None:
        """AC-003.14. One sentence, supplied by F3, so F7 and F9 do not write two."""
        items = [{"completion": "we guarantee it"} for _ in range(5)]
        payload = post(client, {"config": RULE, "items": items}).json()
        signals = {s["name"]: s["message"] for s in payload["health"]["signals"]}
        assert signals["selection_rejects_everything"] == (
            "A selector that accepts everything is not selecting; one that rejects "
            "everything will stall the run."
        )

    def test_the_configuration_identity_comes_back_with_its_provenance(self, client) -> None:
        payload = post(client, {"config": RULE, "items": [{"completion": "abc"}]}).json()
        assert len(payload["config_sha256"]) == 64
        assert payload["config_kind"] == "rule"
        assert payload["hash_provenance"] == "local"
        assert payload["hashes"]["selector_hash"] == payload["config_sha256"]
        assert payload["hashes"]["selector_hash_provenance"] == "local"

    def test_a_reference_is_taken_from_the_item_not_the_configuration(self, client) -> None:
        config = {
            "kind": "reference_match",
            "name": "ref",
            "config_sha256": None,
            "mode": "exact",
        }
        payload = post(
            client,
            {
                "config": config,
                "items": [
                    {"completion": "Paris", "reference": "Paris"},
                    {"completion": "Lyon", "reference": "Paris"},
                ],
            },
        ).json()
        assert [row["value"] for row in payload["verdicts"]] == ["accept", "reject"]
        assert payload["verdicts"][0]["match_mode"] == "exact"

    def test_a_judged_verdict_carries_who_judged(self, client) -> None:
        """AC-003.21, over the transport - so F9 can show it and an audit can read it."""
        server = StubJudgeServer(reply="SCORE: 0.90")
        provider_port.bind_provider_resolver(StubProviderResolver(server.provider()))
        payload = post(
            client, {"config": JUDGE, "items": [{"prompt": "Q", "completion": "A"}]}
        ).json()
        judge = payload["verdicts"][0]["judge"]
        assert judge["kind"] == "openai_compatible"
        assert judge["model"] == "judge-7b"
        assert judge["base_url"] == "http://judge.test:8000"


class TestRefusalsComeBackInTheEnvelope:
    def test_an_unknown_type_is_a_named_refusal_not_a_validation_dump(self, client) -> None:
        response = post(
            client,
            {"config": {"kind": "bm25", "name": "x", "config_sha256": None}, "items": []},
        )
        assert response.status_code == 400
        body = response.json()
        assert set(body["error"]) == {"code", "message", "details"}
        assert body["error"]["code"] == "selector_config_invalid"

    def test_an_invalid_body_names_the_failing_pointer(self, client) -> None:
        response = post(
            client,
            {
                "config": {"kind": "rule", "name": "x", "config_sha256": None},
                "items": [{"completion": "a"}],
            },
        )
        assert response.json()["error"]["details"]["pointer"]

    def test_a_misspelled_request_field_is_refused_rather_than_ignored(self, client) -> None:
        """``extra="forbid"``: a silently-ignored field means the verdicts came from a
        different configuration than the caller believes."""
        response = post(client, {"config": RULE, "itemz": [{"completion": "a"}], "items": []})
        assert response.status_code == 422
        assert response.json()["error"]["code"] == "validation_failed"

    def test_an_unconfigured_judge_refuses_503_naming_the_service(self, client) -> None:
        provider_port.bind_provider_resolver(None)
        response = post(client, {"config": JUDGE, "items": [{"completion": "A"}]})
        assert response.status_code == 503
        assert response.json()["error"]["code"] == "judge_provider_not_configured"

    def test_an_unconfigured_review_queue_refuses_503_naming_midataworks(self, client) -> None:
        config = {
            "kind": "human_review",
            "name": "h",
            "config_sha256": None,
            "queue": "legal",
        }
        response = post(client, {"config": config, "items": [{"completion": "A"}]})
        assert response.status_code == 503
        assert response.json()["error"]["code"] == "midataworks_not_configured"

    def test_a_judge_outage_refuses_rather_than_accepting_everything(self, client) -> None:
        """**Mutation control 3's end-to-end form.** A judge outage must not come back as
        five accepts."""
        import httpx

        server = StubJudgeServer(raise_error=httpx.ConnectError("down"))
        provider_port.bind_provider_resolver(StubProviderResolver(server.provider()))
        response = post(
            client, {"config": JUDGE, "items": [{"completion": f"A{i}"} for i in range(5)]}
        )
        assert response.status_code == 400
        assert response.json()["error"]["code"] == "judge_provider_unavailable"
        assert "verdicts" not in response.json()

    def test_an_oversized_batch_is_refused_with_the_cap_named(self, client, monkeypatch) -> None:
        monkeypatch.setenv("EVALUATE_MAX_COMPLETIONS", "2")
        reset_selector_settings()
        response = post(
            client, {"config": RULE, "items": [{"completion": "abc"} for _ in range(3)]}
        )
        assert response.json()["error"]["code"] == "evaluation_too_large"
        assert response.json()["error"]["details"]["limit"] == 2

    def test_a_composite_cycle_is_refused_naming_the_cycle(self, client) -> None:
        loop = {
            "kind": "composite",
            "name": "loop",
            "config_sha256": None,
            "children": ["loop"],
            "combine": "all",
        }
        response = post(client, {"config": loop, "catalog": [loop], "items": []})
        assert response.json()["error"]["code"] == "composite_cycle"
        assert response.json()["error"]["details"]["cycle"] == ["loop", "loop"]


class TestUndecidedOverTheWire:
    def test_a_pending_human_comes_back_as_undecided_with_its_item_id(self, client) -> None:
        """Never converted to accept, and the item id is what lets the caller reconcile."""
        human_review.bind_review_queue(StubReviewQueue())
        config = {
            "kind": "human_review",
            "name": "h",
            "config_sha256": None,
            "queue": "legal",
        }
        payload = post(
            client,
            {"config": config, "items": [{"prompt": "Q", "completion": "A"}]},
        ).json()
        row = payload["verdicts"][0]
        assert row["value"] == "undecided"
        assert row["review_item_id"]
        assert row["score"] is None, "an undecided verdict carries no score, by the contract"
        assert payload["health"]["undecided"] == 1
        assert payload["health"]["accepted"] == 0
