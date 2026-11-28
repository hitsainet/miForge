"""``api.v1.endpoints.selectors``'s routes answer as the contract says (021 FR-021.13).

The evaluate call is a composite of a rule and a judge, the judge answered by the stub
judge endpoint bound at ``services.selectors.provider_port`` (``tests.support.selector_stubs``,
as ``test_selector_evaluate_endpoint.py`` does), so the verdicts carry both a rule's reason
and a judge's record.
"""

from __future__ import annotations

import json
from collections.abc import Iterator

import pytest

from tests.support.answer_cases import answered, assert_covered, case
from tests.support.approval_world import World

ROUTER = "api.v1.endpoints.selectors"
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


@pytest.fixture
def judge(monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    from core.config import reset_settings
    from services.selectors import provider_port
    from services.selectors.settings import reset_selector_settings
    from tests.support.selector_stubs import StubJudgeServer, StubProviderResolver

    monkeypatch.setenv("JUDGE_PROMPT_TEMPLATE", TEMPLATE)
    monkeypatch.setenv("JUDGE_PROVIDER_BASE_URL", "http://judge.test:8000")
    monkeypatch.setenv("JUDGE_PROVIDER_MODEL", "judge-7b")
    reset_settings()
    reset_selector_settings()
    server = StubJudgeServer(reply="SCORE: 0.80\nREASON: coherent")
    provider_port.bind_provider_resolver(StubProviderResolver(server.provider()))
    yield
    provider_port.bind_provider_resolver(None)
    reset_selector_settings()
    reset_settings()


@case(ROUTER, "GET /api/v1/selectors")
def test_list_selectors(world: World) -> None:
    body = answered("GET /api/v1/selectors", world.client.get("/api/v1/selectors"))
    assert len(body) == 5


@case(ROUTER, "POST /api/v1/selectors/evaluate")
def test_evaluate_completions(world: World, judge: None) -> None:
    items = [
        {"prompt": "Q0", "completion": "A sufficiently long answer"},
        {"prompt": "Q1", "completion": "we guarantee a refund"},
    ]
    body = answered(
        "POST /api/v1/selectors/evaluate",
        world.client.post(
            "/api/v1/selectors/evaluate",
            json={"config": COMPOSITE, "catalog": [RULE, JUDGE], "items": items},
        ),
    )
    assert [row["value"] for row in body["verdicts"]] == ["accept", "reject"]


def test_every_route_of_the_router_has_a_case() -> None:
    assert_covered(ROUTER)
