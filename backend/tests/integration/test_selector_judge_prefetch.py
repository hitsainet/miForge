"""``judge_concurrency`` is wired: the judge prefetches its batch in bounded windows (4.28).

``SelectorSettings.judge_concurrency`` was read by nothing until 2026-10-16: per-item judging
calls ``generate([rendered])`` with ONE prompt, so the provider's bounded fan-out never
engaged. The seam is ``services/selectors/service.py::_prefetch_batch``; the design is a
cache **with miss semantics** - a prefetched reply is consumed by the item it was fetched
for, a pair with no entry is judged live - so prefetch changes latency, never a verdict.

Asserted against the real resolver, the real provider and an HTTP stub: the **payload
(window sizes) and the call count**, verdict equivalence with the window-of-one path, the
miss path, and that nothing outlives its batch. Cancellation promptness is in
``test_selector_judge_over_stub_endpoint.py::TestCancellationDuringAJudgedBatch``.
"""

from __future__ import annotations

import json
import re

import httpx
import pytest

from core.errors import Refusal
from services.selectors import provider_port
from services.selectors.service import evaluate
from services.selectors.settings import reset_selector_settings
from services.selectors.types import EvaluationContext
from tests.support.selector_stubs import (
    StubJudgeServer,
    StubProviderResolver,
    judge_config,
    resolved,
)

TEMPLATE = "Grade.\nQ: {prompt}\nA: {completion}\nReply 'SCORE: <0..1>'."


class ScoringServer(StubJudgeServer):
    """Replies with a score derived from the completion (``A7`` -> 0.7), so a verdict that
    was paired with the wrong reply is visible; ``BAD`` gets an unparseable reply."""

    def handler(self, request: httpx.Request) -> httpx.Response:
        payload = json.loads(request.content.decode())
        self.calls.append(payload)
        text = payload["messages"][-1]["content"]
        answer = re.search(r"A: (\S+)", text)
        token = answer.group(1) if answer else ""
        reply = "no score here" if token == "BAD" else f"SCORE: 0.{token[1:]}"
        return httpx.Response(
            200, json={"id": "x", "choices": [{"index": 0, "message": {"content": reply}}]}
        )


@pytest.fixture(autouse=True)
def _configured(monkeypatch):  # type: ignore[no-untyped-def]
    monkeypatch.setenv("JUDGE_PROMPT_TEMPLATE", TEMPLATE)
    monkeypatch.setenv("JUDGE_PROVIDER_BASE_URL", "http://judge.test:8000")
    monkeypatch.setenv("JUDGE_PROVIDER_MODEL", "judge-7b")
    reset_selector_settings()
    yield
    provider_port.bind_provider_resolver(None)
    reset_selector_settings()


def _setup(monkeypatch, window: int):  # type: ignore[no-untyped-def]
    monkeypatch.setenv("JUDGE_CONCURRENCY", str(window))
    reset_selector_settings()
    server = ScoringServer()
    provider = server.provider()
    sizes: list[int] = []
    real = provider.generate

    def spy(prompts, params):  # type: ignore[no-untyped-def]
        sizes.append(len(prompts))
        return real(prompts, params)

    provider.generate = spy  # type: ignore[method-assign]
    provider_port.bind_provider_resolver(StubProviderResolver(provider))
    return server, sizes, resolved(judge_config(threshold=0.5))


COMPLETIONS = ["A9", "A1", "A7", "A3", "A8"]


class TestTheWindowIsTheSetting:
    def test_five_completions_at_two_go_as_two_two_one_and_nothing_after(
        self, monkeypatch
    ) -> None:  # type: ignore[no-untyped-def]
        server, sizes, selector = _setup(monkeypatch, window=2)
        evaluate(COMPLETIONS, selector, EvaluationContext(), prompts=["Q"] * 5)
        assert sizes == [2, 2, 1], "every judge call came from the prefetch, in windows of 2"
        assert server.call_count == 5, "each completion judged exactly once"

    def test_the_verdicts_are_the_window_of_one_verdicts(self, monkeypatch) -> None:  # type: ignore[no-untyped-def]
        """Equivalence: prefetch pairs each reply with ITS completion."""
        results = {}
        for window in (1, 4):
            _, _, selector = _setup(monkeypatch, window=window)
            result = evaluate(COMPLETIONS, selector, EvaluationContext(), prompts=["Q"] * 5)
            results[window] = [(r.verdict.value.value, r.verdict.score) for r in result.records]
        assert results[1] == results[4]
        assert [score for _, score in results[4]] == [0.9, 0.1, 0.7, 0.3, 0.8]


class TestMissSemantics:
    def test_an_oversized_text_is_a_miss_and_refuses_at_its_own_item(
        self, monkeypatch
    ) -> None:  # type: ignore[no-untyped-def]
        monkeypatch.setenv("SELECTOR_MAX_TEXT_CHARS", "50")
        _server, sizes, selector = _setup(monkeypatch, window=4)
        with pytest.raises(Refusal) as caught:
            evaluate(["A9", "A" + "x" * 80, "A7"], selector, EvaluationContext(), prompts=["Q"] * 3)
        assert caught.value.code == "selector_text_too_long"
        assert sizes == [2], "the oversized pair was never sent to the judge"

    def test_an_unparseable_reply_refuses_the_evaluation_as_before(self, monkeypatch) -> None:  # type: ignore[no-untyped-def]
        _, _, selector = _setup(monkeypatch, window=4)
        with pytest.raises(Refusal) as caught:
            evaluate(["A9", "BAD", "A7"], selector, EvaluationContext(), prompts=["Q"] * 3)
        assert caught.value.code == "judge_reply_unparseable"

    def test_nothing_prefetched_outlives_its_batch(self, monkeypatch) -> None:  # type: ignore[no-untyped-def]
        server, _sizes, selector = _setup(monkeypatch, window=4)
        with pytest.raises(Refusal):
            evaluate(["A9", "BAD", "A7", "A3"], selector, EvaluationContext(), prompts=["Q"] * 4)
        assert selector.plugin._prefetched == {}  # the property under test
        before = server.call_count
        evaluate(["A7"], selector, EvaluationContext(), prompts=["Q"])
        assert server.call_count == before + 1, "a new batch is judged, not served stale"

    def test_a_repeated_pair_takes_one_reply_each(self, monkeypatch) -> None:  # type: ignore[no-untyped-def]
        server, sizes, selector = _setup(monkeypatch, window=4)
        evaluate(["A7", "A7", "A7"], selector, EvaluationContext(), prompts=["Q"] * 3)
        assert sizes == [3]
        assert server.call_count == 3
