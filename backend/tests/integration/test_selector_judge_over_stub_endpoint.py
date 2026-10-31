"""The judge over a stub OpenAI-compatible server  (AC-003.4, AC-003.7; E-4, E-5).

**Four scenarios the FPRD names:** success, timeout, malformed reply, and a reply that
parses but scores below threshold. The fourth matters as much as the first three: it is the
only one of the four that is a **verdict** rather than a refusal, and the line between the
two is the line this feature's correctness sits on.

A **stub server**, not a mock: the real request builder serialises, the real parser reads,
and only the socket is replaced. *A mock never runs the real request builder, and ADR-016
names over-mocked stubs among the reasons a suite stays green over a bug.*
"""

from __future__ import annotations

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


@pytest.fixture(autouse=True)
def _configured(monkeypatch):
    monkeypatch.setenv("JUDGE_PROMPT_TEMPLATE", TEMPLATE)
    monkeypatch.setenv("JUDGE_PROVIDER_BASE_URL", "http://judge.test:8000")
    monkeypatch.setenv("JUDGE_PROVIDER_MODEL", "judge-7b")
    reset_selector_settings()
    yield
    provider_port.bind_provider_resolver(None)
    reset_selector_settings()


def batch(server: StubJudgeServer, completions, *, threshold=0.6):
    provider_port.bind_provider_resolver(StubProviderResolver(server.provider()))
    selector = resolved(judge_config(threshold=threshold))
    return evaluate(
        completions,
        selector,
        EvaluationContext(batch_id="b1", run_id="run-1"),
        prompts=["Q"] * len(completions),
    )


class TestSuccess:
    def test_a_batch_is_judged_and_the_health_record_is_right(self) -> None:
        server = StubJudgeServer(reply="SCORE: 0.80\nREASON: solid")
        result = batch(server, ["A1", "A2", "A3", "A4", "A5"])
        assert [record.verdict.value.value for record in result.records] == ["accept"] * 5
        assert result.health.n == 5
        assert result.health.accepted == 5
        assert result.health.acceptance_rate == 1.0
        assert server.call_count == 5, "one judge call per completion"

    def test_the_payload_is_asserted_field_by_field_for_every_call(self) -> None:
        """ "Was called" passes against a call sending the wrong template."""
        server = StubJudgeServer(reply="SCORE: 0.80")
        batch(server, ["first answer", "second answer"])
        assert len(server.calls) == 2
        for index, call in enumerate(server.calls):
            assert call["model"] == "judge-7b"
            assert call["temperature"] == 0.0
            assert call["stream"] is False
            assert len(call["messages"]) == 1
            content = call["messages"][0]["content"]
            assert content.startswith("Grade.")
            assert "Q: Q" in content
            assert ["first answer", "second answer"][index] in content

    def test_every_record_carries_the_judge_identity(self) -> None:
        server = StubJudgeServer(reply="SCORE: 0.80")
        result = batch(server, ["A"] * 3)
        for record in result.records:
            assert record.judge is not None
            assert record.judge.model == "judge-7b"
            assert record.judge.base_url == "http://judge.test:8000"

    def test_the_grader_distribution_is_computed_from_real_scores(self) -> None:
        replies = iter(["SCORE: 0.1", "SCORE: 0.5", "SCORE: 0.9", "SCORE: 0.7", "SCORE: 0.3"])

        class Varying(StubJudgeServer):
            def handler(self, request):
                self.reply = next(replies)
                return super().handler(request)

        result = batch(Varying(), ["A"] * 5, threshold=0.4)
        stats = result.health.grader_stats
        assert stats is not None
        assert stats.n == 5
        assert stats.min == pytest.approx(0.1)
        assert stats.max == pytest.approx(0.9)
        assert stats.variance > 0.0, "real spread, so no degenerate signal"
        assert result.health.signals == ()


class TestBelowThreshold:
    def test_a_parsing_reply_below_threshold_is_a_verdict_not_a_refusal(self) -> None:
        """The fourth scenario, and the one that distinguishes the two categories."""
        server = StubJudgeServer(reply="SCORE: 0.20\nREASON: off topic")
        result = batch(server, ["A1", "A2"], threshold=0.6)
        assert [record.verdict.value.value for record in result.records] == ["reject", "reject"]
        assert result.health.rejected == 2

    def test_a_whole_batch_below_threshold_raises_the_named_signal(self) -> None:
        """*"one that rejects everything will stall the run."*"""
        server = StubJudgeServer(reply="SCORE: 0.10")
        result = batch(server, ["A"] * 6, threshold=0.6)
        assert "selection_rejects_everything" in result.health.signal_names

    def test_a_whole_batch_at_one_score_raises_the_no_spread_signal_too(self) -> None:
        """A judge answering identically for every candidate gives GRPO zero advantage."""
        server = StubJudgeServer(reply="SCORE: 0.80")
        result = batch(server, ["A"] * 6, threshold=0.6)
        assert "selection_accepts_everything" in result.health.signal_names
        assert "grader_scores_have_no_spread" in result.health.signal_names


class TestFailuresRefuseTheBatch:
    """AC-003.7, E-4, E-5. **No verdict is defaulted** - mutation control 3's targets."""

    def test_a_timeout_refuses_the_batch_rather_than_accepting_it(self) -> None:
        server = StubJudgeServer(raise_error=httpx.ReadTimeout("slow"))
        with pytest.raises(Refusal) as exc:
            batch(server, ["A"] * 5)
        assert exc.value.code == "judge_provider_unavailable"

    def test_an_unreachable_endpoint_refuses_the_batch(self) -> None:
        server = StubJudgeServer(raise_error=httpx.ConnectError("down"))
        with pytest.raises(Refusal) as exc:
            batch(server, ["A"] * 5)
        assert exc.value.code == "judge_provider_unavailable"

    def test_a_malformed_reply_refuses_the_batch(self) -> None:
        server = StubJudgeServer(reply="Honestly, pretty good. 8/10.")
        with pytest.raises(Refusal) as exc:
            batch(server, ["A"] * 5)
        assert exc.value.code == "judge_reply_unparseable"

    def test_a_sustained_failure_part_way_through_refuses_the_whole_batch(self) -> None:
        """A short batch silently drops candidates, which looks like a strict selector.

        The endpoint dies from the third call onward - so the one permitted retry cannot
        paper over it, and the batch must refuse rather than return two verdicts and
        pretend the other three were rejected.
        """
        state = {"n": 0}

        class DiesOnThird(StubJudgeServer):
            def handler(self, request):
                state["n"] += 1
                if state["n"] >= 3:
                    raise httpx.ConnectError("down for good")
                return super().handler(request)

        with pytest.raises(Refusal) as exc:
            batch(DiesOnThird(reply="SCORE: 0.8"), ["A"] * 5)
        assert exc.value.code == "judge_provider_unavailable"

    def test_a_single_transient_flap_is_recovered_by_the_one_retry(self) -> None:
        """The documented retry policy, asserted so its *boundary* is known.

        One dropped connection is common and is retried; a systematic failure is not
        smoothed into latency, which the test above pins from the other side.
        """
        state = {"n": 0}

        class FlapsOnce(StubJudgeServer):
            def handler(self, request):
                state["n"] += 1
                if state["n"] == 3:
                    raise httpx.ConnectError("one flap")
                return super().handler(request)

        result = batch(FlapsOnce(reply="SCORE: 0.8"), ["A"] * 5)
        assert len(result.records) == 5
        assert state["n"] == 6, "five calls plus the one retry"

    def test_no_failure_mode_yields_a_batch_of_accepts(self) -> None:
        """**The one assertion mutation control 3 must break.**

        Returning ``accept`` on a transport failure is the plausible-looking "robustness"
        change that converts this feature into the accept-everything selector it exists to
        prevent, and it would be invisible: the run would train on everything, loss would
        fall, and the test score would not move.
        """
        for server in (
            StubJudgeServer(raise_error=httpx.ConnectError("x")),
            StubJudgeServer(raise_error=httpx.ReadTimeout("x")),
            StubJudgeServer(status=503),
            StubJudgeServer(reply="no score"),
            StubJudgeServer(reply="SCORE: 42"),
        ):
            with pytest.raises(Refusal):
                batch(server, ["A"] * 3)


class TestCancellationDuringAJudgedBatch:
    """E-16: evaluated and recorded, or not started - and the judge stops PROMPTLY.

    *Rewritten 2026-10-16 (suite tracker 4.28).* This counted calls to ``is_cancelled`` and
    expected the per-item loop to be the only caller. The judge now prefetches the batch in
    windows of ``judge_concurrency`` and checks cancellation before each window, so the
    cancellation here is driven the way it arrives in production - **while judge calls are
    in flight** (the server raises the flag on its Nth call) - and the bound asserted is the
    real promise: no record is written after the request, and no judge call is STARTED after
    the window in flight when it arrived.
    """

    def _run(self, monkeypatch: pytest.MonkeyPatch, window: int, cancel_on_call: int):
        from services.selectors.settings import reset_selector_settings

        monkeypatch.setenv("JUDGE_CONCURRENCY", str(window))
        reset_selector_settings()
        server = StubJudgeServer(reply="SCORE: 0.8")
        flag = {"cancelled": False}
        handler = server.handler

        def cancelling(request):  # type: ignore[no-untyped-def]
            response = handler(request)
            if server.call_count >= cancel_on_call:
                flag["cancelled"] = True
            return response

        server.handler = cancelling  # type: ignore[method-assign]
        provider_port.bind_provider_resolver(StubProviderResolver(server.provider()))
        selector = resolved(judge_config())
        try:
            with pytest.raises(Refusal) as exc:
                evaluate(
                    ["A"] * 12,
                    selector,
                    EvaluationContext(is_cancelled=lambda: flag["cancelled"]),
                    prompts=["Q"] * 12,
                )
        finally:
            reset_selector_settings()
        return server, exc.value

    def test_with_a_window_of_one_it_stops_after_the_call_in_flight(self, monkeypatch) -> None:  # type: ignore[no-untyped-def]
        """``judge_concurrency=1``: exactly the old per-item promptness."""
        server, refusal = self._run(monkeypatch, window=1, cancel_on_call=2)
        assert refusal.code == "evaluation_cancelled"
        assert server.call_count == 2, "no judge call may start after the cancellation"
        assert refusal.details["completed"] == 0, "nothing judged ahead is recorded"

    def test_with_a_window_of_four_at_most_one_window_runs(self, monkeypatch) -> None:  # type: ignore[no-untyped-def]
        server, refusal = self._run(monkeypatch, window=4, cancel_on_call=2)
        assert refusal.code == "evaluation_cancelled"
        assert (
            server.call_count == 4
        ), "the window in flight finishes and no further window starts: 4 calls, not 12"
        assert refusal.details["completed"] == 0
