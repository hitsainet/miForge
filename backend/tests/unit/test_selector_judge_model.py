"""The judge-model selector  (AC-003.4, AC-003.5, AC-003.7, AC-003.21; E-4, E-5, E-6, E-15).

Two things matter most here, and both are silent failures if they go wrong:

1. **No default verdict on any failure.** An unreachable endpoint, a timeout, an unparseable
   reply and an out-of-range score each refuse the **evaluation**. **Mutation control 3**
   makes a transport failure return ``accept`` - the plausible-looking "robustness" change
   that converts this feature into the accept-everything selector it exists to prevent -
   and the tests here must go red.
2. **With no endpoint configured the type is unavailable BY NAME**, never falling back to
   anything. There is nothing to fall back *to*, which is R-02.11 enforced by absence: the
   only candidate would be the model being trained, and a model grading itself is not a
   measurement.

The judge runs over a **stub server**, not a mock, so the real request builder and the real
response parser run; and the **payload is asserted field by field with the call count**,
because "was called" passes against a call sending the wrong template - and the template is
what the judge actually judges.
"""

from __future__ import annotations

import httpx
import pytest

from core.errors import Refusal
from providers.base import ProviderIdentity
from schemas.selector import VerdictValue
from services.selectors import provider_port
from services.selectors.plugins.judge_model import parse_judge_reply, render_template
from services.selectors.settings import get_selector_settings, reset_selector_settings
from tests.support.selector_stubs import (
    StubJudgeServer,
    StubProviderResolver,
    judge_config,
    resolved,
)

TEMPLATE = (
    "Rate this answer.\nQ: {prompt}\nA: {completion}\n"
    "Reply with a line 'SCORE: <0..1>' and optionally 'REASON: <why>'."
)


@pytest.fixture(autouse=True)
def _judge_settings(monkeypatch: pytest.MonkeyPatch):
    """A configured template and endpoint. Unset afterwards, so the standalone case is
    reachable rather than being masked by a leaked environment."""
    monkeypatch.setenv("JUDGE_PROMPT_TEMPLATE", TEMPLATE)
    monkeypatch.setenv("JUDGE_PROVIDER_BASE_URL", "http://judge.test:8000")
    monkeypatch.setenv("JUDGE_PROVIDER_MODEL", "judge-7b")
    reset_selector_settings()
    yield
    provider_port.bind_provider_resolver(None)
    reset_selector_settings()


def judge(server: StubJudgeServer, *, threshold: float | None = 0.6, run_target=None):
    resolver = StubProviderResolver(server.provider())
    provider_port.bind_provider_resolver(resolver)
    return resolved(judge_config(threshold=threshold), run_target=run_target)


class TestThePayloadAndTheCallCount:
    def test_the_rendered_template_is_what_is_sent(self) -> None:
        """AC-003.4. Field by field: ``model``, the message roles, the decode parameters."""
        server = StubJudgeServer(reply="SCORE: 0.90")
        selector = judge(server)
        selector.plugin.evaluate(prompt="What is 2+2?", completion="4", context={})

        assert server.call_count == 1, "exactly one call per completion"
        body = server.calls[0]
        assert body["model"] == "judge-7b"
        assert body["stream"] is False
        assert body["temperature"] == 0.0, "a sampled judge makes a run irreproducible"
        assert [message["role"] for message in body["messages"]] == ["user"]
        content = body["messages"][0]["content"]
        assert "What is 2+2?" in content
        assert "A: 4" in content
        assert "SCORE:" in content, "the reply form must be asked for, not assumed"

    def test_one_call_per_completion_and_no_more(self) -> None:
        server = StubJudgeServer(reply="SCORE: 0.70")
        selector = judge(server)
        for completion in ("a", "b", "c"):
            selector.plugin.evaluate(prompt="q", completion=completion, context={})
        assert server.call_count == 3

    def test_no_api_key_means_no_authorization_header(self) -> None:
        server = StubJudgeServer()
        selector = judge(server)
        selector.plugin.evaluate(prompt="q", completion="a", context={})
        assert "authorization" not in server.headers[0]


class TestTheVerdictRecordsBothNumbers:
    def test_above_threshold_accepts_and_names_both_numbers(self) -> None:
        server = StubJudgeServer(reply="SCORE: 0.80\nREASON: coherent")
        verdict = judge(server, threshold=0.6).plugin.evaluate(
            prompt="q", completion="a", context={}
        )
        assert verdict.value is VerdictValue.ACCEPT
        assert verdict.label == "judge_above_threshold"
        assert "judge 0.80/1.00, threshold 0.60" in (verdict.reason or "")
        assert "coherent" in (verdict.reason or "")
        assert verdict.score == pytest.approx(0.8)

    def test_below_threshold_rejects_and_names_both_numbers(self) -> None:
        server = StubJudgeServer(reply="SCORE: 0.20")
        verdict = judge(server, threshold=0.6).plugin.evaluate(
            prompt="q", completion="a", context={}
        )
        assert verdict.value is VerdictValue.REJECT
        assert "judge 0.20/1.00, threshold 0.60" in (verdict.reason or "")

    def test_every_judged_verdict_records_who_judged(self) -> None:
        """AC-003.21, FR-003.18. Without it, BRD-02 AC3 is not auditable afterwards."""
        server = StubJudgeServer()
        selector = judge(server)
        identity = selector.plugin.judge_identity()
        assert identity.kind == "openai_compatible"
        assert identity.model == "judge-7b"
        assert identity.base_url == "http://judge.test:8000"

    def test_the_resolved_provider_is_on_the_bindings_for_the_record(self) -> None:
        server = StubJudgeServer()
        selector = judge(server)
        assert selector.bindings["provider"].identity.model == "judge-7b"


class TestNoDefaultVerdictOnAnyFailure:
    """AC-003.7 (E-4, E-5). **Mutation control 3's targets.**"""

    def test_an_unreachable_endpoint_refuses_rather_than_accepting(self) -> None:
        server = StubJudgeServer(raise_error=httpx.ConnectError("no route"))
        selector = judge(server)
        with pytest.raises(Refusal) as exc:
            selector.plugin.evaluate(prompt="q", completion="a", context={})
        assert exc.value.code == "judge_provider_unavailable"

    def test_a_timeout_refuses_rather_than_accepting(self) -> None:
        server = StubJudgeServer(raise_error=httpx.ReadTimeout("slow"))
        selector = judge(server)
        with pytest.raises(Refusal) as exc:
            selector.plugin.evaluate(prompt="q", completion="a", context={})
        assert exc.value.code == "judge_provider_unavailable"
        assert exc.value.details["read_timeout_s"] == 2.0

    def test_a_4xx_refuses_and_is_not_retried(self) -> None:
        """A 4xx is a statement about the request; retrying turns a wrong model name into a
        slow failure instead of a fast one."""
        server = StubJudgeServer(status=404)
        selector = judge(server)
        with pytest.raises(Refusal) as exc:
            selector.plugin.evaluate(prompt="q", completion="a", context={})
        assert exc.value.code == "judge_provider_unavailable"
        assert server.call_count == 1, "a 4xx must not be retried"

    def test_a_connection_error_is_retried_exactly_once(self) -> None:
        server = StubJudgeServer(raise_error=httpx.ConnectError("no route"))
        selector = judge(server)
        with pytest.raises(Refusal):
            selector.plugin.evaluate(prompt="q", completion="a", context={})
        assert server.call_count == 2, "one retry on a connection error, and only one"

    def test_a_reply_with_no_score_line_refuses(self) -> None:
        """Strict parsing: a regex scraping the first integer anywhere would read a score
        out of the model's own prose."""
        server = StubJudgeServer(reply="I think this answer is about a 7 out of 10, honestly.")
        selector = judge(server)
        with pytest.raises(Refusal) as exc:
            selector.plugin.evaluate(prompt="q", completion="a", context={})
        assert exc.value.code == "judge_reply_unparseable"
        assert "7" not in str(exc.value.details.get("expected", ""))

    def test_the_unparseable_refusal_truncates_the_reply(self) -> None:
        server = StubJudgeServer(reply="x" * 5000)
        selector = judge(server)
        with pytest.raises(Refusal) as exc:
            selector.plugin.evaluate(prompt="q", completion="a", context={})
        assert len(exc.value.details["reply"]) < 600

    def test_a_reply_with_no_choices_refuses(self) -> None:
        server = StubJudgeServer(body={"id": "x"})
        selector = judge(server)
        with pytest.raises(Refusal) as exc:
            selector.plugin.evaluate(prompt="q", completion="a", context={})
        assert exc.value.code == "judge_reply_unparseable"


class TestTheScoreRangeIsTheContracts:
    """FPRD OQ-5, E-15: the range is 0..1 and an outlier is refused, never clamped."""

    @pytest.mark.parametrize("raw", ["SCORE: 7", "SCORE: -0.5", "SCORE: 1.0001"])
    def test_an_out_of_range_score_refuses_with_the_value_and_the_range(self, raw) -> None:
        with pytest.raises(Refusal) as exc:
            parse_judge_reply(raw)
        assert exc.value.code == "grader_score_out_of_range"
        assert exc.value.details["range"] == [0.0, 1.0]

    def test_no_clamping_happens(self) -> None:
        """A clamp turns a judge that misread the scale into a confident verdict."""
        server = StubJudgeServer(reply="SCORE: 9")
        selector = judge(server)
        with pytest.raises(Refusal) as exc:
            selector.plugin.evaluate(prompt="q", completion="a", context={})
        assert exc.value.details["value"] == 9.0

    @pytest.mark.parametrize("raw", ["SCORE: 0", "SCORE: 1", "SCORE:0.5", "score: 0.33"])
    def test_the_bounds_and_the_spellings_that_are_accepted(self, raw) -> None:
        score, _ = parse_judge_reply(raw)
        assert 0.0 <= score <= 1.0


class TestRefusalsAtPrepare:
    def test_a_missing_threshold_is_refused_not_defaulted(self) -> None:
        """ADR-018's named shape for this feature: ``getattr(config, 'threshold', 0.5)``
        is a fabricated verdict policy."""
        server = StubJudgeServer()
        provider_port.bind_provider_resolver(StubProviderResolver(server.provider()))
        with pytest.raises(Refusal) as exc:
            resolved(judge_config(threshold=None))
        assert exc.value.code == "judge_threshold_missing"

    def test_no_configured_template_refuses_by_name(self, monkeypatch) -> None:
        """A hard-coded template would be a verdict policy chosen by whoever wrote the
        line; the contract has no field for one, so it is explicit configuration."""
        monkeypatch.delenv("JUDGE_PROMPT_TEMPLATE", raising=False)
        reset_selector_settings()
        server = StubJudgeServer()
        provider_port.bind_provider_resolver(StubProviderResolver(server.provider()))
        with pytest.raises(Refusal) as exc:
            resolved(judge_config())
        assert exc.value.code == "judge_template_not_configured"
        assert "SCORE" in str(exc.value.details["reply_form"])

    def test_a_template_naming_an_unknown_placeholder_refuses_at_prepare(self, monkeypatch) -> None:
        monkeypatch.setenv("JUDGE_PROMPT_TEMPLATE", "Rate {answer}")
        reset_selector_settings()
        server = StubJudgeServer()
        provider_port.bind_provider_resolver(StubProviderResolver(server.provider()))
        with pytest.raises(Refusal) as exc:
            resolved(judge_config())
        assert exc.value.code == "judge_template_invalid"
        assert server.call_count == 0, "refused before any completion was judged"

    def test_an_in_process_judge_is_refused_before_any_call(self) -> None:
        """AC-003.5: pre-flight, before the first step - not at the first completion."""

        class InProcessProvider:
            @property
            def identity(self):
                return ProviderIdentity(kind="in_process", model="lfm2-1.2b")

            def available(self) -> bool:
                return True

            def generate(self, prompts, params, **options):  # pragma: no cover
                raise AssertionError("the guard must refuse before any generation")

        provider_port.bind_provider_resolver(StubProviderResolver(InProcessProvider()))
        with pytest.raises(Refusal) as exc:
            resolved(judge_config())
        assert exc.value.code == "judge_is_resident_model"

    def test_a_judge_matching_the_run_target_is_refused(self) -> None:
        """AC-003.6, through the resolver rather than the guard directly."""
        server = StubJudgeServer()
        provider_port.bind_provider_resolver(StubProviderResolver(server.provider()))
        target = ProviderIdentity(
            kind="openai_compatible", model="judge-7b", base_url="http://judge.test:8000/"
        )
        with pytest.raises(Refusal) as exc:
            resolved(judge_config(), run_target=target)
        assert exc.value.code == "judge_is_resident_model"

    def test_a_same_lineage_judge_records_the_caveat_on_the_resolution(self) -> None:
        """E-7: returned and recorded, never only logged."""
        server = StubJudgeServer()
        provider_port.bind_provider_resolver(StubProviderResolver(server.provider()))
        target = ProviderIdentity(
            kind="openai_compatible", model="judge-7b", base_url="http://other:8000"
        )
        selector = resolved(judge_config(), run_target=target)
        assert selector.caveats == ("judge_shares_lineage",)

    def test_an_unreadable_run_target_refuses_rather_than_disabling_the_guard(self) -> None:
        server = StubJudgeServer()
        provider_port.bind_provider_resolver(StubProviderResolver(server.provider()))
        with pytest.raises(Refusal) as exc:
            resolved(judge_config(), run_target=object())
        assert exc.value.code == "judge_run_target_unreadable"


class TestUnavailableByName:
    """**The question the coordinator asked**: what happens with no endpoint configured."""

    def test_with_no_resolver_bound_the_judge_refuses_by_name(self) -> None:
        provider_port.bind_provider_resolver(None)
        with pytest.raises(Refusal) as exc:
            resolved(judge_config())
        assert exc.value.code == "judge_provider_not_configured"
        assert exc.value.details["service"] == "judge_provider"

    def test_the_refusal_says_miforge_will_not_fall_back_to_the_trained_model(self) -> None:
        """Stated in the message because the operator's next step is to configure an
        endpoint, not to wonder why nothing happened."""
        provider_port.bind_provider_resolver(None)
        with pytest.raises(Refusal) as exc:
            resolved(judge_config())
        assert "will NOT fall back" in exc.value.message
        assert "R-02.11" in exc.value.message

    def test_a_resolver_that_does_not_know_the_key_refuses_by_name(self) -> None:
        server = StubJudgeServer()
        provider_port.bind_provider_resolver(StubProviderResolver(server.provider(), key="other"))
        with pytest.raises(Refusal) as exc:
            resolved(judge_config(provider="judge"))
        assert exc.value.code == "judge_provider_not_configured"
        assert exc.value.details["provider_key"] == "judge"

    def test_the_listing_marks_the_judge_unavailable_naming_the_service(self) -> None:
        provider_port.bind_provider_resolver(None)
        from services.selectors import registry

        rows = {row.name: row for row in registry.list_capabilities(get_selector_settings())}
        assert rows["judge_model"].available is False
        assert rows["judge_model"].unavailable_reason == "judge_provider_not_configured"

    def test_the_listing_marks_the_judge_available_once_wired(self) -> None:
        server = StubJudgeServer()
        provider_port.bind_provider_resolver(StubProviderResolver(server.provider()))
        from services.selectors import registry

        rows = {row.name: row for row in registry.list_capabilities(get_selector_settings())}
        assert rows["judge_model"].available is True
        assert rows["judge_model"].unavailable_reason is None


class TestBounds:
    def test_an_oversized_completion_refuses_with_the_bound_named(self) -> None:
        """Truncation changes the verdict: a judge shown a truncated answer is judging a
        different answer."""
        server = StubJudgeServer()
        selector = judge(server)
        with pytest.raises(Refusal) as exc:
            selector.plugin.evaluate(prompt="q", completion="x" * 70_000, context={})
        assert exc.value.code == "selector_text_too_long"
        assert exc.value.details["limit"] == 65_536
        assert server.call_count == 0, "bounded BEFORE the call, not after"


class TestTemplateRendering:
    def test_a_stray_brace_is_refused_with_advice(self) -> None:
        with pytest.raises(Refusal) as exc:
            render_template("rate {", prompt="p", completion="c")
        assert exc.value.code == "judge_template_invalid"

    def test_both_placeholders_are_substituted(self) -> None:
        assert render_template("{prompt}|{completion}", prompt="p", completion="c") == "p|c"
