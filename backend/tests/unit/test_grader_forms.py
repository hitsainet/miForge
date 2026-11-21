"""Graders  (AC-003.20; FR-003.9, E-15).

**A grader is a form, not a sixth type**: the same object, under the same resolution path,
out of the same registry. That is what makes it impossible for a grader and a selector over
one configuration to disagree about what a completion scored - two code paths would be two
answers, and the symptom would be a run whose reward and whose admission disagreed with no
way to tell which was right.

**The range is 0..1 and it is the contract's**, not a per-configuration choice: it is fixed
at both ends by ``Verdict.score``'s bounds and by ``Grader.grade``'s documented range. That
answers FPRD OQ-5. A value outside it is **refused with the value and the range, never
clamped** - a clamp turns a grader that misread its scale into a confident number.
"""

from __future__ import annotations

import pytest

from core.errors import Refusal
from schemas.selector import Grader
from services.graders import SCORE_RANGE, grader_for
from services.selectors import registry
from services.selectors.settings import reset_selector_settings
from services.selectors.types import SelectorForm
from tests.support.selector_stubs import (
    StubJudgeServer,
    StubProviderResolver,
    judge_config,
    reference_config,
    resolved,
    rule_config,
)

TEMPLATE = "Rate {prompt} / {completion}. Reply 'SCORE: <0..1>'."


@pytest.fixture
def judge_settings(monkeypatch):
    monkeypatch.setenv("JUDGE_PROMPT_TEMPLATE", TEMPLATE)
    monkeypatch.setenv("JUDGE_PROVIDER_BASE_URL", "http://judge.test:8000")
    monkeypatch.setenv("JUDGE_PROVIDER_MODEL", "judge-7b")
    reset_selector_settings()
    yield
    from services.selectors import provider_port

    provider_port.bind_provider_resolver(None)
    reset_selector_settings()


class TestEveryDeclaredGraderFormIsReal:
    """A capability with nothing behind it is the prose-without-mechanism shape.

    This test found one: ``rule`` was registered with a grader form and has no ``grade``
    method, which would have made ``grader_for`` refuse on a type the listing advertised.
    """

    @pytest.mark.parametrize("kind", sorted(registry.registered_kinds()))
    def test_a_declared_grader_form_has_a_grade_method(self, kind: str) -> None:
        capability = registry.capability(kind)
        if SelectorForm.GRADER not in capability.forms:
            pytest.skip(f"{kind} declares no grader form")
        from services.selectors.plugins import judge_model, reference_match

        implementations = {
            "judge_model": judge_model.JudgeModelSelector,
            "reference_match": reference_match.ReferenceMatchSelector,
        }
        assert hasattr(implementations[kind], "grade"), (
            f"{kind} declares a grader form but its plugin has no grade(); the listing "
            f"would advertise a capability that refuses when called"
        )

    def test_exactly_the_two_scoring_types_offer_a_grader_form(self) -> None:
        offering = {
            kind
            for kind in registry.registered_kinds()
            if SelectorForm.GRADER in registry.capability(kind).forms
        }
        assert offering == {"judge_model", "reference_match"}

    def test_the_listing_reports_the_grader_form_on_exactly_those_types(self) -> None:
        from services.selectors.settings import get_selector_settings

        rows = {row.name: row for row in registry.list_capabilities(get_selector_settings())}
        assert "grader" in rows["reference_match"].forms
        assert "grader" not in rows["rule"].forms
        assert "grader" not in rows["human_review"].forms
        assert "grader" not in rows["composite"].forms


class TestTheGraderAndTheSelectorCannotDisagree:
    def test_reference_match_grades_the_number_it_thresholds(self) -> None:
        selector = resolved(reference_config(mode="fuzzy", threshold=0.6))
        context = {"reference": "the cat sat on the mat"}
        verdict = selector.plugin.evaluate(
            prompt="", completion="the cat sat on a mat", context=context
        )
        score = grader_for(selector).grade(
            prompt="", completion="the cat sat on a mat", context=context
        )
        assert verdict.score == pytest.approx(score)

    def test_the_judge_grades_the_number_it_thresholds(self, judge_settings) -> None:
        from services.selectors import provider_port

        server = StubJudgeServer(reply="SCORE: 0.42")
        provider_port.bind_provider_resolver(StubProviderResolver(server.provider()))
        selector = resolved(judge_config(threshold=0.6))
        verdict = selector.plugin.evaluate(prompt="q", completion="a", context={})
        score = grader_for(selector).grade(prompt="q", completion="a", context={})
        assert verdict.score == pytest.approx(score) == pytest.approx(0.42)


class TestTheRangeCheck:
    def test_the_range_is_the_contracts_zero_to_one(self) -> None:
        assert SCORE_RANGE == (0.0, 1.0)

    def test_an_out_of_range_value_is_refused_with_the_value_and_the_range(self) -> None:
        """AC-003.20, E-15. Through the real adapter over a plugin that misbehaves."""
        selector = resolved(reference_config(mode="exact", threshold=None))

        class Misbehaving:
            name = "bad-grader"

            def grade(self, *, prompt, completion, context):
                return 1.7

        selector.plugin = Misbehaving()
        with pytest.raises(Refusal) as exc:
            grader_for(selector).grade(prompt="", completion="x", context={})
        assert exc.value.code == "grader_score_out_of_range"
        assert exc.value.details["value"] == 1.7
        assert exc.value.details["range"] == [0.0, 1.0]

    def test_no_clamping_happens(self) -> None:
        selector = resolved(reference_config(mode="exact", threshold=None))

        class Negative:
            name = "bad-grader"

            def grade(self, *, prompt, completion, context):
                return -0.2

        selector.plugin = Negative()
        with pytest.raises(Refusal) as exc:
            grader_for(selector).grade(prompt="", completion="x", context={})
        assert exc.value.details["value"] == -0.2

    @pytest.mark.parametrize("value", [0.0, 0.5, 1.0])
    def test_the_bounds_themselves_are_in_range(self, value: float) -> None:
        selector = resolved(reference_config(mode="exact", threshold=None))

        class Fixed:
            name = "g"

            def grade(self, *, prompt, completion, context):
                return value

        selector.plugin = Fixed()
        assert grader_for(selector).grade(prompt="", completion="x", context={}) == value


class TestRefusingATypeWithNoGraderForm:
    @pytest.mark.parametrize("config", [rule_config(min_length=1)], ids=["rule"])
    def test_a_selector_only_type_refuses_rather_than_synthesising_a_grade(self, config) -> None:
        """A 1.0/0.0 derived from an admission decision is not a grade - and a GRPO group
        of those has zero variance by construction."""
        with pytest.raises(Refusal) as exc:
            grader_for(resolved(config))
        assert exc.value.code == "grader_form_unavailable"
        assert exc.value.details["kind"] == "rule"
        assert exc.value.details["forms"] == ["selector"]


class TestItSatisfiesTheContractProtocol:
    def test_the_adapter_is_a_grader(self) -> None:
        selector = resolved(reference_config(mode="exact", threshold=None))
        assert isinstance(grader_for(selector), Grader)
