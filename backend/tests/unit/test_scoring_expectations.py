"""Mapping R-02.16's three forms onto F3  (FR-004.13; E-9, E-10, E-11; AC-004.26, AC-004.27).

Three refusals carry this file, and each exists because the alternative is a confident number:

* **``human_review`` as a test-set expectation** (E-10). It writes candidates to a queue and
  reads decisions back, so it cannot answer at a step. F3's verdict being **three-valued** is
  what makes this detectable rather than a hang.
* **embedding-mode reference matching** (E-9). Refused by the **contract**, at validation.
* **any ``undecided``** (E-11). Never counted as a reject to make the arithmetic close.

F4 implements no selector, no grader and no matcher: every verdict here comes from F3's live
registry, and a test asserts the package constructs nothing of its own.
"""

from __future__ import annotations

import pytest

from core.errors import Refusal
from schemas.selector import VerdictValue
from services.scoring import expectations
from services.selectors import provider_port
from tests.support.selector_stubs import StubJudgeServer, StubProviderResolver


def rule_config(name: str = "len", **body):
    """A rule selector - pure, in-process, no model and no network."""
    config = {
        "kind": "rule",
        "name": name,
        "config_sha256": None,
        # `expression` is a STRING holding JSON - F3's contract gives this type one opaque
        # field and its plugin parses it. Written the contract's way rather than as a dict,
        # because a fixture that disagrees with the schema is a fixture testing nothing.
        "expression": '{"min_length": 5}',
    }
    config.update(body)
    return config


def reference_config(mode: str = "exact", **body):
    config = {
        "kind": "reference_match",
        "name": "ref",
        "config_sha256": None,
        "mode": mode,
    }
    if mode == "fuzzy":
        config["threshold"] = 0.8
    config.update(body)
    return config


class TestAHumanReviewExpectationIsRefusedUpFront:
    """E-10, AC-004.26: **refused by name, not left to hang.**"""

    def test_a_direct_human_review_expectation_refuses(self) -> None:
        with pytest.raises(Refusal) as caught:
            expectations.resolve_expectation(
                {
                    "kind": "human_review",
                    "name": "people",
                    "config_sha256": None,
                    "question": "is this good?",
                    "label_set": ["yes", "no"],
                },
                source="pack",
            )
        assert caught.value.code == "expectation_not_synchronous"
        assert caught.value.details["type"] == "human_review"

    def test_a_composite_with_a_human_child_refuses_and_names_where(self) -> None:
        """**The one a reader would miss.** A composite containing a human child cannot
        answer at a step either, and the refusal has to say which child so the operator can
        act on it."""
        catalog = {
            "people": {
                "kind": "human_review",
                "name": "people",
                "config_sha256": None,
                "question": "q",
                "label_set": ["a", "b"],
            },
            "len": rule_config(),
        }
        with pytest.raises(Refusal) as caught:
            expectations.resolve_expectation(
                {
                    "kind": "composite",
                    "name": "both",
                    "config_sha256": None,
                    "mode": "all",
                    "children": ["len", "people"],
                },
                catalog=catalog,
                source="pack",
            )
        assert caught.value.code == "expectation_not_synchronous"
        assert "both" in caught.value.details["at"]

    def test_the_refusal_says_what_the_type_IS_for(self) -> None:
        """ADR-021: the message says what to do next."""
        with pytest.raises(Refusal) as caught:
            expectations.resolve_expectation(
                {
                    "kind": "human_review",
                    "name": "people",
                    "config_sha256": None,
                    "question": "q",
                    "label_set": ["a", "b"],
                },
                source="pack",
            )
        assert "signal-source selector" in caught.value.message


class TestEmbeddingModeIsRefusedByTheContract:
    """E-9, AC-004.27, and a **divergence from this feature's documents**.

    F4's FPRD, FTDD, FTID and FTASKS all say this surfaces F3's ``embeddings_unsupported``
    code. **There is no such code**: F3's ``reference_match`` module says in terms *"There is
    no ``embeddings_unsupported`` code here and no embeddings capability on the provider
    port, because a refusal code for a mode the contract does not define is a refusal nothing
    can produce."* The schema refuses the mode first, which is strictly more useful - it names
    the ``/mode`` pointer - and the property that matters holds either way: **nothing falls
    back to fuzzy.**
    """

    def test_embedding_mode_refuses(self) -> None:
        with pytest.raises(Refusal) as caught:
            expectations.resolve_expectation(reference_config(mode="embedding"), source="pack")
        assert caught.value.code == "selector_config_invalid"

    def test_it_does_not_fall_back_to_fuzzy(self) -> None:
        """*A selector that silently fell back to fuzzy matching when asked for embeddings
        would be the fabricated value ADR-018 forbids* - the schema's own words."""
        with pytest.raises(Refusal):
            expectations.resolve_expectation(reference_config(mode="embedding"), source="pack")

    def test_f4_adds_no_embeddings_capability(self) -> None:
        """DEF-3: R-02.16 asks for *"exact or fuzzy match"*, so F4 needs none."""
        from providers import registry

        for row in registry.list_rows():
            assert "embeddings" not in row.capabilities

    def test_exact_and_fuzzy_both_resolve(self) -> None:
        for mode in ("exact", "fuzzy"):
            resolved = expectations.resolve_expectation(reference_config(mode=mode), source="pack")
            assert resolved.kind == "reference_match"


class TestScoringThroughF3sRegistry:
    def test_a_selector_verdict_becomes_an_admitted_indicator(self) -> None:
        resolved = expectations.resolve_expectation(rule_config(), source="pack")
        scored = expectations.score_completions(resolved, ["p", "p"], ["long enough", "no"])
        assert [outcome.outcome for outcome in scored.outcomes] == [1.0, 0.0]
        assert [outcome.verdict for outcome in scored.outcomes] == ["accept", "reject"]

    def test_the_selector_hash_f3_published_is_carried(self) -> None:
        """F3's FR-003.16 values, so a score names the configuration that judged it."""
        resolved = expectations.resolve_expectation(rule_config(), source="pack")
        scored = expectations.score_completions(resolved, ["p"], ["long enough"])
        assert scored.selector_hash is not None

    def test_a_grader_scalar_is_used_as_the_outcome(self) -> None:
        resolved = expectations.resolve_expectation(
            reference_config(mode="fuzzy"), form=expectations.FORM_GRADER, source="pack"
        )
        scored = expectations.score_completions(
            resolved, ["p"], ["hello world"], references=["hello world"]
        )
        assert scored.outcomes[0].outcome == pytest.approx(1.0)

    def test_a_form_the_type_does_not_offer_refuses(self) -> None:
        """*A rule's 1.0/0.0 admission flag read as a graded scalar is a different
        measurement wearing the same number.*"""
        with pytest.raises(Refusal) as caught:
            expectations.resolve_expectation(
                rule_config(), form=expectations.FORM_GRADER, source="pack"
            )
        assert caught.value.code == "expectation_form_unsupported"

    def test_an_unknown_form_refuses(self) -> None:
        with pytest.raises(Refusal):
            expectations.resolve_expectation(rule_config(), form="vibes", source="pack")


class TestUndecidedIsNeverCountedAsAReject:
    """E-11. *Counting an undecided as a reject to make the arithmetic close is a fabricated
    number with extra steps.*"""

    def test_an_undecided_verdict_refuses_the_evaluation_with_the_counts(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        resolved = expectations.resolve_expectation(rule_config(), source="pack")

        from schemas.selector import Verdict

        real = resolved.resolved.plugin.evaluate

        def sometimes_undecided(*, prompt, completion, context):
            if context["index"] == 1:
                return Verdict(value=VerdictValue.UNDECIDED, selector="len", label="pending")
            return real(prompt=prompt, completion=completion, context=context)

        monkeypatch.setattr(resolved.resolved.plugin, "evaluate", sometimes_undecided)

        with pytest.raises(Refusal) as caught:
            expectations.score_completions(resolved, ["a", "b"], ["long enough", "also long"])
        assert caught.value.code == "evaluation_incomplete"
        assert caught.value.details["undecided"] == 1
        assert caught.value.details["total"] == 2
        assert caught.value.details["prompt_indexes"] == [1]

    def test_the_refusal_says_why_coercion_is_forbidden(self) -> None:
        resolved = expectations.resolve_expectation(rule_config(), source="pack")
        from schemas.selector import Verdict

        resolved.resolved.plugin.evaluate = lambda **kw: Verdict(  # type: ignore[method-assign]
            value=VerdictValue.UNDECIDED, selector="len", label="pending"
        )
        with pytest.raises(Refusal) as caught:
            expectations.score_completions(resolved, ["a"], ["x"])
        assert "half-answered set" in caught.value.message


class TestF3sRefusalsPropagateUnchanged:
    def test_a_judge_transport_failure_keeps_f3s_code(self) -> None:
        """*A re-coded refusal loses its owner* - the re-coding that does happen is in
        :mod:`services.scoring.service`, which keeps the original in ``details``."""
        import httpx

        server = StubJudgeServer(raise_error=httpx.ConnectError("down"))
        provider_port.bind_provider_resolver(StubProviderResolver(server.provider()))
        try:
            resolved = expectations.resolve_expectation(
                {
                    "kind": "judge_model",
                    "name": "j",
                    "config_sha256": None,
                    "provider": "judge",
                    "threshold": 0.6,
                },
                source="pack",
            )
        except Refusal as caught:
            # No judge template configured is itself a legitimate named refusal; either way
            # the point is that F3's code arrives unchanged.
            assert caught.code.startswith("judge_")
            return
        finally:
            provider_port.bind_provider_resolver(None)
        with pytest.raises(Refusal) as caught:
            expectations.score_completions(resolved, ["p"], ["c"])
        assert caught.value.code.startswith("judge_")

    def test_an_invalid_configuration_keeps_f3s_pointer(self) -> None:
        with pytest.raises(Refusal) as caught:
            expectations.resolve_expectation(
                {"kind": "rule", "name": "x", "config_sha256": None}, source="pack"
            )
        assert caught.value.code == "selector_config_invalid"
        assert "pointer" in caught.value.details
