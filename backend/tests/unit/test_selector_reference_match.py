"""The reference-match selector  (FR-003.6; R-02.9, E-8 as the contract settled it).

**Embedding similarity is CUT, and this file is where that stays true.** The FPRD described
three modes and an ``embeddings_unsupported`` refusal; the settled contract's ``mode`` enum
is ``["exact", "fuzzy"]``, because R-02.49's provider abstraction is scoped to generation
and the word *"embedding"* occurs **exactly once in the whole of BRD-02** - in the amendment
that offered it. No provider, no source, no mechanism.

So the test for E-8 is not *"embedding mode refuses ``embeddings_unsupported``"* - that
refusal would be dead code guarding a mode the contract cannot express. It is
:func:`test_embedding_mode_is_refused_by_the_schema_itself`, which is the stronger
statement: a configuration asking for it never reaches this module.

**And there is no fuzzy fallback.** A silent mode change would alter every verdict while the
R-02.10 configuration hash stayed identical - the worst available outcome for a feature
whose job is that selection is observable.
"""

from __future__ import annotations

import pytest

from core.errors import Refusal
from schemas.selector import VerdictValue
from services.selectors.plugins.reference_match import FUZZY_METRIC, references_from
from tests.support.selector_stubs import reference_config, resolved


def verdict_for(completion: str, reference, *, mode: str = "fuzzy", threshold=0.6):
    selector = resolved(reference_config(mode=mode, threshold=threshold))
    context = {"reference": reference} if reference is not None else {}
    return selector.plugin.evaluate(prompt="", completion=completion, context=context)


class TestExactMode:
    def test_an_identical_string_is_accepted(self) -> None:
        verdict = verdict_for("exactly this", "exactly this", mode="exact", threshold=None)
        assert verdict.value is VerdictValue.ACCEPT
        assert verdict.score == 1.0
        assert verdict.reason == "exact match to reference"

    def test_a_different_string_is_rejected(self) -> None:
        verdict = verdict_for("almost this", "exactly this", mode="exact", threshold=None)
        assert verdict.value is VerdictValue.REJECT
        assert verdict.score == 0.0

    @pytest.mark.parametrize(
        "completion",
        ["Exactly This", " exactly this", "exactly this "],
        ids=["case", "leading-space", "trailing-space"],
    )
    def test_exact_means_exact_with_no_normalisation(self, completion: str) -> None:
        """*"exact after case-folding is a different verdict from exact."* The contract
        gives this type no normalisation field, so choosing one here would be a verdict
        policy chosen by whoever wrote the line."""
        verdict = verdict_for(completion, "exactly this", mode="exact", threshold=None)
        assert verdict.value is VerdictValue.REJECT

    def test_a_threshold_in_exact_mode_is_refused_rather_than_ignored(self) -> None:
        """An operator who set a threshold expects tolerance; silently giving them
        byte-equality would reject every near-miss while the configuration said otherwise."""
        with pytest.raises(Refusal) as exc:
            resolved(reference_config(mode="exact", threshold=0.8))
        assert exc.value.code == "reference_match_threshold_not_applicable"
        assert exc.value.details["threshold"] == 0.8


class TestFuzzyMode:
    def test_a_near_match_above_the_threshold_is_accepted(self) -> None:
        verdict = verdict_for("the cat sat on a mat", "the cat sat on the mat", threshold=0.6)
        assert verdict.value is VerdictValue.ACCEPT

    def test_a_poor_match_is_rejected_and_the_reason_names_both_numbers(self) -> None:
        verdict = verdict_for("banana", "the cat sat on the mat", threshold=0.6)
        assert verdict.value is VerdictValue.REJECT
        assert "no reference within 0.60" in (verdict.reason or "")
        assert "best 0." in (verdict.reason or "")

    def test_a_missing_threshold_is_refused_not_defaulted(self) -> None:
        """The similarity score means nothing without the line it is compared against."""
        with pytest.raises(Refusal) as exc:
            resolved(reference_config(mode="fuzzy", threshold=None))
        assert exc.value.code == "reference_match_threshold_missing"
        assert exc.value.details["metric"] == FUZZY_METRIC


class TestTheMetricIsPartOfTheVerdict:
    def test_the_metric_is_named_in_the_reason(self) -> None:
        """Swapping it rescopes every fuzzy decision, so it is recorded, not implied."""
        verdict = verdict_for("close enough", "close enought", threshold=0.6)
        assert FUZZY_METRIC in (verdict.reason or "")

    def test_the_metric_is_pinned_to_one_named_implementation(self) -> None:
        assert FUZZY_METRIC == "difflib.SequenceMatcher.ratio"

    def test_the_mode_is_on_every_verdict_label(self) -> None:
        """A verdict is not comparable across modes."""
        fuzzy = verdict_for("a", "a", mode="fuzzy", threshold=0.5)
        exact = verdict_for("a", "a", mode="exact", threshold=None)
        assert "fuzzy" in (fuzzy.label or "")
        assert "exact" in (exact.label or "")


class TestEmbeddingModeIsCut:
    def test_embedding_mode_is_refused_by_the_schema_itself(self) -> None:
        """The stronger form of E-8: the configuration never reaches this module."""
        config = reference_config(mode="embedding", threshold=0.8)
        with pytest.raises(Refusal) as exc:
            resolved(config)
        assert exc.value.code == "selector_config_invalid"
        assert "/mode" in exc.value.details["pointer"]

    @pytest.mark.parametrize("mode", ["exact", "fuzzy"])
    def test_the_mode_in_force_is_always_the_mode_configured(self, mode: str) -> None:
        """**The behavioural form of "no fallback".**

        A silent mode substitution would alter every verdict while the R-02.10
        configuration hash stayed identical - the worst available outcome for this
        feature. So the assertion is that the mode recorded on the resolution, the mode the
        plugin reports and the mode in the configuration are the same string, with no code
        path that can change one of them.
        """
        threshold = 0.6 if mode == "fuzzy" else None
        selector = resolved(reference_config(mode=mode, threshold=threshold))
        assert selector.body["mode"] == mode
        assert selector.plugin.mode == mode
        assert selector.prepared["mode"] == mode

    def test_the_contract_declares_exactly_two_modes(self) -> None:
        import json
        from pathlib import Path

        schema = json.loads(
            (
                Path(__file__).resolve().parents[3] / "docs" / "schemas" / "selector-config-v1.json"
            ).read_text()
        )
        modes = schema["$defs"]["ReferenceMatchSelector"]["properties"]["mode"]["enum"]
        assert modes == ["exact", "fuzzy"]


class TestTheReferenceComesFromTheItem:
    def test_a_missing_reference_refuses_rather_than_rejecting(self) -> None:
        """A reject would discard the completion and look like the model failing a test it
        was never given."""
        with pytest.raises(Refusal) as exc:
            verdict_for("anything", None)
        assert exc.value.code == "reference_missing"
        assert "reference" in exc.value.details["looked_for"]

    @pytest.mark.parametrize("key", ["reference", "reference_answer", "references"])
    def test_both_spellings_are_accepted_and_neither_is_a_guess(self, key: str) -> None:
        assert references_from({key: "answer"}) == ("answer",)

    def test_a_list_of_references_matches_any_of_them(self) -> None:
        """A test item may have several acceptable answers."""
        verdict = verdict_for("beta", ["alpha", "beta", "gamma"], mode="exact", threshold=None)
        assert verdict.value is VerdictValue.ACCEPT

    def test_the_best_score_among_several_is_the_one_used(self) -> None:
        verdict = verdict_for(
            "the cat sat on a mat", ["banana", "the cat sat on the mat"], threshold=0.6
        )
        assert verdict.value is VerdictValue.ACCEPT

    def test_an_empty_context_yields_no_references(self) -> None:
        assert references_from({}) == ()


class TestItNeedsNothingConfigured:
    def test_it_is_available_with_no_sibling_and_no_endpoint(self) -> None:
        """One of the four types that work on the standalone path (AC20)."""
        from services.selectors import registry
        from services.selectors.settings import SelectorSettings

        rows = {
            row.name: row for row in registry.list_capabilities(SelectorSettings(_env_file=None))
        }
        assert rows["reference_match"].available is True
        assert rows["reference_match"].requires == "none"
