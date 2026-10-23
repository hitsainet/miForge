"""The rule selector  (AC-003.3; FR-003.4, E-2).

Determinism and a reason that **names the term or pattern that fired**. A verdict whose
reason is the word "failed" is indistinguishable from a selector that is not really
deciding, which is the first of the two silent failures this feature exists to make visible.
"""

from __future__ import annotations

import json

import pytest

from core.errors import Refusal
from schemas.selector import VerdictValue, is_admitted
from services.selectors.plugins.rule import parse_expression
from tests.support.selector_stubs import resolved, rule_config


def verdict_for(completion: str, **checks):
    selector = resolved(rule_config(**checks))
    return selector.plugin.evaluate(prompt="", completion=completion, context={})


class TestTheReasonNamesWhatFired:
    def test_a_forbidden_term_is_named_in_the_reason(self) -> None:
        verdict = verdict_for("we guarantee results", forbidden_terms=["guarantee"])
        assert verdict.value is VerdictValue.REJECT
        assert verdict.label == "rule_forbidden_term"
        assert verdict.reason == 'forbidden term "guarantee"'

    def test_a_missing_required_term_is_named(self) -> None:
        verdict = verdict_for("no citation here", required_terms=["see also"])
        assert verdict.reason == 'missing required term "see also"'

    def test_a_length_bound_reports_both_numbers(self) -> None:
        verdict = verdict_for("hi", min_length=10)
        assert verdict.label == "rule_too_short"
        assert verdict.reason == "2 characters, minimum 10"

    def test_an_unmatched_pattern_names_the_pattern(self) -> None:
        verdict = verdict_for("nothing numeric", patterns=[{"regex": r"\d+", "must_match": True}])
        assert verdict.label == "rule_pattern_not_matched"
        assert r'no match for pattern "\d+"' == verdict.reason

    def test_a_forbidden_pattern_names_the_pattern(self) -> None:
        verdict = verdict_for(
            "call 555-1234", patterns=[{"regex": r"\d{3}-\d{4}", "must_match": False}]
        )
        assert verdict.label == "rule_pattern_matched"

    def test_a_structure_failure_says_what_structure(self) -> None:
        verdict = verdict_for("not json", structure={"json_parsable": True})
        assert verdict.label == "rule_structure_invalid"
        assert "does not parse as JSON" in (verdict.reason or "")

    def test_an_accept_also_carries_a_reason(self) -> None:
        """AC-003.2: an accept with no reason is indistinguishable from not running."""
        verdict = verdict_for('{"ok": true}', structure={"json_parsable": True})
        assert is_admitted(verdict)
        assert verdict.reason
        assert verdict.label == "rule_passed"


class TestDeterminism:
    def test_the_same_input_gives_the_same_verdict_twice(self) -> None:
        selector = resolved(rule_config(forbidden_terms=["a", "b"], min_length=3)).plugin
        first = selector.evaluate(prompt="", completion="bcd", context={})
        second = selector.evaluate(prompt="", completion="bcd", context={})
        assert (first.value, first.label, first.reason) == (
            second.value,
            second.label,
            second.reason,
        )

    def test_check_order_is_fixed_so_the_reason_is_stable(self) -> None:
        """forbidden -> required -> length -> patterns -> structure.

        A completion failing several checks reports the **first** in that order, every
        time. A reason that depended on dict ordering is a reason a UI cannot map.
        """
        verdict = verdict_for(
            "x",
            forbidden_terms=["x"],
            required_terms=["never"],
            min_length=99,
            patterns=[{"regex": "zzz", "must_match": True}],
        )
        assert verdict.label == "rule_forbidden_term"


class TestCaseSensitivity:
    def test_matching_is_case_sensitive_by_default(self) -> None:
        verdict = verdict_for("We Guarantee", forbidden_terms=["guarantee"])
        assert is_admitted(verdict)

    def test_case_insensitivity_is_explicit(self) -> None:
        verdict = verdict_for("We Guarantee", forbidden_terms=["guarantee"], case_sensitive=False)
        assert verdict.value is VerdictValue.REJECT


class TestRefusalsAtPrepare:
    def test_a_rule_with_no_checks_is_refused(self) -> None:
        """A rule that accepts everything is not selecting - refused at the type."""
        with pytest.raises(Refusal) as exc:
            resolved(
                {
                    "kind": "rule",
                    "name": "empty",
                    "config_sha256": None,
                    "expression": json.dumps({}),
                }
            )
        assert exc.value.code == "rule_expression_invalid"
        assert "accepts every completion" in exc.value.message

    def test_an_unparseable_expression_is_refused_not_treated_as_a_substring(self) -> None:
        with pytest.raises(Refusal) as exc:
            resolved(
                {
                    "kind": "rule",
                    "name": "oops",
                    "config_sha256": None,
                    "expression": "forbidden: guarantee",
                }
            )
        assert exc.value.code == "rule_expression_invalid"
        assert "literal substring" in exc.value.message

    def test_a_misspelled_check_key_is_refused_rather_than_ignored(self) -> None:
        """A silently-ignored key means the operator believes a check is in force."""
        with pytest.raises(Refusal) as exc:
            resolved(rule_config(forbiden_terms=["guarantee"]))
        assert exc.value.code == "rule_expression_invalid"

    def test_impossible_length_bounds_are_refused(self) -> None:
        with pytest.raises(Refusal):
            resolved(rule_config(min_length=10, max_length=5))

    def test_a_pathological_regex_is_refused_at_prepare_not_run_per_completion(self) -> None:
        """FTDD §8: refused at ``prepare``, so it never runs against a completion."""
        with pytest.raises(Refusal) as exc:
            resolved(rule_config(patterns=[{"regex": "(a+)+$", "must_match": True}]))
        assert exc.value.code == "rule_pattern_invalid"
        assert exc.value.details["construct"] == "nested quantifier"

    def test_an_overlong_pattern_is_refused_with_the_bound_named(self) -> None:
        with pytest.raises(Refusal) as exc:
            resolved(rule_config(patterns=[{"regex": "a" * 600, "must_match": True}]))
        assert exc.value.code == "rule_pattern_invalid"
        assert exc.value.details["limit"] == 512

    def test_an_uncompilable_pattern_is_refused_not_read_as_a_literal(self) -> None:
        with pytest.raises(Refusal) as exc:
            resolved(rule_config(patterns=[{"regex": "a[", "must_match": True}]))
        assert exc.value.code == "rule_pattern_invalid"

    def test_must_match_has_no_default(self) -> None:
        """ "regex" alone does not say which way; guessing picks a verdict policy."""
        with pytest.raises(Refusal):
            resolved(rule_config(patterns=[{"regex": "x"}]))


class TestTheExpressionParser:
    def test_a_non_object_expression_is_refused(self) -> None:
        with pytest.raises(Refusal) as exc:
            parse_expression("[1, 2, 3]")
        assert exc.value.details["parsed_type"] == "list"

    def test_min_lines_counts_only_non_empty_lines(self) -> None:
        verdict = verdict_for("one\n\n\n", structure={"min_lines": 2})
        assert verdict.value is VerdictValue.REJECT
        assert "1 non-empty lines" in (verdict.reason or "")
