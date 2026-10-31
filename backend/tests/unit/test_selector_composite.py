"""The composite selector  (AC-003.12, AC-003.13; FR-003.8, E-13, E-14).

**``undecided`` propagates, and this is the one place the three-valued logic is non-obvious**
- which is why the contract states the rule rather than leaving it to an implementer:
*"``all`` with one undecided child cannot be accept (the child may yet reject) and must not
be reject (nobody rejected)."*

The truth tables below are the whole point of this file. Every row is a case where
collapsing three values into two would produce a plausible-looking wrong answer.

**Short-circuit is a correctness feature and a kindness**: an ``all`` composite of a rule and
a human review rejects on a forbidden term **without posting the candidate to a person**,
and that is asserted by a queue call count of zero.
"""

from __future__ import annotations

import pytest

from core.errors import Refusal
from schemas.selector import VerdictValue
from services.selectors.plugins import human_review
from services.selectors.plugins.composite import order_children
from tests.support.selector_stubs import (
    StubReviewQueue,
    composite_config,
    human_config,
    reference_config,
    resolved,
    rule_config,
)

#: A rule that accepts anything at least one character long.
ACCEPTS = rule_config("accepts", min_length=1)
#: A rule that rejects anything at least one character long.
REJECTS = rule_config("rejects", required_terms=["__never_present__"])


def verdict(combine: str, children: list[dict], completion: str = "x", **context):
    names = [child["name"] for child in children]
    selector = resolved(composite_config("c", names, combine), catalog=children)
    return selector.plugin.evaluate(prompt="", completion=completion, context=context)


class TestAllCombine:
    def test_every_child_accepting_accepts(self) -> None:
        assert verdict("all", [ACCEPTS, rule_config("a2", min_length=1)]).value is (
            VerdictValue.ACCEPT
        )

    def test_one_child_rejecting_rejects_and_names_it(self) -> None:
        result = verdict("all", [ACCEPTS, REJECTS])
        assert result.value is VerdictValue.REJECT
        assert "rejects" in (result.reason or "")

    def test_it_short_circuits_at_the_first_reject(self) -> None:
        """Cheapest-first ordering puts the rule before the human, so the queue is untouched."""
        queue = StubReviewQueue()
        human_review.bind_review_queue(queue)
        try:
            result = verdict("all", [REJECTS, human_config("human")])
            assert result.value is VerdictValue.REJECT
            assert queue.post_count == 0, (
                "an AND composite must reject on a rule WITHOUT posting anybody's "
                "candidate - nobody should be asked to read something a rule settled"
            )
        finally:
            human_review.bind_review_queue(None)


class TestAnyCombine:
    def test_one_child_accepting_accepts(self) -> None:
        result = verdict("any", [REJECTS, ACCEPTS])
        assert result.value is VerdictValue.ACCEPT

    def test_every_child_rejecting_rejects(self) -> None:
        result = verdict("any", [REJECTS, rule_config("r2", required_terms=["__never_present__"])])
        assert result.value is VerdictValue.REJECT
        assert "all 2 children rejected" in (result.reason or "")

    def test_it_short_circuits_at_the_first_accept(self) -> None:
        queue = StubReviewQueue()
        human_review.bind_review_queue(queue)
        try:
            assert verdict("any", [ACCEPTS, human_config("human")]).value is VerdictValue.ACCEPT
            assert queue.post_count == 0
        finally:
            human_review.bind_review_queue(None)


class TestUndecidedPropagates:
    """E-14, and the rule the contract spells out."""

    @pytest.fixture
    def pending_queue(self):
        queue = StubReviewQueue()  # no decisions -> every candidate is undecided
        human_review.bind_review_queue(queue)
        yield queue
        human_review.bind_review_queue(None)

    def test_all_with_one_undecided_child_is_undecided_not_accept(self, pending_queue) -> None:
        """*"cannot be accept - the child may yet reject."*"""
        result = verdict("all", [ACCEPTS, human_config("human")])
        assert result.value is VerdictValue.UNDECIDED

    def test_all_with_one_undecided_child_is_undecided_not_reject(self, pending_queue) -> None:
        """*"must not be reject - nobody rejected."*"""
        result = verdict("all", [ACCEPTS, human_config("human")])
        assert result.value is not VerdictValue.REJECT

    def test_the_reason_names_the_pending_child(self, pending_queue) -> None:
        result = verdict("all", [ACCEPTS, human_config("human")])
        assert "waiting on human" in (result.reason or "")
        assert "1 of 2 accepted" in (result.reason or "")

    def test_any_with_one_undecided_and_the_rest_rejecting_is_undecided(
        self, pending_queue
    ) -> None:
        result = verdict("any", [REJECTS, human_config("human")])
        assert result.value is VerdictValue.UNDECIDED

    def test_an_undecided_composite_carries_no_score(self, pending_queue) -> None:
        """The contract refuses the combination at the type: a score on a pending candidate
        is how a consumer reconstructs the forbidden path to admitted."""
        result = verdict("all", [ACCEPTS, human_config("human")])
        assert result.score is None


class TestMajorityCombine:
    """``majority`` is what the contract has instead of the FPRD's weighted sum."""

    def test_a_strict_majority_accepting_accepts(self) -> None:
        children = [ACCEPTS, rule_config("a2", min_length=1), REJECTS]
        result = verdict("majority", children)
        assert result.value is VerdictValue.ACCEPT
        assert "2 of 3" in (result.reason or "")

    def test_a_strict_majority_rejecting_rejects(self) -> None:
        children = [REJECTS, rule_config("r2", required_terms=["__never_present__"]), ACCEPTS]
        result = verdict("majority", children)
        assert result.value is VerdictValue.REJECT

    def test_it_short_circuits_once_the_outcome_cannot_change(self) -> None:
        queue = StubReviewQueue()
        human_review.bind_review_queue(queue)
        try:
            children = [ACCEPTS, rule_config("a2", min_length=1), human_config("human")]
            assert verdict("majority", children).value is VerdictValue.ACCEPT
            assert queue.post_count == 0, "2 of 3 accepted settles it; do not ask a person"
        finally:
            human_review.bind_review_queue(None)

    def test_a_pending_child_that_could_still_tip_it_leaves_it_undecided(self) -> None:
        queue = StubReviewQueue()
        human_review.bind_review_queue(queue)
        try:
            children = [ACCEPTS, REJECTS, human_config("human")]
            result = verdict("majority", children)
            assert result.value is VerdictValue.UNDECIDED
        finally:
            human_review.bind_review_queue(None)

    def test_a_decided_deadlock_rejects_and_says_so_with_the_counts(self) -> None:
        """Every child decided, no majority either way: a **decided** outcome, because
        nothing further will arrive and nothing admitted it."""
        children = [
            ACCEPTS,
            rule_config("a2", min_length=1),
            REJECTS,
            rule_config("r2", required_terms=["__never_present__"]),
        ]
        result = verdict("majority", children)
        assert result.value is VerdictValue.REJECT
        assert "no majority (2 of 4 accepted, needs 3)" in (result.reason or "")


class TestCycles:
    def test_a_direct_self_reference_is_refused_naming_the_cycle(self) -> None:
        """AC-003.13, E-13. Refused at **resolution**, before any completion."""
        config = composite_config("loop", ["loop"], "all")
        with pytest.raises(Refusal) as exc:
            resolved(config, catalog=[config])
        assert exc.value.code == "composite_cycle"
        assert exc.value.details["cycle"] == ["loop", "loop"]

    def test_an_indirect_cycle_is_refused_naming_the_path(self) -> None:
        outer = composite_config("outer", ["inner"], "all")
        inner = composite_config("inner", ["outer"], "all")
        with pytest.raises(Refusal) as exc:
            resolved(outer, catalog=[outer, inner])
        assert exc.value.code == "composite_cycle"
        assert exc.value.details["cycle"] == ["outer", "inner", "outer"]

    def test_a_shared_child_is_not_mistaken_for_a_cycle(self) -> None:
        """A diamond is legal: the visiting set is a path, not a visited set.

        A guard that cried wolf on a shared child would get switched off, and then it
        protects nothing.
        """
        left = composite_config("left", ["accepts"], "all")
        right = composite_config("right", ["accepts"], "all")
        top = composite_config("top", ["left", "right"], "all")
        selector = resolved(top, catalog=[top, left, right, ACCEPTS])
        assert len(selector.children) == 2


class TestResolutionRefusals:
    def test_an_unknown_child_is_refused_naming_what_is_available(self) -> None:
        config = composite_config("c", ["nope"], "all")
        with pytest.raises(Refusal) as exc:
            resolved(config, catalog=[config, ACCEPTS])
        assert exc.value.code == "composite_child_unknown"
        assert "accepts" in exc.value.details["available"]

    def test_an_empty_children_list_is_refused_by_the_schema(self) -> None:
        """``minItems: 1`` - so this never reaches ``prepare``."""
        config = composite_config("c", [], "all")
        with pytest.raises(Refusal) as exc:
            resolved(config, catalog=[config])
        assert exc.value.code == "selector_config_invalid"

    def test_an_unknown_combine_is_refused_by_the_schema(self) -> None:
        config = composite_config("c", ["accepts"], "weighted_sum")
        with pytest.raises(Refusal) as exc:
            resolved(config, catalog=[config, ACCEPTS])
        assert exc.value.code == "selector_config_invalid"
        assert "/combine" in exc.value.details["pointer"]


class TestCheapestFirstOrdering:
    def test_children_are_ordered_cheapest_first(self) -> None:
        """So ``rule`` runs before ``judge_model`` and both before ``human_review``."""
        human_review.bind_review_queue(StubReviewQueue())
        try:
            children = [
                human_config("h"),
                reference_config("r", mode="exact", threshold=None),
                rule_config("ru", min_length=1),
            ]
            selector = resolved(composite_config("c", ["h", "r", "ru"], "all"), catalog=children)
        finally:
            human_review.bind_review_queue(None)
        assert [child.kind for child in selector.children] == [
            "rule",
            "reference_match",
            "human_review",
        ]

    def test_equal_cost_children_keep_their_configured_order(self) -> None:
        """Stability matters: the reason names the deciding child, and a reason that
        depended on sort instability would change between runs over one input."""
        a = rule_config("first", min_length=1)
        b = rule_config("second", min_length=1)
        selector = resolved(composite_config("c", ["first", "second"], "all"), catalog=[a, b])
        assert [child.name for child in selector.children] == ["first", "second"]

    def test_order_children_is_pure_and_testable_on_its_own(self) -> None:
        selector = resolved(
            composite_config("c", ["accepts", "rejects"], "all"), catalog=[ACCEPTS, REJECTS]
        )
        assert order_children(selector.children) == selector.children


class TestCaveatsAreInherited:
    def test_a_childs_caveat_becomes_the_composites_caveat(self) -> None:
        """A judge sharing the model's lineage three levels down still bears on every
        verdict the composite produces."""
        selector = resolved(composite_config("c", ["accepts"], "all"), catalog=[ACCEPTS])
        selector.children[0].caveats = ("judge_shares_lineage",)
        selector.plugin.prepare(selector)
        assert "judge_shares_lineage" in selector.caveats
