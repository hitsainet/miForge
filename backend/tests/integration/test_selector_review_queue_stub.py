"""In-run review against a stub review API  (AC-003.9, AC-003.11; FR-003.7, E-10, E-11).

**Four scenarios the FPRD names:** post, an idempotent repost of the same ``external_id``, a
partially decided queue, and an unreachable service. Plus the one that matters most for
correctness: a batch that stays undecided past its deadline and is **never** converted to
accept.

This is the **one data-egress path in this feature** - prompt and completion text leaving
the process - so the payload is asserted field by field with the call count. *A privacy
guarantee verified by reading is a guarantee verified by nothing.*
"""

from __future__ import annotations

import pytest

from core.errors import Refusal
from services.selectors.plugins import human_review
from services.selectors.plugins.human_review import derive_external_id
from services.selectors.service import evaluate
from services.selectors.settings import reset_selector_settings
from services.selectors.types import EvaluationContext
from tests.support.selector_stubs import (
    StubReviewQueue,
    composite_config,
    human_config,
    resolved,
    rule_config,
)


@pytest.fixture(autouse=True)
def _clean():
    reset_selector_settings()
    yield
    human_review.bind_review_queue(None)
    reset_selector_settings()


def run_batch(queue: StubReviewQueue, completions, *, batch_id="b1", timeout=None):
    human_review.bind_review_queue(queue)
    selector = resolved(human_config(timeout_seconds=timeout))
    return evaluate(
        completions,
        selector,
        EvaluationContext(batch_id=batch_id, run_id="run-1"),
        prompts=[f"Q{i}" for i in range(len(completions))],
    )


class TestPostingAndReading:
    def test_a_batch_is_posted_once_and_every_field_is_asserted(self) -> None:
        queue = StubReviewQueue()
        result = run_batch(queue, ["A0", "A1", "A2"])

        assert queue.post_count == 1, "one post for the batch, not one per candidate"
        assert len(queue.all_posted) == 3
        for index, candidate in enumerate(queue.all_posted):
            assert candidate.prompt == f"Q{index}"
            assert candidate.completion == f"A{index}"
            assert candidate.external_id == derive_external_id("b1", index, f"A{index}")
            assert candidate.provenance == {"selector": "human", "batch_id": "b1"}

        assert all(record.verdict.value.value == "undecided" for record in result.records)

    def test_the_whole_batch_comes_back_undecided_and_counted(self) -> None:
        queue = StubReviewQueue()
        result = run_batch(queue, ["A"] * 6)
        assert result.health.undecided == 6
        assert result.health.accepted == 0
        assert result.health.acceptance_rate == 0.0

    def test_an_undecided_batch_raises_no_degenerate_signal(self) -> None:
        """Nothing is wrong: a queue nobody has read yet is a normal state, not a
        selector that rejects everything."""
        queue = StubReviewQueue()
        result = run_batch(queue, ["A"] * 6)
        assert result.health.signals == ()

    def test_each_record_carries_its_queue_item_for_reconciliation(self) -> None:
        queue = StubReviewQueue()
        result = run_batch(queue, ["A0", "A1"])
        item_ids = set(queue.items.values())
        for record in result.records:
            assert record.review_item_id in item_ids

    def test_decisions_are_read_back_by_the_ids_that_were_posted(self) -> None:
        queue = StubReviewQueue()
        run_batch(queue, ["A0", "A1"])
        assert queue.read_calls == [
            [derive_external_id("b1", 0, "A0"), derive_external_id("b1", 1, "A1")]
        ]


class TestIdempotentRepost:
    def test_a_second_pass_over_the_same_batch_creates_no_duplicate(self) -> None:
        """miDataworks returns the existing item for a repeated ``external_id`` - verified -
        so a retried post cannot ask a person to read the same candidate twice."""
        queue = StubReviewQueue()
        run_batch(queue, ["A0", "A1"])
        run_batch(queue, ["A0", "A1"])
        assert queue.post_count == 2, "two posts were made"
        assert len(queue.items) == 2, "but only two items exist"

    def test_a_second_pass_picks_up_a_decision_made_in_between(self) -> None:
        queue = StubReviewQueue()
        first = run_batch(queue, ["A0", "A1"])
        assert [record.verdict.value.value for record in first.records] == [
            "undecided",
            "undecided",
        ]

        queue.decide(derive_external_id("b1", 0, "A0"), "accept")
        second = run_batch(queue, ["A0", "A1"])
        assert [record.verdict.value.value for record in second.records] == [
            "accept",
            "undecided",
        ]


class TestAPartiallyDecidedQueue:
    def test_the_health_record_counts_all_three_states(self) -> None:
        queue = StubReviewQueue()
        run_batch(queue, ["A0", "A1", "A2", "A3", "A4"])
        queue.decide(derive_external_id("b1", 0, "A0"), "accept")
        queue.decide(derive_external_id("b1", 1, "A1"), "reject")
        queue.decide(derive_external_id("b1", 2, "A2"), "override", label="needs-citation")

        result = run_batch(queue, ["A0", "A1", "A2", "A3", "A4"])
        assert [record.verdict.value.value for record in result.records] == [
            "accept",
            "reject",
            "undecided",
            "undecided",
            "undecided",
        ]
        assert (result.health.accepted, result.health.rejected, result.health.undecided) == (
            1,
            1,
            3,
        )

    def test_an_override_is_undecided_and_keeps_its_label(self) -> None:
        """Neither admitted nor counted as rejected. Their ``label_set`` is 2-64 **free
        strings**, so reading "override" as an admission would train on an answer nobody
        gave."""
        queue = StubReviewQueue()
        run_batch(queue, ["A0"])
        queue.decide(derive_external_id("b1", 0, "A0"), "override", label="partially-correct")
        result = run_batch(queue, ["A0"])
        verdict = result.records[0].verdict
        assert verdict.value.value == "undecided"
        assert verdict.label == "partially-correct"

    def test_decision_ids_are_available_for_the_batch_record(self) -> None:
        """R-02.36's last sentence: the decision identifier travels with the batch."""
        queue = StubReviewQueue()
        human_review.bind_review_queue(queue)
        selector = resolved(human_config())
        selector.plugin.prefetch(["Q"], ["A"], batch_id="b1")
        queue.decide(derive_external_id("b1", 0, "A"), "accept")
        selector.plugin.prefetch(["Q"], ["A"], batch_id="b1")
        ids = selector.plugin.decision_ids()
        assert list(ids.values()) == [f"d-{derive_external_id('b1', 0, 'A')}"]


class TestTheDeadlineNeverAdmits:
    @pytest.mark.parametrize("timeout", [0, 1, 3600], ids=["zero", "one-second", "an-hour"])
    def test_whatever_the_timeout_an_undecided_candidate_stays_undecided(
        self, timeout: int
    ) -> None:
        """AC-003.11, E-10, E-11. There is no code path from a deadline to an accept.

        Defaulting silence to accept would turn an unattended queue into an unselected run;
        defaulting it to reject would discard every candidate a person had not reached yet.
        """
        queue = StubReviewQueue()
        result = run_batch(queue, ["A"] * 3, timeout=timeout)
        assert all(record.verdict.value.value == "undecided" for record in result.records)
        assert result.health.accepted == 0

    def test_no_undecided_record_is_admitted_by_is_admitted(self) -> None:
        from schemas.selector import is_admitted

        queue = StubReviewQueue()
        result = run_batch(queue, ["A"] * 3, timeout=0)
        assert not any(is_admitted(record.verdict) for record in result.records)


class TestAnUnreachableService:
    def test_a_configured_but_unreachable_queue_refuses_the_batch(self) -> None:
        queue = StubReviewQueue(unreachable=True)
        with pytest.raises(Refusal) as exc:
            run_batch(queue, ["A"] * 3)
        assert exc.value.code == "midataworks_unreachable"

    def test_no_verdict_is_defaulted_when_the_service_is_unreachable(self) -> None:
        """Not accept, not reject, not undecided-as-a-silent-shrug: a named refusal, so the
        operator sees an outage rather than a quality signal."""
        queue = StubReviewQueue(unreachable=True)
        with pytest.raises(Refusal):
            run_batch(queue, ["A"] * 3)

    def test_an_absent_service_refuses_differently_from_a_broken_one(self) -> None:
        human_review.bind_review_queue(None)
        with pytest.raises(Refusal) as absent:
            resolved(human_config())
        human_review.bind_review_queue(StubReviewQueue(unreachable=True))
        selector = resolved(human_config())
        with pytest.raises(Refusal) as broken:
            selector.plugin.prefetch(["Q"], ["A"], batch_id="b")
        assert absent.value.code != broken.value.code, (
            '"absent" and "broken" are different facts; one code for both would send the '
            "operator to the wrong place"
        )


class TestACompositeKeepsCandidatesOutOfTheQueue:
    def test_a_rule_rejection_never_reaches_a_person(self) -> None:
        """FTID I-9, and the kindness half of short-circuiting: fewer candidates in a
        queue, and nobody asked to read something a rule already settled."""
        queue = StubReviewQueue()
        human_review.bind_review_queue(queue)
        children = [rule_config("gate", forbidden_terms=["guarantee"]), human_config("human")]
        selector = resolved(composite_config("c", ["gate", "human"], "all"), catalog=children)

        result = evaluate(
            ["we guarantee it"] * 3,
            selector,
            EvaluationContext(batch_id="b1"),
            prompts=["Q"] * 3,
        )
        assert all(record.verdict.value.value == "reject" for record in result.records)
        assert (
            queue.post_count == 0
        ), "an AND composite must reject on a rule WITHOUT posting anybody's candidate"
        assert queue.all_posted == []

    def test_a_passing_completion_does_reach_the_person(self) -> None:
        """The other side of the same test, so the zero above is not a broken wiring."""
        queue = StubReviewQueue()
        human_review.bind_review_queue(queue)
        children = [rule_config("gate", forbidden_terms=["guarantee"]), human_config("human")]
        selector = resolved(composite_config("c", ["gate", "human"], "all"), catalog=children)

        result = evaluate(
            ["a careful answer"],
            selector,
            EvaluationContext(batch_id="b1"),
            prompts=["Q"],
        )
        assert result.records[0].verdict.value.value == "undecided"
        assert queue.post_count == 1
        assert len(queue.all_posted) == 1
