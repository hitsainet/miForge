"""The verdict partition, exhaustively  (FR-002.12, FR-002.13, AC-002.4).

**The two tests this file exists for:**

* :meth:`TestTheThreeValueTable.test_each_verdict_value_goes_to_exactly_one_destination` -
  mutation controls M-1 and M-2: route ``undecided`` into the admitted set, or count it as
  a reject, and this goes red.
* :meth:`TestTheBoundedWait.test_an_expired_deadline_admits_nothing` - the feature's
  privacy-of-judgement guarantee. *Defaulting silence to accept turns an unattended queue
  into an unselected run.*

The wait is driven through injected ``sleep`` and ``now``, so the expiry behaviour is
asserted in milliseconds rather than being approximated by a shorter deadline.
"""

from __future__ import annotations

from typing import Any

import pytest

from core.errors import Refusal
from schemas.selector import Verdict, VerdictValue
from services.selectors.types import (
    EvaluationResult,
    PublishedHashes,
    SelectionHealth,
    VerdictRecord,
)
from training.signals import selection
from training.signals.selection import Candidate, apply, partition
from training.signals.settings import SignalSettings


def candidate(index: int = 0, completion: str = "an answer") -> Candidate:
    return Candidate(prompt_id=f"p-{index}", prompt="q", completion=completion, index=index)


def verdict(value: str, score: float | None = None) -> Verdict:
    return Verdict(value=VerdictValue(value), selector="rule", label="because", score=score)


def result_for(
    values: list[str], *, review_ids: list[str | None] | None = None
) -> EvaluationResult:
    """An ``EvaluationResult`` in F3's own shape, built through F3's own models."""
    records = []
    for index, value in enumerate(values):
        records.append(
            VerdictRecord(
                verdict=verdict(value),
                reason_code="because",
                config_sha256="a" * 64,
                config_kind="rule",
                review_item_id=(review_ids[index] if review_ids else None),
            )
        )
    accepted = values.count("accept")
    return EvaluationResult(
        records=tuple(records),
        health=SelectionHealth(
            n=len(values),
            accepted=accepted,
            rejected=values.count("reject"),
            undecided=values.count("undecided"),
            acceptance_rate=accepted / len(values) if values else 0.0,
            reason_histogram={"because": len(values)},
        ),
        hashes=PublishedHashes(selector_hash="b" * 64, selector_hash_provenance="local"),
    )


class Evaluator:
    """A recording stand-in for F3's one entry point, with a scripted sequence of answers.

    It records the **payload and the call count** - "was called" passes against a call
    sending the wrong completions, which is the whole reason ADR-016 words its first rule
    the way it does.
    """

    def __init__(self, answers: list[list[str]]) -> None:
        self.answers = answers
        self.calls: list[tuple[tuple[str, ...], tuple[str, ...]]] = []

    def __call__(
        self, completions: Any, resolved: Any, ctx: Any = None, *, prompts: Any = None
    ) -> Any:
        self.calls.append((tuple(completions), tuple(prompts or ())))
        values = self.answers[min(len(self.calls) - 1, len(self.answers) - 1)]
        return result_for(values)


@pytest.fixture
def settings() -> SignalSettings:
    return SignalSettings(pending_deadline_s=0.0)


class TestTheThreeValueTable:
    @pytest.mark.parametrize(
        "value,expected",
        [
            ("accept", {"admitted": 1, "rejected": 0, "pending": 0}),
            ("reject", {"admitted": 0, "rejected": 1, "pending": 0}),
            ("undecided", {"admitted": 0, "rejected": 0, "pending": 1}),
        ],
    )
    def test_each_verdict_value_goes_to_exactly_one_destination(
        self, value: str, expected: dict[str, int]
    ) -> None:
        """**Mutation controls M-1 and M-2.** Three values, three destinations, no fourth."""
        admitted, rejected, pending = partition([candidate()], [verdict(value)])
        assert {
            "admitted": len(admitted),
            "rejected": len(rejected),
            "pending": len(pending),
        } == expected

    def test_undecided_is_counted_in_neither_other_destination(self) -> None:
        """Stated separately from the table because it is the rule, not a case of it."""
        admitted, rejected, pending = partition(
            [candidate(0), candidate(1), candidate(2)],
            [verdict("accept"), verdict("reject"), verdict("undecided")],
        )
        assert [c.index for c in admitted] == [0]
        assert [c.index for c in rejected] == [1]
        assert [c.index for c in pending] == [2]

    def test_a_verdict_count_mismatch_refuses_rather_than_zipping(self) -> None:
        """A positional join over mismatched lists misattributes every verdict after the
        gap."""
        with pytest.raises(Refusal) as caught:
            partition([candidate(0), candidate(1)], [verdict("accept")])
        assert caught.value.code == "verdict_count_mismatch"


class TestTheBoundedWait:
    def test_an_expired_deadline_admits_nothing(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """**AC-002.4.** The deadline bounds the wait and produces only a count.

        There is no branch, flag, timeout or shortfall path from pending to admitted - so
        when the clock runs out with candidates still pending, the admitted set is
        unchanged and the pending count is what the record carries.
        """
        # The second entry is the re-ask: only the still-pending candidate is sent, and it
        # is still undecided - nobody answered. The deadline then expires.
        evaluator = Evaluator([["undecided", "accept"], ["undecided"]])
        monkeypatch.setattr(selection, "f3_evaluate", evaluator)
        clock = iter([0.0, 0.0, 0.2, 0.4, 0.6, 1.0, 1.0, 1.0, 1.0])
        outcome = apply(
            [candidate(0), candidate(1)],
            resolved=None,
            settings=SignalSettings(pending_deadline_s=0.5),
            sleep=lambda _seconds: None,
            now=lambda: next(clock),
        )
        assert [c.index for c in outcome.admitted] == [1]
        assert [c.index for c in outcome.pending] == [0]
        assert outcome.counts == {"admitted": 1, "rejected": 0, "pending": 1}

    def test_a_decision_that_arrives_during_the_wait_is_honoured(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The wait is a wait, not a formality: a human who answers in time is heard.

        That is what makes the expiry test above meaningful - it fails for the right
        reason, rather than because nothing was ever re-asked.
        """
        evaluator = Evaluator([["undecided"], ["accept"]])
        monkeypatch.setattr(selection, "f3_evaluate", evaluator)
        clock = iter([0.0, 0.0, 0.1, 0.2, 0.3, 0.4, 0.5])
        outcome = apply(
            [candidate(0)],
            resolved=None,
            settings=SignalSettings(pending_deadline_s=5.0),
            sleep=lambda _seconds: None,
            now=lambda: next(clock),
        )
        assert len(outcome.admitted) == 1
        assert outcome.pending == ()
        assert len(evaluator.calls) == 2

    def test_no_wait_at_all_when_the_deadline_is_zero(
        self, monkeypatch: pytest.MonkeyPatch, settings: SignalSettings
    ) -> None:
        evaluator = Evaluator([["undecided"]])
        monkeypatch.setattr(selection, "f3_evaluate", evaluator)
        outcome = apply([candidate(0)], resolved=None, settings=settings)
        assert len(evaluator.calls) == 1
        assert len(outcome.pending) == 1


class TestWhatIsCarriedAndWhatIsNot:
    def test_the_health_record_is_f3s_and_is_not_recomputed(
        self, monkeypatch: pytest.MonkeyPatch, settings: SignalSettings
    ) -> None:
        """Recomputing any of F3's numbers produces a second number that can disagree."""
        evaluator = Evaluator([["accept", "reject"]])
        monkeypatch.setattr(selection, "f3_evaluate", evaluator)
        outcome = apply([candidate(0), candidate(1)], resolved=None, settings=settings)
        assert outcome.health.reason_histogram == {"because": 2}
        assert outcome.health.acceptance_rate == 0.5
        assert outcome.health.n == 2

    def test_f3_is_called_once_with_the_completions_and_the_prompts(
        self, monkeypatch: pytest.MonkeyPatch, settings: SignalSettings
    ) -> None:
        """The payload **and** the call count: "was called" passes against a call sending
        the wrong completions."""
        evaluator = Evaluator([["accept"]])
        monkeypatch.setattr(selection, "f3_evaluate", evaluator)
        apply([candidate(0, "the answer")], resolved=None, settings=settings)
        assert evaluator.calls == [(("the answer",), ("q",))]

    def test_review_identifiers_are_carried_and_nothing_else_is(
        self, monkeypatch: pytest.MonkeyPatch, settings: SignalSettings
    ) -> None:
        """R-02.36's closing sentence: the decision identifiers are recorded on the batch
        they admitted. **Identifiers only** - a reviewer's label and reason stay with F10."""

        def evaluate(completions, resolved, ctx=None, *, prompts=None):  # type: ignore[no-untyped-def]
            return result_for(["undecided"], review_ids=["item-77"])

        monkeypatch.setattr(selection, "f3_evaluate", evaluate)
        outcome = apply([candidate(0)], resolved=None, settings=settings)
        assert outcome.review_ids == ("item-77",)

    def test_the_hashes_are_f3s_published_pair(
        self, monkeypatch: pytest.MonkeyPatch, settings: SignalSettings
    ) -> None:
        evaluator = Evaluator([["accept"]])
        monkeypatch.setattr(selection, "f3_evaluate", evaluator)
        outcome = apply([candidate(0)], resolved=None, settings=settings)
        assert outcome.hashes.selector_hash == "b" * 64
        assert outcome.hashes.selector_hash_provenance == "local"
