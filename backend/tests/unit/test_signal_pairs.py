"""A scalar orders a pair; a tie is never broken  (FR-002.19, FR-002.22; AC-002.8).

**Mutation control M-5:** break a tie by returning the first completion as chosen and
:meth:`TestOrdering.test_a_tie_produces_no_pair` must go red.

*A pair ordered by a coin flip trains the model toward noise, and its loss curve looks
exactly like one that is working* - which is why the tie is a counted outcome rather than a
judgement call.
"""

from __future__ import annotations

import pytest

from core.errors import Refusal
from training.signals import pairs
from training.signals.base import PreferenceExample


class TestOrdering:
    def test_a_higher_scalar_is_the_chosen_side(self) -> None:
        result = pairs.from_generations(
            "p-1",
            "q",
            "worse",
            "better",
            first_scalar=0.2,
            second_scalar=0.9,
            ordered_by="grader_score",
        )
        assert isinstance(result, PreferenceExample)
        assert (result.chosen, result.rejected) == ("better", "worse")
        assert result.ordered_by == "grader_score"
        assert result.margin == pytest.approx(0.7)

    def test_the_order_of_the_arguments_does_not_decide_it(self) -> None:
        result = pairs.from_generations(
            "p-1",
            "q",
            "better",
            "worse",
            first_scalar=0.9,
            second_scalar=0.2,
            ordered_by="grader_score",
        )
        assert isinstance(result, PreferenceExample)
        assert result.chosen == "better"

    def test_a_tie_produces_no_pair(self) -> None:
        """**Mutation control M-5.** Equal scalars separate nothing."""
        result = pairs.from_generations(
            "p-1",
            "q",
            "a",
            "b",
            first_scalar=0.5,
            second_scalar=0.5,
            ordered_by="verdict_score",
        )
        assert isinstance(result, pairs.NoPair)
        assert result.reason_code == pairs.TIE

    def test_an_absent_scalar_produces_no_pair_and_is_counted_separately(self) -> None:
        """A missing scalar is not a tie: a tie means the selector answered and could not
        separate them, this means nothing answered at all - which is a configuration
        problem the operator can fix."""
        result = pairs.from_generations(
            "p-1",
            "q",
            "a",
            "b",
            first_scalar=None,
            second_scalar=0.5,
            ordered_by="verdict_score",
        )
        assert isinstance(result, pairs.NoPair)
        assert result.reason_code == pairs.NO_SCALAR

    def test_the_result_is_a_union_not_an_optional(self) -> None:
        """``NoPair`` carries a reason that lands on the record as a count; ``None`` would
        invite a caller to read it as *nothing happened*."""
        result = pairs.from_generations(
            "p-1", "q", "a", "b", first_scalar=None, second_scalar=None, ordered_by="x"
        )
        assert result is not None
        assert isinstance(result, pairs.NoPair)


class TestMalformedPairs:
    def test_byte_identical_sides_are_refused_and_counted(self) -> None:
        """E-10: an identical pair expresses no preference and has no gradient to give."""
        result = pairs.from_generations(
            "p-1", "q", "same", "same", first_scalar=0.9, second_scalar=0.1, ordered_by="x"
        )
        assert isinstance(result, pairs.NoPair)
        assert result.reason_code == pairs.IDENTICAL

    @pytest.mark.parametrize("chosen,rejected", [("", "b"), ("a", ""), ("   ", "b")])
    def test_an_empty_side_is_refused(self, chosen: str, rejected: str) -> None:
        result = pairs.from_generations(
            "p-1", "q", chosen, rejected, first_scalar=0.9, second_scalar=0.1, ordered_by="x"
        )
        assert isinstance(result, pairs.NoPair)
        assert result.reason_code == pairs.EMPTY


class TestPairsFromAFile:
    def test_a_row_binds_by_the_declared_columns(self) -> None:
        result = pairs.from_row(
            "p-1", {"good": "yes", "bad": "no"}, chosen_column="good", rejected_column="bad"
        )
        assert isinstance(result, PreferenceExample)
        assert (result.chosen, result.rejected, result.ordered_by) == ("yes", "no", "file")
        assert result.margin is None

    def test_a_non_text_cell_is_refused_rather_than_coerced(self) -> None:
        """``str(None)`` is ``"None"``: a training example nobody wrote, which trains as
        happily as a real one (ADR-018)."""
        with pytest.raises(Refusal) as caught:
            pairs.from_row(
                "p-1", {"good": None, "bad": "no"}, chosen_column="good", rejected_column="bad"
            )
        assert caught.value.code == "pair_column_not_text"


class TestCounts:
    def test_dropped_and_ties_are_counted_apart(self) -> None:
        """Collapsing them would make a selector that never discriminates look like a
        dataset full of empty rows."""
        results = [
            PreferenceExample(prompt_id="1", prompt="", chosen="a", rejected="b"),
            pairs.NoPair(prompt_id="2", reason_code=pairs.TIE),
            pairs.NoPair(prompt_id="3", reason_code=pairs.NO_SCALAR),
            pairs.NoPair(prompt_id="4", reason_code=pairs.IDENTICAL),
            pairs.NoPair(prompt_id="5", reason_code=pairs.EMPTY),
        ]
        assert pairs.split_counts(results) == (2, 2)

    def test_an_unknown_reason_code_cannot_be_constructed(self) -> None:
        """F6 and F9 key on these codes; a code invented at a call site is a count nobody
        renders."""
        with pytest.raises(ValueError):
            pairs.NoPair(prompt_id="1", reason_code="because")
