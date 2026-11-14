"""Reward groups: the statistics, the declared range, and the signal  (R-02.7).

**Mutation control M-7:** drop ``grader_scores_have_no_spread`` from the record and
:meth:`TestZeroSpread.test_zero_spread_is_named_on_the_record` must go red.

*A group whose rewards are all equal yields a zero advantage for every member: the step runs
in full and applies numerically nothing.* It is the one failure in this feature that costs a
full GPU step and reports a healthy loss, which is why it is a named outcome on the record
rather than a log line or a skipped group.
"""

from __future__ import annotations

import pytest

from core.errors import Refusal
from services.graders import SCORE_RANGE
from services.selectors.types import SIGNAL_NO_SPREAD
from training.objectives.grpo import advantages
from training.signals import reward


class TestTheStatistics:
    def test_min_max_mean_and_variance_are_on_the_group(self) -> None:
        group = reward.group("p-1", "q", ["a", "b", "c"], [0.0, 0.5, 1.0])
        assert (group.minimum, group.maximum) == (0.0, 1.0)
        assert group.mean == pytest.approx(0.5)
        # Population variance, not the sample estimator: the group IS the population the
        # advantage is computed against.
        assert group.variance == pytest.approx(0.1666667, rel=1e-5)

    def test_the_declared_range_travels_with_every_group(self) -> None:
        """ADR-012: two runs' rewards are not comparable unless both say their scale."""
        group = reward.group("p-1", "q", ["a", "b"], [0.1, 0.9])
        assert group.declared_range == SCORE_RANGE
        assert reward.statistics_for([group])[0]["declared_range"] == [0.0, 1.0]

    def test_the_statistics_name_their_sample(self) -> None:
        group = reward.group("p-1", "q", ["a", "b"], [0.1, 0.9])
        assert reward.statistics_for([group])[0]["n"] == 2


class TestZeroSpread:
    def test_zero_spread_is_named_on_the_record(self) -> None:
        """**Mutation control M-7**, and the name is F3's, imported rather than retyped."""
        group = reward.group("p-1", "q", ["a", "b", "c"], [0.4, 0.4, 0.4])
        assert group.zero_spread is True
        assert reward.signals_for([group]) == (SIGNAL_NO_SPREAD,)
        assert SIGNAL_NO_SPREAD == "grader_scores_have_no_spread"

    def test_a_group_with_spread_raises_no_signal(self) -> None:
        group = reward.group("p-1", "q", ["a", "b"], [0.1, 0.9])
        assert group.zero_spread is False
        assert reward.signals_for([group]) == ()

    def test_the_advantage_of_a_zero_spread_group_is_zero_for_every_member(self) -> None:
        """The arithmetic behind the signal, asserted rather than described: this is why
        the step applies numerically nothing."""
        assert advantages((0.4, 0.4, 0.4)) == pytest.approx([0.0, 0.0, 0.0], abs=1e-12)

    def test_the_group_is_recorded_and_not_skipped(self) -> None:
        """Skipping it would hide the one failure that costs a full step and reports a
        healthy loss. F2 records; stopping is R-02.25's and F6's."""
        groups = [reward.group("p-1", "q", ["a", "b"], [0.4, 0.4])]
        assert len(reward.statistics_for(groups)) == 1
        assert reward.statistics_for(groups)[0]["zero_spread"] is True


class TestRefusals:
    def test_a_group_of_one_is_refused_naming_the_size(self) -> None:
        """E-16: a single completion has no relative advantage by construction."""
        with pytest.raises(Refusal) as caught:
            reward.group("p-1", "q", ["only"], [0.5])
        assert caught.value.code == "reward_group_too_small"
        assert caught.value.details["size"] == 1
        assert caught.value.details["minimum"] == 2

    def test_a_score_count_mismatch_is_refused(self) -> None:
        with pytest.raises(Refusal) as caught:
            reward.group("p-1", "q", ["a", "b"], [0.5])
        assert caught.value.code == "reward_score_count_mismatch"

    def test_f2_declares_no_range_of_its_own_and_clamps_nothing(self) -> None:
        """FR-002.24: the range is the grader configuration's, and F2 clamps nothing.

        Asserted by **behaviour**, not by scanning this module's own text for the word
        "clamp" - a string scan reports a module's prose as the violation, which has cost
        this project three rewritten structural tests. A value outside the declared range
        comes back exactly as it went in: the refusal is F3's
        (``services.graders.GraderAdapter.grade``), and a clamp here would turn a grader
        that misread its scale into a confident number.
        """
        group = reward.group("p-1", "q", ["a", "b"], [-3.0, 7.5])
        assert group.scores == (-3.0, 7.5)
        assert (group.minimum, group.maximum) == (-3.0, 7.5)
        assert group.declared_range == SCORE_RANGE

    def test_an_out_of_range_value_is_f3s_refusal_and_is_not_re_checked_here(self) -> None:
        """E-14: F2 propagates F3's refusal and offers no second opinion.

        ``reward.group`` takes scores that have already been through
        ``services.graders.GraderAdapter.grade``, which refuses out of range **and never
        clamps**. This asserts F2 adds no second check that would disagree with it.
        """
        group = reward.group("p-1", "q", ["a", "b"], [0.0, 1.0])
        assert group.scores == (0.0, 1.0)


class TestAdvantages:
    def test_the_advantage_is_centred_on_the_group_mean(self) -> None:
        assert advantages((0.0, 1.0)) == [-0.5, 0.5]

    def test_no_standard_deviation_division(self) -> None:
        """Normalising by the spread turns a group that learned nothing into a group with
        enormous advantages for whichever member rounded up."""
        assert advantages((0.5, 0.5000001))[0] < 0
        assert abs(advantages((0.5, 0.5000001))[0]) < 1e-6
