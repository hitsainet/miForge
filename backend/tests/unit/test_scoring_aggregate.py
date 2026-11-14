"""Aggregation to one number, by a named method, with nothing clamped  (FR-004.15; E-8).

Two things this file protects.

**The method name is the contract's.** ``step-record-v1.json``'s ``EvaluationBlock.aggregation``
is an enum of five values whose own description says why it exists: *"'mean' and 'pass@k' are
not the same claim."* F4's documents spell the methods ``accept_rate`` and
``mean_normalised``; neither is in the enum, so the run log would refuse them. The contract
wins, and the agreement is asserted **against the schema file** so the next change moves this
test rather than breaking it silently.

**Nothing is clamped.** ``ScoreRecord.score`` is bounded 0..1. A clamp makes the manifest's
number differ from the computation and hides the real fault, which is an aggregation or a
declared grader range that is wrong.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from core.errors import Refusal
from services.scoring import aggregate

SCHEMA = Path(__file__).resolve().parents[3] / "docs" / "schemas" / "step-record-v1.json"


class TestTheMethodVocabularyIsTheContracts:
    def test_the_declared_methods_are_exactly_the_schemas_enum(self) -> None:
        """Read from the file, not retyped.

        A hardcoded list goes stale the moment the contract moves - which is the defect
        class this project keeps paying for, and the reason the ScoreRecord key set is read
        from its schema too.
        """
        document = json.loads(SCHEMA.read_text("utf-8"))
        declared = document["$defs"]["EvaluationBlock"]["properties"]["aggregation"]["enum"]
        assert sorted(aggregate.DECLARED_METHODS) == sorted(declared)

    def test_f4s_own_document_spellings_are_NOT_methods(self) -> None:
        """The divergence, pinned so it cannot be reintroduced by a reader of the FTID.

        ``accept_rate`` and ``mean_normalised`` appear in F4's FPRD, FTDD, FTID and FTASKS.
        Neither is in the contract's enum, so a block carrying one would be refused by the
        run log's validator. An accept rate over accept/reject verdicts **is** ``mean`` over
        the admitted indicator - the arithmetic is identical and the contract's name for it
        is ``mean``.
        """
        for stale in ("accept_rate", "mean_normalised"):
            assert stale not in aggregate.DECLARED_METHODS
            with pytest.raises(Refusal) as caught:
                aggregate.aggregate(stale, [1.0])
            assert caught.value.code == "aggregation_method_unknown"

    def test_an_unknown_method_is_never_defaulted_to_mean(self) -> None:
        with pytest.raises(Refusal):
            aggregate.aggregate("whatever", [0.5])

    def test_pass_at_k_is_declared_and_refused_by_name(self) -> None:
        """Declared by the contract, **not implemented**, and said so.

        It needs k independent samples per prompt and no requirement in R-02.16 - R-02.19
        asks for repeated sampling. Computing it over one sample would be a number that
        claims pass@k and is not.
        """
        assert "pass_at_k" in aggregate.DECLARED_METHODS
        with pytest.raises(Refusal) as caught:
            aggregate.aggregate("pass_at_k", [1.0, 0.0])
        assert caught.value.code == "aggregation_method_unsupported"


class TestTheArithmetic:
    def test_mean_over_an_admitted_indicator_is_the_accept_rate(self) -> None:
        assert aggregate.aggregate("mean", [1.0, 1.0, 0.0, 0.0]) == 0.5

    def test_mean_over_grader_scalars(self) -> None:
        assert aggregate.aggregate("mean", [0.2, 0.4, 0.6]) == pytest.approx(0.4)

    def test_median(self) -> None:
        assert aggregate.aggregate("median", [0.1, 0.9, 0.5]) == 0.5

    def test_exact_match_rate_counts_only_exact_matches(self) -> None:
        """Different from ``mean`` over a continuous scalar, which is why both names exist."""
        assert aggregate.aggregate("exact_match_rate", [1.0, 0.99, 0.0]) == pytest.approx(1 / 3)
        assert aggregate.aggregate("mean", [1.0, 0.99, 0.0]) == pytest.approx(0.6633333, rel=1e-5)

    def test_weighted_mean(self) -> None:
        assert aggregate.aggregate("weighted_mean", [1.0, 0.0], weights=[3.0, 1.0]) == 0.75

    def test_weighted_mean_without_weights_refuses(self) -> None:
        """*Weighting everything equally is 'mean' under another name*, and the recorded
        method would then describe a computation that did not happen."""
        with pytest.raises(Refusal) as caught:
            aggregate.aggregate("weighted_mean", [1.0, 0.0])
        assert caught.value.code == "aggregation_weights_missing"

    def test_mis_sized_weights_refuse_rather_than_padding(self) -> None:
        with pytest.raises(Refusal) as caught:
            aggregate.aggregate("weighted_mean", [1.0, 0.0], weights=[1.0])
        assert caught.value.code == "aggregation_weights_mismatch"

    def test_weights_passed_to_a_method_that_ignores_them_refuse(self) -> None:
        """*A caller that passed weights believes they are in force.*"""
        with pytest.raises(Refusal) as caught:
            aggregate.aggregate("mean", [1.0, 0.0], weights=[1.0, 1.0])
        assert caught.value.code == "aggregation_weights_not_applicable"

    def test_zero_sum_weights_refuse_rather_than_falling_back(self) -> None:
        with pytest.raises(Refusal) as caught:
            aggregate.aggregate("weighted_mean", [1.0, 0.0], weights=[0.0, 0.0])
        assert caught.value.code == "aggregation_weights_invalid"


class TestNothingIsEverClamped:
    @pytest.mark.parametrize("boundary", [0.0, 1.0])
    def test_the_boundary_values_are_accepted(self, boundary: float) -> None:
        """Inclusive at both ends, matching the schema's ``minimum``/``maximum``."""
        assert aggregate.aggregate("mean", [boundary]) == boundary

    def test_an_out_of_range_input_refuses_with_the_value_and_the_range(self) -> None:
        with pytest.raises(Refusal) as caught:
            aggregate.aggregate("mean", [0.5, 1.4])
        assert caught.value.code == "score_out_of_range"
        assert caught.value.details["value"] == 1.4
        assert caught.value.details["range"] == [0.0, 1.0]

    def test_a_negative_input_refuses(self) -> None:
        with pytest.raises(Refusal) as caught:
            aggregate.aggregate("mean", [-0.1])
        assert caught.value.code == "score_out_of_range"

    def test_the_refusal_names_the_fault_rather_than_the_bound(self) -> None:
        """The message has to send the reader to the aggregation or the grader's declared
        range, which is the true fault - not to the bound, which is correct."""
        with pytest.raises(Refusal) as caught:
            aggregate.aggregate("mean", [2.0])
        assert "clamp" in caught.value.message

    def test_a_nan_refuses_rather_than_reaching_the_serialiser(self) -> None:
        """``json`` would emit the literal ``NaN``, which no parser is required to accept,
        so the log would validate going out and fail coming back."""
        with pytest.raises(Refusal) as caught:
            aggregate.aggregate("mean", [float("nan")])
        assert caught.value.code == "score_out_of_range"

    def test_no_outcomes_refuses(self) -> None:
        with pytest.raises(Refusal) as caught:
            aggregate.aggregate("mean", [])
        assert caught.value.code == "evaluation_set_empty"
