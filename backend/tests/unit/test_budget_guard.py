"""The budget seam and the computed estimator  (FR-001.8, R-02.42; OQ-1, OQ-5).

**What the operator decided, and what it means for F8.** F1's OQ-1 asked whether R-02.42's
estimator moves forward into F1's wave or whether F1 ships a pass-through and risks an
allocator error for two waves. It was pulled forward (2026-10-14), so
:class:`training.budget_guard.ComputedBudgetGuard` exists now, implementing
:class:`training.budget.BudgetGuard` **exactly as F1 declared it**. F8's work at this seam
becomes *confirm, not rebuild*.

Three properties are load-bearing and each has its own class below:

1. **A guard that cannot estimate must not permit.** Neither input present is a **refusal**,
   not a pass (F8's FR-008.10). The asymmetry matters: the refusing case is obvious and gets
   tested, while *"I had nothing to work with, so I allowed it"* is green against every test
   of the refusing path and silently useless in production.
2. **The guard runs before ``acquire()``.** Asserted with the **payload and the call count**,
   because "the guard was called" is satisfied by a call made *after* the allocation it was
   meant to precede.
3. **The pass-through logs loudly.** A silent pass-through is indistinguishable from a
   working guard.
"""

from __future__ import annotations

import logging
from pathlib import Path

import pytest

from core.errors import Refusal
from core.gpu import Device
from training.acquisition.base import AcquisitionMode, AcquisitionSpec
from training.budget import BudgetGuard, PassThroughGuard, require_exactly_one_input
from training.budget_guard import (
    GIB,
    OPTIMIZER_STATE_BYTES,
    ComputedBudgetGuard,
    estimate,
    parameter_count_from_config,
)
from training.run_config import RunConfig

#: BRD-02 §6's worked example, as a config: LFM2 1.2B-ish shape.
BILLION_ISH = {
    "model_type": "llama",
    "hidden_size": 2048,
    "num_hidden_layers": 24,
    "num_attention_heads": 16,
    "num_key_value_heads": 8,
    "intermediate_size": 8192,
    "vocab_size": 65536,
}

CFG = RunConfig(learning_rate=1e-5, warmup_steps=10, batch_size=1, gradient_accumulation=4)


def card(free_gib: float) -> Device:
    return Device(
        uuid="GPU-12345678-1234-1234-1234-123456789abc",
        name="NVIDIA GeForce RTX 3090",
        index=0,
        free_memory_bytes=int(free_gib * GIB),
    )


class TestTheProtocolIsSatisfiedByBoth:
    def test_both_guards_satisfy_the_declared_protocol(self) -> None:
        """**Do not change this signature.** F8 adopted it verbatim rather than proposing
        one, so the ``Protocol`` is not edited and the call site is not moved; the only
        change at this seam when F8 lands is the binding in ``services.acquisition``."""
        assert isinstance(PassThroughGuard(), BudgetGuard)
        assert isinstance(ComputedBudgetGuard(card(24)), BudgetGuard)

    def test_check_returns_none_rather_than_a_verdict(self) -> None:
        """A boolean return is a value a caller can forget to read, and the one thing this
        guard must never be is called and ignored."""
        assert ComputedBudgetGuard(card(24)).check(_load_spec(1_000_000), CFG) is None


class TestAGuardThatCannotEstimateMustNotPermit:
    """F8's FR-008.10, recorded on this side so the two cannot drift."""

    def test_NEITHER_input_is_refused_not_passed(self) -> None:
        bare = AcquisitionSpec(path=Path("/tmp/x"))
        assert bare.param_count is None and bare.architecture_config is None
        with pytest.raises(Refusal) as caught:
            require_exactly_one_input(bare)
        assert caught.value.code == "budget_spec_incomplete"
        assert "must not permit" in caught.value.message

    def test_BOTH_inputs_are_refused_as_ambiguous(self) -> None:
        """Two independent estimates of the same thing, with nothing to say which to trust."""
        both = AcquisitionSpec(
            mode=AcquisitionMode.CREATE, config=dict(BILLION_ISH), param_count=1_000
        )
        with pytest.raises(Refusal) as caught:
            require_exactly_one_input(both)
        assert caught.value.code == "budget_spec_ambiguous"
        assert "precedence" in caught.value.message

    def test_the_estimator_refuses_the_same_two_cases_before_any_arithmetic(self) -> None:
        with pytest.raises(Refusal) as caught:
            estimate(AcquisitionSpec(path=Path("/tmp/x")), CFG)
        assert caught.value.code == "budget_spec_incomplete"

    def test_the_computed_guard_refuses_when_the_cards_free_memory_is_UNKNOWN(self) -> None:
        """*"I could not read the limit, so I allowed it"* is the pass-through failure
        wearing the estimator's name. This guard's purpose is a comparison against a real
        number."""
        unknown = Device(uuid="GPU-1", name="card", index=0, free_memory_bytes=None)
        with pytest.raises(Refusal) as caught:
            ComputedBudgetGuard(unknown).check(_load_spec(1_000_000), CFG)
        assert caught.value.code == "budget_unknown_capacity"
        assert "unmeasured run" in caught.value.message


class TestTheEstimateIsArithmeticInTheParameterCount:
    def test_the_four_fixed_components_are_what_BRD_section_6_lists(self) -> None:
        """Gradients, FP32 master and optimizer states, all per parameter - §6's worked
        example for a 1.2B model is gradients 2.4 GB, FP32 master 4.8 GB, 8-bit states
        2.4 GB, and this arithmetic reproduces that ratio exactly."""
        projection = estimate(_load_spec(1_200_000_000), CFG)
        assert projection.weights_bytes == 1_200_000_000 * 2
        assert projection.gradient_bytes == 1_200_000_000 * 2
        assert projection.master_bytes == 1_200_000_000 * 4
        assert projection.optimizer_state_bytes == 1_200_000_000 * 2
        record = projection.as_record()
        assert record["gradients_gib"] == pytest.approx(2.23, abs=0.02)
        assert record["fp32_master_gib"] == pytest.approx(4.47, abs=0.02)
        assert record["optimizer_states_gib"] == pytest.approx(2.23, abs=0.02)

    def test_a_32_bit_optimizer_costs_four_times_the_state_of_an_8_bit_one(self) -> None:
        """Which is the whole reason the precision is configurable **and recorded**."""
        eight = estimate(_load_spec(1_000_000), CFG)
        thirty_two = estimate(
            _load_spec(1_000_000), CFG.model_copy(update={"optimizer_state_precision": "32bit"})
        )
        assert thirty_two.optimizer_state_bytes == 4 * eight.optimizer_state_bytes
        assert OPTIMIZER_STATE_BYTES["32bit"] == 4 * OPTIMIZER_STATE_BYTES["8bit"]

    def test_activations_are_a_RANGE_and_the_refusal_compares_the_pessimistic_end(
        self,
    ) -> None:
        """Gradient checkpointing is on (R-02.13), so this is recompute-bounded and depends
        on sequence length - neither of which R-02.13 puts in the run config. Admitting a
        run on its optimistic estimate means it dies in the allocator minutes in, which is
        the precise failure R-02.42 exists to prevent."""
        projection = estimate(_load_spec(1_000_000_000), CFG)
        assert projection.activation_low_bytes < projection.activation_high_bytes
        assert projection.total_high_bytes > projection.total_low_bytes
        assert projection.total_high_bytes == (
            projection.fixed_bytes + projection.activation_high_bytes
        )

    def test_accumulation_does_not_multiply_activations_but_batch_size_does(self) -> None:
        """Four micro-batches of 2 hold the activations of 2, not of 8 - accumulation trades
        activations for steps, which is the reason it exists."""
        accum = estimate(
            _load_spec(1_000_000), CFG.model_copy(update={"gradient_accumulation": 16})
        )
        bigger = estimate(_load_spec(1_000_000), CFG.model_copy(update={"batch_size": 4}))
        baseline = estimate(_load_spec(1_000_000), CFG)
        assert accum.activation_high_bytes == baseline.activation_high_bytes
        assert bigger.activation_high_bytes == 4 * baseline.activation_high_bytes

    def test_the_basis_says_whether_the_count_was_measured_or_derived(self) -> None:
        """A reader comparing two runs needs to know: a count from weight headers is a
        measurement, one from a config is a derivation."""
        assert estimate(_load_spec(1_000), CFG).basis == "weights"
        assert estimate(_create_spec(), CFG).basis == "config"


class TestTheParameterCountFromAConfig:
    """R-02.42 as corrected: *from the architecture config, before ``instantiate``*."""

    def test_a_billion_scale_config_estimates_in_the_right_order_of_magnitude(self) -> None:
        count = parameter_count_from_config(BILLION_ISH)
        assert 1.0e9 < count < 2.0e9, f"expected ~1.2B, got {count/1e9:.2f}B"

    def test_grouped_query_attention_is_cheaper_than_full_attention(self) -> None:
        """Assuming ``kv_heads == heads`` overstates a GQA model by a wide margin, which on a
        24 GB card is the difference between a refusal and a run."""
        gqa = parameter_count_from_config(BILLION_ISH)
        mha = parameter_count_from_config({**BILLION_ISH, "num_key_value_heads": 16})
        assert gqa < mha

    def test_tied_embeddings_drop_the_output_head(self) -> None:
        tied = parameter_count_from_config({**BILLION_ISH, "tie_word_embeddings": True})
        untied = parameter_count_from_config(BILLION_ISH)
        assert untied - tied == BILLION_ISH["vocab_size"] * BILLION_ISH["hidden_size"]

    def test_an_omitted_intermediate_size_falls_back_to_the_documented_ratio(self) -> None:
        """Omitting ``intermediate_size`` is ordinary; omitting ``hidden_size`` means the
        config does not describe a model - so only the derived field has a fallback."""
        without = {k: v for k, v in BILLION_ISH.items() if k != "intermediate_size"}
        assert parameter_count_from_config(without) > 0

    @pytest.mark.parametrize("field", ["hidden_size", "num_hidden_layers", "vocab_size"])
    def test_a_missing_DIMENSION_refuses_and_names_the_field(self, field: str) -> None:
        """ADR-018: a guessed size produces a confident estimate of a model that does not
        exist."""
        incomplete = {k: v for k, v in BILLION_ISH.items() if k != field}
        with pytest.raises(Refusal) as caught:
            parameter_count_from_config(incomplete)
        assert caught.value.code == "architecture_config_invalid"
        assert caught.value.details["field"] == field

    def test_a_boolean_is_not_mistaken_for_a_dimension(self) -> None:
        """``True`` is an ``int`` in Python, so a config carrying ``hidden_size: true``
        would otherwise estimate a one-dimensional model without complaint."""
        with pytest.raises(Refusal):
            parameter_count_from_config({**BILLION_ISH, "hidden_size": True})


class TestTheRefusalIsActionable:
    def test_an_oversized_run_names_the_estimate_AND_the_free_figure(self) -> None:
        """FR-001.8: the message carries the estimate and the free figure, so the operator
        knows what to change rather than only that something was too big."""
        with pytest.raises(Refusal) as caught:
            ComputedBudgetGuard(card(4)).check(_create_spec(), CFG)
        assert caught.value.code == "budget_exceeded"
        assert caught.value.details["free_gib"] == pytest.approx(4.0)
        assert caught.value.details["claimable_gib"] < caught.value.details["free_gib"]
        assert caught.value.details["estimate"]["total_gib_high"] > 0
        assert "Reduce batch_size" in caught.value.message
        assert "before any allocation" in caught.value.message

    def test_a_run_that_fits_passes(self) -> None:
        ComputedBudgetGuard(card(24)).check(_create_spec(), CFG)

    def test_headroom_is_left_unclaimed(self) -> None:
        """The allocator fragments, CUDA's context costs hundreds of megabytes, and this
        node's cards are **shared** - none of the three pods requests ``nvidia.com/gpu``, so
        all three can be resident at once."""
        projection = estimate(_create_spec(), CFG)
        exact = projection.total_high_bytes / GIB
        with pytest.raises(Refusal):
            ComputedBudgetGuard(card(exact)).check(_create_spec(), CFG)
        ComputedBudgetGuard(card(exact / 0.85)).check(_create_spec(), CFG)

    def test_every_number_in_the_record_names_its_scale(self) -> None:
        """ADR-012. A bare float is not a measurement; a key ending in ``_gib`` is."""
        record = estimate(_create_spec(), CFG).as_record()
        unscaled = {
            key
            for key in record
            if not key.endswith(("_gib", "_gib_low", "_gib_high"))
            and key not in {"param_count", "basis"}
        }
        assert unscaled == set(), f"these numbers name no scale: {unscaled}"


class TestThePassThroughIsLoudAndIsKept:
    def test_it_permits_and_logs_a_WARNING_naming_the_requirement(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        """**A silent pass-through is indistinguishable from a working guard**, which is the
        exact shape of defect this project's review discipline exists to catch."""
        guard = PassThroughGuard()
        with caplog.at_level(logging.WARNING):
            guard.check(_load_spec(1_000), CFG)
        assert any(
            record.levelno == logging.WARNING
            and getattr(record, "code", None) == "budget_guard_pass_through"
            and getattr(record, "requirement", None) == "R-02.42"
            for record in caplog.records
        ), f"expected a loud pass-through warning, got {[r.message for r in caplog.records]}"

    def test_it_records_its_calls_so_F8_can_use_it_as_a_SPY(self) -> None:
        """**Keep this class after F8 lands**: F8's ordering test uses it as a spy, so
        deleting it breaks another feature's test."""
        guard = PassThroughGuard()
        spec = _load_spec(1_000)
        guard.check(spec, CFG)
        assert guard.calls == [(spec, CFG)]

    def test_it_is_the_ONE_deliberate_exception_to_the_neither_input_rule(self) -> None:
        """It declines to estimate *by design*, which is not the same as being handed a spec
        it cannot read - and refusing here would make a pass-through stricter than the
        estimator that replaces it."""
        PassThroughGuard().check(AcquisitionSpec(path=Path("/tmp/x")), CFG)


def _load_spec(param_count: int) -> AcquisitionSpec:
    return AcquisitionSpec(path=Path("/tmp/x"), param_count=param_count)


def _create_spec() -> AcquisitionSpec:
    return AcquisitionSpec(mode=AcquisitionMode.CREATE, config=dict(BILLION_ISH))
