"""R-02.42's adapter branch: *"which is why adapters fit where full-parameter does not."*

**FR-008.12.** The base term stays; gradients, the FP32 master copy and the optimizer states
cover **the adapter**. Pure arithmetic, so all of this runs with no card and no ML stack
(ADR-023) - which is also the property that lets R-02.42's estimator be pulled forward as a
scheduling decision rather than a refactor.

## The deviation this file also pins

The estimate is reachable through ``POST /api/v1/budget/estimate`` and **not** through the
guard bound in ``services.acquisition``: ``guard_for`` constructs F1's
``ComputedBudgetGuard``, which has no run-type input, and that function and
``training/budget_guard.py`` are outside F8's write scope. The consequence is bounded and in
the safe direction, and :class:`TestTheGuardStillUsesTheFullParameterEstimate` asserts the
state of affairs rather than leaving it implied - so the day the binding lands, this file
goes red and says why.
"""

from __future__ import annotations

from typing import Any

import pytest

from core.errors import Refusal
from tests.support.test_adapter_doubles import adapter_config
from training import budget_guard
from training.run_config import RunConfig
from training.runtypes import adapter_budget

GIB = 1024**3

#: LFM2 1.2B's shape, close enough for BRD-02 section 6's worked example: 2.4 GB of FP16
#: weights means ~1.2e9 parameters.
LFM2_CONFIG: dict[str, Any] = {
    "model_type": "llama",
    "hidden_size": 2048,
    "intermediate_size": 8192,
    "num_hidden_layers": 16,
    "num_attention_heads": 32,
    "num_key_value_heads": 8,
    "vocab_size": 65536,
    "tie_word_embeddings": True,
}


def cfg(**overrides: Any) -> RunConfig:
    values: dict[str, Any] = {"learning_rate": 1e-4, "warmup_steps": 10, "batch_size": 1}
    values.update(overrides)
    return RunConfig(**values)


def adapter_estimate(**overrides: Any) -> adapter_budget.AdapterBudgetEstimate:
    values: dict[str, Any] = {
        "cfg": cfg(),
        "adapter": adapter_config(rank=8, alpha=16.0, target_modules=("q_proj", "v_proj")),
        "architecture_config": LFM2_CONFIG,
    }
    values.update(overrides)
    return adapter_budget.estimate(**values)


class TestTheAdapterSizedTermsAreTheWholePoint:
    def test_gradients_master_and_optimizer_cover_the_ADAPTER_not_the_base(self) -> None:
        estimate = adapter_estimate()
        assert estimate.gradient_bytes == estimate.adapter_param_count * 2
        assert estimate.master_bytes == estimate.adapter_param_count * 4
        assert estimate.optimizer_state_bytes == estimate.adapter_param_count * 2

    def test_the_BASE_term_stays_because_on_a_shared_card_it_decides(self) -> None:
        estimate = adapter_estimate()
        assert estimate.weights_bytes == estimate.base_param_count * 2
        assert estimate.weights_bytes > estimate.fixed_bytes / 2

    def test_an_adapter_run_is_MUCH_cheaper_than_the_full_parameter_one(self) -> None:
        """*"The adapter path is the cheapest run type, not the dearest."* Asserted, because
        people get it backwards and then budget it as the expensive one."""
        from training.acquisition.base import AcquisitionMode, AcquisitionSpec

        full = budget_guard.estimate(
            AcquisitionSpec(mode=AcquisitionMode.CREATE, config=LFM2_CONFIG), cfg()
        )
        adapter = adapter_estimate()
        assert adapter.total_high_bytes < full.total_high_bytes
        # The three parameter-proportional terms collapse; only the base and the
        # activations remain, so the saving is most of the fixed footprint.
        assert adapter.fixed_bytes < full.fixed_bytes / 3

    def test_an_adapter_fits_on_one_24GB_card_where_full_parameter_is_tight(self) -> None:
        """R-02.42's last sentence, as arithmetic rather than as prose."""
        adapter = adapter_estimate(cfg=cfg(batch_size=2))
        assert adapter.total_high_bytes < 12 * GIB

    def test_the_activations_are_UNCHANGED_because_the_pass_is_through_the_base(self) -> None:
        from training.acquisition.base import AcquisitionMode, AcquisitionSpec

        full = budget_guard.estimate(
            AcquisitionSpec(mode=AcquisitionMode.CREATE, config=LFM2_CONFIG), cfg()
        )
        adapter = adapter_estimate()
        assert adapter.activation_low_bytes == full.activation_low_bytes
        assert adapter.activation_high_bytes == full.activation_high_bytes

    def test_a_32_bit_optimizer_costs_four_times_the_adapters_state(self) -> None:
        eight = adapter_estimate()
        thirty_two = adapter_estimate(cfg=cfg(optimizer_state_precision="32bit"))
        assert thirty_two.optimizer_state_bytes == 4 * eight.optimizer_state_bytes


class TestTheAdapterParameterFigureIsABoundAndSaysSo:
    def test_an_estimate_from_a_config_alone_is_labelled_a_bound(self) -> None:
        """ADR-012: every number names its scale and sample, and a bound is a different
        kind of number from a count."""
        assert adapter_estimate().adapter_param_basis == "bound"

    def test_a_supplied_realised_count_is_labelled_a_count(self) -> None:
        estimate = adapter_estimate(adapter_param_count=3_000_000)
        assert estimate.adapter_param_basis == "count"
        assert estimate.adapter_param_count == 3_000_000

    def test_the_bound_is_CONSERVATIVE_rather_than_exact(self) -> None:
        """Every targeted module is assumed to be the architecture's widest linear. That
        over-counts a run targeting only ``q_proj``, deliberately and in the safe
        direction."""
        exact_q_proj = 8 * (2048 + 2048) * 16  # rank * (in + out) * layers
        bound = adapter_budget.adapter_parameter_bound(
            LFM2_CONFIG, adapter_config(rank=8, alpha=16.0, target_modules=("q_proj",))
        )
        assert bound > exact_q_proj

    def test_the_bound_grows_with_rank_and_with_the_number_of_target_modules(self) -> None:
        one = adapter_budget.adapter_parameter_bound(
            LFM2_CONFIG, adapter_config(rank=8, alpha=16.0, target_modules=("q_proj",))
        )
        two = adapter_budget.adapter_parameter_bound(
            LFM2_CONFIG, adapter_config(rank=8, alpha=16.0, target_modules=("q_proj", "v_proj"))
        )
        rank_sixteen = adapter_budget.adapter_parameter_bound(
            LFM2_CONFIG, adapter_config(rank=16, alpha=32.0, target_modules=("q_proj",))
        )
        assert two == 2 * one
        assert rank_sixteen == 2 * one

    def test_the_bound_counts_BOTH_lora_matrices_and_not_twice(self) -> None:
        """``rank * (in + out)`` is A and B together. Doubling it again would inflate every
        adapter budget by 2x - in the safe direction, and therefore silently."""
        bound = adapter_budget.adapter_parameter_bound(
            {"hidden_size": 100, "num_hidden_layers": 1, "intermediate_size": 100},
            adapter_config(rank=4, alpha=8.0, target_modules=("q_proj",)),
        )
        assert bound == 4 * (100 + 100)

    @pytest.mark.parametrize("missing", ["hidden_size", "num_hidden_layers"])
    def test_a_config_missing_a_dimension_REFUSES_rather_than_guessing(self, missing: str) -> None:
        """A guessed dimension produces a confident estimate of a model that does not
        exist (ADR-018)."""
        config = {k: v for k, v in LFM2_CONFIG.items() if k != missing}
        with pytest.raises(Refusal) as caught:
            adapter_budget.adapter_parameter_bound(
                config, adapter_config(rank=8, alpha=16.0, target_modules=("q_proj",))
            )
        assert caught.value.code == "architecture_config_invalid"


class TestExactlyOneBaseInput:
    """FR-008.10, E-5. **A guard that cannot estimate must not permit.**"""

    def test_BOTH_inputs_is_refused_as_ambiguous(self) -> None:
        with pytest.raises(Refusal) as caught:
            adapter_estimate(base_param_count=1_200_000_000)
        assert caught.value.code == "budget_spec_ambiguous"

    def test_NEITHER_input_is_refused_and_NOT_passed(self) -> None:
        with pytest.raises(Refusal) as caught:
            adapter_estimate(architecture_config=None)
        assert caught.value.code == "budget_spec_incomplete"
        assert "cannot be estimated is refused" in caught.value.message

    def test_a_parameter_count_with_no_adapter_count_is_refused(self) -> None:
        """The adapter's size depends on each targeted module's shape, which a base
        parameter count does not describe. Refusing rather than bounding it from a number
        that cannot bound it."""
        with pytest.raises(Refusal) as caught:
            adapter_estimate(architecture_config=None, base_param_count=1_200_000_000)
        assert caught.value.code == "budget_spec_incomplete"

    def test_a_parameter_count_WITH_an_adapter_count_works(self) -> None:
        estimate = adapter_estimate(
            architecture_config=None,
            base_param_count=1_200_000_000,
            adapter_param_count=4_000_000,
        )
        assert estimate.basis == "weights"
        assert estimate.base_param_count == 1_200_000_000

    def test_the_basis_says_whether_the_base_count_was_derived_or_measured(self) -> None:
        assert adapter_estimate().basis == "config"


class TestTheRecordNamesEveryTermAndItsScale:
    def test_the_breakdown_names_each_term_separately(self) -> None:
        """FR-008.9: *an operator who is over by 3 GB needs to know which term to change.*"""
        record = adapter_estimate().as_record()
        for key in (
            "weights_gib",
            "gradients_gib",
            "fp32_master_gib",
            "optimizer_states_gib",
            "activations_gib_low",
            "activations_gib_high",
            "total_gib_low",
            "total_gib_high",
        ):
            assert key in record, key

    def test_every_number_names_its_scale(self) -> None:
        record = adapter_estimate().as_record()
        for key, value in record.items():
            if isinstance(value, int | float) and not isinstance(value, bool):
                # `_gib_low` / `_gib_high` name the scale AND the end of the range, which
                # is the stronger form: an activation figure is a range, not a point.
                assert "_gib" in key or key.endswith("_count"), f"{key} does not name its scale"

    def test_the_record_says_it_is_an_adapter_run_and_how_the_figures_were_derived(self) -> None:
        record = adapter_estimate().as_record()
        assert record["run_type"] == "adapter"
        assert record["basis"] == "config"
        assert record["adapter_param_basis"] == "bound"

    def test_the_pessimistic_end_is_what_a_guard_would_compare(self) -> None:
        estimate = adapter_estimate()
        assert estimate.total_high_bytes > estimate.total_low_bytes
        assert estimate.total_high_bytes == estimate.fixed_bytes + estimate.activation_high_bytes


class TestTheModuleIsPureArithmetic:
    def test_it_imports_no_torch_and_makes_no_cuda_call(self) -> None:
        """What makes 4.0 shippable alone (FR-008.17) and runnable with no card (ADR-023).
        Asserted on the module's imports, not by convention."""
        import ast
        from pathlib import Path

        source = (
            Path(__file__).resolve().parents[3] / "backend/src/training/runtypes/adapter_budget.py"
        ).read_text(encoding="utf-8")
        imported: set[str] = set()
        for node in ast.walk(ast.parse(source)):
            if isinstance(node, ast.Import):
                imported |= {alias.name.split(".")[0] for alias in node.names}
            if isinstance(node, ast.ImportFrom) and node.module:
                imported.add(node.module.split(".")[0])
        for forbidden in ("torch", "peft", "transformers", "safetensors"):
            assert forbidden not in imported, forbidden

    def test_the_whole_module_runs_with_no_gpu_present(self) -> None:
        """Not a tautology on this host: ``torch`` is not installed here, so an accidental
        import would make every test in this file an error rather than a failure."""
        assert adapter_estimate().total_high_bytes > 0

    def test_free_vram_is_NOT_an_input_to_the_estimator(self) -> None:
        """The estimator is pure; the comparison lives in the guard. That keeps the
        arithmetic unit-testable with no card and the refusal in one place."""
        import inspect

        parameters = inspect.signature(adapter_budget.estimate).parameters
        assert not any("free" in name or "vram" in name for name in parameters)


class TestTheGuardStillUsesTheFullParameterEstimate:
    """The reported deviation, pinned so it cannot be forgotten or silently half-fixed.

    When the binding lands - ``guard_for`` choosing the adapter estimate for an adapter run -
    these two assertions go red and name exactly what changed. That is the point: a
    deviation nothing asserts is a deviation the next session rediscovers.
    """

    def test_the_bound_guard_takes_no_run_type_and_so_cannot_select_this_branch(self) -> None:
        import inspect

        parameters = inspect.signature(budget_guard.ComputedBudgetGuard.check).parameters
        assert set(parameters) == {"self", "spec", "cfg"}
        assert not any(
            "run_type" in name or "adapter" in name
            for name in inspect.signature(budget_guard.estimate).parameters
        )

    def test_f1s_estimator_has_no_adapter_branch(self) -> None:
        import inspect

        source = inspect.getsource(budget_guard.estimate)
        assert "adapter" not in source.lower(), (
            "if F1's estimator has grown an adapter branch, F8's FR-008.12 deviation is "
            "closed and this test plus the module docstring must be updated"
        )

    def test_the_gate_an_adapter_run_faces_today_is_CONSERVATIVE_not_permissive(self) -> None:
        """Fail-closed: the full-parameter figure is larger, so the guard refuses some runs
        that would have fitted and admits none that would not. The wrong answer, in the
        safe direction - which is why this is a reported deviation and not a defect that
        had to block the feature."""
        from training.acquisition.base import AcquisitionMode, AcquisitionSpec

        full = budget_guard.estimate(
            AcquisitionSpec(mode=AcquisitionMode.CREATE, config=LFM2_CONFIG), cfg()
        )
        assert full.total_high_bytes > adapter_estimate().total_high_bytes
