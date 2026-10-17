"""R-02.42's adapter branch: the term that decides on a shared card  (FR-008.12).

R-02.42's last sentence: *"for an adapter run the estimate covers the adapter and its
optimizer state, not full-parameter gradients - which is why adapters fit where
full-parameter does not."*

**Pure arithmetic, and it must stay that way.** No ``torch`` import, no CUDA call, no model
load: a config and some integers in, bytes out. That is what makes it runnable on a machine
with no card (ADR-023) and what lets R-02.42's estimator be pulled forward as a scheduling
decision rather than a refactor (FPRD D11).

## What changes against the full-parameter estimate, and what does not

The **base term stays**, because on a shared card the base copy is the term that decides.
What shrinks is gradients, the FP32 master copy and the optimizer states: they cover the
**adapter's** parameters, which are tens of megabytes rather than gigabytes. Activations are
unchanged - a forward and backward pass through the base still materialises them.

## The adapter's own parameter count is a BOUND, and it says so

A LoRA pair on a linear of shape ``(in, out)`` holds ``rank * (in + out)`` parameters, so an
exact count needs each targeted module's shape - which is only knowable once the model
exists. Before instantiation the only inputs are the architecture config and the target
module *names*, so :func:`adapter_parameter_bound` returns a deliberately **conservative
upper bound**: every targeted module is assumed to be the widest linear the architecture has
(``hidden x intermediate``). It is reported as ``bound`` rather than ``count`` in the
estimate, because an operator who is over by 300 MB needs to know whether the figure was
measured or bounded - and because a conservative estimate is a product decision worth
recording beside the number rather than hiding inside it.

## Where this is reached from, honestly

``POST /api/v1/budget/estimate`` calls it, so an operator sees an adapter run's real
footprint before submitting (FR-008.18). **The guard bound in** ``services.acquisition``
**does not**: ``guard_for`` constructs F1's ``ComputedBudgetGuard``, which has no run-type
input, and both that function and ``training/budget_guard.py`` are outside this feature's
write scope. The consequence is bounded and in the safe direction - an adapter run is
gated on the *full-parameter* estimate, so the guard refuses some runs that would have
fitted and admits none that would not - but it is a real deviation from FR-008.12 and it is
reported rather than papered over. The one-line binding it needs is in the feature report.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from core.errors import refuse
from training.budget_guard import (
    ACTIVATION_HIGH_FACTOR,
    ACTIVATION_LOW_FACTOR,
    BYTES_FP32_MASTER,
    BYTES_GRADIENTS,
    BYTES_WEIGHTS,
    GIB,
    OPTIMIZER_STATE_BYTES,
    parameter_count_from_config,
)
from training.run_config import RunConfig
from training.runtypes.adapters.config import AdapterConfigIn

#: A LoRA pair holds two matrices, ``A`` (in x rank) and ``B`` (rank x out), so one
#: targeted linear costs ``rank * (in + out)`` parameters - both matrices together, which
#: is why nothing below multiplies by two again.
LORA_MATRICES_PER_MODULE = 2


@dataclass(frozen=True)
class AdapterBudgetEstimate:
    """An adapter run's projected footprint, term by term.

    ``adapter_param_basis`` is ``"bound"`` or ``"count"``: whether the adapter's parameter
    figure was bounded from the architecture config or measured from a built model. ADR-012
    - every number names its scale and sample, and a bound is a different kind of number
    from a count.
    """

    base_param_count: int
    adapter_param_count: int
    adapter_param_basis: str
    weights_bytes: int
    gradient_bytes: int
    master_bytes: int
    optimizer_state_bytes: int
    activation_low_bytes: int
    activation_high_bytes: int
    basis: str

    @property
    def fixed_bytes(self) -> int:
        return (
            self.weights_bytes
            + self.gradient_bytes
            + self.master_bytes
            + self.optimizer_state_bytes
        )

    @property
    def total_low_bytes(self) -> int:
        return self.fixed_bytes + self.activation_low_bytes

    @property
    def total_high_bytes(self) -> int:
        """The pessimistic end - the figure a guard would compare against.

        Same rule as the full-parameter estimate: admitting a run on its optimistic figure
        means it dies in the allocator minutes in, which is the failure R-02.42 exists to
        prevent.
        """
        return self.fixed_bytes + self.activation_high_bytes

    def as_record(self) -> dict[str, Any]:
        """The estimate for the run log, the preview and a refusal's ``details``.

        **The breakdown by term is not optional** (FR-008.9): an operator who is over by 3
        GB needs to know which term to change, and the base term is named separately
        because on a shared card it is the one that decides.
        """
        return {
            "run_type": "adapter",
            "base_param_count": self.base_param_count,
            "adapter_param_count": self.adapter_param_count,
            "adapter_param_basis": self.adapter_param_basis,
            "basis": self.basis,
            "weights_gib": round(self.weights_bytes / GIB, 3),
            "gradients_gib": round(self.gradient_bytes / GIB, 3),
            "fp32_master_gib": round(self.master_bytes / GIB, 3),
            "optimizer_states_gib": round(self.optimizer_state_bytes / GIB, 3),
            "activations_gib_low": round(self.activation_low_bytes / GIB, 3),
            "activations_gib_high": round(self.activation_high_bytes / GIB, 3),
            "total_gib_low": round(self.total_low_bytes / GIB, 3),
            "total_gib_high": round(self.total_high_bytes / GIB, 3),
        }


def _dimension(config: Mapping[str, Any], names: tuple[str, ...]) -> int | None:
    for name in names:
        value = config.get(name)
        if isinstance(value, bool):
            continue
        if isinstance(value, int) and value > 0:
            return value
    return None


def adapter_parameter_bound(config: Mapping[str, Any], adapter: AdapterConfigIn) -> int:
    """A conservative upper bound on the adapter's parameter count, from the config alone.

    Every targeted module is assumed to be the architecture's widest linear
    (``hidden x intermediate``), on every layer. That over-counts a run targeting only
    ``q_proj`` and ``v_proj``, deliberately and in the safe direction, and the estimate
    labels the figure ``bound``.

    Raises:
        Refusal: ``architecture_config_invalid`` when the config states no hidden size or
            layer count. A guessed dimension produces a confident estimate of a model that
            does not exist (ADR-018), which is worse here than refusing, because the whole
            value of the figure is that it is about *this* architecture.
    """
    hidden = _dimension(config, ("hidden_size", "n_embd", "d_model"))
    layers = _dimension(config, ("num_hidden_layers", "n_layer", "num_layers"))
    if hidden is None or layers is None:
        raise refuse(
            "architecture_config_invalid",
            "The architecture config states no hidden size or layer count, so the "
            "adapter's size cannot be bounded. Refusing before any allocation rather than "
            "assuming a dimension.",
            field="hidden_size" if hidden is None else "num_hidden_layers",
        )
    intermediate = _dimension(config, ("intermediate_size", "n_inner", "ffn_dim")) or 4 * hidden
    widest = max(hidden, intermediate)
    # `rank * (in + out)` is A and B together; see LORA_MATRICES_PER_MODULE.
    per_module = adapter.rank * (hidden + widest)
    return per_module * len(adapter.target_modules) * layers


def estimate(
    *,
    cfg: RunConfig,
    adapter: AdapterConfigIn,
    architecture_config: Mapping[str, Any] | None = None,
    base_param_count: int | None = None,
    adapter_param_count: int | None = None,
) -> AdapterBudgetEstimate:
    """An adapter run's footprint. **Exactly one** base input, as R-02.42 requires.

    Args:
        architecture_config: for a base that does not exist yet, or whose count is unknown.
        base_param_count: for a base that was loaded, read from the weight headers.
        adapter_param_count: the realised adapter count, when the model exists. ``None``
            bounds it from the config and labels the figure ``bound``.

    Raises:
        Refusal: ``budget_spec_ambiguous`` when both base inputs are given - they are two
            independent estimates of the same thing and nothing says which to trust;
            ``budget_spec_incomplete`` when neither is. **A guard that cannot estimate must
            not permit**, so neither is a refusal rather than a pass (FR-008.10, E-5).
    """
    has_config = architecture_config is not None
    has_count = base_param_count is not None
    if has_config and has_count:
        raise refuse(
            "budget_spec_ambiguous",
            "An adapter budget was given both an architecture config and a parameter "
            "count. They are two independent estimates of the same base and nothing says "
            "which to trust, so this is refused rather than resolved by precedence.",
            base_param_count=base_param_count,
        )
    if not has_config and not has_count:
        raise refuse(
            "budget_spec_incomplete",
            "An adapter budget needs either the base's parameter count or its architecture "
            "config; it was given neither, so the run's footprint cannot be estimated. "
            "**A guard that cannot estimate must not permit**: refusing here rather than "
            "letting an unmeasured run reach the allocator.",
            run_type="adapter",
        )

    if architecture_config is not None:
        base_params = parameter_count_from_config(architecture_config)
        basis = "config"
        adapter_params = (
            adapter_param_count
            if adapter_param_count is not None
            else adapter_parameter_bound(architecture_config, adapter)
        )
        adapter_basis = "count" if adapter_param_count is not None else "bound"
    else:
        assert base_param_count is not None  # established above
        base_params = base_param_count
        basis = "weights"
        if adapter_param_count is None:
            raise refuse(
                "budget_spec_incomplete",
                "An adapter budget estimated from a parameter count needs the adapter's "
                "own parameter count too: the adapter's size depends on each targeted "
                "module's shape, which a base parameter count does not describe. Supply "
                "the architecture config instead, or the realised adapter count.",
                run_type="adapter",
            )
        adapter_params = adapter_param_count
        adapter_basis = "count"

    state_bytes = OPTIMIZER_STATE_BYTES[cfg.optimizer_state_precision]
    weights = base_params * BYTES_WEIGHTS
    return AdapterBudgetEstimate(
        base_param_count=base_params,
        adapter_param_count=adapter_params,
        adapter_param_basis=adapter_basis,
        weights_bytes=weights,
        # THE WHOLE POINT: these three cover the ADAPTER, not the base.
        gradient_bytes=adapter_params * BYTES_GRADIENTS,
        master_bytes=adapter_params * BYTES_FP32_MASTER,
        optimizer_state_bytes=adapter_params * state_bytes,
        # Unchanged: a forward and backward pass through the base still materialises them.
        activation_low_bytes=int(weights * ACTIVATION_LOW_FACTOR * cfg.batch_size),
        activation_high_bytes=int(weights * ACTIVATION_HIGH_FACTOR * cfg.batch_size),
        basis=basis,
    )
