"""The computed memory budget  (R-02.42), pulled forward into F1 by operator decision.

**Whose file this is, and why it is here now.** R-02.42 belongs to F8, which PPRD section 6
places two waves after F1, and F1 is the feature that *allocates*. F1's own open question
OQ-1 put the ordering to the operator: either the estimator moves forward, or F1 ships a
pass-through and an allocator error is possible until F8 lands. **The operator pulled it
forward (2026-10-14.)** The seam is unchanged - this class implements
:class:`training.budget.BudgetGuard` exactly as declared, binds in
:mod:`services.acquisition`, and F8's work at this seam becomes *confirm, not rebuild*.

## Estimated from the architecture config, before anything is instantiated

R-02.42 as corrected on 2026-10-13 reads *"from the architecture config, before
``instantiate``"*. The original wording estimated from *"the resident model's parameter
count"*, and **a from-scratch run has no resident model to measure** - so the guard fired
too late for the run type that most needs it. Hence two inputs, exactly one of which is
present: a count (a load, read from the weight headers) or a config (a creation, read from
JSON). Neither is a refusal, not a pass.

## What is counted, and the one number that is a range

BRD-02 section 6's worked example for LFM2 1.2B full-parameter is the calibration:
gradients 2.4 GB, FP32 master 4.8 GB, 8-bit AdamW states 2.4 GB, checkpointed activations
1-3 GB. The first three are arithmetic in the parameter count. **Activations are not** -
they depend on sequence length, batch size and what the model recomputes, so they are a
*range* here and the refusal quotes the range rather than a false point estimate. ADR-012:
every number names its scale and sample.

**The ceiling is not hard-coded.** Section 6's ~2B ceiling was demoted to a worked example
on 2026-10-12 precisely so this computation governs instead, so the limit is the card's
**actually free** memory as ``nvidia-smi`` reports it, read at submit.
"""

from __future__ import annotations

import logging
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from core.errors import refuse
from core.gpu import Device
from training.acquisition.base import AcquisitionSpec
from training.budget import require_exactly_one_input
from training.run_config import RunConfig

log = logging.getLogger(__name__)

#: Bytes per parameter, per component, for a full-parameter update (BRD-02 section 6).
#:
#: ``weights`` is 2 because R-02.46 admits only FP16/BF16 (the dtype gate enforces it), and
#: ``gradients`` matches the weights' dtype. ``master`` is the FP32 copy R-02.13 requires
#: *in miForge's process*. The optimizer's two moments are 1 byte each at 8-bit and 4 each
#: at 32-bit - which is the whole reason the precision is configurable and recorded.
BYTES_WEIGHTS = 2
BYTES_GRADIENTS = 2
BYTES_FP32_MASTER = 4
OPTIMIZER_STATE_BYTES: Mapping[str, int] = {"8bit": 2, "32bit": 8}

#: Activation range as a multiple of the parameter *bytes*, calibrated against section 6's
#: worked example: 1-3 GB against a 1.2B model's 2.4 GB of FP16 weights. A range, not a
#: point: gradient checkpointing is on (R-02.13), so this is recompute-bounded and depends
#: on sequence length and batch size, neither of which R-02.13 puts in the run config.
ACTIVATION_LOW_FACTOR = 0.4
ACTIVATION_HIGH_FACTOR = 1.25

#: Headroom left unclaimed. The allocator fragments, CUDA's own context costs hundreds of
#: megabytes, and the node's cards are **shared** with a serving process and an
#: interpretation process - none of the three requests ``nvidia.com/gpu``, so all three can
#: be resident at once (verified in this cluster's manifests, not assumed).
SAFETY_FRACTION = 0.90

GIB = 1024**3


@dataclass(frozen=True)
class BudgetEstimate:
    """A run's projected footprint, with the activation figure kept as a range."""

    param_count: int
    weights_bytes: int
    gradient_bytes: int
    master_bytes: int
    optimizer_state_bytes: int
    activation_low_bytes: int
    activation_high_bytes: int
    #: Where the parameter count came from: ``"weights"`` for a load, ``"config"`` for a
    #: creation. Recorded because an estimate from a config is a *derivation* and one from
    #: weight headers is a *measurement*, and a reader comparing two runs needs to know.
    basis: str

    @property
    def fixed_bytes(self) -> int:
        """Everything that does not depend on sequence length or batch size."""
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
        """**The figure the guard compares against.** The pessimistic end, deliberately.

        Admitting a run on its optimistic estimate means it dies in the allocator at some
        later step - after the lease, after acquisition, usually minutes in - which is the
        precise failure R-02.42 exists to prevent. Refusing a run that *might* have fitted
        costs the operator a retry with a smaller batch; admitting one that does not costs
        them the run.
        """
        return self.fixed_bytes + self.activation_high_bytes

    def as_record(self) -> dict[str, Any]:
        """The estimate, in GiB, for the run log and for a refusal's ``details``.

        GiB rather than bytes because this is read by a person deciding what to change, and
        **named** rather than positional so a reader does not have to know the order.
        """
        return {
            "param_count": self.param_count,
            "basis": self.basis,
            "weights_gib": round(self.weights_bytes / GIB, 2),
            "gradients_gib": round(self.gradient_bytes / GIB, 2),
            "fp32_master_gib": round(self.master_bytes / GIB, 2),
            "optimizer_states_gib": round(self.optimizer_state_bytes / GIB, 2),
            "activations_gib_low": round(self.activation_low_bytes / GIB, 2),
            "activations_gib_high": round(self.activation_high_bytes / GIB, 2),
            "total_gib_low": round(self.total_low_bytes / GIB, 2),
            "total_gib_high": round(self.total_high_bytes / GIB, 2),
        }


def parameter_count_from_config(config: Mapping[str, Any]) -> int:
    """Estimate a transformer's parameter count from its architecture config.

    The standard decoder-only arithmetic, from the fields every Hugging Face config carries:
    embeddings, then per layer the attention projections and the MLP, then the output head.

    **Refuses on a config that does not state its shape.** ADR-018 again: a default hidden
    size would produce a confident estimate of a model that does not exist, and the guard's
    whole value is that its answer is about *this* architecture.
    """
    hidden = _positive_int(config, ("hidden_size", "n_embd", "d_model"))
    layers = _positive_int(config, ("num_hidden_layers", "n_layer", "num_layers"))
    vocab = _positive_int(config, ("vocab_size",))
    intermediate = _optional_positive_int(config, ("intermediate_size", "n_inner", "ffn_dim"))
    if intermediate is None:
        # The near-universal ratio when a config omits it. Stated as a documented fallback
        # for one derived field rather than for a *dimension*: omitting `intermediate_size`
        # is ordinary, omitting `hidden_size` means the config does not describe a model.
        intermediate = 4 * hidden

    heads = _optional_positive_int(config, ("num_attention_heads", "n_head")) or 1
    kv_heads = _optional_positive_int(config, ("num_key_value_heads",)) or heads
    head_dim = max(1, hidden // heads)

    # Attention: q is hidden x hidden; k and v are hidden x (kv_heads * head_dim), which is
    # what makes grouped-query attention cheaper and is why kv_heads is read separately -
    # assuming kv_heads == heads overstates a GQA model by a wide margin.
    attention = hidden * hidden + 2 * hidden * (kv_heads * head_dim) + hidden * hidden
    mlp = 3 * hidden * intermediate  # gate, up, down: the SwiGLU shape modern models use
    per_layer = attention + mlp + 2 * hidden  # + two norms

    embeddings = vocab * hidden
    head = 0 if _is_truthy(config.get("tie_word_embeddings")) else vocab * hidden

    return embeddings + layers * per_layer + hidden + head


def _positive_int(config: Mapping[str, Any], names: tuple[str, ...]) -> int:
    value = _optional_positive_int(config, names)
    if value is None:
        raise refuse(
            "architecture_config_invalid",
            f"The architecture config states none of {list(names)}, so the run's memory "
            f"footprint cannot be estimated. Refusing before any allocation rather than "
            f"assuming a dimension - a guessed size produces a confident estimate of a "
            f"model that does not exist.",
            field=names[0],
        )
    return value


def _optional_positive_int(config: Mapping[str, Any], names: tuple[str, ...]) -> int | None:
    for name in names:
        value = config.get(name)
        if isinstance(value, bool):
            continue
        if isinstance(value, int) and value > 0:
            return value
    return None


def _is_truthy(value: Any) -> bool:
    return bool(value) and value is not None


def estimate(spec: AcquisitionSpec, cfg: RunConfig) -> BudgetEstimate:
    """The projected footprint of ``spec`` under ``cfg`` (R-02.42).

    Raises:
        Refusal: ``budget_spec_ambiguous`` / ``budget_spec_incomplete`` from
            :func:`training.budget.require_exactly_one_input`, before any arithmetic.
    """
    require_exactly_one_input(spec)

    config = spec.architecture_config
    if config is not None:
        params = parameter_count_from_config(config)
        basis = "config"
    else:
        # `require_exactly_one_input` has established exactly one is present, so a None
        # here would be a bug in that function rather than a possible state.
        assert spec.param_count is not None
        params = spec.param_count
        basis = "weights"

    state_bytes = OPTIMIZER_STATE_BYTES[cfg.optimizer_state_precision]
    weights = params * BYTES_WEIGHTS
    return BudgetEstimate(
        param_count=params,
        weights_bytes=weights,
        gradient_bytes=params * BYTES_GRADIENTS,
        master_bytes=params * BYTES_FP32_MASTER,
        optimizer_state_bytes=params * state_bytes,
        # Scaled by the *effective* batch size relative to one, because accumulation
        # trades activations for steps: four micro-batches of 2 hold the activations of 2,
        # not of 8. `batch_size` alone is what is resident at once.
        activation_low_bytes=int(weights * ACTIVATION_LOW_FACTOR * cfg.batch_size),
        activation_high_bytes=int(weights * ACTIVATION_HIGH_FACTOR * cfg.batch_size),
        basis=basis,
    )


class ComputedBudgetGuard:
    """R-02.42's guard: refuse a run that will not fit, **before any allocation**.

    Args:
        device: the card the run is pinned to, already resolved by ``core.gpu`` at submit.
            Passed in rather than resolved here so the device is resolved **once** per
            submit - and so this guard never names a device of its own, which is what keeps
            the hard-coded-GPU ratchet's allowlist empty (ADR-008, foundation 1.6).
        safety_fraction: how much of the free memory may be claimed.
    """

    name = "computed"

    def __init__(self, device: Device, *, safety_fraction: float = SAFETY_FRACTION) -> None:
        self.device = device
        self.safety_fraction = safety_fraction
        #: Every call, for the ordering test that asserts the guard ran before `acquire()`.
        self.calls: list[tuple[AcquisitionSpec, RunConfig]] = []

    def check(self, spec: AcquisitionSpec, cfg: RunConfig) -> None:
        """Refuse ``budget_exceeded``, naming the estimate **and** the free figure."""
        self.calls.append((spec, cfg))
        projection = estimate(spec, cfg)

        free = self.device.free_memory_bytes
        if free is None:
            # The card's free memory is unknown. **Refused, not permitted**: this guard's
            # entire purpose is to compare against a real number, and "I could not read the
            # limit, so I allowed it" is the pass-through failure wearing the estimator's
            # name.
            raise refuse(
                "budget_unknown_capacity",
                f"The free memory on {self.device.uuid} could not be read, so the run's "
                f"projected {projection.as_record()['total_gib_high']} GiB cannot be "
                f"compared with anything. Refusing rather than admitting an unmeasured run.",
                gpu_uuid=self.device.uuid,
                estimate=projection.as_record(),
            )

        allowed = int(free * self.safety_fraction)
        if projection.total_high_bytes > allowed:
            raise refuse(
                "budget_exceeded",
                f"This run projects {projection.as_record()['total_gib_low']}-"
                f"{projection.as_record()['total_gib_high']} GiB on "
                f"{self.device.name} ({self.device.uuid}), which has "
                f"{round(free / GIB, 2)} GiB free - {round(allowed / GIB, 2)} GiB of it "
                f"claimable after headroom. Reduce batch_size, or use a smaller model. "
                f"Refusing before any allocation (R-02.42).",
                gpu_uuid=self.device.uuid,
                free_gib=round(free / GIB, 2),
                claimable_gib=round(allowed / GIB, 2),
                estimate=projection.as_record(),
            )

        log.info(
            "budget check passed",
            extra={
                "code": "budget_ok",
                "gpu_uuid": self.device.uuid,
                "free_gib": round(free / GIB, 2),
                **projection.as_record(),
            },
        )
