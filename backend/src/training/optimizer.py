"""AdamW, 8-bit states by default, FP32 master weights  (R-02.13; FR-001.21 - FR-001.23).

Four things R-02.13 fixes, and one it leaves configurable:

* **AdamW**, full-parameter. Adapter runs are R-02.43 and F8's.
* **8-bit optimizer states by default** (``bitsandbytes``, ADR-002) - a *default*, not the
  only value (FR-001.23), which is why the precision is in
  :class:`training.run_config.RunConfig` and recorded in the optimizer identity.
* **FP32 master weights in miForge's process**, kept explicitly rather than implied by a
  library default. A silently FP16 master set is a defect no reading catches: training
  proceeds, the loss falls, and small updates are lost to rounding for the whole run.
* **Gradient checkpointing on**, with ``use_cache`` off beside it. Not configurable -
  BRD-02 section 6's memory budget assumes it, and the two settings conflict so the library
  only warns.

The identity returned alongside is what F5 writes into
``checkpoint-manifest-v1.json``'s ``optimizer`` block: ``name`` like ``adamw-8bit`` with a
``config_hash`` over the hyperparameters. F1 records; F5 writes.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Literal

from core.errors import refuse
from training.run_config import RunConfig

log = logging.getLogger(__name__)

#: The two values ``OptimizerIdentity.state`` admits in the vendored schema. ``restored``
#: when trainer state was loaded for the resumed step, ``fresh`` when a new optimizer was
#: started on loaded weights - R-02.22 requires the distinction be *recorded*, because a
#: fresh optimizer on a mid-run checkpoint produces a loss discontinuity that is otherwise
#: unexplained.
OptimizerStateProvenance = Literal["restored", "fresh"]


@dataclass(frozen=True)
class OptimizerIdentity:
    """Which optimizer produced a step, and where its state came from.

    Mirrors the vendored schema's ``OptimizerIdentity`` exactly - three fields, two
    required - and deliberately carries no field the schema lacks. R-02.35's rule is that a
    manifest's fields are *exactly* the schema's, and a dataclass with a fourth field is how
    a writer ends up asserting one that does not exist.
    """

    name: str
    state: OptimizerStateProvenance
    config_hash: str | None = None

    def as_manifest_block(self) -> dict[str, Any]:
        """The ``optimizer`` block F5 writes into a checkpoint manifest."""
        return {"name": self.name, "state": self.state, "config_hash": self.config_hash}


def build(
    model: Any,
    cfg: RunConfig,
    *,
    state: OptimizerStateProvenance = "fresh",
) -> tuple[Any, OptimizerIdentity]:
    """Construct the optimizer R-02.13 requires, and its recorded identity.

    Args:
        model: the acquired parameter set.
        cfg: the run configuration. Already validated at the API edge - **refused there,
            never clamped**, because a clamped learning rate trains a model the operator did
            not ask for and reports success.
        state: ``"fresh"`` or ``"restored"``. Passed in rather than inferred: only the
            caller knows whether trainer state was found, and inferring it from whether the
            optimizer has any state loaded would read ``fresh`` for a restore that failed.

    Returns:
        ``(optimizer, identity)``.

    Raises:
        Refusal: ``ml_runtime_unavailable`` when torch is absent;
            ``optimizer_precision_unavailable`` when 8-bit states are asked for and
            ``bitsandbytes`` is not installed. **Refused, not downgraded to 32-bit**: a
            silent downgrade quadruples the optimizer's memory on a card budgeted to the
            megabyte, and the recorded identity would then name a precision that was not
            used.
    """
    torch = _torch()
    enable_gradient_checkpointing(model)
    params = [p for p in model.parameters() if p.requires_grad]
    if not params:
        raise refuse(
            "model_has_no_trainable_parameters",
            "No parameter of this model has requires_grad set, so an optimizer would have "
            "nothing to update and every step would report a falling loss of zero. "
            "R-02.13's updates are full-parameter.",
        )

    if cfg.optimizer_state_precision == "8bit":
        optimizer = _adamw_8bit(params, cfg)
    else:
        optimizer = torch.optim.AdamW(params, lr=cfg.learning_rate)

    assert_fp32_master_weights(optimizer, params)

    identity = OptimizerIdentity(
        name=cfg.optimizer_name(), state=state, config_hash=cfg.config_hash()
    )
    log.info(
        "optimizer built",
        extra={
            "code": "optimizer_built",
            "optimizer": identity.name,
            "state": identity.state,
            "config_hash": identity.config_hash,
            "trainable_parameters": sum(p.numel() for p in params),
        },
    )
    return optimizer, identity


def _adamw_8bit(params: list[Any], cfg: RunConfig) -> Any:
    """``bitsandbytes.optim.AdamW8bit``, refused by name when the wheel is absent."""
    try:
        from bitsandbytes.optim import AdamW8bit
    except ImportError as exc:
        raise refuse(
            "optimizer_precision_unavailable",
            "8-bit optimizer states need bitsandbytes, which is not installed in this "
            "process. Install the 'bnb' extra, or set optimizer_state_precision to "
            "'32bit' deliberately - miForge will not downgrade silently, because that "
            "quadruples the optimizer's memory and the recorded identity would name a "
            "precision that was not used.",
            requested="8bit",
            detail=str(exc),
        ) from exc
    return AdamW8bit(params, lr=cfg.learning_rate)


def enable_gradient_checkpointing(model: Any) -> None:
    """Turn gradient checkpointing on and the KV cache off (R-02.13).

    ``use_cache`` has to go off with it: the two conflict, the library only *warns*, and the
    warning is easy to miss in a worker's log - after which activations are cached and the
    memory gradient checkpointing was enabled to save is spent anyway.

    **And on a frozen base the checkpointed segment needs an input that carries a
    gradient.** ``gradient_checkpointing_enable()`` is *reentrant* by default, and reentrant
    ``torch.utils.checkpoint`` builds no autograd graph for a segment whose **inputs** all
    have ``requires_grad=False``. On a full-parameter run the input embedding is trainable,
    so its output requires grad and the graph is built. On an **adapter** run (R-02.43) the
    base - the embedding included - is frozen, so nothing entering the first checkpointed
    block requires grad, no graph is built, and ``loss.backward()`` raises *"element 0 of
    tensors does not require grad and does not have a grad_fn"*. The adapter run, the
    cheapest run type and the one that holds no barrier, could not complete a single
    backward pass.

    :func:`_needs_input_grad_hook` decides, and ``enable_input_require_grads()`` is the fix
    - the same remedy ``peft.prepare_model_for_kbit_training`` and ``transformers``'
    ``Trainer`` use, and in the same order (hook first, then enable).

    **Chosen over ``gradient_checkpointing_kwargs={"use_reentrant": False}``, which also
    works, because that changes checkpointing semantics for *every* model** - including the
    full-parameter path, which is the path BRD-02 §6's memory budget was reasoned about and
    the one this defect does not touch. The hook is conditional on the frozen-input case, so
    a full-parameter run executes exactly the code it executed before: the no-regression
    property is structural, not merely tested.
    """
    enable = getattr(model, "gradient_checkpointing_enable", None)
    if enable is None:
        raise refuse(
            "gradient_checkpointing_unavailable",
            "This model exposes no gradient_checkpointing_enable(), so R-02.13's "
            "requirement that gradient checkpointing be on cannot be satisfied. BRD-02 "
            "§6's memory budget assumes it; refusing rather than training without it.",
        )

    reason = _needs_input_grad_hook(model)
    if reason is not None:
        require = getattr(model, "enable_input_require_grads", None)
        if require is not None:
            require()
            log.info(
                "input gradients enabled for a checkpointed frozen base",
                extra={"code": "input_require_grads_enabled", "reason": reason},
            )
        else:
            # **Warned, not refused, and the checkpointing mode is not switched either.**
            # Both of those would assert something this function has not established: that
            # a backward pass on *this* model will fail. All that is known here is that
            # some parameters are frozen and the remedy is unavailable - and a model with
            # no `enable_input_require_grads` also has no `get_input_embeddings`, so it
            # cannot have reached the one case that *is* established
            # (`input_embeddings_frozen`); it is in the `unidentifiable` fallback, which is
            # "cannot tell".
            #
            # Refusing here turns a partially-frozen model that may train perfectly well
            # into a build failure - a worse defect than D3, because D3 failed loudly at
            # backward while this would fail a working configuration at construction. And
            # the failure, if it does come, is autograd's own loud RuntimeError, not a
            # silently wrong answer, so correctness is not what is at stake and ADR-013's
            # fail-closed rule does not reach it.
            #
            # `gradient_checkpointing_kwargs={"use_reentrant": False}` *would* work here -
            # the three-way probe proved it - and was deliberately not used: every real
            # `transformers` or `peft` model has `enable_input_require_grads`, so this
            # branch cannot be reached by one, and a fallback no caller can reach reads as
            # protection while protecting nothing.
            log.warning(
                "a partly frozen model exposes no enable_input_require_grads(); if a "
                "checkpointed block receives no gradient-carrying input, the first "
                "backward will raise 'element 0 of tensors does not require grad'",
                extra={"code": "input_require_grads_unavailable", "reason": reason},
            )

    enable()
    config = getattr(model, "config", None)
    if config is not None:
        config.use_cache = False


def _needs_input_grad_hook(model: Any) -> str | None:
    """Why a reentrant checkpointed segment would receive no gradient-carrying input.

    Returns the reason, so the log names **which** condition fired rather than only that
    one did. ``None`` means the hook is unnecessary and must not be installed - the
    full-parameter path.

    Two conditions, and *both* are false for a full-parameter run:

    * ``input_embeddings_frozen`` - the precise case. Something is trainable, the input
      embedding is not, so the first checkpointed block's input has no ``grad_fn``.
    * ``input_embeddings_unidentifiable`` - the fallback, for a partially-frozen model whose
      input embedding this function cannot find. A partially-frozen model is already outside
      the full-parameter path, so erring towards the hook here cannot reach it.

    The ``trainable``/``frozen`` pass is a **fast path, not a guard**, and is recorded as
    such because a mutation sweep proved it: deleting it leaves every case below answered
    correctly by the embedding check, and a model with nothing trainable still reaches
    :func:`build`'s ``model_has_no_trainable_parameters``, which is the better diagnosis. All
    it buys is not installing a forward hook on a model that is about to be refused. Do not
    read it as the thing that protects the full-parameter path - that is the
    ``requires_grad`` check below.
    """
    trainable = False
    frozen = False
    for parameter in model.parameters():
        if parameter.requires_grad:
            trainable = True
        else:
            frozen = True
        if trainable and frozen:
            break
    if not trainable or not frozen:
        return None

    embeddings = _input_embeddings(model)
    if embeddings is None:
        return "input_embeddings_unidentifiable"
    if any(parameter.requires_grad for parameter in embeddings.parameters()):
        return None
    return "input_embeddings_frozen"


def _input_embeddings(model: Any) -> Any:
    """The module whose output enters the first checkpointed block, or ``None``.

    ``None`` is *"cannot tell"*, never *"there are none"* - the caller must not read an
    absent answer as a negative one.
    """
    getter = getattr(model, "get_input_embeddings", None)
    if getter is None:
        return None
    try:
        embeddings = getter()
    except (AttributeError, NotImplementedError):
        return None
    if embeddings is None or not hasattr(embeddings, "parameters"):
        return None
    return embeddings if list(embeddings.parameters()) else None


def assert_fp32_master_weights(optimizer: Any, params: list[Any]) -> None:
    """Assert the optimizer's master parameters are FP32 (R-02.13), or refuse.

    **Asserted, not assumed.** R-02.13 requires FP32 master weights *in miForge's process*,
    and whether a given optimizer keeps them depends on the library and the version. A
    silently FP16 master set still trains - which is why only a check catches it.

    ``bitsandbytes``' 8-bit optimizers keep an FP32 master copy internally and expose it
    only after the first step, so a parameter that is itself FP16/BF16 with an 8-bit
    optimizer is the expected and correct state; what is refused is a **32-bit** optimizer
    holding non-FP32 parameters, where there is no master copy anywhere.
    """
    torch = _torch()
    is_eight_bit = type(optimizer).__name__.endswith("8bit")
    if is_eight_bit:
        return
    # Counted rather than listed: the actionable fact is "there is no master copy", and a
    # list of shapes for a 24-layer model is noise in a refusal the operator has to read.
    offenders = [p for p in params if p.dtype is not torch.float32]
    if offenders:
        raise refuse(
            "fp32_master_weights_absent",
            f"A 32-bit AdamW was built over {len(offenders)} non-FP32 parameters, so there "
            f"is no FP32 master copy anywhere and small updates will be lost to rounding "
            f"for the whole run. R-02.13 requires FP32 master weights in miForge's "
            f"process. Cast the model's parameters to float32, or use the 8-bit optimizer, "
            f"which keeps its own master copy.",
            non_fp32_parameters=len(offenders),
        )


def _torch() -> Any:
    try:
        import torch
    except ImportError as exc:
        raise refuse(
            "ml_runtime_unavailable",
            "torch is not installed in this process, so no optimizer can be built. "
            "Everything in miForge except a training step runs without it (ADR-023).",
            detail=str(exc),
        ) from exc
    return torch
