"""Wrap a module with ``peft`` so only the adapter is trainable  (FR-008.19, FR-008.22).

**Let ``peft`` own the parts ``peft`` owns.** Its saver writes ``adapter_config.json`` and
the PEFT tensor keys (``base_model.model.<module>.lora_A.weight`` / ``.lora_B.weight``) for
free, so an adapter is loadable by standard tooling and not only by this suite (FR-008.30).
Hand-rolling LoRA to satisfy a two-file layout statement is how the directory stops being
standard - which is the defect FPRD D5 reports, one layer down.

``scaling`` is **not** a ``peft`` field. ``peft``'s behaviour *is* alpha over rank; miForge
records the rule explicitly in its own manifest so a reader can refuse a future rule it does
not implement. Do not try to make ``peft`` carry it.
"""

from __future__ import annotations

import logging
from typing import Any

from core.errors import refuse
from training.runtypes.adapters.config import AdapterConfigIn
from training.runtypes.adapters.config import validate as validate_config

log = logging.getLogger(__name__)


def assert_only_adapter_requires_grad(wrapped: Any) -> list[Any]:
    """Return the trainable parameters, having asserted the base carries none.

    **A production assertion, not a test helper.** AC14's guarantee is that the base is
    byte-unchanged after an adapter run, and the cheapest place to enforce it is at
    construction: an optimizer built over a parameter whose ``requires_grad`` is set *will*
    write to it, and the hash comparison would then find out at the end of a run instead of
    at the start of one. M9 flips a ``requires_grad`` and
    ``test_adapter_requires_grad.py`` must go red.

    Raises:
        Refusal: ``adapter_base_is_trainable`` naming the offending parameters;
            ``adapter_has_no_trainable_parameters`` when nothing is trainable at all, which
            would make every step report a falling loss of zero while the model did not move.
    """
    trainable: list[Any] = []
    leaked: list[str] = []
    for name, parameter in wrapped.named_parameters():
        if not parameter.requires_grad:
            continue
        trainable.append(parameter)
        # PEFT names every adapter tensor with a `lora_` segment. A trainable parameter
        # without one is a base tensor, and an optimizer built over it would change the
        # bytes AC14 requires to be identical.
        if "lora_" not in name:
            leaked.append(name)

    if leaked:
        raise refuse(
            "adapter_base_is_trainable",
            f"{len(leaked)} base parameter(s) carry requires_grad in an adapter run, so "
            f"the optimizer would update the base and acceptance criterion 14's guarantee "
            f"- the base's bytes are unchanged - would be false. Refusing at construction "
            f"rather than discovering it from a hash comparison at the end of the run.",
            parameters=leaked[:10],
            leaked_count=len(leaked),
        )
    if not trainable:
        raise refuse(
            "adapter_has_no_trainable_parameters",
            "No parameter of the wrapped model is trainable, so every step would report a "
            "falling loss of zero while the model did not change. Check that the target "
            "modules name modules this architecture actually has.",
        )
    return trainable


def build(module: Any, config: Any) -> tuple[Any, AdapterConfigIn]:
    """Return ``(peft-wrapped module, validated config)``.

    Raises:
        Refusal: ``adapter_config_invalid`` from validation (E-7);
            ``ml_runtime_unavailable`` when ``peft`` is not installed - **refused by name**,
            never substituted with a hand-rolled implementation, because the thing ``peft``
            buys is a directory the rest of the world can load.
    """
    validated = validate_config(config)

    try:
        from peft import LoraConfig, get_peft_model
    except ImportError as exc:
        raise refuse(
            "ml_runtime_unavailable",
            "peft is not installed in this process, so a low-rank adapter cannot be built. "
            "Everything else in miForge runs without it (ADR-023), and miForge will not "
            "hand-roll LoRA instead: peft's saver is what writes adapter_config.json and "
            "the PEFT tensor keys that make the directory loadable by standard tooling.",
            detail=str(exc),
        ) from exc

    lora = LoraConfig(
        r=validated.rank,
        # peft annotates `lora_alpha: int`; the vendored schema allows a float and peft
        # computes the scale as alpha / r in floating point, so a float is correct here.
        lora_alpha=validated.alpha,  # type: ignore[arg-type]
        lora_dropout=validated.dropout,
        target_modules=list(validated.target_modules),
        bias="none",
        task_type="CAUSAL_LM",
    )
    wrapped = get_peft_model(module, lora)
    trainable = assert_only_adapter_requires_grad(wrapped)

    log.info(
        "adapter built",
        extra={
            "code": "adapter_built",
            "rank": validated.rank,
            "alpha": validated.alpha,
            "dropout": validated.dropout,
            "target_modules": list(validated.target_modules),
            "trainable_parameters": len(trainable),
        },
    )
    return wrapped, validated
