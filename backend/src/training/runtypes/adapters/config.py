"""The adapter's run configuration, bounded by the schema  (FR-008.22; R-02.43).

Rank, alpha, target modules and dropout are **run configuration** (R-02.43), validated at
submit. **The bounds are the schema's** - read out of ``$defs.AdapterConfig`` at import
rather than restated as independent numbers, because two copies of a bound are two answers.

``scaling`` is not a field here. It is alpha over rank for this release and is written into
the manifest from :func:`training.runtypes.adapters.schema.scaling_default`; it is not an
operator input, because nothing in this release implements a second rule and an input the
code ignores is worse than no input.
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict, field_validator, model_validator

from core.errors import refuse
from training.runtypes.adapters import schema as adapter_schema

_DEF = adapter_schema.ADAPTER_CONFIG_DEF

#: Read from the vendored schema. ``rank`` carries ``minimum``, ``alpha``
#: ``exclusiveMinimum`` - the asymmetry is the schema's and is preserved rather than
#: flattened to "> 0" for both, which would admit ``rank: 0``.
RANK_MINIMUM: int = int(adapter_schema.bound(_DEF, "rank", "minimum"))
ALPHA_EXCLUSIVE_MINIMUM: float = float(adapter_schema.bound(_DEF, "alpha", "exclusiveMinimum"))
TARGET_MODULES_MIN_ITEMS: int = int(adapter_schema.bound(_DEF, "target_modules", "minItems"))
DROPOUT_MINIMUM: float = float(adapter_schema.bound(_DEF, "dropout", "minimum"))
DROPOUT_MAXIMUM: float = float(adapter_schema.bound(_DEF, "dropout", "maximum"))
DROPOUT_DEFAULT: float = float(adapter_schema.bound(_DEF, "dropout", "default"))


class AdapterConfigIn(BaseModel):
    """What an operator supplies for an adapter run.

    Frozen and closed: a misspelled ``lora_alpha`` is refused rather than ignored, which
    would otherwise leave ``alpha`` at a value the operator did not choose and record it in
    a manifest as though they had.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    rank: int
    alpha: float
    target_modules: tuple[str, ...]
    dropout: float = DROPOUT_DEFAULT

    @field_validator("rank")
    @classmethod
    def _rank_in_bounds(cls, value: int) -> int:
        if value < RANK_MINIMUM:
            raise ValueError(
                f"rank must be at least {RANK_MINIMUM} (adapter-manifest-v1.json's bound); "
                f"a rank of {value} describes no adapter"
            )
        return value

    @field_validator("alpha")
    @classmethod
    def _alpha_in_bounds(cls, value: float) -> float:
        if not value > ALPHA_EXCLUSIVE_MINIMUM:
            raise ValueError(
                f"alpha must be greater than {ALPHA_EXCLUSIVE_MINIMUM} "
                f"(adapter-manifest-v1.json's bound); the scale is alpha over rank, so "
                f"alpha={value} scales the adapter's contribution to nothing or inverts it"
            )
        return value

    @field_validator("dropout")
    @classmethod
    def _dropout_in_bounds(cls, value: float) -> float:
        if not DROPOUT_MINIMUM <= value <= DROPOUT_MAXIMUM:
            raise ValueError(
                f"dropout must be within {DROPOUT_MINIMUM}..{DROPOUT_MAXIMUM} "
                f"(adapter-manifest-v1.json's bounds); got {value}"
            )
        return value

    @model_validator(mode="after")
    def _at_least_one_target_module(self) -> AdapterConfigIn:
        if len(self.target_modules) < TARGET_MODULES_MIN_ITEMS:
            raise ValueError(
                f"at least {TARGET_MODULES_MIN_ITEMS} target module is required "
                f"(adapter-manifest-v1.json's minItems); an adapter attached to nothing "
                f"trains nothing and the loss would fall to zero while the model did not "
                f"change"
            )
        return self

    def manifest_block(self) -> dict[str, Any]:
        """The manifest's ``adapter`` object - **and the only source of its numbers**.

        ``rank`` and ``alpha`` here and ``r`` and ``lora_alpha`` in the ``LoraConfig``
        handed to ``peft`` come from **this one object** (task 9.2). miLLM's Feature 033
        loader cross-checks the two files and refuses on a mismatch, so a writer deriving
        the manifest's numbers independently of what it gave ``save_pretrained`` would
        produce a directory that validates against both schemas, loads under plain ``peft``
        and is **refused by miLLM** - failing acceptance criterion 15 for a reason neither
        schema can see. The agreement is asserted in
        :func:`training.runtypes.adapters.save.assert_config_agrees`.
        """
        return {
            "rank": self.rank,
            "alpha": self.alpha,
            "target_modules": list(self.target_modules),
            "dropout": self.dropout,
            "scaling": adapter_schema.scaling_default(),
        }

    @property
    def scale(self) -> float:
        """``alpha / rank`` - the ``alpha_over_r`` rule, computed in one place.

        Exposed so the merge does not recompute it: applying the scale twice, or not at
        all, produces a model that loads fine and answers differently.
        """
        return self.alpha / self.rank


def validate(payload: Any) -> AdapterConfigIn:
    """Coerce and validate an adapter configuration, refusing by name.

    Raises:
        Refusal: ``adapter_config_invalid`` (E-7), naming the field and the schema bound it
            violated. A Pydantic ``ValidationError`` reaching the API is a 500 where a 400
            naming the field belongs.
    """
    if isinstance(payload, AdapterConfigIn):
        return payload
    try:
        return AdapterConfigIn.model_validate(payload)
    except Exception as exc:
        raise refuse(
            "adapter_config_invalid",
            "This adapter configuration is outside the bounds adapter-manifest-v1.json "
            "declares. Refusing at submit rather than clamping: a clamped rank trains an "
            "adapter the operator did not ask for and records it as though they had.",
            detail=str(exc),
            bounds={
                "rank_minimum": RANK_MINIMUM,
                "alpha_exclusive_minimum": ALPHA_EXCLUSIVE_MINIMUM,
                "target_modules_min_items": TARGET_MODULES_MIN_ITEMS,
                "dropout_range": [DROPOUT_MINIMUM, DROPOUT_MAXIMUM],
            },
        ) from exc
