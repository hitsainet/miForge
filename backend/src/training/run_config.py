"""The run configuration R-02.13 names, validated and never clamped  (FR-001.22, FR-001.23).

Exactly four numbers - learning rate, warmup, batch size, gradient accumulation - plus the
optimizer-state precision, because **8-bit is R-02.13's default and not its only value**
(FR-001.23). R-02.42 likewise estimates *"optimizer states at their configured
precision"*, which only means something if the precision is configurable.

## Refused, never clamped

A clamped learning rate trains a model the operator did not ask for and reports success.
So every bound below raises, naming the field, and the refusal reaches the client as
ADR-013's envelope with ``run_config_invalid``. ADR-018's rule - *a value that is quietly
replaced is a fabricated value* - applies to a number as much as to a missing attribute.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator

from core.canonical import sha256_of

#: The optimizer-state precisions this project supports. ``8bit`` is R-02.13's default;
#: ``32bit`` is the escape hatch for a run small enough not to need the saving, and for
#: diagnosing whether an 8-bit state is itself the cause of a training problem.
OptimizerPrecision = Literal["8bit", "32bit"]

#: Upper bounds that exist to catch a typo, not to express a policy. A learning rate of 1.0
#: on a full-parameter update destroys the model in one step, and `1e-2` is already far past
#: anything sane - but the bound is deliberately loose, because a *narrow* bound that
#: refuses a legitimate value is indistinguishable, to the operator, from a bug.
MAX_LEARNING_RATE = 1.0


class RunConfig(BaseModel):
    """R-02.13's four values, the state precision, and the creation seed.

    Frozen and closed: an unknown key is refused rather than ignored, so a misspelled
    ``grad_accum`` cannot silently leave the real field at its default.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    learning_rate: float = Field(gt=0.0, le=MAX_LEARNING_RATE)
    warmup_steps: int = Field(ge=0)
    batch_size: int = Field(ge=1)
    gradient_accumulation: int = Field(default=1, ge=1)

    #: R-02.13's **default**, expressed as a default rather than hard-coded, which is what
    #: makes the recorded optimizer identity (`adamw-8bit`) a fact about this run rather
    #: than a constant.
    optimizer_state_precision: OptimizerPrecision = "8bit"

    #: Mirrors :attr:`training.acquisition.base.AcquisitionSpec.seed` so the seed appears in
    #: the optimizer's config hash as well as in the acquisition record: two runs that
    #: differ only by seed are two different runs, and a config hash that collapsed them
    #: would make the manifests claim they were the same.
    seed: int | None = None

    @field_validator("learning_rate")
    @classmethod
    def _finite(cls, value: float) -> float:
        """Refuse a non-finite learning rate.

        ``float("nan")`` passes ``gt=0`` checks in some orderings and would then poison the
        first update silently - the loss becomes NaN and nothing says why. ``canonical_bytes``
        refuses NaN too, so a NaN here would also make the config unhashable, several steps
        later and far from the cause.
        """
        if value != value or value in (float("inf"), float("-inf")):
            raise ValueError("learning_rate must be a finite number")
        return value

    @property
    def effective_batch_size(self) -> int:
        """``batch_size * gradient_accumulation`` - the number a loss is *per*.

        ADR-012: every number names its scale and sample. A loss without its effective
        batch size is not a measurement, and the two factors are configured separately, so
        the product is computed in one place rather than at each reader.
        """
        return self.batch_size * self.gradient_accumulation

    def optimizer_name(self) -> str:
        """``adamw-8bit`` or ``adamw-32bit``.

        The string shape is the one ``checkpoint-manifest-v1.json``'s ``optimizer.name``
        field exemplifies, so F5 writes this value through unchanged rather than deriving
        its own spelling.
        """
        return f"adamw-{self.optimizer_state_precision}"

    def config_hash(self) -> str:
        """Canonical hash of the hyperparameters, for ``optimizer.config_hash``.

        Through ``core.canonical`` (ADR-005), over the model dump rather than a hand-built
        dict: a hand-built dict is a second list of the fields, and it would stop including
        a field the moment one was added here.
        """
        return sha256_of(self.model_dump(mode="json"))
