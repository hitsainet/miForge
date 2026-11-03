"""``create`` - a model that did not exist until now  (R-02.41; FR-008.1 - FR-008.8).

**Small on purpose.** R-02.41's whole content is *"step 0 is random instead of pretrained;
everything after it is the section 5.7 loop unchanged"*, plus three lineage facts - and F1's
:class:`training.acquisition.scratch.ScratchAcquirer` already did the instantiating. A large
``create.py`` would mean the loop had been duplicated, so :meth:`CreateRunType.prepare`
delegating to :func:`training.runtypes.base.all_parameters` is the design and not laziness.

## The least coupled capability in the project

No base model, no attachment, no sibling, nothing upstream at all - only a config, a
tokenizer and a corpus (FR-008.8). BRD-02 section 5.9: *"The architecture's most
tightly-coupled design was carrying its least-coupled capability."* It is what acceptance
criteria 12 and 20 pin, which is why a convenience that reached a sibling from this path
would break the criterion that proves the 2026-10-13 standalone reframe.

## Two inputs this run type needs and no shipped field carries

**The tokenizer.** BRD-02 section 8 says miForge never trains one, so a creation run is
*given* one - and F1's ``_instantiate`` deliberately returns ``tokenizer=None``, with
*"the caller supplies it, rather than a default tokenizer being invented for a model that
has none."* F8 is that caller, and **no field on ``AcquisitionSpec`` or ``RunConfig``
carries a tokenizer** (both are frozen and closed). So the refusal is raised at
:meth:`CreateRunType.prepare`, the earliest point at which the absence is a fact rather
than a prediction, and the field it should come from is owed to F6's run definition
(R-02.53).

**The signal source.** FR-008.6 requires a run-log note - not a refusal - when a creation
run's source selects from the model's own output. ``signal_source`` is likewise on neither
frozen input, so :func:`random_start_note` is a pure function and ``preflight`` takes the
value as an optional keyword. Its production caller is F6's run definition, which has not
landed; see the feature report.
"""

from __future__ import annotations

import logging
from typing import Any

from core.errors import refuse
from training.acquisition.base import (
    AcquiredModel,
    AcquisitionMode,
    AcquisitionSpec,
    lineage_for_creation,
)
from training.run_config import RunConfig
from training.runtypes.base import STEP, SaveContext, TrainablePlan, all_parameters
from training.runtypes.change import save_checkpoint

log = logging.getLogger(__name__)

#: The one signal source that makes sense at step 0 (R-02.41 bullet 4). Every other source
#: selects from the model's own output, and at step 0 that output is noise.
SENSIBLE_AT_STEP_ZERO = "corpus"


def random_start_note(signal_source: str | None) -> dict[str, Any] | None:
    """The run-log note for a creation run whose source learns from its own output.

    ``None`` when the source is a corpus or is not stated. **A note, never a refusal**
    (FR-008.6, R-02.41 bullet 4): refusing would be miForge overruling an operator who may
    want exactly this, and staying silent would be miForge letting a run learn from noise
    without saying so. Neither is acceptable, so it says so and proceeds.
    """
    if signal_source is None or signal_source == SENSIBLE_AT_STEP_ZERO:
        return None
    return {
        "code": "create_run_starts_from_random_weights",
        "signal_source": signal_source,
        "note": (
            "step 0 is random and this source selects from the model's own output, so the "
            "first batches are selected from noise. Not refused - a corpus source is the "
            "one that makes sense at step 0, and this run may be deliberate."
        ),
    }


class CreateRunType:
    """R-02.41's from-scratch run type. **Not** the acquisition, which is F1's."""

    name = "create"
    #: The same object every run type carries - the assertion that defends BRD-02 section
    #: 5.9's first sentence. A creation run introduces **no new step path**.
    step = staticmethod(STEP)

    def preflight(
        self,
        spec: AcquisitionSpec,
        cfg: RunConfig,
        *,
        signal_source: str | None = None,
        adapter: Any = None,
    ) -> None:
        """Refuse what this run type cannot start without, and note what it will not refuse.

        Only the architecture config can be checked from these two inputs; see the module
        docstring for the tokenizer and the signal source.

        Raises:
            Refusal: ``architecture_config_required`` when the spec carries no config.
        """
        if spec.mode is not AcquisitionMode.CREATE:
            raise refuse(
                "run_type_mode_mismatch",
                f"A 'create' run instantiates its parameters in this process, so its "
                f"acquisition mode must be 'create' and not {spec.mode.value!r}. Refusing "
                f"rather than overriding the mode: the two would then disagree about what "
                f"was trained and the acquisition record is what a manifest cites.",
                run_type=self.name,
                mode=spec.mode.value,
            )
        if spec.architecture_config is None and spec.config_path is None:
            raise refuse(
                "architecture_config_required",
                "A 'create' run needs an architecture config: it is what defines the model "
                "and what its GPU memory is estimated from, before anything is "
                "instantiated. Supply it inline or as a path.",
                run_type=self.name,
            )

        note = random_start_note(signal_source)
        if note is not None:
            log.info(note["note"], extra=note)

    def prepare(self, acquired: AcquiredModel, *, adapter: Any = None) -> TrainablePlan:
        """Every parameter of the model F1 instantiated - **identical to ``change``'s**.

        That identity is correct and should stay visible: R-02.41's content is that step 0
        is random, which the acquisition already did.

        Raises:
            Refusal: ``tokenizer_required`` (E-2, FR-008.7). miForge does not train a
                tokenizer, and a default one invented for a model that has none produces a
                run that trains and generates nonsense.
        """
        if acquired.tokenizer is None:
            raise refuse(
                "tokenizer_required",
                "This creation run was given no tokenizer. miForge does not train one "
                "and will not invent a default: a mismatched tokenizer "
                "produces a model that trains happily and generates nonsense. Supply a "
                "tokenizer, or reuse one from an existing model.",
                run_type=self.name,
            )
        return all_parameters(acquired)

    def save(self, ctx: SaveContext) -> Any:
        """``parent: null``, ``base_model`` = the config identity, ``domain`` written.

        The pair comes from F1's :func:`~training.acquisition.base.lineage_for_creation`
        rather than being spelled here, which is what makes M10 a one-line mutation: there
        is exactly one line that could write the model's own name into ``base_model`` or a
        non-null ``parent``.

        ``domain`` reaches F5 as ``ctx.domain`` and is **written either way** - null is an
        assertion that this run biases toward nothing, an absent key is an omission, and
        the schema keeps the field required precisely so the two are different (FR-008.4,
        M11). There is no ``or``, no ``getattr`` and no ``.get`` on that path.
        """
        base_model, parent = lineage_for_creation(config_identity=ctx.record.base_model)
        if ctx.record.parent is not None:
            raise refuse(
                "create_lineage_invalid",
                f"This creation run's acquisition record names a parent "
                f"({ctx.record.parent!r}). A created model descends from nothing: "
                f"'parent' is null and 'base_model' is the architecture config it came "
                f"from. Refusing rather than writing a lineage that would root the tree "
                f"at a checkpoint the model never descended from.",
                run_id=ctx.run_id,
                parent=ctx.record.parent,
            )
        return save_checkpoint(ctx, base_model=base_model, parent=parent)
