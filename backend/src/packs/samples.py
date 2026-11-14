"""The try-before-committing panels  (task 9.4; FR-007.19, R-02.39).

R-02.39: *"Each step that can be tried before it is committed is tried."* A sample of the
prompts at step 2, of the corpus or pairs at step 3, the dry-run at step 4, a **scored**
sample at step 5, and the **current** score against the proposed floor at step 6. A step
with nothing to sample (1 and 7) has no panel.

**Where a panel exists and cannot run, it says so by name and does not block the step**
(E7). A dry-run is advice; advice that becomes a wall in front of authoring is a worse
failure than no advice.

## Everything here reads through another feature's published entry point

Nothing in this module reads a part's rows, samples them, scores them or aggregates a score
itself. Each of those already exists and each has a rule inside it that would be lost by a
second implementation:

* ``services.scoring.sets.prompt_rows_from_part`` reads rows through F10's **validated**
  manifest view - F4 *"never calls miDataworks and never parses a handoff manifest itself"*,
  and neither does this.
* ``services.scoring.sets.resolve`` with a :class:`~services.scoring.sets.SampleConfig` does
  the **seeded, stratified** draw. A head slice would move when the file's order changed and
  nothing else did, which is why F4 refuses to truncate; a panel that took the first five
  rows would be showing the operator a different sample from the one a run would score.
* ``services.scoring.expectations`` + ``services.scoring.aggregate`` score and aggregate -
  including the refusal that an ``undecided`` verdict has **no** score, which is the rule
  that keeps a pending candidate out of a number.

So a sample panel and a run see the same prompts by construction, which is the only way the
panel's advice is about the run that will happen.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from packs.definition import PackVersionDefinition
from packs.dryrun import DRY_RUN_PARAMS, resolve_provider
from packs.errors import PACK_PART_UNRESOLVED, PackRefusal
from packs.references import ContractPartRef
from packs.resolver import Resolution, resolve_slots
from packs.settings import get_pack_settings
from packs.slots import SlotKind
from services.scoring import aggregate as aggregate_module
from services.scoring import expectations, sets

#: The seed every panel draws with. **Stated, not absent and not random**: an absent seed
#: makes the sample irreproducible while reading as reproducible, and a random one would show
#: the author a different five prompts every time they re-opened the step - which reads as
#: the data changing. ``GenerationParams.seed`` carries the same argument for decoding.
PANEL_SEED = 0

#: The aggregation method a panel's score uses. Named rather than defaulted, because *"the
#: method is recorded beside the number and two scores aggregated differently are not the
#: same measurement"* (F4's refusal when it is unknown).
PANEL_AGGREGATION = "mean"


@dataclass(frozen=True, slots=True)
class SlotSample:
    """What a try-panel shows for one slot.

    ``n`` **and** ``population`` together, always: ADR-012 requires every number to name its
    scale and sample, so "5 of 640 prompts" is expressible and a bare "5" is not.
    """

    slot: SlotKind
    rendered: str
    prompts: tuple[str, ...]
    references: tuple[str | None, ...]
    population: int
    seed: int

    @property
    def n(self) -> int:
        return len(self.prompts)

    def payload(self) -> dict[str, Any]:
        return {
            "slot": self.slot.value,
            "set": self.rendered,
            "n": self.n,
            "population": self.population,
            "seed": self.seed,
            "prompts": [
                {"prompt": prompt, "reference": reference}
                for prompt, reference in zip(self.prompts, self.references, strict=True)
            ],
        }


@dataclass(frozen=True, slots=True)
class ScoredSample:
    """A scored sample for step 5, or the current score against a floor for step 6."""

    sample: SlotSample
    score: float
    provider: str
    aggregation: str
    #: The proposed floor, for step 6. ``None`` at step 5, where no floor is in play.
    floor: float | None = None

    @property
    def breached(self) -> bool | None:
        """Whether the floor is **already** breached - the thing step 6 exists to show.

        ``None`` when no floor was proposed: *"is it breached"* has no answer without one,
        and ``False`` would read as "it is fine".
        """
        return None if self.floor is None else self.score < self.floor

    def payload(self) -> dict[str, Any]:
        return {
            **self.sample.payload(),
            "score": self.score,
            "provider": self.provider,
            "aggregation": self.aggregation,
            "floor": self.floor,
            "floor_breached": self.breached,
        }


def _why(filled: Any, resolution: Resolution | None) -> str:
    """Why a slot cannot be sampled, in one clause - empty, or the resolution's own reason."""
    if filled is None:
        return "it is empty"
    if resolution is None:
        return "it was not resolved"
    return resolution.reason or resolution.state


def slot_sample(
    session: Any,
    definition: PackVersionDefinition,
    slot: SlotKind,
    *,
    n: int | None = None,
    seed: int = PANEL_SEED,
    resolutions: Mapping[SlotKind, Resolution] | None = None,
) -> SlotSample:
    """A seeded, stratified sample of one slot's rows.

    Raises:
        PackRefusal: ``pack_part_unresolved`` when the slot is empty or its part does not
            resolve - naming the slot, because a panel that simply showed nothing would be
            indistinguishable from a set that is empty.
        Refusal: F4's ``evaluation_set_empty`` / ``evaluation_set_unresolvable`` for a part
            whose rows cannot be read, **unchanged** - the diagnosis belongs to the module
            that read the file.
    """
    size = n if n is not None else get_pack_settings().dry_run_sample_size
    filled = definition.slots.get(slot)
    resolution = (resolutions or resolve_slots(session, definition)).get(slot)
    if filled is None or resolution is None or resolution.state != "resolved":
        raise PackRefusal(
            PACK_PART_UNRESOLVED,
            f"The {slot.value!r} slot cannot be sampled: {_why(filled, resolution)}. Fill "
            f"it, or fix the reference, and the panel will show a sample.",
            slot=slot.value,
            service=None if resolution is None else resolution.service,
        )

    rows, markers, reference = _rows_for(resolution, slot)
    resolved_set = sets.resolve(
        slot.value,
        reference,
        rows,
        sample=sets.SampleConfig(size=min(size, len(rows)), seed=seed),
        markers=markers,
    )
    return SlotSample(
        slot=slot,
        rendered=resolved_set.reference.rendered,
        prompts=tuple(prompt.text for prompt in resolved_set.prompts),
        references=tuple(prompt.reference for prompt in resolved_set.prompts),
        population=resolved_set.population,
        seed=seed,
    )


def scored_sample(
    session: Any,
    definition: PackVersionDefinition,
    slot: SlotKind,
    *,
    expectation_config: Mapping[str, Any],
    catalog: Mapping[str, Mapping[str, Any]] | None = None,
    form: str = expectations.FORM_SELECTOR,
    n: int | None = None,
    seed: int = PANEL_SEED,
    provider_name: str | None = None,
    floor: float | None = None,
) -> ScoredSample:
    """Score a sample of one slot, for steps 5 and 6.

    The same path a run's evaluation takes - generate through the provider, score through
    F3, aggregate through F4 - so the number the author sees at step 6 is the number the run
    will compare against the floor.

    Raises:
        PackRefusal: ``provider_not_available`` naming the provider (E7) - **the step still
            saves**; ``pack_part_unresolved`` for an unsamplable slot.
        Refusal: F4's ``evaluation_incomplete`` when any verdict is ``undecided``, which is
            the rule that keeps a pending candidate out of a score, and F3's configuration
            refusals unchanged.
    """
    # THE CONFIGURATION IS RESOLVED BEFORE ANYTHING IS GENERATED, and the order is the
    # point: resolution is where an unusable expectation is refused - a human-review
    # configuration, which cannot score a set at all, or an unknown type - and refusing
    # after a generation pass would have spent a provider call (and, on a real endpoint,
    # money and a minute) to reach the same refusal. The same argument F6 makes for putting
    # the pack gate first in its submit order: every later step costs something.
    expectation = expectations.resolve_expectation(
        expectation_config,
        form=form,
        catalog=catalog,
        # R-02.40's honesty rule: a configuration cited by a pack is a pack-sourced
        # configuration, and F4 names that provenance `"pack"` itself.
        source="pack",
    )
    sample = slot_sample(session, definition, slot, n=n, seed=seed)
    provider = resolve_provider(provider_name)
    completions = provider.generate(list(sample.prompts), DRY_RUN_PARAMS)
    scored = expectations.score_completions(
        expectation,
        list(sample.prompts),
        [completion.text for completion in completions],
        references=list(sample.references),
        batch_id="pack-authoring-panel",
    )
    score = aggregate_module.aggregate(PANEL_AGGREGATION, scored.values)
    return ScoredSample(
        sample=sample,
        score=score,
        provider=provider.identity.record(),
        aggregation=PANEL_AGGREGATION,
        floor=floor,
    )


def _rows_for(
    resolution: Resolution, slot: SlotKind
) -> tuple[list[dict[str, Any]], tuple[str, ...] | None, sets.SetReference]:
    """Rows, split markers and the pinned reference for a resolved slot.

    Built from the resolution rather than re-queried: the resolution already came from F10's
    one lookup, and a second query could answer differently - which is how a panel ends up
    showing a sample of a part the run will not use.
    """
    part_id = resolution.part_id
    local_path = resolution.local_path
    if not local_path:
        raise PackRefusal(
            PACK_PART_UNRESOLVED,
            f"The part pinned in the {slot.value!r} slot is indexed without a directory, so "
            f"its rows cannot be read. Reindex, or re-import the part.",
            slot=slot.value,
            part_id=part_id,
        )

    structured: Mapping[str, Any] | None
    ref = resolution.ref
    if isinstance(ref, ContractPartRef):
        structured = ref.as_contract_dict()
    else:
        structured = ref.as_external_dict()

    rows, markers, reference = _load_slot(
        {
            "local_path": local_path,
            "part_id": part_id,
            "rendered": resolution.rendered,
            "external_provenance": None if isinstance(ref, ContractPartRef) else structured,
        }
    )
    return rows, markers, reference


def _load_slot(
    slot: Mapping[str, Any],
) -> tuple[list[dict[str, Any]], tuple[str, ...], sets.SetReference]:
    """F4's production slot loader, imported lazily.

    Lazily because :mod:`services.scoring.entrypoint` imports the run model and the run log,
    and a pack panel must not drag the training side of the application into the request
    path that lists packs.
    """
    from services.scoring.entrypoint import load_slot

    return load_slot(slot)


def sample_prompts(
    session: Any,
    definition: PackVersionDefinition,
    *,
    n: int | None = None,
    seed: int = PANEL_SEED,
    resolutions: Mapping[SlotKind, Resolution] | None = None,
) -> SlotSample:
    """The dry-run's prompts: **the pack's own prompt set** (AC11).

    A separate name because AC11 is explicit about *whose* prompts a dry-run answers, and a
    caller reading ``slot_sample(..., SlotKind.PROMPT_SET)`` at a call site could as easily
    have passed another slot.
    """
    return slot_sample(
        session, definition, SlotKind.PROMPT_SET, n=n, seed=seed, resolutions=resolutions
    )
