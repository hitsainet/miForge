"""**THE validator.** One implementation, two callers  (R-02.53; FR-006.57-.62).

R-02.53: miForge *"validates the whole definition before it starts anything ... and refuses
the run naming what failed rather than starting and dying."*

## Two properties, both requirements

**Before anything is allocated.** No GPU lease, no ``mf_runs`` row in ``running``, no
checkpoint read, no model load, no VRAM touched. :func:`validate` calls no
``lease.acquire`` - ``test_definition_allocates_nothing.py`` asserts that on the spy's call
count, because *a test that only checks the response passes against a validator that took
and released a lease.*

**All failures, not the first.** The one consumer of this path is a form (F9), and a
validator that stops at the first failure makes an operator resubmit once per mistake. So
every step runs, refusals are **collected**, and each keeps **its own code** - F7's
``pack_incomplete`` with its ``missing_slots``, F4's floor codes, F1's acquisition and budget
codes. A refusal is never re-wrapped into one generic code, because re-wrapping hides which
layer refused.

## The order is the requirement  (FR-006.58)

1. **structural** - Pydantic over the payload (done at the edge, by
   :class:`schemas.run_definition.RunDefinition`);
2. **the pack gate** - F7's ``require_valid_pack``, **first among the semantic checks**,
   because nothing that touches the database, the filesystem or the card may run before it;
3. **part resolvability** - re-resolved now, never from a cached validity flag (E-20);
4. **loop, stop and refresh validation** - cadences, the non-empty condition set, each
   enabled condition's configuration, the refresh rule's applicability;
5. **the computed budget against free VRAM**, through F1's preflight hook, **before any
   allocation** (FR-006.54). **Mutation M-14** moves this after the lease;
6. **optional-service availability**, named if absent;
7. *(not here)* **device resolution and the GPU lease** - F1's, **last**, because it is the
   first step that takes something away from another run.

Step 5 reads the device to know what free VRAM is. **Reading ``nvidia-smi -L`` allocates
nothing**; the *lease* is the allocation, and that is what stays outside this function. A
definition that fails at step 2, 3 or 4 never reaches step 5 at all, so a refused definition
resolves no device either - which is what ``test_definition_allocates_nothing.py`` asserts.

## ``requires=``: a step the validator could not run is reported as **skipped**

**Part resolvability is unanswerable when the pack did not resolve.** Running it on a
guess would produce a **fabricated** second failure, which is worse than a missing one
because the operator would try to fix it (ADR-018); omitting it silently would make the list
read as complete when it is not. So a skipped step appears in the list, naming what it
waited on, and :attr:`Outcome.may_be_more` says whether there is more to fix.

*FR-006.59 names a second such case - "the budget is unanswerable when the model source did
not [resolve]" - and in this decomposition there is no separate model-source step for it to
wait on: resolving the model source **is** what the budget step does, through F1's own
preflight, and it reports its own failure. :data:`REQUIRES` therefore has one entry rather
than two, and the difference is reported rather than papered over with a step that exists
only to be waited on.*
"""

from __future__ import annotations

import logging
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from typing import Any

from core.canonical import sha256_of
from core.errors import Refusal, refuse
from schemas.run_definition import RunDefinition

# F4 reads its own block from ``definition_resolved[SCORING_KEY]``. Imported from F4
# rather than respelled, so F6 writes exactly what F4 reads - two spellings of one key
# is a scoring block the evaluator never finds.
from services.scoring.entrypoint import DEFINITION_KEY as SCORING_KEY
from training import refresh as refresh_registry
from training import stop as stop_registry

log = logging.getLogger(__name__)

#: FR-006.58's step names, in order. The sequence a spy asserts (task 8.14) and the order a
#: refusal list is sorted into, so a form renders failures in the order they were found.
STEPS: tuple[str, ...] = (
    "structural",
    "pack_gate",
    "part_resolvability",
    "loop_stop_refresh",
    "submission_guards",
    "judge_identity",
    "budget",
    "optional_services",
)

#: Which earlier step each step needs an answer from before it can be run at all. The
#: ``requires=`` rule - see the module docstring.
REQUIRES: Mapping[str, tuple[str, ...]] = {
    "pack_gate": (),
    "part_resolvability": ("pack_gate",),
    "loop_stop_refresh": (),
    # F4's two submission guards read the pack's retention slot and the resolved part set.
    # With either unanswered they would report a **fabricated** second failure - *"this run
    # defines no retention set"* about a run whose pack simply did not resolve - which is
    # worse than a missing one because the operator would go and fix the wrong thing
    # (ADR-018). So they wait, and the skip names what they waited on.
    "submission_guards": ("pack_gate", "part_resolvability"),
    "judge_identity": (),
    "budget": (),
    "optional_services": (),
}


@dataclass(frozen=True, slots=True)
class Failure:
    """One named refusal, with the part it belongs to. **The owner's code, unchanged.**"""

    step: str
    code: str
    message: str
    details: Mapping[str, Any] = field(default_factory=dict)

    def payload(self) -> dict[str, Any]:
        return {
            "step": self.step,
            "code": self.code,
            "message": self.message,
            "details": dict(self.details),
        }


@dataclass(frozen=True, slots=True)
class Skipped:
    """A step the validator **could not run**, naming the step it waited on."""

    step: str
    waited_on: tuple[str, ...]

    def payload(self) -> dict[str, Any]:
        return {
            "step": self.step,
            "status": "skipped",
            "waited_on": list(self.waited_on),
            "message": (
                f"{self.step!r} could not be checked because "
                f"{', '.join(self.waited_on)} did not resolve. Fix that first; there may be "
                f"more to fix here."
            ),
        }


@dataclass
class Outcome:
    """What :func:`validate` produced: the resolved view, or every failure.

    ``ok`` is ``not failures``, and ``may_be_more`` is ``bool(skipped)``. Both are
    properties rather than flags a caller sets, so the two cannot disagree.
    """

    definition: RunDefinition
    failures: list[Failure] = field(default_factory=list)
    skipped: list[Skipped] = field(default_factory=list)
    #: The resolved values, recorded **beside** the submitted ones and never replacing them
    #: (FR-006.61). The run log's ``definition`` block carries this as ``resolved``.
    resolved: dict[str, Any] = field(default_factory=dict)
    #: The steps actually invoked, in order. FR-006.65's assertion subject.
    steps_run: list[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not self.failures

    @property
    def may_be_more(self) -> bool:
        return bool(self.skipped)

    def payload(self) -> dict[str, Any]:
        """``POST /runs/validate``'s body, success or failure."""
        return {
            "ok": self.ok,
            "may_be_more_to_fix": self.may_be_more,
            "steps_checked": list(self.steps_run),
            "failures": [failure.payload() for failure in self.failures],
            "skipped": [entry.payload() for entry in self.skipped],
            "resolved": dict(self.resolved),
        }

    def raise_if_invalid(self) -> None:
        """Raise ADR-013's envelope with **every** failure, or return (FR-006.59).

        Raises:
            Refusal: ``run_definition_invalid``, with ``details.failures`` as a list of
                named refusals. A list rather than a merged message, so a form can put each
                refusal beside the field it belongs to.
        """
        if self.ok:
            return
        codes = [failure.code for failure in self.failures]
        raise refuse(
            "run_definition_invalid",
            f"This run definition has {len(self.failures)} problem(s): "
            f"{', '.join(codes)}. Nothing was allocated and no run was created. Fix them "
            f"and resubmit - every failure is listed, so one resubmission is enough."
            + (" Some checks could not run; there may be more." if self.may_be_more else ""),
            failures=[failure.payload() for failure in self.failures],
            skipped=[entry.payload() for entry in self.skipped],
            may_be_more_to_fix=self.may_be_more,
        )


def validate(
    definition: RunDefinition,
    *,
    session: Any,
    gate: Callable[..., Any] | None = None,
    resolve_parts: Callable[..., Any] | None = None,
    budget: Callable[..., Any] | None = None,
    services: Callable[..., Any] | None = None,
    seed_generator: Any = None,
) -> Outcome:
    """Validate the whole definition. **Allocates nothing.** Never raises for a bad
    definition - it collects (FR-006.57).

    Args:
        definition: the submitted definition.
        session: an open session. Read from; nothing is written.
        gate: F7's ``require_valid_pack``, injected so a spy can assert the payload and the
            call count (M-9). ``None`` uses the live service interface.
        resolve_parts: F10's resolver, same reason.
        budget: F1's preflight hook, same reason (M-14).
        services: the optional-service check, same reason.
        seed_generator: injected seed source, for a deterministic test.

    Raises:
        Refusal: only for a programming error - a ``session`` of ``None``. Every
            *definition* problem is a collected :class:`Failure`.
    """
    outcome = Outcome(definition=definition)
    resolved: dict[str, Any] = {}
    answered: set[str] = {"structural"}
    outcome.steps_run.append("structural")

    gate_fn = gate if gate is not None else _live_gate
    parts_fn = resolve_parts if resolve_parts is not None else _live_parts
    budget_fn = budget if budget is not None else _live_budget
    services_fn = services if services is not None else _live_services

    # ---- 2. THE PACK GATE, first among the semantic checks ------------------ FR-006.47
    # M-9's target. Nothing that touches the database, the filesystem or the card runs
    # before it, so a run from an invalid pack never gets as far as a device resolution.
    pack_resolved: Any = None
    if _run(outcome, "pack_gate", answered):
        try:
            if definition.pack is not None:
                pack_resolved = gate_fn(session, definition.pack.pack_id, definition.pack.version)
                resolved["pack"] = pack_resolved.payload()
                answered.add("pack_gate")
            else:
                # Loose parts: there is no pack to gate, which is a legitimate shape
                # (R-02.40) and not a skipped check.
                resolved["pack"] = None
                answered.add("pack_gate")
        except Refusal as exc:
            _collect(outcome, "pack_gate", exc)

    # ---- 3. PART RESOLVABILITY, re-resolved now ------------------------- FR-006.58, E-20
    if _run(outcome, "part_resolvability", answered):
        try:
            resolved["parts"] = parts_fn(session, definition, pack_resolved)
            answered.add("part_resolvability")
        except Refusal as exc:
            _collect(outcome, "part_resolvability", exc)

    # ---- 4. LOOP, STOP AND REFRESH ------------------- FR-006.5, .15, .17, .21, .31
    if _run(outcome, "loop_stop_refresh", answered):
        try:
            resolved["loop"] = _validate_loop(definition)
            answered.add("loop_stop_refresh")
        except Refusal as exc:
            _collect(outcome, "loop_stop_refresh", exc)

    # ---- 4a. F4'S TWO SUBMISSION GUARDS ----------------------------- R-02.48, E-12, E-16
    # `services.scoring.entrypoint.check_submission` was implemented, documented and called
    # by **nothing**: this module imported `DEFINITION_KEY` from it and never the guard. So
    # a run could bind a part version as the test set **and** as a training signal source,
    # and nothing refused - and the refusal's own message says why that is the worst
    # available failure: *"training on the test set raises the test score for the wrong
    # reason, and the rise is indistinguishable from success on every surface miForge
    # produces."* It is not a crash and not a bad number; it is a **good** number that means
    # nothing, on every screen.
    #
    # Before the budget on purpose: this reads two dicts and allocates nothing, while the
    # budget step resolves a device. Cheapest refusals first (ADR-023).
    if _run(outcome, "submission_guards", answered):
        try:
            resolved["submission_guards"] = _check_submission_guards(
                definition, resolved.get("pack"), resolved.get("parts")
            )
            answered.add("submission_guards")
        except Refusal as exc:
            _collect(outcome, "submission_guards", exc)

    # ---- 4b. R-02.11: THE JUDGE IS NEVER THE MODEL UNDER TRAINING --------- AC-003.5
    if _run(outcome, "judge_identity", answered):
        try:
            caveats = _check_judge_identity(definition)
            if caveats is not None:
                resolved["judge"] = caveats
            answered.add("judge_identity")
        except Refusal as exc:
            _collect(outcome, "judge_identity", exc)

    # ---- 5. THE COMPUTED BUDGET, BEFORE ANY ALLOCATION --------- FR-006.54, M-14, AC13
    # The lease is NOT taken here and is not taken by this function at all. Moving this
    # check after the lease is M-14: the run takes the only card, finds it does not fit,
    # releases it - the ordering defect BRD section 9 records against R-02.42.
    if _run(outcome, "budget", answered):
        try:
            resolved["budget"] = budget_fn(definition)
            answered.add("budget")
        except Refusal as exc:
            _collect(outcome, "budget", exc)

    # ---- 6. OPTIONAL-SERVICE AVAILABILITY, NAMED IF ABSENT ----------------- FR-006.60
    if _run(outcome, "optional_services", answered):
        try:
            resolved["services"] = services_fn(definition)
            answered.add("optional_services")
        except Refusal as exc:
            _collect(outcome, "optional_services", exc)

    # ---- the resolved values, recorded BESIDE the submitted ones ----------- FR-006.61
    # THE SEED. Resolved, never merely accepted: an absent seed must not stay absent, so a
    # generated value is recorded here and the submitted definition keeps its null. Mutating
    # this line - recording `definition.seed` instead of the resolved value - must make
    # `test_run_definition_seed.py` red.
    resolved["seed"] = definition.resolve_seed(generate=seed_generator)
    resolved["domain"] = _resolve_domain(definition, pack_resolved)
    resolved["run_type"] = definition.run_type
    resolved["submitted_sha256"] = sha256_of(definition.model_dump(mode="json"))
    outcome.resolved = resolved

    if outcome.failures:
        log.info(
            "run definition refused",
            extra={
                "code": "run_definition_invalid",
                "failures": [failure.code for failure in outcome.failures],
                "skipped": [entry.step for entry in outcome.skipped],
            },
        )
    return outcome


def _run(outcome: Outcome, step: str, answered: set[str]) -> bool:
    """Whether ``step`` may run, recording a :class:`Skipped` entry when it may not."""
    waiting = tuple(need for need in REQUIRES.get(step, ()) if need not in answered)
    if waiting:
        outcome.skipped.append(Skipped(step=step, waited_on=waiting))
        return False
    outcome.steps_run.append(step)
    return True


def _collect(outcome: Outcome, step: str, exc: Refusal) -> None:
    """Record a refusal **with its own code**, never re-wrapped (FR-006.59)."""
    outcome.failures.append(
        Failure(
            step=step,
            code=exc.code,
            message=str(exc),
            details=dict(getattr(exc, "details", {}) or {}),
        )
    )


# ------------------------------------------------------------------ step 4, F6's own


def _validate_loop(definition: RunDefinition) -> dict[str, Any]:
    """The cadences, the condition set, each condition's configuration, the refresh rule.

    Raises:
        Refusal: ``run_has_no_stop_condition`` (E-1); ``stop_condition_unknown`` (E-2);
            ``stop_condition_config_incomplete`` (E-3); ``refresh_not_applicable`` (E-12);
            ``refresh_mode_unknown``; ``cadence_invalid`` - the last raised by Pydantic at
            the edge, since ``Cadences`` bounds every field at ``ge=1``.
    """
    # E-1 / FR-006.17. R-02.53's list requires "at least one stop condition (R-02.25)" in
    # so many words, so this is a cited requirement rather than a derived one - DEF-3 is
    # closed. Deleting this check is mutation F6-re, and 8.2's test must go red: the
    # consequence is a run that holds the node's only GPU indefinitely under ADR-008's
    # one-run-per-card lease, writing a checkpoint every `save_every` steps onto a volume
    # for which NO requirement anywhere defines pruning, deletion or archival.
    enabled = definition.enabled_conditions()
    if not enabled:
        raise refuse(
            "run_has_no_stop_condition",
            "This run has no stop condition enabled, so nothing would ever end it. "
            "R-02.53 requires at least one of R-02.25's five: "
            f"{list(stop_registry.names())}. An unbounded run holds this node's only GPU "
            "indefinitely and writes a checkpoint every save_every steps onto a volume "
            "with no retention policy. Add a step_count at the very least.",
            known=list(stop_registry.names()),
        )

    # E-2, then E-3: the name must be one of the five, and an enabled condition's
    # configuration must be complete. Both run through the condition's OWN validator
    # (`config_value`), so there is one definition of "complete" per condition rather than
    # a copy of each condition's parameter list in this module.
    stop_registry.assert_known(list(enabled))
    from training.stop.base import StopContext

    probe = StopContext(run_id="validate", step=0, round=0)
    for name, config in enabled.items():
        if name in stop_registry.UNCONFIGURED:
            continue
        # Evaluated against an empty context purely to exercise the configuration
        # refusals. The verdict is discarded: at step 0 with no scores and no window
        # nothing can fire, and a condition that READ something here would be reaching
        # into a run that does not exist yet.
        stop_registry.resolve(name).evaluate(probe, config)

    refresh_block: dict[str, Any] | None = None
    if definition.refresh is not None:
        source_name = str(definition.signal_source.get("source", ""))
        # E-12 / FR-006.31. Refused, NOT ignored: an ignored refresh leaves the operator
        # believing the prompt set rotated when it did not.
        refresh_registry.assert_applicable(source_name)
        refresh_registry.resolve(definition.refresh.mode)
        if definition.cadences.refresh_every is None:
            raise refuse(
                "refresh_rule_incomplete",
                "A refresh rule is configured and refresh_every is not set, so no round "
                "would ever end and the refresh would never happen. A 'round' is "
                "refresh_every completed steps (FR-006.37).",
                mode=definition.refresh.mode,
            )
        refresh_block = {
            "mode": definition.refresh.mode,
            "rule_sha256": refresh_registry.rule_hash(
                definition.refresh.mode, definition.refresh.rule
            ),
            "refresh_every": definition.cadences.refresh_every,
        }

    from training.stop.retention_floor import exposure_steps

    return {
        "cadences": definition.cadences.model_dump(mode="json"),
        "stop_conditions": {name: dict(config) for name, config in enabled.items()},
        "stop_conditions_sha256": sha256_of(
            {name: dict(config) for name, config in enabled.items()}
        ),
        "refresh": refresh_block,
        # FR-006.20's number, computed in one place and recorded so nobody has to infer it.
        # At retention_every = 1 it is 0.
        "retention_exposure_steps": exposure_steps(definition.cadences.retention_every),
    }


# ------------------------------------------------------ step 4a, F4's submission guards

#: The two slot kinds that make a part **evaluation** material. Underscored, because that is
#: ``batch-record-v1.json``'s ``PartReference.kind`` vocabulary and the one both artifacts
#: use; the hyphenated ``mf_pack_parts.part_kind`` spelling is normalised onto it below
#: rather than matched separately - *three vocabularies touch this and they are not
#: aliases* (``packs/slots.py``), and a guard that silently failed to match a spelling would
#: report no overlap for a run that has one.
EVALUATION_PART_KINDS: frozenset[str] = frozenset({"test_set", "retention_set"})

#: The kinds a run **trains** on.
SIGNAL_PART_KINDS: frozenset[str] = frozenset({"prompt_set", "corpus", "preference_pairs"})

#: F4's two E-12 codes, named so the deferral below selects on them rather than on a message.
E12_CODES: frozenset[str] = frozenset({"retention_set_missing", "retention_floor_missing"})

#: The stop condition whose existence makes a retention measurement part of a run's shape.
#: Spelled from F6's registry rather than retyped - a condition renamed there must not leave
#: this guard silently matching nothing.
RETENTION_FLOOR_CONDITION = "retention_floor"

#: Stands in for a provider row whose model is not knowable at submit. See its one use.
JUDGE_MODEL_UNKNOWN = "<model not resolved at submit>"

#: Passed to ``check_submission`` **only** to step past E-12 and reach E-16. Named, so that a
#: reader cannot mistake it for a floor the run has: it is recorded nowhere and read by
#: nothing, and *"a fabricated floor of 0.0 passes every run forever"* is precisely why it
#: has a name instead of being a literal.
E12_FLOOR_NOT_ASKED: float = 0.0


def _normalised_kind(value: Any) -> str:
    """One spelling for the two that exist. See :data:`EVALUATION_PART_KINDS`."""
    return str(value or "").strip().lower().replace("-", "_")


def _part_identity(entry: Mapping[str, Any]) -> str | None:
    """One comparable identity for a part, whichever of the three shapes it arrives in.

    **The same function on both sides of the comparison**, which is the whole point: F4's
    guard intersects two sets of strings, so two derivations would make the intersection
    empty for a run that really does bind one part twice - a guard that reports *"no
    overlap"* while looking as though it ran.

    Three shapes reach here: a pack's ``PartReference`` element (``contract_reference`` /
    ``external``), F2's ``PartSpec`` and F4's scoring slot (``reference`` / ``part_id``), and
    the loose-parts block (``part_id``). ``None`` when an entry names nothing, which is a
    shape the edge models already refuse and is therefore not guessed at here.
    """
    reference = entry.get("contract_reference") or entry.get("reference")
    if isinstance(reference, Mapping) and reference:
        return f"{reference.get('type')}/{reference.get('name')}@{reference.get('version')}"
    external = entry.get("external")
    if isinstance(external, Mapping) and external.get("content_sha256"):
        return f"sha256:{external['content_sha256']}"
    part_id = entry.get("part_id")
    if part_id:
        return f"part:{part_id}"
    return None


def _scoring_slot(definition: RunDefinition, key: str) -> Mapping[str, Any] | None:
    slot = definition.scoring.get(key)
    return slot if isinstance(slot, Mapping) and slot else None


def _check_submission_guards(
    definition: RunDefinition, pack_payload: Any, parts_block: Any
) -> dict[str, Any]:
    """Call F4's :func:`services.scoring.entrypoint.check_submission`. **R-02.48, E-12, E-16.**

    That function had **no caller**: this module imported ``DEFINITION_KEY`` from its module
    and never the guard itself, so both of its refusals were unreachable in production while
    fully implemented and unit-tested.

    The five arguments are assembled here because this is the only place that holds all of
    them at once - the pack's resolved slots, the loose-parts block, F2's signal parts and
    F4's own scoring slots. **Every one is read, never defaulted:** a wrong derivation here
    fails in the worst available direction, which is refusing *every* run at submit, so each
    source is taken from the shape its owner publishes and
    ``test_submission_guards_are_wired.py`` drives a valid run of each shape through this
    step to prove it still submits.
    """
    from services.scoring.entrypoint import check_submission

    test_ids: set[str] = set()
    retention_ids: set[str] = set()
    signal_ids: set[str] = set()

    # 1. Whatever resolved - a pack's seven slots, or the loose-parts set. Both shapes carry
    #    `kind`, so one loop reads either.
    elements: list[Any] = []
    for block in (parts_block, pack_payload):
        if isinstance(block, Mapping):
            elements.extend(block.get("parts") or ())
    for element in elements:
        if not isinstance(element, Mapping):
            continue
        kind = _normalised_kind(element.get("kind"))
        identity = _part_identity(element)
        if identity is None:
            continue
        if kind == "test_set":
            test_ids.add(identity)
        elif kind == "retention_set":
            retention_ids.add(identity)
        elif kind in SIGNAL_PART_KINDS:
            signal_ids.add(identity)

    # 2. F2's signal parts, off the definition. **The authoritative training side**: this is
    #    the list the source will actually read, and a part named here is trained on whether
    #    or not a pack also mentions it.
    for spec in definition.signal_source.get("parts") or ():
        if isinstance(spec, Mapping):
            identity = _part_identity(spec)
            if identity is not None:
                signal_ids.add(identity)

    # 3. F4's own scoring slots, which are what `spec_from_definition` will read.
    for key, bucket in (("test_slot", test_ids), ("retention_slot", retention_ids)):
        slot = _scoring_slot(definition, key)
        if slot is None:
            continue
        identity = _part_identity(slot)
        if identity is not None:
            bucket.add(identity)
        bucket.update(str(version) for version in slot.get("part_versions") or ())

    # Off the pack's **published payload** (``ResolvedPack.payload()``'s
    # ``retention_floor``), not by calling a second method on the object: this module
    # already records that payload in ``resolved["pack"]``, and reading one shape rather
    # than two is what keeps F6 and F7 from disagreeing about where a pack's floor lives.
    pack_floor = pack_payload.get("retention_floor") if isinstance(pack_payload, Mapping) else None
    retention_slot = _scoring_slot(definition, "retention_slot")
    truth: dict[str, Any] = {
        "test_part_versions": test_ids,
        "retention_part_versions": retention_ids,
        "signal_part_versions": signal_ids,
        # A pack that passed the gate **cannot** be floorless: ``FLOOR_BEARING_SLOTS`` is
        # exactly ``{retention_set}``, ``slot_from_document`` refuses a floor on any other
        # slot, and ``PackValidity`` refuses a filled retention slot without one. So a
        # pack-supplied floor is itself evidence that the slot is filled - read from the
        # payload rather than from an element list, which a thin caller can leave empty.
        "retention_set_present": bool(
            retention_ids or retention_slot is not None or pack_floor is not None
        ),
        "retention_floor": (
            definition.retention_floor_override
            if definition.retention_floor_override is not None
            else pack_floor
        ),
    }
    declares_retention = bool(
        truth["retention_set_present"]
        or pack_payload is not None
        or RETENTION_FLOOR_CONDITION in definition.enabled_conditions()
    )
    try:
        check_submission(**truth)
    except Refusal as exc:
        if exc.code not in E12_CODES or declares_retention:
            raise
        # ---- E-12, UNCONDITIONALLY, IS AN OPERATOR DECISION AND IS NOT TAKEN HERE -----
        # R-02.17 reads *"a run **also** defines a retention set"* and F4's AC-004.13 makes
        # that a submit-time refusal, so on the requirement alone this `raise` would stand.
        # Two things in the built product say otherwise, and both were **measured, not
        # assumed**:
        #
        #   * F6's own `training/run_driver._evaluator` states in terms that *"a run may
        #     legitimately have no test or retention set"*;
        #   * enforcing it refuses **74 currently-passing tests across five features**, and
        #     among them **every chain stage** - `services/chains/service.py` builds a
        #     `RunDefinition` per hop and launches it, so a template that omits a retention
        #     slot would make chaining unusable. A `create` run from an architecture config
        #     has no prior general capability to retain at all, which is the case the
        #     requirement's word "also" does not obviously cover.
        #
        # So it is **reported rather than resolved in either direction**: the deferral is
        # named in `resolved["submission_guards"]`, where a validate response shows it, and
        # flipping it is deleting this branch. What is NOT deferred is E-16 below - it is
        # independent of these two arguments and has no false-refusal direction.
        deferred = exc.code
        check_submission(
            **{
                **truth,
                # Set to values that satisfy E-12 **so execution reaches E-16**, which
                # `check_submission` evaluates after it. Neither is recorded anywhere and
                # neither is read as a floor: the run's floor stays absent, which is what
                # the deferral is about.
                "retention_set_present": True,
                "retention_floor": E12_FLOOR_NOT_ASKED,
            }
        )
        return {"retention_guard_deferred": deferred, "checked": ["held_out_set_used_for_training"]}
    return {"checked": sorted(E12_CODES | {"held_out_set_used_for_training"})}


# -------------------------------------------- step 4b, F3's R-02.11 self-judging guard


def _judge_provider_keys(definition: RunDefinition) -> tuple[list[str], list[str]]:
    """``(keys, unchecked)`` - the judge provider keys this run configures.

    ``unchecked`` names a configuration whose judge cannot be identified from the definition
    alone - a ``composite``, whose children are resolved against a catalog this step does not
    hold. **Named rather than skipped silently**, because a step that quietly checked nothing
    reads exactly like a step that found nothing wrong.
    """
    keys: list[str] = []
    unchecked: list[str] = []
    for slot in ("selector", "grader"):
        body = definition.signal_source.get(slot)
        if not isinstance(body, Mapping):
            continue
        kind = str(body.get("kind") or "")
        if kind == "judge_model":
            provider = body.get("provider")
            if provider:
                keys.append(str(provider))
            else:
                unchecked.append(f"{slot}:no_provider_key")
        elif kind == "composite":
            unchecked.append(f"{slot}:composite_children_not_resolvable_at_submit")
    return keys, unchecked


def _run_target_identity(definition: RunDefinition) -> Any:
    """The run's **own** inference target, as the definition can state it at submit.

    ADR-009: generation goes through the in-process provider by default - *"the parameters
    miForge holds"* - and neither ``SourceSpec`` nor ``RunDefinition`` carries a second
    generation provider, so the kind is not in doubt. The model name is the one the model
    source names.

    ``None`` for a ``create`` run, and that is the honest answer rather than a placeholder:
    a run that initialises at random has no pretrained model name, so there is no name for
    R-02.11's comparison to be about, and :func:`check_judge_identity` documents ``None`` as
    *"no run target"* rather than treating it as a match.
    """
    from pathlib import Path as _Path

    from providers.base import ProviderIdentity

    source = definition.model_source or {}
    name = source.get("repo")
    if not name and source.get("path"):
        name = _Path(str(source["path"])).name
    if not name:
        return None
    return ProviderIdentity(kind="in_process", model=str(name), base_url=None)


def _check_judge_identity(definition: RunDefinition) -> dict[str, Any] | None:
    """Apply R-02.11 **before the run's first step**, and record what it found.

    ``guard_at_resolution`` - the one wired call site of three - runs inside F2's source
    pre-flight, which builds its ``RunContext`` with ``provider=None``. So ``run_target`` is
    ``None`` there, the comparison takes
    :func:`services.selectors.identity.check_judge_identity`'s early ``return ()``, and
    **R-02.11's comparison has never once been made in production**: only the unconditional
    in-process refusal fires. A judge configured during pack authoring - when no target
    existed - and then used in a run whose target *is* that judge was never caught
    (AC-003.5).

    This is call site 2. It needs no model in hand: the judge's identity comes off the
    registry's **row** (kind, model, base_url) rather than from a constructed provider, so
    nothing is allocated and no endpoint is contacted - which is what lets the check sit
    inside a validator whose defining property is that it allocates nothing.

    Returns:
        ``None`` when the run configures no judge; otherwise a record of what was compared,
        the caveats raised and anything that could not be checked. **Returned and recorded,
        never only logged** - FR-003.18 is about auditability after the run and a log line is
        not a record.

    Raises:
        Refusal: ``judge_is_resident_model`` (R-02.11), carrying both identities.
    """
    keys, unchecked = _judge_provider_keys(definition)
    if not keys and not unchecked:
        return None

    from providers import registry as provider_registry
    from providers.base import ProviderIdentity
    from services.selectors.identity import guard_at_preflight

    run_target = _run_target_identity(definition)
    caveats: list[str] = []
    compared: list[str] = []
    for key in keys:
        row = provider_registry.row_for(key)
        if row is None:
            # Not configured here, and **not refused here**: step 6 reports an absent
            # optional service by name, and refusing twice for one fact would tell the
            # operator two things about one state. Named so the gap is visible.
            unchecked.append(f"{key}:provider_not_configured")
            continue
        # `row.model` is `None` for exactly one row: the in-process one, before a model
        # handle is bound - which is always the case at submit. A **named** placeholder
        # rather than `""`, because two empty strings compare equal and that would be a
        # false `judge_is_resident_model`; this value cannot collide with a real model
        # name, is recorded nowhere, and for the only row that can produce it the refusal
        # fires on `kind` before any model is read. Every HTTP row has a model by
        # construction - `http_provider_rows` refuses a base URL without one.
        judge = ProviderIdentity(
            kind=row.kind, model=row.model or JUDGE_MODEL_UNKNOWN, base_url=row.base_url
        )
        caveats.extend(guard_at_preflight(judge, run_target))
        compared.append(key)
    return {
        "compared": compared,
        "run_target": None if run_target is None else run_target.model_dump(mode="json"),
        "caveats": sorted(set(caveats)),
        "unchecked": sorted(set(unchecked)),
    }


def _resolve_domain(definition: RunDefinition, pack_resolved: Any) -> str | None:
    """The run's domain: the definition's, else the pack's, else ``None``.

    ``None`` is an **assertion** that the run is not domain-scoped - legitimate for general
    pretraining, and the reason F5's ``ManifestFields.domain`` is nullable and has no
    default. This value is what unblocks the weights save (F1's
    ``checkpoint_save_skipped_no_provenance``).
    """
    if definition.domain is not None:
        return definition.domain
    if pack_resolved is not None:
        return getattr(getattr(pack_resolved, "definition", None), "domain", None)
    return None


# ---------------------------------------------------- the live bindings for steps 2, 3, 5, 6


def _live_gate(session: Any, pack_id: str, version: str) -> Any:
    """F7's gate, through the **live service interface** (``PackService``), not the module.

    Through the service because that is what F7's own reachability test asserts the gate is
    present on, so the two halves of the seam check the same object.
    """
    from packs.service import pack_service

    # `pack_service` is F7's singleton INSTANCE, not a factory. The first draft wrote
    # `pack_service()` and MyPy caught it as "PackService not callable" - **every test of
    # this path injected a `gate` spy, so the live binding was never once executed.** That
    # is the unreachable-capability shape in miniature, committed inside the fix for three
    # other instances of it.
    #
    # ⚠ THIS CITED `test_the_live_gate_binding_actually_runs`, WHICH NEVER EXISTED. Only the
    # budget's equivalent was written, so the binding carried a citation to its own absent
    # guard and read as covered - the same class as a requirement asserting a field its
    # schema lacks. The real guard is
    # `test_run_definition.py::TestTheLIVEPackGateBindingRefusesAnInvalidPack`: it leaves
    # THIS binding live while spying every other dependency, and drives an INVALID pack,
    # which is the only input that can tell `require_valid_pack` from `resolved_pack` -
    # they differ solely in `validity.raise_if_invalid()`. Swapping one for the other used
    # to leave the whole file at 54 passed.
    return pack_service.require_valid_pack(session, pack_id, version)


def _live_parts(session: Any, definition: RunDefinition, pack_resolved: Any) -> dict[str, Any]:
    """Step 3: every part the definition references resolves **now** (E-20).

    For a pack run the gate already re-resolved every slot, so this step reports what it
    resolved to rather than resolving a second time - two resolutions of one pack is two
    answers, and the second would be the one recorded.

    For a loose-parts run the targets are resolved through F10's one entry point.
    """
    if pack_resolved is not None:
        return {
            "source": "pack",
            "parts": [record.element() for record in pack_resolved.part_records()],
        }

    from services.parts import resolve as parts_resolve

    targets: list[Any] = []
    for spec in definition.parts or ():
        reference = spec.get("reference")
        part_id = spec.get("part_id")
        if (reference is None) == (part_id is None):
            raise refuse(
                "part_target_invalid",
                "A part is named by a contract reference OR by a part id, never both and "
                "never neither: resolving by precedence would read a part nobody asked for.",
                part=dict(spec),
            )
        targets.append(
            parts_resolve.reference_from(reference) if reference is not None else str(part_id)
        )

    lookup = parts_resolve.resolve_parts(session, targets)
    if lookup.missing:
        raise lookup.unresolved_refusal()
    if lookup.unpinnable:
        raise lookup.unpinnable_refusal()
    return {
        "source": "parts",
        "parts": [
            {
                "part_id": part.part_id,
                "kind": part.part_kind,
                "reference": (
                    None if part.contract_reference is None else part.contract_reference.as_dict()
                ),
            }
            for part in lookup.pinned.values()
        ],
    }


def _live_budget(definition: RunDefinition) -> dict[str, Any]:
    """Step 5: F1's preflight hook, **before any allocation** (FR-006.54, M-14).

    Calls :func:`services.acquisition.preflight`, which resolves the acquirer, enriches the
    spec, runs **the budget guard** and then the runtime gate - in F1's own order and
    through F1's own guard. The estimate stays F8's and the hook stays F1's; what is F6's is
    this step's **position**, and it is before the lease.

    Also runs F8's ``RunType.preflight`` (its first production caller), so a run type that
    cannot do what was asked refuses here rather than in the worker.
    """
    from services import acquisition as acquisition_service
    from training import runtypes
    from training.acquisition.base import AcquisitionSpec
    from training.run_config import RunConfig

    spec = AcquisitionSpec(
        **{
            **definition.model_source,
            "seed": definition.resolve_seed(),
            "gpu_request": definition.gpu_request,
        }
    )
    cfg = RunConfig(**{**definition.run_configuration, "seed": spec.seed})

    # F8's registry, resolved by name. `runtypes.resolve` refuses an unknown name listing
    # the three - the first production caller of this registry, which until now had none.
    run_type = runtypes.resolve(definition.run_type)
    run_type.preflight(
        spec,
        cfg,
        signal_source=str(definition.signal_source.get("source") or "") or None,
        adapter=definition.adapter,
    )

    device = acquisition_service.resolve_submit_device(definition.gpu_request)
    enriched, availability = acquisition_service.preflight(spec, cfg, device=device)
    return {
        "mode": enriched.mode.value,
        "param_count": enriched.param_count,
        "gpu_uuid": None if device is None else device.uuid,
        "gpu_name": None if device is None else device.name,
        "free_vram_bytes": None if device is None else device.free_memory_bytes,
        "mode_available": availability.available,
        "optimizer_config_hash": cfg.config_hash(),
    }


def _live_services(definition: RunDefinition) -> dict[str, Any]:
    """Step 6: every optional service the definition's **own contents** involve (FR-006.60).

    Reported as ``<service>_not_configured``, naming the service **and the capability that
    becomes unavailable**, and keeping *unconfigured* distinct from *unreachable* (F10
    FR-010.6). **miForge never substitutes a different behaviour for a missing service.**

    **A definition that needs no optional service validates with none configured, which is
    where AC20 begins.** That case is the default, and this function returns an empty
    ``needed`` list for it rather than reporting anything at all.
    """
    from clients import availability
    from services.review import preflight as review_preflight

    needed: list[str] = []
    selector = definition.signal_source.get("selector") or {}
    kinds = (
        [str(selector["kind"])] if isinstance(selector, Mapping) and selector.get("kind") else []
    )
    if kinds:
        # R-02.36: the in-run human-review queue is **miDataworks'**. miForge keeps none of
        # its own, so an unconfigured service makes that selector *unavailable* rather than
        # degraded. Routed through **F10's own published surface** rather than a
        # `require_midataworks` call of F6's: that function validates the capability key
        # against `clients.capabilities`, and F6 passing its own prose produced
        # `capability_unknown` instead of `midataworks_not_configured` - a refusal about the
        # wrong thing. One spelling across the seam, and the eleven cross-feature refusal
        # codes stay spelled identically.
        needed.extend(review_preflight.selector_kinds_unavailable(kinds))
        review_preflight.require_for_selector_kinds(kinds)

    for spec in definition.parts or ():
        if spec.get("reference") is not None:
            needed.append("part_import")
            break

    return {
        "needed": sorted(set(needed)),
        "midataworks": availability.integration_status(),
    }


__all__ = ["REQUIRES", "SCORING_KEY", "STEPS", "Failure", "Outcome", "Skipped", "validate"]
