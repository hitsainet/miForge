"""R-02.53's run definition - **the envelope, and nothing else**  (FR-006.56).

R-02.53 had to be *added* to BRD-02 on 2026-10-13: **nothing in the document defined
creating or starting a run.** R-02.24 defined the loop's body, R-02.38 said a run "cannot
start" from an invalid pack, R-02.42 "refuses to start" an oversized one, R-02.22 said to
"start miForge with" a ``run_id`` - every one presupposed a run coming into existence and
none defined it. That is why the New-run form, the product's primary action, had three
features each naming another as its owner: there was nothing to point at.

## This module defines **no field of another feature's block**

FR-006.56's table names, per part, the feature whose values fill it. So each part here is
either a thin reference object (a pack identity, a cadence - F6's own), or an opaque
``Mapping`` handed straight to the owning feature's constructor:

===========================  =======================================================
Part                         Validated by
===========================  =======================================================
``model_source``             :class:`training.acquisition.base.AcquisitionSpec` (F1)
``run_configuration``        :class:`training.run_config.RunConfig` (F1, R-02.13)
``signal_source``            :class:`training.signals.base.SourceSpec` (F2)
``run_type``                 :func:`training.runtypes.resolve` (F8)
``pack`` / ``parts``         F7's gate and F10's resolver
``scoring``                  F4's ``spec_from_definition`` (its ``DEFINITION_KEY``)
``stop_conditions``          :mod:`training.stop` (**F6's own**)
cadences, ``refresh``        **F6's own**
===========================  =======================================================

Writing F1's learning rate or F2's batch size again here would be a second declaration of
each, and the one that drifted would be the one the form posts against.

## Nothing is defaulted into existence  (E-24, ADR-018)

No default model source, no default floor, no default run type, no default stop-condition
parameter. A missing required part is a **refusal naming the part**, not a silently filled
blank: *"a fabricated model source, floor or cadence is an invisible wrong answer."*

The one line this feature draws, stated once because it recurs: a **cadence** may have a
declared default, because it changes only *when* a recorded action happens and the record
says when it happened. Anything that decides *whether a run ends* - a plateau's window, a
threshold, a floor - has **no default and is refused if absent** (FR-006.5, FR-006.21).

## The seed is resolved, never merely accepted  (operator decision, 2026-10-14)

R-02.47 requires the content hash of the weights a run started from, for **both** modes, and
no requirement anywhere recorded a seed - so a ``create`` run's random initialisation had a
hash that identified *which* init happened without letting anyone repeat it.

``seed`` is therefore part of the definition, and **if the operator supplies none, one is
generated and the generated value is recorded** (:meth:`RunDefinition.resolve_seed`). An
absent seed must not stay absent: a run whose seed is null is a run that cannot be repeated,
and here ``null`` would be an omission rather than an assertion. The submitted definition
keeps the ``null``; the resolved definition carries the number, which is precisely the
difference ``definition_submitted`` and ``definition_resolved`` exist to show.

**What the seed does and does not cover - stated because an overclaimed seed is worse than
none.** It covers a ``create`` run's random initialisation (``torch.manual_seed`` in
:mod:`training.acquisition.scratch`, folded into ``base_model`` and the starting hash) and
it appears in :attr:`training.run_config.RunConfig.seed` so the optimizer config hash
distinguishes two runs that differ only by seed. It does **not** cover batch/data order (no
source in this release shuffles, so order is the part's own and already reproducible, but
nothing *enforces* that), generation sampling (``GenerationParams.seed`` is its own field,
per source and per scoring pass), the scoring subsample (``ScoreRecord`` still owes an
``subsample_seed`` field - BRD-02 §9) or cuDNN/kernel nondeterminism. Same seed, same
definition hash, and a ``create`` run's starting weights repeat; a self-generation run's
completions do not unless its own decode seed is set too.
"""

from __future__ import annotations

import secrets
from collections.abc import Mapping
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, model_validator

#: Cadence fields and their declared defaults. ``None`` means *never* for the two optional
#: ones, which is a statement: a run with no save cadence still saves at run end
#: (FR-006.8), and a run with no refresh rule has no rounds past the first.
DEFAULT_TEST_EVERY = 1
DEFAULT_RETENTION_EVERY = 1

#: The widest value a generated seed may take. ``2**31 - 1`` rather than ``2**64``, because
#: ``torch.manual_seed`` accepts a 64-bit value but numpy's legacy seeding and several
#: tokenizer paths do not, and a seed that some library silently truncates is a seed that
#: does not reproduce the run it claims to.
MAX_SEED = 2**31 - 1


class PackReference(BaseModel):
    """A pack identity and version, as F7's gate takes them (R-02.38)."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    pack_id: str = Field(min_length=1)
    version: str = Field(min_length=1)


class Cadences(BaseModel):
    """The four cadences, validated at submit (FR-006.5, E-4).

    Each is a **positive integer or null**; zero, negative and non-integer are refused by
    name. Zero would make ``step % n`` a division by zero on the hottest path in the
    product, and a negative cadence names an action that never happens while reading as
    though it does.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    #: F4's (R-02.16). Defaulted, because it changes only *when* a recorded evaluation
    #: happens and the record says when it happened.
    test_every: int = Field(default=DEFAULT_TEST_EVERY, ge=1)
    #: F4's (R-02.17). **Defaults to 1 whenever a floor is set** (F4 FR-004.26), which is
    #: also what acceptance criterion 3 asks for - and at 1 the exposure window
    #: FR-006.20 names is **0** steps.
    retention_every: int = Field(default=DEFAULT_RETENTION_EVERY, ge=1)
    #: F5's writer, F6's cadence (R-02.20, R-02.24). ``None`` writes no cadenced
    #: checkpoint; the run-end save still happens.
    save_every: int | None = Field(default=None, ge=1)
    #: **F6's own** (R-02.26). A "round" is this many completed steps (FR-006.37, AMB-4).
    #: ``None`` means the run has one round and never refreshes.
    refresh_every: int | None = Field(default=None, ge=1)


class RefreshRule(BaseModel):
    """R-02.26's refresh rule: a mode and its body, recorded and hashed (FR-006.36)."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    mode: str = Field(min_length=1)
    #: The mode's own body. Opaque here and validated by the mode, so a third mode cannot
    #: be smuggled in by a field name.
    rule: dict[str, Any] = Field(default_factory=dict)


class StopConditionSpec(BaseModel):
    """One enabled stop condition: its name and its configuration (FR-006.15, .21).

    ``config`` has **no defaults and no shape here**: each condition validates its own
    parameters through :func:`training.stop.base.config_value`, which refuses a missing one
    by name. Declaring a ``window`` or a ``threshold`` field with a default in this module
    would be exactly the fabricated value ADR-018 forbids, in the place where it is
    invisible - a fabricated cadence shows up in the record, a fabricated stop parameter
    does not.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    condition: str = Field(min_length=1)
    config: dict[str, Any] = Field(default_factory=dict)


class RunDefinition(BaseModel):
    """R-02.53's run definition, and the submit payload (FR-006.56).

    Closed to unknown keys: a misspelled ``retention_every`` must not silently leave the
    real field at its default, which would train a different run than was described - and
    on this object it would change *when a run stops*.
    """

    model_config = ConfigDict(extra="forbid")

    #: Supplied by the caller, or minted by the service. Accepted because R-02.22 continues
    #: a run by *starting miForge with the same ``run_id``*, so the identifier cannot always
    #: be the server's to choose.
    run_id: str | None = Field(default=None, min_length=1, max_length=64)

    #: F1's (R-02.46). Handed to ``AcquisitionSpec``, which owns "exactly one source".
    model_source: dict[str, Any]
    #: F8's (R-02.41, R-02.43). One of ``change | create | adapter`` - resolved through
    #: F8's registry, never matched against a literal here.
    run_type: str = Field(min_length=1)
    #: F1's R-02.13 "run configuration". Handed to ``RunConfig``.
    run_configuration: dict[str, Any]
    #: F2's (R-02.4-R-02.7). Handed to ``SourceSpec``.
    signal_source: dict[str, Any]

    #: F7's (R-02.38), or loose parts. **Exactly one of the two**, refused rather than
    #: resolved by precedence.
    pack: PackReference | None = None
    parts: list[dict[str, Any]] | None = None

    #: F4's block, under the key F4 reads it from. Handed to ``spec_from_definition``.
    scoring: dict[str, Any] = Field(default_factory=dict)
    #: F4's (R-02.17), when the run overrides the pack's slot floor. ``None`` means *use
    #: the pack's*, and F4's ``resolve_floor`` records which layer supplied it.
    retention_floor_override: float | None = Field(default=None, ge=0.0, le=1.0)

    cadences: Cadences = Field(default_factory=Cadences)
    #: **F6's own** (R-02.25). At least one, enforced by the validator - R-02.53's list
    #: says "at least one stop condition (R-02.25)" in so many words.
    stop_conditions: list[StopConditionSpec] = Field(default_factory=list)
    refresh: RefreshRule | None = None

    #: F8's adapter block, for an ``adapter`` run. Opaque; F8's run type validates it.
    adapter: dict[str, Any] | None = None

    #: The run's domain, or ``None`` asserting it is not domain-scoped. **F6's**, and the
    #: value F5's ``ManifestFields.domain`` has no default for: *null is an assertion and an
    #: absent key is an omission*. Resolution prefers the pack's own domain.
    domain: str | None = None

    #: The reproducibility seed (operator decision, 2026-10-14). ``None`` here is *"I did
    #: not choose one"*, and resolution **generates one** - see the module docstring.
    seed: int | None = Field(default=None, ge=0, le=MAX_SEED)

    #: ``auto`` or a ``GPU-...`` UUID. Refused by name inside ``core.gpu``; no device logic
    #: lives here.
    gpu_request: str = "auto"

    @model_validator(mode="after")
    def _one_part_source(self) -> RunDefinition:
        """A pack **or** loose parts, never both and never neither.

        Refused rather than merged: a run whose material came half from a pack version and
        half from a loose list is a run whose inputs no pack version names, so *"a pack
        version names a reproducible run input"* would stop being true without anything
        saying so.
        """
        if (self.pack is None) == (self.parts is None):
            raise ValueError(
                "a run draws its parts from a pack version OR from an explicit part list, "
                "never both and never neither: merging the two produces a run whose inputs "
                "no pack version names"
            )
        if self.parts is not None and not self.parts:
            raise ValueError(
                "'parts' is an empty list, which names no material at all. Omit it and name "
                "a pack, or list the parts the run should read"
            )
        return self

    def resolve_seed(self, *, generate: Any = None) -> int:
        """The seed in force, generating one when the operator supplied none.

        Args:
            generate: injected for tests; defaults to :func:`secrets.randbelow`-backed
                generation. Injected rather than monkeypatched so the "two runs with
                different seeds differ" test is deterministic without reaching into this
                module.

        Returns:
            An ``int``, **never ``None``**. That is the whole point: an absent seed must not
            stay absent, because a run whose seed is null is a run that cannot be repeated.
        """
        if self.seed is not None:
            return int(self.seed)
        return int((generate or _generated_seed)())

    def enabled_conditions(self) -> dict[str, Mapping[str, Any]]:
        """``condition name -> configuration``, as :func:`training.stop.evaluate_all` takes it.

        Raises:
            ValueError: on a duplicate condition name. Refused rather than
                last-one-wins: two configurations for one condition means which one decides
                the run's end depends on list order.
        """
        enabled: dict[str, Mapping[str, Any]] = {}
        for spec in self.stop_conditions:
            if spec.condition in enabled:
                raise ValueError(
                    f"stop condition {spec.condition!r} is configured twice; which one "
                    f"ends the run would then depend on list order"
                )
            enabled[spec.condition] = spec.config
        return enabled


def _generated_seed() -> int:
    """A seed miForge chose. ``secrets`` rather than ``random``, so a seeded test process
    cannot make every run in it share one 'generated' seed - which would make two runs look
    reproducible against each other for the wrong reason."""
    return secrets.randbelow(MAX_SEED + 1)


__all__ = [
    "DEFAULT_RETENTION_EVERY",
    "DEFAULT_TEST_EVERY",
    "MAX_SEED",
    "Cadences",
    "PackReference",
    "RefreshRule",
    "RunDefinition",
    "StopConditionSpec",
]
