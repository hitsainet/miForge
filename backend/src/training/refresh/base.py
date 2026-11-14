"""What a prompt refresh is  (R-02.26; FR-006.32-.37).

R-02.26, in full: *"For self-generation runs, the domain prompt set may be **refreshed**
between rounds: a new prompt set is loaded, or the selector's accepted completions become
seeds for new prompts through a templating rule. Refresh rules are configuration and are
recorded. The resident model does not write its own prompt set."*

Four things in that sentence are requirements and all four are here: **self-generation
only** (FR-006.31), **exactly two modes** (FR-006.32), **recorded** (FR-006.36), and **the
model does not write its own prompts** (FR-006.35).

## ``training/refresh/`` imports nothing from ``providers/``

That last clause, together with BRD §8's non-goal - *"the resident model selecting its own
training data, writing its own prompts, or grading itself"* - makes it a **prohibition, not
a preference**. A template is a mechanical transformation of an accepted completion's text;
asking a model to invent prompts from one is the forbidden thing with an extra step.

So the absence is asserted by a static test
(``backend/tests/unit/test_refresh_no_provider_import.py``) in the shape of ADR-001's
sibling-import guard, covering ``import providers``, ``from providers import``, the relative
forms and the absolute package path, **with a self-test proving the scanner sees each form**
- *a scanner that cannot see one form is a guard that is not there.* **Mutation M-6** adds
the call.

## A "round" is ``refresh_every`` completed steps, and F6 defines it

**"Round" is defined nowhere in BRD-02.** Verified: it appears in §1's prose and in
acceptance criterion 3's *"over three rounds"* and is never given a length. Reported as
**AMB-4**. F6 defines it as ``refresh_every`` completed steps, increments it at each
refresh, and carries the number on every boundary record - which makes criterion 3's "three
rounds" a checkable statement. The alternative reading - a round as a test-cadence interval
- is rejected because R-02.26 is the only requirement that uses the word normatively and it
is about refresh.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any, Protocol, runtime_checkable

from core.canonical import sha256_of
from schemas.selector import VerdictValue

#: The one signal source a refresh applies to (R-02.26's opening clause). The four source
#: names are ``checkpoint-manifest-v1.json``'s closed enum, and this is the one that
#: generates - cited from F2's registry rather than respelled would be better still, but the
#: manifest spells it with a hyphen and F2's registry key is the authority a test compares
#: against (``test_refresh_applicability.py``).
SELF_GENERATION = "self-generation"


class RefreshModeName(StrEnum):
    """R-02.26's two modes, by name. **There is no third.**"""

    NEW_PROMPT_SET = "new_prompt_set"
    TEMPLATE_SEEDS = "template_seeds"


MODE_NAMES: tuple[str, ...] = tuple(mode.value for mode in RefreshModeName)


@dataclass(frozen=True, slots=True)
class Seed:
    """One accepted completion offered as a seed for a new prompt.

    Only ever built from an **``accept``** verdict. There is no constructor path from an
    ``undecided`` candidate, and that is the point: R-02.9 carries the prohibition
    *"no consumer may construct a path from `undecided` to admitted"*, and seeding a prompt
    from a pending completion is that same path wearing a different hat - the unattended
    review queue becomes an unselected run.
    """

    prompt_id: str
    prompt: str
    completion: str
    #: The verdict that admitted it, carried so the filter is auditable rather than assumed.
    verdict: VerdictValue = VerdictValue.ACCEPT


@runtime_checkable
class AcceptedSeedSource(Protocol):
    """A signal source that can publish the completions its selector **accepted**.

    **This seam is declared here and is not implemented by any source in this release.**
    F2's ``SelfGenerationSource`` builds ``SelectionOutcome.admitted`` inside ``next()``
    and retains it nowhere, and the batch record carries identifiers and counts but **no
    text** (ADR-020, F2 FR-002.17) - so there is no existing path from which accepted
    completion text can be read.

    F6 therefore declares the interface, calls it, and **refuses by name** when the live
    source does not implement it (``refresh_seeds_unavailable``), rather than inventing a
    seed from a batch record's hashes or from a pending candidate. Reported to the
    orchestrator as owed by F2; mode (a) is a complete path meanwhile.
    """

    def accepted_seeds(self, limit: int) -> Sequence[Seed]:
        """Up to ``limit`` of this source's most recently accepted completions."""
        ...


@dataclass
class RefreshContext:
    """What a refresh may read, injected.

    ``source`` is the live :class:`training.signals.base.SignalSource`. It is here so mode
    (b) can ask it for seeds; **mode (a) never touches it**, and neither mode calls
    ``next()`` on it - the loop does not reach inside the step (FR-006.2, M-12).
    """

    run_id: str
    step: int
    #: The round **after** this refresh. FR-006.37: incremented at the refresh.
    round: int
    session: Any = None
    source: Any = None
    #: ``SourceSpec`` of the run, so mode (a) can state what it replaced.
    current_parts: Sequence[Mapping[str, Any]] = ()


@dataclass(frozen=True, slots=True)
class RefreshOutcome:
    """What one refresh produced, and the record of it  (FR-006.36).

    Every field R-02.26's *"refresh rules are configuration and are recorded"* implies, and
    nothing content-bearing: no prompt text, no completion text (ADR-020, FR-006.50's
    security half).
    """

    mode: str
    #: The rule's identity through ``core/canonical.py`` - **the one** canonical function
    #: (ADR-005, foundation 2.6). Not a second hasher.
    rule_sha256: str
    round: int
    #: The resulting prompt set's identity: a contract reference, or an external
    #: provenance (R-02.40). **Never promoted** from one to the other (FR-006.33).
    set_reference: Mapping[str, Any] | None
    #: How many prompts the refreshed set holds. ``0`` is a failure, not a refresh
    #: (FR-006.38, E-13) and is refused by the caller.
    size: int
    #: Seeds by verdict. ``undecided`` is counted and **never used** - the count exists so
    #: an operator can see how many were pending, which is the number that explains a
    #: small refresh.
    seed_counts: Mapping[str, int] = field(default_factory=dict)
    #: The part specs the source should read after this refresh. Handed to the loop, which
    #: rebuilds the source - the refresh constructs no source of its own.
    parts: Sequence[Mapping[str, Any]] = ()

    def payload(self) -> dict[str, Any]:
        """The ``refresh`` socket event's body and the structured log line's fields."""
        return {
            "mode": self.mode,
            "rule_sha256": self.rule_sha256,
            "round": self.round,
            "set": dict(self.set_reference) if self.set_reference is not None else None,
            "size": self.size,
            "seed_counts": dict(self.seed_counts),
        }


@runtime_checkable
class RefreshMode(Protocol):
    """One of R-02.26's two. Resolved from one registry, never by module path."""

    #: One plain sentence for ``GET /refresh-modes``. Not the docstring: docstrings carry
    #: requirement ids and history, which the operator's screen must not.
    SUMMARY: str
    name: str

    def refresh(self, ctx: RefreshContext, rule: Mapping[str, Any]) -> RefreshOutcome:
        """Produce the new prompt set's identity, or refuse by name."""
        ...


def rule_hash(mode: str, rule: Mapping[str, Any]) -> str:
    """The rule's canonical identity, mode included.

    The mode is part of the hash because the same body under the two modes is two different
    rules, and a run log recording one hash for both could not say which happened.
    """
    return sha256_of({"mode": mode, "rule": dict(rule)})


__all__ = [
    "MODE_NAMES",
    "SELF_GENERATION",
    "AcceptedSeedSource",
    "RefreshContext",
    "RefreshMode",
    "RefreshModeName",
    "RefreshOutcome",
    "Seed",
    "rule_hash",
]
