"""Mode (a): a new prompt set is loaded  (R-02.26 as amended 2026-10-06; FR-006.33).

R-02.26's amendment is explicit: *"A refresh that loads a new prompt set loads a miDataworks
prompt-set version (BRD-03 R-03.48), and the run log records its contract reference."*

## This module imports no sibling client, and that is a guard rather than a style

``test_standalone_without_siblings.py::test_F1_has_no_outbound_sibling_call_site`` forbids
any import of ``clients`` from ``backend/src/training/``, including a function-local one -
it is acceptance criterion 20's guard, and the criterion that pins the whole standalone
reframe. The first draft of this module imported ``clients.availability`` directly and the
guard caught it on a clean full-suite run.

So the resolution and the absent-service refusal live in :mod:`services.prompt_sets`, beside
:mod:`services.review.preflight` (F4's ``human_review`` selector) and
:mod:`services.parts.acquire` (F10's HTTP import), which is where every other
optional-sibling decision in this codebase already sits. **What stays here is the only thing
that is this mode's: deciding which set the next round reads.** The refusal is inherited,
not re-written - one spelling of ``midataworks_not_configured`` across the seam.

## Three outcomes, kept apart, because F10 kept them apart for a reason

* **pinned** - indexed and carrying a ``{type, name, version}``. Recorded as the contract
  reference.
* **unpinnable** - indexed, with a ``null`` contract reference. That is *conformant*;
  miDataworks permits it and F10 recorded the null faithfully. The refresh records the
  part's **own provenance** and **does not promote it** (R-02.40, FR-006.33).
* **absent** - ``midataworks_not_configured`` when the service is unset,
  ``dataset_version_unresolved`` when it is set and the version is not imported. *Configure
  the service* and *import the version* are different jobs. **Neither is ever answered by
  carrying on with the old set**: that would train the next round on the previous round's
  prompts while the run log recorded a fresh one.
"""

from __future__ import annotations

from collections.abc import Mapping

from core.errors import refuse
from services import prompt_sets
from training.refresh.base import (
    RefreshContext,
    RefreshModeName,
    RefreshOutcome,
    rule_hash,
)

NAME = RefreshModeName.NEW_PROMPT_SET.value

#: Re-exported so a caller naming the slot does not reach past this module for the string.
PROMPT_SET_KIND = prompt_sets.PROMPT_SET_KIND


class NewPromptSet:
    """R-02.26's first mode."""

    name = NAME

    def refresh(self, ctx: RefreshContext, rule: Mapping[str, object]) -> RefreshOutcome:
        if ctx.session is None:
            raise refuse(
                "refresh_rule_incomplete",
                "Loading a new prompt set needs a session to resolve the part against "
                "F10's index. Refusing rather than reporting an empty refresh.",
                mode=NAME,
            )

        target = prompt_sets.target_from_rule(rule)
        resolved = prompt_sets.resolve_refreshed_set(ctx.session, target)

        return RefreshOutcome(
            mode=NAME,
            rule_sha256=rule_hash(NAME, rule),
            round=ctx.round,
            set_reference=resolved.reference,
            size=resolved.size,
            # No seeds in this mode, and the zeros are stated rather than left absent: an
            # absent count is indistinguishable from a count of nothing.
            seed_counts={"accept": 0, "reject": 0, "undecided": 0},
            parts=[resolved.spec],
        )


MODE = NewPromptSet()

__all__ = ["MODE", "NAME", "PROMPT_SET_KIND", "NewPromptSet"]
