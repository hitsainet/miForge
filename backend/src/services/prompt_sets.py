"""Resolving a refreshed prompt set  (R-02.26 as amended 2026-10-06; FR-006.33, FR-006.60).

## Why this is in ``services/`` and not in ``training/refresh/``

``test_standalone_without_siblings.py::test_F1_has_no_outbound_sibling_call_site`` forbids
**any** import of ``clients`` from ``backend/src/training/``, by a static AST scan that sees
function-local imports too. That guard is acceptance criterion 20's - the criterion that
pins the whole 2026-10-13 standalone reframe - and it must stay able to fail: if it cannot,
none of eleven features' AC20 claims means anything.

The first draft of :mod:`training.refresh.new_prompt_set` imported ``clients.availability``
directly and the guard caught it on a clean full-suite run. The fix is not to relax the
guard or to hide the import inside a function; it is to put the work where its neighbours
already are. **ADR-017 puts the loop and the run types in ``training/`` and one module per
optional sibling in ``clients/``, and the layer between them is where every other
optional-sibling decision in this codebase already lives** -
:mod:`services.review.preflight` does exactly this for F4's ``human_review`` selector, and
:mod:`services.parts.acquire` for F10's HTTP part import. Resolving a part against F10's
index and reporting an absent service is service-layer work; deciding *which* set a round
should load is the refresh mode's. So ``training/refresh/`` imports this module and has no
sibling-reachable import of its own, which is also the shape FR-006.35's ``providers/``
guard wants.

## Unconfigured and unreachable are different facts  (F10 FR-010.6, ADR-010)

* **unconfigured** - no ``MIDATAWORKS_BASE_URL``. A *decision*, refused by name
  (``midataworks_not_configured``) with the capability that becomes unavailable listed. An
  unset base URL means **absent-and-normal** (ADR-010); it does not mean broken.
* **not imported** - the service is configured and the version simply is not in this
  installation's index yet: ``dataset_version_unresolved``, naming the reference.

*Configure the service* and *import the version* are different jobs, and collapsing them
sends the operator to do the wrong one. **Neither is ever answered by substituting a
different prompt set** - no stale set, no empty set, no local file (R-02.50, FR-006.60).
A silent fallback here would train on the previous round's prompts while the run log
recorded a fresh round.
"""

from __future__ import annotations

import logging
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from clients import availability
from core.errors import refuse
from services.parts import resolve as parts_resolve

log = logging.getLogger(__name__)

#: The slot a refresh replaces. A refresh replaces the **prompt set** and nothing else:
#: R-02.26 is about the domain prompt set, and rotating a test or retention set mid-run
#: would change what a score means without saying so.
PROMPT_SET_KIND = "prompt_set"

#: What an absent miDataworks costs, as a key of **F10's** closed registry
#: (:data:`clients.capabilities.CAPABILITIES`). That registry refuses an unregistered key
#: by name - *"a lenient lookup would let health and the guard disagree"* - which is right,
#: and it means F6 cannot coin one.
#:
#: **``part_import_http`` is the accurate key, not a fallback.** A refresh that names a
#: prompt-set version this installation does not hold needs that version *imported*, and
#: importing it over miDataworks' export routes is exactly what the key describes. With the
#: service unset, that import is what becomes unavailable.
#:
#: *Reported to the orchestrator:* the registry has no key naming R-02.26's refresh
#: specifically, so the refusal says "importing a part" where an operator asked for a
#: refreshed round. The message says what to do next either way, and adding a key is an
#: edit to F10's module, which F6 does not make.
CAPABILITY = "part_import_http"


@dataclass(frozen=True, slots=True)
class ResolvedSet:
    """A resolved prompt set: its identity, its size, and how to ask for it again.

    ``reference`` is **either** a miDataworks contract reference **or** an external
    provenance, and the two are never interchanged: miForge *"does not silently promote an
    external part to a contract reference, and does not claim miDataworks provenance for
    anything miDataworks did not publish"* (R-02.40).
    """

    part_id: str
    #: ``{type, name, version}`` for a pinned contract part, or
    #: ``{external_provenance, part_id, source}`` for an unpinnable one.
    reference: Mapping[str, Any]
    #: Whether the reference is a miDataworks contract reference.
    pinned: bool
    size: int
    #: The ``PartSpec`` the source should read after the refresh.
    spec: Mapping[str, Any]


def resolve_refreshed_set(session: Any, target: Any) -> ResolvedSet:
    """Resolve one prompt-set target through F10's single entry point, or refuse by name.

    Args:
        session: an open session.
        target: a :class:`services.parts.contract_validator.ContractReference` or a part id
            string - the two forms F10's index is keyed by.

    Raises:
        Refusal: ``midataworks_not_configured`` when nothing matches **and** the service is
            unset; ``dataset_version_unresolved`` when it is set and the version is simply
            not imported; ``refresh_rule_incomplete`` when the part is not a prompt set.
    """
    lookup = parts_resolve.resolve_parts(session, [target])
    key = parts_resolve.target_key(target)

    if key in lookup.pinned:
        part = lookup.pinned[key]
        assert part.contract_reference is not None  # `pinned` means it has one
        reference: Mapping[str, Any] = part.contract_reference.as_dict()
        spec: Mapping[str, Any] = {"kind": PROMPT_SET_KIND, "reference": dict(reference)}
        pinned = True
    elif key in lookup.unpinnable:
        part = lookup.unpinnable[key]
        # NOT PROMOTED (R-02.40). The part carries no contract reference and miForge does
        # not invent one: what is recorded is its own provenance, and the spec names it by
        # the id of the row that carries that provenance.
        reference = {
            "external_provenance": dict(part.payload.get("provenance") or {}),
            "part_id": part.part_id,
            "source": part.source,
        }
        spec = {"kind": PROMPT_SET_KIND, "part_id": part.part_id}
        pinned = False
    else:
        raise _absent(target, lookup)

    if part.part_kind != PROMPT_SET_KIND:
        raise refuse(
            "refresh_rule_incomplete",
            f"The refresh names a part of kind {part.part_kind!r}; a prompt refresh "
            f"replaces the run's {PROMPT_SET_KIND!r} and nothing else. Rotating a test or "
            f"retention set mid-run would change what a score means without saying so.",
            part_kind=part.part_kind,
            expected=PROMPT_SET_KIND,
        )

    return ResolvedSet(
        part_id=part.part_id,
        reference=reference,
        pinned=pinned,
        size=declared_size(part.payload),
        spec=spec,
    )


def _absent(target: Any, lookup: Any) -> Exception:
    """``midataworks_not_configured`` or ``dataset_version_unresolved`` - by name.

    **It does not probe.** Reachability is deliberately not checked: a sibling that is down
    this second may be up by the next round, and refusing for a transient network state
    would be worse than the wait. *Unconfigured* is a decision and is refused here;
    *unreachable* is a fault and is reported where it happens, with a different code
    (F10 FR-010.6).
    """
    if not availability.is_configured():
        log.warning(
            "a prompt refresh needs a part this installation does not hold, and "
            "miDataworks is not configured",
            extra={
                "code": "midataworks_not_configured",
                "capability": CAPABILITY,
                "target": parts_resolve.target_key(target),
            },
        )
        return availability.not_configured_refusal(
            "Loading a refreshed prompt set for the next round",
            CAPABILITY,
        )
    refusal: Exception = lookup.unresolved_refusal()
    return refusal


def declared_size(payload: Mapping[str, Any]) -> int:
    """How many rows the set declares, from **F10's own part payload**.

    Read from the part's declared splits rather than by opening the files: the count is
    provenance F10 already verified against the bytes at import, and re-counting here would
    be a second answer to how big a set is.
    """
    total = 0
    for split in payload.get("splits") or ():
        rows = split.get("rows")
        if isinstance(rows, int):
            total += rows
    return total


def target_from_rule(rule: Mapping[str, Any]) -> Any:
    """``{reference: {...}}`` or ``{part_id: "..."}`` - **exactly one** (FR-006.33).

    Refused rather than resolved by precedence: picking one would load a prompt set the
    operator did not ask for and report success, which is ADR-018's fabricated value at the
    scale of a whole round of training.

    Raises:
        Refusal: ``refresh_rule_incomplete``.
    """
    reference = rule.get("reference")
    part_id = rule.get("part_id")
    if (reference is None) == (part_id is None):
        raise refuse(
            "refresh_rule_incomplete",
            "A new-prompt-set refresh names the set EITHER by miDataworks contract "
            "reference ({type, name, version}) OR by part id - never both and never "
            "neither. Resolving by precedence would load a prompt set nobody asked for.",
            reference=reference,
            part_id=part_id,
        )
    if reference is not None:
        return parts_resolve.reference_from(reference)
    return str(part_id)


__all__ = [
    "CAPABILITY",
    "PROMPT_SET_KIND",
    "ResolvedSet",
    "declared_size",
    "resolve_refreshed_set",
    "target_from_rule",
]
