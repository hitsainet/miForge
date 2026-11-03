"""The check a run's validation calls before anything allocates  (FR-010.26; R-02.48).

**F10 supplies the check; F6 calls it.** R-02.48's closing sentence is the requirement: *"a
run that cannot proceed without one says so before it starts, not partway through."* A run
whose selector is human review and whose miDataworks is unconfigured must be refused **at
submission** - not at the first candidate, after a model has been loaded onto a card and a
lease taken.

The call site is **F6's FR-006.60**, step 6 of R-02.53's validate-before-anything-allocates
order (*"every optional service the definition needs is checked at validation and named if
absent"*), carried there as edge case E-22 and pinned by its AC-006.42 - including that
*unconfigured* is distinguished from *unreachable*.

This module exists as its own file for a findability reason worth stating. A whole-corpus
scan found that F10's documents name a moment another feature owns - "run submission" is
F6's - while **never naming F6 anywhere in its four documents**, though F6 names F10 twenty
two times. The substance was right on both sides; what was missing was that an implementer of
this feature could not find the call site from this feature's own documents. **A seam has to
be findable from the side that depends on it, not only from the side that provides it.**
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence

from clients import availability

#: The selector kinds that need miDataworks. ``human_review`` is the only one: a rule, a
#: reference match and a composite of those reach nothing, and a judge model needs a
#: **generation provider** (F4's), which is a different optional service with its own
#: refusal. Naming them here rather than inferring from a requirement enum keeps this
#: module honest about what *this* feature's absence costs.
SELECTOR_KINDS_NEEDING_MIDATAWORKS: frozenset[str] = frozenset({"human_review"})


def selector_kinds_unavailable(kinds: Iterable[str]) -> list[str]:
    """Which of ``kinds`` cannot run in this installation right now. Never raises.

    For a listing: an option that is unavailable is reported **by name** rather than omitted,
    because an option missing from a response is indistinguishable from one that is broken
    (ADR-013).
    """
    if availability.is_configured():
        return []
    return sorted(set(kinds) & SELECTOR_KINDS_NEEDING_MIDATAWORKS)


def require_for_selector_kinds(kinds: Sequence[str]) -> None:
    """Refuse a run definition whose selector needs miDataworks while it is unconfigured.

    Args:
        kinds: The selector kinds this run's definition names.

    Raises:
        Refusal: ``midataworks_not_configured``, naming the **selector** as the capability
            that needs it - so the message is about the thing the operator chose rather than
            about an integration they may not have known was involved. ``details`` carry the
            setting, the action and the capability key, and the message carries the
            standalone alternative.

    **It does not probe.** Reachability is deliberately not checked here: a sibling that is
    down at submission may be up by the first candidate, and refusing a run for a transient
    network state would be worse than the wait. *Unconfigured* is a decision and is refused;
    *unreachable* is a fault and is reported when it happens, with a different code (FR-010.6).
    """
    unavailable = selector_kinds_unavailable(kinds)
    if not unavailable:
        return
    raise availability.not_configured_refusal(
        f"Starting a run whose selector is {', '.join(unavailable)}",
        "human_review_selector",
    )
