"""The one registry of what miDataworks is needed *for*  (F10 task 2.2; FR-010.4).

**Two readers, one list.** ``GET /api/health`` and ``GET /api/v1/midataworks/status`` both
read :data:`CAPABILITIES`. Writing the list twice is the failure the vendored-schema
discipline exists to prevent, applied to a smaller thing: two lists drift, and *health
reporting a capability the guard does not gate is worse than no report at all*, because an
operator who reads "available" and then gets a refusal stops trusting the whole response.

**Deliberately not a framework.** This is a plain ``dict`` keyed by capability, not a
registry of "available siblings" with a plugin interface. One sibling is not a pattern;
miDataworks keeps its own ``require_mistudio`` concrete and so does this. When a second
optional sibling arrives the shape is **copied**, not generalised (ADR-010, and the F10
FTID's §11 rule).

**Why the keys are checked.** :func:`describe` refuses an unknown key rather than returning
a placeholder. A typo'd capability string passed to ``require_midataworks`` would otherwise
make health report on a capability nothing gates - a fabricated value in the sense ADR-018
means it, and silent.
"""

from __future__ import annotations

from core.errors import refuse

#: Capability key -> what it is, in the operator's words.
#:
#: These three are **all** of F10's capabilities that need the sibling. Everything else in
#: this feature - taking in a part from a directory, validating a manifest, reading the
#: part index - works with ``MIDATAWORKS_BASE_URL`` unset, because a
#: ``midataworks.dataset-version/v1`` document is self-describing. That is the whole point
#: of a vendored contract and the reason "optional" is the feature rather than a caveat
#: (R-02.50; BRD-02 AC20).
CAPABILITIES: dict[str, str] = {
    "part_import_http": "Importing a part over miDataworks' HTTP export routes",
    "available_parts_listing": "Listing the versions miDataworks offers for a set kind",
    "human_review_selector": "The in-run human-review selector",
}


def describe(capability: str) -> str:
    """What ``capability`` is for.

    Args:
        capability: A key of :data:`CAPABILITIES`.

    Returns:
        The operator-facing description.

    Raises:
        Refusal: ``capability_unknown`` when the key is not registered. Refused rather
            than defaulted: the whole value of this registry is that health and the guard
            cannot disagree, and a lenient lookup would let them.
    """
    try:
        return CAPABILITIES[capability]
    except KeyError:
        raise refuse(
            "capability_unknown",
            f"{capability!r} is not a registered miDataworks capability. Add it to "
            f"clients/capabilities.py so health and the availability guard agree about "
            f"it. Known: {sorted(CAPABILITIES)}.",
            capability=capability,
            known=sorted(CAPABILITIES),
        ) from None


def all_keys() -> list[str]:
    """Every capability key, sorted. The list health reports as unavailable when unset."""
    return sorted(CAPABILITIES)
