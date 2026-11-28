"""Pre-flight orchestration and the capability listing  (FR-002.1, FR-002.4).

Two questions, answered without a GPU and without a run:

* **what can this installation train with right now?** - :func:`listing`, which backs F9's
  New-run select so a source is never *shown and then refused on submit*;
* **can this particular specification proceed?** - :func:`preflight`, which F7's pack
  validation calls before a run exists and F6 calls at submit.

**Availability is computed per call from injected settings, never cached at import.** A
module global would freeze the answer, so an operator who configures a provider would be
told the source is still unavailable until something restarted. F3's ``GET /selectors``
does the same; this copies the arrangement rather than inventing a second one.

**An unavailable source is listed, named and explained - never omitted.** An option missing
from a response is indistinguishable from one that is broken (ADR-013).
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from core.errors import Refusal
from training.signals import ensure_registered, registry
from training.signals.base import RunContext, SourceSpec

#: The fallback code when no provider can generate and the registry names no reason of its
#: own. Spelled with ADR-010's ``_not_configured`` suffix, which the app's ``status_for``
#: maps to 503 by suffix - so this needs no edit there to be reported as "not configured"
#: rather than "your request was bad". **F4's own ``inference_provider_unavailable`` is
#: preferred wherever the registry supplies it**: one spelling per fact.
INFERENCE_PROVIDER_NOT_CONFIGURED = "inference_provider_not_configured"


def generation_available() -> tuple[bool, str | None]:
    """Whether any provider can generate **right now**, and why not when none can.

    Asked of **F4's registry** - the one resolution path ADR-009 gives this project - rather
    than of a second table of F2's own. Two registries would be two answers to *which model
    answered*, which is precisely what R-02.49 records provider identity to prevent, and F2
    builds no provider (FPRD §15 finding 6: consume it, never fork it).

    Computed from **live** rows on every call. In the API process the in-process row is
    listed and unavailable, because a web process holds no parameters to generate from -
    saying so by name beats offering a row that would refuse on first use.
    """
    from providers import registry as provider_registry

    rows = provider_registry.list_rows()
    for row in rows:
        if row.available and "generation" in row.capabilities:
            return True, None
    # The registry's own spelling, not a second one. ADR-010's `_not_configured` suffix is
    # for an absent **sibling**; an unbound in-process model is a local fact, and F4 already
    # names it.
    reason = next((row.unavailable_reason for row in rows if row.unavailable_reason), None)
    return False, reason or INFERENCE_PROVIDER_NOT_CONFIGURED


def listing() -> list[dict[str, Any]]:
    """Every source, with what it needs and whether that is satisfied, **by name**."""
    ensure_registered()
    available_provider, provider_reason = generation_available()
    rows: list[dict[str, Any]] = []
    for name in registry.names():
        capability = registry.capability(name)
        unavailable: str | None = None
        if capability.needs_generation and not available_provider:
            unavailable = provider_reason
        rows.append(
            {
                "name": name,
                "part_kind": capability.part_kind,
                "objective": capability.objective,
                "requires": {
                    "part_kind": capability.part_kind,
                    "selector": capability.needs_selector,
                    "grader": capability.needs_grader,
                    "generation_provider": capability.needs_generation,
                },
                "available": unavailable is None,
                # `None` exactly when available: a reason beside an available source would
                # be a second answer to the same question.
                "unavailable_reason": unavailable,
                "summary": capability.summary,
            }
        )
    return rows


def preflight(spec: SourceSpec, *, context: RunContext | None = None) -> dict[str, Any]:
    """Resolve a specification, or refuse. **Never partially succeeds** (FR-002.4).

    Returns a summary of what resolved - the parts, their splits, the column binding, the
    objective and the configuration hashes - which is what F9's form shows before submit and
    what F7's pack validation checks a pack's slots against.

    Raises:
        Refusal: every configuration refusal in this feature, each naming what to fix: an
            unresolvable part, a kind mismatch, an inadmissible split, no usable column
            semantic, an unconfigured masking policy or DPO reference, a group size below
            two, a source the gate contract cannot record.
    """
    ensure_registered()
    source = registry.get(spec.source)(context or RunContext())
    resolved = source.preflight(spec)
    try:
        return {
            "source": resolved.source,
            "objective": resolved.objective,
            "parts": [_part_summary(part) for part in resolved.parts],
            "selector": _config_summary(resolved.selector),
            "grader": _config_summary(getattr(resolved.grader, "_resolved", None)),
            "batch_size": spec.batch_size,
            "group_size": spec.group_size,
            "groups_per_batch": spec.groups_per_batch,
        }
    finally:
        source.close()


def _part_summary(part: Any) -> dict[str, Any]:
    """One resolved part, as the form and the pack validator read it.

    The **binding** is here because it is the thing an operator most needs to see before
    submitting: it says which declared semantic will fill which role, and a corpus trained
    from ``messages`` and one trained from ``text`` are not the same run.
    """
    return {
        "part_id": part.payload.get("id"),
        "kind": part.kind,
        "contract_reference": part.payload.get("contract_reference"),
        "external": part.payload.get("external_provenance") is not None,
        "splits": [split["name"] for split in part.splits],
        "rows": sum(int(split.get("rows") or 0) for split in part.splits),
        "binding": part.binding.as_record(),
        # Carried to the point of use, never only shown at import: a caveat an operator
        # does not see when starting a run is a caveat that did not exist (F10 FR-010.17).
        "caveats": list(part.payload.get("caveats") or []),
        "licence_classes": list(part.payload.get("licence_classes") or []),
    }


def _config_summary(resolved: Any) -> Mapping[str, Any] | None:
    """A resolved selector or grader configuration, by **hash and provenance**.

    The provenance is reported because ``local`` and ``contract`` are different claims and
    R-02.40 forbids miForge rendering one as the other.
    """
    if resolved is None:
        return None
    return {
        "kind": resolved.kind,
        "name": resolved.name,
        "config_sha256": resolved.config_sha256,
        "provenance": resolved.hash.provenance,
        "caveats": list(resolved.caveats),
    }


def refusal_payload(refusal: Refusal) -> dict[str, Any]:
    """A refusal as the pre-flight route returns it. One envelope, ADR-013's."""
    return refusal.envelope()
