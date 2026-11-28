"""``domain`` has THREE states on a checkpoint, and a FOURTH on an adapter  (task 4.2a).

**The distinction, and why a nullable column is the wrong shape.** BRD-02 made
``domain`` *nullable but still required* in ``checkpoint-manifest-v1.json``, and the
schema says why in its own description: *"null is a statement, an absent key is an
omission."* A general from-scratch pretraining run (R-02.41) biases toward nothing, and
asserting that is different from forgetting to say. A single ``VARCHAR NULL`` column
stores both as ``NULL`` and the two facts become one - at which point the BRD's fix is
unpinned and F9's mutation control over it asserts nothing.

So the index carries **two** columns: the value, and the *state* of the record. Four
states, enumerated by :class:`DomainState`:

====================  ============================================================
``recorded``          the key is present and carries a string
``asserted_null``     the key is present and is JSON ``null`` - *this run has no
                      domain*, said deliberately
``absent``            the key is **missing** from a checkpoint manifest. The schema
                      requires it, so this manifest is nonconforming: an omission,
                      not an assertion
``not_in_kind``       the artifact kind has no ``domain`` field at all. Adapters.
                      ``adapter-manifest-v1.json`` declares no ``domain`` property
                      and sets ``additionalProperties: false``, so one cannot
                      conform while carrying it
====================  ============================================================

**``not_in_kind`` is not "unknown" and must never be filled in from the base.** An
adapter's base is a checkpoint and that checkpoint does have a domain - which is exactly
the inference 4.2a forbids, because an adapter trained on a different pack than its base
would then be labelled with the base's domain and nothing would say so. The table
constraints below make the inference unrepresentable rather than merely discouraged: a
``mf_adapters`` row is refused by CHECK unless its state is ``not_in_kind``.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from enum import StrEnum
from typing import Any


class ArtifactKind(StrEnum):
    """Which manifest schema an artifact is described by."""

    CHECKPOINT = "checkpoint"
    ADAPTER = "adapter"


class DomainState(StrEnum):
    """The state of an artifact's ``domain`` record. See the module docstring."""

    RECORDED = "recorded"
    ASSERTED_NULL = "asserted_null"
    ABSENT = "absent"
    NOT_IN_KIND = "not_in_kind"


#: The states a checkpoint row may hold. ``not_in_kind`` is excluded: a checkpoint
#: manifest always has the field, conformingly or not.
CHECKPOINT_STATES = frozenset({DomainState.RECORDED, DomainState.ASSERTED_NULL, DomainState.ABSENT})
#: The only state an adapter row may hold.
ADAPTER_STATES = frozenset({DomainState.NOT_IN_KIND})


@dataclass(frozen=True, slots=True)
class DomainClassification:
    """A classified ``domain`` record: the value, the state, and anything odd about it."""

    value: str | None
    state: DomainState
    #: A named anomaly for the caller to log, or ``None``. Carried rather than raised so
    #: that reindex can record a nonconforming artifact *and* say so: refusing it would
    #: make a checkpoint that exists on disk invisible, which is a worse failure than an
    #: indexed row with an honest ``absent``.
    anomaly: str | None = None


def classify_domain(manifest: Mapping[str, Any], *, kind: ArtifactKind) -> DomainClassification:
    """Classify a manifest's ``domain`` record into one of :class:`DomainState`'s four.

    Args:
        manifest: The parsed manifest, **as loaded from disk**. It must not have been
            through a defaulting layer: a Pydantic model with ``domain: str | None = None``
            would have already collapsed ``absent`` into ``asserted_null`` before this
            function saw it, which is the whole defect this function exists to prevent.
        kind: Which manifest schema describes the artifact.

    Returns:
        The value, its state, and a named anomaly when the manifest is nonconforming.
    """
    if kind is ArtifactKind.ADAPTER:
        # An adapter manifest has no `domain` property and refuses unknown keys. A key
        # here is therefore nonconforming - reported, never adopted, and never inferred
        # from `base`.
        anomaly = "adapter_manifest_carries_domain" if "domain" in manifest else None
        return DomainClassification(value=None, state=DomainState.NOT_IN_KIND, anomaly=anomaly)

    # THE LOAD-BEARING LINE: key presence, not truthiness and not `.get()`. `.get()`
    # returns None for both a missing key and an explicit null, which is the collapse.
    if "domain" not in manifest:
        return DomainClassification(
            value=None,
            state=DomainState.ABSENT,
            anomaly="checkpoint_manifest_missing_required_domain",
        )

    raw = manifest["domain"]
    if raw is None:
        return DomainClassification(value=None, state=DomainState.ASSERTED_NULL)
    if not isinstance(raw, str):
        # The schema says string-or-null. Anything else is not a domain; refusing to
        # coerce it is ADR-018 - `str(raw)` here would fabricate a domain name.
        return DomainClassification(
            value=None,
            state=DomainState.ABSENT,
            anomaly=f"checkpoint_manifest_domain_not_a_string:{type(raw).__name__}",
        )
    return DomainClassification(value=raw, state=DomainState.RECORDED)


def render_domain(value: str | None, state: DomainState) -> str:
    """How a ``domain`` record reads to a human. Used by the API and so by F9.

    Exists so the four states have **one** rendering rather than one per screen - the
    drift that produced four spellings of the verdict value (foundation 3.1a). In
    particular ``not_in_kind`` renders as *not recorded in this artifact kind* and never
    as a blank, because a blank reads as "no domain", which is what ``asserted_null``
    means.
    """
    match state:
        case DomainState.RECORDED:
            if value is None:  # pragma: no cover - CHECK constraint forbids it
                raise ValueError("state 'recorded' with no value; the CHECK should have refused")
            return value
        case DomainState.ASSERTED_NULL:
            return "no domain"
        case DomainState.ABSENT:
            return "not recorded"
        case DomainState.NOT_IN_KIND:
            return "not recorded in this artifact kind"
