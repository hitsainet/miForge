"""The in-process part lookup F7 binds pack slots against  (F10 S-1; FR-010.15, FR-010.19).

**This is the entry point F7's task 8.1 waits on, and it is deliberately not
``kind_map.resolve``.** That function maps a *target kind* to a miForge part kind; this one
answers *"which part is this reference, and is it usable as a pack slot?"* - a different
question with a different failure mode. Publishing one and calling it the other is how a
feature ends up blocked on something that already exists under a name it does not fit.

## One batched lookup

:func:`resolve_parts` takes **all** of a pack's targets and issues one query. A pack has six
slots; six round trips to answer one validation is six chances for the answer to change
half way through, and F7 validates a whole pack as a unit.

## Three outcomes, kept apart

================  ======================================================================
Outcome           What it means, and whose refusal it is
================  ======================================================================
**pinned**        The part is indexed **and** carries a ``{type, name, version}``. Usable
                  as a pack slot. This is the only outcome F7 may bind.
**unpinnable**    The part is indexed and its ``contract_reference`` is **null**. That is
                  **conformant** - ``Target.contract_reference`` is
                  ``anyOf [ContractReference, null]`` and is in ``required``, parsed from
                  the vendored file - so F10 took it in and recorded the null faithfully.
                  It names nothing reproducible, so **F7 refuses the binding** with
                  ``part_unpinnable``. Both halves are needed and neither is sufficient:
                  F10's keeps miForge from inventing contract identity, F7's keeps an
                  unpinnable part out of a run.
**missing**       Nothing in the index matches. ``dataset_version_unresolved``, naming the
                  reference.
================  ======================================================================

Collapsing *unpinnable* into *missing* would be the tempting simplification and it would
be wrong in the expensive direction: the part exists, its bytes are verified, and an
operator told it is "not found" will re-import it forever.

**Nothing here synthesises a reference.** Not from the version id, not from the dataset
name, not from the directory name. A ``null`` is surfaced as ``None``, and the only thing
built from it is a refusal.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

import sqlalchemy as sa
from sqlalchemy.orm import Session

from core.errors import Refusal, refuse
from services.parts.contract_validator import ContractReference
from services.parts.index import part_payload
from services.parts.tables import CONTRACT_PARTS, PART_SPLITS

#: A target to look up: either miDataworks' contract reference, or a part id miForge
#: issued. Both are accepted by the one entry point because a pack slot may be pinned
#: either way - a contract part by its reference, an external part by the id of the row
#: that carries its provenance - and two entry points would be two batches.
PartTarget = ContractReference | str


def reference_from(value: Mapping[str, Any]) -> ContractReference:
    """Build a :class:`ContractReference` from ``{type, name, version}``.

    Raises:
        Refusal: ``contract_reference_malformed`` when a field is missing or empty.
            **Refused, not defaulted**: a reference with two of three fields names nothing,
            and the schema's own ``ContractReference`` requires all three with
            ``minLength: 1``.
    """
    missing = [key for key in ("type", "name", "version") if not value.get(key)]
    if missing:
        raise refuse(
            "contract_reference_malformed",
            f"A miDataworks contract reference is {{type, name, version}} and this one is "
            f"missing {missing}. All three are required - two of three names nothing.",
            missing=missing,
            given=dict(value),
        )
    return ContractReference(
        type=str(value["type"]), name=str(value["name"]), version=str(value["version"])
    )


def target_key(target: PartTarget) -> str:
    """The stable key a result is filed under.

    ``<type>/<name>@<version>`` for a reference - the **contract's** rendering, inner layer
    only - and the part id itself for an id. F7 prefixes its own pack identity for display;
    this never does.
    """
    if isinstance(target, str):
        return target
    return target.render()


@dataclass(frozen=True, slots=True)
class ResolvedPart:
    """One part the index knows, with the question F7 actually asks answered.

    ``contract_reference`` is ``None`` for an unpinnable part and that ``None`` is the
    answer, not a missing value.
    """

    part_id: str
    part_kind: str
    source: str
    contract_reference: ContractReference | None
    payload: Mapping[str, Any]

    @property
    def pinned(self) -> bool:
        return self.contract_reference is not None


@dataclass(frozen=True, slots=True)
class PartLookup:
    """The result of one batched lookup. Three outcomes, three collections.

    Separate collections rather than one list with a status field, so a caller cannot
    iterate the whole thing and treat every element the same way - which is how
    *unpinnable* would silently become *pinned*.
    """

    pinned: Mapping[str, ResolvedPart] = field(default_factory=dict)
    unpinnable: Mapping[str, ResolvedPart] = field(default_factory=dict)
    missing: tuple[str, ...] = ()

    def all_usable(self) -> bool:
        """Whether every target resolved to a pinned part."""
        return not self.unpinnable and not self.missing

    def unresolved_refusal(self) -> Refusal:
        """``dataset_version_unresolved`` for the missing targets, naming each.

        Raises:
            ValueError: if nothing is missing - building a refusal for an empty set would
                be a refusal with no reason, which :class:`core.errors.Refusal` forbids
                anyway.
        """
        if not self.missing:
            raise ValueError("unresolved_refusal() called with nothing missing")
        return refuse(
            "dataset_version_unresolved",
            f"No part in this installation matches {list(self.missing)}. Import the "
            f"version first - from a miDataworks export directory, or over miDataworks' "
            f"export routes when it is configured.",
            unresolved=list(self.missing),
        )

    def unpinnable_refusal(self) -> Refusal:
        """``part_unpinnable``, built here so the **code spelling** is shared.

        **The refusal is F7's to raise**, at pack-slot binding, and the message says why in
        F7's terms. It is constructed here only so there is one spelling of the code across
        the seam: eleven cross-feature refusal codes are spelled identically across the
        feature documents, and that agreement is load-bearing.
        """
        if not self.unpinnable:
            raise ValueError("unpinnable_refusal() called with nothing unpinnable")
        names = sorted(self.unpinnable)
        return refuse(
            "part_unpinnable",
            f"Part(s) {names} carry no miDataworks contract reference, so they name "
            f"nothing reproducible and cannot pin a pack slot. The version is conformant - "
            f"miDataworks permits a null reference - and miForge recorded the null rather "
            f"than inventing one. Ask miDataworks to publish the version with its contract "
            f"reference, or pin an external part with its own provenance instead.",
            unpinnable=names,
            part_ids=[self.unpinnable[name].part_id for name in names],
        )


def resolve_parts(session: Session, targets: Sequence[PartTarget]) -> PartLookup:
    """Resolve every target in **one** query. The entry point F7 binds slots against.

    Args:
        session: An open session.
        targets: Contract references and/or miForge part ids, in any mix. Duplicates are
            collapsed; the result is keyed by :func:`target_key`.

    Returns:
        A :class:`PartLookup` with the three outcomes kept apart. **Never raises for a
        missing or unpinnable part** - the caller decides, because the refusal that fits
        depends on what the caller was doing: F7 refuses a pack binding, a listing just
        reports.

    Example:
        >>> lookup = resolve_parts(session, [reference_from(slot), "part-abc123"])
        >>> if not lookup.all_usable():
        ...     raise (
        ...         lookup.unpinnable_refusal() if lookup.unpinnable
        ...         else lookup.unresolved_refusal()
        ...     )
        >>> part = lookup.pinned["prompt-set/support-tickets@3"]
    """
    if not targets:
        return PartLookup()

    references = [t for t in targets if not isinstance(t, str)]
    ids = [t for t in targets if isinstance(t, str)]

    clauses: list[Any] = []
    for reference in references:
        clauses.append(
            sa.and_(
                CONTRACT_PARTS.c.ref_type == reference.type,
                CONTRACT_PARTS.c.ref_name == reference.name,
                CONTRACT_PARTS.c.ref_version == reference.version,
            )
        )
    if ids:
        clauses.append(CONTRACT_PARTS.c.id.in_(ids))

    rows = [
        dict(row._mapping)
        for row in session.execute(
            sa.select(CONTRACT_PARTS).where(sa.or_(*clauses)).order_by(CONTRACT_PARTS.c.id)
        ).all()
    ]

    split_rows: dict[str, list[Mapping[str, Any]]] = {}
    if rows:
        for split_row in session.execute(
            sa.select(PART_SPLITS)
            .where(PART_SPLITS.c.part_id.in_([str(r["id"]) for r in rows]))
            .order_by(PART_SPLITS.c.part_id, PART_SPLITS.c.name)
        ).all():
            mapping = dict(split_row._mapping)
            split_rows.setdefault(str(mapping["part_id"]), []).append(mapping)

    by_reference: dict[str, dict[str, Any]] = {}
    by_id: dict[str, dict[str, Any]] = {}
    for part_row in rows:
        by_id[str(part_row["id"])] = part_row
        if part_row["ref_type"] is not None:
            key = f"{part_row['ref_type']}/{part_row['ref_name']}@{part_row['ref_version']}"
            by_reference[key] = part_row

    pinned: dict[str, ResolvedPart] = {}
    unpinnable: dict[str, ResolvedPart] = {}
    missing: list[str] = []

    for target in targets:
        key = target_key(target)
        if key in pinned or key in unpinnable or key in missing:
            continue
        found = by_id.get(key) if isinstance(target, str) else by_reference.get(key)
        if found is None:
            missing.append(key)
            continue
        resolved = _resolved(found, split_rows.get(str(found["id"]), []))
        if resolved.pinned:
            pinned[key] = resolved
        else:
            # Conformant, verified, indexed - and it names nothing reproducible. Kept in
            # its own collection so no caller can iterate past the distinction.
            unpinnable[key] = resolved

    return PartLookup(pinned=pinned, unpinnable=unpinnable, missing=tuple(missing))


def _resolved(row: Mapping[str, Any], splits: list[Mapping[str, Any]]) -> ResolvedPart:
    reference = (
        None
        if row["ref_type"] is None
        else ContractReference(
            type=str(row["ref_type"]), name=str(row["ref_name"]), version=str(row["ref_version"])
        )
    )
    return ResolvedPart(
        part_id=str(row["id"]),
        part_kind=str(row["part_kind"]),
        source=str(row["source"]),
        contract_reference=reference,
        payload=part_payload(row, splits),
    )


def resolve_one(session: Session, target: PartTarget) -> ResolvedPart:
    """One target, refusing by name for either failure. For a caller with a single slot.

    Raises:
        Refusal: ``dataset_version_unresolved`` or ``part_unpinnable``.
    """
    lookup = resolve_parts(session, [target])
    key = target_key(target)
    if key in lookup.pinned:
        return lookup.pinned[key]
    if lookup.unpinnable:
        raise lookup.unpinnable_refusal()
    raise lookup.unresolved_refusal()
