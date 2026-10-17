"""The registry: the listing, and the index behind it  (task 5.x; FR-007.11 - FR-007.13, FR-007.16).

The registry lists every pack with its identity, validity, part count, prompt count, the
runs using it, and its derivation. The **pack store on disk is the truth**
(:mod:`packs.store`); ``mf_packs`` and ``mf_pack_parts`` are an index over it, which is what
ADR-004 says every other artifact's database rows are.

## That makes a pack rebuildable, and the reindex question answerable

The 2026-10-13 review left an instruction: *"a pack assertion must not be added to the
reindex-equivalence test without first deciding whether packs get a disk mirror."* The gate
contract decided it - ``pack.json`` on disk - so :func:`rebuild_index` exists and a pack
**is** now rebuildable from the filesystem alone. **It is wired:** ``services/reindex.py``
calls :func:`rebuild_index` and the rows are read back by
``tests/integration/test_pack_reindex_wiring.py``, which goes red if that call is removed.
*(Until 2026-10-14 this paragraph said the wiring was owed to another owner; it was then
built, and this docstring was corrected on 2026-10-16.)*

## Two numbers that must not be invented

* **``prompt_count`` is ``None`` when it cannot be resolved - never 0** (ADR-018,
  FR-007.11). A prompt count of 0 because miDataworks is unreachable is a fabricated value,
  and the UI renders ``None`` as "unknown". ADR-012 then requires the number to name its
  scale and sample, which is why the listing carries ``parts_filled`` **and**
  ``parts_total``.
* **``runs_using`` is read from run records, read-only.** Run rows are F1's and F6's; this
  feature reads them and must never write one.

## Three fields that live only in the index, and why

``is_example`` (FR-007.16's Example pill), ``derived_from_pack_id`` (FR-007.12's derivation
record) and ``mf_pack_parts.addressed_through_pack_id`` (FR-007.25's cross-pack addressing)
have **no field in ``pack-definition-v1.json``** - its top-level object and ``PackSlot`` are
closed - so each is a **column**, added by migration ``0015`` (operator decision,
2026-10-14). Each has a writer: :func:`clone` records the derivation,
:func:`mark_example` sets the flag (its production caller is
``backend/scripts/seed_example_packs.py``), and :func:`record_addressing` stores the pack a
slot's part is addressed through - reached from ``POST /api/v1/packs``'s
``addressed_through`` and read back by the lineage route. *That last writer did not exist
until 2026-10-17 (operator decision Q3): the column was null on every row.* Because none can ride in
the artifact, :func:`rebuild_index` **preserves** them across a rebuild (``DB_ONLY_PACK_FIELDS``,
``DB_ONLY_PART_FIELDS``); they are the one piece of pack state the filesystem is not
authoritative for. *(Until 2026-10-16 this section said the columns did not exist and the
listing always reported ``derived_from: None``; both stopped being true at ``0015``.)*
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any

import sqlalchemy as sa
from sqlalchemy.orm import Session

from core.errors import refuse
from models.pack import Pack, PackPart
from packs import store
from packs.definition import FilledSlot, PackVersionDefinition
from packs.errors import PACK_IN_USE, PackRefusal
from packs.references import ContractPartRef
from packs.rendering import contract_form, display_form, external_form
from packs.resolver import Resolution, resolve_slots
from packs.slots import SLOT_ORDER, SLOT_TO_STORAGE_KIND, SlotKind, storage_part_kind
from packs.validation import PackValidity, validate

#: The tables ``services.reindex`` may rebuild from disk, named here rather than in that
#: module so a rename cannot leave it citing a table that no longer exists - the same reason
#: F10 exports ``PART_INDEX_TABLES``. A silently-stale name would make foundation task 4.4's
#: equivalence assertion skip the very rows it was added to cover.
#:
#: These became **index** tables on 2026-10-14. A pack was the one artifact with no on-disk
#: manifest, which is why the 2026-10-13 review forbade adding a pack assertion to that
#: equivalence test until someone decided whether packs get a disk mirror. The gate contract
#: decided it: ``pack-definition-v1.json`` says ``pack.json`` is an on-disk file, and
#: :func:`rebuild_index` reconstructs both tables from those files alone.
PACK_INDEX_TABLES: tuple[str, ...] = (Pack.__tablename__, PackPart.__tablename__)


def row_id(name: str, version: str) -> str:
    """``mf_packs.id`` for a pack version.

    One row per **version**, because that table's unique key is ``(name, version)`` and
    because ``mf_runs.pack_id`` points at it: a run cites the version it ran, not the pack
    in general, and R-02.35's guarantee is about a version.
    """
    return f"{name}@{version}"


@dataclass(frozen=True, slots=True)
class RegistryRow:
    """One line of the registry listing (FR-007.11).

    Every number carries its scale and sample (ADR-012): ``parts_filled`` of
    ``parts_total``, ``prompt_count`` which may be ``None`` for unknown, and
    ``runs_using`` as a list rather than a count so the UI can name them.
    """

    pack_id: str
    version: str
    name: str
    domain: str | None
    parts_filled: int
    parts_total: int
    prompt_count: int | None
    runs_using: tuple[str, ...]
    validity: PackValidity
    #: FR-007.12's derivation record. ~~``None`` always, today; there is nowhere to record
    #: it.~~ **Recorded since migration ``0015``** (operator decision, 2026-10-14).
    #: ``None`` now means **not derived**; a value naming a pack that no longer exists means
    #: *derived from something no longer here*, and the two must render differently or the
    #: fact that a derivation happened is erased.
    derived_from: str | None
    #: FR-007.16's Example pill. ~~``None`` always, today; the Example pill is owed a
    #: field.~~ **Recorded since migration ``0015``.** ``False``, not ``None``, for a pack
    #: nobody has marked - every pack either is or is not an example.
    is_example: bool | None
    definition_sha256: str
    created_at: datetime

    def payload(self) -> dict[str, Any]:
        return {
            "pack_id": self.pack_id,
            "version": self.version,
            "name": self.name,
            "domain": self.domain,
            "parts_filled": self.parts_filled,
            "parts_total": self.parts_total,
            "prompt_count": self.prompt_count,
            "runs_using": list(self.runs_using),
            "validity": self.validity.payload(),
            "derived_from": self.derived_from,
            "is_example": self.is_example,
            "definition_sha256": self.definition_sha256,
            "created_at": self.created_at.isoformat().replace("+00:00", "Z"),
        }


def create(session: Session, definition: PackVersionDefinition) -> PackVersionDefinition:
    """Commit a pack version: write the file, then index it.

    **The file first, then the index.** If the index write fails, a rebuild recovers the
    pack; if the order were reversed, a crash between the two would leave a row pointing at
    a pack that does not exist - and ADR-004's rule is that where they disagree the
    filesystem wins, which only works if the filesystem is written first.
    """
    store.write_committed(definition)
    index(session, definition)
    return definition


def index(session: Session, definition: PackVersionDefinition) -> Pack:
    """Project one pack version into ``mf_packs`` / ``mf_pack_parts``.

    Validity is stored **and never trusted**: :func:`packs.gate.require_valid_pack`
    re-resolves. The column exists because ``mf_packs`` has it and its check constraint
    requires a reason beside a false, so an invalid pack's row says *why* - a ``valid=False``
    row with no reason tells an operator only that something is wrong.
    """
    identifier = row_id(definition.name, definition.version)
    validity = validate(definition, resolve_slots(session, definition))

    row = session.get(Pack, identifier)
    if row is None:
        row = Pack(id=identifier, name=definition.name, version=definition.version)
        session.add(row)
    row.valid = validity.is_valid
    row.invalid_reason = None if validity.is_valid else _reason(validity)
    row.definition_sha256 = definition.document()["definition_sha256"]

    session.execute(sa.delete(PackPart).where(PackPart.pack_id == identifier))
    session.flush()
    for slot in SLOT_ORDER:
        filled = definition.slots.get(slot)
        if filled is None:
            continue
        session.add(_part_row(identifier, slot, filled))
    session.flush()
    return row


def _part_row(identifier: str, slot: SlotKind, filled: FilledSlot) -> PackPart:
    """One ``mf_pack_parts`` row.

    **Typed reference columns, not a JSON blob** for a contract part: ``mf_pack_parts``
    carries ``ref_type``/``ref_name``/``ref_version`` and a check constraint requiring all
    three, so a wrong-shaped contract reference is impossible at the storage layer and
    FAC-6's "no borrowed provenance" is queryable rather than only tested. An external
    part's provenance goes in that table's own ``external_provenance`` column, in the
    contract's ``ExternalProvenance`` shape - **never** in the ref columns, which is the
    mutation FAC-6's negative control performs.
    """
    ref = filled.ref
    if isinstance(ref, ContractPartRef):
        return PackPart(
            pack_id=identifier,
            part_kind=storage_part_kind(slot),
            source="midataworks",
            ref_type=ref.type,
            ref_name=ref.name,
            ref_version=ref.version,
            external_provenance=None,
        )
    return PackPart(
        pack_id=identifier,
        part_kind=storage_part_kind(slot),
        source="external",
        ref_type=None,
        ref_name=None,
        ref_version=None,
        # The contract's `ExternalProvenance` shape and nothing added: this column is
        # documented as holding it "opaquely so this table does not become a second
        # definition of it", and a miForge-only key here would be exactly that.
        external_provenance=ref.as_external_dict(),
    )


def _reason(validity: PackValidity) -> str:
    """One line saying why a pack is not valid, for the index row's own record."""
    parts: list[str] = []
    if validity.missing_slots:
        parts.append("missing " + ", ".join(slot.value for slot in validity.missing_slots))
    if validity.floor_missing:
        parts.append("retention floor missing")
    if validity.unresolved:
        parts.append("unresolved " + ", ".join(item.slot.value for item in validity.unresolved))
    if validity.unavailable:
        parts.append(
            "service unavailable for " + ", ".join(item.slot.value for item in validity.unavailable)
        )
    return "; ".join(parts) or "not valid"


#: The columns that live **only** in ``mf_packs`` / ``mf_pack_parts`` and cannot be rebuilt
#: from ``pack.json``. Added by migration ``0015`` on an operator decision, 2026-10-14.
#:
#: **These three break ADR-004's "the database is a rebuildable index" for themselves, and
#: that is not a defect in the decision — it is a consequence that had to be handled.**
#: `pack-definition-v1.json`'s top-level object and its ``PackSlot`` are both closed to
#: unknown keys, which is precisely why each needed a column: none of the three can ride in
#: the artifact. So a rebuild that read only the files would **silently erase all three** —
#: seed an example pack, run ``reindex``, lose the Example flag and the derivation history.
#:
#: :func:`rebuild_index` therefore preserves them across a rebuild instead of dropping them.
#: That makes them the one piece of pack state the filesystem is *not* authoritative for,
#: which is stated here rather than discovered later, and pinned by
#: ``test_pack_reindex_preserves_db_only_columns``.
DB_ONLY_PACK_FIELDS: tuple[str, ...] = (
    "is_example",
    "derived_from_pack_id",
)
DB_ONLY_PART_FIELDS: tuple[str, ...] = ("addressed_through_pack_id",)


def _db_only(session: Session) -> dict[str, tuple[bool, str | None]]:
    """``{pack row id: (is_example, derived_from_pack_id)}`` for every indexed pack.

    Read in one query rather than per pack: the registry listing is budgeted at 500 ms for
    50 packs and a per-row lookup is how a listing becomes N+1 without anyone noticing.
    """
    return {
        str(row[0]): (bool(row[1]), row[2])
        for row in session.execute(
            sa.select(Pack.id, Pack.is_example, Pack.derived_from_pack_id)
        ).all()
    }


def listing(session: Session) -> list[RegistryRow]:
    """Every pack version, read from the store and resolved once per pack (FR-007.11)."""
    rows: list[RegistryRow] = []
    usage = usage_map(session)
    db_only = _db_only(session)
    for definition in store.iter_versions():
        resolutions = resolve_slots(session, definition)
        rows.append(_row(definition, resolutions, usage, db_only))
    return rows


def get(session: Session, name: str, version: str) -> tuple[PackVersionDefinition, RegistryRow]:
    """One pack version and its registry line."""
    definition = store.read(name, version)
    resolutions = resolve_slots(session, definition)
    return definition, _row(definition, resolutions, usage_map(session), _db_only(session))


def _row(
    definition: PackVersionDefinition,
    resolutions: Mapping[SlotKind, Resolution],
    usage: Mapping[str, tuple[str, ...]],
    db_only: Mapping[str, tuple[bool, str | None]] | None = None,
) -> RegistryRow:
    validity = validate(definition, resolutions)
    prompt = resolutions.get(SlotKind.PROMPT_SET)
    identifier = row_id(definition.name, definition.version)
    # A pack present on disk but not yet indexed has no row to read, which is an ordinary
    # state during a rebuild. `is_example` is False there rather than None: a pack nobody
    # has marked is not an example, and that is an assertion rather than an absence.
    example, derived = (db_only or {}).get(identifier, (False, None))
    return RegistryRow(
        pack_id=definition.name,
        version=definition.version,
        name=definition.name,
        domain=definition.domain,
        parts_filled=len(definition.filled()),
        parts_total=len(SLOT_ORDER),
        # None, never 0: an unresolvable prompt set has an UNKNOWN count, and 0 would be a
        # fabricated measurement the operator would act on (ADR-018).
        prompt_count=None if prompt is None else prompt.rows,
        runs_using=usage.get(row_id(definition.name, definition.version), ()),
        validity=validity,
        derived_from=derived,
        is_example=example,
        definition_sha256=definition.document()["definition_sha256"],
        created_at=definition.created_at,
    )


def usage_map(session: Session) -> dict[str, tuple[str, ...]]:
    """Which runs used which pack version. **Read-only** over F1's and F6's rows.

    One query for the whole listing rather than one per pack: ``GET /packs`` has a 500 ms
    budget for fifty packs and a per-row query is the easiest way to lose it.
    """
    from models.run import Run

    result = session.execute(
        sa.select(Run.pack_id, Run.id).where(Run.pack_id.is_not(None)).order_by(Run.id)
    ).all()
    usage: dict[str, list[str]] = {}
    for pack_id, run_id in result:
        usage.setdefault(str(pack_id), []).append(str(run_id))
    return {key: tuple(value) for key, value in usage.items()}


def clone(
    session: Session, name: str, version: str, *, into: str, new_version: str = "1"
) -> PackVersionDefinition:
    """A new pack whose slots start as this one's references (FR-007.12).

    References, not resolution facts: a clone is a new composition pinning the same parts,
    and copying a row count into it would be copying a measurement taken at another time.

    ~~**The derivation is not recorded**, and that is reported rather than hidden.~~
    **RECORDED since migration ``0015``** (operator decision, 2026-10-14): the clone's row
    carries ``derived_from_pack_id`` naming the exact source version it was taken from.

    **The column was added and the capability still did not exist until this line.** A column
    nothing writes is the same unreachable-capability failure as a function nothing calls,
    and the test that pinned the gap - ``test_a_clone_records_no_derivation_...`` - went red
    on the column's arrival and stayed red until the writer did too. That is the tripwire
    working: the schema change alone would have shipped a registry column permanently null.

    It cannot ride in ``pack.json`` (the document's top-level object is closed), so it lives
    only in the index - which is why :func:`rebuild_index` carries it across a rebuild.
    """
    source = store.read(name, version)
    clone_definition = PackVersionDefinition(
        name=into,
        version=new_version,
        domain=source.domain,
        slots=dict(source.slots),
        created_at=datetime.now(tz=UTC),
    )
    created = create(session, clone_definition)

    # AFTER create(), because create() writes the file and then indexes it; the row does not
    # exist before that. The source is named by its ROW ID (`<name>@<version>`), not by name
    # alone: a clone of `notary-md@1` and a clone of `notary-md@2` are different derivations,
    # and recording only the pack name would lose which version was copied.
    row = session.get(Pack, row_id(into, new_version))
    if row is not None:
        row.derived_from_pack_id = row_id(name, version)
        session.flush()
    return created


def mark_example(session: Session, name: str, version: str, *, is_example: bool = True) -> bool:
    """Set (or clear) ``mf_packs.is_example`` for one pack version (FR-007.16).

    Returns:
        ``True`` if the row changed, ``False`` if it already said this. The two are reported
        separately rather than collapsed into "done", because the seeder's whole job is to
        say what it did and *"already marked"* and *"marked just now"* are different facts.

    Raises:
        PackRefusal: ``pack_version_committed`` is **not** raised. A committed version's
            *slots* are frozen (FR-007.2) and this is not a slot: it is index-only state,
            it is absent from the document the digest covers, and marking a pack as an
            example changes nothing about the run input the version names.

    **The writer exists because the column alone is not the capability.** ``0015`` added
    ``is_example`` and nothing wrote it, which is the same unreachable capability as a
    function nothing calls - the registry would have shipped a permanently-``false`` column
    and an Example pill that could never appear. :func:`clone` learned the same lesson for
    ``derived_from_pack_id`` on the same day.

    **Its production caller is ``backend/scripts/seed_example_packs.py``.** No API route sets
    this today, which is stated rather than papered over: the pill is produced by an operator
    running the seeder, and if a route is ever wanted it calls this function rather than
    writing the column a second time.
    """
    row = session.get(Pack, row_id(name, version))
    if row is None:
        # Not a refusal taxonomy code: a caller that has not created the pack yet has a
        # sequencing bug, not a condition an operator can act on. `create()` writes the file
        # and then indexes it, so the row exists by the time it returns.
        raise refuse(
            "pack_not_indexed",
            f"Pack version {name}@{version} has no index row, so there is nothing to mark. "
            f"Create it first, or run reindex if the index was dropped.",
            pack_id=name,
            version=version,
        )
    if bool(row.is_example) == is_example:
        return False
    row.is_example = is_example
    session.flush()
    return True


def delete(session: Session, name: str, version: str) -> None:
    """Delete a pack version, **refusing while a run cites it** (E11).

    Raises:
        PackRefusal: ``pack_in_use``, naming the runs. A run's history must not lose the
            thing that defined it: R-02.35's guarantee is that a pack version names a
            reproducible run input, and a deleted pack version makes every run that cited
            it unreproducible. ``mf_runs.pack_id`` is ``ON DELETE RESTRICT`` for the same
            reason, so this refusal is the readable form of a constraint that would
            otherwise surface as an integrity error.
    """
    identifier = row_id(name, version)
    runs = usage_map(session).get(identifier, ())
    if runs:
        raise PackRefusal(
            PACK_IN_USE,
            f"Pack version {identifier} was used by run(s) {list(runs)}, so deleting it "
            f"would leave those runs unable to say what they trained on. Export it if you "
            f"want it elsewhere; it stays here while those runs do.",
            pack_id=name,
            version=version,
            runs=list(runs),
        )
    session.execute(sa.delete(PackPart).where(PackPart.pack_id == identifier))
    row = session.get(Pack, identifier)
    if row is not None:
        session.delete(row)
    session.flush()
    store.delete(name, version)


def rebuild_index(
    session: Session, *, definitions: Iterable[PackVersionDefinition] | None = None
) -> int:
    """Drop the pack index and rebuild it **from the files alone**. Returns the row count.

    This is the pack half of ADR-004's reindex, and it is what makes the 2026-10-13 review's
    open question answerable: with ``pack.json`` on disk, losing the database loses no pack.

    **WIRED 2026-10-14.** This read *"not wired into ``services/reindex.py``: that module is
    another owner's file, and whether foundation 4.4's equivalence test should now assert
    packs is the coordinator's call. Reported as owed wiring rather than claimed as done."*
    That was right when written, and the coordinator closed it: ``reindex`` calls this, and
    both pack tables moved from ``RECORD_TABLES`` to ``INDEX_TABLES``.

    The finding is struck through here rather than deleted, because the pattern is worth more
    than the line: this was the **fourth** rebuilder in that module with no production caller.
    ``test_pack_reindex_wiring.py`` is what keeps the call there - it seeds packs, empties the
    index, runs the production ``reindex`` and reads the rows back, so it fails on a call
    passing the wrong session as well as on no call at all. Proven by deleting the call and
    seeing three of its assertions go red.
    """
    # CARRIED ACROSS THE REBUILD, because `pack.json` cannot hold them and a rebuild that
    # read only the files would erase them. See `DB_ONLY_PACK_FIELDS`. Captured BEFORE the
    # delete, which is the only moment they still exist.
    carried_packs = _db_only(session)
    carried_parts = {
        (str(row[0]), str(row[1])): row[2]
        for row in session.execute(
            sa.select(PackPart.pack_id, PackPart.part_kind, PackPart.addressed_through_pack_id)
        ).all()
        if row[2] is not None
    }

    session.execute(sa.delete(PackPart))
    session.execute(sa.delete(Pack))
    session.flush()
    count = 0
    for definition in definitions if definitions is not None else store.iter_versions():
        index(session, definition)
        count += 1
    session.flush()

    # Restored only for packs the rebuild actually produced. A carried value for a pack the
    # store no longer holds is DROPPED rather than re-inserted: the filesystem is
    # authoritative for *which packs exist*, and resurrecting a row for a deleted pack would
    # make this function the one place a pack can come back from the dead.
    for identifier, (example, derived) in carried_packs.items():
        row = session.get(Pack, identifier)
        if row is None:
            continue
        row.is_example = example
        row.derived_from_pack_id = derived
    for (pack_id, part_kind), through in carried_parts.items():
        part = session.execute(
            sa.select(PackPart).where(PackPart.pack_id == pack_id, PackPart.part_kind == part_kind)
        ).scalar_one_or_none()
        if part is not None:
            part.addressed_through_pack_id = through
    session.flush()
    return count


def check_addressing(
    session: Session, definition: PackVersionDefinition, addressed_through: Mapping[SlotKind, str]
) -> None:
    """Refuse an addressing that cannot be recorded, **before** anything is written.

    Run before :func:`create` writes the pack file, so a refused addressing leaves no pack
    behind (the file is the authority and is written first - a refusal after it would leave
    a pack without the addressing its author asked for).

    Raises:
        Refusal: ``pack_addressed_through_self`` (a slot addressed through its own pack is
            not cross-pack), ``pack_addressed_through_unknown`` (no such pack version is
            indexed) and ``pack_addressed_through_empty_slot`` (no part fills that slot).
    """
    identifier = row_id(definition.name, definition.version)
    for slot, through in addressed_through.items():
        if through == identifier:
            raise refuse(
                "pack_addressed_through_self",
                f"Slot {slot.value} of {identifier} is addressed through {through}, which is "
                f"the pack itself. Cross-pack addressing names a DIFFERENT pack (FR-007.25).",
                slot=slot.value,
                pack_id=identifier,
            )
        if session.get(Pack, through) is None:
            raise refuse(
                "pack_addressed_through_unknown",
                f"Slot {slot.value} of {identifier} is addressed through {through!r}, and no "
                f"such pack version is indexed. Name it as <pack>@<version>.",
                slot=slot.value,
                through=through,
            )
        if definition.slots.get(slot) is None:
            raise refuse(
                "pack_addressed_through_empty_slot",
                f"Slot {slot.value} of {identifier} holds no part, so it cannot be addressed "
                f"through another pack.",
                slot=slot.value,
                pack_id=identifier,
            )


def record_addressing(
    session: Session, name: str, version: str, addressed_through: Mapping[SlotKind, str]
) -> int:
    """Record which pack each slot's part is addressed THROUGH (FR-007.25). Returns the count.

    **The writer ``mf_pack_parts.addressed_through_pack_id`` never had.** Migration ``0015``
    added the column on the operator's decision and nothing set it, so cross-pack addressing
    was rendered from a parameter nobody passed and stored nowhere. The value is the other
    pack's **row id** (``<name>@<version>``), like ``derived_from_pack_id``: the version is
    part of the identity, and a name alone would lose which version the part came through.
    Callers validate first with :func:`check_addressing`.

    Index-only state, like ``is_example``: the artifact cannot carry it (``PackSlot`` is
    closed), so :func:`rebuild_index` preserves it across a rebuild.
    """
    identifier = row_id(name, version)
    count = 0
    for slot, through in addressed_through.items():
        part = session.execute(
            sa.select(PackPart).where(
                PackPart.pack_id == identifier, PackPart.part_kind == storage_part_kind(slot)
            )
        ).scalar_one()
        part.addressed_through_pack_id = through
        count += 1
    session.flush()
    return count


def addressing(session: Session, name: str, version: str) -> dict[SlotKind, str]:
    """``{slot: <pack>@<version>}`` for every slot addressed through another pack."""
    by_kind = {kind: slot for slot, kind in SLOT_TO_STORAGE_KIND.items()}
    rows = session.execute(
        sa.select(PackPart.part_kind, PackPart.addressed_through_pack_id).where(
            PackPart.pack_id == row_id(name, version),
            PackPart.addressed_through_pack_id.is_not(None),
        )
    ).all()
    return {by_kind[str(kind)]: str(through) for kind, through in rows if kind in by_kind}


def lineage(
    definition: PackVersionDefinition,
    resolutions: Mapping[SlotKind, Resolution],
    *,
    addressed_through: Mapping[SlotKind, str] | None = None,
) -> list[dict[str, Any]]:
    """Per slot: where the part came from, in full (FR-007.26).

    The record that makes FR-007.24 **auditable** rather than merely asserted, and the
    artifact of US-8: six months on, an operator reads this and can say where every part came
    from without opening another application.

    ``addressed_through`` names the pack a part is addressed **through** when it is not this
    one (FR-007.25's cross-pack mixing). It is a parameter rather than a stored field because
    ``PackSlot`` is closed on ``{contract_reference, external, floor}`` - the artifact cannot
    carry it, which is reported - and because the display form needs the pack anyway.
    """
    through = addressed_through or {}
    rows: list[dict[str, Any]] = []
    for slot in SLOT_ORDER:
        filled = definition.slots.get(slot)
        if filled is None:
            rows.append({"slot": slot.value, "source": None, "state": "empty"})
            continue
        ref = filled.ref
        pack_id = through.get(slot, definition.name)
        resolution = resolutions.get(slot)
        rows.append(
            {
                "slot": slot.value,
                "source": ref.source,
                # The reference in full, in ITS OWN layer: a contract reference as
                # `{type,name,version}`, an external part as its own provenance. Never one
                # rendered as the other (FR-007.24).
                "reference": (
                    ref.as_contract_dict()
                    if isinstance(ref, ContractPartRef)
                    else dict(ref.as_external_dict())
                ),
                "inner_form": (
                    contract_form(ref) if isinstance(ref, ContractPartRef) else external_form(ref)
                ),
                # The display form, computed here because this view is read by a screen.
                # It is display only and is never written back into a reference field.
                "display_form": str(display_form(ref, pack_id)),
                "addressed_through": pack_id,
                "floor": filled.floor,
                "state": "unresolved" if resolution is None else resolution.state,
                "rows": None if resolution is None else resolution.rows,
                "service": None if resolution is None else resolution.service,
            }
        )
    return rows


def payloads(rows: Sequence[RegistryRow]) -> list[dict[str, Any]]:
    """The listing as the API serves it."""
    return [row.payload() for row in rows]
