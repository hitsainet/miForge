"""``mf_packs`` and ``mf_pack_parts`` - the pack registry  (task 4.2; R-02.38, R-02.39).

A pack is a named, versioned composition holding **references** to parts, never copies.
Parts come from miDataworks under ``midataworks.dataset-version/v1`` or from outside the
suite with their own provenance (R-02.40).

**These tables are an index like every other, as of 2026-10-14.** They were not, and the
reversal is recorded here rather than silently applied.

This paragraph read: *"These tables are the one place the ADR-004 story does not hold... A
pack is the only artifact with no on-disk manifest, so losing the database loses every
unexported pack - and therefore **a pack assertion must not be added to task 4.4's
reindex-equivalence test** until someone decides whether packs get a disk mirror."* That was
the 2026-10-13 review's finding 10 and it was correct when written.

**Someone decided: the gate contract did.** ``pack-definition-v1.json`` reads *"THIS IS AN
ON-DISK FILE, 'pack.json', beside the pack's directory - not a database-only record"*, F7
implemented it atomically through ``core.canonical.write_canonical``, and
``packs.registry.rebuild_index`` reconstructs both tables from those files alone. So losing
the database loses no pack, these are ``INDEX_TABLES``, and the prohibition above is lifted.

**Left in struck-through form on purpose.** A prohibition is the most expensive kind of stale
finding: deleting it costs the next reader the reasoning, and leaving it live means a later
session honours a rule that no longer applies - which is exactly how a correction that lands
in three documents and not the fourth makes the fourth look reviewed.
``test_pack_reindex_wiring.py`` is the assertion that keeps this true.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any

from sqlalchemy import (
    Boolean,
    CheckConstraint,
    DateTime,
    ForeignKey,
    Index,
    Integer,
    String,
    Text,
    false,
    func,
)
from sqlalchemy.orm import Mapped, mapped_column, relationship

from db.base import Base, JsonColumn

#: The six part kinds that define a run (AGENT.md, "Who owns packs"). A seventh slot kind
#: is a schema change, not a row, which is why the set is stated rather than free text.
PART_KINDS = frozenset(
    {"prompt-set", "corpus", "preference-pairs", "selector", "grader", "test-set", "retention-set"}
)
PART_SOURCES = frozenset({"midataworks", "external"})


class Pack(Base):
    """A named, versioned composition of the parts one domain's runs need (R-02.38)."""

    __tablename__ = "mf_packs"

    id: Mapped[str] = mapped_column(String(64), primary_key=True)
    name: Mapped[str] = mapped_column(String(128), nullable=False)
    version: Mapped[str] = mapped_column(String(64), nullable=False)
    #: R-02.38: a run cannot start from an invalid pack. Stored rather than recomputed so
    #: the reason a pack is invalid survives the validation that found it.
    valid: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False)
    invalid_reason: Mapped[str | None] = mapped_column(Text, nullable=True)
    definition_sha256: Mapped[str | None] = mapped_column(String(64), nullable=True)

    #: FR-007.16's Example pill. Added by migration ``0015`` on an operator decision, 2026-10-14.
    #:
    #: **It had no field anywhere, and the consequence was not cosmetic:** F7 shipped **no
    #: example packs at all**, because seeding unmarked packs that are indistinguishable from
    #: real ones is worse than shipping none. Two worked examples are how an operator who has
    #: never seen a pack learns what one is, so the missing boolean cost the product its
    #: onboarding.
    #:
    #: ``False`` rather than nullable: every pack either is or is not an example, and there is
    #: no third state to represent. Existing rows take the server default, which is an
    #: assertion ("not an example") rather than an omission.
    is_example: Mapped[bool] = mapped_column(
        Boolean, nullable=False, server_default=false(), default=False
    )

    #: FR-007.12's derivation record: the pack this one was cloned from. Null means **not
    #: derived**, which is different from a dangling id meaning *derived from something no
    #: longer here* - the registry must render those differently or it erases the fact that a
    #: derivation happened.
    #:
    #: **No database-level foreign key, and that is deliberate rather than an oversight.** It
    #: is a self-reference, SQLite cannot add a constraint to an existing table, and the only
    #: way to gain one is a table rebuild - which would risk silently dropping the CHECK
    #: constraint below, because SQLAlchemy does not reflect SQLite CHECK constraints.
    #: Declaring the FK in the ORM alone would make the two engines disagree about what is
    #: enforced, which ``tests/support/db.py`` already names as its blind spot. So the
    #: reference is enforced in the writer, identically on both engines, and the column is
    #: indexed by ``0015`` so the derivation walk is cheap. ADR-004 makes this table an index
    #: over ``pack.json``: a dangling id is recoverable by ``reindex``; a dropped constraint
    #: is not.
    derived_from_pack_id: Mapped[str | None] = mapped_column(String(64), nullable=True)

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=func.now()
    )

    parts: Mapped[list[PackPart]] = relationship(
        back_populates="pack", cascade="all, delete-orphan", passive_deletes=True
    )

    __table_args__ = (
        Index("uq_mf_packs_name_version", "name", "version", unique=True),
        Index("ix_mf_packs_derived_from", "derived_from_pack_id"),
        # An invalid pack says why. A `valid=False` row with no reason tells an operator
        # only that something is wrong, which is the refusal-without-a-reason ADR-013
        # forbids.
        CheckConstraint("valid OR invalid_reason IS NOT NULL", name="invalid_names_a_reason"),
    )


class PackPart(Base):
    """One slot of one pack: which part kind, from where, pinned to which version.

    ``{type, name, version}`` is miDataworks' **contract** reference form. The
    ``<pack>/<type>/<name>@<version>`` rendering is miForge prefixing its own pack
    identity for display - two layers, and this table stores the inner one.
    """

    __tablename__ = "mf_pack_parts"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    pack_id: Mapped[str] = mapped_column(
        ForeignKey("mf_packs.id", ondelete="CASCADE"), nullable=False
    )
    part_kind: Mapped[str] = mapped_column(String(32), nullable=False)
    source: Mapped[str] = mapped_column(String(16), nullable=False)

    #: The contract reference, for a ``midataworks`` part: ``{type, name, version}``.
    ref_type: Mapped[str | None] = mapped_column(String(64), nullable=True)
    ref_name: Mapped[str | None] = mapped_column(String(128), nullable=True)
    ref_version: Mapped[str | None] = mapped_column(String(64), nullable=True)
    #: An external part's own provenance (R-02.40). Shape is the pack definition
    #: contract's, which foundation task 3.4 settles; stored opaquely here so this table
    #: does not become a second definition of it.
    external_provenance: Mapped[dict[str, Any] | None] = mapped_column(JsonColumn, nullable=True)

    #: FR-007.25's cross-pack addressing: which pack this part was addressed **through**,
    #: when a slot cites a part belonging to another pack. Null is the ordinary case - the
    #: part belongs to this pack. Added by migration ``0015``, 2026-10-14.
    #:
    #: **Display already worked and storage did not**, which is the half that survives a
    #: restart: ``packs.rendering`` takes the pack as an argument, so the
    #: ``<pack>/<type>/<name>@<version>`` form rendered correctly while nothing recorded
    #: which pack had been named. `PackSlot` in ``pack-definition-v1.json`` is closed, so it
    #: could not ride in the artifact either.
    addressed_through_pack_id: Mapped[str | None] = mapped_column(String(64), nullable=True)

    pack: Mapped[Pack] = relationship(back_populates="parts")

    __table_args__ = (
        CheckConstraint("source IN ('midataworks','external')", name="source_known"),
        # A part is pinned or it is not a part. A miDataworks part carries all three
        # contract-reference fields; an external one carries its own provenance. Neither
        # may be empty, because an unpinned part makes a pack version meaningless.
        CheckConstraint(
            "(source = 'midataworks' AND ref_type IS NOT NULL AND ref_name IS NOT NULL "
            "AND ref_version IS NOT NULL) OR "
            "(source = 'external' AND external_provenance IS NOT NULL)",
            name="part_is_pinned",
        ),
        Index("uq_mf_pack_parts_pack_id_part_kind", "pack_id", "part_kind", unique=True),
        Index("ix_mf_pack_parts_addressed_through", "addressed_through_pack_id"),
    )
