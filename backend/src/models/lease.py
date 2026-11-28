"""``mf_leases`` - one run per card, with an expiry  (task 4.2; ADR-008).

**A bare boolean is the anti-pattern.** ADR-008: *one run per card via a lease with an
expiry, not a bare boolean.* A boolean has no holder, so a crashed run leaves the card
marked busy forever and the only recovery is someone editing the database; and it has no
expiry, so nothing can distinguish "held" from "abandoned".

**The resource is a GPU UUID, never an index and never a bus address** (ADR-008). Indices
inverted once on this node, and a copied bus address unbound the root NVMe. The ratchet
test bans hard-coded GPU sites; this column's *content* is resolved at submit from live
``nvidia-smi -L``.
"""

from __future__ import annotations

from datetime import datetime

from sqlalchemy import CheckConstraint, DateTime, Index, Integer, String, func, text
from sqlalchemy.orm import Mapped, mapped_column

from db.base import Base


class Lease(Base):
    """An exclusive, expiring claim on one resource by one run."""

    __tablename__ = "mf_leases"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    #: The GPU UUID, as ``nvidia-smi -L`` reports it.
    resource: Mapped[str] = mapped_column(String(64), nullable=False)
    #: Which run holds it. Not nullable: a lease with no holder is the boolean again.
    holder_run_id: Mapped[str] = mapped_column(String(64), nullable=False)

    acquired_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=func.now()
    )
    #: NOT NULL. An unexpiring lease is a card lost to a crash.
    expires_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    released_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)

    __table_args__ = (
        # Exclusivity, in the database rather than in a service's good intentions: a
        # partial unique index on the unreleased rows. Two workers racing for the same
        # card both pass an application-level check; only one passes this.
        Index(
            "uq_mf_leases_resource_unreleased",
            "resource",
            unique=True,
            # Written as SQL text, not as `released_at.is_(None)`: inside
            # `__table_args__` the name still refers to the MappedColumn descriptor
            # rather than to a Column, so the expression form does not compile here.
            sqlite_where=text("released_at IS NULL"),
            postgresql_where=text("released_at IS NULL"),
        ),
        CheckConstraint("expires_at > acquired_at", name="expiry_after_acquisition"),
        Index("ix_mf_leases_holder_run_id", "holder_run_id"),
    )
