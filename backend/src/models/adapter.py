"""``mf_adapters`` - the index over adapter manifests  (task 4.2, 4.2a, 4.2b; ADR-004).

An adapter is a standard PEFT directory holding ``adapter_model.safetensors`` and
``adapter_manifest.json``. Like :mod:`models.checkpoint` this table is an index over
those manifests and nothing else.

**``domain`` is the fourth case.** ``adapter-manifest-v1.json`` declares no ``domain``
property and sets ``additionalProperties: false``, so a conforming adapter manifest
*cannot* carry one. Every row therefore holds ``domain_state = 'not_in_kind'`` - enforced
by CHECK below, not by convention - and ``domain`` stays NULL. The adapter's base is a
checkpoint that *does* have a domain, and filling this in from there is the specific
inference task 4.2a forbids: an adapter trained on a different pack than its base would be
labelled with the base's domain with nothing saying so.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any

from sqlalchemy import CheckConstraint, DateTime, Index, Integer, String, Text, func
from sqlalchemy.orm import Mapped, mapped_column

from db.base import Base, JsonColumn
from models.domain_state import DomainState


class Adapter(Base):
    """One saved adapter directory, as its manifest describes it."""

    __tablename__ = "mf_adapters"

    #: ``<base_model>:<run_id>:adapter-<step>``, step UNPADDED. The ``adapter-`` segment
    #: is what distinguishes an adapter from a checkpoint of the same run and step in
    #: R-02.29's tree. **Task 4.2b:** the join key, hence the primary key.
    name: Mapped[str] = mapped_column(String(512), primary_key=True)

    run_id: Mapped[str] = mapped_column(String(64), nullable=False)
    step: Mapped[int] = mapped_column(Integer, nullable=False)

    # ------------------------------------------------------------------ domain, 4.2a
    #: Always NULL, and always NULL for a *stated* reason: see the module docstring.
    domain: Mapped[str | None] = mapped_column(String(128), nullable=True)
    domain_state: Mapped[str] = mapped_column(
        String(16), nullable=False, default=DomainState.NOT_IN_KIND.value
    )

    # ------------------------------------------------------------------ lineage
    #: The base's fully qualified name, as the base's own checkpoint manifest gives it.
    #: R-02.29's tree joins an adapter on this rather than on ``parent``: an adapter's
    #: edge is to the base its own manifest names.
    base_name: Mapped[str] = mapped_column(String(512), nullable=False)
    #: NULL when the base was not itself a saved checkpoint - a conforming value, per the
    #: schema's own description, and not the same as the field being missing.
    base_manifest_sha256: Mapped[str | None] = mapped_column(String(64), nullable=True)

    # ------------------------------------------------------------------ identity
    weights_sha256: Mapped[str] = mapped_column(String(64), nullable=False)
    manifest_sha256: Mapped[str] = mapped_column(String(64), nullable=False)

    #: Rank, alpha, target modules, dropout and the scaling rule. Stored opaquely: an
    #: unrecognised ``scaling`` value must be **refused, not guessed** (the schema says
    #: so and says why), and shredding the object into columns would quietly drop the
    #: unknown value instead of carrying it to the refusal.
    adapter_config: Mapped[dict[str, Any] | None] = mapped_column(JsonColumn, nullable=True)
    optimizer: Mapped[dict[str, Any] | None] = mapped_column(JsonColumn, nullable=True)
    #: The pack parts this adapter trained on. The adapter manifest carries these and the
    #: checkpoint manifest deliberately does not, because an adapter is small and
    #: routinely moved and must say what it trained on without its run log alongside.
    parts: Mapped[list[Any] | None] = mapped_column(JsonColumn, nullable=True)

    test: Mapped[dict[str, Any] | None] = mapped_column(JsonColumn, nullable=True)
    retention: Mapped[dict[str, Any] | None] = mapped_column(JsonColumn, nullable=True)

    path: Mapped[str] = mapped_column(Text, nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    indexed_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=func.now()
    )

    __table_args__ = (
        CheckConstraint("step >= 0", name="step_non_negative"),
        # ---- task 4.2a's fourth case, enforced ----
        CheckConstraint(
            f"domain_state = '{DomainState.NOT_IN_KIND}'", name="domain_state_is_not_in_kind"
        ),
        CheckConstraint("domain IS NULL", name="domain_never_inferred_from_base"),
        Index("ix_mf_adapters_base_name", "base_name"),
        Index("ix_mf_adapters_run_id_step", "run_id", "step"),
    )
