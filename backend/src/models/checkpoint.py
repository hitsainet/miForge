"""``mf_checkpoints`` - the index over checkpoint manifests  (task 4.2, 4.2a, 4.2b; ADR-004).

**Every column here is derived from a ``manifest.json`` on disk.** Dropping this table
loses no artifact; it costs a ``reindex`` (ADR-004), and task 4.4 proves that by asserting
the queries answer identically after a drop-and-rebuild. Where a row and its manifest
disagree, **the manifest wins and the disagreement is logged** (task 4.5) - implemented in
``services/reindex.py``, not left as an aspiration here.

**``run_id`` is deliberately NOT a foreign key to ``mf_runs``.** A checkpoint copied from
another machine keeps its meaning (ADR-004) and arrives with a ``run_id`` that has no local
run row; a FK would refuse to index it, making a real on-disk artifact invisible. The
column is indexed instead, so the join is still cheap where the run does exist.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any

from sqlalchemy import CheckConstraint, DateTime, Index, Integer, String, Text, func
from sqlalchemy.orm import Mapped, mapped_column

from db.base import Base, JsonColumn
from models.domain_state import DomainState

SIGNAL_SOURCES = frozenset({"corpus", "self-generation", "preference-pairs", "reward"})


class Checkpoint(Base):
    """One saved checkpoint directory, as its manifest describes it."""

    __tablename__ = "mf_checkpoints"

    #: The fully qualified name, ``<base_model>:<run_id>:step-<step>`` with the step
    #: UNPADDED. **Task 4.2b:** this is the join key R-02.29's tree walks, so it is the
    #: primary key and therefore indexed by construction rather than by remembering to.
    name: Mapped[str] = mapped_column(String(512), primary_key=True)

    run_id: Mapped[str] = mapped_column(String(64), nullable=False)
    step: Mapped[int] = mapped_column(Integer, nullable=False)

    # ------------------------------------------------------------------ domain, 4.2a
    #: The domain value, or NULL. **Meaningless without ``domain_state``** - read them
    #: together, and read :func:`models.domain_state.render_domain` rather than inventing
    #: a second rendering.
    domain: Mapped[str | None] = mapped_column(String(128), nullable=True)
    #: Which of the three checkpoint states this record is in: a value, an explicit null,
    #: or a key absent from the manifest. NOT NULL: there is no fourth answer, and a
    #: nullable state column would reintroduce exactly the collapse 4.2a forbids.
    domain_state: Mapped[str] = mapped_column(String(16), nullable=False)

    signal_source: Mapped[str] = mapped_column(String(32), nullable=False)

    # ------------------------------------------------------------------ lineage
    #: The ``name`` of the checkpoint this one descends from, or NULL for one saved
    #: directly off the base model. The edge BRD-01 R-01.16 walks.
    parent: Mapped[str | None] = mapped_column(String(512), nullable=True)
    #: The ROOT model this lineage descends from - not the thing most recently loaded. A
    #: continue or a branch loads a *checkpoint*, and writing that checkpoint's identity
    #: here makes the root equal the parent and terminates every lineage walk one hop
    #: short (the 2026-10-13 cross-document review's finding 4). It validates and the
    #: tree looks plausible, which is why it is called out here.
    base_model: Mapped[str] = mapped_column(String(512), nullable=False)
    #: A merged checkpoint has TWO ancestors: the base via ``parent`` and the adapter
    #: here (R-02.45). Stored as the manifest's ``merged_adapter`` object so its
    #: three-state ``manifest_sha256`` - a hash, a conforming null, or absent - is not
    #: flattened on the way in.
    merged_adapter: Mapped[dict[str, Any] | None] = mapped_column(JsonColumn, nullable=True)

    # ------------------------------------------------------------------ identity
    weights_sha256: Mapped[str] = mapped_column(String(64), nullable=False)
    #: Digest of the manifest's own canonical bytes. The handle task 4.5 compares on, and
    #: what ``MergedAdapter.manifest_sha256`` and ``AdapterBase.manifest_sha256`` point at.
    manifest_sha256: Mapped[str] = mapped_column(String(64), nullable=False)

    selector_hash: Mapped[str | None] = mapped_column(String(64), nullable=True)
    grader_hash: Mapped[str | None] = mapped_column(String(64), nullable=True)
    optimizer: Mapped[dict[str, Any] | None] = mapped_column(JsonColumn, nullable=True)

    #: The ``ScoreRecord`` objects verbatim. JSON rather than shredded columns because
    #: ``provider`` is nullable AND not ``required``: a conforming score may omit the key
    #: or carry null, and only a round trip through JSON preserves which. Foundation 3.8
    #: says the same thing - "a test is the only thing that catches the omission."
    test: Mapped[dict[str, Any] | None] = mapped_column(JsonColumn, nullable=True)
    retention: Mapped[dict[str, Any] | None] = mapped_column(JsonColumn, nullable=True)

    #: Where the directory is, relative to the checkpoint root. Relative so the index
    #: survives the root being mounted elsewhere - which it is, between a developer's
    #: machine and the GPU node's PVC.
    path: Mapped[str] = mapped_column(Text, nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    indexed_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=func.now()
    )

    __table_args__ = (
        CheckConstraint("step >= 0", name="step_non_negative"),
        CheckConstraint(
            "signal_source IN ('corpus','self-generation','preference-pairs','reward')",
            name="signal_source_known",
        ),
        # ---- task 4.2a, made unrepresentable rather than merely discouraged ----
        # A checkpoint manifest always HAS the field, conformingly or not, so
        # 'not_in_kind' is refused here: adopting it on a checkpoint is how the adapter
        # rendering leaks across artifact kinds.
        CheckConstraint(
            f"domain_state IN ('{DomainState.RECORDED}','{DomainState.ASSERTED_NULL}',"
            f"'{DomainState.ABSENT}')",
            name="domain_state_is_a_checkpoint_state",
        ),
        # The value is present exactly when the state says it is. This is the constraint
        # that keeps 'asserted_null' and 'absent' apart: without it a row could carry
        # 'absent' with a value, or 'recorded' with none, and the two states would stop
        # meaning anything.
        CheckConstraint(
            f"(domain_state = '{DomainState.RECORDED}') = (domain IS NOT NULL)",
            name="domain_value_iff_recorded",
        ),
        Index("ix_mf_checkpoints_parent", "parent"),
        Index("ix_mf_checkpoints_run_id_step", "run_id", "step"),
        Index("ix_mf_checkpoints_domain_state", "domain_state"),
    )
