"""``mf_runs`` and ``mf_steps`` - the run record and the step index  (task 4.2; ADR-003, ADR-007).

**ADR-007: the run record is the protocol.** A run is reached through its row, not through
a Celery control message - a solo worker never reads the control queue (ADR-006) - so
cancellation and liveness are columns here, and that is why this table is in the base
schema rather than in F6's own migration: F1 writes steps, F6 drives the loop, F9 reads
both, and the janitor (ADR-007) reads the heartbeat.

**These two tables are records, not an index.** ``reindex`` does not rebuild them and
task 4.4's equivalence test must not assert over them: a run's status and the gap between
its heartbeats are nowhere in any manifest.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any

from sqlalchemy import (
    BigInteger,
    CheckConstraint,
    DateTime,
    ForeignKey,
    Index,
    Integer,
    String,
    Text,
    func,
    text,
)
from sqlalchemy.orm import Mapped, mapped_column, relationship

from db.base import Base, JsonColumn

#: Terminal statuses. The heartbeat **refuses to move a terminal row** (ADR-007, task
#: 6.4), so the set has to be stated somewhere both the heartbeat and the janitor read.
TERMINAL_STATUSES = frozenset({"completed", "failed", "cancelled"})
RUN_STATUSES = frozenset({"pending", "running", *TERMINAL_STATUSES})
#: The run types ``mf_runs.run_type`` permits. **The CHECK constraint below is rendered from
#: this set**, so the two cannot disagree — which they did until 2026-10-14.
#:
#: This read ``{"create", "full", "adapter", "merge"}`` after migration ``0014`` renamed the
#: value to ``change``: the constraint was updated and **this constant was not**. It was
#: exported from ``models/__init__.py`` and read by nothing, so it was a dead second registry
#: carrying the stale spelling — the exact shape of the ``DB_RUN_TYPE`` translation map that
#: same migration removed, reappearing one file away. Found independently by F6 (which
#: documented *"'full' is the stale spelling in models/run.RUN_TYPES"* and worked around it)
#: and by F9 (which found it exported and unused). Neither could fix it: ``models/`` is the
#: coordinator's.
#:
#: ``merge`` is permitted and nothing should write it — a merge is an operation, not a run,
#: recorded in ``mf_merges`` (F6's FR-006.55: a stepless run row is refused at submit for
#: having no stop condition, never heartbeats, and is reaped).
#:
#: **This is not the authoritative list** — ``training.runtypes``' live registry is, and
#: ``models/`` must not import ``training/`` (a layering inversion). The two are kept in step
#: by a test rather than by an import.
#: ~~``merge`` is permitted and nothing should write it.~~ **Narrowed by migration `0016`,
#: 2026-10-14, because it is now PROVED.** `0014`'s comment left it in *"rather than removed
#: because F6 is not built, so nothing has yet proved no submit path produces one."* F6 is
#: built: `services/run_launch.launch` is the only path that creates an `mf_runs` row,
#: `RunDefinition.run_type` is validated through F8's registry (`change|create|adapter`), and
#: `workers/merge_tasks` writes `mf_merges` and never a run row. A merge is an **operation**,
#: not a run - F6's FR-006.55 gives the mechanical reason: a stepless run row is refused at
#: submit for having no stop condition, never heartbeats, and is reaped by the janitor.
RUN_TYPES = frozenset({"create", "change", "adapter"})


class Run(Base):
    """One training run. R-02.53's definition, resolved, plus its live state."""

    __tablename__ = "mf_runs"

    #: The ``run_id`` carried by every manifest and every log line. Stable for the life of
    #: the run; a new ``run_id`` is a new branch (R-02.23).
    id: Mapped[str] = mapped_column(String(64), primary_key=True)
    status: Mapped[str] = mapped_column(String(16), nullable=False, default="pending")
    run_type: Mapped[str] = mapped_column(String(16), nullable=False)

    #: R-02.53's run definition, as submitted and as resolved, hashed canonically and
    #: durable before the enqueue. Two fields because the difference between them is the
    #: record of what miForge filled in.
    definition_submitted: Mapped[dict[str, Any] | None] = mapped_column(JsonColumn, nullable=True)
    definition_resolved: Mapped[dict[str, Any] | None] = mapped_column(JsonColumn, nullable=True)
    definition_sha256: Mapped[str | None] = mapped_column(String(64), nullable=True)

    pack_id: Mapped[str | None] = mapped_column(
        ForeignKey("mf_packs.id", ondelete="RESTRICT"), nullable=True
    )

    #: ADR-008: pinned by **UUID**, resolved at submit from live ``nvidia-smi -L``.
    #: ``gpu_request`` keeps what was asked for (``auto`` or a UUID) beside what was
    #: resolved, because "auto" and "this exact card" are different requests that can
    #: resolve to the same card.
    gpu_request: Mapped[str | None] = mapped_column(String(64), nullable=True)
    gpu_uuid: Mapped[str | None] = mapped_column(String(64), nullable=True)

    #: THE HEARTBEAT COLUMN (task 6.4a names it here so nothing has to choose a second
    #: name). Its **age** is what a liveness rule reads - never the 60 s write interval,
    #: which is a minimum between writes and not a maximum gap.
    heartbeat_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    #: Task 6.4b: the longest gap actually observed on this run. Nothing else in the
    #: product can produce the measurement that gives the janitor's threshold a value.
    longest_heartbeat_gap_seconds: Mapped[int | None] = mapped_column(Integer, nullable=True)

    #: Cooperative cancellation (ADR-006): set here, read at a step boundary. Never
    #: ``revoke(terminate=True)``.
    #: An operator asked for a checkpoint. Consumed by the loop at a **step boundary**, the
    #: same place cancellation is checked, so a save never lands mid-step.
    #:
    #: On ``mf_runs`` rather than a table of its own, which is FR-005.1's straight reading
    #: ("a flag on the `mf_runs` row") and also the better engineering: the loop reads this
    #: row at every boundary anyway for the heartbeat and the cancel flag, so a separate
    #: table would add a join to the hottest path in the product to carry one timestamp.
    #:
    #: A TIMESTAMP, not a boolean - it answers "when was it asked", which is what
    #: distinguishes a slow save from an ignored request, exactly as ``cancel_requested_at``
    #: does. Two requests before the next boundary **collapse**, because one save is what
    #: was wanted; ``save_request_count`` keeps the *fact* that two were asked for, so an
    #: operator pressing the button twice can see the system noticed both.
    save_requested_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True), nullable=True
    )
    save_request_count: Mapped[int] = mapped_column(Integer, nullable=False, server_default="0")

    cancel_requested_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True), nullable=True
    )

    last_step: Mapped[int | None] = mapped_column(Integer, nullable=True)
    failure_code: Mapped[str | None] = mapped_column(String(64), nullable=True)
    failure_message: Mapped[str | None] = mapped_column(Text, nullable=True)

    # ---------------------------------------------------------------- F6's index columns
    # Added by migration 0016. Every one is a value F6 already writes to the run log's
    # `boundary` block, which ADR-004 makes authoritative; these columns make the run list
    # and the run screen a query instead of a file scan per run. All nullable with no
    # defaults: a run that has not ended has no stop reason, and NULL says exactly that.

    #: Which of R-02.25's five conditions ended the run. Constrained, because a sixth value
    #: here would be a sixth stop condition arriving without anyone deciding - what F6's
    #: explicit registry exists to prevent, enforced again at the storage layer.
    #:
    #: **The derivation behind this column is load-bearing, not a convenience.** F6's
    #: mutation M-19b proved it: the boundary block's fired-condition array is written in
    #: *registry* order, which puts `test_threshold` before `retention_floor`, so a reader
    #: taking the first element reports *"the run reached its target"* for a run that fell
    #: below its floor. The reason must be derived by **precedence**, never by array order.
    stop_reason: Mapped[str | None] = mapped_column(String(32), nullable=True)
    stop_detail: Mapped[str | None] = mapped_column(Text, nullable=True)

    #: R-02.26's round counter, incremented at a refresh boundary.
    round: Mapped[int | None] = mapped_column(Integer, nullable=True)

    #: FR-006.24's audit label: **who** asked for the stop. `cancel_requested_at` records
    #: that someone did and never which someone, so an operator stop was unattributable.
    stop_requested_by: Mapped[str | None] = mapped_column(String(120), nullable=True)

    #: FR-006.53's disk projection, **both bounds**. Two columns rather than one, because a
    #: single column would force picking a bound to store and ADR-012 forbids presenting one
    #: bound as the figure - the range for a 10,000-step run at `save_every=100` is
    #: 480-960 GB, and the low end is not the answer.
    projected_bytes_low: Mapped[int | None] = mapped_column(BigInteger, nullable=True)
    projected_bytes_high: Mapped[int | None] = mapped_column(BigInteger, nullable=True)

    # ------------------------------------------------- §5.10 only (migration 0017)
    #: The serving process this run attached to, and the attachment id authorising its
    #: holder-only calls. **Null for every acquisition mode but §5.10**, and that is an
    #: assertion rather than an omission: a run that did not attach has no attachment.
    #:
    #: **These exist because `attach` returns the id exactly ONCE and no read path returns
    #: it** (BRD-01 R-01.31). Seven of BRD-01's fifteen calls are holder-only and carry
    #: `X-miLLM-Attachment` beside the bearer token - including `commit-begin`,
    #: `commit-end`, `save` and `detach` - so **losing the id is losing the ability to
    #: commit**, and therefore to finish a step, save, or detach cleanly. Holding it in
    #: worker memory "for now" is F11's MUT-12 shipped on purpose: it passes every test,
    #: works until the first worker restart, then produces a refused commit nothing explains
    #: and a run that dies looking like a crash.
    hotupdate_base_url: Mapped[str | None] = mapped_column(String(255), nullable=True)
    #: **SECRET-HANDLED.** It authorises the holder-only calls, so it is a credential in
    #: everything but name: excluded from every API serialisation, **only ever logged as a
    #: digest**, and `repr=False` so it stays out of `repr(Run)`, out of a traceback frame
    #: and out of a debugger watch. F11's secret-hygiene test drives a full run with a log
    #: capture and asserts the value never appears.
    #:
    #: ⚠ **`repr=False` DOES NOT WORK HERE and was tried.** F11's handover suggested it, and
    #: SQLAlchemy refuses: *"includes dataclasses argument(s): 'repr' but class does not
    #: specify SQLAlchemy native dataclass configuration."* `Run` is a plain declarative
    #: model, not `MappedAsDataclass`. So the protection is the explicit `__repr__` below -
    #: **a mechanism rather than a keyword that silently is not there.** Had the kwarg been
    #: accepted-and-ignored instead of refused, this column would have looked protected and
    #: not been; the refusal is the better failure.
    attachment_id: Mapped[str | None] = mapped_column(String(128), nullable=True)
    #: The attachment's state as last observed from miLLM's status endpoint, written by the
    #: trainer (migration 0020; operator decision 2026-10-16) and served read-only to the
    #: shell's attachment chip: ``{weights_state, resident_model, holder, stale, attached,
    #: observed_at}``. Null = not reported, never a guessed ``clean``.
    hotupdate_status: Mapped[dict[str, Any] | None] = mapped_column(JsonColumn, nullable=True)

    def __repr__(self) -> str:
        """Identity and state only. **`attachment_id` is deliberately absent.**

        A `repr` reaches places nobody chose to send it: a traceback frame, a debugger watch,
        a structured log line that interpolates the object, pytest's assertion rewriting on
        an unrelated failure. The attachment id authorises every holder-only call to miLLM,
        so it is a credential in everything but name and must not ride along.

        `hotupdate_base_url` is included: it is a location, not a secret, and knowing which
        serving process a run attached to is exactly what a reader of a traceback needs.
        """
        return (
            f"Run(id={self.id!r}, status={self.status!r}, run_type={self.run_type!r}, "
            f"hotupdate_base_url={self.hotupdate_base_url!r})"
        )

    #: R-02.27: this run's place in a curriculum, or null for a standalone run.
    #:
    #: **No database-level foreign key to `mf_chains`, deliberately.** That table lives on
    #: F6's own `MetaData` (foundation 4.2's ratchet keeps `Base.metadata` to the seven
    #: base-schema tables), so a `ForeignKey` declared here would be an unresolvable
    #: cross-metadata reference that `create_all` over `Base.metadata` alone cannot satisfy.
    #: SQLite also cannot add a constraint to an existing table, so gaining one would mean a
    #: second rebuild. Same reasoning as `mf_packs.derived_from_pack_id`: the reference is
    #: enforced in the writer, identically on both engines, and indexed by `0016`.
    chain_id: Mapped[str | None] = mapped_column(String(64), nullable=True)
    stage_index: Mapped[int | None] = mapped_column(Integer, nullable=True)

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=func.now()
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=func.now()
    )
    finished_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)

    steps: Mapped[list[Step]] = relationship(
        back_populates="run", cascade="all, delete-orphan", passive_deletes=True
    )

    __table_args__ = (
        CheckConstraint(
            "status IN ('pending','running','completed','failed','cancelled')",
            name="status_known",
        ),
        # ``change``, not ``full``. BRD-02 §4 names the three run types as *"direct weight
        # **change** on an existing model"*, from-scratch creation, and adapter training, and
        # F8's registry refuses anything but ``change|create|adapter``. Until migration
        # ``0014`` this constraint said ``full`` and F8 bridged the gap with a
        # ``DB_RUN_TYPE = {"change": "full"}`` map - a translation table between two names
        # for one concept, which is a second source of truth and the defect class this
        # project has paid for repeatedly. ``change`` also simply says more: a row reading
        # ``run_type='full'`` does not say full *what*.
        #
        # ``merge`` is permitted and nothing should write it: a merge is an operation, not a
        # run, and gets no row here (F6's FR-006.55 - a stepless run would be refused at
        # submit for having no stop condition, never heartbeat, and be reaped). It is left in
        # rather than removed because F6 is not built, so nothing has yet *proved* no submit
        # path produces one, and removing a value something needs costs a migration under
        # load. ``mf_merges`` is where a merge is recorded.
        # RENDERED FROM `RUN_TYPES`, not written out again. The literal form is exactly what
        # let the constant and the constraint disagree for a day after migration 0014.
        CheckConstraint(
            "run_type IN ({})".format(", ".join(f"'{value}'" for value in sorted(RUN_TYPES))),
            name="run_type_known",
        ),
        # R-02.25's five, and nothing else. See `stop_reason`'s own note.
        CheckConstraint(
            "stop_reason IS NULL OR stop_reason IN "
            "('step_count','test_threshold','retention_floor','loss_plateau','operator_stop')",
            name="stop_reason_known",
        ),
        # NO `completed_names_its_reason` CONSTRAINT, and its absence is deliberate.
        # F6 asked for `CHECK (status <> 'completed' OR stop_reason IS NOT NULL)` on the
        # sound argument that a completed row which cannot say why it stopped is a record an
        # operator cannot act on. It was built and then dropped: migration `0016`'s downgrade
        # drops `stop_reason`, so every completed run returns with NULL there and the next
        # upgrade's rebuild REFUSES ITS OWN ROWS - the revision becomes non-reversible on any
        # installation with a finished run. Backfilling a reason would be a fabricated
        # history in the table an operator reads to find out what happened.
        #
        # The property moved rather than vanished: F6 derives the reason by precedence at the
        # boundary, and mutation M-19 (`stop_reason` permanently null) is red. Enforced where
        # the value is written, on both engines, by a control seen to fail.
        # A range is a range in both directions.
        CheckConstraint(
            "projected_bytes_low IS NULL OR projected_bytes_high IS NULL "
            "OR projected_bytes_low <= projected_bytes_high",
            name="projection_is_ordered",
        ),
        # A run is a chain stage or it is not. Half of either is a run whose place in a
        # curriculum nothing can state.
        CheckConstraint(
            "(chain_id IS NULL) = (stage_index IS NULL)",
            name="chain_membership_is_whole",
        ),
        Index("ix_mf_runs_chain_id", "chain_id"),
        # A terminal run has a finish time and a live one does not. Stated as a constraint
        # because a terminal row with no `finished_at` is how a janitor ends up unable to
        # tell a finished run from one it should reap.
        CheckConstraint(
            "(status IN ('completed','failed','cancelled')) = (finished_at IS NOT NULL)",
            name="terminal_iff_finished",
        ),
        Index("ix_mf_runs_status_created_at", "status", "created_at"),
    )


class Step(Base):
    """One step of one run - the index over the run log's **step** block (R-02.14, F1).

    The authoritative record is the run log on disk; this table exists so the run view's
    loss curve is a query rather than a file scan.
    """

    __tablename__ = "mf_steps"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    run_id: Mapped[str] = mapped_column(
        ForeignKey("mf_runs.id", ondelete="CASCADE"), nullable=False
    )
    #: THE CANONICAL STEP, as an integer. Every rendered form derives from this one value
    #: (BRD-01 R-01.24): paths pad it to six digits, names do not pad it. Nothing carries
    #: the step as a string of its own.
    step: Mapped[int] = mapped_column(Integer, nullable=False)

    loss: Mapped[float | None] = mapped_column(nullable=True)
    #: WHICH objective produced ``loss``. Added 2026-10-14, and it is the load-bearing one
    #: of the three columns added with it: **a loss figure is interpretable against another
    #: only when both say which objective produced it**, and F9 draws loss curves from this
    #: table. F1 dispatches on ``batch.objective`` through F2's registry rather than
    #: hard-coding a cross-entropy, so a run's objective can change between stages - and a
    #: curve that silently splices two objectives together looks like a regression.
    objective: Mapped[str | None] = mapped_column(String(64), nullable=True)
    grad_norm: Mapped[float | None] = mapped_column(nullable=True)
    learning_rate: Mapped[float | None] = mapped_column(nullable=True)
    wall_time_ms: Mapped[int | None] = mapped_column(Integer, nullable=True)

    #: SECTION 5.10 ONLY, and nullable for a reason that is a requirement defect worth
    #: knowing. R-02.14 demands both on *every* step record - written when every step held a
    #: commit bracket. Standalone there is no quiesce and no bracket, so demanding them would
    #: make the core product unable to satisfy its own requirement. They are present-and-null
    #: on every step F1 writes, and F11 fills them.
    #:
    #: ``commit_duration_ms`` is **not local telemetry**: miLLM Feature 31's acceptance gate
    #: is set from its p99.9, so this column is a cross-repo product obligation. The
    #: percentile query reads the run log by nearest-rank until this column is populated -
    #: never interpolating a duration that was never measured, because the gate is 10x it.
    quiesce_hold_ms: Mapped[float | None] = mapped_column(nullable=True)
    commit_duration_ms: Mapped[float | None] = mapped_column(nullable=True)
    #: Hashes of the batch this step consumed, so a step can be traced to its provenance
    #: without the batch text ever being stored (ADR-020).
    batch_source_hashes: Mapped[dict[str, Any] | None] = mapped_column(JsonColumn, nullable=True)

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=func.now()
    )

    run: Mapped[Run] = relationship(back_populates="steps")

    __table_args__ = (
        CheckConstraint("step >= 0", name="step_non_negative"),
        Index("uq_mf_steps_run_id_step", "run_id", "step", unique=True),
        # Supports FR-001.26's commit-duration distribution: a percentile over one run's
        # non-null durations. Partial, because the column is null on every standalone step
        # and an index over mostly-null rows is mostly wasted.
        Index(
            "ix_mf_steps_commit_duration",
            "run_id",
            "commit_duration_ms",
            postgresql_where=text("commit_duration_ms IS NOT NULL"),
        ),
    )
