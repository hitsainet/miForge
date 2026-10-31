"""F8's one table: ``mf_merges``, the record of a merge  (R-02.45; ADR-003, ADR-004).

## Core ``Table`` on its own ``MetaData``, not ``db.base.Base``

Foundation task 4.2's ratchet asserts ``set(Base.metadata.tables)`` **equals** the seven
base-schema tables, with the message *"a table only one feature reads belongs in that
feature's own migration."* So a feature's tables are deliberately not on ``Base.metadata``.
``services/parts/tables.py`` (F10) and ``services/scoring/tables.py`` (F4) are the worked
examples and this follows them: the same :data:`db.base.NAMING_CONVENTION`, because unnamed
CHECK constraints cannot be dropped by name - which makes a migration reversible on
PostgreSQL and irreversible on the SQLite the unit tests run against - and the same ``mf_``
prefix.

## Why a merge gets a table of its own, and not an ``mf_runs`` row

**A merge is an operation, not a run.** F6's FR-006.55 supplies the mechanical reason F8's
own answer lacked: a stepless run row would be refused at submit for having no stop
condition, would never heartbeat and so would be reaped by the janitor, and would be a run
whose step series is empty - which every stop condition and every curve in F6 assumes is
impossible.

So F8 built ``MergeOutcome`` as a frozen dataclass and reported, accurately, that *"it is not
persisted: ``mf_merges`` does not exist in the schema, and adding it is a migration this
feature does not own."* It does now, via ``0014_merges_and_run_type_vocabulary.py``.

## This table is an index, not the record  (ADR-004)

A merged checkpoint's own ``manifest.json`` carries ``merged_adapter`` and is authoritative -
a merged checkpoint has **two** ancestors and ``parent`` carries only the base, which is why
that field was added to the schema on 2026-10-13.

**But the authoritative copy covers only one of the three outcomes.** A refused or
interrupted merge writes no checkpoint and therefore leaves no manifest, so there is nothing
on disk for ``reindex`` to rebuild those rows from. That is the case this table exists for: a
merge is irreversible, and a merge that **refused** is the outcome an operator most needs to
find afterwards - it is the difference between "the base was never saved so I stopped" and
"nothing happened and I do not know why". Rows with ``outcome = 'written'`` are an index over
the manifests; rows with the other two outcomes are the only record there is.

Stated plainly because it is an exception to ADR-004's shape, and an exception nobody wrote
down is how a later session prunes the table for disk space.

## The constraints, and why each is a constraint rather than a convention

``result_checkpoint_name`` is nullable because two of the three outcomes write no checkpoint.
That nullability is the only place a row could lie about an irreversible operation, so the
pairing is enforced **in both directions**: ``written`` iff a result name is present. A
``written`` row with no checkpoint name says a merge happened and cannot say to what; a
``refused`` row carrying one says a merge that did not happen produced an artifact. Either
would store cleanly without the constraint.

``refusal_code`` is permitted only on a ``refused`` row - not the converse, since
``interrupted`` carries no code and neither does ``written``.
"""

from __future__ import annotations

import sqlalchemy as sa

from db.base import NAMING_CONVENTION

# Imported rather than respelled, so the table's CHECK and the dataclass cannot drift - two
# spellings of one vocabulary is precisely the defect that produced the `run_type`
# divergence which this table's own migration also settled. Verified to pull in none of
# torch/peft/transformers, so declaring a table does not drag in the ML stack.
from training.runtypes.adapters.merge import (
    OUTCOME_INTERRUPTED,
    OUTCOME_REFUSED,
    OUTCOME_WRITTEN,
)

#: F8's own metadata. See the module docstring for why this is not ``Base.metadata``.
metadata = sa.MetaData(naming_convention=NAMING_CONVENTION)

#: The three outcomes R-02.45 allows to be *recorded*.
OUTCOMES: tuple[str, ...] = (OUTCOME_WRITTEN, OUTCOME_REFUSED, OUTCOME_INTERRUPTED)

_OUTCOME_LIST = ",".join(f"'{value}'" for value in OUTCOMES)


MERGES = sa.Table(
    "mf_merges",
    metadata,
    sa.Column("id", sa.Integer, primary_key=True, autoincrement=True),
    # The adapter that was merged, by the identity its own manifest carries. `adapter_name`
    # is a qualified checkpoint name or a Hub repo id, so it is wide and may contain slashes.
    sa.Column("adapter_name", sa.String(255), nullable=False),
    sa.Column("adapter_step", sa.Integer, nullable=False),
    # The adapter manifest's hash, so the row names the exact artifact merged rather than a
    # name that may since have been rewritten. Required: a merge whose input cannot be
    # identified afterwards is a lineage claim nothing can check.
    sa.Column("adapter_manifest_sha256", sa.String(64), nullable=False),
    # The unmerged base. AC17 requires it to survive the merge, so recording which checkpoint
    # it was is what makes that survival checkable afterwards rather than merely promised.
    sa.Column("base_checkpoint_name", sa.String(255), nullable=False),
    # NULL for 'refused' and 'interrupted'. The pairing CHECK below is what stops this
    # nullability becoming a place to hide a lie.
    sa.Column("result_checkpoint_name", sa.String(255), nullable=True),
    sa.Column("outcome", sa.String(16), nullable=False),
    # Set from the RAISED refusal's code. R-02.45's refusals are raised rather than returned,
    # deliberately - a caller can ignore a return value and cannot ignore an exception - so
    # this column records what was raised, written by whoever catches it.
    sa.Column("refusal_code", sa.String(64), nullable=True),
    sa.Column(
        "created_at",
        sa.DateTime(timezone=True),
        server_default=sa.text("(CURRENT_TIMESTAMP)"),
        nullable=False,
    ),
    sa.CheckConstraint(f"outcome IN ({_OUTCOME_LIST})", name="outcome_known"),
    sa.CheckConstraint(
        f"(outcome = '{OUTCOME_WRITTEN}') = (result_checkpoint_name IS NOT NULL)",
        name="written_iff_result",
    ),
    sa.CheckConstraint(
        f"refusal_code IS NULL OR outcome = '{OUTCOME_REFUSED}'",
        name="code_only_when_refused",
    ),
    # The two lookups an operator actually performs: "what happened to this adapter" and
    # "what has been merged onto this base, and did any of it refuse".
    sa.Index("ix_mf_merges_adapter", "adapter_name", "adapter_step"),
    sa.Index("ix_mf_merges_base", "base_checkpoint_name"),
)
