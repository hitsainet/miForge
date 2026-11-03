"""F10's four tables  (F10 tasks 5.1-5.4; ADR-003, ADR-004).

**One declaration, and it is this one.** F10 owns these four tables, their migration and
their reads. ``workers/run_record.py`` records what happens when a module re-declares a
table whose owner declares it elsewhere - *"when 4.2 landed, six of its seven column names
were wrong ... every test of it passed, because the tests built its tables"* - so these
tables are declared once, here, and ``backend/tests/unit/test_parts_tables.py`` asserts the
**migrated** database's columns match this declaration. That test is the one that would
have caught those six names.

## Why Core ``Table`` on its own ``MetaData``, not ``db.base.Base``

Foundation task 4.2 draws the boundary and its test enforces it:
``test_the_base_schema_is_exactly_these_seven_tables`` asserts
``set(Base.metadata.tables)`` **equals** the seven base-schema tables, with the message
*"a table only one feature reads belongs in that feature's own migration."* So a feature's
tables are deliberately **not** on ``Base.metadata``. They carry the same
:data:`db.base.NAMING_CONVENTION` - unnamed CHECK and UNIQUE constraints cannot be dropped
by name, which makes a migration reversible on PostgreSQL and irreversible on the SQLite
the unit tests run against - and the same ``mf_`` prefix, which this module's own ratchet
test checks since the base-schema ratchet cannot see them.

Core rather than ORM for the same reason ``workers/run_record.py`` is Core: the writes here
are conditional ``INSERT``/``UPDATE`` statements whose ``WHERE`` clause carries the policy -
the idempotent re-import, the watermark advance - and an ORM attribute assignment cannot
express a predicate two racing writers are serialised by.

## Two groups, two different rules, stated before the columns

**Group A - index, rebuildable from disk.** :data:`CONTRACT_PARTS`, :data:`PART_SPLITS`.
The authority is ``<part_dir>/midataworks-dataset-version.json``, exactly as a checkpoint's
authority is its ``manifest.json`` (ADR-004). :func:`services.parts.index.reindex_parts`
rebuilds both from manifests alone.

**Group B - durable run state, NOT rebuildable.** :data:`REVIEW_BINDINGS`,
:data:`REVIEW_CANDIDATE_REFS`. No local file can reconstruct a miDataworks queue id or an
``external_id`` -> ``item_id`` mapping, because the authority for a decision is
**miDataworks**. These rows are run state in the same sense an attachment id is run state:
*losing it means losing the ability to read decisions back.* A reindex test must assert
these are **out of scope, in words**, so that a later agent does not "fix" the test by
inventing a rebuild for them.

**Holding a queue id is not holding a queue** (FR-010.25, as corrected 2026-10-13). The
prohibition in R-02.36 is on the *thing* - the queue, the decision history, the review
screen - not on a table-name prefix. R-02.36's own closing sentence *requires* the
``external_id`` -> decision-id map, because the decisions' identifiers are recorded with
the batch they admitted.
"""

from __future__ import annotations

import sqlalchemy as sa

from db.base import NAMING_CONVENTION, JsonColumn

#: F10's own metadata. See the module docstring for why this is not ``Base.metadata``.
metadata = sa.MetaData(naming_convention=NAMING_CONVENTION)

#: The two sources a part can come from. **Never conflated** (R-02.40): an external part
#: has null contract-reference columns, and that null is the enforcement of "miForge does
#: not silently promote an external part to a contract reference."
SOURCE_CONTRACT = "midataworks_contract"
SOURCE_EXTERNAL = "external_files"
PART_SOURCES: frozenset[str] = frozenset({SOURCE_CONTRACT, SOURCE_EXTERNAL})

#: miDataworks' decision vocabulary **as read**, plus ``none`` for *no decision yet*.
#: Verified in their ``services/review/decision_rules.py``:
#: ``DECISIONS = ("accept", "override", "flag", "reject")``.
#:
#: This column stores **what was read**, never miForge's verdict value. The single
#: conversion lives in :mod:`services.review.verdicts`. A column holding a third
#: vocabulary would be a second mapping that some caller re-derives - and ``pending`` reads
#: as a terminal value where ``undecided`` reads as a non-terminal one, which is the whole
#: of *silence is not consent*.
DECISION_NONE = "none"
DECISION_VALUES: tuple[str, ...] = ("accept", "override", "flag", "reject", DECISION_NONE)


CONTRACT_PARTS = sa.Table(
    "mf_contract_parts",
    metadata,
    sa.Column("id", sa.String(64), primary_key=True),
    # The gate contract's `PartReference.kind` vocabulary (batch-record-v1.json), not a
    # fourth spelling. `services.parts.kind_map` asserts the two agree.
    sa.Column("part_kind", sa.String(32), nullable=False),
    sa.Column("source", sa.String(24), nullable=False),
    # The contract reference, `{type, name, version}`. NULL for an external part, and
    # **null for a conformant version that carries none** - `Target.contract_reference`
    # is `anyOf [ContractReference, null]` and is in `required`, so "no reference" is an
    # assertion miDataworks deliberately allows. Nothing may synthesise one; F7 refuses
    # such a part at pack-slot binding with `part_unpinnable`.
    sa.Column("ref_type", sa.String(64), nullable=True),
    sa.Column("ref_name", sa.String(128), nullable=True),
    sa.Column("ref_version", sa.String(64), nullable=True),
    sa.Column("dataset", sa.String(128), nullable=True),
    sa.Column("version_id", sa.String(64), nullable=True),
    sa.Column("version_number", sa.Integer, nullable=True),
    sa.Column("recipe_sha256", sa.String(64), nullable=True),
    sa.Column("content_kind", sa.String(24), nullable=False),
    sa.Column("config_sha256", sa.String(64), nullable=True),
    sa.Column("local_path", sa.Text, nullable=False),
    # The digest of the manifest's bytes **as received**. Not of a re-serialisation:
    # re-serialising would digest miForge's serialisation rather than what miDataworks
    # sent, which makes the value useless as the cross-repo check it exists to be.
    sa.Column("manifest_sha256", sa.String(64), nullable=False),
    sa.Column("row_count", sa.Integer, nullable=False, server_default="0"),
    # Verbatim, both of them. Not filtered, not summarised, not reduced to a boolean: the
    # shape is miDataworks' and normalising it would be miForge taking an opinion on
    # someone else's vocabulary. FR-010.17 requires them carried, not understood.
    sa.Column("licence_classes", JsonColumn, nullable=True),
    sa.Column("licences", JsonColumn, nullable=True),
    sa.Column("caveats", JsonColumn, nullable=True),
    # An external part's own provenance - the gate contract's `ExternalProvenance`
    # (batch-record-v1.json). Stored opaquely so this table does not become a second
    # definition of it.
    sa.Column("external_provenance", JsonColumn, nullable=True),
    sa.Column("imported_at", sa.DateTime(timezone=True), nullable=False),
    sa.CheckConstraint(
        f"source IN ('{SOURCE_CONTRACT}', '{SOURCE_EXTERNAL}')", name="source_known"
    ),
    # R-02.40 at the database. An external part may not carry a contract reference, which
    # is the enforcement of "provenance is never a lie" rather than a convention three
    # call sites are trusted to honour.
    sa.CheckConstraint(
        f"source <> '{SOURCE_EXTERNAL}' OR "
        "(ref_type IS NULL AND ref_name IS NULL AND ref_version IS NULL)",
        name="external_part_has_no_contract_reference",
    ),
    # A reference is whole or absent. Two of three fields is an identity nothing can
    # resolve, and the schema's `ContractReference` requires all three.
    sa.CheckConstraint(
        "(ref_type IS NULL AND ref_name IS NULL AND ref_version IS NULL) OR "
        "(ref_type IS NOT NULL AND ref_name IS NOT NULL AND ref_version IS NOT NULL)",
        name="contract_reference_is_whole_or_absent",
    ),
    # Idempotent re-import as a **database guarantee**, not a service convention. Partial,
    # because external parts and reference-less conformant versions both have null
    # columns and several of them must be able to coexist.
    sa.Index(
        "uq_mf_contract_parts_reference",
        "ref_type",
        "ref_name",
        "ref_version",
        unique=True,
        sqlite_where=sa.text(f"source = '{SOURCE_CONTRACT}'"),
        postgresql_where=sa.text(f"source = '{SOURCE_CONTRACT}'"),
    ),
    # The parts list filters by kind; it is the only filter the index serves.
    sa.Index("ix_mf_contract_parts_part_kind", "part_kind"),
)


PART_SPLITS = sa.Table(
    "mf_part_splits",
    metadata,
    sa.Column("id", sa.Integer, primary_key=True, autoincrement=True),
    sa.Column(
        "part_id",
        sa.String(64),
        sa.ForeignKey("mf_contract_parts.id", ondelete="CASCADE"),
        nullable=False,
    ),
    sa.Column("name", sa.String(128), nullable=False),
    sa.Column("path", sa.Text, nullable=False),
    sa.Column("rows", sa.Integer, nullable=False),
    sa.Column("bytes", sa.Integer, nullable=False),
    sa.Column("sha256", sa.String(64), nullable=False),
    # miDataworks' row-level digest, which depends on its key scheme. **Recorded, never
    # recomputed**: recomputing it would mean reimplementing someone else's key scheme,
    # which is a second answer to their question.
    sa.Column("logical_digest", sa.String(64), nullable=False),
    # Required booleans on the schema's `Split` - so F2's refusal has something to read
    # rather than a field it has to default (FR-010.18).
    sa.Column("held_out", sa.Boolean, nullable=False),
    sa.Column("evaluation_only", sa.Boolean, nullable=False),
    sa.Column("label_counts", JsonColumn, nullable=True),
    sa.Index("uq_mf_part_splits_part_id_name", "part_id", "name", unique=True),
)


REVIEW_BINDINGS = sa.Table(
    "mf_review_bindings",
    metadata,
    sa.Column("id", sa.String(64), primary_key=True),
    # FK to mf_runs by name rather than by object: that table is on `Base.metadata` and
    # this one is not, so the constraint is emitted by the migration and never resolved
    # by a `create_all` over this metadata alone.
    sa.Column("run_id", sa.String(64), nullable=False, unique=True),
    # miDataworks' queue id. **Losing it loses read-back**, which is why it is a column on
    # a durable row and not a value held in a worker's memory: every worker is
    # `--pool=solo -c 1`, so anything that must reach a running job goes through the
    # database (ADR-006, ADR-007).
    sa.Column("queue_id", sa.String(64), nullable=False),
    sa.Column("question", sa.Text, nullable=False),
    sa.Column("label_set", JsonColumn, nullable=False),
    sa.Column("posted", sa.Integer, nullable=False, server_default="0"),
    sa.Column("decided", sa.Integer, nullable=False, server_default="0"),
    sa.Column("pending", sa.Integer, nullable=False, server_default="0"),
    # The `since` watermark. **timezone-aware**: a naive column silently shifts the
    # watermark, and a shifted watermark loses decisions permanently.
    sa.Column("read_back_cursor", sa.DateTime(timezone=True), nullable=True),
    sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
    sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False),
    sa.CheckConstraint("posted >= 0 AND decided >= 0 AND pending >= 0", name="counts_non_negative"),
)


REVIEW_CANDIDATE_REFS = sa.Table(
    "mf_review_candidate_refs",
    metadata,
    sa.Column("id", sa.Integer, primary_key=True, autoincrement=True),
    sa.Column(
        "binding_id",
        sa.String(64),
        sa.ForeignKey("mf_review_bindings.id", ondelete="CASCADE"),
        nullable=False,
    ),
    sa.Column("step", sa.Integer, nullable=False),
    # Derived deterministically from (run_id, step, candidate_index, prompt_hash) through
    # the one canonical-JSON function, so a re-post after a crash is idempotent:
    # miDataworks returns the existing item for a repeated external id.
    sa.Column("external_id", sa.String(256), nullable=False),
    sa.Column("item_id", sa.String(64), nullable=True),
    sa.Column("decision_id", sa.String(64), nullable=True),
    # miDataworks' vocabulary as read, with `none` meaning *no decision yet*. A stored
    # value, not an absence: an absence invites a default and a value does not, and
    # "how many is this run waiting on" becomes a COUNT over an indexed column rather
    # than a set difference.
    sa.Column("decision", sa.String(16), nullable=False, server_default=DECISION_NONE),
    sa.Column("override_label", sa.Text, nullable=True),
    # Untrusted data, stored as data (FR-010.29). Never interpreted as an instruction,
    # never used to build a path or a command.
    sa.Column("reason", sa.Text, nullable=True),
    sa.Column("decided_by", sa.String(128), nullable=True),
    sa.Column("decided_by_origin", sa.String(16), nullable=True),
    sa.Column("decided_at", sa.DateTime(timezone=True), nullable=True),
    sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
    sa.CheckConstraint(
        "decision IN ('" + "', '".join(DECISION_VALUES) + "')", name="decision_known"
    ),
    # Only an override carries a label - their own `validate()` enforces the same rule on
    # the way in, and a label beside an `accept` here would mean miForge recorded something
    # miDataworks never said.
    sa.CheckConstraint(
        "override_label IS NULL OR decision = 'override'", name="only_override_carries_a_label"
    ),
    sa.Index(
        "uq_mf_review_candidate_refs_binding_id_external_id",
        "binding_id",
        "external_id",
        unique=True,
    ),
    # "How many undecided" runs **every step**; it is the hot query of this feature.
    sa.Index("ix_mf_review_candidate_refs_binding_id_decision", "binding_id", "decision"),
    # The watermark read.
    sa.Index("ix_mf_review_candidate_refs_binding_id_decided_at", "binding_id", "decided_at"),
)


#: **Group C - operator state, NOT rebuildable and NOT an index.** Which parts the operator
#: archived (operator decision 2026-10-20; migration 0022): hidden from ``GET /parts`` and the
#: part pickers unless ``include_archived``, never deleted. A table of its own because
#: :data:`CONTRACT_PARTS` is emptied and rebuilt by ``reindex_parts`` - a column there would
#: silently unarchive every part on the next reindex. **No foreign key** to
#: :data:`CONTRACT_PARTS` for the same reason: a cascade would erase the archive on every
#: reindex. Part ids are deterministic (``index.part_id_for``), so a row keeps naming the same
#: part across a rebuild.
PART_ARCHIVE = sa.Table(
    "mf_part_archive",
    metadata,
    sa.Column("part_id", sa.String(64), primary_key=True),
    sa.Column("archived_at", sa.DateTime(timezone=True), nullable=False),
)


#: The tables ``reindex`` may rebuild from disk (Group A).
PART_INDEX_TABLES: tuple[str, ...] = (CONTRACT_PARTS.name, PART_SPLITS.name)

#: The tables ``reindex`` must **not** touch (Group B). Named, so a test can assert the
#: exclusion in words rather than a later agent inventing a rebuild for them.
REVIEW_STATE_TABLES: tuple[str, ...] = (REVIEW_BINDINGS.name, REVIEW_CANDIDATE_REFS.name)

#: The operator-state tables ``reindex`` must **not** touch (Group C).
ARCHIVE_STATE_TABLES: tuple[str, ...] = (PART_ARCHIVE.name,)

ALL_TABLES: tuple[str, ...] = PART_INDEX_TABLES + REVIEW_STATE_TABLES + ARCHIVE_STATE_TABLES
