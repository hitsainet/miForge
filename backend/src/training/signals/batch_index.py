"""The batch index: one row per batch, rebuilt from the run log  (FR-002.32; ADR-004).

**The log is the truth and this is a projection of it.** ``services/reindex.py`` rebuilds
the checkpoint and adapter indexes from on-disk *manifests*; a batch has no manifest and its
authoritative copy is the run log's ``batch`` block, so :func:`rebuild_batch_index` is the
rebuild for this table - exactly the shape F1's ``rebuild_step_index`` takes for ``mf_steps``,
for the same reason and with the same delete-then-insert idempotence.

## Every column is a projection of the block, and three obvious ones are therefore absent

:func:`row_from_block` is the **single** projection: the live writer and the rebuild both
call it, which is what makes ADR-016 #4's equivalence test meaningful. Two projections would
agree until one of them changed, and the symptom would be a reindex that quietly returns
different numbers.

That rule decides the column list, and it excludes three fields a reader will look for:

* **``objective``** - ``batch-record-v1.json`` has no objective field (the step record
  requires one instead), so the authoritative copy of a batch's objective is
  ``mf_steps.objective``, beside the loss it produced.
* **the named health signals** - the record's ``SelectionHealth`` is four fields and has no
  room for F3's reason histogram, grader statistics or its three named signals. F6 reads
  those from the **in-process** batch (``TrainingBatch.health``) within the step that
  produced them; a column here could not be rebuilt from the log and would make this index
  unrebuildable, which is a worse defect than a missing convenience.
* **hash provenance** - ``selector_config_sha256`` is a bare string in the contract with no
  ``contract``/``local`` provenance beside it. A column for it could not be filled by the
  rebuild, and a *default* of ``contract`` would be the fabricated provenance claim R-02.40
  forbids. F3 publishes the provenance in process and F5's manifest write takes it from
  there. **Reported as owed on the gate contract** (see this feature's report).

**No text column.** Not ``prompt``, not ``completion``, not ``sample``. The absence *is* the
guarantee (ADR-020, FR-002.17), and a test asserts the table's column set contains none.

## Its own ``MetaData``, and no migration in this feature

Foundation task 4.2 draws the boundary and its test enforces it: the base schema holds the
tables **more than one feature reads**, and ``set(Base.metadata.tables)`` must equal exactly
those seven. A batch row has one writer and one reader, so it lives on this module's own
metadata with ``db.base.NAMING_CONVENTION`` - F10's ``services/parts/tables.py`` is the
worked example. **The Alembic revision is the coordinator's**, because the revision graph is
shared state and three features are in flight: two revisions off one parent give everybody
multiple heads, on someone else's clean clone.
"""

from __future__ import annotations

from collections.abc import Mapping
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import sqlalchemy as sa

from core.errors import refuse
from db.base import NAMING_CONVENTION, JsonColumn
from training import run_log

#: F2's own metadata. See the module docstring for why this is not ``Base.metadata``.
metadata = sa.MetaData(naming_convention=NAMING_CONVENTION)

BATCHES = sa.Table(
    "mf_batches",
    metadata,
    # No surrogate key: `(run_id, batch_sha256)` IS the identity, because the hash is what
    # the step record cites and what joins a step to the material it consumed.
    sa.Column("run_id", sa.String(64), primary_key=True),
    sa.Column("batch_sha256", sa.String(64), primary_key=True),
    # NULLABLE, and not an error state: a batch can exist with no step - assembled and then
    # discarded by a cancellation, or consumed by a step that failed. A queryable state.
    sa.Column("step", sa.Integer, nullable=True),
    sa.Column("source", sa.String(32), nullable=False),
    # Which model produced the candidates. NULL for a corpus batch, and that null is the
    # statement "nothing was generated" rather than a missing value.
    sa.Column("resolved_model", sa.String(256), nullable=True),
    # Copied onto the row, not joined: a batch keeps its meaning if a part row is removed,
    # and parts are immutable once imported (F10), so the copy is truthful.
    sa.Column("part_refs", JsonColumn, nullable=False),
    sa.Column("prompt_ids", JsonColumn, nullable=False),
    sa.Column("n_prompts", sa.Integer, nullable=False),
    # THREE verdict counts, never two. `undecided` is a real outcome of an asynchronous
    # human-review selector, and a two-column tally would force every pending candidate to
    # be miscounted as accepted or rejected.
    sa.Column("n_accept", sa.Integer, nullable=False),
    sa.Column("n_reject", sa.Integer, nullable=False),
    sa.Column("n_undecided", sa.Integer, nullable=False),
    sa.Column("candidates_generated", sa.Integer, nullable=False),
    sa.Column("candidates_admitted", sa.Integer, nullable=False),
    sa.Column("acceptance_rate", sa.Float, nullable=False),
    sa.Column("exhausted", sa.Boolean, nullable=False),
    sa.Column("selector_config_sha256", sa.String(64), nullable=True),
    sa.Column("grader_config_sha256", sa.String(64), nullable=True),
    sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
    # The paginated newest-first listing is the only hot read.
    sa.Index("ix_mf_batches_run_id_created_at", "run_id", "created_at"),
    # F9's and F6's filters.
    sa.Index("ix_mf_batches_run_id_source", "run_id", "source"),
)


def row_from_block(block: Mapping[str, Any]) -> dict[str, Any]:
    """One ``batch`` block as one ``mf_batches`` row. **The single projection.**

    Raises:
        Refusal: ``batch_block_malformed`` when the line is not a batch block. Refused
            rather than skipped: a skipped line makes a rebuilt index disagree with the log
            it was built from, which is the divergence ADR-004's equivalence test exists to
            detect.
    """
    if block.get("block") != "batch" or not isinstance(block.get("batch"), Mapping):
        raise refuse(
            "batch_block_malformed",
            "This run-log line is not a 'batch' block carrying a batch record. The "
            "discriminator is in the line because reindex classifies each line it reads.",
            block=block.get("block"),
        )
    record: Mapping[str, Any] = block["batch"]
    verdicts: Mapping[str, Any] = record.get("verdicts") or {}
    health: Mapping[str, Any] = record.get("selection_health") or {}
    prompt_ids = list(record.get("prompt_ids") or [])
    return {
        "run_id": str(block["run_id"]),
        "batch_sha256": str(record["batch_sha256"]),
        "step": block.get("step"),
        "source": str(record["source"]),
        "resolved_model": record.get("resolved_model"),
        "part_refs": list(record.get("parts") or []),
        "prompt_ids": prompt_ids,
        "n_prompts": len(prompt_ids),
        "n_accept": int(verdicts.get("accept", 0)),
        "n_reject": int(verdicts.get("reject", 0)),
        "n_undecided": int(verdicts.get("undecided", 0)),
        "candidates_generated": int(health.get("candidates_generated", 0)),
        "candidates_admitted": int(health.get("candidates_admitted", 0)),
        "acceptance_rate": float(health.get("acceptance_rate", 0.0)),
        "exhausted": bool(health.get("exhausted", False)),
        "selector_config_sha256": record.get("selector_config_sha256"),
        "grader_config_sha256": record.get("grader_config_sha256"),
        "created_at": _timestamp(block.get("ts")),
    }


def _timestamp(value: Any) -> datetime:
    """The block's ``ts`` as an aware datetime. Refuses an unparseable one.

    Refused rather than replaced with "now": a row whose timestamp is the moment of the
    *rebuild* would make a reindexed index disagree with the live one on every ordering
    query, and newest-first is the only hot read this table has.
    """
    if isinstance(value, datetime):
        return value if value.tzinfo else value.replace(tzinfo=UTC)
    if isinstance(value, str):
        try:
            parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
        except ValueError as exc:
            raise refuse(
                "batch_block_malformed",
                f"A batch block's ts is {value!r}, which is not an ISO-8601 timestamp.",
                ts=value,
            ) from exc
        return parsed if parsed.tzinfo else parsed.replace(tzinfo=UTC)
    raise refuse(
        "batch_block_malformed",
        "A batch block carries no ts, so its row could not be ordered. The schema requires "
        "one on every line.",
    )


def upsert(session: Any, block: Mapping[str, Any]) -> dict[str, Any]:
    """Write (or replace) the row for one batch block, in the caller's transaction.

    Replace rather than insert-only, because the rebuild reruns over the same log: an
    insert-only path makes a second rebuild fail, which would make ADR-016 #4's equivalence
    test pass once and never again.
    """
    row = row_from_block(block)
    session.execute(
        sa.delete(BATCHES)
        .where(BATCHES.c.run_id == row["run_id"])
        .where(BATCHES.c.batch_sha256 == row["batch_sha256"])
    )
    session.execute(sa.insert(BATCHES).values(**row))
    return row


def rebuild_batch_index(session: Any, *, log_root: Path, run_id: str) -> int:
    """Rebuild ``mf_batches`` for one run from its run log. Returns the row count.

    Delete-then-insert for the whole run, so a rebuild is idempotent and a row the log no
    longer contains disappears - an upsert-only rebuild would leave a batch that was rolled
    back visible forever.

    Raises:
        Refusal: ``run_log_absent`` when there is no log. An empty index would assert that
            the run drew no batches, and the log is the only thing that could say otherwise.
    """
    path = run_log.run_log_path(log_root, run_id)
    blocks = list(run_log.read_blocks(path, block="batch"))
    session.execute(sa.delete(BATCHES).where(BATCHES.c.run_id == run_id))
    for block in blocks:
        session.execute(sa.insert(BATCHES).values(**row_from_block(block)))
    if not blocks and not path.is_file():
        raise refuse(
            "run_log_absent",
            f"No run log at {path}, so {run_id}'s batches cannot be rebuilt. The log is the "
            f"authoritative record: without it the batch provenance is gone, and "
            f"an empty index would assert the run drew no batches.",
            run_id=run_id,
            path=str(path),
        )
    return len(blocks)


def list_batches(
    session: Any, *, run_id: str, limit: int, before: datetime | None = None
) -> list[dict[str, Any]]:
    """A run's batch rows, **newest first**, keyset-paginated on ``created_at``.

    Keyset rather than offset: a run writes rows while the operator is paging, and an offset
    would silently skip or repeat batches as the tail grows.
    """
    query = sa.select(BATCHES).where(BATCHES.c.run_id == run_id)
    if before is not None:
        query = query.where(BATCHES.c.created_at < before)
    rows = session.execute(
        query.order_by(BATCHES.c.created_at.desc(), BATCHES.c.batch_sha256.desc()).limit(limit)
    ).all()
    return [dict(row._mapping) for row in rows]


def text_free_columns() -> tuple[str, ...]:
    """Every column name, for the assertion that none of them could hold a text field.

    A helper rather than an inline list in the test, because the **test must read the table
    rather than a copy of it**: a list in the test would keep passing after a column was
    added, which is the construction-green failure this project keeps finding.
    """
    return tuple(column.name for column in BATCHES.columns)


def ddl(dialect: str = "postgresql") -> str:
    """The ``CREATE TABLE`` this feature owes the coordinator's migration.

    Printed rather than applied: the Alembic revision graph is shared state and three
    features are in flight, so a revision created here would give everyone multiple heads
    at the worst moment - on someone else's clean clone. This function is what the report
    quotes, and it is generated **from the table** so the quoted DDL cannot drift from the
    declaration the tests run against.
    """
    from sqlalchemy.schema import CreateIndex, CreateTable

    engine = sa.create_engine(f"{dialect}://", strategy="mock", executor=lambda *a, **k: None)
    statements = [str(CreateTable(BATCHES).compile(engine)).strip()]
    statements += [str(CreateIndex(index).compile(engine)).strip() for index in BATCHES.indexes]
    return ";\n".join(statements) + ";"


# **No ``create_all`` helper lives in this package, deliberately.** F10's structural test
# refuses one anywhere under ``backend/src`` and its reason is right: a table built from the
# declaration tests the models against themselves, so a drifted migration would still pass.
# The migration is the authority and it is the coordinator's; a test that needs this table
# reaches for ``metadata`` above, in the test, where the choice is visible.
