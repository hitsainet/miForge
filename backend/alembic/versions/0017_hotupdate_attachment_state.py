"""F11's two run-state columns: where the §5.10 attachment lives across a worker restart.

Revision ID: 0017_hotupdate
Revises: 0016_f6_loop
Create Date: 2026-10-14

F11's owed DDL, handed over in its report because the Alembic graph is serialised through one
owner: two revisions off the same parent make ``alembic upgrade head`` fail for everyone on a
clean clone.

## Why these two columns are not optional, and why holding the id in memory was not an option

**``attach`` returns the attachment id exactly once and no read path returns it** (BRD-01
R-01.31). Seven of BRD-01's fifteen calls are holder-only and carry
``X-miLLM-Attachment: <attachment id>`` alongside the bearer token, and the four that matter
most are ``commit-begin``, ``commit-end``, ``save`` and ``detach``. **Losing the id is losing
the ability to commit** — and therefore the ability to finish a step, save a checkpoint, or
detach cleanly.

So keeping it in a request-scoped variable or a worker's memory *"for now"* is **MUT-12
shipped on purpose**: it passes every test, works until the first worker restart, and then
produces a refused commit that nothing can explain. The run dies looking like a crash. F11
therefore calls ``run_state.remember()`` on the attach path — before the rebuild, per
FR-011.5 — and **refuses by name** (``hotupdate_run_columns_missing``, carrying this DDL)
until these columns exist. That refusal is what this revision switches off.

## ``attachment_id`` is secret-handled

It authorises the holder-only calls, so it is a credential in everything but name:

* excluded from every API serialisation — nothing in ``api/`` may put it in a response body;
* **only a digest reference may be logged**, never the value. F11's secret-hygiene test drives
  a full run with a log capture and asserts the value never appears;
* kept out of ``repr(Run)`` by an **explicit ``Run.__repr__``** that omits it — so it stays
  out of a traceback frame and out of a debugger watch. *This read ``repr=False`` on the
  ORM attribute until 2026-10-14, and that mechanism is not there: SQLAlchemy raises
  ``ArgumentError`` for ``repr=False`` on a model that is not a ``MappedAsDataclass``, so
  the attribute was never declared that way. A docstring naming a mechanism the code
  does not have is worse than silence — a reviewer checking the claim finds the field
  undefended and a reviewer trusting it stops looking.*

Note what ``repr=False`` does **not** cover, learned the expensive way in foundation 2.5:
Pydantic's validation errors echo rejected input, which is why ``Settings`` carries
``hide_input_in_errors=True``. This column is not a settings field, so that path does not
apply — but the lesson generalises, and **the error path is an output**.

## Additive, and no rebuild

Two nullable columns, no constraint, no index, no backfill. SQLite can ``ALTER TABLE ADD
COLUMN`` for both, so unlike ``0014`` and ``0016`` this revision needs **no table rebuild** —
which is the whole reason it is cheap and separate. Null is the correct and permanent value
for every acquisition mode but §5.10, and that is an assertion rather than an omission: a run
that did not attach has no attachment.

**Reindex is unaffected.** This is *run state*, not artifact state — ADR-004's rebuildable
index covers what a manifest or a run log can reproduce, and an attachment id is neither. It
exists only for the lifetime of one attachment and is deliberately unrecoverable afterwards.
"""

from __future__ import annotations

import sqlalchemy as sa
from alembic import op

revision: str = "0017_hotupdate"
down_revision: str | None = "0016_f6_loop"
branch_labels: str | None = None
depends_on: str | None = None


def upgrade() -> None:
    # The serving process this run attached to. Recorded beside the id because an id without
    # the host it belongs to cannot be used: the holder-only calls go to a specific base URL,
    # and a worker that restarts has neither unless both are durable.
    op.add_column("mf_runs", sa.Column("hotupdate_base_url", sa.String(length=255), nullable=True))
    # SECRET-HANDLED. See the module docstring: never serialised, only ever logged as a
    # digest, and `repr=False` on the ORM attribute.
    op.add_column("mf_runs", sa.Column("attachment_id", sa.String(length=128), nullable=True))


def downgrade() -> None:
    # Dropping these loses the live attachment of any §5.10 run in flight, which is
    # unrecoverable - `attach` returns the id once. That is acceptable only because §5.10 is
    # the optional mode and a downgrade is a deliberate act; a run mid-attachment must be
    # stopped first, and it will refuse its next commit rather than do something silent.
    op.drop_column("mf_runs", "attachment_id")
    op.drop_column("mf_runs", "hotupdate_base_url")
