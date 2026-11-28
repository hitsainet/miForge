"""WSP submitted check: the editor's claimed figures beside the measured ones (018 FU-2, operator
decision 2026-11-25; ``0xcc/reviews/run_contract_r2_2026-11-25.md``).

Revision ID: 0040_wsp_submitted_check
Revises: 0039_wsp_hosted
Create Date: 2026-11-25

* ``mf_wsp_runs.submitted_check`` - the ``claims`` object of the run's ``submitted_measured``
  transcript line, as the intake stored it: whether ``submit`` stated figures, whether they were
  compared with the measurement of the submitted weights, whether any differs, and each figure.
  ``run_document`` serves it as ``submitted_check`` (``miforge-wsp-run-v1.json``) when it is set.
  Null on every run taken in before this migration: nothing is backfilled, so every stored
  verdict's ``run_sha256`` still reproduces from its document.

Additive and nullable: neither engine rebuilds a table.

**The downgrade refuses while any run holds a check**, naming the count: the column is part of
those runs' documents, so dropping it would change every such document after its verdict, and
miWorkbench's import would refuse them ``run_sha256_mismatch``.
"""

from __future__ import annotations

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

revision: str = "0040_wsp_submitted_check"
down_revision: str | None = "0039_wsp_hosted"
branch_labels: str | None = None
depends_on: str | None = None

#: What ``models.approval.NullableJson`` renders: JSON on SQLite, JSONB on PostgreSQL.
JSON_TYPE = sa.JSON(none_as_null=True).with_variant(
    postgresql.JSONB(none_as_null=True), "postgresql"
)


def upgrade() -> None:
    op.add_column("mf_wsp_runs", sa.Column("submitted_check", JSON_TYPE, nullable=True))


def downgrade() -> None:
    held = (
        op.get_bind()
        .execute(sa.text("SELECT COUNT(*) FROM mf_wsp_runs WHERE submitted_check IS NOT NULL"))
        .scalar_one()
    )
    if held:
        raise RuntimeError(
            f"Refusing to downgrade 0040_wsp_submitted_check: {held} run(s) hold a submitted "
            "check, which is part of their run documents; dropping it would change each document "
            "after its verdict (run_sha256)."
        )
    op.drop_column("mf_wsp_runs", "submitted_check")
