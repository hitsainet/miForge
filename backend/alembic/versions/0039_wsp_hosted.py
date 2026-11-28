"""WSP on hosted models (feature 020, phase P1; FR-020.1, FR-020.4, FR-020.5, FR-020.20; TDD §4;
F20-D8, D12, D13, D18, D22).

Revision ID: 0039_wsp_hosted
Revises: 0038_wsp_pool
Create Date: 2026-11-25

1. **Additive columns** (column-level CHECKs on the ``ADD COLUMN``, the ``0032`` pattern):
   * ``mf_wsp_endpoints.history_mode`` - ``paging`` (the default) or ``append_only``;
   * ``mf_wsp_usage.cache_read_tokens`` and ``cache_write_tokens`` - null on every earlier call
     ("not recorded").
2. **The thinking CHECK widens** to the three effort levels (F20-D22): PostgreSQL drops and
   re-creates it; SQLite rebuilds ``mf_wsp_endpoints`` inside ``batch_alter_table`` with
   ``copy_from`` describing the table as it stands after step 1 (:func:`_mf_wsp_endpoints`).
   SQLAlchemy does not reflect SQLite CHECKs, so a reflected rebuild would drop the other sixteen
   in silence; ``test_migration_0039.py`` asserts every one survives on both engines.
3. **The data step** (F20-D8, F20-D13), on the stored locality: every hosted Anthropic row becomes
   ``append_only``, and one reading thinking ``off`` reads ``default`` (on such a row ``off`` sent
   nothing and ``default`` sends nothing either; what changes is the NO_THINKING sentence).
   Windows and prices are left as they are: a hosted row without a window, or a hosted Anthropic
   row without both cache prices, refuses runs by name (``wsp.endpoints.run_refusal``).

**The downgrade refuses while any endpoint's thinking is an effort**, naming the count: narrowing
the CHECK under such a row would fail half-way on PostgreSQL and drop the row on a SQLite rebuild.
The ``off`` to ``default`` change is not reverted: ``default`` is valid under the old CHECK.

**A referenced table (SQLite):** ``mf_wsp_runs.endpoint_id`` references this table, and the
suite's connections enforce foreign keys, so the rebuild's ``DROP TABLE`` would delete through
the references and fail. :func:`_rebuild` follows SQLite's own procedure for that: foreign keys
off around the rebuild, ``PRAGMA foreign_key_check`` after, the rebuild committed and foreign
keys on again, each pragma read back.
"""

from __future__ import annotations

from typing import Any

import sqlalchemy as sa
from alembic import op

revision: str = "0039_wsp_hosted"
down_revision: str | None = "0038_wsp_pool"
branch_labels: str | None = None
depends_on: str | None = None

OLD_THINKING = "thinking IN ('off', 'on', 'default')"
NEW_THINKING = "thinking IN ('off', 'on', 'default', 'low', 'medium', 'high')"
THINKING_CHECK = "ck_mf_wsp_endpoints_thinking_known"
HISTORY_CHECK = "ck_mf_wsp_endpoints_history_mode_known"
EFFORTS = ("low", "medium", "high")

#: ``(table, column, DDL after the column name)``, in upgrade order. The CHECK names are what
#: ``db.base.NAMING_CONVENTION`` renders for the models' ``CheckConstraint`` names.
COLUMNS: tuple[tuple[str, str, str], ...] = (
    (
        "mf_wsp_endpoints",
        "history_mode",
        "VARCHAR(16) DEFAULT 'paging' NOT NULL "
        "CONSTRAINT ck_mf_wsp_endpoints_history_mode_known "
        "CHECK (history_mode IN ('paging', 'append_only'))",
    ),
    (
        "mf_wsp_usage",
        "cache_read_tokens",
        "INTEGER CONSTRAINT ck_mf_wsp_usage_cache_read_non_negative "
        "CHECK (cache_read_tokens >= 0)",
    ),
    (
        "mf_wsp_usage",
        "cache_write_tokens",
        "INTEGER CONSTRAINT ck_mf_wsp_usage_cache_write_non_negative "
        "CHECK (cache_write_tokens >= 0)",
    ),
)

#: ``mf_wsp_endpoints``' CHECKs as ``0038`` left them (generated from the model at ``0038`` and
#: checked against a SQLite database migrated to ``0038``, FTASKS 3.1), written out, never
#: reflected. ``0028``'s five are table constraints; every later one was added with its column
#: (``ADD COLUMN ... CHECK``) and is a COLUMN constraint, which the rebuild keeps as one: SQLite's
#: own ``DROP COLUMN`` (every earlier downgrade's) refuses a column a table constraint names.
TABLE_CHECKS: tuple[tuple[str, str], ...] = (
    ("ck_mf_wsp_endpoints_kind_known", "kind IN ('openai_compatible', 'anthropic')"),
    (
        "ck_mf_wsp_endpoints_external_has_prices",
        "locality <> 'external' OR "
        "(price_in_per_mtok IS NOT NULL AND price_out_per_mtok IS NOT NULL)",
    ),
    ("ck_mf_wsp_endpoints_locality_known", "locality IN ('local', 'external')"),
    (
        "ck_mf_wsp_endpoints_output_cap_field_known",
        "output_cap_field IN ('max_tokens', 'max_completion_tokens')",
    ),
    ("ck_mf_wsp_endpoints_serving_known", "serving IN ('generic', 'miserver')"),
)
#: Column -> its own CHECK (name, text); ``thinking``'s is :func:`_mf_wsp_endpoints`' argument.
COLUMN_CHECKS: dict[str, tuple[str, str]] = {
    "action_protocol": (
        "ck_mf_wsp_endpoints_action_protocol_known",
        "action_protocol IN ('tool_calls', 'structured')",
    ),
    "context_window_tokens": (
        "ck_mf_wsp_endpoints_context_window_positive",
        "context_window_tokens > 0",
    ),
    "thinking_tokens": ("ck_mf_wsp_endpoints_thinking_tokens_positive", "thinking_tokens > 0"),
    "output_cap": ("ck_mf_wsp_endpoints_output_cap_known", "output_cap IN ('on', 'off')"),
    "paging": ("ck_mf_wsp_endpoints_paging_known", "paging IN ('on', 'off')"),
    "output_ceiling_tokens": (
        "ck_mf_wsp_endpoints_output_ceiling_at_least_256",
        "output_ceiling_tokens >= 256",
    ),
    "registration_window_tokens": (
        "ck_mf_wsp_endpoints_registration_window_positive",
        "registration_window_tokens > 0",
    ),
    "protocol_test_result": (
        "ck_mf_wsp_endpoints_protocol_test_result_known",
        "protocol_test_result IS NULL OR protocol_test_result IN ('passed', 'failed')",
    ),
    "protocol_tested_action_protocol": (
        "ck_mf_wsp_endpoints_protocol_tested_action_protocol_known",
        "protocol_tested_action_protocol IS NULL OR "
        "protocol_tested_action_protocol IN ('tool_calls', 'structured')",
    ),
    "history_mode": (HISTORY_CHECK, "history_mode IN ('paging', 'append_only')"),
}


def _mf_wsp_endpoints(thinking: str, *, history_mode: bool = True) -> sa.Table:
    """``mf_wsp_endpoints`` as the rebuild leaves it: every column of ``0028`` to ``0038`` (plus
    ``history_mode`` unless ``history_mode=False``, the downgrade's), with the thinking CHECK
    ``thinking`` and every other CHECK where ``0038`` had it. Batch's ``copy_from``, recreated as
    described: the table has no index besides its primary key and no foreign key of its own."""
    checks = {**COLUMN_CHECKS, "thinking": (THINKING_CHECK, thinking)}

    def column(name: str, type_: Any, *, nullable: bool, default: str | None = None) -> sa.Column:
        own = [sa.CheckConstraint(checks[name][1], name=checks[name][0])] if name in checks else []
        return sa.Column(name, type_, *own, nullable=nullable, server_default=default)

    columns = [
        column("id", sa.String(40), nullable=False),
        column("kind", sa.String(24), nullable=False),
        column("base_url", sa.String(512), nullable=False),
        column("model", sa.String(256), nullable=False),
        column("locality", sa.String(16), nullable=False),
        column("price_in_per_mtok", sa.Numeric(12, 6), nullable=True),
        column("price_out_per_mtok", sa.Numeric(12, 6), nullable=True),
        column("price_cache_write_per_mtok", sa.Numeric(12, 6), nullable=True),
        column("price_cache_read_per_mtok", sa.Numeric(12, 6), nullable=True),
        column("output_cap_field", sa.String(32), nullable=False),
        column("serving", sa.String(16), nullable=False),
        column("key_ciphertext", sa.LargeBinary(), nullable=True),
        column("key_fingerprint", sa.String(8), nullable=True),
        column("created_at", sa.DateTime(timezone=True), nullable=False),
        column("created_by", sa.String(64), nullable=False),
        column("removed_at", sa.DateTime(timezone=True), nullable=True),
        column("action_protocol", sa.String(16), nullable=False, default="tool_calls"),
        column("context_window_tokens", sa.Integer(), nullable=True),
        column("thinking", sa.String(8), nullable=False, default="off"),
        column("thinking_tokens", sa.Integer(), nullable=True),
        column("output_cap", sa.String(4), nullable=False, default="on"),
        column("paging", sa.String(4), nullable=False, default="on"),
        column("output_ceiling_tokens", sa.Integer(), nullable=True),
        column("registration_window_tokens", sa.Integer(), nullable=True),
        column("registration_window_read_at", sa.DateTime(timezone=True), nullable=True),
        column("protocol_test_result", sa.String(8), nullable=True),
        column("protocol_test_run_id", sa.String(16), nullable=True),
        column("protocol_tested_base_url", sa.String(512), nullable=True),
        column("protocol_tested_model", sa.String(256), nullable=True),
        column("protocol_tested_action_protocol", sa.String(16), nullable=True),
        column("protocol_test_at", sa.DateTime(timezone=True), nullable=True),
    ]
    if history_mode:
        columns.append(column("history_mode", sa.String(16), nullable=False, default="paging"))
    return sa.Table(
        "mf_wsp_endpoints",
        sa.MetaData(),
        *columns,
        sa.PrimaryKeyConstraint("id", name="pk_mf_wsp_endpoints"),
        *(sa.CheckConstraint(text, name=name) for name, text in TABLE_CHECKS),
    )


def _pragma(name: str) -> Any:
    return op.get_bind().exec_driver_sql(f"PRAGMA {name}").scalar()


def _rebuild(table: sa.Table) -> None:
    """SQLite: recreate ``mf_wsp_endpoints`` as ``table`` describes it, copying every row (the
    columns ``table`` names), by SQLite's own procedure for a table other tables reference
    (``mf_wsp_runs.endpoint_id``): foreign keys off around the rebuild, then
    ``foreign_key_check`` and foreign keys on again. With foreign keys on, the rebuild's ``DROP
    TABLE`` deletes through every reference and fails. The pragma is a no-op inside a
    transaction, so each is read back, and the rebuild refuses rather than run with one ignored.
    A failed rebuild raises before the commit and leaves the connection to be discarded."""
    enforced = _pragma("foreign_keys")
    op.execute("PRAGMA foreign_keys = OFF")
    if _pragma("foreign_keys"):
        raise RuntimeError(
            "0039_wsp_hosted cannot rebuild mf_wsp_endpoints on SQLite inside a transaction that "
            "enforces foreign keys: PRAGMA foreign_keys = OFF did not take. Run the migration on "
            "its own connection."
        )
    with op.batch_alter_table("mf_wsp_endpoints", copy_from=table, recreate="always"):
        pass
    broken = op.get_bind().exec_driver_sql("PRAGMA foreign_key_check").fetchall()
    if broken:
        raise RuntimeError(f"0039_wsp_hosted left broken foreign keys: {broken[:5]}")
    if enforced:
        # The rebuild's copy opened a transaction, inside which the pragma is a no-op (measured):
        # commit the finished rebuild, then enforce foreign keys again, and read it back.
        dbapi = op.get_bind().connection.dbapi_connection
        if dbapi is not None and dbapi.in_transaction:
            dbapi.commit()
        op.execute("PRAGMA foreign_keys = ON")
        if not _pragma("foreign_keys"):
            raise RuntimeError("0039_wsp_hosted could not enforce foreign keys again on SQLite")


def _sqlite() -> bool:
    return bool(op.get_bind().dialect.name == "sqlite")


def _set_thinking(*, to_check: str) -> None:
    if _sqlite():
        _rebuild(_mf_wsp_endpoints(to_check))
        return
    op.drop_constraint(op.f(THINKING_CHECK), "mf_wsp_endpoints", type_="check")
    op.create_check_constraint(op.f(THINKING_CHECK), "mf_wsp_endpoints", to_check)


def upgrade() -> None:
    # Raw DDL, because Alembic's add_column does not render a column-level CHECK.
    for table, column, ddl in COLUMNS:
        op.execute(f"ALTER TABLE {table} ADD COLUMN {column} {ddl}")
    _set_thinking(to_check=NEW_THINKING)
    hosted_anthropic = "kind = 'anthropic' AND locality = 'external'"
    op.execute(f"UPDATE mf_wsp_endpoints SET history_mode = 'append_only' WHERE {hosted_anthropic}")
    op.execute(
        f"UPDATE mf_wsp_endpoints SET thinking = 'default' WHERE {hosted_anthropic} "
        "AND thinking = 'off'"
    )


def downgrade() -> None:
    in_use = (
        op.get_bind()
        .execute(
            sa.text(
                "SELECT COUNT(*) FROM mf_wsp_endpoints WHERE thinking IN ('low', 'medium', 'high')"
            )
        )
        .scalar_one()
    )
    if in_use:
        raise RuntimeError(
            f"Refusing to downgrade 0039_wsp_hosted: {in_use} endpoint(s) use a thinking effort; "
            "set them to off, on or default first. Narrowing the thinking CHECK would fail on "
            "PostgreSQL and drop them on a SQLite rebuild."
        )
    if _sqlite():
        # One rebuild: the thinking CHECK narrowed and ``history_mode`` (with its CHECK) gone.
        _rebuild(_mf_wsp_endpoints(OLD_THINKING, history_mode=False))
    else:
        _set_thinking(to_check=OLD_THINKING)
        op.drop_column("mf_wsp_endpoints", "history_mode")
    for table, column, _ddl in reversed(COLUMNS):
        if table != "mf_wsp_endpoints":
            op.drop_column(table, column)
