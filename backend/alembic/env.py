"""Alembic environment  (foundation task 4.2; ADR-003, ADR-020).

**The URL comes from settings, never from ``alembic.ini``.** One source of truth for the
database URL, and no credential in a tracked file. ``-x url=...`` overrides it for a
one-off run against a scratch database.

**Every model module is imported, not just the base.** ``target_metadata`` is only as
complete as the classes that have been imported, so a model nobody imports is a table
``--autogenerate`` would propose *dropping*. Importing the package is what prevents that.
"""

from __future__ import annotations

from logging.config import fileConfig

from alembic import context
from sqlalchemy import pool

import models  # noqa: F401  - registers every table on Base.metadata; see the docstring
from core.config import get_settings
from db.base import Base
from db.session import make_engine

config = context.config
# Guarded, and the guard matters: `fileConfig` REPLACES the root logger's handlers, so a
# migration run programmatically from a test fixture would tear out pytest's log-capture
# handler and the suite could no longer assert on a log line - including task 4.5's
# divergence line, which is the one place a log IS the behaviour. A caller that owns its
# own logging sets `cfg.attributes["configure_logger"] = False`.
if config.config_file_name is not None and config.attributes.get("configure_logger", True):
    fileConfig(config.config_file_name, disable_existing_loggers=False)

target_metadata = Base.metadata


def _url() -> str:
    override = context.get_x_argument(as_dictionary=True).get("url")
    return override or get_settings().database_url


def run_migrations_offline() -> None:
    context.configure(
        url=_url(),
        target_metadata=target_metadata,
        literal_binds=True,
        dialect_opts={"paramstyle": "named"},
        # Needed for SQLite, which cannot ALTER a column: without it a later migration
        # that changes a column is simply impossible on the engine the unit tests use.
        render_as_batch=True,
    )
    with context.begin_transaction():
        context.run_migrations()


def run_migrations_online() -> None:
    engine = make_engine(_url(), poolclass=pool.NullPool)
    with engine.connect() as connection:
        context.configure(
            connection=connection,
            target_metadata=target_metadata,
            render_as_batch=True,
            # Compare types and server defaults so a drift between the models and the
            # migrations is reported rather than invisible - the drift test in
            # `test_db_migration.py` depends on this being on.
            compare_type=True,
            compare_server_default=True,
        )
        with context.begin_transaction():
            context.run_migrations()


if context.is_offline_mode():
    run_migrations_offline()
else:
    run_migrations_online()
