"""Test support: a MIGRATED database, and manifests written by the real writer.

Two decisions here are the ones worth reading.

**The tables come from the Alembic migration, never from ``create_all``.** ``create_all``
builds them from the models, so a migration that has drifted from the models would still
pass every test - the models would simply be tested against themselves. Running the
migration means the suite exercises the artefact that production applies.

**Manifests are written by :func:`core.canonical.write_canonical` and read back off the
disk**, and every fixture manifest is **validated against the vendored schema** before a
test uses it. ADR-016: *fixtures whose fields agree by construction are how a suite stays
green over a real bug.* A hand-built dict passed straight to a row builder would agree
with that builder by construction and would never notice that the schema had moved.

**The engine.** SQLite by default so the suite runs with no service (ADR-023's spirit:
everything bar a training step must run on a laptop). Set
``MIFORGE_TEST_DATABASE_URL`` to a PostgreSQL URL - which CI's ``backend-tests.yml`` will
do - to run the same tests against the engine production uses. The two are not equivalent
and that is a known limitation, recorded rather than hidden: a PostgreSQL-only type or a
``JSONB`` operator would pass here and fail there. **Two of those have now actually
happened** - ``0013`` typed three columns ``sa.JSON()`` where the models said JSONB, and
the two-database allocation below - so the paragraph is a record of real defects rather
than a caveat.

**One test may ask for two databases**, and on PostgreSQL that used to hand back the same
one twice. See :func:`_configured_url_for`.
"""

from __future__ import annotations

import os
from collections.abc import Iterator
from contextlib import contextmanager
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from alembic import command
from alembic.config import Config
from sqlalchemy import Engine
from sqlalchemy.orm import Session, sessionmaker

from core.canonical import write_canonical
from db.session import make_engine

REPO_ROOT = Path(__file__).resolve().parents[3]
TEST_DB_URL_ENV = "MIFORGE_TEST_DATABASE_URL"


#: ``(test id, requested path) -> database name``. See :func:`_configured_url_for`.
_SLOTS: dict[tuple[str, str], str] = {}


def _current_test() -> str:
    """The running test's id, or ``""`` outside a test.

    ``PYTEST_CURRENT_TEST`` is set by pytest for the duration of each test, which is
    exactly the scope over which two requested paths must not collide.
    """
    return os.environ.get("PYTEST_CURRENT_TEST", "").split(" ")[0]


def _configured_url_for(configured: str, tmp_path: Path) -> str:
    """The configured URL, or a **sibling database** when one test asks for a second.

    **This is the fix for a defect only PostgreSQL could show.** ``database_url`` used to
    return the configured URL for every requested path, so a test that asked for two
    databases - ``migrated_engine(tmp_path / "db-here")`` and ``.../"db-there"``, meaning
    *"a second installation, empty, so the import is about the document that travelled
    rather than rows that happened to be shared"* - got **one**. The second
    ``migrated_engine`` then downgraded it to base and re-upgraded it, wiping the first's
    rows, and the test failed on a primary-key collision. On SQLite the two paths are two
    files, so the whole class was invisible locally and appeared only in CI.

    Each distinct path a single test requests gets its own slot: slot 0 is the configured
    database, slot *n* is ``<name>_altn``, created on demand and reused by later tests.
    Only the handful of tests that want two databases ever create one, so the cost is a
    couple of ``CREATE DATABASE`` statements per session rather than one per test.
    """
    from sqlalchemy import make_url

    key = (_current_test(), str(tmp_path))
    if key in _SLOTS:
        return _SLOTS[key]
    taken = {url for (test, _), url in _SLOTS.items() if test == key[0]}
    if not taken:
        _SLOTS[key] = configured
        return configured

    url = make_url(configured)
    if url.get_backend_name() != "postgresql":
        raise AssertionError(
            f"{TEST_DB_URL_ENV} names a {url.get_backend_name()} database and this test "
            f"asked for a second one. Only PostgreSQL has a supported way to allocate a "
            f"sibling database here; refusing rather than handing back the same database "
            f"under a second name, which is the defect this function exists to fix."
        )
    # `render_as_string(hide_password=False)`, never `str(url)`. SQLAlchemy's `str()` of a
    # URL returns the MASKED form (`postgres:***@...`), which is precisely the trap
    # backend-tests.yml documents when it explains why the CI service sets a real password
    # rather than `trust`: the masked form authenticates fine under trust and fails under
    # password auth. The first draft of this function used `str()` and failed with
    # "password authentication failed", which is the right failure and the reason the CI
    # job is configured the way it is.
    sibling = url.set(database=f"{url.database}_alt{len(taken)}").render_as_string(
        hide_password=False
    )
    _create_database_if_absent(url, f"{url.database}_alt{len(taken)}")
    _SLOTS[key] = sibling
    return sibling


def _create_database_if_absent(template: Any, name: str) -> None:
    """``CREATE DATABASE name``, ignoring "already exists".

    Connects to ``postgres`` rather than to the target, because a database cannot be
    created from inside itself, and in AUTOCOMMIT because ``CREATE DATABASE`` cannot run in
    a transaction.
    """
    from sqlalchemy import create_engine, text

    admin = create_engine(
        template.set(database="postgres").render_as_string(hide_password=False),
        isolation_level="AUTOCOMMIT",
    )
    try:
        with admin.connect() as conn:
            exists = conn.execute(
                text("SELECT 1 FROM pg_database WHERE datname = :n"), {"n": name}
            ).first()
            if not exists:
                conn.execute(text(f'CREATE DATABASE "{name}"'))
    finally:
        admin.dispose()


def database_url(tmp_path: Path) -> str:
    """The URL this run's tests use. See the module docstring."""
    configured = os.environ.get(TEST_DB_URL_ENV)
    if configured:
        return _configured_url_for(configured, tmp_path)
    # mkdir here rather than at each call site: SQLite's "unable to open database file"
    # for a missing parent directory is one of the least informative errors in the suite.
    tmp_path.mkdir(parents=True, exist_ok=True)
    return f"sqlite+pysqlite:///{tmp_path / 'miforge-test.db'}"


def migrated_engine(tmp_path: Path) -> Engine:
    """An engine whose schema was built by ``alembic upgrade head``."""
    url = database_url(tmp_path)
    cfg = Config(str(REPO_ROOT / "alembic.ini"))
    cfg.set_main_option("script_location", str(REPO_ROOT / "backend" / "alembic"))
    # Do not let Alembic reconfigure logging out from under pytest - see env.py.
    cfg.attributes["configure_logger"] = False
    cfg.cmd_opts = _XArgs(url)  # type: ignore[assignment]
    if _already_stamped(url):
        # A reused database would otherwise keep the previous test's rows, and a
        # reindex-equivalence test seeded twice compares a superset with a subset.
        command.downgrade(cfg, "base")
    command.upgrade(cfg, "head")
    return make_engine(url)


class _XArgs:
    """The shape Alembic's ``get_x_argument`` reads, so ``-x url=...`` works in-process."""

    def __init__(self, url: str) -> None:
        self.x = [f"url={url}"]


def _already_stamped(url: str) -> bool:
    """True when this database already carries an Alembic version row.

    Matters only for a shared PostgreSQL named by the environment: a fresh SQLite file per
    ``tmp_path`` is never stamped, but a reused CI database is, and ``upgrade head`` on an
    already-migrated database is a no-op that leaves the previous test's rows behind.
    """
    engine = make_engine(url)
    try:
        with engine.connect() as conn:
            return engine.dialect.has_table(conn, "alembic_version")
    finally:
        engine.dispose()


@contextmanager
def session_for(engine: Engine) -> Iterator[Session]:
    """A session on ``engine``, committed on success."""
    factory = sessionmaker(bind=engine, expire_on_commit=False)
    session = factory()
    try:
        yield session
        session.commit()
    except BaseException:
        session.rollback()
        raise
    finally:
        session.close()


# ====================================================================== manifest fixtures


def _validate(manifest: dict[str, Any], kind: str) -> None:
    """Refuse to hand a test a manifest its own schema rejects.

    This is the line that stops a fixture drifting away from the contract. Without it a
    test asserting over ``domain`` would keep passing after ``domain`` left the schema.
    """
    import jsonschema

    from models.domain_state import ArtifactKind
    from services.reindex import load_schema

    jsonschema.Draft202012Validator(load_schema(ArtifactKind(kind))).validate(manifest)


def checkpoint_manifest(
    *,
    run_id: str = "run-1",
    step: int = 100,
    base_model: str = "LFM2-1.2B",
    parent: str | None = None,
    domain: str | object | None = "support",
    signal_source: str = "corpus",
    test: dict[str, Any] | None = None,
    retention: dict[str, Any] | None = None,
    merged_adapter: dict[str, Any] | None = None,
    created_at: str = "2026-10-14T00:00:00Z",
    **extra: Any,
) -> dict[str, Any]:
    """A conforming checkpoint manifest.

    ``domain`` accepts a string or ``None``; **omitting the key entirely** is done by the
    caller deleting it, deliberately, because a default here could not express "absent"
    without becoming the very collapse task 4.2a is about.
    """
    manifest: dict[str, Any] = {
        "kind": "miforge.checkpoint-manifest",
        "schema_version": "1",
        "name": f"{base_model}:{run_id}:step-{step}",
        "run_id": run_id,
        "step": step,
        "domain": domain,
        "signal_source": signal_source,
        "parent": parent,
        "base_model": base_model,
        "optimizer": {"name": "adamw-8bit", "state": "restored", "config_hash": "a" * 64},
        "weights_sha256": "b" * 64,
        "created_at": created_at,
    }
    if test is not None:
        manifest["test"] = test
    if retention is not None:
        manifest["retention"] = retention
    if merged_adapter is not None:
        manifest["merged_adapter"] = merged_adapter
    manifest.update(extra)
    _validate(manifest, "checkpoint")
    return manifest


def adapter_manifest(
    *,
    run_id: str = "run-2",
    step: int = 50,
    base_model: str = "LFM2-1.2B",
    base_name: str | None = None,
    created_at: str = "2026-10-14T00:00:00Z",
    **extra: Any,
) -> dict[str, Any]:
    """A conforming adapter manifest. Note it carries **no** ``domain`` field at all."""
    manifest: dict[str, Any] = {
        "kind": "miforge.adapter-manifest",
        "schema_version": "1",
        "name": f"{base_model}:{run_id}:adapter-{step}",
        "run_id": run_id,
        "step": step,
        "base": {"name": base_name or f"{base_model}:run-1:step-100", "manifest_sha256": "c" * 64},
        "adapter": {
            "rank": 16,
            "alpha": 32.0,
            "target_modules": ["q_proj", "v_proj"],
            "dropout": 0.05,
            "scaling": "alpha_over_r",
        },
        "weights_sha256": "d" * 64,
        "created_at": created_at,
        "parts": [{"type": "prompt-set", "name": "support-tickets", "version": "3"}],
    }
    manifest.update(extra)
    _validate(manifest, "adapter")
    return manifest


def write_checkpoint(root: Path, manifest: dict[str, Any], *, relative: str | None = None) -> Path:
    """Write ``manifest`` into a checkpoint directory under ``root``, as the real writer does.

    The directory renders the step **zero-padded to six digits** (BRD-01 R-01.24) while
    the manifest's ``name`` renders it unpadded. Both come from the one integer; nothing
    carries the step as a string of its own.
    """
    rel = relative or f"{manifest['run_id']}/step-{int(manifest['step']):06d}"
    directory = root / rel
    write_canonical(directory / "manifest.json", manifest)
    # The HF directory's own files. Not fakes with content that matters - what matters is
    # that `manifest.json` is the only non-standard file beside them.
    (directory / "model.safetensors").write_bytes(b"\0")
    (directory / "config.json").write_text("{}", encoding="utf-8")
    return directory


def write_adapter(root: Path, manifest: dict[str, Any], *, relative: str | None = None) -> Path:
    """Write ``manifest`` into a PEFT adapter directory under ``root``."""
    rel = relative or f"{manifest['run_id']}/adapter-{int(manifest['step']):06d}"
    directory = root / rel
    write_canonical(directory / "adapter_manifest.json", manifest)
    (directory / "adapter_model.safetensors").write_bytes(b"\0")
    # THREE files, per the schema's normative layout. `adapter_config.json` is PEFT's own
    # configuration and `PeftConfig.from_pretrained` reads it: a directory without it is
    # not loadable by standard tooling, which is what R-02.51 promises. The fixture writes
    # the real layout so a test cannot pass against a directory nothing could load.
    (directory / "adapter_config.json").write_text("{}", encoding="utf-8")
    return directory


def score(
    *, set_name: str = "support-test", value: float = 0.81, n: int = 200, **extra: Any
) -> dict[str, Any]:
    """A ``ScoreRecord``. ``provider`` is **not** included by default, on purpose.

    The field is nullable and **not** ``required``, so a payload omitting it validates
    clean and the schema-sync test - which compares schema *files* - cannot see the
    omission either. Only a test can, which is why the default fixture omits it and the
    tests that care pass it explicitly.
    """
    record: dict[str, Any] = {"set": set_name, "score": value, "n": n}
    record.update(extra)
    return record


def utc_now() -> datetime:
    return datetime.now(tz=UTC)
