# Origin: miStudio (Onegaishimas/miStudio)
#   backend/tests/unit/test_migration_lock_safety.py @ 67b9ab1f
# Mode: adapt (docs/REUSE.md). The incident replay is the same; the migration replayed is this
# repository's own 0022 (ALTER TABLE mf_runs ADD COLUMN archived_at); the budget is the
# Deployment's progress deadline (miForge migrates in an init container, which no probe kills);
# and SQLite gets its own test, because the guards must not reach it.
"""A migration blocked on a lock fails fast, names its blocker, and leaves nothing queued.

miStudio's INCIDENT of 2026-10-21 (05:05-05:16 ET): a bare ``alembic upgrade`` reached an
``ALTER TABLE … ADD COLUMN`` while a hung, two-day-old COPY held ACCESS SHARE on that table. With
no ``lock_timeout`` the ALTER waited forever, and each killed attempt's ALTER stayed queued in
Postgres behind the COPY, in front of every later reader of the table. miForge's ``migrate`` init
container ran the same bare command, so the same lock would have held its pod in ``Init``
indefinitely, every container down, with nothing in the log.

The PostgreSQL tests replay that against a REAL PostgreSQL 15 (``MIFORGE_TEST_DATABASE_URL``; CI's
``postgres`` job; a scratch database per test, dropped WITH (FORCE) afterwards), through the real
``backend/alembic/env.py`` and the real ``python -m db.migrate upgrade`` the init container runs.
The migration is this repository's 0022, which opens with ``ALTER TABLE mf_runs ADD COLUMN
archived_at`` (ACCESS EXCLUSIVE); the blocker is a second session holding ACCESS SHARE on
``mf_runs`` in an open transaction, exactly the COPY's lock. On the SQLite job they skip, naming
the job that runs them; the SQLite and budget tests run on both.

MUTATION CONTROLS (2026-10-21; each alone, restored from saved bytes, sha256-checked) — all red;
the record is 0xcc/reviews/migrate_safely_2026-10-21.md:
  M1 lock_timeout set to "0" (disabled)            M6 the deadline cancel removed from the watch
  M2 client_connection_check_interval not set      M7 MIGRATION_DEADLINE 600s on the init container
  M3 env.py installs no guards on PostgreSQL       M8 statement_timeout not set
  M4 a lock timeout is not retryable               M9 idle_in_transaction_session_timeout not set
  M5 the watch never reads blockers                S1 env.py applies the guards on SQLite too
"""

from __future__ import annotations

import os
import signal
import subprocess
import sys
import time
import uuid
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest
from sqlalchemy import Engine, create_engine, make_url, text
from sqlalchemy.exc import DBAPIError

from db import migration_safety as ms
from tests.support.db import TEST_DB_URL_ENV

yaml = pytest.importorskip("yaml")

REPO_ROOT = Path(__file__).resolve().parents[3]
GITOPS = REPO_ROOT / "gitops" / "pilot"
#: The revision before the replayed migration, and the replayed migration.
BEFORE = "0021_approvals_and_settings"
BLOCKED_TABLE = "mf_runs"
BLOCKED_COLUMN = "archived_at"
POSTGRES_ONLY = (
    f"replays a PostgreSQL lock queue; needs {TEST_DB_URL_ENV} naming PostgreSQL "
    f"(CI's `postgres` job runs it)"
)


def _server_url() -> Any:
    configured = os.environ.get(TEST_DB_URL_ENV, "")
    if not configured or make_url(configured).get_backend_name() != "postgresql":
        pytest.skip(POSTGRES_ONLY)
    return make_url(configured)


def _autocommit(url: Any) -> Engine:
    return create_engine(
        url.render_as_string(hide_password=False), isolation_level="AUTOCOMMIT", pool_size=1
    )


@pytest.fixture
def scratch_db() -> Iterator[Any]:
    server = _server_url()
    name = f"mig_lock_{uuid.uuid4().hex[:12]}"
    admin = _autocommit(server)
    try:
        with admin.connect() as conn:
            conn.execute(text(f'CREATE DATABASE "{name}"'))
        yield server.set(database=name)
    finally:
        with admin.connect() as conn:
            conn.execute(text(f'DROP DATABASE IF EXISTS "{name}" WITH (FORCE)'))
        admin.dispose()


def _env(url: Any, **extra: object) -> dict[str, str]:
    env = dict(os.environ)
    env["DATABASE_URL"] = url.render_as_string(hide_password=False)
    env["PYTHONUNBUFFERED"] = "1"
    for name in (*ms.DEFAULTS, "MIGRATION_DEADLINE_AT", "ALEMBIC_CONFIG"):
        env.pop(name, None)
    env.update({k: str(v) for k, v in extra.items()})
    return env


def _alembic(url: Any, *args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, "-m", "alembic", *args],
        cwd=REPO_ROOT,
        env=_env(url),
        capture_output=True,
        text=True,
        timeout=240,
    )


def _migrate(
    url: Any, timeout: float, **extra: object
) -> tuple[subprocess.CompletedProcess[str], float]:
    started = time.monotonic()
    result = subprocess.run(
        [sys.executable, "-m", "db.migrate", "upgrade"],
        cwd=REPO_ROOT,
        env=_env(url, **extra),
        capture_output=True,
        text=True,
        timeout=timeout,
    )
    return result, time.monotonic() - started


def _head() -> str:
    from alembic.config import Config
    from alembic.script import ScriptDirectory

    cfg = Config(str(REPO_ROOT / "alembic.ini"))
    cfg.set_main_option("script_location", str(REPO_ROOT / "backend" / "alembic"))
    (head,) = ScriptDirectory.from_config(cfg).get_heads()
    return str(head)


def _version(observer: Any) -> set[str]:
    return {
        str(v) for v in observer.execute(text("SELECT version_num FROM alembic_version")).scalars()
    }


def _migration_sessions(observer: Any, dbname: str) -> list[tuple[Any, ...]]:
    rows = observer.execute(
        text(
            "SELECT pid, state, wait_event_type, query FROM pg_stat_activity "
            "WHERE datname = :d AND application_name = :a"
        ),
        {"d": dbname, "a": ms.APPLICATION_NAME},
    ).all()
    return [tuple(r) for r in rows]


@pytest.fixture
def at_the_blocked_migration(scratch_db: Any) -> Iterator[tuple[Any, Any, int, Any]]:
    """A scratch database one revision before 0022's ALTER, and a session holding ACCESS SHARE on
    the table that ALTER needs, as miStudio's hung COPY did."""
    result = _alembic(scratch_db, "upgrade", BEFORE)
    assert result.returncode == 0, result.stdout + result.stderr
    blocker_engine = create_engine(scratch_db.render_as_string(hide_password=False))
    observer_engine = _autocommit(scratch_db)
    blocker = blocker_engine.connect()
    blocker.execute(text(f"SELECT count(*) FROM {BLOCKED_TABLE}"))  # opens the transaction
    pid = int(blocker.execute(text("SELECT pg_backend_pid()")).scalar_one())
    observer = observer_engine.connect()
    try:
        yield scratch_db, blocker, pid, observer
    finally:
        try:
            blocker.rollback()
            blocker.close()
            observer.close()
        finally:
            blocker_engine.dispose()
            observer_engine.dispose()


class TestTheIncidentReplayed:
    def test_a_blocked_migration_gives_up_NAMES_THE_BLOCKER_and_then_succeeds(
        self, at_the_blocked_migration: tuple[Any, Any, int, Any]
    ) -> None:
        url, blocker, blocker_pid, observer = at_the_blocked_migration
        # Budget: 2 attempts x 2 s lock_timeout + 1 s backoff. Without lock_timeout this run
        # would wait until the 150 s statement_timeout; the 60 s subprocess timeout turns that
        # into a failure rather than a hung suite.
        result, elapsed = _migrate(
            url,
            timeout=60,
            MIGRATION_LOCK_TIMEOUT="2s",
            MIGRATION_ATTEMPTS=2,
            MIGRATION_RETRY_BACKOFF="1s",
        )
        out = result.stdout + result.stderr
        assert result.returncode == 3, out
        assert elapsed < 30, f"took {elapsed:.1f}s for a ~5s budget"
        assert out.count("gave up waiting for a lock") >= 2, "the lock timeout was not retried"
        assert f"pid={blocker_pid} " in out, f"the blocker (pid {blocker_pid}) is not named:\n{out}"
        assert "WAITING FOR A LOCK" in out, "the blocker was not logged during the wait"
        assert "pg_terminate_backend" in out and "GIVING UP" in out
        # The values the SERVER accepted for the migration session (set_config's own result).
        for applied in (
            "lock_timeout=2s",
            "statement_timeout=150s",
            "idle_in_transaction_session_timeout=5min",
            "client_connection_check_interval=2s",
        ):
            assert applied in out, f"{applied} was not set on the migration session:\n{out}"
        assert _version(observer) == {BEFORE}, "a blocked run must apply nothing"
        assert _migration_sessions(observer, url.database) == [], "a statement was left queued"

        blocker.rollback()  # the COPY finishes (or the operator terminates it)
        result, _ = _migrate(url, timeout=120, MIGRATION_LOCK_TIMEOUT="2s")
        assert result.returncode == 0, result.stdout + result.stderr
        assert _version(observer) == {_head()}
        assert observer.execute(
            text(
                "SELECT 1 FROM information_schema.columns "
                "WHERE table_name = :t AND column_name = :c"
            ),
            {"t": BLOCKED_TABLE, "c": BLOCKED_COLUMN},
        ).first(), "0022's column was not added once the lock was free"

    def test_a_killed_migration_LEAVES_NO_WAITER_BEHIND(
        self, at_the_blocked_migration: tuple[Any, Any, int, Any]
    ) -> None:
        """A killed init container must not leave its ALTER in the lock queue (miStudio,
        2026-10-21: two did). client_connection_check_interval reaps it long before lock_timeout
        would."""
        url, _blocker, _pid, observer = at_the_blocked_migration
        proc = subprocess.Popen(
            [sys.executable, "-m", "db.migrate", "upgrade"],
            cwd=REPO_ROOT,
            env=_env(url, MIGRATION_LOCK_TIMEOUT="120s", MIGRATION_ATTEMPTS=1),
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        try:
            deadline = time.monotonic() + 60
            while not any(row[2] == "Lock" for row in _migration_sessions(observer, url.database)):
                assert proc.poll() is None, "the migration exited instead of waiting on the lock"
                assert time.monotonic() < deadline, "the migration never queued on the lock"
                time.sleep(0.2)
            proc.send_signal(signal.SIGKILL)
            proc.wait(timeout=10)
            killed = time.monotonic()
            while _migration_sessions(observer, url.database):
                assert time.monotonic() - killed < 15, (
                    "the killed migration's statement is still queued: "
                    f"{_migration_sessions(observer, url.database)}"
                )
                time.sleep(0.25)
        finally:
            if proc.poll() is None:
                proc.kill()

    def test_the_deadline_cancels_a_run_SERVER_SIDE(
        self, at_the_blocked_migration: tuple[Any, Any, int, Any]
    ) -> None:
        url, _blocker, _pid, observer = at_the_blocked_migration
        result, elapsed = _migrate(
            url,
            timeout=60,
            MIGRATION_LOCK_TIMEOUT="120s",
            MIGRATION_ATTEMPTS=1,
            MIGRATION_DEADLINE="3s",
        )
        out = result.stdout + result.stderr
        assert result.returncode == 4, out
        assert "passed its deadline" in out and elapsed < 30, out
        assert _migration_sessions(observer, url.database) == []

    def test_a_long_statement_is_cancelled_and_REPORTED(
        self, at_the_blocked_migration: tuple[Any, Any, int, Any]
    ) -> None:
        url, _blocker, _pid, _observer = at_the_blocked_migration
        result, _ = _migrate(
            url,
            timeout=60,
            MIGRATION_LOCK_TIMEOUT="120s",
            MIGRATION_STATEMENT_TIMEOUT="1s",
            MIGRATION_ATTEMPTS=3,
        )
        out = result.stdout + result.stderr
        assert result.returncode == 4, out
        assert "statement_timeout=1s" in out and "MIGRATION_STATEMENT_TIMEOUT=0" in out
        assert out.count("Running upgrade") == 1, "a statement timeout must not be retried"


class TestTheSessionGuards:
    def test_the_guards_HOLD_ON_A_REAL_SESSION(self, scratch_db: Any) -> None:
        """Each guard is a real server setting on the session, and the idle one ends a transaction
        that stops talking (a migration process that hangs mid-transaction cannot keep its
        locks)."""
        engine = create_engine(scratch_db.render_as_string(hide_password=False))
        settings = ms.MigrationSettings(
            lock_timeout="3s",
            statement_timeout="7s",
            idle_timeout="1s",
            client_check_interval="2s",
            attempts=1,
            retry_backoff_s=0,
            deadline_s=10,
        )
        try:
            with engine.connect() as conn:
                guards = ms.apply_session_guards(conn, settings)
                names = (
                    "application_name",
                    "lock_timeout",
                    "statement_timeout",
                    "idle_in_transaction_session_timeout",
                    "client_connection_check_interval",
                )
                shown = {n: conn.execute(text(f"SHOW {n}")).scalar_one() for n in names}
                assert shown == {
                    "application_name": ms.APPLICATION_NAME,
                    "lock_timeout": "3s",
                    "statement_timeout": "7s",
                    "idle_in_transaction_session_timeout": "1s",
                    "client_connection_check_interval": "2s",
                }, shown
                assert guards.backend_pid > 0
                conn.commit()
                conn.execute(text("SELECT 1"))  # opens a transaction, then goes quiet
                time.sleep(2.5)
                with pytest.raises(DBAPIError):
                    conn.execute(text("SELECT 1"))
        finally:
            engine.dispose()


class TestSQLiteIsUntouched:
    def test_the_runner_migrates_SQLITE_WITHOUT_THE_GUARDS(self, tmp_path: Path) -> None:
        """SQLite has none of these settings (``SHOW`` is not even SQL there): the runner must
        migrate it to head exactly as before, guards skipped, in both CI jobs."""
        url = make_url(f"sqlite+pysqlite:///{tmp_path / 'miforge.db'}")
        result, _ = _migrate(url, timeout=240)
        out = result.stdout + result.stderr
        assert result.returncode == 0, out
        assert "migrations applied" in out
        assert "migration session pid=" not in out, "the PostgreSQL guards reached SQLite"
        engine = create_engine(url)
        try:
            with engine.connect() as conn:
                assert _version(conn) == {_head()}
        finally:
            engine.dispose()

    def test_a_missing_alembic_ini_is_refused_by_name(self, tmp_path: Path) -> None:
        result = subprocess.run(
            [sys.executable, "-m", "db.migrate", "upgrade"],
            cwd=tmp_path,
            env=_env(make_url(f"sqlite+pysqlite:///{tmp_path / 'x.db'}")),
            capture_output=True,
            text=True,
            timeout=120,
        )
        assert result.returncode == 1 and "no alembic.ini" in result.stderr, result.stderr


class TestTheBudget:
    def _docs(self, name: str) -> list[dict[str, Any]]:
        return [d for d in yaml.safe_load_all((GITOPS / name).read_text("utf-8")) if d]

    def test_two_bounded_runs_fit_inside_the_ROLLOUT_PROGRESS_DEADLINE(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """No probe kills an init container, so the deadline is not forced from outside: it is
        chosen. A blocked migration exits non-zero with its blocker named, and the kubelet
        restarts it with backoff (10 s first). Two whole runs (each the deadline plus 60 s for the
        image's imports) and that first backoff must fit inside the Deployment's progress
        deadline, so the blocker is in the log, twice, before ArgoCD marks the rollout Degraded.
        Read with the manifests' own MIGRATION_* values applied: the ConfigMap first, the
        container's env over it, as Kubernetes resolves them."""
        (deployment,) = [
            d
            for d in self._docs("backend.yaml")
            if d.get("kind") == "Deployment" and d["metadata"]["name"] == "miforge-backend"
        ]
        (migrate,) = [
            c
            for c in deployment["spec"]["template"]["spec"]["initContainers"]
            if c["name"] == "migrate"
        ]
        assert migrate["command"] == ["python", "-m", "db.migrate", "upgrade"], migrate["command"]
        for name in ms.DEFAULTS:
            monkeypatch.delenv(name, raising=False)
        maps = {
            d["metadata"]["name"]: d.get("data") or {}
            for d in self._docs("config.yaml")
            if d.get("kind") == "ConfigMap"
        }
        for source in migrate.get("envFrom", []):
            for key, value in maps[source["configMapRef"]["name"]].items():
                if key.startswith("MIGRATION_"):
                    monkeypatch.setenv(key, str(value))
        for env in migrate.get("env", []):
            if env["name"].startswith("MIGRATION_") and "value" in env:
                monkeypatch.setenv(env["name"], env["value"])
        settings = ms.settings_from_env()
        progress_deadline = deployment["spec"].get("progressDeadlineSeconds", 600)
        assert 2 * (settings.deadline_s + 60) + 10 <= progress_deadline, (
            settings.deadline_s,
            progress_deadline,
        )
        assert settings.statement_timeout_s < settings.deadline_s
        assert (
            settings.lock_retry_budget_s() <= settings.deadline_s
        ), "the retries cannot all run inside the deadline"
        assert settings.attempts >= 2, "a single attempt cannot ride out a short-lived lock"


class TestParsing:
    @pytest.mark.parametrize(
        ("value", "seconds"), [("15s", 15), ("500ms", 0.5), ("4min", 240), ("2000", 2), ("0", 0)]
    )
    def test_postgres_durations(self, value: str, seconds: float) -> None:
        assert ms.parse_duration(value) == pytest.approx(seconds)

    def test_a_malformed_setting_is_refused_by_name(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("MIGRATION_LOCK_TIMEOUT", "15 seconds; DROP TABLE x")
        with pytest.raises(ValueError, match="not a duration"):
            ms.settings_from_env()

    def test_the_sqlstate_is_found_through_the_wrapping(self) -> None:
        class InnerError(Exception):
            sqlstate = "55P03"

        class WrappedError(Exception):
            def __init__(self) -> None:
                super().__init__("wrapped")
                self.orig = Exception("adapter")
                self.orig.__cause__ = InnerError()

        assert ms.sqlstate_of(WrappedError()) == ms.LOCK_NOT_AVAILABLE
        assert ms.sqlstate_of(ValueError("no sql")) is None
