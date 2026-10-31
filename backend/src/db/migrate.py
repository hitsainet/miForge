# Origin: miStudio (Onegaishimas/miStudio) backend/src/db/migrate.py @ 67b9ab1f
# Mode: adapt (docs/REUSE.md). Kept: the bounded lock-timeout retry inside one deadline and the
# exit codes (0 / 1 / 3 / 4). Changed: `upgrade` only — miForge migrates in an init container,
# so no app container can start before the schema and miStudio's `wait-for-head` has no caller
# here; the target is `head`; alembic.ini is read from the working directory, as the `alembic`
# command reads it (the image's WORKDIR /app holds it, and this module is installed elsewhere).
"""What the ``migrate`` init container runs instead of a bare ``alembic upgrade head``.

``python -m db.migrate upgrade``
    Runs the migrations with the session guards of :mod:`db.migration_safety` (PostgreSQL only;
    SQLite migrates exactly as before). A lock timeout is retried up to ``MIGRATION_ATTEMPTS``
    times with doubling backoff (``MIGRATION_RETRY_BACKOFF``), all inside ``MIGRATION_DEADLINE`` —
    after which the run is cancelled server-side. Exit codes: 0 applied (or nothing to do), 3 still
    blocked after every attempt (the blockers are in the log), 4 deadline or statement timeout, 1
    any other failure.

    A non-zero exit leaves the pod in ``Init:Error`` and the kubelet restarts the init container
    with its exponential backoff (10 s, 20 s, 40 s, … capped at 5 min), so each restart is a fresh,
    bounded attempt that names its blocker again. Before this, a held lock kept the pod in ``Init``
    indefinitely with nothing in the log.
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path
from typing import Any

#: As the ``alembic`` command: ``ALEMBIC_CONFIG``, else ``alembic.ini`` in the working directory.
DEFAULT_INI = Path(os.environ.get("ALEMBIC_CONFIG", "alembic.ini"))
TARGET = "head"

EXIT_OK = 0
EXIT_FAILED = 1
EXIT_BLOCKED = 3
EXIT_TIMED_OUT = 4


def _say(message: str) -> None:
    print(f"[migrate] {message}", file=sys.stderr, flush=True)


def _config(ini: Path) -> Any:
    from alembic.config import Config

    return Config(str(ini))


def upgrade(ini: Path = DEFAULT_INI, target: str = TARGET) -> int:
    from alembic import command

    from db import migration_safety as ms

    if not ini.is_file():
        _say(f"refusing to migrate: no {ini} in {Path.cwd()}")
        return EXIT_FAILED
    try:
        settings = ms.settings_from_env()
    except ValueError as exc:
        _say(f"refusing to migrate: {exc}")
        return EXIT_FAILED
    started = time.time()
    deadline = started + settings.deadline_s
    os.environ["MIGRATION_DEADLINE_AT"] = repr(deadline)
    backoffs = settings.backoffs()
    _say(
        f"upgrading to {target}: up to {settings.attempts} attempt(s), "
        f"lock_timeout={settings.lock_timeout}, backoff {backoffs}, "
        f"deadline {settings.deadline_s:.0f}s"
    )
    last: Exception | None = None
    for attempt in range(1, settings.attempts + 1):
        try:
            command.upgrade(_config(ini), target)
        except ms.MigrationSafetyError as exc:
            last = exc
            if not exc.retryable:
                _say(f"attempt {attempt} FAILED and will not be retried: {exc}")
                return EXIT_TIMED_OUT
            if attempt == settings.attempts:
                break
            pause = backoffs[attempt - 1]
            if time.time() + pause + settings.lock_timeout_s > deadline:
                _say(
                    f"attempt {attempt} blocked; no time left for another before the "
                    f"{settings.deadline_s:.0f}s deadline"
                )
                break
            _say(
                f"attempt {attempt}/{settings.attempts} blocked on a lock; "
                f"retrying in {pause:.0f}s"
            )
            time.sleep(pause)
            continue
        except Exception as exc:
            # Report every failure the same way, then exit non-zero.
            _say(f"migration FAILED (not a lock timeout; not retried): {type(exc).__name__}: {exc}")
            return EXIT_FAILED
        _say(f"migrations applied in {time.time() - started:.1f}s (attempt {attempt})")
        return EXIT_OK
    _say(
        f"GIVING UP after {time.time() - started:.0f}s: the migration could not get its locks. "
        f"Nothing was applied and no statement is left waiting in Postgres. {last}"
    )
    return EXIT_BLOCKED


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m db.migrate")
    parser.add_argument("action", choices=["upgrade"])
    parser.add_argument(
        "--config",
        type=Path,
        default=DEFAULT_INI,
        help="alembic.ini (default: $ALEMBIC_CONFIG or ./alembic.ini)",
    )
    args = parser.parse_args(argv)
    return upgrade(args.config)


if __name__ == "__main__":
    sys.exit(main())
