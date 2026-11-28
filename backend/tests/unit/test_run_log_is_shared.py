"""A run's log is writable by every process of the app's group, whatever the creator's umask.

Found live 2026-10-17 (F11 window 2): the API created `<runs>/<run_id>/` 0755 and `run.jsonl`
0644 under umask 0022; the train worker, running as another uid with the same group during a
hot-update session, could not append, and the run failed before its first step.
"""

from __future__ import annotations

import os
import stat
from pathlib import Path

import pytest

from training import run_log


@pytest.fixture
def strict_umask():  # type: ignore[no-untyped-def]
    previous = os.umask(0o022)
    yield
    os.umask(previous)


def _block() -> dict:
    return {
        "block": "step",
        "run_id": "r1",
        "step": 1,
        "ts": "2026-10-17T00:00:00Z",
        "loss": 1.0,
        "objective": "cross_entropy",
        "learning_rate": 1e-4,
        "wall_time_ms": 1.0,
    }


def test_the_directory_and_file_are_group_writable_under_umask_022(
    tmp_path: Path, strict_umask  # type: ignore[no-untyped-def]
) -> None:
    path = run_log.run_log_path(tmp_path, "r1")
    run_log.append(path, _block())
    # Literal values, NOT the module's constants: comparing against the constant would pass
    # against any mode the constant were changed to (a test reimplementing what it tests).
    assert stat.S_IMODE(path.parent.stat().st_mode) == 0o2775
    assert stat.S_IMODE(path.stat().st_mode) == 0o664


def test_an_existing_files_mode_is_not_rewritten_by_a_later_appender(
    tmp_path: Path, strict_umask  # type: ignore[no-untyped-def]
) -> None:
    path = run_log.run_log_path(tmp_path, "r1")
    run_log.append(path, _block())
    os.chmod(path, 0o600)
    run_log.append(path, {**_block(), "step": 2})
    assert stat.S_IMODE(path.stat().st_mode) == 0o600
    assert len(path.read_text().splitlines()) == 2
