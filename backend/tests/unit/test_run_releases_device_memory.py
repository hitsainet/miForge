"""A finished run returns its GPU memory, on every exit  (found live, F11 window 2).

The train worker is a long-lived solo process. After a 1.2B attached run it still held 5.3 GB
(gradients, optimizer states, cache), so R-02.42's budget guard refused the next run on the
card. On a failure, the exception's traceback frames kept the run's locals alive as well.
Driven through the real task body (`run_step_loop.run`).
"""

from __future__ import annotations

import gc
import weakref
from contextlib import contextmanager
from typing import Any

import pytest

from training import run_driver, tasks


class _Big:
    pass


@pytest.fixture
def wired(monkeypatch: pytest.MonkeyPatch) -> list[str]:
    import db.session as db_session

    calls: list[str] = []

    @contextmanager
    def scope():  # type: ignore[no-untyped-def]
        yield object()

    monkeypatch.setattr(db_session, "session_scope", scope)
    monkeypatch.setattr(tasks, "release_device_memory", lambda run_id=None: calls.append(run_id))
    return calls


def test_a_completed_run_releases_once(wired, monkeypatch) -> None:  # type: ignore[no-untyped-def]
    monkeypatch.setattr(run_driver, "drive_recorded_run", lambda *a, **k: {"status": "completed"})
    assert tasks.run_step_loop.run("r1")["status"] == "completed"
    assert wired == ["r1"]


def test_a_failed_run_releases_and_its_frames_no_longer_hold_the_run(wired, monkeypatch) -> None:  # type: ignore[no-untyped-def]
    ref: list[Any] = []

    def boom(*args: Any, **kwargs: Any) -> Any:
        model = _Big()  # stands in for the module, the optimizer, the gradients
        ref.append(weakref.ref(model))
        raise RuntimeError("the run failed")

    monkeypatch.setattr(run_driver, "drive_recorded_run", boom)
    with pytest.raises(RuntimeError) as caught:
        tasks.run_step_loop.run("r2")
    assert wired == ["r2"]
    gc.collect()
    assert ref[0]() is None, "the failure's traceback still holds the run's locals"
    assert caught.value is not None
