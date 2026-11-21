"""A commit the watchdog cut off and this process then completed LATE is recorded.

Found live 2026-10-18 (F11 14.6, run f11w2-r7 step 1): the trainer was paused mid-commit
past miLLM's 800 ms watchdog; miLLM declared the set ``torn`` and answered ``/v1`` with 503
``weights_torn``; on resuming, the trainer's commit-end was accepted late (R-01.23a) with
``overran: true`` and serving came back ``clean``. Path (a) worked - and miForge recorded
nothing but a "creep" warning, because the bracket never read commit-end's answer.
"""

from __future__ import annotations

import logging
import time

import pytest

from tests.support.fixtures_hotupdate import Harness
from training.hotupdate import barrier


def test_a_late_commit_end_is_recorded_as_an_overrun(
    harness: Harness, caplog: pytest.LogCaptureFixture
) -> None:
    harness.attach()
    harness.fake.commit_watchdog_ms = 1.0  # the watchdog fires during the write
    with (
        caplog.at_level(logging.ERROR),
        barrier.commit_bracket(harness.client, harness.timings),
    ):
        time.sleep(0.02)
    assert harness.fake.commit_overran is True, "the fake did not take the late path"
    assert harness.fake.weights_state == "clean"
    assert len(harness.timings.overruns) == 1
    assert harness.timings.overruns[0]["commit_duration_ms"] >= 20.0
    codes = [getattr(r, "code", None) for r in caplog.records]
    assert codes.count("weights_commit_overran") == 1


def test_an_in_time_commit_records_no_overrun(harness: Harness) -> None:
    harness.attach()
    with barrier.commit_bracket(harness.client, harness.timings):
        pass
    assert harness.timings.overruns == []
