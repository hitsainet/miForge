"""Every beat entry keeps firing when beat restarts with an empty schedule file.

**The defect this pins (2026-10-22).** The beat container runs with
``--schedule=/tmp/celerybeat-schedule`` on its writable layer, so every pod restart starts
from an empty schedule file. Celery's ``ScheduleEntry.__init__`` then sets
``last_run_at = last_run_at or self.default_now()``, so an *interval* entry counts its
interval from beat's start. ``checkpoint-prune`` was a 3,600 s interval: on a pod redeployed
more often than hourly - every push to ``main`` rolls it - it never fired, with no error
anywhere. miStudio's and miDataworks' daily prunes had the same defect.

**How it is tested.** Nothing here reads source. Each test drives Celery's real
:class:`~celery.beat.PersistentScheduler` - the class ``celery beat`` uses by default -
against this app's real ``beat_schedule``, ticking a simulated clock (``app.now`` is what
every schedule consults) and opening a **fresh schedule file at every simulated restart**,
exactly as the container does. The only thing replaced is the send: the scheduler records
the entry instead of publishing to the broker.
"""

from __future__ import annotations

import itertools
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import pytest
from celery.beat import PersistentScheduler

from workers.celery_app import CHECKPOINT_PRUNE_INTERVAL_SECONDS, celery_app

#: An arbitrary instant that is not aligned to any minute or hour, so a crontab entry
#: cannot pass by coinciding with a restart.
START = datetime(2026, 10, 22, 13, 41, 17, 250000, tzinfo=UTC)

#: Restart cadence for the all-entries test. Ten minutes is a busy deploy day: every push
#: to ``main`` rolls the pod, and the operator pushes in batches.
BUSY_DAY_RESTART_SECONDS = 600.0


class _RecordingScheduler(PersistentScheduler):
    """Celery's PersistentScheduler with the broker send replaced by a record."""

    #: ``tick`` passes ``self.producer`` to ``apply_entry``; the base class's cached
    #: property would open a broker connection. Nothing is published here.
    producer = None

    def __init__(self, *args: Any, sent: list[tuple[str, datetime]], **kwargs: Any) -> None:
        self._sent = sent
        super().__init__(*args, **kwargs)

    def apply_async(
        self, entry: Any, producer: Any = None, advance: bool = True, **kwargs: Any
    ) -> None:
        if advance:
            entry = self.reserve(entry)
        self._sent.append((entry.name, self.app.now()))


def _simulate(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    *,
    span_seconds: float,
    restart_every_seconds: float | None,
) -> dict[str, list[datetime]]:
    """Run beat for ``span_seconds`` of simulated time, restarting it with a fresh schedule
    file every ``restart_every_seconds`` (``None``: never). Returns send times per entry."""
    tmp_path.mkdir(parents=True, exist_ok=True)
    clock = [START]
    monkeypatch.setattr(celery_app, "now", lambda: clock[0])
    end = START + timedelta(seconds=span_seconds)
    sent: list[tuple[str, datetime]] = []
    restart = 0
    while clock[0] < end:
        window = restart_every_seconds if restart_every_seconds is not None else span_seconds
        window_end = min(clock[0] + timedelta(seconds=window), end)
        # A FRESH file each start: the container's writable layer does not survive a restart.
        scheduler = _RecordingScheduler(
            app=celery_app, schedule_filename=str(tmp_path / f"celerybeat-{restart}"), sent=sent
        )
        restart += 1
        while clock[0] < window_end:
            delay = float(scheduler.tick())
            clock[0] = min(clock[0] + timedelta(seconds=max(delay, 1e-3)), window_end)
        scheduler.close()
    by_entry: dict[str, list[datetime]] = {name: [] for name in celery_app.conf.beat_schedule}
    for name, at in sent:
        by_entry.setdefault(name, []).append(at)
    return by_entry


def _longest_gap(sends: list[datetime], span_seconds: float) -> float:
    """The longest stretch of the simulated span with no send, edges included."""
    points = [START, *sends, START + timedelta(seconds=span_seconds)]
    return max((b - a).total_seconds() for a, b in itertools.pairwise(points))


class TestCheckpointPruneSurvivesRestarts:
    @pytest.mark.parametrize(
        "restart_every_seconds",
        [None, 600.0, 1800.0, 3540.0],
        ids=["never", "every-10-min", "every-30-min", "every-59-min"],
    )
    def test_a_restarted_beat_still_prunes_within_its_period(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
        restart_every_seconds: float | None,
    ) -> None:
        """**The defect, directly.** Restarted more often than hourly, the sweep must still
        run at least once in every hour. As a 3,600 s interval it never runs at all: each
        restart puts ``last_run_at`` back to "now"."""
        span = 8 * 3600.0
        sends = _simulate(
            tmp_path, monkeypatch, span_seconds=span, restart_every_seconds=restart_every_seconds
        )["checkpoint-prune"]
        assert sends, "checkpoint-prune never fired across the restarts"
        assert _longest_gap(sends, span) <= CHECKPOINT_PRUNE_INTERVAL_SECONDS + 1.0, sends

    def test_the_period_is_still_hourly(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The crontab keeps the configured meaning: one sweep per
        ``CHECKPOINT_PRUNE_INTERVAL_SECONDS``, not more often and not less."""
        sends = _simulate(
            tmp_path, monkeypatch, span_seconds=6 * 3600.0, restart_every_seconds=None
        )["checkpoint-prune"]
        gaps = {(b - a).total_seconds() for a, b in itertools.pairwise(sends)}
        assert gaps == {CHECKPOINT_PRUNE_INTERVAL_SECONDS}, gaps


class TestEveryEntrySurvivesABusyDeployDay:
    def test_every_entry_keeps_firing_when_beat_restarts_every_ten_minutes(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """**A guard for the next entry, not only this one.** Each entry's natural period is
        measured with no restarts; then beat is restarted every ten minutes and no entry may
        go more than two of its periods without firing. A short interval (the janitor's 60 s)
        passes - a restart delays it by one interval. A long interval fails: it is never
        due before the next restart resets it. Make it a ``crontab``."""
        span = 6 * 3600.0
        steady = _simulate(
            tmp_path / "steady", monkeypatch, span_seconds=span, restart_every_seconds=None
        )
        busy = _simulate(
            tmp_path / "busy",
            monkeypatch,
            span_seconds=span,
            restart_every_seconds=BUSY_DAY_RESTART_SECONDS,
        )
        assert set(steady) == set(celery_app.conf.beat_schedule)
        for name, sends in steady.items():
            assert len(sends) >= 2, f"{name}: fired {len(sends)} times in {span:.0f} s"
            period = max((b - a).total_seconds() for a, b in itertools.pairwise(sends))
            gap = _longest_gap(busy[name], span)
            assert gap <= 2 * period + 1.0, (
                f"{name}: period {period:.0f} s, but {gap:.0f} s without firing while beat "
                f"restarts every {BUSY_DAY_RESTART_SECONDS:.0f} s - an interval longer than "
                "the restart cadence never comes due; schedule it with a crontab"
            )


class TestThePremise:
    def test_a_long_interval_entry_never_fires_across_frequent_restarts(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """**The Celery behaviour the fix rests on**, pinned so a Celery change that makes
        it untrue is noticed. ``ScheduleEntry.__init__``: ``self.last_run_at = last_run_at
        or self.default_now()`` - a fresh schedule file counts an interval from beat's start.
        If this ever goes red, the reasoning in ``CHECKPOINT_PRUNE_SCHEDULE`` is stale."""
        monkeypatch.setitem(
            celery_app.conf,
            "beat_schedule",
            {"hourly-interval": {"task": "miforge.maintenance.never", "schedule": 3600.0}},
        )
        sends = _simulate(
            tmp_path, monkeypatch, span_seconds=8 * 3600.0, restart_every_seconds=1800.0
        )
        assert sends["hourly-interval"] == []
