"""The janitor is **reachable in production**  (ADR-016 #1; foundation 6.1, 6.5, 5.3).

*A capability is not shipped until a test FAILS when its wiring is removed.* This file is
that test for the janitor, and it is here rather than folded into ``test_janitor.py``
because every test in that file passes by importing :func:`workers.janitor.sweep`
directly - which is exactly how a sibling shipped sixteen fully-implemented, unit-tested,
documented MCP tools that were never registered with the server. The suite was green and
the documentation said ✅ while the feature did not exist for any caller.

So these assertions are made against the **live Celery registry** and the **live beat
schedule**, never against the module. Removing the ``@celery_app.task`` decorator, the beat
entry, or the maintenance route must each turn this file red.
"""

from __future__ import annotations

from workers.celery_app import (
    QUEUE_MAINTENANCE,
    QUEUES,
    TASK_JANITOR_SWEEP,
    celery_app,
    queue_for,
)


class TestTheLiveRegistry:
    def test_the_sweep_task_is_registered_with_the_app(self):
        """The live registry, not the module. An unregistered task is unreachable however
        well tested its body is."""
        assert TASK_JANITOR_SWEEP in celery_app.tasks

    def test_the_sweep_task_routes_to_the_maintenance_queue(self):
        """Resolved through Celery's own router, so the queue asserted here is the queue
        the broker uses rather than a second implementation of the glob match."""
        assert queue_for(TASK_JANITOR_SWEEP) == QUEUE_MAINTENANCE

    def test_every_registered_miforge_task_resolves_to_a_queue(self):
        """A task with no route silently lands on ``default``, where a worker with no card
        picks up a training task and fails. The route table is explicit so that a feature
        adding a task under an unrouted area turns this red."""
        unrouted = [
            name
            for name in celery_app.tasks
            if name.startswith("miforge.") and queue_for(name) is None
        ]
        assert unrouted == []


class TestTheBeatSchedule:
    def test_the_sweep_is_on_the_schedule(self):
        """**The janitor's only production caller.** Without an entry here the sweep is a
        function nothing ever calls, and no run is ever reaped."""
        schedule = celery_app.conf.beat_schedule
        entries = [e for e in schedule.values() if e["task"] == TASK_JANITOR_SWEEP]
        assert len(entries) == 1, "exactly one scheduled caller, not zero and not two"

    def test_the_interval_is_shorter_than_the_lapse_threshold(self):
        """Otherwise worst-case detection is the sum of the two rather than roughly one
        interval past lapsing."""
        from core.config import get_settings
        from workers.celery_app import JANITOR_INTERVAL_SECONDS

        assert get_settings().janitor_lapse_seconds > JANITOR_INTERVAL_SECONDS

    def test_the_scheduled_entry_targets_the_maintenance_queue(self):
        """The payload, not merely the presence. A scheduled task sent to ``train`` queues
        behind a ten-hour training job and reaps nothing - which is precisely the kind of
        wrong-argument call that a 'was it called' assertion passes against (ADR-016)."""
        entry = next(
            e for e in celery_app.conf.beat_schedule.values() if e["task"] == TASK_JANITOR_SWEEP
        )
        assert entry["options"]["queue"] == QUEUE_MAINTENANCE


class TestTheQueues:
    def test_all_four_queues_are_declared(self):
        declared = {q.name for q in celery_app.conf.task_queues}
        assert declared == set(QUEUES) == {"default", "train", "generate", "maintenance"}

    def test_the_default_queue_is_default(self):
        assert celery_app.conf.task_default_queue == "default"
