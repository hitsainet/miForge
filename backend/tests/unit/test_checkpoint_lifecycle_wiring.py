"""The lifecycle sweep is **reachable in production**  (ADR-016 #1; OQ-4).

*A capability is not shipped until a test FAILS when its wiring is removed.* Every
assertion here is made against the **live** Celery registry, the **live** router and the
**live** beat schedule - never against the module, because importing
:mod:`workers.checkpoint_lifecycle` proves only that a file parses. That is exactly how a
sibling shipped sixteen fully-implemented, unit-tested, documented MCP tools that were
never registered with the server: the suite was green and the docs said ✅ while the feature
did not exist for any caller.

Removing the ``@celery_app.task`` decorator, the beat entry, or the ``maintenance`` route
must each turn this file red. The schedule entry matters more here than for most tasks:
**the beat entry is the sweep's only production caller.** There is no endpoint - F5's route
module deliberately exposes no delete - so without it the policy is a tested function that
never runs and the volume still fills at step 9,000.
"""

from __future__ import annotations

from workers.celery_app import (
    CHECKPOINT_PRUNE_INTERVAL_SECONDS,
    QUEUE_MAINTENANCE,
    TASK_CHECKPOINT_PRUNE,
    celery_app,
    queue_for,
)


class TestTheLiveRegistry:
    def test_the_prune_task_is_registered_with_the_app(self) -> None:
        assert TASK_CHECKPOINT_PRUNE in celery_app.tasks

    def test_it_routes_to_the_maintenance_queue(self) -> None:
        """Resolved through Celery's own router, so the queue asserted is the one the
        broker uses rather than a second implementation of the glob match.

        ``maintenance`` rather than ``default`` because the volume fills **while** a
        ten-hour training task is running, which is exactly when the sweep has work to do -
        and a prune queued behind that task cannot free anything."""
        assert queue_for(TASK_CHECKPOINT_PRUNE) == QUEUE_MAINTENANCE

    def test_the_plausible_unrouted_spelling_still_routes_nowhere(self) -> None:
        """**The tripwire, and this project has paid for it three times.**
        ``TASK_ROUTES`` is an explicit table keyed by area prefix, so the name this task
        wants to be called - ``miforge.checkpoints.prune`` - matches no entry and lands on
        ``default``: F1's ``miforge.training.run_step_loop``, F8's merge and F6's
        ``miforge.chain.sweep``, all over again. If a later edit adds a
        ``miforge.checkpoints.*`` route this goes red and says so, rather than silently
        making the wrong name work."""
        assert queue_for("miforge.checkpoints.prune") is None
        assert TASK_CHECKPOINT_PRUNE.startswith("miforge.maintenance.")


class TestTheBeatSchedule:
    def test_the_prune_is_on_the_schedule_exactly_once(self) -> None:
        """Its only production caller. Not zero, and not two - a second entry would double
        every sweep's manifest reads for nothing."""
        entries = [
            e for e in celery_app.conf.beat_schedule.values() if e["task"] == TASK_CHECKPOINT_PRUNE
        ]
        assert len(entries) == 1

    def test_the_scheduled_entry_targets_the_maintenance_queue(self) -> None:
        """**The payload, not merely the presence** (ADR-016). A reachability test that
        only checked "it is scheduled" passes against an entry sending the task to
        ``train``, where it queues behind the run whose checkpoints it exists to prune."""
        entry = next(
            e for e in celery_app.conf.beat_schedule.values() if e["task"] == TASK_CHECKPOINT_PRUNE
        )
        assert entry["options"]["queue"] == QUEUE_MAINTENANCE
        assert entry["schedule"] == CHECKPOINT_PRUNE_INTERVAL_SECONDS

    def test_it_sweeps_far_less_often_than_the_janitor(self) -> None:
        """Different clocks, not different degrees of care. A lapsed run holds a GPU and
        must be reaped within an interval; a volume fills over hours. A per-minute sweep
        would re-read every manifest under the root sixty times per checkpoint it could
        possibly act on - and nothing is protected by this number, which is why it is
        allowed to be coarse."""
        from workers.celery_app import JANITOR_INTERVAL_SECONDS

        assert CHECKPOINT_PRUNE_INTERVAL_SECONDS > JANITOR_INTERVAL_SECONDS * 10


class TestTheTaskBodyReachesTheSweep:
    def test_calling_the_registered_task_calls_the_sweep_with_a_session(self, monkeypatch) -> None:
        """**The payload and the call count**, because "was called" passes against a call
        sending the wrong arguments.

        The task is invoked through ``celery_app.tasks[...]``, i.e. the registry entry a
        worker would use, rather than by importing the function - so a decorator that
        registered some *other* callable under this name would not pass."""
        from contextlib import contextmanager

        import workers.checkpoint_lifecycle as lifecycle_worker

        calls: list[dict[str, object]] = []

        class FakeReport:
            def as_dict(self) -> dict[str, object]:
                return {"outcome": "disabled", "kept": 7}

        def fake_sweep(session, **kwargs):
            calls.append({"session": session, "kwargs": kwargs})
            return FakeReport()

        @contextmanager
        def fake_scope():
            yield "the-session"

        monkeypatch.setattr(lifecycle_worker, "sweep", fake_sweep)
        monkeypatch.setattr("workers.run_record.default_session_scope", fake_scope)

        result = celery_app.tasks[TASK_CHECKPOINT_PRUNE].run()

        assert len(calls) == 1, "once per sweep, not zero and not twice"
        assert calls[0]["session"] == "the-session"
        assert calls[0]["kwargs"] == {}, "the production sweep takes the configured root"
        assert result == {"outcome": "disabled", "kept": 7}


class TestTheSettingsThatGateIt:
    def test_both_gates_default_to_deleting_nothing(self) -> None:
        """Shipped **disabled and dry-run**, deliberately. miStudio shipped its own
        step-granular checkpoint retention the same way, and a policy that starts deleting
        on upgrade is a policy nobody consented to on the installation it first runs on."""
        from core.config import Settings

        settings = Settings()
        assert settings.checkpoint_lifecycle_enabled is False
        assert settings.checkpoint_lifecycle_dry_run is True

    def test_n_and_m_are_configuration_with_defaults_not_literals(self) -> None:
        from core.config import Settings
        from services.checkpoints.lifecycle import LifecyclePolicy

        settings = Settings(checkpoint_lifecycle_keep_last_n=4, checkpoint_lifecycle_keep_every_m=9)
        # `assert (x, 4)` was written here first: a two-tuple, always truthy, asserting
        # nothing. pytest's own PytestAssertRewriteWarning is what caught it.
        assert settings.checkpoint_lifecycle_keep_last_n == 4
        assert LifecyclePolicy().keep_last_n == Settings().checkpoint_lifecycle_keep_last_n
        assert LifecyclePolicy().keep_every_m == Settings().checkpoint_lifecycle_keep_every_m
        assert settings.checkpoint_lifecycle_keep_every_m == 9

    def test_the_sweep_reads_n_and_m_from_settings_and_not_from_the_dataclass(
        self, monkeypatch
    ) -> None:
        """A policy built from ``LifecyclePolicy()`` would ignore the operator entirely
        while every default-valued test still passed."""
        import core.config as config_module
        from workers.checkpoint_lifecycle import policy_from_settings

        monkeypatch.setattr(
            config_module,
            "_settings",
            config_module.Settings(
                checkpoint_lifecycle_keep_last_n=3, checkpoint_lifecycle_keep_every_m=7
            ),
        )
        policy = policy_from_settings()
        assert (policy.keep_last_n, policy.keep_every_m) == (3, 7)


class TestAnUnparseableFlagFailsToItsDefault:
    """**miStudio's durable lesson, and it is not a style preference:** *a boolean setting
    must fail to its DEFAULT, not to False - for ``dry_run``, False means delete.*

    Two plausible behaviours are both wrong here. Pydantic's own is to raise, which stops
    the process over a typo on a field whose safe value is already the default. Falling
    through to ``False`` is the one outcome that must never happen: it deletes checkpoints
    while the operator believes the opposite.
    """

    def test_a_typo_in_dry_run_does_not_mean_delete(self) -> None:
        from core.config import Settings

        assert Settings(checkpoint_lifecycle_dry_run="ture").checkpoint_lifecycle_dry_run is True
        assert Settings(checkpoint_lifecycle_dry_run="nope!").checkpoint_lifecycle_dry_run is True

    def test_an_empty_value_means_unset_which_means_the_default(self) -> None:
        """``CHECKPOINT_LIFECYCLE_DRY_RUN=`` in a ``.env`` file says *I did not set this*,
        which is the default rather than ``False`` - the same reading
        ``_empty_string_is_unset`` already gives the sibling URLs."""
        from core.config import Settings

        assert Settings(checkpoint_lifecycle_dry_run="").checkpoint_lifecycle_dry_run is True
        assert Settings(checkpoint_lifecycle_enabled="").checkpoint_lifecycle_enabled is False

    def test_a_typo_in_enabled_does_not_turn_it_on(self) -> None:
        from core.config import Settings

        assert Settings(checkpoint_lifecycle_enabled="ya").checkpoint_lifecycle_enabled is False

    def test_the_recognised_spellings_are_still_pydantics_own(self) -> None:
        """The accepted vocabulary must not become a second one declared in the validator:
        a value Pydantic understands is passed straight through."""
        from core.config import Settings

        for text, expected in (
            ("false", False),
            ("0", False),
            ("no", False),
            ("off", False),
            ("true", True),
            ("1", True),
            ("yes", True),
            ("on", True),
        ):
            assert Settings(checkpoint_lifecycle_dry_run=text).checkpoint_lifecycle_dry_run is (
                expected
            ), text

    def test_it_warns_rather_than_falling_back_in_silence(self, caplog) -> None:
        """A silent fallback is how a misconfigured flag survives for months. The log line
        names the field and the value, so the fix is one edit."""
        import logging

        from core.config import Settings

        with caplog.at_level(logging.WARNING, logger="miforge.config"):
            Settings(checkpoint_lifecycle_dry_run="maybe")
        assert any(
            record.__dict__.get("code") == "checkpoint_lifecycle_flag_unparseable"
            and record.__dict__.get("value") == "maybe"
            for record in caplog.records
        )
