"""The protocol heartbeat: the interval from the hint, and never behind the step.

**MUT-18's target.** Calling the heartbeat from inside the step loop starves it under load
*only* - a short step looks fine - so the test needs a **long** fake step and a heartbeat
**count**. A single-call assertion passes against a heartbeat that fired once before the
step and never again, which is exactly the shape of the defect.

**FR-011.16:** the interval comes from miLLM's own hint, never a constant here, because the
staleness timeout it must beat is miLLM's configuration and a local constant drifts from it
in silence.
"""

from __future__ import annotations

import threading
import time
from collections.abc import Callable

import pytest

from tests.support.fixtures_hotupdate import Harness
from training.hotupdate import heartbeat as heartbeat_module
from training.hotupdate.heartbeat import Heartbeat


class TestTheIntervalComesFromMiLLM:
    def test_the_hint_is_used(self) -> None:
        """BRD-01 R-01.30 returns *the staleness window and a suggested interval*."""
        assert (
            heartbeat_module.interval_from(
                {"heartbeat": {"timeout_s": 60.0, "interval_hint_s": 7.0}}
            )
            == 7.0
        )

    def test_the_OLD_invented_names_are_no_longer_honoured(self) -> None:
        """The regression that stops the wrong names coming back.

        `staleness_window_seconds` and `suggested_interval_seconds` are F11's own coinage
        and appear in no version of BRD-01 or of miLLM's FTDD. A reader that still
        honoured them would pass every test above **and** keep working against the fake,
        so the only thing that can catch their return is an assertion that they are
        ignored. Without this, re-adding them as a "compatibility" fallback would be
        invisible — and a translation table between two names for one concept is a second
        source of truth, which is the family behind `step-`, `full` vs `change` and
        `queued` vs `pending` in this project.
        """
        assert (
            heartbeat_module.interval_from(
                {"staleness_window_seconds": 60.0, "suggested_interval_seconds": 7.0}
            )
            == heartbeat_module.FALLBACK_INTERVAL_SECONDS
        )

    def test_without_a_hint_half_the_window_is_used(self) -> None:
        """Half, not the whole: an interval equal to the window makes every heartbeat the
        last one that could still arrive in time, so one slow call marks the attachment
        stale."""
        assert heartbeat_module.interval_from({"heartbeat": {"timeout_s": 60.0}}) == 30.0

    def test_the_configured_override_wins(self) -> None:
        assert (
            heartbeat_module.interval_from({"heartbeat": {"interval_hint_s": 7.0}}, override=2.0)
            == 2.0
        )

    def test_with_neither_a_warning_names_the_last_resort(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        with caplog.at_level("WARNING"):
            value = heartbeat_module.interval_from({})
        assert value == heartbeat_module.FALLBACK_INTERVAL_SECONDS
        assert any(r.__dict__.get("code") == "heartbeat_interval_unstated" for r in caplog.records)

    def test_the_attach_response_carries_what_the_interval_is_read_from(
        self, harness: Harness
    ) -> None:
        """A fake that omitted the window could not test FR-011.16 at all.

        **These assertions now name miLLM's field names rather than F11's own.** They
        previously asserted `staleness_window_seconds` and `suggested_interval_seconds`,
        which the upstream never sends — so this test, whose stated purpose is that the
        fake carries what the interval is read from, was pinning the WRONG CONTRACT and
        passing. It is the clearest case in this feature of a fixture and a reader being
        wrong together: both sides agreed, so nothing could disagree with them.
        """
        response = harness.attach()
        assert (
            response["heartbeat"]["timeout_s"] > 0
        ), "miLLM FTDD 031 §5.3: the attach response carries heartbeat.timeout_s"
        assert response["heartbeat"]["interval_hint_s"] > 0
        beat = harness.client.call("heartbeat")
        assert beat["interval_hint_s"] > 0
        assert beat["timeout_s"] > 0


#: How long the "step" (this thread, which never calls the heartbeat) holds while waiting for
#: beats. A DEADLINE, not a duration: the wait ends as soon as the beats are seen.
STARVED_AFTER_SECONDS = 20.0


def _hold_until(condition: Callable[[], bool], deadline: float = STARVED_AFTER_SECONDS) -> bool:
    """Hold this thread until ``condition`` is true or ``deadline`` seconds pass.

    These tests used to sleep a fixed 80 or 150 ms and then count beats at a 10 ms interval,
    which asserts how fast the machine schedules a thread, not that the heartbeat is
    independent of the step: under a loaded full-suite run the count came up short once
    (2026-11-11) and the test passed alone. Waiting on the count itself keeps what is being
    proved - beats arrive while this thread does nothing to cause them - and leaves the clock
    out of it. A heartbeat that really is starved never satisfies the condition and fails at
    the deadline.
    """
    end = time.monotonic() + deadline
    while time.monotonic() < end:
        if condition():
            return True
        time.sleep(0.005)
    return condition()


class TestItIsNotBehindTheStep:
    def test_a_long_step_does_not_starve_it(self, harness: Harness) -> None:
        """**MUT-18.** Its own thread, its own clock: the beats are counted *while* a long
        step is in flight, and a hook in the loop would produce one or zero."""
        harness.attach()
        beat = Heartbeat(client=harness.client, interval_seconds=0.01)
        beat.start()
        try:
            # The "long step": this thread holds, as a forward pass would, and never calls
            # the heartbeat. One called from inside the loop could not fire at all here.
            _hold_until(lambda: beat.beats >= 3)
        finally:
            beat.stop()
        assert beat.beats >= 3, f"the heartbeat was starved by the step: {beat.beats} beats"
        assert harness.fake.call_count("heartbeat") == beat.beats

    def test_it_keeps_beating_during_a_commit(self, harness: Harness) -> None:
        """The commit watchdog and the staleness timeout are independent: *a quiet trainer
        mid-commit looks dead.*"""
        from training.hotupdate import barrier

        harness.attach()
        beat = Heartbeat(client=harness.client, interval_seconds=0.01)
        beat.start()
        try:
            with barrier.commit_bracket(harness.client, harness.timings):
                # Only beats sent AFTER the bracket opened count: the first beat is sent at
                # start, before it, and would otherwise be credited to the commit.
                at_open = beat.beats
                _hold_until(lambda: beat.beats >= at_open + 2)
                beats_inside = beat.beats - at_open
        finally:
            beat.stop()
        assert beats_inside >= 2, (
            f"the heartbeat stopped while the bracket was open: {beats_inside} beat(s) in "
            f"{STARVED_AFTER_SECONDS:.0f} s at a 10 ms interval"
        )

    def test_it_runs_on_its_own_thread(self, harness: Harness) -> None:
        harness.attach()
        with Heartbeat(client=harness.client, interval_seconds=0.01) as beat:
            time.sleep(0.03)
            names = [t.name for t in threading.enumerate()]
            assert "hotupdate-heartbeat" in names
        assert beat.stopped


class TestItStopsBeforeDetach:
    def test_detach_stops_it_first(self, harness: Harness) -> None:
        """FR-011.18: it never outlives the attachment it proves."""
        from training.hotupdate.attach import AttachAcquirer

        acquirer = AttachAcquirer(transport=harness.fake.transport())
        acquirer.client = harness.client
        harness.attach()
        beat = Heartbeat(client=harness.client, interval_seconds=0.01)
        beat.start()
        acquirer.detach(heartbeat=beat)
        assert beat.stopped is True
        ops = harness.fake.ops()
        assert ops[-1] == "detach"
        assert "heartbeat" not in ops[ops.index("detach") :]

    def test_a_beat_after_stop_does_not_happen(self, harness: Harness) -> None:
        harness.attach()
        beat = Heartbeat(client=harness.client, interval_seconds=0.01, sleep=lambda _s: None)
        beat.stop()
        beat.run_until_stopped()
        assert beat.beats == 0


class TestARevivedAttachment:
    def test_a_heartbeat_from_a_stale_but_uncleared_attachment_revives_it(
        self, harness: Harness
    ) -> None:
        """BRD-01 R-01.30's own sentence. Asserted so the fake's stale behaviour cannot
        contradict it - a fake that refused a heartbeat forever after one stall would make
        every recovery test agree with a rule miLLM does not have."""
        harness.attach()
        harness.fake.stale = True
        Heartbeat(client=harness.client, interval_seconds=1.0).beat_once()
        assert harness.fake.stale is False
