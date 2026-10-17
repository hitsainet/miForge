"""The two §5.10 step-record fields, and the figure another repository's gate reads.

**FR-011.57 - FR-011.59; MUT-15.** Dropping the two fields **validates cleanly** - both are
nullable - and silently removes the input to miLLM Feature 31's acceptance gate. So the
schema is not what catches MUT-15: a test asserting the fields are *populated* is.

The hold is **measured locally** (FPRD DEF-B): nothing on miLLM's surface reports a hold
duration. The quiesce response carries a held-request count, the state, a timestamp and a
deadline - *and a deadline is a limit, not a hold.*
"""

from __future__ import annotations

from pathlib import Path

import pytest

from core.errors import Refusal
from tests.support.fixtures_hotupdate import Harness
from training import run_log
from training.hotupdate import barrier
from training.hotupdate import timings as timings_module
from training.hotupdate.timings import StepTimings


class TestBothFieldsArePopulatedAndNeitherIsDefinedHere:
    def test_a_bracketed_step_fills_both_fields(self, harness: Harness) -> None:
        """**MUT-15.**"""
        harness.attach()
        with barrier.commit_bracket(harness.client, harness.timings):
            pass
        last = harness.timings.last
        assert last["quiesce_hold_ms"] is not None
        assert last["commit_duration_ms"] is not None
        assert last["quiesce_hold_ms"] >= last["commit_duration_ms"], (
            "the hold contains the commit, so a hold shorter than its own commit means the "
            "two clocks are measuring different windows"
        )

    def test_fill_puts_them_into_F1s_block_and_the_block_still_validates(
        self, harness: Harness
    ) -> None:
        harness.attach()
        with barrier.commit_bracket(harness.client, harness.timings):
            pass
        block = run_log.step_block(
            run_id="hot-run",
            step=1,
            loss=0.5,
            objective="causal_lm",
            learning_rate=1e-5,
            wall_time_ms=120.0,
        )
        assert block["quiesce_hold_ms"] is None, "F1 writes them present and unset"
        filled = timings_module.fill(block, harness.timings)
        assert filled["commit_duration_ms"] == harness.timings.commits_ms[-1]
        run_log.validate_block(filled)
        assert set(filled) == set(block), "F11 defines no field and adds no key"

    def test_filling_a_field_the_contract_does_not_have_is_refused(self, harness: Harness) -> None:
        """The schema closes the block to unknown keys, so a new key here would be F11
        defining a field it does not own."""
        block = run_log.step_block(
            run_id="r", step=0, loss=1.0, objective="o", learning_rate=1e-5, wall_time_ms=1.0
        )
        block.pop("quiesce_hold_ms")
        with pytest.raises(Refusal) as caught:
            timings_module.fill(block, harness.timings)
        assert caught.value.code == "step_record_field_unknown"

    def test_the_default_path_is_unaffected(self) -> None:
        """A standalone step writes them **present and null** - an assertion that this run
        had no barrier to measure, where an absent key would be an omission."""
        block = run_log.step_block(
            run_id="r", step=0, loss=1.0, objective="o", learning_rate=1e-5, wall_time_ms=1.0
        )
        run_log.validate_block(block)
        assert block["quiesce_hold_ms"] is None and block["commit_duration_ms"] is None


class TestTheCreepWarning:
    def test_a_commit_past_the_fraction_warns_and_names_the_watchdog(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        """*A run whose commits creep toward the watchdog is the early warning that the
        watchdog is set too tight* - and a duration without its limit cannot be judged."""
        timings = StepTimings(watchdog_ms=2000.0, creep_fraction=0.5, run_id="r")
        with caplog.at_level("WARNING"):
            timings.record_commit(1200.0)
        assert any(
            record.__dict__.get("code") == "commit_duration_creep" for record in caplog.records
        )
        warned = next(
            r for r in caplog.records if r.__dict__.get("code") == "commit_duration_creep"
        )
        assert warned.__dict__["watchdog_ms"] == 2000.0
        assert warned.__dict__["fraction_of_watchdog"] == 0.6

    def test_a_healthy_commit_does_not_warn(self, caplog: pytest.LogCaptureFixture) -> None:
        timings = StepTimings(watchdog_ms=2000.0, creep_fraction=0.5)
        with caplog.at_level("WARNING"):
            timings.record_commit(40.0)
        assert [
            r for r in caplog.records if r.__dict__.get("code") == "commit_duration_creep"
        ] == []


class TestTheReportAnotherRepositorysGateReads:
    def _log_with(self, tmp_path: Path, durations: list[float]) -> Path:
        path = run_log.run_log_path(tmp_path, "hot-run")
        path.parent.mkdir(parents=True, exist_ok=True)
        for step, duration in enumerate(durations):
            block = run_log.step_block(
                run_id="hot-run",
                step=step,
                loss=1.0,
                objective="causal_lm",
                learning_rate=1e-5,
                wall_time_ms=100.0,
            )
            block["commit_duration_ms"] = duration
            block["quiesce_hold_ms"] = duration + 5.0
            run_log.append(path, block)
        return path

    def test_the_report_names_its_scale_and_sample(self, tmp_path: Path) -> None:
        """ADR-012, and this is the number miLLM's gate is set from: *never a bare
        millisecond figure.*"""
        path = self._log_with(tmp_path, [40.0] * 99 + [900.0])
        report = timings_module.report(path, conditions="serving concurrently, 1 card")
        assert report["sample_size"] == 100
        assert report["percentile"] == 99.9
        assert report["value_ms"] == 900.0
        assert report["watchdog_gate_ms"] == 9000.0
        assert "x4" in report["topology"], "a number without its topology is not a measurement"
        assert "concurrently" in report["conditions"]

    def test_no_measurement_is_None_rather_than_zero(self, tmp_path: Path) -> None:
        """Zero would be a measured instantaneous commit, and ten times zero is a watchdog
        of nothing - which is the standalone case, where there is no bracket to time."""
        path = self._log_with(tmp_path, [])
        run_log.append(
            path,
            run_log.step_block(
                run_id="hot-run",
                step=0,
                loss=1.0,
                objective="o",
                learning_rate=1e-5,
                wall_time_ms=1.0,
            ),
        )
        report = timings_module.report(path)
        assert report["value_ms"] is None
        assert report["watchdog_gate_ms"] is None
        assert report["sample_size"] == 0

    def test_the_percentile_is_the_one_F1_already_defines(self, tmp_path: Path) -> None:
        """F1's ``run_log.commit_duration_percentile`` is the single query, reading the
        authoritative log. A second percentile implementation here would be a second answer
        to the question another project's gate asks."""
        path = self._log_with(tmp_path, [10.0, 20.0, 30.0])
        assert timings_module.report(path, percentile=100.0)["value_ms"] == 30.0
        assert run_log.commit_duration_percentile(path, 100.0) == 30.0

    def test_the_held_request_count_travels_with_the_hold(self, harness: Harness) -> None:
        """A hold of 40 ms holding nothing and one holding 200 requests are different facts."""
        harness.attach()
        with barrier.commit_bracket(harness.client, harness.timings):
            pass
        assert harness.timings.held_requests == [3]


class TestTheWatchdogMirrorsTheMeasuredValue:
    """miForge holds miLLM's commit watchdog only to judge a commit against it, so it must be
    the value miLLM actually runs: **200 ms since miLLM `db44fea`** (10,000 full-parameter
    commits on the RTX 3090, p99.9 14.4 ms miLLM-side / 15.9 ms trainer-side, x10 rounded up).
    The provisional 2,000 ms made every creep warning fire ten times too late."""

    def test_the_default_is_200_ms(self) -> None:
        from core.config import Settings

        assert Settings(_env_file=None).hotupdate_commit_watchdog_ms == 200.0  # type: ignore[call-arg]

    def test_the_env_example_says_the_same(self) -> None:
        example = Path(__file__).resolve().parents[3] / ".env.example"
        lines = [
            line.strip()
            for line in example.read_text("utf-8").splitlines()
            if line.strip().startswith("HOTUPDATE_COMMIT_WATCHDOG_MS=")
        ]
        assert lines == ["HOTUPDATE_COMMIT_WATCHDOG_MS=200"]
