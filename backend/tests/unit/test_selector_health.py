"""Selection health and the three named signals  (AC-003.14; FR-003.10, E-17, E-18).

The two silent failures this feature exists to make visible, plus their scalar twin:

* **accepts everything** - the run trains on raw output and drifts toward whatever the model
  already produces; loss falls, the test score does not move, nothing errors;
* **rejects everything** - batches never fill and the run looks slow rather than broken;
* **a grader with no spread** - GRPO computes an advantage from the spread of rewards within
  a group, so a constant reward gives **zero advantage for every member** and the update is
  numerically nothing. The run burns GPU hours, records a loss, and learns nothing.

Each gets a **name**, on the record and on the room. The copy is compared as a **string**,
which is only possible because it is fixed here once - so F7 and F9 render one sentence
rather than inventing two.
"""

from __future__ import annotations

import pytest

from schemas.selector import Verdict, VerdictValue
from services.selectors.health import (
    GRADER_NO_SPREAD_SENTENCE,
    SELECTION_DEGENERATE_SENTENCE,
    compute_health,
    publish_signals,
    score_stats,
    signal_events,
)
from services.selectors.settings import SelectorSettings
from services.selectors.types import (
    SIGNAL_ACCEPTS_EVERYTHING,
    SIGNAL_NO_SPREAD,
    SIGNAL_REJECTS_EVERYTHING,
)

SETTINGS = SelectorSettings(
    selection_min_n_for_signals=5, grader_variance_epsilon=1e-9, _env_file=None
)


def verdicts(*values: str, label: str = "code") -> list[Verdict]:
    return [
        Verdict(value=VerdictValue(value), selector="s", label=label, reason="because")
        for value in values
    ]


class TestTheCountsAndTheRate:
    def test_the_rate_never_travels_without_n(self) -> None:
        """ADR-012: every number names its scale and sample. ``Accepted 3 of 5``."""
        health = compute_health(
            verdicts("accept", "accept", "accept", "reject", "reject"), settings=SETTINGS
        )
        assert (health.accepted, health.n) == (3, 5)
        assert health.acceptance_rate == pytest.approx(0.6)

    def test_undecided_is_counted_separately_and_is_not_accepted(self) -> None:
        health = compute_health(verdicts("accept", "undecided", "reject"), settings=SETTINGS)
        assert (health.accepted, health.rejected, health.undecided) == (1, 1, 1)
        assert health.acceptance_rate == pytest.approx(1 / 3)

    def test_an_empty_batch_has_a_zero_rate_and_an_n_that_says_why(self) -> None:
        health = compute_health([], settings=SETTINGS)
        assert (health.n, health.acceptance_rate) == (0, 0.0)
        assert health.signals == ()

    def test_the_reason_histogram_counts_codes(self) -> None:
        batch = verdicts("reject", "reject", label="rule_forbidden_term") + verdicts(
            "accept", label="rule_passed"
        )
        health = compute_health(batch, settings=SETTINGS)
        assert health.reason_histogram == {"rule_forbidden_term": 2, "rule_passed": 1}

    def test_admission_is_counted_through_is_admitted_only(self) -> None:
        """One place decides admission (foundation 3.1a); counting it a second way here
        would be a second answer."""
        health = compute_health(verdicts("undecided", "undecided"), settings=SETTINGS)
        assert health.accepted == 0


class TestTheThreeNamedSignals:
    def test_all_accept_raises_the_named_signal(self) -> None:
        health = compute_health(verdicts(*["accept"] * 6), settings=SETTINGS)
        assert health.signal_names == (SIGNAL_ACCEPTS_EVERYTHING,)

    def test_all_reject_raises_the_named_signal(self) -> None:
        health = compute_health(verdicts(*["reject"] * 6), settings=SETTINGS)
        assert health.signal_names == (SIGNAL_REJECTS_EVERYTHING,)

    def test_a_mixed_batch_raises_none(self) -> None:
        health = compute_health(
            verdicts("accept", "reject", "accept", "reject", "accept"), settings=SETTINGS
        )
        assert health.signals == ()

    def test_a_batch_with_pending_candidates_is_not_accepting_everything(self) -> None:
        """The claim would be false: the undecided candidates have not been accepted."""
        health = compute_health(
            verdicts("accept", "accept", "accept", "accept", "undecided"), settings=SETTINGS
        )
        assert health.signals == ()

    def test_a_batch_under_the_n_floor_raises_nothing(self) -> None:
        """A 1-of-1 batch is not evidence of anything."""
        health = compute_health(verdicts("accept", "accept"), settings=SETTINGS)
        assert health.signals == ()

    def test_zero_variance_raises_the_grader_signal(self) -> None:
        """E-18, the GRPO zero-advantage case."""
        health = compute_health(
            verdicts("accept", "reject", "accept", "reject", "accept"),
            settings=SETTINGS,
            scores=[0.5] * 5,
        )
        assert SIGNAL_NO_SPREAD in health.signal_names

    def test_real_spread_raises_no_grader_signal(self) -> None:
        health = compute_health(
            verdicts("accept", "reject", "accept", "reject", "accept"),
            settings=SETTINGS,
            scores=[0.1, 0.4, 0.6, 0.8, 0.95],
        )
        assert SIGNAL_NO_SPREAD not in health.signal_names

    def test_a_grader_batch_under_the_floor_raises_nothing(self) -> None:
        health = compute_health(verdicts("accept", "reject"), settings=SETTINGS, scores=[0.5, 0.5])
        assert health.signals == ()

    def test_no_scores_means_no_grader_stats_and_no_grader_signal(self) -> None:
        """An empty list and an absent one are different facts."""
        health = compute_health(verdicts(*["accept"] * 6), settings=SETTINGS, scores=None)
        assert health.grader_stats is None
        assert SIGNAL_NO_SPREAD not in health.signal_names


class TestTheCopyIsFixedHereOnce:
    def test_the_selector_sentence_is_verbatim_from_the_requirement(self) -> None:
        """R-02.39's own words, which BRD-02 quotes as the rationale for the dry-run."""
        assert SELECTION_DEGENERATE_SENTENCE == (
            "A selector that accepts everything is not selecting; one that rejects "
            "everything will stall the run."
        )

    @pytest.mark.parametrize(
        ("values", "signal"),
        [(["accept"] * 6, SIGNAL_ACCEPTS_EVERYTHING), (["reject"] * 6, SIGNAL_REJECTS_EVERYTHING)],
        ids=["accepts-everything", "rejects-everything"],
    )
    def test_both_selector_signals_carry_that_sentence(self, values, signal) -> None:
        health = compute_health(verdicts(*values), settings=SETTINGS)
        found = {item.name: item.message for item in health.signals}
        assert found[signal] == SELECTION_DEGENERATE_SENTENCE

    def test_the_grader_signal_carries_its_own_sentence(self) -> None:
        """Two sentences, not one. The accept/reject sentence does not describe this case -
        a grader with no spread is not accepting or rejecting anything - and *a signal shown
        under a sentence that does not describe it is worse than no sentence.*"""
        health = compute_health(
            verdicts("accept", "reject", "accept", "reject", "accept"),
            settings=SETTINGS,
            scores=[0.5] * 5,
        )
        found = {item.name: item.message for item in health.signals}
        assert found[SIGNAL_NO_SPREAD] == GRADER_NO_SPREAD_SENTENCE
        assert found[SIGNAL_NO_SPREAD] != SELECTION_DEGENERATE_SENTENCE

    def test_the_two_sentences_are_different_strings(self) -> None:
        assert GRADER_NO_SPREAD_SENTENCE != SELECTION_DEGENERATE_SENTENCE


class TestTheSignalCarriesItsCounts:
    def test_the_accepts_everything_signal_carries_the_counts(self) -> None:
        health = compute_health(verdicts(*["accept"] * 7), settings=SETTINGS)
        details = health.signals[0].details
        assert details == {"accepted": 7, "n": 7, "acceptance_rate": 1.0}

    def test_the_grader_signal_carries_the_variance_and_the_epsilon(self) -> None:
        health = compute_health(
            verdicts("accept", "reject", "accept", "reject", "accept"),
            settings=SETTINGS,
            scores=[0.25] * 5,
        )
        details = {item.name: item.details for item in health.signals}[SIGNAL_NO_SPREAD]
        assert details["variance"] == 0.0
        assert details["epsilon"] == 1e-9
        assert details["min"] == details["max"] == 0.25


class TestTheEmit:
    def test_the_payload_and_the_call_count_are_both_asserted(self) -> None:
        """ADR-016: "was called" passes against a call sending the wrong arguments."""
        health = compute_health(verdicts(*["reject"] * 6), settings=SETTINGS)
        calls: list[tuple[str, dict]] = []
        emitted = publish_signals(health, run_id="run-1", emit=lambda e, p: calls.append((e, p)))

        assert emitted == [SIGNAL_REJECTS_EVERYTHING]
        assert len(calls) == 1, "one emit per signal, not per verdict"
        event, payload = calls[0]
        assert event == SIGNAL_REJECTS_EVERYTHING
        assert payload["signal"] == SIGNAL_REJECTS_EVERYTHING
        assert payload["message"] == SELECTION_DEGENERATE_SENTENCE
        assert payload["n"] == 6
        assert payload["rejected"] == 6
        assert payload["acceptance_rate"] == 0.0

    def test_a_clean_batch_emits_nothing(self) -> None:
        health = compute_health(
            verdicts("accept", "reject", "accept", "reject", "accept"), settings=SETTINGS
        )
        calls: list = []
        assert publish_signals(health, run_id=None, emit=lambda e, p: calls.append(e)) == []
        assert calls == []

    def test_the_record_holds_the_signal_even_with_no_emitter(self) -> None:
        """ADR-007: the record is the truth and the socket is a convenience.

        miStudio reaped a live 5.8-hour job as "worker lost" because progress reported only
        over a socket is invisible to anything that was not listening.
        """
        health = compute_health(verdicts(*["accept"] * 6), settings=SETTINGS)
        assert publish_signals(health, run_id=None, emit=None) == [SIGNAL_ACCEPTS_EVERYTHING]
        assert health.signal_names == (SIGNAL_ACCEPTS_EVERYTHING,)

    def test_signal_events_is_pure_so_the_payload_is_testable(self) -> None:
        health = compute_health(verdicts(*["accept"] * 6), settings=SETTINGS)
        assert [name for name, _ in signal_events(health)] == [SIGNAL_ACCEPTS_EVERYTHING]


class TestDetectionLivesHereNotInAPlugin:
    def test_no_plugin_module_computes_a_degenerate_signal(self) -> None:
        """A plugin sees one completion; the property is a property of a batch. Putting it
        in a plugin is how it ends up implemented five times and tested once.
        """
        from pathlib import Path

        plugins = Path(__file__).resolve().parents[2] / "src" / "services" / "selectors" / "plugins"
        offenders = [
            path.name
            for path in plugins.glob("*.py")
            for signal in (SIGNAL_ACCEPTS_EVERYTHING, SIGNAL_REJECTS_EVERYTHING, SIGNAL_NO_SPREAD)
            if signal in path.read_text(encoding="utf-8")
        ]
        assert offenders == [], f"degenerate detection leaked into plugins: {offenders}"


class TestScoreStats:
    def test_the_population_variance_is_used(self) -> None:
        """The batch IS the population whose spread the objective uses."""
        stats = score_stats([0.0, 1.0])
        assert stats is not None
        assert stats.variance == pytest.approx(0.25)

    def test_a_single_value_has_zero_variance(self) -> None:
        stats = score_stats([0.7])
        assert stats is not None
        assert (stats.n, stats.variance, stats.min, stats.max) == (1, 0.0, 0.7, 0.7)

    def test_an_empty_sequence_gives_none_rather_than_zeros(self) -> None:
        assert score_stats([]) is None


class TestTheSocketEmitterReachesTheRunRoom:
    """FTASKS 8.4: on the record **and** on ``miforge/run/<id>``.

    The socket is the convenience and the record is the truth (ADR-007) - but a convenience
    that was never wired is a feature that does not exist, so the emitter is exercised
    rather than assumed. miStudio reaped a live 5.8-hour job as "worker lost" because
    progress reported only over a socket is invisible to anything not listening; the mirror
    of that lesson is that a socket nobody tested is invisible to everyone.
    """

    def test_it_emits_each_signal_into_the_runs_room(self, monkeypatch) -> None:
        from services.selectors.health import socket_emitter

        sent: list[tuple[str, str, dict]] = []

        async def fake_emit(run_id: str, event: str, payload: dict) -> None:
            sent.append((run_id, event, payload))

        monkeypatch.setattr("api.sockets.emit_run_event", fake_emit)

        health = compute_health(verdicts(*["accept"] * 6), settings=SETTINGS)
        emitted = publish_signals(health, run_id="run-7", emit=socket_emitter("run-7"))

        assert emitted == [SIGNAL_ACCEPTS_EVERYTHING]
        assert len(sent) == 1, "one emit per signal"
        run_id, event, payload = sent[0]
        assert run_id == "run-7"
        assert event == SIGNAL_ACCEPTS_EVERYTHING
        assert payload["message"] == SELECTION_DEGENERATE_SENTENCE
        assert payload["n"] == 6

    def test_the_room_spelling_is_the_foundations(self) -> None:
        """ADR-007 fixes it and F9's store joins by it; never built by string here."""
        from api.sockets import room_for

        assert room_for("run-7") == "miforge/run/run-7"

    def test_a_failing_emit_does_not_fail_the_evaluation(self, monkeypatch) -> None:
        """The record already holds the signal. Losing the socket must not fail an
        evaluation whose verdicts are correct."""
        from services.selectors.health import socket_emitter

        async def exploding(run_id: str, event: str, payload: dict) -> None:
            raise RuntimeError("no redis")

        monkeypatch.setattr("api.sockets.emit_run_event", exploding)
        health = compute_health(verdicts(*["reject"] * 6), settings=SETTINGS)
        assert publish_signals(health, run_id="run-7", emit=socket_emitter("run-7")) == [
            SIGNAL_REJECTS_EVERYTHING
        ]
        assert health.signal_names == (SIGNAL_REJECTS_EVERYTHING,)
