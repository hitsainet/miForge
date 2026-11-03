"""The floor: the four lines this feature exists to protect  (FR-004.23, 25, 26, 27; E-12, E-13).

**M-1 and M-3 live here**, and they are the sharpest controls in the feature:

* invert ``score < floor`` and **every run passes its floor forever** - the loss curve still
  falls, the test score still rises, the suite stays green, and general capability is spent
  invisibly. That is the project risk the PPRD names with this feature as its only
  mitigation;
* default a missing floor to ``0.0`` and a fabricated threshold passes every run - ADR-018's
  named anti-pattern in the place it costs most.

Both failures are invisible on every surface this feature produces, which is why they are
negative controls and not code-review notes.
"""

from __future__ import annotations

import pytest

from core.errors import Refusal
from providers.base import ProviderIdentity
from services.scoring import retention
from services.scoring.settings import exposure_window_steps
from services.scoring.tables import FLOOR_LAYER_NONE, FLOOR_LAYER_PACK, FLOOR_LAYER_RUN

PROVIDER = ProviderIdentity(kind="in_process", model="lfm2-1.2b")


def floor(value: float | None, layer: str = FLOOR_LAYER_PACK) -> retention.ResolvedFloor:
    return retention.ResolvedFloor(value=value, layer=layer)


def evaluate(score: float, resolved: retention.ResolvedFloor):
    return retention.evaluate_floor(
        score,
        resolved,
        run_id="run-1",
        step=42,
        n=180,
        provider=PROVIDER,
        set_ref="retention_set/general@1",
    )


class TestTheComparisonIsStrict:
    def test_below_the_floor_is_a_breach(self) -> None:
        signal = evaluate(0.70, floor(0.72))
        assert signal is not None
        assert (signal.score, signal.floor) == (0.70, 0.72)

    def test_EQUAL_to_the_floor_is_NOT_a_breach(self) -> None:
        """E-13, stated in the design so it is not left to whoever writes the line.

        A run sitting exactly on its floor has not fallen below it, and stopping it would be
        the mirror defect of M-1: a false positive on the one signal an operator must be able
        to trust.
        """
        assert evaluate(0.72, floor(0.72)) is None

    def test_above_the_floor_is_not_a_breach(self) -> None:
        assert evaluate(0.80, floor(0.72)) is None

    def test_M1_THE_INVERSION_WOULD_FAIL_HERE(self) -> None:
        """M-1's assertion written against the defect, not against the fix.

        With ``score > floor`` the first assertion returns ``None`` and the second returns a
        signal - so this one test fails in **both** directions, which is what makes the
        mutation control unambiguous rather than merely likely.
        """
        assert evaluate(0.10, floor(0.90)) is not None, "a clear breach must be detected"
        assert evaluate(0.90, floor(0.10)) is None, "a clear pass must not be a breach"

    def test_none_returned_rather_than_a_signal_with_a_false_flag(self) -> None:
        """A caller cannot publish a non-breach by forgetting to read a boolean."""
        assert evaluate(0.99, floor(0.10)) is None


class TestAMissingFloorRefuses:
    def test_a_null_floor_at_evaluation_refuses(self) -> None:
        """M-3. ``floor or 0.0`` here is that mutation shipped on purpose."""
        with pytest.raises(Refusal) as caught:
            evaluate(0.50, floor(None, FLOOR_LAYER_NONE))
        assert caught.value.code == "retention_floor_missing"

    def test_a_null_floor_is_never_read_as_nothing_to_breach(self) -> None:
        with pytest.raises(Refusal) as caught:
            evaluate(0.0, floor(None, FLOOR_LAYER_NONE))
        assert "nothing to breach" in caught.value.message

    def test_resolution_refuses_a_retention_set_with_no_floor(self) -> None:
        """E-12, at **submission** - R-02.48's *"says so before it starts"*."""
        with pytest.raises(Refusal) as caught:
            retention.resolve_floor(kind="retention", pack_slot_floor=None, run_override_floor=None)
        assert caught.value.code == "retention_floor_missing"
        assert "0.0 passes every run" in caught.value.message

    def test_a_test_set_legitimately_has_no_floor(self) -> None:
        resolved = retention.resolve_floor(
            kind="test", pack_slot_floor=None, run_override_floor=None
        )
        assert (resolved.value, resolved.layer) == (None, FLOOR_LAYER_NONE)


class TestTheFloorsLayerIsRecorded:
    """FPRD OQ-2 / DEF-7 is unsettled, so the record says what was **in force**."""

    def test_a_pack_slot_floor_records_its_layer(self) -> None:
        resolved = retention.resolve_floor(
            kind="retention", pack_slot_floor=0.72, run_override_floor=None
        )
        assert (resolved.value, resolved.layer) == (0.72, FLOOR_LAYER_PACK)

    def test_a_run_override_wins_and_records_its_layer(self) -> None:
        resolved = retention.resolve_floor(
            kind="retention", pack_slot_floor=0.72, run_override_floor=0.65
        )
        assert (resolved.value, resolved.layer) == (0.65, FLOOR_LAYER_RUN)

    def test_the_layer_travels_on_the_breach_signal(self) -> None:
        signal = evaluate(0.50, floor(0.72, FLOOR_LAYER_RUN))
        assert signal is not None and signal.floor_layer == FLOOR_LAYER_RUN

    @pytest.mark.parametrize("bad", [-0.1, 1.1])
    def test_a_floor_outside_the_schemas_range_refuses(self, bad: float) -> None:
        """It could not be written to a manifest, so it is refused rather than clamped."""
        with pytest.raises(Refusal) as caught:
            retention.resolve_floor(kind="retention", pack_slot_floor=bad, run_override_floor=None)
        assert caught.value.code == "retention_floor_out_of_range"


class TestTheBreachSignalSaysEverythingAnOperatorNeeds:
    def test_the_payload_carries_score_floor_n_provider_step_and_set(self) -> None:
        """AC-004.9's field list, asserted field by field rather than as a shape."""
        signal = evaluate(0.60, floor(0.72))
        assert signal is not None
        payload = signal.payload()
        assert payload["event"] == "retention_floor_breached"
        assert payload["run_id"] == "run-1"
        assert payload["step"] == 42
        assert payload["kind"] == "retention"
        assert payload["score"] == 0.60
        assert payload["floor"] == 0.72
        assert payload["n"] == 180
        assert payload["provider"] == "in_process:lfm2-1.2b"
        assert payload["set"] == "retention_set/general@1"

    def test_the_message_names_the_sample_and_says_what_happens_next(self) -> None:
        """ADR-012 and ADR-021 together: every number names its sample, and the sentence
        says what to do next rather than how to feel."""
        signal = evaluate(0.60, floor(0.72))
        assert signal is not None
        message = signal.message()
        assert "180 prompts" in message
        assert "stop condition" in message
        assert "Continue from the last checkpoint" in message

    def test_the_payload_carries_no_credential(self) -> None:
        signal = evaluate(0.60, floor(0.72))
        assert signal is not None
        assert "api_key" not in repr(signal.payload())


class TestTheExposureWindowIsANumber:
    """FR-004.26, AC-004.15. **Reported, never implied.**"""

    def test_a_cadence_of_one_exposes_nothing(self) -> None:
        assert exposure_window_steps(1) == 0

    def test_a_coarse_cadence_reports_n_minus_one(self) -> None:
        """With a cadence of 10, **nine** steps of training can run on a model already below
        its floor - and an operator who was not told that reads the floor as a guarantee it
        is not."""
        assert exposure_window_steps(10) == 9

    def test_a_cadence_below_one_is_not_a_cadence(self) -> None:
        with pytest.raises(ValueError):
            exposure_window_steps(0)

    def test_the_default_retention_cadence_is_every_step(self) -> None:
        """FR-004.26's safe default: the consequence is a safety property, not a preference."""
        from services.scoring.settings import ScoringSettings

        assert ScoringSettings().retention_every == 1


class TestTheWorkerHoldsValidatesLine:
    """Found 2026-10-18 (F11 window 2): a run with no retention slot passed validate (E-12
    is deliberately deferred there) and was refused ``retention_floor_missing`` in the
    worker, after a lease. The floor is required where a retention set exists - not
    otherwise."""

    @staticmethod
    def _spec(definition):  # type: ignore[no-untyped-def]
        from services.scoring.entrypoint import spec_from_definition

        def loader(raw):  # type: ignore[no-untyped-def]
            rows = [{"prompt": "p", "completion": "c"}]
            return rows, None, {"kind": "test", "name": "t", "version": "1"}

        return spec_from_definition(definition, run_id="r", load_rows=loader)

    def test_no_retention_set_and_no_floor_is_not_refused(self) -> None:
        spec = self._spec({})
        assert spec.floor.value is None

    def test_a_retention_set_without_a_floor_is_still_refused(self) -> None:
        with pytest.raises(Refusal) as caught:
            self._spec({"retention_slot": {"local_path": "/x"}})
        assert caught.value.code == "retention_floor_missing"
