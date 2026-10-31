"""The acquired parameters go on the leased card, and the record says which  (ADR-008, ADR-023).

**The defect this file exists for.** Nothing in ``backend/src/`` ever moved a model onto a
GPU. ``local_dir._load_pretrained`` called ``from_pretrained`` with no ``device_map``,
``scratch._instantiate`` called ``from_config``, ``training/tasks.py`` placed nothing - and
``objectives/sft.py`` builds its batch tensors on *the model's* device, so a run resolved a
card by UUID, **took its lease**, and then computed on the CPU without raising anything.
``grep -rE 'device_map|\\.to\\("cuda"\\)|\\.cuda\\(\\)' backend/src/`` returned zero hits.

The only visible trace was in the suite itself: all three GPU integration tests wrote
``acquired.module.to("cuda")`` by hand before stepping. **A test hand-doing what production
omits is the signal**, which is why those three now assert the placement instead of
performing it.

## Why these cases are meaningful on a machine with no card

Every branch of the decision is driven through :class:`training.acquisition.placement.CudaProbe`,
a protocol over this process's CUDA view. So the ``cuda`` branches are exercised here on a
CPU-only box with a fake probe, asserting **the decision and the recorded device** rather
than that CUDA was used. A placement defect that is only testable on the GPU node is a
placement defect that survives a green suite - which is precisely what happened.

The controls that prove these tests bite are in
``tests/unit/test_acquisition_placement_is_wired.py``.
"""

from __future__ import annotations

import logging

import pytest

from core.errors import Refusal
from training.acquisition import placement
from training.acquisition.placement import (
    CUDA_UNAVAILABLE,
    GPU_NOT_PINNED,
    KIND_CPU,
    KIND_CUDA,
    LEASED_UUID,
    SOLE_VISIBLE_DEVICE,
    Decision,
    decide,
    normalise_device_uuid,
    place,
)

CARD_A = "GPU-aaaaaaaa-1111-2222-3333-444444444444"
CARD_B = "GPU-bbbbbbbb-5555-6666-7777-888888888888"


class FakeDevice:
    """Stands in for ``torch.device``, carrying the two fields the read-back reads."""

    def __init__(self, kind: str, index: int | None = None) -> None:
        self.type = kind
        self.index = index

    def __repr__(self) -> str:  # pragma: no cover - diagnostics only
        return f"device(type={self.type!r}, index={self.index!r})"


class FakeProbe:
    """A :class:`CudaProbe` whose whole CUDA view is declared by the test.

    ``uuids`` is the driver's answer **per index**, so an inverted enumeration is
    expressible - which is the incident ADR-008 was written after and therefore the case
    most worth being able to write down.
    """

    def __init__(self, uuids: list[str | None] | None = None) -> None:
        self._uuids = uuids or []
        self.devices_handed_out: list[int] = []

    def is_available(self) -> bool:
        return bool(self._uuids)

    def device_count(self) -> int:
        return len(self._uuids)

    def uuid_of(self, index: int) -> str | None:
        return self._uuids[index]

    def device(self, index: int):
        self.devices_handed_out.append(index)
        return FakeDevice(KIND_CUDA, index)


class FakeParam:
    def __init__(self, device) -> None:
        self.device = device

    def numel(self) -> int:
        return 7


class FakeModule:
    """A module that records what it was asked to move to, and really 'moves'."""

    def __init__(self, device=None, *, obey: bool = True) -> None:
        self._device = device if device is not None else FakeDevice(KIND_CPU)
        self._obey = obey
        self.moved_to: list = []

    def to(self, target):
        self.moved_to.append(target)
        if self._obey:
            self._device = target
        return self

    def parameters(self):
        return iter([FakeParam(self._device)])


# ======================================================================= the decision


class TestTheUuidSpellingIsNormalisedInOnePlace:
    """``torch`` reports a bare UUID and ``nvidia-smi`` prefixes it, and the two are
    compared. This project has paid four times for one value with two spellings."""

    @pytest.mark.parametrize(
        "given",
        [
            "GPU-AAAAAAAA-1111-2222-3333-444444444444",
            "gpu-aaaaaaaa-1111-2222-3333-444444444444",
            "aaaaaaaa-1111-2222-3333-444444444444",
            "  AAAAAAAA-1111-2222-3333-444444444444 ",
        ],
    )
    def test_every_spelling_normalises_to_nvidia_smis_form(self, given: str) -> None:
        assert normalise_device_uuid(given) == CARD_A

    def test_a_torch_reported_uuid_matches_a_submitted_one(self) -> None:
        """The join these two spellings have to make: the spec carries ``nvidia-smi``'s
        form and the probe reports ``torch``'s. If they did not normalise to one string the
        leased card would never be found and every run would refuse."""
        probe = FakeProbe([normalise_device_uuid("aaaaaaaa-1111-2222-3333-444444444444")])
        assert decide(CARD_A, probe=probe).uuid == CARD_A


class TestANamedCardIsMatchedByUuidAndNeverByIndex:
    def test_the_leased_uuid_is_found_wherever_the_driver_enumerated_it(self) -> None:
        """**The inverted-enumeration case.** The leased card is index 1 here, so a
        placement that took index 0 would run the work on the other card - which is the
        incident, verbatim: *nothing failed; the work just ran on the wrong device.*"""
        probe = FakeProbe([CARD_B, CARD_A])
        decision = decide(CARD_A, probe=probe)

        assert decision == Decision(kind=KIND_CUDA, uuid=CARD_A, index=1, reason=LEASED_UUID)

    def test_a_named_card_that_is_not_visible_is_refused_and_never_replaced(self) -> None:
        """*A run pinned to a card it did not get is unreproducible*, and the card it would
        fall back to is held by another run's lease."""
        with pytest.raises(Refusal) as caught:
            decide(CARD_A, probe=FakeProbe([CARD_B]))

        assert caught.value.code == "leased_gpu_not_visible"
        assert caught.value.details["visible"] == [CARD_B]
        assert caught.value.details["request"] == CARD_A

    def test_a_named_card_with_no_cuda_at_all_is_refused_rather_than_run_on_the_cpu(
        self,
    ) -> None:
        """The one case acquisition *can* close without knowing the lease: the operator
        asked for a specific card. Answering with the CPU would be the substitution
        R-02.48 and ADR-013 both forbid - and it is the defect this module fixes,
        reappearing one layer up."""
        with pytest.raises(Refusal) as caught:
            decide(CARD_A, probe=FakeProbe([]))

        assert caught.value.code == "leased_gpu_not_visible"
        assert caught.value.details["visible"] == []


class TestAutoIsUnambiguousOnlyWithOneVisibleDevice:
    def test_one_visible_device_is_taken_and_its_uuid_read_back(self) -> None:
        decision = decide("auto", probe=FakeProbe([CARD_A]))
        assert decision == Decision(
            kind=KIND_CUDA, uuid=CARD_A, index=0, reason=SOLE_VISIBLE_DEVICE
        )

    def test_auto_with_several_visible_devices_FAILS_CLOSED(self) -> None:
        """**ADR-013: fail closed where correctness is at stake.** ``auto`` was resolved to
        a UUID at submit and a lease was taken on it, but neither reaches this layer -
        ``run_driver`` builds the spec from ``definition.gpu_request``, still ``"auto"``,
        while the resolved value sits on ``mf_runs.gpu_uuid``. Taking the first enumerated
        device would place the work on a card another run holds, with both leases claiming
        exclusivity and both runs' numbers wrong.

        The refusal names the owed wiring, which is the part that makes it actionable
        rather than merely safe.
        """
        with pytest.raises(Refusal) as caught:
            decide("auto", probe=FakeProbe([CARD_A, CARD_B]))

        assert caught.value.code == GPU_NOT_PINNED
        assert caught.value.details["devices_visible"] == 2
        assert caught.value.details["visible"] == [CARD_A, CARD_B]
        assert "mf_runs.gpu_uuid" in caught.value.message, (
            "the refusal must name where the leased UUID lives; a fail-closed guard that "
            "does not say what to do next is an outage with no remedy (ADR-021)"
        )

    def test_a_driver_that_will_not_name_the_card_is_refused(self) -> None:
        """A card the record cannot name is a card a score cannot name either (R-02.16),
        and ``device_uuid=None`` already means *CPU* to ``in_process._check_budget``.
        Recording 'unknown' while using it would make those two states one."""
        with pytest.raises(Refusal) as caught:
            decide("auto", probe=FakeProbe([None]))

        assert caught.value.code == "gpu_uuid_unreadable"


class TestTheRequestItselfIsClassifiedAgainstOnePattern:
    @pytest.mark.parametrize("asked", ["0", "cuda:0", "01:00.0", "GPU-nope", "the-fast-one"])
    def test_anything_that_is_neither_auto_nor_a_uuid_is_refused_by_name(self, asked: str) -> None:
        """Against ``core.gpu.UUID_PATTERN`` - **the one pattern** - rather than with a
        second taxonomy of indices and bus addresses, which ``core.gpu.resolve_device``
        owns. A second classifier is a second answer."""
        with pytest.raises(Refusal) as caught:
            decide(asked, probe=FakeProbe([CARD_A]))

        assert caught.value.code == "gpu_request_not_a_uuid"
        assert caught.value.details["request"] == asked

    def test_an_empty_request_is_refused_rather_than_treated_as_auto(self) -> None:
        with pytest.raises(Refusal) as caught:
            decide("", probe=FakeProbe([CARD_A]))
        assert caught.value.code == "gpu_request_empty"


# ======================================================================= the placement


class TestPlacingMovesTheModuleAndReadsBackWhereItLanded:
    def test_the_module_is_moved_to_the_decided_device(self) -> None:
        probe = FakeProbe([CARD_B, CARD_A])
        module = FakeModule()

        placed = place(module, CARD_A, probe=probe)

        assert len(module.moved_to) == 1, "the move happens exactly once"
        assert module.moved_to[0].type == KIND_CUDA
        assert module.moved_to[0].index == 1, "the index of the LEASED card, not zero"
        assert probe.devices_handed_out == [1]
        assert placed.device_kind == KIND_CUDA
        assert placed.device_uuid == CARD_A
        assert placed.reason == LEASED_UUID

    def test_a_module_that_does_not_actually_move_is_REFUSED(self) -> None:
        """**The positive check that the measurement happened.** A ``.to(device)`` whose
        result is never inspected is not evidence of anything - this project's
        false-measurement family is four members long and every one was a signal
        structurally indistinguishable from the thing it was mistaken for. So the device is
        read off a real parameter afterwards, and a module whose tensors stayed behind is
        refused rather than recorded as placed."""
        module = FakeModule(obey=False)

        with pytest.raises(Refusal) as caught:
            place(module, CARD_A, probe=FakeProbe([CARD_A]))

        assert caught.value.code == "model_placement_failed"
        assert caught.value.details["device_uuid"] == CARD_A
        assert "cpu" in caught.value.details["landed_on"]

    def test_a_module_that_will_not_say_where_its_parameters_are_is_REFUSED(self) -> None:
        class Opaque:
            def to(self, target):
                return self

            def parameters(self):
                return iter([])

        with pytest.raises(Refusal) as caught:
            place(Opaque(), CARD_A, probe=FakeProbe([CARD_A]))

        assert caught.value.code == "model_placement_unverifiable"

    def test_a_module_that_raises_on_the_move_is_refused_by_name(self) -> None:
        class Immovable:
            def to(self, target):
                raise RuntimeError("CUDA out of memory")

            def parameters(self):  # pragma: no cover - never reached
                return iter([])

        with pytest.raises(Refusal) as caught:
            place(Immovable(), CARD_A, probe=FakeProbe([CARD_A]))

        assert caught.value.code == "model_placement_failed"
        assert caught.value.details["error"] == "RuntimeError"


class TestTheNoGpuCaseIsCpuAndIsNamedRatherThanSilent:
    """**ADR-023 and ADR-013 pull in opposite directions here and both are honoured.**

    ADR-023 requires a developer without a GPU to run everything except a training step, and
    this codebase already treats a CPU-resident parameter set as legitimate -
    ``model_handle.free_vram_bytes()`` returns ``0`` for one *by design*. So acquisition
    must not refuse. ADR-013 says never substitute a different behaviour for a missing
    service. So the CPU outcome is **recorded as the CPU** and logged at warning: it is
    reported, it is never claimable as a card, and the one case this layer cannot refuse is
    named in the warning rather than hidden.
    """

    def test_no_cuda_device_places_nothing_and_records_the_cpu(self) -> None:
        module = FakeModule()

        placed = place(module, "auto", probe=FakeProbe([]))

        assert placed.device_kind == KIND_CPU
        assert placed.device_uuid is None, (
            "a CPU placement must never carry a device UUID: that value is what a score's "
            "provider provenance reports (R-02.16), so a UUID here is a claim that a card "
            "produced the number"
        )
        assert placed.reason == CUDA_UNAVAILABLE
        assert module.moved_to == [], "nothing is moved; the parameters are already here"

    def test_the_cpu_outcome_is_logged_at_WARNING_and_names_the_consequence(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        """A silent CPU run is the defect being fixed, reappearing one layer up. The log
        line is the difference between 'degraded' and 'degraded and nobody knows'."""
        with caplog.at_level(logging.WARNING, logger=placement.__name__):
            place(FakeModule(), "auto", probe=FakeProbe([]))

        records = [r for r in caplog.records if getattr(r, "code", None) == "model_placed_on_cpu"]
        assert len(records) == 1, f"expected exactly one named warning, got {caplog.records}"
        consequence = records[0].consequence
        assert "mf_runs.gpu_uuid" in consequence, (
            "the warning must name the case this layer cannot refuse - gpu_request='auto' "
            "while a lease is held - rather than leaving a reader to rediscover it"
        )

    def test_a_cpu_outcome_tolerates_a_module_with_no_to_at_all(self) -> None:
        """Nothing moves on this path, so a stub module is a legitimate caller. Three
        existing tests in ``test_acquisition_scratch.py`` monkeypatch ``_instantiate`` to
        return ``object()``, and a placement that demanded ``.to`` would have broken them -
        which would have been a fixture problem masquerading as a design constraint."""
        placed = place(object(), "auto", probe=FakeProbe([]))
        assert placed.device_kind == KIND_CPU


class TestPlacementOnARealTorchModule:
    """One pass through the **real** ``torch`` runtime, so the fakes above are not the only
    thing that has ever executed this code. Asserts the recorded device, not that CUDA was
    used, so it runs here and on the GPU node alike."""

    def test_a_real_module_reports_a_real_device_after_placement(self) -> None:
        torch = pytest.importorskip("torch")

        module = torch.nn.Linear(4, 4)
        probe = placement._TorchCudaProbe(torch)

        if probe.is_available():  # pragma: no cover - needs a card
            placed = place(module, "auto", probe=FakeProbe([probe.uuid_of(0)]))
            assert placed.device_kind == KIND_CUDA
            assert next(module.parameters()).device.type == KIND_CUDA
        else:
            placed = place(module, "auto", probe=probe)
            assert placed.device_kind == KIND_CPU
            assert placed.device_uuid is None
            assert next(module.parameters()).device.type == KIND_CPU

    def test_the_real_probe_reports_this_machine_consistently(self) -> None:
        """``is_available()`` and ``device_count()`` must agree, because
        :func:`decide` treats a disagreement as *no card* and would then place on the CPU
        while a card was present."""
        torch = pytest.importorskip("torch")
        probe = placement._TorchCudaProbe(torch)
        assert probe.is_available() == (probe.device_count() > 0)
