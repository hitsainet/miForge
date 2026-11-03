"""An existing attachment refuses at PRE-FLIGHT, by the same name the attach would give.

Found live 2026-10-18 (F11 14.8, run f11w2-r5): ``POST /runs/validate`` answered ``ok`` and
``mode_available`` over a STALE attachment, and the refusal arrived only from the worker -
after a lease and a queue slot. The status read pre-flight already makes carries the
attachment, so R-02.33's first row (held, E-1) and third row (stale, E-3) are refused there,
with the codes, holder and operator action ``refusals.interpret`` gives the attach itself.
"""

from __future__ import annotations

import pytest

from core.errors import Refusal
from tests.support.fixtures_hotupdate import Harness
from training.acquisition.base import AcquisitionMode, AcquisitionSpec


def _preflight(harness: Harness):  # type: ignore[no-untyped-def]
    from training.hotupdate.attach import AttachAcquirer

    acquirer = AttachAcquirer(transport=harness.fake.transport())
    return acquirer.preflight(AcquisitionSpec(mode=AcquisitionMode.ATTACH))


def test_a_stale_attachment_refuses_with_the_operator_action(harness: Harness) -> None:
    harness.fake.make_stale("trainer:miforge@lnxhost02")
    harness.fake.calls.clear()
    with pytest.raises(Refusal) as caught:
        _preflight(harness)
    refusal = caught.value
    assert refusal.code == "attachment_stale_operator_action_required"
    assert refusal.details["holder"] == "trainer:miforge@lnxhost02"
    assert refusal.details["operator_action"] == "POST /api/weights/clear-stale (operator identity)"
    assert refusal.details["miforge_can_perform_it"] is False
    assert refusal.details["stage"] == "preflight"
    assert harness.fake.call_count("attach") == 0
    assert harness.fake.call_count("clear-stale") == 0, "R-02.34: never clear it ourselves"


def test_an_active_attachment_held_by_another_trainer_refuses_naming_it(harness: Harness) -> None:
    harness.fake.hold("trainer:nightly-forge")
    with pytest.raises(Refusal) as caught:
        _preflight(harness)
    assert caught.value.code == "attachment_held_by_other"
    assert caught.value.details["holder"] == "trainer:nightly-forge"
    assert "operator_action" not in caught.value.details
    assert harness.fake.call_count("detach") == 0, "R-02.33: never force a detach"


def test_nothing_attached_passes(harness: Harness) -> None:
    """The positive control: without it, 'refuses' could be unconditional."""
    _preflight(harness)


@pytest.mark.parametrize("label", ["Q4", "Q8", "q2", "GPTQ", "AWQ", "BITNET"])
def test_a_quantized_resident_model_is_named_not_attachable(harness: Harness, label: str) -> None:
    """R-02.33 row 2 (E-2). Found live 2026-10-18: over miLLM's Q4 GGUF model, pre-flight
    said ``attach_resident_size_unknown`` - true, and the wrong diagnosis."""
    harness.fake.quantization = label
    harness.fake.num_parameters = None  # as live: a GGUF model reports no count
    with pytest.raises(Refusal) as caught:
        _preflight(harness)
    assert caught.value.code == "resident_model_not_attachable"
    assert caught.value.details["quantization"] == label
    assert "FP16 or BF16" in caught.value.message
    assert harness.fake.call_count("attach") == 0


@pytest.mark.parametrize("label", ["FP16", "FP32", None, "SOMETHING-NEW"])
def test_an_unquantized_or_unknown_label_is_not_refused_for_quantization(
    harness: Harness, label: str | None
) -> None:
    """The positive control, and the unknown case: an absent or unrecognised label is left
    to the attach's own NOT_ATTACHABLE rather than refused on a guess."""
    harness.fake.quantization = label
    _preflight(harness)
