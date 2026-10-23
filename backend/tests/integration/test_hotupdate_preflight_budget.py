"""An attach pre-flight yields a spec R-02.42's budget guard can estimate  (found live).

F11 window 2 (2026-10-17): every attach submission to ``POST /runs/define`` was refused
``budget_spec_incomplete``: a load fills ``param_count`` from the weights on disk, an attach has
no files, and the attach pre-flight returned the spec unchanged - so the guard had neither a
count nor a config. The pre-flight now reads the resident model's ``num_parameters`` from
miLLM's catalogue and refuses by name when it cannot.
"""

from __future__ import annotations

import pytest

from core.errors import Refusal
from tests.support.fixtures_hotupdate import Harness
from training.acquisition.base import AcquisitionMode, AcquisitionSpec, Availability
from training.budget import require_exactly_one_input


def _preflight(harness: Harness, monkeypatch: pytest.MonkeyPatch) -> AcquisitionSpec:
    from training.hotupdate.attach import AttachAcquirer

    acquirer = AttachAcquirer(transport=harness.fake.transport())
    monkeypatch.setattr(type(acquirer), "available", lambda self: Availability(available=True))
    return acquirer.preflight(AcquisitionSpec(mode=AcquisitionMode.ATTACH))


def test_the_spec_carries_the_resident_models_parameter_count(harness, monkeypatch) -> None:  # type: ignore[no-untyped-def]
    spec = _preflight(harness, monkeypatch)
    assert spec.param_count == harness.fake.num_parameters
    require_exactly_one_input(spec)  # the guard's own gate: estimable, not ambiguous


def test_it_asked_miLLMs_catalogue_once(harness, monkeypatch) -> None:  # type: ignore[no-untyped-def]
    _preflight(harness, monkeypatch)
    assert harness.fake.call_count("models") == 1


def test_an_unknown_size_is_refused_by_name_not_permitted(harness, monkeypatch) -> None:  # type: ignore[no-untyped-def]
    harness.fake.num_parameters = None
    with pytest.raises(Refusal) as caught:
        _preflight(harness, monkeypatch)
    assert caught.value.code == "attach_resident_size_unknown"
    assert harness.fake.call_count("attach") == 0, "refused before attaching anything"


def test_a_trainer_that_is_pid_1_is_refused_before_attaching(harness, monkeypatch) -> None:  # type: ignore[no-untyped-def]
    """CUDA refuses a handle whose creator PID equals the opener's; miLLM's server is PID 1
    of its namespace, so a PID-1 trainer fails every rebuild (found live 2026-10-17)."""
    import os

    from training.hotupdate.attach import AttachAcquirer

    acquirer = AttachAcquirer(transport=harness.fake.transport())
    monkeypatch.setattr(type(acquirer), "available", lambda self: Availability(available=True))
    monkeypatch.setattr(os, "getpid", lambda: 1)
    with pytest.raises(Refusal) as caught:
        acquirer.acquire(AcquisitionSpec(mode=AcquisitionMode.ATTACH))
    assert caught.value.code == "attach_trainer_is_pid_1"
    assert harness.fake.call_count("attach") == 0
