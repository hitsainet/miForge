"""Placement is REACHABLE from all three acquirers  (ADR-016 #1).

**ADR-016 #1: a capability is not shipped until a test FAILS when its wiring is removed.**
That rule exists in this repository because a sibling once shipped sixteen fully-implemented,
fully-unit-tested MCP tools that were never registered with its server: every test passed by
importing the module directly, the suite was green, and the feature did not exist for any
caller. ``training/acquisition/placement.py`` is in exactly that position - a correct module
that three ``acquire()`` methods have to call - so the module's own tests
(``test_acquisition_placement.py``) are not sufficient evidence of anything.

So this file asserts **the call, the payload and the call count** for each acquirer, and then
asserts the *recorded* device independently of the spy, because a reachability test that only
checks "was called" passes against a call sending the wrong arguments.

## The mutations these cases are the control for

Deleting ``placed = place(module, spec.gpu_request)`` from any one of ``local_dir.acquire``,
``scratch.acquire`` or ``hub.acquire`` must turn this file red twice over: the spy is never
called, and ``AcquisitionRecord.device_kind`` falls back to ``None``, which is why that field
defaults to ``None`` rather than to ``"cpu"``. **A default that reads as a legitimate state
would have hidden the same defect a second time** - and this defect's whole character was
that the degraded state was indistinguishable from the intended one.

The red/green output for each is recorded in the review notes for 2026-10-16.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from tests.unit.test_acquisition_local_dir import build_model_dir, rooted  # noqa: F401
from training.acquisition import hub, local_dir, scratch
from training.acquisition.base import AcquisitionMode, AcquisitionSpec
from training.acquisition.placement import KIND_CPU, Placement

CARD = "GPU-aaaaaaaa-1111-2222-3333-444444444444"
LLAMA = {
    "model_type": "llama",
    "hidden_size": 64,
    "num_hidden_layers": 2,
    "num_attention_heads": 4,
    "vocab_size": 256,
}


class _Tokenizer:
    """A tokenizer double with a ``__len__``, because that is what the vocabulary gate
    reads - a double that cannot express the defect makes its assertion vacuous."""

    def __len__(self) -> int:
        return 128


@pytest.fixture
def tokenizer_dir(rooted: Path) -> Path:  # noqa: F811
    """A tokenizer directory inside the configured model root, sized for ``LLAMA``.

    A ``create`` run names the tokenizer its checkpoint ships (2026-10-16), and both
    ``preflight`` and ``acquire`` resolve it - so a creation spec without one is refused
    ``tokenizer_required`` and these placement cases would never reach ``place()``.
    """
    from tests.support.tiny_hf_model import write_tokenizer_json_only

    return write_tokenizer_json_only(rooted / "tok", vocab_size=LLAMA["vocab_size"])


class StubModule:
    """A module that can be identified by the spy, so the *payload* is assertable."""

    def to(self, target):  # pragma: no cover - the spy replaces placement entirely
        return self

    def parameters(self):
        return iter([])


class PlacementSpy:
    """Records every call, and answers with a placement the record must then carry."""

    def __init__(self, answer: Placement) -> None:
        self.answer = answer
        self.calls: list[tuple] = []

    def __call__(self, module, gpu_request, **kwargs):
        self.calls.append((module, gpu_request, kwargs))
        return self.answer


ON_THE_CARD = Placement(device_kind="cuda", device_uuid=CARD, reason="leased_uuid")


def _spy_on(monkeypatch: pytest.MonkeyPatch, module) -> PlacementSpy:
    """Replace the acquirer module's own ``place`` reference.

    The **module's attribute**, not ``placement.place``: each acquirer imports the name, so
    patching per-module is what makes a missing call site in one of the three show up as
    that one failing rather than as all three.
    """
    spy = PlacementSpy(ON_THE_CARD)
    monkeypatch.setattr(module, "place", spy)
    return spy


def _assert_called_once_with_the_module(spy: PlacementSpy, module, gpu_request: str) -> None:
    assert len(spy.calls) == 1, (
        f"placement must be called exactly once per acquisition, got {len(spy.calls)}; "
        f"zero means the acquirer never places the model and the run computes on the CPU "
        f"while holding a card's lease, which is the defect this file is the control for"
    )
    called_module, called_request, _kwargs = spy.calls[0]
    assert called_module is module, (
        "the placed object must be the acquired module itself - placing a copy or a "
        "different object leaves the trained tensors where they were"
    )
    assert called_request == gpu_request, (
        "the device request must be passed through unchanged; a hard-coded 'auto' here "
        "would discard the leased UUID and reintroduce ADR-008's inverted enumeration"
    )


class TestEachAcquirerPlacesTheModelItAcquired:
    def test_a_local_directory_load_places_and_records(
        self, rooted: Path, monkeypatch: pytest.MonkeyPatch  # noqa: F811
    ) -> None:
        module = StubModule()
        monkeypatch.setattr(local_dir, "_load_pretrained", lambda d, dt: (module, object()))
        spy = _spy_on(monkeypatch, local_dir)
        directory = build_model_dir(rooted)
        spec = AcquisitionSpec(mode=AcquisitionMode.LOAD, path=directory, gpu_request=CARD)

        acquired = local_dir.LocalDirAcquirer().acquire(spec)

        _assert_called_once_with_the_module(spy, module, CARD)
        assert acquired.record.device_kind == "cuda"
        assert acquired.record.device_uuid == CARD
        assert acquired.record.device_placement_reason == "leased_uuid"
        assert acquired.device_uuid == CARD, (
            "AcquiredModel.device_uuid is what providers/model_handle.device_uuid() reads, "
            "and therefore what a score's provider provenance reports (R-02.16). It was "
            "never assigned before 2026-10-16, so every in-process score recorded no device "
            "and providers/in_process._check_budget skipped its KV-cache guard entirely"
        )

    def test_a_creation_places_and_records(
        self, monkeypatch: pytest.MonkeyPatch, tokenizer_dir: Path
    ) -> None:
        module = StubModule()
        monkeypatch.setattr(
            scratch, "_instantiate", lambda parsed, seed, tok: (module, _Tokenizer())
        )
        spy = _spy_on(monkeypatch, scratch)
        spec = AcquisitionSpec(
            mode=AcquisitionMode.CREATE,
            config=dict(LLAMA),
            gpu_request=CARD,
            seed=7,
            tokenizer_path=tokenizer_dir,
        )

        acquired = scratch.ScratchAcquirer().acquire(spec)

        _assert_called_once_with_the_module(spy, module, CARD)
        assert acquired.record.device_kind == "cuda"
        assert acquired.record.device_uuid == CARD
        assert acquired.device_uuid == CARD

    def test_a_hub_load_places_and_records(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        module = StubModule()
        snapshot = tmp_path / "snapshot"
        snapshot.mkdir()
        (snapshot / "config.json").write_text(json.dumps({"model_type": "llama"}))
        (snapshot / "tokenizer.json").write_text("{}")
        from tests.unit.test_acquisition_dtype_gate import write_safetensors

        write_safetensors(snapshot / "model.safetensors", {"w": ("F16", [8, 8])})

        monkeypatch.setattr(hub, "download_snapshot", lambda repo, commit: snapshot)
        monkeypatch.setattr(local_dir, "_load_pretrained", lambda d, dt: (module, object()))
        spy = _spy_on(monkeypatch, hub)
        spec = AcquisitionSpec(
            mode=AcquisitionMode.LOAD,
            repo="org/model",
            revision="main",
            revision_resolved="c" * 40,
            gpu_request=CARD,
        )

        acquired = hub.HubAcquirer().acquire(spec)

        _assert_called_once_with_the_module(spy, module, CARD)
        assert acquired.record.device_kind == "cuda"
        assert acquired.record.device_uuid == CARD
        assert acquired.device_uuid == CARD


class TestTheRecordedDeviceIsTheREALONEOnThisMachine:
    """Without the spy, through the real placement. **Asserts the recorded decision, not
    that CUDA was used**, so these are meaningful on a CPU-only box - which is the whole
    reason the defect survived: every assertion that could have caught it needed a card.

    On a machine with no CUDA device the honest answer is ``cpu``/``None``, and that is
    exactly what must be recorded: ADR-023 makes the CPU a supported developer state, and
    ADR-013 forbids claiming a card that was not used.
    """

    @staticmethod
    def _expected_kind() -> str:
        import torch

        return "cuda" if torch.cuda.is_available() else KIND_CPU

    def test_a_creation_records_the_device_it_is_actually_on(
        self, monkeypatch: pytest.MonkeyPatch, tokenizer_dir: Path
    ) -> None:
        import torch

        # A real `torch` module so the real read-back runs: `_instantiate` is stubbed only
        # to keep the case fast, not to avoid the measurement.
        monkeypatch.setattr(
            scratch,
            "_instantiate",
            lambda parsed, seed, tok: (torch.nn.Linear(4, 4), _Tokenizer()),
        )
        spec = AcquisitionSpec(
            mode=AcquisitionMode.CREATE, config=dict(LLAMA), tokenizer_path=tokenizer_dir
        )

        acquired = scratch.ScratchAcquirer().acquire(spec)

        assert acquired.record.device_kind == self._expected_kind(), (
            "device_kind defaults to None, which means *nothing placed these parameters*. "
            "A concrete value here is only possible if `acquire` called placement"
        )
        assert acquired.record.device_placement_reason is not None
        if acquired.record.device_kind == KIND_CPU:
            assert acquired.record.device_uuid is None
            assert acquired.device_uuid is None
            assert next(acquired.module.parameters()).device.type == KIND_CPU

    def test_a_local_load_records_the_device_it_is_actually_on(
        self, rooted: Path, monkeypatch: pytest.MonkeyPatch  # noqa: F811
    ) -> None:
        import torch

        monkeypatch.setattr(
            local_dir, "_load_pretrained", lambda d, dt: (torch.nn.Linear(4, 4), object())
        )
        directory = build_model_dir(rooted)
        spec = AcquisitionSpec(mode=AcquisitionMode.LOAD, path=directory)

        acquired = local_dir.LocalDirAcquirer().acquire(spec)

        assert acquired.record.device_kind == self._expected_kind()
        assert acquired.record.device_placement_reason is not None


class TestTheRecordCannotClaimADeviceItWasNotGiven:
    def test_device_kind_defaults_to_None_rather_than_to_cpu(self) -> None:
        """**The field that makes the mutation visible.** If ``device_kind`` defaulted to
        ``"cpu"``, deleting a ``place()`` call would produce a record that reads as a
        perfectly ordinary CPU acquisition - the same indistinguishability that let the
        original defect run a leased card's run on the CPU for the project's whole life."""
        from training.acquisition.base import AcquisitionRecord

        for name in ("device_kind", "device_uuid", "device_placement_reason"):
            assert AcquisitionRecord.model_fields[name].default is None, (
                f"{name} must default to None: None means 'nothing recorded this', and a "
                f"default naming a real state would make an unplaced model look placed"
            )

    def test_a_cpu_record_carries_no_device_uuid(self) -> None:
        """``device_uuid`` is what ``in_process._check_budget`` reads to decide whether a
        VRAM budget applies at all, so a UUID on a CPU record would make the scoring guard
        compare against a card that is not computing."""
        from training.acquisition.base import AcquisitionMode, AcquisitionRecord, SourceKind

        record = AcquisitionRecord(
            mode=AcquisitionMode.CREATE,
            source_kind=SourceKind.CONFIG,
            starting_weights_sha256="a" * 64,
            base_model="config:llama@abc+seed=none",
            device_kind=KIND_CPU,
            device_uuid=None,
            device_placement_reason="cuda_unavailable",
        )
        assert record.device_uuid is None
        assert record.model_dump(mode="json")["device_kind"] == KIND_CPU, (
            "the device travels in the record's dump, which is what training/tasks.py logs "
            "as `acquisition_recorded` before the first step (R-02.47)"
        )
