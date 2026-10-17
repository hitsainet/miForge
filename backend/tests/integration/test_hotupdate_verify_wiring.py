"""R-02.2's reconstruction check runs in the production loop, before step 1  (FR-011.9-13).

Until 2026-10-16 ``training/hotupdate/verify.py`` had no production caller and no provider
filled log-probabilities, so an attached run would have trained a rebuilt-from-handles graph
it had never compared with the serving process. These tests drive ``tasks.drive_run`` and
the real ``AttachAcquirer.acquire``; the local side runs a real (tiny) torch module through
``Verifier``; the serving side is a stub whose numbers are computed by hand from the same
logits - so the comparison is between two independent computations, not one function and
itself.
"""

# ruff: noqa: F811 - `rig` is a pytest fixture imported by name; parameters receive it.

from __future__ import annotations

import math
from typing import Any, ClassVar

import pytest

from core.errors import Refusal
from tests.support.fake_weights_surface import FakeWeightsSurface
from training import tasks
from training.acquisition.base import AcquiredModel
from training.hotupdate import verify

from .test_step_loop import RECORD, drive, rig  # noqa: F401 - `rig` is a fixture

torch = pytest.importorskip("torch")

#: The tiny model's next-token logits for any input: fixed, so the "served" side can be
#: computed by hand from the same numbers.
LOGITS = [0.5, 2.0, -1.0, 3.0, 1.0, 0.0]


class TinyLM(torch.nn.Module):  # type: ignore[misc,name-defined]
    """A real torch module with a parameter, whose last-position logits are LOGITS + bias."""

    def __init__(self, bias: float = 0.0, skew: int | None = None) -> None:
        super().__init__()
        values = list(LOGITS)
        if skew is not None:
            values[skew] += 0.5  # a reconstruction defect in one row of the output layer
        self.out = torch.nn.Parameter(torch.tensor(values) + bias)

    def forward(self, input_ids: Any) -> Any:
        batch, length = input_ids.shape
        return (self.out.expand(batch, length, -1),)


class Tokenizer:
    def __call__(self, text: str, add_special_tokens: bool = True) -> dict[str, list[int]]:
        return {"input_ids": [1, 2, 3] if add_special_tokens else [2, 3]}


def served(token_ids: list[int]) -> dict[int, float]:
    """miLLM's scoring-mode answer, by hand: log-softmax over the allowed set at temp 0."""
    z = [LOGITS[i] for i in token_ids]
    log_norm = math.log(sum(math.exp(v) for v in z))
    return {i: LOGITS[i] - log_norm for i in token_ids}


class Serving:
    identity = "openai_compatible:LFM2-1.2B@http://millm.test:8000"

    def __init__(self, delta: float = 0.0) -> None:
        self.delta = delta
        self.calls: list[tuple[str, list[int]]] = []

    def next_token_logprobs(
        self, prompt: str, token_ids: list[int], *, add_special_tokens: bool = True
    ) -> dict[int, float]:
        self.calls.append((prompt, list(token_ids)))
        answer = served(token_ids)
        return {k: v + (self.delta if n == 0 else 0.0) for n, (k, v) in enumerate(answer.items())}


def _verifier(module: Any, serving: Serving, size: int = 4) -> verify.Verifier:
    return verify.Verifier(
        module=module,
        tokenizer_loader=Tokenizer,
        provider=serving,
        prompt="The capital of France is",
        token_set_size=size,
        tolerance=0.01,
        tolerance_is_provisional=True,
    )


class TestTheCheckItself:
    def test_restricted_logprobs_is_miLLMs_arithmetic(self) -> None:
        got = verify.restricted_logprobs(torch.tensor(LOGITS), [3, 1])
        want = served([3, 1])
        assert got == pytest.approx({"3": want[3], "1": want[1]}, abs=1e-6)

    def test_a_faithful_reconstruction_matches(self) -> None:
        serving = Serving()
        result = _verifier(TinyLM(), serving)()
        assert result.matched
        # The set is the rebuilt model's own top-4, scored on both sides over exactly it.
        assert serving.calls == [("The capital of France is", [3, 1, 4, 0])]
        assert result.token_set == ("3", "1", "4", "0")
        assert result.max_abs_difference is not None and result.max_abs_difference < 1e-5

    def test_a_shift_common_to_every_logit_still_matches(self) -> None:
        """Log-probabilities over a set are shift-invariant; logits are not. Comparing the
        right quantity is what lets a correct reconstruction pass (R-02.2)."""
        assert _verifier(TinyLM(bias=7.0), Serving())().matched

    def test_a_faulty_reconstruction_refuses_the_run(self) -> None:
        with pytest.raises(Refusal) as caught:
            _verifier(TinyLM(skew=1), Serving())()
        assert caught.value.code == "reconstruction_mismatch"
        assert caught.value.details["tolerance_is_provisional"] is True

    def test_a_serving_side_disagreement_refuses_the_run(self) -> None:
        with pytest.raises(Refusal) as caught:
            _verifier(TinyLM(), Serving(delta=0.05))()
        assert caught.value.code == "reconstruction_mismatch"


class TestTheLoopRunsItFirst:
    def _acquire(self, rig, monkeypatch, verifier, release) -> None:
        monkeypatch.setattr(
            tasks,
            "acquire_for_run",
            lambda spec: AcquiredModel(
                rig["model"], rig["tokenizer"], RECORD, verifier=verifier, release=release
            ),
        )

    def test_it_runs_once_before_the_first_batch(self, rig, monkeypatch) -> None:
        events: list[str] = []
        real_next = rig["source"].next

        def next_batch() -> Any:
            events.append("batch")
            return real_next()

        rig["source"].next = next_batch
        self._acquire(rig, monkeypatch, lambda: events.append("verify"), None)
        drive(rig, max_steps=2)
        assert events[0] == "verify"
        assert events.count("verify") == 1

    def test_a_mismatch_stops_the_run_before_any_step_and_releases(self, rig, monkeypatch) -> None:
        released: list[str] = []

        def mismatch() -> None:
            raise verify.refuse("reconstruction_mismatch", "rebuilt graph disagrees")

        self._acquire(rig, monkeypatch, mismatch, lambda: released.append("release"))
        with pytest.raises(Refusal) as caught:
            drive(rig, max_steps=3)
        assert caught.value.code == "reconstruction_mismatch"
        assert rig["source"].calls == 0, "no batch may be drawn for an unverified graph"
        assert rig["optimizer"].steps == 0
        assert released == ["release"]


class TestTheAcquirerSuppliesIt:
    def _acquirer(self, rig, monkeypatch, serving: Serving, module: Any):  # type: ignore[no-untyped-def]
        from tests.support.fixtures_hotupdate import configure
        from training.acquisition.base import Availability
        from training.hotupdate import attach as attach_module
        from training.hotupdate import rebuild

        configure(monkeypatch)
        surface = FakeWeightsSurface()
        acquirer = attach_module.AttachAcquirer(transport=surface.transport())
        monkeypatch.setattr(type(acquirer), "available", lambda self: Availability(available=True))

        class Base:
            device_uuid = "GPU-test"
            tensors: ClassVar[dict[str, Any]] = {}

        monkeypatch.setattr(rebuild, "build", lambda response, **kw: Base())
        monkeypatch.setattr(rebuild, "check_build_identity", lambda response: None)
        monkeypatch.setattr(attach_module, "build_module", lambda response, base: module)
        monkeypatch.setattr(attach_module, "acquisition_record", lambda **kw: RECORD)
        monkeypatch.setattr(
            attach_module.AttachAcquirer, "_verify_provider", lambda self, *a: serving
        )
        monkeypatch.setattr(
            attach_module.AttachAcquirer, "_tokenizer_loader", lambda self, *a: Tokenizer
        )
        return acquirer, surface

    def test_the_acquisition_carries_the_real_verifier(self, rig, monkeypatch) -> None:
        from training.acquisition.base import AcquisitionMode, AcquisitionSpec

        acquirer, _ = self._acquirer(rig, monkeypatch, Serving(), TinyLM())
        try:
            acquired = acquirer.acquire(AcquisitionSpec(mode=AcquisitionMode.ATTACH))
            assert isinstance(acquired.verifier, verify.Verifier)
            assert acquired.verifier is acquirer.verifier
            assert acquired.verifier.module is acquired.module
        finally:
            if acquirer.heartbeat is not None:
                acquirer.heartbeat.stop()

    def test_a_mismatch_through_the_real_acquirer_trains_nothing_and_detaches(
        self, rig, monkeypatch
    ) -> None:
        from training.acquisition.base import AcquisitionMode, AcquisitionSpec

        acquirer, surface = self._acquirer(rig, monkeypatch, Serving(), TinyLM(skew=1))
        monkeypatch.setattr(
            tasks,
            "acquire_for_run",
            lambda spec: acquirer.acquire(AcquisitionSpec(mode=AcquisitionMode.ATTACH)),
        )
        try:
            with pytest.raises(Refusal) as caught:
                drive(rig, max_steps=2)
            assert caught.value.code == "reconstruction_mismatch"
            ops = surface.ops()
            assert "quiesce" not in ops and "commit-begin" not in ops
            assert ops[-1] == "detach"
        finally:
            if acquirer.heartbeat is not None:
                acquirer.heartbeat.stop()

    def test_the_default_provider_and_tokenizer_name_the_resident_model(self, monkeypatch) -> None:
        from tests.support.fixtures_hotupdate import BASE_URL, configure
        from training.hotupdate import attach as attach_module
        from training.hotupdate import settings as settings_module

        configure(monkeypatch)
        live = settings_module.load()
        acquirer = attach_module.AttachAcquirer()
        provider = acquirer._verify_provider("LFM2-1.2B", live)
        assert provider.base_url.rstrip("/") == BASE_URL
        assert provider.identity.model == "LFM2-1.2B"
        captured: list[str] = []
        import transformers

        monkeypatch.setattr(
            transformers.AutoTokenizer,
            "from_pretrained",
            classmethod(lambda cls, source, **kw: captured.append(source)),
        )
        acquirer._tokenizer_loader("LFM2-1.2B", live)()
        monkeypatch.setenv("HOTUPDATE_VERIFY_TOKENIZER", "/data/hf-cache/lfm2")
        from core.config import reset_settings

        reset_settings()
        acquirer._tokenizer_loader("LFM2-1.2B", settings_module.load())()
        assert captured == ["LFM2-1.2B", "/data/hf-cache/lfm2"]
