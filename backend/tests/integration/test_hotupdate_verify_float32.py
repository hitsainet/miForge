"""R-02.2 in float32 on both sides separates a ~1 % reconstruction error  (F11 14.3b, D41).

Window 2 measured (LFM2.5-1.2B, bf16, RTX 3090): a clean reconstruction differs from what
miLLM serves by 0.059-0.221 in log-probability, a reconstruction with one block's output
scaled by 1.01 by 0.072-0.220 - **inside** the clean range, so no bf16 tolerance separates
them; D41's 0.5 catches gross errors (>= 2.29) only. The clean gap is cross-process kernel
numerics, which float32 on both sides shrinks by orders of magnitude while a 1 % error in the
graph stays the size it is.

**Reproduced here on a synthetic graph, through the production code:** the real
:class:`training.hotupdate.verify.Verifier` (its float32 path is ``torch.func.functional_call``
over float32 copies), the real :class:`providers.openai_compatible.OpenAICompatibleProvider`
(its ``scoring_precision`` request and echo check), and a fake ``/v1/completions`` that
implements the **proposed** miLLM contract (FTDD §6.10): it scores the same weights with a
different kernel (chunked accumulation - a stand-in for another process's kernels), in bf16
or, when asked and able, in float32 and says so. A second fake behaves as miLLM does today:
it accepts the unknown field and ignores it.

Measured on this graph over four prompts (recorded in F11 FTASKS 14.3b): **native** - clean
0.050-0.113, 1 % error 0.064-0.082 (on two prompts the error is SMALLER than the clean gap);
**float32** - clean 1.9e-6 - 6.4e-6, 1 % error 0.014-0.056.

**Since 2026-10-18 the fake is miLLM's published contract, not a proposal.** miLLM shipped
FR-25.15 (main ce8046b; D61) and generates ``docs/schemas/millm-scoring-api-v1.json`` from its
route; miForge vendors it byte-identical (``test_schema_sync.py``) and every request this fake
receives and every answer it gives - the echo and the 409 refusal - is validated against it
here. The float32 bar is the MEASURED 5e-4 (FTASKS 14.3b), and a 409
``scoring_precision_unavailable`` falls back to the native comparison under D41's 0.5 - named
in a WARNING and in the result, never silent.
"""

from __future__ import annotations

import json
import logging
import math
from pathlib import Path
from typing import Any

import httpx
import pytest
import torch
import torch.nn.functional as F  # noqa: N812 - torch's own convention

from core.errors import Refusal
from providers.openai_compatible import OpenAICompatibleProvider
from training.hotupdate import verify

V, D, LAYERS = 64, 64, 6
ERROR_BLOCK, ERROR_SCALE = 2, 1.01
NATIVE_TOLERANCE = 0.5  # D41, provisional
FLOAT32_TOLERANCE = 5e-4  # the float32 mode's MEASURED bar (FTASKS 14.3b, 2026-10-18)
PROMPT = "The capital of France is"

jsonschema = pytest.importorskip("jsonschema")

#: miLLM's generated scoring contract, as vendored (byte-identity: test_schema_sync.py).
CONTRACT = json.loads(
    (
        Path(__file__).resolve().parents[3] / "docs" / "schemas" / "millm-scoring-api-v1.json"
    ).read_text("utf-8")
)
REFUSAL = CONTRACT["scoring_precision"]["refusal"]


def _contract_validator(ref: dict[str, str]) -> Any:
    return jsonschema.Draft202012Validator({**ref, "$defs": CONTRACT["$defs"]})


REQUEST_VALIDATOR = _contract_validator(CONTRACT["request"])
RESPONSE_VALIDATOR = _contract_validator(CONTRACT["response"])


def _rms(x: torch.Tensor) -> torch.Tensor:
    return x * torch.rsqrt(x.float().pow(2).mean(-1, keepdim=True) + 1e-6).to(x.dtype)


class Toy(torch.nn.Module):
    """A residual MLP stack: enough depth for kernel rounding to compound as it does in a
    real decoder, small enough to run on a laptop. ``chunked`` is the serving side's kernel."""

    def __init__(self) -> None:
        super().__init__()
        self.emb = torch.nn.Embedding(V, D)
        self.w1 = torch.nn.ParameterList(
            [torch.nn.Parameter(torch.randn(4 * D, D) / D**0.5) for _ in range(LAYERS)]
        )
        self.w2 = torch.nn.ParameterList(
            [torch.nn.Parameter(torch.randn(D, 4 * D) / (4 * D) ** 0.5) for _ in range(LAYERS)]
        )
        self.head = torch.nn.Parameter(torch.randn(V, D) * 3 / D**0.5)
        self.chunked = False
        self.error_scale = 1.0

    def _lin(self, x: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
        if not self.chunked:
            return F.linear(x, w)
        width = w.shape[1] // 4
        out = F.linear(x[..., :width], w[:, :width])
        for i in range(1, 4):
            out = out + F.linear(
                x[..., i * width : (i + 1) * width], w[:, i * width : (i + 1) * width]
            )
        return out

    def forward(self, input_ids: torch.Tensor) -> tuple[torch.Tensor]:
        h = self.emb(input_ids)
        for i in range(LAYERS):
            delta = self._lin(F.gelu(self._lin(_rms(h), self.w1[i])), self.w2[i])
            if i == ERROR_BLOCK:
                delta = delta * self.error_scale
            h = h + delta
        return (self._lin(_rms(h), self.head),)


def _ids(text: str) -> list[int]:
    return [ord(c) % V for c in text]


class Tokenizer:
    def __call__(self, text: str, add_special_tokens: bool = True) -> dict[str, list[int]]:
        return {"input_ids": _ids(text)}


@pytest.fixture(scope="module")
def weights() -> dict[str, torch.Tensor]:
    """One set of bf16 weights - what miLLM holds and what the rebuild aliases."""
    torch.manual_seed(0)
    return {k: v.to(torch.bfloat16) for k, v in Toy().state_dict().items()}


def _model(weights: dict[str, torch.Tensor], *, error: bool = False) -> Toy:
    model = Toy().to(torch.bfloat16)
    model.load_state_dict(weights)
    model.error_scale = ERROR_SCALE if error else 1.0
    return model


def _serving(
    weights: dict[str, torch.Tensor],
    *,
    honours_float32: bool,
    refuses_float32: str | None = None,
    seen: list[dict[str, Any]] | None = None,
) -> httpx.Client:
    """``/v1/completions`` scoring mode as miLLM serves it (``next_token_scores.py``'s
    arithmetic; FR-25.15's ``scoring_precision``): float32 honoured and echoed; or ignored
    (miLLM before FR-25.15 - an unknown field accepted silently); or REFUSED with the 409 the
    contract declares, its reason in the header (``refuses_float32``: e.g. a bitsandbytes model).
    Every request and every answer is validated against the vendored contract."""
    native = _model(weights)
    native.chunked = True
    wide = _model(weights).float()
    wide.chunked = True

    def handle(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        REQUEST_VALIDATOR.validate(body)
        if seen is not None:
            seen.append(body)
        asked = body.get("scoring_precision") == "float32"
        if asked and refuses_float32 is not None:
            assert refuses_float32 in REFUSAL["reasons"]
            # miLLM's envelope, as its own test_float32_scoring.py pins it.
            return httpx.Response(
                REFUSAL["status"],
                json={
                    "error": {
                        "message": f"float32 scoring is unavailable (reason: {refuses_float32})"
                        ": the resident model is quantized",
                        "type": REFUSAL["type"],
                        "param": REFUSAL["param"],
                        "code": REFUSAL["code"],
                    }
                },
                headers={REFUSAL["reason_header"]: refuses_float32},
            )
        float32 = honours_float32 and asked
        model = wide if float32 else native
        with torch.no_grad():
            z = model(torch.tensor([_ids(body["prompt"])]))[0][0, -1].float()
        allowed = torch.tensor(body["allowed_token_ids"])
        values = torch.log_softmax(z[allowed], dim=0)
        top = {f"token_id:{int(i)}": float(v) for i, v in zip(allowed, values, strict=True)}
        first = next(iter(top))
        answer: dict[str, Any] = {
            "id": "cmpl-fake",
            "object": "text_completion",
            "created": 0,
            "model": body["model"],
            "choices": [
                {
                    "index": 0,
                    "text": "",
                    "finish_reason": "length",
                    "logprobs": {
                        "tokens": [first],
                        "token_logprobs": [top[first]],
                        "top_logprobs": [top],
                        "text_offset": [0],
                    },
                }
            ],
            "usage": {"prompt_tokens": len(body["prompt"]), "completion_tokens": 1},
        }
        headers: dict[str, str] = {}
        if float32:
            answer[CONTRACT["scoring_precision"]["echo_body_field"]] = "float32"
            headers[CONTRACT["scoring_precision"]["echo_header"]] = "float32"
        RESPONSE_VALIDATOR.validate(answer)
        return httpx.Response(200, json=answer, headers=headers)

    return httpx.Client(transport=httpx.MockTransport(handle))


def _provider(
    weights: dict[str, torch.Tensor],
    *,
    honours_float32: bool = True,
    refuses_float32: str | None = None,
    seen: list[dict[str, Any]] | None = None,
) -> Any:
    return OpenAICompatibleProvider(
        base_url="http://millm.test:8000",
        model="toy",
        connect_timeout_s=5.0,
        read_timeout_s=5.0,
        client=_serving(
            weights, honours_float32=honours_float32, refuses_float32=refuses_float32, seen=seen
        ),
    )


def _verify(module: Toy, provider: Any, precision: str) -> verify.VerifyResult:
    checker = verify.Verifier(
        module=module,
        tokenizer_loader=Tokenizer,
        provider=provider,
        prompt=PROMPT,
        token_set_size=15,
        tolerance=FLOAT32_TOLERANCE if precision == "float32" else NATIVE_TOLERANCE,
        tolerance_is_provisional=True,
        precision=precision,
    )
    try:
        return checker()
    except Refusal as refusal:
        if refusal.code != "reconstruction_mismatch":
            raise
        assert checker.result is not None
        return checker.result


class TestTheSyntheticGraphReproducesWindowTwo:
    def test_in_bf16_a_1pct_error_hides_inside_the_kernel_spread(self, weights) -> None:
        clean = _verify(_model(weights), _provider(weights), "native")
        broken = _verify(_model(weights, error=True), _provider(weights), "native")
        assert clean.max_abs_difference is not None and broken.max_abs_difference is not None
        # The clean gap is real (the kernels differ) and the 1 % error is not separable
        # from it: within 3x, as window 2's ranges overlapped outright.
        assert clean.max_abs_difference > 1e-3
        assert broken.max_abs_difference < 3 * clean.max_abs_difference
        # And D41's bf16 bar waves both through: the blind spot this mode exists to close.
        assert clean.matched and broken.matched
        assert clean.precision == broken.precision == "native"

    def test_in_float32_the_same_error_is_refused_and_the_clean_rebuild_passes(
        self, weights
    ) -> None:
        clean = _verify(_model(weights), _provider(weights), "float32")
        broken = _verify(_model(weights, error=True), _provider(weights), "float32")
        assert clean.max_abs_difference is not None and broken.max_abs_difference is not None
        assert clean.matched, clean.record()
        assert clean.max_abs_difference < 1e-4
        assert not broken.matched
        assert broken.max_abs_difference > FLOAT32_TOLERANCE
        # Separated by orders of magnitude, not by a margin the next kernel change erases.
        assert broken.max_abs_difference > 100 * clean.max_abs_difference
        assert broken.record()["precision"] == "float32"

    def test_a_float32_mismatch_refuses_the_run_by_name(self, weights) -> None:
        checker = verify.Verifier(
            module=_model(weights, error=True),
            tokenizer_loader=Tokenizer,
            provider=_provider(weights),
            prompt=PROMPT,
            token_set_size=15,
            tolerance=FLOAT32_TOLERANCE,
            tolerance_is_provisional=True,
            precision="float32",
        )
        with pytest.raises(Refusal) as caught:
            checker()
        assert caught.value.code == "reconstruction_mismatch"
        assert caught.value.details["precision"] == "float32"
        assert caught.value.details["tolerance_is_provisional"] is True


class TestTheServedTensorsAreNeverTouched:
    def test_the_float32_forward_leaves_the_rebuilt_parameters_as_they_were(self, weights) -> None:
        module = _model(weights)
        before = {k: (v.dtype, v.data_ptr(), v.clone()) for k, v in module.state_dict().items()}
        _verify(module, _provider(weights), "float32")
        for key, tensor in module.state_dict().items():
            dtype, pointer, values = before[key]
            assert tensor.dtype == dtype == torch.bfloat16
            assert tensor.data_ptr() == pointer, f"{key} was replaced"
            assert torch.equal(tensor, values), f"{key} was written"


class TestTodaysMiLLMIsRefusedNotTrusted:
    def test_an_endpoint_that_ignores_the_request_is_refused(self, weights) -> None:
        """miLLM accepts unknown fields silently, so a float32 request it does not honour
        answers in bf16 and looks the same. Only the echo can tell."""
        with pytest.raises(Refusal) as caught:
            _verify(_model(weights), _provider(weights, honours_float32=False), "float32")
        assert caught.value.code == "provider_scoring_precision_unconfirmed"
        assert caught.value.details["requested"] == "float32"
        assert caught.value.details["echoed"] is None

    def test_the_native_mode_sends_no_precision_field(self, weights) -> None:
        seen: list[dict[str, Any]] = []

        def handle(request: httpx.Request) -> httpx.Response:
            body = json.loads(request.content)
            seen.append(body)
            values = [-math.log(len(body["allowed_token_ids"]))] * len(body["allowed_token_ids"])
            top = {
                f"token_id:{i}": v for i, v in zip(body["allowed_token_ids"], values, strict=True)
            }
            return httpx.Response(200, json={"choices": [{"logprobs": {"top_logprobs": [top]}}]})

        provider = OpenAICompatibleProvider(
            base_url="http://millm.test:8000",
            model="toy",
            connect_timeout_s=5.0,
            read_timeout_s=5.0,
            client=httpx.Client(transport=httpx.MockTransport(handle)),
        )
        _verify(_model(weights), provider, "native")
        assert len(seen) == 1 and "scoring_precision" not in seen[0]

    def test_an_unknown_precision_is_refused_before_any_forward(self, weights) -> None:
        with pytest.raises(Refusal) as caught:
            verify.Verifier(
                module=_model(weights),
                tokenizer_loader=Tokenizer,
                provider=_provider(weights),
                prompt=PROMPT,
                token_set_size=15,
                tolerance=0.1,
                tolerance_is_provisional=True,
                precision="float16",
            )()
        assert caught.value.code == "verification_precision_unknown"


@pytest.fixture
def fresh_settings():  # type: ignore[no-untyped-def]
    """Drop the cached settings after the test, so its environment does not leak into the next
    (it did: a 0.5 set here was read by test_hotupdate_verify.py's default check)."""
    from core.config import reset_settings

    yield
    reset_settings()


class TestTheSettingReachesTheRunsVerifier:
    """The acquisition builds the run's verifier from the live settings - so the precision
    and ITS bar arrive together, never the bf16 bar on a float32 comparison."""

    def _verifier(self, monkeypatch: pytest.MonkeyPatch, **env: str) -> verify.Verifier:
        from tests.support.fixtures_hotupdate import configure
        from training.hotupdate import attach as attach_module
        from training.hotupdate import settings as settings_module

        configure(monkeypatch, **env)
        acquirer = attach_module.AttachAcquirer()
        built = acquirer._verifier(object(), "toy", settings_module.load())
        assert isinstance(built, verify.Verifier)
        return built

    def test_the_default_is_float32_at_the_measured_bar(self, monkeypatch, fresh_settings) -> None:
        built = self._verifier(monkeypatch)
        assert (built.precision, built.tolerance, built.float32_device) == ("float32", 5e-4, "cpu")
        assert built.tolerance_is_provisional is False
        # The bar it falls back to on a 409 is D41's native one, still provisional.
        assert (built.native_tolerance, built.native_tolerance_is_provisional) == (0.5, True)

    def test_float32_carries_its_own_tolerance_and_device(
        self, monkeypatch, fresh_settings
    ) -> None:
        built = self._verifier(
            monkeypatch,
            HOTUPDATE_VERIFICATION_PRECISION="float32",
            HOTUPDATE_VERIFICATION_TOLERANCE="0.4",
            HOTUPDATE_VERIFICATION_TOLERANCE_FLOAT32="0.001",
            HOTUPDATE_VERIFICATION_TOLERANCE_FLOAT32_MEASURED="false",
            HOTUPDATE_VERIFY_FLOAT32_DEVICE="cpu",
        )
        assert (built.precision, built.tolerance, built.float32_device) == ("float32", 0.001, "cpu")
        assert built.tolerance_is_provisional is True
        assert built.native_tolerance == 0.4

    def test_native_keeps_d41s_tolerance(self, monkeypatch, fresh_settings) -> None:
        built = self._verifier(monkeypatch, HOTUPDATE_VERIFICATION_PRECISION="native")
        assert (built.precision, built.tolerance) == ("native", 0.5)
        assert built.tolerance_is_provisional is True


class TestTheRunsVerifierScoresInFloat32:
    """End to end through the acquirer's own construction: the verifier the run calls asks miLLM
    for float32 - ONE request, carrying ``scoring_precision: "float32"`` - and judges the answer
    at the measured 5e-4. Removing the line that requests float32, or the one that reads the
    float32 bar, turns this red (FTASKS 14.3b controls)."""

    def _run_verifier(
        self, monkeypatch, weights, *, error: bool = False, **serving: Any
    ) -> tuple[verify.Verifier, list[dict[str, Any]]]:
        from tests.support.fixtures_hotupdate import configure
        from training.hotupdate import attach as attach_module
        from training.hotupdate import settings as settings_module

        configure(monkeypatch)
        seen: list[dict[str, Any]] = []
        provider = _provider(weights, seen=seen, **serving)
        monkeypatch.setattr(
            attach_module.AttachAcquirer, "_verify_provider", lambda self, *a: provider
        )
        monkeypatch.setattr(
            attach_module.AttachAcquirer, "_tokenizer_loader", lambda self, *a: Tokenizer
        )
        built = attach_module.AttachAcquirer()._verifier(
            _model(weights, error=error), "toy", settings_module.load()
        )
        return built, seen

    def test_a_clean_rebuild_is_verified_in_float32(
        self, monkeypatch, weights, fresh_settings
    ) -> None:
        built, seen = self._run_verifier(monkeypatch, weights)
        result = built()
        assert len(seen) == 1
        assert seen[0]["scoring_precision"] == "float32"
        assert result.matched
        record = result.record()
        assert record["precision"] == record["requested_precision"] == "float32"
        assert record["tolerance"] == 5e-4 and record["tolerance_is_provisional"] is False
        assert record["precision_fallback_reason"] is None

    def test_a_1pct_error_is_refused_at_the_measured_bar(
        self, monkeypatch, weights, fresh_settings
    ) -> None:
        built, seen = self._run_verifier(monkeypatch, weights, error=True)
        with pytest.raises(Refusal) as caught:
            built()
        assert caught.value.code == "reconstruction_mismatch"
        assert caught.value.details["precision"] == "float32"
        assert caught.value.details["tolerance"] == 5e-4
        assert [body.get("scoring_precision") for body in seen] == ["float32"]


class TestARefusedFloat32FallsBackByName:
    """miLLM refuses float32 by name when it cannot build the copy (409
    ``scoring_precision_unavailable``: bitsandbytes/GGUF, memory, device, busy). The check then
    runs NATIVE under D41's provisional 0.5 - and says so: a WARNING with the reason, and the
    result's ``requested_precision`` / ``precision_fallback_reason``. Never silently."""

    def _verifier(self, weights, provider: Any, *, error: bool = False, native: float | None):
        return verify.Verifier(
            module=_model(weights, error=error),
            tokenizer_loader=Tokenizer,
            provider=provider,
            prompt=PROMPT,
            token_set_size=15,
            tolerance=FLOAT32_TOLERANCE,
            tolerance_is_provisional=False,
            precision="float32",
            native_tolerance=native,
            native_tolerance_is_provisional=True,
        )

    def test_the_fallback_runs_native_at_d41s_bar_and_logs_why(self, weights, caplog) -> None:
        seen: list[dict[str, Any]] = []
        provider = _provider(weights, refuses_float32="model_not_float32_castable", seen=seen)
        with caplog.at_level(logging.WARNING, logger="training.hotupdate.verify"):
            result = self._verifier(weights, provider, native=NATIVE_TOLERANCE)()
        # Asked for float32 once, refused, then ONE native request with no precision field.
        assert [body.get("scoring_precision") for body in seen] == ["float32", None]
        assert result.matched
        record = result.record()
        assert record["precision"] == "native"
        assert record["requested_precision"] == "float32"
        assert record["precision_fallback_reason"] == "model_not_float32_castable"
        assert record["tolerance"] == NATIVE_TOLERANCE
        assert record["tolerance_is_provisional"] is True
        warnings = [
            r
            for r in caplog.records
            if getattr(r, "code", None) == "reconstruction_verification_precision_fallback"
        ]
        assert len(warnings) == 1
        assert warnings[0].levelno == logging.WARNING
        assert warnings[0].fallback_reason == "model_not_float32_castable"
        assert warnings[0].fallback_tolerance == NATIVE_TOLERANCE

    def test_a_fallen_back_mismatch_names_the_fallback(self, weights) -> None:
        provider = _provider(weights, refuses_float32="memory")
        checker = self._verifier(weights, provider, native=1e-9)
        with pytest.raises(Refusal) as caught:
            checker()
        assert caught.value.code == "reconstruction_mismatch"
        assert caught.value.details["precision_fallback_reason"] == "memory"
        assert "refused float32 scoring (reason: memory)" in caught.value.message

    def test_with_no_native_bar_the_refusal_aborts_the_run(self, weights) -> None:
        provider = _provider(weights, refuses_float32="busy")
        with pytest.raises(Refusal) as caught:
            self._verifier(weights, provider, native=None)()
        assert caught.value.code == "provider_scoring_precision_unavailable"
        assert caught.value.details["reason"] == "busy"

    def test_a_missing_echo_is_still_refused_never_fallen_back_from(self, weights) -> None:
        """An answer without the echo may be bf16 dressed as float32 - a silent downgrade by
        the SERVER, which is exactly what the echo exists to catch. No fallback."""
        provider = _provider(weights, honours_float32=False)
        with pytest.raises(Refusal) as caught:
            self._verifier(weights, provider, native=NATIVE_TOLERANCE)()
        assert caught.value.code == "provider_scoring_precision_unconfirmed"
