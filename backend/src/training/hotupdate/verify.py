"""Verifying the reconstruction against the serving process  (R-02.2; FR-011.9 - FR-011.13).

One forward pass on a fixed prompt, compared with what the serving process answers for the
same prompt. **A mismatch aborts the run** - before any step, any test and any save, with
nothing written. Without it, a wrongly reconstructed graph trains silently and *every score
is against a different model than the one being served.*

## Log-probabilities, not logits

R-02.2 compares **log-probabilities over an explicit token set**. It said *"logits"* until
2026-10-13, and that reading was **unperformable**: the `/v1` surface returns ``logprobs``,
``top_logprobs`` (capped at 20) and ``allowed_token_ids`` - log-probabilities, never logits,
the two differing by the log-partition term. Both repositories corrected it the same day
(BRD-01's criteria 1, 11 and 13 too; its criterion 4 keeps "logits" and is right to, being a
miLLM-internal comparison). **The token set is passed explicitly** rather than taking
whatever ``top_logprobs`` returns, because a top-k truncation makes the comparison's domain
depend on the model being compared.

## The tolerance is R-02.2's, and it is provisional

Provisionally **1x10^-2 in log-probability space, as the maximum absolute difference over
the compared token set**, configurable - and to be replaced, before §5.10's acceptance, by
**the observed spread across repeated identical calls on unchanged weights, times a
margin**. That measurement is this feature's to produce (OQ-1) and needs a live serving
process, so until then the value is **labelled provisional in every recorded result**: *a
provisional bar that stops reading as provisional is MUT-14 shipping without a mutation.*

The statistic is named, so it is the one used: ``max(abs(local - remote)) <= tol`` over the
set. **Not a mean, and not ``allclose`` with its own hidden ``rtol``** - a different
statistic is a different bar at the same number.

## It goes through R-02.49's provider - and since 2026-10-16 the provider can answer

FR-011.10 requires the serving process to be reached through the inference provider, not
through a second private HTTP path. Until 2026-10-16 the shipped OpenAI-compatible provider
filled no log-probabilities, so this module could only refuse - and nothing in the run
called it. Now ``OpenAICompatibleProvider.next_token_logprobs`` uses the legacy
``/v1/completions`` scoring mode miLLM serves (``logprobs`` + ``allowed_token_ids``,
renormalised over the set), :func:`restricted_logprobs` computes the local side with the same
arithmetic, and :class:`Verifier` is called by the run before step 1 (``AcquiredModel.verifier``).
"""

from __future__ import annotations

import logging
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

from core.errors import Refusal, refuse

log = logging.getLogger(__name__)

#: The quantity compared, recorded with every result. A verification whose terms are not
#: recorded cannot be re-run later against the same bar.
QUANTITY = "log_probability"


@dataclass(frozen=True)
class VerifyResult:
    """The comparison and **its terms**: quantity, token set, tolerance, provider."""

    matched: bool
    quantity: str
    token_set: tuple[str, ...]
    tolerance: float
    tolerance_is_provisional: bool
    provider: str
    max_abs_difference: float | None
    statistic: str = "max_absolute_difference"
    missing_tokens: tuple[str, ...] = ()
    #: ``native`` | ``float32`` - which precision BOTH sides computed in (F11 14.3b).
    precision: str = "native"
    #: The precision that was ASKED for, when it differs from :attr:`precision`: float32 was
    #: requested, the serving side refused it by name, and the native comparison ran instead.
    requested_precision: str | None = None
    #: miLLM's stated reason for that refusal (``model_not_float32_castable``, ``memory``,
    #: ``device``, ``busy``). ``None`` when nothing fell back.
    precision_fallback_reason: str | None = None

    def record(self) -> dict[str, Any]:
        """The run-log form. Every term, because the bar has to be re-readable."""
        return {
            "event": "reconstruction_verified" if self.matched else "reconstruction_mismatch",
            "quantity": self.quantity,
            "statistic": self.statistic,
            "token_set": list(self.token_set),
            "token_set_size": len(self.token_set),
            "tolerance": self.tolerance,
            # NOT decoration. Until the spread measurement exists, every reader of this
            # record must be able to see that the bar is a starting point for measurement
            # rather than a measured result (R-02.2, OQ-1).
            "tolerance_is_provisional": self.tolerance_is_provisional,
            "provider": self.provider,
            "max_abs_difference": self.max_abs_difference,
            "missing_tokens": list(self.missing_tokens),
            "matched": self.matched,
            "precision": self.precision,
            "requested_precision": self.requested_precision or self.precision,
            "precision_fallback_reason": self.precision_fallback_reason,
        }


@dataclass
class LocalLogprobs:
    """miForge's own log-probabilities over the token set, from the rebuilt graph.

    A tiny value object rather than a torch call inlined into :func:`check`, so the
    comparison - the half that carries the tolerance and the refusal - is testable with no
    card and no serving process. The forward pass itself needs both.
    """

    values: dict[str, float] = field(default_factory=dict)


def logprobs_from_provider(
    provider: Any,
    prompt: str,
    token_ids: Sequence[int],
    *,
    add_special_tokens: bool = True,
    precision: str | None = None,
) -> dict[str, float]:
    """The serving process's log-probabilities over ``token_ids``, **through R-02.49's provider**.

    Uses the provider's ``next_token_logprobs`` capability - the legacy ``/v1/completions``
    scoring mode, renormalised over the set (miLLM Feature 25; vLLM ``processed_logprobs``).
    Keyed by ``str(token_id)``, the keys :func:`check` compares.

    Raises:
        Refusal: ``provider_logprobs_unavailable`` when the provider has no such capability.
            **The refusal is the honest answer**: the alternative is a second, private HTTP
            path, which FR-011.10 forbids for the reason R-02.49 exists.
    """
    scorer = getattr(provider, "next_token_logprobs", None)
    if scorer is None:
        raise refuse(
            "provider_logprobs_unavailable",
            "The configured inference provider cannot return next-token log-probabilities, "
            "so the reconstruction check cannot be made. miForge will NOT open a second HTTP path "
            "to the serving process to get them.",
            provider=str(getattr(provider, "identity", None)),
            capability="next-token logprobs over an explicit token set (allowed_token_ids)",
        )
    # The precision is passed only when one is ASKED for, so a provider that predates the
    # float32 mode (and every native comparison) is called exactly as before.
    extra = {} if precision is None else {"precision": precision}
    values = scorer(
        prompt, [int(i) for i in token_ids], add_special_tokens=add_special_tokens, **extra
    )
    return {str(token): float(value) for token, value in values.items()}


def restricted_logprobs(logits: Any, token_ids: Sequence[int]) -> dict[str, float]:
    """``log_softmax`` of one position's logits over EXACTLY ``token_ids``, in float32.

    The same arithmetic miLLM's scoring mode uses at temperature 0
    (``millm/services/next_token_scores.py``: no temperature scaling, renormalised over the
    allowed set). Computing it any other way - full-vocabulary log-softmax, or a dtype other
    than float32 - would make a correct reconstruction disagree by the log-partition term.
    """
    import torch

    z = logits.float()
    ids = torch.tensor([int(i) for i in token_ids], dtype=torch.long, device=z.device)
    values = torch.log_softmax(z[ids], dim=0)
    return {str(int(i)): float(v) for i, v in zip(token_ids, values.tolist(), strict=True)}


def _last_position_logits(module: Any, input_ids: Sequence[int]) -> Any:
    import torch

    device = next(iter(module.parameters())).device
    with torch.no_grad():
        output = module(input_ids=torch.tensor([list(input_ids)], device=device))
    logits = output.logits if hasattr(output, "logits") else output[0]
    return logits[0, -1]


#: The precisions R-02.2's comparison can run in (F11 14.3b).
NATIVE = "native"
FLOAT32 = "float32"


def _last_position_logits_float32(module: Any, input_ids: Sequence[int], device: str) -> Any:
    """The last position's logits from a **float32 copy** of ``module``'s tensors (14.3b).

    ``torch.func.functional_call`` runs the module's own forward with every floating
    parameter and buffer replaced by a float32 copy on ``device`` - so the rebuilt graph is
    the one evaluated, and **the served tensors are never cast, moved or written**: they are
    miLLM's, and an in-place ``.float()`` would change what miLLM serves. Integer buffers
    keep their dtype. ~4 bytes per parameter on ``device`` for the duration of one forward.
    """
    import torch
    from torch.func import functional_call

    target = torch.device(device)
    tensors: dict[str, Any] = {}
    for name, tensor in module.named_parameters():
        tensors[name] = tensor.detach().to(device=target, dtype=torch.float32)
    for name, buffer in module.named_buffers():
        tensors[name] = buffer.detach().to(
            device=target, dtype=torch.float32 if buffer.is_floating_point() else buffer.dtype
        )
    with torch.no_grad():
        output = functional_call(
            module,
            tensors,
            args=(),
            kwargs={"input_ids": torch.tensor([list(input_ids)], device=target)},
        )
    logits = output.logits if hasattr(output, "logits") else output[0]
    return logits[0, -1]


def choose_token_set(logits: Any, size: int) -> list[int]:
    """The explicit token set: the local model's top ``size`` next tokens, by id.

    Chosen from the RECONSTRUCTION, then scored on both sides over that fixed set - so the
    domain does not depend on the serving side's own top-k (the truncation R-02.2 avoids).
    """
    import torch

    # Clamped to the vocabulary: a set larger than the vocabulary cannot be chosen.
    k = max(1, min(int(size), int(logits.numel())))
    return [int(i) for i in torch.topk(logits.float(), k=k).indices.tolist()]


@dataclass
class Verifier:
    """R-02.2's check as the run calls it: once, after the rebuild, **before step 1**.

    Supplied on the acquisition (``AcquiredModel.verifier``) by
    :class:`training.hotupdate.attach.AttachAcquirer` and called by ``training/tasks.py``
    on its presence (ADR-022). Until 2026-10-16 nothing called this module outside tests,
    and no provider filled log-probabilities - so a wrongly reconstructed graph would have
    trained silently against a model other than the one served.
    """

    module: Any
    tokenizer_loader: Any
    provider: Any
    prompt: str
    token_set_size: int
    tolerance: float
    tolerance_is_provisional: bool
    add_special_tokens: bool = True
    #: ``native`` (both sides in the served dtype) or ``float32`` (both sides float32, the
    #: serving side REQUIRED to confirm it). F11 14.3b.
    precision: str = NATIVE
    #: Where the float32 forward runs (``cpu`` by default: the card is shared).
    float32_device: str = "cpu"
    #: The NATIVE bar, used only when float32 was asked for and the serving side REFUSED it by
    #: name (``provider_scoring_precision_unavailable``: miLLM's 409, e.g. a bitsandbytes model
    #: it cannot copy to float32). ``None`` means no fallback - that refusal then aborts the
    #: run. Never used for an answer that merely lacks the echo: that is refused, not fallen
    #: back from, because it may be a bf16 answer dressed as float32.
    native_tolerance: float | None = None
    native_tolerance_is_provisional: bool = True
    result: VerifyResult | None = None

    def __call__(self) -> VerifyResult:
        if self.precision not in (NATIVE, FLOAT32):
            raise refuse(
                "verification_precision_unknown",
                f"The reconstruction check runs in {NATIVE!r} or {FLOAT32!r}; got "
                f"{self.precision!r}. Refusing rather than guessing which bar applies.",
                precision=self.precision,
            )
        tokenizer = self.tokenizer_loader()
        input_ids = list(
            tokenizer(self.prompt, add_special_tokens=self.add_special_tokens)["input_ids"]
        )
        try:
            self.result = self._compare(input_ids, self.precision)
        except Refusal as refusal:
            if not (
                self.precision == FLOAT32
                and refusal.code == "provider_scoring_precision_unavailable"
                and self.native_tolerance is not None
            ):
                raise
            reason = _refusal_reason(refusal)
            # NOT silent: the operator reads that this run's reconstruction was checked at the
            # bf16 bar, which cannot see a ~1 % error, and why.
            log.warning(
                "float32 verification unavailable - falling back to the native comparison",
                extra={
                    "code": "reconstruction_verification_precision_fallback",
                    "requested_precision": FLOAT32,
                    "fallback_precision": NATIVE,
                    "fallback_reason": reason,
                    "fallback_tolerance": self.native_tolerance,
                    "fallback_tolerance_is_provisional": self.native_tolerance_is_provisional,
                    "detail": refusal.message,
                },
            )
            self.result = self._compare(input_ids, NATIVE, fallback_reason=reason)
        abort_on_mismatch(self.result)
        return self.result

    def _compare(
        self, input_ids: list[int], precision: str, *, fallback_reason: str | None = None
    ) -> VerifyResult:
        """One comparison in ``precision``, judged against THAT precision's bar."""
        if precision == FLOAT32:
            logits = _last_position_logits_float32(self.module, input_ids, self.float32_device)
            tolerance, provisional = self.tolerance, self.tolerance_is_provisional
        else:
            logits = _last_position_logits(self.module, input_ids)
            if fallback_reason is not None and self.native_tolerance is not None:
                tolerance = self.native_tolerance
                provisional = self.native_tolerance_is_provisional
            else:
                tolerance, provisional = self.tolerance, self.tolerance_is_provisional
        token_ids = choose_token_set(logits, self.token_set_size)
        local = LocalLogprobs(values=restricted_logprobs(logits, token_ids))
        remote = logprobs_from_provider(
            self.provider,
            self.prompt,
            token_ids,
            add_special_tokens=self.add_special_tokens,
            precision=FLOAT32 if precision == FLOAT32 else None,
        )
        identity: Any = self.provider.identity
        return check(
            local,
            remote,
            token_set=[str(i) for i in token_ids],
            tolerance=tolerance,
            tolerance_is_provisional=provisional,
            provider=str(
                identity.record() if callable(getattr(identity, "record", None)) else identity
            ),
            precision=precision,
            requested_precision=self.precision if fallback_reason is not None else None,
            precision_fallback_reason=fallback_reason,
        )


def check(
    local: LocalLogprobs,
    remote: dict[str, float],
    *,
    token_set: Sequence[str],
    tolerance: float,
    tolerance_is_provisional: bool,
    provider: str,
    precision: str = "native",
    requested_precision: str | None = None,
    precision_fallback_reason: str | None = None,
) -> VerifyResult:
    """Compare, by the statistic R-02.2 names. **The bar, not a bar.**

    A token the two sides do not both report is **not** skipped: it is recorded in
    ``missing_tokens`` and makes the result a mismatch. Skipping it is how a comparison over
    an empty intersection returns "matched" - the vacuity failure this project keeps finding.
    """
    missing = tuple(
        token for token in token_set if token not in local.values or token not in remote
    )
    differences = [
        abs(local.values[token] - remote[token])
        for token in token_set
        if token in local.values and token in remote
    ]
    worst = max(differences) if differences else None
    matched = not missing and worst is not None and worst <= tolerance
    result = VerifyResult(
        matched=matched,
        quantity=QUANTITY,
        token_set=tuple(str(token) for token in token_set),
        tolerance=tolerance,
        tolerance_is_provisional=tolerance_is_provisional,
        provider=provider,
        max_abs_difference=worst,
        missing_tokens=missing,
        precision=precision,
        requested_precision=requested_precision,
        precision_fallback_reason=precision_fallback_reason,
    )
    log.info(
        "reconstruction verification",
        extra={"code": "reconstruction_verification", **result.record()},
    )
    return result


def abort_on_mismatch(result: VerifyResult) -> None:
    """Abort the run on a mismatch, **before any step, test or save** (FR-011.12).

    Raises:
        Refusal: ``reconstruction_mismatch``, naming the tolerance used, the quantity
            compared and whether the bar was the provisional one. *Every number names its
            scale and sample* (ADR-012), and a tolerance is a number.
    """
    if result.matched:
        return
    raise refuse(
        "reconstruction_mismatch",
        f"The model rebuilt over miLLM's handles does not agree with what miLLM answers: "
        f"max absolute difference in {result.quantity} space over "
        f"{len(result.token_set)} tokens was {_rendered(result.max_abs_difference)}, "
        f"tolerance {result.tolerance}{_provisional_note(result)}"
        f"{_fallback_note(result)}{_missing_note(result)}. "
        f"The run is aborted before its first step and nothing is written: training a "
        f"wrongly reconstructed graph scores every test against a different model than the "
        f"one being served.",
        **result.record(),
    )


def _rendered(value: float | None) -> str:
    """The worst difference, or the word for not having one. **Never ``0`` for absent**: a
    zero difference is a perfect match, which is the opposite of what an unmeasurable
    comparison means."""
    return "unmeasurable" if value is None else str(round(value, 6))


def _provisional_note(result: VerifyResult) -> str:
    """So the bar reads as provisional **wherever it appears** (R-02.2, OQ-1)."""
    if not result.tolerance_is_provisional:
        return ""
    return " (PROVISIONAL - the tolerance awaits a measured spread)"


#: miLLM's header naming why it refused float32 scoring (its FR-25.15; the vendored
#: scoring contract ``millm-scoring-api-v1.json`` -> ``scoring_precision.refusal.reason_header``).
SCORING_PRECISION_REASON_HEADER = "X-miLLM-Scoring-Precision-Reason"


def _refusal_reason(refusal: Refusal) -> str:
    """The serving side's stated reason: its header first, then the message's ``(reason: x)``."""
    headers = refusal.details.get("response_headers") or {}
    reason = headers.get(SCORING_PRECISION_REASON_HEADER.lower()) or refusal.details.get("reason")
    return str(reason) if reason else "unstated"


def _fallback_note(result: VerifyResult) -> str:
    """Say when the comparison is not the one asked for, and why."""
    if result.precision_fallback_reason is None:
        return ""
    return (
        f" - compared in {result.precision} because miLLM refused "
        f"{result.requested_precision} scoring (reason: {result.precision_fallback_reason})"
    )


def _missing_note(result: VerifyResult) -> str:
    if not result.missing_tokens:
        return ""
    return f"; tokens missing from one side: {list(result.missing_tokens)}"
