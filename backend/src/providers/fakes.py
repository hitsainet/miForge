"""The fake handle and provider that make this feature laptop-testable  (ADR-023; FR-004.11).

**Written before :mod:`providers.in_process` uses it, deliberately.** ADR-023 is a
requirement on the layout, not an accident: *a developer without a GPU can run everything
except a training step.* For F4 that means everything except in-process generation on real
weights, and the only way to get there is a handle whose ``generate`` is arithmetic.

## A stub, not a mock

ADR-016 names over-mocked stubs among the reasons a suite stays green over a real bug. So
:class:`FakeModelHandle` runs the **real** :class:`providers.in_process.InProcessProvider`
code path - the budget check, the identity assembly, the ordering guarantee - and replaces
only the tensors. The analogue on the HTTP side is F3's ``StubJudgeServer``, which replaces
only the socket.

## It records what it was asked

``calls`` and ``prompts_seen`` exist because ADR-016 requires the **payload and the call
count** to be asserted: *"was called" passes against a call sending the wrong arguments*,
and for a scoring pass the arguments are what the number means.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence

from core.errors import Refusal
from providers.base import Completion, GenerationParams, ProviderIdentity

#: The digest a fake handle reports unless told otherwise. A readable constant rather than
#: 64 hex characters: a test asserting provenance should read as provenance.
FAKE_DIGEST = "fake-parameter-set@step0"


class FakeModelHandle:
    """A :class:`providers.model_handle.ModelHandle` with no tensors and no card.

    Args:
        answers: Completion per prompt. A callable receives the prompt and its index, so a
            test can make the ``k``-th answer wrong without hand-building a list whose
            fields agree with the expectation by construction.
        free_vram_bytes: What :meth:`free_vram_bytes` reports. Large by default so the
            budget check passes; set it small to exercise E-24's refusal.
        digest: What :meth:`parameter_set_digest` reports.
        model: What :meth:`model_identity` reports.
        device_uuid: ADR-008's UUID, or ``None`` for a CPU-resident fake.
        fail_after: Raise ``raise_error`` once this many prompts have been answered, for
            M-6's partial-failure control. ``None`` never fails.
    """

    def __init__(
        self,
        *,
        answers: Callable[[str, int], str] | Sequence[str] | str = "ok",
        free_vram_bytes: int = 64 * 1024**3,
        digest: str = FAKE_DIGEST,
        model: str = "fake-model",
        device_uuid: str | None = "GPU-fake-0000",
        fail_after: int | None = None,
        raise_error: BaseException | None = None,
    ) -> None:
        self._answers = answers
        self._free = free_vram_bytes
        self._digest = digest
        self._model = model
        self._device_uuid = device_uuid
        self._fail_after = fail_after
        self._raise = raise_error
        #: One entry per :meth:`generate` call: ``(prompts, params)``.
        self.calls: list[tuple[tuple[str, ...], GenerationParams]] = []
        #: Every prompt this handle has been asked, flattened and in order.
        self.prompts_seen: list[str] = []
        self.answered = 0

    # ------------------------------------------------------------------ the port
    def parameter_set_digest(self) -> str:
        return self._digest

    def model_identity(self) -> str:
        return self._model

    def device_uuid(self) -> str | None:
        return self._device_uuid

    def free_vram_bytes(self) -> int:
        return self._free

    def generate(self, prompts: Sequence[str], params: GenerationParams) -> list[str]:
        """Answer each prompt, in order, failing after ``fail_after`` if configured."""
        self.calls.append((tuple(prompts), params))
        out: list[str] = []
        for index, prompt in enumerate(prompts):
            if self._fail_after is not None and self.answered >= self._fail_after:
                raise self._raise or Refusal(
                    "inference_provider_unreachable",
                    "The fake handle was configured to fail partway through this pass.",
                )
            self.prompts_seen.append(prompt)
            self.answered += 1
            out.append(self._answer_for(prompt, index))
        return out

    def _answer_for(self, prompt: str, index: int) -> str:
        if callable(self._answers):
            return self._answers(prompt, index)
        if isinstance(self._answers, str):
            return self._answers
        # Positional, and **refused rather than wrapped** when it runs out: a modulo here
        # would silently answer prompt 7 with answer 0 and the test would still pass.
        if index >= len(self._answers):
            raise AssertionError(
                f"FakeModelHandle was given {len(self._answers)} answers and asked for "
                f"prompt {index}. Refusing rather than wrapping: a wrapped answer makes a "
                f"mis-sized fixture look like a working pass."
            )
        return self._answers[index]


class FakeProvider:
    """A :class:`providers.base.InferenceProvider` that answers without any model at all.

    For the cases where the subject is not generation - the registry's rows, the pinning
    refusal, the manifest payload - so those tests do not drag the in-process path's budget
    arithmetic in with them.
    """

    def __init__(
        self,
        *,
        kind: str = "in_process",
        model: str = "fake-model",
        base_url: str | None = None,
        answers: Callable[[str, int], str] | Sequence[str] | str = "ok",
        available: bool = True,
        capabilities: frozenset[str] = frozenset({"generation"}),
    ) -> None:
        self._identity = ProviderIdentity(kind=kind, model=model, base_url=base_url)
        self._answers = answers
        self._available = available
        self._capabilities = capabilities
        self.calls: list[tuple[tuple[str, ...], GenerationParams]] = []

    @property
    def identity(self) -> ProviderIdentity:
        return self._identity

    def available(self) -> bool:
        return self._available

    def capabilities(self) -> frozenset[str]:
        return self._capabilities

    def generate(
        self, prompts: list[str], params: GenerationParams, **options: object
    ) -> list[Completion]:
        if options:
            raise Refusal(
                "inference_provider_option_unknown",
                f"Unknown generation options {sorted(options)}.",
            )
        self.calls.append((tuple(prompts), params))
        handle = FakeModelHandle(answers=self._answers)
        return [
            Completion(text=text, provider=self._identity, params=params)
            for text in handle.generate(prompts, params)
        ]
