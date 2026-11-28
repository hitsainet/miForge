"""The default provider: generate from the parameters miForge already holds  (FR-004.2, FR-004.11).

**This is the implementation that makes acceptance criterion 20 true for scoring.** ADR-009:
*"one provider interface, in-process by default or any OpenAI-compatible endpoint."* The
in-process path needs no endpoint, no credential and no sibling, which is why it is the
default and why a full test-and-retention pass completes with nothing else running.

## Budget first, then generate

E-24, in acceptance criterion 13's shape. A scoring pass allocates a KV cache on the card
**already** holding gradients, FP32 master weights and 8-bit optimizer states (BRD-02 §6).
Failing in the allocator partway through a pass is the worse outcome: it kills a run at an
evaluation rather than at a step, with a `torch` OOM instead of a sentence an operator can
act on. So the pass declares what it needs and refuses by name, naming **both** numbers -
the estimate and the free figure - because either alone is unreadable.

The estimate is deliberately crude and deliberately explicit. It is not a model of
attention; it is ``prompts x tokens x bytes_per_token``, with ``bytes_per_token``
configuration. A crude estimate that is stated is a budget; a precise one that is guessed is
the fabricated value ADR-018 forbids.

## It writes nothing

No optimizer, no ``backward``, no ``.step()``, no save is imported or reachable from this
module (FR-004.20), and a static test asserts the absence. Generation during training shares
the process with a live optimizer state, and the cheapest way to corrupt a run is to measure
it carelessly.

## "ours" is not an identity

:attr:`identity` carries the **parameter-set digest**, not just the model name. Two steps of
one run are two different parameter sets and their scores are not interchangeable; for every
other provider the model name settles it, and in-process is the one place that is easy to
forget.
"""

from __future__ import annotations

from typing import Any

from core.errors import refuse
from providers.base import Completion, GenerationParams, InferenceProvider, ProviderIdentity
from providers.model_handle import ModelHandle

#: ADR-009's name for the provider that generates from the parameters miForge holds. Never a
#: sibling's name, and never "local": the distinction that matters is *whose memory*.
PROVIDER_KIND = "in_process"

#: Capability names this provider declares. **Generation only.** It exposes no embeddings
#: and - since R-02.19's amendment - no activations: *"not a capability of R-02.49's
#: inference provider, whose scope is generation and which exposes no activations."* A
#: declared-but-unused capability slot is how the next reader concludes the provider is the
#: route, which is exactly what DEF-10 reported.
CAPABILITY_GENERATION = "generation"

#: Default bytes of KV cache per generated token, per prompt in flight. A stated figure for
#: a 1-2B model in bf16 with a modest number of layers; configuration overrides it. It is an
#: order-of-magnitude budget, which is what a refusal-before-allocation needs.
DEFAULT_KV_BYTES_PER_TOKEN = 128 * 1024


class InProcessProvider(InferenceProvider):
    """Generate through a :class:`providers.model_handle.ModelHandle`.

    Args:
        handle: The held parameter set. :class:`providers.fakes.FakeModelHandle` satisfies
            this with no card, which is what makes every test below this line GPU-free.
        kv_bytes_per_token: The budget coefficient. Explicit so a benchmark can state it -
            ADR-024: a number quoted without its parameters is not a measurement.
        batch_bound: The most prompts sent to the handle in one call. Configuration with a
            declared bound rather than "as many as fit": the free figure moves while a run
            is stepping, so a bound computed from it is stale the moment it is taken.
    """

    def __init__(
        self,
        handle: ModelHandle,
        *,
        kv_bytes_per_token: int = DEFAULT_KV_BYTES_PER_TOKEN,
        batch_bound: int = 8,
    ) -> None:
        if batch_bound < 1:
            raise refuse(
                "inference_provider_unavailable",
                f"A batch bound must be at least one prompt; got {batch_bound}. Refusing "
                f"rather than clamping to 1: a clamped bound is a configuration the "
                f"operator did not choose.",
                provider=PROVIDER_KIND,
                batch_bound=batch_bound,
            )
        self._handle = handle
        self._kv_bytes_per_token = kv_bytes_per_token
        self._batch_bound = batch_bound

    # ---------------------------------------------------------------- identity
    @property
    def identity(self) -> ProviderIdentity:
        """Who this is. ``base_url`` is ``None`` **and that is the distinction**, not a gap.

        The parameter-set digest rides in ``engine_version``, which the interface documents
        as *"what the provider reports about itself, when it does. Free-form because each
        engine reports something different and inventing a common shape would discard
        detail."* For this provider what it reports about itself **is** which parameters it
        holds, and the field is the contract's own place for it - rather than a sixth field
        on a frozen gate type F4 does not own.
        """
        return ProviderIdentity(
            kind=PROVIDER_KIND,
            model=self._handle.model_identity(),
            base_url=None,
            engine_version=self._handle.parameter_set_digest(),
        )

    @property
    def parameter_set_digest(self) -> str:
        """Which parameters answered (FR-004.4). Recorded on every score."""
        return self._handle.parameter_set_digest()

    @property
    def device_uuid(self) -> str | None:
        """The card, by UUID (ADR-008). Never an index, never a bus address."""
        return self._handle.device_uuid()

    def available(self) -> bool:
        """True when a parameter set is held. **No network, no allocation.**

        Availability is asked by a listing endpoint, so it must not probe: a listing whose
        latency depends on a remote host is a listing that times out, and a listing that
        allocates is a listing that can OOM a run.
        """
        try:
            return bool(self._handle.model_identity())
        except Exception:
            # An unidentifiable parameter set is not available for scoring - it could not
            # name what produced the number (R-02.16). Reported as unavailable rather than
            # raised, because `available()` answers a listing.
            return False

    def capabilities(self) -> frozenset[str]:
        """``{generation}``. Declared, never inferred."""
        return frozenset({CAPABILITY_GENERATION})

    # ---------------------------------------------------------------- budget
    def kv_cache_estimate_bytes(self, prompt_count: int, max_tokens: int) -> int:
        """The declared KV-cache requirement for a batch. Named so a test can assert it."""
        return min(prompt_count, self._batch_bound) * max_tokens * self._kv_bytes_per_token

    def _check_budget(self, prompt_count: int, params: GenerationParams) -> None:
        """Refuse **before** allocating (E-24, criterion 13's shape).

        Skipped when the handle reports no card: a CPU-resident parameter set has no device
        budget to spend, and checking against a fabricated figure would be the silent
        default ADR-018 forbids - in the direction that refuses a legitimate pass.
        """
        if self._handle.device_uuid() is None:
            return
        free = self._handle.free_vram_bytes()
        need = self.kv_cache_estimate_bytes(prompt_count, params.max_tokens)
        if need > free:
            raise refuse(
                "provider_budget_exceeded",
                f"Scoring needs about {need} bytes of KV cache and {free} bytes are free "
                f"on this card. Refusing before allocating: an allocator failure partway "
                f"through a pass kills the run at an evaluation instead of reporting that "
                f"the pass does not fit. Lower the batch bound, lower max_tokens, or "
                f"subsample the set.",
                provider=PROVIDER_KIND,
                estimate_bytes=need,
                free_bytes=free,
                prompts=prompt_count,
                max_tokens=params.max_tokens,
                batch_bound=self._batch_bound,
                device_uuid=self._handle.device_uuid(),
            )

    # ---------------------------------------------------------------- generation
    def generate(
        self, prompts: list[str], params: GenerationParams, **options: Any
    ) -> list[Completion]:
        """One completion per prompt, in input order, inside the declared budget.

        Raises:
            Refusal: ``provider_budget_exceeded`` before any allocation (E-24);
                ``inference_provider_option_unknown`` for an option this provider does not
                honour - **refused rather than ignored**, because FR-004.8's whole point is
                that a silently dropped parameter produces a number whose meaning depends
                on which server answered;
                ``inference_provider_reply_unparseable`` when the handle returns a list of
                a different length than it was asked for.
        """
        if options:
            raise refuse(
                "inference_provider_option_unknown",
                f"Unknown generation options {sorted(options)} were passed to the "
                f"in-process provider. Refusing rather than ignoring them: a silently "
                f"dropped option is a request the caller believes it made, and a score "
                f"taken under a parameter the engine ignored is not the score it claims.",
                provider=PROVIDER_KIND,
                unknown=sorted(options),
            )
        if not prompts:
            raise refuse(
                "evaluation_set_empty",
                "A generation call was made with no prompts. ScoreRecord.n has minimum 1, "
                "so there is no score to record over zero prompts.",
                provider=PROVIDER_KIND,
            )
        self._check_budget(len(prompts), params)

        identity = self.identity
        out: list[Completion] = []
        for start in range(0, len(prompts), self._batch_bound):
            chunk = prompts[start : start + self._batch_bound]
            texts = self._handle.generate(chunk, params)
            if len(texts) != len(chunk):
                raise refuse(
                    "inference_provider_reply_unparseable",
                    f"The model handle returned {len(texts)} completions for "
                    f"{len(chunk)} prompts. Refusing rather than padding: results are "
                    f"joined to prompts positionally, so a short list misattributes every "
                    f"score after the gap.",
                    provider=PROVIDER_KIND,
                    asked=len(chunk),
                    received=len(texts),
                )
            out.extend(Completion(text=text, provider=identity, params=params) for text in texts)
        return out
