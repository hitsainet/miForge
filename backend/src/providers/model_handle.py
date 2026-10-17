"""The port the in-process provider generates through  (F4 FR-004.11; R-02.49, ADR-009, ADR-023).

**F4 declares this; F1's acquisition (R-02.46) satisfies it.** Declared here rather than in
F1 because ADR-009's in-process provider is F4's, and because a fake implementation is what
keeps this whole feature buildable and testable without a GPU (ADR-023): a developer without
a card runs all of F4 except one GPU-marked file.

## Keep it this small

Every method added here is a method F1 owes, and F1 is already written and landed. If the
in-process provider wants more, it belongs inside :mod:`providers.in_process`, built out of
what the handle already gives. The four below are the minimum a held parameter set must
offer to be generated from and recorded honestly:

* ``parameter_set_digest`` - **which** parameters answered. ADR-009 says every score records
  its provider, and for in-process "ours" is not an identity: two steps of one run are two
  different parameter sets and their scores are not interchangeable.
* ``device_uuid`` - ADR-008: by UUID, never an index or a bus address. Indices inverted once
  on this cluster and a copied bus address unbound the root NVMe.
* ``free_vram_bytes`` - for E-24's refusal *before* allocating. A scoring pass allocates a KV
  cache on the card already holding gradients, FP32 master weights and 8-bit optimizer
  states (BRD-02 §6), and an allocator failure partway through a pass is the worse outcome.
* ``generate`` - one completion per prompt, in order.

## Why an adapter rather than making F1 implement the Protocol

:class:`training.acquisition.base.AcquiredModel` is a dataclass holding ``module``,
``tokenizer`` and an :class:`~training.acquisition.base.AcquisitionRecord`. It already
carries everything this port needs, so F4 adapts it instead of obliging F1 to grow methods -
which keeps the obligation on F4's side of the seam and means F1's file is not edited by a
second feature. ``AcquiredModel`` is imported lazily, inside the adapter, so this module
imports with no torch and no F1 import cost on the HTTP path.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any, Protocol, runtime_checkable

from core.errors import refuse
from providers.base import GenerationParams


@runtime_checkable
class ModelHandle(Protocol):
    """The minimum a held parameter set must offer to be generated from."""

    def parameter_set_digest(self) -> str:
        """A digest naming *which* parameters answered. Recorded on every score."""
        ...

    def device_uuid(self) -> str | None:
        """The card the parameters sit on, by UUID (ADR-008). ``None`` on CPU."""
        ...

    def free_vram_bytes(self) -> int:
        """Free device memory right now, for the E-24 budget refusal."""
        ...

    def model_identity(self) -> str:
        """What to record as ``ProviderIdentity.model`` for this parameter set."""
        ...

    def generate(self, prompts: Sequence[str], params: GenerationParams) -> list[str]:
        """One completion per prompt, **in input order**.

        Order is part of the contract: callers join results to prompts positionally, and a
        handle that reordered or dropped one would misattribute every score after the gap.
        """
        ...


class AcquiredModelHandle:
    """A :class:`ModelHandle` over F1's :class:`~training.acquisition.base.AcquiredModel`.

    The adapter exists so the obligation F4 placed on F1 (FPRD §10) is discharged on F4's
    side of the seam: F1 holds the parameters and defines no generation surface, and this is
    the smallest thing that turns the one into the other.

    **Generation here is GPU work and is the one GPU-marked path in this feature.** Nothing
    else in F4 needs a card (ADR-023), which is why :class:`providers.fakes.FakeModelHandle`
    exists and why it is written before this is used.
    """

    def __init__(self, acquired: Any, *, step: int | None = None) -> None:
        """Wrap an acquired model.

        Args:
            acquired: F1's ``AcquiredModel``. Typed ``Any`` for the reason F1 gives for its
                own ``module``/``tokenizer`` fields: naming the type would pull torch into
                this module's import, which ADR-023's layout forbids here.
            step: The step whose parameters these are, folded into the digest. ``None``
                means "as acquired", which is step 0's answer and is a fact rather than a
                missing value.
        """
        self._acquired = acquired
        self._step = step

    # ------------------------------------------------------------------ identity
    def parameter_set_digest(self) -> str:
        """``<starting_weights_sha256>@step<N>``.

        The starting hash is R-02.47's recorded value and the step is what makes two
        parameter sets of one run distinguishable. **Not** the starting hash alone: every
        score of a run would then claim the same parameters, which is the "ours is not an
        identity" failure this field exists to prevent.
        """
        record = getattr(self._acquired, "record", None)
        base = getattr(record, "starting_weights_sha256", None)
        if not base:
            raise refuse(
                "inference_provider_unavailable",
                "The acquired model carries no starting weights hash, so a score taken "
                "from it could not name which parameters produced it (R-02.16). Refusing "
                "rather than recording an unidentified parameter set.",
                provider="in_process",
            )
        suffix = "as-acquired" if self._step is None else f"step{self._step}"
        return f"{base}@{suffix}"

    def model_identity(self) -> str:
        """The run's model identity: ``base_model`` from F1's acquisition record."""
        record = getattr(self._acquired, "record", None)
        identity = getattr(record, "base_model", None)
        if not identity:
            raise refuse(
                "inference_provider_unavailable",
                "The acquired model names no base model, so there is no model identity to "
                "record beside a score. Refusing rather than recording 'ours'.",
                provider="in_process",
            )
        return str(identity)

    def device_uuid(self) -> str | None:
        """ADR-008's UUID, straight off F1's acquisition. Never an index."""
        value = getattr(self._acquired, "device_uuid", None)
        return None if value is None else str(value)

    # ------------------------------------------------------------------ budget
    def free_vram_bytes(self) -> int:
        """Free bytes on the card holding the parameters, or ``0`` when there is no card.

        Zero for a CPU-resident model is deliberate and is **not** "unknown": a CPU handle
        has no device budget to spend, so the in-process provider's budget check is
        skipped explicitly rather than passing against a fabricated figure.
        """
        if self.device_uuid() is None:
            return 0
        import torch  # local: ADR-023 keeps torch out of this module's import

        if not torch.cuda.is_available():  # pragma: no cover - needs a card
            return 0
        free, _total = torch.cuda.mem_get_info()  # pragma: no cover - needs a card
        return int(free)  # pragma: no cover - needs a card

    # ------------------------------------------------------------------ generation
    #: The only tokenizer outputs forwarded to ``module.generate``. **Not** the whole
    #: ``BatchEncoding``, and the difference is a fixed crash - see :meth:`generate`.
    _GENERATE_INPUTS: tuple[str, ...] = ("input_ids", "attention_mask")

    def generate(self, prompts: Sequence[str], params: GenerationParams) -> list[str]:
        """Generate through the held module and tokenizer. **GPU path.**

        ``do_sample`` follows ``params.temperature``: zero means greedy, which is
        FR-004.5's default and the only case BRD-02's acceptance criterion 2 is written
        against ("with deterministic decoding and the same provider, exactly").

        *The lines below still carry ``pragma: no cover`` because CI has no card. They are
        no longer unexecuted:* ``tests/integration/test_in_process_generation_gpu.py`` is
        ``gpu``-marked and drives all of them on the 3090. **Three defects were found the
        first time this body ever ran** (2026-10-16, F4 task 8.3). All three were invisible
        to ``providers.fakes.FakeModelHandle``, which the only previous test used.

        **1. ``**encoded`` splatted ``token_type_ids`` and ``generate`` raised.** A
        ``BatchEncoding``'s keys come from the tokenizer's ``model_input_names``, and the
        **base** ``PreTrainedTokenizerFast`` declares
        ``['input_ids', 'token_type_ids', 'attention_mask']``. Llama- and GPT-family
        tokenizers override it to drop ``token_type_ids``; a generic fast tokenizer does
        not. So this raised ``ValueError: The following model_kwargs are not used by the
        model: ['token_type_ids']`` - a bare transformers internal, with no refusal code -
        for every such tokenizer, and worked for the rest.

        **That conditionality is why nothing caught it, and it is worse than a flat crash:**
        ADR-009's *default* provider generated fine on some tokenizers and hard-failed on
        others. A ``create`` run is the exposed case, because ``scratch.py`` deliberately
        builds no tokenizer - *"a creation run is given one"* - so whichever one it is given
        decides whether scoring works. Forwarding only :data:`_GENERATE_INPUTS` fixes it and
        also decouples this provider from a tokenizer-class attribute it does not own.

        **2. ``params.stop`` was accepted and dropped.** ``openai_compatible`` sends it
        (lines 206-212). FR-004.8 forbids the asymmetry: *"Where a scoring pass depends on a
        parameter being respected, the provider must declare support for it; otherwise F4
        refuses rather than sending it hopefully."* Two providers disagreeing about one
        parameter **recorded beside every score** is that failure with the disagreement
        inside miForge. Honoured by **truncating the decoded text** at the earliest stop
        sequence, excluding it - which is what an OpenAI-compatible server returns for
        ``stop``, and therefore what makes the two providers' numbers comparable rather than
        merely both non-silent.

        **``stop_strings=`` was tried first and deliberately rejected**, which is worth
        recording because it looks like the better answer: it ends generation early instead
        of generating and discarding. It raised
        ``ValueError: substring not found`` from ``StopStringCriteria.clean_tokenizer_vocab``,
        which round-trips a static ``"abcdef"`` probe through the tokenizer and cannot with a
        word-level vocabulary. **That is defect 1's shape again** - behaviour conditional on
        the tokenizer class, working on BPE and hard-failing elsewhere - introduced by the
        fix for it, in the same function, an hour later. So the cost is paid instead and
        named: generation runs to ``max_tokens`` and the tail is discarded, so ``stop``
        bounds the **returned text** and not the compute. E-24's budget already estimates
        from ``max_tokens``, so no budget claim is overstated by this; a scoring pass is
        simply no faster for setting ``stop``.

        **3. ``params.seed`` was accepted and dropped**, so sampled in-process scores were
        irreproducible while recording a seed asserting otherwise. FR-004.8 names ``seed``
        explicitly, the field's own comment says it *"determines the subsample **and any
        sampling**"*, and this project's rule is that an overclaimed seed is worse than
        none. Seeded **immediately before the loop and nowhere else**, which is the pattern
        ``acquisition/scratch.py`` states for the same reason: a seed set earlier is
        consumed by whatever drew from the RNG in between, so the recorded seed would not
        describe these completions.
        """
        import torch  # pragma: no cover - needs a card

        module = self._acquired.module  # pragma: no cover - needs a card
        tokenizer = self._acquired.tokenizer  # pragma: no cover - needs a card
        sampling = params.temperature > 0.0  # pragma: no cover - needs a card
        if params.seed is not None:  # pragma: no cover - needs a card
            # Defect 3. Before the loop, not inside it: per-prompt reseeding would make
            # every prompt draw the same sequence, which is reproducible and wrong.
            torch.manual_seed(params.seed)
        out: list[str] = []  # pragma: no cover - needs a card
        with torch.inference_mode():  # pragma: no cover - needs a card
            for prompt in prompts:
                encoded = tokenizer(prompt, return_tensors="pt").to(module.device)
                # Defect 1. Only the inputs the model's forward accepts.
                inputs = {name: encoded[name] for name in self._GENERATE_INPUTS if name in encoded}
                generated = module.generate(
                    **inputs,
                    max_new_tokens=params.max_tokens,
                    do_sample=sampling,
                    # Sampling parameters only when sampling. Passing `top_p` under greedy
                    # decoding is what produced transformers' "not valid for do_sample=
                    # False" warning on every call, and a warning nobody reads is how a
                    # real one gets missed.
                    temperature=params.temperature if sampling else None,
                    top_p=params.top_p if sampling else None,
                )
                text = tokenizer.decode(
                    generated[0][inputs["input_ids"].shape[-1] :], skip_special_tokens=True
                )
                out.append(_truncate_at_stop(text, params.stop))
        return out  # pragma: no cover - needs a card


def _truncate_at_stop(text: str, stop: Sequence[str]) -> str:
    """Cut ``text`` at the **earliest** stop sequence in it, excluding the sequence.

    Defect 2's whole mechanism. An OpenAI-compatible server stops at ``stop`` and excludes
    it from the reply, so truncating here is what makes an in-process score and an HTTP
    score under the same ``stop`` mean the same thing - without it the two providers would
    differ by exactly the stop sequence, which is the kind of quiet incomparability R-02.16
    records provenance to prevent.

    **Tokenizer-independent on purpose.** See :meth:`AcquiredModelHandle.generate` for why
    ``stop_strings=`` is not used: it works on a BPE vocabulary and raises on a word-level
    one, and behaviour conditional on the tokenizer class is the defect this file was opened
    to fix.

    **The earliest match, not the first in the tuple.** Scanning ``stop`` in order and
    cutting at the first one found would make the result depend on the order the caller
    happened to list them in, and ``GenerationParams`` is frozen but its ``stop`` is a plain
    tuple with no ordering contract.
    """
    cuts = [text.index(sequence) for sequence in stop if sequence and sequence in text]
    return text[: min(cuts)] if cuts else text
