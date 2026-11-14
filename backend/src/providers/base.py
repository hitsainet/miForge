"""The inference-provider interface  (foundation task 3.8; R-02.49, ADR-009).

**This is a gate contract with four consumers - F2, F3, F4 and F7 - and it is the one
that nearly shipped without an owner.** It appeared in the whole foundation backlog once,
in a Relevant Files note saying implementations *"land with F4"*, with no task behind it,
while F2 and F4 are scheduled in the same wave. The ratified split, which this module
implements rather than redesigns:

* **the interface** - here, the gate;
* **the OpenAI-compatible HTTP implementation** - F3's, because R-02.11 makes the
  in-process provider unusable as a judge, so F3 cannot wait on F4 for it;
* **the registry, the provider identity and the in-process path** - F4's.

## Why this abstraction exists at all

R-02.49 was written because *"a test path wired to one sibling by name is what made the
whole document depend on it."* Before it, miForge's test path called miLLM, and that one
call site is what turned an optional integration into a hard runtime dependency on an
application that did not exist. So: **miLLM is one row beside vLLM, Ollama and the OpenAI
API - never a named dependency.** The default is in-process, which needs nothing.

## Every score records its provider

ADR-009, and R-02.16 requires it on the artifact: *a score is only comparable against
another taken the same way.* ``ScoreRecord.provider`` exists in both vendored manifests as
of 2026-10-13 - but it is **deliberately not ``required``**, so a payload omitting it
still validates and **a test is the only thing that catches the omission.**

Three further fields change what a score means and the manifests do **not** carry them, so
:class:`GenerationParams` and :class:`ProviderIdentity` carry them to the run log's
**evaluation block** instead (foundation task 3.3a): the **aggregation method**, the
**decode parameters** and the **subsample seed**. Recording "0.82" without them describes
nothing comparable.
"""

from __future__ import annotations

from typing import Any, Protocol, runtime_checkable

from pydantic import BaseModel, ConfigDict, Field


class GenerationParams(BaseModel):
    """The decode parameters, which are **part of what a score means**.

    A score at ``temperature=0`` is not comparable with one at ``0.8``, so these are
    recorded in the evaluation block beside every score rather than being an incidental
    call argument. ``seed`` is here for the same reason: it is the subsample seed that
    determines *which* ``n`` prompts were used, and a score over an unnamed subsample is
    not reproducible.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    max_tokens: int = Field(default=256, gt=0)
    temperature: float = Field(default=0.0, ge=0.0, le=2.0)
    top_p: float = Field(default=1.0, gt=0.0, le=1.0)
    stop: tuple[str, ...] = ()
    #: Determines the subsample and any sampling. **Null is an assertion that no seed was
    #: set**, which makes the result irreproducible - and that is worth recording as a
    #: fact rather than hiding. A `create` run has the same problem at a larger scale:
    #: R-02.47 requires the content hash of the weights it started from, and random init
    #: has no reproducible hash without a seed.
    seed: int | None = None


class ProviderIdentity(BaseModel):
    """Who produced a completion, in enough detail to compare two scores honestly.

    ``kind`` is the provider family and ``model`` what it served. Both are needed: "the
    in-process provider" does not identify a model, and a model name does not say whether
    it was reached in-process or over HTTP - and the two can disagree, which is the
    cross-plane equivalence problem a sibling already documented as tech debt after a
    measure-only sweep misled it.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    #: ``in_process`` | ``openai_compatible``. Never a sibling's name: the *endpoint* may
    #: happen to be miLLM, which belongs in ``base_url``, not in the type system.
    kind: str
    model: str
    #: Present only for an HTTP provider. ``None`` for in-process - and that is the
    #: distinction, not a missing value.
    base_url: str | None = None
    #: What the provider reports about itself, when it does. Free-form because each engine
    #: reports something different and inventing a common shape would discard detail.
    engine_version: str | None = None

    def record(self) -> str:
        """The single string written to ``ScoreRecord.provider``.

        One string rather than a nested object because ``ScoreRecord.provider`` is typed
        ``anyOf [string, null]`` in both vendored manifests - the schema decides the shape
        here, not this module. The full identity goes to the evaluation block.
        """
        if self.base_url:
            return f"{self.kind}:{self.model}@{self.base_url}"
        return f"{self.kind}:{self.model}"


class Completion(BaseModel):
    """One generated continuation, with the provenance a score needs."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    text: str
    provider: ProviderIdentity
    params: GenerationParams
    #: Token counts when the provider reports them; ``None`` rather than ``0``, because
    #: zero is a measurement and absence is not.
    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    #: Log-probabilities over an explicit token set, when requested. R-02.2 compares
    #: these - **not logits**: the `/v1` surface returns log-probabilities and never
    #: logits, which is why four of BRD-01's acceptance criteria had to be corrected.
    logprobs: dict[str, float] | None = None


@runtime_checkable
class InferenceProvider(Protocol):
    """Generate completions. The one way miForge reaches a model it is not training.

    **Implementations:** in-process (F4, the default - needs no endpoint and no sibling)
    and OpenAI-compatible HTTP (F3). An OpenAI-compatible endpoint covers miLLM, vLLM,
    Ollama and the OpenAI API with one implementation and one ``base_url``, which is the
    shape miStudio already proved.
    """

    @property
    def identity(self) -> ProviderIdentity:
        """Who this is, recorded on every score it produces."""
        ...

    def available(self) -> bool:
        """Whether this provider can be used **right now**.

        Separate from construction so that an unconfigured provider can be *listed* as
        unavailable by name rather than being absent from the list. An option missing from
        a response is indistinguishable from one that is broken (ADR-013), and R-02.9's
        selector types must each report whether their requirement is currently satisfied.
        """
        ...

    def generate(
        self, prompts: list[str], params: GenerationParams, **options: Any
    ) -> list[Completion]:
        """Generate one completion per prompt, in order.

        **Order is part of the contract**: callers join results to prompts positionally,
        and a provider that reordered or dropped one would silently misattribute every
        score after the gap. A provider that cannot answer raises a named ``Refusal``
        rather than returning a short list.
        """
        ...
