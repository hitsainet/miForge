"""The authoring dry-run  (task 10.x; FR-007.20, FR-007.21, R-02.39, AC11).

Take **a sample of the pack's own prompts**, answer them through the configured inference
provider, apply the chosen selector or grader configuration, and show **a verdict with a
reason for each, before the step is saved**. R-02.39's own argument for why this is a
requirement and not a nicety:

> *"A selector that accepts everything is not selecting; one that rejects everything will
> stall the run."*

The only other signals are R-02.25's loss plateau and the selector's own accept rate, **and
both cost a run to discover.** The cheapest place to find out is before the run starts.

## Synchronous, in-request, and not a queue

**This is not the in-run human-review queue, and conflating the two is a mistake this
project already made once** (R-02.36's 2026-10-12 clarification). The dry-run happens before
any run exists, judges the *selector's verdicts* on a sample, and **returns within the
request**. miForge keeps no review queue or review screen of its own: no queue, no inbox, no
pending-decision store, no reviewer identity, no notion of a decision coming back later. If
this ever needs a worker, **shrink the sample; do not add a queue.**

A **human-review configuration therefore dry-runs to ``undecided`` for every candidate, and
that is the correct result, not a failure.** It answers asynchronously from a queue this
feature is forbidden to have, so there is nothing to wait on. It must not fall back to
``accept``, must not block, and must not reach F10's client to get an answer - which F10
asserts from its own side on the call count being zero.

## Three counts, because the verdict is three-valued

F3's :class:`~services.selectors.types.SelectionHealth` is returned **whole** and nothing
here recomputes a rate from the verdict list: *two places computing an acceptance rate is two
answers*, and F3 built the record so there would be one. ``accept`` / ``reject`` /
``undecided``, with the denominator arriving beside the numerators, which is ADR-012's "every
number names its scale and sample" expressed in the payload.

**Never ``verdict != "reject"`` as a stand-in for accepted, and never a truthiness test on a
verdict.** :func:`schemas.selector.is_admitted` is the only route to "admitted", and one
place treating ``undecided`` as either other value makes the health record lie.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from core.errors import Refusal
from packs.errors import PROVIDER_NOT_AVAILABLE, PackRefusal
from providers.base import GenerationParams, ProviderIdentity
from providers.registry import IN_PROCESS_NAME, list_rows
from providers.registry import get as get_provider
from schemas.selector import Verdict, is_admitted
from services.selectors.resolve import resolve as resolve_selector
from services.selectors.service import evaluate
from services.selectors.types import EvaluationContext, SelectionHealth

#: Decode parameters for a dry-run. ``temperature=0`` so two dry-runs of the same
#: configuration over the same sample are comparable - a verdict that moves because the
#: sampler moved is advice about nothing. The seed is stated for the same reason: **null
#: would be an assertion that the result is irreproducible** (``GenerationParams.seed``).
DRY_RUN_PARAMS = GenerationParams(max_tokens=256, temperature=0.0, seed=0)


@dataclass(frozen=True, slots=True)
class DryRunVerdict:
    """One prompt, its answer, and what the configuration said about it, with the reason."""

    prompt: str
    completion: str
    #: ``accept`` | ``reject`` | ``undecided`` - F3's three-valued verdict, carried as its
    #: own value and never folded into two.
    value: str
    #: The stable machine code a UI maps on; a mapping keyed on prose breaks on a wording
    #: change.
    reason_code: str
    #: Operator-facing, naming the thing that fired.
    reason: str | None
    score: float | None
    #: Whether this candidate **would be admitted**, from
    #: :func:`schemas.selector.is_admitted` and from nowhere else.
    admitted: bool


@dataclass(frozen=True, slots=True)
class DryRunResult:
    """What the panel renders. F3's health record travels **whole**."""

    #: Who answered the prompts (ADR-009: every score records its provider).
    provider: ProviderIdentity
    #: Who judged, for a judge-model configuration. ``None`` for a type with no judge -
    #: and the absence is the distinction, not a missing value. Surfaced separately from
    #: ``provider`` so a reviewer can see that the judge is **not** the model being sampled
    #: (R-02.11 as a constraint on this feature).
    judge: ProviderIdentity | None
    config_sha256: str
    config_kind: str
    hash_provenance: str
    health: SelectionHealth
    verdicts: tuple[DryRunVerdict, ...]
    #: Set when every verdict is ``undecided`` because the configuration answers
    #: asynchronously. **Named rather than inferred**: a panel of all-``undecided`` with no
    #: explanation reads as a broken dry-run, and this says which configuration kind did it
    #: and why the step still saves.
    asynchronous_note: str | None = None

    def payload(self) -> dict[str, Any]:
        """The response body (FTDD §5).

        **No separate ``accepted``/``sample_n`` pair**: the counts live in ``health`` and a
        second place holding them is a second answer. The UI renders the record.
        """
        return {
            "provider": self.provider.model_dump(mode="json"),
            "judge": None if self.judge is None else self.judge.model_dump(mode="json"),
            "config_sha256": self.config_sha256,
            "config_kind": self.config_kind,
            "hash_provenance": self.hash_provenance,
            "health": self.health.model_dump(mode="json"),
            "verdicts": [
                {
                    "prompt": verdict.prompt,
                    "completion": verdict.completion,
                    "verdict": verdict.value,
                    "reason_code": verdict.reason_code,
                    "reason": verdict.reason,
                    "score": verdict.score,
                    "admitted": verdict.admitted,
                }
                for verdict in self.verdicts
            ],
            "asynchronous_note": self.asynchronous_note,
            "advisory": True,
        }


def dry_run(
    prompts: Sequence[str],
    config: Mapping[str, Any],
    *,
    catalog: Sequence[Mapping[str, Any]] = (),
    provider_name: str | None = None,
    source: str = "inline",
) -> DryRunResult:
    """Answer ``prompts`` and judge them with ``config``. **Advisory; never a wall.**

    Args:
        prompts: A sample of the **pack's own** prompts (AC11 says so explicitly: a dry-run
            on arbitrary prompts measures nothing about this domain).
        config: A ``selector-config-v1.json`` body. **Selected, not authored** here
            (FR-007.22): rule terms and judge templates are miDataworks' and are cited by
            hash; this feature runs the plugin.
        catalog: The other configured selectors by name. **Required for a composite**,
            whose ``children`` are *names of other configured selectors* and cannot be
            resolved from its own body.
        provider_name: Which provider answers. Defaults to the first available row, so a
            laptop with nothing configured gets a named refusal rather than a silent choice.
        source: Where a locally supplied configuration came from (R-02.40's honesty rule).
            ``"inline"`` for a dry-run, which is what F3's ``resolve`` documents.

    Returns:
        The result, with F3's health record whole.

    Raises:
        PackRefusal: ``provider_not_available`` naming the provider and what it costs. **The
            step still saves** (E7) - this refusal is raised by the dry-run call alone, and
            the draft's step commit is a different call that never invokes it. A dry-run is
            advice, and advice that becomes a wall in front of authoring is a worse failure
            than no advice.
        Refusal: F3's configuration refusals, unchanged - an invalid configuration, an
            unknown type, a composite cycle, a judge that is the model being sampled.
    """
    provider = resolve_provider(provider_name)
    completions = provider.generate(list(prompts), DRY_RUN_PARAMS)
    texts = [completion.text for completion in completions]

    resolved = resolve_selector(
        # A copy, because F3's `resolve` takes a `dict` it is free to read as it likes and
        # this one arrives as a request body's mapping.
        dict(config),
        catalog={str(body["name"]): dict(body) for body in catalog if "name" in body},
        source=source,
    )

    result = evaluate(
        texts,
        resolved,
        EvaluationContext(batch_id="dry-run"),
        prompts=list(prompts),
    )

    verdicts = tuple(
        DryRunVerdict(
            prompt=prompt,
            completion=text,
            value=record.verdict.value.value,
            reason_code=record.reason_code,
            reason=record.verdict.reason,
            score=record.verdict.score,
            # `is_admitted` and nothing else. Never `value != "reject"`: that is the
            # mutation that silently admits an undecided candidate.
            admitted=is_admitted(record.verdict),
        )
        for prompt, text, record in zip(prompts, texts, result.records, strict=True)
    )

    return DryRunResult(
        provider=provider.identity,
        judge=_judge_identity(resolved),
        config_sha256=resolved.hash.sha256,
        config_kind=resolved.kind,
        hash_provenance=resolved.hash.provenance,
        health=result.health,
        verdicts=verdicts,
        asynchronous_note=_asynchronous_note(resolved.kind, result.verdicts),
    )


def resolve_provider(name: str | None) -> Any:
    """The provider that answers, or a refusal **naming it** (E7).

    With no name given, the first *available* row is used - and if none is available the
    refusal lists every row with the reason it cannot be used, because an option missing
    from a response is indistinguishable from one that is broken (ADR-013).
    """
    rows = list_rows()
    if name is None:
        available = [row for row in rows if row.available]
        if not available:
            raise PackRefusal(
                PROVIDER_NOT_AVAILABLE,
                "No inference provider is available, so a dry-run cannot answer the "
                "sample. The dry-run is advice: the authoring step still saves without it. "
                "Configure an OpenAI-compatible endpoint, or run the dry-run from the "
                "training worker, which holds the in-process model.",
                capability="dry_run",
                providers=[
                    {"name": row.name, "unavailable_reason": row.unavailable_reason} for row in rows
                ],
            )
        name = available[0].name
    try:
        return get_provider(name)
    except Refusal as refusal:
        raise PackRefusal(
            PROVIDER_NOT_AVAILABLE,
            f"Provider {name!r} cannot answer this dry-run: {refusal.message} The "
            f"authoring step still saves without the dry-run.",
            capability="dry_run",
            provider=name,
            provider_refusal=refusal.code,
        ) from refusal


def _judge_identity(resolved: Any) -> ProviderIdentity | None:
    """The judge's identity, when the configuration has one.

    Read off F3's resolved bindings rather than re-resolved here: F3 binds the judge, applies
    R-02.11's guard that the judge is not the resident model, and records the identity on
    every verdict. A second resolution here could pick a different provider from the one that
    actually judged, and the panel would then name the wrong instrument.
    """
    binding = resolved.bindings.get("provider")
    if binding is None:
        return None
    identity = getattr(binding, "identity", None)
    return identity if isinstance(identity, ProviderIdentity) else None


def _asynchronous_note(kind: str, verdicts: Sequence[Verdict]) -> str | None:
    """Name the all-``undecided`` case by its cause, or return ``None``.

    Only when **every** verdict is undecided: a mixture is a configuration partly answering,
    which is a different fact and belongs to F3's own signals rather than to a note here.
    """
    if not verdicts or any(is_admitted(verdict) for verdict in verdicts):
        return None
    if not all(verdict.value.value == "undecided" for verdict in verdicts):
        return None
    return (
        f"The {kind!r} configuration ran over every prompt in the sample and reached no "
        f"synchronous verdict: it answers from a review queue, which is miDataworks', "
        f"and which pack authoring deliberately does not have. Nothing is wrong "
        f"and this step saves - the verdicts arrive during a run, not here."
    )


def in_process_is_the_only_option() -> bool:
    """Whether the only configured provider is the in-process one.

    Used by the authoring panels to say *why* a dry-run is unavailable in the API process:
    the in-process row holds no parameters outside the training worker, which is a fact worth
    naming rather than a provider that looks broken.
    """
    rows = list_rows()
    return bool(rows) and all(row.name == IN_PROCESS_NAME for row in rows)
