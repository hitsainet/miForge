"""The evaluation service  (FR-003.11; R-02.9).

**One entry point.** :func:`evaluate` takes a sequence of completions and a resolved
configuration and returns verdicts **in input order** plus the health record. It is the call
F2's signal sources make, the call F4's scoring makes, the call F7's dry-run makes, and the
call behind ``POST /api/v1/selectors/evaluate``. One implementation, so a verdict does not
depend on which caller asked.

## F3 declares no Celery task and no queue, and must stay that way

``evaluate`` is a **synchronous callable whose placement is its caller's.** F6's AMB-1
settled a three-document queue conflict as *"the loop calls both in-process and
synchronously, with ``generate`` carrying only work the loop does not await"*, and F2 and F4
accepted it. So this runs in the ``train`` worker when F2's in-process assembly calls it, in
a ``generate`` task when an HTTP-provider pass calls it, and in a request handler for F7's
dry-run.

A Celery task added **inside** this service would make the placement F3's decision rather
than its caller's, and would hand a ``--pool=solo -c 1`` worker a cross-queue ``.get()`` -
a worker that is busy never reads its control queue, so that deadlocks rather than queues.
A static test asserts no ``workers/`` module imports this package in order to wrap it.

## Input order, and why it is a list

Verdicts come back as a **list, positionally joined to the completions** - never a mapping
keyed on completion text. Two identical completions are two candidates, and a dict would
silently merge them, which would misattribute every verdict after the collision.

## Cancellation is cooperative, at a completion boundary

A completion is **evaluated and recorded, or not started** - never half (ADR-006, ADR-007).
``revoke(terminate=True)`` is never used: a solo-pool worker has no pool child to signal, so
a running job is reached through the database instead.
"""

from __future__ import annotations

import logging
from collections.abc import Sequence
from typing import Any

from core.errors import Refusal, refuse
from providers.base import ProviderIdentity
from schemas.selector import Verdict
from services.selectors import hashing
from services.selectors.health import compute_health, publish_signals
from services.selectors.resolve import ResolvedSelector
from services.selectors.settings import get_selector_settings
from services.selectors.types import (
    EvaluationContext,
    EvaluationResult,
    SelectorForm,
    VerdictRecord,
)

log = logging.getLogger(__name__)


class EvaluationCancelled(Refusal):
    """A cooperative cancellation, as a named refusal rather than a bare exception.

    A refusal because the caller needs to tell "the run was cancelled" from "the judge is
    down": both stop the evaluation, and only one of them is a problem.
    """

    def __init__(self, completed: int, total: int) -> None:
        super().__init__(
            "evaluation_cancelled",
            f"The evaluation was cancelled after {completed} of {total} completions. Each "
            f"completion was either evaluated and recorded or not started; none is half "
            f"evaluated.",
            {"completed": completed, "total": total},
        )


def evaluate(
    completions: Sequence[str],
    resolved: ResolvedSelector,
    ctx: EvaluationContext | None = None,
    *,
    prompts: Sequence[str] | None = None,
) -> EvaluationResult:
    """Evaluate a batch. **The public entry point, and the one F7's dry-run calls.**

    Args:
        completions: The completions to judge, in the order the caller wants them back.
        resolved: A configuration from :func:`services.selectors.resolve.resolve`. Resolved,
            not raw: resolution is where a configuration error is refused **once** rather
            than once per completion.
        ctx: Run id, batch id, per-item extras (``reference`` for reference-match), the
            cancellation check and the signal sink. Defaults to a context with none of
            them, which is the dry-run case.
        prompts: The prompts the completions answer, positionally. Defaults to empty
            strings - a rule selector does not read the prompt, but a judge does, so a
            judge configuration with no prompts is judging an answer to nothing.

    Returns:
        :class:`EvaluationResult` - the per-completion records in input order, the health
        record with its named signals, and the published hashes for F2 and F5.

    Raises:
        Refusal: ``evaluation_cancelled``; or any configuration or transport refusal, which
            fails the **whole** evaluation. A per-completion failure is never converted into
            a verdict: a transport failure is not a verdict, and defaulting it to accept
            would turn this into the accept-everything selector the feature exists to
            prevent.
    """
    ctx = ctx or EvaluationContext()
    settings = get_selector_settings()
    total = len(completions)
    if total > settings.evaluate_max_completions:
        raise refuse(
            "evaluation_too_large",
            f"{total} completions exceeds the {settings.evaluate_max_completions} this "
            f"endpoint evaluates synchronously. Split the batch, or raise "
            f"EVALUATE_MAX_COMPLETIONS deliberately: an unbounded synchronous evaluation "
            f"over a judge is a request that outlives its client.",
            count=total,
            limit=settings.evaluate_max_completions,
        )

    prompt_list = list(prompts) if prompts is not None else [""] * total
    if len(prompt_list) != total:
        raise refuse(
            "evaluation_prompt_count_mismatch",
            f"{len(prompt_list)} prompts were supplied for {total} completions. Refusing "
            f"rather than padding: a positional join over mismatched lists misattributes "
            f"every verdict after the gap.",
            prompts=len(prompt_list),
            completions=total,
        )

    records: list[VerdictRecord] = []
    scores: list[float] = []
    has_grader = SelectorForm.GRADER in resolved.capability.forms

    try:
        _prefetch_batch(resolved, prompt_list, list(completions), ctx)
        for index, completion in enumerate(completions):
            # The cancellation boundary. Checked BEFORE starting a completion, so the
            # invariant is "evaluated and recorded, or not started" rather than "stopped
            # somewhere". A prefetch that stopped on cancellation lands here next.
            if ctx.is_cancelled is not None and ctx.is_cancelled():
                raise EvaluationCancelled(completed=len(records), total=total)

            item_context = _context_for(index, ctx)
            verdict = resolved.plugin.evaluate(
                prompt=prompt_list[index], completion=completion, context=item_context
            )
            records.append(_record(verdict, resolved, item_context))
            if has_grader and verdict.score is not None:
                scores.append(verdict.score)
    finally:
        # Prefetched-but-unconsumed work never outlives its batch: a resolved selector is
        # reused across batches, and a reply cached for one must not answer another.
        release = getattr(resolved.plugin, "release_batch", None)
        if release is not None:
            release()

    verdicts: list[Verdict] = [record.verdict for record in records]
    health = compute_health(verdicts, settings=settings, scores=scores if has_grader else None)
    publish_signals(health, run_id=ctx.run_id, emit=ctx.emit_signal)

    return EvaluationResult(
        records=tuple(records),
        health=health,
        hashes=hashing.publish(
            selector=resolved.hash if SelectorForm.SELECTOR in resolved.capability.forms else None,
            grader=resolved.hash if has_grader else None,
        ),
    )


def _context_for(index: int, ctx: EvaluationContext) -> dict[str, Any]:
    """The per-completion context a plugin sees.

    ``index`` and ``batch_id`` are always present because ``human_review`` derives its
    deterministic ``external_id`` from them, and a missing one would make a retried post
    create a duplicate for a person to read twice.
    """
    item: dict[str, Any] = {"index": index, "batch_id": ctx.batch_id, "run_id": ctx.run_id}
    item.update(ctx.extra)
    if index < len(ctx.items):
        item.update(ctx.items[index])
    return item


def _record(
    verdict: Verdict, resolved: ResolvedSelector, item_context: dict[str, Any]
) -> VerdictRecord:
    """Pair a verdict with the provenance an audit needs (FR-003.18)."""
    del item_context
    judge: ProviderIdentity | None = None
    provider = resolved.bindings.get("provider")
    if provider is not None:
        judge = provider.identity

    match_mode: str | None = None
    match_metric: str | None = None
    if resolved.kind == "reference_match":
        prepared = resolved.prepared or {}
        if isinstance(prepared, dict):
            match_mode = prepared.get("mode")
            match_metric = prepared.get("metric") if match_mode == "fuzzy" else None

    review_item_id: str | None = None
    if resolved.kind == "human_review" and verdict.value.value == "undecided":
        review_item_id = verdict.label

    return VerdictRecord(
        verdict=verdict,
        # `label` doubles as the machine reason code - the contract's own comment says so.
        # Mirrored here so a reader of this record does not have to know that.
        reason_code=verdict.label or "unlabelled",
        config_sha256=resolved.config_sha256,
        config_kind=resolved.kind,
        judge=judge,
        caveats=resolved.caveats,
        match_mode=match_mode,
        match_metric=match_metric,
        review_item_id=review_item_id,
    )


def _prefetch_batch(
    resolved: ResolvedSelector,
    prompts: list[str],
    completions: list[str],
    ctx: EvaluationContext,
) -> None:
    """Let an asynchronous plugin post its whole batch before the per-completion loop.

    Two plugins offer it: ``human_review`` (posting 64 candidates one at a time would be 64
    round trips where miDataworks accepts 500 in one) and, since 2026-10-16, ``judge_model``
    (judging ``judge_concurrency`` completions per provider call - the setting was read by
    nothing until then). Both receive ``is_cancelled`` and stop at their own boundary.

    **It is an optimisation, not the enqueue path.** ``human_review.evaluate`` posts a
    candidate nothing has posted yet, so correctness does not depend on this call - which
    is why it is deliberately *not* walked into a composite's children. A composite must be
    able to reject on a rule **without posting anybody's candidate**, and pre-posting the
    whole batch for a human child would put candidates in front of a person that the
    cheaper sibling was about to settle. That is asserted by a queue call count of zero.
    """
    prefetch = getattr(resolved.plugin, "prefetch", None)
    if prefetch is not None:
        # `is_cancelled` travels in, because a prefetch is work done AHEAD of the per-item
        # cancellation boundary and must stop at its own (suite tracker 4.28).
        prefetch(prompts, completions, batch_id=ctx.batch_id, is_cancelled=ctx.is_cancelled)
