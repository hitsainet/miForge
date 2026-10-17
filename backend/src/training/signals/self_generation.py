"""Self-generation: the model's own completions, filtered by a selector  (R-02.5).

FR-002.10 - FR-002.17. Prompts from a **prompt set**, generation through the configured
provider, verdicts from F3, and the **accepted** completions become supervised examples.

## The loop that feeds the model its own output

This is the source whose dangerous failure is quiet. A selector that is not selecting trains
on raw output: the loss falls, the test score does not move, nothing errors. So three things
are on every record rather than in a log line - the resolved generating model, the three
verdict counts, and the selection health with its acceptance rate and its ``n``.

## ``undecided`` is held, never admitted, never counted as a rejection

The partition is :mod:`training.signals.selection`'s, in one place, and the deadline bounds
the **wait** and produces only a **count**. There is no branch, flag, timeout or shortfall
path from pending to admitted (FR-002.13, AC-002.4).

## What is deliberately NOT here

**R-02.26's prompt refresh is F6's**, not this module's. When the prompt set is exhausted
this source says so on the record and refuses to invent more; whether to refresh, rotate or
stop is the loop's decision (R-02.24 - R-02.27). Adding it here would put a stop condition
in two places.
"""

from __future__ import annotations

from collections.abc import Iterator, Mapping
from typing import Any

from core.errors import refuse
from training.signals import generation
from training.signals.base import (
    BatchCounts,
    RunContext,
    SourceCapability,
    SupervisedExample,
    TrainingBatch,
)
from training.signals.common import BaseSource
from training.signals.parts_reader import read_rows
from training.signals.selection import Candidate, apply
from training.signals.settings import require_masking_policy

NAME = "self-generation"

CAPABILITY = SourceCapability(
    name=NAME,
    part_kind="prompt_set",
    needs_selector=True,
    needs_grader=False,
    needs_generation=True,
    objective="sft",
    summary=(
        "the model answers its own domain prompts and a selector decides which answers "
        "become training examples - the accepted ones are supervised examples"
    ),
)


class SelfGenerationSource(BaseSource):
    """R-02.5. Generate, select, admit, record."""

    NAME = NAME
    CAPABILITY = CAPABILITY

    def __init__(self, context: RunContext | None = None) -> None:
        super().__init__(context)
        self._prompts: Iterator[tuple[str, Mapping[str, Any]]] | None = None
        self._batch_number = 0

    def resolve_extra(self, resolved: Any) -> None:
        """Pre-flight refuses a missing masking policy and an absent provider (FR-002.4)."""
        require_masking_policy(self.settings)
        if self.context.provider is None:
            raise refuse(
                "generation_provider_not_configured",
                "Self-generation needs an inference provider to answer its prompts. The "
                "default is in-process and needs no sibling application; configure one, or "
                "use the corpus source, which reads material rather than generating it.",
                source=self.NAME,
                capabilities_unavailable=["self-generation"],
            )

    def next(self) -> TrainingBatch:
        """Generate a round of candidates, select, and record what was admitted."""
        resolved = self.require_resolved()
        self.check_cancelled()
        policy = require_masking_policy(self.settings)
        wanted = resolved.spec.batch_size
        self._batch_number += 1

        prompts, exhausted = self._take(wanted)
        if not prompts:
            raise refuse(
                "prompt_set_exhausted",
                "This prompt set has no prompts left. Refusing rather than reusing them "
                "silently: R-02.26's prompt refresh is the loop's decision (F6), and a "
                "source that quietly cycled would make every later round train on material "
                "the model has already been fitted to.",
                run_id=self.context.run_id,
            )

        completions, identity, resolved_model = generation.generate(
            [text for _id, text in prompts],
            self.context.provider,
            resolved.spec.params,
            settings=self.settings,
            is_cancelled=self.context.is_cancelled,
            current_step=self.context.current_step,
        )
        self.check_cancelled()

        candidates = [
            Candidate(prompt_id=prompt_id, prompt=text, completion=completion.text, index=index)
            for index, ((prompt_id, text), completion) in enumerate(
                zip(prompts, completions, strict=True)
            )
        ]
        empty = [candidate for candidate in candidates if not candidate.completion.strip()]
        # E-9: an empty completion is recorded and rejected with a named reason, never
        # admitted as a zero-length supervised example. It is dropped BEFORE selection
        # rather than being judged, because asking a judge to score nothing produces a
        # verdict about nothing.
        candidates = [candidate for candidate in candidates if candidate.completion.strip()]

        outcome = apply(
            candidates,
            resolved.selector,
            settings=self.settings,
            run_id=self.context.run_id,
            batch_id=f"{self.context.run_id}:{self._batch_number}",
            is_cancelled=self.context.is_cancelled,
            emit_signal=None,
        )

        examples = [
            SupervisedExample(
                prompt_id=candidate.prompt_id,
                prompt=candidate.prompt,
                completion=candidate.completion,
            )
            for candidate in outcome.admitted
        ]
        counts = BatchCounts(
            generated=len(completions),
            admitted=len(outcome.admitted),
            # The empty completions are rejections, counted as such: they were generated
            # and they are not training material. They are NOT pending, and nothing in this
            # path can make them so.
            rejected=len(outcome.rejected) + len(empty),
            pending=len(outcome.pending),
            exhausted=exhausted,
            requested=wanted,
        )
        if not examples:
            # E-7. A round in which nothing was admitted cannot fill the batch, and
            # **never** fills it with the rejected completions. Refused even under
            # `short_batch`, because a zero-example batch is not a short batch: it is a step
            # that trains on nothing while every log line looks healthy. F3's own named
            # signal is carried in the refusal so an operator reads the cause rather than
            # the symptom - and stopping the run is still F6's (R-02.25).
            signals = list(outcome.health.signal_names)
            raise refuse(
                "selection_admitted_nothing",
                f"The selector admitted none of {len(completions)} candidates "
                f"({counts.rejected} rejected, {counts.pending} awaiting review). The "
                f"batch is refused rather than padded with rejected completions - a "
                f"zero-example step trains on nothing and reports a healthy loss. "
                + (f"Signals: {', '.join(signals)}. " if signals else "")
                + "Loosen the selector, or check that its configuration is the one you "
                "meant.",
                generated=len(completions),
                rejected=counts.rejected,
                pending=counts.pending,
                signals=signals,
            )
        self.handle_shortfall(counts, requested=wanted)
        return self.record(
            examples=examples,
            counts=counts,
            resolved_model=resolved_model,
            health=outcome.health,
            hashes=outcome.hashes,
            signals=outcome.health.signal_names,
            provider=identity,
            review_ids=outcome.review_ids,
            masking_policy=policy,
        )

    def _take(self, count: int) -> tuple[list[tuple[str, str]], bool]:
        """``count`` prompts, and whether the set ran out. **Identifiers travel with them.**"""
        resolved = self.require_resolved()
        if self._prompts is None:
            self._prompts = self._prompt_stream()
        binding = resolved.parts[0].binding
        column = binding.column("prompt")
        taken: list[tuple[str, str]] = []
        for prompt_id, row in self._prompts:
            value = row[column]
            if not isinstance(value, str):
                raise refuse(
                    "prompt_column_not_text",
                    f"The column bound to 'prompt' holds {type(value).__name__}, not text. "
                    f"Refusing rather than coercing it: a prompt nobody wrote produces "
                    f"candidates about nothing.",
                    value_type=type(value).__name__,
                )
            taken.append((prompt_id, value))
            if len(taken) >= count:
                return taken, False
        return taken, True

    def _prompt_stream(self) -> Iterator[tuple[str, Mapping[str, Any]]]:
        resolved = self.require_resolved()
        for part in resolved.parts:
            columns = [name for name, _semantic in part.binding.roles.values()]
            yield from read_rows(part, columns=columns)


def factory(context: RunContext) -> SelfGenerationSource:
    return SelfGenerationSource(context)
