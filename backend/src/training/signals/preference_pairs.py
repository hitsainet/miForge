"""Preference pairs, both halves  (R-02.6; FR-002.18 - FR-002.22).

R-02.6 admits two materials and this one source reads both:

* **from a pairs part** - chosen and rejected from the columns whose **declared** semantics
  are ``chosen`` and ``rejected``, never from their names;
* **from two generations** - two completions per prompt, ordered by a **scalar**.

## The second half names an operation the selector interface does not have

*"the selector ranking two generations"* - and a selector as R-02.9 defines it returns a
verdict, which does not order two completions (FPRD §15 finding 2, reported upstream). So
the ordering comes from a scalar: a grader's score, or the optional ``score`` a verdict
carries where its configuration produces one. **Equal or absent scalars produce no pair**
and the tie is counted. A tie is never broken: *a pair ordered by a coin flip trains the
model toward noise, and its loss curve looks exactly like one that is working.*

## Which half runs is decided by the part's kind, not by a flag

A ``preference_pairs`` part is the file half; a ``prompt_set`` part is the from-generations
half. Deciding it from the material means a run cannot be configured into the half its part
cannot serve.
"""

from __future__ import annotations

from collections.abc import Iterator, Mapping
from typing import Any

from core.errors import refuse
from training.signals import generation, pairs
from training.signals.base import (
    BatchCounts,
    PartSpec,
    PreferenceExample,
    RunContext,
    SourceCapability,
    TrainingBatch,
)
from training.signals.common import BaseSource
from training.signals.parts_reader import read_rows
from training.signals.selection import Candidate, apply
from training.signals.settings import require_reference_source

NAME = "preference-pairs"

CAPABILITY = SourceCapability(
    name=NAME,
    part_kind="preference_pairs",
    needs_selector=False,
    needs_grader=False,
    needs_generation=False,
    objective="dpo",
    summary=(
        "chosen/rejected pairs from a dataset version, or two generations ordered by a "
        "scalar - trained with a direct preference objective"
    ),
)


class PreferencePairsSource(BaseSource):
    """R-02.6. Two materials, one objective, and a tie that is never broken."""

    NAME = NAME
    CAPABILITY = CAPABILITY

    def __init__(self, context: RunContext | None = None) -> None:
        super().__init__(context)
        self._rows: Iterator[tuple[str, Mapping[str, Any]]] | None = None
        self._batch_number = 0

    def required_kind(self, part_spec: PartSpec) -> str | None:
        """``prompt_set`` for the from-generations half; the source's default otherwise."""
        return "prompt_set" if part_spec.kind == "prompt_set" else None

    @property
    def from_generations(self) -> bool:
        resolved = self.require_resolved()
        return any(part.kind == "prompt_set" for part in resolved.parts)

    def resolve_extra(self, resolved: Any) -> None:
        """Pre-flight refuses an absent reference source, and an unorderable configuration.

        Both are configuration refusals and both belong **before the run starts**: a DPO run
        with no reference is not a preference objective, and a from-generations run with
        nothing to order by produces a batch of no pairs at step 1.
        """
        require_reference_source(self.settings)
        if any(part.kind == "prompt_set" for part in resolved.parts):
            if self.context.provider is None:
                raise refuse(
                    "generation_provider_not_configured",
                    "Building pairs from two generations needs an inference provider. The "
                    "default is in-process and needs no other miSuite app.",
                    source=self.NAME,
                    capabilities_unavailable=["preference-pairs from generations"],
                )
            if resolved.grader is None and resolved.selector is None:
                raise refuse(
                    "pair_ordering_not_configured",
                    "Pairs built from two generations need something that produces a "
                    "SCALAR to order them: a grader's score, or a selector whose "
                    "configuration produces a score. A verdict - accept, reject or undecided "
                    "- does not order two completions, so a ranking cannot be derived from "
                    "one. Configure a grader, or use a pairs part.",
                    source=self.NAME,
                )

    def next(self) -> TrainingBatch:
        """One batch of pairs, from whichever material this run resolved."""
        resolved = self.require_resolved()
        self.check_cancelled()
        reference = require_reference_source(self.settings)
        wanted = resolved.spec.batch_size
        self._batch_number += 1

        if self.from_generations:
            examples, counts, extra = self._from_generations(wanted)
        else:
            examples, counts, extra = self._from_file(wanted)

        if not examples:
            raise refuse(
                "no_pairs_produced",
                f"This batch produced no usable pair out of {counts.generated or wanted} "
                f"candidates ({counts.ties} unorderable, {counts.dropped} malformed). A "
                f"zero-pair batch is a step that trains on nothing while the log looks "
                f"healthy. A tie is never broken to make up the number.",
                ties=counts.ties,
                dropped=counts.dropped,
                run_id=self.context.run_id,
            )
        self.handle_shortfall(counts, requested=wanted)
        return self.record(examples=examples, counts=counts, reference_source=reference, **extra)

    # ------------------------------------------------------------------ the file half
    def _from_file(
        self, wanted: int
    ) -> tuple[list[PreferenceExample], BatchCounts, dict[str, Any]]:
        resolved = self.require_resolved()
        if self._rows is None:
            self._rows = self._row_stream()
        binding = resolved.parts[0].binding
        chosen_column = binding.column("chosen")
        rejected_column = binding.column("rejected")

        examples: list[PreferenceExample] = []
        results: list[Any] = []
        exhausted = False
        for prompt_id, row in self._rows:
            result = pairs.from_row(
                prompt_id, row, chosen_column=chosen_column, rejected_column=rejected_column
            )
            results.append(result)
            if isinstance(result, PreferenceExample):
                examples.append(result)
            if len(examples) >= wanted:
                break
        else:
            exhausted = True

        dropped, ties = pairs.split_counts(results)
        counts = BatchCounts(
            generated=0,
            admitted=len(examples),
            dropped=dropped,
            ties=ties,
            exhausted=exhausted,
            requested=wanted,
        )
        return examples, counts, {}

    def _row_stream(self) -> Iterator[tuple[str, Mapping[str, Any]]]:
        resolved = self.require_resolved()
        for part in resolved.parts:
            columns = [name for name, _semantic in part.binding.roles.values()]
            yield from read_rows(part, columns=columns)

    # ------------------------------------------------------ the from-generations half
    def _from_generations(
        self, wanted: int
    ) -> tuple[list[PreferenceExample], BatchCounts, dict[str, Any]]:
        resolved = self.require_resolved()
        prompts, exhausted = self._take_prompts(wanted)
        if not prompts:
            raise refuse(
                "prompt_set_exhausted",
                "This prompt set has no prompts left, so no further pairs can be generated.",
                run_id=self.context.run_id,
            )
        # TWO completions per prompt, generated in one call so both come from the same
        # parameter set: generating them in two calls a step apart would order a pair whose
        # two sides came from two different models.
        doubled = [text for _id, text in prompts for _ in range(2)]
        completions, identity, resolved_model = generation.generate(
            doubled,
            self.context.provider,
            resolved.spec.params,
            settings=self.settings,
            is_cancelled=self.context.is_cancelled,
            current_step=self.context.current_step,
        )
        self.check_cancelled()

        scalars, ordered_by, health, hashes, review_ids = self._scalars(prompts, completions)

        examples: list[PreferenceExample] = []
        results: list[Any] = []
        for index, (prompt_id, text) in enumerate(prompts):
            first, second = completions[2 * index].text, completions[2 * index + 1].text
            result = pairs.from_generations(
                prompt_id,
                text,
                first,
                second,
                first_scalar=scalars[2 * index],
                second_scalar=scalars[2 * index + 1],
                ordered_by=ordered_by,
            )
            results.append(result)
            if isinstance(result, PreferenceExample):
                examples.append(result)

        dropped, ties = pairs.split_counts(results)
        counts = BatchCounts(
            generated=len(completions),
            admitted=len(examples),
            dropped=dropped,
            ties=ties,
            exhausted=exhausted,
            requested=wanted,
        )
        return (
            examples,
            counts,
            {
                "resolved_model": resolved_model,
                "provider": identity,
                "health": health,
                "hashes": hashes,
                "review_ids": review_ids,
                "signals": health.signal_names if health is not None else (),
            },
        )

    def _scalars(
        self, prompts: list[tuple[str, str]], completions: tuple[Any, ...]
    ) -> tuple[list[float | None], str, Any, Any, tuple[str, ...]]:
        """The scalar for each completion, and **which kind of scalar it is**.

        A grader's score where one is configured; otherwise the optional ``score`` a verdict
        carries. Recorded either way, because *a pair ordered by a grader and one ordered by
        a verdict's confidence are not the same evidence*.
        """
        resolved = self.require_resolved()
        if resolved.grader is not None:
            scores = [
                float(
                    resolved.grader.grade(
                        prompt=prompts[index // 2][1], completion=completion.text, context={}
                    )
                )
                for index, completion in enumerate(completions)
            ]
            return list(scores), "grader_score", None, None, ()

        candidates = [
            Candidate(
                prompt_id=prompts[index // 2][0],
                prompt=prompts[index // 2][1],
                completion=completion.text,
                index=index,
            )
            for index, completion in enumerate(completions)
        ]
        outcome = apply(
            candidates,
            resolved.selector,
            settings=self.settings,
            run_id=self.context.run_id,
            batch_id=f"{self.context.run_id}:{self._batch_number}",
            is_cancelled=self.context.is_cancelled,
        )
        # `None` where the verdict carried no score, which `pairs.from_generations` turns
        # into `no_scalar` - a counted outcome, not a silently dropped pair.
        scalars: list[float | None] = [record.verdict.score for record in outcome.records]
        return scalars, "verdict_score", outcome.health, outcome.hashes, outcome.review_ids

    def _take_prompts(self, count: int) -> tuple[list[tuple[str, str]], bool]:
        resolved = self.require_resolved()
        if self._rows is None:
            self._rows = self._row_stream()
        column = resolved.parts[0].binding.column("prompt")
        taken: list[tuple[str, str]] = []
        for prompt_id, row in self._rows:
            value = row[column]
            if isinstance(value, str):
                taken.append((prompt_id, value))
            if len(taken) >= count:
                return taken, False
        return taken, True


def factory(context: RunContext) -> PreferencePairsSource:
    return PreferencePairsSource(context)
