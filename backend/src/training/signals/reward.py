"""Reward: generate in groups, score with a grader, record the statistics  (R-02.7).

FR-002.23 - FR-002.27. A reward batch is prompts from a **prompt set** - verified:
``target.miforge_set_kind`` has **no reward value** - N completions generated per prompt
through the provider, each scored by a **grader resolved from F3's registry**. F2
implements no grader.

## F2 declares no range, normalises nothing, clamps nothing

R-02.7 says only *"a scored set of completions."* The range is the **grader's** - F3 fixes
it at 0..1 in its own contract and **refuses a value outside it rather than clamping**, and
F2 propagates that refusal without a second opinion. The declared range is recorded beside
the statistics **always**, because two runs' rewards are not comparable unless both say
their scale (ADR-012: every number names its scale and sample).

## Zero spread is named on the record, and the group is NOT skipped

A group whose rewards are all equal yields a **zero advantage for every member**: a
group-relative objective subtracts the group mean, so the step runs in full - a forward
pass, a backward pass, an optimizer step - and applies **numerically nothing**. Hours of
GPU time, a recorded loss, no learning.

The next reader will want to "fix" this by skipping the group. **Do not.** Skipping hides
the one failure in this feature that costs a full GPU step and reports a healthy loss.
Recording it is what lets F6 stop on R-02.25 and F9 show it; stopping is **not** F2's.
"""

from __future__ import annotations

from collections.abc import Iterator, Mapping, Sequence
from typing import Any

from core.errors import refuse
from services.graders import SCORE_RANGE
from services.selectors.types import SIGNAL_NO_SPREAD
from training.signals import generation
from training.signals.base import (
    BatchCounts,
    RewardGroup,
    RunContext,
    SourceCapability,
    TrainingBatch,
)
from training.signals.common import BaseSource
from training.signals.parts_reader import read_rows


def group(
    prompt_id: str,
    prompt: str,
    completions: Sequence[str],
    scores: Sequence[float],
    *,
    declared_range: tuple[float, float] = SCORE_RANGE,
    variance_epsilon: float = 1e-9,
) -> RewardGroup:
    """One prompt's group, with min, max, mean and variance on the record.

    Args:
        declared_range: **The grader configuration's**, defaulting to the range F3's
            contract fixes. F2 declares none of its own and clamps nothing - a clamp turns
            a grader that misread its scale into a confident number.
        variance_epsilon: Below this, the spread counts as zero. Explicit because "zero" in
            floating point is a choice, and it is the same epsilon F3 raises its own signal
            at so the two cannot disagree about one group.

    Raises:
        Refusal: ``reward_group_too_small`` (E-16) - **a group of one has no relative
            advantage by construction**, so it is refused rather than contributing a zero
            update that looks like a step; ``reward_score_count_mismatch`` when the scores
            and the completions do not line up positionally.
    """
    if len(completions) < 2:
        raise refuse(
            "reward_group_too_small",
            f"A reward group needs at least two completions and this one has "
            f"{len(completions)}. A group-relative objective computes each member's "
            f"advantage against the group's own mean, so a group of one has an advantage of "
            f"zero by construction: the step would run in full and apply nothing. Raise "
            f"SIGNAL_REWARD_GROUP_SIZE, or configure group_size on the run.",
            prompt_id=prompt_id,
            size=len(completions),
            minimum=2,
        )
    if len(scores) != len(completions):
        raise refuse(
            "reward_score_count_mismatch",
            f"{len(scores)} scores were produced for {len(completions)} completions. "
            f"Refusing rather than joining what arrived: the join is positional, so a gap "
            f"misattributes every score after it.",
            prompt_id=prompt_id,
            completions=len(completions),
            scores=len(scores),
        )

    values = [float(score) for score in scores]
    mean = sum(values) / len(values)
    # Population variance, not the sample estimator: the group IS the population a
    # group-relative advantage is computed against, and an n-1 denominator would report a
    # spread the objective does not see.
    variance = sum((value - mean) ** 2 for value in values) / len(values)
    return RewardGroup(
        prompt_id=prompt_id,
        prompt=prompt,
        completions=tuple(completions),
        scores=tuple(values),
        declared_range=declared_range,
        minimum=min(values),
        maximum=max(values),
        mean=mean,
        variance=variance,
        zero_spread=variance <= variance_epsilon,
    )


def signals_for(groups: Sequence[RewardGroup]) -> tuple[str, ...]:
    """The named signals a batch of groups raises. Today: ``grader_scores_have_no_spread``.

    **F3's name, imported rather than retyped** (``SIGNAL_NO_SPREAD``). F6 stops on these
    strings and F9 renders them, so a second spelling here would be a signal that fires and
    is never seen - which is worse than one that does not fire at all, because the run looks
    healthy and the record claims it was checked.
    """
    return (SIGNAL_NO_SPREAD,) if any(item.zero_spread for item in groups) else ()


def statistics_for(groups: Sequence[RewardGroup]) -> list[dict[str, object]]:
    """Per-group statistics as the run log carries them, **with the declared range**.

    The range travels with every group rather than once per batch: a batch may be scored by
    one grader today and the figures are read months later beside another run's, and a mean
    of 0.7 means nothing without the scale it was taken on.
    """
    return [
        {
            "prompt_id": item.prompt_id,
            "n": len(item.scores),
            "min": item.minimum,
            "max": item.maximum,
            "mean": item.mean,
            "variance": item.variance,
            "declared_range": list(item.declared_range),
            "zero_spread": item.zero_spread,
        }
        for item in groups
    ]


# ======================================================================== the source


NAME = "reward"

CAPABILITY = SourceCapability(
    name=NAME,
    part_kind="prompt_set",
    needs_selector=False,
    needs_grader=True,
    needs_generation=True,
    objective="grpo",
    summary=(
        "N completions per prompt, each scored by a grader, trained with a group-relative "
        "policy gradient and no value network"
    ),
)


class RewardSource(BaseSource):
    """R-02.7. Generate in groups, score with a grader, record the statistics.

    **This source records its batch normally, through the real contract.**
    ``batch-record-v1.json``'s ``source`` enum is
    ``corpus | self-generation | preference-pairs | reward`` and
    ``checkpoint-manifest-v1.json``'s ``signal_source`` - which R-02.7 and FR-002.1 name -
    carries the same four values with the same spellings, so a registry name needs no
    translation to become either and ``reward`` is recordable like any other source.

    *This docstring read* **"This source cannot write its batch record today"** *and quoted
    the enum as* ``corpus | self_generation | preference_pairs | mixed`` *until 2026-10-16.
    Both halves were true when written and both became false on 2026-10-14, when the two
    enums were made one vocabulary:* ``reward`` *was added,* ``mixed`` *removed (no
    requirement describes a mixed signal source) and the separators aligned. The paragraph
    went on to say that pre-flight "therefore refuses* ``batch_source_not_in_contract``
    *before a run starts, naming the gap" - which is the sentence most worth correcting,
    because it told the next reader that this source is unusable and that its refusal is
    the expected outcome of configuring it.*

    **The refusal itself is NOT dead, and it is not about this source.**
    :meth:`training.signals.common.BaseSource.preflight` calls
    :func:`training.signals.batch_record.record_source` before it resolves a single part, and
    that guard still refuses any name outside the contract's closed enum - which no built-in
    source can be, but a plugin's fifth source can, since
    ``training.signals.registry.register`` is public and takes any name. Writing such a batch
    down as ``corpus`` or ``reward`` instead would be a false claim about what trained the
    checkpoint, which is the one question the record exists to answer six months later.
    """

    NAME = NAME
    CAPABILITY = CAPABILITY

    def __init__(self, context: RunContext | None = None) -> None:
        super().__init__(context)
        self._prompts: Iterator[tuple[str, Mapping[str, Any]]] | None = None

    def resolve_extra(self, resolved: Any) -> None:
        """Pre-flight refuses a group below two and an absent provider (E-16, FR-002.4)."""
        size = resolved.spec.group_size or self.settings.reward_group_size
        if size < 2:
            raise refuse(
                "reward_group_too_small",
                f"The configured group size is {size}. A group-relative objective computes "
                f"each member's advantage against the group's own mean, so a group of one "
                f"has an advantage of zero by construction: every step would run in full "
                f"and apply nothing. Set SIGNAL_REWARD_GROUP_SIZE, or group_size on the "
                f"run, to at least 2.",
                size=size,
                minimum=2,
            )
        if self.context.provider is None:
            raise refuse(
                "generation_provider_not_configured",
                "The reward source generates the completions it scores. The default "
                "provider is in-process and needs no sibling application.",
                source=self.NAME,
                capabilities_unavailable=["reward"],
            )

    def next(self) -> TrainingBatch:
        """One batch of groups, with per-group statistics and the declared range."""
        resolved = self.require_resolved()
        self.check_cancelled()
        size = resolved.spec.group_size or self.settings.reward_group_size
        wanted = resolved.spec.groups_per_batch or self.settings.reward_groups_per_batch

        prompts, exhausted = self._take(wanted)
        if not prompts:
            raise refuse(
                "prompt_set_exhausted",
                "This prompt set has no prompts left, so no further reward groups can be "
                "generated.",
                run_id=self.context.run_id,
            )

        # Every completion of every group in ONE call: a group's advantage is relative to
        # its own members, so two calls a step apart would compute an advantage across two
        # different parameter sets.
        expanded = [text for _id, text in prompts for _ in range(size)]
        completions, identity, resolved_model = generation.generate(
            expanded,
            self.context.provider,
            resolved.spec.params,
            settings=self.settings,
            is_cancelled=self.context.is_cancelled,
            current_step=self.context.current_step,
        )
        self.check_cancelled()

        grader = resolved.grader
        if grader is None:  # pragma: no cover - pre-flight refuses this
            raise refuse(
                "grader_not_configured",
                "This reward source reached assembly with no grader resolved. Pre-flight "
                "refuses that, so reaching here means something bypassed it - and scoring "
                "with no grader would mean inventing a reward.",
                source=self.NAME,
            )

        groups = []
        for index, (prompt_id, text) in enumerate(prompts):
            window = completions[index * size : (index + 1) * size]
            scores = [
                # The grader's refusal is propagated, never caught and never clamped: F3
                # refuses a value outside its declared range, and a second opinion here
                # would be a second answer about what a completion scored.
                float(grader.grade(prompt=text, completion=item.text, context={}))
                for item in window
            ]
            groups.append(
                group(
                    prompt_id,
                    text,
                    [item.text for item in window],
                    scores,
                    variance_epsilon=self.settings.reward_variance_epsilon,
                )
            )

        counts = BatchCounts(
            generated=len(completions),
            admitted=len(groups),
            exhausted=exhausted,
            requested=wanted,
        )
        self.handle_shortfall(counts, requested=wanted)
        return self.record(
            examples=groups,
            counts=counts,
            resolved_model=resolved_model,
            provider=identity,
            # The named signal rides on the batch and in the log line. The record's
            # SelectionHealth has four fields and no room for it - reported as owed.
            signals=signals_for(groups),
        )

    def _take(self, count: int) -> tuple[list[tuple[str, str]], bool]:
        resolved = self.require_resolved()
        if self._prompts is None:
            self._prompts = self._prompt_stream()
        column = resolved.parts[0].binding.column("prompt")
        taken: list[tuple[str, str]] = []
        for prompt_id, row in self._prompts:
            value = row[column]
            if isinstance(value, str):
                taken.append((prompt_id, value))
            if len(taken) >= count:
                return taken, False
        return taken, True

    def _prompt_stream(self) -> Iterator[tuple[str, Mapping[str, Any]]]:
        resolved = self.require_resolved()
        for part in resolved.parts:
            columns = [name for name, _semantic in part.binding.roles.values()]
            yield from read_rows(part, columns=columns)


def factory(context: RunContext) -> RewardSource:
    return RewardSource(context)
