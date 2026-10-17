"""Mode (b): accepted completions become seeds through a templating rule  (FR-006.34, .35).

R-02.26: *"the selector's accepted completions become seeds for new prompts through a
templating rule"*, closing with *"The resident model does not write its own prompt set."*

## Only ``accept`` seeds, and the comparison is written positively

``== VerdictValue.ACCEPT``, **never** ``!= REJECT``. The verdict is three-valued in R-02.9's
own text - *"accept, reject, or undecided"* - so the negative spelling admits ``undecided``,
and R-02.9 carries the prohibition this applies: *"no consumer may construct a path from
``undecided`` to admitted."* F2 writes its own half the same way and records that *"there is
no configuration value, deadline or timeout from which an ``undecided`` becomes an accept."*

Seeding a prompt from a pending completion is that same path wearing a different hat: the
unattended review queue becomes an unselected run, one round later and with nothing in the
record to say so. **Mutation M-5** includes ``undecided`` in the seed set.

**No deadline, timeout or shortfall path yields a seed either.** There is no parameter in
this module that could; the filter is one positive comparison and there is no branch around
it. ``E-14``: if the only available seeds are ``undecided``, the refresh **refuses**.

## Deterministic and non-generative  (FR-006.35)

The template is ``str.format``-style substitution over a fixed, closed set of fields. No
model is called, nothing is sampled, and this module **imports nothing from
``providers/``** - asserted by a static test with its own self-test (**M-6**). The same
rule and the same seeds produce the same prompts, which is what makes the recorded
``rule_sha256`` mean anything.

## The seam this mode needs, and who owes it

This mode reads its seeds from the live signal source through
:class:`training.refresh.base.AcceptedSeedSource`. **No source in this release implements
it**: F2's self-generation source builds ``SelectionOutcome.admitted`` inside ``next()`` and
retains it nowhere, and the batch record carries identifiers and counts but no text
(ADR-020). So this mode **refuses by name** - ``refresh_seeds_unavailable`` - rather than
seeding from a batch hash or from a pending candidate. Reported to the orchestrator as owed
by F2. *Refuse with a reason rather than guess* (ADR-018).
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from core.errors import refuse
from schemas.selector import VerdictValue
from training.refresh.base import (
    AcceptedSeedSource,
    RefreshContext,
    RefreshModeName,
    RefreshOutcome,
    Seed,
    rule_hash,
)

NAME = RefreshModeName.TEMPLATE_SEEDS.value

#: The only fields a template may reference. **Closed**, so a template naming an unknown
#: field is refused rather than rendered with an empty string - a prompt with a silently
#: missing clause is a prompt nobody wrote.
TEMPLATE_FIELDS: frozenset[str] = frozenset({"prompt", "completion", "prompt_id"})


class TemplateSeeds:
    """R-02.26's second mode."""

    name = NAME

    def refresh(self, ctx: RefreshContext, rule: Mapping[str, Any]) -> RefreshOutcome:
        template = rule.get("template")
        if not isinstance(template, str) or not template.strip():
            raise refuse(
                "refresh_rule_incomplete",
                "A template-seeds refresh needs a 'template' string. It is the rule, and "
                "a default template would invent the prompts the next round trains on.",
                mode=NAME,
            )
        _assert_fields(template)
        limit = _limit(rule)

        offered = _seeds_from(ctx, limit)
        counts = _counts(offered)

        # THE FILTER. One positive comparison, `== ACCEPT`. Written this way rather than
        # `!= REJECT` because the verdict is THREE-valued and the negative spelling admits
        # `undecided`. M-5 breaks this line.
        accepted = tuple(seed for seed in offered if seed.verdict == VerdictValue.ACCEPT)

        if not accepted:
            # E-14. Refused, in every configuration, past any deadline. There is no
            # parameter that relaxes this and no branch that could.
            raise refuse(
                "refresh_produced_no_prompts",
                f"No accepted completion is available to seed a refreshed prompt set "
                f"({counts['undecided']} awaiting review, {counts['reject']} rejected). "
                f"The refresh is refused rather than seeded from a pending verdict: "
                f"R-02.9 forbids any consumer constructing a path from 'undecided' to "
                f"admitted, and a prompt seeded from an unreviewed completion is that path "
                f"one round later. Wait for the review queue, or loosen the selector.",
                mode=NAME,
                round=ctx.round,
                seed_counts=counts,
            )

        prompts = tuple(_render(template, seed) for seed in accepted)
        return RefreshOutcome(
            mode=NAME,
            rule_sha256=rule_hash(NAME, rule),
            round=ctx.round,
            # An externally-sourced set with miForge's own provenance (R-02.40). NOT a
            # contract reference: miDataworks did not publish these prompts and claiming
            # its provenance for them would be the promotion FR-006.33 forbids.
            set_reference={
                "external_provenance": {
                    "source": "miforge:refresh/template_seeds",
                    "run_id": ctx.run_id,
                    "round": ctx.round,
                    "step": ctx.step,
                    "rule_sha256": rule_hash(NAME, rule),
                }
            },
            size=len(prompts),
            seed_counts=counts,
            # The rendered prompts travel to the loop as material, not as a part reference:
            # they are generated here and there is nothing in the index to point at.
            parts=[{"kind": "prompt_set", "inline_prompt_count": len(prompts)}],
        )


#: How many seeds a refresh asks the source for when the rule does not say. A **cap, not a
#: target**: the refreshed set is as large as the accepts allow, and a rule asking for more
#: than this gets this - because an unbounded request on a run that has accepted a hundred
#: thousand completions would build a prompt set nobody asked for.
DEFAULT_SEED_LIMIT = 512


def _limit(rule: Mapping[str, Any]) -> int:
    """How many seeds to ask for. A cadence-shaped value, so it **may** have a default.

    The line this feature draws on defaults (FR-006.5): a value that changes only *how
    much* of a recorded action happens may be defaulted, because the record says what
    happened. A value that decides *whether a run ends* may not. A seed limit is the first
    kind - the refresh record carries the resulting set's size either way.
    """
    value = rule.get("limit", DEFAULT_SEED_LIMIT)
    if not isinstance(value, int) or isinstance(value, bool) or value < 1:
        raise refuse(
            "refresh_rule_incomplete",
            f"A seed limit of {value!r} is not a positive integer. Omit it for the default "
            f"of {DEFAULT_SEED_LIMIT}, or give a count of at least 1.",
            mode=NAME,
            parameter="limit",
            value=repr(value),
        )
    return value


def _seeds_from(ctx: RefreshContext, limit: int) -> Sequence[Seed]:
    """Ask the live source for its accepted completions, or refuse naming who owes it.

    Raises:
        Refusal: ``refresh_seeds_unavailable``. **Not an empty list** - an empty list would
            fall through to E-14's refusal and blame the review queue for a mechanism that
            does not exist, which is the fabricated diagnosis ADR-018 warns about.
    """
    source = ctx.source
    if source is None or not isinstance(source, AcceptedSeedSource):
        raise refuse(
            "refresh_seeds_unavailable",
            "This run's signal source publishes no accepted-completion seeds, so a "
            "template-seeds refresh has nothing to template. F2's self-generation source "
            "builds its admitted set inside next() and retains it nowhere, and the batch "
            "record carries identifiers and counts but no text (ADR-020) - so there is no "
            "path to read the completions from. Use the new_prompt_set mode, which is "
            "complete, until the seed surface exists.",
            mode=NAME,
            run_id=ctx.run_id,
            needs="training.refresh.base.AcceptedSeedSource.accepted_seeds",
            owed_by=["F2 (signal sources)"],
        )
    seeds: Sequence[Seed] = source.accepted_seeds(limit)
    return seeds


def _counts(seeds: Sequence[Seed]) -> dict[str, int]:
    """Seeds by verdict, all three keys always present (FR-006.36).

    All three stated, including the zeros: *an absent count is indistinguishable from a
    count of nothing*, and the ``undecided`` count is the number that explains a small
    refresh to an operator.
    """
    counts = {"accept": 0, "reject": 0, "undecided": 0}
    for seed in seeds:
        counts[str(seed.verdict)] = counts.get(str(seed.verdict), 0) + 1
    return counts


def _assert_fields(template: str) -> None:
    """Refuse a template naming a field outside :data:`TEMPLATE_FIELDS`."""
    import string

    used = {field for _literal, field, _spec, _conv in string.Formatter().parse(template) if field}
    unknown = sorted(used - TEMPLATE_FIELDS)
    if unknown:
        raise refuse(
            "refresh_rule_incomplete",
            f"The template names {unknown}, which a seed does not carry. Known fields: "
            f"{sorted(TEMPLATE_FIELDS)}. Refused rather than rendered with a blank: a "
            f"prompt with a silently missing clause is a prompt nobody wrote.",
            mode=NAME,
            unknown_fields=unknown,
            known_fields=sorted(TEMPLATE_FIELDS),
        )


def _render(template: str, seed: Seed) -> str:
    """One prompt from one accepted seed. Pure substitution; nothing is generated."""
    return template.format(
        prompt=seed.prompt,
        completion=seed.completion,
        prompt_id=seed.prompt_id,
    )


MODE = TemplateSeeds()

__all__ = ["DEFAULT_SEED_LIMIT", "MODE", "NAME", "TEMPLATE_FIELDS", "TemplateSeeds"]
