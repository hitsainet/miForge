"""Where the held-out rule lives, and nowhere else  (F10 task 4.10; FR-010.18).

**One function, one rule, one refusal.** F2 (signal sources) and F4 (test and retention
scoring) both call :func:`splits_for`. If the rule appears in F2's loader as well, one of
them will drift - and miDataworks' guarantee that *"no code path merges them into another
split"* would be broken by **miForge** being that code path. The whole point of putting a
rule in one place is that the second copy cannot exist to be wrong.

**An excluded split that was asked for by name is refused, not filtered out.** Returning an
empty list would become a zero-row batch, and a zero-row batch is a step that trains on
nothing while every log line looks healthy. That is the silent-failure shape this feature's
design criterion rejects: *where there was a choice between a design that could be used
correctly and one that cannot be used incorrectly, the second was taken.*

``purpose="evaluate"`` includes held-out splits, because for a test or retention set
held-out is the **correct** answer - which is why this is one function with a purpose rather
than two functions, one of which someone would call from the wrong place.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Literal

from core.errors import refuse
from services.parts.contract_validator import Split

Purpose = Literal["train", "evaluate"]


def splits_for(
    splits: Sequence[Split], purpose: Purpose, *, named: Sequence[str] | None = None
) -> list[Split]:
    """The splits usable for ``purpose``.

    Args:
        splits: Every split of one part, as the manifest describes them.
        purpose: ``"train"`` excludes every ``held_out`` or ``evaluation_only`` split;
            ``"evaluate"`` includes them.
        named: Specific split names the caller wants. When given, an excluded one is
            **refused by name** rather than silently dropped, and an unknown one is
            refused too - a typo'd split name that quietly yielded nothing is the same
            silent zero-row batch by another route.

    Returns:
        The usable splits, in the manifest's order.

    Raises:
        Refusal: ``split_not_available_for_training`` when a named split is held out or
            evaluation-only and the purpose is training; ``split_unknown`` when a named
            split is not in the part.
    """
    by_name = {split.name: split for split in splits}

    if named is not None:
        unknown = [name for name in named if name not in by_name]
        if unknown:
            raise refuse(
                "split_unknown",
                f"This part has no split named {unknown[0]!r}. It has "
                f"{sorted(by_name)}. Refusing rather than returning nothing: an empty "
                f"result becomes a zero-row batch, which trains on nothing while every "
                f"step looks healthy.",
                unknown=unknown,
                available=sorted(by_name),
            )
        chosen = [by_name[name] for name in named]
    else:
        chosen = list(splits)

    if purpose == "evaluate":
        return chosen

    excluded = [split for split in chosen if not split.training_eligible]
    if named is not None and excluded:
        first = excluded[0]
        reasons = []
        if first.held_out:
            reasons.append("held_out")
        if first.evaluation_only:
            reasons.append("evaluation_only")
        raise refuse(
            "split_not_available_for_training",
            f"Split {first.name!r} is {' and '.join(reasons)} and cannot be training "
            f"material. miDataworks writes a held-out split as its own file and no code "
            f"path merges it into another; miForge will not be that code path. Use it as "
            f"a test or retention set instead.",
            split=first.name,
            held_out=first.held_out,
            evaluation_only=first.evaluation_only,
            purpose=purpose,
        )

    usable = [split for split in chosen if split.training_eligible]
    if chosen and not usable:
        raise refuse(
            "no_training_splits",
            "Every split of this part is held out or evaluation-only, so it offers no "
            "training material at all. Refusing rather than returning an empty list, "
            "which a caller would turn into a zero-row batch.",
            splits=[split.name for split in chosen],
        )
    return usable


def training_splits(splits: Sequence[Split]) -> list[Split]:
    """Shorthand for ``splits_for(splits, "train")``. Same rule, same place."""
    return splits_for(splits, "train")


def evaluation_splits(splits: Sequence[Split]) -> list[Split]:
    """Shorthand for ``splits_for(splits, "evaluate")``."""
    return splits_for(splits, "evaluate")
