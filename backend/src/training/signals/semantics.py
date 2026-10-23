"""Binding a column by its **declared semantic**, never by its name  (FR-002.7; ADR-018).

ADR-018's rule in the form it takes here: *a default that always fires is a fabricated
value, and "the first string column" is that default wearing a hat.* A corpus bound to the
wrong column trains perfectly happily on the wrong field, and nothing in the loss curve
says so - which is why the decisive test is a part whose column **names** look right
(``text``, ``chosen``) while its declared semantics are ``other``. It must refuse.

**The enum is read from the vendored schema at import**, through F10's own ``enum_at``
helper, so a semantic added upstream produces a loud startup failure here instead of
falling through a default. That is F10's ``kind_map`` arrangement, cited rather than
copied (ADR-001), and :func:`_assert_exhaustive` is what makes it bite.

## Three roles, four sources

A corpus accepts plain text, chat messages, or a prompt/completion pair. A prompt set
offers prompts. A pairs part offers chosen and rejected, with the prompt where it has one.
The table below is the whole policy; there is no second one anywhere in the feature.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from core.errors import refuse
from services.parts.contract_validator import enum_at
from training.signals.base import ColumnBinding

#: Every ``content.columns[].semantic`` the vendored dataset-version schema defines, read
#: from the file. Never re-typed: see the module docstring.
SCHEMA_SEMANTICS: frozenset[str] = enum_at("$defs", "Column", "properties", "semantic")

#: The roles a source can fill from a part, and which declared semantics can fill each.
#: ``messages`` is its own role rather than an alias of ``text`` because a chat-formatted
#: example is rendered through a tokenizer chat **template**, whose identity then belongs on
#: the batch - two runs rendered by different templates are not the same run.
ROLE_SEMANTICS: Mapping[str, tuple[str, ...]] = {
    "text": ("text",),
    "messages": ("messages",),
    "prompt": ("prompt",),
    "completion": ("completion",),
    "chosen": ("chosen",),
    "rejected": ("rejected",),
}

#: Semantics this feature deliberately fills **no** role with, each with the reason. Stated
#: rather than omitted, so :func:`_assert_exhaustive` can tell "classified as unusable"
#: from "nobody has looked at it yet" - which is the distinction that makes an upstream
#: addition a failure instead of a silent fall-through.
UNUSED_SEMANTICS: Mapping[str, str] = {
    # A labelled set is miDataworks' classification material, not a training signal F2
    # consumes: no R-02.4-R-02.7 source trains on a label.
    "label": "labels are classification material, not a signal source's input",
    "labels": "as `label`",
    "probability": "a labeller's confidence, not a training target",
    # `reference` is reference-match's comparison text and reaches F3 through the
    # evaluation context's per-item extras, not through a batch's material.
    "reference": "reference-match's comparison text, carried to the selector as per-item context",
    # `completions` (plural) is a pre-generated set. R-02.5 and R-02.7 generate their own
    # candidates through the provider, and consuming a shipped set would make the recorded
    # `resolved_model` a claim about a model miForge never called.
    "completions": (
        "a shipped candidate set; generating sources make their own and record which " "model did"
    ),
    "group": "miDataworks' grouping key; a reward group is one prompt's completions here",
    "other": "declared as uninterpreted by the contract; binding it would be a guess",
}

#: Which roles each source can use, and the refusal when a part offers none of them. The
#: tuple of tuples reads as *"any one of these role sets satisfies this source"*.
SOURCE_ROLE_SETS: Mapping[str, tuple[tuple[str, ...], ...]] = {
    "corpus": (("text",), ("messages",), ("prompt", "completion")),
    "self-generation": (("prompt",),),
    "preference-pairs": (("prompt", "chosen", "rejected"), ("chosen", "rejected")),
    "reward": (("prompt",),),
}

#: One refusal code per source, so the operator reads the same code the FPRD names.
NO_COLUMN_REFUSAL: Mapping[str, str] = {
    "corpus": "corpus_no_content_column",
    "self-generation": "prompt_set_no_prompt_column",
    "preference-pairs": "pairs_no_preference_columns",
    "reward": "prompt_set_no_prompt_column",
}


def _assert_exhaustive() -> None:
    """Every schema semantic is either bound to a role or recorded as unused.

    Runs at **import**, so an upstream addition fails at startup rather than reaching a
    ``resolve_columns`` call that would otherwise never consider it. Refused rather than
    defaulted to "unusable": a new semantic may well be a content column, and deciding
    that silently is how a feature ends up ignoring the field a run was meant to train on.
    """
    classified = {semantic for members in ROLE_SEMANTICS.values() for semantic in members}
    classified |= set(UNUSED_SEMANTICS)
    unknown = sorted(SCHEMA_SEMANTICS - classified)
    if unknown:
        raise refuse(
            "unmapped_column_semantic",
            f"midataworks-dataset-version-v1.json declares column semantics {unknown} that "
            f"this release classifies neither as a role nor as unused. Decide each "
            f"deliberately in training/signals/semantics.py: a new semantic may be content, "
            f"and treating it as unusable by default would silently ignore the field a run "
            f"was meant to train on.",
            unknown=unknown,
            roles=sorted(classified),
        )
    stale = sorted(classified - SCHEMA_SEMANTICS)
    if stale:
        raise refuse(
            "unmapped_column_semantic",
            f"This release maps column semantics {stale} that the vendored schema no longer "
            f"declares. A role bound to a semantic nobody publishes can never be filled, so "
            f"the refusal would arrive as 'no content column' on a part that is fine.",
            stale=stale,
        )


_assert_exhaustive()


def usable_role_sets(source: str) -> tuple[tuple[str, ...], ...]:
    """The role sets ``source`` accepts. Refuses an unknown source rather than returning ()."""
    sets = SOURCE_ROLE_SETS.get(source)
    if sets is None:
        raise refuse(
            "signal_source_unknown",
            f"No column policy for a source named {source!r}. Known: "
            f"{sorted(SOURCE_ROLE_SETS)}.",
            name=source,
            registered=sorted(SOURCE_ROLE_SETS),
        )
    return sets


def declared_semantics(columns: Sequence[Mapping[str, Any]]) -> list[str]:
    """Every semantic a part declares, sorted - the list a refusal shows the operator.

    Shown because a refusal that does not say what the part *does* declare makes the
    operator guess, which is the thing ADR-018 forbids the code from doing and is no better
    when a person has to do it (ADR-013: say what to do next).
    """
    return sorted({str(column.get("semantic")) for column in columns})


def _candidates(columns: Sequence[Mapping[str, Any]], role: str) -> list[Mapping[str, Any]]:
    """Columns whose **declared** semantic can fill ``role``. Never matched on name."""
    allowed = ROLE_SEMANTICS[role]
    return [column for column in columns if str(column.get("semantic")) in allowed]


def resolve_columns(
    columns: Sequence[Mapping[str, Any]],
    source: str,
    *,
    configured: Mapping[str, str] | None = None,
    part_kind: str | None = None,
) -> ColumnBinding:
    """Bind the roles ``source`` needs, from the contract's declared semantics only.

    Args:
        columns: ``content.columns`` of the part's manifest, as miDataworks published it.
        source: One of the four canonical source names.
        configured: ``role -> column name``, for a part declaring more than one candidate
            for a role. Choosing among candidates is **configuration, not a heuristic**.
        part_kind: Named in the refusal, so the operator knows which part objected.

    Returns:
        A :class:`ColumnBinding` naming, per role, the column and the semantic that filled
        it. Recorded on the batch, because a corpus trained from ``messages`` and one
        trained from ``text`` are not the same run.

    Raises:
        Refusal: the source's own ``*_no_*_column`` code listing what the part declares
            (E-4, E-12), or ``column_semantic_ambiguous`` listing the candidates (FR-002.7).
    """
    configured = dict(configured or {})
    role_sets = usable_role_sets(source)

    for role_set in role_sets:
        if all(_candidates(columns, role) for role in role_set):
            roles: dict[str, tuple[str, str]] = {}
            for role in role_set:
                roles[role] = _bind_one(columns, role, configured.get(role), source)
            return ColumnBinding(roles=roles)

    raise refuse(
        NO_COLUMN_REFUSAL[source],
        f"This part declares no column that can fill {source!r}'s material. It needs "
        f"{' or '.join('+'.join(role_set) for role_set in role_sets)}, matched on the "
        f"contract's DECLARED semantic - never on a column's name, a file's name or a "
        f"column's position. It declares: {', '.join(declared_semantics(columns)) or 'nothing'}. "
        f"Export the version with a content column, or choose a different part.",
        source=source,
        part_kind=part_kind,
        required=[list(role_set) for role_set in role_sets],
        declared=declared_semantics(columns),
    )


def _bind_one(
    columns: Sequence[Mapping[str, Any]], role: str, chosen: str | None, source: str
) -> tuple[str, str]:
    """One role's ``(column name, semantic)``. Refuses ambiguity; **never picks the first**."""
    candidates = _candidates(columns, role)
    if chosen is not None:
        for column in candidates:
            if str(column.get("name")) == chosen:
                return chosen, str(column.get("semantic"))
        raise refuse(
            "column_binding_unknown",
            f"The configuration binds {role!r} to a column named {chosen!r}, which this "
            f"part does not declare with a usable semantic. Usable candidates: "
            f"{[str(c.get('name')) for c in candidates]}.",
            role=role,
            configured=chosen,
            candidates=[str(column.get("name")) for column in candidates],
        )
    if len(candidates) > 1:
        raise refuse(
            "column_semantic_ambiguous",
            f"This part declares {len(candidates)} columns that could fill {role!r}: "
            f"{[str(c.get('name')) for c in candidates]}. Name the one to use in the run's "
            f"column configuration. Refusing rather than picking the first: 'the first "
            f"usable column' is a default that always fires, and a corpus bound to the "
            f"wrong field trains happily and says nothing.",
            role=role,
            source=source,
            candidates=[str(column.get("name")) for column in candidates],
        )
    column = candidates[0]
    return str(column.get("name")), str(column.get("semantic"))
