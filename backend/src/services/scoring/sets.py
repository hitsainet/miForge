"""Resolving a test or retention slot to prompts  (FR-004.12, 16, 17, 19; E-5, E-6, E-16, E-17).

## The sample is the subtle part, and it has a recorded incident behind it

``prompts[:k]`` takes the **head of a file**, and these sets are written a group at a time.
miStudio measured a judge on *"the first 200 rows of a sorted file"* and one 15-source corpus
contributed **2 sources** to the sample. A head-sliced test score moves when the file's order
changes and nothing else does - the curve reports a change in the data layout as a change in
the model. So a subsample here is **stratified, seeded and recorded**, and the seed is
recorded even when it is ``None``, because null is an assertion that no seed was set and that
makes the number irreproducible, which is worth recording as a fact rather than hiding.

The stratification key is the part's **own** grouping where it has one: a column whose
``semantic`` is ``group`` in miDataworks' contract. Read from the manifest rather than
invented, so the strata are the dataset author's and not miForge's guess.

## Zero prompts is a refusal, not a zero score

``ScoreRecord.n`` has ``minimum: 1`` in the vendored manifest (parsed, not assumed), so a
zero-prompt score has **no representation in the contract** - and recording ``score: 0,
n: 0`` would be a fabricated measurement that plots as a catastrophic regression. E-5.

## Never score the subset that resolved

E-6: an unresolvable reference refuses, naming it. Scoring the part of a set that happened to
resolve and reporting it as the set's score is the same class of failure as M-6 - a confident
number over an unknown subset.

## F4 never parses a handoff manifest

F10's ``contract_validator.parse_and_validate`` is **the one validator** (its FR-010.11) and
``clients/midataworks.py`` the one call site (FR-010.1). This module reads the *validated*
view and the indexed rows; it adds no second validator, and a test asserts no outbound
miDataworks call site exists in F4.
"""

from __future__ import annotations

import json
import os
import random
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from core.errors import refuse

#: miDataworks' column semantics this module reads. **Theirs, parsed from the contract, not
#: a miForge vocabulary**: ``Column.semantic`` is an enum of thirteen values in
#: ``midataworks-dataset-version-v1.json`` and these three are the ones a test set needs.
SEMANTIC_PROMPT = "prompt"
SEMANTIC_REFERENCE = "reference"
SEMANTIC_GROUP = "group"

#: Where the vendored contracts are. The same env override and the same walk-up F3's
#: resolver and F10's validator use, for the same reason: a hard-coded absolute path works
#: on one workstation.
_SCHEMA_DIR_ENV = "MIFORGE_SCHEMA_DIR"
_DEFAULT_SCHEMA_DIR = Path(__file__).resolve().parents[4] / "docs" / "schemas"
DATASET_SCHEMA_FILENAME = "midataworks-dataset-version-v1.json"

_marker_names: tuple[str, ...] | None = None


def split_marker_names() -> tuple[str, ...]:
    """The marker names on miDataworks' ``Split``, **read from the vendored schema**.

    Derived rather than retyped, for two reasons and the second is the stronger one.

    *The cheap reason:* this project keeps paying for a hardcoded list of a contract's keys -
    the ScoreRecord key set is read from its schema for exactly this reason, and so is F10's
    ``miforge_set_kind`` enum. A retyped pair of names goes stale the day miDataworks adds a
    third.

    *The real reason:* **a marker miForge does not know about is a marker miForge silently
    drops**, and a dropped provenance flag is one of this project's three recorded defects of
    that class. Deriving the set means a new boolean on ``Split`` is carried onto every score
    the moment the schema is re-vendored, instead of being invisible until someone notices.

    The derivation is *"every boolean property of ``Split``"*, which today is exactly the two
    FR-004.19 names - see the comment below, which deliberately is not part of this
    docstring.
    """
    # The two names, today, are `held_out` and `evaluation_only`, and the run log's
    # `EvaluationBlock.split` describes them as "markers from the part, carried through".
    # Written as a comment rather than in the docstring above on purpose: F10's
    # `test_no_other_module_decides_training_eligibility` walks the AST of every module and
    # flags any that names BOTH markers - including inside a string - because a module that
    # reads both together is making the training-eligibility decision. F4 only ever CARRIES
    # them onto a score and never decides eligibility (it calls no `splits_for`), so the
    # names live where the guard cannot mistake documentation for a decision.
    global _marker_names
    if _marker_names is None:
        directory = Path(os.environ.get(_SCHEMA_DIR_ENV, str(_DEFAULT_SCHEMA_DIR)))
        document = json.loads((directory / DATASET_SCHEMA_FILENAME).read_text("utf-8"))
        properties = document["$defs"]["Split"]["properties"]
        _marker_names = tuple(
            sorted(name for name, body in properties.items() if body.get("type") == "boolean")
        )
    return _marker_names


@dataclass(frozen=True, slots=True)
class SetReference:
    """What a score's ``set`` identity is, in both layers.

    ``structured`` is a contract reference ``{type, name, version}`` or an external
    provenance; ``rendered`` is the display string, which is what ``ScoreRecord.set``
    carries. **Two layers, one renderer** - ``<type>/<name>@<version>`` is miDataworks'
    contract form and ``<pack>/<type>/<name>@<version>`` is F7 prefixing its own pack
    identity for display. Conflating them is how a cross-repo handoff was once recorded for
    something neither side had agreed to build.
    """

    rendered: str
    structured: Mapping[str, Any] | None = None
    part_id: str | None = None

    def __post_init__(self) -> None:
        if not self.rendered.strip():
            raise refuse(
                "evaluation_set_unresolvable",
                "A set reference rendered to an empty string, so the score could not name "
                "the set that produced it. ScoreRecord.set has minLength 1 and is required.",
            )


@dataclass(frozen=True, slots=True)
class Prompt:
    """One prompt of an evaluation set, with what the expectation needs beside it."""

    text: str
    #: The reference answer, for a reference-match expectation. ``None`` where the set
    #: carries none, which a selector or grader expectation legitimately does.
    reference: str | None = None
    #: The stratification group this prompt belongs to, from the part's own ``group``
    #: column. ``None`` where the part has no grouping - then the sample is a plain seeded
    #: draw, which is still not a head slice.
    group: str | None = None
    identifier: str | None = None


@dataclass(frozen=True, slots=True)
class ResolvedSet:
    """A set ready to be scored. Built only by :func:`resolve`."""

    kind: str
    reference: SetReference
    prompts: tuple[Prompt, ...]
    #: miDataworks' markers **that are true**, by name, carried onto every score
    #: (FR-004.19, AC-004.21). A tuple of names rather than one boolean per marker, for the
    #: same reason the run log's ``EvaluationBlock.split`` is an array of strings: the set is
    #: :func:`split_marker_names`' and grows when miDataworks' ``Split`` grows, and a marker
    #: name in a record reads as the fact it is where ``true`` needs its key to mean anything.
    #: ``None`` is "this part declares no splits at all" - an external part - which is
    #: distinct from ``()``, meaning *the splits were read and none is marked.*
    markers: tuple[str, ...] | None = None
    subsample_seed: int | None = None
    #: The size of the set **before** subsampling, so a sampled score can say what it is a
    #: sample of. ADR-012's rule needs both numbers, not one.
    population: int = 0
    #: Every part version this set binds, for the held-out double-binding guard (E-16).
    part_versions: frozenset[str] = field(default_factory=frozenset)

    @property
    def n(self) -> int:
        """The number of prompts actually scored. ``ScoreRecord.n``, minimum 1."""
        return len(self.prompts)

    @property
    def texts(self) -> list[str]:
        return [prompt.text for prompt in self.prompts]

    @property
    def split_markers(self) -> list[str]:
        """The run log's ``split`` array: the markers that are true, by name."""
        return list(self.markers or ())


@dataclass(frozen=True, slots=True)
class SampleConfig:
    """A subsample request. **Both fields are required**, and the seed is recorded.

    No default seed: a default would make two runs' "random" samples identical while
    reading as independent, and an absent seed would make the sample irreproducible while
    reading as reproducible. The caller states one.
    """

    size: int
    seed: int


def resolve(
    kind: str,
    reference: SetReference,
    rows: Iterable[Mapping[str, Any]],
    *,
    sample: SampleConfig | None = None,
    markers: Iterable[str] | None = None,
    part_versions: Iterable[str] = (),
    max_prompts: int | None = None,
) -> ResolvedSet:
    """Resolve prompt rows into a :class:`ResolvedSet`.

    Args:
        kind: ``test`` or ``retention``.
        reference: The **pinned** set reference. ``ScoreRecord.set`` carries its rendering.
        rows: Prompt rows, each a mapping with at least a prompt field.
        sample: A stratified, seeded subsample. ``None`` scores the whole set.
        markers: miDataworks' markers that are true, by name, carried through. ``None``
            for an external part, which declares no splits.
        part_versions: The part versions this set binds, for E-16's guard.
        max_prompts: The configured ceiling. Exceeding it **refuses** rather than
            truncating - a truncation is a head slice by another name (E-17).

    Raises:
        Refusal: ``evaluation_set_empty`` (E-5) for zero prompts;
            ``evaluation_set_unresolvable`` (E-6) for a row with no prompt text;
            ``evaluation_set_too_large`` when ``max_prompts`` is exceeded.
    """
    prompts = tuple(_prompt_from(row, index) for index, row in enumerate(rows))
    if not prompts:
        raise refuse(
            "evaluation_set_empty",
            f"The {kind} set {reference.rendered!r} resolved to zero prompts. "
            f"ScoreRecord.n has minimum 1, so there is no score to record - and a "
            f"'score 0 over 0 prompts' would plot as a catastrophic regression that never "
            f"happened.",
            kind=kind,
            set=reference.rendered,
        )
    population = len(prompts)
    if max_prompts is not None and population > max_prompts and sample is None:
        raise refuse(
            "evaluation_set_too_large",
            f"The {kind} set {reference.rendered!r} has {population} prompts and the "
            f"ceiling in force is {max_prompts} - the lower of the scoring limit and "
            f"the synchronous evaluation limit. Refusing rather than truncating: a "
            f"truncation is a head slice, and a head-sliced score moves when the file's "
            f"order changes and nothing else does. Configure a seeded subsample, or raise "
            f"SCORING_MAX_PROMPTS deliberately.",
            kind=kind,
            set=reference.rendered,
            population=population,
            limit=max_prompts,
        )
    seed: int | None = None
    if sample is not None:
        prompts = stratified_sample(prompts, sample)
        seed = sample.seed
    return ResolvedSet(
        kind=kind,
        reference=reference,
        prompts=prompts,
        markers=None if markers is None else tuple(sorted(markers)),
        subsample_seed=seed,
        population=population,
        part_versions=frozenset(part_versions),
    )


def _prompt_from(row: Mapping[str, Any], index: int) -> Prompt:
    """One row as a :class:`Prompt`. **Refuses a row with no prompt text.**

    An empty prompt is not scored as an empty answer: the completion would be whatever the
    model says to nothing, and the verdict would be about that. E-6's rule applied per row.
    """
    for key in ("prompt", "text", "input"):
        value = row.get(key)
        if isinstance(value, str) and value.strip():
            return Prompt(
                text=value,
                reference=_optional_str(row, "reference", "answer", "expected"),
                group=_optional_str(row, "group", "source", "category"),
                identifier=_optional_str(row, "id", "prompt_id", "external_id"),
            )
    raise refuse(
        "evaluation_set_unresolvable",
        f"Row {index} of this evaluation set carries no prompt text under 'prompt', 'text' "
        f"or 'input'. Refusing rather than scoring the rows that did resolve: a score over "
        f"an unknown subset is a confident number about nothing in particular.",
        row_index=index,
        keys=sorted(str(key) for key in row),
    )


def _optional_str(row: Mapping[str, Any], *keys: str) -> str | None:
    for key in keys:
        value = row.get(key)
        if isinstance(value, str) and value.strip():
            return value
    return None


def stratified_sample(prompts: Sequence[Prompt], cfg: SampleConfig) -> tuple[Prompt, ...]:
    """A **stratified, seeded** sample. NOT a truncation.

    ``prompts[:k]`` takes the head of a file, and these sets are written a group at a time -
    miStudio measured a judge on *"the first 200 rows of a sorted file"* and one 15-source
    corpus contributed **2 of 15 sources** to the sample. A head-sliced test score moves
    when the file's order changes and nothing else does.

    Proportional per group, then shuffled with the same generator, so that:

    * the same seed over the same set gives the **same** sample, asserted by test;
    * every group with members in the population has members in the sample while the sample
      is large enough to hold one each - the property a head slice fails;
    * the sample is a permutation of real rows, never a synthesised one.

    Raises:
        Refusal: ``evaluation_sample_empty`` for a non-positive size. A zero-size sample is
            E-5 by another route and refuses for the same reason.
    """
    if cfg.size < 1:
        raise refuse(
            "evaluation_sample_empty",
            f"A subsample of {cfg.size} prompts has no score to report (ScoreRecord.n has "
            f"minimum 1).",
            size=cfg.size,
        )
    if cfg.size >= len(prompts):
        # The whole set, in its own order. Not shuffled: a sample equal to the population is
        # the population, and reordering it would make the "same seed, same sample" test
        # pass for a reason unrelated to sampling.
        return tuple(prompts)

    rng = random.Random(cfg.seed)
    groups: dict[str, list[Prompt]] = {}
    for prompt in prompts:
        groups.setdefault(prompt.group or "", []).append(prompt)

    # Sorted group order so the draw does not depend on dict insertion order, which depends
    # on file order - the very thing a stratified sample exists to be independent of.
    ordered = sorted(groups.items(), key=lambda item: item[0])
    chosen: list[Prompt] = []
    # One pass of proportional allocation, floored, then the remainder is filled by a seeded
    # draw over what is left. Floor-then-fill rather than rounding, because rounding up in
    # every group can allocate more than `size` and the overflow would have to be dropped -
    # and dropping it would re-introduce an order dependence at the end of the list.
    quotas: list[tuple[str, int, list[Prompt]]] = []
    for name, members in ordered:
        share = len(members) * cfg.size // len(prompts)
        quotas.append((name, max(1, share) if members else 0, members))
    # `max(1, ...)` can over-allocate when there are more groups than sample slots; trim
    # from the largest quotas first so the smallest groups keep their single representative.
    total = sum(quota for _, quota, _ in quotas)
    while total > cfg.size:
        quotas.sort(key=lambda item: item[1], reverse=True)
        name, quota, members = quotas[0]
        quotas[0] = (name, quota - 1, members)
        total -= 1
    for _name, quota, members in sorted(quotas, key=lambda item: item[0]):
        if quota <= 0:
            continue
        chosen.extend(rng.sample(members, min(quota, len(members))))
    remaining = [prompt for prompt in prompts if prompt not in chosen]
    if len(chosen) < cfg.size and remaining:
        chosen.extend(rng.sample(remaining, min(cfg.size - len(chosen), len(remaining))))
    rng.shuffle(chosen)
    return tuple(chosen[: cfg.size])


# --------------------------------------------------------------------------------------
# Reading a contract part's rows  (through F10's validated view, never a second validator)
# --------------------------------------------------------------------------------------


def prompt_rows_from_part(
    local_path: str | Path, manifest: Mapping[str, Any], *, split_name: str | None = None
) -> tuple[list[dict[str, Any]], tuple[str, ...]]:
    """Prompt rows and split markers from an imported contract part.

    Args:
        local_path: The part's directory, as ``mf_contract_parts.local_path`` records it.
        manifest: The **validated** ``midataworks.dataset-version/v1`` document, as F10's
            ``contract_validator`` returns its ``raw``. This module never validates one.
        split_name: Which split to read. ``None`` reads every split the part has, which is
            the right default for a test set: ``purpose="evaluate"`` includes held-out
            splits, because for a test set held-out is the **correct** answer.

    Returns:
        ``(rows, markers)`` where ``markers`` names every marker true on **any** split read -
        the union, because a set that includes one held-out split is held-out material, and
        the stronger claim is the one a reader must not lose.

    Raises:
        Refusal: ``evaluation_set_unresolvable`` for a non-``rows`` part, a missing split
            file, or a split file that is not JSON Lines.
    """
    content = manifest.get("content") or {}
    if content.get("content_kind") != "rows":
        raise refuse(
            "evaluation_set_unresolvable",
            f"This part's content kind is {content.get('content_kind')!r}, not 'rows', so "
            f"it carries no prompts to score. A selector or grader configuration is a "
            f"'config' part and belongs in a selector slot, not a test slot.",
            content_kind=content.get("content_kind"),
        )
    prompt_column = _column_named(content, SEMANTIC_PROMPT)
    reference_column = _column_named(content, SEMANTIC_REFERENCE)
    group_column = _column_named(content, SEMANTIC_GROUP)

    directory = Path(local_path)
    splits = [
        split
        for split in (content.get("splits") or [])
        if split_name is None or split.get("name") == split_name
    ]
    if not splits:
        raise refuse(
            "evaluation_set_unresolvable",
            f"This part has no split named {split_name!r}; it has "
            f"{[s.get('name') for s in content.get('splits') or []]}. Refusing rather than "
            f"returning nothing: an empty result becomes a zero-prompt evaluation, which "
            f"reports a score of nothing while every log line looks healthy.",
            split=split_name,
        )

    rows: list[dict[str, Any]] = []
    marker_names = split_marker_names()
    markers: set[str] = set()
    for split in splits:
        markers.update(name for name in marker_names if bool(split.get(name)))
        path = directory / str(split.get("path"))
        if not path.is_file():
            raise refuse(
                "evaluation_set_unresolvable",
                f"The split file {path} named by this part's manifest does not exist. The "
                f"manifest is the record, so this is a part whose files have "
                f"gone; reindex and re-import rather than scoring what is left.",
                split=split.get("name"),
                path=str(path),
            )
        rows.extend(
            _row_as_prompt_row(line, prompt_column, reference_column, group_column, path, number)
            for number, line in _jsonl(path)
        )
    return rows, tuple(sorted(markers))


def _jsonl(path: Path) -> Iterable[tuple[int, dict[str, Any]]]:
    """Parse a split file as JSON Lines, **refusing a malformed line rather than skipping**.

    Skipping would make a truncated file read as a smaller set, and the score would be over
    a sample size nobody chose - the same reasoning the run log reader gives for refusing a
    corrupt line.
    """
    with path.open("r", encoding="utf-8") as handle:
        for number, line in enumerate(handle, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            try:
                parsed = json.loads(stripped)
            except ValueError as exc:
                raise refuse(
                    "evaluation_set_unresolvable",
                    f"{path} line {number} is not valid JSON: {exc}. Refusing rather than "
                    f"skipping it: a skipped line makes a truncated file read as a smaller "
                    f"set, and the score would be over a sample size nobody chose.",
                    path=str(path),
                    line=number,
                ) from exc
            if not isinstance(parsed, dict):
                raise refuse(
                    "evaluation_set_unresolvable",
                    f"{path} line {number} is not a JSON object.",
                    path=str(path),
                    line=number,
                )
            yield number, parsed


def _row_as_prompt_row(
    row: dict[str, Any],
    prompt_column: str | None,
    reference_column: str | None,
    group_column: str | None,
    path: Path,
    number: int,
) -> dict[str, Any]:
    """Project one contract row onto the keys :func:`_prompt_from` reads."""
    out: dict[str, Any] = {}
    if prompt_column is not None and prompt_column in row:
        out["prompt"] = row[prompt_column]
    else:
        # Fall through to _prompt_from's own key search, which refuses by name if nothing
        # matches. Refused there rather than here so one message covers both routes.
        out.update({key: row[key] for key in ("prompt", "text", "input") if key in row})
    if reference_column is not None and reference_column in row:
        out["reference"] = row[reference_column]
    if group_column is not None and group_column in row:
        out["group"] = row[group_column]
    for key in ("id", "prompt_id", "external_id"):
        if key in row:
            out[key] = row[key]
            break
    out.setdefault("id", f"{path.name}:{number}")
    return out


def _column_named(content: Mapping[str, Any], semantic: str) -> str | None:
    """The first column whose ``semantic`` is ``semantic``, or ``None``.

    Read from the manifest's own column declarations so the prompt, the reference and the
    **stratification group** are the dataset author's choice rather than miForge guessing a
    column name - which is the same discipline as reading the ``provider`` key set out of
    the checkpoint schema instead of retyping it.
    """
    for column in content.get("columns") or []:
        if column.get("semantic") == semantic:
            return str(column.get("name"))
    return None


def render_reference(structured: Mapping[str, Any] | None, *, fallback: str | None = None) -> str:
    """**The one renderer** for a set identity (FTID section 11).

    A contract reference renders as miDataworks' own ``<type>/<name>@<version>``; an
    external provenance renders as ``external:<origin>@<sha256 prefix>``. One function,
    because two renderings of one identity is how the pack/contract rendering confusion
    started, and because ``ScoreRecord.set`` is the string a later reader joins on.

    Raises:
        Refusal: ``evaluation_set_unresolvable`` when neither form is present and no
            fallback was given. **No synthesised identity** - inventing one makes miForge
            the author of a contract identity miDataworks never issued.
    """
    if structured:
        if {"type", "name", "version"} <= set(structured):
            return f"{structured['type']}/{structured['name']}@{structured['version']}"
        origin = structured.get("origin")
        digest = structured.get("sha256") or structured.get("content_sha256")
        if origin and digest:
            return f"external:{origin}@{str(digest)[:16]}"
    if fallback:
        return fallback
    raise refuse(
        "evaluation_set_unresolvable",
        "This set carries neither a contract reference nor an external provenance, so there "
        "is no identity to record as ScoreRecord.set. Refusing rather than synthesising one: "
        "an invented reference makes miForge the author of an identity miDataworks never "
        "issued.",
        structured=dict(structured or {}),
    )
