"""Three refusals before a byte is read  (FR-002.5, FR-002.6, FR-002.7, FR-002.9).

``resolve`` answers *"can this part fill this source's batches?"* and refuses at each of
three layers rather than guessing:

1. **kind** - ``target.miforge_set_kind`` must be the kind the source reads, else
   ``part_kind_mismatch`` naming both;
2. **admissibility** - every split must be usable as training material, through **F10's
   one rule** (:func:`services.parts.reader.splits_for`), which refuses a held-out split
   **by name** rather than filtering it out;
3. **columns** - bound by declared semantic only (:mod:`training.signals.semantics`).

**Layer 2 is the one whose violation is invisible in training and fatal in evaluation.** A
held-out split used as training material makes every subsequent test score meaningless and
nothing in the loss curve says so. ``held_out`` and ``evaluation_only`` are **required
booleans** on the schema's ``Split`` - parsed, not assumed - so there is no unknown case to
fall through and **no ``getattr(split, "held_out", False)`` to write**. That line is the
most plausible-looking one in this feature and it is a named mutation control.

## F10 validates; F2 admits. No second validator and no second opinion

The manifest is parsed through F10's ``parse_and_validate`` and the splits through F10's
``splits_for``. ADR-010 gives miDataworks exactly one outbound call site and it is F10's;
this module performs no HTTP and imports no ``httpx``. The part's **columns** are read from
the manifest on disk because F10's index does not carry them - that is reading F10's
authority with F10's parser, not re-validating it with a second one.
"""

from __future__ import annotations

import csv
import json
from collections.abc import Iterator, Mapping, Sequence
from pathlib import Path
from typing import Any

from core.errors import refuse
from services.parts import ingest, reader
from services.parts.contract_validator import DatasetVersion, Split, parse_and_validate
from training.signals.base import PartSpec, ResolvedPart
from training.signals.semantics import resolve_columns

#: Which ``PartReference.kind`` each source reads. ``reward`` reads a **prompt set**:
#: ``target.miforge_set_kind`` has **no reward value** (verified by parsing the vendored
#: schema), so a reward run's inputs are a prompt set plus a grader configuration. The
#: contract's ``reward_bundle`` target is *not* consumed - it ships executable code from
#: another application and pins ``trl_version``, and TRL is not in ADR-002's stack
#: (FPRD §15 finding 3).
SOURCE_PART_KIND: Mapping[str, str] = {
    "corpus": "corpus",
    "self-generation": "prompt_set",
    "preference-pairs": "preference_pairs",
    "reward": "prompt_set",
}

#: Split file formats this release can read, by suffix. **Dispatched on the declared path's
#: suffix, never on the bytes**: sniffing content is the same guess as choosing a column by
#: its name, one layer down.
#:
#: ``.parquet`` is what miDataworks' export service writes, and reading it needs
#: ``pyarrow``, which is **not in ADR-002's dependency list**. ``pyproject.toml`` has one
#: owner and this feature does not edit it, so the import is lazy and its absence is a
#: **named refusal** that says which dependency is owed - not a crash, and not a silent
#: fall-back to another reader that would read the wrong bytes.
SUPPORTED_SPLIT_SUFFIXES: tuple[str, ...] = (".parquet", ".jsonl", ".ndjson", ".json", ".csv")


def resolve(
    part: Mapping[str, Any],
    spec: PartSpec,
    *,
    source: str,
    required_kind: str | None = None,
) -> ResolvedPart:
    """Resolve one part for ``source``, refusing at kind, admissibility or columns.

    Args:
        part: F10's row as ``services.parts.index.part_payload`` serves it.
        spec: What the run asked for - the named splits and any column bindings.
        source: One of the four canonical source names.
        required_kind: The kind this particular part must be, when it is not the source's
            default. ``preference-pairs`` is the one caller: its from-generations half
            reads a **prompt set** (R-02.6's second clause), while its file half reads a
            pairs part, and the two are one source with two materials.

    Returns:
        A :class:`ResolvedPart` carrying the usable splits, the column binding and the
        gate contract's ``PartReference`` object for the record.

    Raises:
        Refusal: ``part_kind_mismatch`` (E-2), F10's ``split_not_available_for_training``
            or ``no_training_splits`` (E-3), or a column refusal (E-4, E-12).
    """
    required = required_kind or SOURCE_PART_KIND[source]
    actual = str(part.get("part_kind"))
    if actual != required:
        raise refuse(
            "part_kind_mismatch",
            f"Source {source!r} reads a {required!r} part and this one is {actual!r}. "
            f"Refusing rather than reading it anyway: the kinds describe different material, "
            f"and a test set offered to a corpus would train on the very rows the run is "
            f"scored against.",
            source=source,
            required_kind=required,
            supplied_kind=actual,
            part_id=part.get("id"),
        )

    version = _manifest_of(part)
    # Layer 2, through F10's ONE rule. `named=` matters: a split asked for by name that is
    # held out is refused BY NAME, where an unnamed request simply takes the eligible ones.
    # Returning an empty list instead would become a zero-row batch - a step that trains on
    # nothing while every log line looks healthy.
    usable: list[Split] = reader.splits_for(
        version.splits, "train", named=list(spec.splits) or None
    )

    # The column policy follows the KIND that was read, not the source's default: the
    # from-generations half of `preference-pairs` reads a prompt set, which declares
    # prompts and no chosen/rejected columns at all.
    column_source = "self-generation" if required == "prompt_set" else source
    binding = resolve_columns(
        _columns_of(version),
        column_source,
        configured=spec.columns,
        part_kind=actual,
    )

    return ResolvedPart(
        payload=part,
        kind=actual,
        splits=tuple(_split_payload(split) for split in usable),
        binding=binding,
        reference=part_reference(part, kind=actual),
    )


def part_reference(part: Mapping[str, Any], *, kind: str) -> dict[str, Any]:
    """The gate contract's ``PartReference`` for a part. **Exactly one of the two forms.**

    ``contract_reference`` and ``external`` are both in the contract's ``required`` list and
    both nullable, so the record states which form this part has rather than leaving the
    reader to infer it from an absent key. An **external part carries null contract fields**
    - not an empty string and not a synthesised reference (R-02.40, FR-002.31): a batch
    record never claims miDataworks provenance for anything miDataworks did not publish.

    Raises:
        Refusal: ``part_provenance_absent`` when a part has neither form. The schema
            permits a conformant version with no contract reference, and F10 records that
            null faithfully - but a part with no provenance of **either** kind makes the run
            unreproducible and nothing downstream could say where a behaviour came from.
    """
    reference = part.get("contract_reference")
    external = part.get("external_provenance")
    if reference is None and external is None:
        raise refuse(
            "part_provenance_absent",
            f"Part {part.get('id')!r} carries neither a miDataworks contract reference nor "
            f"external provenance, so a batch drawn from it could not say where its material "
            f"came from. Re-import it from a miDataworks export, or import it as an external "
            f"part with its own source, digest and retrieval time.",
            part_id=part.get("id"),
        )
    return {
        "kind": kind,
        "contract_reference": None if reference is None else dict(reference),
        "external": None if external is None else dict(external),
    }


def _manifest_of(part: Mapping[str, Any]) -> DatasetVersion:
    """The part's manifest, parsed through **F10's** validator.

    Read from disk because ``mf_contract_parts`` does not carry the column declarations and
    a batch cannot be bound without them. The authority for a part is
    ``<part_dir>/midataworks-dataset-version.json`` (ADR-004, F10's own words), so this
    reads the authority with the owner's parser - it is not a second validator with a second
    opinion about conformance.
    """
    directory = Path(str(part.get("local_path") or ""))
    if not directory.is_dir():
        raise refuse(
            "part_directory_missing",
            f"Part {part.get('id')!r} is indexed at {directory} and that directory is not "
            f"there. The index is a rebuildable projection and the filesystem is the truth "
            f"(ADR-004): re-import the version, or run reindex to drop the stale row.",
            part_id=part.get("id"),
            path=str(directory),
        )
    version, _digest = parse_and_validate(ingest.read_manifest(directory), source=str(directory))
    return version


def _columns_of(version: DatasetVersion) -> list[Mapping[str, Any]]:
    """``content.columns``, or a refusal for a content kind that declares none.

    Read off ``raw`` deliberately: F10's typed view does not name ``columns``, and its own
    docstring says ``raw`` *"is the whole document, so nothing downstream has to come back
    here for a field this view does not name."*
    """
    content = version.raw.get("content") or {}
    columns = content.get("columns")
    if not isinstance(columns, list) or not columns:
        raise refuse(
            "part_declares_no_columns",
            f"This part's content kind is {version.content_kind!r} and it declares no "
            f"columns, so nothing can be bound to a training role. A signal source needs a "
            f"'rows' version with declared column semantics.",
            content_kind=version.content_kind,
        )
    return [column for column in columns if isinstance(column, Mapping)]


def _split_payload(split: Split) -> dict[str, Any]:
    """One split as this feature carries it.

    **The two admissibility flags are deliberately NOT copied here.** They are F10's
    (``services/parts/reader.py``), the splits in this payload have already been through
    its one rule, and a module that holds both flags together is a module that could decide
    eligibility a second time - which is what F10's own structural test looks for and
    refuses by name. Carrying them "for display" is how the second decision site gets
    written six months later.
    """
    return {
        "name": split.name,
        "path": split.path,
        "rows": split.rows,
        "sha256": split.sha256,
        "logical_digest": split.logical_digest,
    }


# ======================================================================== reading rows


def row_identifier(part_id: str, split: str, index: int) -> str:
    """The identifier a prompt or example is recorded under.

    ``<part id>:<split>:<row index>`` - deterministic, and **resolvable against the part it
    came from**, which is what ``batch-record-v1.json`` asks of ``prompt_ids``. It is derived
    rather than read from a column because no column semantic in the contract declares a row
    key: ``id`` is a *name*, and binding a role to a name is the guess this feature exists to
    refuse. If miDataworks later publishes row keys as a declared semantic, this is the one
    function that changes.
    """
    return f"{part_id}:{split}:{index}"


def read_rows(
    part: ResolvedPart, *, columns: Sequence[str] | None = None, limit: int | None = None
) -> Iterator[tuple[str, Mapping[str, Any]]]:
    """Every row of every usable split, as ``(row identifier, row)``.

    Lazily, split by split, in the manifest's order - so a batch takes the rows it needs
    rather than materialising a table to take one column (ADR-024).

    Raises:
        Refusal: ``split_format_unsupported`` for a suffix this release cannot read, or
            ``parquet_reader_unavailable`` when a Parquet split is asked for without
            ``pyarrow`` installed.
    """
    part_id = str(part.payload.get("id"))
    base = Path(str(part.payload.get("local_path")))
    produced = 0
    wanted = list(columns) if columns is not None else None
    for split in part.splits:
        # `confine` is F10's traversal guard: a manifest-declared path is untrusted input,
        # and `../../etc` in a `path` field must not escape the part's directory.
        path = ingest.confine(base, str(split["path"]))
        for index, row in enumerate(_read_file(path, wanted)):
            yield row_identifier(part_id, str(split["name"]), index), row
            produced += 1
            if limit is not None and produced >= limit:
                return


def _read_file(path: Path, columns: list[str] | None) -> Iterator[Mapping[str, Any]]:
    """One split file's rows, dispatched on the declared suffix."""
    suffix = path.suffix.lower()
    if suffix not in SUPPORTED_SPLIT_SUFFIXES:
        raise refuse(
            "split_format_unsupported",
            f"{path.name} has suffix {suffix!r}, which this release cannot read. Supported: "
            f"{list(SUPPORTED_SPLIT_SUFFIXES)}. Refusing rather than guessing the format "
            f"from the bytes: a misread split is a batch of the wrong material, and the "
            f"batch record would describe it as though it were right.",
            path=str(path),
            suffix=suffix,
            supported=list(SUPPORTED_SPLIT_SUFFIXES),
        )
    if not path.is_file():
        raise refuse(
            "split_file_missing",
            f"{path} is declared by the part's manifest and is not on disk. F10 verifies a "
            f"split's digest at import, so a missing file means the part directory changed "
            f"after it was imported; re-import it or reindex.",
            path=str(path),
        )
    if suffix == ".parquet":
        yield from _read_parquet(path, columns)
    elif suffix == ".csv":
        with path.open("r", encoding="utf-8", newline="") as handle:
            yield from csv.DictReader(handle)
    elif suffix == ".json":
        document = json.loads(path.read_text("utf-8"))
        if not isinstance(document, list):
            raise refuse(
                "split_format_unsupported",
                f"{path.name} is a .json split and holds {type(document).__name__}, not an "
                f"array of row objects.",
                path=str(path),
            )
        yield from (row for row in document if isinstance(row, Mapping))
    else:
        with path.open("r", encoding="utf-8") as handle:
            for number, line in enumerate(handle, start=1):
                stripped = line.strip()
                if not stripped:
                    continue
                try:
                    row = json.loads(stripped)
                except ValueError as exc:
                    # Refused, not skipped. A skipped line makes a split read as shorter
                    # than its manifest says, and the batch record would carry a row count
                    # the part contradicts.
                    raise refuse(
                        "split_corrupt",
                        f"{path.name} line {number} is not valid JSON: {exc}. Refusing "
                        f"rather than skipping it - a skipped row makes the split read "
                        f"shorter than the manifest it was verified against.",
                        path=str(path),
                        line=number,
                    ) from exc
                if isinstance(row, Mapping):
                    yield row


def _read_parquet(path: Path, columns: list[str] | None) -> Iterator[Mapping[str, Any]]:
    """A Parquet split, **resolved columns only**, through ``pyarrow`` imported lazily.

    Lazy because ``pyarrow`` is not in ADR-002's list and ``pyproject.toml`` has one owner:
    the dependency is **reported as owed**, and until it is installed a Parquet split
    refuses by name instead of failing with an ``ImportError`` nobody can act on.
    """
    try:
        import pyarrow.parquet as parquet
    except ImportError as exc:
        raise refuse(
            "parquet_reader_unavailable",
            "This split is Parquet - the format miDataworks' export service writes - and "
            "pyarrow is not installed. It is the one dependency this feature needs beyond "
            "ADR-002's list, and it is owed in pyproject.toml. A JSON-lines or CSV export "
            "reads without it.",
            path=str(path),
            dependency="pyarrow",
        ) from exc
    table = parquet.read_table(path, columns=columns)
    yield from table.to_pylist()
