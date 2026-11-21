"""The only way a part comes in  (F10 tasks 3.1-3.3; FR-010.9, FR-010.11, FR-010.12).

``midataworks.dataset-version/v1`` is **miDataworks'** contract (BRD-03 R-03.47). miForge
vendors ``docs/schemas/midataworks-dataset-version-v1.json`` byte-identical and **never
edits it**; the file's own header says so. A needed change is *requested* of miDataworks,
and nothing in this feature can change it.

## The schema validates; a model only reads

A Pydantic model of this manifest would be a **second description of someone else's
contract living in miForge** - exactly the restatement ADR-014 forbids and the mechanism by
which the two BRDs' manifest field lists came to disagree in the first place. So:

* the **vendored JSON Schema is the validator** (``jsonschema``, draft 2020-12, constructed
  for that draft **explicitly** rather than by auto-detection, so a future ``$schema``
  change cannot silently fall back to a laxer draft);
* :class:`DatasetVersion` is a typed **read view** over an already-validated document. It
  extracts the fields F10 uses and **cannot reject anything the schema accepts**. The
  moment it rejects something, miForge has a second contract.

## The path is the one the sync test checks

:data:`SCHEMA_PATH` is the file ``backend/tests/contract/test_schema_sync.py`` compares
against miDataworks' normative copy, and a test asserts those two paths are the same
object. A validator reading an embedded copy would pass while the vendored file drifted -
the exact failure the sync test exists to prevent, routed around.

## All the failures, not the first

An operator fixing an export wants every problem at once. :func:`validate` collects
``iter_errors`` sorted by instance path, and each entry carries the failing **JSON
Pointer**, the failing keyword, the schema path and the expected value where the schema
states one - taken from ``jsonschema``'s own ``json_path``, ``validator``,
``validator_value`` and ``absolute_schema_path`` rather than parsed out of the message
string.
"""

from __future__ import annotations

import json
import os
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any

import jsonschema

from core.errors import Refusal, refuse

#: The **instance** filename miDataworks writes beside the split files. Note there is no
#: ``-v1``: ``MANIFEST_FILE = "midataworks-dataset-version.json"`` in their
#: ``services/publishing/manifest_builder.py``, while ``-v1`` is the *schema* filename.
#: Verified by reading that module, not inferred from the schema's name. Hard-coding the
#: wrong one fails only at integration time, when a real export is in front of you.
MANIFEST_FILENAME = "midataworks-dataset-version.json"

#: The vendored schema's filename. Different from :data:`MANIFEST_FILENAME` on purpose;
#: see above.
SCHEMA_FILENAME = "midataworks-dataset-version-v1.json"

#: Where the vendored contracts are. Resolved by walking up from this file, with the same
#: environment override ``services/reindex.py`` uses and for the same reason: a hard-coded
#: absolute path works on one workstation.
_SCHEMA_DIR_ENV = "MIFORGE_SCHEMA_DIR"
_DEFAULT_SCHEMA_DIR = Path(__file__).resolve().parents[4] / "docs" / "schemas"


def schema_dir() -> Path:
    return Path(os.environ.get(_SCHEMA_DIR_ENV, str(_DEFAULT_SCHEMA_DIR)))


def schema_path() -> Path:
    """The vendored schema file. **The same path the sync test checks** - asserted."""
    return schema_dir() / SCHEMA_FILENAME


@lru_cache(maxsize=1)
def load_schema() -> Mapping[str, Any]:
    """The vendored schema, parsed once.

    Parsed, not described. This module's job is to agree with a file, and the only way to
    agree with a file is to read it.

    Raises:
        Refusal: ``schema_not_vendored`` when the file is absent, naming where it looked.
    """
    path = schema_path()
    if not path.is_file():
        raise refuse(
            "schema_not_vendored",
            f"{SCHEMA_FILENAME} is not present in {schema_dir()}. miDataworks owns this "
            f"contract and miForge vendors it byte-identical; set "
            f"{_SCHEMA_DIR_ENV} if the contracts live elsewhere in this deployment.",
            schema=SCHEMA_FILENAME,
            searched=str(schema_dir()),
        )
    parsed: Mapping[str, Any] = json.loads(path.read_text("utf-8"))
    return parsed


@lru_cache(maxsize=1)
def validator() -> jsonschema.Draft202012Validator:
    """The compiled validator, built once at first use.

    Compiled once because per-request compilation of a 1,600-line schema is the common
    case being slow for nothing. **Draft 2020-12 explicitly**, matching the ``$schema``
    the vendored file declares: ``validator_for`` would auto-detect, and auto-detection is
    a silent downgrade waiting for a day when the declaration moves.
    """
    return jsonschema.Draft202012Validator(dict(load_schema()))


def enum_at(*pointer: str) -> frozenset[str]:
    """The ``enum`` values at a ``$defs`` pointer in the **vendored** schema.

    The single most important helper in this package. Reading the enums from the file
    rather than re-typing them in Python is what makes an upstream addition a loud startup
    failure instead of a value that falls through a default (ADR-018, and the F10 TDD's
    T-2).

    Args:
        *pointer: Keys to walk, e.g. ``("$defs", "Target", "properties", "kind")``.

    Returns:
        The enum's members.

    Raises:
        Refusal: ``schema_shape_unexpected`` when the pointer does not lead to an enum.
            Refused rather than defaulted to an empty set, because an empty set would
            make every exhaustiveness check below pass vacuously - the construction-green
            failure this project keeps finding.
    """
    node: Any = load_schema()
    walked: list[str] = []
    for key in pointer:
        walked.append(key)
        if not isinstance(node, Mapping) or key not in node:
            raise refuse(
                "schema_shape_unexpected",
                f"{SCHEMA_FILENAME} has no node at /{'/'.join(walked)}. The vendored "
                f"contract's shape changed; re-read it rather than guessing.",
                pointer="/".join(pointer),
            )
        node = node[key]

    # `anyOf [{enum: [...]}, {type: null}]` is how their generator renders an optional
    # enum. Both shapes are walked here so a caller never has to know which one a field
    # happens to use.
    candidates: list[Any] = [node]
    if isinstance(node, Mapping) and isinstance(node.get("anyOf"), list):
        candidates.extend(node["anyOf"])
    for candidate in candidates:
        if isinstance(candidate, Mapping) and isinstance(candidate.get("enum"), list):
            return frozenset(str(value) for value in candidate["enum"])

    raise refuse(
        "schema_shape_unexpected",
        f"{SCHEMA_FILENAME} declares no enum at /{'/'.join(pointer)}; this package reads "
        f"its vocabularies from the file and will not invent one.",
        pointer="/".join(pointer),
    )


# ====================================================================== refusal detail


@dataclass(frozen=True, slots=True)
class SchemaFailure:
    """One way a candidate manifest fails the contract.

    Every field here comes from ``jsonschema``'s structured attributes. The message string
    is carried for a human and is never parsed.
    """

    instance_path: str
    schema_keyword: str
    schema_path: str
    expected: Any
    message: str

    def as_dict(self) -> dict[str, Any]:
        return {
            "instance_path": self.instance_path,
            "schema_keyword": self.schema_keyword,
            "schema_path": self.schema_path,
            "expected": self.expected,
            "message": self.message,
        }


def _pointer(error: jsonschema.ValidationError) -> str:
    """The failing instance location as an RFC 6901 JSON Pointer.

    ``jsonschema``'s ``json_path`` renders ``$.content.splits[0].sha256``, which is a
    different notation from the one the schema's own ``$ref``s use and from the one an
    operator's editor understands. A pointer is built from ``absolute_path`` instead, so
    the refusal says ``/content/splits/0/sha256``.
    """
    if not error.absolute_path:
        return ""
    return "/" + "/".join(str(part) for part in error.absolute_path)


def _expected(error: jsonschema.ValidationError) -> Any:
    """What the schema said, where it says something quotable."""
    keyword = str(error.validator)
    if keyword in {
        "const",
        "enum",
        "type",
        "pattern",
        "minimum",
        "maximum",
        "minItems",
        "minLength",
        "maxLength",
        "required",
        "additionalProperties",
    }:
        return error.validator_value
    return None


#: Properties that discriminate a ``oneOf`` branch in this contract: ``content`` on
#: ``content_kind``, ``publication`` on ``state``, ``sources[].origin`` on ``kind``. Read
#: from the schema would be better; they are listed because ``discriminator`` is an OpenAPI
#: extension that ``jsonschema`` does not interpret, so there is nothing to read *through*.
#: A test asserts this list against the schema's own ``discriminator.propertyName`` values,
#: which is what stops it going stale.
_DISCRIMINATORS: frozenset[str] = frozenset({"content_kind", "state", "kind"})


def _descend(error: jsonschema.ValidationError) -> list[jsonschema.ValidationError]:
    """Expand a ``oneOf``/``anyOf`` failure into the failures that actually matter.

    **Without this, every failure inside ``content`` collapses into one error.** The
    contract's ``content`` is a discriminated ``oneOf`` over ``rows``/``config``/
    ``reward_bundle``, so a missing ``sha256`` on one split reports as *"{the entire content
    object} is not valid under any of the given schemas"* at ``/content`` - naming neither
    the split nor the field, and dumping a page of JSON into the message. That refusal
    cannot be acted on, which is precisely what FR-010.12 forbids.

    So the branch the document *meant* is identified by its discriminator - the branch whose
    own sub-errors are not about ``content_kind`` being the wrong constant - and its
    sub-errors are reported instead. If no branch matches the discriminator, the original
    error is kept: the document named a content kind that does not exist, and *that* is the
    useful thing to say.
    """
    if str(error.validator) not in {"oneOf", "anyOf"} or not error.context:
        return [error]

    branches: dict[int, list[jsonschema.ValidationError]] = {}
    for sub in error.context:
        index = sub.schema_path[0] if sub.schema_path else 0
        branches.setdefault(int(index), []).append(sub)

    def fails_on_discriminator(errors: list[jsonschema.ValidationError]) -> bool:
        return any(
            str(sub.validator) == "const" and sub.path and str(sub.path[-1]) in _DISCRIMINATORS
            for sub in errors
        )

    candidates = {
        index: errors for index, errors in branches.items() if not fails_on_discriminator(errors)
    }
    if not candidates:
        # Every branch was rejected on its discriminator: the document names a content kind
        # the contract does not define. The original message says exactly that.
        return [error]

    # The branch with the fewest complaints is the one the document was trying to be. With
    # a discriminator present there is normally exactly one candidate; the tie-break matters
    # only for the `publication` union, where both branches share a `state` property.
    chosen = min(candidates.values(), key=len)
    expanded: list[jsonschema.ValidationError] = []
    for sub in chosen:
        expanded.extend(_descend(sub))
    return expanded


def failures(document: Any) -> list[SchemaFailure]:
    """Every way ``document`` fails the vendored contract, sorted by instance path.

    Sorted so two runs over one bad manifest report the same order - a refusal whose
    ``details`` reorder between calls is a refusal nobody can diff.
    """
    collected: list[jsonschema.ValidationError] = []
    for error in validator().iter_errors(document):
        collected.extend(_descend(error))
    errors = sorted(collected, key=lambda e: (list(e.absolute_path), str(e.validator), e.message))
    return [
        SchemaFailure(
            instance_path=_pointer(error),
            schema_keyword=str(error.validator),
            schema_path="/" + "/".join(str(part) for part in error.absolute_schema_path),
            expected=_expected(error),
            message=error.message,
        )
        for error in errors
    ]


def invalid_refusal(found: Sequence[SchemaFailure], *, source: str) -> Refusal:
    """The ``dataset_version_invalid`` refusal. **The refusal is the product** (FR-010.12).

    The message names the first failure in full, because that is what fits in a toast and
    is usually the only one that matters; ``details['errors']`` carries all of them, so the
    operator can fix the export from the refusal alone without opening this document.
    """
    first = found[0]
    where = first.instance_path or "<root>"
    return refuse(
        "dataset_version_invalid",
        f"This is not a conformant `midataworks.dataset-version/v1` document. "
        f"`{where}` fails `{first.schema_keyword}`: {first.message}"
        f"{'' if len(found) == 1 else f' (and {len(found) - 1} more)'}",
        source=source,
        schema=SCHEMA_FILENAME,
        error_count=len(found),
        errors=[failure.as_dict() for failure in found],
    )


# ====================================================================== the read view


@dataclass(frozen=True, slots=True)
class ContractReference:
    """miDataworks' contract form for a part: ``{type, name, version}``.

    **Their shape, not restated or extended.** ``<type>/<name>@<version>`` is the
    contract's rendering; ``<pack>/<type>/<name>@<version>`` is F7 prefixing *its own* pack
    identity for **display only**. Two layers, and conflating them is how a cross-repo
    handoff was once recorded for something neither side had agreed to build.

    The same three fields are the ``ContractReference`` of
    ``docs/schemas/batch-record-v1.json``; this is the parsed form of the manifest's copy,
    and :meth:`as_dict` produces exactly that contract's object.
    """

    type: str
    name: str
    version: str

    def as_dict(self) -> dict[str, str]:
        """The gate contract's object, key for key (``batch-record-v1.json``)."""
        return {"type": self.type, "name": self.name, "version": self.version}

    def render(self) -> str:
        """``<type>/<name>@<version>`` - the **inner** layer only. F7 adds its prefix."""
        return f"{self.type}/{self.name}@{self.version}"


@dataclass(frozen=True, slots=True)
class Split:
    """One split file as the manifest describes it.

    ``held_out`` and ``evaluation_only`` are **required booleans** on the schema's
    ``Split`` (parsed, not assumed), which is what gives
    :func:`services.parts.reader.splits_for` something to read rather than a field it has
    to default.
    """

    name: str
    path: str
    rows: int
    bytes: int
    sha256: str
    logical_digest: str
    held_out: bool
    evaluation_only: bool
    label_counts: Mapping[str, int]

    @property
    def training_eligible(self) -> bool:
        """Whether this split may fill a training batch. See :mod:`services.parts.reader`."""
        return not (self.held_out or self.evaluation_only)


@dataclass(frozen=True, slots=True)
class ConfigBody:
    """A selector or grader configuration, as a part.

    miForge **authors none of these** (R-02.9 as amended; BRD-03 R-03.48 owns authoring)
    and this feature ships no editor for them. F10 validates, resolves ``path``, verifies
    ``sha256`` and hands F3 the body and the hash.
    """

    kind: str
    name: str
    number: int
    plugin: str
    body_format: str
    sha256: str
    path: str


@dataclass(frozen=True, slots=True)
class Caveat:
    """Something a reader must know. Carried **verbatim**, never summarised.

    An ``amber`` caveat does not refuse the part - grading it is miDataworks' judgement and
    accepting it is the operator's - but it must be visible **at the point of use**, not
    only at import. Dropping one is how a shortcut warning becomes a silent training
    defect, and a dropped licence provenance is one of this project's recorded defects of
    exactly this class.
    """

    code: str
    severity: str
    message: str
    detail: Mapping[str, Any]

    def as_dict(self) -> dict[str, Any]:
        return {
            "code": self.code,
            "severity": self.severity,
            "message": self.message,
            "detail": dict(self.detail),
        }


@dataclass(frozen=True, slots=True)
class DatasetVersion:
    """A validated ``midataworks.dataset-version/v1`` document, as F10 reads it.

    **Cannot reject anything the schema accepts.** Built only by :func:`validate`, which
    has already run the schema; every field here is read out of a document that passed. If
    this class ever grows a check of its own, miForge has a second contract (TDD T-1).

    ``raw`` is the whole document, so nothing downstream has to come back here for a field
    this view does not name - and ``extensions`` in particular is passed through
    **unparsed**, because it is miDataworks' extension point and reading it would be
    miForge building on an unversioned surface.
    """

    raw: Mapping[str, Any]

    # -------------------------------------------------------------- version identity
    @property
    def dataset(self) -> str:
        return str(self.raw["version"]["dataset"])

    @property
    def version_id(self) -> str:
        return str(self.raw["version"]["version_id"])

    @property
    def version_number(self) -> int:
        return int(self.raw["version"]["number"])

    @property
    def recipe_sha256(self) -> str:
        return str(self.raw["version"]["recipe"]["sha256"])

    # -------------------------------------------------------------- target
    @property
    def target_kind(self) -> str:
        return str(self.raw["target"]["kind"])

    @property
    def miforge_set_kind(self) -> str | None:
        value = self.raw["target"]["miforge_set_kind"]
        return None if value is None else str(value)

    @property
    def contract_reference(self) -> ContractReference | None:
        """The contract reference, **or ``None``, which is conformant.**

        ``Target.contract_reference`` is ``anyOf [ContractReference, null]`` *and* is in
        ``Target.required`` - parsed from the vendored file, not taken on trust. So "no
        reference" is an assertion miDataworks deliberately allows, and refusing it here
        would refuse a conformant version.

        **Nothing may synthesise one.** Not from the version id, not from the dataset
        name, not from the directory name. Inventing one makes miForge the author of a
        contract identity miDataworks never issued (FR-010.5, FR-010.15). The refusal
        belongs one layer up: **F7** rejects such a part at pack-slot binding with
        ``part_unpinnable``, because a pack version's guarantee is that it names a
        reproducible run input.
        """
        value = self.raw["target"].get("contract_reference")
        if value is None:
            return None
        return ContractReference(
            type=str(value["type"]), name=str(value["name"]), version=str(value["version"])
        )

    # -------------------------------------------------------------- content
    @property
    def content_kind(self) -> str:
        return str(self.raw["content"]["content_kind"])

    @property
    def splits(self) -> tuple[Split, ...]:
        """The split files, or an empty tuple for a non-``rows`` content kind."""
        if self.content_kind != "rows":
            return ()
        return tuple(
            Split(
                name=str(entry["name"]),
                path=str(entry["path"]),
                rows=int(entry["rows"]),
                bytes=int(entry["bytes"]),
                sha256=str(entry["sha256"]),
                logical_digest=str(entry["logical_digest"]),
                held_out=bool(entry["held_out"]),
                evaluation_only=bool(entry["evaluation_only"]),
                label_counts=dict(entry.get("label_counts") or {}),
            )
            for entry in self.raw["content"]["splits"]
        )

    @property
    def config(self) -> ConfigBody | None:
        """The selector/grader configuration, for a ``config`` content kind."""
        if self.content_kind != "config":
            return None
        body = self.raw["content"]["config"]
        return ConfigBody(
            kind=str(body["kind"]),
            name=str(body["name"]),
            number=int(body["number"]),
            plugin=str(body["plugin"]),
            body_format=str(body["body_format"]),
            sha256=str(body["sha256"]),
            path=str(body["path"]),
        )

    # -------------------------------------------------------------- provenance
    @property
    def caveats(self) -> tuple[Caveat, ...]:
        return tuple(
            Caveat(
                code=str(entry["code"]),
                severity=str(entry["severity"]),
                message=str(entry["message"]),
                detail=dict(entry.get("detail") or {}),
            )
            for entry in self.raw["caveats"]
        )

    @property
    def licence_classes(self) -> tuple[str, ...]:
        """Every licence class across ``sources[]``, sorted and de-duplicated.

        Carried because **a dropped licence provenance is one of this project's three
        recorded defects of this class.** ``forbids_redistribution`` and ``private_only``
        are informational for miForge, which does not redistribute parts - but
        informational is not the same as absent, and only a present value lets an operator
        answer the question later.
        """
        return tuple(
            sorted({str(source["licence"]["licence_class"]) for source in self.raw["sources"]})
        )

    @property
    def licences(self) -> tuple[Mapping[str, Any], ...]:
        """Each source's licence block verbatim: displayed, raw, origin, class, table."""
        return tuple(dict(source["licence"]) for source in self.raw["sources"])

    @property
    def extensions(self) -> Mapping[str, Any]:
        """miDataworks' extension point. **Passed through, never parsed.**"""
        return dict(self.raw["extensions"])

    @property
    def publication(self) -> Mapping[str, Any] | None:
        value = self.raw.get("publication")
        return None if value is None else dict(value)


def validate(document: Any, *, source: str) -> DatasetVersion:
    """Validate a candidate manifest and return the typed read view.

    **Every path that takes a part in goes through here.** There is no trusted path: a
    version that fails the contract is a defect wherever it came from, and a bypass for
    "parts we fetched ourselves" is how an invalid part gets a part row.

    Args:
        document: The parsed manifest.
        source: Where it came from, for the refusal's ``details`` - a directory, an export
            id, a Hub repo. An operator with three imports in flight needs to know which
            one refused.

    Returns:
        The read view over the validated document.

    Raises:
        Refusal: ``dataset_version_invalid`` carrying every failure, each with its JSON
            Pointer, keyword, schema path and expected value.
    """
    found = failures(document)
    if found:
        raise invalid_refusal(found, source=source)
    return DatasetVersion(raw=document)


def parse_and_validate(manifest_bytes: bytes, *, source: str) -> tuple[DatasetVersion, str]:
    """Validate manifest **bytes** and return the view plus the digest of those bytes.

    The digest is of the bytes **as received** (FR-010.14, FR-010.7's rule in the FTID's
    §7). Re-serialising first would digest miForge's serialisation rather than what
    miDataworks sent, which makes the value useless as a cross-repo check - and that value
    is the only thing that lets two repositories agree they are looking at one manifest.

    Raises:
        Refusal: ``dataset_version_not_json`` when the bytes are not JSON (named, rather
            than a ``JSONDecodeError`` reaching the application's last-resort handler as a
            500); ``dataset_version_invalid`` when they do not conform.
    """
    import hashlib

    try:
        document = json.loads(manifest_bytes.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise refuse(
            "dataset_version_not_json",
            f"{source} does not contain JSON: {exc}. A `midataworks.dataset-version/v1` "
            f"manifest is a JSON document; nothing is guessed from the bytes.",
            source=source,
        ) from exc
    return validate(document, source=source), hashlib.sha256(manifest_bytes).hexdigest()
