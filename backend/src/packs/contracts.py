"""The vendored contracts this feature reads  (ADR-014; FR-007.7, FR-007.32).

**Parsed, never described.** Three files matter to F7 and not one of them is F7's to write:

* ``pack-definition-v1.json`` - the pack artifact (foundation task 3.4). F7 validates
  against it on write, export and import.
* ``batch-record-v1.json`` - whose ``$defs.PartReference`` **settles the shape of the
  element a ``parts`` array carries.** See :mod:`packs.records` for why that matters here.
* ``adapter-manifest-v1.json`` - whose ``parts`` items F8 writes from F7's producer.

Reading the enums and subschemas out of the files, rather than re-typing them in Python, is
what makes an upstream change a **loud failure at import** instead of a value that falls
through a default (ADR-018; the same discipline as ``services/parts/kind_map.py``). A
hard-coded Python copy would simply not know.

**Nothing here edits a schema**, and nothing in this feature may: ``docs/schemas/`` is
single-owner shared state, ``adapter-manifest-v1.json`` is owed byte-identical vendoring
into miLLM under BRD-01 R-01.28, and an edit here would change a contract another repository
is waiting to take a copy of.
"""

from __future__ import annotations

import json
import os
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from core.errors import refuse

#: ``docs/schemas/``, found relative to this file rather than from configuration, so the
#: contracts travel with the repository exactly as F1's run log and F2's batch record find
#: theirs. It is the one ``docs/`` path published to the public mirror, because these are
#: public contracts.
#: ``MIFORGE_SCHEMA_DIR`` overrides it, and **that override is what makes the backend image
#: able to start.** The default below is correct when the package is run from the repository
#: and WRONG the moment it is pip-installed: ``parents[N]`` counts up from this file, so from
#: ``/usr/local/lib/python3.13/site-packages/`` it lands on ``/usr/local/lib/docs/schemas``,
#: which does not exist. The image puts the contracts at ``/app/docs/schemas`` and sets the
#: variable; eight modules in this tree compute this directory independently and five already
#: honoured the variable, so this is adopting the existing convention rather than inventing
#: one. Found by running the image, not by reading it: the container crashed on import with
#: ``schema_not_vendored`` naming ``/usr/local/lib/docs/schemas``.
_SCHEMA_DIR_ENV = "MIFORGE_SCHEMA_DIR"
_DEFAULT_SCHEMA_DIR = Path(__file__).resolve().parents[3] / "docs" / "schemas"


def schema_dir() -> Path:
    """The vendored contract directory, honouring ``MIFORGE_SCHEMA_DIR``."""
    return Path(os.environ.get(_SCHEMA_DIR_ENV, str(_DEFAULT_SCHEMA_DIR)))


#: Kept as a module attribute because tests and sibling modules read it. It is the
#: resolved value at import time; :func:`schema_dir` is the live one.
SCHEMA_DIR = schema_dir()

PACK_DEFINITION_SCHEMA = "pack-definition-v1.json"
BATCH_RECORD_SCHEMA = "batch-record-v1.json"
ADAPTER_MANIFEST_SCHEMA = "adapter-manifest-v1.json"
MIDATAWORKS_SCHEMA = "midataworks-dataset-version-v1.json"

_cache: dict[str, Mapping[str, Any]] = {}


def load(name: str) -> Mapping[str, Any]:
    """One vendored schema, parsed once.

    Raises:
        Refusal: ``schema_not_vendored`` when the file is absent. Refused rather than
            defaulted to an empty schema, which would make every conformance check below
            pass **vacuously** - the construction-green failure this project keeps finding
            (a review harness once passed against a tree containing zero documents).
    """
    if name not in _cache:
        path = schema_dir() / name
        if not path.is_file():
            raise refuse(
                "schema_not_vendored",
                f"{name} is not present at {path}. It is a gate contract and this feature "
                f"validates against it rather than describing it; without the file there "
                f"is nothing to agree with.",
                schema=name,
                searched=str(path),
            )
        _cache[name] = json.loads(path.read_text("utf-8"))
    return _cache[name]


def at(name: str, *pointer: str) -> Mapping[str, Any]:
    """The subschema at a key path inside ``name``.

    Args:
        name: A vendored schema's filename.
        *pointer: Keys to walk, e.g. ``("$defs", "PartReference")``.

    Raises:
        Refusal: ``schema_shape_unexpected`` when the path does not lead to an object.
            A missing pointer means the contract moved, and silently returning ``{}``
            would turn a moved contract into a check that always passes.
    """
    node: Any = load(name)
    walked: list[str] = []
    for key in pointer:
        walked.append(key)
        if not isinstance(node, Mapping) or key not in node:
            raise refuse(
                "schema_shape_unexpected",
                f"{name} has no {'/'.join(walked)}. The contract has moved and the code "
                f"reading it has not; refusing rather than treating the absence as an "
                f"empty constraint.",
                schema=name,
                pointer=list(pointer),
            )
        node = node[key]
    if not isinstance(node, Mapping):
        raise refuse(
            "schema_shape_unexpected",
            f"{name}'s {'/'.join(pointer)} is not a schema object.",
            schema=name,
            pointer=list(pointer),
        )
    return node


def enum_at(name: str, *pointer: str) -> frozenset[str]:
    """The ``enum`` members at a key path, as a set.

    Raises:
        Refusal: ``schema_shape_unexpected`` when the path leads to something with no
            ``enum``. **Never an empty set on failure** - an empty set makes an
            exhaustiveness check pass vacuously.
    """
    node = at(name, *pointer)
    # `anyOf [{enum: [...]}, {type: null}]` is how a nullable enum renders - miDataworks'
    # `miforge_set_kind` is exactly that shape. Both forms are walked here so a caller never
    # has to know which one a field happens to use, and so a nullable enum is not mistaken
    # for a field with no vocabulary at all.
    candidates: list[Any] = [node]
    branches = node.get("anyOf")
    if isinstance(branches, list):
        candidates.extend(branches)
    for candidate in candidates:
        members = candidate.get("enum") if isinstance(candidate, Mapping) else None
        if isinstance(members, list) and members:
            return frozenset(str(member) for member in members)
    raise refuse(
        "schema_shape_unexpected",
        f"{name}'s {'/'.join(pointer)} carries no non-empty enum, so there is nothing "
        f"to check exhaustiveness against. Refusing rather than returning an empty set, "
        f"which would make every exhaustiveness check pass vacuously.",
        schema=name,
        pointer=list(pointer),
    )


_validators: dict[str, Any] = {}


def validator(name: str) -> Any:
    """A compiled validator for ``name``, with **cross-file ``$ref`` resolved locally**.

    ``pack-definition-v1.json`` is the first vendored contract in this repository to
    ``$ref`` *another file* - its slots reference
    ``batch-record-v1.json#/$defs/ContractReference`` and ``#/$defs/ExternalProvenance``,
    which is right on the merits (the element shape is settled there and must not be
    restated) and means a plain ``Draft202012Validator`` tries to **fetch the reference over
    the network** and fails with ``Unresolvable``. So every vendored file is registered
    under its own ``$id`` and resolution stays on disk.

    Two consequences worth stating rather than discovering:

    * A pack document cannot be validated without ``batch-record-v1.json`` beside it. If a
      pack definition ever becomes portable across repositories (FPRD OQ-4) it carries that
      dependency with it - vendoring one file would not be enough.
    * Nothing is fetched from the network at validation time, which is what keeps this
      feature's whole test path offline and laptop-runnable (ADR-023).
    """
    if name not in _validators:
        from jsonschema import Draft202012Validator
        from referencing import Registry
        from referencing.jsonschema import DRAFT202012

        resources = [
            DRAFT202012.create_resource(dict(load(filename)))
            for filename in (
                PACK_DEFINITION_SCHEMA,
                BATCH_RECORD_SCHEMA,
                ADAPTER_MANIFEST_SCHEMA,
                MIDATAWORKS_SCHEMA,
            )
        ]
        registry: Registry[Any] = Registry().with_resources(
            (str(resource.id() or ""), resource) for resource in resources
        )
        _validators[name] = Draft202012Validator(dict(load(name)), registry=registry)
    return _validators[name]


def pack_definition_schema() -> Mapping[str, Any]:
    """``pack-definition-v1.json`` - the pack artifact's contract (foundation 3.4)."""
    return load(PACK_DEFINITION_SCHEMA)


def part_reference_subschema() -> Mapping[str, Any]:
    """``batch-record-v1.json#/$defs/PartReference`` - **the settled element shape.**

    Returned with the whole batch-record document as its ``$defs`` scope would normally be
    needed for ``$ref`` resolution; this subschema's two object references are inlined by
    the generator, so it validates standalone. A test asserts that - if the file ever
    starts using ``$ref`` here, the test goes red rather than the validator silently
    accepting anything.
    """
    return at(BATCH_RECORD_SCHEMA, "$defs", "PartReference")


def adapter_parts_item_subschema() -> Mapping[str, Any]:
    """The item subschema of ``adapter-manifest-v1.json``'s ``parts`` array.

    **Taken out of the loaded file and never copied into a test.** It is ``{"type":
    "object"}`` today, so a copied constraint would agree with itself and prove nothing;
    read from the file, the conformance check in :mod:`packs.records`'s tests fails the day
    the schema constrains the item.
    """
    return at(ADAPTER_MANIFEST_SCHEMA, "$defs", "AdapterManifestV1", "properties", "parts", "items")
