"""The adapter manifest's schema, **read** - never restated  (R-02.44; FR-008.28).

``docs/schemas/adapter-manifest-v1.json`` is normative and miForge owns it. This module is
the only place F8 reads it, and F8 **defines no field of it**: the bounds the edge model
validates, the keys the conformance test compares and the scaling values the read path
accepts all come out of the file at import.

## Why read rather than retype

Two copies of a bound are two answers, and the one that is wrong is whichever nobody runs.
This is not hypothetical here: ``ScoreRecord`` had **four** properties when F8's documents
were written and gained a fifth (``provider``) the same day. A retyped tuple would have
agreed with itself and gone on agreeing, which is the fixture failure ADR-016 names.

The reader is :func:`services.reindex.load_schema`, reused rather than reimplemented so
that the schema directory is resolved once (it has an environment override for a container
that installs the package elsewhere) and parsed once.
"""

from __future__ import annotations

from collections.abc import Mapping
from functools import cache
from typing import Any

from core.errors import refuse
from models.domain_state import ArtifactKind
from services.reindex import load_schema, schema_violations

#: The ``$defs`` entries this feature reads bounds and key sets out of.
ADAPTER_CONFIG_DEF = "AdapterConfig"
SCORE_RECORD_DEF = "ScoreRecord"
ADAPTER_BASE_DEF = "AdapterBase"


def schema() -> Mapping[str, Any]:
    """The parsed adapter manifest schema."""
    return load_schema(ArtifactKind.ADAPTER)


def definition(name: str) -> Mapping[str, Any]:
    """One ``$defs`` entry, or a refusal naming it.

    Refuses rather than returning ``{}`` so that a vendored file which lost a definition
    fails loudly at import of the first caller instead of silently relaxing every bound
    this module is supposed to enforce.
    """
    defs = schema().get("$defs")
    if not isinstance(defs, Mapping) or name not in defs:
        raise refuse(
            "schema_definition_missing",
            f"adapter-manifest-v1.json declares no $defs/{name}, so the bounds this "
            f"feature validates against cannot be read. The schema is normative and is "
            f"not restated in code; fix the vendored file.",
            definition=name,
        )
    entry = defs[name]
    if not isinstance(entry, Mapping):
        raise refuse(
            "schema_definition_missing",
            f"$defs/{name} in adapter-manifest-v1.json is not an object.",
            definition=name,
        )
    return entry


@cache
def declared_keys(name: str) -> frozenset[str]:
    """Every key a ``$defs`` object declares - including the non-required ones.

    **The non-required ones are the point.** Since the 2026-10-13 re-vendor every object in
    this schema sets ``additionalProperties: false``, so an *invented* key is refused by
    ``jsonschema`` itself. What a validator still cannot see is a **declared, optional key
    being left out**, and ``provider`` is exactly that: ``anyOf [string, null]``, defaulted
    ``null``, and absent from ``required``. A manifest missing it **validates clean** while
    leaving R-02.16's obligation unmet in the one artifact built to travel without its run
    log. This set is what the conformance test compares in the *missing* direction (M17).
    """
    properties = definition(name).get("properties")
    if not isinstance(properties, Mapping) or not properties:
        raise refuse(
            "schema_definition_missing",
            f"$defs/{name} declares no properties.",
            definition=name,
        )
    return frozenset(str(key) for key in properties)


@cache
def required_keys(name: str) -> frozenset[str]:
    """The ``required`` list of a ``$defs`` object."""
    required = definition(name).get("required") or ()
    return frozenset(str(key) for key in required)


@cache
def known_scaling() -> frozenset[str]:
    """The ``scaling`` values this release recognises, **from the schema's enum**.

    An enum of one today. Read rather than written as ``{"alpha_over_r"}`` so that
    vendoring a schema whose enum grew cannot leave the code quietly narrower than the
    contract - and so the refusal below lists what the contract permits rather than what
    one module remembers.
    """
    prop = definition(ADAPTER_CONFIG_DEF).get("properties", {}).get("scaling", {})
    values = prop.get("enum") if isinstance(prop, Mapping) else None
    if not values:
        raise refuse(
            "schema_definition_missing",
            "adapter-manifest-v1.json declares no enum for adapter.scaling, so an "
            "unrecognised scaling rule could not be refused by name. Guessing the scale "
            "changes the served model's behaviour silently.",
            definition=ADAPTER_CONFIG_DEF,
        )
    return frozenset(str(value) for value in values)


@cache
def scaling_default() -> str:
    """The schema's **own** declared default for ``scaling``.

    Reading this is not the fabricated value ADR-018 forbids: ``scaling`` is not in
    ``AdapterConfig.required`` and the schema declares ``"default": "alpha_over_r"``, so an
    absent key means alpha-over-rank **by contract**. An unrecognised *value* is a
    different thing and is refused. Stated because "not alpha-over-rank by default" reads
    as forbidding this, and someone would otherwise turn an absent key into a refusal the
    contract does not authorise.
    """
    prop = definition(ADAPTER_CONFIG_DEF).get("properties", {}).get("scaling", {})
    value = prop.get("default") if isinstance(prop, Mapping) else None
    if not isinstance(value, str) or not value:
        raise refuse(
            "schema_definition_missing",
            "adapter-manifest-v1.json declares no default for adapter.scaling.",
            definition=ADAPTER_CONFIG_DEF,
        )
    return value


def bound(def_name: str, prop: str, keyword: str) -> Any:
    """One JSON Schema keyword off one property, e.g. ``("AdapterConfig","rank","minimum")``.

    ``None`` when the keyword is absent, so a caller can tell "no such bound" from a bound
    of zero.
    """
    properties = definition(def_name).get("properties")
    if not isinstance(properties, Mapping):
        return None
    entry = properties.get(prop)
    if not isinstance(entry, Mapping):
        return None
    return entry.get(keyword)


def validate(doc: Mapping[str, Any]) -> None:
    """Validate one adapter manifest against the vendored schema, or refuse by name.

    Called **before** the staging directory is renamed into place, never after: once
    ``atomic_dir`` renames, the directory is published, and a document that does not
    conform must fail the write rather than ship (FR-008.28).

    Raises:
        Refusal: ``adapter_manifest_invalid``, listing every violation. Every one of them,
            not the first: a writer that is wrong about two fields should be told about
            both rather than fixed twice.
    """
    violations = schema_violations(doc, ArtifactKind.ADAPTER)
    if violations:
        raise refuse(
            "adapter_manifest_invalid",
            "This adapter manifest does not conform to adapter-manifest-v1.json, which is "
            "normative. Nothing is published: a non-conforming manifest fails the write "
            "rather than becoming a file a consumer has to cope with.",
            violations=violations,
        )
