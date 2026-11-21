"""Configuration resolution  (FR-003.2, FR-003.8; R-02.9, R-02.10).

**Five steps, in this order, and the order is the point:**

1. validate the body against the vendored ``selector-config-v1.json`` (foundation 3.1);
2. resolve the ``kind`` from the **one** registry;
3. resolve the hash and its provenance (R-02.10);
4. for a composite, resolve children **depth-first with a visiting set** so a cycle is
   refused before evaluation, not discovered during it;
5. call the plugin's ``prepare()`` - compile regexes with bounds, require a threshold where
   one is needed, check that a child set is non-empty.

**Return a fully resolved object or raise. Never half-resolved.** A partially resolved
configuration is a configuration whose failure surfaces at completion 37, by which point a
batch has been half-evaluated and the operator is reading a transport error instead of a
configuration error.

## Composite children are names, not nested bodies

``selector-config-v1.json`` says a composite's ``children`` are *"names of other configured
selectors"*. So resolution needs a **catalog**: the other configured selectors, by name.
That is what makes a cycle possible at all - and why the visiting set is here rather than
being an implementation detail of the composite plugin.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import jsonschema

from core.errors import refuse
from schemas.selector import SelectorKind
from services.selectors import registry
from services.selectors.hashing import resolve_hash
from services.selectors.types import Capability, ResolvedHash

#: Where the vendored contracts are. The same env override and the same walk-up as
#: :mod:`services.reindex`, for the same reason: a hard-coded absolute path works on one
#: workstation.
_SCHEMA_DIR_ENV = "MIFORGE_SCHEMA_DIR"
_DEFAULT_SCHEMA_DIR = Path(__file__).resolve().parents[4] / "docs" / "schemas"
SCHEMA_FILENAME = "selector-config-v1.json"

_schema: dict[str, Any] | None = None


def schema_dir() -> Path:
    return Path(os.environ.get(_SCHEMA_DIR_ENV, str(_DEFAULT_SCHEMA_DIR)))


def load_schema() -> dict[str, Any]:
    """The vendored selector-config schema, parsed once.

    **Parsed, not described.** This module's job is to agree with the file, and the only
    way to agree with a file is to read it - which is also how the two capabilities the
    contract deliberately cut (embedding similarity, a composite weighted sum) stay cut
    rather than being reintroduced from a document that predates it.
    """
    global _schema
    if _schema is None:
        path = schema_dir() / SCHEMA_FILENAME
        if not path.is_file():
            raise refuse(
                "schema_not_vendored",
                f"{SCHEMA_FILENAME} is not present in {schema_dir()}; set "
                f"{_SCHEMA_DIR_ENV} to the directory holding the vendored contracts.",
                schema=SCHEMA_FILENAME,
                searched=str(schema_dir()),
            )
        _schema = json.loads(path.read_text("utf-8"))
    return _schema


def _pointer(error: jsonschema.ValidationError) -> str:
    """The failing location as a JSON pointer, for ``details`` (E-2).

    A refusal that says only "invalid" makes the operator guess which field; ADR-018
    forbids the code guessing and it is no better to make the reader do it.
    """
    return "/" + "/".join(str(part) for part in error.absolute_path)


#: ``kind`` -> the ``$defs`` entry that describes it. Derived from the schema at load time
#: rather than hard-coded, so a type added to the contract does not need an edit here.
def _branch_for(kind: Any, schema: dict[str, Any]) -> dict[str, Any] | None:
    """The ``$defs`` branch whose ``kind`` const equals ``kind``, or ``None``.

    Exists because ``oneOf`` reports *"is not valid under any of the given schemas"* at the
    document root, which names nothing. A configuration that declares its ``kind`` has
    already told us which branch it meant, so validating against that branch gives the
    **failing field** rather than the failing document - and a refusal that does not name
    the field makes the operator guess, which is the thing ADR-018 forbids the code from
    doing and is no better when the reader has to do it.
    """
    definitions: dict[str, Any] = schema.get("$defs") or {}
    for definition in definitions.values():
        if not isinstance(definition, dict):
            continue
        properties = definition.get("properties") or {}
        declared = (properties.get("kind") or {}).get("const")
        if declared is not None and declared == kind:
            return definition
    return None


def validate_config(config: dict[str, Any]) -> None:
    """Validate one selector configuration. **Refuses naming the failing pointer** (E-2).

    Raises:
        Refusal: ``selector_config_invalid``, with the JSON pointer and the schema message.
    """
    schema = load_schema()
    errors = sorted(
        jsonschema.Draft202012Validator(schema).iter_errors(config),
        key=lambda e: list(e.absolute_path),
    )
    if not errors:
        return

    branch = _branch_for(config.get("kind"), schema) if isinstance(config, dict) else None
    if branch is not None:
        resolver_schema = dict(branch)
        resolver_schema["$defs"] = schema.get("$defs", {})
        branch_errors = sorted(
            jsonschema.Draft202012Validator(resolver_schema).iter_errors(config),
            key=lambda e: list(e.absolute_path),
        )
        if branch_errors:
            errors = branch_errors

    deepest = max(errors, key=lambda e: len(e.absolute_path))
    raise refuse(
        "selector_config_invalid",
        f"The selector configuration does not satisfy {SCHEMA_FILENAME}: "
        f"{deepest.message}. Correct the field named in details.pointer.",
        pointer=_pointer(deepest),
        validator=deepest.validator,
        all_messages=[e.message for e in errors][:8],
    )


@dataclass
class ResolvedSelector:
    """A configuration that is ready to evaluate. Built only by :func:`resolve`.

    Not a Pydantic model: it holds compiled regexes, a resolved child list and a plugin
    instance, none of which are serialisable, and pretending otherwise would invite someone
    to round-trip it through JSON and get a different object back.
    """

    kind: SelectorKind
    name: str
    body: dict[str, Any]
    hash: ResolvedHash
    capability: Capability
    #: Whatever the plugin's ``prepare`` produced: compiled patterns, a parsed rule body, a
    #: resolved child list. Opaque to this module by design - resolution owns the *order*
    #: of the steps, not the contents of step 5.
    prepared: Any = None
    #: Resolved children, for a composite. Empty for every other kind.
    children: tuple[ResolvedSelector, ...] = ()
    #: The plugin instance, satisfying :class:`schemas.selector.Selector`.
    plugin: Any = None
    #: Caveats raised at resolution - today only ``judge_shares_lineage`` (R-02.11, E-7).
    #: Carried here so every verdict this configuration produces records them.
    caveats: tuple[str, ...] = ()
    #: Extras a plugin needs that are not configuration: the resolved provider, the review
    #: port. Set by ``prepare``.
    bindings: dict[str, Any] = field(default_factory=dict)

    @property
    def config_sha256(self) -> str:
        return self.hash.sha256


def resolve(
    config: dict[str, Any],
    *,
    catalog: dict[str, dict[str, Any]] | None = None,
    handoff: dict[str, Any] | None = None,
    source: str | None = None,
    authored_bytes: bytes | None = None,
    run_target: Any = None,
    _visiting: tuple[str, ...] = (),
) -> ResolvedSelector:
    """Resolve one configuration, recursively for a composite.

    Args:
        config: The selector configuration body.
        catalog: The other configured selectors by name, for a composite's ``children``.
        handoff: A miDataworks handoff version, when the configuration came from one.
        source: Where a local configuration came from (R-02.40). Required without
            ``handoff``; ``"inline"`` for F7's dry-run.
        authored_bytes: The authored bytes, digested only to verify against the contract.
        run_target: The run's own inference target, for the R-02.11 guard. ``None`` before
            a run exists.
        _visiting: The composite resolution path, for cycle detection. Internal.

    Raises:
        Refusal: any of ``selector_config_invalid``, ``selector_type_unknown``,
            ``selector_config_hash_mismatch``, ``composite_cycle``,
            ``composite_child_unknown``, ``judge_is_resident_model``, or a plugin's own
            ``prepare`` refusal.
    """
    validate_config(config)

    kind = str(config["kind"])
    name = str(config["name"])
    factory = registry.get(kind)
    declared = registry.capability(kind)
    resolved_hash = resolve_hash(
        config, handoff=handoff, source=source, authored_bytes=authored_bytes
    )

    resolved = ResolvedSelector(
        kind=kind,  # type: ignore[arg-type]  # validated against the schema's `kind` enum
        name=name,
        body=dict(config),
        hash=resolved_hash,
        capability=declared,
    )

    if kind == "composite":
        resolved.children = _resolve_children(
            config,
            catalog=catalog or {},
            handoff=None,
            source=source,
            run_target=run_target,
            visiting=(*_visiting, name),
        )

    plugin = factory(resolved)
    resolved.plugin = plugin
    prepare = getattr(plugin, "prepare", None)
    if prepare is not None:
        # `prepare` refuses here, once per configuration - not once per completion. A
        # configuration error that surfaces 64 times is 63 copies of one fact, and it
        # arrives after the batch has been half-evaluated.
        prepare(resolved, run_target=run_target)
    return resolved


def _resolve_children(
    config: dict[str, Any],
    *,
    catalog: dict[str, dict[str, Any]],
    handoff: dict[str, Any] | None,
    source: str | None,
    run_target: Any,
    visiting: tuple[str, ...],
) -> tuple[ResolvedSelector, ...]:
    """Resolve a composite's named children, refusing a cycle **naming the path** (E-13)."""
    children: list[ResolvedSelector] = []
    for child_name in config["children"]:
        if child_name in visiting:
            cycle = [*visiting, child_name]
            raise refuse(
                "composite_cycle",
                "A composite selector refers to itself, directly or through a child, so "
                "resolution would not terminate. Break the cycle named in details.cycle.",
                cycle=cycle,
            )
        child_body = catalog.get(child_name)
        if child_body is None:
            raise refuse(
                "composite_child_unknown",
                f"Composite {config['name']!r} names child {child_name!r}, which is not "
                f"among the configured selectors. Supply it in the catalog, or correct the "
                f"composite's children.",
                child=child_name,
                available=sorted(catalog),
            )
        children.append(
            resolve(
                child_body,
                catalog=catalog,
                handoff=handoff,
                source=source,
                run_target=run_target,
                _visiting=visiting,
            )
        )
    return tuple(children)
