"""Build, validate, and the two assertions a validator cannot make  (FR-005.7 - FR-005.16).

``docs/schemas/checkpoint-manifest-v1.json`` is **normative** and this module cites it
rather than restating it. Restating a field list in prose is what produced September's
gaps 2, 3 and 4 in this project, so the declared keys below are **read out of the schema
file at import and never retyped** - a copied list agrees with itself and would pass the
day the schema changes.

## Seventeen fields, six of them other features' values

F5 owns the **write**, never the value. Every borrowed field is an explicit keyword
parameter of :func:`build` and an absent one is a named refusal, because the failure mode
here is a *plausible default* - ``signal_source: "corpus"``, a zeroed score, an omitted
``domain`` - written permanently into the one artifact meant to outlive the database, the
run log and the process. ADR-018 puts it sharply: *a ``getattr`` default that always fires
is a fabricated value, and that is worse than a refusal because it is silent.* So there is
**no** ``getattr`` default, **no** ``or <literal>`` and **no** ``.get(k, <literal>)``
anywhere in this module's assembly path.

| field | value comes from |
|---|---|
| ``kind``, ``schema_version`` | the schema's own constants |
| ``run_id``, ``step`` | the run record and the completed step (F1 / F6) |
| ``domain`` | run configuration, or ``null`` (F7 / F6) |
| ``signal_source`` | the run's configured source (**F2**) |
| ``parent``, ``base_model`` | the lineage rule below (F1's functions + the carry-forward) |
| ``optimizer`` | F1 records the identity; ``state`` is the resume decision's (F5) |
| ``selector_hash``, ``grader_hash`` | **F3** (R-02.10) |
| ``test``, ``retention`` | **F4**'s ``ScoreRecord``s, verbatim (R-02.16 - R-02.18) |
| ``merged_adapter`` | **F8**'s merge record (R-02.45) |
| ``weights_sha256``, ``created_at``, ``name`` | F5 |

## The key-set assertion has two clauses, and only one of them is defence in depth

**(a) no key the schema does not declare.** The schema now closes every object
(``additionalProperties: false`` at every level, verified by parsing the live file), so the
validator catches this too. Kept anyway, because it fires **first** and refuses naming the
key in miForge's own vocabulary rather than as a JSON-Schema error string.

**(b) no key *dropped* from the record F5 was handed. This is the breach nothing else can
see.** ``ScoreRecord.provider`` is **declared and not in ``required``**, so a manifest that
silently loses the provider F4 recorded **validates clean** - and the schema-sync test
cannot see it either, because that test compares schema *files*, not manifests. A test is
the only thing that catches the omission, and the provenance R-02.16 demands would be gone
with no error on any surface.

*This clause is where the requirement's weight moved on 2026-10-13. Before the three
operator-approved schema changes landed, (a) was the only enforcement and the mutation
control mutated by **writing** ``provider`` - so the moment the schema changed, the control
would have been recorded as surviving against correct code.*

## The lineage rule that looks right and is not

``base_model`` is *"the root model this lineage descends from"* - the schema's own words. A
continue or a branch **loads a checkpoint**, so setting the field to the thing just loaded
roots the tree at the parent and terminates every lineage walk one hop short. It validates,
it saves, and the tree looks plausible. F1 settled it in three named functions and
:func:`resolve_lineage` **calls them rather than re-deriving the rule**.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from core.canonical import canonical_bytes
from core.errors import refuse
from models.domain_state import ArtifactKind
from services.checkpoints.paths import qualified_name
from services.reindex import load_schema
from training.acquisition.base import (
    lineage_for_creation,
    lineage_for_first_load,
    lineage_for_loaded_checkpoint,
)

__all__ = [
    "ManifestFields",
    "assert_no_dropped_keys",
    "assert_no_undeclared_keys",
    "build",
    "declared_keys",
    "read",
    "resolve_base_model",
    "resolve_lineage",
    "validate",
]

#: The root object's key in :func:`declared_keys`. A sentinel rather than ``""`` so a bug
#: that passed an empty property name could not silently address the top level.
ROOT = "<root>"

_validator: Any = None
_declared: dict[str, frozenset[str]] | None = None


# ======================================================================== the schema


def _schema() -> Mapping[str, Any]:
    return load_schema(ArtifactKind.CHECKPOINT)


def _root_def(schema: Mapping[str, Any]) -> Mapping[str, Any]:
    """The object the top-level ``$ref`` points at."""
    return _deref(schema, str(schema["$ref"]))


def _deref(schema: Mapping[str, Any], ref: str) -> Mapping[str, Any]:
    if not ref.startswith("#/$defs/"):
        raise refuse(
            "checkpoint_schema_unexpected",
            f"The vendored checkpoint manifest schema contains a reference this module "
            f"cannot resolve ({ref!r}). Refusing rather than guessing at the field list: "
            f"the schema is normative and a guessed key set would enforce nothing.",
            ref=ref,
        )
    name = ref[len("#/$defs/") :]
    definition = schema["$defs"][name]
    assert isinstance(definition, Mapping)
    return definition


def _object_def_of(schema: Mapping[str, Any], prop: Mapping[str, Any]) -> Mapping[str, Any] | None:
    """The object definition a property resolves to, if it resolves to one.

    Handles both shapes the schema actually uses: a bare ``$ref`` (``optimizer``) and an
    ``anyOf`` of a ``$ref`` with ``null`` (``test``, ``retention``, ``merged_adapter``).
    Derived from the file rather than from a hand-written map of property names to ``$def``
    names - that map is exactly the kind of retyped list this module refuses to keep.
    """
    if "$ref" in prop:
        return _deref(schema, str(prop["$ref"]))
    for branch in prop.get("anyOf", ()):
        if isinstance(branch, Mapping) and "$ref" in branch:
            return _deref(schema, str(branch["$ref"]))
    return None


def declared_keys() -> dict[str, frozenset[str]]:
    """Every object's declared property names, **read from the schema file**.

    Keyed by the property that holds the object (``optimizer``, ``test``, ``retention``,
    ``merged_adapter``) plus :data:`ROOT` for the top level. Collected once; see
    :func:`validate` for why "once" is lazy rather than at import.
    """
    global _declared
    if _declared is None:
        schema = _schema()
        root = _root_def(schema)
        declared: dict[str, frozenset[str]] = {ROOT: frozenset(root["properties"])}
        for name, prop in root["properties"].items():
            if not isinstance(prop, Mapping):
                continue
            nested = _object_def_of(schema, prop)
            if nested is not None:
                declared[name] = frozenset(nested["properties"])
        _declared = declared
    return _declared


def _constant(name: str) -> str:
    """A ``const`` property's value, from the schema. ``kind`` and ``schema_version``.

    Read rather than written down: these two are the only fields whose value *is* the
    contract, so typing them here would be the one restatement with no upside at all.
    """
    prop = _root_def(_schema())["properties"][name]
    return str(prop["const"])


def _compiled() -> Any:
    """The draft-2020-12 validator, compiled **once per process** (ADR-024).

    *Once*, not *at import*, and the difference is deliberate. The feature documents say
    "compiled once at import"; the budget they are protecting is per-save recompilation,
    which this satisfies. Compiling at import would (1) make this module unimportable
    whenever ``docs/schemas/`` is not where it is expected - the refusal path, which is
    where most of this feature's defect risk lives, would become untestable - and (2) freeze
    the schema directory before ``MIFORGE_SCHEMA_DIR`` is read, which is the override a
    container uses. :func:`services.reindex.load_schema` made the same call for the same
    reason.
    """
    global _validator
    if _validator is None:
        import jsonschema

        _validator = jsonschema.Draft202012Validator(dict(_schema()))
    return _validator


# ======================================================================== the assertions


def assert_no_undeclared_keys(doc: Mapping[str, Any]) -> None:
    """Clause (a): no key the schema does not declare, at the top level and in each nested
    object.

    **Defence in depth, kept on purpose.** The schema closes every object, so the validator
    refuses an undeclared key too - but this fires first and names the key in miForge's own
    refusal vocabulary (``checkpoint_manifest_invalid`` with the key in ``details``) rather
    than as a JSON-Schema error string a caller has to parse.
    """
    declared = declared_keys()
    _assert_subset(doc, declared[ROOT], where=ROOT)
    for name, allowed in declared.items():
        if name == ROOT:
            continue
        nested = doc.get(name)
        if isinstance(nested, Mapping):
            _assert_subset(nested, allowed, where=name)


def _assert_subset(obj: Mapping[str, Any], allowed: frozenset[str], *, where: str) -> None:
    extra = sorted(set(obj) - allowed)
    if extra:
        raise refuse(
            "checkpoint_manifest_invalid",
            f"The manifest carries {extra} in {where}, which the checkpoint manifest "
            f"contract does not declare. The manifest's fields are exactly the "
            f"schema's. Remove the key, or change the contract in its owning repository "
            f"and re-vendor it - never from here.",
            undeclared=extra,
            where=where,
            declared=sorted(allowed),
        )


def assert_no_dropped_keys(doc: Mapping[str, Any], supplied: Mapping[str, Any]) -> None:
    """Clause (b): every key present in a record F5 was handed is present in what it wrote.

    **The breach nothing else catches.** ``ScoreRecord.provider`` is declared and *not* in
    ``required``, so a manifest that lost it validates clean, and the schema-sync test
    compares schema files rather than manifests. Losing the provider loses the provenance
    R-02.16 demands, with no error anywhere - so this assertion, and the test behind it, is
    the whole of the enforcement.

    Args:
        doc: the assembled manifest.
        supplied: the nested records as the caller handed them over, keyed by the manifest
            property they belong to. A ``None`` entry means "no record", which is a
            statement rather than an omission and is therefore not a dropped key.
    """
    for name, record in supplied.items():
        if not isinstance(record, Mapping):
            continue
        written = doc.get(name)
        if not isinstance(written, Mapping):
            raise refuse(
                "checkpoint_manifest_dropped_record",
                f"The {name!r} record handed to the manifest writer is not in the manifest "
                f"it wrote. A record is written verbatim or as null; silently discarding "
                f"one loses provenance with no error on any surface.",
                where=name,
                supplied=sorted(record),
            )
        dropped = sorted(set(record) - set(written))
        if dropped:
            raise refuse(
                "checkpoint_manifest_dropped_keys",
                f"The manifest writer dropped {dropped} from the {name!r} record it was "
                f"handed. Records pass through verbatim: a dropped optional "
                f"field - 'provider' above all - validates clean against the schema, so "
                f"nothing but this check would ever notice.",
                where=name,
                dropped=dropped,
                supplied=sorted(record),
                written=sorted(written),
            )


def validate(doc: Mapping[str, Any]) -> None:
    """Validate against the vendored contract **and** assert clause (a).

    Two checks, not one, and in this order: the named refusal first where it applies, the
    contract second so nothing non-conforming can get past on the strength of having
    declared keys only.
    """
    assert_no_undeclared_keys(doc)
    errors = sorted(_compiled().iter_errors(dict(doc)), key=lambda e: list(e.path))
    if errors:
        rendered = [
            f"{'/'.join(str(p) for p in e.absolute_path) or ROOT}: {e.message}" for e in errors
        ]
        raise refuse(
            "checkpoint_manifest_invalid",
            f"The manifest does not conform to checkpoint-manifest-v1.json: "
            f"{'; '.join(rendered[:4])}. Fix the value at its source; the writer does not "
            f"coerce one.",
            violations=rendered,
        )


# ======================================================================== lineage


def resolve_lineage(
    *,
    loaded_identity: str | None = None,
    config_identity: str | None = None,
    parent_manifest: Mapping[str, Any] | None = None,
) -> tuple[str, str | None]:
    """``(base_model, parent)`` for the checkpoint about to be written (FR-005.13, DEF-5).

    **Dispatches to F1's three named functions and derives nothing of its own**, because the
    rule has one definition and this is the second reader of it:

    * ``parent_manifest`` given - a **continue or a branch**: the loaded thing *is* a
      checkpoint, so :func:`training.acquisition.base.lineage_for_loaded_checkpoint` carries
      that checkpoint's **own** ``base_model`` forward unchanged and makes it the ``parent``.
      A checkpoint whose manifest carries no ``base_model`` **refuses** there rather than
      substituting itself - substituting would root the tree at the parent.
    * ``loaded_identity`` given - a first load of a Hugging Face directory or a Hub repo:
      that *is* the root, and there is no parent checkpoint.
    * ``config_identity`` given - a ``create`` run: the architecture config is the root and
      ``parent`` is ``null`` (R-02.41, AC12). No schema change was ever needed for it.
    """
    if parent_manifest is not None:
        name = parent_manifest.get("name")
        if not isinstance(name, str) or not name:
            raise refuse(
                "checkpoint_manifest_invalid",
                "The parent checkpoint's manifest carries no 'name', so it cannot be named "
                "as a parent. Refusing rather than synthesising one from its directory.",
                where="parent_manifest",
            )
        return lineage_for_loaded_checkpoint(
            checkpoint_name=name,
            checkpoint_base_model=str(parent_manifest.get("base_model") or ""),
        )
    if config_identity is not None:
        return lineage_for_creation(config_identity=config_identity)
    if loaded_identity:
        return lineage_for_first_load(model_identity=loaded_identity)
    raise refuse(
        "save_refused_missing_field",
        "A checkpoint's lineage root is not optional: name the loaded model's identity, the "
        "architecture config's identity, or the parent checkpoint's manifest. Writing a "
        "guess here would root the tree at something the run never started from.",
        field="base_model",
    )


def resolve_base_model(
    *,
    loaded_identity: str | None = None,
    config_identity: str | None = None,
    parent_manifest: Mapping[str, Any] | None = None,
) -> str:
    """Just the lineage root. See :func:`resolve_lineage`, which is the whole rule."""
    return resolve_lineage(
        loaded_identity=loaded_identity,
        config_identity=config_identity,
        parent_manifest=parent_manifest,
    )[0]


# ======================================================================== the fields object


@dataclass(frozen=True)
class ManifestFields:
    """Everything a manifest needs except what F5 computes itself.

    **The one place another feature's value enters a manifest** (TID section 10, seam 4).
    A frozen dataclass rather than a loose ``**kwargs`` so that a field's type is agreed
    once here instead of in a manifest literal in each feature - and so that a feature
    which has not landed shows up as a missing *argument* rather than as a missing *key*.

    ``domain`` and ``signal_source`` have **no defaults**: ``domain=None`` has to be passed
    deliberately, because *null is a statement and an absent key is an omission* (the
    schema's own words), and ``signal_source`` is required and **not** nullable, so there is
    nothing to default it to that would not be a fabricated provenance claim.
    """

    run_id: str
    step: int
    domain: str | None
    signal_source: str
    base_model: str
    parent: str | None
    optimizer: Mapping[str, Any]
    selector_hash: str | None = None
    grader_hash: str | None = None
    test: Mapping[str, Any] | None = None
    retention: Mapping[str, Any] | None = None
    merged_adapter: Mapping[str, Any] | None = None
    created_at: str | None = None
    extra: dict[str, Any] = field(default_factory=dict)

    def as_kwargs(self) -> dict[str, Any]:
        """The keyword arguments :func:`build` takes. ``extra`` is **not** flattened in.

        ``extra`` exists so a caller can hand over a value whose home is not yet agreed and
        have it **refused by name** at clause (a) instead of silently written. It is passed
        through as ``extra=`` and never merged here, so a reader of this method cannot
        mistake it for a way in.
        """
        return {
            "run_id": self.run_id,
            "step": self.step,
            "domain": self.domain,
            "signal_source": self.signal_source,
            "base_model": self.base_model,
            "parent": self.parent,
            "optimizer": self.optimizer,
            "selector_hash": self.selector_hash,
            "grader_hash": self.grader_hash,
            "test": self.test,
            "retention": self.retention,
            "merged_adapter": self.merged_adapter,
            "created_at": self.created_at,
            "extra": self.extra,
        }


# ======================================================================== build


def _require(value: Any, *, field_name: str) -> Any:
    if value is None or (isinstance(value, str) and not value):
        raise refuse(
            "save_refused_missing_field",
            f"The run carries no {field_name!r}, which the checkpoint manifest requires and "
            f"the schema does not allow to be null. Refusing the save rather than writing a "
            f"default: a fabricated value in the one artifact meant to outlive everything is "
            f"worse than a refusal, because it is silent. Set it on the run.",
            field=field_name,
        )
    return value


def _check_score(record: Mapping[str, Any] | None, *, where: str) -> dict[str, Any] | None:
    """A ``ScoreRecord``, **verbatim**, or ``None``. Never rounded, renormalised or rebuilt.

    The refusals here duplicate what the schema would say, deliberately: ``n: 0`` is the
    case E-6 names, and *"a zero-prompt score is not a score"* is a sentence worth putting in
    the refusal the operator reads rather than leaving as ``n: 0 is less than the minimum
    of 1``.
    """
    if record is None:
        return None
    if not isinstance(record, Mapping):
        raise refuse(
            "checkpoint_manifest_invalid",
            f"The {where!r} score is {type(record).__name__}, not a record. Scoring supplies a "
            f"ScoreRecord or nothing; null means 'no test ran at this step', which is a "
            f"different statement from 'a test ran and scored nothing'.",
            where=where,
        )
    missing = [k for k in ("set", "score", "n") if k not in record]
    if missing:
        raise refuse(
            "checkpoint_manifest_invalid",
            f"The {where!r} score is missing {missing}. A score without its set or its "
            f"sample size is not comparable against another one, which is the whole reason "
            f"the record carries them.",
            where=where,
            missing=missing,
        )
    if not isinstance(record["n"], int) or isinstance(record["n"], bool) or record["n"] < 1:
        raise refuse(
            "checkpoint_manifest_invalid",
            f"The {where!r} score reports n={record['n']!r}. A zero-prompt score is not a "
            f"score, so the save is refused rather than the value coerced. Write null "
            f"if no evaluation ran at this step.",
            where=where,
            n=record["n"],
        )
    # Written verbatim. `dict()` copies the mapping so a later mutation by the caller
    # cannot change what was written, and copies nothing else: no rounding, no
    # renormalisation, no reconstruction, and above all no defaulted `provider`.
    return dict(record)


def build(
    *,
    run_id: str,
    step: int,
    domain: str | None,
    signal_source: str,
    base_model: str,
    parent: str | None,
    optimizer: Mapping[str, Any],
    weights_sha256: str,
    selector_hash: str | None = None,
    grader_hash: str | None = None,
    test: Mapping[str, Any] | None = None,
    retention: Mapping[str, Any] | None = None,
    merged_adapter: Mapping[str, Any] | None = None,
    created_at: str | None = None,
    extra: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Assemble, validate and return one manifest. **Raises rather than guessing.**

    ``domain`` is keyword-only with **no default**, so ``None`` must be passed deliberately
    and an absent key is unrepresentable. ``signal_source`` has no default either: it is
    required and not nullable, its value is F2's, and ``"corpus"`` would be a fabricated
    provenance claim (FPRD section 10's ordering note).

    Args:
        extra: keys whose home is not agreed. Written, so clause (a) **refuses them by
            name** - the alternative is a caller inventing a field and nothing noticing.

    Raises:
        Refusal: ``save_refused_missing_field`` for an absent required value;
            ``checkpoint_manifest_invalid`` for a non-conforming one.
    """
    scores = {
        "test": _check_score(test, where="test"),
        "retention": _check_score(retention, where="retention"),
    }
    supplied: dict[str, Any] = {
        "optimizer": optimizer,
        "merged_adapter": merged_adapter,
        **{k: v for k, v in scores.items() if v is not None},
    }

    doc: dict[str, Any] = {
        "kind": _constant("kind"),
        "schema_version": _constant("schema_version"),
        "run_id": _require(run_id, field_name="run_id"),
        "step": step,
        # Written EXPLICITLY, as a string or as null, and never omitted. The schema keeps it
        # required precisely so that "no domain" is asserted: null is a statement, an absent
        # key is an omission, and a general from-scratch pretraining run biases toward
        # nothing. There is no `or`, no `getattr` and no `.get` on this line on purpose.
        "domain": domain,
        "signal_source": _require(signal_source, field_name="signal_source"),
        "parent": parent,
        "base_model": _require(base_model, field_name="base_model"),
        "optimizer": dict(optimizer) if isinstance(optimizer, Mapping) else optimizer,
        "weights_sha256": _require(weights_sha256, field_name="weights_sha256"),
        "selector_hash": selector_hash,
        "grader_hash": grader_hash,
        # Verbatim or null. `provider` rides in here as F4 recorded it - F5 takes no
        # `provider` argument of its own, derives none from the run and defaults none,
        # because a guessed provider is a fabricated provenance (ADR-018).
        "test": scores["test"],
        "retention": scores["retention"],
        "merged_adapter": dict(merged_adapter) if isinstance(merged_adapter, Mapping) else None,
        "created_at": created_at if created_at is not None else _now(),
        # Both step forms from the one integer, at the point of use: unpadded here, padded
        # in the path (FR-005.12).
        "name": qualified_name(_require(base_model, field_name="base_model"), run_id, step),
    }
    if extra:
        doc.update(extra)

    validate(doc)
    assert_no_dropped_keys(doc, supplied)
    return doc


def _now() -> str:
    """RFC 3339, UTC, to the second. F5's own value, at write time."""
    return datetime.now(UTC).replace(microsecond=0).isoformat().replace("+00:00", "Z")


# ======================================================================== read


#: The manifest's filename, from the one place that already declares it per artifact kind.
MANIFEST_NAME = "manifest.json"


def read(path: Path) -> dict[str, Any]:
    """Read and **validate** a manifest from disk.

    Validated because a checkpoint directory can arrive from anywhere (OQ-4) - copied in
    from another installation, restored from a backup, written by a version of this code
    that no longer exists - so a manifest read from disk is **untrusted input until
    validated**, exactly like a request body.

    Args:
        path: the ``manifest.json`` file, or the directory containing it.

    Raises:
        Refusal: ``checkpoint_not_found`` when there is no manifest there;
            ``checkpoint_manifest_invalid`` when it does not parse or does not conform.
    """
    import json

    file = path / MANIFEST_NAME if path.is_dir() else path
    if not file.is_file():
        raise refuse(
            "checkpoint_not_found",
            f"No {MANIFEST_NAME} at {file}. A directory without one is reported unusable "
            f"rather than indexed with invented fields; name a different checkpoint.",
            path=str(file),
        )
    try:
        parsed = json.loads(file.read_text("utf-8"))
    except ValueError as exc:
        raise refuse(
            "checkpoint_manifest_invalid",
            f"{file} is not valid JSON ({exc}). Fix or remove the file; nothing is inferred "
            f"from the directory around it.",
            path=str(file),
        ) from exc
    if not isinstance(parsed, dict):
        raise refuse(
            "checkpoint_manifest_invalid",
            f"{file} parses to {type(parsed).__name__}, not an object.",
            path=str(file),
        )
    validate(parsed)
    return parsed


def digest(doc: Mapping[str, Any]) -> str:
    """SHA-256 of the manifest's **canonical** bytes, through the one serialiser (ADR-005).

    Of the canonical bytes rather than the file's: two manifests differing only in
    whitespace are the same manifest, and hashing the file would say otherwise. This is the
    value ``MergedAdapter.manifest_sha256`` and the index's ``manifest_sha256`` point at,
    and :func:`services.reindex.scan_artifact` computes it the same way for the same reason.
    """
    import hashlib

    return hashlib.sha256(canonical_bytes(dict(doc))).hexdigest()
