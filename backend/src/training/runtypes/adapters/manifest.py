"""Assemble an adapter manifest. **This module defines no field.**

``docs/schemas/adapter-manifest-v1.json`` is normative (R-02.44; FR-008.28): 13 properties,
9 required. Every value below comes from a named source and none is invented here -

===========================  ====================================================
Field                        Source
===========================  ====================================================
``kind``, ``schema_version``  constants the schema fixes, read from the schema
``name``                      :func:`~.paths.adapter_name`, step **unpadded**
``run_id``, ``step``          the run; ``step`` is one integer, padded only in the path
``base``                      F1's acquisition record (:func:`base_block`)
``adapter``                   the run configuration's own ``manifest_block``
``weights_sha256``            streamed SHA-256 of the file **on disk**, added by the saver
``created_at``                RFC 3339, at save
``optimizer``                 F1's optimizer identity - same shape as the checkpoint's
``parts``                     F7's producer, never constructed here
``test``, ``retention``       F4's ``ScoreRecord``s, **including** ``provider``
===========================  ====================================================

## ``provider`` is optional in the schema, and that decides how it is written

``provider`` is in neither ``ScoreRecord``'s ``required`` list, so a manifest omitting it
**validates clean**: the conformance test's validation step stays green over the omission
and only a key-set assertion in the *missing* direction catches it (M17). So the score
payloads are built by F4's :func:`services.scoring.manifest_payload.score_record_payload`,
which refuses a score that carries no value for a declared key - **one builder for both
manifests**, with a contract test asserting the two schemas' ``ScoreRecord`` key sets are
equal so the shared builder cannot silently become wrong for one of them.

## No ``domain``, no ``signal_source``, no ``selector_hash``, no ``grader_hash``

This schema declares none of the four and closes its objects, so a conforming adapter
manifest **cannot** carry them - which means a travelling adapter cannot state the domain it
was trained toward or the source that produced its batches. That asymmetry is reported
(FPRD D7), not patched: inventing a field here would fail validation, and widening the
schema is a cross-repo act for the operator (ADR-014). They go in the run log.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from datetime import UTC, datetime
from typing import Any

from core.errors import refuse
from services.scoring.manifest_payload import score_record_payload
from training.acquisition.base import AcquisitionRecord
from training.runtypes.adapters import paths as adapter_paths
from training.runtypes.adapters import schema as adapter_schema
from training.runtypes.adapters.config import AdapterConfigIn


def _constant(name: str) -> str:
    """A ``const`` the schema pins, read from the file rather than retyped."""
    entry = adapter_schema.schema().get("$defs", {}).get("AdapterManifestV1", {})
    value = entry.get("properties", {}).get(name, {}).get("const")
    if not isinstance(value, str) or not value:
        raise refuse(
            "schema_definition_missing",
            f"adapter-manifest-v1.json pins no const for {name!r}.",
            field=name,
        )
    return value


def base_block(
    record: AcquisitionRecord,
    *,
    manifest_sha256: str | None = None,
) -> dict[str, Any]:
    """The ``base`` object: which model this adapter is bound to (``$defs.AdapterBase``).

    **Two cases, both defined by the schema, and the resolution order matters.** A consumer
    resolves the base by ``manifest_sha256`` when it is present - the exact base, even if
    its directory moved - otherwise by matching ``name`` against a registered repo id,
    otherwise against a row name.

    1. **The base was a saved checkpoint.** ``name`` is that checkpoint's own manifest
       ``name``, which F1 recorded as ``record.parent``, and ``manifest_sha256`` is its
       manifest hash.
    2. **The base was a Hub model.** ``name`` is **the repo id**, e.g.
       ``LiquidAI/LFM2.5-1.2B-Instruct``, and ``manifest_sha256`` is ``null`` - which is
       what distinguishes the case.

    A local directory that is not a miForge checkpoint falls under case 2's shape: the
    recorded model identity, with a null hash.

    Raises:
        Refusal: ``adapter_base_unresolved`` when the base *is* a checkpoint and no manifest
            hash was supplied. ``null`` there would be a **false assertion** - the schema's
            null means *"the base was not itself a saved checkpoint"*, not *"its hash was
            not looked up"* - and an adapter that cannot be rebound to its exact base is the
            failure BRD-01 R-01.27 refuses on.
    """
    if record.parent:
        if not manifest_sha256:
            raise refuse(
                "adapter_base_unresolved",
                f"This adapter's base is the saved checkpoint {record.parent!r} and its "
                f"manifest hash could not be resolved. Refusing rather than writing null: "
                f"null in this field asserts the base was never a saved checkpoint, which "
                f"would be false, and the exact base would stop being identifiable if its "
                f"directory moved.",
                base=record.parent,
            )
        return {"name": record.parent, "manifest_sha256": manifest_sha256}

    name = record.source_repo or record.base_model
    if not name:
        raise refuse(
            "adapter_base_unresolved",
            "This adapter's acquisition record names no base. An adapter is meaningless "
            "without one and its base must not be inferred from its directory.",
            mode=record.mode.value,
        )
    return {"name": name, "manifest_sha256": None}


def build(
    *,
    run_id: str,
    step: int,
    base: Mapping[str, Any],
    adapter: AdapterConfigIn,
    optimizer: Mapping[str, Any],
    parts: Sequence[Mapping[str, Any]] | None = None,
    test: Mapping[str, Any] | None = None,
    retention: Mapping[str, Any] | None = None,
    created_at: str | None = None,
) -> dict[str, Any]:
    """Assemble one adapter manifest **without** ``weights_sha256``.

    The digest is added by :func:`training.runtypes.adapters.save.save`, from the file as
    written: hashing the in-memory state dict would describe bytes no consumer ever reads.
    So the document this returns is deliberately **not yet valid**, and the saver validates
    after adding it and before the rename.

    Args:
        base: the ``base`` object from :func:`base_block`.
        adapter: the run configuration. **The only source** of ``rank`` and ``alpha``, which
            must agree with ``adapter_config.json``'s ``r`` and ``lora_alpha``.
        parts: contract references from F7's ``records.part_reference_record``. ``None``
            when the run had no pack; an **empty list** when a pack resolved to no parts -
            two different facts, so ``None`` omits the optional key and ``[]`` writes it.
        test, retention: F4's ``ScoreRecord``s, or ``None`` where nothing ran at this step.

    Raises:
        Refusal: ``adapter_parts_invalid`` when a ``parts`` element is not an object, or
            carries no ``source`` discriminator.
    """
    base_name = base.get("name")
    if not isinstance(base_name, str) or not base_name:
        raise refuse(
            "adapter_base_unresolved",
            "The base object carries no name.",
            base=dict(base),
        )

    doc: dict[str, Any] = {
        "kind": _constant("kind"),
        "schema_version": _constant("schema_version"),
        "name": adapter_paths.adapter_name(base_name, run_id, step),
        "run_id": run_id,
        "step": step,
        "base": dict(base),
        "adapter": adapter.manifest_block(),
        "optimizer": dict(optimizer),
        # Verbatim or null, through F4's builder, which refuses a score missing a declared
        # key rather than defaulting `provider`. See the module docstring.
        "test": score_record_payload(test),
        "retention": score_record_payload(retention),
        "created_at": created_at if created_at is not None else _now(),
    }
    if parts is not None:
        doc["parts"] = _checked_parts(parts)
    return doc


def part_element_subschema() -> Mapping[str, Any]:
    """The **settled** element shape, read from the contract that settles it.

    ``adapter-manifest-v1.json`` constrains ``parts`` items only to ``{"type": "object"}``
    (verified), so whichever feature spelled the element first would have defined it for the
    project - which is why F8 refuses rather than spelling one. It is settled as a **gate
    contract**: ``batch-record-v1.json#/$defs/PartReference``, whose own description names
    this exact seam - *"two features each deferred to the other: one named it an open
    question and the other refuses rather than guess a spelling, so the silence left an
    adapter writer unbuildable."*

    Read through F7's own accessor so there is one reader, not a path spelled twice.

    Raises:
        Refusal: ``pack_part_shape_unavailable`` when F7's module is absent - F8 refuses
            rather than inventing a spelling (FPRD D10, FR-008.32).
    """
    try:
        from packs.contracts import part_reference_subschema
    except ImportError as exc:
        raise refuse(
            "pack_part_shape_unavailable",
            "The pack part-reference element shape cannot be read, so the adapter's "
            "'parts' array cannot be written. Refusing rather than inventing a spelling: "
            "the adapter manifest constrains the item only to 'object', so a guessed shape "
            "would validate and would become this project's second answer.",
            detail=str(exc),
        ) from exc
    return part_reference_subschema()


def _checked_parts(parts: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """Pass F7's elements through **unreshaped**, validated against the settled contract.

    F7's ``packs.records.part_reference_record`` is the **only producer** of an element, and
    its ``PartReferenceRecord.element()`` is what lands here. F8 writes what comes back,
    reshapes nothing and constructs nothing (FR-008.32, M16).

    **The discriminator is what must not be lost**, and it is *exactly one of*
    ``contract_reference`` and ``external`` being non-null - not a ``source`` string. That
    correction is worth recording: F8's own documents describe the element as
    ``{slot, pack, source, type, name, version}``, taken from F7's **documents**, and the
    shape F7 **shipped** is ``{kind, contract_reference, external}``. A ``source`` check
    would have refused every element F7's producer emits. Checked against the contract
    rather than against either feature's prose, which is the only reading that could have
    caught it.

    Validation is the whole guarantee here: without the discriminator an external part's
    provenance would read as miDataworks', and the contract refuses both the
    neither-reference case (*"an unattributed training input makes the run unreproducible"*)
    and the both-references case.

    Raises:
        Refusal: ``adapter_parts_invalid`` naming the index and the violations;
            ``pack_part_shape_unavailable`` when F7's producer's contract cannot be read.
    """
    import jsonschema

    validator = jsonschema.Draft202012Validator(part_element_subschema())

    checked: list[dict[str, Any]] = []
    for index, part in enumerate(parts):
        if not isinstance(part, Mapping):
            raise refuse(
                "adapter_parts_invalid",
                f"parts[{index}] is {type(part).__name__}, not an object. A part reference "
                f"comes from the pack records and is written unreshaped; "
                f"the adapter writer constructs no element of its own.",
                index=index,
            )
        element = dict(part)
        violations = [
            f"{'/'.join(str(p) for p in error.absolute_path) or '<root>'}: {error.message}"
            for error in sorted(validator.iter_errors(element), key=lambda e: list(e.path))
        ]
        if violations:
            raise refuse(
                "adapter_parts_invalid",
                f"parts[{index}] is not a part reference as the gate contract settles it "
                f"(batch-record-v1.json#/$defs/PartReference). It is written unreshaped or "
                f"not at all: an element adjusted here would be a second spelling, and "
                f"an element with no provenance makes the run unreproducible.",
                index=index,
                element=element,
                violations=violations,
                # The DIRECTION, which `violations` cannot give: a `oneOf` failure reports
                # "is not valid under any of the given schemas" and names neither the
                # element nor whether it carried no provenance or two. An operator reading
                # a refusal needs that, so it is computed here and attached.
                references_present=[
                    name for name in ("contract_reference", "external") if element.get(name)
                ],
            )

        # EXACTLY ONE of the two references is **the schema's** constraint, and that is why
        # there is no second check for it here.
        #
        # F8 originally enforced it in code, because `PartReference`'s description claimed
        # "exactly one of the two" while the schema enforced NEITHER half: `required` named
        # all three members and each reference was typed `anyOf [object, null]`, with no
        # `oneOf`, `not` or `dependentSchemas`. So an element carrying neither reference, or
        # both, validated clean. F8 reported it rather than editing a contract (ADR-014) and
        # pinned the gap with a test that said in its own words *"if a later re-vendor adds
        # the constraint, this goes red and says so."*
        #
        # The coordinator added the `oneOf` on 2026-10-14 and it did go red - to **both**
        # copies of the definition, since it is inlined in the batch record as well as
        # sitting in `$defs`, and patching one would have left the other permissive. The
        # duplicate check that used to stand here was then removed rather than kept "for
        # safety": the validator above runs first, so it was unreachable for the two cases
        # it existed to catch, and an unreachable guard reads as protection while protecting
        # nothing. What it contributed that the validator cannot - *which* direction failed -
        # is now `references_present` in that refusal's details.
        #
        # Relaxing the schema is caught immediately: the flipped test asserts `oneOf` is
        # present, and says what an element with no provenance would cost in an artifact
        # built to travel without its run log.
        #
        # It matters in both directions. Neither: the contract's own words are that such a
        # part is refused "because an unattributed training input makes the run
        # unreproducible and nothing downstream can say where a behaviour came from" - and
        # the adapter manifest is the one artifact built to travel without its run log, so
        # an unattributed part there is unattributable for good. Both: nothing says which
        # provenance holds, and an external part reading as a contract reference is exactly
        # the borrowed provenance F7's producer exists to prevent.
        checked.append(element)
    return checked


def _now() -> str:
    """RFC 3339, UTC, to the second - the same rendering F5's manifest writer uses."""
    return datetime.now(UTC).replace(microsecond=0).isoformat().replace("+00:00", "Z")
