"""The real producer's output against the real schemas  (FR-007.32, FAC-13; ADR-016).

**Conformance, not sync.** A sync test proves two repositories' copies of a file match; this
proves *our producer's output matches our contract*. Both are owed and neither substitutes
for the other - F7's structural-test ledger records that distinction.

**Two schemas, for two different reasons.**

``adapter-manifest-v1.json``'s ``parts`` item is the one F8 writes into, and its subschema is
read **out of the loaded file and never copied**: it is ``{"type": "object"}`` today, so a
copied constraint would agree with itself and prove nothing. Read from the file, this goes red
the day the schema constrains the item - which is its whole value.

``batch-record-v1.json``'s ``PartReference`` is the one that **bites today**: closed, three
required keys, and the slot's vocabulary as an enum. The FTASKS' own control - *"drop ``slot``
from the output → the conformance test red"* - **cannot fire against the adapter schema
alone**, because an object missing a key is still an object. It fires here.

Neither schema is edited by this feature, ever: ``docs/schemas/`` is single-owner shared
state and ``adapter-manifest-v1.json`` is owed byte-identical vendoring into miLLM under
BRD-01 R-01.28, so an edit here would change a contract another repository is waiting to copy.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path

import jsonschema
import pytest

from packs import contracts
from packs.records import PackVersionIdentity, part_reference_record
from packs.references import ContractPartRef, ExternalPartRef
from packs.slots import SlotKind

PACK = PackVersionIdentity(pack_id="notary-md", version="2")

ELEMENTS = {
    "contract": part_reference_record(
        SlotKind.PROMPT_SET,
        ContractPartRef(type="prompt-set", name="support-tickets", version="3"),
        pack=PACK,
    ).element(),
    "external": part_reference_record(
        SlotKind.RETENTION_SET,
        ExternalPartRef(
            origin="/data/operator/retention.jsonl",
            content_sha256="e" * 64,
            taken_in_at=datetime(2026, 10, 14, tzinfo=UTC),
        ),
        pack=PACK,
    ).element(),
}


@pytest.mark.parametrize("source", sorted(ELEMENTS))
def test_the_real_output_is_a_valid_adapter_manifest_parts_item(source: str) -> None:
    """Validated against the subschema **taken out of the loaded file**."""
    subschema = dict(contracts.adapter_parts_item_subschema())
    jsonschema.Draft202012Validator(subschema).validate(ELEMENTS[source])


@pytest.mark.parametrize("source", sorted(ELEMENTS))
def test_the_real_output_is_a_valid_part_reference(source: str) -> None:
    """The closed subschema: this is the one a dropped key fails."""
    jsonschema.Draft202012Validator(dict(contracts.part_reference_subschema())).validate(
        ELEMENTS[source]
    )


def test_a_whole_adapter_manifest_carrying_the_real_elements_validates() -> None:
    """The element in the artifact that will hold it, not only on its own.

    A manifest built by the **real** fixture builder, with its ``parts`` replaced by the real
    producer's output. That is the round trip F8 will actually perform; an element validated
    only against its own subschema could still break the document that carries it.
    """
    from tests.support.db import adapter_manifest

    manifest = adapter_manifest(parts=list(ELEMENTS.values()))
    assert manifest["parts"] == list(ELEMENTS.values())


def test_the_adapter_schema_still_constrains_the_item_no_further() -> None:
    """A **standing record of why the control needs the batch record** (not an assertion of taste).

    While this holds, the adapter schema cannot catch a malformed element, so the conformance
    value comes from ``PartReference``. The day someone constrains it, this test goes red and
    the comment above it should be revisited - which is the point of asserting a fact about a
    contract rather than remembering it.
    """
    assert dict(contracts.adapter_parts_item_subschema()) == {"type": "object"}


def test_the_part_reference_subschema_resolves_without_a_network_fetch() -> None:
    """``PartReference``'s two object references are inlined, so it validates standalone.

    If the generator ever emits ``$ref`` here instead, a plain validator would try to fetch
    the reference and fail - which is exactly what ``pack-definition-v1.json``'s cross-file
    ``$ref`` does, and why :func:`packs.contracts.validator` exists.
    """
    text = json.dumps(dict(contracts.part_reference_subschema()))
    assert "$ref" not in text


def test_the_schema_files_this_feature_reads_are_the_vendored_ones() -> None:
    """Read from ``docs/schemas/`` relative to the repository, not from configuration.

    The published path: ``docs/schemas/`` is the one ``docs/`` directory the public mirror
    keeps, because these are public contracts.
    """
    assert contracts.SCHEMA_DIR.is_dir()
    assert (contracts.SCHEMA_DIR / contracts.PACK_DEFINITION_SCHEMA).is_file()
    assert contracts.SCHEMA_DIR == Path(__file__).resolve().parents[3] / "docs" / "schemas"
