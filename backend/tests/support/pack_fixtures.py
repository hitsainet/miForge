"""Pack fixtures that go through the real writer  (F7; ADR-016).

**Nothing here hand-builds a pack document.** A literal dict agrees with the reader by
construction and would keep passing after the writer changed - *fixtures whose fields agree
by construction are how a suite stays green over a real bug.* So every pack in the suite is
built as a :class:`packs.definition.PackVersionDefinition`, written by
:func:`packs.registry.create`, and read back off the disk through
:func:`packs.store.read`.

The slot **documents** below are the one hand-written layer, and deliberately: they are the
request shape a client sends, and :func:`packs.definition.slot_from_document` - the real
reader - is what turns them into slots. A test that built :class:`FilledSlot` objects
directly would skip the reader that production uses.

Parts are imported through **F10's** real ingest path, so a resolved slot in a test resolves
against the same index a run resolves against.
"""

from __future__ import annotations

from collections.abc import Mapping
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from packs import registry
from packs.definition import PackVersionDefinition, slot_from_document
from packs.slots import SLOT_ORDER, SlotKind
from services.parts import acquire
from services.parts.tables import SOURCE_EXTERNAL
from tests.support import part_fixtures as part_fx


def contract_slot(
    type_: str = "prompt-set", name: str = "support-tickets", version: str = "3"
) -> dict[str, Any]:
    """A slot document pinning a miDataworks part. **Both source keys stated**, one null."""
    return {
        "contract_reference": {"type": type_, "name": name, "version": version},
        "external": None,
        "floor": None,
    }


def external_slot(
    origin: str = "/data/operator/prompts.jsonl",
    content_sha256: str = "a" * 64,
    retrieved_at: str = "2026-10-14T00:00:00Z",
    floor: float | None = None,
) -> dict[str, Any]:
    """A slot document pinning an external part, in the contract's ``ExternalProvenance`` shape."""
    return {
        "contract_reference": None,
        "external": {
            "source": origin,
            "content_sha256": content_sha256,
            "retrieved_at": retrieved_at,
        },
        "floor": floor,
    }


def retention_slot(
    *, floor: float | None = 0.72, type_: str = "retention-set", name: str = "general"
) -> dict[str, Any]:
    """The retention slot, with the floor that makes a pack valid (R-02.38, E14)."""
    return {
        "contract_reference": {"type": type_, "name": name, "version": "2"},
        "external": None,
        "floor": floor,
    }


def definition(
    *,
    name: str = "notary-md",
    version: str = "1",
    domain: str | None = "maryland notary language",
    slots: Mapping[str, Any] | None = None,
    created_at: datetime | None = None,
) -> PackVersionDefinition:
    """A pack version, built through the real model and the real slot reader.

    ``slots`` is ``{slot kind: slot document | None}``; anything left out is ``None``, which
    is what the artifact states for an empty slot.
    """
    documents = dict(slots or {})
    return PackVersionDefinition(
        name=name,
        version=version,
        domain=domain,
        slots={slot: slot_from_document(slot, documents.get(slot.value)) for slot in SLOT_ORDER},
        created_at=created_at or datetime(2026, 10, 14, tzinfo=UTC),
    )


def valid_slots(**overrides: Any) -> dict[str, Any]:
    """The three required slots filled: prompt set, test set, retention set with its floor."""
    slots = {
        SlotKind.PROMPT_SET.value: contract_slot(),
        SlotKind.TEST_SET.value: contract_slot("test-set", "notary-test", "1"),
        SlotKind.RETENTION_SET.value: retention_slot(),
    }
    slots.update(overrides)
    return slots


def commit(session: Any, pack: PackVersionDefinition) -> PackVersionDefinition:
    """Write and index a pack through the **real** writer (``registry.create``)."""
    return registry.create(session, pack)


def import_contract_part(
    session: Any,
    *,
    parts_root: Path,
    export_root: Path,
    set_kind: str = "prompt_set",
    reference: tuple[str, str, str] | None = ("prompt-set", "support-tickets", "3"),
    rows: int = 3,
) -> str:
    """Import a conformant part through **F10's** real ingest path. Returns its part id.

    ``reference=None`` imports a conformant version that carries **no** contract reference -
    which is the ``part_unpinnable`` case (E5), and which has to be built this way because
    the null is upstream's assertion rather than something a test can fake in the index.
    """
    bodies = {
        "train": b"".join(f'{{"prompt": "p{index}"}}\n'.encode() for index in range(rows)),
    }
    manifest, split_bodies = part_fx.miforge_set_manifest(
        miforge_set_kind=set_kind, reference=reference, split_bodies=bodies
    )
    directory = part_fx.write_export(export_root, manifest, split_bodies)
    result = acquire.from_dir(session, directory, parts_root=parts_root)
    return str(result.part_id)


def import_external_part(
    session: Any,
    *,
    parts_root: Path,
    export_root: Path,
    origin: str = "/data/operator/prompts.jsonl",
    content_sha256: str = "b" * 64,
    retrieved_at: str = "2026-10-14T00:00:00Z",
    set_kind: str = "prompt_set",
    rows: int = 3,
) -> tuple[str, dict[str, Any]]:
    """Import an external part through F10's path. Returns ``(part_id, provenance)``.

    BRD-02's 2026-10-06 amendment accepts an operator's files *"only as a contract-conformant
    export"*, so an external part is a conformant export directory taken in with
    ``source=external_files`` and its own provenance - and its three contract-reference
    columns stay null, which F10 enforces with a CHECK constraint.
    """
    provenance = {
        "source": origin,
        "content_sha256": content_sha256,
        "retrieved_at": retrieved_at,
    }
    # The split bodies are varied by ORIGIN, deliberately. F10 derives a part id from the
    # manifest, so two imports of a byte-identical manifest collapse into one row - and the
    # second one's provenance is silently dropped, keeping the first's. That is reasonable
    # idempotence on F10's side and a trap for a fixture: without this, three "different"
    # external parts in one test are one part with one provenance, and the pack's other two
    # slots resolve to nothing for a reason that looks like a bug in resolution.
    bodies = {
        "train": b"".join(f'{{"prompt": "{origin}-{index}"}}\n'.encode() for index in range(rows))
    }
    manifest, split_bodies = part_fx.miforge_set_manifest(
        miforge_set_kind=set_kind, reference=None, split_bodies=bodies
    )
    directory = part_fx.write_export(export_root, manifest, split_bodies)
    result = acquire.from_dir(
        session,
        directory,
        parts_root=parts_root,
        source=SOURCE_EXTERNAL,
        external_provenance=provenance,
    )
    return str(result.part_id), provenance
