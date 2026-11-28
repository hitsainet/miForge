"""Export and import  (task 6.x; FR-007.14, FR-007.15, AC10's second half).

**By value.** An export carries the pack's own content - identity, version, domain, slots,
sources, provenance, floor - in full, with **no pointer back to this installation's
database, API or filesystem**. It does not carry part *content* (FR-007.4): a reference
travelling intact is the point, and a pack that copied its parts would fork from them
immediately.

**The bytes are the canonical bytes.** :func:`export_bytes` returns exactly what
:func:`core.canonical.canonical_bytes` produces for the document, which is the same byte
string ``definition_sha256`` was taken over and the same one :mod:`packs.store` wrote. So an
export is **verifiable**: a reader can hash what it received and compare. Two serialisers
would be two answers (ADR-005).

**Import never rewrites a reference to make it resolve** (E8). A pack whose parts this
installation cannot resolve is imported, stored, and *reported* unresolvable - because
rewriting a reference would make the imported pack a different pack, and AC10's claim is
that the imported version *yields the same run inputs*.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from core.canonical import canonical_bytes
from packs import registry, store
from packs.definition import PackVersionDefinition
from packs.errors import PACK_DOCUMENT_INVALID, PackRefusal


#: The filename an export is offered under, in ``Content-Disposition``. Carries the pack's
#: identity so a directory of exports is readable; nothing parses it back.
def export_filename(definition: PackVersionDefinition) -> str:
    """``<name>@<version>.pack.json``."""
    return f"{definition.name}@{definition.version}.pack.json"


def export_bytes(definition: PackVersionDefinition) -> bytes:
    """The pack document as canonical bytes - **the same bytes a hash is taken over**."""
    return canonical_bytes(definition.document())


def import_document(
    session: Any, payload: bytes | str | Mapping[str, Any]
) -> PackVersionDefinition:
    """Import an exported pack. Validated, then stored, then indexed - **or nothing**.

    Args:
        session: An open session.
        payload: The export's bytes, its text, or an already-parsed document.

    Returns:
        The imported pack version.

    Raises:
        PackRefusal: ``pack_document_invalid`` for anything that is not a conforming pack
            document - including a future ``schema_version`` and a digest that does not
            match its own body (E9) - naming the schema and what failed;
            ``pack_version_committed`` when this installation already holds that version,
            because an import that silently overwrote one would make every run citing it
            mean something else.

    **Atomicity** (FPRD §8: *"an import is atomic: a pack version appears whole or not at
    all"*): the document is validated before anything is written, and if indexing fails the
    file just written is removed. The alternative - a file on disk with no row - would be
    repaired by a rebuild, but "repaired later" is not the same promise as "whole or not at
    all", and an operator who saw an import fail must not find the pack half present.
    """
    document = _parse(payload)
    # `from_document` validates - the declared version first, then the schema, then the
    # digest - and nothing here validates ahead of it. An extra `validate_document` call
    # before this line looks like belt and braces and is not: the schema's `schema_version`
    # is a `const`, so it would report a future document as *"'2' was expected to be '1'"* at
    # a JSON pointer and the version-specific refusal would never be reached.
    definition = PackVersionDefinition.from_document(document)

    store.write_committed(definition)
    try:
        registry.index(session, definition)
    except BaseException:
        store.delete(definition.name, definition.version)
        raise
    return definition


def _parse(payload: bytes | str | Mapping[str, Any]) -> Mapping[str, Any]:
    """Read the payload as a JSON object, refusing anything else by name.

    A file that is not JSON, or is JSON but not an object, is refused here rather than
    reaching the schema validator - whose message would be about a type mismatch at the
    document root and would not tell the operator they handed over the wrong file (E9).
    """
    if isinstance(payload, Mapping):
        return payload
    import json

    text = payload.decode("utf-8") if isinstance(payload, bytes) else payload
    try:
        parsed: Any = json.loads(text)
    except ValueError as error:
        raise PackRefusal(
            PACK_DOCUMENT_INVALID,
            f"This file is not JSON, so it is not a pack document: {error}. An export is a "
            f"single JSON object written by miForge; export the pack again from the "
            f"installation that holds it.",
            schema="pack-definition-v1.json",
            detail=str(error),
        ) from error
    if not isinstance(parsed, dict):
        raise PackRefusal(
            PACK_DOCUMENT_INVALID,
            f"This file holds a JSON {type(parsed).__name__}, not an object, so it is not "
            f"a pack document.",
            schema="pack-definition-v1.json",
        )
    return parsed
