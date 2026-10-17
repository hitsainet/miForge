"""F7's refusals  (FTDD §5; ADR-013, ADR-018, FR-007.28).

**One exception type and nine named codes, each spelled exactly once.** A literal code at a
raise site is a code that can be misspelled, and a misspelled code is a refusal no client
matches on - so every raise in this package names a constant from this module. A static test
asserts there is no bare string literal refusal code and no ``HTTPException`` anywhere under
``packs/``.

:class:`PackRefusal` subclasses :class:`core.errors.Refusal` so the application-level
handler installed in :mod:`api.app` renders it as the one envelope. A bare
``HTTPException`` here would bypass that handler and put a **second** error shape on the
same API, and a client that has to recognise two shapes is a client that will recognise one.

## The distinction this module exists to keep

``MIDATAWORKS_NOT_CONFIGURED`` is a **503** and ``PACK_PART_UNRESOLVED`` is a **422**.
One says *the service is absent*; the other says *your pack is wrong*. A standalone operator
must never be told the second when the first is true - that is R-02.50's convention, and the
FTDD calls it *"the single most likely thing to be implemented carelessly"* in this feature.
``validity`` therefore reports an absent service in its own field, and nothing folds it into
the missing-slot list.
"""

from __future__ import annotations

from typing import Any

from core.errors import Refusal

# ---------------------------------------------------------------------------- the codes
#
# Nine, from FTDD §5's taxonomy. `MIDATAWORKS_NOT_CONFIGURED` is ADR-010's
# `<service>_not_configured` spelling and is **not** F7's invention: it is the same string
# F10 raises, which is why it is written out here rather than built from the service name -
# a built string is a string a grep for the code does not find.

#: A required slot is empty. Carries **every** missing slot, never the first (AC10).
PACK_INCOMPLETE = "pack_incomplete"
#: A pinned reference does not resolve at its pinned version. The pack must change.
PACK_PART_UNRESOLVED = "pack_part_unresolved"
#: The service that could resolve a contract part is absent. **Nothing is wrong with the
#: pack.** 503, naming the service and the capability.
MIDATAWORKS_NOT_CONFIGURED = "midataworks_not_configured"
#: A dry-run cannot generate. Advisory: the step still saves (E7).
PROVIDER_NOT_AVAILABLE = "provider_not_available"
#: An edit to a committed version. Editing produces a new version (E10).
PACK_VERSION_COMMITTED = "pack_version_committed"
#: Deleting a pack a run used (E11).
PACK_IN_USE = "pack_in_use"
#: An import that is not a conforming pack document, or one of a future schema version
#: (E9). Names the schema and what failed.
PACK_DOCUMENT_INVALID = "pack_document_invalid"
#: A part whose ``contract_reference`` is null: conformant upstream, and it names nothing
#: reproducible, so it cannot pin a slot (E5). **F10 spells this code too** - it builds the
#: refusal in `services/parts/resolve.py` so the spelling is shared across the seam - and
#: F7 is the side that raises it.
PART_UNPINNABLE = "part_unpinnable"
#: The retention slot is filled and its floor is absent. A **distinct** reason from a
#: missing retention set, because the operator's fix is different (E14).
RETENTION_FLOOR_MISSING = "retention_floor_missing"

#: Every code this feature raises. Used by the taxonomy test, which asserts the set is
#: exactly nine and that each appears at a raise site - a code nothing raises is a code a
#: client was told to handle for nothing.
PACK_REFUSAL_CODES: frozenset[str] = frozenset(
    {
        PACK_INCOMPLETE,
        PACK_PART_UNRESOLVED,
        MIDATAWORKS_NOT_CONFIGURED,
        PROVIDER_NOT_AVAILABLE,
        PACK_VERSION_COMMITTED,
        PACK_IN_USE,
        PACK_DOCUMENT_INVALID,
        PART_UNPINNABLE,
        RETENTION_FLOOR_MISSING,
    }
)

#: Codes that mean *a service is absent*, not *the pack is wrong*. Kept as a set so the
#: 503-vs-422 split is a property of the code rather than a judgement at each call site.
SERVICE_ABSENT_CODES: frozenset[str] = frozenset(
    {MIDATAWORKS_NOT_CONFIGURED, PROVIDER_NOT_AVAILABLE}
)


class PackRefusal(Refusal):
    """A named refusal from the pack feature. The only exception type ``packs/`` raises."""

    def __init__(self, code: str, message: str, **details: Any) -> None:
        if code not in PACK_REFUSAL_CODES:
            # Refused rather than allowed through: an unregistered code would be a tenth
            # refusal nobody documented, and the taxonomy test could not see it.
            raise ValueError(
                f"{code!r} is not one of F7's nine refusal codes. Add it to "
                f"packs/errors.py and to the FTDD's taxonomy table first, so the API's "
                f"documented refusals and the ones it raises cannot disagree. "
                f"Known: {sorted(PACK_REFUSAL_CODES)}."
            )
        super().__init__(code, message, details or None)

    @property
    def names_an_absent_service(self) -> bool:
        """Whether this refusal is *the service is absent* rather than *the pack is wrong*."""
        return self.code in SERVICE_ABSENT_CODES
