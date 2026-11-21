"""The façade the API and F2, F4, F6 and F8 hold  (FR-007.10, FR-007.28; ADR-016 #1).

**One object, so a capability can be asserted on a live interface rather than by importing a
module.** That distinction is the whole of ADR-016 #1 and it is not pedantry: a sibling
shipped sixteen fully-implemented, unit-tested, documented MCP tools that were never
registered, and every one of those tests passed by importing the module directly. So the
gate, the resolved view and the part-reference record are **methods on this class**, the
reachability test reads them off the class, and a negative control deletes one and requires a
red.

## What each consumer takes from here

* **F6** - :meth:`PackService.require_valid_pack`, **first** in its submit order, before any
  allocation (R-02.53's step 2 of six). F6's own task asserts the call, its payload and its
  call count; until that test exists the seam is **owed, not assumed**.
* **F2 and F4** - :meth:`PackService.resolved_pack`, so neither re-implements resolution.
  This is the seam that stops three features inventing three resolvers.
* **F8** - :meth:`PackService.part_records`, whose elements are the only thing written into
  an artifact's ``parts`` array.
* **F9** - :meth:`PackService.lineage` and :mod:`packs.rendering`'s two forms.

Nothing here adds an outbound call site (ADR-010): contract parts are reached through F10's
one module, and this feature adds none of its own.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from sqlalchemy.orm import Session

from packs import authoring, dryrun, gate, registry, samples, store, transfer
from packs.definition import PackVersionDefinition
from packs.gate import ResolvedPack
from packs.records import PartReferenceRecord
from packs.resolver import Resolution
from packs.slots import SlotKind
from packs.validation import PackValidity, validate


class PackService:
    """Every pack capability, in one place, held by the API and by other features.

    Stateless: each method takes the session it needs. A service object holding a session
    would make a worker's long-lived instance hold a long-lived transaction, which is how a
    row lock outlives the work that took it.
    """

    # ------------------------------------------------------------------ the run-start gate

    def require_valid_pack(self, session: Session, pack_id: str, version: str) -> ResolvedPack:
        """**The gate** (FR-007.10). Refuses unless valid; re-resolves, reads no cached flag."""
        return gate.require_valid_pack(session, pack_id, version)

    def resolved_pack(self, session: Session, pack_id: str, version: str) -> ResolvedPack:
        """The resolved view F2 and F4 consume, without the gate's refusal."""
        return gate.resolved_pack(session, pack_id, version)

    def part_records(
        self, session: Session, pack_id: str, version: str
    ) -> list[PartReferenceRecord]:
        """One record per filled slot - **F8's only route to a ``parts`` element**."""
        return self.resolved_pack(session, pack_id, version).part_records()

    # ------------------------------------------------------------------------ the registry

    def listing(self, session: Session) -> list[dict[str, Any]]:
        return registry.payloads(registry.listing(session))

    def get(self, session: Session, pack_id: str, version: str) -> dict[str, Any]:
        _definition, row = registry.get(session, pack_id, version)
        return row.payload()

    def definition(self, pack_id: str, version: str) -> PackVersionDefinition:
        return store.read(pack_id, version)

    def validity(self, session: Session, pack_id: str, version: str) -> PackValidity:
        """The verdict. Asking whether a pack is valid is a **successful question**: the
        route serving this returns 200 describing invalidity, and the refusals fire where an
        action is attempted."""
        resolved = gate.resolved_pack(session, pack_id, version)
        return resolved.validity

    def resolutions(
        self, session: Session, pack_id: str, version: str
    ) -> Mapping[SlotKind, Resolution]:
        return gate.resolved_pack(session, pack_id, version).resolutions

    def lineage(self, session: Session, pack_id: str, version: str) -> list[dict[str, Any]]:
        """Per slot: source kind, the reference in full, resolution state, floor (FR-007.26)."""
        resolved = gate.resolved_pack(session, pack_id, version)
        # The STORED addressing (FR-007.25), rendered by pack name - the display prefix.
        through = {
            slot: row.split("@", 1)[0]
            for slot, row in registry.addressing(session, pack_id, version).items()
        }
        return registry.lineage(
            resolved.definition, resolved.resolutions, addressed_through=through
        )

    def usage(self, session: Session, pack_id: str, version: str) -> tuple[str, ...]:
        return gate.usage(session, pack_id, version)

    def create(
        self,
        session: Session,
        definition: PackVersionDefinition,
        addressed_through: Mapping[SlotKind, str] | None = None,
    ) -> PackVersionDefinition:
        if addressed_through:
            registry.check_addressing(session, definition, addressed_through)
        created = registry.create(session, definition)
        if addressed_through:
            registry.record_addressing(
                session, definition.name, definition.version, addressed_through
            )
        return created

    def clone(
        self, session: Session, pack_id: str, version: str, *, into: str, new_version: str = "1"
    ) -> PackVersionDefinition:
        return registry.clone(session, pack_id, version, into=into, new_version=new_version)

    def delete(self, session: Session, pack_id: str, version: str) -> None:
        registry.delete(session, pack_id, version)

    def rebuild_index(self, session: Session) -> int:
        """Rebuild the pack index from the files alone. See :func:`packs.registry.rebuild_index`."""
        return registry.rebuild_index(session)

    # ------------------------------------------------------------------ export and import

    def export_bytes(self, pack_id: str, version: str) -> bytes:
        return transfer.export_bytes(store.read(pack_id, version))

    def export_filename(self, pack_id: str, version: str) -> str:
        return transfer.export_filename(store.read(pack_id, version))

    def import_document(
        self, session: Session, payload: bytes | str | Mapping[str, Any]
    ) -> PackVersionDefinition:
        return transfer.import_document(session, payload)

    # --------------------------------------------------------------------------- authoring

    def create_draft(
        self, *, name: str | None = None, version: str = "1", domain: str | None = None
    ) -> authoring.Draft:
        return authoring.create(name=name, version=version, domain=domain)

    def draft(self, draft_id: str) -> authoring.Draft:
        return authoring.get(draft_id)

    def drafts(self) -> list[dict[str, Any]]:
        return authoring.listing()

    def commit_step(self, draft_id: str, number: int, **answers: Any) -> authoring.Draft:
        return authoring.commit_step(draft_id, number, **answers)

    def review_draft(self, session: Session, draft_id: str) -> tuple[authoring.Draft, PackValidity]:
        return authoring.review(session, draft_id)

    def commit_draft(self, session: Session, draft_id: str) -> PackVersionDefinition:
        return authoring.commit(session, draft_id)

    def delete_draft(self, draft_id: str) -> None:
        authoring.delete(draft_id)

    # ----------------------------------------------------------------- panels and the dry-run

    def sample(
        self, session: Session, draft_id: str, slot: SlotKind, *, n: int | None = None
    ) -> samples.SlotSample:
        draft = authoring.get(draft_id)
        return samples.slot_sample(session, authoring.definition_of(draft), slot, n=n)

    def scored_sample(
        self,
        session: Session,
        draft_id: str,
        slot: SlotKind,
        *,
        expectation_config: Mapping[str, Any],
        catalog: Mapping[str, Mapping[str, Any]] | None = None,
        form: str = "selector",
        n: int | None = None,
        provider_name: str | None = None,
        floor: float | None = None,
    ) -> samples.ScoredSample:
        draft = authoring.get(draft_id)
        return samples.scored_sample(
            session,
            authoring.definition_of(draft),
            slot,
            expectation_config=expectation_config,
            catalog=catalog,
            form=form,
            n=n,
            provider_name=provider_name,
            floor=floor,
        )

    def dry_run(
        self,
        session: Session,
        draft_id: str,
        *,
        config: Mapping[str, Any],
        catalog: Sequence[Mapping[str, Any]] = (),
        provider_name: str | None = None,
        n: int | None = None,
    ) -> dryrun.DryRunResult:
        """Sample the pack's **own** prompts, answer them, judge them (AC11).

        The sample comes from the draft's ``prompt_set`` slot, which is what AC11 requires -
        a dry-run on arbitrary prompts measures nothing about this domain. The result is
        cached on the draft **for display only** and is never an input to validity.
        """
        draft = authoring.get(draft_id)
        definition = authoring.definition_of(draft)
        sample = samples.sample_prompts(session, definition, n=n)
        result = dryrun.dry_run(
            sample.prompts, config, catalog=catalog, provider_name=provider_name
        )
        authoring.record_dry_run(draft_id, result.payload())
        return result

    # ------------------------------------------------------------------------- validation

    def validate(
        self, definition: PackVersionDefinition, resolutions: Mapping[SlotKind, Resolution]
    ) -> PackValidity:
        """Pure validity, exposed so a caller holding a resolution does not resolve twice."""
        return validate(definition, resolutions)


#: The process-wide instance. A module-level singleton rather than a dependency-injected one
#: because it is stateless; the reachability test reads the **class**, so a test cannot be
#: fooled by a stub bound here.
pack_service = PackService()
