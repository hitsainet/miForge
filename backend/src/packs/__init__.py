"""Domain packs  (F7; BRD-02 R-02.35, R-02.38, R-02.39, R-02.40).

> **A pack is a named, versioned composition of the parts one domain's runs need:** a prompt
> set, optionally a corpus, optionally preference pairs, a selector and/or grader
> configuration, a test set, and a retention set with its floor (R-02.35).

**A pack holds references to parts, not copies of them.** A pack version pins each part's
exact version, so a pack version names a **reproducible run input**. That single sentence is
the feature.

## The one mistake this package must not repeat

**The word "pack" was coined in miForge's UI mockup, not in a requirement.** It appears
nowhere in BRD-02 as originally supplied, and twice it was then treated as settled
requirement it was not - once by inferring a "BRD-02 v0.3" that never existed, once by
writing that a pack *"was a versioned miDataworks bundle delivered under
midataworks.dataset-version/v1 (BRD-03 R-03.47, R-03.48)"* when **the cited requirements
never said that**: R-03.48 lists six set kinds and no pack, R-03.47's schema has no pack
field, and miDataworks ships a test asserting its absence. The operator settled it on
2026-10-12: **packs are real and miForge assembles them.**

The check that catches that class of error is cheap and it is the one used throughout this
package: **verify that a cited requirement, or a cited schema, says what you claim it says.**
Several claims in F7's own documents did not survive it, and each correction is recorded in
the module that found it rather than in a summary:

* :mod:`packs.records` - the ``parts`` element shape is settled by ``batch-record-v1.json``
  and the documents' shape would not validate.
* :mod:`packs.references` - there is no ``local_version`` field to put an ordinal in; the
  content hash is the pin.
* :mod:`packs.external` - F10 already takes external parts in; F7 does not rebuild it.
* :mod:`packs.store` - the gate contract made a pack an on-disk file, so ``reindex`` **can**
  rebuild one, which reverses the documents' standing note.

## Three distinctions carry this feature, and all three have been collapsed once already

1. **Reference vs copy** - :mod:`packs.references` has no content field, so there is nowhere
   to put a copy.
2. **Contract form vs display form** - :mod:`packs.rendering` has two functions, one
   direction and no parser.
3. **The authoring dry-run vs the in-run review queue** - :mod:`packs.dryrun` is
   synchronous, in-request, and has no queue. miForge keeps no review queue of its own
   (R-02.36); that is miDataworks'.

The code makes each one hard to collapse rather than relying on a reviewer noticing.
"""

from __future__ import annotations

from packs.gate import ResolvedPack, require_valid_pack, resolved_pack
from packs.records import PackVersionIdentity, PartReferenceRecord, part_reference_record
from packs.references import ContractPartRef, ExternalPartRef, PartRef
from packs.rendering import DisplayRef, contract_form, display_form, external_form
from packs.service import PackService, pack_service
from packs.slots import REQUIRED_SLOTS, SLOT_ORDER, SlotKind
from packs.validation import PackValidity, validate

__all__ = [  # noqa: RUF022 - grouped by the seam each name serves, not alphabetised
    # The gate (FR-007.10) - F6's call site, asserted on `PackService` as well as here.
    "require_valid_pack",
    "resolved_pack",
    "ResolvedPack",
    # The part-reference record (FR-007.32) - F8's only route to a `parts` element.
    "part_reference_record",
    "PartReferenceRecord",
    "PackVersionIdentity",
    # The two rendering layers (FR-007.27) - F9's part renderer.
    "contract_form",
    "display_form",
    "external_form",
    "DisplayRef",
    # The reference types (FR-007.6) - distinct, with no conversion between them.
    "ContractPartRef",
    "ExternalPartRef",
    "PartRef",
    # Slots and validity.
    "SlotKind",
    "SLOT_ORDER",
    "REQUIRED_SLOTS",
    "validate",
    "PackValidity",
    # The façade.
    "PackService",
    "pack_service",
]
