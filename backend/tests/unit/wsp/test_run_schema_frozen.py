"""The run contract changes only by a recorded revision (W-9, FR-019.21; 2026-11-25).

``docs/schemas/miforge-wsp-run-v1.json`` is vendored byte-identical by miWorkbench
(``backend/tests/unit/test_vendored_wsp_schemas.py`` there), so any change to it is a contract
revision that miWorkbench must re-vendor, in the push order miForge first. This pins the file's
sha256, so an unrecorded change goes red here before it goes red in miWorkbench's cross-repo job.

History:

- ``a3fc4010...`` (read 2026-11-24 at ``32b6fa8``): the form 019 kept unchanged. 019 put its
  window and ceiling on miForge's own rows and never in ``run_document``; a run with every 019
  column set still validates (``integration/wsp/test_dispatch.py``,
  ``test_a_run_with_every_019_column_set_is_a_valid_run_document``).
- Revision 2 (2026-11-25, ``0xcc/reviews/run_contract_r2_2026-11-25.md``), additive:
  ``FrozenRequest.python`` (018 FU-8, F18-D15; present only as ``"off"``) and the document's
  ``submitted_check`` with ``SubmittedCheck`` and ``FigureCheck`` (018 FU-2; present only when
  the intake stored one). Both optional, so every earlier document and request still validates.
- Revision 3 (2026-11-26, feature 021, FR-021.9; operator's choice (b)): the root ``$id`` became
  the absolute ``https://hitsai.net/schemas/miforge-wsp-run-v1.json`` (it was the bare file name),
  so the published answer contract can reference the document by URI. No validation rule changed;
  every document and request valid before is valid after. miWorkbench re-vendors both copies.
- Revision 4 (2026-11-27, ``0xcc/reviews/wsp_window_read_not_resident_2026-11-27.md``),
  descriptions only: ``submitted_check`` is now also stored, with ``measured`` false, for a
  completed inspect that submitted with ``forward_passes_used`` 0 (run ``wsp_0971152b6843``), and
  the three descriptions that said inspect never carries one say so. No validation rule changed:
  the check uses the existing ``SubmittedCheck`` shape. miWorkbench re-vendors, miForge first.
  **Folded into revision 4 before it was pushed** (2026-11-28, the same branch, fix 8; operator
  2026-11-27 ~22:10 ET): the additive, optional ``Head.identical_to_parent`` (a ``baseline:*``
  head's flag that its ``scores_sha256`` equals the parent head's on the same set). Not
  required, so every stored verdict and snapshot still validates. Revision 4 was never pushed
  and miWorkbench never vendored its first form (``e9d33e5d...``), so one re-vendor takes both.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

SCHEMA = Path(__file__).resolve().parents[4] / "docs" / "schemas" / "miforge-wsp-run-v1.json"
PRE_019_SHA256 = "a3fc40105d095d32737e8ea3976d94b743df047790cd4d5ae6ba17218e88c358"
REVISION_2_SHA256 = "5a8870118fcdad34fbea2e59e29e6a14e41feee74bb7eb071f4b20ca08a66351"
REVISION_3_SHA256 = "476d34efeef0797e720ca893991f109c8b02fc1777ca3c62089bae1e31bcec3e"
#: Revision 4 as first built (2026-11-27), never pushed or vendored.
REVISION_4_FIRST_SHA256 = "e9d33e5d8e34de86a19944ef46c32ea1f731582ee81e116ebbb7b94c43bb68f7"
REVISION_4_SHA256 = "05fc289aa9420d1388a1646f9df3dc739a03cb1e4fd058e28636a81ab8e4a5bb"


def test_the_run_schema_is_byte_identical_to_its_recorded_revision() -> None:
    assert hashlib.sha256(SCHEMA.read_bytes()).hexdigest() == REVISION_4_SHA256
