"""Answer models for archive, unarchive and purge (feature 021, P3).

Documentation only: each model describes what ``api/v1/endpoints/archive.py`` and ``purge.py``
answer today and is declared on the route's ``responses``; no route validates against it (MF9).
"""

from pydantic import BaseModel

from schemas.answers_common import may_be_absent

# ===================================================================== archive


class ArchiveRunAnswer(BaseModel):
    """A run's archive state, for archive and unarchive (``services/archive.py:_run_answer``)."""

    run_id: str
    status: str
    archived: bool
    archived_at: str | None
    changed: bool


class ArchivePartAnswer(BaseModel):
    """A part's archive state, for archive and unarchive (``services/archive.py:_part_answer``)."""

    part_id: str
    archived: bool
    archived_at: str | None
    changed: bool


# ======================================================================= purge


class PurgeBlocker(BaseModel):
    """One thing that stops a purge (``services/purge/service.py:_Plan.block``, ``_missing``)."""

    code: str
    message: str
    #: ``run_active``, ``lease_held``, ``run_active_reference``, ``run_reference``,
    #: ``lineage_reference``.
    run_id: str | None = may_be_absent()
    #: ``lineage_reference``.
    artifact: str | None = may_be_absent()
    #: ``approval_pending``.
    approval_id: str | None = may_be_absent()
    #: ``loaded_in_millm``.
    resident_model: str | None = may_be_absent()
    #: ``pack_pins_it``.
    pack: str | None = may_be_absent()
    #: ``chain_active`` (``services/chains/service.py:purge_blockers``, a contributed check).
    chain_id: str | None = may_be_absent()


class PurgeNotArchived(BaseModel):
    """Why an unarchived item cannot be purged (``services/purge/service.py:_Plan.payload``)."""

    code: str
    message: str


class PurgeRemoves(BaseModel):
    """What a purge would remove (``services/purge/service.py:_Plan.payload``, ``_missing``)."""

    #: Rows per table name; a run's and a part's tables differ.
    rows: dict[str, int]
    row_total: int
    files: int
    bytes: int
    directories: list[str]


class PurgeKept(BaseModel):
    """A path a purge leaves in place, and why (``services/purge/service.py:_plan_run``,
    ``_plan_part``)."""

    path: str
    why: str


class PurgeMillm(BaseModel):
    """What miServer is serving, as read for the loaded-checkpoint blocker
    (``services/purge/service.py:millm_resident``); empty for a part, and for an item not
    planned (``_missing``)."""

    checked: bool | None = may_be_absent()
    #: When ``checked`` is false.
    note: str | None = may_be_absent()
    #: When ``checked`` is true.
    resident_model: str | None = may_be_absent()


class PurgePlanItem(BaseModel):
    """One item's purge plan (``services/purge/service.py:_Plan.payload``, or ``_missing`` for
    an item not found or not plannable)."""

    kind: str
    id: str
    found: bool
    name: str
    #: Planned items only; null for a part.
    status: str | None = may_be_absent()
    archived: bool
    #: Planned items only.
    archived_at: str | None = may_be_absent()
    purgeable: bool
    not_archived: PurgeNotArchived | None
    blockers: list[PurgeBlocker]
    removes: PurgeRemoves
    kept: list[PurgeKept]
    millm: PurgeMillm


class PurgePreviewAnswer(BaseModel):
    """``POST /runs/purge/plan`` and ``/parts/purge/plan``
    (``api/v1/endpoints/purge.py:preview_runs``, ``preview_parts``)."""

    kind: str
    items: list[PurgePlanItem]


class PurgeTombstoneDetail(BaseModel):
    """A tombstone's ``detail`` (``services/purge/service.py:_tombstone``)."""

    status: str | None
    archived_at: str | None
    directories: list[str]
    kept: list[PurgeKept]
    millm: PurgeMillm


class PurgeTombstone(BaseModel):
    """One tombstone (``services/purge/service.py:tombstone_payload``)."""

    tombstone_id: str
    kind: str
    id: str
    name: str
    purged_by: str
    reason: str | None
    #: Rows removed per table name.
    rows_removed: dict[str, int]
    files_removed: int
    bytes_removed: int
    detail: PurgeTombstoneDetail
    purged_at: str


class PurgeItemAnswer(PurgeTombstone):
    """One purged item (``services/purge/service.py:_finish``), the answer of
    ``POST /runs/{run_id}/purge`` and ``/parts/{part_id}/purge``."""

    outcome: str
    files_left: list[str]


class PurgeManyResult(BaseModel):
    """One item of a selection purge: ``_finish``'s answer when purged, or the blocked shape
    (``services/purge/service.py:purge_many``)."""

    kind: str
    id: str
    outcome: str
    #: Purged: the tombstone's reason (may be null). Blocked: the refusal's message.
    reason: str | None
    #: Blocked only.
    code: str | None = may_be_absent()
    #: Blocked only.
    blockers: list[PurgeBlocker] | None = may_be_absent()
    #: Purged only.
    tombstone_id: str | None = may_be_absent()
    #: Purged only.
    name: str | None = may_be_absent()
    #: Purged only.
    purged_by: str | None = may_be_absent()
    #: Purged only.
    rows_removed: dict[str, int] | None = may_be_absent()
    #: Purged only.
    files_removed: int | None = may_be_absent()
    #: Purged only.
    bytes_removed: int | None = may_be_absent()
    #: Purged only.
    detail: PurgeTombstoneDetail | None = may_be_absent()
    #: Purged only.
    purged_at: str | None = may_be_absent()
    #: Purged only.
    files_left: list[str] | None = may_be_absent()


class PurgeManyAnswer(BaseModel):
    """``POST /runs/purge`` and ``/parts/purge`` (``services/purge/service.py:purge_many``)."""

    kind: str
    results: list[PurgeManyResult]
    purged: int
    blocked: int
    bytes_removed: int


class PurgeListPurgesAnswer(BaseModel):
    """``GET /purges`` (``api/v1/endpoints/purge.py:list_purges``)."""

    items: list[PurgeTombstone]
    total: int
