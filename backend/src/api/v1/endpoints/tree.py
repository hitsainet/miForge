"""``GET /api/v1/tree`` - the forest across runs  (R-02.29; FR-009.21; task 3.4).

**Thin: parse the filters, query, return.** The join is
:func:`services.tree.compose`, reached through :func:`services.reindex.checkpoint_tree`,
which is the one query path into it - so the response this route serves is the same object
foundation task 4.4's reindex-equivalence test compares across a drop-and-rebuild. A route
that assembled its own body from the ORM rows would be outside that guarantee and would
pass every test while breaking the thing the guarantee protects.

## Three things that are explicitly NOT refusals

Each is a legitimate state the tree must render, and a 404 or an empty page for any of them
would hide an artifact while R-02.29 promises the tree **across runs**:

* a node whose run has no row (``run_indexed: false``, FR-009.28);
* a chain that ends ``truncated`` (FR-009.27);
* an adapter that is ``unparented`` (FPRD E-11).

## And one thing that is

**An unparseable filter is refused, never widened to no filter.** A silently widened query
returns more than was asked for and looks like success - which is ADR-018's silent
fabricated value wearing a 200.
"""

from __future__ import annotations

import logging
from typing import Annotated, Any

from fastapi import APIRouter, Depends, Query
from sqlalchemy import select
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.orm import Session

from api.deps import session_dependency
from core.errors import refuse
from models.adapter import Adapter
from models.checkpoint import Checkpoint
from models.run import Run
from services.reindex import checkpoint_tree
from services.tree import (
    DEFAULT_LIMIT,
    INDEX_UNAVAILABLE,
    INVALID_FILTER,
    MAX_LIMIT,
    RUN_NOT_FOUND,
    DomainFilter,
    parse_kind,
)

log = logging.getLogger("miforge.api.tree")

router = APIRouter(tags=["tree"])

#: The three refusal codes, as constants rather than as literals at each raise site. They
#: are re-exported from :mod:`services.tree` where the filter parser also needs them: two
#: spellings of one code is how a client's ``switch`` loses a branch, and this project has
#: paid for a renamed enum value five times.
REFUSAL_CODES: tuple[str, ...] = (INVALID_FILTER, RUN_NOT_FOUND, INDEX_UNAVAILABLE)

#: What a request-level log line carries. FR-009.29's divergences must be visible **in
#: aggregate** and not only per node, so the count rides here; a per-node line already
#: comes from ``services.reindex``' divergence logging.
TREE_SERVED_EVENT = "tree_served"


@router.get("/tree")
def get_tree(  # five filters, each from FTDD section 5.2's signature
    session: Annotated[Session, Depends(session_dependency)],
    run_id: Annotated[str | None, Query(max_length=64)] = None,
    domain: Annotated[str | None, Query(max_length=256)] = None,
    kind: Annotated[str | None, Query(max_length=32)] = None,
    limit: Annotated[int, Query(ge=1, le=MAX_LIMIT)] = DEFAULT_LIMIT,
    offset: Annotated[int, Query(ge=0)] = 0,
) -> dict[str, Any]:
    """The forest: every checkpoint and adapter node with its edges.

    Args:
        run_id: filter to one run's artifacts.
        domain: a domain name, or the literal ``null`` for artifacts asserting no domain.
        kind: ``checkpoint`` or ``adapter``.
        limit, offset: flat pagination over nodes. **Not by subtree**: a node whose edge
            target falls outside the page resolves ``not_loaded``, which is a different
            value from ``truncated``.

    Raises:
        Refusal: ``invalid_filter`` for an unparseable ``domain`` or ``kind``;
            ``run_not_found`` when a ``run_id`` filter names neither a run row nor any
            artifact; ``index_unavailable`` when the index cannot be read.
    """
    domain_filter = DomainFilter.parse(domain)
    kind_filter = parse_kind(kind)

    if run_id is not None:
        _require_known_run(session, run_id)

    try:
        payload = checkpoint_tree(
            session,
            run_id=run_id,
            domain=domain_filter,
            kind=kind_filter,
            limit=limit,
            offset=offset,
        )
    except SQLAlchemyError as exc:
        raise refuse(
            INDEX_UNAVAILABLE,
            "The artifact index (mf_checkpoints, mf_adapters) could not be read. The "
            "manifests on disk are the record, so run reindex to rebuild the "
            "index from them.",
            dependency="postgresql:mf_checkpoints,mf_adapters",
            recovery="reindex",
            detail=str(exc),
        ) from exc

    log.info(
        TREE_SERVED_EVENT,
        extra={
            "nodes": len(payload["nodes"]),
            "total": payload["page"]["total"],
            "index_stale": len(payload["index_stale"]),
            "filters": {"run_id": run_id, "domain": domain, "kind": kind},
        },
    )
    return payload


def _require_known_run(session: Session, run_id: str) -> None:
    """Refuse a ``run_id`` filter that names **neither** a run row nor any artifact.

    Two lookups, and the refusal names which of the two failed. A run row alone is not the
    test: ``reindex`` indexes a valid manifest whose ``run_id`` has no run row (F5's OQ-4),
    and refusing on the row alone would make FR-009.28's whole case unreachable through the
    one filter an operator would use to look at it.
    """
    has_run = session.execute(select(Run.id).where(Run.id == run_id).limit(1)).first() is not None
    if has_run:
        return
    has_artifact = (
        session.execute(select(Checkpoint.name).where(Checkpoint.run_id == run_id).limit(1)).first()
        is not None
        or session.execute(select(Adapter.name).where(Adapter.run_id == run_id).limit(1)).first()
        is not None
    )
    if has_artifact:
        return
    raise refuse(
        RUN_NOT_FOUND,
        f"No run {run_id!r}: it has no run row and no indexed checkpoint or adapter. If "
        f"its artifacts exist on disk, run reindex; otherwise check the id.",
        run_id=run_id,
        lookups_failed=["mf_runs.id", "mf_checkpoints.run_id/mf_adapters.run_id"],
    )
