"""Purge: permanently delete an ARCHIVED run or part  (operator decision 2026-10-20; 0023).

Archive (``services.archive``) hides; purge removes. The rules, each enforced here, in the
one place both the single-item and the batch routes reach:

1. **Only an archived item.** Anything else is refused ``not_archived`` (409). The run's
   final ``DELETE`` carries ``archived_at IS NOT NULL AND status IN terminal`` in its
   ``WHERE``, so a run continued (which unarchives it) between the plan and the delete is
   refused rather than removed.
2. **Operator only.** Enforced at the route (``api.v1.endpoints.purge``): a request carrying
   ``X-Miforge-Agent`` is refused ``purge_operator_only`` (403), and no MCP tool calls it.
3. **The item and what belongs only to it.** A run: its row, steps, checkpoint and adapter
   index rows, scores and score items, batch records, review binding and candidate refs,
   released lease rows, the merge records whose result it owns, and on disk its checkpoint
   directory (checkpoints, adapters), its trainer-state directory and its run-log directory
   (``run.jsonl``: steps, events, definitions). A part: its index row, split rows, archive
   row and its directory under the parts root. Bytes are measured before anything moves.
4. **Refused while anything outside the purge still needs it** - ``purge_blocked`` (409),
   every blocker named: a pending or running run (itself, or one that names it), a
   non-archived run that resumes, loads or branches from it, a checkpoint or adapter of
   another live run whose lineage runs through it, an active chain that contains or names
   it, a pending approval that names it, an unreleased lease, a checkpoint miLLM is serving
   (when miLLM is configured and answers - otherwise the answer says it was not checked),
   and for a part, every pack that pins it and every live run that used it.
5. **One append-only tombstone per purge** (``mf_purges``): what, by whom, why, the rows,
   files and bytes.

**Order of destruction** - the checkpoint pruner's, for the same reason (see
``services.checkpoints.pruning``): every directory is first **renamed** to a hidden
``.tmp-save-purge-<uuid>`` sibling, which every discovery path already skips, then the rows
are deleted and the tombstone written in **one commit**, then the staged directories are
removed. A database failure renames them back, so a refused or failed purge leaves both the
rows and the files exactly as they were. A directory that cannot be removed after the commit
is reported by path (``files_left``) and stays hidden.

**Only what miForge owns is deleted.** A checkpoint directory under a read-only
``CHECKPOINT_READ_ROOTS`` root (miLLM's, for an attached run) is listed under ``kept``, never
touched. A path that is a symlink, or resolves outside its root, is refused
``purge_path_unsafe`` before anything moves.
"""

from __future__ import annotations

import logging
import os
import shutil
import uuid
from collections.abc import Iterable, Iterator, Mapping, Sequence
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, cast

import sqlalchemy as sa
from sqlalchemy.orm import Session

from core.atomic_dir import TMP_PREFIX
from core.config import get_settings
from core.errors import Refusal, refuse
from core.paths import single_segment
from models.adapter import Adapter
from models.approval import Approval
from models.checkpoint import Checkpoint
from models.pack import Pack, PackPart
from models.run import Step
from services.checkpoints.paths import parse_qualified_name
from services.parts.tables import (
    CONTRACT_PARTS,
    PART_ARCHIVE,
    PART_SPLITS,
    REVIEW_BINDINGS,
    REVIEW_CANDIDATE_REFS,
    SOURCE_EXTERNAL,
)
from services.purge import blockers as contributed_blockers
from services.purge.tables import PURGES
from services.scoring.tables import SCORE_ITEMS, SCORES
from training.runtypes.tables import MERGES
from training.signals.batch_index import BATCHES
from workers.run_record import LEASES, RUNS, TERMINAL_STATES, as_aware, rowcount_of, utcnow

log = logging.getLogger("miforge.purge")

#: Who purged. There is no login (suite decision 2026-10-06) and an agent is refused, so
#: every purge that reaches the service is the operator's.
OPERATOR = "operator"
#: The staging prefix: F5's ``TMP_PREFIX`` so every discovery path skips it, extended so an
#: operator finding an orphan can tell a purge from a save or a prune.
STAGE_PREFIX = f"{TMP_PREFIX}purge-"
#: A batch is a selection on one screen; the run list's largest page is 200.
MAX_BATCH = 200

ACTIVE_RUN_STATES: frozenset[str] = frozenset({"pending", "running"})
OPEN_APPROVAL_STATES: frozenset[str] = frozenset({"pending", "executing"})

_STEPS = cast(sa.Table, Step.__table__)
_CHECKPOINTS = cast(sa.Table, Checkpoint.__table__)
_ADAPTERS = cast(sa.Table, Adapter.__table__)
_APPROVALS = cast(sa.Table, Approval.__table__)
_PACKS = cast(sa.Table, Pack.__table__)
_PACK_PARTS = cast(sa.Table, PackPart.__table__)


# ============================================================================ helpers


def _iso(value: Any) -> str | None:
    aware = as_aware(value)
    return None if aware is None else aware.isoformat()


def _strings(value: Any) -> Iterator[str]:
    """Every string leaf of a JSON value, keys excluded."""
    if isinstance(value, str):
        yield value
    elif isinstance(value, Mapping):
        for item in value.values():
            yield from _strings(item)
    elif isinstance(value, (list, tuple)):
        for item in value:
            yield from _strings(item)


def _run_id_values(value: Any) -> Iterator[str]:
    """Every value stored under a ``run_id`` key anywhere in a JSON value."""
    if isinstance(value, Mapping):
        for key, item in value.items():
            if key == "run_id" and isinstance(item, str):
                yield item
            else:
                yield from _run_id_values(item)
    elif isinstance(value, (list, tuple)):
        for item in value:
            yield from _run_id_values(item)


def names_run(name: str | None, run_id: str) -> bool:
    """Whether a qualified artifact name (``<base>:<run>:step-N`` / ``:adapter-N``) is the run's."""
    if not name:
        return False
    parsed = parse_qualified_name(name)
    if parsed is not None:
        return parsed[1] == run_id
    parts = name.rsplit(":", 2)
    return len(parts) == 3 and parts[1] == run_id and parts[2].startswith("adapter-")


def _run_roots() -> list[Path]:
    """Every root under which a directory named for a run is that run's."""
    from training.tasks import run_log_root

    settings = get_settings()
    return [
        settings.checkpoint_root,
        *settings.checkpoint_read_root_paths,
        settings.trainer_state_root,
        run_log_root(),
    ]


def _path_under_run(text: str, run_id: str, roots: Sequence[Path]) -> bool:
    if "/" not in text:
        return False
    candidate = Path(text)
    for root in roots:
        try:
            relative = candidate.relative_to(root)
        except ValueError:
            continue
        if relative.parts and relative.parts[0] == run_id:
            return True
    return False


def mentions_run(value: Any, run_id: str, roots: Sequence[Path]) -> bool:
    """Whether a stored definition or payload names ``run_id``.

    By a ``run_id`` key (``resume_from``, a branch source), a qualified checkpoint or adapter
    name, or a path into the run's own directory under one of the run roots. Never by bare
    substring: run ids are short and free-form, and ``base`` inside ``base-2`` is not a
    reference.
    """
    if any(found == run_id for found in _run_id_values(value)):
        return True
    return any(
        names_run(text, run_id) or _path_under_run(text, run_id, roots) for text in _strings(value)
    )


def _measure(directory: Path) -> tuple[int, int]:
    """``(files, bytes)`` under ``directory``, never following a link."""
    files = 0
    total = 0
    for current, dirnames, filenames in os.walk(directory, followlinks=False):
        for name in filenames + [d for d in dirnames if os.path.islink(os.path.join(current, d))]:
            try:
                total += os.lstat(os.path.join(current, name)).st_size
            except OSError:
                continue
            files += 1
    return files, total


def _owned_dir(root: Path, name: str, *, what: str) -> Path:
    """``root/name`` if it is safe to delete: one segment, not a link, inside ``root``."""
    single_segment(name, code="purge_path_unsafe", what=what)
    candidate = root / name
    if candidate.is_symlink():
        raise refuse(
            "purge_path_unsafe",
            f"{candidate} is a symbolic link. A purge deletes only directories miForge wrote, "
            f"and never follows a link to whatever it points at.",
            path=str(candidate),
        )
    base = os.path.realpath(root)
    resolved = os.path.realpath(candidate)
    if not resolved.startswith(base.rstrip(os.sep) + os.sep):
        raise refuse(
            "purge_path_unsafe",
            f"{candidate} resolves outside {base}; refusing to delete {what}.",
            path=str(candidate),
            root=base,
        )
    return candidate


@dataclass
class _Plan:
    """What one purge would remove and what stops it. Rendered by :meth:`payload`."""

    kind: str
    item_id: str
    name: str
    archived: bool
    archived_at: str | None
    status: str | None = None
    blockers: list[dict[str, Any]] = field(default_factory=list)
    rows: dict[str, int] = field(default_factory=dict)
    directories: list[Path] = field(default_factory=list)
    files: int = 0
    bytes: int = 0
    kept: list[dict[str, str]] = field(default_factory=list)
    millm: dict[str, Any] = field(default_factory=dict)

    def block(self, code: str, message: str, **details: Any) -> None:
        entry = {"code": code, "message": message, **details}
        if entry not in self.blockers:
            self.blockers.append(entry)

    def payload(self) -> dict[str, Any]:
        refusal = None
        if not self.archived:
            refusal = {
                "code": "not_archived",
                "message": f"{self.name} is not archived. Archive it first; only an archived "
                f"{self.kind} can be deleted permanently.",
            }
        return {
            "kind": self.kind,
            "id": self.item_id,
            "found": True,
            "name": self.name,
            "status": self.status,
            "archived": self.archived,
            "archived_at": self.archived_at,
            "purgeable": self.archived and not self.blockers,
            "not_archived": refusal,
            "blockers": list(self.blockers),
            "removes": {
                "rows": dict(self.rows),
                "row_total": sum(self.rows.values()),
                "files": self.files,
                "bytes": self.bytes,
                "directories": [str(path) for path in self.directories],
            },
            "kept": list(self.kept),
            "millm": dict(self.millm),
        }


def _count(session: Session, table: sa.Table, *where: Any) -> int:
    return int(
        session.execute(sa.select(sa.func.count()).select_from(table).where(*where)).scalar_one()
    )


def millm_resident() -> dict[str, Any]:
    """What miLLM is serving now, for the "checkpoint loaded in miLLM" blocker.

    ``checked: false`` with a note when that is not knowable - miLLM not configured, or not
    answering. Read once per request and shared by every item in it.
    """
    from training import acquisition

    try:
        status = acquisition.attachment_status()
    except Exception as exc:  # a read that cannot be made is "not checked"
        return {"checked": False, "note": f"miLLM status read failed ({type(exc).__name__})"}
    if not status.get("configured"):
        return {"checked": False, "note": "miLLM is not configured; nothing can be loaded there"}
    if status.get("reachable") is False:
        return {
            "checked": False,
            "note": f"miLLM did not answer ({status.get('reason')}); not checked",
        }
    resident = status.get("resident_model")
    return {"checked": True, "resident_model": resident if isinstance(resident, str) else None}


# =============================================================================== runs


def _run_name(row: Any) -> str:
    return f"run {row.id} ({row.run_type}, {row.status})"


def _plan_run(session: Session, row: Any, millm: Mapping[str, Any]) -> _Plan:
    run_id = str(row.id)
    plan = _Plan(
        kind="run",
        item_id=run_id,
        name=_run_name(row),
        archived=row.archived_at is not None,
        archived_at=_iso(row.archived_at),
        status=str(row.status),
        millm=dict(millm),
    )
    roots = _run_roots()

    # ---- blockers ----
    if row.status in ACTIVE_RUN_STATES:
        plan.block("run_active", f"Run {run_id} is {row.status}.", run_id=run_id)
    if session.execute(
        sa.select(LEASES.c.id).where(
            LEASES.c.holder_run_id == run_id, LEASES.c.released_at.is_(None)
        )
    ).first():
        plan.block(
            "lease_held",
            f"Run {run_id} still holds a GPU lease that has not been released.",
            run_id=run_id,
        )

    others = session.execute(
        sa.select(
            RUNS.c.id,
            RUNS.c.status,
            RUNS.c.archived_at,
            RUNS.c.definition_submitted,
            RUNS.c.definition_resolved,
        ).where(RUNS.c.id != run_id)
    ).all()
    run_state = {str(o.id): (str(o.status), o.archived_at) for o in others}
    for other in others:
        named = [other.definition_submitted, other.definition_resolved]
        if not any(mentions_run(value, run_id, roots) for value in named if value is not None):
            continue
        if other.status in ACTIVE_RUN_STATES:
            plan.block(
                "run_active_reference",
                f"Run {other.id} is {other.status} and resumes, loads or branches from this run.",
                run_id=str(other.id),
            )
        elif other.archived_at is None:
            plan.block(
                "run_reference",
                f"Run {other.id} (not archived) resumes, loads or branches from this run. "
                f"Archive or purge it first.",
                run_id=str(other.id),
            )

    def lineage(kind: str, name: str, owner: str, through: str) -> None:
        state = run_state.get(owner)
        if state is not None and state[1] is not None and state[0] not in ACTIVE_RUN_STATES:
            return  # an archived run's artifact; its own purge decides about it
        whose = "has no run row" if state is None else f"is {state[0]}, not archived"
        plan.block(
            "lineage_reference",
            f"{kind} {name} of run {owner} ({whose}) descends from {through}.",
            run_id=owner,
            artifact=name,
        )

    for child in session.execute(
        sa.select(_CHECKPOINTS.c.name, _CHECKPOINTS.c.run_id, _CHECKPOINTS.c.parent).where(
            _CHECKPOINTS.c.run_id != run_id, _CHECKPOINTS.c.parent.is_not(None)
        )
    ).all():
        if names_run(child.parent, run_id):
            lineage("Checkpoint", str(child.name), str(child.run_id), str(child.parent))
    for adapter in session.execute(
        sa.select(_ADAPTERS.c.name, _ADAPTERS.c.run_id, _ADAPTERS.c.base_name).where(
            _ADAPTERS.c.run_id != run_id
        )
    ).all():
        if names_run(adapter.base_name, run_id):
            lineage("Adapter", str(adapter.name), str(adapter.run_id), str(adapter.base_name))

    # Optional features (chaining) contribute their own checks; see services.purge.blockers.
    for code, message, details in contributed_blockers.contributed(
        session, "run", run_id, lambda value: mentions_run(value, run_id, roots)
    ):
        plan.block(code, message, **details)

    for approval in session.execute(
        sa.select(_APPROVALS.c.id, _APPROVALS.c.action, _APPROVALS.c.payload).where(
            _APPROVALS.c.status.in_(sorted(OPEN_APPROVAL_STATES))
        )
    ).all():
        if mentions_run(approval.payload, run_id, roots):
            plan.block(
                "approval_pending",
                f"Approval {approval.id} ({approval.action}) is waiting and names this run.",
                approval_id=str(approval.id),
            )

    resident = millm.get("resident_model")
    if millm.get("checked") and isinstance(resident, str) and names_run(resident, run_id):
        plan.block(
            "loaded_in_millm",
            f"miLLM is serving {resident}, a checkpoint of this run.",
            resident_model=resident,
        )

    # ---- what goes ----
    scores = [
        int(s)
        for s in session.execute(sa.select(SCORES.c.id).where(SCORES.c.run_id == run_id)).scalars()
    ]
    bindings = [
        str(b)
        for b in session.execute(
            sa.select(REVIEW_BINDINGS.c.id).where(REVIEW_BINDINGS.c.run_id == run_id)
        ).scalars()
    ]
    plan.rows = {
        "mf_runs": 1,
        "mf_steps": _count(session, _STEPS, _STEPS.c.run_id == run_id),
        "mf_checkpoints": _count(session, _CHECKPOINTS, _CHECKPOINTS.c.run_id == run_id),
        "mf_adapters": _count(session, _ADAPTERS, _ADAPTERS.c.run_id == run_id),
        "mf_scores": len(scores),
        "mf_score_items": (
            _count(session, SCORE_ITEMS, SCORE_ITEMS.c.score_id.in_(scores)) if scores else 0
        ),
        "mf_batches": _count(session, BATCHES, BATCHES.c.run_id == run_id),
        "mf_review_bindings": len(bindings),
        "mf_review_candidate_refs": (
            _count(session, REVIEW_CANDIDATE_REFS, REVIEW_CANDIDATE_REFS.c.binding_id.in_(bindings))
            if bindings
            else 0
        ),
        "mf_leases": _count(
            session,
            LEASES,
            LEASES.c.holder_run_id == run_id,
            LEASES.c.released_at.is_not(None),
        ),
        "mf_merges": len(_owned_merges(session, run_id)),
    }

    from training.tasks import run_log_root

    settings = get_settings()
    owned = [
        (settings.checkpoint_root, "the run's checkpoint directory"),
        (settings.trainer_state_root, "the run's trainer state"),
        (run_log_root(), "the run's log directory"),
    ]
    for root, what in owned:
        directory = _owned_dir(root, run_id, what=what)
        if directory.is_dir():
            files, size = _measure(directory)
            plan.directories.append(directory)
            plan.files += files
            plan.bytes += size
    for root in settings.checkpoint_read_root_paths:
        candidate = root / run_id
        if candidate.exists():
            plan.kept.append(
                {
                    "path": str(candidate),
                    "why": "under a read-only checkpoint root miForge does not own (miLLM's)",
                }
            )
    # An indexed checkpoint whose directory is neither one of ours nor already listed.
    covered = [os.path.realpath(path) + os.sep for path in plan.directories]
    covered += [os.path.realpath(entry["path"]) + os.sep for entry in plan.kept]
    for path in session.execute(
        sa.select(_CHECKPOINTS.c.path).where(_CHECKPOINTS.c.run_id == run_id)
    ).scalars():
        real = os.path.realpath(str(path)) + os.sep
        if not any(real.startswith(prefix) for prefix in covered):
            plan.kept.append({"path": str(path), "why": "indexed outside the run's own directory"})
            covered.append(real)
    return plan


def _owned_merges(session: Session, run_id: str) -> list[int]:
    """Merge records this run owns: its result is the run's checkpoint, or (refused or
    interrupted, so no result) the adapter merged was the run's."""
    owned: list[int] = []
    for merge in session.execute(
        sa.select(MERGES.c.id, MERGES.c.adapter_name, MERGES.c.result_checkpoint_name)
    ).all():
        result = merge.result_checkpoint_name
        if (result is not None and names_run(result, run_id)) or (
            result is None and names_run(merge.adapter_name, run_id)
        ):
            owned.append(int(merge.id))
    return owned


def _run_row(session: Session, run_id: str, *, lock: bool) -> Any:
    query = sa.select(RUNS.c.id, RUNS.c.status, RUNS.c.run_type, RUNS.c.archived_at).where(
        RUNS.c.id == run_id
    )
    if lock:
        query = query.with_for_update()
    return session.execute(query).one_or_none()


def _run_not_found(run_id: str) -> Refusal:
    return refuse(
        "run_not_found", f"No run with id {run_id!r}. Check the identifier.", run_id=run_id
    )


def plan_runs(session: Session, run_ids: Sequence[str]) -> list[dict[str, Any]]:
    """What purging each run would remove and what stops it. Changes nothing."""
    millm = millm_resident()
    out: list[dict[str, Any]] = []
    for run_id in _unique(run_ids):
        row = _run_row(session, run_id, lock=False)
        if row is None:
            out.append(_missing("run", run_id, _run_not_found(run_id)))
            continue
        try:
            out.append(_plan_run(session, row, millm).payload())
        except Refusal as refusal:
            out.append(_missing("run", run_id, refusal, found=True))
    return out


def purge_run(
    session: Session,
    run_id: str,
    *,
    reason: str | None = None,
    purged_by: str = OPERATOR,
    millm: Mapping[str, Any] | None = None,
    now: datetime | None = None,
) -> dict[str, Any]:
    """Permanently delete one archived run. Commits. See the module docstring.

    Raises:
        Refusal: ``run_not_found`` (404); ``not_archived`` (409); ``purge_blocked`` (409)
            with every blocker; ``purge_path_unsafe`` (409).
    """
    row = _run_row(session, run_id, lock=True)
    if row is None:
        raise _run_not_found(run_id)
    plan = _plan_run(session, row, millm if millm is not None else millm_resident())
    _refuse_unless_purgeable(plan)

    staged = _stage(plan.directories)
    try:
        removed = _delete_run_rows(session, run_id)
        tombstone = _tombstone(
            session, plan, removed, reason=reason, purged_by=purged_by, now=now, staged=staged
        )
        session.commit()
    except BaseException:
        session.rollback()
        _unstage(staged)
        raise
    return _finish(plan, tombstone, staged)


def _delete_run_rows(session: Session, run_id: str) -> dict[str, int]:
    removed: dict[str, int] = {}

    def gone(table: sa.Table, *where: Any) -> None:
        removed[table.name] = removed.get(table.name, 0) + rowcount_of(
            session.execute(sa.delete(table).where(*where))
        )

    scores = list(
        session.execute(sa.select(SCORES.c.id).where(SCORES.c.run_id == run_id)).scalars()
    )
    if scores:
        gone(SCORE_ITEMS, SCORE_ITEMS.c.score_id.in_(scores))
    else:
        removed[SCORE_ITEMS.name] = 0
    gone(SCORES, SCORES.c.run_id == run_id)
    bindings = list(
        session.execute(
            sa.select(REVIEW_BINDINGS.c.id).where(REVIEW_BINDINGS.c.run_id == run_id)
        ).scalars()
    )
    if bindings:
        gone(REVIEW_CANDIDATE_REFS, REVIEW_CANDIDATE_REFS.c.binding_id.in_(bindings))
    else:
        removed[REVIEW_CANDIDATE_REFS.name] = 0
    gone(REVIEW_BINDINGS, REVIEW_BINDINGS.c.run_id == run_id)
    gone(BATCHES, BATCHES.c.run_id == run_id)
    gone(_STEPS, _STEPS.c.run_id == run_id)
    gone(_CHECKPOINTS, _CHECKPOINTS.c.run_id == run_id)
    gone(_ADAPTERS, _ADAPTERS.c.run_id == run_id)
    gone(LEASES, LEASES.c.holder_run_id == run_id, LEASES.c.released_at.is_not(None))
    merges = _owned_merges(session, run_id)
    if merges:
        gone(MERGES, MERGES.c.id.in_(merges))
    else:
        removed[MERGES.name] = 0
    # The run row LAST, and conditionally: a run continued since the plan was read is
    # unarchived and running, and this matches nothing.
    gone(
        RUNS,
        RUNS.c.id == run_id,
        RUNS.c.archived_at.is_not(None),
        RUNS.c.status.in_(sorted(TERMINAL_STATES)),
    )
    if removed[RUNS.name] != 1:
        raise refuse(
            "not_archived",
            f"Run {run_id} stopped being an archived, finished run while it was being purged; "
            f"nothing was deleted.",
            run_id=run_id,
        )
    return removed


# ============================================================================== parts


def _part_rows(session: Session, part_id: str) -> tuple[Any, Any]:
    index_row = (
        session.execute(sa.select(CONTRACT_PARTS).where(CONTRACT_PARTS.c.id == part_id))
        .mappings()
        .one_or_none()
    )
    archive_row = session.execute(
        sa.select(PART_ARCHIVE.c.archived_at).where(PART_ARCHIVE.c.part_id == part_id)
    ).one_or_none()
    return index_row, archive_row


def _part_name(part_id: str, row: Any) -> str:
    if row is None:
        return f"part {part_id} (no longer indexed)"
    if row["ref_type"] is not None:
        return f"part {row['ref_type']}/{row['ref_name']}@{row['ref_version']}"
    if row["dataset"] is not None:
        number = row["version_number"]
        return f"part {row['dataset']}{'' if number is None else f' v{number}'}"
    return f"part {part_id}"


def _mentions_part(value: Any, part_id: str, reference: Mapping[str, Any] | None) -> bool:
    if value is None:
        return False
    if any(part_id in text for text in _strings(value)):
        return True
    if reference is None:
        return False
    rendered = f"{reference['type']}/{reference['name']}@{reference['version']}"
    if any(text == rendered or text.endswith("/" + rendered) for text in _strings(value)):
        return True
    return _contains_reference(value, reference)


def _contains_reference(value: Any, reference: Mapping[str, Any]) -> bool:
    if isinstance(value, Mapping):
        if all(value.get(key) == reference[key] for key in ("type", "name", "version")):
            return True
        return any(_contains_reference(item, reference) for item in value.values())
    if isinstance(value, (list, tuple)):
        return any(_contains_reference(item, reference) for item in value)
    return False


def _plan_part(session: Session, part_id: str, row: Any, archive_row: Any) -> _Plan:
    plan = _Plan(
        kind="part",
        item_id=part_id,
        name=_part_name(part_id, row),
        archived=archive_row is not None,
        archived_at=None if archive_row is None else _iso(archive_row.archived_at),
    )
    reference = None
    digest = None
    if row is not None:
        if row["ref_type"] is not None:
            reference = {
                "type": row["ref_type"],
                "name": row["ref_name"],
                "version": row["ref_version"],
            }
        if row["source"] == SOURCE_EXTERNAL:
            digest = (row["external_provenance"] or {}).get("content_sha256")

    # ---- blockers: packs pin it ----
    for slot in session.execute(
        sa.select(
            _PACKS.c.name,
            _PACKS.c.version,
            _PACK_PARTS.c.part_kind,
            _PACK_PARTS.c.source,
            _PACK_PARTS.c.ref_type,
            _PACK_PARTS.c.ref_name,
            _PACK_PARTS.c.ref_version,
            _PACK_PARTS.c.external_provenance,
        ).join(_PACKS, _PACKS.c.id == _PACK_PARTS.c.pack_id)
    ).all():
        pinned = (
            reference is not None
            and slot.ref_type == reference["type"]
            and slot.ref_name == reference["name"]
            and slot.ref_version == reference["version"]
        ) or (
            digest is not None
            and isinstance(slot.external_provenance, Mapping)
            and slot.external_provenance.get("content_sha256") == digest
        )
        if pinned:
            plan.block(
                "pack_pins_it",
                f"Pack {slot.name}@{slot.version} pins this part in its {slot.part_kind} slot.",
                pack=f"{slot.name}@{slot.version}",
            )

    # ---- runs that used it ----
    scored = {
        str(r)
        for r in session.execute(
            sa.select(SCORES.c.run_id).where(SCORES.c.part_id == part_id).distinct()
        ).scalars()
    }
    for run in session.execute(
        sa.select(
            RUNS.c.id,
            RUNS.c.status,
            RUNS.c.archived_at,
            RUNS.c.definition_submitted,
            RUNS.c.definition_resolved,
        )
    ).all():
        used = str(run.id) in scored or any(
            _mentions_part(value, part_id, reference)
            for value in (run.definition_submitted, run.definition_resolved)
        )
        if not used:
            continue
        if run.status in ACTIVE_RUN_STATES:
            plan.block(
                "run_active_reference",
                f"Run {run.id} is {run.status} and uses this part.",
                run_id=str(run.id),
            )
        elif run.archived_at is None:
            plan.block(
                "run_reference",
                f"Run {run.id} (not archived) used this part. Archive or purge it first.",
                run_id=str(run.id),
            )
    for code, message, details in contributed_blockers.contributed(
        session, "part", part_id, lambda value: _mentions_part(value, part_id, reference)
    ):
        plan.block(code, message, **details)
    for approval in session.execute(
        sa.select(_APPROVALS.c.id, _APPROVALS.c.action, _APPROVALS.c.payload).where(
            _APPROVALS.c.status.in_(sorted(OPEN_APPROVAL_STATES))
        )
    ).all():
        if _mentions_part(approval.payload, part_id, reference):
            plan.block(
                "approval_pending",
                f"Approval {approval.id} ({approval.action}) is waiting and names this part.",
                approval_id=str(approval.id),
            )

    # ---- what goes ----
    plan.rows = {
        "mf_contract_parts": 0 if row is None else 1,
        "mf_part_splits": _count(session, PART_SPLITS, PART_SPLITS.c.part_id == part_id),
        "mf_part_archive": 0 if archive_row is None else 1,
    }
    if row is not None:
        parts_root = get_settings().parts_root
        local = Path(str(row["local_path"]))
        try:
            relative = local.relative_to(parts_root)
        except ValueError:
            relative = None
        if relative is None or len(relative.parts) != 1:
            if local.exists():
                plan.kept.append(
                    {"path": str(local), "why": "not a directory of its own under the parts root"}
                )
        else:
            directory = _owned_dir(parts_root, relative.parts[0], what="the part's directory")
            if directory.is_dir():
                files, size = _measure(directory)
                plan.directories.append(directory)
                plan.files += files
                plan.bytes += size
    return plan


def _part_not_found(part_id: str) -> Refusal:
    return refuse(
        "part_not_found",
        f"No part with id {part_id!r} in the index or the archive.",
        part_id=part_id,
    )


def plan_parts(session: Session, part_ids: Sequence[str]) -> list[dict[str, Any]]:
    """What purging each part would remove and what stops it. Changes nothing."""
    out: list[dict[str, Any]] = []
    for part_id in _unique(part_ids):
        row, archive_row = _part_rows(session, part_id)
        if row is None and archive_row is None:
            out.append(_missing("part", part_id, _part_not_found(part_id)))
            continue
        try:
            out.append(_plan_part(session, part_id, row, archive_row).payload())
        except Refusal as refusal:
            out.append(_missing("part", part_id, refusal, found=True))
    return out


def purge_part(
    session: Session,
    part_id: str,
    *,
    reason: str | None = None,
    purged_by: str = OPERATOR,
    now: datetime | None = None,
) -> dict[str, Any]:
    """Permanently delete one archived part. Commits. See the module docstring.

    Raises:
        Refusal: ``part_not_found`` (404); ``not_archived`` (409); ``purge_blocked`` (409);
            ``purge_path_unsafe`` (409).
    """
    row, archive_row = _part_rows(session, part_id)
    if row is None and archive_row is None:
        raise _part_not_found(part_id)
    plan = _plan_part(session, part_id, row, archive_row)
    _refuse_unless_purgeable(plan)

    staged = _stage(plan.directories)
    try:
        removed: dict[str, int] = {}
        removed[PART_SPLITS.name] = rowcount_of(
            session.execute(sa.delete(PART_SPLITS).where(PART_SPLITS.c.part_id == part_id))
        )
        removed[CONTRACT_PARTS.name] = rowcount_of(
            session.execute(sa.delete(CONTRACT_PARTS).where(CONTRACT_PARTS.c.id == part_id))
        )
        removed[PART_ARCHIVE.name] = rowcount_of(
            session.execute(sa.delete(PART_ARCHIVE).where(PART_ARCHIVE.c.part_id == part_id))
        )
        if removed[PART_ARCHIVE.name] != 1:
            raise refuse(
                "not_archived",
                f"Part {part_id} was unarchived while it was being purged; nothing was deleted.",
                part_id=part_id,
            )
        tombstone = _tombstone(
            session, plan, removed, reason=reason, purged_by=purged_by, now=now, staged=staged
        )
        session.commit()
    except BaseException:
        session.rollback()
        _unstage(staged)
        raise
    return _finish(plan, tombstone, staged)


# ============================================================================= shared


def _unique(ids: Iterable[str]) -> list[str]:
    seen: dict[str, None] = {}
    for item in ids:
        seen.setdefault(str(item), None)
    return list(seen)


def _missing(kind: str, item_id: str, refusal: Refusal, *, found: bool = False) -> dict[str, Any]:
    return {
        "kind": kind,
        "id": item_id,
        "found": found,
        "name": f"{kind} {item_id}",
        "purgeable": False,
        "archived": False,
        "blockers": [{"code": refusal.code, "message": refusal.message}],
        "not_archived": None,
        "removes": {"rows": {}, "row_total": 0, "files": 0, "bytes": 0, "directories": []},
        "kept": [],
        "millm": {},
    }


def _refuse_unless_purgeable(plan: _Plan) -> None:
    if not plan.archived:
        raise refuse(
            "not_archived",
            f"{plan.name} is not archived. Only an archived {plan.kind} can be deleted "
            f"permanently; archive it first.",
            kind=plan.kind,
            id=plan.item_id,
        )
    if plan.blockers:
        raise refuse(
            "purge_blocked",
            f"{plan.name} cannot be deleted: "
            + "; ".join(blocker["message"] for blocker in plan.blockers),
            kind=plan.kind,
            id=plan.item_id,
            blockers=plan.blockers,
        )


def _stage(directories: Sequence[Path]) -> list[tuple[Path, Path]]:
    """Rename each directory to a hidden sibling. All or none."""
    staged: list[tuple[Path, Path]] = []
    try:
        for directory in directories:
            target = directory.with_name(f"{STAGE_PREFIX}{uuid.uuid4().hex}")
            os.rename(directory, target)
            staged.append((directory, target))
    except OSError as exc:
        _unstage(staged)
        raise refuse(
            "purge_files_unmovable",
            f"Could not move {directory} aside before deleting it ({type(exc).__name__}); "
            f"nothing was deleted.",
            path=str(directory),
        ) from exc
    return staged


def _unstage(staged: Sequence[tuple[Path, Path]]) -> None:
    for original, target in reversed(staged):
        try:
            os.rename(target, original)
        except OSError:
            log.error(
                "purge could not restore a staged directory",
                extra={
                    "code": "purge_unstage_failed",
                    "path": str(original),
                    "staged": str(target),
                },
            )


def _tombstone(
    session: Session,
    plan: _Plan,
    removed: Mapping[str, int],
    *,
    reason: str | None,
    purged_by: str,
    now: datetime | None,
    staged: Sequence[tuple[Path, Path]],
) -> dict[str, Any]:
    values: dict[str, Any] = {
        "id": f"purge-{uuid.uuid4().hex}",
        "kind": plan.kind,
        "item_id": plan.item_id,
        "name": plan.name,
        "purged_by": purged_by,
        "reason": (reason or "").strip()[:2000] or None,
        "rows_removed": dict(removed),
        "files_removed": plan.files,
        "bytes_removed": plan.bytes,
        "detail": {
            "status": plan.status,
            "archived_at": plan.archived_at,
            "directories": [str(original) for original, _ in staged],
            "kept": list(plan.kept),
            "millm": dict(plan.millm),
        },
        "purged_at": now or utcnow(),
    }
    session.execute(sa.insert(PURGES).values(**values))
    return values


def _finish(
    plan: _Plan, tombstone: Mapping[str, Any], staged: Sequence[tuple[Path, Path]]
) -> dict[str, Any]:
    left: list[str] = []
    for original, target in staged:
        try:
            shutil.rmtree(target)
        except OSError as exc:
            left.append(str(target))
            log.error(
                "purge left bytes behind; they are staged and undiscoverable",
                extra={
                    "code": "purge_files_left",
                    "path": str(original),
                    "staged": str(target),
                    "error": type(exc).__name__,
                },
            )
    log.info(
        "purged",
        extra={
            "code": "purged",
            "kind": plan.kind,
            "item_id": plan.item_id,
            "bytes_removed": plan.bytes,
        },
    )
    return {
        **tombstone_payload(tombstone),
        "outcome": "purged",
        "files_left": left,
    }


def tombstone_payload(row: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "tombstone_id": row["id"],
        "kind": row["kind"],
        "id": row["item_id"],
        "name": row["name"],
        "purged_by": row["purged_by"],
        "reason": row["reason"],
        "rows_removed": dict(row["rows_removed"] or {}),
        "files_removed": int(row["files_removed"]),
        "bytes_removed": int(row["bytes_removed"]),
        "detail": dict(row["detail"] or {}),
        "purged_at": _iso(row["purged_at"]),
    }


def purge_many(
    session: Session, kind: str, ids: Sequence[str], *, reason: str | None = None
) -> dict[str, Any]:
    """Purge each item **independently**: one blocked item never stops the rest.

    Each item is its own transaction (the single-item function commits); a refusal rolls
    back only that item and is reported with its code, message and blockers.
    """
    millm = millm_resident() if kind == "run" else None
    results: list[dict[str, Any]] = []
    for item_id in _unique(ids):
        try:
            if kind == "run":
                result = purge_run(session, item_id, reason=reason, millm=millm)
            else:
                result = purge_part(session, item_id, reason=reason)
        except Refusal as refusal:
            session.rollback()
            results.append(
                {
                    "kind": kind,
                    "id": item_id,
                    "outcome": "blocked",
                    "code": refusal.code,
                    "reason": refusal.message,
                    "blockers": list(refusal.details.get("blockers") or []),
                }
            )
            continue
        except Exception as exc:  # one failure must not abort the batch
            session.rollback()
            log.exception(
                "purge failed", extra={"code": "purge_failed", "kind": kind, "item_id": item_id}
            )
            results.append(
                {
                    "kind": kind,
                    "id": item_id,
                    "outcome": "blocked",
                    "code": "purge_failed",
                    "reason": f"The purge failed ({type(exc).__name__}); nothing was deleted "
                    f"for this item. See the server log.",
                    "blockers": [],
                }
            )
            continue
        results.append(result)
    purged = [r for r in results if r["outcome"] == "purged"]
    return {
        "kind": kind,
        "results": results,
        "purged": len(purged),
        "blocked": len(results) - len(purged),
        "bytes_removed": sum(int(r["bytes_removed"]) for r in purged),
    }


def list_tombstones(session: Session, *, limit: int = 100) -> list[dict[str, Any]]:
    rows = (
        session.execute(
            sa.select(PURGES).order_by(PURGES.c.purged_at.desc(), PURGES.c.id).limit(limit)
        )
        .mappings()
        .all()
    )
    return [tombstone_payload(dict(row)) for row in rows]
