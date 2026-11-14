"""Rebuilding ``mf_scores`` from what is on disk  (FR-004.32; ADR-004, foundation 4.3 - 4.5).

ADR-004: **the filesystem is the source of truth and the database is a rebuildable index.**
So ``mf_scores`` must be reconstructible, and the two sources are not interchangeable:

| Source | What it can rebuild | For which steps |
|---|---|---|
| a checkpoint's ``manifest.json`` | the five ``ScoreRecord`` fields | **saved steps only** |
| the on-disk run log's ``evaluation`` blocks | everything the block carries | **every** step |

**A score at a non-save step exists only in the run log**, which is therefore part of the
durability story and not telemetry - stated here so nobody prunes it for disk space.

## Where they disagree, the manifest wins and the disagreement is logged

ADR-004, and F10's ``resolve_part`` is the worked example of the same rule. The manifest is
the artifact that travels; a row that disagrees with it is a stale cache, and the log line
is what makes a silent divergence visible. This module logs under the **same event name**
F10 and F5 use, so one query finds every divergence in the product rather than three.

## What a rebuilt row cannot know, named rather than guessed

``EvaluationBlock`` closes itself to unknown keys and has no field for the floor's **layer**,
the selector and grader hashes, the reproducibility marking or the per-prompt detail. A
rebuilt row therefore reports ``floor_layer: "none"`` and nulls, and **that is a reported
gap** (DEF-2's neighbours) rather than a guess: defaulting the layer to ``pack_slot`` would
fabricate provenance for the one value FPRD OQ-2 says is unsettled.

## NOT YET CALLED BY ``services/reindex.py``

That module is the foundation's and outside this feature's write scope, so the one line that
calls :func:`reindex_scores` from it is **owed, and reported as owed**. Until it lands this
is a capability with no production caller - which is, by this project's own rule, a finding
and not a pass.
"""

from __future__ import annotations

import json
import logging
from collections.abc import Iterable, Mapping
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import sqlalchemy as sa

from services.scoring import evaluation_block as blocks
from services.scoring.tables import FLOOR_LAYER_NONE, KIND_RETENTION, SCORE_ITEMS, SCORES
from training import run_log

log = logging.getLogger("miforge.scoring")

#: The event name a manifest/index disagreement is logged under. **The same string F5 and
#: F10 use**, so one query over the logs finds every divergence in the product.
DIVERGENCE_EVENT = "index_manifest_divergence"

#: The checkpoint manifest's filename, normative at R-02.20.
MANIFEST_FILENAME = "manifest.json"

#: The ``kind`` const the checkpoint schema pins. **Read from the vendored file**, not
#: retyped: a directory can contain a ``manifest.json`` that is not ours, and treating it as
#: a checkpoint would rebuild scores from an artifact that is not one. ``services/reindex.py``
#: reads its own copy the same way and for the same reason - a hardcoded ``"miforge.checkpoint"``
#: (the value this module first carried) matches nothing, so every manifest was skipped and
#: the manifest pass silently did nothing while reporting success.


def checkpoint_kind() -> str:
    """The ``kind`` const of ``checkpoint-manifest-v1.json``, parsed from the file."""
    from services.scoring.manifest_payload import SCHEMA_FILENAME, schema_dir

    document = json.loads((schema_dir() / SCHEMA_FILENAME).read_text("utf-8"))
    return str(document["$defs"]["CheckpointManifestV1"]["properties"]["kind"]["const"])


def drop_scores(session: Any) -> None:
    """Empty both tables, for the reindex-equivalence test (foundation 4.4)."""
    session.execute(sa.delete(SCORE_ITEMS))
    session.execute(sa.delete(SCORES))


def reindex_scores(
    session: Any,
    *,
    run_log_root: Path,
    checkpoint_root: Path | None = None,
    run_ids: Iterable[str] | None = None,
) -> dict[str, Any]:
    """Rebuild ``mf_scores`` from the run logs, then let saved manifests correct it.

    Args:
        session: An open session. The caller commits.
        run_log_root: The directory holding ``<run_id>/run.jsonl``.
        checkpoint_root: Where checkpoints live. ``None`` skips the manifest pass, which is
            the right behaviour for a rebuild of a run that has saved nothing - and not the
            same as there being no manifests, which is why it is a parameter.
        run_ids: Limit the rebuild to these runs. ``None`` rebuilds everything found.

    Returns:
        A report: how many blocks were read, how many rows were written, and the
        divergences the manifest pass corrected.
    """
    drop_scores(session)
    report: dict[str, Any] = {"blocks": 0, "rows": 0, "divergences": []}

    for log_path in _run_logs(run_log_root, run_ids):
        for block in run_log.read_blocks(log_path, block=blocks.BLOCK):
            report["blocks"] += 1
            for row in blocks.score_rows_from_block(block):
                session.execute(sa.insert(SCORES).values(**_row_values(row)))
                report["rows"] += 1

    if checkpoint_root is not None:
        report["divergences"] = _apply_manifests(session, checkpoint_root)
    return report


def _run_logs(root: Path, run_ids: Iterable[str] | None) -> list[Path]:
    """Every run log under ``root``, optionally filtered. Sorted, so a rebuild is stable."""
    if not root.is_dir():
        return []
    wanted = None if run_ids is None else set(run_ids)
    out: list[Path] = []
    for path in sorted(root.glob(f"*/{run_log.RUN_LOG_NAME}")):
        if wanted is None or path.parent.name in wanted:
            out.append(path)
    return out


def _row_values(row: Mapping[str, Any]) -> dict[str, Any]:
    """One projected block row as ``mf_scores`` column values.

    The columns the block cannot supply are set to their honest unknown - ``floor_layer`` to
    ``none``, the hashes and the reproducibility reason to null - rather than to a plausible
    value. See the module docstring.
    """
    return {
        "run_id": row["run_id"],
        "step": row["step"],
        "kind": row["kind"],
        "set_ref": None,
        "set_rendered": row["set_rendered"],
        "part_id": None,
        "score": row["score"],
        "n": row["n"],
        "floor": row["floor"],
        "floor_layer": FLOOR_LAYER_NONE,
        "breached": row["breached"],
        "provider": row["provider"],
        "provider_kind": _kind_of(row["provider"]),
        "provider_model": _model_of(row["provider"]),
        "provider_base_url": _base_url_of(row["provider"]),
        "parameter_set_digest": None,
        "aggregation": row["aggregation"],
        "decode": row["decode"],
        "subsample_seed": row["subsample_seed"],
        "split_markers": row["split_markers"],
        "selector_hash": None,
        "grader_hash": None,
        "reproducible": True,
        "not_reproducible_reason": None,
        "out_of_cadence": False,
        "provider_discontinuity": False,
        "rescore_reason": None,
        "created_at": _timestamp(row.get("ts")),
    }


def _timestamp(value: Any) -> datetime:
    """The block's ``ts``, or now. A rebuilt row keeps the time the pass happened.

    ``created_at`` is part of ``mf_scores``' unique key - the re-score rule - so a rebuild
    that stamped everything with the rebuild time would collapse two genuine re-scores of one
    step into a constraint violation.
    """
    if isinstance(value, str):
        try:
            return datetime.fromisoformat(value)
        except ValueError:
            pass
    return datetime.now(UTC)


def _kind_of(provider: str) -> str:
    """``kind`` out of the rendered provider string ``kind:model[@base_url]``.

    Parsed rather than stored twice: ``ProviderIdentity.record()`` is the one place the
    rendering is defined, and a second encoding of the same three facts would drift.
    """
    return provider.split(":", 1)[0] if ":" in provider else provider


def _model_of(provider: str) -> str:
    rest = provider.split(":", 1)[1] if ":" in provider else provider
    return rest.split("@", 1)[0]


def _base_url_of(provider: str) -> str | None:
    rest = provider.split(":", 1)[1] if ":" in provider else provider
    return rest.split("@", 1)[1] if "@" in rest else None


def _apply_manifests(session: Any, checkpoint_root: Path) -> list[dict[str, Any]]:
    """Let each saved step's manifest correct the rows the log produced.

    **The manifest wins** (ADR-004) and the disagreement is logged with both answers, then
    the row is corrected so the next read is cheap and the same line is not emitted forever -
    exactly F10's ``resolve_part`` shape.
    """
    divergences: list[dict[str, Any]] = []
    if not checkpoint_root.is_dir():
        return divergences
    for manifest_path in sorted(checkpoint_root.rglob(MANIFEST_FILENAME)):
        try:
            manifest = json.loads(manifest_path.read_text("utf-8"))
        except ValueError:
            # A manifest that does not parse is F5's problem to report, not a reason to
            # abandon a score rebuild that is otherwise complete.
            continue
        if manifest.get("kind") != checkpoint_kind():
            continue
        run_id = manifest.get("run_id")
        step = manifest.get("step")
        if run_id is None or step is None:
            continue
        for key, kind in (("test", "test"), ("retention", KIND_RETENTION)):
            score = manifest.get(key)
            if not isinstance(score, Mapping):
                continue
            divergences.extend(
                _reconcile(session, run_id=str(run_id), step=int(step), kind=kind, score=score)
            )
    return divergences


def _reconcile(
    session: Any, *, run_id: str, step: int, kind: str, score: Mapping[str, Any]
) -> list[dict[str, Any]]:
    """One manifest ScoreRecord against the rebuilt row. Manifest wins; divergence logged."""
    row = (
        session.execute(
            sa.select(SCORES)
            .where(SCORES.c.run_id == run_id, SCORES.c.step == step, SCORES.c.kind == kind)
            .order_by(SCORES.c.created_at.desc())
        )
        .mappings()
        .first()
    )

    missing = [key for key in ("set", "score", "n") if score.get(key) is None]
    if missing:
        # The three fields `ScoreRecord` makes REQUIRED. A manifest without them is
        # non-conforming, which is F5's to report; here it is skipped and named rather than
        # coerced - `float(None)` would be a type error but `float(None or 0.0)` would be a
        # fabricated score, and a fabricated score plots as a real one.
        log.warning(
            DIVERGENCE_EVENT,
            extra={
                "artifact": f"{run_id}@{step}",
                "divergence": "manifest_score_incomplete",
                "divergent_fields": missing,
                "resolution": "skipped",
                "kind": kind,
            },
        )
        return [
            {
                "artifact": f"{run_id}@{step}",
                "divergence": "manifest_score_incomplete",
                "divergent_fields": missing,
                "resolution": "skipped",
                "kind": kind,
            }
        ]

    truth: dict[str, Any] = {
        "set_rendered": str(score["set"]),
        "score": float(score["score"]),
        "n": int(score["n"]),
        "floor": score.get("floor"),
    }
    # `provider` is NOT in the manifest's `required` list, so a manifest that omits it is
    # conformant and says nothing - which is different from one that asserts null. An absent
    # key must not overwrite a provider the run log recorded.
    if "provider" in score and score["provider"] is not None:
        truth["provider"] = score["provider"]

    if row is None:
        # A saved step with no run-log block: the log was pruned or the run predates it.
        # Insert what the manifest can say and mark the rest unknown rather than absent.
        session.execute(
            sa.insert(SCORES).values(
                run_id=run_id,
                step=step,
                kind=kind,
                set_ref=None,
                set_rendered=truth["set_rendered"],
                score=truth["score"],
                n=truth["n"],
                floor=truth["floor"],
                floor_layer=FLOOR_LAYER_NONE,
                breached=False,
                provider=str(truth.get("provider") or "unrecorded:unrecorded"),
                provider_kind=_kind_of(str(truth.get("provider") or "unrecorded:unrecorded")),
                provider_model=_model_of(str(truth.get("provider") or "unrecorded:unrecorded")),
                provider_base_url=None,
                parameter_set_digest=None,
                aggregation="mean",
                decode={},
                subsample_seed=None,
                split_markers=None,
                selector_hash=None,
                grader_hash=None,
                reproducible=True,
                not_reproducible_reason=None,
                out_of_cadence=False,
                provider_discontinuity=False,
                rescore_reason="rebuilt from manifest; no run-log evaluation block",
                created_at=datetime.now(UTC),
            )
        )
        return [
            {
                "artifact": f"{run_id}@{step}",
                "divergence": "row_missing",
                "resolution": "manifest_wins",
                "kind": kind,
            }
        ]

    diffs = {
        name: {"index": row[name], "manifest": value}
        for name, value in truth.items()
        if row[name] != value
    }
    if not diffs:
        return []
    log.warning(
        DIVERGENCE_EVENT,
        extra={
            "artifact": f"{run_id}@{step}",
            "divergence": "field_mismatch",
            "divergent_fields": sorted(diffs),
            "resolution": "manifest_wins",
            "detail": diffs,
            "kind": kind,
        },
    )
    session.execute(
        sa.update(SCORES)
        .where(SCORES.c.id == row["id"])
        .values(**{name: answers["manifest"] for name, answers in diffs.items()})
    )
    return [
        {
            "artifact": f"{run_id}@{step}",
            "divergence": "field_mismatch",
            "divergent_fields": sorted(diffs),
            "resolution": "manifest_wins",
            "kind": kind,
        }
    ]
