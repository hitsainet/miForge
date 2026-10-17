"""``change`` - full-parameter updates to a model that exists. F1's default.

**Thin on purpose, and it is written first and smallest.** A ``change`` run *is* F1's step
over all parameters, written by F5's checkpoint writer; there is nothing for F8 to add. If
:mod:`training.runtypes.create` ever grows larger than this, the loop has been duplicated
somewhere (TID section 7).
"""

from __future__ import annotations

import logging
from collections.abc import Mapping
from typing import Any

from services.checkpoints import service as checkpoint_service
from services.checkpoints.manifest import ManifestFields
from services.scoring import record as score_record
from training.acquisition.base import AcquiredModel, AcquisitionSpec
from training.run_config import RunConfig
from training.runtypes.base import STEP, SaveContext, TrainablePlan, all_parameters

log = logging.getLogger(__name__)


def joined_scores(ctx: SaveContext) -> tuple[Mapping[str, Any] | None, Mapping[str, Any] | None]:
    """The two ``ScoreRecord`` payloads this step's manifest carries, **joined from F4**.

    ``SaveContext.test`` and ``SaveContext.retention`` are a *superset* field with no
    production filler: nothing in ``backend/src`` constructed a ``SaveContext`` carrying
    them, so every manifest this run type wrote recorded ``test: null, retention: null`` -
    **"no test ran at this step"** - while ``mf_scores`` and the run log held real numbers.
    Both properties are nullable, so the manifest *validated clean*, and the schema-sync
    test compares schema **files** and cannot see a payload at all. R-02.16's provenance was
    simply absent from the one artifact that travels.

    So the pair is **derived** here through F4's own published seam,
    :func:`services.scoring.record.publish_to_manifest`, which reads the index for this
    exact ``(run_id, step)`` and hands back both keys - ``None`` where that kind did not run
    at this step, which is the one case ``null`` is an honest answer for (E-22).

    An explicitly supplied ``ctx.test``/``ctx.retention`` still wins: a caller that already
    holds the pass should not make the save re-read the index. Removing the
    ``publish_to_manifest`` call below must make
    ``backend/tests/integration/test_runtype_manifest_scores.py`` red (observed:
    6 of its 9 tests, the three survivors being the ones that do not depend on the join).

    ## A ``None`` session is NAMED, not silent

    ``drive_run`` supports ``session=None`` on purpose - the run log is the authoritative
    record (ADR-004) - but the scores are joined from the **index**, so with no session
    there is nothing to join from. The ``null`` that then reaches the manifest means *"not
    known here"*, which is a **different fact** from the *"nothing ran at this step"* the
    schema defines it as, and the difference is unreadable from the artifact. So it is
    logged rather than left to look like an evaluation that did not happen.

    *This guard was in* ``training/tasks.py``'s *own projection of these two payloads until
    2026-10-16, when the loop was routed through* :meth:`RunType.save` *and the projection
    collapsed into this one join. It belongs here: "there is nothing to join from" is the
    join's own fact, and a guard left at one of two call sites protects one of them.*
    """
    if ctx.test is not None or ctx.retention is not None:
        return ctx.test, ctx.retention

    if ctx.session is None:
        log.info(
            "manifest written with no joined scores",
            extra={
                "code": "checkpoint_scores_unjoinable_no_session",
                "run_id": ctx.run_id,
                "step": ctx.step,
                "reason": (
                    "this run was driven with no database session, so `mf_scores` could "
                    "not be read. `test` and `retention` are null in the manifest and that "
                    "null is NOT evidence that no evaluation ran."
                ),
            },
        )
        return None, None

    captured: dict[str, Mapping[str, Any] | None] = {}

    def publish(test: Mapping[str, Any] | None, retention: Mapping[str, Any] | None) -> None:
        captured["test"] = test
        captured["retention"] = retention

    score_record.publish_to_manifest(ctx.session, run_id=ctx.run_id, step=ctx.step, publish=publish)
    return captured["test"], captured["retention"]


def manifest_fields_from(
    ctx: SaveContext, *, base_model: str, parent: str | None
) -> ManifestFields:
    """Build F5's ``ManifestFields`` from a :class:`SaveContext`.

    **The one place a checkpoint manifest's borrowed values are assembled in F8**, shared
    by ``change`` and ``create`` and by the merge. ``base_model`` and ``parent`` are
    arguments rather than read off the context because each run type derives them from F1's
    record through F1's own lineage functions - and the merge derives them from the adapter's
    base, which is a third rule again.

    ``test`` and ``retention`` come from :func:`joined_scores`, not off the context - see
    there for why reading the context alone wrote ``null`` into every manifest.
    """
    test, retention = joined_scores(ctx)
    return ManifestFields(
        run_id=ctx.run_id,
        step=ctx.step,
        domain=ctx.domain,
        signal_source=ctx.signal_source,
        base_model=base_model,
        parent=parent,
        optimizer=ctx.optimizer,
        selector_hash=ctx.selector_hash,
        grader_hash=ctx.grader_hash,
        test=test,
        retention=retention,
        merged_adapter=ctx.merged_adapter,
    )


def save_checkpoint(ctx: SaveContext, *, base_model: str, parent: str | None) -> Any:
    """Hand F5's writer the lineage pair and let it publish. **No second writer.**

    F5's ``service.save`` owns the resave rule, the index upsert, the socket event and the
    heartbeat; F8 supplies values and calls it. A second checkpoint writer would mean a
    second manifest, and the one that is wrong is whichever nobody runs (TDD R-2).
    """
    if ctx.tokenizer is None:
        from core.errors import refuse

        raise refuse(
            "tokenizer_required",
            "A checkpoint carries the tokenizer's own files at its root, so a save needs "
            "the tokenizer the run was given. Refusing rather than writing a directory no "
            "standard loader could read (R-02.21, acceptance criterion 2).",
            run_id=ctx.run_id,
            step=ctx.step,
        )
    outcome = checkpoint_service.save(
        ctx.module,
        ctx.tokenizer,
        session=ctx.session,
        fields=manifest_fields_from(ctx, base_model=base_model, parent=parent),
        root=ctx.root,
        emit=ctx.emit,
        heartbeat=ctx.heartbeat,
    )
    # WHAT WAS PUBLISHED, reported through the context rather than through the return value
    # - see `SaveContext.artifacts`. F5's writer knows the directory; the loop must not
    # re-derive it and must not branch on which writer ran to find it.
    ctx.artifacts.append(str(outcome.directory))
    return outcome


class ChangeRunType:
    """R-02.12's step over every parameter of an acquired model."""

    name = "change"
    #: The same object every run type carries. See :data:`training.runtypes.base.STEP`.
    step = staticmethod(STEP)

    def preflight(
        self,
        spec: AcquisitionSpec,
        cfg: RunConfig,
        *,
        signal_source: str | None = None,
        adapter: Any = None,
    ) -> None:
        """Nothing of its own. F1's acquirer pre-flight already refuses E-1 through E-6.

        An empty body stated explicitly rather than omitted: *this run type adds no
        refusal* is a fact worth being able to read, and a ``change`` run that acquired
        a new refusal here would be F8 constraining F1's default path.
        """

    def prepare(self, acquired: AcquiredModel, *, adapter: Any = None) -> TrainablePlan:
        return all_parameters(acquired)

    def save(self, ctx: SaveContext) -> Any:
        """F1's lineage pair, verbatim, into F5's writer.

        ``record.base_model`` and ``record.parent`` were already derived by F1 through
        :func:`training.acquisition.base.lineage_for_loaded_checkpoint` or
        :func:`~training.acquisition.base.lineage_for_first_load` - the functions that fix
        the carry-forward defect where a continue rooted the tree at its parent. F8 does
        not re-derive them.
        """
        return save_checkpoint(ctx, base_model=ctx.record.base_model, parent=ctx.record.parent)
