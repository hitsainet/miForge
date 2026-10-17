"""``miforge.train.merge_adapter`` - the merge, on the card, with the record as the truth.

**F8's merge had no enqueue and its table had no writer**, and both were owed to F6 (its
tasks 11.5 and 12.5, both ``[?]`` naming this feature). F8 said so plainly rather than
disguising it: *"the enqueue is not wired, and that is stated rather than implied"* - a route
that enqueued a task nothing consumed would be the unreachable-capability failure this
project's review discipline exists to catch.

## Why the ``train`` queue and not ``default``

A merge touches the card, and the card is leased one run at a time (ADR-008). So the
arithmetic belongs where the GPU work is serialised. **The task name must begin
``miforge.train.``** - ``TASK_ROUTES`` is an *explicit* table keyed by area prefix, and a
task matching no entry lands on ``default``, where a worker with **no card** picks it up and
fails. F1 hit exactly this and recorded it: its documents all spelled the loop task
``miforge.training.…``, which routes nowhere.

*F6's own documents name a chain sweeper ``miforge.chain.sweep``, which has the same defect
and no entry in that table; it is reported rather than shipped under a name that does not
route.*

## A merge is an operation, not a run  (FR-006.55)

It gets no ``mf_runs`` row, so it holds **no lease of its own** and takes no stop
conditions. It runs on the ``train`` queue, where the solo worker's one-at-a-time execution
is what keeps it off a card another run is training on. That is a weaker guarantee than a
lease and it is stated rather than implied: a merge enqueued while a run holds the card
waits for the worker, not for the lease. Reported.

## The record is written for the outcome that happened - all **three** of them

``written`` with the result checkpoint, ``refused`` with the raised refusal's code, or
``interrupted``. A **preflight** refusal - the route's AC17 check, before anything is
enqueued - writes no row; see :mod:`services.merge_record` for why that is deliberate
rather than an omission.

**``interrupted`` had no producer until 2026-10-14.** The constant, the ``outcome_known``
CHECK built from it and three docstrings explaining it all existed while
``grep -rn OUTCOME_INTERRUPTED backend/src`` matched the definition and nothing else - an
outcome the schema admits, the documents promise and no path could write, which is the same
unreachable capability as a function with no caller. It is produced in exactly two places
now, both of them here: the cooperative cancellation point **before the write**, and the
arm that catches a merge which started and did not finish. Without the second one a merge
killed on the card leaves **no row at all**, which an operator cannot tell apart from a
merge that was never enqueued - and that indistinguishability is the thing this table was
built to remove.

**One case records ``written`` from inside that arm, and it has to.** The directory write
is atomic (ADR-015), so an exception after the ``rename`` means the merged checkpoint is on
disk; an ``interrupted`` row then denies an artifact that exists, which ``written_iff_result``
exists to make unstorable. So the filesystem is probed and the manifest supplies the name.
"""

from __future__ import annotations

import logging
from typing import Any

from core.config import get_settings
from core.errors import Refusal
from core.logging import configure_logging
from workers.cancellation import RunCancelled
from workers.celery_app import QUEUE_TRAIN, celery_app

log = logging.getLogger(__name__)

#: The task's name. A constant so the route table, the enqueue call and the reachability
#: test refer to one string rather than three spellings of it.
TASK_MERGE_ADAPTER = "miforge.train.merge_adapter"


@celery_app.task(name=TASK_MERGE_ADAPTER, queue=QUEUE_TRAIN)  # type: ignore[untyped-decorator]
def merge_adapter(adapter_name: str, run_id: str, step: int) -> dict[str, Any]:
    """Perform one merge and record its outcome.

    Args:
        adapter_name: the adapter's qualified name, as its own manifest carries it.
        run_id: where the **merged checkpoint** is written - a merged checkpoint is a
            checkpoint of the run that asked for it.
        step: the step the merged checkpoint is written at.

    Returns:
        The recorded outcome. ``interrupted`` is returned rather than raised when the
        merge was cancelled before the write: the operator asked for it, so it is not a
        failure, and a task that raised would be a stop reported as an error.

    Raises:
        Refusal: ``acquisition_mode_unavailable`` on a build with no ML runtime - **named
            rather than guessed**, and recorded as a ``refused`` row so the request does
            not simply vanish; plus anything F8's ``merge`` raises, each recorded before it
            propagates.
        BaseException: anything else, re-raised **after** an ``interrupted`` row is
            recorded for it. The row is written first because the exception is the reason
            the row is the only evidence that remains.
    """
    configure_logging(get_settings().log_level)
    from db.session import session_scope
    from services import merge_record, runtypes_service

    with session_scope() as session:
        # AC17 again, in the worker. It ran at the route too, and running it twice is
        # deliberate: the route's check answered against a filesystem state that may have
        # changed, and the thing about to be made irreversible must be the thing that was
        # checked. It is a manifest read and a directory stat - cheap against a merge.
        admitted = runtypes_service.preflight_merge(session, adapter_name)

        try:
            outcome = _perform(
                session,
                adapter_name=adapter_name,
                run_id=run_id,
                step=step,
                admitted=admitted,
            )
        except RunCancelled:
            # Cancelled at one of the two points BEFORE the write (task 12.5). Nothing
            # irreversible happened, so this is not a failure and is not raised onward:
            # the operator asked for it. It is recorded, because a merge that was
            # enqueued and then silently did not happen is the "nothing happened and I
            # do not know why" case this table exists to prevent.
            recorded = merge_record.interrupted(
                adapter_name=str(admitted["name"]),
                adapter_step=int(admitted["step"]),
                adapter_manifest_sha256=_manifest_sha256(session, adapter_name),
                base_checkpoint_name=str(admitted["base_checkpoint"]),
            )
            merge_record.record(session, recorded)
            session.commit()
            return {
                "outcome": recorded.outcome,
                "adapter": recorded.adapter_name,
                "base_checkpoint": recorded.base_checkpoint_name,
                "result_checkpoint": None,
                "run_id": run_id,
                "step": step,
            }
        except Refusal as exc:
            # A merge that was ADMITTED and then refused on the card. This is the outcome
            # the table exists for: no checkpoint was written, so there is no manifest for
            # `reindex` to rebuild the row from, and this row is the only record there is.
            recorded = merge_record.refused(
                adapter_name=str(admitted["name"]),
                adapter_step=int(admitted["step"]),
                adapter_manifest_sha256=_manifest_sha256(session, adapter_name),
                base_checkpoint_name=str(admitted["base_checkpoint"]),
                refusal_code=exc.code,
            )
            merge_record.record(session, recorded)
            session.commit()
            raise
        except BaseException:
            # **Admitted, started, and did not finish for a reason that is not one of
            # ours**: a crash on the card, an OOM, a `SystemExit` from the worker's warm
            # shutdown, a `KeyboardInterrupt`. Without this arm such a merge leaves NO
            # ROW AT ALL, which is indistinguishable from a merge that was never
            # enqueued - and `interrupted` exists in the vocabulary, in the
            # `outcome_known` CHECK and in three docstrings precisely for it.
            #
            # `session.rollback()` first, and here it is right where in `reindex` it was
            # wrong: that bug discarded checkpoints ALREADY FLUSHED by work that had
            # succeeded, whereas this merge did **not** complete, so its partial index
            # work is not a record of anything. The artifact, if there is one, is on disk
            # and authoritative (ADR-004) and `reindex` rebuilds the row from its
            # manifest. Rolling back is also the only way the record survives a failure
            # that came FROM the database and poisoned the transaction - without it the
            # insert would raise on top of the exception and the row would be lost in the
            # one case where it is the only evidence.
            session.rollback()
            _record_the_interrupt(
                session,
                adapter_name=adapter_name,
                run_id=run_id,
                step=step,
                admitted=admitted,
            )
            raise

        merge_record.record(session, outcome)
        session.commit()
        return {
            "outcome": outcome.outcome,
            "adapter": outcome.adapter_name,
            "base_checkpoint": outcome.base_checkpoint_name,
            "result_checkpoint": outcome.result_checkpoint_name,
            "run_id": run_id,
            "step": step,
        }


def _check_cancelled(run_id: str) -> None:
    """Raise :class:`RunCancelled` if a stop has been requested for ``run_id``.

    **Its own fresh transaction**, which is the whole point and is F6's mutation M-2: the
    task's session was opened before a load that takes minutes, so a read through it
    returns the snapshot from before the operator pressed stop and the flag is invisible
    for the entire merge. That failure is silent - no error, no log line, just a merge
    that was cancelled and ran anyway, irreversibly.

    **An absent ``mf_runs`` row is reported, not treated as "not cancelled".** A merge is
    an operation and not a run (FR-006.55), so it has no row of its own; the ``run_id``
    it carries is the run the merged checkpoint is written into, and that run may or may
    not have a row - ``POST /adapters/{name}/merge`` does not require one. Where there is
    no row there is no ``cancel_requested_at`` column to read, so there is **no
    cancellation channel for this merge at all**, and that is logged by name rather than
    reported as a negative answer to a question nothing could have answered.
    """
    from workers.cancellation import stop_requested
    from workers.run_record import default_session_scope

    with default_session_scope() as fresh:
        try:
            requested = stop_requested(fresh, run_id)
        except Refusal as exc:
            if exc.code != "run_not_found":
                raise
            log.info(
                "this merge has no cancellation channel: its run has no mf_runs row",
                extra={
                    "code": "merge_cancellation_unavailable",
                    "run_id": run_id,
                    "note": (
                        "a merge is an operation and not a run, so cancellation is read "
                        "from the row of the run the merged checkpoint is written into; "
                        "there is none, so nothing could have asked for a stop"
                    ),
                },
            )
            return

    if requested:
        log.info(
            "cancellation observed before the merge wrote anything",
            extra={"code": "merge_cancelled", "run_id": run_id},
        )
        raise RunCancelled(run_id)


def _record_the_interrupt(
    session: Any,
    *,
    adapter_name: str,
    run_id: str,
    step: int,
    admitted: dict[str, Any],
) -> None:
    """Record an interrupted merge - as ``written`` if the artifact is nevertheless there.

    **The probe is not defensiveness, it is the constraint.** ``written_iff_result`` holds
    in both directions, so an ``interrupted`` row asserts *no checkpoint was produced*. F5's
    writer is atomic - temp directory, ``fsync``, ``rename`` (ADR-015) - which means the
    result either exists whole or does not exist, and an exception raised after the rename
    and before the outcome was constructed would otherwise be recorded as an interrupt
    that produced nothing while the merged checkpoint sat on disk. That is a lie about an
    irreversible operation, which is the one thing this table's constraints exist to make
    unstorable.

    If the directory is there but its manifest cannot be read, **nothing is recorded** and
    the reason is logged by name. Both available rows would be false - ``interrupted``
    denies an artifact that exists, ``written`` must name one and there is no name to give
    - so this refuses rather than guesses, and the original exception carries on.
    """
    from services import merge_record
    from training.runtypes.adapters.merge import OUTCOME_WRITTEN, MergeOutcome

    try:
        result_name = _result_checkpoint_on_disk(run_id, step)
        recorded = (
            merge_record.interrupted(
                adapter_name=str(admitted["name"]),
                adapter_step=int(admitted["step"]),
                adapter_manifest_sha256=_manifest_sha256(session, adapter_name),
                base_checkpoint_name=str(admitted["base_checkpoint"]),
            )
            if result_name is None
            else MergeOutcome(
                outcome=OUTCOME_WRITTEN,
                adapter_name=str(admitted["name"]),
                adapter_step=int(admitted["step"]),
                adapter_manifest_sha256=_manifest_sha256(session, adapter_name),
                base_checkpoint_name=str(admitted["base_checkpoint"]),
                result_checkpoint_name=result_name,
            )
        )
    except Exception:
        log.exception(
            "an interrupted merge could not be recorded; no row was written",
            extra={
                "code": "merge_outcome_unrecordable",
                "run_id": run_id,
                "adapter": adapter_name,
                "step": step,
            },
        )
        return

    merge_record.record(session, recorded)
    session.commit()


def _result_checkpoint_on_disk(run_id: str, step: int) -> str | None:
    """The merged checkpoint's name if the atomic write landed, else ``None``.

    Read from the manifest rather than rendered from ``(base_model, run_id, step)``,
    because a name this module composed would be a claim about an artifact it never saw -
    and the manifest on disk is authoritative (ADR-004).
    """
    from services.checkpoints import manifest as manifest_module
    from services.checkpoints.paths import checkpoint_dir

    directory = checkpoint_dir(run_id, step)
    if not (directory / manifest_module.MANIFEST_NAME).is_file():
        return None
    return str(manifest_module.read(directory)["name"])


def _perform(
    session: Any,
    *,
    adapter_name: str,
    run_id: str,
    step: int,
    admitted: dict[str, Any],
) -> Any:
    """Load the base, merge, write. Needs the ML runtime; refuses by name without it.

    **The two cooperative cancellation points are here, and both are before the write**
    (task 12.5). Not inside the merge and not inside F5's writer: a merge is irreversible
    and the directory write is atomic, so between those two facts there is no safe place
    to stop once the arithmetic has begun. ADR-006 forbids the alternative outright -
    ``revoke(terminate=True)`` signals a pool child, the solo pool has none, and the call
    returns successfully having done nothing.

    The first check is before the load, so a cancelled merge costs no VRAM at all. The
    second is after it, and it is the one that earns its keep: loading a multi-gigabyte
    base is the long part of a merge, and a stop pressed during it would otherwise be
    seen only after the irreversible step had already run.
    """
    from training.runtypes.adapters import merge as merge_module

    _check_cancelled(run_id)

    base_directory = admitted["base_checkpoint_path"]
    acquired = _load_base(base_directory)

    # The second point. The base is in hand and nothing has been written.
    _check_cancelled(run_id)

    return merge_module.merge(
        session=session,
        adapter_name=adapter_name,
        base_module=acquired.module,
        tokenizer=acquired.tokenizer,
        run_id=run_id,
        step=step,
        # Off the merged checkpoint's own lineage: the base checkpoint's manifest carries
        # both, and neither is defaulted - `domain=None` is an assertion and a defaulted
        # `signal_source` would be a fabricated provenance claim in a published artifact.
        domain=acquired.domain,
        signal_source=acquired.signal_source,
        optimizer=acquired.optimizer,
    )


class _LoadedBase:
    """The base checkpoint in hand, with the manifest values the merged save needs."""

    def __init__(self, module: Any, tokenizer: Any, manifest: dict[str, Any]) -> None:
        self.module = module
        self.tokenizer = tokenizer
        self.domain = manifest.get("domain")
        self.signal_source = manifest["signal_source"]
        self.optimizer = manifest.get("optimizer") or {}


def _load_base(directory: Any) -> _LoadedBase:
    """Load the unmerged base **through F1's own acquisition function**.

    :func:`training.tasks.acquire_for_run` rather than a ``from_pretrained`` of this
    module's own, for three reasons and one of them is structural:

    * it runs the acquirer's pre-flight, which confines the directory to a configured model
      root, refuses a missing file and refuses a **quantized** source - all of which a
      merge needs and none of which belongs in a worker task;
    * it refuses ``acquisition_mode_unavailable`` by name on a build with no ML runtime,
      instead of an ``ImportError``;
    * **it is the single ``acquirer.acquire`` call site in the product**, which
      ``test_budget_guard_ordering.py`` asserts structurally. A second call site here would
      have made that guard red - and it did, on the first draft of this module, which is how
      the next paragraph was found.

    **A merge allocates the base's VRAM with no budget check, and that is reported rather
    than guessed.** R-02.42's estimate is a *run*'s footprint: F8's ``adapter_budget``
    estimates weights plus adapter gradients plus optimizer state, which a merge has none
    of, and no requirement estimates a merge. Inventing a number here would be a fabricated
    figure in the one place a refusal is irreversible, so the parameter count the
    acquirer's pre-flight actually read is logged instead and the gap is in the feature
    report.
    """
    from pathlib import Path as _Path

    from services.checkpoints import manifest as manifest_module
    from training import tasks
    from training.acquisition.base import AcquisitionMode, AcquisitionSpec

    path = _Path(str(directory))
    doc = manifest_module.read(path)
    spec = AcquisitionSpec(mode=AcquisitionMode.LOAD, path=path)
    acquired = tasks.acquire_for_run(spec)
    log.info(
        "merge base loaded; no budget estimate exists for a merge",
        extra={
            "code": "merge_base_loaded",
            "base_checkpoint": doc.get("name"),
            "param_count": acquired.record.param_count,
            "note": (
                "R-02.42 estimates a run's footprint and no requirement estimates a "
                "merge, so this allocation is unguarded by design rather than by omission"
            ),
        },
    )
    return _LoadedBase(acquired.module, acquired.tokenizer, dict(doc))


def _manifest_sha256(session: Any, adapter_name: str) -> str:
    """The adapter manifest's hash, for a ``refused`` row's required field.

    Required by the table, because *a merge whose input cannot be identified afterwards is
    a lineage claim nothing can check*. Read through F8's own reader, so the hash on a
    refused row is the same value a written row would have carried.
    """
    from core.canonical import sha256_of
    from services import runtypes_service

    return sha256_of(runtypes_service.read_adapter(session, adapter_name)["manifest"])


__all__ = ["TASK_MERGE_ADAPTER", "merge_adapter"]
