"""``miforge.train.run_step_loop`` and the driver it calls  (R-02.12, R-02.15; ADR-006).

**The task name is ``miforge.train.run_step_loop``, not ``miforge.training.…``.** All four
of this feature's documents say the latter, and it routes to **no queue**: the foundation's
``TASK_ROUTES`` is an *explicit* table keyed ``miforge.train.*`` / ``miforge.generate.*`` /
``miforge.maintenance.*`` / ``miforge.index.*`` / ``miforge.packs.*``, and ADR-006 names the
areas after the queues they contend on. A task matching no entry lands on ``default``, where
a worker with **no card** picks it up and fails - precisely the failure the foundation made
the table explicit to prevent. Reported as a document defect; the code uses the name that
routes.

## Two layers, deliberately

:func:`drive_run` is a **plain function** taking its batch source as an argument, and
:func:`run_step_loop` is the Celery task that resolves one and calls it. The split is what
makes F1's whole path - acquire, record, optimizer, step, record, trainer state - testable
without a broker and without F2, while keeping the production entry point a real registered
task whose absence a test can detect.

## The repeat here is a temporary driver, and it is F6's to replace

R-02.24's loop is **F6's** (its FR-006.3 extracts this repeat into ``training/loop.py`` and
has this task delegate to it - declared ripple work, behaviour-preserving, with F1's tests as
the regression). F6 also fixes the boundary order as **step -> evaluate -> save -> stop ->
refresh** and adds an **evaluation at step 0, before the first step**, so the first
*completed* step is step 1 and no save happens at step 0. F1 asserts no ordering against that
sequence: it supplies **one turn** (:func:`training.step.run_one`) and the save-point hook.
**There is no second loop task** - F6 replaces this body rather than wrapping it, so the task
name stays stable for everything that enqueues it.

## Cancellation goes through the database, and nothing else can work

``revoke(terminate=True)`` signals a pool child and the solo pool has none, so the call
returns successfully and nothing happens - the worst available failure mode, because the
operator sees a cancellation that was accepted. A busy solo worker also never reads the
control queue. So the stop is a column on the run row, re-read in **its own fresh
transaction** at every step boundary (:class:`workers.cancellation.CancellationToken`), and
it lands **between** two steps: a step either completes and is recorded, or has not started.
"""

from __future__ import annotations

import logging
import sys
import traceback
from collections.abc import Mapping
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any

from core.config import get_settings
from core.errors import refuse
from core.logging import configure_logging, run_context
from core.runtime_settings import effective_settings

# ADR-009's provider registry. Imported at module scope because the in-process handle is
# bound on the critical path of every run: a lazy import inside the loop would make an
# unimportable registry fail at the first save point rather than when the worker starts.
from providers import registry as provider_registry

# F5's save path. Imported at module scope rather than inside the loop: a save that
# failed to import at the save boundary would fail a run that had already done its
# work, and a missing module should stop the worker starting, not stop step 100.
#
# `services.checkpoints.service` is **deliberately not among them any more**: since the loop
# was routed through `RunType.save` (2026-10-16) the only caller of F5's writer is
# `training/runtypes/change.py`, which imports it at ITS module scope for the same reason. A
# second import here would read as a second call site.
from services.checkpoints import manifest as checkpoint_manifest
from services.checkpoints import requests as save_requests
from services.checkpoints import resume as resume_module

# F8's ONE manifest-score join, reused rather than re-derived. `training/runtypes/change.py`
# owns `joined_scores`, shared by `change`, `create` (through `save_checkpoint`) and
# `adapter`; a fourth projection of the same two payloads here is how one of the four ends
# up disagreeing, and `ScoreRecordData.score_record` / `manifest_payload.required_keys` were
# both checked and are NOT alternative seams.
from training import acquisition, boundary, loop, runtypes, trainer_state
from training import optimizer as optimizer_module
from training import step as step_module
from training.acquisition.base import AcquiredModel, AcquisitionSpec
from training.run_config import RunConfig
from training.runtypes import base as runtypes_base
from training.runtypes import change as change_runtype
from training.step_record import StepWriter
from workers.celery_app import QUEUE_TRAIN, celery_app

log = logging.getLogger(__name__)

#: The task's name. A constant so the route table, the enqueue call and the reachability
#: test refer to one string rather than three spellings of it.
TASK_RUN_STEP_LOOP = "miforge.train.run_step_loop"

#: Run logs live in a sibling of the artifact roots. **Not inside the checkpoint root**: a
#: checkpoint directory that gains unrelated files stops being a standard Hugging Face
#: directory (R-02.51, acceptance criterion 23).
RUN_LOG_DIRNAME = "runs"


def run_log_root(root: Path | None = None) -> Path:
    """Where the authoritative run logs live (ADR-004)."""
    base = root if root is not None else get_settings().trainer_state_root.parent
    return base / RUN_LOG_DIRNAME


@dataclass
class RunResult:
    """What a driven run produced. A summary, never the record itself.

    Returned to Celery as the task's result, and Redis is the broker and the progress
    channel - **never a store of record** (ADR-003). The record is the run log.
    """

    run_id: str
    steps_completed: int
    last_step: int | None
    acquisition: dict[str, Any]
    trainer_states: list[str]
    #: **The artifact directories this run's save points published**, in order - checkpoint
    #: directories for a ``change`` or ``create`` run, adapter directories for an
    #: ``adapter`` one, because that is what the run type wrote (R-02.44). Separate from
    #: ``trainer_states`` because they are two artifacts with two lifetimes: the published
    #: artifact is the product and a trainer state exists only to continue from.
    checkpoints: list[str]
    #: One of R-02.25's five stop reasons, or ``step_budget_reached`` / ``cancelled``
    #: (:mod:`training.loop`). F6's addition: a run that ended because a condition fired
    #: says **which** condition, which is what an operator acts on.
    stopped_because: str
    #: The firing condition's own sentence, naming the number, its scale and its sample.
    stop_detail: str | None = None
    #: **Every** condition that fired at the terminating boundary, in precedence order
    #: (FR-006.27). Only the first becomes ``stopped_because``; none of the rest is lost.
    all_fired: tuple[str, ...] = ()
    #: How many rounds the run completed (FR-006.37: a round is ``refresh_every`` steps).
    rounds: int = 1
    #: One entry per boundary. Not the record - the run log is - but what the loop's own
    #: tests assert the phase order and the condition evaluations against.
    boundaries: list[Any] = field(default_factory=list)


def acquire_for_run(spec: AcquisitionSpec) -> AcquiredModel:
    """Acquire the parameters and return them with their record (R-02.46, R-02.47).

    Pre-flight has already run at submit; it runs again here because the worker is a
    different process and the thing it is about to allocate must be the thing that was
    checked. Re-running it is cheap - it reads a config and a header - and skipping it would
    mean the only check happened against a filesystem state that may have changed.

    **The acquirer's pre-flight is not the whole re-check, and this function is not where
    the rest of it lives.** It proves the weight files are present and the dtype loads; it
    never proves the bytes hash to what the manifest records. The hash half is
    :func:`verify_checkpoint_source`, called by :func:`drive_run` **before** this - because
    R-02.21 wants the verification before a single tensor is read, and this function reads
    them all. It is not folded in here because this function is also
    :mod:`workers.merge_tasks`' acquirer and :mod:`tests` double it by name with one
    positional argument; a second parameter list would silently un-double those.
    """
    acquirer = acquisition.resolve(spec)
    return acquirer.acquire(acquirer.preflight(spec))


def verify_checkpoint_source(
    spec: AcquisitionSpec,
    *,
    run_id: str,
    cfg: RunConfig,
    root: Path | None = None,
    trainer_state_root: Path | None = None,
) -> dict[str, Any] | None:
    """The worker's re-run of F5's checkpoint cases (R-02.21, R-02.22). The report, or ``None``.

    ``None`` when the source is not a miForge checkpoint - a ``create`` run and a plain
    Hugging Face ``load`` are untouched, which is :func:`services.acquisition.
    is_miforge_checkpoint`'s discrimination and not a second copy of it.

    **The gap this closes.** The submit path verifies (F5 task 7.8, wired into
    :func:`services.acquisition.preflight`), and the worker re-ran only *the acquirer's*
    pre-flight - present files and a loadable dtype, never the hash. So a checkpoint
    corrupted, truncated or swapped **between submit and acquire** - a half-finished rsync,
    a bad disk, a restore from the wrong backup - was loaded and trained on with no refusal
    anywhere. The worker is the process that matters: it is the one that allocates, and
    submit's verdict is minutes old by the time it does.

    **One implementation, not a second one.** This calls
    :func:`services.acquisition.preflight_checkpoint_source`, which is the same function the
    submit path calls, so the two paths cannot drift in what they accept. The spec is the
    one ``drive_run`` was handed rather than the acquirer's enriched copy: the only fields
    read are ``mode`` and ``path``, and the enrichment's job - confining the path to a model
    root - still happens inside :func:`acquire_for_run` before anything is opened.

    Raises:
        Refusal: :data:`services.acquisition.CHECKPOINT_PREFLIGHT_REFUSALS` - in particular
            ``checkpoint_hash_mismatch``, having read nothing into the model.
    """
    from services import acquisition as acquisition_service

    return acquisition_service.preflight_checkpoint_source(
        spec,
        run_id=run_id,
        configured_config_hash=cfg.config_hash(),
        root=root,
        trainer_state_root=trainer_state_root,
    )


def planned_restore_step(report: Mapping[str, Any] | None) -> int | None:
    """Which step's trainer state the **plan** decided to restore, or ``None``.

    The decision is **read, not retaken**: :func:`services.checkpoints.resume.resume_plan`
    already weighed the four trainer-state outcomes (present and complete, absent, present
    for another step, torn) and a second judgement here would be a second answer. Three
    conditions have to hold, and each is a distinct fact:

    * the trainer-state case was actually **checked** - without a ``run_id`` the plan
      reports it unchecked rather than deciding, and "unchecked" is not "fresh";
    * the plan is **not a branch** - trainer state is addressed by ``run_id`` (R-02.15), so
      a branch has none of its own and is ``fresh`` by construction (FR-005.38);
    * the verdict is ``restored``.
    """
    if not report:
        return None
    if not report.get("trainer_state_checked") or report.get("is_branch"):
        return None
    if report.get("optimizer_state") != resume_module.RESTORED:
        return None
    step = report.get("step")
    return int(step) if isinstance(step, int) else None


def resume_target(report: Mapping[str, Any] | None) -> dict[str, Any] | None:
    """The ``resume_from`` block of a checkpoint report, or ``None`` when this is no resume.

    Set by :func:`services.acquisition.preflight_checkpoint_source` for a spec naming
    ``resume_from`` (D52). Read by presence - no acquisition mode is compared here.
    """
    if not report:
        return None
    block = report.get("resume_from")
    return dict(block) if isinstance(block, Mapping) else None


def require_resume_verified(target: Mapping[str, Any], verified: Any, *, run_id: str) -> None:
    """Refuse a resume whose acquisition did not prove the weights are the checkpoint.

    Raises:
        Refusal: ``resume_identity_unverified`` - before any optimizer state is restored.
    """
    proved = verified.record() if callable(getattr(verified, "record", None)) else None
    if (
        proved is not None
        and proved.get("run_id") == target.get("run_id")
        and proved.get("step") == target.get("step")
        and proved.get("manifest_file_sha256") == target.get("manifest_file_sha256")
    ):
        return
    raise refuse(
        "resume_identity_unverified",
        f"This run resumes from checkpoint {target.get('name')!r}, and nothing proved the "
        f"acquired weights are that checkpoint, so its optimizer state will not be restored "
        f"onto them. The acquisition must check the served model against the checkpoint "
        f"before the loop restores anything.",
        run_id=run_id,
        expected=dict(target),
        verified=proved,
    )


def resume_start_step(
    target: Mapping[str, Any] | None, start_step: int, *, run_id: str, log_path: Path
) -> int:
    """The step a resume continues from: the checkpoint's. ``start_step`` when no resume.

    Steps the run log already holds after it are SUPERSEDED - the new attempt writes them
    again, and the step index keeps the later record (:func:`training.run_log.step_rows`).
    Logged by name with the range; the continue route records the same range in the run
    log's definition block, which is the durable record of the discontinuity.
    """
    if target is None:
        return start_step
    from training import run_log

    step = int(target["step"])
    logged = [record["step"] for record in run_log.read_blocks(log_path, block="step")]
    last = max(logged) if logged else None
    if last is not None and last > step:
        log.warning(
            "resuming before the run's last recorded step; the later steps are superseded",
            extra={
                "code": "resume_discontinuity",
                "run_id": run_id,
                "resumed_from_step": step,
                "superseded_first": step + 1,
                "superseded_last": last,
                "checkpoint": target.get("name"),
            },
        )
    return step


def restore_optimizer_state(
    optimizer: Any,
    identity: optimizer_module.OptimizerIdentity,
    *,
    run_id: str,
    step: int,
    root: Path | None = None,
) -> optimizer_module.OptimizerIdentity:
    """Load ``step``'s optimizer state into ``optimizer``. Returns the identity **as it is**.

    R-02.22 and acceptance criterion 4: *"a run continued from a saved checkpoint with
    restored optimizer state produces a loss curve continuous with the original."*
    Continuity is a property of the AdamW moment buffers, so without this call the curve
    steps by the width of one un-warmed optimizer - measured at ~0.4 in loss units on
    ``test_continuity_across_a_resume.py``'s model, about 40,000x its tolerance.

    Read through F1's own :func:`training.trainer_state.read`, which refuses an incomplete
    directory by name: continuing from a torn state does not crash, it *runs*, and produces
    the unexplained discontinuity R-02.22 exists to prevent.

    **The returned identity is what happened, not what was planned.** A manifest claiming
    ``restored`` over an optimizer that is in fact fresh is worse than one admitting
    ``fresh`` - it makes a real discontinuity unexplainable. So a state dict that comes back
    carrying no per-parameter entries (the state was written before the optimizer had taken
    a step, so there is nothing in it to restore) is **recorded as** ``fresh`` and logged by
    name. Every other failure raises, and a raise writes no manifest at all.
    """
    state = trainer_state.read(run_id, step, root=root)
    optimizer.load_state_dict(state["optimizer"])

    loaded = (optimizer.state_dict() or {}).get("state") or {}
    if not loaded:
        log.warning(
            "trainer state restored but it carries no optimizer state; recording fresh",
            extra={
                "code": "trainer_state_restored_empty",
                "run_id": run_id,
                "step": step,
                "reason": (
                    "the restored state_dict has no per-parameter entries, so the saved "
                    "optimizer had taken no step. The manifest records `fresh`, because "
                    "`restored` over an effectively fresh optimizer makes a real loss "
                    "discontinuity unexplainable."
                ),
            },
        )
        return replace(identity, state=resume_module.FRESH)

    log.info(
        "optimizer state restored",
        extra={
            "code": "optimizer_state_restored",
            "run_id": run_id,
            "step": step,
            "parameters_with_state": len(loaded),
        },
    )
    return identity


def drive_run(
    *,
    run_id: str,
    spec: AcquisitionSpec,
    cfg: RunConfig,
    batch_source: step_module.BatchSource,
    objectives: step_module.ObjectiveRegistry | None = None,
    session: Any = None,
    cancel: step_module.Cancellation | None = None,
    heartbeat: Any = None,
    emit: Any = None,
    max_steps: int = 1,
    save_every: int | None = None,
    log_root: Path | None = None,
    trainer_state_root: Path | None = None,
    domain: str | None = None,
    signal_source: str | None = None,
    # ---- F8's dispatch (R-02.41, R-02.43). `"change"` is F1's default path, so this
    # ---- function's existing tests are the regression for the routing.
    run_type: str = "change",
    adapter: Any = None,
    # ---- F6's parameters (R-02.24 - R-02.27). Every one defaults to F1's prior
    # ---- behaviour, so this function's existing tests are the regression for the
    # ---- delegation and the production driver (`run_step_loop`) sets the real values.
    stop_conditions: Mapping[str, Mapping[str, Any]] | None = None,
    test_every: int = 1,
    retention_every: int = 1,
    refresh_every: int | None = None,
    refresh_mode: str | None = None,
    refresh_rule: Mapping[str, Any] | None = None,
    apply_refresh: Any = None,
    evaluate: Any = None,
    session_scope: Any = None,
    start_step: int = 0,
    evaluate_step_zero: bool = False,
    run_end_save: bool = False,
) -> RunResult:
    """Acquire, then take ``max_steps`` steps, recording each one.

    **F1's whole path, in the order the requirements fix it:**

    0. verify the checkpoint source, **before a tensor is read** (R-02.21) -
       :func:`verify_checkpoint_source`;
    1. acquire (R-02.46) and **record the acquisition before the first step** (R-02.47) -
       a run's lineage must exist even if step 1 fails;
    2. build the optimizer (R-02.13), **restoring its state when the plan said restored**
       (R-02.22, acceptance criterion 4) - :func:`restore_optimizer_state`;
    2a. ask the **run type** what is trainable (R-02.41, R-02.43) -
       :meth:`training.runtypes.base.RunType.prepare`, **before** the optimizer, because
       that is what freezes an adapter run's base;
    3. per step: the run type's own ``step`` - the single :data:`training.runtypes.base.STEP`
       object all three carry (R-02.12) - then the step record (R-02.14) in the order
       log -> index -> socket -> heartbeat;
    4. on a save point, the trainer state (R-02.15), **for the same step number as the
       weights save**, then the run type's artifact
       (:meth:`training.runtypes.base.RunType.save`).

    ## The dispatch, and what it was before  (ADR-016 #1)

    ``RunType.prepare`` and ``RunType.save`` had **no production caller** until 2026-10-16.
    ``services.run_definition`` called ``preflight`` and this loop went straight to
    :func:`training.step.run_one` over ``acquired.module`` and to F5's writer, so:

    * **an adapter run optimised every parameter.** ``optimizer_module.build`` was handed
      the whole module and :func:`training.runtypes.adapters.build.
      assert_only_adapter_requires_grad` never ran, so R-02.43's entire economy - no
      full-parameter gradients, no FP32 master copy for the base, no barrier - did not
      happen. *The cheapest run type was the dearest.*
    * **an adapter run's save wrote a full checkpoint** through F5 rather than the PEFT
      directory R-02.44 defines and acceptance criterion 15 is about.

    Both were reported here rather than fixed, which by this project's own rule means
    neither run type was shipped. The dispatch is three calls - ``resolve``, ``prepare``,
    ``save`` - and **no ``if run_type ==`` anywhere**: the registry is the conditional
    (:mod:`training.runtypes`).

    Args:
        max_steps: how many steps this driver takes. **F6's stop conditions (R-02.25)
            replace this**; it exists so the temporary driver terminates at all rather than
            spinning against a loop condition this feature does not own.
        save_every: the save cadence. **R-02.24's own name for it and F6's to validate and
            carry on the run definition** - F1 writes trainer state at the save points the
            loop signals and defines no cadence field of its own. ``None`` writes none.
        domain: the run's domain, or ``None`` asserting it is not domain-scoped. **F6's**,
            off R-02.53's run definition. Nullable because general pretraining has no
            domain, and null is a statement where an absent key would be an omission.
        signal_source: which source filled the batches. **F2's.** Required and **not**
            nullable in the manifest, so without it a weights checkpoint cannot be written
            at all - the save point writes trainer state, logs
            ``checkpoint_save_skipped_no_provenance`` naming who owes the value, and the run
            stays continuable. Passing a placeholder here would write a fabricated
            provenance claim into a published artifact, which is why there is no default.
        run_type: one of ``change | create | adapter``, **resolved through F8's registry**
            and never matched against a literal. Defaults to ``"change"``, which is F1's
            path and this function's prior behaviour, so the existing tests are the
            regression for the routing rather than a rewrite of them.
        adapter: F8's adapter block, for an ``adapter`` run. Opaque here - the run type
            validates it, twice: once at submit in ``preflight`` and once at the save, from
            **one object**, because miLLM's loader cross-checks the manifest's numbers
            against ``adapter_config.json``'s.
    """
    writer = StepWriter(
        run_id=run_id,
        log_root=run_log_root(log_root),
        heartbeat=heartbeat,
        emit=emit,
    )

    # THE CHECKPOINT RE-CHECK, **BEFORE A TENSOR IS READ** (R-02.21). The acquirer's own
    # pre-flight, inside `acquire_for_run` below, checks presence and dtype and never the
    # hash - so this is the only thing between a checkpoint corrupted after submit and a run
    # that trains on it. It returns `None` for a `create` run and for a plain Hugging Face
    # `load`, which is why it sits unconditionally on the path rather than behind a mode
    # test here.
    checkpoint = verify_checkpoint_source(
        spec, run_id=run_id, cfg=cfg, trainer_state_root=trainer_state_root
    )

    acquired = acquire_for_run(spec)
    # R-02.2, BEFORE ANYTHING ELSE TOUCHES THE PARAMETERS (F11 FR-011.12): a rebuilt-from-
    # handles graph that disagrees with what the serving process answers is refused here,
    # before any step, test or save. Presence dispatch, as with `stepper` and `release`.
    # The attachment is released on a refusal - the `finally` below has not been entered yet.
    # *Until 2026-10-16 nothing called the check: the attach mode trained an unverified graph.*
    # And `bind_run` first, inside the same guard: it persists the attachment and publishes
    # its state, and a failure there must release the attachment just as a refused check does.
    resuming = resume_target(checkpoint)
    if acquired.verifier is not None or acquired.bind_run is not None or resuming is not None:
        try:
            if acquired.bind_run is not None:
                acquired.bind_run(run_id=run_id, session_scope=session_scope, session=session)
            if acquired.verifier is not None:
                acquired.verifier()
            # D52: A RESUME RESTORES OPTIMIZER STATE ONLY ONTO WEIGHTS PROVED TO BE THE
            # CHECKPOINT. The acquisition's own check (`training/hotupdate/resume.py`, run on
            # the attach response) is what proves it; this is the loop refusing to restore
            # without that proof, so removing either half still refuses. After R-02.2, so a
            # graph that disagrees with the serving process is refused for that first.
            if resuming is not None:
                require_resume_verified(resuming, acquired.resume_verified, run_id=run_id)
        except BaseException as exc:
            if acquired.release is not None:
                _release_acquisition(acquired.release, run_id=run_id, failing=exc)
            raise
    # THE ACQUIRED TOKENIZER REACHES THE BATCHES. The source's `RunContext` is built by the
    # run driver BEFORE acquisition (it has to pre-flight first), so it carries no tokenizer,
    # and every batch it builds takes `context.tokenizer` - which the objectives encode with.
    # Until 2026-10-17 nothing bound it: every production run, of every mode, would have
    # refused `tokenizer_unavailable` at step 1, while every test injected a tokenizer into
    # its own RunContext or used a stub source that never tokenizes (found preparing F11
    # 14.4). An explicitly supplied tokenizer is kept.
    bind_tokenizer(batch_source, acquired.tokenizer)
    # BEFORE THE FIRST STEP. R-02.47: the lineage root has to exist even if step 1 fails,
    # which is why this is not deferred to the first save.
    record = acquired.record.model_dump(mode="json")
    log.info("acquisition recorded", extra={"code": "acquisition_recorded", **record})

    # ---------------------------------------------------------------------------------
    # F8'S DISPATCH, STEP ONE OF THREE: WHAT IS TRAINABLE  (R-02.41, R-02.43)
    #
    # Resolved by name through the ONE registry - `runtypes.resolve` refuses an unknown name
    # listing the three - and `prepare` runs **before the optimizer is built**, which is the
    # whole point: for an `adapter` run it is `prepare` that wraps the base with `peft` and
    # freezes it, so an optimizer built first would cover every parameter. `change` and
    # `create` return `all_parameters`, whose `module` IS `acquired.module`, so the default
    # path is byte-identical to what it was.
    # ---------------------------------------------------------------------------------
    run_type_impl = runtypes.resolve(run_type)
    plan = run_type_impl.prepare(acquired, adapter=adapter)

    # The provider names the parameter set the STEP forwards through, which is `plan.module`
    # and not always `acquired.module`: for an adapter run the step forwards through the
    # `peft` wrapper, and a score taken from the unwrapped base would measure a model without
    # the adapter the run exists to train. Identity-compared rather than replaced
    # unconditionally so that `change`/`create` bind the very object they always did.
    stepped = acquired if plan.module is acquired.module else replace(acquired, module=plan.module)

    # THE OPTIMIZER STATE RESTORE (R-02.22, acceptance criterion 4). The plan's verdict is
    # honoured rather than retaken - see `planned_restore_step` - and the identity handed to
    # F5's manifest writer below is `restore_optimizer_state`'s return value, which is what
    # actually happened rather than what was planned.
    #
    # Built over `plan.module`, so `build`'s own `requires_grad` filter collects the adapter's
    # parameters ALONE on an adapter run (FR-008.19) - gradients and state tens of megabytes
    # rather than gigabytes - and every parameter on the other two.
    restore_step = planned_restore_step(checkpoint)
    # A RESUME CONTINUES FROM THE CHECKPOINT'S STEP, not from the run log's last step: the
    # weights ARE that step's, so numbering from a later one would label step N+1's update
    # as step L+1. Steps after it that the log already holds are superseded and named.
    start_step = resume_start_step(resuming, start_step, run_id=run_id, log_path=writer.log_path)
    optimizer, identity = optimizer_module.build(
        plan.module,
        cfg,
        state=resume_module.RESTORED if restore_step is not None else resume_module.FRESH,
    )
    if restore_step is not None:
        identity = restore_optimizer_state(
            optimizer, identity, run_id=run_id, step=restore_step, root=trainer_state_root
        )

    # A killed writer leaves its in-flight directory behind; the run's own next save tidies
    # the wreck of its last one (E-10). Done here rather than on a schedule so nothing
    # sweeps a directory another process is writing into.
    trainer_state.clean_stale_temp_dirs(run_id, root=trainer_state_root)

    written_states: list[str] = []
    #: The steps this process already saved. See `save_at`'s first lines.
    saved_steps: set[int] = set()
    written_checkpoints: list[str] = []

    def save_context(step: int, source: str) -> runtypes_base.SaveContext:
        """F8's :class:`~training.runtypes.base.SaveContext` for one save point.

        **The one construction**, handed to whichever run type is resolved and read by
        :func:`training.runtypes.change.manifest_fields_from` for the attached path too. A
        second one built per branch is how two manifests drift on the first field anyone
        adds, with each branch's own tests passing.

        ``module`` is ``plan.module`` rather than ``acquired.module``: for an ``adapter`` run
        the thing with ``save_pretrained`` that writes **the adapter and not the base** is
        the ``peft`` wrapper, and for the other two they are the same object.
        """
        return runtypes_base.SaveContext(
            run_id=run_id,
            step=step,
            session=session,
            module=plan.module,
            # Passed explicitly, never defaulted. `None` here is an assertion that this run
            # is not domain-scoped, which is a legitimate state for general pretraining and
            # is why the field is nullable.
            domain=domain,
            signal_source=source,
            record=acquired.record,
            optimizer=identity.as_manifest_block(),
            tokenizer=acquired.tokenizer,
            adapter=adapter,
            # `None` is the configured root for BOTH artifact kinds - an adapter lives at
            # `<checkpoint_root>/<run_id>/adapters/step-NNNNNN`, inside the tree `reindex`
            # walks, not in a root of its own.
            root=None,
            emit=emit,
            heartbeat=heartbeat,
        )

    def manifest_fields_for(step: int, source: str) -> checkpoint_manifest.ManifestFields:
        """F5's ``ManifestFields`` for one save point - **with R-02.16's two score records.**

        ## The gap this closes

        Both save branches below built their own ``ManifestFields`` and **neither passed
        ``test`` or ``retention``**, so every production checkpoint manifest recorded
        ``test: null, retention: null`` - which the schema defines as *"no test ran at this
        step"* - while ``mf_scores`` and the run log held real numbers for that very step.
        Both properties are nullable, so the manifest **validated clean**; the schema-sync
        test compares schema *files* and cannot see a payload at all. R-02.16's provenance
        was absent from the one artifact that travels without its run log.

        ## One construction, not two, and now not three either

        The two blocks were identical apart from the writer they fed, so a field added to
        one would have been missing from the other - and that change had to add two.
        **Since 2026-10-16 this function is reached only by F11's attached path**: a
        ``change``/``create`` save goes through
        :meth:`training.runtypes.change.ChangeRunType.save`, which builds the same object
        from the same context through :func:`training.runtypes.change.manifest_fields_from`.
        So this delegates to that function rather than restating its field list - otherwise
        the attached manifest and the standalone one would be two projections again, which
        is the shape the paragraph above is about.

        ``test`` and ``retention`` are both annotated *"Written by the trainer"* in
        ``checkpoint-manifest-v1.json``, so :func:`training.hotupdate.save.caller_manifest`
        passes them through rather than rejecting them.

        Args:
            step: the save point's step. The join reads the index for **this** step, so a
                save at a step evaluated earlier in the boundary still carries that step's
                scores.
            source: the batch's provenance, passed narrowed rather than read off the
                closure - the caller has already refused the ``None`` case.
        """
        # F1's lineage pair verbatim, which is `ChangeRunType.save`'s own rule: `record`'s
        # two fields were already derived by F1's lineage functions and are not re-derived.
        return change_runtype.manifest_fields_from(
            save_context(step, source),
            base_model=acquired.record.base_model,
            parent=acquired.record.parent,
        )

    # ---------------------------------------------------------------------------------
    # THE REPEAT IS F6'S  (R-02.24, F6 FR-006.1)
    #
    # This function's `for step in range(...)` moved to `training/loop.py` on 2026-10-14.
    # F1's own TDD declared the ripple ("F6 owns the repeating") and F1's task tests are the
    # regression for it - behaviour here is unchanged, which is why the two closures below
    # are the old body verbatim rather than a rewrite. There is now exactly ONE step loop in
    # the product and `test_one_repeat.py` asserts it.
    #
    # F1 keeps the two phases that are its own - one turn, and the save point - and hands
    # them in as callables. F6 decides *when*, *how often* and *when to stop*.
    # ---------------------------------------------------------------------------------

    # ---------------------------------------------------------------------------------
    # WHICH STEP: the run type's, or the acquisition's own when it writes another process's
    # tensors (R-02.52). Dispatched on the PRESENCE of `acquired.stepper` - never on the mode
    # - exactly as the save path below dispatches on `acquired.saver` (ADR-022).
    #
    # **And only when the step updates the acquired parameters themselves.** An adapter run
    # trains a `peft` wrapper (`plan.module is not acquired.module`) whose base is frozen, so
    # it writes no resident tensor and must open no bracket (AC-011.18 asserts zero) -
    # quiescing serving for a write that does not happen would be an outage for nothing.
    #
    # *Until 2026-10-16 this line did not exist and every step was `run_type_impl.step`*:
    # an attached `change` run called `optimizer.step()` on miLLM's resident tensors with no
    # quiesce and no commit bracket, while F11's `hot_update_step` - the whole of R-02.52 -
    # had no production caller. Its tests drove it directly, so the suite was green.
    # ---------------------------------------------------------------------------------
    step_impl = (
        acquired.stepper
        if acquired.stepper is not None and plan.module is acquired.module
        else run_type_impl.step
    )

    def take_step(step: int) -> Any:
        """F1's one turn: the run type's step, the divergence check, then the step record.

        **Through** ``run_type_impl.step`` **rather than** :func:`training.step.run_one`
        **directly.** All three run types carry the same object
        (:data:`training.runtypes.base.STEP` is ``run_one``, asserted by identity in
        ``test_runtype_registry.py``), so this is the same call it always was - and it is
        the registry's attribute that is reached, which is what makes "three run types share
        one loop" a wiring a test can remove rather than a comment.
        """
        outcome = step_impl(
            plan.module,
            optimizer,
            batch_source,
            cancel=cancel,
            objectives=objectives,
            # The acquisition's stepper is told which step it is taking, so an interrupted
            # commit's abandonment record names it (FR-011.42). The run type's own step
            # (`run_one`) does not take the argument.
            **({"step": step} if step_impl is not run_type_impl.step else {}),
        )

        # ---- THE PARAMETERS HAVE MOVED, SO THE DIGEST MOVES WITH THEM  (ADR-009) ----
        # Immediately after `run_one` returns, and **before** the divergence check and the
        # step record below. `step._apply_update` calls `optimizer.step()` on every path -
        # including a non-finite loss, which is returned rather than raised - so by the
        # time control is here the held parameter set IS step `step`'s, whether or not the
        # step is recorded as completed. Rebinding after `writer.write` instead would leave
        # the in-process provider naming step `step - 1`'s parameters on the divergence
        # path and on any writer failure: a score attributed to weights that no longer
        # exist, which is the one thing `parameter_set_digest` exists to prevent.
        provider_registry.bind_acquired_model(stepped, step=step)

        if not outcome.finite:
            # E-12: *a missing number is indistinguishable from a healthy one*, so the value
            # is REPORTED rather than dropped - but it cannot go in the run log. JSON has no
            # NaN, `canonical_bytes` refuses it with `allow_nan=False`, and emitting the
            # literal `NaN` would produce a log that validates on the way out and fails on
            # the way back in. So the number is carried in the structured log at ERROR and
            # in the refusal's details (ADR-021), both as a string, and the step is not
            # recorded as completed.
            log.error(
                "step diverged: a non-finite loss or gradient norm",
                extra={
                    "code": "step_diverged",
                    "run_id": run_id,
                    "step": step,
                    "loss": str(outcome.loss),
                    "grad_norm": str(outcome.grad_norm),
                    "objective": outcome.objective,
                    "effective_batch_size": cfg.effective_batch_size,
                },
            )
            raise refuse(
                "step_diverged",
                f"Step {step} produced a non-finite loss ({outcome.loss}) or gradient norm "
                f"({outcome.grad_norm}). The step is not recorded as completed and the run "
                f"stops here. Lower the learning rate or check the batch.",
                run_id=run_id,
                step=step,
                loss=str(outcome.loss),
                grad_norm=str(outcome.grad_norm),
            )

        writer.write(session=session, step=step, outcome=outcome, cfg=cfg)
        return outcome

    def save_requested(step: int) -> Any:
        """Whether an operator asked for a save, **consumed** here (F5 FR-005.1)."""
        del step
        return save_requests.consume_save_request(session, run_id) if session is not None else None

    def save_at(step: int, requested: Any = None) -> None:
        """One save point: trainer state, then the weights."""
        # ONE SAVE PER STEP. F6 calls save twice when the last step lands on the cadence
        # (the cadence save, then the run-end save) and leaves de-duplication to the save
        # (S-2). F5's weights save is idempotent on a matching hash, but trainer state's
        # `write` refuses to overwrite a published step - so every run whose final step
        # was a multiple of save_every was marked FAILED `artifact_already_exists` after
        # its last step (found live 2026-10-18, F11 run f11w2-r7: 1,000 steps, save_every
        # 50). Nothing trained between the two calls, so the second is the same save.
        if step in saved_steps:
            log.info(
                "save point already written at this step; not writing it again",
                extra={"code": "save_point_already_written", "run_id": run_id, "step": step},
            )
            return
        saved_steps.add(step)
        # TRAINER STATE FIRST, then the weights. The order matters on a crash between
        # them: trainer state without a checkpoint is harmless - a continue refuses on
        # the missing checkpoint - while a checkpoint without trainer state is a
        # checkpoint that cannot be continued from, which looks complete and is not.
        written_states.append(
            str(
                trainer_state.write(
                    run_id=run_id,
                    step=step,
                    optimizer=optimizer,
                    identity=identity,
                    root=trainer_state_root,
                )
            )
        )

        # THE WEIGHTS SAVE (R-02.20), F5's path. It needs two values F1 does not own, and
        # **F6 supplies both now** (2026-10-14): `signal_source` names which source filled
        # the batch (F2's) and `domain` comes off R-02.53's run definition (F6's).
        # `ManifestFields` gives neither a default, deliberately - *null is a statement and
        # an absent key is an omission*, and a defaulted `signal_source` would be a
        # fabricated provenance claim in a published artifact.
        #
        # The loudly-logging pass-through below therefore no longer fires on the production
        # path: `run_step_loop` reads both from the recorded definition. It is kept for the
        # case it was written for - a caller that has neither - because skipping silently is
        # the one thing not done: an operator whose cadence passed with no checkpoint must
        # be told which feature owes the value.
        if signal_source is None:
            log.warning(
                "save point reached with no weights checkpoint written",
                extra={
                    "code": "checkpoint_save_skipped_no_provenance",
                    "run_id": run_id,
                    "step": step,
                    "reason": (
                        "signal_source is unknown, and the manifest's field is required "
                        "and not nullable. F2 names the source that filled the batch; "
                        "F6 carries `domain` on the run definition. Trainer state WAS "
                        "written, so the run is continuable."
                    ),
                    "owed_by": ["F2 signal_source", "F6 domain"],
                },
            )
        elif acquired.saver is not None:
            # THIS ACQUISITION BRINGS ITS OWN SAVE PATH, and dispatching on that is what
            # keeps ADR-022's fence - there is no `mode == "attach"` here, and this branch
            # is dead for every mode in F1 because `load` and `create` leave `saver` None.
            #
            # It exists because an attached acquisition holds NEITHER the tensors NOR the
            # tokenizer: `AcquiredModel.tokenizer` is None in that mode, so the call below
            # raised `AttributeError: 'NoneType' object has no attribute 'save_pretrained'`
            # at the first save point of every attached run - a traceback, not a refusal,
            # exactly where a run first tries to checkpoint. (OQ-5 / AMB-M, settled
            # 2026-10-14: F11 owns the attached-mode save. It calls F5's own indexer and
            # writes no bytes; the artifact is the serving process's under R-01.10-13.)
            saved = acquired.saver(
                fields=manifest_fields_for(step, signal_source),
                session=session,
                emit=emit,
                heartbeat=heartbeat,
            )
            written_checkpoints.append(str(saved.directory))
        else:
            # -------------------------------------------------------------------------
            # F8'S DISPATCH, STEP THREE OF THREE: WHICH ARTIFACT KIND  (R-02.43, R-02.44)
            #
            # `change`/`create` -> F5's checkpoint writer, a standard Hugging Face directory
            # with the tokenizer's own files at its ROOT (R-02.21, ADR-015).
            # `adapter`        -> a PEFT directory: `adapter_model.safetensors`,
            # `adapter_config.json` and miForge's additive `adapter_manifest.json`, tensor
            # keys in PEFT's convention so the thing loads under plain `peft`.
            #
            # **Not a branch here.** The run type was resolved once, at the top, and this is
            # one call; before 2026-10-16 this line was F5's writer unconditionally, so an
            # adapter run published a full checkpoint of its base and R-02.44's artifact was
            # not what any run produced.
            ctx = save_context(step, signal_source)
            run_type_impl.save(ctx)
            # WHERE IT LANDED, reported by the writer through the context - the loop reads
            # neither return value, because the two writers return different shapes and
            # reaching for `.directory` on whichever came back would be the dispatch's
            # conditional wearing a `getattr`.
            written_checkpoints.extend(ctx.artifacts)

        if requested is not None:
            log.info(
                "operator-requested save completed",
                extra={
                    "code": "checkpoint_save_request_served",
                    "run_id": run_id,
                    "step": step,
                    # How many requests collapsed into this one save. The effect
                    # collapses; the fact that two were asked for does not.
                    "request_count": requested["request_count"],
                    "also_on_cadence": save_every is not None and step % save_every == 0,
                },
            )

    #: What the last `save_requested` consumed, so `save_at` can log the collapse count.
    pending_request: dict[str, Any] = {}

    def _consume(step: int) -> Any:
        pending_request["value"] = save_requested(step)
        return pending_request["value"]

    def _save(step: int) -> None:
        save_at(step, requested=pending_request.pop("value", None))

    boundary_ctx = boundary.BoundaryContext(
        run_id=run_id,
        session=session,
        session_scope=session_scope,
        heartbeat=heartbeat,
        emit=emit,
        log_path=writer.log_path,
        take_step=take_step,
        evaluate=evaluate,
        save=_save,
        save_requested=_consume,
        stop_conditions=dict(stop_conditions or {}),
        test_every=test_every,
        retention_every=retention_every,
        save_every=save_every,
        refresh_every=refresh_every,
        refresh_mode=refresh_mode,
        refresh_rule=dict(refresh_rule or {}),
        source=batch_source,
        apply_refresh=apply_refresh,
    )

    # ---------------------------------------------------------------------------------
    # ADR-009'S DEFAULT INFERENCE PROVIDER, MADE TO EXIST  (R-02.16, R-02.17; AC20)
    #
    # `providers.registry.get("in_process")` refused `inference_provider_unavailable` in
    # **every** process before this line, because `bind_model_handle` had no production
    # caller - so F4's test and retention scoring refused in the one worker that holds the
    # parameters, and its refusal message claimed the provider "is available in the
    # training worker once the model is acquired", which nothing made true.
    #
    # It sits HERE rather than beside the acquisition so that nothing can raise between
    # the bind and the `try` below: the handle must not outlive the run. A solo worker
    # (ADR-006) runs one run after another in one process, and a leaked handle would make
    # the next process-level listing - and any score taken before the next bind - report
    # the **previous** run's parameters by name.
    #
    # `step=start_step` is the honest name for what is loaded: 0 for a fresh run's
    # as-acquired weights, the resumed step on a continue. `take_step` rebinds after every
    # update, which is what keeps `assert_pin_unchanged`'s own note - that the in-process
    # digest "changes at **every** step by design" - a true statement about this process.
    # ---------------------------------------------------------------------------------
    provider_registry.bind_acquired_model(stepped, step=start_step)
    try:
        outcome = loop.run(
            boundary_ctx,
            max_steps=max_steps,
            start_step=start_step,
            # Both default OFF here and are set by `run_step_loop`, which is the production
            # driver. F1's own task tests assert this function's save counts exactly (two
            # saves for `max_steps=4, save_every=2`, none for `save_every=None`), and they
            # are the declared regression for this delegation - so the defaults preserve
            # what they measure and the production path turns the two loop-owned
            # behaviours on.
            evaluate_step_zero=evaluate_step_zero,
            run_end_save=run_end_save,
        )
    finally:
        # THE HANDLE DOES NOT OUTLIVE THE RUN. In a `finally` rather than after the call,
        # because a refused or cancelled run leaves the worker process alive and holding
        # the next run - and `get("in_process")` must refuse by name there rather than
        # resolve a dead run's parameters. Unbinding is also what makes the standalone
        # control assertable: a test that bound a provider and could not bind it away
        # could not express the unconfigured world (AC20).
        provider_registry.bind_model_handle(None)
        if acquired.release is not None:
            _release_acquisition(acquired.release, run_id=run_id, failing=sys.exc_info()[1])

    return RunResult(
        run_id=run_id,
        steps_completed=outcome.steps_completed,
        last_step=outcome.last_step,
        acquisition=record,
        trainer_states=written_states,
        checkpoints=written_checkpoints,
        stopped_because=outcome.stopped_because,
        stop_detail=outcome.stop_detail,
        all_fired=outcome.all_fired,
        rounds=outcome.rounds,
        boundaries=outcome.boundaries,
    )


def bind_tokenizer(batch_source: Any, tokenizer: Any) -> bool:
    """Give the source's context the acquired tokenizer unless it already has one.

    Returns whether it bound one. A source with no ``context`` (a stub) is left alone.
    """
    context = getattr(batch_source, "context", None)
    if context is None or tokenizer is None or getattr(context, "tokenizer", None) is not None:
        return False
    context.tokenizer = tokenizer
    return True


def _release_acquisition(release: Any, *, run_id: str, failing: BaseException | None) -> None:
    """End an acquisition that holds something (§5.10's attachment), on every exit.

    **On the failure path a release error is logged, never raised**: raising from a
    ``finally`` replaces the run's own error, and that error is the diagnosis. On the
    success path it propagates - a run that finished but could not detach (for instance
    because the set is not ``clean``) must not report a clean finish.
    """
    try:
        release()
    except Exception as exc:
        if failing is None:
            raise
        log.error(
            "the acquisition could not be released after the run failed",
            extra={
                "code": "acquisition_release_failed",
                "run_id": run_id,
                "error": f"{type(exc).__name__}: {exc}",
                "run_error": type(failing).__name__,
            },
        )


@celery_app.task(name=TASK_RUN_STEP_LOOP, queue=QUEUE_TRAIN)  # type: ignore[untyped-decorator]
def run_step_loop(run_id: str, max_steps: int | None = None) -> dict[str, Any]:
    """The production entry point: read the recorded definition, then drive the run.

    **Wired to F6's driver on 2026-10-14.** Until then this task raised
    ``signal_source_unavailable`` by name - the honest refusal for a seam with nothing in
    it, since a task that silently returned "done" would report a run that never happened.
    R-02.4-R-02.8 (F2) and R-02.24-R-02.27 (F6) are both built now, so the seam is filled:
    :func:`training.run_driver.drive_recorded_run` reads the definition R-02.53 requires,
    resolves F2's source from it, drives F6's loop, and transitions the row.

    **A run with no recorded definition is refused** (``run_definition_missing``). That is
    what makes F1's pre-R-02.53 ``POST /api/v1/runs`` unable to produce a run that trains,
    so there is exactly one path to a running run and R-02.53's gate sits on it.
    """
    configure_logging(effective_settings().log_level)
    with run_context(run_id):
        log.info(
            "run_step_loop entered",
            extra={"code": "run_started", "run_id": run_id, "max_steps": max_steps},
        )
        from db.session import session_scope
        from training import run_driver

        try:
            with session_scope() as session:
                return run_driver.drive_recorded_run(
                    run_id,
                    session=session,
                    session_scope=session_scope,
                    max_steps=max_steps,
                    emit=_emitter(),
                )
        except BaseException as exc:
            # The traceback's frames hold the run's locals - the model, the optimizer and its
            # states, the gradients - and Celery keeps the exception. Cleared, so the memory
            # can actually be returned below.
            traceback.clear_frames(exc.__traceback__)
            raise
        finally:
            release_device_memory(run_id)


def release_device_memory(run_id: str | None = None) -> None:
    """Return the run's GPU memory to the card when a run ends - on EVERY exit.

    The worker is a long-lived solo process. Until 2026-10-17 nothing released anything when a
    run finished, so the caching allocator kept the last run's gradients and optimizer states
    (5.3 GB after a 1.2B attached run) and R-02.42's budget guard - correctly - refused the
    next run on the same card (found live, F11 window 2).
    """
    import gc

    gc.collect()
    torch = sys.modules.get("torch")
    if torch is None:
        return
    try:
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()
    except Exception as exc:  # never let cleanup replace the run's own outcome
        log.warning(
            "device memory not released after the run",
            extra={"code": "device_memory_release_failed", "run_id": run_id, "error": str(exc)},
        )


def _emitter() -> Any:
    """The socket emitter, or ``None`` when the transport is not importable.

    ``None`` rather than a raise: the socket is a projection (ADR-007) and a worker with no
    Redis reachable must still complete a run whose record is durable. The absence is
    logged so a screen that stopped updating has a reason recorded somewhere.

    **IT RETURNS F6'S SYNC BRIDGE, NOT ``emit_run_event`` ITSELF** - and this line is the
    whole reason :mod:`training.run_events` exists. ``api.sockets.emit_run_event`` is an
    ``async def``, while every emit site below here calls ``emit(run_id, event, payload)``
    **bare and discards the result**: F1's :class:`training.step_record.StepWriter`, F4's
    score recorder, F5's save and F6's boundary. Handing them the coroutine function builds
    a coroutine per event, awaits none of them, and **delivers nothing at all** - the
    emitter looks wired, the subscriber looks connected, and the screen simply never
    updates. Invisible from both ends.

    This function returned ``emit_run_event`` until 2026-10-16, so every run event in
    production was lost, and the suite was green the entire time because every emit test
    injects its own recorder and none asked what production passes.
    :func:`training.run_events.run_emitter` - the bridge written for exactly this - had **no
    production caller and no test**. Found by ``0xcc/scripts/check_reachability.py``.
    ``backend/tests/unit/test_run_event_emitter_wiring.py`` is the control: point this back
    at ``emit_run_event`` and three tests go red, one of them because Python itself reports
    ``coroutine 'emit_run_event' was never awaited``.
    """
    try:
        # The transport is probed HERE even though the bridge imports it lazily per event.
        # Without the probe, an unimportable `api.sockets` stopped returning ``None`` and
        # started raising inside the first ``emit`` instead - which would fail a run over a
        # projection, the exact trade this function's second paragraph refuses. The import
        # is what the probe is for; `run_emitter` is what is returned.
        import api.sockets  # noqa: F401
        from training.run_events import run_emitter
    except Exception as exc:
        log.warning(
            "no socket emitter available; the run record is still durable",
            extra={"code": "run_emitter_unavailable", "error": type(exc).__name__},
        )
        return None
    return run_emitter()


def queue_of(task_name: str = TASK_RUN_STEP_LOOP) -> str | None:
    """Which queue the task routes to, resolved through Celery's **own** router.

    Through the router rather than by re-implementing the glob match: a second
    implementation of the routing rule is a second answer, and the one a hand-rolled test
    checked would not be the one the broker uses.
    """
    from workers.celery_app import queue_for

    return queue_for(task_name)


__all__ = [
    "TASK_RUN_STEP_LOOP",
    "RunResult",
    "acquire_for_run",
    "drive_run",
    "planned_restore_step",
    "queue_of",
    "require_resume_verified",
    "restore_optimizer_state",
    "resume_start_step",
    "resume_target",
    "run_log_root",
    "run_step_loop",
    "verify_checkpoint_source",
]
