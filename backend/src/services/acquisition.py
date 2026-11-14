"""Submit: validate, resolve the device, pre-flight, lease, enqueue  (FR-001.16, FR-001.7).

**The order is the requirement**, and every element of it comes from a decision already
taken:

1. **Validate** at the edge (Pydantic v2) - a type or range error never reaches a worker.
2. **Resolve the device** by UUID from live ``nvidia-smi -L`` (ADR-008). Resolved **once**
   per submit and passed down, so nothing below names a device of its own.
3. **Pre-flight** - mode availability, the source, the dtype gate, the artifact roots'
   atomicity, the budget guard, then **F5's checkpoint cases**.
   R-02.48: *"a run that cannot proceed says so before it starts, not partway through."*
4. **The budget guard, before the lease and before any allocation** (ADR-008). Acquisition
   is the first allocation, so this is the last moment the check means anything.
5. **Take the lease** - one run per card, by a lease with an expiry (ADR-008). A boolean
   flag is the anti-pattern that ADR names.
6. **Enqueue** on the ``train`` queue and return ``202``. The request path **never loads a
   model**: that would allocate outside the lease and block the API for a multi-gigabyte
   read.

## What this module does *not* own

R-02.53's **run definition** - the pack, at least one stop condition, the cadences, the
retention floor - is **F6's**, and F6 validates the definition as a whole. This is the
*acquisition half*: the mode, the source, the dtype, the device and the budget. F1 adds no
second validator for the definition's completeness.

## F5's checkpoint cases live here, and that is task 7.8

F5's resume cases - resolve, validate, **verify the hash**, decide on trainer state - all
exist, as :func:`services.checkpoints.resume.resume_plan`. ``POST
/runs/{run_id}/continue`` calls it, so the API's *planning* route refuses correctly. But
that route **only returns a plan**: the path that actually creates a run row and enqueues
``miforge.train.run_step_loop`` is :func:`services.run_launch.launch` through
:mod:`services.run_definition` step 5, which lands in :func:`preflight` here - and nothing
on it verified a checkpoint. A run defined with ``model_source = {"mode": "load", "path":
<a corrupted checkpoint>}`` was accepted and the worker re-ran only the *acquirer's*
preflight, which checks that the weight files exist and never that they hash to what the
manifest records. **It would have trained, silently, on corrupted weights.**

:func:`preflight_checkpoint_source` folds those cases in **by calling F5's own functions**,
not by restating them, and :func:`preflight_artifact_roots` adds the atomicity probe the
worker never had because it never calls ``create_app``.
"""

from __future__ import annotations

import logging
import threading
from pathlib import Path
from typing import Any

from core.config import get_settings
from core.errors import Refusal, refuse
from core.gpu import Device, resolve_device
from training import acquisition
from training.acquisition.base import AcquisitionMode, AcquisitionSpec, Availability
from training.budget import BudgetGuard, PassThroughGuard
from training.budget_guard import ComputedBudgetGuard
from training.run_config import RunConfig

log = logging.getLogger(__name__)

#: Roots this process has already proved publishable, and the lock that serialises the
#: proof. See :func:`preflight_artifact_roots` for why a per-submit probe needs both.
_ATOMIC_ROOTS: set[Path] = set()
_ATOMICITY_LOCK = threading.Lock()

#: The refusals F5's resume planning raises, folded in below (F5 task 7.8). Named as a
#: constant so a test can assert the set rather than re-spelling five strings, and so a
#: refusal added to :func:`services.checkpoints.resume.resume_plan` without being considered
#: here shows up as a disagreement rather than as silence.
CHECKPOINT_PREFLIGHT_REFUSALS = (
    "checkpoint_not_found",
    "checkpoint_manifest_invalid",
    "checkpoint_hash_mismatch",
    "trainer_state_incomplete",
    "run_not_continuable",
)


def guard_for(device: Device | None) -> BudgetGuard:
    """The budget guard for this submit.

    **R-02.42's estimator, pulled forward into F1 by operator decision (2026-10-14).** F1's
    OQ-1 put the ordering question - F1 allocates in wave 2, R-02.42 is F8's in wave 4 - and
    the answer was to move the estimator rather than ship an unguarded allocation for two
    waves. The seam is untouched: :class:`training.budget_guard.ComputedBudgetGuard`
    implements :class:`training.budget.BudgetGuard` exactly as declared, and **this function
    is the binding** F8's documents describe as its only change here.

    :class:`training.budget.PassThroughGuard` is returned when no device could be resolved -
    on a machine with no card, where no allocation can happen either - and it **logs loudly**
    that no estimate was made. It is kept rather than deleted because F8's ordering test uses
    it as a spy.
    """
    return PassThroughGuard() if device is None else ComputedBudgetGuard(device)


def is_miforge_checkpoint(path: Path | None) -> bool:
    """Whether ``path`` is a directory miForge itself wrote as a checkpoint.

    The discriminator is the presence of ``manifest.json``, and it is deliberately the only
    one. A ``load`` may name **either** a miForge checkpoint or a plain Hugging Face
    directory (R-02.46 case 1 covers both), and only the first has a recorded
    ``weights_sha256`` to verify against. Treating every directory as a checkpoint would
    refuse ``checkpoint_not_found`` for a perfectly good third-party model; treating none as
    one is the gap this function exists to close.

    ``manifest.json`` rather than a guess over several plausible names: *a getattr default
    that always fires is a fabricated value* (ADR-018), and the filename comes from the one
    module that declares it.
    """
    from services.checkpoints import manifest as manifest_module

    if path is None:
        return False
    return (Path(path) / manifest_module.MANIFEST_NAME).is_file()


def preflight_artifact_roots(
    *,
    checkpoint_root: Path | None = None,
    trainer_state_root: Path | None = None,
) -> dict[str, str]:
    """The filesystem-atomicity case, for the roots this run will publish into (FR-005.5).

    **Why it is here and not only at startup.** ``create_app`` already probes the checkpoint
    root and *logs* - deliberately, because an unwritable mount must not take down
    ``/api/health`` and every screen that touches no artifact. But a **Celery worker never
    calls** ``create_app``, so on the task path nothing probed anything, and the first
    discovery was ``atomic_dir`` refusing at a save point hours into the run. Refusing at
    submit is what F1's own rule asks for: *a run that cannot proceed says so before it
    starts, not partway through* (R-02.48).

    **A root that does not exist is reported, never silently passed.** The probe in
    :func:`core.atomic_dir.verify_root_is_atomic` *creates* the root, which is right for a
    startup check and wrong in a request handler - on a default configuration it would
    mkdir ``/data/checkpoints`` from an API process and refuse every submit with
    ``checkpoint_root_unwritable``, which is precisely the 38-collection-error failure
    ``verify_root_is_atomic``'s own comment records. So an absent root is returned as
    ``not_checked_root_absent`` and logged at warning: *a skip that names its reason is the
    honest form*, and the alternative is a check that reports a verdict it never reached.

    **Probed once per process per root, under a lock, and that is a correctness fix rather
    than an optimisation.** :func:`core.atomic_dir.verify_root_is_atomic` names its probe
    directory ``.atomicity-probe-<pid>``, which is unique per *process* because it was
    written as a startup check. Called per submit it is no longer unique per *call*: two
    concurrent submits in one uvicorn worker would race on ``probe.mkdir()``, the loser
    would get ``FileExistsError`` - an ``OSError``, so the primitive reports
    ``checkpoint_root_unwritable`` - and a perfectly healthy run would be refused for a
    collision with itself. *Found by asking what a second simultaneous submit would do, not
    by a test; a false refusal under load is exactly the kind of defect a single-threaded
    suite cannot see.* A filesystem's atomicity does not change between two requests, so
    memoising the success is also the honest answer and not merely the cheap one.

    Only **successes** are cached. A refusal is re-raised on every submit, because an
    operator who fixes the mount must not have to restart the process to be believed, and an
    absent root may appear later.

    Returns:
        One entry per root: ``"atomic"`` or ``"not_checked_root_absent"``.

    Raises:
        Refusal: ``checkpoint_root_not_atomic`` or ``checkpoint_root_unwritable``, naming
            the root, from the primitive. Nothing is allocated and no lease is taken.
    """
    from core import atomic_dir

    settings = get_settings()
    roots = {
        "checkpoint_root": (
            checkpoint_root if checkpoint_root is not None else settings.checkpoint_root
        ),
        # The trainer-state root is checked too, because R-02.15's write is atomic as well
        # and a run whose optimizer state cannot be published is a run that cannot be
        # continued - which looks complete and is not.
        "trainer_state_root": (
            trainer_state_root if trainer_state_root is not None else settings.trainer_state_root
        ),
    }

    report: dict[str, str] = {}
    for label, root in roots.items():
        if root in _ATOMIC_ROOTS:
            report[label] = "atomic"
            continue
        if not root.is_dir():
            log.warning(
                "an artifact root does not exist, so its atomicity was not checked",
                extra={
                    "code": "artifact_root_atomicity_not_checked",
                    "root_name": label,
                    "root": str(root),
                    "consequence": (
                        "The run is not refused for this. The root is created by the first "
                        "writer and `atomic_dir` still refuses by name there, so the "
                        "guarantee holds; what is lost is the early discovery."
                    ),
                },
            )
            report[label] = "not_checked_root_absent"
            continue
        with _ATOMICITY_LOCK:
            # Re-checked inside the lock: two threads can both miss the cache above, and
            # without this the second would still race the first on the probe directory.
            if root not in _ATOMIC_ROOTS:
                atomic_dir.verify_root_is_atomic(root)
                _ATOMIC_ROOTS.add(root)
        report[label] = "atomic"
    return report


def reset_artifact_root_atomicity_cache() -> None:
    """Forget which roots have been probed. **For tests**, which repoint the roots.

    Exported rather than reaching into the module global from a test file: a test that
    mutates a private set is a second definition of the cache's shape, and this project has
    paid four times for a second spelling of one thing.
    """
    with _ATOMICITY_LOCK:
        _ATOMIC_ROOTS.clear()


def preflight_checkpoint_source(
    spec: AcquisitionSpec,
    *,
    run_id: str | None = None,
    configured_config_hash: str | None = None,
    root: Path | None = None,
    trainer_state_root: Path | None = None,
) -> dict[str, Any] | None:
    """F5's resume cases, on the submit path  (**task 7.8**; R-02.21, R-02.22).

    Returns ``None`` when the source is not a miForge checkpoint, so a ``create`` run and a
    plain Hugging Face ``load`` are untouched.

    **The gap this closes.** F5's cases all exist and all run - as
    :func:`services.checkpoints.resume.resume_plan`, which
    ``POST /runs/{run_id}/continue`` calls. But that route only *returns a plan*: the thing
    that actually creates a run and enqueues ``miforge.train.run_step_loop`` is
    :func:`services.run_launch.launch`, through
    :func:`services.run_definition` step 5, which calls :func:`preflight` - and nothing on
    that path verified the checkpoint. So a run defined with
    ``model_source = {"mode": "load", "path": <a corrupted checkpoint>}`` was accepted, and
    the worker re-ran only the *acquirer's* preflight, which checks that the weight files
    exist and never that they hash to what the manifest records. **A corrupted checkpoint
    would have loaded and trained, silently** - the wrong-answer failure ADR-013 forbids,
    and the one R-02.21 was amended to prevent.

    **One implementation, not a second one.** The three cases that are addressable by
    ``(run_id, step)`` go through F5's own ``continue_plan`` / ``branch_plan``; the two that
    are addressable by path alone call the same two primitives F5 calls
    (:func:`services.checkpoints.manifest.read`,
    :func:`services.checkpoints.writer.verify_weights`) rather than restating what they do.
    A translation layer between two spellings of one check is a second source of truth, and
    this project has paid for that four times.

    Args:
        spec: the enriched spec, **after** the acquirer's preflight has confined the path.
        run_id: the run that will own the resumed work, when the caller knows it.
            :func:`submit` knows it; the define path does not, because
            :func:`services.run_launch.launch` mints the id after validation.
        configured_config_hash: the resumed configuration's optimizer hash. A difference is
            restored anyway and logged with both values - R-02.22 says restore (FR-005.34).

    Raises:
        Refusal: one of :data:`CHECKPOINT_PREFLIGHT_REFUSALS`, each **before step 1** and
            each having changed nothing - no tensor read into a model, no lease, no run row.
    """
    from services.checkpoints import manifest as manifest_module
    from services.checkpoints import resume as resume_module
    from services.checkpoints import writer as writer_module
    from services.checkpoints.paths import checkpoint_dir

    if spec.resume_from is not None:
        # AN ATTACHED RUN RESUMING FROM A CHECKPOINT IT SAVED (D52, F11 FTASKS 18.0). The
        # same F5 cases as a load continue - resolve, verify the hash, decide trainer state -
        # through the same functions. Dispatched on the presence of `resume_from`, never on
        # the mode: the spec's own validator already admits it on the attach mode alone.
        return _preflight_resume_from(
            spec,
            run_id=run_id,
            configured_config_hash=configured_config_hash,
            root=root,
            trainer_state_root=trainer_state_root,
        )

    if spec.mode is not AcquisitionMode.LOAD or not is_miforge_checkpoint(spec.path):
        return None

    directory = Path(spec.path)  # type: ignore[arg-type]  # guarded above
    # RESOLVE AND VALIDATE. A manifest read off disk is untrusted input until validated,
    # exactly like a request body: the directory may have been copied in from another
    # installation or restored from a backup.
    doc = manifest_module.read(directory)
    source_run = str(doc["run_id"])
    source_step = int(doc["step"])

    checkpoint_root = root if root is not None else get_settings().checkpoint_root
    addressed_by_run_and_step = False
    try:
        addressed_by_run_and_step = (
            checkpoint_dir(source_run, source_step, root=checkpoint_root).resolve()
            == directory.resolve()
        )
    except (Refusal, OSError):
        # A manifest whose own run_id will not render a path - a copied-in checkpoint from
        # an installation with a different naming rule. Not a refusal: the path-addressed
        # branch below still verifies the hash, which is the case that matters.
        addressed_by_run_and_step = False

    report: dict[str, Any] = {
        "source_run_id": source_run,
        "step": source_step,
        "directory": str(directory),
        "addressed_by": "run_and_step" if addressed_by_run_and_step else "path_only",
    }

    if addressed_by_run_and_step and run_id is not None:
        # THE FULL FOLD. `continue_plan` for the same run, `branch_plan` for a new one - and
        # the distinction is load-bearing rather than tidy: trainer state is addressed by
        # run_id (R-02.15), so a branch is `fresh` BY CONSTRUCTION (FR-005.38) and refusing
        # it over the *source* run's torn directory would block legitimate work for a state
        # it is never going to read.
        if run_id == source_run:
            plan = resume_module.continue_plan(
                source_run,
                step=source_step,
                configured_config_hash=configured_config_hash,
                root=checkpoint_root,
                trainer_state_root=trainer_state_root,
            )
        else:
            plan = resume_module.branch_plan(
                source_run,
                source_step,
                new_run_id=run_id,
                configured_config_hash=configured_config_hash,
                root=checkpoint_root,
                trainer_state_root=trainer_state_root,
            )
        report.update(
            {
                "verified_weights_sha256": plan.verified_weights_sha256,
                "optimizer_state": plan.optimizer_state,
                "reason": plan.reason,
                "is_branch": plan.is_branch,
                "trainer_state_checked": True,
            }
        )
    else:
        # VERIFY, which is the case that must never be skipped: R-02.21 requires the hash
        # check "before using a single tensor", and this is the last point on the submit
        # path where nothing has been read.
        report["verified_weights_sha256"] = writer_module.verify_weights(
            directory, str(doc["weights_sha256"])
        )
        # THE TRAINER-STATE DECISION IS REPORTED AS UNCHECKED, not passed over. Without a
        # run id there is no answer to *whose* optimizer state this is: the same directory
        # is a continue for one caller and a branch point for another, and the two want
        # opposite verdicts on a torn directory. Naming the gap is the same choice
        # `test_schema_sync.py` makes by classing a contract `Internal` BY NAME rather than
        # giving it an empty vendor list, which would make its assertion pass vacuously.
        report["trainer_state_checked"] = False
        report["trainer_state_not_checked_because"] = (
            "no run_id was supplied, so whether this is a continue of "
            f"{source_run!r} or a branch point for a new run is unknown, and the two "
            "require opposite verdicts on a torn directory"
            if addressed_by_run_and_step
            else (
                "the directory is not addressed as "
                f"{checkpoint_root}/<run_id>/step-NNNNNN, so this installation holds no "
                "trainer state for it"
            )
        )

    log.info(
        "checkpoint source pre-flighted",
        extra={"code": "checkpoint_source_preflighted", "run_id": run_id, **report},
    )
    return report


def _preflight_resume_from(
    spec: AcquisitionSpec,
    *,
    run_id: str | None,
    configured_config_hash: str | None,
    root: Path | None,
    trainer_state_root: Path | None,
) -> dict[str, Any]:
    """F5's checkpoint cases for ``spec.resume_from``. The report, as for a load.

    With ``run_id`` the full fold: the same run is a **continue** (trainer state decided,
    and refused unless the step is the run's latest artifact - its saves would collide
    otherwise); another run is a **branch** (``fresh`` by construction, FR-005.38). Without
    one, verify only, with the trainer-state decision reported as unchecked - the same
    shape and the same reason as the load branch below.
    """
    from services.checkpoints import resume as resume_module
    from services.checkpoints import writer as writer_module

    target = resume_module.resolve_resume_from(spec.resume_from, root=root)
    report: dict[str, Any] = {
        "source_run_id": target.run_id,
        "step": target.step,
        "directory": str(target.directory),
        "addressed_by": "resume_from",
        # What the served model must match, carried so the worker's identity check and the
        # run log name the same checkpoint the plan verified.
        "resume_from": {
            "run_id": target.run_id,
            "step": target.step,
            "name": target.name,
            "manifest_file_sha256": target.manifest_file_sha256,
        },
    }
    holding = target.directory.parent.parent
    if run_id is not None:
        if run_id == target.run_id:
            resume_module.refuse_unless_latest(
                target.run_id,
                target.step,
                root=holding,
                trainer_state_root=trainer_state_root,
            )
            plan = resume_module.continue_plan(
                target.run_id,
                step=target.step,
                configured_config_hash=configured_config_hash,
                root=holding,
                trainer_state_root=trainer_state_root,
            )
        else:
            plan = resume_module.branch_plan(
                target.run_id,
                target.step,
                new_run_id=run_id,
                configured_config_hash=configured_config_hash,
                root=holding,
                trainer_state_root=trainer_state_root,
            )
        report.update(
            {
                "verified_weights_sha256": plan.verified_weights_sha256,
                "optimizer_state": plan.optimizer_state,
                "reason": plan.reason,
                "is_branch": plan.is_branch,
                "trainer_state_checked": True,
            }
        )
    else:
        report["verified_weights_sha256"] = writer_module.verify_weights(
            target.directory, target.weights_sha256
        )
        report["trainer_state_checked"] = False
        report["trainer_state_not_checked_because"] = (
            "no run_id was supplied, so whether this resume continues "
            f"{target.run_id!r} or branches from it into a new run is unknown"
        )
    log.info(
        "resume source pre-flighted",
        extra={"code": "resume_source_preflighted", "run_id": run_id, **report},
    )
    return report


def preflight(
    spec: AcquisitionSpec,
    cfg: RunConfig,
    *,
    guard: BudgetGuard | None = None,
    device: Device | None = None,
    run_id: str | None = None,
) -> tuple[AcquisitionSpec, Availability]:
    """Everything that can refuse, before the lease and before any allocation.

    Returns the spec as preflight enriched it - a confined path, a resolved commit, a
    parameter count - and the mode's availability, which the caller records.

    Args:
        run_id: the run that will own this work, when the caller knows it. Used only by
            F5's checkpoint cases, where it decides continue-versus-branch. Optional
            because :func:`services.run_launch.launch` mints the id after validation, and
            its absence narrows one check and is **reported** rather than hidden.

    Raises:
        Refusal: every refusal of E-1 through E-6 from the acquirer - including
            ``acquisition_mode_unavailable`` for a mode this build does not have, which the
            registry's own entry raises so there is no ``if mode == ...`` here; then
            ``acquisition_mode_unavailable`` for a mode whose ML runtime is absent; then
            ``budget_exceeded`` from the guard; then **F5's checkpoint cases**
            (:data:`CHECKPOINT_PREFLIGHT_REFUSALS`). **Nothing is allocated and no lease is
            taken on any of those paths.**
    """
    availability = acquisition.availability(spec.mode)
    acquirer = acquisition.resolve(spec)

    # THE ACQUIRER'S OWN PRE-FLIGHT FIRST, and the ordering is deliberate rather than
    # incidental. An unavailable *mode* refuses here by itself - `_UnavailableMode.preflight`
    # raises `acquisition_mode_unavailable` naming the mode and the capabilities, which is
    # E-6 - so no `if mode == ...` is needed and R-02.48's "never substituted for a different
    # behaviour" holds through the registry.
    #
    # For a mode that exists but whose ML runtime is absent, running pre-flight first is what
    # makes the refusal useful: ADR-023 puts the path confinement, the missing-file check and
    # the dtype gate on the no-torch side precisely so they can run here, and "this model is
    # quantized" is a more actionable answer than "this build cannot load models" when both
    # are true. The runtime refusal follows immediately below, so neither is lost.
    enriched = acquirer.preflight(spec)

    # THE FILESYSTEM-ATOMICITY CASE, HERE, and early because it is the cheapest of the
    # four and applies to every mode. A run whose artifact roots cannot publish atomically
    # cannot checkpoint at all, so there is nothing for the budget guard to usefully say
    # about it. On the task path this is the ONLY probe there is: a Celery worker never
    # calls `create_app`, which is where the startup check lives (F5 task 7.8).
    preflight_artifact_roots()

    # THE BUDGET CHECK, HERE. After the source is known (so there is a parameter count or a
    # config to estimate from) and before the lease and before `acquire()`. A test asserts
    # this ordering with the payload and the call count, because "the guard was called" is
    # satisfied by a call made after the allocation it was meant to precede.
    (guard if guard is not None else guard_for(device)).check(enriched, cfg)

    # THE RUNTIME GATE LAST, because the refusals are ordered **most specific first** and
    # this is the least specific of the three. "This model is quantized" tells the operator
    # which file to replace; "this run projects 31 GiB on a 24 GiB card" tells them which
    # number to lower; "this build cannot load models" tells them to install something. All
    # three can be true at once, and reporting the installation problem first would send
    # someone to fix their environment for a run that was never going to fit anyway.
    #
    # It also keeps the budget guard's call site exercisable in CI, where there is no ML
    # runtime - and that ordering assertion is the one this feature most needs a test for.
    if not availability.available:
        raise refuse(
            "acquisition_mode_unavailable",
            availability.message
            or f"The {spec.mode.value!r} acquisition mode cannot run on this deployment.",
            mode=spec.mode.value,
            reason=availability.reason,
            capabilities_unavailable=list(availability.unavailable_capabilities),
        )

    # F5'S CHECKPOINT CASES, LAST, and the position is a cost decision that is worth
    # stating rather than leaving to be rediscovered. Every refusal above is cheap - a
    # config read, a safetensors header, an arithmetic estimate - while this one computes
    # SHA-256 over every weight byte, which is seconds for a 2.4 GB checkpoint. All four
    # refuse having changed nothing, so the order cannot affect correctness; it affects
    # only which cheap answers an operator gets without paying for the expensive one.
    #
    # It is **after** the availability gate on purpose too: a build that cannot load models
    # at all will not be fixed by hashing a checkpoint for it.
    preflight_checkpoint_source(
        enriched,
        run_id=run_id,
        configured_config_hash=cfg.config_hash(),
    )
    return enriched, availability


def resolve_submit_device(gpu_request: str) -> Device | None:
    """The card this run is pinned to, or ``None`` when there is no card at all.

    ``None`` rather than a refusal, so that every non-training capability of miForge works
    on a machine with no GPU (ADR-023). The refusal belongs at the point a training step
    actually needs a device, where the message can say so; refusing at submit would make a
    developer unable to exercise the submit path at all.

    An index or a bus address is refused inside ``core.gpu`` **by name**, so this feature
    adds no device-resolution logic and no entry to the hard-coded-GPU ratchet's allowlist.
    """
    try:
        return resolve_device(gpu_request)
    except Refusal as exc:
        if exc.code in {
            "nvidia_smi_unavailable",
            "no_cuda_device_visible",
            "nvidia_smi_failed",
            "nvidia_smi_timeout",
        }:
            log.warning(
                "no GPU could be resolved; the submit path continues and a training step "
                "will refuse by name",
                extra={"code": exc.code, "gpu_request": gpu_request},
            )
            return None
        # A malformed request, an index, a bus address or a UUID that is not present: the
        # operator asked for a specific card and did not get it. Refused, never silently
        # replaced with another - a run pinned to a card it did not get is unreproducible.
        raise


def submit(
    *,
    session: Any,
    run_id: str,
    spec: AcquisitionSpec,
    cfg: RunConfig,
    guard: BudgetGuard | None = None,
    enqueue: Any | None = None,
) -> dict[str, Any]:
    """Run the whole submit sequence. Returns the accepted run's acquisition summary.

    Args:
        session: the request-scoped session. The caller commits.
        run_id: the identifier every manifest, log line and socket room will carry.
        spec: what to acquire.
        cfg: R-02.13's run configuration.
        guard: override the budget guard. The ordering test injects a spy.
        enqueue: override the enqueue call. ``None`` sends the real Celery task.
    """
    device = resolve_submit_device(spec.gpu_request)
    # `run_id` is passed, which is what gives this path F5's trainer-state case as well as
    # its verification: only the caller knows whether the resumed work belongs to the run
    # the checkpoint came from (a continue) or to a new one (a branch).
    enriched, availability = preflight(spec, cfg, guard=guard, device=device, run_id=run_id)

    lease_uuid: str | None = None
    if device is not None:
        from workers import lease as lease_module

        # The device is passed in rather than re-resolved: resolving twice invites the two
        # answers to differ, and the second answer would be the one the work actually ran on
        # while the first is the one recorded.
        held = lease_module.acquire(session, run_id, spec.gpu_request, device=device)
        lease_uuid = held.gpu_uuid

    payload = acquisition_summary(run_id, enriched, cfg, device, availability)

    if enqueue is None:
        from training.tasks import run_step_loop

        enqueue = run_step_loop.delay
    enqueue(run_id)

    log.info(
        "run accepted",
        extra={
            "code": "run_accepted",
            "run_id": run_id,
            "acquisition_mode": enriched.mode.value,
            "gpu_uuid": lease_uuid,
            "optimizer": cfg.optimizer_name(),
        },
    )
    return payload


def acquisition_summary(
    run_id: str,
    spec: AcquisitionSpec,
    cfg: RunConfig,
    device: Device | None,
    availability: Availability,
) -> dict[str, Any]:
    """What ``POST /api/v1/runs`` returns, and what F9's resource strip renders.

    The mockup's resource strip was built for an attachment that no longer exists; its
    handoff says the honest content on the default path is *"what model miForge holds and
    where it came from"*. That is this payload - the mode, the source and the device - and
    the starting hash joins it once acquisition has happened in the worker.
    """
    return {
        "run_id": run_id,
        "acquisition": {
            "mode": spec.mode.value,
            "source": _source_summary(spec),
            # Recorded even when available, so the success and failure shapes are one type.
            "mode_available": availability.available,
            "mode_unavailable_reason": availability.reason,
            "seed": spec.seed,
        },
        "optimizer": {
            "name": cfg.optimizer_name(),
            "state_precision": cfg.optimizer_state_precision,
            "config_hash": cfg.config_hash(),
        },
        "run_config": cfg.model_dump(mode="json"),
        "gpu": {
            # `gpu_request` beside `gpu_uuid`: "auto" and "this exact card" are different
            # requests that can resolve to the same card, and only keeping both can say
            # "you asked for auto and got this one".
            "request": spec.gpu_request,
            "uuid": None if device is None else device.uuid,
            "name": None if device is None else device.name,
        },
    }


def _source_summary(spec: AcquisitionSpec) -> dict[str, Any]:
    """The source, as the run record and the UI name it. **No secret, ever** (ADR-020)."""
    return {
        "path": None if spec.path is None else str(spec.path),
        "repo": spec.repo,
        "revision_requested": spec.revision,
        "revision_resolved": spec.revision_resolved,
        "architecture": _architecture_of(spec),
    }


def _architecture_of(spec: AcquisitionSpec) -> str | None:
    config = spec.architecture_config
    if config is None:
        return None
    value = config.get("model_type")
    return value if isinstance(value, str) else None


def model_roots() -> list[Path]:
    """The directories a ``local_dir`` load may read from. Reported by the API.

    Published so a refused path is actionable: ``model_source_outside_root`` names the roots
    in its details, and the endpoint can show them before the operator types a path.
    """
    settings = get_settings()
    return [settings.checkpoint_root, settings.parts_root]


def assert_mode_known(value: str) -> AcquisitionMode:
    """Parse an acquisition mode, refusing by name rather than raising ``ValueError``."""
    try:
        return AcquisitionMode(value)
    except ValueError as exc:
        raise refuse(
            "acquisition_mode_unknown",
            f"{value!r} is not an acquisition mode. miForge acquires parameters by 'load', "
            f"'create' or 'attach'.",
            mode=value,
            known=[m.value for m in AcquisitionMode],
        ) from exc
