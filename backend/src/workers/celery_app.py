"""The Celery application: four queues, explicit routes, and the solo pool (task 6.1, 6.2).

ADR-006. Four queues, each named for what contends on it rather than for who sends to it:

* ``default`` - CRUD and ``reindex``. Cheap, latency-sensitive.
* ``train`` - the GPU run loop. **Concurrency 1 per GPU** (ADR-008), enforced by the lease.
* ``generate`` - self-generation and test-set generation when out of process. Also
  GPU-bound, and under the node's **time-slicing with no MPS** it contends for the whole
  card, which is why it is a queue of its own and not part of ``train``.
* ``maintenance`` - the janitor and retention sweeps. Must keep running while ``train`` is
  saturated, which is the reason it is separate: a janitor queued behind a ten-hour
  training task cannot reap anything.

Task names are ``miforge.<area>.<task>`` and the routes are **explicit**. A route derived
from a prefix match is one typo away from sending a training task to ``default``, where a
worker with no card picks it up and fails - so the mapping is a table, and the test asserts
every registered task resolves to a queue through it.

--------------------------------------------------------------------------------------
EVERY WORKER RUNS ``--pool=solo -c 1``, AND THE CONSEQUENCES ARE THE DESIGN (task 6.2)
--------------------------------------------------------------------------------------

The pool and the concurrency are set **in this configuration**, not only on the command
line, because a flag is a thing an operator can omit. ADR-006 inherits the choice from
miStudio deliberately, and it has exactly two consequences, both of which shape every
other module in this package:

1. **``revoke(terminate=True)`` does not work.** It signals a *pool child*, and the solo
   pool has none - the task runs in the worker's own process. The call returns
   successfully and nothing happens, which is the worst available failure mode because the
   operator sees a cancellation that was accepted. **Cancellation is therefore cooperative
   and goes through the database** (:mod:`workers.cancellation`); a guard test asserts the
   string ``terminate=True`` appears nowhere in ``backend/src/``.
2. **A busy solo worker never reads the control queue.** There is no second thread to read
   it. So *anything* that needs to reach a running job - a stop, a parameter change, a
   question about progress - goes through the ``mf_runs`` row (ADR-007: the run record is
   the protocol). This is also why the heartbeat is a database write and not a Celery
   event: an event nobody reads is invisible to the janitor.
"""

from __future__ import annotations

import logging
from typing import Any

from celery import Celery
from celery.schedules import crontab
from celery.signals import task_prerun

# `kombu` is in pyproject's MyPy `ignore_missing_imports` override list, so no per-line
# ignore is needed here (MyPy reported the one that stood here as unused).
from kombu import Queue

from core.config import get_settings
from core.logging import bind_run_id, configure_logging
from core.runtime_settings import effective_settings

log = logging.getLogger(__name__)

QUEUE_DEFAULT = "default"
QUEUE_TRAIN = "train"
QUEUE_GENERATE = "generate"
QUEUE_MAINTENANCE = "maintenance"

QUEUES: tuple[str, ...] = (QUEUE_DEFAULT, QUEUE_TRAIN, QUEUE_GENERATE, QUEUE_MAINTENANCE)

#: The janitor's task name. Named here rather than inline so the beat schedule, the route
#: table and the reachability test all refer to one string.
TASK_JANITOR_SWEEP = "miforge.maintenance.janitor_sweep"

#: F6's chain sweeper (R-02.27, FR-006.45). **``maintenance``, and the prefix is the
#: point:** F6's four documents all name this task ``miforge.chain.sweep``, and there is no
#: ``miforge.chain.*`` entry in ``TASK_ROUTES`` - so under that name it would route
#: NOWHERE and land on ``default``, queued behind ``reindex``, which is precisely the defect
#: F1 recorded for ``miforge.training.run_step_loop`` and F8 recorded again for the merge.
#: ``test_run_loop_reachability.py`` carries a tripwire asserting the documents' name does
#: not route; this is the name that does.
#:
#: ``maintenance`` rather than ``default`` because a chain advance only ever has work to do
#: while ``train`` is saturated by the run it is waiting on - the same reason the janitor is
#: there.
TASK_CHAIN_SWEEP = "miforge.maintenance.chain_sweep"

#: The checkpoint lifecycle sweep (OQ-4, settled 2026-10-16). **``miforge.maintenance.``,
#: and the prefix is the whole reachability question again:** ``TASK_ROUTES`` is an explicit
#: table keyed by area prefix, so ``miforge.checkpoints.prune`` - which is what this task
#: wants to be called - matches nothing and lands on ``default``, queued behind ``reindex``
#: on a worker that may not even have the checkpoint volume mounted. That is F1's recorded
#: ``miforge.training.run_step_loop`` defect, F8's merge defect and F6's
#: ``miforge.chain.sweep`` defect, three times over. ``test_checkpoint_lifecycle_wiring.py``
#: carries a tripwire asserting the unrouted spelling does not route.
#:
#: ``maintenance`` for the janitor's reason, and more sharply: a prune walks the whole
#: checkpoint root reading manifests, so it must not run on a queue a ten-hour training task
#: can block - the volume fills *while* that task is running, which is exactly when the
#: sweep has work to do.
TASK_CHECKPOINT_PRUNE = "miforge.maintenance.checkpoint_prune"

#: **Explicit** routes, task name -> queue. Globs are used for the per-area prefixes a
#: feature will add tasks under, and every registered task must match exactly one entry -
#: which the reachability test asserts, so a task added without a route is a red suite
#: rather than a task that quietly runs on a worker with no card.
TASK_ROUTES: dict[str, dict[str, str]] = {
    "miforge.train.*": {"queue": QUEUE_TRAIN},
    "miforge.generate.*": {"queue": QUEUE_GENERATE},
    "miforge.maintenance.*": {"queue": QUEUE_MAINTENANCE},
    "miforge.index.*": {"queue": QUEUE_DEFAULT},
    "miforge.packs.*": {"queue": QUEUE_DEFAULT},
}

#: Modules that define tasks, imported by the worker at startup.
#:
#: **Without this a task is registered nowhere that matters.** The worker runs
#: ``celery -A workers.celery_app:celery_app``, which imports *this* module and nothing
#: else, so a task defined in a module no one imports is a fully-implemented,
#: fully-unit-tested capability that does not exist for any caller - the sibling's sixteen
#: unregistered MCP tools, in this package. Every one of those tests passed by importing
#: the module directly.
#:
#: A feature adds its module here, and ``backend/tests/unit/test_step_task_wiring.py``
#: asserts the task is in the **live** registry after ``import_default_modules()``, so
#: removing a line turns the suite red rather than turning the task off quietly.
TASK_MODULES: tuple[str, ...] = (
    "training.tasks",  # F1: miforge.train.run_step_loop
    # F10: miforge.index.part_import. Added 2026-10-14. It was reachable in the API
    # process only, because `endpoints/parts.py` imports it - and a worker started with
    # `-A workers.celery_app` imports this tuple and nothing else, so an enqueued import
    # was received by a worker that did not have the task. Precisely the shape this tuple's
    # docstring is about, arriving within a day of the docstring being written.
    "services.parts.tasks",
    # F4: miforge.generate.evaluate. Added 2026-10-14, and it closes a SECOND gap beside
    # the task: importing this module is also what binds F3's `ProviderResolver`
    # (`services/scoring/wiring.py` runs as an import side effect of
    # `services.scoring.__init__`). Without this line a `train` or `generate` worker has no
    # resolver bound, so the judge-model selector refuses whatever the operator configured -
    # which is the gap F4 found called in fourteen test files and nowhere in `backend/src/`.
    "services.scoring.tasks",
    # F6: miforge.train.merge_adapter. Added 2026-10-14, and it is F8's owed wiring: F8
    # built the merge, the three outcomes and `mf_merges` and reported that the enqueue was
    # F6's - *"a route that enqueued a task nothing consumed would be the
    # unreachable-capability failure this project's review discipline exists to catch."*
    # Without this line a worker started with `-A workers.celery_app` would not have the
    # task, so an enqueued merge would be received by a worker that does not have it.
    "workers.merge_tasks",
    # F6: miforge.maintenance.chain_sweep. Added 2026-10-14 with F6's task 7.0.
    # **The sweep is the ONLY advance path a chain has** (FR-006.45) - the finishing stage's
    # own task deliberately does not enqueue its successor, because a worker dying between
    # the terminal transition and the enqueue would stall the chain forever. So without this
    # line and the beat entry below, `POST /chains` returns a 202, stage one runs, and the
    # curriculum sits at stage one forever with nothing anywhere saying why.
    "workers.chain_tasks",
    # F2 deliberately declares NO Celery task - its in-process assembly is synchronous
    # inside the step on the `train` queue (F6's AMB-1), and it carries a structural test
    # that no module of its own imports Celery at all, so routing it through `generate`
    # cannot be done silently. There is nothing to add here for it, and that absence is
    # asserted rather than assumed.
)

#: How often the janitor sweeps. Comfortably shorter than ``JANITOR_LAPSE_SECONDS`` so a
#: lapsed run is noticed within roughly one interval of lapsing rather than within one
#: threshold of it - otherwise the worst-case detection time is the sum of the two.
JANITOR_INTERVAL_SECONDS = 60.0

#: How often the chain sweeper looks. Longer than the janitor's interval because the cost of
#: being late is different in kind: a lapsed run holds a card and must be reaped promptly,
#: whereas a chain hop is late by at most one interval and nothing is held up but the
#: curriculum. It is off the run's critical path by construction - the sweep never runs
#: inside a step boundary (FR-006.50's 5 % budget).
#:
#: *Set to 120 rather than the janitor's 60 so the sentence above is TRUE. It was written
#: beside a value of 60.0, which would have made the comment a claim the number denied -
#: this project's commonest defect, in two adjacent lines.*
CHAIN_SWEEP_INTERVAL_SECONDS = 120.0

#: How often the checkpoint lifecycle sweeps. **Hourly**, which is two orders of magnitude
#: slower than the janitor, because the two measure different clocks: a lapsed run holds a
#: GPU and must be reaped within an interval, whereas a volume fills over hours and a save
#: happens every ``save_every`` steps - minutes at best. A sweep per minute would re-read
#: every manifest under the root sixty times for each checkpoint it could possibly act on.
#:
#: It is also *not* a safety interval. Running late costs disk; running often costs reads.
#: Nothing is protected by this number, which is why it is allowed to be coarse.
CHECKPOINT_PRUNE_INTERVAL_SECONDS = 3600.0

#: **The hourly period above, scheduled on the wall clock (minute 0 of every hour, UTC) -
#: NOT as a 3,600 s interval. The difference is whether the sweep ever runs.**
#:
#: Beat keeps each entry's ``last_run_at`` in its schedule file, and the beat container
#: keeps that file on its writable layer (``--schedule=/tmp/celerybeat-schedule``), so every
#: pod restart starts from an empty file. Celery 5.6 ``celery/beat.py``,
#: ``ScheduleEntry.__init__``::
#:
#:     self.last_run_at = last_run_at or self.default_now()
#:
#: and ``default_now`` is ``self.schedule.now() if self.schedule else self.app.now()``. So
#: an INTERVAL entry is counted from beat's start: as a plain 3,600 s interval the sweep
#: first fires one hour after each restart, and a pod redeployed more often than hourly
#: (every push to ``main`` rolls it) never prunes at all - silently, with the volume still
#: filling. A crontab entry is due at the next wall-clock match after ``last_run_at``, so a
#: restart costs at most the one slot the pod was down across, never the whole period.
#: The same defect was found in miStudio's daily prune and miDataworks' daily
#: ``prune-agent-requests`` (2026-10-22).
#:
#: ``test_beat_survives_restarts.py`` drives Celery's real ``PersistentScheduler`` against
#: this app's ``beat_schedule`` with a fresh schedule file at every simulated restart; it is
#: red when this goes back to the plain interval.
CHECKPOINT_PRUNE_SCHEDULE = crontab(minute="0")


def _build() -> Celery:
    settings = get_settings()
    app = Celery("miforge", broker=settings.redis_url, backend=settings.redis_url)

    app.conf.update(
        # ------------------------------------------------------------------- the solo pool
        # SET HERE, not only on the command line. See the module docstring: solo has no
        # pool child to signal, so `revoke(terminate=True)` silently does nothing, and a
        # busy solo worker never reads the control queue. Everything that must reach a
        # running job goes through the mf_runs row instead (ADR-006, ADR-007).
        worker_pool="solo",
        worker_concurrency=1,
        # One task in flight, none held in reserve. Prefetching on a queue whose tasks run
        # for hours means a task sitting in a worker's buffer while another worker is idle.
        worker_prefetch_multiplier=1,
        # ------------------------------------------------------------- never retry a step
        # `task_acks_late=True` would redeliver a task whose worker died mid-run, and
        # R-02.31 forbids exactly that: re-applying a non-idempotent AdamW update to a
        # half-written parameter set compounds the damage rather than repairing it. An
        # interrupted run is recovered by the janitor marking it failed and an operator
        # continuing from the last checkpoint (R-02.22) - never by an automatic retry.
        task_acks_late=False,
        task_reject_on_worker_lost=False,
        task_default_retry_delay=0,
        # ------------------------------------------------------------------------- queues
        task_queues=tuple(Queue(name) for name in QUEUES),
        task_default_queue=QUEUE_DEFAULT,
        task_routes=dict(TASK_ROUTES),
        include=TASK_MODULES,
        # ------------------------------------------------------------------------- hygiene
        task_serializer="json",
        result_serializer="json",
        accept_content=["json"],
        timezone="UTC",
        enable_utc=True,
        # A result backend that never expires fills Redis with the output of every step of
        # every run. Redis is the broker and the progress channel, never a store of record
        # (ADR-003), so results are disposable by design.
        result_expires=3600,
        # ⚠ AN INTERVAL ENTRY RESTARTS ITS COUNT AT EVERY BEAT RESTART (the schedule file is
        # disposable; see CHECKPOINT_PRUNE_SCHEDULE). That is harmless for the janitor's
        # 60 s and the chain sweep's 120 s - a restart delays them by one interval - and
        # fatal for anything longer than the time between redeploys, which never fires.
        # So a long period is a `crontab`. `test_beat_survives_restarts.py` restarts beat
        # every ten minutes and requires every entry here to keep firing.
        beat_schedule={
            "janitor-sweep": {
                "task": TASK_JANITOR_SWEEP,
                "schedule": JANITOR_INTERVAL_SECONDS,
                "options": {"queue": QUEUE_MAINTENANCE},
            },
            # F6, FR-006.45. **Without this entry the sweep never fires and a chain never
            # advances** - there is no second path by design, so the schedule is not a
            # convenience here, it is the capability. A chain would accept a 202, run its
            # first stage, and stop.
            "chain-sweep": {
                "task": TASK_CHAIN_SWEEP,
                "schedule": CHAIN_SWEEP_INTERVAL_SECONDS,
                "options": {"queue": QUEUE_MAINTENANCE},
            },
            # OQ-4. **This entry is the sweep's only production caller**, exactly as the
            # janitor's is: there is no endpoint and no other path, so without it the
            # policy is a fully-tested function that never runs and the volume still fills
            # at step 9,000. It is safe to schedule unconditionally because both gates
            # (`CHECKPOINT_LIFECYCLE_ENABLED`, `CHECKPOINT_LIFECYCLE_DRY_RUN`) default to
            # "plan and log, delete nothing" - so an installation that never configures it
            # gets a readable plan in its logs and keeps every byte.
            "checkpoint-prune": {
                "task": TASK_CHECKPOINT_PRUNE,
                # A crontab, not the interval: see CHECKPOINT_PRUNE_SCHEDULE - an interval
                # restarts its count at every beat restart and may never come due.
                "schedule": CHECKPOINT_PRUNE_SCHEDULE,
                "options": {"queue": QUEUE_MAINTENANCE},
            },
        },
    )
    return app


celery_app = _build()


@task_prerun.connect(weak=False)  # type: ignore[untyped-decorator]
def refresh_runtime_settings(**_signal: Any) -> None:
    """Read the runtime settings overrides before every task  (F12, FR-012.8; FTID I-4).

    **The workers' only path to an override.** A run, a merge, a chain sweep and the pruning
    sweep each start with ``mf_settings`` as stored, in this worker's own process, so a
    tolerance or a pruning setting changed through ``PUT /api/v1/settings/{key}`` reaches
    the code that uses it at the next task. A failure to read is logged and the environment
    values apply. ``test_settings_api.py``'s junction test fails when this connection is
    removed.
    """
    from core.runtime_settings import refresh_from_database

    refresh_from_database()


def queue_for(task_name: str) -> str | None:
    """Which queue ``task_name`` routes to under :data:`TASK_ROUTES`, or ``None``.

    Resolved through Celery's own router rather than by re-implementing the glob match,
    because a second implementation of the routing rule is a second answer - and the one
    the test would then be checking is not the one the broker uses.
    """
    route: Any = celery_app.conf.task_routes
    if not route:
        return None
    from celery.app.routes import MapRoute

    resolved = MapRoute(route)(task_name)
    if not resolved:
        return None
    queue = resolved.get("queue")
    return None if queue is None else str(queue)


# Celery's task decorator is untyped; the ignore is on the decorator, not on the
# function, so the function's own signature is still checked.
@celery_app.task(name=TASK_JANITOR_SWEEP, queue=QUEUE_MAINTENANCE)  # type: ignore[untyped-decorator]
def janitor_sweep() -> dict[str, Any]:
    """Reap runs whose heartbeat has lapsed  (task 6.5).

    **This registration is the janitor's only production caller.** ADR-016's first
    structural rule - *a capability is not shipped until a test fails when its wiring is
    removed* - applies to it literally: a ``janitor.sweep`` with nothing on a schedule
    calling it is a fully-tested function that never runs, which is the shape of the
    sibling's sixteen unregistered MCP tools. ``backend/tests/unit/test_janitor_wiring.py``
    asserts this task is in the **live** registry, routes to ``maintenance``, and is on the
    beat schedule.
    """
    from workers.janitor import sweep
    from workers.run_record import default_session_scope

    configure_logging(effective_settings().log_level)
    bind_run_id(None)  # a sweep belongs to no single run
    with default_session_scope() as session:
        report = sweep(session)
    return {
        "reaped": report.reaped,
        "alive": len(report.alive),
        "running_without_heartbeat": report.running_without_heartbeat,
        "lapse_threshold_seconds": report.lapse_threshold_seconds,
    }


@celery_app.task(name=TASK_CHECKPOINT_PRUNE, queue=QUEUE_MAINTENANCE)  # type: ignore[untyped-decorator]
def checkpoint_prune() -> dict[str, Any]:
    """Sweep the checkpoint root against the lifecycle policy  (OQ-4).

    **Registered here for the janitor's reason**, which ADR-016 #1 states as a rule: *a
    capability is not shipped until a test fails when its wiring is removed.* A
    ``lifecycle.plan`` with nothing on a schedule calling it is the sibling's sixteen
    unregistered MCP tools - unit-tested, documented, and absent for every caller.
    ``backend/tests/unit/test_checkpoint_lifecycle_wiring.py`` asserts this task is in the
    **live** registry, routes to ``maintenance``, is on the beat schedule with that queue
    in its payload, and that the plausible-but-unrouted name does not route.

    Deletes nothing unless both settings say so; see
    :mod:`workers.checkpoint_lifecycle`.
    """
    from workers.checkpoint_lifecycle import sweep
    from workers.run_record import default_session_scope

    configure_logging(effective_settings().log_level)
    bind_run_id(None)  # a sweep belongs to no single run
    with default_session_scope() as session:
        report = sweep(session)
    return report.as_dict()
