"""``miforge.generate.evaluate`` - a **thin wrapper**, not a second implementation  (FR-004.14a).

## Which queue, and why it is not simply "generation goes on generate"

FR-004.14a settled a conflict three feature documents each stated consistently and
contradicted each other on: **where a scoring pass runs is decided by whose VRAM it generates
from, not by whether it is generation.**

* The **in-cadence pass F6's loop awaits runs in the ``train`` worker, synchronously**, by a
  direct call to :func:`services.scoring.service.evaluate`. ADR-009's in-process provider
  generates *"from the parameters miForge holds"*, and those exist **only** in that worker's
  process. A ``generate`` worker would need a second copy of the parameter set on the leased
  card - against ADR-008's one-run-per-card lease and BRD-02 §6's budget, which has no room
  for it - and a cross-queue ``.get()`` from a ``--pool=solo -c 1`` task blocks the only
  worker thread on a queue it cannot drain.
* This task carries exactly two things: a pass whose provider is an **HTTP** one, where no
  VRAM of miForge's is involved at all, and work the loop **does not await** - the
  out-of-cadence route.

So this module is a wrapper and nothing else. ``backend/tests/unit/test_scoring_task.py``
asserts its body contains no scoring logic of its own, and that all three callers reach
**one** implementation by **call count** rather than by "something was called".

## Two deviations from this feature's documents, recorded rather than silent

**The task is named ``miforge.generate.evaluate``, not ``miforge.scoring.evaluate``.**
``workers.celery_app.TASK_ROUTES`` has no ``miforge.scoring.*`` entry, and
``test_janitor_wiring.test_every_registered_miforge_task_resolves_to_a_queue`` fails for any
registered task that resolves to no route - *"a task with no route silently lands on default,
where a worker with no card picks up a training task and fails."* ``miforge.generate.*``
routes to the ``generate`` queue, which is the queue this task needs. Adding a route is a
one-line edit to a file this feature does not own; the name is a constant here so the rename
is one line on this side too.

**The module is here, not under ``workers/tasks/``.** ``backend/src/workers/`` is the
foundation's package and is single-owner.

**And the gap that is the foundation's to close**, reported rather than worked around: a
worker started with ``-A workers.celery_app`` imports ``TASK_MODULES`` and nothing else, so
``"services.scoring.tasks"`` must be added to that tuple or an enqueued evaluation is
received by a worker that does not have the task. That is the same defect F10 hit within a
day of the tuple's docstring being written, and it is reported for the same reason.
"""

from __future__ import annotations

import logging
from typing import Any

from workers.celery_app import QUEUE_GENERATE, celery_app

log = logging.getLogger("miforge.scoring")

#: The task's name. A constant so the route table, the enqueue call and the reachability test
#: refer to one string rather than three spellings of it. See the module docstring on why it
#: is under ``miforge.generate.*``.
TASK_EVALUATE = "miforge.generate.evaluate"


@celery_app.task(name=TASK_EVALUATE, queue=QUEUE_GENERATE)  # type: ignore[untyped-decorator]
def evaluate_task(run_id: str, step: int, **options: Any) -> dict[str, Any]:
    """Run one out-of-cadence or HTTP-provider evaluation. **Calls the service and returns.**

    Deliberately holds no scoring logic: the resolution, the generation, the aggregation, the
    floor comparison and the record are all :mod:`services.scoring.service`'s, so a pass run
    through the worker and a pass run in the ``train`` worker cannot produce different
    numbers.

    The spec is assembled by the caller's own run definition (F6's R-02.53 resolution), which
    is why this signature takes the run and the step rather than a serialised spec: a
    ``GenerationParams`` and a ``ResolvedSet`` are not JSON, and serialising them would make
    the broker a second place a run's configuration lives.
    """
    from services.scoring import entrypoint

    return entrypoint.evaluate_run(run_id=run_id, step=step, out_of_cadence=True, **options)
