"""The part-import job  (F10 task 4.11; ADR-006, ADR-007).

Import is background work: digesting a multi-gigabyte split is not something a request can
wait on. It runs on the **``default``** queue, never ``train`` - nothing here touches a GPU,
and a GPU queue is for what contends on the card.

**Cooperative cancellation only** (ADR-006). Every worker is ``--pool=solo -c 1``, so
``revoke(terminate=True)`` has no pool child to signal and a busy solo worker never reads
the control queue. An import checks for a stop at the one safe boundary it has - before the
atomic placement - and after that point the rename either happened or it did not.

## Two deviations from this feature's task list, both recorded rather than silent

**The module is here, not under ``workers/``.** ``backend/src/workers/`` is the foundation's
package and is single-owner; this feature does not edit it. The task is registered by its
decorator, so importing this module registers it - and the API's ``parts`` endpoint imports
it, which is the production import path.

**The task is named ``miforge.index.part_import``, not ``miforge.parts.import``.**
``celery_app.TASK_ROUTES`` has no ``miforge.parts.*`` entry, and
``test_janitor_wiring.test_every_registered_miforge_task_resolves_to_a_queue`` fails for any
registered task that resolves to no route - *"a task with no route silently lands on
default, where a worker with no card picks up a training task and fails."* Adding the route
is a one-line edit to a file this feature does not own, so until that lands the task is
named under the existing ``miforge.index.*`` area, which routes to ``default`` - the queue
it needs. The name is a constant here, so the rename is one line on this side too.

**And one gap that is the foundation's to close**, reported rather than worked around: a
Celery worker started with ``-A workers.celery_app`` does not import this module, so the
task is registered in the API process and not in the worker's. ``celery_app`` needs an
``include=[...]`` naming the feature task modules. Until then an enqueued import is received
by a worker that does not know the task. The synchronous path (``wait=true``) runs the same
function and is what the tests exercise.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from workers.celery_app import QUEUE_DEFAULT, celery_app

log = logging.getLogger("miforge.parts.tasks")

#: The task name. See the module docstring for why it is under ``miforge.index.*``.
TASK_PART_IMPORT = "miforge.index.part_import"


@celery_app.task(name=TASK_PART_IMPORT, queue=QUEUE_DEFAULT)  # type: ignore[untyped-decorator]
def part_import(
    source: str,
    locator: str,
    *,
    recorded_source: str = "midataworks_contract",
    external_provenance: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Import one part. **The same function the synchronous route calls.**

    One implementation, so a part imported through the worker and a part imported in-process
    produce identical rows by construction rather than by two code paths agreeing (which is
    AC-010.5's point and the reason the integration test can compare them at all).

    Returns:
        The import summary - part id, kind, digest, splits, caveats, licence classes.
    """
    from core.config import get_settings
    from db.session import session_scope
    from services.parts import acquire

    parts_root = Path(get_settings().parts_root)
    with session_scope() as session:
        if source == "dir":
            result = acquire.from_dir(
                session,
                Path(locator),
                parts_root=parts_root,
                source=recorded_source,
                external_provenance=external_provenance,
            )
        elif source == "http":
            result = acquire.from_http(session, locator, parts_root=parts_root)
        else:
            result = acquire.from_hub()

    log.info(
        "part_import_job_complete",
        extra={
            "code": "part_import_job_complete",
            "part_id": result.part_id,
            "part_kind": result.part_kind,
            "created": result.created,
        },
    )
    return {
        "part_id": result.part_id,
        "created": result.created,
        "part_kind": result.part_kind,
        "manifest_sha256": result.manifest_sha256,
        "splits": list(result.splits),
        "caveats": [dict(caveat) for caveat in result.caveats],
        "licence_classes": list(result.licence_classes),
    }


def enqueue_import(
    *,
    source: str,
    locator: str,
    recorded_source: str = "midataworks_contract",
    external_provenance: dict[str, Any] | None = None,
) -> str:
    """Queue an import and return its job id.

    A function rather than ``.delay()`` at the call site so the route does not import Celery
    and so there is one place the task's arguments are assembled - a second assembly is how
    a keyword gets dropped and an external part is recorded as a contract one.
    """
    async_result = part_import.apply_async(
        args=[source, locator],
        kwargs={
            "recorded_source": recorded_source,
            "external_provenance": external_provenance,
        },
        queue=QUEUE_DEFAULT,
    )
    return str(async_result.id)
