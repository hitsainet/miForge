"""``miforge.maintenance.chain_sweep`` - **the one path a chain has**  (FR-006.45).

## The name, and why the name is the defect this module exists to avoid

``TASK_ROUTES`` is an **explicit table keyed by area prefix**, so a task name matching no
entry lands on ``default`` - where a worker with no card picks it up, or in this case where
the sweep competes with CRUD instead of running on the queue that must keep moving while
``train`` is saturated.

**F6's own four documents name this sweeper ``miforge.chain.sweep``, which routes
NOWHERE**: there is no ``miforge.chain.*`` entry and there is no reason to add one, because
a sweep is maintenance. F1 recorded exactly this defect for its own loop task
(``miforge.training.run_step_loop``), F8's merge recorded it again, and
``test_run_loop_reachability.py`` carries a live tripwire asserting
``queue_for("miforge.chain.sweep") is None``. So the shipped name is
:data:`workers.celery_app.TASK_CHAIN_SWEEP`, and ``test_chain_task_wiring.py`` asserts it
resolves to ``maintenance`` rather than asserting it exists.

## Why ``maintenance`` and not ``default``

ADR-006 names the queue for what contends on it. A chain advance must happen while the
``train`` queue is saturated by the very run it is waiting on - that is the *only* time it
ever has work to do. On ``default`` it would queue behind ``reindex``; on ``train`` it would
queue behind the ten-hour task whose completion it exists to notice. The janitor is on
``maintenance`` for the same reason, and it is the same reason.

## Nothing here holds chain state

ADR-007: the record is the protocol. The sweep reads ``mf_chains`` and
``mf_chain_stages``, decides, and writes. **A worker that dies mid-sweep loses nothing**,
which is the property FR-006.45 is about and the reason the finishing stage's own task does
not enqueue its successor.

## One transaction per chain

:func:`services.chains.service.sweep` takes a session *factory*, not a session, so one
broken chain's transaction rolls back on its own and the next chain gets a clean one. That
is ``services/reindex.py``'s recorded lesson - *one subsystem must not kill the whole
operation* - and a sweep over a dozen curricula has the same exposure.
"""

from __future__ import annotations

import logging
from typing import Any

from core.logging import bind_run_id, configure_logging
from core.runtime_settings import effective_settings
from workers.celery_app import QUEUE_MAINTENANCE, TASK_CHAIN_SWEEP, celery_app

log = logging.getLogger(__name__)

#: Re-exported so a caller importing this module gets the one spelling. The constant is
#: **defined in** :mod:`workers.celery_app` beside ``TASK_JANITOR_SWEEP`` because the beat
#: schedule and the route table are there, and a name defined here and read there would be
#: a circular import solved by a second spelling.
TASK_NAME = TASK_CHAIN_SWEEP


# Celery's task decorator is untyped; the ignore is on the decorator, not the function, so
# the function's own signature is still checked.
@celery_app.task(name=TASK_CHAIN_SWEEP, queue=QUEUE_MAINTENANCE)  # type: ignore[untyped-decorator]
def chain_sweep() -> dict[str, Any]:
    """Advance or halt every live chain  (FR-006.45, R-02.27).

    **This registration plus the beat entry are the sweep's only production callers**, and
    ADR-016's first structural rule applies to them literally: a ``sweep`` with nothing on a
    schedule calling it is a fully-tested function that never runs, and since this is the
    *only* advance path a chain has, an unwired sweep makes chaining a feature that creates
    a curriculum and never advances it - green tests, a 202, and a chain that sits at stage
    one forever.

    ``test_chain_task_wiring.py`` asserts this task is in the **live** registry, that it
    routes to ``maintenance``, and that it is on the beat schedule; removing any of the
    three turns the suite red.
    """
    from services.chains import service as chain_service
    from workers.run_record import default_session_scope

    configure_logging(effective_settings().log_level)
    bind_run_id(None)  # a sweep belongs to no single run
    return chain_service.sweep(default_session_scope)


__all__ = ["TASK_NAME", "chain_sweep"]
