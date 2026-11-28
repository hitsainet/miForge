"""Drive the WSP pool in a test as production does, in one call (feature 019 P3; TID §8).

Since P3 the pass only CLAIMS (``dispatch.drive``) and the push is its own Celery task
(``wsp.tasks.push`` -> ``dispatch.push_claimed``) on the solo ``wsp_push`` queue. ``drive_inline``
runs one pass, then each push the pass enqueued, in order, against the worker the claim chose,
which is what a broker with one push consumer does. One ``read_window`` serves both the claim's
residency read and the push's read, as ``wsp.window_read.read_window`` does in production.

By default each ``claimed:<run>:<index>`` action is replaced IN PLACE by the push's outcome
(``pushed:<run>``, ``push_failed:<run>``, ``lapsed:<run>``, ``not_pushing:<run>``,
``requeued:<run>``), so a test
written against 016's single-step pass reads the same. ``keep_claims=True`` keeps both: the claim
action, then the outcome.
"""

from __future__ import annotations

import datetime as dt
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

import sqlalchemy as sa

from wsp import dispatch
from wsp.models import WspRun


def address(index: int) -> str:
    return f"http://wsp-worker-{index}.test:9000"


def slots(workers: Any) -> list[dispatch.WorkerSlot]:
    """One slot per worker: a single worker is worker 0."""
    listed = list(workers) if isinstance(workers, (list, tuple)) else [workers]
    return [dispatch.WorkerSlot(i, address(i), w) for i, w in enumerate(listed)]


#: The window a miServer reports with a model resident in these tests (``resident_window``).
RESIDENT_WINDOW = 32_768


def resident_window(endpoint: Any) -> int | None:
    """The default reader of ``drive_inline`` (2026-11-27): a miServer endpoint's server has a
    model resident and reports :data:`RESIDENT_WINDOW`; every other server reports nothing, so a
    generic endpoint is ``assumed`` unless it states a window. Before 2026-11-27 the default read
    nothing for every endpoint, which is the state of a miServer with no model resident: a run on
    such an endpoint now waits (``dispatch._await_resident``), so a test of that wait passes
    ``read_window=dispatch.no_window`` (or its own reader) explicitly."""
    return RESIDENT_WINDOW if getattr(endpoint, "serving", None) == "miserver" else None


def drive_inline(
    factory: Any,
    workers: Any,
    settings: Any,
    *,
    read_sensitivity: Callable[[Any, Any], str],
    files_for: Callable[[Any, Any], list[tuple[str, Path]]] = lambda run, bundle: [],
    read_window: Callable[[Any], int | None] = resident_window,
    pool_size: int = 1,
    now: Callable[[], dt.datetime] = dispatch.utcnow,
    keep_claims: bool = False,
    pushes: list[str] | None = None,
    recycle_pod: Any = None,
    backoff: Any = None,
) -> list[str]:
    """One pass, then its pushes. ``pushes`` (when given) collects every enqueued run id."""
    the_slots: Sequence[dispatch.WorkerSlot] = slots(workers)
    enqueued: list[str] = []
    actions = dispatch.drive(
        factory,
        the_slots,
        settings,
        pool_size=pool_size,
        read_sensitivity=read_sensitivity,
        files_for=files_for,
        enqueue_push=enqueued.append,
        now=now,
        recycle_pod=recycle_pod,
        read_window=read_window,
        backoff=backoff,
    ).actions
    if pushes is not None:
        pushes.extend(enqueued)
    outcomes: dict[str, str] = {}
    for run_id in enqueued:
        with factory() as session:
            index = session.scalar(sa.select(WspRun.worker_index).where(WspRun.run_id == run_id))
        worker = the_slots[int(index or 0)].client
        outcome = dispatch.push_claimed(
            factory,
            worker,
            settings,
            run_id,
            read_window=read_window,
            files_for=files_for,
            now=now,
        )
        outcomes[run_id] = f"{outcome}:{run_id}"
    out: list[str] = []
    for action in actions:
        if action.startswith("claimed:"):
            run_id = action.split(":")[1]
            if keep_claims:
                out.append(action)
            out.append(outcomes.get(run_id, action))
        else:
            out.append(action)
    return out
