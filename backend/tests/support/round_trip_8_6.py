"""F10 task 8.6's WRITE half: post + read-back against a live miDataworks, closed afterwards.

**Why this is prepared and not run.** The read half of 8.6 is measured
(``tests/performance/test_midataworks_round_trip_live.py``). The write half was refused on
2026-10-16 because a live post leaves a permanent, human-facing review queue and
miDataworks offered no way to remove or close one. **Suite decision D2 (operator,
2026-10-16):** miDataworks adds a queue *close* transition (a route, and a UI filter of
closed queues); then this measurement runs and closes the queues it opened.

**The close is miDataworks' real D2 route** (its ``e2e/outstanding`` ``989038f``):
``POST /api/v1/review-queues/{id}/close`` with ``{"reason": ...}``, answering the queue with
``state: "closed"``; idempotent; ``403 QUEUE_NOT_YOURS`` for a queue miForge did not create;
``409 AUDIT_IN_PROGRESS``. It is called through miForge's single outbound call site,
``MiDataworksClient.close_queue``. *Until 2026-10-17 the route was a run-time parameter
(an environment variable) because the API shape was not published; it is now.*

**What it guarantees, and tests in CI against the fake**
(``tests/unit/test_f10_8_6_measurement.py``):

* every queue it opens is closed - **including when a leg fails mid-way** - and the close
  is **verified by reading the queue back**, not assumed from a 2xx;
* if any queue could not be closed it fails loudly **naming the ids**, so a human never
  inherits an unexplained synthetic queue;
* every figure carries its batch size, its sample, and the statement that **no human
  waiting is included** (a decision is never awaited; the read-back reads what exists).
"""

from __future__ import annotations

import statistics
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from functools import partial
from typing import Any

#: The state a closed queue reads back as - miDataworks' ``ReviewQueueState.CLOSED``.
CLOSED_STATE = "closed"
#: The reason recorded on every queue this measurement closes.
CLOSE_REASON = "miForge F10 8.6 measurement - synthetic rows, closed automatically"
#: The operator's explicit go for a write to a live service. Must be exactly ``1``.
ALLOW_ENV = "MIFORGE_F10_86_ALLOW_LIVE_WRITE"
BATCH_ENV = "MIFORGE_F10_86_BATCH"
REPS_ENV = "MIFORGE_F10_86_REPS"

DEFAULT_BATCH = 100
DEFAULT_REPS = 5

#: Written into every queue this opens, so a human who sees one knows what it is.
MEASUREMENT_QUESTION = (
    "[miForge F10 8.6 measurement - synthetic rows, closed automatically; do not label]"
)


def close_queue(client: Any, queue_id: str) -> dict[str, Any]:
    """The one call that closes a queue: miDataworks' D2 route, through the production client."""
    closed: dict[str, Any] = client.close_queue(queue_id, reason=CLOSE_REASON)
    return closed


@dataclass
class Leg:
    name: str
    samples_ms: list[float] = field(default_factory=list)

    @property
    def n(self) -> int:
        return len(self.samples_ms)

    @property
    def median(self) -> float | None:
        return statistics.median(self.samples_ms) if self.samples_ms else None


@dataclass
class Report:
    batch: int
    reps: int
    legs: dict[str, Leg]
    opened: list[str]
    closed: list[str]

    def lines(self) -> list[str]:
        out = [
            "=== F10 task 8.6 - post + read-back against a LIVE miDataworks ===",
            f"  batch {self.batch} candidates per post; {self.reps} reps; no warm-up "
            f"(each rep opens a real queue); NO human waiting in any figure",
            "  topology: state it with the figure (pod-to-pod across namespaces, or not)",
        ]
        for leg in self.legs.values():
            if leg.n:
                out.append(
                    f"  {leg.name:<34} median {leg.median:7.1f} ms  "
                    f"min/max {min(leg.samples_ms):7.1f} / {max(leg.samples_ms):7.1f}  "
                    f"(n={leg.n})"
                )
        out.append(f"  queues opened {len(self.opened)}, closed and verified {len(self.closed)}")
        return out


def _timed(leg: Leg, call: Callable[[], Any]) -> Any:
    started = time.perf_counter()
    result = call()
    leg.samples_ms.append((time.perf_counter() - started) * 1000.0)
    return result


def candidates(batch: int, rep: int) -> list[dict[str, Any]]:
    return [
        {
            "external_id": f"f10-86-r{rep:02d}-{index:04d}",
            "prompt": "Synthetic measurement prompt - not for labelling.",
            "completion": f"synthetic candidate {index}",
            "provenance": {"part": "prompt-set/f10-8-6-measurement@1"},
        }
        for index in range(batch)
    ]


def measure(
    client: Any,
    *,
    batch: int = DEFAULT_BATCH,
    reps: int = DEFAULT_REPS,
) -> Report:
    """Create -> post -> read back, ``reps`` times; then close and verify EVERY queue opened.

    ``client`` is the production :class:`clients.midataworks.MiDataworksClient`, so every
    leg - the close included - goes through miForge's single outbound call site.
    """
    legs = {
        name: Leg(name)
        for name in (
            "create_external_queue",
            f"post_candidates (batch of {batch})",
            "read_decisions (read-back)",
            "create + post + read-back",
            "close (D2 route)",
        )
    }
    create_leg, post_leg, read_leg, whole_leg, close_leg = legs.values()
    opened: list[str] = []
    closed: list[str] = []
    unclosed: dict[str, str] = {}
    try:
        for rep in range(reps):
            started = time.perf_counter()
            created = _timed(
                create_leg,
                partial(
                    client.create_external_queue,
                    question=MEASUREMENT_QUESTION,
                    label_set=["good", "needs-rework"],
                    external_ref={"purpose": "f10-8.6-measurement", "rep": rep},
                ),
            )
            queue_id = str(created["id"])
            opened.append(queue_id)
            accepted = _timed(
                post_leg,
                partial(client.post_candidates, queue_id, candidates(batch, rep)),
            )
            if len(accepted) != batch:
                raise AssertionError(f"posted {batch}, miDataworks acknowledged {len(accepted)}")
            _timed(read_leg, partial(client.read_decisions, queue_id))
            whole_leg.samples_ms.append((time.perf_counter() - started) * 1000.0)
    finally:
        # EVERY queue opened, including when a leg above raised. Verified by reading back.
        for queue_id in opened:
            try:
                # A refusal (404/403/409) raises and lands in `unclosed` below.
                _timed(close_leg, partial(close_queue, client, queue_id))
                # Verified by READING BACK, not trusted from the 200.
                state = client.get_queue(queue_id).get("state")
                if state != CLOSED_STATE:
                    unclosed[queue_id] = f"reads back state={state!r}, not {CLOSED_STATE!r}"
                    continue
                closed.append(queue_id)
            except Exception as exc:
                unclosed[queue_id] = f"{type(exc).__name__}: {exc}"
    if unclosed:
        raise AssertionError(
            "F10 8.6 opened review queues it could NOT close - an operator must close these "
            f"by hand in miDataworks: {unclosed}"
        )
    return Report(batch=batch, reps=reps, legs=legs, opened=opened, closed=closed)
