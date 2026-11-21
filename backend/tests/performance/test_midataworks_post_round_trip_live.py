"""F10 task 8.6's WRITE half against a LIVE miDataworks - prepared, gated, run by the coordinator.

**Runnable once miDataworks' D2 close route (``POST /api/v1/review-queues/{id}/close``,
its ``e2e/outstanding`` ``989038f``) is DEPLOYED.** Everything is in
:mod:`tests.support.round_trip_8_6`; this file only gates it.
Three gates, all required, because this is the one measurement in miForge that writes to a
live sibling:

1. ``@pytest.mark.upstream`` - deselected by default and in CI;
2. ``MIDATAWORKS_BASE_URL`` set - the live service;
3. ``MIFORGE_F10_86_ALLOW_LIVE_WRITE=1`` - the operator's explicit go.

**To run it (coordinator, after D2 lands and with the operator's go):**

    MIDATAWORKS_BASE_URL=http://<midataworks-service>:<port> \\
    MIFORGE_F10_86_ALLOW_LIVE_WRITE=1 \\
    pytest -m upstream -s backend/tests/performance/test_midataworks_post_round_trip_live.py

Optional: ``MIFORGE_F10_86_BATCH`` (default 100), ``MIFORGE_F10_86_REPS`` (default 5). Run
it **pod-to-pod** (``kubectl exec`` in the miForge backend pod), as the read half was, and
quote the topology with the figure. Then record the medians in F10 FTASKS 8.6 beside the
read half, and confirm in miDataworks' UI that the closed queues are filtered out.
"""

from __future__ import annotations

import os

import pytest

from tests.support import round_trip_8_6 as rt

BASE_URL = os.environ.get("MIDATAWORKS_BASE_URL", "").strip()

pytestmark = [
    pytest.mark.upstream,
    pytest.mark.skipif(not BASE_URL, reason="MIDATAWORKS_BASE_URL is unset: no live miDataworks"),
    pytest.mark.skipif(
        os.environ.get(rt.ALLOW_ENV) != "1",
        reason=f"{rt.ALLOW_ENV}=1 is the operator's explicit go for a write to a live service",
    ),
]


def test_measure_the_post_and_read_back_and_close_every_queue() -> None:
    from clients.midataworks import MiDataworksClient

    batch = int(os.environ.get(rt.BATCH_ENV, rt.DEFAULT_BATCH))
    reps = int(os.environ.get(rt.REPS_ENV, rt.DEFAULT_REPS))
    with MiDataworksClient(
        action="measuring F10 8.6's post round trip", capability="human_review_selector"
    ) as client:
        report = rt.measure(client, batch=batch, reps=reps)
    print()
    print("\n".join(report.lines()))
    assert report.closed == report.opened
