"""The verdict's timing NFR, measured on the changed path (feature 016; FTASKS 14.4; FPRD §8).

3 sets x the full head set x 10^4 rows x B = 2,000 through ``wsp.verdict.decide`` itself, under
60 s. A wall-clock bound is a measurement, not a CI gate (suite lesson: never assert a bare
wall-clock threshold CI must meet), so this runs only with ``MIFORGE_MEASURE_PERF=1`` and skips
by name otherwise. Measured 2026-11-16 on the workstation: 6.7 s.
"""

from __future__ import annotations

import os
import time

import numpy as np
import pytest

from tests.unit.wsp.test_verdict import _head, world
from wsp.verdict import HeldoutSet, decide

pytestmark = pytest.mark.skipif(
    os.environ.get("MIFORGE_MEASURE_PERF") != "1",
    reason="test_verdict_timing: a wall-clock measurement; set MIFORGE_MEASURE_PERF=1 to run it",
)


def test_the_verdict_meets_its_timing_budget() -> None:
    sets = world(rows=10_000)
    big = []
    for s in sets:
        if s.role == "eval":
            big.append(s)
            continue
        fa = np.zeros(10_000, dtype=np.int64)
        big.append(
            HeldoutSet(
                s.name,
                s.role,
                fa,
                {
                    h: _head(
                        h, np.random.default_rng(1).standard_normal(10_000), fa, hd.threshold or 0.0
                    )
                    for h, hd in s.heads.items()
                },
            )
        )
    started = time.monotonic()
    out = decide(big, ["eval"], changed_entries=5, seed=1, b=2000)
    elapsed = time.monotonic() - started
    assert out.value in ("improved", "not_significant", "regressed")
    assert elapsed < 60, f"{elapsed:.1f} s"
