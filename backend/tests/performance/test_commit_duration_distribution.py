"""The commit-duration distribution - **the one measurement another project's gate reads.**

``gpu`` **and** ``upstream``-marked, deselected by default, and **blocked on another
repository**: miLLM has implemented none of BRD-01's weights surface.

**Why this is a deliverable rather than telemetry.** miLLM Feature 31's acceptance gate sets
its commit watchdog to **at least ten times the measured 99.9th-percentile commit duration
under concurrent serving** (BRD-01 §7's resolution). The measurement is this feature's
(R-02.14), so F11 owes another repository a number - and ten times a figure measured in
isolation, on a link nobody named, is not a watchdog.

**The ordering is not circular, and that was checked.** Feature 031 ships with the
provisional 2,000 ms, set from its own synthetic harness of bracketed updates; F11 then runs
and produces this distribution; the gate is re-set from it. So nothing in miLLM waits on
miForge, and AC-011.16 is a prerequisite of another project's acceptance *after* it has
shipped once. Confirmed as the intended order (OQ-6).

**What the figure must carry**, every time (ADR-012, ADR-024):

* the **sample size** - *a distribution from fifty steps is not one*;
* the **conditions** - measured **under contention**, one card, time-sliced, **no MPS**, so
  the commit contends with serving rather than overlapping it;
* the **topology** - both links on this node are **x4, not x16**, and a host-to-device
  figure taken from an x16 number is wrong by three to six times.
"""

from __future__ import annotations

import os

import pytest

pytestmark = [
    pytest.mark.gpu,
    pytest.mark.upstream,
    pytest.mark.skipif(
        not os.environ.get("MILLM_BASE_URL"),
        reason=(
            "MILLM_BASE_URL is unset and BRD-01 is unimplemented. This is the measurement "
            "miLLM Feature 31's acceptance gate reads; it cannot be taken against a fake, "
            "and a fake's numbers would be a fabricated input to another project's gate."
        ),
    ),
]


# **The report's SHAPE is asserted where it will actually run**:
# ``tests/unit/test_hotupdate_timings.py::TestTheReportAnotherRepositorysGateReads``, which
# is not ``upstream``-marked and therefore runs in CI. A copy of it here would be deselected
# with this file and would never execute - a test that cannot run is not a check, and the
# shape is exactly the half that must not fail for the first time on the node at acceptance.
class TestTheLiveMeasurement:
    def test_p99_9_under_concurrent_serving(self) -> None:
        """**AC-011.16.** At least a few thousand bracketed commits with ``/v1`` driven
        concurrently, reported with its sample size, conditions and topology, and handed to
        the operator for miLLM Feature 31's gate. **Reported; nothing in that repository is
        edited from here.**"""
        pytest.skip("15.1: needs the real surface, and the node.")

    def test_the_brackets_overhead_is_inside_the_five_percent_budget(self) -> None:
        """ADR-024: *measure the path you changed.* The budget is measured **on the
        bracketed path**, which has four extra HTTP calls in it - a step-overhead number
        taken from the default path says nothing about this one."""
        pytest.skip("15.2: needs the real surface, and the node.")

    def test_the_commit_creep_warning_fires_before_the_watchdog_does(self) -> None:
        """The early warning, live: *a run whose commits creep toward the watchdog is the
        early warning that the watchdog is set too tight* - and a spurious ``torn`` set is a
        serving outage, so this must fire before the operator finds out that way."""
        pytest.skip("15.1: needs the real surface, and the node.")
