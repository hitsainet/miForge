"""The §5.10 heartbeat is actually started by the attach path.

**The defect.** Three things were true at once: `interval_from` had no production caller,
`Heartbeat` was never constructed anywhere in `backend/src/`, and
`hotupdate_heartbeat_interval_seconds` was declared in `core/config.py`, plumbed into F11's
settings dataclass and **read by nobody**. `AttachAcquirer.detach` took a `heartbeat`
argument and stopped it — so the code read as though a heartbeat existed somewhere.

So a §5.10 run attached and never beat. miLLM's as-built timeout is 30 s; R-01.4 marks the
attachment stale and the run **dies looking like a crash**. That is the same wrong diagnosis
`Heartbeat.beat_once` already warns about for the credential case, arriving by a different
route — and it is worse than an unreachable capability, because the mode could not work at
all rather than merely being untested.

**Why this test can exist on a machine with no GPU**, which is what makes it worth having:
the heartbeat starts *before* `rebuild.build`, so the rebuild raising (as it must with a
fake surface and no CUDA IPC handles) does not stop this from being observable. The
placement is itself the requirement — a multi-gigabyte rebuild can plausibly outlast the
staleness window, so starting the heartbeat afterwards would let the attachment go stale
*during the operation the attachment exists for*. F11 already learned this once: FR-011.5's
`run_state.remember` was placed after the rebuild and was "unreachable on a machine without
the ML wheels, which is every machine in CI".
"""

from __future__ import annotations

import pytest

from core.errors import Refusal
from tests.support.fixtures_hotupdate import Harness
from training.acquisition.base import AcquisitionMode, AcquisitionSpec


def _attempt(harness: Harness):
    """Drive the real acquirer until the rebuild refuses, and hand back the acquirer."""
    from training.hotupdate.attach import AttachAcquirer

    acquirer = AttachAcquirer(transport=harness.fake.transport())
    spec = AcquisitionSpec(mode=AcquisitionMode.ATTACH)
    with pytest.raises(Refusal) as caught:
        acquirer.acquire(spec, run_id="hot-run")
    # The ATTACH must have succeeded, or every assertion below is vacuous - a refusal from
    # `attach` itself would leave no heartbeat for the honest reason. Asserted on the call
    # count rather than inferred from the refusal code.
    assert harness.fake.call_count("attach") == 1, (
        f"attach did not happen exactly once (calls: {harness.fake.call_count('attach')}), "
        f"so 'no heartbeat' would be correct behaviour and this test proves nothing"
    )
    # Which POST-attach step refused is not the subject. On a machine with no CUDA IPC
    # handles it is `incompatible_torch_build` (this workstation's torch differs from the
    # fake's stated producer) or the rebuild itself; both come after the heartbeat starts,
    # and that ordering is exactly the property under test. Named rather than left open, so
    # a future refusal moving in FRONT of the heartbeat fails here instead of silently
    # making the assertions vacuous.
    assert caught.value.code in (
        "incompatible_torch_build",
        "weights_rebuild_failed",
        "weights_rebuild_unavailable",
    ), (
        f"unexpected refusal {caught.value.code!r}. If a new check now refuses BEFORE the "
        f"heartbeat starts, that is the defect this file exists to prevent: the heartbeat "
        f"must precede every slow or refusing step after attach."
    )
    return acquirer


class TestTheAttachPathStartsIt:
    def test_a_heartbeat_exists_after_an_attach(self, harness: Harness) -> None:
        acquirer = _attempt(harness)
        assert acquirer.heartbeat is not None, (
            "the attach path started no heartbeat, so the attachment goes stale at miLLM's "
            "timeout and the run dies as a crash"
        )

    def test_it_actually_BEAT_and_not_merely_was_constructed(self, harness: Harness) -> None:
        """The call count, not "was called": a constructed-but-unstarted thread beats zero.

        `run_until_stopped` beats first and waits second, precisely so an attachment is
        proved live at once — so one beat must have reached the surface by now.
        """
        acquirer = _attempt(harness)
        acquirer.heartbeat.stop()
        beats = harness.fake.call_count("heartbeat")
        assert beats >= 1, (
            f"no heartbeat reached miLLM (count={beats}). A constructed-but-unstarted "
            f"Heartbeat beats zero times, which is why this asserts the COUNT and not that "
            f"an object exists - the distinction ADR-016 #1 draws between 'was called' and "
            f"the call count."
        )
        assert acquirer.heartbeat.beats >= 1, "the heartbeat recorded no beat of its own"

    def test_the_interval_came_from_miLLMs_HINT_not_the_fallback(self, harness: Harness) -> None:
        """FR-011.16: read the hint rather than a local constant.

        The fake's `interval_hint_s` is deliberately set to something that is **not** the
        fallback, so this distinguishes "read the hint" from "fell through". Without that
        difference the assertion would hold against the defect.
        """
        from training.hotupdate import heartbeat as heartbeat_module

        hinted = harness.fake.suggested_interval_seconds
        assert (
            hinted != heartbeat_module.FALLBACK_INTERVAL_SECONDS
        ), "the fake's hint equals the fallback, so this test cannot tell them apart"
        acquirer = _attempt(harness)
        acquirer.heartbeat.stop()
        assert acquirer.heartbeat.interval_seconds == hinted

    def test_detach_with_no_argument_stops_the_one_acquire_started(self, harness: Harness) -> None:
        """`detach()` took a heartbeat parameter and defaulted to stopping nothing."""
        acquirer = _attempt(harness)
        beating = acquirer.heartbeat
        acquirer.detach()
        assert beating.stopped, "detach() left the heartbeat running"
        assert acquirer.heartbeat is None, "the acquirer still holds a stopped heartbeat"
