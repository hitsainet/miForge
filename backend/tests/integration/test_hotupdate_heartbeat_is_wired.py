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


def _attempt(harness: Harness, monkeypatch: pytest.MonkeyPatch):
    """Drive the real acquirer until the rebuild refuses; hand back the acquirer and the
    heartbeat it STARTED (captured at construction, because a failed acquire now stops it
    and detaches - the leak this file's old shape depended on, closed 2026-10-17)."""
    from training.hotupdate import heartbeat as heartbeat_module
    from training.hotupdate.attach import AttachAcquirer

    created: list = []
    real = heartbeat_module.Heartbeat

    class Recording(real):  # type: ignore[misc, valid-type]
        def __init__(self, *args, **kwargs) -> None:
            super().__init__(*args, **kwargs)
            created.append(self)

    monkeypatch.setattr(heartbeat_module, "Heartbeat", Recording)
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
    # Which POST-attach step refused is not the subject; that it comes after the heartbeat
    # starts is. Named, so a refusal moving IN FRONT of the heartbeat fails here.
    assert caught.value.code in (
        "incompatible_torch_build",
        "weights_rebuild_failed",
        "weights_rebuild_unavailable",
    ), (
        f"unexpected refusal {caught.value.code!r}. If a new check now refuses BEFORE the "
        f"heartbeat starts, that is the defect this file exists to prevent: the heartbeat "
        f"must precede every slow or refusing step after attach."
    )
    assert len(created) == 1, f"expected one heartbeat, the attach path made {len(created)}"
    return acquirer, created[0]


class TestTheAttachPathStartsIt:
    def test_a_heartbeat_was_started_after_the_attach(self, harness, monkeypatch) -> None:
        _, beating = _attempt(harness, monkeypatch)
        assert beating is not None

    def test_it_actually_BEAT_and_not_merely_was_constructed(self, harness, monkeypatch) -> None:
        """The call count, not "was called": a constructed-but-unstarted thread beats zero."""
        _, beating = _attempt(harness, monkeypatch)
        beats = harness.fake.call_count("heartbeat")
        assert beats >= 1, f"no heartbeat reached miLLM (count={beats})"
        assert beating.beats >= 1, "the heartbeat recorded no beat of its own"

    def test_the_interval_came_from_miLLMs_HINT_not_the_fallback(
        self, harness, monkeypatch
    ) -> None:
        """FR-011.16: read the hint rather than a local constant (the fake's hint differs
        from the fallback, so this distinguishes the two)."""
        from training.hotupdate import heartbeat as heartbeat_module

        hinted = harness.fake.suggested_interval_seconds
        assert hinted != heartbeat_module.FALLBACK_INTERVAL_SECONDS
        _, beating = _attempt(harness, monkeypatch)
        assert beating.interval_seconds == hinted


class TestAFailedAcquireReleasesTheAttachment:
    def test_the_heartbeat_stops_and_the_attachment_is_detached_once(
        self, harness, monkeypatch
    ) -> None:
        """Before 2026-10-17 a failure after `attach` left a LIVE attachment beating in the
        worker, holding miLLM's one slot until the worker restarted: `drive_run` releases
        only what `acquire` returned, and a raise returns nothing."""
        acquirer, beating = _attempt(harness, monkeypatch)
        assert beating.stopped, "the heartbeat kept the failed attachment alive"
        assert harness.fake.call_count("detach") == 1
        assert acquirer.heartbeat is None and acquirer.client is None
