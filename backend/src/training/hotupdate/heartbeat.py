"""The §5.10 protocol heartbeat - off the step's path  (R-02.3; FR-011.14 - FR-011.19).

**This is not ``workers/heartbeat.py``.** That one is the *database* heartbeat ADR-007
requires of every job, throttled to 60 s and read by the janitor. This one proves to
*miLLM* that the attached process is alive (BRD-01 R-01.4, R-01.30). Two different things at
two paths, with one basename - *a shared basename is not drift; a shared basename at two
paths for one file is.*

## Its own task, never a hook in the loop

**A heartbeat that waits behind a forward pass is a missed heartbeat**, and a missed
heartbeat marks the attachment stale while the trainer is alive and working. So it runs on
its own thread with its own clock, and **MUT-18** - calling it from inside the step loop -
must turn the test red. That test therefore needs a *long* fake step and a heartbeat
**count**: a single-call assertion passes against a heartbeat that fired once before the
step and never again.

## The interval comes from miLLM, not from a constant here

BRD-01 R-01.30's response carries **the staleness window and a suggested interval**, and
FR-011.16 reads the hint. A local constant drifts from the timeout it must beat, in silence,
and the symptom is a stale attachment under load only. The configured override exists for an
operator who needs to force one; unset is normal.

## It keeps sending during a commit

The commit watchdog (R-01.22) and the staleness timeout (R-01.4) are independent: a quiet
trainer mid-commit looks dead. It is stopped **before** ``detach`` and never outlives the
attachment it proves.

## A rejected heartbeat stops the run

It carries the credential and the attachment header like any other holder-only gated call,
through ``client.call`` with no special path. A ``401``/``403`` **stops the run and names
the identity** - it is never swallowed as a transient network error, because *a heartbeat
refused for a missing token is indistinguishable from a dead trainer*: miLLM marks the
attachment stale and the run dies one interval later as the wrong diagnosis. **MUT-5 is the
control, and it is the negative control for the previous review round's own fix.**
"""

from __future__ import annotations

import logging
import threading
from dataclasses import dataclass, field
from typing import Any

from core.errors import Refusal

log = logging.getLogger(__name__)

#: Used only when miLLM states neither a hint nor a window **and** no override is set.
#: Named as a last resort rather than a default: FR-011.16's whole point is that the
#: interval is the serving side's to state.
FALLBACK_INTERVAL_SECONDS = 10.0


def interval_from(response: Any, *, override: float | None = None) -> float:
    """The interval to use: the override, else miLLM's hint, else half its window.

    Half the staleness window rather than the whole of it: an interval equal to the window
    means every heartbeat is the last one that could still arrive in time, so a single slow
    call marks the attachment stale.
    """
    if override is not None:
        return float(override)
    if isinstance(response, dict):
        # `heartbeat: {timeout_s, interval_hint_s}` per miLLM's committed FTDD 031 §5.3.
        # F11 read `suggested_interval_seconds`/`interval_seconds`, which the upstream does
        # not send, so FR-011.16's "read the hint rather than a local constant" silently
        # BECAME a local constant - it fell through to FALLBACK_INTERVAL_SECONDS on every
        # real response. Survivable (miLLM's as-built timeout is 30 s against a 10 s
        # fallback) and wrong, which is the combination that stays hidden.
        block = response.get("heartbeat")
        hint = block.get("interval_hint_s") if isinstance(block, dict) else None
        if isinstance(hint, (int, float)) and hint > 0:
            return float(hint)
        # And the window is `timeout_s`, in the SAME block - not a top-level
        # `staleness_window_seconds`. Two invented names in one function, both of which
        # fell through to the fallback against every real response.
        window = block.get("timeout_s") if isinstance(block, dict) else None
        if isinstance(window, (int, float)) and window > 0:
            return float(window) / 2.0
    log.warning(
        "miLLM stated no heartbeat interval; using the last-resort value",
        extra={
            "code": "heartbeat_interval_unstated",
            "fallback_seconds": FALLBACK_INTERVAL_SECONDS,
            "note": (
                "BRD-01 R-01.30 returns the staleness window and a suggested interval; "
                "miLLM FTDD 031 §5.3 spells them heartbeat.timeout_s and "
                "heartbeat.interval_hint_s"
            ),
        },
    )
    return FALLBACK_INTERVAL_SECONDS


@dataclass
class Heartbeat:
    """The interval task. **Started after attach, stopped before detach.**

    Args:
        client: the one call site. The heartbeat gets no private path.
        interval_seconds: from :func:`interval_from`.
        on_identity_rejected: called with the refusal when miLLM rejects the credential, so
            the run can be **stopped** rather than left to die as a stale attachment one
            interval later. Injected rather than imported so the stop is the caller's
            decision and is observable in a test.
        sleep: injected, so a test does not really wait an interval. A test that sleeps
            through an interval is a test nobody runs.
    """

    client: Any
    interval_seconds: float
    on_identity_rejected: Any = None
    sleep: Any = None
    #: Every beat sent. A **count**, because MUT-18's symptom is fewer beats rather than
    #: none: a single-call assertion passes against a starved heartbeat.
    beats: int = 0
    failures: list[str] = field(default_factory=list)
    _stop: threading.Event = field(default_factory=threading.Event, repr=False)
    _thread: threading.Thread | None = field(default=None, repr=False)

    def beat_once(self) -> dict[str, Any] | None:
        """One heartbeat. Returns miLLM's response, or ``None`` if it was refused."""
        try:
            body = self.client.call("heartbeat")
        except Refusal as exc:
            self.failures.append(exc.code)
            if exc.code == "weights_identity_rejected":
                log.error(
                    "the heartbeat was refused for the credential; stopping the run",
                    extra={
                        "code": "heartbeat_identity_rejected",
                        "upstream_refusal": exc.code,
                        "note": (
                            "a heartbeat refused for a missing token is indistinguishable "
                            "from a dead trainer: miLLM would mark the attachment stale and "
                            "the run would die one interval later as a crash. Reported as "
                            "the identity failure it is, and NOT retried."
                        ),
                    },
                )
                if self.on_identity_rejected is not None:
                    self.on_identity_rejected(exc)
                self._stop.set()
                return None
            log.warning(
                "a heartbeat did not reach miLLM",
                extra={"code": "heartbeat_failed", "upstream_refusal": exc.code},
            )
            return None
        self.beats += 1
        return dict(body) if isinstance(body, dict) else None

    def run_until_stopped(self) -> None:
        """The loop. **Beats first, then waits** - so an attachment is proved live at once."""
        waiter = self.sleep if self.sleep is not None else self._stop.wait
        while not self._stop.is_set():
            self.beat_once()
            if self._stop.is_set():
                break
            waiter(self.interval_seconds)

    def start(self) -> None:
        """Start the task. Daemon, so it can never hold a worker open past its run."""
        if self._thread is not None:
            return
        self._thread = threading.Thread(
            target=self.run_until_stopped, name="hotupdate-heartbeat", daemon=True
        )
        self._thread.start()

    def stop(self, timeout: float = 5.0) -> None:
        """Stop **before** ``detach``, never after (FR-011.18)."""
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=timeout)
            self._thread = None

    @property
    def stopped(self) -> bool:
        return self._stop.is_set()

    def __enter__(self) -> Heartbeat:
        self.start()
        return self

    def __exit__(self, *exc: object) -> None:
        self.stop()
