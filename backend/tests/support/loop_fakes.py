"""Fakes for the loop  (F6 task 3.1).

**The most valuable file in F6's test tree:** it makes the whole feature testable with no
GPU, because every collaborator the loop orchestrates is injected as a callable
(:class:`training.boundary.BoundaryContext`) rather than imported.

Everything here **records its calls**, and records the arguments as well as the count.
"Was called" passes against a call sending the wrong arguments (ADR-016 #1), and the two
mutations these fakes exist for - M-12 (a second ``source.next()``) and M-11 (no step-0
evaluation) - are both *count* defects that a boolean spy cannot see.

**What is deliberately not faked:** F4's breach, F5's save and F2's batch record, wherever a
test is about the seam. *Fixtures whose fields agree by construction are how a suite stays
green over a real bug*, so the cross-seam tests round-trip the real writer and the real
reader and these fakes are used only where the subject is the loop's own arithmetic.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

from training.stop.base import StopContext, Verdict


@dataclass
class FakeStep:
    """F1's one turn, recorded.

    Args:
        losses: the loss each step reports, in order. The last value repeats once the list
            runs out, so a plateau test can state a flat tail without padding it.
        raise_at: a step at which to raise ``boom``, for E-9's no-run-end-save test.
        cancel_at: a step at which to raise ``RunCancelled``, for the cancellation path.
    """

    losses: Sequence[float] = (0.9, 0.7, 0.5, 0.45, 0.44, 0.44, 0.44, 0.44)
    raise_at: int | None = None
    cancel_at: int | None = None
    #: Every step this fake was asked for, in order. **A list, not a counter** - the
    #: sequence is what catches a loop that skipped or repeated a step, and a counter of
    #: the right length can be built from the wrong steps.
    calls: list[int] = field(default_factory=list)

    def __call__(self, step: int) -> Any:
        self.calls.append(step)
        if self.cancel_at is not None and step >= self.cancel_at:
            from workers.cancellation import RunCancelled

            raise RunCancelled("run", "operator_stop")
        if self.raise_at is not None and step >= self.raise_at:
            raise RuntimeError("boom")
        index = min(step - 1, len(self.losses) - 1)
        return _Outcome(loss=float(self.losses[max(0, index)]))


@dataclass
class _Outcome:
    """The one field the loop reads off F1's step outcome, and no more.

    Narrow on purpose: the loop must not depend on F1's outcome shape, so a fake that
    carried every field would hide a loop that had started reading one.
    """

    loss: float


@dataclass
class FakeSource:
    """F2's source. **The loop must never call this** (FR-006.2, M-12).

    F1's ``step.run_one`` pulls the batch, exactly once per boundary. This fake exists so a
    test can assert ``source.calls == 0`` from the loop's side: two pulls per step means
    every other batch is trained on nothing and the step series still looks perfect.
    """

    calls: int = 0
    health_signals: tuple[str, ...] = ()

    def next(self) -> Any:
        self.calls += 1
        return object()

    def close(self) -> None:
        pass


@dataclass
class FakeEvaluate:
    """F4's evaluation, recorded by step.

    ``calls`` includes **step 0** when the baseline ran, which is the whole subject of
    M-11: a run whose first data point is already post-update has nothing to compare
    against (acceptance criterion 12).
    """

    calls: list[int] = field(default_factory=list)
    raise_at: int | None = None

    def __call__(self, step: int) -> Any:
        self.calls.append(step)
        if self.raise_at is not None and step == self.raise_at:
            from core.errors import refuse

            raise refuse("evaluation_failed", "the provider refused", step=step)
        return {"step": step}


@dataclass
class FakeSave:
    """F5's save, recorded by step. ``calls`` is the subject of M-13 and of S-2."""

    calls: list[int] = field(default_factory=list)

    def __call__(self, step: int) -> Any:
        self.calls.append(step)
        return {"step": step}


@dataclass
class FakeEmit:
    """The socket emitter, recorded as ``(run_id, event, payload)`` triples.

    The **payload** is kept, not just the event name: M-17's control renames an event or
    drops an emission, and a spy that recorded only names would pass against an emission
    carrying the wrong step.
    """

    events: list[tuple[str, str, dict[str, Any]]] = field(default_factory=list)
    raise_on: str | None = None

    def __call__(self, run_id: str, event: str, payload: dict[str, Any]) -> None:
        if self.raise_on is not None and event == self.raise_on:
            raise RuntimeError("transport down")
        self.events.append((run_id, event, dict(payload)))

    def named(self, event: str) -> list[dict[str, Any]]:
        return [payload for _run, name, payload in self.events if name == event]


@dataclass
class FakeHeartbeat:
    """The database heartbeat, counting **writes** and **attempts** separately.

    Both, because the 60 s throttle means most attempts do not write - so a test about
    M-3 (the advances around the long phases) asserts on *attempts*, while a test about the
    throttle asserts on *writes*. One counter would conflate a throttled heartbeat with an
    absent one, which is the difference between a healthy run and a reaped one.
    """

    attempts: int = 0
    writes: int = 0
    started: int = 0
    write_every: int = 1

    def start(self, session: Any) -> bool:
        self.started += 1
        return self.touch(session, force=True)

    def touch(self, session: Any, run_id: str | None = None, *, force: bool = False) -> bool:
        del session, run_id
        self.attempts += 1
        if force or self.attempts % self.write_every == 0:
            self.writes += 1
            return True
        return False


@dataclass
class AlwaysFires:
    """A stop condition that always fires, registered only in a test.

    Used for the precedence tests, which need two conditions firing at one boundary without
    arranging two real causes. **Never registered in the production registry** - that set is
    closed at five (FR-006.15).
    """

    name: str
    detail: str = "fired"

    def evaluate(self, ctx: StopContext, config: Mapping[str, Any]) -> Verdict:
        del config
        return Verdict(
            condition=self.name,
            triggered=True,
            observed_at_step=ctx.step,
            detail=self.detail,
        )


@dataclass
class NeverFires:
    """A stop condition that never fires. The other half of the no-short-circuit test."""

    name: str

    def evaluate(self, ctx: StopContext, config: Mapping[str, Any]) -> Verdict:
        del config
        return Verdict(
            condition=self.name,
            triggered=False,
            observed_at_step=ctx.step,
            detail="not fired",
        )


@dataclass
class SeedSource:
    """A signal source that publishes accepted seeds  (:class:`AcceptedSeedSource`).

    **No production source implements that Protocol** - F2's self-generation source builds
    its admitted set inside ``next()`` and retains it nowhere, and the batch record carries
    no text (ADR-020). So this double is what makes mode (b)'s verdict filter testable, and
    the gap is reported rather than papered over: without a real publisher the mode refuses
    by name (``refresh_seeds_unavailable``).
    """

    seeds: Sequence[Any] = ()
    asked: list[int] = field(default_factory=list)

    def accepted_seeds(self, limit: int) -> Sequence[Any]:
        self.asked.append(limit)
        return list(self.seeds)[:limit]


def boundary_context(**overrides: Any) -> Any:
    """A :class:`training.boundary.BoundaryContext` wired to these fakes.

    One builder so a test states only what it is about. Everything it does not name is a
    recording fake rather than ``None``, because a ``None`` collaborator silently skips a
    phase and a test about a different phase would then pass for the wrong reason.
    """
    from training.boundary import BoundaryContext

    defaults: dict[str, Any] = {
        "run_id": "run-f6",
        "take_step": FakeStep(),
        "evaluate": FakeEvaluate(),
        "save": FakeSave(),
        "emit": FakeEmit(),
        "heartbeat": FakeHeartbeat(),
        "stop_conditions": {},
        "test_every": 1,
        "retention_every": 1,
        "save_every": None,
        "source": FakeSource(),
    }
    defaults.update(overrides)
    return BoundaryContext(**defaults)


__all__ = [
    "AlwaysFires",
    "FakeEmit",
    "FakeEvaluate",
    "FakeHeartbeat",
    "FakeSave",
    "FakeSource",
    "FakeStep",
    "NeverFires",
    "SeedSource",
    "boundary_context",
]
