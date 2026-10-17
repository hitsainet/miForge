"""THE COMMIT BRACKET  (R-02.52; FR-011.53 - FR-011.56; BRD-01 R-01.21, R-01.22).

**Why this module exists at all.** Everywhere else in miForge an update is made atomic by
the filesystem: temp directory, ``fsync``, ``rename`` (ADR-015). Here the parameters are in
another process's GPU memory and a multi-tensor write to them is **not atomic** - it writes
tensor by tensor. BRD-01 R-01.8 gives quiesce a hold timeout so a hung trainer cannot take
serving down, and a self-resume partway through a write leaves miLLM serving a parameter
set whose layers sit at two different steps: **a model that was never tested, has no
checkpoint and cannot be reproduced.** The timeout that protects availability would
silently destroy correctness. The bracket separates the two waits: ``commit-begin``
**suspends** the hold timeout (R-01.21) and starts a separate, longer watchdog (R-01.22)
which **fails closed** - ``503 WEIGHTS_TORN``, never a served answer.

## It implements a requirement; it does not invent one

R-02.52 defines the §5.10 step and this bracket. **It had to be added**: R-02.12's rewrite
note said the four protocol stages *"move to §5.10, which keeps them verbatim"* and §5.10
did not contain them, so for a time nothing in BRD-02 obliged miForge to open a bracket at
all. The endpoints, the hold-timeout suspension and the watchdog stay BRD-01 R-01.21 and
R-01.22's, cited and never restated.

## The seven stages, and where they are

FR-011.53's order is: assemble a batch -> forward and backward -> ``quiesce`` ->
``commit-begin`` -> **apply the update in place** -> ``commit-end`` -> ``resume``.

:func:`hot_update_step` composes it out of F1's own step rather than re-implementing it.
**The bracket is applied by wrapping the optimizer**, not by copying F1's three stages: the
only thing that writes a resident tensor is ``optimizer.step()``, so
:class:`BracketedOptimizer` opens the bracket around exactly that call and nothing else.
The gradient-norm computation, ``zero_grad`` and the whole forward and backward stay
**outside the quiesce** - which is the entire reason the hold is tens of milliseconds
rather than seconds. A copy of ``run_one`` here would have been a second definition of the
step, diverging from F1's the first time either changed.

## commit-end on every exit - except a write that failed partway

``commit-end`` is sent on every **successful** write, **before** ``resume``: the set must be
``clean`` before serving is released. **MUT-1** (write before ``commit-begin``) and **MUT-2**
(drop the commit-end after a successful write) are its controls.

**One exit withholds it, by operator decision (2026-10-17): the write itself raising
partway.** Until then the bracket sent ``commit-end`` on that path too - which told miLLM
the set was ``clean`` when some tensors held the new step and others the old one, so miLLM
would serve a model that was never trained. Now the commit is left open, miLLM's commit
watchdog (BRD-01 R-01.22, 200 ms) declares the set ``torn`` and refuses serving, and
:class:`AttachedStepper` waits for that and recovers (rollback or stop - never a late
commit-end, which would bless the half-written set). Pinned through ``drive_run`` by
``tests/integration/test_hotupdate_write_failure.py``; restoring the always-commit-end line
is its negative control.

## It owns no decision about retry

On an interrupted commit it records and **raises**; ``recovery.py`` decides. A bracket that
could retry is a bracket that can re-apply a non-idempotent AdamW update to a half-written
parameter set, which **compounds the damage rather than correcting it** (R-02.31). There is
also **no cancellation check inside** - ADR-007's boundary is outside the bracket, and a
cancellation honoured inside one is a half-applied step and a ``torn`` set.
"""

from __future__ import annotations

import logging
import time
from collections.abc import Iterator
from contextlib import contextmanager
from typing import Any

from core.errors import Refusal, refuse
from training.hotupdate import state
from training.hotupdate.timings import StepTimings

log = logging.getLogger(__name__)

#: The stage order, as data, so the test asserts a **sequence** rather than a result. An
#: inversion is only visible as a sequence: every one of these calls succeeds in the wrong
#: order too.
STAGE_ORDER: tuple[str, ...] = ("quiesce", "commit-begin", "commit-end", "resume")

#: The ops whose refusal, with ``commit-begin`` not yet returned, means nothing was written:
#: the one retryable interruption (BRD AC7, R-02.31).
PRECOMMIT_OPS: frozenset[str] = frozenset({"quiesce", "commit-begin"})


@contextmanager
def commit_bracket(
    client: Any, timings: StepTimings, *, what: str = "optimizer update"
) -> Iterator[None]:
    """Bracket one write to miLLM's resident tensors. **Nothing else goes in here.**

    Args:
        client: the one outbound call site.
        timings: receives the hold and the commit duration for this step.
        what: what is being written, for the log. Two writers use this bracket - F1's
            optimizer update and FR-011.61's resident merge - **on identical terms**.

    Raises:
        Refusal: ``weights_commit_interrupted`` when ``commit-end`` does not succeed, and
            ``weights_write_failed`` when the write itself raised - in which case
            ``commit-end`` is **withheld** (operator decision 2026-10-17) so miLLM's watchdog
            marks the set ``torn``. Either way the step is **abandoned, never retried**
            (R-02.31) and R-02.30's recovery decides.
    """
    quiesce = client.call("quiesce")
    t_quiesce = time.monotonic()
    in_flight: BaseException | None = None
    #: Set when the write itself raised: the commit is deliberately left open, and neither
    #: commit-end nor resume is sent (miLLM refuses a resume while the set is not clean).
    write_failed: BaseException | None = None
    try:
        # commit-begin BEFORE the first byte. It is what suspends R-01.8's hold timeout; a
        # write that starts before it can be cut in half by a self-resume, which is the
        # exact defect this bracket exists for. MUT-1 moves the write above this line.
        client.call("commit-begin")
        timings.commits_begun += 1
        t_commit = time.monotonic()
        try:
            yield
        except BaseException as exc:
            # THE WRITE ITSELF FAILED PARTWAY. ``commit-end`` IS WITHHELD (operator decision,
            # 2026-10-17). Some resident tensors may hold the new step and others the old
            # one; a commit-end would tell miLLM the set is ``clean`` and it would serve a
            # model that was never trained. Left open, miLLM's commit watchdog (BRD-01
            # R-01.22; 200 ms since miLLM ``db44fea``) declares the set ``torn`` and serving
            # is refused - the honest state - and R-02.30's recovery takes over (rollback or
            # stop; completion is not offered, because completing would bless the
            # half-written set). Every OTHER exit still closes the bracket with commit-end.
            write_failed = exc
            in_flight = exc
            log.error(
                "the write failed partway; commit-end WITHHELD so the set goes torn",
                extra={
                    "code": "weights_write_failed",
                    "run_id": timings.run_id,
                    "what": what,
                    "error": f"{type(exc).__name__}: {exc}",
                },
            )
            if isinstance(exc, Exception) and not isinstance(exc, Refusal):
                raise refuse(
                    "weights_write_failed",
                    f"The {what} failed partway ({type(exc).__name__}: {exc}). commit-end was "
                    f"withheld so miLLM's commit watchdog declares the set torn rather than "
                    f"serving a half-written model. THE STEP IS NOT RETRIED; recovery is "
                    f"R-02.30's rollback to the last checkpoint, or a stop.",
                    what=what,
                    commit_left_open=True,
                    retried=False,
                ) from exc
            raise
        else:
            try:
                client.call("commit-end")
            except Refusal as exc:
                log.error(
                    "commit-end failed; miLLM's parameter set may be torn",
                    extra={
                        "code": "weights_commit_interrupted",
                        "run_id": timings.run_id,
                        "what": what,
                        "upstream_refusal": exc.code,
                    },
                )
                interrupted = refuse(
                    "weights_commit_interrupted",
                    "The commit bracket could not be closed: miLLM did not accept "
                    "commit-end, so the parameter set is torn and serving is refused "
                    "until it is clean again. THE STEP IS NOT RETRIED - re-applying a "
                    "non-idempotent optimizer update to a half-written set compounds "
                    "the damage. Recovery is R-02.30's: complete the commit, or load "
                    "the last checkpoint.",
                    what=what,
                    upstream_refusal=exc.code,
                    retried=False,
                )
                in_flight = interrupted
                raise interrupted from exc
            timings.record_commit((time.monotonic() - t_commit) * 1000.0)
    except BaseException as exc:
        in_flight = exc
        raise
    finally:
        # Resume AFTER commit-end, never before: the set must be clean before serving is
        # released. Combining the two `finally` blocks would reverse this on the exception
        # path, which is where it matters.
        #
        # AND `resume` MUST NOT MASK WHAT IS ALREADY PROPAGATING. miLLM refuses a resume
        # unless the set is clean (BRD-01 R-01.7), so a failed commit makes this call fail
        # *correctly* - and an exception raised from a `finally` replaces the one in flight.
        # Without this guard the operator is told "the set is not clean" instead of "the
        # commit was interrupted and the step was abandoned", which is a true sentence
        # standing where the diagnosis belongs. Found by the test for the commit-end
        # failure path, not by reading.
        try:
            if write_failed is None:
                client.call("resume")
        except Refusal as exc:
            if in_flight is None:
                raise
            log.warning(
                "resume was refused while another failure was already propagating",
                extra={
                    "code": "weights_resume_refused",
                    "run_id": timings.run_id,
                    "what": what,
                    "upstream_refusal": exc.code,
                    "propagating": type(in_flight).__name__,
                    "note": (
                        "miLLM refuses a resume unless the set is clean (R-01.7); the "
                        "interruption is the diagnosis and this must not replace it"
                    ),
                },
            )
        timings.record_hold((time.monotonic() - t_quiesce) * 1000.0, quiesce)


class BracketedOptimizer:
    """F1's optimizer, with its ``step()`` inside the bracket and nothing else.

    **Why wrap the optimizer rather than copy the step.** ``training.step.run_one`` is F1's
    and is one function: batch, forward/backward, apply. The only call in it that writes a
    resident tensor is ``optimizer.step()``. Wrapping that gives the bracket *exactly* the
    write - tighter than wrapping "stage 3", since the gradient norm is computed and
    ``zero_grad`` is called outside the hold - and leaves F1's step as the single definition
    of a step. The alternative, a second copy of the three stages inside this package, would
    diverge from F1's the first time either changed, and F1's own ordering tests would not
    cover this path.

    Everything else is delegated, so an optimizer attribute F11 has never heard of still
    works: a trainer-state write reads ``state_dict``, the scheduler reads ``param_groups``.
    """

    def __init__(self, inner: Any, client: Any, timings: StepTimings) -> None:
        self._inner = inner
        self._client = client
        self._timings = timings
        #: How many times the bracket was opened. Asserted by the adapter-read test as
        #: **zero** (AC-011.18): a hash-only assertion passes whenever no gradient step
        #: happens to land on the base, so the count is what bites.
        self.bracketed_steps = 0

    def step(self, *args: Any, **kwargs: Any) -> Any:
        """One update, inside the bracket. The only resident-tensor write F1 makes."""
        with commit_bracket(self._client, self._timings):
            self.bracketed_steps += 1
            return self._inner.step(*args, **kwargs)

    def __getattr__(self, name: str) -> Any:
        # Delegation, not inheritance: the optimizer is torch's and this class must not
        # model it. `__getattr__` is only consulted for attributes this object lacks, so
        # `step` above always wins.
        return getattr(self._inner, name)


def hot_update_step(
    model: Any,
    optimizer: Any,
    batch_source: Any,
    *,
    client: Any,
    timings: StepTimings,
    expected_identity: state.ResidentIdentity | None = None,
    cancel: Any = None,
    objectives: Any = None,
) -> Any:
    """One §5.10 step: FR-011.53's seven stages, in order (R-02.52).

    The precondition first, **before the batch** - assembling a batch against a torn set
    wastes the work and records a batch for a step that cannot run. Then F1's own step,
    with the optimizer bracketed.

    Returns F1's :class:`training.step.StepOutcome`, unchanged. The two §5.10 step-record
    fields come from ``timings.last`` and are put into the block by
    :func:`training.hotupdate.timings.fill`; **the record is written after the bracket
    closes**, because a record written inside is a record of a step that has not finished.
    """
    from training.step import run_one

    # The cancellation boundary is OUTSIDE the bracket (ADR-007, TDD §7). run_one checks it
    # before stage 1; this is the same boundary, and there is none inside the bracket.
    state.require_clean(client, expected_identity)
    return run_one(
        model,
        BracketedOptimizer(optimizer, client, timings),
        batch_source,
        cancel=cancel,
        objectives=objectives,
    )


class AttachedStepper:
    """The §5.10 step as F1's loop calls it: :func:`hot_update_step`, plus this step's timings.

    Supplied on :class:`training.acquisition.base.AcquiredModel` as ``stepper`` by
    :class:`training.hotupdate.attach.AttachAcquirer`, and called by
    ``training/tasks.py``'s ``take_step`` **on its presence** (ADR-022: a value on the
    acquisition, not a ``mode == "attach"`` branch in the core path). Same signature as
    :func:`training.step.run_one`, so the loop needs to know nothing else about it.

    **Why it exists.** Until 2026-10-16 :func:`hot_update_step` had no production caller:
    the loop stepped every acquisition through the run type's plain ``run_one``, so an
    attached ``change`` run wrote miLLM's resident tensors with **no quiesce and no commit
    bracket** - the exact failure R-02.52 and BRD-01 R-01.21/22 exist to make impossible.
    Every F11 test drove :func:`hot_update_step` directly, so the suite was green.

    **The baseline is the saver's, read at call time.** ``save.AttachedSaver.expected`` is
    re-read after every save (a save can rename the resident model), and the module
    docstring of ``save.py`` records that the step must take its baseline from there rather
    than from the frozen snapshot taken at attach. Passing ``saver=None`` (no save path)
    falls back to the attach-time identity.

    **The timings are this step's or ``None`` - never the previous step's.**
    :attr:`StepTimings.last` returns the last value recorded *by any step*, so a step that
    held no bracket (cancelled at the boundary) would otherwise be recorded with its
    predecessor's commit duration: a fabricated sample in the distribution another
    repository's watchdog is set from.
    """

    def __init__(
        self,
        client: Any,
        timings: StepTimings,
        *,
        expected: state.ResidentIdentity | None,
        saver: Any = None,
        precommit_retries: int = 1,
    ) -> None:
        self.client = client
        self.timings = timings
        self._expected = expected
        self._saver = saver
        #: Steps taken through the bracket path. Asserted by the wiring test with its count.
        self.calls = 0
        self.precommit_retries = precommit_retries
        self.precommit_retries_taken = 0
        #: Every R-02.30 recovery taken, as its abandonment record (FR-011.42).
        self.recoveries: list[dict[str, Any]] = []
        #: Writes the observed attachment state into the run record; set by ``bind_run``.
        #: Published at most every ``publish_interval_s`` after a step, and always after a
        #: recovery - the moment the operator most needs the chip to be right.
        self.publish_status: Any = None
        self.publish_interval_s = 15.0
        self._published_at: float | None = None

    @property
    def expected(self) -> state.ResidentIdentity | None:
        saver_expected = getattr(self._saver, "expected", None) if self._saver else None
        return saver_expected if saver_expected is not None else self._expected

    def __call__(
        self,
        model: Any,
        optimizer: Any,
        batch_source: Any,
        *,
        cancel: Any = None,
        objectives: Any = None,
        step: int | None = None,
    ) -> Any:
        """One step, with R-02.30/R-02.31's recovery around it.

        * A refusal **before** ``commit-begin`` (quiesce or commit-begin refused, e.g. miLLM's
          hold timeout expired first - BRD AC7): nothing was written, so the step is retried
          up to ``precommit_retries`` times and each timeout is recorded.
        * A refusal **after** ``commit-begin`` is an interrupted commit: the step is
          **abandoned, never retried**, and :mod:`training.hotupdate.recovery` decides -
          COMPLETE (late ``commit-end``, BRD-01 R-01.23a), else ROLLBACK (load the last
          checkpoint, R-01.25) or STOP. Either way the discontinuity is recorded and the
          run ends with a named refusal carrying the record; a resumed run is a ``continue``
          from the step named in it.
        """
        from dataclasses import replace

        from core.errors import Refusal

        retries = 0
        while True:
            holds_before = len(self.timings.holds_ms)
            commits_before = len(self.timings.commits_ms)
            begun_before = self.timings.commits_begun
            self.calls += 1
            try:
                outcome = hot_update_step(
                    model,
                    optimizer,
                    batch_source,
                    client=self.client,
                    timings=self.timings,
                    expected_identity=self.expected,
                    cancel=cancel,
                    objectives=objectives,
                )
            except Refusal as exc:
                begun = self.timings.commits_begun > begun_before
                op = (exc.details or {}).get("op")
                if not begun and op in PRECOMMIT_OPS:
                    if retries < self.precommit_retries:
                        retries += 1
                        self.precommit_retries_taken += 1
                        log.warning(
                            "refused before commit-begin; nothing was written, retrying",
                            extra={
                                "code": "hotupdate_precommit_retry",
                                "run_id": self.timings.run_id,
                                "op": op,
                                "upstream": exc.code,
                                "attempt": retries,
                            },
                        )
                        continue
                    raise
                if begun:
                    self._recover(exc, step=step)
                raise
            hold = self.timings.holds_ms[-1] if len(self.timings.holds_ms) > holds_before else None
            commit = (
                self.timings.commits_ms[-1]
                if len(self.timings.commits_ms) > commits_before
                else None
            )
            self._publish(force=False)
            return replace(outcome, quiesce_hold_ms=hold, commit_duration_ms=commit)

    def _publish(self, *, force: bool) -> None:
        if self.publish_status is None:
            return
        now = time.monotonic()
        if (
            not force
            and self._published_at is not None
            and now - self._published_at < self.publish_interval_s
        ):
            return
        self._published_at = now
        try:
            self.publish_status(state.read_status(self.client))
        except Exception as exc:
            log.warning(
                "could not record the attachment state",
                extra={"code": "hotupdate_status_write_failed", "error": str(exc)},
            )

    def _await_torn(self) -> bool:
        """Poll miLLM's status until its commit watchdog declares the set ``torn``.

        Bounded: five watchdog periods and at least one second. Returns whether ``torn``
        was observed - the caller does not roll back against a set still ``writing``.
        """
        deadline = time.monotonic() + max(1.0, 5 * self.timings.watchdog_ms / 1000.0)
        while True:
            try:
                weights_state, _ = state.read(self.client)
            except Exception:  # a failed read is "not observed", not a crash
                weights_state = None
            if weights_state == state.TORN:
                return True
            if time.monotonic() >= deadline:
                return False
            time.sleep(min(0.02, max(0.001, self.timings.watchdog_ms / 10000.0)))

    def _recover(self, cause: Any, *, step: int | None) -> None:
        """R-02.30 after an interrupted commit. Always raises; the step is never retried."""
        from training.hotupdate import recovery

        saved = getattr(self._saver, "last_saved", None) if self._saver else None
        interruption = recovery.Interruption(
            same_process=True,
            holds_update=True,
            attachment_id=getattr(self.client, "attachment_id", None),
            commit_begun=True,
            last_checkpoint_step=saved[0] if saved else None,
        )
        write_failed = getattr(cause, "code", None) == "weights_write_failed"
        torn_observed: bool | None = None
        if write_failed:
            # commit-end was WITHHELD (the write failed partway). Completion is not offered:
            # a late commit-end would bless a half-written set. Wait for miLLM's watchdog to
            # declare it torn - the state in which a rollback load is the documented
            # recovery (R-01.23b, R-01.25) - then roll back or stop.
            interruption.holds_update = False
            torn_observed = self._await_torn()
        decided = recovery.plan(interruption)
        if write_failed and not torn_observed:
            decided = recovery.RecoveryPlan(
                outcome=recovery.Recovery.STOP,
                reason=(
                    "the write failed partway and commit-end was withheld, but the serving "
                    "process never reported the set torn within the wait - its commit "
                    "watchdog did not fire as expected, so no recovery is attempted against a "
                    "set whose state is unknown"
                ),
                unavailable_conditions=("watchdog_declared_torn",),
                operator_action=(
                    "check miLLM's WEIGHTS_COMMIT_WATCHDOG_MS and /api/weights/status; load a "
                    "known checkpoint into the serving process once the set is torn"
                ),
            )
        if decided.outcome is recovery.Recovery.COMPLETE:
            try:
                recovery.complete_the_commit(self.client, self.timings)
            except Refusal:
                # Completion was an optimisation; fall through to rollback without it.
                interruption.holds_update = False
                decided = recovery.plan(interruption)
        if decided.outcome is recovery.Recovery.ROLLBACK and saved is not None:
            self.client.call("load", {"name": saved[1]})
        record = recovery.abandonment_record(
            run_id=self.timings.run_id or "",
            step=-1 if step is None else int(step),
            plan_taken=decided,
            cause=getattr(cause, "code", type(cause).__name__),
        )
        record["watchdog_declared_torn"] = torn_observed
        self.recoveries.append(record)
        self._publish(force=True)
        log.error(
            "commit interrupted; step abandoned and recovered",
            extra={"code": "hotupdate_recovery", **record},
        )
        raise refuse(
            "hotupdate_step_abandoned",
            f"The commit was interrupted ({record['cause']}). The step is abandoned and NOT "
            f"retried (R-02.31); recovery took the {decided.outcome.value} path: "
            f"{decided.reason}. Continue the run from step {decided.resume_from_step} "
            f"if one is named."
            + (f" Operator action: {decided.operator_action}" if decided.operator_action else ""),
            **record,
        ) from cause
