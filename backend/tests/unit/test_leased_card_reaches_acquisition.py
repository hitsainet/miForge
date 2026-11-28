"""The card the run LEASED is the card acquisition is told to use.  (ADR-008; ADR-016 #1)

## The wiring, and why it is load-bearing

``mf_runs`` carries two GPU columns and they are **not interchangeable**:

* ``gpu_request`` - what was *asked for* at submit. Usually the literal ``"auto"``, which is
  a request to **choose**.
* ``gpu_uuid`` - the card that was actually *leased*, i.e. the choice, already made and
  recorded, minutes before any worker runs.

``training.run_driver.drive_recorded_run`` reads the **second** and hands it to
:class:`training.acquisition.base.AcquisitionSpec` as ``gpu_request``. Before 2026-10-16 it
handed over ``definition.gpu_request`` - the submitted value - so acquisition was told
``"auto"`` for a run whose card had already been decided.

**What that costs.** The placement layer must not guess: on a multi-card host an ``"auto"``
spec is refused as ``gpu_not_pinned``, because taking index 0 would put two runs on one card
with *both* leases claiming exclusivity and both runs' numbers wrong. ADR-008's recorded
incident is exactly this shape - an inverted index, nothing failing, the work simply running
on the wrong device. And before placement existed to refuse at all, nothing in
``backend/src/`` moved a model onto a card, so a run held a lease and trained on the CPU in
silence.

## Why this file exists rather than a grep

ADR-016 #1: *a capability is not shipped until a test FAILS when its wiring is removed* - and
**a source-level grep is explicitly the weaker form this project rejects**. It proves a read
*exists*, not that anything depends on it; this repository has a recorded case where a
reachability test written over a source string survived its own mutation because the mutated
branch still contained the string it searched for, leaving 20 tests passing against a branch
that could never fire.

So both cases below drive the **real** ``drive_recorded_run`` over a **real** ``mf_runs`` row
and capture the actual ``AcquisitionSpec`` object it builds, at the seam where it hands that
object downstream. The assertion is on **the payload and the call count**, never on "was
called": a spy that only records a call passes against a call sending the wrong arguments.

## Where the spy sits, and what that does and does not cover

On ``training.tasks.drive_run`` - resolved from ``run_driver``'s imported module object at
call time, which is the single seam every acquisition on this path goes through. It is
deliberately **not** on ``AcquisitionSpec``'s constructor: a constructor spy would record a
spec that was built and prove nothing about it being the one handed on. What it does not
cover is ``drive_run``'s own forwarding of that spec into ``acquire_for_run``, which
``tests/integration/test_acquisition_gpu.py`` exercises on the card and
``test_acquisition_placement_is_wired.py`` covers for the placement half.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from tests.support.jobs import insert_run, memory_engine, session_for
from training import run_driver, tasks
from training.acquisition.base import AcquisitionSpec

RUN = "run-leased-card"

#: A real-shaped card UUID. ADR-008: by UUID, never an index or a bus address.
LEASED = "GPU-7f3c1d2e-4b5a-6c7d-8e9f-0a1b2c3d4e5f"

#: What the operator submitted. ``auto`` is the normal case - the request to choose.
SUBMITTED_DEFINITION: dict[str, Any] = {
    "run_id": RUN,
    "model_source": {"mode": "load", "path": "/data/checkpoints/base"},
    "run_type": "change",
    "run_configuration": {
        "learning_rate": 1e-5,
        "warmup_steps": 2,
        "batch_size": 2,
        "gradient_accumulation": 1,
    },
    "signal_source": {"source": "corpus"},
    "parts": [{"part_id": "part-1", "kind": "corpus"}],
    "cadences": {"test_every": 2, "retention_every": 1, "save_every": 10},
    "stop_conditions": [{"condition": "step_count", "config": {"total": 1}}],
    "domain": "legal",
    "gpu_request": "auto",
}


class _DriveRunSpy:
    """Records every ``drive_run`` call with its keywords, and counts them.

    Returns a real :class:`training.tasks.RunResult` so ``drive_recorded_run`` runs its
    whole terminal path afterwards - mark completed, release the lease, announce. A spy that
    raised instead would leave the rest of the function unexercised and would make this a
    test of the first half of a function rather than of the function.
    """

    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []

    def __call__(self, **kwargs: Any) -> tasks.RunResult:
        self.calls.append(kwargs)
        return tasks.RunResult(
            run_id=kwargs["run_id"],
            steps_completed=1,
            last_step=1,
            acquisition={},
            trainer_states=[],
            checkpoints=[],
            stopped_because="step_count",
        )

    @property
    def count(self) -> int:
        return len(self.calls)

    @property
    def spec(self) -> AcquisitionSpec:
        assert self.count == 1, (
            f"drive_run was called {self.count} time(s); exactly one acquisition per run, "
            f"and a second call would mean a second spec nothing here asserted"
        )
        captured = self.calls[0]["spec"]
        assert isinstance(captured, AcquisitionSpec), (
            f"the driver handed drive_run a {type(captured).__name__}, not an "
            f"AcquisitionSpec; acquisition would refuse or guess"
        )
        return captured


def _seed(session: Any, **row: Any) -> None:
    """One ``mf_runs`` row carrying R-02.53's recorded definition as the index over it.

    No run log is written, so ``run_launch.read_definition`` falls back to these columns -
    which is a supported production path (ADR-004: the log is authoritative, the row is a
    rebuildable index over it) and keeps this a unit test with no filesystem record.
    """
    insert_run(
        session,
        RUN,
        status="pending",
        definition_submitted=dict(SUBMITTED_DEFINITION),
        definition_resolved={"domain": "legal", "seed": 4242},
        definition_sha256="c" * 64,
        **row,
    )


@pytest.fixture
def spy(monkeypatch: pytest.MonkeyPatch) -> _DriveRunSpy:
    probe = _DriveRunSpy()
    # Patched on the `tasks` MODULE, which is how `run_driver` resolves it: the driver does
    # `tasks.drive_run(...)`, looking the attribute up at call time rather than holding an
    # early reference, so this is the production call site and not a parallel one.
    monkeypatch.setattr(tasks, "drive_run", probe)
    return probe


class TestTheLeasedCardIsWhatAcquisitionIsTold:
    def test_the_spec_carries_the_LEASED_uuid_not_the_submitted_request(
        self, spy: _DriveRunSpy, tmp_path: Path
    ) -> None:
        """Delete the ``mf_runs.gpu_uuid`` read in ``drive_recorded_run`` - or revert it to
        ``definition.gpu_request`` - and this goes red.

        The payload is the assertion. A spec carrying ``"auto"`` reaches placement as a run
        whose card was never pinned, and on a multi-card host placement must then refuse
        ``gpu_not_pinned`` rather than take index 0 - so the run that *did* hold a lease
        fails, and the reason names a condition the operator did not create.
        """
        engine = memory_engine()
        with session_for(engine) as session:
            _seed(session, gpu_request="auto", gpu_uuid=LEASED)
            session.commit()
            run_driver.drive_recorded_run(
                RUN, session=session, batch_source=object(), log_root=tmp_path
            )

        spec = spy.spec
        assert spec.gpu_request == LEASED, (
            f"acquisition was told {spec.gpu_request!r}. The lease is on {LEASED!r} and that "
            f"value was on the row all along - this same function reads the column twice "
            f"more to RELEASE the lease"
        )
        # And it is not merely *different* from the submitted value: it is the row's.
        assert spec.gpu_request != SUBMITTED_DEFINITION["gpu_request"]
        # The rest of the spec still comes off the definition, so the fix is not a wholesale
        # replacement of the model source with row state.
        assert spec.path is not None and str(spec.path).endswith("base")
        assert spec.seed == 4242, "the resolved seed, not the submitted null (R-02.47)"

    def test_a_pinned_uuid_SUBMITTED_is_still_overridden_by_the_lease(
        self, spy: _DriveRunSpy, tmp_path: Path
    ) -> None:
        """The lease wins even when the submit named a card, and it has to.

        A request naming card A that was leased card B is a request the lease has already
        answered: the lease is what makes *one run per card* true, and acquisition placing
        on A would put this run on a card another run's lease claims exclusively. Without
        this case the test above would pass against an implementation that simply preferred
        *any* non-``auto`` value, which is a different rule with the same symptom here.
        """
        other = "GPU-00000000-1111-2222-3333-444444444444"
        engine = memory_engine()
        with session_for(engine) as session:
            insert_run(
                session,
                RUN,
                status="pending",
                definition_submitted={**SUBMITTED_DEFINITION, "gpu_request": other},
                definition_resolved={"domain": "legal", "seed": 4242},
                definition_sha256="c" * 64,
                gpu_request=other,
                gpu_uuid=LEASED,
            )
            session.commit()
            run_driver.drive_recorded_run(
                RUN, session=session, batch_source=object(), log_root=tmp_path
            )

        assert spy.spec.gpu_request == LEASED


class TestTheFallbackWhenNoCardWasLeased:
    def test_with_no_lease_on_the_row_the_SUBMITTED_request_is_used(
        self, spy: _DriveRunSpy, tmp_path: Path
    ) -> None:
        """A machine with no card leases none (ADR-023), and the submitted value is then the
        only thing there is to pass - ``"auto"``, which acquisition resolves or refuses by
        name. The fallback is ``or``, so a null column cannot turn into a null request.
        """
        engine = memory_engine()
        with session_for(engine) as session:
            _seed(session, gpu_request="auto", gpu_uuid=None)
            session.commit()
            run_driver.drive_recorded_run(
                RUN, session=session, batch_source=object(), log_root=tmp_path
            )

        spec = spy.spec
        assert spec.gpu_request == "auto"
        assert spec.gpu_request is not None, (
            "a null gpu_request would reach acquisition as an absent choice rather than as "
            "the request to choose that `auto` is"
        )

    def test_the_fallback_is_the_DEFINITIONS_value_rather_than_a_literal(
        self, spy: _DriveRunSpy, tmp_path: Path
    ) -> None:
        """Without this, the case above passes against ``gpu_request="auto"`` hard-coded on
        the fallback branch - which would silently discard a submitted pin on exactly the
        machine that has no lease to override it with."""
        pinned = "GPU-99999999-8888-7777-6666-555555555555"
        engine = memory_engine()
        with session_for(engine) as session:
            insert_run(
                session,
                RUN,
                status="pending",
                definition_submitted={**SUBMITTED_DEFINITION, "gpu_request": pinned},
                definition_resolved={"domain": "legal", "seed": 4242},
                definition_sha256="c" * 64,
                gpu_request=pinned,
                gpu_uuid=None,
            )
            session.commit()
            run_driver.drive_recorded_run(
                RUN, session=session, batch_source=object(), log_root=tmp_path
            )

        assert spy.spec.gpu_request == pinned
