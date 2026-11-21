"""A three-stage curriculum, end to end, through the sweep  (AC-006.32; FR-006.39 - .46).

R-02.27's worked example, as the requirement itself gives it: *"corpus first, then
self-generation, then preference pairs."* Three stages, three signal sources, three runs,
one card, one lease - and the whole thing advanced only ever by
``miforge.maintenance.chain_sweep``.

**Nothing is faked but the GPU.** The real validator, the real ``launch``, the real run log,
the real checkpoint writer, the real ``branch_plan``, the real lease. The step itself is not
run - that needs a card (ADR-023) - so each stage is ended by writing a checkpoint and a
terminal row, which is what the loop does at its last boundary. What that leaves untested
here is the step, and the step is F1's.

## Why the sweep is driven rather than the Celery task

``chain_sweep()`` opens ``default_session_scope`` against the configured database and calls
exactly :func:`services.chains.service.sweep` - asserted on the **payload and the call
count** in ``test_chain_wiring.py``, including that what it passes is a *factory* and not a
session. Driving the task here would add a Postgres-and-Redis dependency to this file and
would report a connection failure as a chaining failure.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
import sqlalchemy as sa

from core.errors import Refusal
from schemas.chain import ChainSpec
from services.chains import service as chain_service
from services.chains.tables import CHAINS, STAGES
from tests.support import checkpoints as ckpt
from tests.support.db import migrated_engine, session_for
from tests.unit.test_chain_service import GPU, _set_terminal, _stage, spies, stage_definition
from training import run_log, tasks
from workers.run_record import RUNS

#: R-02.27's own example curriculum. Three stages, three sources - which is also why a hop
#: must be a new ``run_id``: a run records exactly one ``signal_source`` and the manifest's
#: field is a required single enum value, so one run cannot carry two (FR-006.41).
CURRICULUM: tuple[str, ...] = ("corpus", "self_generation", "preference_pairs")


@pytest.fixture
def engine(tmp_path: Path):
    built = migrated_engine(tmp_path / "db")
    yield built
    built.dispose()


@pytest.fixture
def roots(tmp_path: Path) -> dict[str, Path]:
    return {"logs": tmp_path / "logs", "checkpoints": tmp_path / "ck"}


def _spec(advance_on: tuple[str, ...] = ("step_count",)) -> ChainSpec:
    return ChainSpec(
        name="bias-through-a-curriculum",
        stages=[
            {
                "definition": stage_definition(signal_source={"source": source}),
                "advance_on": list(advance_on),
            }
            for source in CURRICULUM
        ],
    )


def _scope(engine: Any) -> Any:
    def scope() -> Any:
        return session_for(engine)

    return scope


def _finish_stage(engine: Any, chain_id: str, index: int, roots: dict[str, Path]) -> str:
    """End the stage the way the loop's last boundary does: a save, then a terminal row."""
    with session_for(engine) as session:
        run_id = str(_stage(session, chain_id, index).run_id)
        ckpt.write_chain(roots["checkpoints"], run_id=run_id, steps=(100, 200))
        _set_terminal(session, run_id, stop_reason="step_count")
    return run_id


class TestAThreeStageChainCompletes:
    def test_the_whole_curriculum_runs_and_the_chain_completes(self, engine, roots) -> None:
        with session_for(engine) as session:
            chain_id = str(
                chain_service.create(
                    session, _spec(), log_root=roots["logs"], **spies(gpu_uuid=GPU)
                )["chain_id"]
            )

        run_ids = [_finish_stage(engine, chain_id, 0, roots)]
        for index in (1, 2):
            report = chain_service.sweep(
                _scope(engine),
                log_root=roots["logs"],
                checkpoint_root=roots["checkpoints"],
                **spies(gpu_uuid=GPU),
            )
            assert report["advanced"] == 1, report
            run_ids.append(_finish_stage(engine, chain_id, index, roots))

        final = chain_service.sweep(
            _scope(engine),
            log_root=roots["logs"],
            checkpoint_root=roots["checkpoints"],
            **spies(gpu_uuid=GPU),
        )
        assert final["completed"] == 1

        with session_for(engine) as session:
            chain = session.execute(sa.select(CHAINS)).one()
            assert chain.status == "completed"
            assert chain.halt_code is None
            assert chain.finished_at is not None
            assert chain.current_stage == 2

            # Three runs, three ids, each recording its place. R-02.27's curriculum is a
            # sequence of BRANCHES, necessarily - a run records one `signal_source`.
            assert len(set(run_ids)) == 3
            rows = session.execute(
                sa.select(RUNS).where(RUNS.c.chain_id == chain_id).order_by(RUNS.c.stage_index)
            ).all()
            assert [row.stage_index for row in rows] == [0, 1, 2]
            assert [row.id for row in rows] == run_ids

            # Every stage carries its own recorded definition (R-02.53), and the three
            # sources are the three the curriculum asked for - not one repeated.
            sources = [
                (
                    row.definition_resolved["source"]["source"]
                    if "source" in (row.definition_resolved or {})
                    else row.definition_submitted["signal_source"]["source"]
                )
                for row in rows
            ]
            assert sources == list(CURRICULUM)

            # Each stage's row is closed out in order.
            stages = session.execute(
                sa.select(STAGES)
                .where(STAGES.c.chain_id == chain_id)
                .order_by(STAGES.c.stage_index)
            ).all()
            assert all(stage.started_at is not None for stage in stages)
            assert all(stage.finished_at is not None for stage in stages)

    def test_every_stage_has_a_durable_definition_block_in_its_own_run_log(
        self, engine, roots
    ) -> None:
        """FR-006.61: the run log says what each stage was asked to do.

        A chain's record of a stage is ``mf_chain_stages.definition``; the **run's** record
        is its own log's ``definition`` block, written before the enqueue. Both exist, and
        this asserts the second, because a stage whose run log cannot state its shape is a
        stage an operator cannot reproduce from the artifact.
        """
        with session_for(engine) as session:
            chain_id = str(
                chain_service.create(session, _spec(), log_root=roots["logs"], **spies())[
                    "chain_id"
                ]
            )
        _finish_stage(engine, chain_id, 0, roots)
        chain_service.sweep(
            _scope(engine),
            log_root=roots["logs"],
            checkpoint_root=roots["checkpoints"],
            **spies(),
        )
        with session_for(engine) as session:
            for index in (0, 1):
                run_id = str(_stage(session, chain_id, index).run_id)
                path = run_log.run_log_path(tasks.run_log_root(roots["logs"]), run_id)
                assert path.is_file(), f"stage {index} wrote no run log"
                blocks = list(run_log.read_blocks(path, block="definition"))
                assert len(blocks) == 1
                assert blocks[0]["definition_sha256"]

    def test_the_curriculum_holds_one_lease_for_its_whole_duration(self, engine, roots) -> None:
        """FR-006.44, over three stages. **One row, never released, never re-acquired.**"""
        from tests.support.jobs import read_leases

        with session_for(engine) as session:
            chain_id = str(
                chain_service.create(
                    session, _spec(), log_root=roots["logs"], **spies(gpu_uuid=GPU)
                )["chain_id"]
            )
        for index in (0, 1):
            _finish_stage(engine, chain_id, index, roots)
            chain_service.sweep(
                _scope(engine),
                log_root=roots["logs"],
                checkpoint_root=roots["checkpoints"],
                **spies(gpu_uuid=GPU),
            )
        with session_for(engine) as session:
            rows = read_leases(session, GPU)
            assert len(rows) == 1
            assert rows[0]["holder_run_id"] == f"chain:{chain_id}"
            assert rows[0]["released_at"] is None


class TestAFloorBreachMidCurriculumHaltsIt:
    def test_the_second_stage_breaching_its_floor_stops_the_curriculum(self, engine, roots) -> None:
        """AC-006.31 at curriculum scale, and the risk the PPRD names as a project risk.

        Advancing here would start the third stage **from the checkpoint whose general
        capability had just collapsed** and push it further in the same direction -
        compounding the one failure in the loop whose symptom is invisible on every curve,
        *and the suite would be green for every stage.*
        """
        with session_for(engine) as session:
            chain_id = str(
                chain_service.create(session, _spec(), log_root=roots["logs"], **spies())[
                    "chain_id"
                ]
            )
        _finish_stage(engine, chain_id, 0, roots)
        chain_service.sweep(
            _scope(engine),
            log_root=roots["logs"],
            checkpoint_root=roots["checkpoints"],
            **spies(),
        )
        with session_for(engine) as session:
            second = str(_stage(session, chain_id, 1).run_id)
            ckpt.write_chain(roots["checkpoints"], run_id=second, steps=(100,))
            _set_terminal(session, second, stop_reason="retention_floor")

        report = chain_service.sweep(
            _scope(engine),
            log_root=roots["logs"],
            checkpoint_root=roots["checkpoints"],
            **spies(),
        )
        assert report["halted"] == 1
        with session_for(engine) as session:
            chain = session.execute(sa.select(CHAINS)).one()
            assert chain.status == "halted"
            assert chain.halt_code == "chain_halted_on_floor_breach"
            assert _stage(session, chain_id, 2).run_id is None
            assert session.execute(sa.select(sa.func.count()).select_from(RUNS)).scalar() == 2


class TestStoppingAChain:
    def test_the_running_stage_is_stopped_through_the_single_run_mechanism(
        self, engine, roots
    ) -> None:
        """FR-006.24, AMB-3: **no second flag and no second mechanism.**

        The assertion is on ``mf_runs.cancel_requested_at`` - F1's own column, set by
        ``stop_service.request_stop`` - because a chain stop that set some chain-only flag
        would halt the chain and leave the stage training.
        """
        with session_for(engine) as session:
            chain_id = str(
                chain_service.create(session, _spec(), log_root=roots["logs"], **spies())[
                    "chain_id"
                ]
            )
        with session_for(engine) as session:
            outcome = chain_service.stop_chain(session, chain_id, label="operator:sean")
        assert outcome["action"] == "halted"
        assert outcome["reason"] == "chain_halted_on_operator_stop"
        assert outcome["stage_stop"]["stop_pending"] is True

        with session_for(engine) as session:
            chain = session.execute(sa.select(CHAINS)).one()
            assert chain.status == "halted"
            assert chain.halt_code == "chain_halted_on_operator_stop"
            run = session.execute(sa.select(RUNS)).one()
            assert run.cancel_requested_at is not None

    def test_a_finished_chain_refuses_the_stop(self, engine, roots) -> None:
        """A success for a stop that stopped nothing would tell the operator they had
        stopped a curriculum that had already ended."""
        with session_for(engine) as session:
            chain_id = str(
                chain_service.create(session, _spec(), log_root=roots["logs"], **spies())[
                    "chain_id"
                ]
            )
            chain_service.halt(session, chain_id, "chain_no_checkpoint", detail="for the test")
        with session_for(engine) as session, pytest.raises(Refusal) as caught:
            chain_service.stop_chain(session, chain_id)
        assert caught.value.code == "chain_not_stoppable"

    def test_a_sweep_does_not_resurrect_a_halted_chain(self, engine, roots) -> None:
        """``halted`` is terminal, and the sweep must leave it alone.

        A sweep that advanced a halted chain would undo an operator's stop one interval
        later, which is the same class of wrong as advancing past a floor breach.
        """
        with session_for(engine) as session:
            chain_id = str(
                chain_service.create(session, _spec(), log_root=roots["logs"], **spies())[
                    "chain_id"
                ]
            )
        with session_for(engine) as session:
            chain_service.stop_chain(session, chain_id)
        _finish_stage(engine, chain_id, 0, roots)
        report = chain_service.sweep(
            _scope(engine),
            log_root=roots["logs"],
            checkpoint_root=roots["checkpoints"],
            **spies(),
        )
        # A terminal chain is not even a candidate.
        assert report["looked_at"] == 0
        with session_for(engine) as session:
            assert _stage(session, chain_id, 1).run_id is None


class TestTheDetailViewAgreesWithTheSweep:
    def test_the_advance_decision_is_computed_by_the_same_filter(self, engine, roots) -> None:
        """A screen rendering the rule a second way would show *"this advances"* beside a
        chain the sweep had halted."""
        with session_for(engine) as session:
            chain_id = str(
                chain_service.create(
                    session,
                    _spec(advance_on=("test_threshold",)),
                    log_root=roots["logs"],
                    **spies(),
                )["chain_id"]
            )
        with session_for(engine) as session:
            run_id = str(_stage(session, chain_id, 0).run_id)
            _set_terminal(session, run_id, stop_reason="retention_floor")
            payload = chain_service.detail(session, chain_id)

        decision = payload["stages"][0]["advance_decision"]
        assert decision == {
            "stop_reason": "retention_floor",
            "advances": False,
            "halt_code": "chain_halted_on_floor_breach",
        }
        assert payload["stages"][1]["advance_decision"] is None
        assert payload["stage_count"] == 3
