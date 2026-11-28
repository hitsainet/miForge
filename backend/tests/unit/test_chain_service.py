"""The chain orchestrator  (FR-006.39 - .46; AC-006.29 - .33; **M-8**, M-20, M-21, M-22).

Everything here drives the **real** ``services.run_launch.launch`` with the validator's
injection points spied, rather than a fake submit. ADR-016: *fixtures whose fields agree by
construction are how a suite stays green over a real bug* - a fake launch that inserted a
run row would agree with these assertions and would never notice that a hop stopped
recording ``chain_id``, or that a stage started without a definition record.

The three halts are tested **in every configuration** (AC-006.31), including the one that
declares the reason, because that is the case the type and the insert-time check cannot
reach: a row written by hand, a migration, or a later edit that widened the type.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
import sqlalchemy as sa

from core.errors import Refusal
from schemas.chain import ChainSpec, StageSpec
from services.chains import service as chain_service
from services.chains.tables import CHAINS, STAGES
from tests.support.db import migrated_engine, session_for
from workers.run_record import RUNS, utcnow

GPU = "GPU-11111111-2222-3333-4444-555555555555"


def stage_definition(**overrides: Any) -> dict[str, Any]:
    """A definition valid in every respect, with **loose parts and no optional service**.

    The standalone case is the default case (FR-006.60), which is where AC20 begins.
    """
    body: dict[str, Any] = {
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
        "stop_conditions": [{"condition": "step_count", "config": {"total": 100}}],
        "domain": "legal",
    }
    body.update(overrides)
    return body


def spec(
    *, stages: int = 3, advance_on: tuple[str, ...] = ("step_count",), **over: Any
) -> ChainSpec:
    built = []
    for index in range(stages):
        definition = stage_definition(**(over if index == 0 else {}))
        if index > 0:
            definition = stage_definition()
        built.append({"definition": definition, "advance_on": list(advance_on)})
    return ChainSpec(name="curriculum", stages=built)


class _Spy:
    def __init__(self, result: Any = None) -> None:
        self.calls: list[tuple[tuple[Any, ...], dict[str, Any]]] = []
        self._result = result

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        self.calls.append((args, kwargs))
        return self._result

    @property
    def count(self) -> int:
        return len(self.calls)


def spies(*, gpu_uuid: str | None = None) -> dict[str, Any]:
    """The validator's four injection points, as the create path passes them through."""

    class _Pack:
        def payload(self) -> dict[str, Any]:
            return {"pack_id": None, "retention_floor": None}

    return {
        "gate": _Spy(result=_Pack()),
        "resolve_parts": _Spy(result={"source": "parts", "parts": []}),
        "budget": _Spy(result={"param_count": 1_200_000_000, "gpu_uuid": gpu_uuid, "mode": "load"}),
        "services": _Spy(result={"needed": [], "midataworks": {"configured": False}}),
    }


@pytest.fixture
def engine(tmp_path: Path):
    built = migrated_engine(tmp_path / "db")
    yield built
    built.dispose()


@pytest.fixture
def logs(tmp_path: Path) -> Path:
    return tmp_path / "logs"


def _set_terminal(
    session: Any, run_id: str, *, status: str = "completed", stop_reason: str | None = "step_count"
) -> None:
    """End a stage's run the way the loop would: a terminal status and a stop reason.

    Through an ``UPDATE`` on the production table so ``terminal_iff_finished`` and
    ``stop_reason_known`` both apply - a hand-built row that skipped them would be a fixture
    agreeing with itself.
    """
    if status == "completed" and stop_reason is not None:
        # THROUGH THE PRODUCTION WRITER. This helper set the column by hand, so the chain's
        # tests passed while `run_state.mark_completed` never wrote `stop_reason` at all and
        # no real chain could advance (fixed 2026-10-17).
        from training import run_state

        if not run_state.mark_completed(session, run_id, stop_reason=stop_reason):
            # Already terminal (e.g. the chain's stop took it there first): the real writer
            # refuses to move a terminal row, and so does this helper now.
            status_now = session.execute(
                sa.select(RUNS.c.status).where(RUNS.c.id == run_id)
            ).scalar_one()
            assert status_now in ("completed", "failed"), status_now
        session.flush()
        return
    session.execute(
        sa.update(RUNS)
        .where(RUNS.c.id == run_id)
        .values(status=status, stop_reason=stop_reason, finished_at=utcnow())
    )
    session.flush()


def _stage(session: Any, chain_id: str, index: int) -> Any:
    return session.execute(
        sa.select(STAGES).where(STAGES.c.chain_id == chain_id, STAGES.c.stage_index == index)
    ).one()


# ============================================================== creation validates it all


class TestCreateValidatesEveryStage:
    def test_a_fourth_stages_invalid_configuration_is_caught_at_creation(
        self, engine, logs
    ) -> None:
        """FR-006.39, and the reason the whole stage plan is durable at creation.

        **Not three hours in.** A curriculum whose fourth stage has no stop condition would
        otherwise run stages one to three - hours of GPU time and three checkpoints - and
        then refuse.
        """
        stages = [
            {"definition": stage_definition(), "advance_on": ["step_count"]} for _ in range(4)
        ]
        stages[3]["definition"] = stage_definition(stop_conditions=[])
        with session_for(engine) as session, pytest.raises(Refusal) as caught:
            chain_service.create(
                session, ChainSpec(name="four", stages=stages), log_root=logs, **spies()
            )
        assert caught.value.code == "chain_stage_invalid"
        bad = caught.value.details["stages"]
        assert [entry["stage_index"] for entry in bad] == [3]
        # FR-006.59: the owning feature's code, unchanged.
        assert any(failure["code"] == "run_has_no_stop_condition" for failure in bad[0]["failures"])

    def test_nothing_is_created_when_a_stage_is_invalid(self, engine, logs) -> None:
        """The refusal leaves no chain, no stage and no run. Asserted on the **state**.

        *A test that only checks the response passes against a create that inserted three
        stages and then raised* - and a chain row with a partial plan is a curriculum
        nothing can state the shape of.
        """
        stages = [
            {"definition": stage_definition(), "advance_on": ["step_count"]} for _ in range(3)
        ]
        stages[2]["definition"] = stage_definition(cadences={"test_every": 0})
        with session_for(engine) as session:
            with pytest.raises(Refusal):
                chain_service.create(
                    session, ChainSpec(name="three", stages=stages), log_root=logs, **spies()
                )
            session.rollback()
        with session_for(engine) as session:
            assert session.execute(sa.select(sa.func.count()).select_from(CHAINS)).scalar() == 0
            assert session.execute(sa.select(sa.func.count()).select_from(STAGES)).scalar() == 0
            assert session.execute(sa.select(sa.func.count()).select_from(RUNS)).scalar() == 0

    def test_a_stage_may_not_name_its_own_run_id(self, engine, logs) -> None:
        """A hop is a branch and the chain mints the id (FR-006.41).

        Two stages naming one ``run_id`` would make the second silently ``UPDATE`` the
        first's row - one run row, two stages, and a lineage that cannot be read.
        """
        stages = [
            {"definition": stage_definition(), "advance_on": ["step_count"]} for _ in range(2)
        ]
        stages[1]["definition"] = stage_definition(run_id="borrowed")
        with session_for(engine) as session, pytest.raises(Refusal) as caught:
            chain_service.create(
                session, ChainSpec(name="two", stages=stages), log_root=logs, **spies()
            )
        assert caught.value.code == "chain_stage_invalid"
        assert caught.value.details["stages"][0]["failures"][0]["field"] == "run_id"

    @pytest.mark.parametrize("mode", ["create", "attach"])
    def test_a_later_stage_must_declare_load(self, engine, logs, mode: str) -> None:
        """R-02.27 *determines* a hop's source, so a later stage cannot choose one.

        Refused rather than silently overridden: replacing a ``create`` source with the
        previous stage's checkpoint would train something the operator did not ask for and
        report success.
        """
        stages = [
            {"definition": stage_definition(), "advance_on": ["step_count"]} for _ in range(2)
        ]
        stages[1]["definition"] = stage_definition(
            model_source={"mode": mode, "config": {"model_type": "llama"}}
        )
        with session_for(engine) as session, pytest.raises(Refusal) as caught:
            chain_service.create(
                session, ChainSpec(name="two", stages=stages), log_root=logs, **spies()
            )
        assert caught.value.code == "chain_stage_invalid"
        assert caught.value.details["stages"][0]["failures"][0]["field"] == "model_source.mode"

    def test_the_insert_time_check_refuses_a_forbidden_advance_reason(self, engine, logs) -> None:
        """**MC-14's regression**, and it is FR-006.43's *second* layer.

        ``StageSpec.advance_on`` is a ``Literal``, so no spec built through the type can
        carry ``retention_floor`` - which is why the call to
        :func:`schemas.chain.assert_advanceable` in ``create`` survived a mutation sweep
        with every test still green. The layer exists for the case the type cannot see:
        **a configuration that arrived as plain strings** - from JSON, from a migration,
        from a later edit that widened the type - which is what ``model_construct``
        reproduces here by skipping validation exactly as those routes do.

        Without this test the second layer is a call nothing depends on, and FR-006.43's
        *"three layers"* would be two.
        """
        stage = StageSpec.model_construct(
            definition=stage_definition(), advance_on=("step_count", "retention_floor")
        )
        chain = ChainSpec.model_construct(
            name="widened",
            stages=(stage, StageSpec(definition=stage_definition(), advance_on=("step_count",))),
        )
        with session_for(engine) as session:
            with pytest.raises(Refusal) as caught:
                chain_service.create(session, chain, log_root=logs, **spies())
            session.rollback()
        assert caught.value.code == "chain_stage_invalid"
        assert caught.value.details["offenders"] == ["retention_floor"]
        with session_for(engine) as session:
            assert session.execute(sa.select(sa.func.count()).select_from(CHAINS)).scalar() == 0

    def test_a_chain_of_one_is_refused_by_the_type(self) -> None:
        """``ChainSpec.stages`` has ``min_length=2``: a chain of one is a run.

        Accepting one would make the chain machinery the path a single run took, which is
        the opposite of FR-006.46's deletability property.
        """
        from pydantic import ValidationError

        with pytest.raises(ValidationError):
            ChainSpec(
                name="one",
                stages=[{"definition": stage_definition(), "advance_on": ["step_count"]}],
            )


class TestCreateStartsStageZeroAndOnlyStageZero:
    def test_stage_zero_runs_and_the_rest_are_recorded_unstarted(self, engine, logs) -> None:
        with session_for(engine) as session:
            payload = chain_service.create(session, spec(), log_root=logs, **spies())
            chain_id = payload["chain_id"]
        with session_for(engine) as session:
            rows = session.execute(
                sa.select(STAGES)
                .where(STAGES.c.chain_id == chain_id)
                .order_by(STAGES.c.stage_index)
            ).all()
            assert len(rows) == 3
            assert rows[0].run_id is not None and rows[0].started_at is not None
            # `started_iff_run`: an unstarted stage has NEITHER, and the plan is still durable.
            assert [row.run_id for row in rows[1:]] == [None, None]
            assert [row.started_at for row in rows[1:]] == [None, None]
            assert all(row.definition for row in rows)
            assert session.execute(sa.select(sa.func.count()).select_from(RUNS)).scalar() == 1

    def test_the_stage_run_records_its_place_in_the_curriculum(self, engine, logs) -> None:
        """``mf_runs.chain_id`` and ``stage_index``, which ``chain_membership_is_whole``
        requires to move together."""
        with session_for(engine) as session:
            payload = chain_service.create(session, spec(), log_root=logs, **spies())
            chain_id = payload["chain_id"]
        with session_for(engine) as session:
            run = session.execute(sa.select(RUNS)).one()
            assert run.chain_id == chain_id
            assert run.stage_index == 0
            # R-02.53's record, written by the real launch rather than by this test.
            assert run.definition_sha256 is not None
            assert run.status == "pending"

    def test_the_chain_holds_the_lease_not_the_stage(self, engine, logs) -> None:
        """FR-006.44, and the holder is **the chain**.

        A stage lease would be refused by the chain's own, and a chain whose card was held
        by its first run would lose it the moment that run ended.
        """
        from tests.support.jobs import read_leases

        with session_for(engine) as session:
            payload = chain_service.create(session, spec(), log_root=logs, **spies(gpu_uuid=GPU))
            chain_id = payload["chain_id"]
        with session_for(engine) as session:
            rows = read_leases(session, GPU)
            assert [row["holder_run_id"] for row in rows] == [f"chain:{chain_id}"]
            assert len(rows) == 1, "a stage must not take a second lease on the chain's card"
            chain = session.execute(sa.select(CHAINS)).one()
            assert chain.gpu_uuid == GPU


# ============================================================== the advance filter, live


class TestTheThreeThatNeverAdvance:
    """AC-006.31, **M-8**: in every configuration, including one that declares the reason."""

    @pytest.mark.parametrize(
        ("status", "stop_reason", "expected"),
        [
            ("completed", "retention_floor", "chain_halted_on_floor_breach"),
            ("cancelled", "operator_stop", "chain_halted_on_operator_stop"),
            ("failed", None, "chain_halted_on_failed_stage"),
            # A run that completed and recorded no reason. A stage that cannot say why it
            # stopped cannot be the basis for starting another.
            ("completed", None, "chain_halted_on_failed_stage"),
        ],
    )
    @pytest.mark.parametrize(
        "advance_on",
        [
            ("step_count",),
            # THE CASE THE OTHER TWO LAYERS CANNOT REACH: a row declaring the reason, as a
            # hand edit or a widened type would write it.
            ("step_count", "retention_floor", "operator_stop", "failed"),
        ],
    )
    def test_each_halts_the_chain_in_every_configuration(
        self, engine, logs, status: str, stop_reason: str | None, expected: str, advance_on
    ) -> None:
        with session_for(engine) as session:
            chain_id = chain_service.create(session, spec(), log_root=logs, **spies())["chain_id"]
            # Written straight to the row, which is how a widened type or a hand edit would
            # arrive - `StageSpec` would refuse these at the edge.
            session.execute(
                sa.update(STAGES)
                .where(STAGES.c.chain_id == chain_id)
                .values(advance_on=list(advance_on))
            )
            run_id = str(_stage(session, chain_id, 0).run_id)
            _set_terminal(session, run_id, status=status, stop_reason=stop_reason)
            outcome = chain_service.advance_one(session, chain_id, log_root=logs, **spies())

        assert outcome["action"] == "halted"
        assert outcome["reason"] == expected
        with session_for(engine) as session:
            chain = session.execute(sa.select(CHAINS)).one()
            assert chain.status == "halted"
            assert chain.halt_code == expected
            assert chain.halt_detail
            assert chain.finished_at is not None
            # Halted, never completed: a curriculum that stopped because a stage hit its
            # floor did not finish.
            assert chain.status != "completed"
            # And no second stage was started.
            assert _stage(session, chain_id, 1).run_id is None

    def test_a_reason_the_stage_did_not_declare_halts_by_name(self, engine, logs) -> None:
        """E-15: a stop reason outside the advance set halts ``chain_not_advanceable``."""
        with session_for(engine) as session:
            chain_id = chain_service.create(
                session, spec(advance_on=("test_threshold",)), log_root=logs, **spies()
            )["chain_id"]
            run_id = str(_stage(session, chain_id, 0).run_id)
            _set_terminal(session, run_id, stop_reason="loss_plateau")
            outcome = chain_service.advance_one(session, chain_id, log_root=logs, **spies())
        assert outcome["reason"] == "chain_not_advanceable"
        assert outcome["stop_reason"] == "loss_plateau"
        assert outcome["advance_on"] == ["test_threshold"]


class TestAStageStillRunningIsLeftAlone:
    def test_a_pending_or_running_stage_is_not_advanced(self, engine, logs) -> None:
        with session_for(engine) as session:
            chain_id = chain_service.create(session, spec(), log_root=logs, **spies())["chain_id"]
            outcome = chain_service.advance_one(session, chain_id, log_root=logs, **spies())
        assert outcome["action"] == "none"
        assert outcome["reason"] == "stage_running"
        with session_for(engine) as session:
            assert _stage(session, chain_id, 1).run_id is None
            assert session.execute(sa.select(CHAINS)).one().status == "running"
