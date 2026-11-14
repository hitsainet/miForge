"""The five conditions, one by one  (FR-006.18 - .25, .27).

The two properties that recur and are each worth more than the arithmetic they guard:

* **``observed_at_step`` is the step the number was taken at**, not the boundary's step. A
  test threshold observed forty steps ago is a different statement from one observed last
  step (US-8), and a view that showed only the number would render a stale figure as live.
* **nothing in F6 compares a score with a floor.** ``retention_floor`` fires on the
  ``breached`` column F4 wrote; ``test_stop_no_floor_comparison.py`` asserts the absence
  structurally.
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

from core.errors import Refusal
from tests.support.jobs import insert_run, memory_engine, session_for
from training import stop
from training.stop import loss_plateau, precedence
from training.stop.base import StopContext, Verdict
from training.stop.retention_floor import exposure_steps

RUN = "run-stop"


@pytest.fixture
def scored():
    """A run with real ``mf_scores`` rows, written through **F4's own recorder**.

    Not a hand-built insert: *fixtures whose fields agree by construction are how a suite
    stays green over a real bug*, and the fields this reads - ``breached``, ``n``,
    ``provider``, ``aggregation`` - are exactly the ones a hand-built row would get wrong
    in the same direction as the code.
    """
    from services.scoring import tables as scoring_tables

    engine = memory_engine()
    scoring_tables.metadata.create_all(engine)
    with session_for(engine) as session:
        insert_run(session, RUN, status="running")
        session.commit()
    return engine


def _write_score(engine, **values) -> None:
    """One score row through the production table object, with F4's own defaults."""
    import sqlalchemy as sa

    from services.scoring.tables import SCORES
    from workers.run_record import utcnow

    row = {
        "run_id": RUN,
        "step": 10,
        "kind": "test",
        "set_rendered": "test/legal@1",
        "score": 0.5,
        "n": 32,
        "floor": None,
        "floor_layer": "none",
        "breached": False,
        "provider": "in_process:run-stop@10",
        "provider_kind": "in_process",
        "provider_model": "run-stop",
        "aggregation": "mean",
        "decode": {"temperature": 0.0},
        "reproducible": True,
        "out_of_cadence": False,
        "created_at": utcnow(),
    }
    row.update(values)
    with session_for(engine) as session:
        session.execute(sa.insert(SCORES).values(**row))
        session.commit()


# ------------------------------------------------------------------------- step_count


class TestStepCount:
    def test_it_fires_when_the_completed_count_reaches_the_total(self) -> None:
        condition = stop.resolve("step_count")
        assert condition.evaluate(StopContext(RUN, 99, 0), {"total": 100}).triggered is False
        assert condition.evaluate(StopContext(RUN, 100, 0), {"total": 100}).triggered is True
        assert condition.evaluate(StopContext(RUN, 101, 0), {"total": 100}).triggered is True

    def test_it_names_its_headroom(self) -> None:
        """US-8: the view shows how close the run is, not only whether it is there."""
        verdict = stop.resolve("step_count").evaluate(StopContext(RUN, 40, 0), {"total": 100})
        assert verdict.values["headroom"] == 60
        assert "40 of 100" in (verdict.detail or "")

    def test_a_missing_total_is_refused_naming_the_parameter(self) -> None:
        with pytest.raises(Refusal) as caught:
            stop.resolve("step_count").evaluate(StopContext(RUN, 1, 0), {})
        assert caught.value.code == "stop_condition_config_incomplete"
        assert caught.value.details["parameter"] == "total"

    @pytest.mark.parametrize("total", [0, -5, 1.5, "100", True])
    def test_a_total_that_is_not_a_positive_integer_is_refused(self, total) -> None:
        """``True`` is in the list on purpose: ``isinstance(True, int)`` is true in Python,
        so a bare ``isinstance`` check would accept a boolean as a step budget of one."""
        with pytest.raises(Refusal):
            stop.resolve("step_count").evaluate(StopContext(RUN, 1, 0), {"total": total})

    def test_it_reads_the_loops_counter_and_not_a_row_count(self) -> None:
        """FR-006.18: ``mf_steps`` is a rebuildable index, so a row count could be short by
        one for a step whose log line is durable - and a run that stopped at a different
        step after a reindex would be a run whose stopping point depended on a cache."""
        import ast
        from pathlib import Path

        source = (
            Path(__file__).resolve().parents[3]
            / "backend"
            / "src"
            / "training"
            / "stop"
            / "step_count.py"
        ).read_text(encoding="utf-8")
        for node in ast.walk(ast.parse(source)):
            if isinstance(node, (ast.Import, ast.ImportFrom)):
                names = [a.name for a in node.names]
                if isinstance(node, ast.ImportFrom) and node.module:
                    names.append(node.module)
                assert not any("sqlalchemy" in n or n.endswith("tables") for n in names)


# ---------------------------------------------------------------------- test_threshold


class TestTestThreshold:
    def test_it_fires_at_or_above_the_threshold(self, scored) -> None:
        """``>=``: the operator's statement is "stop when it is this good", and a score
        landing exactly on the target has met it."""
        _write_score(scored, score=0.80, step=10)
        with session_for(scored) as session:
            ctx = StopContext(RUN, 12, 0, session=session)
            assert stop.resolve("test_threshold").evaluate(ctx, {"threshold": 0.80}).triggered
            assert not stop.resolve("test_threshold").evaluate(ctx, {"threshold": 0.81}).triggered

    def test_the_verdict_carries_F4s_n_provider_and_aggregation(self, scored) -> None:
        """ADR-012 applies hardest to the number that ended a run: "0.84" from a mean over
        320 prompts and "0.84" from an accept rate over 12 are different claims."""
        _write_score(scored, score=0.9, n=320, aggregation="mean", step=10)
        with session_for(scored) as session:
            verdict = stop.resolve("test_threshold").evaluate(
                StopContext(RUN, 10, 0, session=session), {"threshold": 0.5}
            )
        assert verdict.values["n"] == 320
        assert verdict.values["aggregation"] == "mean"
        assert verdict.values["provider"]
        assert "320 prompts" in (verdict.detail or "")

    def test_observed_at_step_is_the_scores_step_not_the_boundarys(self, scored) -> None:
        """**The exposure window made visible.** A threshold met at step 10 and read at
        step 19 must not read as though it were measured at 19."""
        _write_score(scored, score=0.9, step=10)
        with session_for(scored) as session:
            verdict = stop.resolve("test_threshold").evaluate(
                StopContext(RUN, 19, 0, session=session), {"threshold": 0.5}
            )
        assert verdict.observed_at_step == 10

    def test_an_out_of_cadence_score_does_not_end_the_run(self, scored) -> None:
        """``mf_scores.out_of_cadence`` marks an ad-hoc check, and its own column comment is
        *"an ad-hoc check must never silently join the step series"*. **Ending a run** on
        one would be the loudest possible way of joining it: an operator spot-checking a
        run must not thereby stop it."""
        _write_score(scored, score=0.99, step=11, out_of_cadence=True)
        with session_for(scored) as session:
            verdict = stop.resolve("test_threshold").evaluate(
                StopContext(RUN, 11, 0, session=session), {"threshold": 0.5}
            )
        assert verdict.triggered is False
        assert verdict.values["score"] is None

    def test_a_rescore_supersedes_the_earlier_number(self, scored) -> None:
        """A re-score **appends** (F4's index is over ``(run, step, kind, created_at)``), so
        the condition must read the latest - stopping on a figure F4 has already replaced
        would end a run on a number nothing shows any more."""
        from datetime import timedelta

        from workers.run_record import utcnow

        _write_score(scored, score=0.95, step=10, created_at=utcnow())
        _write_score(scored, score=0.10, step=10, created_at=utcnow() + timedelta(seconds=5))
        with session_for(scored) as session:
            verdict = stop.resolve("test_threshold").evaluate(
                StopContext(RUN, 10, 0, session=session), {"threshold": 0.5}
            )
        assert verdict.triggered is False
        assert verdict.values["score"] == pytest.approx(0.10)

    def test_no_score_yet_is_not_a_stop_and_not_a_refusal(self, scored) -> None:
        """Before the first evaluation there is nothing to compare. Reported as ``None``
        rather than rendered as a stale zero."""
        with session_for(scored) as session:
            verdict = stop.resolve("test_threshold").evaluate(
                StopContext(RUN, 1, 0, session=session), {"threshold": 0.5}
            )
        assert verdict.triggered is False
        assert verdict.values["score"] is None

    @pytest.mark.parametrize("threshold", [-0.1, 1.1])
    def test_a_threshold_outside_the_scales_range_is_refused_not_clamped(self, threshold) -> None:
        """``ScoreRecord.score`` is bounded 0..1, so no score F4 can record would meet it.
        Refused rather than clamped: a clamped threshold is a target nobody set."""
        with pytest.raises(Refusal) as caught:
            stop.resolve("test_threshold").evaluate(
                StopContext(RUN, 1, 0), {"threshold": threshold}
            )
        assert caught.value.details["range"] == [0.0, 1.0]


# ---------------------------------------------------------------------- retention_floor


class TestRetentionFloor:
    def test_it_fires_on_F4s_breached_row_and_compares_nothing(self, scored) -> None:
        """**M-1's unit half.** The condition reads a fact F4 recorded at the pass; it does
        not recompute ``score < floor`` against a floor that may since have been edited."""
        _write_score(
            scored,
            kind="retention",
            step=7,
            score=0.60,
            floor=0.72,
            floor_layer="pack_slot",
            breached=True,
        )
        with session_for(scored) as session:
            verdict = stop.resolve("retention_floor").evaluate(
                StopContext(RUN, 7, 0, session=session), {}
            )
        assert verdict.triggered is True
        assert verdict.values["score"] == pytest.approx(0.60)
        assert verdict.values["floor"] == pytest.approx(0.72)
        assert verdict.values["floor_layer"] == "pack_slot"
        assert "pack slot" in (verdict.detail or "")

    def test_a_retention_score_ABOVE_the_floor_does_not_fire(self, scored) -> None:
        _write_score(scored, kind="retention", step=7, score=0.80, floor=0.72, breached=False)
        with session_for(scored) as session:
            verdict = stop.resolve("retention_floor").evaluate(
                StopContext(RUN, 7, 0, session=session), {}
            )
        assert verdict.triggered is False

    def test_equality_is_not_a_breach_and_the_test_exercises_F4s_comparison(self) -> None:
        """E-6, and the test is deliberately against **F4's** function: F6 has no comparison
        to exercise, so a test of equality here would be a test of nothing."""
        from providers.base import ProviderIdentity
        from services.scoring.retention import ResolvedFloor, evaluate_floor

        floor = ResolvedFloor(value=0.72, layer="pack_slot")
        signal = evaluate_floor(
            0.72,
            floor,
            run_id=RUN,
            step=7,
            n=32,
            provider=ProviderIdentity(kind="in_process", model="m"),
            set_ref="retention/general@1",
        )
        assert signal is None, "a run sitting exactly on its floor has not fallen below it"

    def test_observed_at_step_is_the_breachs_step_and_names_the_exposure(self, scored) -> None:
        """FR-006.20's window, made visible: a breach seen at 7 and acted on at 10 cost
        three steps of training against a model already below its floor."""
        _write_score(scored, kind="retention", step=7, score=0.6, floor=0.72, breached=True)
        with session_for(scored) as session:
            verdict = stop.resolve("retention_floor").evaluate(
                StopContext(RUN, 10, 0, session=session), {}
            )
        assert verdict.observed_at_step == 7
        assert verdict.values["exposure_steps"] == 3

    @pytest.mark.parametrize(("cadence", "expected"), [(1, 0), (2, 1), (10, 9), (0, 0), (-3, 0)])
    def test_the_exposure_window_is_retention_every_minus_one_and_ZERO_at_one(
        self, cadence, expected
    ) -> None:
        """FR-006.20's number, in one place so a screen, a refusal and the run log agree.
        **At F4's default of 1 it is 0** - the breach is seen at the boundary it occurred
        on and the loop stops before the next step."""
        assert exposure_steps(cadence) == expected


class TestNoFloorComparisonExistsAnywhereInF6:
    """AC-006.14. *The division is kept by what the code cannot contain.*"""

    #: The comparison banned is an **ordering** one. ``is None`` and ``==`` against a floor
    #: are null checks and equality, neither of which decides a breach - and a scanner that
    #: flagged them would be a guard that is red for the wrong reason, which is worse than
    #: no guard because the next session relaxes it.
    ORDERING = (ast.Lt, ast.LtE, ast.Gt, ast.GtE)

    def _floor_orderings(self, path) -> list[str]:
        """Every ordering comparison in ``path`` with a floor on either side."""
        found: list[str] = []
        for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
            if not isinstance(node, ast.Compare):
                continue
            if not any(isinstance(op, self.ORDERING) for op in node.ops):
                continue
            for operand in [node.left, *node.comparators]:
                if self._names_a_floor(operand):
                    found.append(f"{path.name}:{node.lineno}")
        return found

    @staticmethod
    def _names_a_floor(operand) -> bool:
        if isinstance(operand, ast.Name):
            return operand.id == "floor"
        if isinstance(operand, ast.Attribute):
            return operand.attr == "floor"
        if isinstance(operand, ast.Subscript):
            return isinstance(operand.slice, ast.Constant) and operand.slice.value == "floor"
        if isinstance(operand, ast.Call):
            return any(
                TestNoFloorComparisonExistsAnywhereInF6._names_a_floor(arg) for arg in operand.args
            )
        return False

    def _f6_modules(self) -> list:
        src = Path(__file__).resolve().parents[3] / "backend" / "src"
        return [
            src / "training" / "loop.py",
            src / "training" / "boundary.py",
            src / "training" / "run_state.py",
            src / "training" / "run_driver.py",
            *sorted((src / "training" / "stop").rglob("*.py")),
            *sorted((src / "training" / "refresh").rglob("*.py")),
        ]

    def test_no_F6_module_compares_a_score_with_a_floor(self) -> None:
        offenders = [site for path in self._f6_modules() for site in self._floor_orderings(path)]
        assert not offenders, (
            f"F6 must contain no floor comparison - F4's FR-004.25 owns it, declares "
            f"equality not a breach, and calls that line the most load-bearing in the "
            f"feature. A second comparison is a second answer: {offenders}"
        )

    @pytest.mark.parametrize(
        "source",
        [
            "x = score < floor",
            "x = score <= floor",
            "x = floor > score",
            "x = floor >= score",
            "x = row['score'] < row['floor']",
            "x = signal.score < signal.floor",
            "x = s < float(row['floor'])",
        ],
    )
    def test_the_scanner_can_actually_see_each_form_of_the_comparison(
        self, source: str, tmp_path
    ) -> None:
        """**The self-test, one case per spelling.**

        A scanner that cannot see one form is a guard that is not there. This project has
        shipped two of those: a ``||`` branch printing "clean" from a grep that had
        **errored**, and a ``replace()`` that matched nothing so a **no-op mutation** looked
        exactly like a survivor.
        """
        path = tmp_path / "planted.py"
        path.write_text(source, encoding="utf-8")
        assert self._floor_orderings(path), f"the scanner would not catch {source!r}"

    @pytest.mark.parametrize(
        "source",
        [
            "x = None if row['floor'] is None else float(row['floor'])",
            "x = floor == other",
            "x = floor is not None",
        ],
    )
    def test_the_scanner_does_NOT_flag_a_null_check_or_an_equality(
        self, source: str, tmp_path
    ) -> None:
        """The other half of the self-test, and it earned its place: the first draft of this
        guard flagged ``None if row['floor'] is None else ...`` in
        ``retention_floor.py`` - a null check, not a breach decision. **A guard that is red
        for the wrong reason is worse than no guard**, because the next session relaxes it.
        """
        path = tmp_path / "planted.py"
        path.write_text(source, encoding="utf-8")
        assert not self._floor_orderings(path)


# ------------------------------------------------------------------------ loss_plateau


class TestLossPlateau:
    @pytest.mark.parametrize(
        "missing", ["metric", "window", "min_relative_improvement", "min_steps"]
    )
    def test_each_of_the_four_parameters_is_required_and_refused_BY_NAME(self, missing) -> None:
        """E-3, FR-006.21. **R-02.25 defines none of it** - ``plateau`` occurs twice in
        BRD-02 and neither use parametrises it (DEF-4) - so all four are required and none
        is defaulted. A default here decides when runs end, invisibly."""
        config = {
            "metric": "raw",
            "window": 4,
            "min_relative_improvement": 0.01,
            "min_steps": 10,
        }
        del config[missing]
        with pytest.raises(Refusal) as caught:
            stop.resolve("loss_plateau").evaluate(StopContext(RUN, 20, 0), config)
        assert caught.value.code == "stop_condition_config_incomplete"
        assert caught.value.details["parameter"] == missing

    def test_no_plateau_field_has_a_default_anywhere_in_the_schema(self) -> None:
        """The edge model must not fill one in either. ``StopConditionSpec.config`` is an
        open mapping precisely so no field of it can carry a default."""
        from schemas.run_definition import StopConditionSpec

        spec = StopConditionSpec(condition="loss_plateau")
        assert spec.config == {}

    def test_a_window_below_two_is_refused(self) -> None:
        """A window of one has no improvement to measure - it would compare a number with
        itself and fire on every boundary."""
        with pytest.raises(Refusal):
            stop.resolve("loss_plateau").evaluate(
                StopContext(RUN, 20, 0),
                {
                    "metric": "raw",
                    "window": 1,
                    "min_relative_improvement": 0.01,
                    "min_steps": 10,
                },
            )

    def test_min_steps_below_the_window_is_refused(self) -> None:
        """The condition could never see a full window before it was allowed to fire."""
        with pytest.raises(Refusal) as caught:
            stop.resolve("loss_plateau").evaluate(
                StopContext(RUN, 20, 0),
                {
                    "metric": "raw",
                    "window": 10,
                    "min_relative_improvement": 0.01,
                    "min_steps": 4,
                },
            )
        assert caught.value.details["parameter"] == "min_steps"

    def test_an_unknown_metric_is_refused_rather_than_treated_as_raw(self) -> None:
        """*"A raw per-step loss and a 50-step mean plateau at different steps"*, so
        guessing the metric changes when the run ends."""
        with pytest.raises(Refusal) as caught:
            stop.resolve("loss_plateau").evaluate(
                StopContext(RUN, 20, 0),
                {
                    "metric": "ema",
                    "window": 4,
                    "min_relative_improvement": 0.01,
                    "min_steps": 10,
                },
            )
        assert caught.value.details["parameter"] == "metric"

    def test_it_does_not_fire_before_min_steps_even_on_a_flat_window(self) -> None:
        """Without ``min_steps`` a run can plateau-stop during warmup, before the learning
        rate has arrived."""
        ctx = StopContext(RUN, 5, 0, window_losses=(0.5, 0.5, 0.5, 0.5), window_steps=(2, 3, 4, 5))
        verdict = stop.resolve("loss_plateau").evaluate(
            ctx,
            {
                "metric": "raw",
                "window": 4,
                "min_relative_improvement": 0.01,
                "min_steps": 50,
            },
        )
        assert verdict.triggered is False

    def test_it_does_not_fire_on_a_partial_window(self) -> None:
        ctx = StopContext(RUN, 60, 0, window_losses=(0.5, 0.5), window_steps=(59, 60))
        verdict = stop.resolve("loss_plateau").evaluate(
            ctx,
            {
                "metric": "raw",
                "window": 4,
                "min_relative_improvement": 0.01,
                "min_steps": 10,
            },
        )
        assert verdict.triggered is False

    def test_it_fires_on_a_full_flat_window_past_min_steps(self) -> None:
        ctx = StopContext(
            RUN, 60, 0, window_losses=(0.50, 0.50, 0.50, 0.50), window_steps=(57, 58, 59, 60)
        )
        verdict = stop.resolve("loss_plateau").evaluate(
            ctx,
            {
                "metric": "raw",
                "window": 4,
                "min_relative_improvement": 0.01,
                "min_steps": 10,
            },
        )
        assert verdict.triggered is True
        assert verdict.values["improvement"] == pytest.approx(0.0)

    def test_it_does_NOT_fire_while_the_loss_is_still_improving(self) -> None:
        ctx = StopContext(
            RUN, 60, 0, window_losses=(0.90, 0.70, 0.50, 0.30), window_steps=(57, 58, 59, 60)
        )
        verdict = stop.resolve("loss_plateau").evaluate(
            ctx,
            {
                "metric": "raw",
                "window": 4,
                "min_relative_improvement": 0.01,
                "min_steps": 10,
            },
        )
        assert verdict.triggered is False

    def test_the_improvement_ratio_uses_ABS_of_the_first_loss(self) -> None:
        """**The one ``abs`` that is the difference between a detector and a coin toss.**

        A DPO or reward objective can legitimately produce a negative loss. Without
        ``abs(first)`` the sign of the whole ratio inverts, so a run that was improving
        reads as a plateau and stops.
        """
        improving = loss_plateau.improvement_ratio((-0.2, -0.8))
        assert improving > 0, (
            "a loss falling from -0.2 to -0.8 is an improvement; without abs(first) this "
            "is negative and the condition fires on a run that is learning"
        )

    def test_a_first_loss_of_exactly_zero_yields_no_ratio_and_no_stop(self) -> None:
        """No relative scale exists. Refusing to answer beats dividing by zero and beats
        inventing an epsilon that decides when runs end."""
        assert loss_plateau.improvement_ratio((0.0, 0.0)) is None
        ctx = StopContext(
            RUN, 60, 0, window_losses=(0.0, 0.0, 0.0, 0.0), window_steps=(57, 58, 59, 60)
        )
        assert (
            not stop.resolve("loss_plateau")
            .evaluate(
                ctx,
                {
                    "metric": "raw",
                    "window": 4,
                    "min_relative_improvement": 0.01,
                    "min_steps": 10,
                },
            )
            .triggered
        )

    def test_F3s_health_signals_are_attached_to_a_plateau_stop(self) -> None:
        """FR-006.23. A flat loss on a ``reward`` run can mean the model converged **or**
        that a zero advantage applied numerically nothing for a hundred steps, with the
        same curve - so the stop reason carries what else was true."""
        ctx = StopContext(
            RUN,
            60,
            0,
            window_losses=(0.5, 0.5, 0.5, 0.5),
            window_steps=(57, 58, 59, 60),
            window_health={"grader_scores_have_no_spread": True},
        )
        verdict = stop.resolve("loss_plateau").evaluate(
            ctx,
            {
                "metric": "raw",
                "window": 4,
                "min_relative_improvement": 0.01,
                "min_steps": 10,
            },
        )
        assert verdict.triggered is True
        assert "grader_scores_have_no_spread" in (verdict.detail or "")

    def test_a_health_signal_alone_stops_NOTHING(self) -> None:
        """**F6 stops on none of them.** F3 FR-003.10: *"F3 measures and names; it does not
        stop a run"*, and R-02.39's rationale relies on it - *"one that rejects everything
        stalls the run."* Adding a sixth condition here is the invention this project has
        twice paid for."""
        ctx = StopContext(
            RUN,
            60,
            0,
            window_losses=(0.9, 0.7, 0.5, 0.3),
            window_steps=(57, 58, 59, 60),
            window_health={
                "selection_rejects_everything": True,
                "grader_scores_have_no_spread": True,
            },
        )
        verdicts = stop.evaluate_all(
            {
                "loss_plateau": {
                    "metric": "raw",
                    "window": 4,
                    "min_relative_improvement": 0.01,
                    "min_steps": 10,
                }
            },
            ctx,
        )
        assert all(v.triggered is False for v in verdicts)


# ------------------------------------------------------------------------- precedence


class TestPrecedence:
    def test_the_order_is_the_fixed_tuple_FR00627_names(self) -> None:
        assert precedence.PRECEDENCE == (
            "operator_stop",
            "retention_floor",
            "test_threshold",
            "loss_plateau",
            "step_count",
        )

    def test_every_stop_reason_has_a_rank(self) -> None:
        """Asserted as a test as well as at import: a condition added to the registry and
        not to the tuple would fall through to "whatever came first", so the same run
        stopped by the same two conditions could report different reasons on two machines."""
        for name in stop.names():
            assert isinstance(precedence.rank(name), int)

    def test_an_unranked_name_is_refused_rather_than_ranked_last(self) -> None:
        """A silent last place is how a sixth condition becomes the least important thing in
        the system without anyone deciding."""
        with pytest.raises(Refusal) as caught:
            precedence.rank("gpu_too_hot")
        assert caught.value.code == "stop_condition_unknown"

    @pytest.mark.parametrize(
        ("a", "b", "winner"),
        [
            ("operator_stop", "retention_floor", "operator_stop"),
            ("operator_stop", "step_count", "operator_stop"),
            ("retention_floor", "test_threshold", "retention_floor"),
            ("retention_floor", "step_count", "retention_floor"),
            ("test_threshold", "loss_plateau", "test_threshold"),
            ("test_threshold", "step_count", "test_threshold"),
            ("loss_plateau", "step_count", "loss_plateau"),
        ],
    )
    def test_every_pair_resolves_to_the_more_significant_reason(self, a, b, winner) -> None:
        """E-5, both orders. The floor outranks the optimistic reasons because *"the run
        stopped because general capability fell below the floor"* is the finding and *"the
        run reached its target"* is not."""
        for pair in ((a, b), (b, a)):
            verdicts = [
                Verdict(condition=name, triggered=True, observed_at_step=1) for name in pair
            ]
            first = precedence.first_fired(verdicts)
            assert first is not None
            assert first.condition == winner

    def test_BOTH_fired_conditions_are_recorded_so_nothing_is_lost(self) -> None:
        """FR-006.27: only ``stop_reason`` is single-valued. ``all_fired`` keeps the rest,
        in precedence order so the record is stable."""
        verdicts = [
            Verdict(condition="step_count", triggered=True, observed_at_step=1),
            Verdict(condition="retention_floor", triggered=True, observed_at_step=1),
            Verdict(condition="loss_plateau", triggered=False, observed_at_step=1),
        ]
        assert precedence.fired(verdicts) == ("retention_floor", "step_count")

    def test_nothing_fired_is_None_rather_than_a_sentinel(self) -> None:
        """ "Nothing fired" is the normal case at almost every boundary, and a caller
        comparing against a sentinel string could mistake it for a reason."""
        verdicts = [Verdict(condition="step_count", triggered=False, observed_at_step=1)]
        assert precedence.first_fired(verdicts) is None
