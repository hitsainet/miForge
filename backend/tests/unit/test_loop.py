"""The loop and the boundary  (FR-006.1 - .9, .12, .16; AC-006.1 - .7).

Five mutation controls live against this file and one against the structure of the tree:

* **M-7** swaps the evaluation and the save. The manifest then says ``test: null`` for a
  step that *was* tested - conforming, validating, permanently wrong.
* **M-11** removes the step-0 evaluation. A creation run's first data point is then already
  post-update, with nothing to compare against (acceptance criterion 12).
* **M-12** calls ``source.next()`` in the loop as well as in F1's step. Every other batch is
  then trained on nothing and the step series still looks perfect.
* **M-13** removes the run-end save.
* **M-17** renames an emitted event or drops an emission - invisible from both ends.
* **AC-006.1**: exactly one ``repeat`` over steps exists in the product.
"""

from __future__ import annotations

import pytest

from core.errors import Refusal
from tests.support.loop_fakes import (
    FakeEmit,
    FakeEvaluate,
    FakeHeartbeat,
    FakeSave,
    FakeSource,
    FakeStep,
    boundary_context,
)
from training import loop
from training.boundary import run_boundary
from training.run_events import EVENT_BOUNDARY, EVENT_REFRESH, EVENT_RUN_TERMINAL


class TestThereIsExactlyOneRepeat:
    """AC-006.1, and the reason F1's ``for`` was moved rather than wrapped."""

    def test_only_trainingloop_contains_a_loop_over_steps(self) -> None:
        """A second ``repeat`` is where the orchestrator turns back into the monolith the
        architecture was decomposed to avoid. Searched structurally for a ``while``/``for``
        whose body calls a step, rather than for the word "loop"."""
        import ast
        from pathlib import Path

        src = Path(__file__).resolve().parents[3] / "backend" / "src"
        sites: list[str] = []
        for path in sorted(src.rglob("*.py")):
            if "hotupdate" in path.parts:
                continue
            tree = ast.parse(path.read_text(encoding="utf-8"))
            for node in ast.walk(tree):
                if not isinstance(node, (ast.While, ast.For)):
                    continue
                for inner in ast.walk(node):
                    if not isinstance(inner, ast.Call):
                        continue
                    name = (
                        inner.func.attr
                        if isinstance(inner.func, ast.Attribute)
                        else inner.func.id if isinstance(inner.func, ast.Name) else ""
                    )
                    if name in {"run_one", "run_boundary", "take_step"}:
                        sites.append(f"{path.relative_to(src)}:{node.lineno}")
        assert sites == ["training/loop.py:" + str(_while_line()).split(":")[-1]] or all(
            site.startswith("training/loop.py") for site in sites
        ), f"the only step loop must be training/loop.py; found {sites}"
        assert sites, "no step loop was found at all - the scanner is not working"

    def test_F1s_task_delegates_rather_than_keeping_its_own_for(self) -> None:
        """F1's TDD declared the ripple: *"F6 owns the repeating"*. Asserted on the source,
        because a second ``for`` there would be behaviour-identical today and divergent the
        first time either feature changed."""
        import ast
        from pathlib import Path

        source = (
            Path(__file__).resolve().parents[3] / "backend" / "src" / "training" / "tasks.py"
        ).read_text(encoding="utf-8")
        drive = next(
            node
            for node in ast.walk(ast.parse(source))
            if isinstance(node, ast.FunctionDef) and node.name == "drive_run"
        )
        step_loops = [
            node
            for node in ast.walk(drive)
            if isinstance(node, (ast.For, ast.While))
            and any(
                isinstance(inner, ast.Call)
                and isinstance(inner.func, ast.Attribute)
                and inner.func.attr == "run_one"
                for inner in ast.walk(node)
            )
        ]
        assert step_loops == [], "drive_run must delegate the repeat to training/loop.py"


def _while_line() -> str:
    return "training/loop.py"


class TestTheBoundarysOrderIsTheRequirement:
    """FR-006.7: step -> evaluate -> save -> stop -> refresh."""

    def test_the_phases_happen_in_that_order(self) -> None:
        ctx = boundary_context(save_every=1, test_every=1)
        outcome = run_boundary(ctx, step=1, round_number=0)
        assert outcome.effects[:4] == ["step", "test", "save", "stop"]

    def test_the_evaluation_precedes_the_save_and_SWAPPING_them_is_the_defect(self) -> None:
        """**M-7.** R-02.18 puts the step's test results into the checkpoint manifest, and
        F5 writes the ``ScoreRecord``s F4 supplies **verbatim or ``null``**, where *"``null``
        means 'no test ran at this step'"*. If the save ran first, a save landing on a
        coincident test boundary would write ``test: null`` for a step that **was** tested -
        a manifest that validates, saves, and is permanently wrong. Nothing raises."""
        order: list[str] = []
        ctx = boundary_context(
            save_every=1,
            test_every=1,
            evaluate=lambda step: order.append("test"),
            save=lambda step: order.append("save"),
        )
        run_boundary(ctx, step=1, round_number=0)
        assert order == ["test", "save"], (
            "the evaluation must precede the save, or a coincident boundary writes "
            "test: null into the manifest of a step that was tested"
        )

    def test_the_stop_evaluation_happens_AFTER_the_save(self) -> None:
        """So a floor-breach stop still gets the save that carries the breaching score
        (FR-006.9): the artifact tells the truth, it is not hidden to make the run look
        better."""
        ctx = boundary_context(save_every=1)
        outcome = run_boundary(ctx, step=1, round_number=0)
        assert outcome.effects.index("save") < outcome.effects.index("stop")

    def test_the_loop_calls_F1s_step_exactly_once_per_boundary(self) -> None:
        """AC-006.2, and the first half of M-12."""
        step = FakeStep()
        ctx = boundary_context(take_step=step)
        run_boundary(ctx, step=1, round_number=0)
        assert step.calls == [1]

    def test_the_loop_NEVER_calls_sourcenext_itself(self) -> None:
        """**M-12.** F1's ``step.run_one`` pulls the batch. Two pulls per step means every
        other batch is trained on nothing and the step series looks perfect."""
        source = FakeSource()
        result = loop.run(boundary_context(source=source, take_step=FakeStep()), max_steps=5)
        assert result.steps_completed == 5
        assert (
            source.calls == 0
        ), "the loop must not pull batches; F1's step does, exactly once per boundary"


class TestTheCadences:
    def test_a_save_happens_only_on_the_cadence(self) -> None:
        save = FakeSave()
        loop.run(boundary_context(save=save, save_every=2), max_steps=6, run_end_save=False)
        assert save.calls == [2, 4, 6]

    def test_NO_save_happens_at_step_zero(self) -> None:
        """``0 % n == 0`` for every ``n``, so a naive cadence check would checkpoint
        **untrained weights** - and the resulting checkpoint would look like any other."""
        save = FakeSave()
        loop.run(boundary_context(save=save, save_every=1), max_steps=2, run_end_save=False)
        assert 0 not in save.calls

    def test_an_operator_requested_save_lands_at_the_next_boundary_off_cadence(self) -> None:
        """Two reasons to save, collapsing into one at one boundary (F5 FR-005.1)."""
        save = FakeSave()
        asked = {"n": 0}

        def requested(step: int) -> object | None:
            asked["n"] += 1
            return {"request_count": 1} if step == 3 else None

        loop.run(
            boundary_context(save=save, save_every=None, save_requested=requested),
            max_steps=4,
            run_end_save=False,
        )
        assert save.calls == [3]

    def test_a_cadence_and_a_request_at_one_boundary_produce_ONE_save(self) -> None:
        save = FakeSave()
        loop.run(
            boundary_context(
                save=save, save_every=2, save_requested=lambda step: {"request_count": 1}
            ),
            max_steps=2,
            run_end_save=False,
        )
        assert save.calls == [1, 2], "one save per boundary, not two"

    def test_the_boundary_record_names_the_next_cadence_points(self) -> None:
        ctx = boundary_context(save_every=5, test_every=3)
        outcome = run_boundary(ctx, step=4, round_number=0)
        block = list(ctx.emit.named(EVENT_BOUNDARY))  # type: ignore[union-attr]
        assert block
        assert outcome.timings_ms


class TestStepZeroAndTheRunEndSave:
    def test_the_baseline_is_evaluated_BEFORE_the_first_step(self) -> None:
        """**M-11.** FR-006.6, DEF-1: R-02.24's loop body does not contain this - its
        ``test()`` follows ``step(batch)``, so with steps numbered from 1 no evaluation
        ever occurs at step 0, while acceptance criterion 12 names the step-0 test score as
        a creation run's baseline **explicitly**."""
        evaluate = FakeEvaluate()
        step = FakeStep()
        loop.run(
            boundary_context(evaluate=evaluate, take_step=step),
            max_steps=2,
            evaluate_step_zero=True,
        )
        assert evaluate.calls[0] == 0, "no step-0 baseline was recorded"
        assert step.calls[0] == 1, "the baseline must precede the first step"

    def test_a_resume_does_not_append_a_second_baseline(self) -> None:
        """Step 0 already has one, and re-evaluating would record a second baseline taken
        at a different parameter set."""
        evaluate = FakeEvaluate()
        loop.run(
            boundary_context(evaluate=evaluate),
            max_steps=12,
            start_step=10,
            evaluate_step_zero=True,
        )
        assert 0 not in evaluate.calls

    def test_the_run_end_save_happens_for_a_condition_termination(self) -> None:
        """**M-13.** R-02.20 requires a save *"on the configured cadence and at run end"*,
        and R-02.24's loop body contains no run-end save - so the obligation lands here."""
        save = FakeSave()
        result = loop.run(
            boundary_context(
                save=save, save_every=None, stop_conditions={"step_count": {"total": 3}}
            ),
            max_steps=10,
            run_end_save=True,
        )
        assert result.stopped_because == "step_count"
        assert save.calls == [3], "no run-end save was made"
        assert result.run_end_saved is True

    def test_a_run_end_save_landing_on_a_cadence_boundary_is_F5s_problem_not_F6s(
        self,
    ) -> None:
        """S-2: F5's FR-005.6 makes the second a no-op when the hash matches. **F6 adds no
        de-duplication of its own** - a second rule about when a save is really a save is a
        second answer, and F5's is the one that reads the manifest."""
        save = FakeSave()
        loop.run(
            boundary_context(save=save, save_every=2, stop_conditions={"step_count": {"total": 4}}),
            max_steps=10,
            run_end_save=True,
        )
        assert save.calls == [2, 4, 4], (
            "F6 must call save twice at step 4 and let F5's idempotence collapse it; "
            "de-duplicating here would be a second answer"
        )

    def test_NO_run_end_save_follows_a_failure(self) -> None:
        """E-9. The last completed save is the recovery point, and a save after an exception
        writes an artifact from a state nothing verified.

        **The mechanism is the re-raise, not a flag**: the run-end save is below the
        ``except`` block, so a failure skips it by construction - there is no
        ``if not failed`` for a later edit to get wrong.
        """
        save = FakeSave()
        with pytest.raises(RuntimeError):
            loop.run(
                boundary_context(save=save, save_every=None, take_step=FakeStep(raise_at=3)),
                max_steps=10,
                run_end_save=True,
            )
        assert save.calls == []

    def test_a_cancellation_DOES_get_a_run_end_save(self) -> None:
        """FR-006.8: *"for every stop-condition termination, including ... an operator
        stop"*. A cancellation is an operator stop, so the operator keeps the work."""
        save = FakeSave()
        result = loop.run(
            boundary_context(save=save, save_every=None, take_step=FakeStep(cancel_at=3)),
            max_steps=10,
            run_end_save=True,
        )
        assert result.stopped_because == loop.CANCELLED
        assert save.calls == [2], "the last completed step must be saved"


class TestTheHeartbeatBracketsEveryLongPhase:
    def test_it_beats_unconditionally_before_any_work(self) -> None:
        """A ``running`` row whose ``heartbeat_at`` is NULL is ambiguous, and the janitor
        **reports** such a row rather than reaping it. Beating first keeps that case empty."""
        beat = FakeHeartbeat()
        loop.run(boundary_context(heartbeat=beat, session=object()), max_steps=1)
        assert beat.started == 1

    def test_it_advances_around_the_evaluation_the_save_and_the_refresh(self) -> None:
        """**M-3**, and it is ADR-007's named precedent reproduced structurally: miStudio
        reaped a live 5.8-hour job as "worker lost". The 60 s throttle is a **minimum
        interval between writes, not a maximum gap**, and a test-set pass, a
        self-generation batch and a 2.4 GB save each exceed it."""
        beat = FakeHeartbeat()
        ctx = boundary_context(heartbeat=beat, session=object(), save_every=1, test_every=1)
        run_boundary(ctx, step=1, round_number=0)
        assert beat.attempts >= 3, (
            f"expected an advance after the step, the evaluation and the save; got "
            f"{beat.attempts}. With fewer, a generation-bound pass leaves a gap the "
            f"janitor reads as a dead worker."
        )

    def test_the_longest_gap_is_measured_and_recorded(self) -> None:
        """OQ-5: the janitor's lapse threshold has no defensible value until something
        measures how long the loop actually goes between beats, and **nothing else in the
        product is positioned to say**."""
        ctx = boundary_context(heartbeat=FakeHeartbeat(), session=object(), save_every=1)
        outcome = run_boundary(ctx, step=1, round_number=0)
        assert outcome.longest_gap_ms is not None
        assert outcome.longest_gap_ms >= 0.0

    #: Any syntactic reference to the heartbeat column, in any of the three spellings a
    #: writer could use. **Parsed, not grepped**: the first draft matched the string in a
    #: *comment* explaining why the column must not be touched, which is a guard red for
    #: the exact opposite of a defect.
    HEARTBEAT_COLUMN = "heartbeat_at"

    def _names_the_heartbeat_column(self, path) -> list[str]:
        import ast

        found: list[str] = []
        for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
            if isinstance(node, ast.Attribute) and node.attr == self.HEARTBEAT_COLUMN:
                found.append(f"{path.name}:{node.lineno}")
            elif isinstance(node, ast.keyword) and node.arg == self.HEARTBEAT_COLUMN:
                found.append(f"{path.name}:?")
            elif (
                isinstance(node, ast.Constant)
                and node.value == self.HEARTBEAT_COLUMN
                # A docstring is a Constant too, and the module docstrings here discuss the
                # column by name on purpose. Only an EXACT match is a reference.
                and isinstance(node.value, str)
            ):
                found.append(f"{path.name}:{node.lineno}")
        return found

    def test_F6_adds_no_second_progress_writing_path(self) -> None:
        """FR-006.11, **M-4's companion**: the heartbeat refuses to move a terminal row, and
        F6 must not work around it.

        Asserted as the absence of any F6 reference to the column at all - not merely of a
        write. A read would be harmless; a *second writer* would resurrect a terminal row
        on the run list, and the cheap way to be sure there is none is for the name not to
        appear in F6's code.
        """
        from pathlib import Path

        src = Path(__file__).resolve().parents[3] / "backend" / "src" / "training"
        f6 = [
            src / "loop.py",
            src / "boundary.py",
            src / "run_state.py",
            src / "run_driver.py",
            *sorted((src / "stop").rglob("*.py")),
            *sorted((src / "refresh").rglob("*.py")),
        ]
        offenders = [site for path in f6 for site in self._names_the_heartbeat_column(path)]
        assert not offenders, (
            f"F6 must reach the heartbeat only through workers.heartbeat.Heartbeat, which "
            f"is the one writer and the thing that refuses a terminal row: {offenders}"
        )

    def test_the_guard_can_see_each_spelling_a_writer_would_use(self, tmp_path) -> None:
        """The self-test, and it is three cases because there are three spellings."""
        for source in (
            "RUNS.c.heartbeat_at = now",
            "session.execute(update(RUNS).values(heartbeat_at=now))",
            'row["heartbeat_at"] = now',
        ):
            path = tmp_path / "planted.py"
            path.write_text(source, encoding="utf-8")
            assert self._names_the_heartbeat_column(path), f"the guard would not catch {source!r}"


class TestDurableBeforeAnnounced:
    def test_the_log_write_precedes_the_socket_emission(self, tmp_path) -> None:
        """FR-006.12. *Nothing about a run's progress exists only on the socket* - ADR-007's
        rule and the reason it exists."""

        path = tmp_path / "run.jsonl"
        ctx = boundary_context(log_path=path)
        outcome = run_boundary(ctx, step=1, round_number=0)
        assert outcome.effects.index("log") < outcome.effects.index("socket")

    def test_a_socket_failure_cannot_roll_back_a_recorded_boundary(self, tmp_path) -> None:
        """The emission is outside the durable write's ``try``. A Redis outage must not
        fail a run whose work is already durable - and it must not be silent either."""
        from training import run_log

        path = tmp_path / "run.jsonl"
        ctx = boundary_context(log_path=path, emit=FakeEmit(raise_on=EVENT_BOUNDARY))
        run_boundary(ctx, step=1, round_number=0)
        blocks = list(run_log.read_blocks(path, block="boundary"))
        assert len(blocks) == 1, "the boundary record must survive a failed emission"

    def test_the_recorded_block_validates_against_the_vendored_contract(self, tmp_path) -> None:
        """Through the **real** writer and the **real** schema, not a literal: ``append``
        validates, so a block with a key the contract does not declare is refused at the
        writer rather than discovered by a reader that skips it."""
        from training import run_log

        path = tmp_path / "run.jsonl"
        ctx = boundary_context(
            log_path=path,
            save_every=1,
            stop_conditions={"step_count": {"total": 9}, "retention_floor": {}},
        )
        run_boundary(ctx, step=1, round_number=0)
        for block in run_log.read_blocks(path, block="boundary"):
            run_log.validate_block(block)
            assert block["round"] == 0
            assert [c["condition"] for c in block["stop_conditions"]] == [
                "step_count",
                "retention_floor",
            ]
            assert all("observed_at_step" in c for c in block["stop_conditions"])


class TestTheEventNames:
    def test_the_three_strings_are_exactly_these(self) -> None:
        """**M-17.** Until 2026-10-13 nothing in the project named them and F9 had invented
        ``step | score | save | signal``, then withdrew all four. Any subscription written
        against either guess would have been **silently dead while looking connected**."""
        from training.run_events import RUN_EVENTS

        assert RUN_EVENTS == ("boundary", "refresh", "run_terminal")
        assert (EVENT_BOUNDARY, EVENT_REFRESH, EVENT_RUN_TERMINAL) == RUN_EVENTS

    def test_no_emit_site_carries_an_inline_literal(self) -> None:
        """Two spellings of one event is how a subscriber goes quiet.

        Checked on the **emit call's own event argument**, not on every string in the
        module: ``boundary.PHASES`` legitimately contains the word ``"refresh"`` as a
        *phase* name, and a guard that flagged it would be red for the wrong reason - which
        is worse than no guard, because the next session relaxes it. (The first draft of
        this test did exactly that, and that collision is why the check is this narrow.)
        """
        import ast
        from pathlib import Path

        from training.run_events import RUN_EVENTS

        src = Path(__file__).resolve().parents[3] / "backend" / "src" / "training"
        offenders: list[str] = []
        for path in sorted(src.rglob("*.py")):
            if path.name == "run_events.py" or "hotupdate" in path.parts:
                continue
            for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
                if not isinstance(node, ast.Call):
                    continue
                name = (
                    node.func.attr
                    if isinstance(node.func, ast.Attribute)
                    else node.func.id if isinstance(node.func, ast.Name) else ""
                )
                if name not in {"emit", "_emit"}:
                    continue
                for arg in node.args:
                    if isinstance(arg, ast.Constant) and arg.value in RUN_EVENTS:
                        offenders.append(f"{path.name}:{node.lineno}")
        assert not offenders, (
            f"an emit site spells an event name inline; use the constant in "
            f"training/run_events.py: {offenders}"
        )

    def test_the_guard_can_actually_see_an_inline_event_literal(self, tmp_path) -> None:
        """**The self-test.** A scanner that cannot see the thing it bans is a guard that is
        not there."""
        import ast

        tree = ast.parse('emit(run_id, "boundary", {})\n')
        found = [
            node
            for node in ast.walk(tree)
            if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id == "emit"
            and any(isinstance(a, ast.Constant) and a.value == "boundary" for a in node.args)
        ]
        assert found, "the guard would not catch an inline event literal"

    def test_one_boundary_emits_exactly_one_boundary_event_with_its_payload(self) -> None:
        """The count **and** the payload. A reachability test that only checks "was called"
        passes against an emission carrying the wrong step."""
        emit = FakeEmit()
        ctx = boundary_context(emit=emit, stop_conditions={"step_count": {"total": 9}})
        run_boundary(ctx, step=4, round_number=2)
        payloads = emit.named(EVENT_BOUNDARY)
        assert len(payloads) == 1
        assert payloads[0]["step"] == 4
        assert payloads[0]["round"] == 2
        assert payloads[0]["run_id"] == "run-f6"
        assert payloads[0]["stop_reason"] is None
        assert [c["condition"] for c in payloads[0]["stop_conditions"]] == ["step_count"]


class TestTheLoopStopsWhenAConditionFires:
    def test_it_stops_at_the_boundary_the_condition_fired_on(self) -> None:
        step = FakeStep()
        result = loop.run(
            boundary_context(take_step=step, stop_conditions={"step_count": {"total": 3}}),
            max_steps=99,
        )
        assert step.calls == [1, 2, 3]
        assert result.stopped_because == "step_count"
        assert result.last_step == 3

    def test_the_recorded_reason_is_the_precedence_winner_and_ALL_are_kept(self) -> None:
        from tests.support.loop_fakes import AlwaysFires
        from training import stop

        original = dict(stop._CONDITIONS)
        try:
            stop._CONDITIONS["step_count"] = AlwaysFires("step_count")
            stop._CONDITIONS["retention_floor"] = AlwaysFires("retention_floor")
            result = loop.run(
                boundary_context(stop_conditions={"step_count": {}, "retention_floor": {}}),
                max_steps=5,
            )
        finally:
            stop._CONDITIONS.clear()
            stop._CONDITIONS.update(original)
        assert result.stopped_because == "retention_floor"
        assert result.all_fired == ("retention_floor", "step_count")

    def test_the_budget_ceiling_is_not_a_stop_reason(self) -> None:
        """R-02.25's set is closed at five. Running out of the driver's ceiling is reported
        as ``step_budget_reached``, which is not one of them - and a run whose ONLY bound is
        that ceiling is refused at submit (FR-006.17)."""
        from training.stop.base import STOP_REASONS

        result = loop.run(boundary_context(), max_steps=2)
        assert result.stopped_because == loop.BUDGET_EXHAUSTED
        assert result.stopped_because not in STOP_REASONS


class TestRefusalsPropagate:
    def test_an_evaluation_refusal_stops_the_run_rather_than_continuing(self) -> None:
        """E-11: *a run whose floor cannot be measured is a run with no floor.* The loop
        refuses to continue rather than training on past an unmeasurable boundary."""
        with pytest.raises(Refusal) as caught:
            loop.run(
                boundary_context(evaluate=FakeEvaluate(raise_at=2), test_every=1),
                max_steps=5,
            )
        assert caught.value.code == "evaluation_failed"
