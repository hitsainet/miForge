"""A full bracketed run against the fake surface: attach -> verify -> N steps -> save -> detach.

**Against a fake, and the fake is a reading of a contract with no implementation.** miLLM
has no weight attach/detach/quiesce/resume/commit/save/load code, so nothing here is
evidence about miLLM - it is evidence that F11 speaks the protocol BRD-01 describes. The
live equivalent is ``test_hotupdate_live.py``, marked ``upstream`` and skipped by default,
and it is the only thing that can prove the other half.

What this file does prove, and nothing else in the suite does:

* the **whole sequence** of calls a run makes, in order, across many steps;
* that the serving fake was **never asked to serve while the set was not clean** (BRD AC9,
  AC-011.1) - and that the probe for that can fire, which is what keeps it from being
  vacuous;
* that a save is refused **locally** before miLLM's own refusal (AC-011.5, **MUT-6**);
* that the exit is clean: ``commit-end`` -> ``resume`` -> ``detach``, heartbeat stopped
  first, never a detach while ``writing``.
"""

from __future__ import annotations

from typing import Any

import pytest

from core.errors import Refusal
from tests.support.fixtures_hotupdate import Harness, RecordingOptimizer
from training import run_log
from training.hotupdate import barrier, state
from training.hotupdate import timings as timings_module
from training.hotupdate.attach import AttachAcquirer
from training.hotupdate.heartbeat import Heartbeat


class Scalar:
    """The shape F1's step consumes: ``backward``, ``detach``, ``item``."""

    def __init__(self, value: float) -> None:
        self.value = value

    def backward(self) -> None:
        return None

    def detach(self) -> Scalar:
        return self

    def item(self) -> float:
        return self.value


class Model:
    def parameters(self) -> list[Any]:
        return []


class Batch:
    objective = "sft"

    @property
    def record(self) -> dict[str, Any]:
        return {"batch_sha256": "f" * 64}


class Source:
    def next(self) -> Batch:
        return Batch()


class Objective:
    name = "sft"

    def __init__(self) -> None:
        self.calls = 0

    def loss(self, model: Any, batch: Any) -> Scalar:
        self.calls += 1
        return Scalar(1.0 - self.calls * 0.01)


class Registry:
    def __init__(self, objective: Objective) -> None:
        self._objective = objective

    def get(self, name: str) -> Objective:
        return self._objective


STEPS = 5


class TestAFullRun:
    def test_the_whole_sequence_in_order_and_serving_is_never_held_wrongly(
        self, harness: Harness
    ) -> None:
        objective = Objective()
        harness.attach()
        expected = state.require_clean(harness.client)
        beat = Heartbeat(client=harness.client, interval_seconds=0.01)
        beat.start()
        optimizer = RecordingOptimizer([])
        try:
            for _step in range(STEPS):
                outcome = barrier.hot_update_step(
                    Model(),
                    optimizer,
                    Source(),
                    client=harness.client,
                    timings=harness.timings,
                    expected_identity=expected,
                    objectives=Registry(objective),
                )
                assert outcome.loss < 1.0
                # The serving process answers between steps, and the fake records any
                # completion served from a non-clean set. It must stay empty.
                assert harness.fake.serve_completion().startswith("served by")
                expected = state.ResidentIdentity(
                    model=expected.model,
                    manifest_sha256=expected.manifest_sha256,
                    steps_since_load=harness.fake.steps_since_load,
                )
        finally:
            beat.stop()

        assert optimizer.steps == STEPS
        assert harness.fake.steps_since_load == STEPS
        assert harness.fake.served_while_not_clean == []
        # Every step's four protocol calls, in R-02.52's order, with the state read first.
        per_step = [op for op in harness.fake.ops() if op in {"status", *barrier.STAGE_ORDER}]
        assert per_step == ["status"] + ["status", *barrier.STAGE_ORDER] * STEPS

    def test_both_section_5_10_fields_are_filled_on_every_step(
        self, harness: Harness, tmp_path: Any
    ) -> None:
        """The two fields are the input to **another repository's** acceptance gate, so the
        assertion is that every step carries them - not that the schema allows them."""
        harness.attach()
        path = run_log.run_log_path(tmp_path, "hot-run")
        path.parent.mkdir(parents=True, exist_ok=True)
        optimizer = RecordingOptimizer([])
        objective = Objective()
        for step in range(STEPS):
            outcome = barrier.hot_update_step(
                Model(),
                optimizer,
                Source(),
                client=harness.client,
                timings=harness.timings,
                objectives=Registry(objective),
            )
            run_log.append(
                path,
                timings_module.fill(
                    run_log.step_block(
                        run_id="hot-run",
                        step=step,
                        loss=outcome.loss,
                        objective=outcome.objective,
                        learning_rate=1e-5,
                        wall_time_ms=outcome.wall_time_ms,
                    ),
                    harness.timings,
                ),
            )
        rows = list(run_log.read_blocks(path, block="step"))
        assert len(rows) == STEPS
        assert all(row["commit_duration_ms"] is not None for row in rows)
        assert all(row["quiesce_hold_ms"] is not None for row in rows)
        assert run_log.commit_duration_percentile(path, 99.9) is not None

    def test_the_serving_probe_can_actually_fire(self, harness: Harness) -> None:
        """Without this, *"served_while_not_clean == []"* above proves nothing - it would be
        empty against a probe that never checked. The vacuity failure this project keeps
        finding, asserted away."""
        harness.attach()
        with barrier.commit_bracket(harness.client, harness.timings):
            harness.fake.serve_completion()
        assert harness.fake.served_while_not_clean == ["writing"]


class TestTheSaveIsRefusedLocally:
    def test_a_save_while_not_clean_is_refused_before_the_call(self, harness: Harness) -> None:
        """**MUT-6**, AC-011.5. miLLM would refuse it too (R-01.11) - *and relying on that
        is the defect*: the local read is what makes "no checkpoint from a set that never
        existed at any step" a property of this process rather than a hope about another."""
        harness.attach()
        harness.fake.weights_state = "torn"
        harness.fake.calls.clear()
        with pytest.raises(Refusal) as caught:
            state.require_clean(harness.client)
            harness.client.call("save", {})
        assert caught.value.code == "weights_not_clean"
        assert harness.fake.call_count("save") == 0, "nothing may be saved from a torn set"
        assert harness.fake.saves == []

    def test_a_save_from_a_clean_set_goes_through(self, harness: Harness) -> None:
        """Through the **real orchestrator**, which is what settles the body's shape.

        *Amended 2026-10-14, and the amendment is the point.* This test used to call
        ``client.call("save", {"run_id": "hot-run"})`` - a flat body with no ``manifest``
        envelope and a field miLLM does not read there - and it **passed**, because nothing
        in miForge sent this endpoint a body at all and so the fake accepted any shape.
        The first real save would have been ``400 INVALID_MANIFEST``, and this test would
        have said the path worked. It now goes through ``save.AttachedSaver``, whose body
        the fake validates for field ownership the way miLLM does.
        """
        from tests.support.checkpoints import fields as manifest_fields
        from training.hotupdate import save as save_module

        harness.attach()
        identity = state.require_clean(harness.client)
        saver = save_module.AttachedSaver(harness.client, expected=identity)
        outcome = saver(fields=manifest_fields(run_id="hot-run"), session=None)
        assert outcome.response["weights_sha256"]
        assert len(harness.fake.saves) == 1
        sent = harness.fake.saves[0]["manifest"]
        assert sent["run_id"] == "hot-run"
        assert not (harness.fake.CALLER_MUST_NOT_SEND & set(sent))


class TestTheCleanExit:
    def test_the_ordering_is_commit_end_then_resume_then_detach(self, harness: Harness) -> None:
        """FR-011.18: *an unclean exit converts an orderly shutdown into a serving outage*,
        because a detach while ``writing`` marks the set torn at once and torn survives
        detach."""
        acquirer = AttachAcquirer(transport=harness.fake.transport())
        acquirer.client = harness.client
        harness.attach()
        beat = Heartbeat(client=harness.client, interval_seconds=0.01)
        beat.start()
        with barrier.commit_bracket(harness.client, harness.timings):
            pass
        acquirer.detach(heartbeat=beat)
        ops = [op for op in harness.fake.ops() if op != "heartbeat" and op != "status"]
        assert ops[-3:] == ["commit-end", "resume", "detach"]
        assert beat.stopped is True

    def test_a_detach_while_writing_is_refused(self, harness: Harness) -> None:
        acquirer = AttachAcquirer(transport=harness.fake.transport())
        acquirer.client = harness.client
        harness.attach()
        harness.client.call("quiesce")
        harness.client.call("commit-begin")
        with pytest.raises(Refusal) as caught:
            acquirer.detach()
        assert caught.value.code == "detach_refused_set_not_clean"
        assert harness.fake.call_count("detach") == 0
