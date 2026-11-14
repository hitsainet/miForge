"""A run's top-level parts reach its signal source through the PRODUCTION driver.

Found 2026-10-17 by tracing one real load run end to end (F11 window 2): a definition names
its material at the top level (`parts`, or a `pack` the validator resolves), and the driver
built the source's spec from the `signal_source` block alone - so every parts- or pack-based
run that VALIDATED CLEAN refused `signal_source_no_part` in the worker. Every test of the
source built its own `SourceSpec(parts=...)`.

Driven through `run_driver.drive_recorded_run` from a recorded definition; the source factory
is replaced only to capture the spec it is handed and stop there.
"""

from __future__ import annotations

from typing import Any

import pytest

from tests.support.jobs import insert_run, memory_engine, session_for
from training import run_driver

RUN = "parts-run"
SUBMITTED = {
    "run_id": RUN,
    "model_source": {"mode": "load", "path": "/data/checkpoints/base"},
    "run_type": "change",
    "run_configuration": {"learning_rate": 1e-5, "warmup_steps": 0, "batch_size": 2},
    "signal_source": {"source": "corpus", "batch_size": 2},
    "parts": [
        {"part_id": "part-corpus", "kind": "corpus", "splits": ["train"]},
        {"part_id": "part-test", "kind": "test_set"},
    ],
    "stop_conditions": [{"condition": "step_count", "config": {"total": 1}}],
    "domain": None,
}
RESOLVED = {
    "parts": {
        "source": "parts",
        "parts": [
            {"part_id": "part-corpus", "kind": "corpus", "reference": {"type": "corpus"}},
            {"part_id": "part-test", "kind": "test_set", "reference": {"type": "test_set"}},
        ],
    },
    "seed": 7,
    "domain": None,
}


class _StopError(Exception):
    pass


@pytest.fixture
def captured(monkeypatch: pytest.MonkeyPatch) -> list[Any]:
    from training import signals

    seen: list[Any] = []

    class Capture:
        def __init__(self, context: Any) -> None:
            self.context = context

        def preflight(self, spec: Any) -> None:
            seen.append(spec)
            raise _StopError

    signals.ensure_registered()
    monkeypatch.setattr(signals.registry, "get", lambda name: Capture)
    return seen


def _drive(submitted: dict[str, Any]) -> None:
    engine = memory_engine()
    with session_for(engine) as session:
        insert_run(
            session,
            RUN,
            definition_submitted=submitted,
            definition_resolved=RESOLVED,
            definition_sha256="x" * 64,
        )
        session.commit()
        with pytest.raises(_StopError):
            run_driver.drive_recorded_run(RUN, session=session)


def test_the_source_gets_the_runs_parts_of_its_own_kind(captured) -> None:  # type: ignore[no-untyped-def]
    _drive(SUBMITTED)
    (spec,) = captured
    assert [(p.kind, p.part_id, tuple(p.splits)) for p in spec.parts] == [
        ("corpus", "part-corpus", ("train",))
    ], "the test set is F4's, not the source's; the corpus part keeps its splits"


def test_parts_stated_on_the_source_block_are_kept_as_given(captured) -> None:  # type: ignore[no-untyped-def]
    submitted = {
        **SUBMITTED,
        "signal_source": {
            "source": "corpus",
            "parts": [{"kind": "corpus", "part_id": "explicit"}],
        },
    }
    _drive(submitted)
    (spec,) = captured
    assert [p.part_id for p in spec.parts] == ["explicit"]


def test_the_run_beats_at_loop_entry_before_any_work(captured) -> None:  # type: ignore[no-untyped-def]
    """`Heartbeat.start()` had no production caller: a run that died during acquisition
    left a `running` row with no heartbeat, which the janitor (rightly) never reaps - so it
    held the GPU lease until expiry (found live 2026-10-17). The source pre-flight here is
    the first work after entry, and it stops the run; the beat must already be written."""
    from tests.support.jobs import read_run

    engine = memory_engine()
    with session_for(engine) as session:
        insert_run(
            session,
            RUN,
            definition_submitted=SUBMITTED,
            definition_resolved=RESOLVED,
            definition_sha256="x" * 64,
        )
        session.commit()
        with pytest.raises(_StopError):
            run_driver.drive_recorded_run(RUN, session=session)
    with session_for(engine) as session:
        assert read_run(session, RUN)["heartbeat_at"] is not None
