"""A run with a recorded history, arranged through the production writers (021 FR-021.13).

Shared by the run-surface answer cases (``runs``, ``runs_index``, ``runs_acquisition``,
``runs_loop``, ``scoring``, ``signal_batches``). Nothing here builds an answer or a run-log
line by hand:

* :func:`define` sends the operator's ``POST /api/v1/runs/define`` on the ``world`` fixture,
  so the run row, its recorded definition (row and run log) and its lease are production's;
* :func:`corpus_batch` takes one batch from the real corpus source over a part imported
  through F10's real ingest, which writes the batch block to the run log and the index row;
* :func:`take_steps` runs F6's real boundary for each step, with F1's real step writer as
  the boundary's step, so the step rows, step blocks and boundary blocks are production's;
* :func:`evaluated` sends the operator's ``POST /api/v1/runs/{id}/evaluate`` over a test set
  imported through F10's ingest, scored by the in-process provider bound to the shipped
  :class:`providers.fakes.FakeModelHandle` (the worker's model handle; there are no weights).
"""

from __future__ import annotations

from typing import Any

import pytest

from tests.support.approval_world import World, definition

RUN = "run-answers"

#: The rule expectation ``test_scoring_endpoints.py`` scores with: at least four characters.
RULE = {"kind": "rule", "name": "len", "config_sha256": None, "expression": '{"min_length": 4}'}


def production_shaped_preflight(monkeypatch: pytest.MonkeyPatch) -> None:
    """Replace the ``world`` fixture's F1 pre-flight stub with one of production's shape.

    ``approval_world.world`` returns an availability with no ``reason``; production returns
    :class:`training.acquisition.base.Availability`, whose ``reason`` the operator's submit
    summary reads (``services.acquisition.acquisition_summary``), so the stub 500s the
    operator's ``POST /api/v1/runs``. Same seam, same enriched spec, production's type.
    """
    from services import acquisition as acquisition_service
    from training.acquisition.base import Availability

    monkeypatch.setattr(
        acquisition_service,
        "preflight",
        lambda spec, cfg, **kw: (
            spec.model_copy(update={"param_count": 10}),
            Availability(available=True),
        ),
    )


def real_lease(world: World, monkeypatch: pytest.MonkeyPatch, acquire: Any) -> None:
    """Put production's ``workers.lease.acquire`` back, recording each call on ``world``.

    ``approval_world.world``'s stand-in takes no ``now=`` (the chains service passes it) and
    writes no ``mf_leases`` row, which the chain's stage-0 hand-over reads (it refuses
    ``chain_lease_lost`` without one). The real function takes the device the budget step
    already resolved, so it touches no card. ``acquire`` is the original, captured by the
    caller before ``world`` replaced it.
    """
    from workers import lease as lease_module

    def recorded(session: Any, run_id: str, *args: Any, **kwargs: Any) -> Any:
        held = acquire(session, run_id, *args, **kwargs)
        world.leases.append((run_id, str(held.gpu_uuid)))
        return held

    monkeypatch.setattr(lease_module, "acquire", recorded)


def define(world: World, run_id: str = RUN) -> str:
    """The operator's define: a real run row with its definition recorded."""
    response = world.client.post("/api/v1/runs/define", json=definition(run_id))
    assert response.status_code == 202, response.text
    return run_id


def _signal_settings(monkeypatch: pytest.MonkeyPatch) -> None:
    from training.signals.settings import reset_signal_settings

    # The corpus source's SFT objective refuses with no masking policy configured.
    monkeypatch.setenv("SIGNAL_SFT_MASKING_POLICY", "completion_only")
    reset_signal_settings()


def corpus_part(world: World, monkeypatch: pytest.MonkeyPatch) -> dict[str, Any]:
    """A corpus part imported through F10's real ingest, as the part index serves it."""
    from tests.support import signal_fixtures

    _signal_settings(monkeypatch)
    with world.session() as session:
        return signal_fixtures.corpus_part(session, world.root, texts=["one", "two", "three"])


def corpus_batch(world: World, monkeypatch: pytest.MonkeyPatch, run_id: str = RUN) -> str:
    """One batch from the real corpus source, recorded to the run log and the index."""
    from tests.support import signal_fixtures
    from training.signals import ensure_registered, registry
    from training.signals.base import PartSpec, RunContext, SourceSpec
    from training.tasks import run_log_root

    part = corpus_part(world, monkeypatch)
    ensure_registered()
    with world.session() as session:
        source = registry.get("corpus")(
            RunContext(
                run_id=run_id,
                session=session,
                log_root=run_log_root(),
                tokenizer=signal_fixtures.FakeTokenizer(),
            )
        )
        source.preflight(
            SourceSpec(
                source="corpus",
                parts=(PartSpec(kind="corpus", part_id=str(part["id"])),),
                batch_size=2,
            )
        )
        batch = source.next()
    return str(batch.record["batch_sha256"])


def take_steps(
    world: World, run_id: str = RUN, *, steps: tuple[int, ...] = (1, 2), batch: str | None = None
) -> None:
    """F6's boundary for each step, F1's step writer as its step, into the run's own log."""
    from training import run_log
    from training.boundary import BoundaryContext, run_boundary
    from training.run_config import RunConfig
    from training.step import StepOutcome
    from training.step_record import StepWriter
    from training.tasks import run_log_root

    cfg = RunConfig(learning_rate=1e-5, warmup_steps=0, batch_size=1, gradient_accumulation=1)
    writer = StepWriter(run_id=run_id, log_root=run_log_root())
    with world.session() as session:
        resolved = world.client.get(f"/api/v1/runs/{run_id}/definition").json()["resolved"]
        conditions = resolved["loop"]["stop_conditions"]
        for step in steps:

            def take(at: int) -> StepOutcome:
                outcome = StepOutcome(
                    loss=1.0 / at,
                    objective="sft",
                    grad_norm=0.5,
                    wall_time_ms=12.0,
                    batch_sha256=batch,
                )
                writer.write(session=session, step=at, outcome=outcome, cfg=cfg)
                return outcome

            run_boundary(
                BoundaryContext(
                    run_id=run_id,
                    session=session,
                    log_path=run_log.run_log_path(run_log_root(), run_id),
                    take_step=take,
                    stop_conditions=conditions,
                    save_every=10,
                ),
                step=step,
                round_number=0,
            )


def evaluated(world: World, monkeypatch: pytest.MonkeyPatch, run_id: str = "run-scored") -> Any:
    """A run whose resolved definition carries a scoring block, evaluated by the operator.

    Returns the ``POST /runs/{run_id}/evaluate`` response.
    """
    from providers import registry
    from providers.fakes import FakeModelHandle
    from tests.support import signal_fixtures
    from tests.support.jobs import insert_run

    _signal_settings(monkeypatch)
    manifest, bodies = signal_fixtures.rows_manifest(
        miforge_set_kind="test_set",
        columns=[signal_fixtures.column("q", "prompt")],
        rows=[{"q": f"prompt {i}"} for i in range(4)],
        evaluation_only=True,
        reference=("test-set", "general", "1"),
    )
    with world.session() as session:
        part = signal_fixtures.import_part(
            session, world.root, manifest=manifest, bodies=bodies, name="test-set"
        )
        insert_run(
            session,
            run_id,
            status="running",
            last_step=4,
            definition_resolved={
                "scoring": {
                    "provider": "in_process",
                    "test_slot": {
                        "part_id": str(part["id"]),
                        "local_path": str(part["local_path"]),
                    },
                    "test_expectation": {"config": RULE, "source": "pack"},
                }
            },
        )
    registry.bind_model_handle(FakeModelHandle(answers="a long enough answer"))
    try:
        return world.client.post(f"/api/v1/runs/{run_id}/evaluate", json={"kinds": ["test"]})
    finally:
        registry.bind_model_handle(None)
