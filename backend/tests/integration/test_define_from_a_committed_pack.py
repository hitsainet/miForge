"""A run defined from a **committed pack** gets a row  (F9 task 14.0's live finding).

The first live run of the New run form's pickers (2026-10-18) chose a valid pack, passed the
whole validator, and then ``POST /runs/define`` answered **500**: ``launch`` wrote the bare
pack id into ``mf_runs.pack_id``, which references ``mf_packs.id`` - one row per **version**,
``registry.row_id(name, version)`` = ``notary-md@1``. The foreign key refused the insert.

No test reached it. The launch tests define runs from loose parts or with the pack gate
stubbed, on engines where no ``mf_packs`` row exists to point at. So this one goes through
the **real** gate and the **real** part resolution, against a migrated database that
enforces foreign keys, with a pack written by the real registry. Only the budget (it reads a
card) and the enqueue (it needs a broker) are stood in for.

And it asserts the reader as well as the writer: the pack's ``runs_using`` - which reads
``mf_runs.pack_id`` through ``row_id`` - names the run.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
import sqlalchemy as sa

from core.config import reset_settings
from packs import registry
from packs.settings import reset_pack_settings
from schemas.run_definition import RunDefinition
from services import run_launch
from tests.support import pack_fixtures as fx
from tests.support.db import migrated_engine, session_for
from workers.run_record import RUNS


@pytest.fixture
def engine(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):  # type: ignore[no-untyped-def]
    monkeypatch.setenv("PACKS_ROOT", str(tmp_path / "packs"))
    monkeypatch.setenv("CHECKPOINT_ROOT", str(tmp_path / "checkpoints"))
    monkeypatch.setenv("PARTS_ROOT", str(tmp_path / "parts"))
    monkeypatch.setenv("TRAINER_STATE_ROOT", str(tmp_path / "trainer-state"))
    monkeypatch.delenv("MIDATAWORKS_BASE_URL", raising=False)
    reset_pack_settings()
    reset_settings()
    engine = migrated_engine(tmp_path / "db")
    with session_for(engine) as session:
        for set_kind, reference in (
            ("prompt_set", ("prompt-set", "support-tickets", "3")),
            ("corpus", ("corpus", "notary-corpus", "1")),
            ("test_set", ("test-set", "notary-test", "1")),
            ("retention_set", ("retention-set", "general", "2")),
        ):
            fx.import_contract_part(
                session,
                parts_root=tmp_path / "parts",
                export_root=tmp_path / f"export-{set_kind}",
                set_kind=set_kind,
                reference=reference,
            )
        fx.commit(
            session,
            fx.definition(
                slots=fx.valid_slots(corpus=fx.contract_slot("corpus", "notary-corpus", "1"))
            ),
        )
    yield engine
    engine.dispose()
    reset_pack_settings()
    reset_settings()


def test_a_run_defined_from_a_committed_pack_cites_its_version_row(
    engine: Any, tmp_path: Path
) -> None:
    enqueued: list[str] = []
    definition = RunDefinition(
        run_id="run-from-pack",
        model_source={"mode": "load", "path": str(tmp_path / "checkpoints" / "base")},
        run_type="change",
        run_configuration={"learning_rate": 1e-5, "warmup_steps": 0, "batch_size": 1},
        signal_source={"source": "corpus"},
        pack={"pack_id": "notary-md", "version": "1"},
        stop_conditions=[{"condition": "step_count", "config": {"total": 40}}],
    )

    with session_for(engine) as session:
        accepted = run_launch.launch(
            definition,
            session=session,
            enqueue=enqueued.append,
            log_root=tmp_path / "runs",
            # Only the budget is stood in for: it reads free VRAM from a card. The gate and
            # the part resolution are the real ones, against the real pack.
            budget=lambda *a, **k: {"param_count": 10, "gpu_uuid": None},
        )

    assert accepted["run_id"] == "run-from-pack"
    assert enqueued == ["run-from-pack"]
    with session_for(engine) as session:
        pack_id = session.execute(
            sa.select(RUNS.c.pack_id).where(RUNS.c.id == "run-from-pack")
        ).scalar_one()
        assert pack_id == registry.row_id("notary-md", "1") == "notary-md@1"
        # The reader agrees: the registry lists this run against the pack version.
        rows = {row.pack_id: row for row in registry.listing(session)}
        assert rows["notary-md"].runs_using == ("run-from-pack",)
