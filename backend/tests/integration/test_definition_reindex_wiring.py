"""``reindex`` rebuilds F6's definition index  (ADR-004; task 9.3, task 8.15).

**Assert the wiring, not the existence.** This project has shipped a
``TestRingPruningIsWired`` that asserted an entry point existed while nothing called it, and
``mf_steps``, ``mf_batches`` and ``mf_scores`` each shipped with a rebuilder **no production
caller reached** - three features independently reported the same one-line gap, each rightly
declining to edit a file that was not theirs, which is exactly how they accumulated.

So this file asserts that **the one command iterates F6's rebuilder**, by nulling the three
columns and requiring ``reindex`` to put them back from the run log. A test that only
imported the function would pass against a fourth unwired rebuilder.

## Three columns, not the row  (ADR-004)

``mf_runs`` is a **record** table - its status, its heartbeat and the gap between its beats
are nowhere in any log - so ``drop_index`` does not empty it and the rebuilder does not
recreate a missing row. What **is** an index is the ``definition`` block's three columns, and
those are repaired. Chain rows are excluded from the reindex-equivalence assertion for the
same reason F7's packs were until ``pack.json`` became normative: there is no on-disk
manifest, so making the assertion pass would mean inventing one.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import sqlalchemy as sa

from core.config import reset_settings
from schemas.run_definition import RunDefinition
from services import reindex as reindex_module
from services import run_launch
from tests.support.db import migrated_engine, session_for
from workers.run_record import RUNS

RUN = "run-reindex-def"

DEFINITION = RunDefinition(
    run_id=RUN,
    model_source={"mode": "load", "path": "/data/checkpoints/base"},
    run_type="change",
    run_configuration={"learning_rate": 1e-5, "warmup_steps": 1, "batch_size": 1},
    signal_source={"source": "corpus"},
    parts=[{"part_id": "p1", "kind": "corpus"}],
    stop_conditions=[{"condition": "step_count", "config": {"total": 10}}],
    domain="legal",
    seed=4242,
)

STUBS = {
    "gate": lambda *a, **k: None,
    "resolve_parts": lambda *a, **k: {"source": "parts", "parts": []},
    "budget": lambda *a, **k: {"param_count": 10, "gpu_uuid": None},
    "services": lambda *a, **k: {"needed": []},
}


@pytest.fixture
def launched(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """A run whose definition is recorded, with the artifact roots in a temp directory."""
    monkeypatch.setenv("CHECKPOINT_ROOT", str(tmp_path / "checkpoints"))
    monkeypatch.setenv("TRAINER_STATE_ROOT", str(tmp_path / "trainer-state"))
    monkeypatch.setenv("PARTS_ROOT", str(tmp_path / "parts"))
    monkeypatch.setenv("PACKS_ROOT", str(tmp_path / "packs"))
    reset_settings()

    # The **migrated** schema, not `Base.metadata.create_all`: `reindex` empties every
    # table in `INDEX_TABLES`, including F10's and F4's, which live on their own metadata
    # and exist only after their migrations have run. An engine built from one metadata
    # would fail on `DELETE FROM mf_part_splits` - a measurement that did not happen,
    # reported as a reindex failure.
    engine = migrated_engine(tmp_path / "db")
    with session_for(engine) as session:
        run_launch.launch(
            DEFINITION,
            session=session,
            enqueue=lambda run_id: None,
            log_root=tmp_path,
            **STUBS,
        )
        session.commit()
    yield engine, tmp_path
    engine.dispose()
    reset_settings()


class TestTheRebuilderItself:
    def test_it_restores_the_three_columns_from_the_run_log(self, launched) -> None:
        engine, root = launched
        with session_for(engine) as session:
            session.execute(
                sa.update(RUNS)
                .where(RUNS.c.id == RUN)
                .values(
                    definition_submitted=None,
                    definition_resolved=None,
                    definition_sha256=None,
                )
            )
            session.commit()

            rebuilt = run_launch.rebuild_definition_index(
                session, log_root=root / "runs", run_id=RUN
            )
            session.commit()
            row = session.execute(
                sa.select(
                    RUNS.c.definition_submitted,
                    RUNS.c.definition_resolved,
                    RUNS.c.definition_sha256,
                ).where(RUNS.c.id == RUN)
            ).one()

        assert rebuilt == 1
        assert row.definition_submitted["run_type"] == "change"
        assert row.definition_resolved["seed"] == 4242
        assert len(row.definition_sha256) == 64

    def test_a_run_with_no_log_rebuilds_NOTHING_rather_than_refusing(self, launched) -> None:
        """In a global rebuild a run row with no log on disk is ordinary - a run that never
        started, or one whose log was moved. Refusing the whole operation because one run
        of fifty has no log would make ADR-004's recovery path unusable, which is the
        opposite of what it is for."""
        engine, root = launched
        with session_for(engine) as session:
            assert (
                run_launch.rebuild_definition_index(
                    session, log_root=root / "runs", run_id="no-such-run"
                )
                == 0
            )

    def test_it_does_NOT_recreate_a_missing_run_row(self, launched) -> None:
        """``mf_runs`` is a **record**, not an index: its status, its heartbeat and the gap
        between its beats are nowhere in any log. Recreating a row from a definition block
        would invent a run whose live state nothing knows."""
        engine, root = launched
        with session_for(engine) as session:
            session.execute(sa.delete(RUNS).where(RUNS.c.id == RUN))
            session.commit()
            assert (
                run_launch.rebuild_definition_index(session, log_root=root / "runs", run_id=RUN)
                == 0
            )
            session.commit()
            assert (
                session.execute(sa.select(RUNS.c.id).where(RUNS.c.id == RUN)).one_or_none() is None
            )


class TestTheCommandITERATESIt:
    """**The wiring assertion.** A fourth unwired rebuilder would pass every test above."""

    def test_reindex_rebuilds_the_definition_index(self, launched, tmp_path) -> None:
        """Nulled columns, then ``reindex`` - **not** the rebuilder directly.

        Deleting the call from ``services/reindex.py`` must make this red. Without it, F6's
        rebuilder joins the three that existed with nothing calling them, and ADR-004's
        *"the database is a rebuildable index"* would be false for the one record that says
        what a run was asked to do.
        """
        engine, _ = launched
        with session_for(engine) as session:
            session.execute(
                sa.update(RUNS)
                .where(RUNS.c.id == RUN)
                .values(definition_resolved=None, definition_sha256=None)
            )
            session.commit()

            report = reindex_module.reindex(session, root=tmp_path / "checkpoints")
            session.commit()
            row = session.execute(
                sa.select(RUNS.c.definition_resolved, RUNS.c.definition_sha256).where(
                    RUNS.c.id == RUN
                )
            ).one()

        assert report.definitions == 1, (
            "reindex did not iterate F6's definition rebuilder. If this is 0 the rebuilder "
            "is a fourth declared-but-unwired mechanism, which is the defect this file "
            "exists to catch."
        )
        assert row.definition_resolved is not None
        assert row.definition_sha256 is not None

    def test_the_report_NAMES_the_count_rather_than_hiding_it(self, launched, tmp_path) -> None:
        """ADR-012: every number names its scale. A rebuild that reported nothing would
        leave an operator unable to tell "nothing to rebuild" from "did not run"."""
        engine, _ = launched
        with session_for(engine) as session:
            report = reindex_module.reindex(session, root=tmp_path / "checkpoints")
        assert hasattr(report, "definitions")
        assert isinstance(report.definitions, int)
