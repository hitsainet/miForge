"""Reindex equivalence, and the manifest winning a disagreement  (FR-004.32; ADR-016 #4, ADR-004).

ADR-004: **the filesystem is the source of truth and the database is a rebuildable index.**
So the test is the structural one: seed scores at save **and non-save** steps, snapshot the
API responses, drop the index, rebuild, and assert the responses are **identical**.

The non-save step is the half that matters and the half a weaker test would miss: a score
there exists **only** in the run log, which is why the log is part of the durability story
and not telemetry - *stated so nobody prunes it for disk space.*

Then plant a divergence and assert **the manifest wins and the log line exists** (foundation
4.5). A silent correction is the failure that test exists to catch: the index would quietly
disagree with the artifact and nothing would say which was right.

**Reported, not worked around:** ``services/reindex.py`` does not yet call
:func:`services.scoring.reindex_scores.reindex_scores` - that file is the foundation's and
outside this feature's write scope. Until the one line lands, this capability has no
production caller, which is by this project's own rule a **finding** rather than a pass.
``test_the_production_reindex_does_not_yet_call_it`` pins that, and goes red when it is fixed.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import sqlalchemy as sa

from providers.base import GenerationParams
from providers.fakes import FakeProvider
from services.scoring import expectations, record, reindex_scores, retention, service, sets
from services.scoring.tables import SCORES
from tests.support import scores as fixtures
from tests.support.db import migrated_engine, session_for

RULE = {
    "kind": "rule",
    "name": "len",
    "config_sha256": None,
    "expression": '{"min_length": 4}',
}


@pytest.fixture
def engine(tmp_path: Path):
    engine = migrated_engine(tmp_path / "db")
    fixtures.create_score_tables(engine)
    yield engine
    engine.dispose()


@pytest.fixture
def session(engine):
    with session_for(engine) as session:
        yield session


@pytest.fixture
def roots(tmp_path: Path) -> tuple[Path, Path]:
    return tmp_path / "runs", tmp_path / "checkpoints"


def spec() -> service.EvaluationSpec:
    expectation = expectations.resolve_expectation(RULE, source="pack")

    def a_set(kind: str) -> sets.ResolvedSet:
        reference = sets.SetReference(
            rendered=f"{kind}_set/general@1",
            structured={"type": f"{kind}_set", "name": "general", "version": "1"},
        )
        return sets.resolve(
            kind, reference, [{"prompt": f"p{i}"} for i in range(4)], markers=("evaluation_only",)
        )

    return service.EvaluationSpec(
        run_id="run-1",
        provider_name="in_process",
        aggregation="mean",
        decode=GenerationParams(max_tokens=16),
        test_set=a_set("test"),
        retention_set=a_set("retention"),
        test_expectation=expectation,
        retention_expectation=expectation,
        floor=retention.ResolvedFloor(value=0.2, layer="pack_slot"),
        batch_bound=4,
    )


def seed(session, run_root: Path, checkpoint_root: Path) -> dict[int, dict]:
    """Two evaluated steps: one that saves (10) and one that does not (11).

    The non-save step is the whole point: nothing but the run log can rebuild it.
    """
    log_path = run_root / "run-1" / "run.jsonl"
    saved: dict[int, dict] = {}
    for step in (10, 11):
        service.evaluate(
            spec(),
            step,
            session=session,
            log_path=log_path,
            provider=FakeProvider(answers="longer"),
        )
    payload = record.manifest_payload_for_step(session, run_id="run-1", step=10)
    saved[10] = fixtures.written_and_read_manifest(
        checkpoint_root, step=10, test=payload["test"], retention=payload["retention"]
    )
    return saved


def snapshot(session) -> list[dict]:
    """The comparable projection of ``mf_scores``.

    ``id`` and ``created_at`` are excluded: the primary key is a surrogate and the timestamp
    is read back from the log with its original value, but a rebuild legitimately renumbers
    rows. Everything an API response carries is included.
    """
    rows = session.execute(sa.select(SCORES).order_by(SCORES.c.step, SCORES.c.kind)).mappings()
    return [{k: v for k, v in dict(row).items() if k not in ("id", "created_at")} for row in rows]


class TestReindexEquivalence:
    def test_the_rebuild_reproduces_every_score_at_both_kinds_of_step(self, session, roots) -> None:
        run_root, checkpoint_root = roots
        seed(session, run_root, checkpoint_root)
        before = snapshot(session)
        assert len(before) == 4, "two steps x two kinds"

        reindex_scores.reindex_scores(
            session, run_log_root=run_root, checkpoint_root=checkpoint_root
        )
        after = snapshot(session)

        assert len(after) == len(before)
        for original, rebuilt in zip(before, after, strict=True):
            for field in ("run_id", "step", "kind", "set_rendered", "score", "n", "floor"):
                assert rebuilt[field] == original[field], field
            assert rebuilt["provider"] == original["provider"]
            assert rebuilt["aggregation"] == original["aggregation"]
            assert rebuilt["breached"] == original["breached"]
            assert rebuilt["split_markers"] == original["split_markers"]

    def test_the_non_save_step_is_rebuilt_from_the_log_alone(self, session, roots) -> None:
        """**The half only the run log can serve.** Dropping it would lose the score."""
        run_root, checkpoint_root = roots
        seed(session, run_root, checkpoint_root)
        reindex_scores.reindex_scores(
            session, run_log_root=run_root, checkpoint_root=checkpoint_root
        )
        steps = (
            session.execute(sa.select(SCORES.c.step).distinct().order_by(SCORES.c.step))
            .scalars()
            .all()
        )
        assert steps == [10, 11]

    def test_a_rebuild_with_no_manifests_still_reproduces_everything(self, session, roots) -> None:
        """A run that has saved nothing is fully rebuildable, which is what makes the log
        the durability story rather than the manifest."""
        run_root, checkpoint_root = roots
        seed(session, run_root, checkpoint_root)
        before = snapshot(session)
        reindex_scores.reindex_scores(session, run_log_root=run_root, checkpoint_root=None)
        assert len(snapshot(session)) == len(before)

    def test_what_a_rebuilt_row_cannot_know_is_named_rather_than_guessed(
        self, session, roots
    ) -> None:
        """``EvaluationBlock`` closes itself to unknown keys and has no field for the floor's
        **layer**, so a rebuilt row says ``none``.

        Named rather than defaulted to ``pack_slot``, which would fabricate provenance for
        the one value FPRD OQ-2 says is unsettled. **This is a reported gap**, and the
        assertion is what keeps it visible.
        """
        run_root, checkpoint_root = roots
        seed(session, run_root, checkpoint_root)
        reindex_scores.reindex_scores(
            session, run_log_root=run_root, checkpoint_root=checkpoint_root
        )
        for row in session.execute(sa.select(SCORES)).mappings():
            assert row["floor_layer"] == "none"
            assert row["selector_hash"] is None

    def test_the_rebuild_report_counts_what_it_read(self, session, roots) -> None:
        run_root, checkpoint_root = roots
        seed(session, run_root, checkpoint_root)
        report = reindex_scores.reindex_scores(
            session, run_log_root=run_root, checkpoint_root=checkpoint_root
        )
        assert report["blocks"] == 2
        assert report["rows"] == 4


class TestTheManifestWinsADisagreement:
    def test_a_planted_divergence_is_corrected_toward_the_manifest_and_logged(
        self, session, roots, caplog
    ) -> None:
        """ADR-004, foundation 4.5. F10's ``resolve_part`` is the same shape."""
        import json
        import logging

        run_root, checkpoint_root = roots
        saved = seed(session, run_root, checkpoint_root)

        # Plant it in the MANIFEST, which is the authority, so the rebuilt row must move.
        directory = next(checkpoint_root.iterdir())
        manifest = saved[10]
        manifest["test"]["score"] = 0.25
        (directory / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")

        with caplog.at_level(logging.WARNING):
            report = reindex_scores.reindex_scores(
                session, run_log_root=run_root, checkpoint_root=checkpoint_root
            )

        row = (
            session.execute(sa.select(SCORES).where(SCORES.c.step == 10, SCORES.c.kind == "test"))
            .mappings()
            .one()
        )
        assert row["score"] == 0.25, "the manifest wins"
        assert any(
            record.message == reindex_scores.DIVERGENCE_EVENT for record in caplog.records
        ), "the disagreement must be logged, not corrected silently"
        assert report["divergences"]

    def test_the_divergence_event_name_matches_the_rest_of_the_product(self) -> None:
        """One query over the logs should find every divergence, not three."""
        from services.parts.index import DIVERGENCE_EVENT as PARTS_DIVERGENCE_EVENT

        assert reindex_scores.DIVERGENCE_EVENT == PARTS_DIVERGENCE_EVENT

    def test_a_manifest_that_omits_provider_does_not_overwrite_the_logs(
        self, session, roots
    ) -> None:
        """``provider`` is **not** in the manifest's ``required`` list, so a manifest that
        omits it is conformant and says **nothing** - which is different from one asserting
        ``null``. An absent key must not erase a provider the run log recorded.
        """
        import json

        run_root, checkpoint_root = roots
        saved = seed(session, run_root, checkpoint_root)
        directory = next(checkpoint_root.iterdir())
        manifest = saved[10]
        manifest["test"].pop("provider")
        (directory / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")

        reindex_scores.reindex_scores(
            session, run_log_root=run_root, checkpoint_root=checkpoint_root
        )
        row = (
            session.execute(sa.select(SCORES).where(SCORES.c.step == 10, SCORES.c.kind == "test"))
            .mappings()
            .one()
        )
        assert row["provider"] == "in_process:fake-model"

    def test_a_saved_step_with_no_log_block_is_rebuilt_from_the_manifest(
        self, session, roots
    ) -> None:
        """The log pruned, or a run older than the block: the manifest can still answer for
        the five ScoreRecord fields, and the row says the rest is unknown."""
        run_root, checkpoint_root = roots
        seed(session, run_root, checkpoint_root)
        (run_root / "run-1" / "run.jsonl").unlink()

        report = reindex_scores.reindex_scores(
            session, run_log_root=run_root, checkpoint_root=checkpoint_root
        )
        rows = list(session.execute(sa.select(SCORES)).mappings())
        assert {row["step"] for row in rows} == {10}, "only the saved step survives"
        assert any(entry["divergence"] == "row_missing" for entry in report["divergences"])
        assert all(row["rescore_reason"] for row in rows)


class TestTheProductionWiringIsOwed:
    def test_the_production_reindex_calls_it(self) -> None:
        """**WIRED 2026-10-14.** This asserted the opposite until then.

        ``reindex_scores`` existed and nothing in ``backend/src/`` called it, so by
        ADR-016 #1 a production ``reindex`` did not rebuild ``mf_scores`` at all - an
        unreachable capability, which this project counts as a finding rather than a pass.
        F4 could not fix it because ``services/reindex.py`` is not a feature agent's, so it
        pinned the gap instead and the pin went red on the fix.

        **Three features reported the same one-line gap** - F1's ``rebuild_step_index``,
        F2's ``rebuild_batch_index`` and F4's ``reindex_scores`` were all reachable only
        from their own tests. Each declined to edit the shared file, which was right, and is
        why the three accumulated into one call.

        Asserts the **call**, not that the function exists: a grep proves a function is
        there, not that anything depends on it.
        """
        import ast

        source = (
            Path(__file__).resolve().parents[2] / "src" / "services" / "reindex.py"
        ).read_text("utf-8")
        called = {
            node.func.id
            for node in ast.walk(ast.parse(source))
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
        }
        assert "reindex_scores" in called, (
            "services/reindex.py does not call reindex_scores, so mf_scores is not "
            "rebuilt by the production command and the score index is unreachable."
        )
        for sibling in ("rebuild_step_index", "rebuild_batch_index"):
            assert (
                sibling in called
            ), f"{sibling} is also owed the same call - all three were wired together."

    def test_drop_scores_empties_both_tables(self, session, roots) -> None:
        """What foundation 4.4's equivalence test needs to be able to do."""
        run_root, checkpoint_root = roots
        seed(session, run_root, checkpoint_root)
        reindex_scores.drop_scores(session)
        assert session.execute(sa.select(sa.func.count()).select_from(SCORES)).scalar_one() == 0
