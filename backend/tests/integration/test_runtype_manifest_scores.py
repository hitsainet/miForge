"""The scores reach the artifact  (ADR-016 #1; R-02.16, R-02.18; FR-004.29, .30).

**The capability this pins was implemented, tested, documented and called by nothing.**
``services.scoring.record.publish_to_manifest`` had no caller in ``backend/src``; a
``SaveContext`` carrying ``test``/``retention`` was never constructed anywhere either
(``grep -rn "SaveContext(" src/`` was empty). So both of F8's artifact writers read
``ctx.test``/``ctx.retention``, both were always ``None``, and **every manifest F8 wrote
recorded ``test: null, retention: null``** - which the contract spells out as *"no test ran
at this step"* - while ``mf_scores`` and the run log held real numbers for that very step.

Three things made it invisible:

* both properties are **nullable**, so the manifest validated clean;
* ``test_schema_sync.py`` compares schema **files**, so it cannot see a payload;
* ``null`` is the honest answer at a non-evaluation step, so the wrong answer and the right
  answer are the same bytes. *A test is the only thing that can catch this.*

For an **adapter** the loss is total rather than inconvenient: that manifest is deliberately
self-describing *so it can travel without its run log*, and a travelling adapter that says
``test: null`` cannot be told from one that was never evaluated.

## The controls, and what each is a control for

Every assertion here is on the **payload and the call count**, never on "was called" - a
reachability test that only checks the latter passes against a call sending the wrong
arguments, and the concrete wrong argument available here is *the wrong step's score*, which
is why :class:`TestTheJoinIsForThisStepAndNoOther` exists.

## The controls that were run, with their exact outcome

* **MUT-A** - ``joined_scores`` reverted to ``return ctx.test, ctx.retention`` (the state
  this file closes). **6 failed, 3 passed.** The three survivors are the right three and
  they are named: the two *"nothing ran at this step"* cases and the explicit-pass
  precedence case, none of which depends on the join. *That asymmetry is the point - the
  defect and the honest answer produce identical bytes at an unevaluated step, which is why
  reading a manifest could never have found this.*
* **MUT-B** - the adapter's own call site reverted to ``test=ctx.test,
  retention=ctx.retention``, so the shared helper stays correct and only F8's second writer
  loses it. **2 failed** - both in :class:`TestTheAdapterManifestCarriesThemToo`, which is
  what proves that writer is pinned separately from the checkpoint one rather than by
  accident of sharing a helper. Its third test survives for the same reason three survive
  MUT-A: at an unevaluated step ``ctx.test`` is ``None`` and so is the join's answer.

Both restored to 9 passed, with the source bytes compared against the backup after each
restore - *a ``replace()`` that matches nothing is indistinguishable from a survivor.*
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from services.scoring import record as score_record
from services.scoring.tables import KIND_RETENTION, KIND_TEST
from tests.support import scores as fixtures
from tests.support.db import migrated_engine, session_for
from tests.support.test_adapter_doubles import (
    FakePeftModel,
    adapter_config,
    hub_record,
    optimizer_block,
)
from training.runtypes.adapter import AdapterRunType
from training.runtypes.base import SaveContext
from training.runtypes.change import joined_scores, manifest_fields_from

RUN = "run-1"


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


def _pass(session: Any, log_path: Path, *, step: int, score: float = 0.84) -> dict[str, Any]:
    """One evaluation pass, written through F4's **real** writer.

    Through ``write_pass`` rather than an ``INSERT``, because a hand-built row agrees with
    this test's expectation by construction - and the field it would agree about omitting is
    ``provider``, which is declared and **not required**, so the suite would stay green over
    exactly the omission R-02.16 exists to make catchable.
    """
    return score_record.write_pass(
        [
            fixtures.score_record(run_id=RUN, step=step, kind=KIND_TEST, score=score, n=200),
            fixtures.score_record(
                run_id=RUN,
                step=step,
                kind=KIND_RETENTION,
                set_rendered="retention_set/general@1",
                score=0.71,
                n=120,
                floor=0.6,
                floor_layer="pack_slot",
            ),
        ],
        session=session,
        log_path=log_path,
    )


def _ctx(session: Any, *, step: int, **overrides: Any) -> SaveContext:
    base = {
        "run_id": RUN,
        "step": step,
        "session": session,
        "module": object(),
        "domain": "legal",
        "signal_source": "corpus",
        "record": hub_record(),
        "optimizer": optimizer_block(),
        "tokenizer": object(),
    }
    return SaveContext(**{**base, **overrides})


class TestTheCheckpointManifestCarriesTheStepsScores:
    def test_the_two_score_records_reach_ManifestFields(self, session, tmp_path: Path) -> None:
        """Field by field, not ``is not None``: the defect writes ``null`` and a presence
        check passes against a payload built from the wrong step."""
        _pass(session, tmp_path / "run.jsonl", step=100)
        fields = manifest_fields_from(_ctx(session, step=100), base_model="LFM2-1.2B", parent=None)
        assert fields.test == {
            "set": "test_set/general@1",
            "score": 0.84,
            "n": 200,
            "floor": None,
            "provider": "in_process:fake-model",
        }
        assert fields.retention == {
            "set": "retention_set/general@1",
            "score": 0.71,
            "n": 120,
            "floor": 0.6,
            "provider": "in_process:fake-model",
        }

    def test_the_written_manifest_on_disk_carries_them(self, session, tmp_path: Path) -> None:
        """Round-tripped through F5's real builder and off the disk. The artifact is the
        thing that travels, so the artifact is what is asserted."""
        _pass(session, tmp_path / "run.jsonl", step=100)
        fields = manifest_fields_from(_ctx(session, step=100), base_model="LFM2-1.2B", parent=None)
        manifest = fixtures.written_and_read_manifest(
            tmp_path / "ckpt", run_id=RUN, step=100, test=fields.test, retention=fields.retention
        )
        assert manifest["test"]["score"] == 0.84
        assert manifest["test"]["provider"] == "in_process:fake-model"
        assert manifest["retention"]["floor"] == 0.6

    def test_a_save_at_a_step_where_nothing_ran_still_writes_both_nulls(
        self, session, tmp_path: Path
    ) -> None:
        """E-22. ``null`` is an assertion that nothing ran; an absent key is an omission.
        **This is the case the defect was indistinguishable from.**"""
        _pass(session, tmp_path / "run.jsonl", step=100)
        fields = manifest_fields_from(_ctx(session, step=101), base_model="LFM2-1.2B", parent=None)
        assert fields.test is None
        assert fields.retention is None


class TestTheJoinIsForThisStepAndNoOther:
    def test_the_call_is_made_exactly_once_with_this_steps_identity(
        self, session, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """**The payload AND the count.** A join for the wrong step produces a manifest
        that validates, reads plausibly and reports another step's number as this
        checkpoint's."""
        _pass(session, tmp_path / "run.jsonl", step=100)
        calls: list[dict[str, Any]] = []
        real = score_record.publish_to_manifest

        def spy(session_arg: Any, *, run_id: str, step: int, publish: Any) -> Any:
            calls.append({"run_id": run_id, "step": step, "session": session_arg})
            return real(session_arg, run_id=run_id, step=step, publish=publish)

        monkeypatch.setattr("training.runtypes.change.score_record.publish_to_manifest", spy)
        manifest_fields_from(_ctx(session, step=100), base_model="LFM2-1.2B", parent=None)

        assert len(calls) == 1, f"one join per save, not {len(calls)}"
        assert calls[0]["run_id"] == RUN
        assert calls[0]["step"] == 100
        assert calls[0]["session"] is session

    def test_another_steps_scores_are_not_borrowed(self, session, tmp_path: Path) -> None:
        log = tmp_path / "run.jsonl"
        _pass(session, log, step=100, score=0.10)
        _pass(session, log, step=200, score=0.90)
        at_100 = manifest_fields_from(_ctx(session, step=100), base_model="LFM2-1.2B", parent=None)
        at_200 = manifest_fields_from(_ctx(session, step=200), base_model="LFM2-1.2B", parent=None)
        assert at_100.test is not None and at_100.test["score"] == 0.10
        assert at_200.test is not None and at_200.test["score"] == 0.90

    def test_an_explicitly_supplied_pass_wins_and_no_join_happens(
        self, session, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A caller that already holds the pass is not made to re-read the index. Asserted
        on the count, so the precedence is a fact rather than a comment."""
        _pass(session, tmp_path / "run.jsonl", step=100)
        calls: list[int] = []
        monkeypatch.setattr(
            "training.runtypes.change.score_record.publish_to_manifest",
            lambda *a, **k: calls.append(1),
        )
        supplied = {"set": "s", "score": 0.5, "n": 1, "floor": None, "provider": None}
        test, retention = joined_scores(_ctx(session, step=100, test=supplied))
        assert test == supplied
        assert retention is None
        assert calls == [], "an explicit pass must not trigger an index read"


class TestTheAdapterManifestCarriesThemToo:
    """The artifact built to travel **without** its run log (R-02.44)."""

    def _save(self, session: Any, tmp_path: Path, *, step: int) -> dict[str, Any]:
        ctx = _ctx(
            session,
            step=step,
            module=FakePeftModel(),
            adapter=adapter_config(),
            root=tmp_path / "adapters",
        )
        return AdapterRunType().save(ctx)

    def test_the_published_adapter_manifest_carries_both_scores(
        self, session, tmp_path: Path
    ) -> None:
        _pass(session, tmp_path / "run.jsonl", step=100)
        published = self._save(session, tmp_path, step=100)
        assert published["test"] == {
            "set": "test_set/general@1",
            "score": 0.84,
            "n": 200,
            "floor": None,
            "provider": "in_process:fake-model",
        }
        assert published["retention"]["floor"] == 0.6

    def test_the_manifest_ON_DISK_carries_them(self, session, tmp_path: Path) -> None:
        """Read off the written directory, because that is the copy that travels."""
        _pass(session, tmp_path / "run.jsonl", step=100)
        self._save(session, tmp_path, step=100)
        written = list((tmp_path / "adapters").rglob("adapter_manifest.json"))
        assert len(written) == 1, f"one adapter manifest, not {len(written)}"
        document = json.loads(written[0].read_text(encoding="utf-8"))
        assert document["test"]["score"] == 0.84
        assert document["test"]["provider"] == "in_process:fake-model"
        assert document["retention"]["score"] == 0.71

    def test_an_adapter_save_at_an_unevaluated_step_writes_both_nulls(
        self, session, tmp_path: Path
    ) -> None:
        _pass(session, tmp_path / "run.jsonl", step=100)
        published = self._save(session, tmp_path, step=101)
        assert published["test"] is None
        assert published["retention"] is None
