"""The production save carries this step's scores into the manifest  (R-02.16, R-02.18).

**The other half of the gap ``test_runtype_manifest_scores.py`` closes.** That file pins
F8's two artifact writers. This one pins the writer the *loop* actually calls:
``training.tasks.drive_run``'s save point, which built its own ``ManifestFields`` - twice,
once per save branch - and passed **neither ``test`` nor ``retention``**. So every
production checkpoint manifest recorded ``test: null, retention: null``, which the contract
defines as *"Null when no test ran at this step"*, while ``mf_scores`` and the run log held
real numbers for that very step. R-02.16's provenance was absent from the one artifact that
travels, and F11's attached save sent the same empty pair upstream.

Three things made it invisible, and the third is why a test is the only instrument that can
find it:

* both properties are **nullable**, so the manifest validated clean;
* ``test_schema_sync.py`` compares schema **files**, so it cannot see a payload;
* **at a step where nothing was evaluated, the defect and the correct answer are identical
  bytes.** Reading a manifest - any manifest, however carefully - can never distinguish
  them. So every assertion that could be satisfied by ``null`` is paired with one at a step
  that *was* evaluated, and the "nothing ran here" case is asserted **separately** so that
  its ``null`` is proved to mean absence rather than loss.

## The scores arrive the way production puts them there

Through F4's real writer (:func:`services.scoring.record.write_pass`) from the loop's own
``evaluate`` hook, which the boundary calls **before** the save at the same step
(FR-006.7's order, and the reason M-7 exists). Not an ``INSERT``: a hand-built row agrees
with this file's expectation by construction, and the field it would agree about omitting is
``provider`` - declared and **not required** - so the suite would stay green over exactly
the omission R-02.16 exists to make catchable.

## THE TWO CONTROLS, AND WHAT EACH SURVIVOR MEANS

⚠ **Both controls name lines that MOVED on 2026-10-16**, when the loop was routed through
``RunType.save`` and ``training/tasks.py``'s own projection of these two payloads collapsed
into F8's one join. Re-point them before running either, or the mutation matches nothing -
*and a ``replace()`` that matches nothing is indistinguishable from a survivor*, which is
the trap this file's last paragraph is about:

* **MUT-10** is now the ``test=test, retention=retention`` pair in
  :func:`training.runtypes.change.manifest_fields_from`, which both the checkpoint writer
  and F11's attached save reach. Its outcome below was measured against the old site.
* **MUT-11** is now the ``if ctx.session is None:`` early return in
  :func:`training.runtypes.change.joined_scores` - force it to ``if True:`` rather than the
  old ``if False:``, since the sense of the guard is inverted at its new home.

* **MUT-10** - ``test=test_record, retention=retention_record`` deleted from
  ``manifest_fields_for`` in ``training/tasks.py``, which is exactly the state this file
  closes. **4 failed, 2 passed**, and both survivors are named here rather than treated as
  gaps:

  * :meth:`~TestTheManifestCarriesThisStepsScores.
    test_a_step_with_no_evaluation_carries_a_null_that_is_present` survives **because
    the defect and the correct answer are the same bytes there.** That asymmetry is the
    whole reason this file exists;
  * :meth:`~TestTheManifestCarriesThisStepsScores.
    test_one_join_per_save_through_f4s_published_seam` survives because the join still
    *ran* - its answer was simply thrown away. **That is the sharpest lesson available
    here:** a call-count assertion is not a payload assertion, and this one passes against
    a join whose result reaches nothing. It is kept because it bites on the other axis -

* **MUT-11** - the ``if session is not None:`` guard forced to ``if False:``, so the join is
  never reached at all. **5 failed, 1 passed**; only the identical-bytes case survives, and
  the call-count test is red. So the two controls between them pin *the join happening* and
  *its answer arriving*, which no single assertion does.

Both restored, with the source bytes compared against the backup after each restore - *a
``replace()`` that matches nothing is indistinguishable from a survivor.*
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from services.checkpoints import service as checkpoint_service
from services.scoring import record as score_record
from services.scoring.tables import KIND_RETENTION, KIND_TEST
from tests.support import scores as fixtures
from tests.support.jobs import session_for
from training import tasks
from training.acquisition.base import AcquiredModel

from .test_save_wiring import _AttachedSaverSpy, _SaveSpy
from .test_step_loop import RECORD, RUN, drive, rig  # noqa: F401 - `rig` is a fixture

pytestmark = pytest.mark.usefixtures("rig")

#: The in-domain score written at the evaluated step. Distinct from the retention score
#: below and from `OTHER_SCORE`, so a payload joined from the wrong kind or the wrong step
#: is a different number rather than a coincidence.
TEST_SCORE = 0.84
RETENTION_SCORE = 0.71
OTHER_SCORE = 0.42


@pytest.fixture
def loop(rig):  # noqa: F811 - the parameter requests the imported fixture
    """The loop rig under a local name.

    A one-line alias so that no test signature below shadows the imported ``rig`` fixture -
    which is a ``# noqa: F811`` on every one of them, and a reader cannot tell a suppressed
    shadow from a suppressed real redefinition.
    """
    return rig


@pytest.fixture
def spy(monkeypatch: pytest.MonkeyPatch) -> _SaveSpy:
    recorder = _SaveSpy()
    # F5's own module, not `tasks.checkpoint_service`: since the loop was routed through
    # `RunType.save` the only importer of the writer is `training/runtypes/change.py`, and
    # `training/tasks.py` no longer names it. Patching the module object is what both
    # reached anyway - this just says so.
    monkeypatch.setattr(checkpoint_service, "save", recorder)
    return recorder


class _JoinSpy:
    """Counts :func:`services.scoring.record.publish_to_manifest` and **calls through**.

    Counting without delegating would replace the thing under test with its own counter -
    the shape this project records as *a test that reimplements what it tests cannot fail.*
    """

    def __init__(self, real: Any) -> None:
        self._real = real
        self.count = 0

    def __call__(self, session: Any, **kwargs: Any) -> Any:
        self.count += 1
        return self._real(session, **kwargs)


def _pass(session: Any, log_path: Path, *, step: int, score: float) -> None:
    """One evaluation pass at ``step``, through F4's real writer."""
    score_record.write_pass(
        [
            fixtures.score_record(run_id=RUN, step=step, kind=KIND_TEST, score=score, n=200),
            fixtures.score_record(
                run_id=RUN,
                step=step,
                kind=KIND_RETENTION,
                set_rendered="retention_set/general@1",
                score=RETENTION_SCORE,
                n=120,
                floor=0.6,
                floor_layer="pack_slot",
            ),
        ],
        session=session,
        log_path=log_path,
    )


def _run(loop, *, evaluated: dict[int, float], steps: int = 2, **overrides):
    """Drive the real loop, evaluating exactly the steps in ``evaluated``.

    ``evaluate`` is the loop's own hook and the boundary calls it **before** the save at the
    same step, so the ordering here is production's rather than a fixture's convenience.
    """
    with session_for(loop["engine"]) as session:
        log_path = loop["log_root"] / "scores.jsonl"

        def evaluate(step: int) -> None:
            if step in evaluated:
                _pass(session, log_path, step=step, score=evaluated[step])

        return (
            drive(
                loop,
                session=session,
                max_steps=steps,
                save_every=1,
                signal_source="corpus",
                domain="legal",
                evaluate=evaluate,
                **overrides,
            ),
            session,
        )


class TestTheManifestCarriesThisStepsScores:
    def test_the_two_score_records_reach_the_save(self, loop, spy) -> None:
        """Field by field, never ``is not None``.

        A presence check would pass against a payload joined from the **wrong step**, which
        is the concrete wrong argument available here - and a manifest carrying another
        step's number is worse than one carrying none, because it is believable.
        """
        _run(loop, evaluated={1: TEST_SCORE}, steps=1)

        assert len(spy.calls) == 1, f"expected one save, got {len(spy.calls)}"
        fields = spy.calls[0]["fields"]
        assert fields.step == 1

        assert fields.test is not None, (
            "the manifest recorded `test: null` for a step that WAS evaluated - which the "
            "contract reads as 'no test ran at this step'. The scores are in `mf_scores` "
            "and the run log and absent from the artifact that travels (R-02.16)."
        )
        assert fields.test["score"] == TEST_SCORE
        assert fields.test["n"] == 200
        assert fields.test["set"] == "test_set/general@1"
        assert fields.test["floor"] is None
        # `provider` is declared and NOT required, so a manifest omitting it validates
        # clean. It is asserted here because that is the only thing that can catch its
        # absence - R-02.16: a score is only comparable against one taken the same way.
        assert fields.test["provider"], "every score records its provider (ADR-009)"

        assert fields.retention is not None
        assert fields.retention["score"] == RETENTION_SCORE
        assert fields.retention["n"] == 120
        assert fields.retention["floor"] == 0.6
        assert fields.retention["set"] == "retention_set/general@1"

    def test_a_step_with_no_evaluation_carries_a_null_that_is_present(self, loop, spy) -> None:
        """The case the defect and the fix agree on - asserted **separately** on purpose.

        This is the one test here that survives the control, and it is what proves the
        ``null`` written at an unevaluated step means *absence* rather than *loss*: both
        keys are **present** in the payload handed to the writer, because ``null`` is an
        assertion and an absent key is an omission (E-22).
        """
        _run(loop, evaluated={}, steps=1)

        fields = spy.calls[0]["fields"]
        assert fields.test is None
        assert fields.retention is None
        kwargs = fields.as_kwargs()
        assert "test" in kwargs and "retention" in kwargs, (
            "both keys must reach `build` even when nothing ran: an absent key is an "
            "omission, a null is a statement that no evaluation happened"
        )

    def test_the_join_is_for_this_step_and_no_other(self, loop, spy) -> None:
        """Two steps, two different scores, each save carrying its own.

        Without this, a join that read *the latest row for the run* rather than the latest
        row **for this step** would pass every other assertion in the file - and would
        stamp step 1's number onto step 2's artifact, where nothing could ever contradict
        it.
        """
        _run(loop, evaluated={1: TEST_SCORE, 2: OTHER_SCORE}, steps=2)

        assert [call["fields"].step for call in spy.calls] == [1, 2]
        assert [call["fields"].test["score"] for call in spy.calls] == [TEST_SCORE, OTHER_SCORE]

    def test_a_save_at_an_unevaluated_step_does_not_inherit_the_last_one(self, loop, spy) -> None:
        """Step 1 evaluated, step 2 not. Step 2's manifest must say ``null``, not 0.84.

        The sharper half of the test above: inheriting is the failure mode a per-run join
        produces, and it is indistinguishable from a real re-score once written.
        """
        _run(loop, evaluated={1: TEST_SCORE}, steps=2)

        assert [call["fields"].step for call in spy.calls] == [1, 2]
        assert spy.calls[0]["fields"].test is not None
        assert spy.calls[0]["fields"].test["score"] == TEST_SCORE
        assert (
            spy.calls[1]["fields"].test is None
        ), "step 2 was not evaluated, so its manifest must not carry step 1's score"
        assert spy.calls[1]["fields"].retention is None

    def test_one_join_per_save_through_f4s_published_seam(
        self, loop, spy, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The call count, and that the join goes through F4's seam rather than a copy.

        ``publish_to_manifest`` is the seam F4 published for this; a second projection of
        the same two payloads here would be a fourth copy of a shape that already has three
        agreeing ones. The spy delegates, so the numbers asserted above are still the real
        join's.
        """
        join = _JoinSpy(score_record.publish_to_manifest)
        monkeypatch.setattr(score_record, "publish_to_manifest", join)

        _run(loop, evaluated={1: TEST_SCORE, 2: OTHER_SCORE}, steps=2)

        assert len(spy.calls) == 2
        assert join.count == 2, (
            f"the join ran {join.count} time(s) for 2 saves. Zero means the manifest's "
            f"scores are not read from the index at all; more than one per save means a "
            f"second read whose answer nothing asserts."
        )


class TestTheAttachedSaveGetsTheSameProvenance:
    """F11's attached save sends the trainer's fields upstream, and they are the same fields.

    Both properties are annotated *"Written by the trainer"* in
    ``checkpoint-manifest-v1.json``, so :func:`training.hotupdate.save.caller_manifest`
    passes them through rather than rejecting them - which means the §5.10 path lost exactly
    the same provenance, from the same two literal blocks, and is fixed by the same one
    construction. Pinned separately because *two call sites sharing a helper today can stop
    sharing it tomorrow.*
    """

    def test_the_attached_saver_receives_the_scores(
        self, loop, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        saver = _AttachedSaverSpy()
        monkeypatch.setattr(
            tasks,
            "acquire_for_run",
            lambda spec: AcquiredModel(loop["model"], None, RECORD, saver=saver),
        )

        _run(loop, evaluated={1: TEST_SCORE}, steps=1)

        assert len(saver.calls) == 1, "the attached branch must be the one that fired"
        fields = saver.calls[0]["fields"]
        assert fields.step == 1
        assert fields.test is not None and fields.test["score"] == TEST_SCORE
        assert fields.retention is not None and fields.retention["score"] == RETENTION_SCORE
