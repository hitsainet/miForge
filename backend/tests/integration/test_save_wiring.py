"""The weights save is reachable from the production loop  (ADR-016 #1).

**Why this file exists.** F5 built the whole save path — the writer, the manifest, the
atomic directory, the index upsert, the event — and reported, correctly, that *"until it
lands, nothing in production calls `save()`"*. By this project's own rule that means the
save path was **not shipped**: a capability is not shipped until a test fails when its
wiring is removed. It is the shape of the sibling that shipped sixteen fully-implemented,
unit-tested, documented MCP tools that were never registered with its server — every one of
those tests passed by importing the module directly, so the suite was green and the docs
said ✅ while the feature did not exist for any caller.

So this file asserts the **call**, its **payload** and its **call count**, not that a
function exists. "Was called" passes against a call sending the wrong arguments, which is
why the payload is checked too.

**It also pins the honest half.** The save needs two values the training feature does not
own — `signal_source` (F2's: which source filled the batch) and `domain` (F6's: off
R-02.53's run definition). `ManifestFields` gives neither a default, deliberately, because
*null is a statement and an absent key is an omission* and a defaulted `signal_source`
would write a fabricated provenance claim into a published artifact. Until F6 supplies
them the save point writes trainer state and **logs loudly which feature owes the value**,
rather than skipping in silence. Both halves are asserted here, because a pass-through
nobody can see is indistinguishable from a feature that was never wired.
"""

from __future__ import annotations

import logging
from typing import Any

import pytest

from services.checkpoints import requests as save_requests
from services.checkpoints import service as checkpoint_service
from training import tasks
from training.acquisition.base import AcquiredModel

from .test_step_loop import RECORD, RUN, drive, rig  # noqa: F401 - `rig` is a fixture

pytestmark = pytest.mark.usefixtures("rig")


class _SaveSpy:
    """Records every call, with its arguments. Counts as well as records.

    A spy rather than a mock: the assertion is about *what was sent*, and a mock that only
    answers "were you called" passes against a save carrying the wrong step.
    """

    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []

    def __call__(self, model: Any, tokenizer: Any, **kwargs: Any) -> Any:
        self.calls.append({"model": model, "tokenizer": tokenizer, **kwargs})

        class _Outcome:
            directory = f"/data/checkpoints/{RUN}/step-{kwargs['fields'].step:06d}"

        return _Outcome()


class _AttachedSaverSpy:
    """Stands in for ``training.hotupdate.save.AttachedSaver``, recording its arguments.

    Deliberately **not** the real class: the real one needs a configured §5.10 client and a
    serving fake, which is what its own unit tests give it. What this file tests is the
    *loop's* dispatch, and a double is the right instrument for that - provided it cannot
    accidentally satisfy the assertion, which is why it records the payload rather than a
    boolean.
    """

    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []

    def __call__(self, **kwargs: Any) -> Any:
        self.calls.append(dict(kwargs))

        class _Outcome:
            directory = f"/data/checkpoints/{RUN}/step-{kwargs['fields'].step:06d}"

        return _Outcome()


@pytest.fixture
def spy(monkeypatch: pytest.MonkeyPatch) -> _SaveSpy:
    recorder = _SaveSpy()
    # F5's own module, not `tasks.checkpoint_service`: since the loop was routed through
    # `RunType.save` the only importer of the writer is `training/runtypes/change.py`, and
    # `training/tasks.py` no longer names it. Patching the module object is what both
    # reached anyway - this just says so.
    monkeypatch.setattr(checkpoint_service, "save", recorder)
    return recorder


class TestTheLoopActuallyCallsSave:
    def test_a_save_point_calls_save_once_with_the_right_step(self, rig, spy) -> None:  # noqa: F811
        """The reachability assertion: the call, the count AND the payload."""
        result = drive(rig, max_steps=4, save_every=2, signal_source="corpus", domain="legal")

        assert len(spy.calls) == 2, (
            f"expected a save at steps 2 and 4, got {len(spy.calls)}. If this is 0 the "
            f"wiring in training/tasks.py is gone and the whole save path is unreachable "
            f"from production - which is exactly what this file exists to catch."
        )
        assert [call["fields"].step for call in spy.calls] == [2, 4]
        assert result.checkpoints == [
            f"/data/checkpoints/{RUN}/step-000002",
            f"/data/checkpoints/{RUN}/step-000004",
        ]

    def test_the_payload_carries_the_provenance_it_was_given(self, rig, spy) -> None:  # noqa: F811
        """Not merely "was called". A save carrying the wrong provenance writes a manifest
        that validates, saves, and misattributes the artifact forever."""
        drive(rig, max_steps=1, save_every=1, signal_source="self_generation", domain=None)

        fields = spy.calls[0]["fields"]
        assert fields.run_id == RUN
        assert fields.signal_source == "self_generation"
        assert fields.domain is None, (
            "None must reach the manifest as an explicit null - it is an assertion that the "
            "run is not domain-scoped, and an absent key would be an omission instead"
        )
        assert fields.base_model, "the lineage root must be carried, not left empty"

    def test_no_save_point_means_no_save(self, rig, spy) -> None:  # noqa: F811
        """The negative half. Without it, a mutation that saved on *every* step would pass
        the count assertion above only by coincidence of the cadence chosen there."""
        drive(rig, max_steps=3, save_every=None, signal_source="corpus", domain="legal")
        assert spy.calls == []

    def test_trainer_state_and_the_checkpoint_land_at_the_same_step(
        self, rig, spy  # noqa: F811
    ) -> None:
        """R-02.15: *"on the same cadence as weight saves"*, and for the same step number.

        A trainer state at step 2 beside a checkpoint at step 3 is a continue that resumes
        the optimizer to a different point than the weights - the unexplained loss
        discontinuity R-02.22 exists to prevent.
        """
        result = drive(rig, max_steps=2, save_every=2, signal_source="corpus", domain="legal")
        assert [call["fields"].step for call in spy.calls] == [2]
        assert len(result.trainer_states) == 1
        assert result.trainer_states[0].endswith("/step-000002")


class TestTheHonestPassThrough:
    def test_without_a_signal_source_no_checkpoint_is_written(self, rig, spy) -> None:  # noqa: F811
        """`signal_source` is required and **not nullable** in the manifest.

        So there is nothing to pass. The alternative - a placeholder - would write a
        fabricated provenance claim into a published artifact, and `ManifestFields` has no
        default precisely to make that impossible rather than merely discouraged.
        """
        result = drive(rig, max_steps=2, save_every=1)  # no signal_source
        assert spy.calls == []
        assert result.checkpoints == []

    def test_the_run_is_still_continuable(self, rig, spy) -> None:  # noqa: F811
        """Trainer state is still written, so the cadence is not simply lost."""
        result = drive(rig, max_steps=2, save_every=1)
        assert len(result.trainer_states) == 2

    def test_the_skip_is_logged_loudly_and_names_who_owes_the_value(
        self,
        rig,  # noqa: F811
        spy,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        """**Skipping in silence is the one thing not done.**

        An operator whose save cadence passed with no checkpoint on disk must be told why
        and by whom, or the only symptom is an empty checkpoint list and a healthy-looking
        run. Asserted on the record's structured fields, not on `caplog.text` - ADR-021's
        `extra` fields never appear in the rendered message, so a text assertion would pass
        against a line that named nothing.
        """
        with caplog.at_level(logging.WARNING):
            drive(rig, max_steps=1, save_every=1)

        skips = [
            r
            for r in caplog.records
            if getattr(r, "code", "") == "checkpoint_save_skipped_no_provenance"
        ]
        assert skips, "the skip was silent"
        record = skips[0]
        assert record.levelno >= logging.WARNING, "a silent-failure path warrants a warning"
        assert "F2" in " ".join(getattr(record, "owed_by", []))
        assert "F6" in " ".join(getattr(record, "owed_by", []))
        assert "continuable" in getattr(record, "reason", ""), (
            "the message must say the run is still continuable, or the reader cannot tell "
            "whether this cost them the run"
        )


class TestTheOperatorRequestedSave:
    def test_a_request_is_consumed_at_the_next_boundary(self, rig, spy) -> None:  # noqa: F811
        """A request produces a save at the next step boundary, off-cadence."""
        from tests.support.jobs import session_for

        with session_for(rig["engine"]) as session:
            save_requests.request_save(session, RUN)
            session.commit()

        drive(rig, max_steps=1, save_every=None, signal_source="corpus", domain="legal")
        assert [call["fields"].step for call in spy.calls] == [
            1
        ], "an operator's request did not produce a save at the next boundary"

    def test_two_requests_collapse_into_one_save(self, rig, spy) -> None:  # noqa: F811
        """The effect collapses; the *fact* that two were asked for does not.

        One save is what was wanted. But `save_request_count` keeps both, so an operator who
        pressed the button twice can see the system noticed - which is the difference between
        a collapsed request and an ignored one.
        """
        from tests.support.jobs import session_for

        with session_for(rig["engine"]) as session:
            save_requests.request_save(session, RUN)
            pending = save_requests.request_save(session, RUN)
            session.commit()
        assert pending["request_count"] == 2

        drive(rig, max_steps=1, save_every=None, signal_source="corpus", domain="legal")
        assert len(spy.calls) == 1

    def test_the_request_does_not_survive_its_boundary(self, rig, spy) -> None:  # noqa: F811
        """Consumed, not flagged — otherwise the next boundary saves again."""
        from tests.support.jobs import session_for

        with session_for(rig["engine"]) as session:
            save_requests.request_save(session, RUN)
            session.commit()

        drive(rig, max_steps=3, save_every=None, signal_source="corpus", domain="legal")
        assert len(spy.calls) == 1, "the request was served more than once"


class TestAnAttachedAcquisitionTakesItsOwnSavePath:
    """**ADR-016 #1, the strong form.** OQ-5 / AMB-M, settled 2026-10-14: F11 owns it.

    An attached acquisition holds **neither the tensors nor the tokenizer**, so each test
    below builds its own ``AcquiredModel(..., None, RECORD, saver=saver)`` with an explicit
    ``None`` tokenizer. In production F5's writer calls ``tokenizer.save_pretrained``, so
    before this dispatch existed the first save point of every attached run raised
    ``AttributeError: 'NoneType' object has no attribute 'save_pretrained'`` - a traceback,
    not a refusal, exactly where a run first tries to checkpoint.

    *This paragraph used to add "the rig above already builds it that way, and the only
    reason that has never raised is that ``_SaveSpy`` stands in for the writer" - true when
    written, and false from 2026-10-16.* The rig now supplies a tokenizer, because the
    standalone save goes through
    :meth:`training.runtypes.change.ChangeRunType.save`, which **refuses**
    ``tokenizer_required`` rather than handing the writer a ``None`` (R-02.21: a checkpoint
    carries the tokenizer's own files at its root). The ``None`` here is therefore the
    attached case asserting itself rather than a default nobody chose - which is the right
    way round, since it is the one mode for which it is true.

    **Why this lives here and not beside the saver's unit tests.** The unit file's first
    attempt at a reachability test asserted ``"acquired.saver" in <source of tasks.py>``,
    and the mutation that removes the dispatch condition (``elif acquired.saver is not
    None:`` -> ``elif False:``) **left that string in place inside the branch body, so the
    mutation survived and 20 tests passed.** A grep proves a call exists, not that anything
    depends on it. These assert the live call, its payload, its count, **and that F5's
    writer is not reached** - and the same mutation turns three of them red.
    """

    def test_the_acquisitions_own_saver_is_called_and_F5s_writer_is_NOT(
        self,
        rig,  # noqa: F811
        spy,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        saver = _AttachedSaverSpy()
        monkeypatch.setattr(
            tasks,
            "acquire_for_run",
            lambda spec: AcquiredModel(rig["model"], None, RECORD, saver=saver),
        )

        result = drive(rig, max_steps=4, save_every=2, signal_source="corpus", domain="legal")

        assert len(saver.calls) == 2, (
            f"expected the attached save path at steps 2 and 4, got {len(saver.calls)}. If "
            f"this is 0 the dispatch in training/tasks.py is gone, and an attached run "
            f"falls through to F5's writer with tokenizer=None."
        )
        assert spy.calls == [], (
            "F5's writer was reached for an acquisition that brought its own save path. In "
            "attached mode the resident tensors are the serving process's and R-01.10-13 "
            "govern the artifact; calling F5 here would write a second checkpoint from a "
            "module that holds neither the tensors nor the tokenizer."
        )
        assert [call["fields"].step for call in saver.calls] == [2, 4]
        assert result.checkpoints == [
            f"/data/checkpoints/{RUN}/step-000002",
            f"/data/checkpoints/{RUN}/step-000004",
        ]

    def test_it_receives_the_SAME_provenance_F5_would_have(
        self,
        rig,  # noqa: F811
        spy,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """One ``ManifestFields``, built once in the loop and handed to whichever path runs.

        If the two branches built their own, the attached manifest would drift from the
        standalone one on the first field anyone added - and the drift would be invisible,
        because each branch's tests would pass.
        """
        saver = _AttachedSaverSpy()
        monkeypatch.setattr(
            tasks,
            "acquire_for_run",
            lambda spec: AcquiredModel(rig["model"], None, RECORD, saver=saver),
        )
        drive(rig, max_steps=1, save_every=1, signal_source="self_generation", domain=None)

        fields = saver.calls[0]["fields"]
        assert fields.run_id == RUN
        assert fields.signal_source == "self_generation"
        assert fields.domain is None
        assert fields.base_model, "the lineage root must be carried, not left empty"

    def test_no_save_point_means_no_attached_save_either(
        self,
        rig,  # noqa: F811
        spy,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """The negative half, so the count above is not satisfied by saving every step."""
        saver = _AttachedSaverSpy()
        monkeypatch.setattr(
            tasks,
            "acquire_for_run",
            lambda spec: AcquiredModel(rig["model"], None, RECORD, saver=saver),
        )
        drive(rig, max_steps=3, save_every=None, signal_source="corpus", domain="legal")
        assert saver.calls == []
        assert spy.calls == []

    def test_an_acquisition_WITHOUT_a_saver_still_goes_to_F5(
        self,
        rig,  # noqa: F811
        spy,
    ) -> None:
        """The other direction, which is the one ten of eleven features depend on.

        ``load`` and ``create`` leave ``saver`` as ``None``, so the default path must be
        untouched by the dispatch. A mutation that routed *everything* through the new
        branch would pass every assertion above and break the whole product.
        """
        drive(rig, max_steps=2, save_every=2, signal_source="corpus", domain="legal")
        assert len(spy.calls) == 1
        assert spy.calls[0]["fields"].step == 2
