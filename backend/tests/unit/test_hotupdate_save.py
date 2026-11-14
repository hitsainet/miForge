"""The attached-mode save: F11 decides, miLLM writes, F5 indexes  (FR-011.26, FR-011.56).

**OQ-5 / AMB-M, settled 2026-10-14: F11 owns it.** ADR-022 decides it - an attached-mode
branch inside F5's save service would put a miLLM endpoint reference on the core path and
turn the standalone control red.

## What these tests are for, beyond the happy path

Three things could each ship a save path that looks right:

1. **The caller's manifest fields are derived from the schema**, not listed in code. If the
   derivation is wrong, miLLM answers ``400 MANIFEST_FIELD_NOT_CALLERS`` on **every** save
   of **every** attached run. So the fake validates ownership with its own **literal** set
   and production derives one - *a test that reimplements what it tests cannot fail*, and
   the disagreement between a literal and a derivation is the whole check.
2. **A successful save can rename the resident model**, which is the run's identity. A
   baseline taken at attach is stale the moment the first save succeeds, so the *second*
   save refuses ``weights_identity_changed`` and blames a serving restart that never
   happened. The fake renames by default because that is what miLLM does; production
   **re-reads** rather than encoding either answer, and both readings are tested.
3. **`require_clean` had only two of its three documented call sites.** The third - before
   a save - had no production caller at all, because there was no save path. The test for
   it performed the sequence itself, so the gate function was protected while **the wiring
   to a save was not** (F11 FTASKS 10.10 recorded exactly this).

Mutation controls, each run and each seen red (2026-10-14):

* **MUT-S1** drop ``require_clean`` from the saver -> ``TestThePrecondition`` red.
* **MUT-S2** send ``fields.as_kwargs()`` whole -> ``TestTheCallersFieldsAreDerived`` red
  **through the fake's own validator**, which is the shape of the real refusal.
* **MUT-S3** keep the attach baseline instead of re-reading -> ``TestTheIdentityIsRebaselined``
  red on the second save.
* **MUT-S4** remove the dispatch in ``training/tasks.py`` -> ``TestTheLoopReachesIt`` red.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from core.errors import Refusal
from tests.support.checkpoints import fields as manifest_fields
from tests.support.fixtures_hotupdate import Harness
from training.hotupdate import save as save_module


@pytest.fixture
def saver(harness: Harness) -> save_module.AttachedSaver:
    """A saver over an attached client, as ``AttachAcquirer.acquire`` builds one."""
    harness.attach()
    from training.hotupdate import state

    _, identity = state.read(harness.client)
    return save_module.AttachedSaver(harness.client, expected=identity)


class _Session:
    """Enough session for the index call, which is asserted by effect and not by mock."""

    def __init__(self) -> None:
        self.flushes = 0

    def get(self, *_: Any, **__: Any) -> None:
        return None

    def add(self, _row: Any) -> None:
        self.rows_added = getattr(self, "rows_added", 0) + 1

    def flush(self) -> None:
        self.flushes += 1


class TestTheCallersFieldsAreDerived:
    """R-01.12: *"each field is marked as written by the caller or written by miLLM."*"""

    def test_the_serving_written_set_comes_out_of_the_schema(self) -> None:
        """The four miLLM writes in this mode, **computed from the normative file**.

        Written as an expected value here only because this is the test: production never
        contains the list. If the schema's wording changes, this assertion is where it
        surfaces - which is better than a save refused upstream on every step.
        """
        assert save_module.serving_written_fields() == frozenset(
            {"base_model", "created_at", "name", "weights_sha256"}
        )

    def test_an_unreadable_marking_refuses_rather_than_sending_everything(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """An empty derivation would classify every field as the caller's."""
        monkeypatch.setattr(save_module, "_SERVING_WRITER_MARKER", "no such sentence")
        with pytest.raises(Refusal) as caught:
            save_module.serving_written_fields()
        assert caught.value.code == "manifest_writer_marking_unreadable"

    def test_the_body_drops_what_milllm_writes_and_keeps_what_it_does_not(self) -> None:
        body = save_module.caller_manifest(manifest_fields())
        assert "base_model" not in body and "created_at" not in body
        assert body["run_id"] == "run-a" and body["step"] == 500
        assert body["signal_source"] == "corpus"

    def test_a_null_the_schema_admits_is_KEPT_not_dropped(self) -> None:
        """*Null is a statement and an absent key is an omission.* A dropped ``domain`` is a
        general-pretraining run that forgot to say so."""
        body = save_module.caller_manifest(manifest_fields(domain=None, grader_hash=None))
        assert body["domain"] is None
        assert "domain" in body and "grader_hash" in body

    def test_a_field_with_no_agreed_home_is_refused_rather_than_sent(self) -> None:
        """``extra`` reaches another repository's validator as ``INVALID_MANIFEST``."""
        fields = manifest_fields()
        object.__setattr__(fields, "extra", {"tolerance": 0.01})
        with pytest.raises(Refusal) as caught:
            save_module.caller_manifest(fields)
        assert caught.value.code == "attached_save_extra_fields"
        assert caught.value.details["extra_fields"] == ["tolerance"]

    def test_the_SERVING_SIDE_accepts_the_derived_body(
        self, saver: save_module.AttachedSaver, harness: Harness
    ) -> None:
        """**MUT-S2.** The derivation is checked against the receiver, not against itself.

        The fake validates field ownership from its own **literal** set, the way miLLM does
        (``400 MANIFEST_FIELD_NOT_CALLERS``). So this fails if the derivation drifts in
        either direction, which no amount of reading the schema would establish.
        """
        saver(fields=manifest_fields(), session=_Session())
        assert len(harness.fake.saves) == 1
        sent = harness.fake.saves[0]["manifest"]
        assert not (harness.fake.CALLER_MUST_NOT_SEND & set(sent))
        assert sent["run_id"] == "run-a"


class TestThePrecondition:
    """FR-011.56, MUT-6/MUT-S1: *no checkpoint from a set that is not ``clean``*."""

    @pytest.mark.parametrize("weights_state", ["writing", "torn"])
    def test_a_non_clean_set_refuses_AND_SENDS_NOTHING(
        self, saver: save_module.AttachedSaver, harness: Harness, weights_state: str
    ) -> None:
        """**The call count is the assertion.** A refusal raised *after* the request would
        read identically from the refusal alone, and would have published a checkpoint of a
        parameter set that never existed at any step. *A save is refused locally rather
        than relying on the other side's refusal.*
        """
        harness.fake.weights_state = weights_state
        harness.fake.calls.clear()
        with pytest.raises(Refusal) as caught:
            saver(fields=manifest_fields(), session=_Session())
        assert caught.value.code == "weights_not_clean"
        assert harness.fake.call_count("save") == 0
        assert harness.fake.saves == []

    def test_a_substituted_parameter_set_refuses_before_saving(
        self, saver: save_module.AttachedSaver, harness: Harness
    ) -> None:
        """The identity half: ``clean`` is not enough if it is not the same model."""
        harness.fake.resident_model = "some-other-model"
        harness.fake.calls.clear()
        with pytest.raises(Refusal) as caught:
            saver(fields=manifest_fields(), session=_Session())
        assert caught.value.code == "weights_identity_changed"
        assert harness.fake.call_count("save") == 0

    def test_the_precondition_runs_FIRST_in_the_recorded_order(
        self, saver: save_module.AttachedSaver
    ) -> None:
        outcome = saver(fields=manifest_fields(), session=_Session())
        assert outcome.effects[0] == "require_clean"
        assert outcome.effects.index("save") < outcome.effects.index("identity_rebaselined")


class TestTheIdentityIsRebaselined:
    """**MUT-S3.** A save can rename the resident model; the baseline must follow."""

    def test_two_consecutive_saves_both_succeed_when_the_resident_is_renamed(
        self, saver: save_module.AttachedSaver, harness: Harness
    ) -> None:
        """The reachable defect. Without the re-read: first checkpoint fine, **second save
        refused** ``weights_identity_changed``, run dead, diagnosis pointing at a serving
        restart that never happened - on a run saving every N steps.
        """
        assert harness.fake.renames_resident_on_save is True
        first = saver(fields=manifest_fields(step=500), session=_Session())
        second = saver(fields=manifest_fields(step=600), session=_Session())
        assert first.identity.model != second.identity.model
        assert saver.expected.model == second.identity.model
        assert harness.fake.call_count("save") == 2

    def test_the_last_accepted_save_is_remembered_for_rollback(
        self, saver: save_module.AttachedSaver, harness: Harness
    ) -> None:
        """R-02.30(b)'s rollback loads the last checkpoint miLLM accepted, BY THE NAME miLLM
        returned - so the saver records (step, name) of each accepted save."""
        assert saver.last_saved is None
        saver(fields=manifest_fields(step=500), session=_Session())
        first = saver.last_saved
        saver(fields=manifest_fields(step=600), session=_Session())
        names = list(harness.fake.bodies_of("save"))
        assert len(names) == 2
        assert first is not None and first[0] == 500
        assert saver.last_saved is not None and saver.last_saved[0] == 600
        assert saver.last_saved[1] and saver.last_saved[1] != first[1]

    def test_it_is_correct_ALSO_when_the_resident_is_NOT_renamed(
        self, saver: save_module.AttachedSaver, harness: Harness
    ) -> None:
        """Published BRD-01 does not state the rename - it is in miLLM's own manual, on a
        branch. So miForge depends on **neither** answer: re-reading is right either way,
        and this is the control that proves the code is not quietly assuming the rename.
        """
        harness.fake.renames_resident_on_save = False
        first = saver(fields=manifest_fields(step=500), session=_Session())
        second = saver(fields=manifest_fields(step=600), session=_Session())
        assert first.identity.model == second.identity.model == "LFM2-1.2B"
        assert harness.fake.call_count("save") == 2

    def test_the_baseline_is_read_from_the_SERVING_PROCESS_each_time(
        self, saver: save_module.AttachedSaver, harness: Harness
    ) -> None:
        """One status read for the precondition and one after the save. *A cached answer
        about another process's memory is a guess with a timestamp.*"""
        harness.fake.calls.clear()
        saver(fields=manifest_fields(), session=_Session())
        assert harness.fake.call_count("status") == 2


class TestItWritesNoBytesAndIndexesThroughF5:
    def test_an_unreadable_published_directory_is_NAMED_not_refused(
        self, saver: save_module.AttachedSaver, tmp_path: Path
    ) -> None:
        """The checkpoint exists and is valid upstream, and ``save`` is retry-unsafe.

        Refusing here would report a failure for a save that succeeded and leave a retry to
        hit ``CHECKPOINT_EXISTS``. So it is an index gap, logged by name, and the outcome
        says which.
        """
        outcome = saver(fields=manifest_fields(), session=_Session(), root=tmp_path)
        assert outcome.status == "written_upstream_only"
        assert outcome.indexed is None
        assert "index_skipped" in outcome.effects
        assert "index" not in outcome.effects

    def test_a_readable_one_is_indexed_through_F5s_OWN_indexer(
        self, saver: save_module.AttachedSaver, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """F5's ``upsert_from_manifest`` reads the manifest from disk, which is exactly what
        lets this index a directory **miForge did not write** - ADR-004's authority rule
        working in the direction it was written for.

        The published directory is produced by **F5's real writer** standing in for the
        serving process, not by a planted dict: *fixtures whose fields agree by construction
        are how a suite stays green over a real bug*, and a hand-built manifest here agreed
        with nothing - the first attempt declared ``kind: "checkpoint"`` and the real scanner
        wants ``miforge.checkpoint-manifest``, which is a disagreement a planted fixture
        hides and a round trip cannot.
        """
        from tests.support.checkpoints import create_feature_tables, saved_checkpoint
        from tests.support.jobs import memory_engine, session_for

        fields = manifest_fields()
        published = saved_checkpoint(tmp_path, run_id=fields.run_id, step=fields.step)

        calls: list[Path] = []
        real = save_module.checkpoint_index.upsert_from_manifest

        def recording(session: Any, directory: Path, **kwargs: Any) -> Any:
            calls.append(directory)
            return real(session, directory, **kwargs)

        monkeypatch.setattr(save_module.checkpoint_index, "upsert_from_manifest", recording)

        engine = memory_engine()
        create_feature_tables(engine)
        with session_for(engine) as session:
            outcome = saver(fields=fields, session=session, root=tmp_path)

        # The PAYLOAD and the call, not just "was called": the directory handed over is the
        # one R-01.10's layout and R-01.24's padding name, derived rather than read from a
        # response field published BRD-01 does not define.
        assert calls == [Path(published.directory)]
        assert outcome.status == "written"
        assert outcome.indexed == published.name
        assert "index" in outcome.effects

    def test_it_writes_no_checkpoint_bytes_of_its_own(
        self, saver: save_module.AttachedSaver, tmp_path: Path
    ) -> None:
        """R-01.13 is miLLM's: no temp dir, no rename, no ``latest``, no hash."""
        saver(fields=manifest_fields(), session=_Session(), root=tmp_path)
        assert list(tmp_path.iterdir()) == []

    def test_no_socket_event_is_emitted_under_F5s_event_name(
        self, saver: save_module.AttachedSaver
    ) -> None:
        """F5 builds that payload from a ``SaveOutcome`` this path cannot construct, and a
        second shape under one event name leaves F9 reading fields that are sometimes
        absent. **Owed by F6**, and asserted as an absence so it cannot be done by halves.
        """
        emitted: list[tuple[Any, ...]] = []
        saver(
            fields=manifest_fields(),
            session=_Session(),
            emit=lambda *args: emitted.append(args),
        )
        assert emitted == []


class TestTheLoopReachesIt:
    """ADR-022's fence over the dispatch, and a tripwire. **Not the reachability control.**

    **The real one is ``tests/integration/test_save_wiring.py::
    TestAnAttachedAcquisitionTakesItsOwnSavePath``**, which drives the production loop and
    asserts the call, the payload, the count and that F5's writer is *not* reached.

    This class once held a source-string reachability test instead, and MUT-S4 - the
    mutation that turns ``elif acquired.saver is not None:`` into ``elif False:`` -
    **survived it**: the string stayed in the branch body, so twenty tests passed against a
    dispatch that could never fire. *A grep proves a call exists, not that anything depends
    on it*, which is the weaker form this project records as insufficient - and this is the
    cheapest demonstration of it yet, because the grep and the mutation were written by the
    same person in the same hour.
    """

    def test_the_source_still_mentions_the_dispatch(self) -> None:
        """A tripwire, and labelled as one: it catches a *deleted* branch, not a dead one."""
        import training.tasks as tasks_module

        assert "acquired.saver" in Path(tasks_module.__file__).read_text(), (
            "the step loop no longer mentions the acquisition's saver at all. The "
            "behavioural control is in tests/integration/test_save_wiring.py; this only "
            "catches the branch being removed outright."
        )

    def test_the_dispatch_names_no_MODE(self) -> None:
        """ADR-022's fence: *a fourth registration, not a branch*. The loop must not test
        for ``attach`` by name - that is what makes the standalone control meaningful.

        **Comments and docstrings are stripped before searching**, because a mention is not
        an operation. The first version of this test matched raw source and was turned red
        by the explanatory comment on the very dispatch it protects - which is the sixth
        time in this project a fresh checker has been wrong about correct work, and the
        same retune F4's migration guard needed when it flagged a citation.
        """
        import io
        import tokenize

        import training.tasks as tasks_module

        source = Path(tasks_module.__file__).read_text()
        tokens = [
            token.string
            for token in tokenize.generate_tokens(io.StringIO(source).readline)
            if token.type not in (tokenize.COMMENT, tokenize.STRING)
        ]
        # Whitespace removed from the haystack and absent from the needles, because the
        # tokenizer does not preserve spacing: joined with spaces it reads `acquired .
        # saver`, and the positive control below is what caught that - a search that had
        # silently stopped matching anything would otherwise have read as a clean fence.
        code = "".join("".join(tokens).split())
        assert 'mode=="attach"' not in code
        assert "mode=='attach'" not in code
        assert "AcquisitionMode.ATTACH" not in code
        # The positive check that the stripping did not strip everything: the dispatch
        # itself must survive it, or this test passes against a file it never examined.
        assert "acquired.saver" in code

    def test_an_attached_acquisition_CARRIES_a_saver(
        self, harness: Harness, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The other half of the wiring: the acquirer must put one there.

        ``acquire`` needs torch and a card for the rebuild, so this drives the one line that
        binds the saver with the client and identity the acquirer has in hand at that point -
        and asserts the resulting object is the live class, which is what the loop will call.
        """
        from training.hotupdate import attach as attach_module
        from training.hotupdate import state

        _, identity = state.read(harness.client)
        acquirer = attach_module.AttachAcquirer()
        acquirer.saver = save_module.AttachedSaver(harness.client, expected=identity)
        assert isinstance(acquirer.saver, save_module.AttachedSaver)
        assert (
            "self.saver = save_module.AttachedSaver" in Path(attach_module.__file__).read_text()
        ), "the acquirer no longer binds a saver onto the acquisition"
