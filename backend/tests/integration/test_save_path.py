"""One save, end to end: the order, the resave rule, and the interrupted save.

Three things are asserted here that a unit test of any one component cannot:

1. **The order of the side effects** - index, then the log line, then the socket. ADR-007:
   *progress reported only over the socket is invisible to the janitor*, which is how
   miStudio reaped a live 5.8-hour job as "worker lost". The sequence is asserted, not the
   end state, because an inversion's consequence appears only after a crash.
2. **The resave rule** (FR-005.6, E-2, MUT-12). ``rename`` onto an existing non-empty
   directory fails with ``ENOTEMPTY``, so the second save of a step does not merely race -
   it breaks; and the obvious repair, remove-then-rename, reopens exactly the window the
   atomic write closes. Run end landing on a cadence boundary makes this **ordinary**.
3. **The interrupted save** (AC-005.3, BRD AC8). With a fault injected after the weights:
   the step directory does not exist, **no listing or reindex surfaces the temp directory**,
   and ``latest`` still resolves to the previous step. Atomicity alone does not give that -
   the temp directory is a sibling of the destination and therefore sits inside the tree a
   walk covers.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from core.atomic_dir import is_temp_dir
from core.errors import Refusal
from models.checkpoint import Checkpoint
from services.checkpoints import requests as requests_module
from services.checkpoints import service as service_module
from services.checkpoints.paths import checkpoint_dir, iter_step_dirs, resolve_latest
from services.reindex import list_checkpoints, reindex
from tests.support.checkpoint_faults import BrokenWriter, InjectedFaultError
from tests.support.checkpoints import (
    TinyModel,
    TinyTokenizer,
    create_feature_tables,
    fields,
    saved_checkpoint,
    score,
)
from tests.support.jobs import insert_run, memory_engine, session_for


@pytest.fixture
def engine():
    engine = memory_engine()
    create_feature_tables(engine)
    return engine


class Recorder:
    """A socket emitter that records rather than publishing. The payload is asserted."""

    def __init__(self) -> None:
        self.events: list[tuple[str, str, dict]] = []

    def __call__(self, run_id, event, payload):
        self.events.append((run_id, event, payload))


class Heartbeat:
    def __init__(self) -> None:
        self.touches = 0

    def touch(self, _session):
        self.touches += 1


def save(session, root, *, emit=None, heartbeat=None, payload=b"tensors-v1", **overrides):
    return service_module.save(
        TinyModel(payload=payload),
        TinyTokenizer(),
        session=session,
        fields=fields(**overrides),
        root=root,
        emit=emit,
        heartbeat=heartbeat,
    )


class TestDurableBeforeAnnounced:
    def test_the_index_and_the_log_precede_the_socket_emit(self, engine, tmp_path: Path):
        recorder, heartbeat = Recorder(), Heartbeat()
        with session_for(engine) as session:
            insert_run(session, "run-a", status="running")
            outcome = save(session, tmp_path, emit=recorder, heartbeat=heartbeat)

        assert outcome.effects == ["index", "log", "socket", "heartbeat"], (
            "the socket is a convenience and the database is the truth; announcing before "
            "recording is how a live job is reaped as 'worker lost'"
        )
        assert heartbeat.touches == 1

    def test_the_event_names_the_one_declared_string(self, engine, tmp_path: Path):
        """A screen subscribed to a string the emitter does not emit is silently dead while
        looking connected, and that typo is invisible in review - so the name is a constant
        and this asserts the constant reached the wire."""
        recorder = Recorder()
        with session_for(engine) as session:
            insert_run(session, "run-a", status="running")
            save(session, tmp_path, emit=recorder)

        run_id, event, payload = recorder.events[0]
        assert event == service_module.CHECKPOINT_SAVED_EVENT == "checkpoint_saved"
        assert run_id == "run-a"
        assert payload["step"] == 500
        assert payload["latest_step"] == 500
        assert payload["name"].endswith(":run-a:step-500")

    def test_the_event_reports_the_pointers_actual_state_not_this_step(
        self, engine, tmp_path: Path
    ):
        """``latest_step`` is **resolved**, never assumed to equal the step just written.

        F9 marks the pointer on its curves, so the event has to say where the pointer
        actually is; computing it as "the step I just saved" would be a second answer that
        agrees until it does not.

        *Finding, recorded here rather than silently accommodated:* writing an **earlier**
        step after a later one moves ``latest`` backwards, because
        ``core.atomic_dir.atomic_dir`` swaps the pointer on every publish. Within a run that
        is unreachable - a continue's steps increase and a branch writes into a different
        run directory - and no requirement describes an out-of-order save, so this test
        pins the behaviour as it is rather than asserting a rule nothing states. The
        pointer's semantics belong to the primitive, which is the foundation's file."""
        recorder = Recorder()
        with session_for(engine) as session:
            insert_run(session, "run-a", status="running")
            save(session, tmp_path, emit=recorder, step=500)
            save(session, tmp_path, emit=recorder, step=100, payload=b"earlier")

        assert recorder.events[-1][2]["step"] == 100
        assert recorder.events[-1][2]["latest_step"] == resolve_latest("run-a", root=tmp_path)

    def test_a_save_completes_with_no_emitter_at_all(self, engine, tmp_path: Path):
        """A worker with no Redis reachable still saves. The socket is injected for exactly
        this reason."""
        with session_for(engine) as session:
            insert_run(session, "run-a", status="running")
            outcome = save(session, tmp_path)
        assert outcome.status == "written" and "socket" not in outcome.effects


class TestTheIndexRowIsAProjection:
    def test_the_row_is_built_by_reindexs_own_builder(self, engine, tmp_path: Path):
        """**Why this matters more than it looks:** AC-005.9 compares responses before and
        after a rebuild. If the upsert built its row a second way, the two would agree today
        and drift on the first column anyone added - and the equivalence test would go red
        long afterwards, pointing at the rebuild."""
        with session_for(engine) as session:
            insert_run(session, "run-a", status="running")
            outcome = save(session, tmp_path, test=score())
            after_save = list_checkpoints(session)

        with session_for(engine) as session:
            reindex(session, root=tmp_path)
            after_rebuild = list_checkpoints(session)

        assert after_save == after_rebuild
        assert after_save[0]["name"] == outcome.name
        assert after_save[0]["test"]["provider"] == "in-process:lfm2-1.2b"

    def test_the_stored_path_is_relative_to_the_root(self, engine, tmp_path: Path):
        """Relative so the index survives the root being mounted elsewhere - which it is,
        between a developer's machine and the GPU node's PVC."""
        with session_for(engine) as session:
            insert_run(session, "run-a", status="running")
            save(session, tmp_path)
            row = session.get(Checkpoint, list_checkpoints(session)[0]["name"])
            assert row is not None and row.path == "run-a/step-000500"


class TestTheResaveRule:
    def test_resave_is_noop_or_refusal(self, engine, tmp_path: Path):
        """**MUT-12's target.** Run end landing on a cadence boundary makes this ordinary,
        not an edge case."""
        with session_for(engine) as session:
            insert_run(session, "run-a", status="running")
            first = save(session, tmp_path, payload=b"same")
            before = (first.directory / "manifest.json").read_bytes()

            # Same parameters -> idempotent. A retried Celery task is harmless.
            again = save(session, tmp_path, payload=b"same")
            assert again.status == "noop"
            assert (first.directory / "manifest.json").read_bytes() == before

            # Different parameters -> refused BY NAME. Two parameter sets cannot claim one
            # name, because the step is part of the identity.
            with pytest.raises(Refusal) as caught:
                save(session, tmp_path, payload=b"different")
            assert caught.value.code == "checkpoint_step_exists"
            assert (first.directory / "manifest.json").read_bytes() == before

    def test_the_existing_directory_is_byte_unchanged_in_both_cases(self, engine, tmp_path: Path):
        with session_for(engine) as session:
            insert_run(session, "run-a", status="running")
            first = save(session, tmp_path, payload=b"same")
            before = {
                p.name: p.read_bytes() for p in sorted(first.directory.iterdir()) if p.is_file()
            }
            save(session, tmp_path, payload=b"same")
            with pytest.raises(Refusal):
                save(session, tmp_path, payload=b"different")
            after = {
                p.name: p.read_bytes() for p in sorted(first.directory.iterdir()) if p.is_file()
            }
        assert before == after

    def test_a_refused_resave_leaves_no_staging_directory_behind(self, engine, tmp_path: Path):
        """The candidate write is a full serialisation into a dot-prefixed staging directory.
        It is removed in a ``finally``, so a refusal does not leave a discoverable artifact
        of the thing it refused."""
        with session_for(engine) as session:
            insert_run(session, "run-a", status="running")
            save(session, tmp_path, payload=b"same")
            with pytest.raises(Refusal):
                save(session, tmp_path, payload=b"different")

        entries = [p.name for p in (tmp_path / "run-a").iterdir()]
        assert not [n for n in entries if is_temp_dir(n)], entries

    def test_a_noop_resave_does_not_re_announce_the_save(self, engine, tmp_path: Path):
        """Nothing was written, so nothing is announced: a second ``checkpoint_saved`` for a
        step that did not change would put a second marker on F9's curve."""
        recorder = Recorder()
        with session_for(engine) as session:
            insert_run(session, "run-a", status="running")
            save(session, tmp_path, emit=recorder, payload=b"same")
            outcome = save(session, tmp_path, emit=recorder, payload=b"same")
        assert outcome.status == "noop"
        assert len(recorder.events) == 1


class TestTheInterruptedSave:
    @pytest.mark.parametrize("fail_after", ["nothing", "weights", "config"])
    def test_nothing_partial_is_discoverable(self, engine, tmp_path: Path, fail_after):
        """**AC-005.3 / BRD AC8.** The step directory does not exist, no listing or reindex
        surfaces the temp directory, and ``latest`` still resolves to the previous step."""
        with session_for(engine) as session:
            insert_run(session, "run-a", status="running")
            previous = save(session, tmp_path, step=400, payload=b"previous")

            with pytest.raises(InjectedFaultError):
                service_module.save(
                    BrokenWriter(fail_after=fail_after),
                    TinyTokenizer(),
                    session=session,
                    fields=fields(step=500),
                    root=tmp_path,
                )

            assert not checkpoint_dir("run-a", 500, root=tmp_path).exists()
            assert [step for step, _ in iter_step_dirs("run-a", root=tmp_path)] == [400]
            assert resolve_latest("run-a", root=tmp_path) == 400
            assert (tmp_path / "run-a" / "latest").resolve() == previous.directory

            entries = [p.name for p in (tmp_path / "run-a").iterdir()]
            assert not [
                n for n in entries if is_temp_dir(n)
            ], f"a staging directory survived the fault: {entries}"

            reindex(session, root=tmp_path)
            assert [r["step"] for r in list_checkpoints(session)] == [400]

    def test_a_temp_directory_left_by_a_KILLED_writer_is_not_indexed(self, engine, tmp_path: Path):
        """The primitive cleans up when it *raises*; a process killed outright never runs
        that cleanup. **This is the case AC8 is actually about** - and it is the one where
        atomicity alone is not enough, because the staging directory is a sibling of the
        destination and therefore inside the tree a walk covers.

        *Reported as a finding: ``services.reindex.scan_root`` walks with ``rglob`` and does
        not skip dot-prefixed entries, so this assertion is about a path F5 does not own.*"""
        with session_for(engine) as session:
            insert_run(session, "run-a", status="running")
            save(session, tmp_path, step=400, payload=b"previous")

        orphan = tmp_path / "run-a" / ".tmp-save-deadbeef"
        orphan.mkdir()
        complete = saved_checkpoint(tmp_path, run_id="donor", step=1)
        for item in complete.directory.iterdir():
            (orphan / item.name).write_bytes(item.read_bytes())

        assert [step for step, _ in iter_step_dirs("run-a", root=tmp_path)] == [
            400
        ], "F5's own walk must be blind to an in-flight write"
        assert resolve_latest("run-a", root=tmp_path) == 400

    # The strict xfail that guarded this is removed: it fired the moment the
    # one-line predicate landed in services.reindex.scan_root, which is what a
    # strict xfail is for. The defect was real and is F5's find.
    def test_reindex_does_not_surface_an_orphaned_staging_directory(self, engine, tmp_path: Path):
        with session_for(engine) as session:
            insert_run(session, "run-b", status="running")
            save(session, tmp_path, step=400, payload=b"previous", run_id="run-b")

        donor = saved_checkpoint(tmp_path, run_id="donor", step=7)
        orphan = tmp_path / "run-b" / ".tmp-save-cafebabe"
        orphan.mkdir()
        for item in donor.directory.iterdir():
            (orphan / item.name).write_bytes(item.read_bytes())

        with session_for(engine) as session:
            reindex(session, root=tmp_path)
            paths = [row["path"] for row in list_checkpoints(session)]
        assert not [p for p in paths if ".tmp-save-" in p], paths

    # The strict xfail that guarded this is removed: it fired the moment the
    # one-line predicate landed in services.reindex.scan_root, which is what a
    # strict xfail is for. The defect was real and is F5's find.
    def test_an_orphan_duplicating_a_published_name_does_not_break_the_whole_reindex(
        self, engine, tmp_path: Path
    ):
        with session_for(engine) as session:
            insert_run(session, "run-c", status="running")
            published = save(session, tmp_path, step=400, payload=b"p", run_id="run-c")

        orphan = tmp_path / "run-c" / ".tmp-save-f00d"
        orphan.mkdir()
        for item in published.directory.iterdir():
            (orphan / item.name).write_bytes(item.read_bytes())

        with session_for(engine) as session:
            reindex(session, root=tmp_path)
            assert len(list_checkpoints(session)) == 1


class TestTheSaveRequestFlag:
    def test_two_requests_for_one_boundary_collapse_into_one_save(self, engine):
        """E-12. A property of the table's primary key rather than of a handler's ``if``:
        there is no way to represent two pending saves for one run."""
        with session_for(engine) as session:
            insert_run(session, "run-a", status="running")
            requests_module.request_save(session, "run-a")
            second = requests_module.request_save(session, "run-a")

            assert second["request_count"] == 2, "the fact that two were asked for survives"
            pending = requests_module.pending_save_request(session, "run-a")
            assert pending is not None and pending["request_count"] == 2

            taken = requests_module.consume_save_request(session, "run-a")
            assert taken is not None
            assert requests_module.pending_save_request(session, "run-a") is None
            assert requests_module.consume_save_request(session, "run-a") is None

    def test_a_request_is_database_state_and_never_a_celery_control_message(self):
        """ADR-006: every worker is ``--pool=solo -c 1``, so a **busy** worker never reads
        the control queue - and a busy worker is exactly the one a save request is for."""
        source = Path(requests_module.__file__).read_text("utf-8")
        for forbidden in ("revoke", "control", "celery", "app.control"):
            assert forbidden not in source.split('"""', 2)[2].lower(), forbidden
