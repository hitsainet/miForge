"""Replay and program mode, backend half and end to end (feature 016; FTASKS 9.1-9.6, 5.2a in part;
FR-016.21, FR-016.23; R1B-A1, R1B-A7, R1B-A9, R1B-A10, R3A-13, R4A-2).

``TestReplay`` and ``TestProgram`` go through the REAL driver (``test_dispatch``'s world) with
``wsp.tasks.run_files``, the production file list. Session results are archives built here; the
program patches in them are made by the worker's own ``wsp_worker.program`` (numpy only, so it
runs in this interpreter). Reruns run END TO END against the real supervisor and the real child
(``LocalProcessLauncher`` under the worker interpreter, ``MIFORGE_WSP_WORKER_PYTHON``): no LLM
is constructed, no gateway exists, and a fixture program that hunts the child's memory for a
gateway address finds none (the fake gateway records zero connections).
"""

from __future__ import annotations

import datetime as dt
import json
import socket
import threading
import time
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest
import sqlalchemy as sa
from sqlalchemy.orm import Session

from core.errors import Refusal
from tests.integration.wsp.test_dispatch import FakeWorker, World, _j, archive
from tests.integration.wsp.test_dispatch import world as world
from tests.integration.wsp.test_runs_lifecycle import _patch_members, _store
from tests.support.wsp_worker_env import worker_python
from wsp import dispatch, gateway_ledger, runs, tasks
from wsp.models import WSPBase, WspCandidate, WspRun
from wsp_worker import bundle as worker_bundle
from wsp_worker import program as worker_program

PROGRAM = (
    "def edit(base, support):\n"
    "    w = np.array(base['weight'])\n"
    "    return {'weight': (w + np.float32(0.5) * np.sign(w)).astype(w.dtype)}\n"
)
UNSEEDED = (
    "def edit(base, support):\n"
    "    w = np.array(base['weight'])\n"
    "    return {'weight': (w + np.random.default_rng().random(w.shape)).astype(w.dtype)}\n"
)
#: A hostile program: escapes the restricted builtins (they are not the containment), then hunts
#: the child's heap for anything shaped like the push's gateway, and connects to it.
HUNTER = """
def _real_import():
    for c in ().__class__.__base__.__subclasses__():
        if c.__name__ == 'catch_warnings':
            b = c.__init__.__globals__['__builtins__']
            return b['__import__'] if isinstance(b, dict) else b.__import__

def edit(base, support):
    imp = _real_import()
    gc, sock = imp('gc'), imp('socket')
    for o in gc.get_objects():
        # Plain dicts only: a dict subclass may override get() (one in the full suite does).
        if o.__class__ is {}.__class__:
            g = o.get('gateway')
            if g.__class__ is {}.__class__ and 'ip' in g and 'port' in g:
                # Keep hunting past a dead target: in a full suite an earlier test's planted dict
                # (its listener long closed) may come first on the heap (review OR1-016-12).
                try:
                    s = sock.create_connection((g['ip'], int(g['port'])), timeout=2)
                    s.sendall(b'GET / HTTP/1.0\\r\\n\\r\\n')
                    s.close()
                except:
                    pass
    return {}
"""


class FakeGateway:
    """A TCP listener that counts connections (the private-CI test's fake gateway)."""

    def __init__(self) -> None:
        self.sock = socket.socket()
        self.sock.bind(("127.0.0.1", 0))
        self.sock.listen(8)
        self.sock.settimeout(0.2)
        self.port = self.sock.getsockname()[1]
        self.connections = 0
        self._stop = False
        self.thread = threading.Thread(target=self._serve, daemon=True)
        self.thread.start()

    def _serve(self) -> None:
        while not self._stop:
            try:
                conn, _ = self.sock.accept()
            except OSError:
                continue
            self.connections += 1
            conn.close()

    def close(self) -> None:
        self._stop = True
        self.thread.join(timeout=5)
        self.sock.close()


@pytest.fixture()
def gateway(monkeypatch: pytest.MonkeyPatch) -> Iterator[FakeGateway]:
    fake = FakeGateway()
    monkeypatch.setenv("WSP_GATEWAY_URL", f"http://127.0.0.1:{fake.port}")
    yield fake
    fake.close()


# ----------------------------------------------------------------------- helpers


def drive(world: World, worker: Any) -> list[str]:
    """One pass with the PRODUCTION file list (``wsp.tasks.run_files``)."""
    from wsp.settings import get_wsp_settings

    get_wsp_settings.cache_clear()
    world.settings = get_wsp_settings()
    return dispatch.drive(
        world.factory,
        worker,
        world.settings,
        read_sensitivity=world.read,
        files_for=tasks.run_files,
    ).actions


def _worker_bundle(world: World, manifest: dict[str, Any]) -> worker_bundle.Bundle:
    from wsp.bundles import store_dir

    root = store_dir(world.settings, manifest["bundle_sha256"])
    files = [{k: e[k] for k in ("name", "role", "sha256")} for e in manifest["files"]]
    return worker_bundle.load(root, manifest, files)


def _program_patch(
    world: World, manifest: dict[str, Any], source: str, kind: str = "practice_train"
) -> dict[str, bytes]:
    b = _worker_bundle(world, manifest)
    request = {
        "kind": "repair",
        "parent": {"kind": "bundle", "ref": manifest["bundle_sha256"]},
        "change_limits": {
            "max_changed_entries": None,
            "allowed_tensors": None,
            "frozen_tensors": [],
        },
    }
    done = worker_program.outcome(
        worker_program.run_once(source, b, worker_program.SupportSet(kind, b), request),
        b,
        request,
        "wsp_session",
        dt.datetime.now(dt.UTC).isoformat(),
        created_by={"provider": "weightspace", "provider_version": "0.3.0"},
    )
    if done.manifest is None or done.data is None:
        return {}  # the program changed nothing
    return {
        "patch/patch.json": json.dumps(done.manifest).encode(),
        "patch/edit.npz": done.data,
    }


TRANSCRIPT = b'{"role": "assistant", "content": null, "tool_calls": []}\n'


def _session(
    world: World,
    tmp_path: Path,
    *,
    mode: str = "session",
    source: str | None = None,
    members: dict[str, bytes] | None = None,
) -> tuple[str, dict[str, Any]]:
    """A completed session (or program) run, its result taken in by the real intake."""
    manifest = _store(world, tmp_path) if not _stored(world) else _manifest(world)
    world.body = {**world.body, "mode": mode}
    run_id = world.queue()
    worker = FakeWorker()
    assert drive(world, worker) == [f"pushed:{run_id}"]
    parts = {"transcript.jsonl": TRANSCRIPT, "result.json": b'{"stop_reason": "submitted"}'}
    if source is not None:
        parts["program.py"] = source.encode()
        parts.update(_program_patch(world, manifest, source))
    parts.update(members or {})
    worker.state, worker.data = "finished", archive(worker.boot, run_id, **parts)
    drive(world, worker)
    return run_id, manifest


def _stored(world: World) -> bool:
    from wsp.bundles import store_dir

    return store_dir(world.settings, _manifest(world)["bundle_sha256"]).is_dir()


def _manifest(world: World) -> dict[str, Any]:
    with world.engine.connect() as conn:
        return dict(_j(conn.execute(sa.text("SELECT manifest FROM mf_wsp_bundles")).scalar_one()))


def _candidates(world: World) -> int:
    with world.factory() as s:
        return int(s.scalar(sa.select(sa.func.count()).select_from(WspCandidate)) or 0)


def _queue_replay(world: World, run_id: str) -> str:
    with world.factory() as s:
        row = runs.create_replay(s, run_id, "operator", world.settings)
        s.commit()
        return row.run_id


def _queue_rerun(world: World, sha: str, kind: str = "practice_train") -> str:
    with world.factory() as s:
        row = runs.create_rerun(s, sha, kind, "operator", world.settings)
        s.commit()
        return row.run_id


def _document(world: World, run_id: str) -> dict[str, Any]:
    with world.factory() as s:
        row = runs.get_run(s, run_id)
        doc = runs.run_document(row, reruns=runs.program_reruns(s, row.program_sha256))
    from wsp.schemas import validate_run

    validate_run(doc)  # every document these tests read is a valid miforge.wsp-run/v1
    return doc


# ======================================================================= replay


class TestReplay:
    def _replay_with(self, world: World, of: str, members: dict[str, bytes]) -> dict[str, Any]:
        replay_id = _queue_replay(world, of)
        worker = FakeWorker()
        assert drive(world, worker) == [f"pushed:{replay_id}"]
        pushed = worker.pushed
        assert pushed is not None
        worker.state = "finished"
        worker.data = archive(worker.boot, replay_id, **members)
        drive(world, worker)
        return {"id": replay_id, "pushed": pushed, "row": world.run(replay_id)}

    def test_a_replay_reproduces_creates_no_candidate_and_has_no_gateway(
        self, world: World, tmp_path: Path, gateway: FakeGateway
    ) -> None:
        manifest = _store(world, tmp_path)
        # A session whose patch is a program's edit stands in for any editor's edit.
        run_id, _ = _session(world, tmp_path, members=_program_patch(world, manifest, PROGRAM))
        original = world.run(run_id)
        assert original["outcome"] == "candidate" and original["content_sha256"]
        before = _candidates(world)
        # The replay's patch: the same edit, made again (another created_at, so another
        # patch_sha256, by construction; R1B-A1).
        time.sleep(0.01)
        again = _program_patch(world, manifest, PROGRAM)
        done = self._replay_with(world, run_id, {"transcript.jsonl": TRANSCRIPT, **again})
        row, pushed = done["row"], done["pushed"]
        assert row["status"] == "completed" and row["outcome"] == "report"
        assert row["replay_reproduced"] in (True, 1)
        assert (
            row["patch_sha256"] is None and row["replay_patch_sha256"] != original["patch_sha256"]
        )
        assert row["content_sha256"] == original["content_sha256"]
        assert _candidates(world) == before  # no candidate from a replay
        # R3A-13: no gateway in the push, no token minted.
        assert pushed["gateway"] is None and pushed["mode"] == "replay"
        assert row["gateway_token_sha256"] is None and row["token_expires_at"] is None
        assert pushed["spec"]["replay_of_mode"] == "session"
        # R1B-A7: the original transcript is a pushed file, with its own role.
        sent = {f["name"]: f for f in pushed["files"]}
        assert sent["transcript.jsonl"]["role"] == "transcript"
        assert sent["transcript.jsonl"]["sha256"] == original["transcript_sha256"]
        doc = _document(world, done["id"])
        assert doc["replay"] == {
            "of_run_id": run_id,
            "reproduced": True,
            "patch_sha256": row["replay_patch_sha256"],
            "content_sha256": original["content_sha256"],
        }
        assert doc["request"]["mode"] == "replay" and doc["request"]["of_run_id"] == run_id
        assert doc["request"]["endpoint"] is None

    def test_an_edited_replay_does_not_reproduce_and_makes_no_candidate(
        self, world: World, tmp_path: Path
    ) -> None:
        manifest = _store(world, tmp_path)
        run_id, _ = _session(world, tmp_path, members=_program_patch(world, manifest, PROGRAM))
        before = _candidates(world)
        other = PROGRAM.replace("0.5", "0.25")
        done = self._replay_with(
            world,
            run_id,
            {"transcript.jsonl": TRANSCRIPT, **_program_patch(world, manifest, other)},
        )
        assert done["row"]["status"] == "completed"
        assert done["row"]["replay_reproduced"] in (False, 0)
        assert _candidates(world) == before
        assert _document(world, done["id"])["replay"]["reproduced"] is False

    def test_a_tampered_stored_transcript_fails_the_replay_before_anything_is_consumed(
        self, world: World, tmp_path: Path
    ) -> None:
        run_id, _ = _session(world, tmp_path)
        stored = runs.run_dir(world.settings, run_id) / "transcript.jsonl"
        stored.write_bytes(stored.read_bytes() + b"{}\n")
        replay_id = _queue_replay(world, run_id)
        boots = world.boots()
        worker = FakeWorker()
        assert drive(world, worker) == [f"refused:{replay_id}:wsp_result_invalid"]
        assert world.boots() == boots and "push" not in worker.names()

    def test_only_a_completed_run_with_a_transcript_replays(
        self, world: World, tmp_path: Path
    ) -> None:
        queued = world.queue()
        with world.factory() as s, pytest.raises(Refusal) as caught:
            runs.create_replay(s, queued, "operator", world.settings)
        assert caught.value.code == "validation_failed"
        with world.factory() as s, pytest.raises(Refusal) as caught:
            runs.create_replay(s, "wsp_ffffffffffff", "operator", world.settings)
        assert caught.value.code == "wsp_run_not_found"

    def test_the_gateway_never_lists_or_resolves_a_replay(
        self, world: World, tmp_path: Path
    ) -> None:
        from api import wsp_internal_app

        run_id, _ = _session(world, tmp_path)
        replay_id = _queue_replay(world, run_id)
        worker = FakeWorker()
        drive(world, worker)
        assert world.run(replay_id)["status"] == "running"
        with world.factory() as s:
            listed = wsp_internal_app.list_sessions(s)["sessions"]
        assert replay_id not in [e["run_id"] for e in listed]


# ----------------------------------------------------------------------- 9.2a: below the CHECK


def _schema_without_the_check() -> sa.Engine:
    """The WSP tables as the models declare them, MINUS ``rerun_replay_have_no_token``: the
    only way to store the row the gateway's style predicate exists to refuse (R4A-2)."""
    metadata = sa.MetaData()
    for table in WSPBase.metadata.sorted_tables:
        table.to_metadata(metadata)
    runs_table = metadata.tables["mf_wsp_runs"]
    dropped = [
        c
        for c in runs_table.constraints
        if isinstance(c, sa.CheckConstraint) and "rerun_replay_have_no_token" in str(c.name)
    ]
    assert len(dropped) == 1
    runs_table.constraints.discard(dropped[0])
    engine = sa.create_engine("sqlite://")
    metadata.create_all(engine)
    return engine


@pytest.mark.parametrize("style", ["replay", "program_rerun"])
def test_by_token_and_reserve_refuse_a_no_gateway_style_carrying_a_token(style: str) -> None:
    from api import wsp_internal_app
    from tests.integration.wsp.wsp_db import NOW, bundle_row, insert, run_row
    from wsp.settings import WSPSettings

    engine = _schema_without_the_check()
    token_sha = "d" * 64
    with engine.begin() as conn:
        bundle_row(conn)
        insert(
            conn,
            "mf_wsp_endpoints",
            id="ep",
            kind="openai_compatible",
            base_url="http://miserver-backend.miserver.svc:8000/v1",
            model="m",
            locality="local",
            output_cap_field="max_tokens",
            serving="miserver",
            created_at=NOW,
            created_by="operator",
        )
        run_row(
            conn,
            style=style,
            status="running",
            endpoint_id="ep",
            gateway_token_sha256=token_sha,
            token_expires_at=NOW + dt.timedelta(days=3650),
            request=json.dumps({"budgets": {"tokens": 1000}}),
        )
    with Session(engine) as s:
        run = s.get(WspRun, "wsp_000000000001")
        assert run is not None and run.gateway_token_sha256 == token_sha
        assert not gateway_ledger.session_active(run, token_sha, NOW)
        with pytest.raises(Refusal) as caught:
            gateway_ledger.reserve(
                s,
                run_id=run.run_id,
                token_sha256=token_sha,
                worst_case_usd=gateway_ledger.ZERO,
                worst_case_tokens=1,
                external=False,
                settings=WSPSettings(),
                now=NOW,
            )
        assert caught.value.code == "wsp_session_not_active"
        with pytest.raises(Refusal) as missed:
            wsp_internal_app.session_by_token(token_sha, s)
        assert missed.value.code == "wsp_run_not_found"
        assert wsp_internal_app.list_sessions(s)["sessions"] == []
    # The positive control: the same row as a session IS live, so the refusals above are the
    # style predicate's, not a broken fixture's.
    with engine.begin() as conn:
        conn.execute(sa.text("UPDATE mf_wsp_runs SET style = 'session'"))
    wsp_internal_app._negative.clear()
    with Session(engine) as s:
        run = s.get(WspRun, "wsp_000000000001")
        assert gateway_ledger.session_active(run, token_sha, NOW)
        assert wsp_internal_app.session_by_token(token_sha, s)["run_id"] == run.run_id


# ======================================================================= program mode


@pytest.fixture()
def real_worker(world: World) -> Iterator[Any]:
    from wsp.launcher import LocalProcessLauncher
    from wsp.worker_client import HttpWorker

    python = worker_python()
    started: list[Any] = []

    class Real:
        def run(self) -> list[str]:
            """Push the next queued run to a FRESH real supervisor and collect it."""
            launcher = LocalProcessLauncher(environment="ci", python=python)
            with launcher.session() as local:
                client = HttpWorker(local.url)
                started.append(local)
                try:
                    actions = drive(world, client)
                    assert actions and actions[-1].startswith("pushed:"), actions
                    deadline = time.monotonic() + 60
                    while client.status()["state"] != "finished":
                        assert time.monotonic() < deadline, "the rerun never finished"
                        time.sleep(0.2)
                    actions += drive(world, client)
                finally:
                    client.close()
                assert local.process.wait(timeout=30) == 0
            return actions

    yield Real()


class TestProgram:
    def _stored_program(
        self, world: World, tmp_path: Path, source: str, *, run_here: bool = True
    ) -> tuple[str, str]:
        """``run_here=False`` stores the program with no patch, never executing it in this
        process (whose heap holds earlier session pushes, gateways included)."""
        if run_here:
            run_id, _ = _session(world, tmp_path, mode="program", source=source)
        else:
            run_id, _ = _session(
                world, tmp_path, mode="program", members={"program.py": source.encode()}
            )
        row = world.run(run_id)
        assert row["status"] == "completed" and row["program_sha256"], row
        return run_id, str(row["program_sha256"])

    def test_a_program_session_stores_its_program_once(self, world: World, tmp_path: Path) -> None:
        run_id, sha = self._stored_program(world, tmp_path, PROGRAM)
        with world.engine.connect() as conn:
            rows = conn.execute(sa.text("SELECT * FROM mf_wsp_programs")).mappings().all()
        assert [(r["program_sha256"], r["first_run_id"]) for r in rows] == [(sha, run_id)]
        assert runs.program_path(world.settings, sha).read_text() == PROGRAM
        # The same program from a second session keeps its first run.
        self._stored_program(world, tmp_path, PROGRAM)
        with world.engine.connect() as conn:
            assert conn.execute(sa.text("SELECT COUNT(*) FROM mf_wsp_programs")).scalar_one() == 1

    def test_two_reruns_agree_and_equal_the_sessions_candidate(
        self, world: World, tmp_path: Path, real_worker: Any, gateway: FakeGateway
    ) -> None:
        """Phase-1 acceptance: a program rerun is deterministic (content_sha256; patch_sha256
        covers created_by and created_at, which differ by construction, FTDD §3.2)."""
        session_id, sha = self._stored_program(world, tmp_path, PROGRAM)
        first, second = _queue_rerun(world, sha), None
        real_worker.run()
        second = _queue_rerun(world, sha)
        real_worker.run()
        a, b, s = world.run(first), world.run(second), world.run(session_id)
        assert a["status"] == b["status"] == "completed", (a["failure"], b["failure"])
        assert a["outcome"] == b["outcome"] == "candidate"
        assert a["content_sha256"] == b["content_sha256"] == s["content_sha256"]
        assert a["patch_sha256"] != b["patch_sha256"]
        # No token, no gateway, no connection (R3A-13).
        assert a["gateway_token_sha256"] is None and a["endpoint_id"] is None
        assert gateway.connections == 0
        doc = _document(world, session_id)
        assert doc["program_sha256"] == sha and doc["program_reruns"] == [first, second]
        assert _document(world, first)["request"]["support_set"]["kind"] == "practice_train"

    def test_a_rerun_gets_no_gateway_so_a_hunting_program_finds_none(
        self, world: World, tmp_path: Path, real_worker: Any, gateway: FakeGateway
    ) -> None:
        """R1B-A9, R3A-13: the hunter escapes the builtins and scans the child's heap for a
        gateway; with ``gateway: null`` there is none, and the fake gateway sees zero
        connections. Control C23a mints a token for the rerun and turns this red."""
        _, sha = self._stored_program(world, tmp_path, HUNTER, run_here=False)
        rerun_id = _queue_rerun(world, sha)
        real_worker.run()
        row = world.run(rerun_id)
        assert row["status"] == "completed", row["failure"]
        assert gateway.connections == 0

    def test_the_hunter_does_connect_when_a_gateway_is_in_reach(self, gateway: FakeGateway) -> None:
        """The positive control for the test above: the same program, run here with a gateway
        dict on the heap, connects (so zero connections there means none was in reach)."""
        planted = {"gateway": {"ip": "127.0.0.1", "port": gateway.port, "token": "t"}}
        namespace: dict[str, Any] = {"__builtins__": worker_program.restricted_builtins()}
        exec(compile(HUNTER, "hunter", "exec"), namespace)
        namespace["edit"]({}, None)
        deadline = time.monotonic() + 5
        while gateway.connections == 0 and time.monotonic() < deadline:
            time.sleep(0.05)
        assert planted and gateway.connections >= 1

    @pytest.mark.parametrize(
        ("source", "code"),
        [(UNSEEDED, "wsp_program_nondeterministic"), ("x = 1\n", "wsp_program_invalid")],
        ids=["unseeded", "no-edit"],
    )
    def test_a_bad_stored_program_fails_its_rerun_by_name(
        self, world: World, tmp_path: Path, real_worker: Any, source: str, code: str
    ) -> None:
        # Stored through an intake (a session may submit what a rerun then refuses), with no
        # patch of its own when it has no edit.
        run_id, _ = _session(
            world,
            tmp_path,
            mode="program",
            members={"program.py": source.encode()},
        )
        sha = world.run(run_id)["program_sha256"]
        assert sha
        before = _candidates(world)
        rerun_id = _queue_rerun(world, sha)
        real_worker.run()
        row = world.run(rerun_id)
        assert row["status"] == "failed" and _j(row["failure"])["code"] == code
        assert _candidates(world) == before

    def test_a_tampered_stored_program_fails_the_rerun_before_anything_is_consumed(
        self, world: World, tmp_path: Path
    ) -> None:
        _, sha = self._stored_program(world, tmp_path, PROGRAM)
        path = runs.program_path(world.settings, sha)
        path.write_text(PROGRAM + "\n# changed\n")
        rerun_id = _queue_rerun(world, sha)
        boots = world.boots()
        worker = FakeWorker()
        assert drive(world, worker) == [f"refused:{rerun_id}:wsp_program_not_found"]
        assert world.boots() == boots

    def test_a_rerun_push_carries_no_gateway_no_token_and_its_program(
        self, world: World, tmp_path: Path, gateway: FakeGateway
    ) -> None:
        _, sha = self._stored_program(world, tmp_path, PROGRAM)
        rerun_id = _queue_rerun(world, sha, "practice")
        worker = FakeWorker()
        assert drive(world, worker) == [f"pushed:{rerun_id}"]
        pushed = worker.pushed
        assert pushed is not None and pushed["gateway"] is None
        assert pushed["mode"] == "program_rerun"
        sent = {f["name"]: f for f in pushed["files"]}
        assert sent["program.py"]["role"] == "program" and sent["program.py"]["sha256"] == sha
        row = world.run(rerun_id)
        assert row["gateway_token_sha256"] is None and row["token_expires_at"] is None
        assert _j(row["request"])["support_set"]["kind"] == "practice"

    @pytest.mark.parametrize("tensor", ["mean", "std"])
    def test_a_reruns_patch_to_a_non_editable_tensor_is_refused(
        self, world: World, tmp_path: Path, tensor: str
    ) -> None:
        """Review OR2-016-1: a ``program_rerun``'s result goes through the same intake check as a
        session's. The stored program runs in the child, so the child's ``/out/patch`` is
        untrusted; one that edits a tensor the bundle marks not editable gets no candidate.
        Control ``R-ed-intake-style`` (the check scoped to ``style == "session"``) turns this
        red."""
        _, sha = self._stored_program(world, tmp_path, PROGRAM, run_here=False)
        before = _candidates(world)
        rerun_id = _queue_rerun(world, sha)
        worker = FakeWorker()
        assert drive(world, worker) == [f"pushed:{rerun_id}"]
        members = _patch_members(_manifest(world), tmp_path, tensor=tensor)
        worker.state, worker.data = "finished", archive(worker.boot, rerun_id, **members)
        drive(world, worker)
        row = world.run(rerun_id)
        assert row["style"] == "program_rerun"
        failure = _j(row["failure"])
        assert row["status"] == "failed" and failure["code"] == "wsp_change_limit_exceeded", row
        assert f"tensor {tensor!r} is not editable" in failure["message"]
        assert _candidates(world) == before

    def test_an_unknown_program_or_support_set_is_refused(
        self, world: World, tmp_path: Path
    ) -> None:
        with world.factory() as s, pytest.raises(Refusal) as caught:
            runs.create_rerun(s, "e" * 64, "practice", "operator", world.settings)
        assert caught.value.code == "wsp_program_not_found"
        _, sha = self._stored_program(world, tmp_path, PROGRAM)
        with world.factory() as s, pytest.raises(Refusal) as caught:
            runs.create_rerun(s, sha, "heldout", "operator", world.settings)
        assert caught.value.code == "validation_failed"


def test_replays_and_reruns_count_toward_the_callers_queue(world: World, tmp_path: Path) -> None:
    run_id, _ = _session(world, tmp_path)
    limit = world.settings.wsp_max_queued_per_caller
    for _ in range(limit):
        _queue_replay(world, run_id)
    with world.factory() as s, pytest.raises(Refusal) as caught:
        runs.create_replay(s, run_id, "operator", world.settings)
    assert caught.value.code == "wsp_queue_limit_reached"
