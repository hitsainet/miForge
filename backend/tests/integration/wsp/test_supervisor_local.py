"""The real WSP supervisor, driven over HTTP as a local (non-PID-1) process (FTASKS 4.4b, 4.5; FTDD
§5.4a, §8.1; FTID §3.2).

The rows of FTDD §8.1's live table that do not need the container: one session per process, the
refusals of a second or stale session and of a wrong-sha ack, the exact child environment, the
safe walk, the heartbeat and the watchdog's death, the connection cap, the deadline, the filter
check, the child's dispositions and real exit code, signals to the supervisor, a restart binding
through ``TIME_WAIT``, the PID-1 check in prod, the start-up wipe of the roots. ``kill(-1)``, the
IPC wipe, ``/dev/mqueue``, ``RLIMIT_NPROC`` and the takeover from ``docker exec`` are the image
job's (FTASKS 4.4).

Every supervisor here is started by the test with a fresh launcher root; the test only signals
the process it started (and the tree under it at cleanup).
"""

from __future__ import annotations

import json
import os
import signal
import socket
import subprocess
import sys
import time
from collections.abc import Callable, Iterator
from pathlib import Path

import pytest

from tests.support import wsp_supervisor as ws
from tests.support import wsp_worker_env
from tests.support.wsp_expected import assert_child_env
from wsp.launcher import make_worker_root
from wsp_worker import spec, supervisor

Start = Callable[..., ws.LocalSupervisor]


@pytest.fixture(autouse=True)
def _seccomp() -> None:
    wsp_worker_env.require_seccomp()


@pytest.fixture
def start() -> Iterator[Start]:
    started: list[ws.LocalSupervisor] = []
    roots: list[Path] = []

    def _start(**kwargs: object) -> ws.LocalSupervisor:
        root = kwargs.pop("root", None) or make_worker_root()
        roots.append(root)  # type: ignore[arg-type]
        sup = ws.start(root, **kwargs)  # type: ignore[arg-type]
        started.append(sup)
        return sup

    yield _start
    import shutil

    for sup in started:
        sup.stop()
    for root in roots:
        shutil.rmtree(root, ignore_errors=True)


def _run(sup: ws.LocalSupervisor, actions: list[str], **kw: object) -> tuple[dict[str, bytes], str]:
    boot = sup.ready().json()["boot_id"]
    reply = sup.session(ws.push_doc(boot, actions=actions, **kw))  # type: ignore[arg-type]
    assert reply.status == 202, reply
    sup.wait_state("finished", timeout=30)
    return sup.result()


def _record(members: dict[str, bytes]) -> dict[str, object]:
    return dict(json.loads(members["supervisor.json"]))


# ----------------------------------------------------------------------------- one session


def test_one_session_then_the_process_exits_and_a_new_one_has_a_new_boot(start: Start) -> None:
    """C10o: after a valid ack the process exits 0 by itself; a second session is refused."""
    a = start()
    boot = a.ready().json()["boot_id"]
    data = b"weights"
    push = ws.push_doc(
        boot, actions=["write_result"], files=[ws.file_entry("heads.safetensors", data)]
    )
    assert a.session(push, [("heads.safetensors", data)]).status == 202
    second = a.session(push, [("heads.safetensors", data)])
    assert second.status == 409 and second.json()["refused"] == "worker_not_idle"
    a.wait_state("finished")
    members, digest = a.result()
    assert members["result.json"] and _record(members)["exit_code"] == 0
    assert (
        a.post_json("/ack", {"boot_id": boot, "run_id": "wsr_test", "result_sha256": digest}).status
        == 202
    )
    assert a.wait_exit(timeout=10) == 0
    assert all(not os.listdir(a.root / d) for d in ("work", "out", "tmp", "shm"))
    b = start()
    assert b.ready().json()["boot_id"] != boot


def test_a_stale_boot_id_is_refused(start: Start) -> None:
    sup = start()
    reply = sup.session(ws.push_doc("0" * 32))
    assert reply.status == 409 and reply.json()["refused"] == "worker_not_idle"
    assert sup.ready().json()["state"] == "idle"


def test_a_file_that_does_not_match_its_sha_is_refused_and_the_process_exits(start: Start) -> None:
    sup = start()
    boot = sup.ready().json()["boot_id"]
    entry = ws.file_entry("heads.safetensors", b"declared")
    reply = sup.session(ws.push_doc(boot, files=[entry]), [("heads.safetensors", b"different")])
    assert reply.status == 422 and reply.json()["refused"] == "worker_file_mismatch"
    assert sup.wait_exit(timeout=10) == 0


def test_the_childs_own_post_session_is_refused(start: Start) -> None:
    sup = start()
    members, _ = _run(sup, [f"post_session:{sup.port}"])
    assert members["post_session.txt"].startswith(b"HTTP/1.0 409")


# ----------------------------------------------------------------------------- the ack


def test_an_ack_with_a_wrong_sha_is_refused_before_and_after_the_result(start: Start) -> None:
    """C10aa: only the served result's sha256 ends the session."""
    sup = start()
    boot = sup.ready().json()["boot_id"]
    assert sup.session(ws.push_doc(boot, actions=["write_result"])).status == 202
    sup.wait_state("finished")
    wrong = {"boot_id": boot, "run_id": "wsr_test", "result_sha256": "f" * 64}
    assert sup.post_json("/ack", wrong).status == 409
    _, digest = sup.result()
    assert sup.post_json("/ack", wrong).status == 409
    assert sup.post_json("/recycle", {"boot_id": boot}).status == 409
    assert sup.ready().json()["state"] == "finished"
    good = dict(wrong, result_sha256=digest)
    assert sup.post_json("/ack", good).status == 202
    assert sup.wait_exit(timeout=10) == 0


def test_recycle_while_idle_exits(start: Start) -> None:
    sup = start()
    boot = sup.ready().json()["boot_id"]
    assert sup.post_json("/recycle", {"boot_id": "0" * 32}).status == 409
    assert sup.post_json("/recycle", {"boot_id": boot}).status == 202
    assert sup.wait_exit(timeout=10) == 0


# ----------------------------------------------------------------------------- the child


def test_the_child_env_is_exactly_child_env(start: Start) -> None:
    """C10r: the environment at exec (``/proc/self/environ``) is the literal, nothing more."""
    members, _ = _run(start(), ["dump_env"])
    assert_child_env(json.loads(members["env.json"]))


def test_the_child_line_carries_the_time_its_kill_leaves(start: Start) -> None:
    """Fix 2026-11-21: the supervisor writes the time budget left (from ``running_mono``, where
    the watchdog's kill is measured) into the child line, for the engine's graceful end."""
    members, _ = _run(start(), ["dump_time_left"], time_budget_s=30)
    left = json.loads(members["time_left.json"])
    # Below the full budget: the child's start and its filter check came off it first.
    assert 25 < left < 30


def test_the_childs_output_reaches_only_its_log(start: Start) -> None:
    """C10t: a printed sentinel is in ``logs/stdout.txt``, never on the supervisor's streams."""
    sup = start()
    members, digest = _run(sup, ["print:SENTINEL-7f3a"])
    assert b"SENTINEL-7f3a" in members["logs/stdout.txt"]
    boot = sup.ready().json()["boot_id"]
    sup.post_json("/ack", {"boot_id": boot, "run_id": "wsr_test", "result_sha256": digest})
    sup.wait_exit(timeout=10)
    assert sup.process.stdout is not None and sup.process.stderr is not None
    assert b"SENTINEL-7f3a" not in sup.process.stdout.read() + sup.process.stderr.read()


def test_the_child_cannot_read_the_supervisors_environ(start: Start) -> None:
    """C10p: ``PR_SET_DUMPABLE`` 0 makes the supervisor's ``/proc`` files unreadable to the child
    (here its parent; in the pod, PID 1)."""
    members, _ = _run(start(extra_env={"WSP_CANARY": "secret-canary"}), ["read_parent_environ"])
    assert "parent_environ.txt" not in members
    assert members["parent_environ_error.txt"] == b"PermissionError"


def test_a_child_without_its_filter_never_gets_its_line(start: Start) -> None:
    """C10ad (R3A-5): a child that skips ``install()`` is refused ``worker_filter_missing``."""
    members, _ = _run(start(script=ws.NO_FILTER), [])
    assert _record(members)["refusal"] == "worker_filter_missing"
    assert "got_line.json" not in members


def _default_dispositions() -> None:
    import contextlib
    import signal

    for sig in signal.valid_signals():
        if sig not in (signal.SIGKILL, signal.SIGSTOP):
            with contextlib.suppress(OSError, ValueError):
                signal.signal(sig, signal.SIG_DFL)


def test_the_childs_dispositions_are_a_fresh_interpreters(start: Start) -> None:
    """C10af (R3A-21, R4A-3): the child's ``SigIgn`` equals a fresh ``python -I``'s."""
    members, _ = _run(start(), ["dump_status"])
    theirs = next(
        ln for ln in members["status.txt"].decode().splitlines() if ln.startswith("SigIgn")
    )
    fresh = subprocess.run(
        [
            sys.executable,
            "-I",
            "-c",
            "print(next(l for l in open('/proc/self/status') if l.startswith('SigIgn')).strip())",
        ],
        capture_output=True,
        text=True,
        check=True,
        # A fresh interpreter from DEFAULT dispositions: without this it inherits whatever the
        # test runner ignores (SIGHUP under nohup or a detached suite, SIGINT under some
        # runners), which made this test fail only in long background runs (review OR1-016-12).
        preexec_fn=_default_dispositions,
    ).stdout.strip()
    assert theirs.split() == fresh.split()


def test_the_child_runs_under_its_file_size_limit(start: Start) -> None:
    """Review OR2-016-5: ``child_preamble`` caps every file the child writes (``RLIMIT_FSIZE``),
    so a runaway write ends with ``SIGXFSZ`` instead of filling ``/out``. Control
    ``S13-iso-fsize`` (the ``setrlimit`` line removed) turns this red."""
    from wsp_worker.__main__ import CHILD_FSIZE

    members, _ = _run(start(), ["dump_limits"])
    assert json.loads(members["limits.json"]) == {"fsize": [CHILD_FSIZE, CHILD_FSIZE]}
    assert CHILD_FSIZE == 4 * 1024**3


def test_the_childs_real_exit_code_is_recorded(start: Start) -> None:
    """C10ag: ``SIGCHLD`` keeps its default, so ``Popen.wait`` reads 3, not 0."""
    sup = start()
    members, _ = _run(sup, ["exit:3"])
    assert _record(members)["exit_code"] == 3
    # The race a reaping handler loses only sometimes is visible in the disposition itself:
    # SIGCHLD is neither caught nor ignored by the supervisor.
    masks = {
        line.split(":")[0]: int(line.split()[1], 16)
        for line in Path(f"/proc/{sup.process.pid}/status").read_text().splitlines()
        if line.startswith(("SigCgt", "SigIgn"))
    }
    bit = 1 << (signal.SIGCHLD - 1)
    assert not masks["SigCgt"] & bit and not masks["SigIgn"] & bit, masks


def test_a_child_ignoring_sigalrm_is_killed_at_the_deadline(start: Start) -> None:
    """R1B-A16: the watchdog's deadline, not an in-process alarm."""
    sup = start()
    began = time.monotonic()
    members, _ = _run(sup, ["ignore_sigalrm_loop"], time_budget_s=2)
    record = _record(members)
    assert record["deadline_hit"] is True and record["exit_code"] == -signal.SIGKILL
    assert time.monotonic() - began < 15


def test_a_setsid_descendant_is_dead_before_the_walk(start: Start) -> None:
    """Locally the supervisor is a child subreaper, so a ``setsid`` survivor re-parents to it and
    teardown kills it (R2A-20's local half; as PID 1 it is ``kill(-1)``, the image job's)."""
    sup = start()
    _run(sup, ["setsid_sleeper", "sleep:0.5"])
    pid = int((sup.root / "tmp" / "sleeper.pid").read_text())
    alive = ws.signal_alive(pid)
    if alive:  # the test's own grandchild: never leave it running when the assertion fails
        os.kill(pid, signal.SIGKILL)
    assert not alive
    from wsp.launcher import tree

    assert tree(sup.process.pid) == {sup.process.pid}


# ----------------------------------------------------------------------------- /out (R1B-M2)


def test_unsafe_out_entries_are_recorded_and_never_read(start: Start) -> None:
    """C10s: a symlink, a FIFO, a hard link, an oversize file and a planted ``supervisor.json``."""
    big = str(64 * 1_048_576 + 1)
    members, _ = _run(
        start(),
        ["write_result", "symlink", "fifo", "hardlink", f"oversize:{big}", "plant_supervisor_json"],
    )
    record = _record(members)
    assert record["unsafe_reason"] == "worker_result_unsafe"
    reasons = " | ".join(str(r) for r in record["unsafe"])  # type: ignore[attr-defined]
    for needle in (
        "transcript.jsonl: a symlink",
        "fifo: a fifo",
        "a.txt: a hard link",
        "b.txt: a hard link",
        "big.bin:",
        "supervisor.json: a name only",
    ):
        assert needle in reasons, reasons
    assert set(members) == {"result.json", "logs/stdout.txt", "logs/stderr.txt", "supervisor.json"}
    assert b"WSP_MODE" not in b"".join(members.values())


# ----------------------------------------------------------------------------- the process


def test_ready_answers_503_when_the_heartbeat_stalls(start: Start) -> None:
    sup = start(extra_env={"WSP_TEST_HOOKS": json.dumps({"heartbeat_stall_after_s": 0.5})})
    assert sup.ready().status == 200
    time.sleep(supervisor.HEARTBEAT_STALE_S + 1.5)
    assert sup.ready().status == 503


def test_the_watchdogs_death_ends_the_process(start: Start) -> None:
    """C10ae: an exception in the watchdog thread exits 70."""
    sup = start(extra_env={"WSP_TEST_HOOKS": json.dumps({"watchdog_raise_after_s": 0.5})})
    assert sup.wait_exit(timeout=10) == spec.EXIT_WATCHDOG


def test_signals_from_the_same_uid_do_not_stop_the_supervisor(start: Start) -> None:
    """C10w: SIGINT, SIGTERM, SIGHUP, SIGUSR1 are ignored (sent to the process the test started)."""
    sup = start()
    boot = sup.ready().json()["boot_id"]
    for sig in (signal.SIGINT, signal.SIGTERM, signal.SIGHUP, signal.SIGUSR1):
        os.kill(sup.process.pid, sig)
    time.sleep(0.5)
    assert sup.process.poll() is None
    assert sup.ready().json()["boot_id"] == boot


def test_the_connection_cap_bounds_handler_threads(start: Start) -> None:
    """R3A-9: 64 idle connections leave at most MAX_CONNECTIONS handler threads."""
    sup = start()
    tasks = Path(f"/proc/{sup.process.pid}/task")
    baseline = len(os.listdir(tasks))
    conns = [socket.create_connection(("127.0.0.1", sup.port)) for _ in range(64)]
    try:
        time.sleep(1.0)
        assert len(os.listdir(tasks)) - baseline <= supervisor.MAX_CONNECTIONS
    finally:
        for c in conns:
            c.close()
    time.sleep(0.5)
    end = time.monotonic() + supervisor.HANDLER_TIMEOUT_S + 5
    while sup.ready().status != 200:
        assert time.monotonic() < end


def test_a_restart_binds_its_port_through_time_wait(start: Start) -> None:
    """C10ac (R3A-2): the server closed connections, leaving TIME_WAIT on its port."""
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        port = s.getsockname()[1]
    a = start(port=port)
    boot = a.ready().json()["boot_id"]
    assert a.post_json("/recycle", {"boot_id": boot}).status == 202
    assert a.wait_exit(timeout=10) == 0
    began = time.monotonic()
    b = start(port=port)
    assert b.ready().json()["boot_id"] != boot
    assert time.monotonic() - began < 5


def test_prod_requires_pid1(start: Start) -> None:
    """C10v: with ``WSP_ENVIRONMENT=prod`` and not PID 1, it refuses to start."""
    sup = start(environment="prod", script=None, expect_listening=False)
    code, refused = ws.refusal(sup.process)
    assert code == spec.EXIT_REFUSED and refused["refused"] == "supervisor_not_pid1"


def test_the_start_up_wipe_empties_every_root(start: Start) -> None:
    """C10q (the directory half; the IPC half is PID 1's and the image job's)."""
    root = make_worker_root()
    for name in ("work", "out", "tmp", "shm"):
        (root / name).mkdir()
        (root / name / "planted").write_text("left by a previous session")
    sup = start(root=root)
    assert sup.ready().status == 200
    assert all(not os.listdir(root / name) for name in ("work", "out", "tmp", "shm"))


def test_pids_max_is_recorded_not_enforced(start: Start) -> None:
    members, _ = _run(start(), ["write_result"])
    assert _record(members)["pids_max"] == supervisor.read_pids_max()


def test_the_real_child_passes_the_filter_check_and_refuses_an_incomplete_session(
    start: Start,
) -> None:
    """No test script: the supervisor's own child argv, the real preamble and filter. A session
    push without the bundle manifest, the editor and a gateway is refused by name BEFORE the
    weightspace adapter is imported (FTASKS 8.9 built the adapter; the spec check guards it)."""
    members, _ = _run(start(script=None), [])
    record = _record(members)
    assert record["refusal"] is None
    assert record["exit_code"] == spec.EXIT_SPEC_INVALID
    assert b"worker_spec_invalid" in members["logs/stderr.txt"]


def test_an_unknown_mode_is_refused() -> None:
    r = subprocess.run(
        [sys.executable, "-m", "wsp_worker", "shell"],
        cwd=str(ws.SRC),
        env={"PATH": os.environ.get("PATH", "/usr/bin:/bin")},
        capture_output=True,
        timeout=30,
    )
    assert r.returncode == 64 and b"worker_mode_unknown" in r.stderr


# ----------------------------------------------------------------------------- feature 018 (7.6)


def test_an_engine_session_runs_end_to_end_under_the_real_supervisor(
    start: Start, tmp_path: Path
) -> None:
    """FR-018.2 and FR-018.14 under the real supervisor and the real child (no test script): an
    ``miwsp_engine`` session against a scripted endpoint reached as the gateway; its kernel
    grandchild is dead before the walk; ``/out`` holds only names the intake accepts."""
    from tests.support.wsp_fixtures import make_bundle
    from tests.support.wsp_scripted_endpoint import ScriptedEndpoint
    from wsp.result_archive import RESULT_MEMBERS

    fx = make_bundle(tmp_path / "bundle", rule="mean")
    data = {e["name"]: (fx.root / e["name"]).read_bytes() for e in fx.manifest["files"]}
    entries = [ws.file_entry(e["name"], data[e["name"]], e["role"]) for e in fx.manifest["files"]]
    record_pid = "import os; open('../kernel.pid', 'w').write(str(os.getpid()))"
    steps = [
        {"calls": [("python", {"code": record_pid})]},
        {"calls": [("python", {"code": "W['weight'][0] += 1.0"})]},
        {"calls": [("submit", {"answer": "done", "confidence": 0.5})]},
    ]
    request = {
        "kind": "repair",
        "task": "Repair the probe.",
        "parent": {"kind": "bundle", "ref": fx.manifest["bundle_sha256"]},
        "provider": {"id": "miwsp_engine", "version": "0.1.0"},
        "budgets": {"tokens": 100_000, "time_s": 60, "max_steps": 10, "forward_passes": 0},
        "change_limits": {
            "max_changed_entries": None,
            "allowed_tensors": None,
            "frozen_tensors": [],
        },
        "thinking": None,
        "observation_tools": [],
    }
    editor = {"kind": "openai_compatible", "model": "m", "serving": "generic", "prices": {}}
    sup = start(script=None)
    with ScriptedEndpoint("openai_compatible", steps) as ep:
        boot = sup.ready().json()["boot_id"]
        push = ws.push_doc(
            boot,
            files=entries,
            gateway=ep.gateway,
            time_budget_s=60,
            request=request,
            manifest=fx.manifest,
            editor=editor,
        )
        reply = sup.session(push, list(data.items()))
        assert reply.status == 202, reply
        sup.wait_state("finished", timeout=90)
        members, _ = sup.result()
    record = _record(members)
    assert record["exit_code"] == 0 and record["unsafe_reason"] is None, (
        record,
        members.get("logs/stderr.txt"),
    )
    assert set(members) <= RESULT_MEMBERS, sorted(members)
    assert {"result.json", "transcript.jsonl", "patch/patch.json", "patch/edit.npz"} <= set(members)
    result = json.loads(members["result.json"])
    assert (result["provider"], result["stop_reason"]) == ("miwsp_engine", "submitted")
    # 018 FR-018.11a: the engine's real result.json, read by the backend's own intake reader
    # (producer and consumer in one assertion: D-4's four keys and reasoning_turns, exactly).
    from wsp import runs

    counters = runs.result_counters(result)
    # Paging (2026-11-21): this request names none (on, the default), but the endpoint states no
    # window, so the engine works in the default 8,192 and switches paging off by itself
    # (window_small, the coordinator's decision): no ledger. The peak is the engine's estimate.
    assert isinstance(counters.pop("peak_prompt_tokens"), int)
    assert (result["paging"], result["paging_off_reason"]) == ("off", "window_small")
    assert counters == {
        "forward_passes_used": 0,
        "action_protocol": "tool_calls",
        "handoffs": 0,
        "refused_turns": 0,
        "kernel_restarts": 0,
        "reasoning_turns": 0,
        "repeated_turns": 0,
        "pages_out": 0,
        "recalls": 0,
        "ledger_rows": 0,
    }
    assert len(ep.requests) == 3
    pid = int((sup.root / "tmp" / "kernel.pid").read_text())
    alive = ws.signal_alive(pid)
    if alive:  # never leave the test's own grandchild running when the assertion fails
        os.kill(pid, signal.SIGKILL)
    assert not alive
    from wsp.launcher import tree

    assert tree(sup.process.pid) == {sup.process.pid}
