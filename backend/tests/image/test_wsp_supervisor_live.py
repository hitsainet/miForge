"""The WSP supervisor as PID 1 of its image, run with the pod's flags and driven from outside
(feature 016; FTDD §8.1's live table; FR-016.10, FR-016.25; FTASKS 4.4; P51, P59, R3A-*).

Runs in the private-CI job ``wsp-worker-image`` (``backend/scripts/wsp_worker_image_job.py``),
which builds ``backend/Dockerfile.wsp-worker --target test`` and sets ``MIFORGE_WSP_WORKER_IMAGE``;
anywhere else every test here skips by name (and fails under ``MIFORGE_REQUIRE_WSP_IMAGE=1``).
``tests/image/harness.py`` says how a pod is emulated. The rows that need no container run
locally in ``integration/wsp/test_supervisor_local.py``; the ones here are those only PID 1 in a
pod-like sandbox can prove: ``kill(-1)``, the IPC wipe and ``/dev/mqueue``, signals from inside the
PID namespace, ``RLIMIT_NPROC``, the kubelet's termination-message mount, restarts in a network
namespace that outlives the process.
"""

from __future__ import annotations

import http.client
import json
import os
import signal
import socket
import time
from collections.abc import Iterator
from pathlib import Path

import pytest

from tests.image import harness as h
from tests.support.wsp_expected import assert_child_env
from tests.support.wsp_supervisor import Reply, encode_session, file_entry, push_doc
from wsp_worker import spec, supervisor

PORT = h.PORT


@pytest.fixture(scope="module")
def image() -> str:
    return h.require_image()


@pytest.fixture
def pod(image: str) -> Iterator[h.Pod]:
    with h.pod(image) as p:
        yield p


def _expect(reply: Reply, status: int) -> Reply:
    assert reply.status == status, reply
    return reply


def _json(members: dict[str, bytes], name: str) -> dict[str, object]:
    assert name in members, f"{name} not in the result: {sorted(members)}"
    return dict(json.loads(members[name]))


# ----------------------------------------------------------------------------- P51


def test_two_sessions_in_sequence(pod: h.Pod) -> None:
    """Session A's ``setsid`` survivor writes ``/dev/shm/alive`` and creates SysV and POSIX IPC;
    after A's ack the container exits 0 BY ITSELF and Docker restarts it with a new ``boot_id``;
    session B's child finds none of it (C10o: a supervisor that serves a second session without
    exiting never exits, and keeps its ``boot_id``)."""
    w = pod.start()
    boot_a, members, digest, record = w.run(["plant", "write_result"])
    planted = _json(members, "planted.json")
    assert all(int(v) >= 0 for v in planted["ids"].values()), planted  # type: ignore[attr-defined]
    assert record["exit_code"] == 0
    assert w.exit_on(lambda: _expect(w.ack(boot_a, digest), 202)) == 0
    boot_b = w.wait_ready(not_boot=boot_a)
    assert boot_b != boot_a
    _, members_b, _, _ = w.run(["probe"])
    assert _json(members_b, "probe.json") == {
        "others": [],
        "dev_shm": [],
        "sysv": {"shm": [], "sem": [], "msg": []},
        "mqueue": [],
    }


PLANT = {
    "work": "open('/work/planted', 'w').write('x')",
    "out": "open('/out/planted', 'w').write('x')",
    "tmp": "open('/tmp/planted', 'w').write('x')",
    "shm": "open('/dev/shm/planted', 'w').write('x')",
    "sysv_shm": "import ctypes; assert ctypes.CDLL(None).shmget(0, 4096, 0o1600) >= 0",
    "sysv_sem": "import ctypes; assert ctypes.CDLL(None).semget(0, 1, 0o1600) >= 0",
    "sysv_msg": "import ctypes; assert ctypes.CDLL(None).msgget(0, 0o1600) >= 0",
    "mqueue": (
        "import ctypes, os; f = ctypes.CDLL(None).mq_open; f.restype = ctypes.c_int; "
        "assert f(b'/wsp-planted', os.O_CREAT | os.O_RDWR, 0o600, None) >= 0"
    ),
}
_LEFT = (
    "import json, os\n"
    "def sysv(k):\n"
    "    rows = open('/proc/sysvipc/' + k).read().splitlines()\n"
    "    u = rows[0].split().index('uid')\n"
    "    return sum(1 for r in rows[1:] if int(r.split()[u]) == os.getuid())\n"
    "print(json.dumps({'work': os.listdir('/work'), 'out': os.listdir('/out'),\n"
    "    'tmp': sorted(set(os.listdir('/tmp')) - {'termination-log'}),\n"
    "    'shm': os.listdir('/dev/shm'), 'sysv_shm': sysv('shm'), 'sysv_sem': sysv('sem'),\n"
    "    'sysv_msg': sysv('msg'), 'mqueue': os.listdir('/dev/mqueue')}))\n"
)
_EMPTY = {
    "work": [],
    "out": [],
    "tmp": [],
    "shm": [],
    "sysv_shm": 0,
    "sysv_sem": 0,
    "sysv_msg": 0,
    "mqueue": [],
}


def _left(w: h.Worker) -> dict[str, object]:
    r = w.exec_python(_LEFT)
    assert r.returncode == 0, r.stderr
    return dict(json.loads(r.stdout))


@pytest.mark.parametrize("where", sorted(PLANT))
def test_wipe_dev_shm_and_ipc(pod: h.Pod, where: str) -> None:
    """A file planted in each named volume (they persist across restarts, as emptyDirs do), a
    SysV object and a POSIX queue, are gone before the next boot's ``/ready`` (C10q, run against
    each path; P62)."""
    w = pod.start()
    boot = w.boot()
    r = w.exec_python(PLANT[where])
    assert r.returncode == 0, r.stderr
    assert _left(w)[where] not in ([], 0), "the plant did not land"
    assert w.exit_on(lambda: _expect(w.post_json("/recycle", {"boot_id": boot}), 202)) == 0
    w.wait_ready(not_boot=boot)
    assert _left(w) == _EMPTY


def test_session_refused_when_not_fresh(pod: h.Pod) -> None:
    w = pod.start()
    stale = _expect(w.session(push_doc("0" * 32)), 409)
    assert stale.json()["refused"] == "worker_not_idle"
    boot = w.boot()
    _expect(w.session(push_doc(boot, actions=["sleep:3"])), 202)
    again = _expect(w.session(push_doc(boot, actions=["sleep:3"])), 409)
    assert again.json()["refused"] == "worker_not_idle"


def test_child_cannot_read_proc1(pod: h.Pod) -> None:
    """``PR_SET_DUMPABLE`` 0: ``/proc/1/environ`` is unreadable to the child (C10p rests on
    ``environ``; ``/proc/1/mem`` is blocked by Yama regardless; R2A-7)."""
    _, members, _, _ = pod.start().run(["read_proc1", "write_result"])
    assert _json(members, "proc1.json") == {"environ": "PermissionError", "mem": "PermissionError"}


# ----------------------------------------------------------------------------- P59: takeover


def test_child_cannot_take_over_the_port(pod: h.Pod) -> None:
    """SIGINT, SIGTERM, SIGHUP and SIGUSR1 to PID 1 from ``docker exec --user 10002`` (inside the
    PID namespace, outside the child's filter) change nothing; the child's own ``kill(1, ...)``,
    its ``bind`` on :9000 (any address, loopback; with and without ``SO_REUSEPORT``) and its
    ``listen`` are ``EPERM``, as are a ``kill`` whose pid hides 1 under the upper 32 bits and a
    ``prlimit64`` on PID 1; the session completes (C10w, C10x, C10ah, C10ai, C10aj)."""
    w = pod.start()
    boot, restarts = w.boot(), w.restart_count()
    sent = w.exec_python(
        "import os, signal\n"
        "for s in (signal.SIGINT, signal.SIGTERM, signal.SIGHUP, signal.SIGUSR1):\n"
        "    os.kill(1, s)\n"
    )
    assert sent.returncode == 0, sent.stderr
    time.sleep(1.5)
    assert w.boot() == boot and w.restart_count() == restarts
    same, members, _, record = w.run(["kill_pid1", f"bind_port:{PORT}", "write_result"])
    assert same == boot
    assert _json(members, "kill_pid1.json") == {
        "SIGINT": "EPERM",
        "SIGTERM": "EPERM",
        "SIGHUP": "EPERM",
        "SIGUSR1": "EPERM",
        "kill_pid1_truncated": "EPERM",
        "prlimit64_pid1": "EPERM",
    }
    bound = _json(members, "bind.json")
    assert len(bound) == 5 and set(bound.values()) == {"EPERM"}, bound
    assert record["exit_code"] == 0 and record["refusal"] is None
    assert w.boot() == boot and w.restart_count() == restarts


def test_child_cannot_drive_the_api(pod: h.Pod) -> None:
    """The child's ``POST /session``, a wrong-sha ``/ack`` and ``/recycle`` are refused; from
    the harness a wrong-sha ack is refused before and after ``/result`` (C10aa)."""
    w = pod.start()
    boot, members, digest, _ = w.run([f"child_api:{PORT}", "write_result"])
    assert _json(members, "child_api.json") == {"session": 409, "ack": 409, "recycle": 409}
    wrong = {"boot_id": boot, "run_id": "wsr_test", "result_sha256": "f" * 64}
    _expect(w.post_json("/ack", wrong), 409)
    assert w.ready().json()["state"] == "finished"
    assert w.exit_on(lambda: _expect(w.ack(boot, digest), 202)) == 0


def test_slowloris_does_not_stop_exit(pod: h.Pod) -> None:
    """A loopback slowloris from the child; the deadline still kills it on time. The harness
    sends NO request from the session's start to 5 s after the deadline, so a deadline enforced
    only on a request thread shows (C10y; R2A-2, R3A-11)."""
    budget = 4
    w = pod.start()
    boot = w.boot()
    _expect(w.session(push_doc(boot, actions=[f"slowloris:{PORT}"], time_budget_s=budget)), 202)
    time.sleep(budget + 5)  # the quiet window
    status = w.status()
    assert status["state"] == "finished" and status["deadline_hit"] is True, status
    members, digest = w.result()
    stamp = _json(members, "alive_ts.txt")
    alive_for = float(stamp["last"]) - float(stamp["first"])  # type: ignore[arg-type]
    assert alive_for < budget + 2, f"the child lived {alive_for:.1f}s past a {budget}s deadline"
    assert w.exit_on(lambda: _expect(w.ack(boot, digest), 202)) == 0


def test_descendants_dead_before_walk(pod: h.Pod) -> None:
    """A ``setsid`` descendant writing ``/out`` and the termination path is dead before the walk
    and the truncation: at ``finished``, before ``/ack``, the container holds only PID 1
    (C10z: ``killpg`` instead of ``kill(-1)``; R2A-20, R3A-11)."""
    w = pod.start()
    boot, _, digest, record = w.run(["setsid_writer", "write_result"])
    assert record["exit_code"] == 0
    procs = w.processes()
    assert len(procs) == 1, procs
    time.sleep(0.5)
    _, again = w.result()
    assert again == digest  # nothing wrote /out after the walk
    assert w.exit_on(lambda: _expect(w.ack(boot, digest), 202)) == 0


def test_resource_limits(pod: h.Pod) -> None:
    """A fork bomb stops at ``RLIMIT_NPROC`` (256, set by the child under PID 1); a reservation
    past ``work_max_bytes`` gives ``worker_disk_limit`` (R2A-13; not a hard cap, P68)."""
    w = pod.start()
    boot, members, digest, _ = w.run(["fork_bomb", "write_result"])
    fork = _json(members, "fork.json")
    assert fork["errno"] == "EAGAIN" and 0 < int(fork["forked"]) < 256, fork  # type: ignore[call-overload]
    limits = " ".join(fork["limits"])  # type: ignore[arg-type]
    assert limits.split()[-3:] == ["256", "256", "processes"], limits
    assert w.exit_on(lambda: _expect(w.ack(boot, digest), 202)) == 0
    w.wait_ready(not_boot=boot)
    _, _, _, record = w.run(["fallocate:33554432"], work_max_bytes=8 * 1024 * 1024)
    assert record["refusal"] == "worker_disk_limit", record


def test_restart_binds_at_once(pod: h.Pod) -> None:
    """Two consecutive restarts, each after an ack over a connection the server closed (its
    ``TIME_WAIT`` lives on in the sandbox's network namespace, as in a pod), each show a new
    ``boot_id`` within 5 s (C10ac: ``allow_reuse_address = False``; R3A-2)."""
    w = pod.start()
    for _ in range(2):
        boot, _, digest, _ = w.run(["write_result"])
        assert w.exit_on(lambda b=boot, d=digest: _expect(w.ack(b, d), 202)) == 0  # type: ignore[misc]
        w.wait_ready(timeout=5, not_boot=boot)


def test_filter_check_refuses_a_child_without_its_filter(pod: h.Pod) -> None:
    """A child that skips ``install()`` is refused ``worker_filter_missing`` and never gets its
    line (C10ad; R3A-5)."""
    _, members, _, record = pod.start(script=h.NO_FILTER).run([])
    assert record["refusal"] == "worker_filter_missing"
    assert "got_line.json" not in members


# ----------------------------------------------------------------------------- R3A-9


def _tasks(w: h.Worker) -> int:
    return len(os.listdir(f"/proc/{w.host_pid()}/task"))


def test_watchdog_death_ends_the_process(pod: h.Pod) -> None:
    """The watchdog raising ends the process (exit 70, then a new ``boot_id``); a stale heartbeat
    makes ``/ready`` answer 503; an idle supervisor recycles itself; a slow ``/session`` body is
    not cut off by that idle recycle; 64 loopback connections from the child leave at most
    ``MAX_CONNECTIONS`` handler threads (C10ae; R3A-9)."""
    w = pod.start(env={"WSP_TEST_HOOKS": json.dumps({"watchdog_raise_after_s": 3})})
    boot = w.boot()
    assert w.exit_on(lambda: None) == spec.EXIT_WATCHDOG
    w.wait_ready(not_boot=boot)

    w = pod.start(env={"WSP_TEST_HOOKS": json.dumps({"heartbeat_stall_after_s": 1})})
    assert w.ready().status == 200
    time.sleep(supervisor.HEARTBEAT_STALE_S + 2)
    assert w.ready().status == 503

    idle = {"WSP_TEST_HOOKS": json.dumps({"idle_recycle_s": 2})}
    w = pod.start(env=idle)
    assert w.exit_on(lambda: None, timeout=15) == 0  # the hook is live: idle, it recycles

    w = pod.start(env=idle)
    boot = w.boot()
    data = os.urandom(4096)
    push = push_doc(boot, actions=["write_result"], files=[file_entry("heads.safetensors", data)])
    ctype, body = encode_session(push, [("heads.safetensors", data)])
    cut = body.index(b"heads.safetensors") + 200  # the push part, then the file part, slowly
    conn = http.client.HTTPConnection(w.pod.ip, PORT, timeout=30)
    conn.putrequest("POST", "/session")
    conn.putheader("Content-Type", ctype)
    conn.putheader("Content-Length", str(len(body)))
    conn.endheaders()
    conn.send(body[:cut])
    for i in range(cut, len(body), 1024):
        time.sleep(1.2)  # 4+ s in all: twice the idle limit
        conn.send(body[i : i + 1024])
    reply = conn.getresponse()
    assert reply.status == 202, reply.read()
    conn.close()
    assert w.wait_state("finished")["boot_id"] == boot

    w = pod.start()
    boot = w.boot()
    _expect(
        w.session(push_doc(boot, actions=["sleep:1.5", f"connections:{PORT}:64:4", "sleep:1"])),
        202,
    )
    time.sleep(0.8)
    baseline, peak = _tasks(w), 0
    end = time.monotonic() + 4.5
    while time.monotonic() < end:
        peak = max(peak, _tasks(w))
        time.sleep(0.1)
    w.wait_state("finished")
    members, _ = w.result()
    assert int(members["connections_opened.txt"]) == 64
    assert peak - baseline <= supervisor.MAX_CONNECTIONS, (baseline, peak)


def test_child_signal_dispositions_default(pod: h.Pod) -> None:
    """The child's ``SigIgn`` equals a fresh ``python -I``'s in the same image; a child exiting 3
    is recorded with ``exit_code`` 3, and PID 1 neither catches nor ignores ``SIGCHLD`` (C10af,
    C10ag; R3A-21, R4A-3)."""
    w = pod.start()
    _, members, _, record = w.run(["dump_status", "exit:3"])
    assert record["exit_code"] == 3
    theirs = next(
        ln for ln in members["status.txt"].decode().splitlines() if ln.startswith("SigIgn")
    )
    fresh = w.exec_python(
        "print(next(l for l in open('/proc/self/status') if l.startswith('SigIgn')).strip())"
    )
    assert fresh.returncode == 0, fresh.stderr
    assert theirs.split() == fresh.stdout.split()
    masks = {
        line.split(":")[0]: int(line.split()[1], 16)
        for line in Path(f"/proc/{w.host_pid()}/status").read_text().splitlines()
        if line.startswith(("SigCgt", "SigIgn"))
    }
    bit = 1 << (signal.SIGCHLD - 1)
    assert not masks["SigCgt"] & bit and not masks["SigIgn"] & bit, masks


@pytest.mark.parametrize("reading", ["256", "max"])
def test_pids_max_recorded_not_enforced(pod: h.Pod, reading: str) -> None:
    """``pids.max`` is recorded in ``supervisor.json`` and never refuses a session, including
    ``max``, what a pod with no pids limit reads (R3A-3). ``256`` is the real cgroup value under
    ``--pids-limit 256``. ``max`` is a file bind-mounted over the container's
    ``/sys/fs/cgroup/pids.max``: this Docker daemon applies a default limit even to
    ``--pids-limit -1``, so an unlimited cgroup cannot be had here, and the reading is what the
    supervisor's decision rests on."""
    mounts: tuple[str, ...] = ()
    if reading == "max":
        fake = pod.termdir / "pids.max"
        fake.write_text("max\n")
        fake.chmod(0o644)
        mounts = (f"type=bind,src={fake},dst=/sys/fs/cgroup/pids.max,readonly",)
    _, _, _, record = pod.start(extra_mounts=mounts).run(["write_result"])
    assert record["refusal"] is None and record["exit_code"] == 0
    assert record["pids_max"] == reading


def test_refuses_without_mqueue(pod: h.Pod) -> None:
    """Without the IPC namespace's ``/dev/mqueue`` the supervisor refuses to serve: it could not
    see, so could not wipe, a POSIX queue (R2A-7)."""
    w = pod.start(mqueue=False, restart="no", wait_ready=False)
    assert w.wait_exit() == spec.EXIT_REFUSED
    refused = w.refusal()
    assert "/dev/mqueue" in str(refused.get("message")), refused


def test_child_env_exact(pod: h.Pod) -> None:
    """The child's environment at exec is exactly ``CHILD_ENV`` (C10r)."""
    _, members, _, _ = pod.start().run(["dump_env"])
    assert_child_env(json.loads(members["env.json"]))


def test_unsafe_out_refused(pod: h.Pod) -> None:
    """A symlink to ``/proc/self/environ``, a FIFO, a hard link and an oversize file each give a
    ``worker_result_unsafe`` record, and none of their targets reaches the result (C10s)."""
    big = str(64 * 1_048_576 + 1)
    _, members, _, record = pod.start().run(
        ["write_result", "symlink", "fifo", "hardlink", f"oversize:{big}"]
    )
    assert record["unsafe_reason"] == "worker_result_unsafe"
    reasons = " | ".join(str(r) for r in record["unsafe"])
    for needle in (
        "transcript.jsonl: a symlink",
        "fifo: a fifo",
        "a.txt: a hard link",
        "big.bin:",
    ):
        assert needle in reasons, reasons
    assert set(members) == {"result.json", "logs/stdout.txt", "logs/stderr.txt", "supervisor.json"}
    assert b"WSP_MODE" not in b"".join(members.values())


def test_child_output_not_in_container_log(pod: h.Pod) -> None:
    """A printed sentinel reaches ``/out/logs/stdout.txt`` and never ``docker logs`` (C10t)."""
    sentinel = "SENTINEL-" + os.urandom(6).hex()
    w = pod.start()
    boot, members, digest, _ = w.run([f"print:{sentinel}", "write_result"])
    assert sentinel.encode() in members["logs/stdout.txt"]
    assert w.exit_on(lambda: _expect(w.ack(boot, digest), 202)) == 0
    assert sentinel not in w.logs()


def test_termination_message_truncated(pod: h.Pod) -> None:
    """The child's sentinel reaches the kubelet's termination file (a bind mount the supervisor
    cannot unlink), and is gone when the container has exited (C10u). ``--restart no`` here, so
    the read happens before any next boot's start-up wipe could empty the file instead."""
    sentinel = "SENTINEL-" + os.urandom(6).hex()
    w = pod.start(restart="no")
    boot, _, digest, _ = w.run([f"write_termination:{sentinel}", "write_result"])
    assert sentinel in pod.termlog.read_text()
    _expect(w.ack(boot, digest), 202)
    assert w.wait_exit() == 0
    assert pod.termlog.read_text() == ""


def test_pid1_required_in_prod(pod: h.Pod) -> None:
    """``WSP_ENVIRONMENT=prod`` and not PID 1 (``--init`` makes ``docker-init`` PID 1): the
    supervisor refuses ``supervisor_not_pid1`` (C10v)."""
    w = pod.start(
        script=None, env={"WSP_ENVIRONMENT": "prod"}, init=True, restart="no", wait_ready=False
    )
    assert w.wait_exit() == spec.EXIT_REFUSED
    assert w.refusal().get("refused") == "supervisor_not_pid1", w.logs()


def test_prod_serves_as_pid1_with_the_real_child(pod: h.Pod) -> None:
    """The positive side: ``prod`` as PID 1, no test script. The real child passes the filter
    check and refuses an incomplete session (no manifest or request in its spec) by name."""
    w = pod.start(script=None, env={"WSP_ENVIRONMENT": "prod"})
    _, members, _, record = w.run([])
    assert record["refusal"] is None
    assert record["exit_code"] == spec.EXIT_SPEC_INVALID
    assert b"worker_spec_invalid" in members["logs/stderr.txt"]


def test_deadline_ignores_sigalrm(pod: h.Pod) -> None:
    """A child that ignores ``SIGALRM`` and loops is killed at the watchdog's deadline
    (R1B-A16)."""
    _, _, _, record = pod.start().run(["ignore_sigalrm_loop"], time_budget_s=3)
    assert record["deadline_hit"] is True and record["exit_code"] == -signal.SIGKILL


def test_the_harness_reaches_only_the_pod(pod: h.Pod) -> None:
    """The emulation itself: the sandbox's network is internal (no route out), the worker joins
    the sandbox's network and IPC namespaces, and runs with the pod's uid and no capability."""
    w = pod.start()
    r = w.exec_python(
        "import socket\n"
        "try:\n"
        "    socket.create_connection(('1.1.1.1', 443), timeout=3); print('out')\n"
        "except OSError as e:\n"
        "    print(type(e).__name__)\n"
    )
    assert r.stdout.strip() != "out", r.stdout
    ids = w.exec_python("import os; print(os.getuid(), os.getgid())")
    assert ids.stdout.split() == ["10002", "10002"]
    caps = w.exec_python("print([l for l in open('/proc/1/status') if l.startswith('CapEff')])")
    assert "0000000000000000" in caps.stdout, caps.stdout
    with socket.create_connection((w.pod.ip, PORT), timeout=5):
        pass
