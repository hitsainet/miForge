"""The scripted child's actions for the image job's live tests (FTASKS 4.4, 8.5; FTDD §8.1).

Each runs INSIDE the child (its seccomp filter installed, its line read) in the worker container,
where the supervisor is PID 1, and records what it saw as JSON under ``/out`` for the harness to
read from the pulled result. Each is named ``<action>:<arg>`` in ``spec.test_actions``.
Standard library only (the image has no test dependency the child may import).
"""

from __future__ import annotations

import contextlib
import ctypes
import errno
import http.client
import json
import os
import signal
import socket
import subprocess
import sys
import time
from collections.abc import Callable
from pathlib import Path

_libc = ctypes.CDLL(None, use_errno=True)
IPC_PRIVATE, IPC_CREAT = 0, 0o1000
SECRETS = ("/var/run/secrets", "/run/secrets")
#: What the writable-path walk never enters: kernel views, not filesystems a file can land on.
NOT_WALKED = ("/proc", "/sys")
#: x86-64 syscall numbers (the child's filter is x86-64 only).
SYS_KILL, SYS_PRLIMIT64 = 62, 302


def _errno_name(exc: OSError) -> str:
    return errno.errorcode.get(exc.errno or 0, str(exc.errno))


def _write(out: Path, name: str, doc: object) -> None:
    (out / name).write_text(json.dumps(doc, sort_keys=True))


# ----------------------------------------------------------------------------- session A / B


PLANTER = r"""
import ctypes, json, os, sys, time
libc = ctypes.CDLL(None, use_errno=True)
ids = {
    "shm": libc.shmget(0, 4096, 0o1600),
    "sem": libc.semget(0, 1, 0o1600),
    "msg": libc.msgget(0, 0o1600),
}
mq_open = libc.mq_open
mq_open.restype = ctypes.c_int
ids["mq"] = mq_open(b"/wsp-plant", os.O_CREAT | os.O_RDWR, 0o600, None)
with open("/dev/shm/alive", "w") as fh:
    fh.write(str(os.getpid()))
with open(sys.argv[1], "w") as fh:
    json.dump({"pid": os.getpid(), "ids": ids}, fh)
time.sleep(600)
"""


def plant(out: Path, tmp: Path, arg: str) -> None:
    """Session A: a ``setsid`` process writes ``/dev/shm/alive`` and creates a SysV segment,
    semaphore set and message queue, and a POSIX queue; it then sleeps, alive at teardown."""
    record = out / "planted.json"
    with open(out / "planter.err", "wb") as err:  # in the result, if the planter fails
        subprocess.Popen(
            [sys.executable, "-I", "-c", PLANTER, str(record)],
            start_new_session=True,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.DEVNULL,
            stderr=err,
        )
    end = time.monotonic() + 10
    while not record.exists() and time.monotonic() < end:
        time.sleep(0.05)


def _sysv(kind: str) -> list[str]:
    lines = Path(f"/proc/sysvipc/{kind}").read_text().splitlines()
    header = lines[0].split()
    uid = header.index("uid")
    return [ln for ln in lines[1:] if int(ln.split()[uid]) == os.getuid()]


def probe(out: Path, tmp: Path, arg: str) -> None:
    """Session B: what of session A (or of a plant) is still here."""
    me, parent = os.getpid(), os.getppid()
    others = sorted(
        int(p) for p in os.listdir("/proc") if p.isdigit() and int(p) not in (1, me, parent)
    )
    _write(
        out,
        "probe.json",
        {
            "others": others,
            "dev_shm": sorted(os.listdir("/dev/shm")),
            "sysv": {k: _sysv(k) for k in ("shm", "sem", "msg")},
            "mqueue": sorted(os.listdir("/dev/mqueue")),
        },
    )


# ----------------------------------------------------------------------------- takeover (P59)


def kill_pid1(out: Path, tmp: Path, arg: str) -> None:
    """The child's own ``kill(1, sig)``: its filter answers ``EPERM`` (C10ai)."""
    seen: dict[str, str] = {}
    for sig in (signal.SIGINT, signal.SIGTERM, signal.SIGHUP, signal.SIGUSR1):
        try:
            os.kill(1, sig)
            seen[sig.name] = "delivered"
        except OSError as exc:
            seen[sig.name] = _errno_name(exc)
    # The same signal with PID 1 hidden in the upper 32 bits: the kernel reads a 32-bit pid, so
    # the filter must compare under a mask (C10ah). And prlimit64 on PID 1, read only (C10aj).
    syscall = _libc.syscall
    syscall.restype = ctypes.c_long
    rc = syscall(ctypes.c_long(SYS_KILL), ctypes.c_long(0x1_0000_0001), ctypes.c_long(0))
    seen["kill_pid1_truncated"] = "delivered" if rc == 0 else _errno_name(_last_error())
    old = (ctypes.c_uint64 * 2)()
    rc = syscall(ctypes.c_long(SYS_PRLIMIT64), ctypes.c_long(1), ctypes.c_long(7), None, old)
    seen["prlimit64_pid1"] = "read" if rc == 0 else _errno_name(_last_error())
    _write(out, "kill_pid1.json", seen)


def _last_error() -> OSError:
    err = ctypes.get_errno()
    return OSError(err, os.strerror(err))


def bind_port(out: Path, tmp: Path, arg: str) -> None:
    """``bind`` on the supervisor's port (any address, loopback; with and without
    ``SO_REUSEPORT``) and ``listen``: each must be ``EPERM`` (C10x)."""
    port = int(arg)
    seen: dict[str, str] = {}
    for host in ("0.0.0.0", "127.0.0.1"):
        for reuse in (False, True):
            s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            try:
                s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
                if reuse:
                    s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEPORT, 1)
                s.bind((host, port))
                seen[f"bind {host} reuseport={reuse}"] = "bound"
            except OSError as exc:
                seen[f"bind {host} reuseport={reuse}"] = _errno_name(exc)
            finally:
                s.close()
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    try:
        s.listen(1)  # an unbound socket auto-binds an ephemeral port: still a listener
        seen["listen"] = "listening"
    except OSError as exc:
        seen["listen"] = _errno_name(exc)
    finally:
        s.close()
    _write(out, "bind.json", seen)


def child_api(out: Path, tmp: Path, arg: str) -> None:
    """The child drives the supervisor's API over loopback: it learns the ``boot_id`` from
    ``/ready``, then tries ``/session``, a wrong-sha ``/ack`` and ``/recycle``."""
    port = int(arg)

    def call(method: str, path: str, body: bytes | None = None, ctype: str = "") -> int:
        conn = http.client.HTTPConnection("127.0.0.1", port, timeout=10)
        try:
            conn.request(method, path, body=body, headers={"Content-Type": ctype} if ctype else {})
            r = conn.getresponse()
            r.read()
            return r.status
        finally:
            conn.close()

    conn = http.client.HTTPConnection("127.0.0.1", port, timeout=10)
    conn.request("GET", "/ready")
    boot = json.loads(conn.getresponse().read())["boot_id"]
    conn.close()
    session = json.dumps(
        {
            "boot_id": boot,
            "run_id": "wsr_child",
            "mode": "session",
            "spec": {"time_budget_s": 5, "grace_s": 1, "pull_timeout_s": 5},
            "files": [],
            "gateway": None,
        }
    ).encode()
    body = (
        b'--b\r\nContent-Disposition: form-data; name="session"\r\n\r\n'
        + session
        + b"\r\n--b--\r\n"
    )
    ack = json.dumps({"boot_id": boot, "run_id": "wsr_test", "result_sha256": "0" * 64}).encode()
    _write(
        out,
        "child_api.json",
        {
            "session": call("POST", "/session", body, "multipart/form-data; boundary=b"),
            "ack": call("POST", "/ack", ack, "application/json"),
            "recycle": call(
                "POST", "/recycle", json.dumps({"boot_id": boot}).encode(), "application/json"
            ),
        },
    )


def slowloris(out: Path, tmp: Path, arg: str) -> None:
    """Hold a loopback connection open with one header byte every 2 s, and stamp the time into
    ``/out/alive_ts.txt`` every 0.2 s, until the deadline kills this process (R2A-2, C10y)."""
    s = socket.create_connection(("127.0.0.1", int(arg)), timeout=5)
    s.sendall(b"GET /ready HTTP/1.1\r\nHost: x\r\n")
    stamp = out / "alive_ts.txt"
    first = time.time()
    last_byte = 0.0
    while True:
        now = time.time()
        stamp.write_text(json.dumps({"first": first, "last": now}))
        if now - last_byte > 2:
            with contextlib.suppress(OSError):
                s.sendall(b"X")
            last_byte = now
        time.sleep(0.2)


def connections(out: Path, tmp: Path, arg: str) -> None:
    """``<port>:<n>:<hold_s>``: open n loopback connections and hold them (the connection cap)."""
    port, n, hold = (float(x) for x in arg.split(":"))
    held = []
    for _ in range(int(n)):
        try:
            held.append(socket.create_connection(("127.0.0.1", int(port)), timeout=5))
        except OSError:
            break
    (out / "connections_opened.txt").write_text(str(len(held)))
    time.sleep(hold)
    for s in held:
        s.close()


WRITER = r"""
import os, sys, time
out, tmp = sys.argv[1], sys.argv[2]
while True:
    for path in (os.path.join(out, "late.txt"), os.path.join(tmp, "termination-log")):
        try:
            with open(path, "a") as fh:
                fh.write("late %f\n" % time.time())
        except OSError:
            pass
    time.sleep(0.05)
"""


def setsid_writer(out: Path, tmp: Path, arg: str) -> None:
    """A ``setsid`` descendant that keeps writing ``/out`` and the termination path; the child
    exits at once, leaving it behind (R2A-20, C10z)."""
    subprocess.Popen(
        [sys.executable, "-I", "-c", WRITER, str(out), str(tmp)],
        start_new_session=True,
        stdin=subprocess.DEVNULL,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    time.sleep(0.5)


def fork_bomb(out: Path, tmp: Path, arg: str) -> None:
    """Fork sleepers until the kernel refuses; record how many and why, and the child's limit."""
    count, why = 0, "none"
    while count < 4096:
        try:
            pid = os.fork()
        except OSError as exc:
            why = _errno_name(exc)
            break
        if pid == 0:
            time.sleep(600)
            os._exit(0)
        count += 1
    limits = [ln for ln in Path("/proc/self/limits").read_text().splitlines() if "processes" in ln]
    _write(out, "fork.json", {"forked": count, "errno": why, "limits": limits})


def fallocate(out: Path, tmp: Path, arg: str) -> None:
    """Reserve ``arg`` bytes in the work root, then wait for the disk watchdog."""
    fd = os.open("big.bin", os.O_WRONLY | os.O_CREAT, 0o600)  # cwd is the work root
    os.posix_fallocate(fd, 0, int(arg))
    os.close(fd)
    time.sleep(30)


def read_proc1(out: Path, tmp: Path, arg: str) -> None:
    seen: dict[str, str] = {}
    for name in ("environ", "mem"):
        try:
            with open(f"/proc/1/{name}", "rb") as fh:
                fh.read(16)
            seen[name] = "read"
        except OSError as exc:
            seen[name] = type(exc).__name__
    _write(out, "proc1.json", seen)


def dump_writable(out: Path, tmp: Path, arg: str) -> None:
    """FR-016.25: every directory the child can create a file in, and whether a secrets mount
    exists. Each probe file is removed at once."""
    writable: list[str] = []
    probe_name = f".wsp-probe-{os.getpid()}"
    for dirpath, dirnames, _ in os.walk("/", topdown=True, followlinks=False):
        if dirpath in NOT_WALKED:
            dirnames.clear()
            continue
        dirnames[:] = [d for d in dirnames if os.path.join(dirpath, d) not in NOT_WALKED]
        target = os.path.join(dirpath, probe_name)
        try:
            fd = os.open(target, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600)
        except OSError:
            continue
        os.close(fd)
        os.unlink(target)
        writable.append(dirpath)
    _write(
        out,
        "writable.json",
        {"writable": writable, "secrets": [p for p in SECRETS if os.path.lexists(p)]},
    )


def dump_limits(out: Path, tmp: Path, arg: str) -> None:
    (out / "limits.txt").write_text(Path("/proc/self/limits").read_text())


IMAGE_ACTIONS: dict[str, Callable[[Path, Path, str], None]] = {
    "plant": plant,
    "probe": probe,
    "kill_pid1": kill_pid1,
    "bind_port": bind_port,
    "child_api": child_api,
    "slowloris": slowloris,
    "connections": connections,
    "setsid_writer": setsid_writer,
    "fork_bomb": fork_bomb,
    "fallocate": fallocate,
    "read_proc1": read_proc1,
    "dump_writable": dump_writable,
    "dump_limits": dump_limits,
}
