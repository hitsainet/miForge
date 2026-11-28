"""The WSP worker's supervisor: PID 1 of the worker pod, one session per process (FR-016.10;
FTDD §5.4a, §8.1; FTID §3.2; D20, P49, P51, P59, P60).

It holds no credential (D20). The backend pushes a session to ``POST /session`` and pulls the
result from ``GET /result``; the supervisor never calls miForge. **The child shares its uid and
network namespace, so every route is treated as reachable by the child** (P59): what stops the
child is its seccomp filter (``child_seccomp``), this process's ignored signals and its
``try/finally os._exit``, and the once-only rules below.

The process, in order (FR-016.10's steps):

1. ``main`` is ``try: run() finally: os._exit(code)``; ``threading.excepthook`` exits 70.
2. Every catchable signal but ``SIGCHLD`` is ``SIG_IGN``; ``SIGCHLD`` keeps ``SIG_DFL`` and no
   handler, so ``Popen.wait`` reads the child's real exit code (R3A-21).
3. In prod it must be PID 1 (``supervisor_not_pid1``); ``WSP_TEST_SCRIPT`` and ``WSP_TEST_HOOKS``
   are refused in prod.
4. ``PR_SET_DUMPABLE`` 0, checked.
5. The wipe roots (R3A-1): as PID 1 the mounts ``/work``, ``/out``, ``/tmp``, ``/dev/shm``, with
   ``WSP_WORKER_ROOT`` unset; otherwise ``work/``, ``out/``, ``tmp/``, ``shm/`` under
   ``WSP_WORKER_ROOT``, refused ``worker_root_refused`` when it is unset, ``/``, ``/tmp`` or
   ``/dev/shm``, lacks the launcher's marker file (OVN-wsp-forge-16), or a root resolves outside
   it. Everything under each root is removed (never the root), and **only as PID 1** every SysV
   and POSIX IPC object of the uid (``ipc_wipe``); then it verifies, and refuses to serve if
   anything remains.
6. ``pids.max`` is read and recorded, never enforced (R3A-3).
7. A threaded server: daemon threads, ``allow_reuse_address = True``, ``allow_reuse_port =
   False``, a 10 s handler timeout, at most ``MAX_CONNECTIONS`` connections (more are closed at
   accept). The listening socket is never closed before ``os._exit``.
8. The watchdog thread owns every time bound and calls ``os._exit`` itself; ``main`` joins it.
9. The child: ``python -I`` running ``wsp_worker``, ``CHILD_ENV`` exactly, piped stdio, its own
   session. It installs its filter and writes a ready byte; only when ``/proc/<child>/status``
   shows ``NoNewPrivs: 1`` and ``Seccomp_filters`` one above this process's own is the spec and
   token written to its stdin; else ``worker_filter_missing``.
10. A disk watchdog on the work root (not a hard cap, P68).
11. Teardown: as PID 1 ``kill(-1, SIGKILL)``; otherwise the child's process group plus every
    process of its session or below it by parent link. Then the reap.
12. The safe walk of the out root into the result (``safe_out``).
13. After a valid ``/ack``, the pull timeout, a recycle, the idle recycle or the lifetime: kill and
    reap again, wipe, truncate the termination message, ``os._exit``. Never ``sys.exit``.
    ``/ready`` reports ``idle_recycle_in_s`` while idle, so the driver retires an idle boot
    through a fresh pod before the idle recycle ends the container (``wsp.dispatch``,
    2026-11-26).

``kill(-1)`` and the IPC wipe run **only when ``os.getpid() == 1``** (R3A-1): outside a container
``kill(-1)`` would kill every process of the developer's uid.
"""

from __future__ import annotations

import contextlib
import ctypes
import hashlib
import hmac
import io
import json
import os
import re
import secrets
import select
import shutil
import signal
import subprocess
import sys
import tarfile
import threading
import time
from collections.abc import Iterator
from datetime import UTC, datetime
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, NoReturn

from wsp_worker import child_seccomp, ipc_wipe, safe_out, spec

#: Module constants, not settings: the pod's exact env list stays unchanged (PF-60).
IDLE_RECYCLE_S = 3600
MAX_CONNECTIONS = 16
HANDLER_TIMEOUT_S = 10
RECEIVE_CHUNK_TIMEOUT_S = 300
HEARTBEAT_STALE_S = 5
READY_BYTE_TIMEOUT_S = 2.0
WATCHDOG_TICK_S = 0.1
DISK_POLL_S = 1.0
MAX_SESSION_BYTES = 64 * 1024**3
MAX_SMALL_BODY = 65536
DEFAULT_PORT = 9000

PID1_ROOTS = {"work": "/work", "out": "/out", "tmp": "/tmp", "shm": "/dev/shm"}
#: The file ``LocalProcessLauncher`` puts in a fresh ``WSP_WORKER_ROOT`` (OVN-wsp-forge-16).
ROOT_MARKER = ".wsp-worker-root"
REFUSED_ROOTS = ("/", "/tmp", "/dev/shm")
#: The pod's ``terminationMessagePath``. The kubelet BIND-MOUNTS a file of its own there, inside
#: the ``/tmp`` emptyDir, so ``unlink`` fails ``EBUSY``: the wipe truncates it instead and accepts
#: it once empty. Found by the image job's pod-like mount (FTASKS 4.4): before, the start-up wipe
#: refused ``worker_wipe_failed`` on it and the pod could never serve.
TERMINATION_NAME = "termination-log"
PID1_TERMINATION_LOG = f"/tmp/{TERMINATION_NAME}"
#: The directory holding ``wsp_worker``: the child runs exactly this package (OVN-wsp-forge-14).
PACKAGE_ROOT = Path(__file__).resolve().parent.parent
_CHILD_BOOT = (
    "import sys;sys.path.insert(0,{root!r});import runpy;"
    "runpy.run_module('wsp_worker',run_name='__main__',alter_sys=True)"
)
PR_GET_DUMPABLE, PR_SET_DUMPABLE, PR_SET_CHILD_SUBREAPER = 3, 4, 36


class StartRefused(Exception):  # noqa: N818 - a refusal, named as the code reads
    def __init__(self, code: str, message: str) -> None:
        super().__init__(message)
        self.code = code


def _refuse_start(code: str, message: str) -> NoReturn:
    os.write(2, (json.dumps({"refused": code, "message": message}) + "\n").encode())
    os._exit(spec.EXIT_REFUSED)


# ----------------------------------------------------------------------------- start-up


def ignore_signals() -> None:
    for sig in spec.supervisor_ignored():
        try:
            signal.signal(sig, signal.SIG_IGN)
        except (OSError, ValueError):
            continue


def set_undumpable() -> None:
    libc = ctypes.CDLL(None, use_errno=True)
    libc.prctl(PR_SET_DUMPABLE, 0, 0, 0, 0)
    if libc.prctl(PR_GET_DUMPABLE, 0, 0, 0, 0) != 0:
        raise StartRefused("worker_dumpable", "PR_SET_DUMPABLE 0 did not hold")


def set_subreaper() -> None:
    """Outside PID 1: orphaned descendants re-parent here, so teardown can find and reap them."""
    libc = ctypes.CDLL(None, use_errno=True)
    if libc.prctl(PR_SET_CHILD_SUBREAPER, 1, 0, 0, 0) != 0:
        raise StartRefused("worker_subreaper", "PR_SET_CHILD_SUBREAPER failed")


def resolve_roots(pid1: bool, worker_root: str | None) -> dict[str, Path]:
    """The four wipe roots, or :class:`StartRefused` ``worker_root_refused`` (R3A-1)."""
    if pid1:
        if worker_root:
            raise StartRefused("worker_root_refused", "WSP_WORKER_ROOT must be unset as PID 1")
        return {k: Path(v) for k, v in PID1_ROOTS.items()}
    if not worker_root or not os.path.isabs(worker_root):
        raise StartRefused("worker_root_refused", "WSP_WORKER_ROOT must be an absolute path")
    real = os.path.realpath(worker_root)
    if os.path.normpath(worker_root) in REFUSED_ROOTS or real in REFUSED_ROOTS:
        raise StartRefused("worker_root_refused", f"WSP_WORKER_ROOT may not be {real}")
    if not os.path.isdir(real):
        raise StartRefused("worker_root_refused", "WSP_WORKER_ROOT is not a directory")
    _require_marker(Path(real))
    roots: dict[str, Path] = {}
    for name in PID1_ROOTS:
        path = Path(real) / name
        if not os.path.lexists(path):
            path.mkdir(mode=0o700)
        if os.path.islink(path) or os.path.dirname(os.path.realpath(path)) != real:
            raise StartRefused("worker_root_refused", f"{name}/ resolves outside WSP_WORKER_ROOT")
        roots[name] = path
    return roots


def _require_marker(root: Path) -> None:
    marker = root / ROOT_MARKER
    try:
        st = os.stat(marker, follow_symlinks=False)
    except OSError as exc:
        raise StartRefused(
            "worker_root_refused", f"WSP_WORKER_ROOT has no {ROOT_MARKER} (not a launcher root)"
        ) from exc
    if not (st.st_mode & 0o170000 == 0o100000 and st.st_uid == os.getuid()):
        raise StartRefused("worker_root_refused", f"{ROOT_MARKER} is not this uid's regular file")


def wipe_roots(roots: dict[str, Path], pid1: bool) -> list[str]:
    """Remove everything under each root (never the root); return what is left."""
    for name, root in roots.items():
        if pid1:
            if str(root) != PID1_ROOTS[name]:
                raise StartRefused("worker_root_refused", f"{root} is not a PID-1 wipe root")
        else:
            # A second, independent guard: never empty a directory that is not a launcher root's.
            _require_marker(root.parent)
            if root.parent.resolve() in {Path(p) for p in REFUSED_ROOTS}:
                raise StartRefused("worker_root_refused", f"{root} is not under a launcher root")
        for entry in os.scandir(root):
            try:
                if entry.is_dir(follow_symlinks=False):
                    shutil.rmtree(entry.path, onexc=_force)
                else:
                    os.unlink(entry.path)
            except OSError:
                if name == "tmp" and entry.name == TERMINATION_NAME:
                    truncate(Path(entry.path))  # the kubelet's bind mount: EBUSY, emptied instead
                continue  # verified below
    return [
        f"{root}/{n}"
        for name, root in roots.items()
        for n in os.listdir(root)
        if not (name == "tmp" and n == TERMINATION_NAME and _empty_file(root / n))
    ]


def _empty_file(path: Path) -> bool:
    """A regular file of no bytes: what an emptied termination message mount looks like."""
    try:
        st = os.lstat(path)
    except OSError:
        return False
    return st.st_mode & 0o170000 == 0o100000 and st.st_size == 0


def _force(func: Any, path: str, exc: BaseException) -> None:
    """``rmtree``'s error hook: a child may leave a directory unwritable; reopen it and retry."""
    try:
        os.chmod(os.path.dirname(path), 0o700)
        if os.path.isdir(path) and not os.path.islink(path):
            os.chmod(path, 0o700)
        func(path)
    except OSError:
        pass


def read_pids_max() -> str | None:
    """This process's cgroup ``pids.max`` (v2 path from ``/proc/self/cgroup`` first)."""
    candidates = []
    try:
        for line in Path("/proc/self/cgroup").read_text().splitlines():
            if line.startswith("0::"):
                candidates.append(f"/sys/fs/cgroup{line[3:].rstrip('/')}/pids.max")
    except OSError:
        pass
    for path in (*candidates, "/sys/fs/cgroup/pids.max", "/sys/fs/cgroup/pids/pids.max"):
        try:
            return Path(path).read_text().strip()
        except OSError:
            continue
    return None


def _test_hooks(environment: str) -> dict[str, float]:
    raw = os.environ.get("WSP_TEST_HOOKS")
    if not raw:
        return {}
    if environment == "prod":
        raise StartRefused("worker_test_hooks_refused", "WSP_TEST_HOOKS is refused in prod")
    hooks = json.loads(raw)
    allowed = {"watchdog_raise_after_s", "heartbeat_stall_after_s", "idle_recycle_s"}
    if not isinstance(hooks, dict) or not set(hooks) <= allowed:
        raise StartRefused("worker_test_hooks_refused", f"WSP_TEST_HOOKS keys are {allowed}")
    return {k: float(v) for k, v in hooks.items()}


# ----------------------------------------------------------------------------- multipart


class MultipartError(ValueError):
    pass


_DISPOSITION = re.compile(r'(\w+)="([^"]*)"')


class Multipart:
    """A streaming ``multipart/form-data`` reader bounded by the request's length."""

    def __init__(
        self, rfile: io.BufferedIOBase, boundary: bytes, length: int, on_chunk: Any
    ) -> None:
        self.rfile, self.left, self.on_chunk = rfile, length, on_chunk
        self.delim = b"\r\n--" + boundary
        self.buf = b"\r\n"

    def _fill(self) -> None:
        if self.left <= 0:
            raise MultipartError("the body ended inside the multipart stream")
        read1 = getattr(self.rfile, "read1", self.rfile.read)
        chunk = read1(min(65536, self.left))
        if not chunk:
            raise MultipartError("the connection closed inside the multipart stream")
        self.left -= len(chunk)
        self.buf += chunk
        self.on_chunk()

    def next_part(self) -> dict[str, str] | None:
        """The next part's headers, or ``None`` at the closing boundary."""
        while (i := self.buf.find(self.delim)) < 0:
            if len(self.buf) > 70000:
                raise MultipartError("no boundary where one was due")
            self._fill()
        if i != 0 and self.buf[:i].strip():
            raise MultipartError("unread bytes before a boundary")
        self.buf = self.buf[i + len(self.delim) :]
        while len(self.buf) < 2:
            self._fill()
        if self.buf[:2] == b"--":
            return None
        while (j := self.buf.find(b"\r\n\r\n")) < 0:
            if len(self.buf) > 16384:
                raise MultipartError("part headers are too long")
            self._fill()
        lines = self.buf[2:j].decode("latin-1").split("\r\n")
        self.buf = self.buf[j + 4 :]
        headers: dict[str, str] = {}
        for line in lines:
            key, _, value = line.partition(":")
            headers[key.strip().lower()] = value.strip()
        return headers

    def data(self) -> Iterator[bytes]:
        """The current part's bytes, up to (not including) the next boundary."""
        keep = len(self.delim) - 1
        while True:
            i = self.buf.find(self.delim)
            if i >= 0:
                if i:
                    yield self.buf[:i]
                self.buf = self.buf[i:]
                return
            if len(self.buf) > keep:
                yield self.buf[:-keep]
                self.buf = self.buf[-keep:]
            self._fill()


def disposition(headers: dict[str, str]) -> dict[str, str]:
    return dict(_DISPOSITION.findall(headers.get("content-disposition", "")))


# ----------------------------------------------------------------------------- the supervisor


class Supervisor:
    def __init__(
        self,
        *,
        environment: str,
        pid1: bool,
        roots: dict[str, Path],
        test_script: str | None,
        hooks: dict[str, float],
        pids_max: str | None,
    ) -> None:
        self.environment, self.pid1, self.roots = environment, pid1, roots
        self.test_script, self.hooks, self.pids_max = test_script, hooks, pids_max
        self.lock = threading.Lock()
        self.boot_id = secrets.token_hex(16)
        self.state = "idle"
        self.session_accepted = False
        self.push: spec.Push | None = None
        self.started_at: str | None = None
        self.idle_since = time.monotonic()
        self.accepted_mono: float | None = None
        self.running_mono: float | None = None
        self.finished_mono: float | None = None
        self.last_chunk_mono: float | None = None
        self.deadline_hit = False
        self.refusal: str | None = None
        self.child: subprocess.Popen[bytes] | None = None
        self.exit_code: int | None = None
        self.result: bytes | None = None
        self.result_sha256: str | None = None
        self.result_served = False
        self.exit_reason: str | None = None
        self.heartbeat = time.monotonic()
        self.log_bytes: dict[str, int] = {}
        self.log_truncated: dict[str, bool] = {}
        self._drains: list[threading.Thread] = []
        self._last_disk_poll = 0.0

    # --- paths -----------------------------------------------------------------------
    @property
    def termination_log(self) -> Path:
        return Path(PID1_TERMINATION_LOG) if self.pid1 else self.roots["tmp"] / TERMINATION_NAME

    def child_argv(self, child_mode: str) -> list[str]:
        if self.test_script:
            return [sys.executable, "-I", self.test_script, child_mode, str(PACKAGE_ROOT)]
        boot = _CHILD_BOOT.format(root=str(PACKAGE_ROOT))
        return [sys.executable, "-I", "-c", boot, child_mode]

    # --- the watchdog ----------------------------------------------------------------
    def watchdog(self) -> None:
        started = time.monotonic()
        while True:
            now = time.monotonic()
            stall = self.hooks.get("heartbeat_stall_after_s")
            if stall is None or now - started < stall:
                self.heartbeat = now
            boom = self.hooks.get("watchdog_raise_after_s")
            if boom is not None and now - started >= boom:
                raise RuntimeError("test hook: the watchdog raises")
            reason = self._due(now)
            if reason is not None:
                self.exit_now(reason)
            time.sleep(WATCHDOG_TICK_S)

    def _due(self, now: float) -> str | None:
        kill = False
        with self.lock:
            if self.exit_reason is not None:
                return self.exit_reason
            push = self.push
            idle_limit = self.hooks.get("idle_recycle_s", IDLE_RECYCLE_S)
            if self.state == "idle" and now - self.idle_since > idle_limit:
                return "idle_recycle"
            receiving = self.state == "receiving" and self.last_chunk_mono is not None
            if receiving and now - (self.last_chunk_mono or now) > RECEIVE_CHUNK_TIMEOUT_S:
                return "receive_timeout"
            # The lifetime runs from the session's start (fix 2026-11-21), the clock the time
            # budget and the backend's deadline use; from acceptance only while receiving. From
            # acceptance always, a push longer than grace + pull timeout (a 34.9 GB bundle took
            # 301 s) ended the worker's lifetime before the backend could pull the result.
            start = self.running_mono if self.running_mono is not None else self.accepted_mono
            if push is not None and start is not None and now - start > push.lifetime_s:
                return "lifetime"
            finished = self.finished_mono if self.state == "finished" else None
            pull_s = None if push is None else float(push.spec["pull_timeout_s"])
            if finished is not None and pull_s is not None and now - finished > pull_s:
                return "pull_timeout"
            if self.state == "running" and push is not None and self.running_mono is not None:
                if not self.deadline_hit and now - self.running_mono > push.time_budget_s:
                    self.deadline_hit = True
                    kill = True
                if now - self._last_disk_poll >= DISK_POLL_S and self.refusal is None:
                    self._last_disk_poll = now
                    if _usage(self.roots["work"]) > push.work_max_bytes:
                        self.refusal = "worker_disk_limit"
                        kill = True
        if kill:
            self.kill_session()
        return None

    def idle_recycle_in(self, now: float) -> float | None:
        """Seconds until the watchdog's idle recycle, for ``/ready``: ``_due``'s own test (the
        state is ``idle``, which it leaves for good when a session is accepted), so ``None``
        whenever the recycle no longer counts. The driver retires an idle boot through its pod
        before this reaches 0 and never claims one about to end (2026-11-26, the residue of the
        019 P4 restart debt). The caller holds the lock."""
        if self.state != "idle":
            return None
        limit = self.hooks.get("idle_recycle_s", IDLE_RECYCLE_S)
        return max(0.0, float(limit) - (now - self.idle_since))

    def time_left(self, now: float) -> float:
        """The session's time budget left at ``now``: the watchdog kills the child at 0
        (``_due``), so the child line carries it and the engine ends before it (fix 2026-11-21)."""
        push = self.push
        if push is None:
            return 0.0
        running = self.running_mono if self.running_mono is not None else now
        return max(0.0, push.time_budget_s - (now - running))

    def exit_now(self, reason: str) -> NoReturn:
        """Kill and reap, wipe, truncate the termination message, ``os._exit``."""
        with self.lock:
            self.state = "exiting"
            self.exit_reason = reason
        try:
            self.kill_session()
            self._reap(block_s=2.0)
            wipe_roots(self.roots, self.pid1)
            if os.getpid() == 1:
                ipc_wipe.wipe_all()
        except Exception:
            pass
        finally:
            truncate(self.termination_log)
            os._exit(spec.EXIT_OK)

    # --- the child -------------------------------------------------------------------
    def kill_session(self) -> None:
        """As PID 1, every other process in the namespace; otherwise the child's tree (R3A-1)."""
        if os.getpid() == 1:
            with contextlib.suppress(ProcessLookupError):
                os.kill(-1, signal.SIGKILL)
            return
        child = self.child
        if child is None:
            return
        # This process is a child subreaper outside PID 1, so a setsid descendant whose parent
        # died is re-parented here, not to the host's init: everything below this process is
        # the session's (the supervisor runs nothing else).
        for pid in descendants(child.pid) | descendants(os.getpid(), by_session=False):
            with contextlib.suppress(OSError):
                os.kill(pid, signal.SIGKILL)
        with contextlib.suppress(OSError):
            os.killpg(child.pid, signal.SIGKILL)

    def _reap(self, block_s: float) -> None:
        end = time.monotonic() + block_s
        while True:
            try:
                pid, _ = os.waitpid(-1, os.WNOHANG)
            except ChildProcessError:
                return
            if pid == 0:
                if time.monotonic() > end:
                    return
                time.sleep(0.02)

    def run_session(self) -> None:
        push = self.push
        assert push is not None
        logs = self.roots["out"] / "logs"
        logs.mkdir(mode=0o700)
        sinks = {
            name: os.open(
                logs / f"{name}.txt",
                os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW | os.O_CLOEXEC,
                0o600,
            )
            for name in ("stdout", "stderr")
        }
        child = subprocess.Popen(
            self.child_argv(push.child_mode),
            env=spec.child_env(push.child_mode),
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            start_new_session=True,
            cwd=str(self.roots["work"]),
            close_fds=True,
        )
        self.child = child
        try:
            if self._handshake(child, push):
                for name, pipe in (("stdout", child.stdout), ("stderr", child.stderr)):
                    t = threading.Thread(
                        target=self._drain,
                        args=(name, pipe, sinks[name], push.log_max_bytes),
                        daemon=True,
                    )
                    t.start()
                    self._drains.append(t)
            else:
                self.kill_session()
            child.wait()
        finally:
            self.exit_code = child.returncode
            self.kill_session()
            self._reap(block_s=1.0)
            for t in self._drains:
                t.join(timeout=2.0)
            for fd in sinks.values():
                os.close(fd)
            self._build_result(push)
            with self.lock:
                if self.state == "running":
                    self.state = "finished"
                    self.finished_mono = time.monotonic()

    def _handshake(self, child: subprocess.Popen[bytes], push: spec.Push) -> bool:
        """The ready byte, then the filter check, then the one stdin line (R3A-5, R4A-6)."""
        assert child.stdout is not None and child.stdin is not None
        fd = child.stdout.fileno()
        ready, _, _ = select.select([fd], [], [], READY_BYTE_TIMEOUT_S)
        if not ready or os.read(fd, 1) != b"R":
            self.refusal = "worker_filter_missing"
            return False
        try:
            theirs = child_seccomp.status_fields(child.pid)
            ours = child_seccomp.status_fields("self")
        except OSError:
            self.refusal = "worker_filter_missing"
            return False
        if (
            theirs.get("NoNewPrivs") != "1"
            or "Seccomp_filters" not in theirs
            or "Seccomp_filters" not in ours
            or int(theirs["Seccomp_filters"]) != int(ours["Seccomp_filters"]) + 1
        ):
            self.refusal = "worker_filter_missing"
            return False
        paths = {k: str(self.roots[k]) for k in ("work", "out", "tmp")}
        try:
            child.stdin.write(spec.child_line(push, paths, self.time_left(time.monotonic())))
            child.stdin.close()
        except OSError:
            return True  # the child died first; its exit code says so
        return True

    def _drain(self, name: str, pipe: Any, sink: int, cap: int) -> None:
        written = 0
        truncated = False
        while True:
            chunk = pipe.read1(65536) if hasattr(pipe, "read1") else pipe.read(65536)
            if not chunk:
                break
            room = cap - written
            if room > 0:
                part = chunk[:room]
                os.write(sink, part)
                written += len(part)
            if len(chunk) > max(room, 0):
                truncated = True
        self.log_bytes[name] = written
        self.log_truncated[name] = truncated

    def _build_result(self, push: spec.Push) -> None:
        walk = safe_out.walk(str(self.roots["out"]))
        record = {
            "boot_id": self.boot_id,
            "run_id": push.run_id,
            "exit_code": self.exit_code,
            "deadline_hit": self.deadline_hit,
            "refusal": self.refusal,
            "unsafe_reason": "worker_result_unsafe" if walk.unsafe else None,
            "unsafe": walk.unsafe,
            "log_bytes": self.log_bytes,
            "log_truncated": self.log_truncated,
            "pids_max": self.pids_max,
        }
        members = dict(walk.files)
        members["supervisor.json"] = (json.dumps(record, sort_keys=True) + "\n").encode()
        buffer = io.BytesIO()
        with tarfile.open(fileobj=buffer, mode="w", format=tarfile.USTAR_FORMAT) as tar:
            for name in sorted(members):
                info = tarfile.TarInfo(name)
                info.size, info.mode, info.mtime = len(members[name]), 0o644, 0
                tar.addfile(info, io.BytesIO(members[name]))
        self.result = buffer.getvalue()
        self.result_sha256 = hashlib.sha256(self.result).hexdigest()

    # --- the API ---------------------------------------------------------------------
    def status(self) -> dict[str, Any]:
        with self.lock:
            elapsed = None if self.accepted_mono is None else time.monotonic() - self.accepted_mono
            return {
                "boot_id": self.boot_id,
                "run_id": None if self.push is None else self.push.run_id,
                "state": self.state,
                "started_at": self.started_at,
                "elapsed_s": elapsed,
                "deadline_hit": self.deadline_hit,
                "pids_max": self.pids_max,
            }


def _usage(root: Path) -> int:
    total = 0
    for dirpath, dirnames, filenames in os.walk(root, followlinks=False):
        for name in dirnames + filenames:
            try:
                total += os.lstat(os.path.join(dirpath, name)).st_blocks * 512
            except OSError:
                continue
    return total


def work_disk(root: Path) -> dict[str, int | None]:
    """The free and total bytes of the filesystem holding the work root, for ``/ready``
    (2026-11-26, the 019 P4 disk gap). In the pod ``/work`` is an emptyDir, so this is the node's
    filesystem that the kubelet keeps emptyDirs on: what a push into ``/work`` can still use, the
    figure FTASKS 21.1 read by hand with ``df``. ``statvfs`` reads no file and walks nothing; the
    emptyDir's ``sizeLimit`` is not part of it (the backend knows it as ``WSP_WORKER_WORK_BYTES``).
    Both ``None`` when the root cannot be read."""
    try:
        st = os.statvfs(root)
    except OSError:
        return {"work_free_bytes": None, "work_total_bytes": None}
    return {
        "work_free_bytes": int(st.f_bavail * st.f_frsize),
        "work_total_bytes": int(st.f_blocks * st.f_frsize),
    }


def descendants(root_pid: int, *, by_session: bool = True) -> set[int]:
    """Every process below ``root_pid`` by parent link and, with ``by_session``, in its session or
    process group; never this process or PID 1."""
    me = os.getpid()
    parent: dict[int, int] = {}
    found: set[int] = set()
    for entry in os.listdir("/proc"):
        if not entry.isdigit():
            continue
        try:
            raw = Path(f"/proc/{entry}/stat").read_text()
        except OSError:
            continue
        fields = raw[raw.rfind(")") + 2 :].split()
        pid, ppid, pgrp, sid = int(entry), int(fields[1]), int(fields[2]), int(fields[3])
        parent[pid] = ppid
        if by_session and (sid == root_pid or pgrp == root_pid):
            found.add(pid)
    found.add(root_pid)
    grew = True
    while grew:
        grew = False
        for pid, ppid in parent.items():
            if ppid in found and pid not in found:
                found.add(pid)
                grew = True
    return {p for p in found if p > 1 and p != me}


def truncate(path: Path) -> None:
    try:
        fd = os.open(path, os.O_WRONLY | os.O_TRUNC | os.O_NOFOLLOW | os.O_NONBLOCK | os.O_CLOEXEC)
        os.close(fd)
    except FileNotFoundError:
        pass
    except OSError:
        with contextlib.suppress(OSError):
            os.unlink(path)


# ----------------------------------------------------------------------------- the server


class Server(ThreadingHTTPServer):
    daemon_threads = True
    allow_reuse_address = True
    allow_reuse_port = False

    def __init__(self, address: tuple[str, int], sup: Supervisor) -> None:
        self.sup = sup
        self.slots = threading.BoundedSemaphore(MAX_CONNECTIONS)
        super().__init__(address, Handler)

    def process_request(self, request: Any, client_address: Any) -> None:
        if not self.slots.acquire(blocking=False):
            self.shutdown_request(request)
            return
        try:
            super().process_request(request, client_address)
        except BaseException:
            self.slots.release()
            raise

    def process_request_thread(self, request: Any, client_address: Any) -> None:
        try:
            super().process_request_thread(request, client_address)
        finally:
            self.slots.release()


class Handler(BaseHTTPRequestHandler):
    timeout = HANDLER_TIMEOUT_S
    server_version = "wsp-worker"
    sys_version = ""
    server: Server

    def log_message(self, format: str, *args: Any) -> None:
        return

    def _send(
        self, code: int, body: dict[str, Any] | bytes, headers: dict[str, str] | None = None
    ) -> None:
        data = body if isinstance(body, bytes) else json.dumps(body).encode()
        self.send_response(code)
        self.send_header(
            "Content-Type", "application/json" if isinstance(body, dict) else "application/x-tar"
        )
        self.send_header("Content-Length", str(len(data)))
        for key, value in (headers or {}).items():
            self.send_header(key, value)
        self.end_headers()
        self.wfile.write(data)
        self.wfile.flush()

    def _refuse(self, status: int, code: str, message: str) -> None:
        self._send(status, {"refused": code, "message": message})

    def _small_json(self) -> dict[str, Any] | None:
        try:
            length = int(self.headers.get("Content-Length", "-1"))
        except ValueError:
            length = -1
        if not 0 <= length <= MAX_SMALL_BODY:
            return None
        try:
            doc = json.loads(self.rfile.read(length))
        except (ValueError, OSError):
            return None
        return doc if isinstance(doc, dict) else None

    def do_GET(self) -> None:
        sup = self.server.sup
        if self.path == "/ready":
            if time.monotonic() - sup.heartbeat > HEARTBEAT_STALE_S:
                self._refuse(503, "worker_watchdog_stale", "the watchdog's heartbeat is stale")
                return
            with sup.lock:
                self._send(
                    200,
                    {
                        "boot_id": sup.boot_id,
                        "state": sup.state,
                        "idle_recycle_in_s": sup.idle_recycle_in(time.monotonic()),
                        **work_disk(sup.roots["work"]),
                    },
                )
        elif self.path == "/status":
            self._send(200, sup.status())
        elif self.path == "/result":
            with sup.lock:
                if sup.state != "finished" or sup.result is None:
                    self._refuse(409, "worker_not_finished", "the session has not finished")
                    return
                sup.result_served = True
                result, digest = sup.result, sup.result_sha256 or ""
            self._send(200, result, {"X-Content-SHA256": digest})
        else:
            self._refuse(404, "worker_route_unknown", "no such route")

    def do_POST(self) -> None:
        if self.path == "/session":
            self._session()
        elif self.path == "/ack":
            self._ack()
        elif self.path == "/recycle":
            self._recycle()
        else:
            self._refuse(404, "worker_route_unknown", "no such route")

    def _ack(self) -> None:
        sup = self.server.sup
        doc = self._small_json() or {}
        with sup.lock:
            ok = (
                sup.state == "finished"
                and sup.result_served
                and sup.push is not None
                and doc.get("boot_id") == sup.boot_id
                and doc.get("run_id") == sup.push.run_id
                and isinstance(doc.get("result_sha256"), str)
                and hmac.compare_digest(str(doc["result_sha256"]), sup.result_sha256 or "")
            )
        if not ok:
            self._refuse(409, "worker_ack_refused", "an ack needs the served result's sha256")
            return
        self._send(202, {"boot_id": sup.boot_id, "state": "exiting"})
        with sup.lock:
            sup.exit_reason = "ack"

    def _recycle(self) -> None:
        sup = self.server.sup
        doc = self._small_json() or {}
        with sup.lock:
            ok = (
                sup.state == "idle"
                and not sup.session_accepted
                and doc.get("boot_id") == sup.boot_id
            )
            if ok:
                sup.state = "exiting"
        if not ok:
            self._refuse(409, "worker_not_idle", "a recycle is accepted only while idle")
            return
        self._send(202, {"boot_id": sup.boot_id, "state": "exiting"})
        with sup.lock:
            sup.exit_reason = "recycle"

    def _session(self) -> None:
        sup = self.server.sup
        with sup.lock:
            busy = sup.session_accepted or sup.state != "idle"
        if busy:
            self._refuse(409, "worker_not_idle", "this supervisor has served its one session")
            return
        match = re.match(
            r'multipart/form-data;\s*boundary="?([^";]{1,70})"?',
            self.headers.get("Content-Type", ""),
        )
        try:
            length = int(self.headers.get("Content-Length", "-1"))
        except ValueError:
            length = -1
        if match is None or not 0 < length <= MAX_SESSION_BYTES:
            self._refuse(
                400, "worker_spec_invalid", "a session is multipart/form-data with a length"
            )
            return
        reader = Multipart(self.rfile, match.group(1).encode("latin-1"), length, self._chunk)
        try:
            first = reader.next_part()
            if first is None or disposition(first).get("name") != "session":
                raise MultipartError("the first part must be the session")
            raw = b"".join(_bounded(reader.data(), spec.MAX_PUSH_JSON_BYTES))
            push = spec.parse_push(raw)
        except (MultipartError, spec.SpecError, OSError) as exc:
            self._refuse(400, getattr(exc, "code", "worker_spec_invalid"), str(exc))
            return
        with sup.lock:
            if sup.session_accepted or sup.state != "idle" or push.boot_id != sup.boot_id:
                refused = True
            else:
                refused = False
                sup.session_accepted = True
                sup.state = "receiving"
                sup.push = push
                sup.accepted_mono = sup.last_chunk_mono = time.monotonic()
                sup.started_at = datetime.now(UTC).isoformat()
        if refused:
            self._refuse(409, "worker_not_idle", "not idle, or not this boot's session")
            return
        self.connection.settimeout(RECEIVE_CHUNK_TIMEOUT_S)
        try:
            mismatch = self._receive_files(reader, push)
        except (MultipartError, OSError) as exc:
            with sup.lock:
                sup.exit_reason = "receive_failed"
            self._refuse(400, "worker_spec_invalid", f"the session body failed: {exc}")
            return
        if mismatch is not None:
            self._refuse(422, "worker_file_mismatch", mismatch)
            with sup.lock:
                sup.exit_reason = "file_mismatch"
            return
        with sup.lock:
            sup.state = "running"
            sup.running_mono = time.monotonic()
        threading.Thread(target=sup.run_session, name="wsp-session", daemon=True).start()
        self._send(202, {"boot_id": sup.boot_id, "run_id": push.run_id, "state": "running"})

    def _chunk(self) -> None:
        self.server.sup.last_chunk_mono = time.monotonic()

    def _receive_files(self, reader: Multipart, push: spec.Push) -> str | None:
        work = self.server.sup.roots["work"]
        for expected in push.files:
            headers = reader.next_part()
            if headers is None or disposition(headers).get("filename") != expected.name:
                return f"the parts do not follow files[]: expected {expected.name}"
            digest = hashlib.sha256()
            size = 0
            fd = os.open(
                work / expected.name,
                os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW | os.O_CLOEXEC,
                0o600,
            )
            try:
                for chunk in reader.data():
                    size += len(chunk)
                    if size > expected.bytes:
                        return f"{expected.name} is longer than its {expected.bytes} bytes"
                    digest.update(chunk)
                    os.write(fd, chunk)
            finally:
                os.close(fd)
            if size != expected.bytes or digest.hexdigest() != expected.sha256:
                return f"{expected.name} does not match its sha256 and size"
        if reader.next_part() is not None:
            return "the body carries a part that files[] does not name"
        return None


def _bounded(chunks: Iterator[bytes], cap: int) -> Iterator[bytes]:
    total = 0
    for chunk in chunks:
        total += len(chunk)
        if total > cap:
            raise MultipartError("the session part is larger than its bound")
        yield chunk


# ----------------------------------------------------------------------------- main


def run() -> int:
    ignore_signals()
    threading.excepthook = lambda args: os._exit(spec.EXIT_WATCHDOG)
    environment = os.environ.get("WSP_ENVIRONMENT", "prod")
    pid1 = os.getpid() == 1
    try:
        if environment == "prod" and not pid1:
            raise StartRefused("supervisor_not_pid1", "in prod the supervisor must be PID 1")
        test_script = os.environ.get("WSP_TEST_SCRIPT") or None
        if test_script and environment == "prod":
            raise StartRefused("worker_test_script_refused", "WSP_TEST_SCRIPT is refused in prod")
        hooks = _test_hooks(environment)
        set_undumpable()
        if not pid1:
            set_subreaper()
        roots = resolve_roots(pid1, os.environ.get("WSP_WORKER_ROOT"))
        left = wipe_roots(roots, pid1)
        if pid1:
            left += ipc_wipe.wipe_all()
        if left:
            raise StartRefused("worker_wipe_failed", f"the wipe left {left[:5]}")
        port = int(os.environ.get("WSP_WORKER_PORT", str(DEFAULT_PORT)))
    except (StartRefused, ipc_wipe.IpcWipeRefusedError, ValueError) as exc:
        _refuse_start(getattr(exc, "code", "worker_start_refused"), str(exc))
    sup = Supervisor(
        environment=environment,
        pid1=pid1,
        roots=roots,
        test_script=test_script,
        hooks=hooks,
        pids_max=read_pids_max(),
    )
    server = Server(("0.0.0.0" if pid1 else "127.0.0.1", port), sup)
    threading.Thread(target=server.serve_forever, name="wsp-server", daemon=True).start()
    watchdog = threading.Thread(target=sup.watchdog, name="wsp-watchdog", daemon=True)
    watchdog.start()
    os.write(
        1, (json.dumps({"event": "listening", "port": server.server_address[1]}) + "\n").encode()
    )
    watchdog.join()
    return spec.EXIT_WATCHDOG


def main() -> NoReturn:
    code = spec.EXIT_WATCHDOG
    try:
        code = run()
    finally:
        os._exit(code)
