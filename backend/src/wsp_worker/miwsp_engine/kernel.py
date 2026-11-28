"""The kernel grandchild, from the session's side (feature 018; FR-018.14; FTDD §6.5; FTID §3.4).

- **Spawn:** a FRESH interpreter (``subprocess.Popen`` of ``sys.executable -I -c BOOT``), never a
  fork of the session after it read the token; environment :data:`KERNEL_ENV` (a literal, pinned
  by a test like ``CHILD_ENV``); its own pipes (never the child's stdout or stderr, which the
  supervisor drains); its own session, so a timeout kills its whole process group; ``close_fds``.
  Not ``multiprocessing``: its forkserver needs ``bind``/``listen``, which the child's filter
  denies.
- **The filter check:** the ready byte ``K`` within :data:`READY_TIMEOUT_S`, then
  ``/proc/<pid>/status``: ``NoNewPrivs: 1`` and ``Seccomp_filters`` one above this process's own,
  as the supervisor checks the child (``supervisor.py:540-552``). Anything else refuses
  ``worker_kernel_filter_missing`` before a byte of the session is sent.
- **Tensors** travel as ``.npy`` files in the session's scratch directory, written to a temporary
  name and renamed, so a map the kernel still holds keeps its old inode.
- **Everything the kernel answers is untrusted:** a bounded line, a JSON object, and deltas whose
  names, indices and values are checked here before the store's limits see them.
- **Timeout:** :data:`EXEC_TIMEOUT_S` per ``exec`` (``weightspace``'s sandbox alarm). On a timeout,
  or if the kernel dies, its process group is killed and a new kernel starts from the store's
  accepted values; the editor is told its variables are gone.
"""

from __future__ import annotations

import contextlib
import json
import os
import selectors
import signal
import subprocess
import sys
import time
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

import wsp_worker
from wsp_worker import child_seccomp

#: The kernel's whole environment (no token, nothing inherited). Pinned by a test.
KERNEL_ENV: dict[str, str] = {
    "PATH": "/usr/local/bin:/usr/bin:/bin",
    "HOME": "/tmp",
    "PYTHONNOUSERSITE": "1",
    "OMP_NUM_THREADS": "1",
    "OPENBLAS_NUM_THREADS": "1",
    "MKL_NUM_THREADS": "1",
}
PACKAGE_ROOT = str(Path(wsp_worker.__file__).resolve().parents[1])
BOOT = (
    "import sys, runpy; sys.path.insert(0, {root!r}); "
    "runpy.run_module('wsp_worker.miwsp_engine.kernel_main', run_name='__main__')"
)
READY_TIMEOUT_S = 10.0
EXEC_TIMEOUT_S = 300.0
CONTROL_TIMEOUT_S = 60.0
MAX_ANSWER_BYTES = 64 * 1024 * 1024


class KernelError(Exception):
    def __init__(self, code: str, message: str) -> None:
        super().__init__(message)
        self.code = code


@dataclass
class ExecResult:
    output: str
    deltas: dict[str, tuple[np.ndarray, np.ndarray]] = field(default_factory=dict)
    restarted: bool = False


class Kernel:
    """One kernel for the whole session (F18-D10), restarted after a timeout."""

    def __init__(
        self,
        directory: Path,
        values: Callable[[], Mapping[str, np.ndarray]],
        *,
        edit: bool,
        timeout_s: float = EXEC_TIMEOUT_S,
        python: str | None = None,
        rebind: bool = False,
    ) -> None:
        self.dir = directory
        self._values = values
        self.names = list(values())
        self.edit = edit
        self.timeout_s = timeout_s
        self.python = python or sys.executable
        self.notes: list[str] = []
        #: 2026-11-27: the kernel binds R, W, notes and np back after every exec
        #: (``kernel_main.RESERVED``); off for a replay of a run recorded before (its header).
        self.rebind = rebind
        self.restarts = 0
        self.proc: subprocess.Popen[bytes] | None = None
        self._pending = b""

    # ----------------------------------------------------------------------------- files

    def _path(self, name: str) -> Path:
        return self.dir / f"t{self.names.index(name)}.npy"

    def write_files(self, names: list[str]) -> None:
        self.dir.mkdir(parents=True, exist_ok=True)
        values = self._values()
        for name in names:
            final = self._path(name)
            temp = final.with_name(final.name + ".tmp.npy")
            np.save(temp, np.ascontiguousarray(values[name]), allow_pickle=False)
            os.replace(temp, final)

    # ----------------------------------------------------------------------------- process

    def start(self) -> None:
        self.write_files(self.names)
        self._pending = b""
        self.proc = subprocess.Popen(
            [self.python, "-I", "-c", BOOT.format(root=PACKAGE_ROOT), "kernel"],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            close_fds=True,
            env=dict(KERNEL_ENV),
            cwd=str(self.dir),
            start_new_session=True,
        )
        ready = self._read(1, time.monotonic() + READY_TIMEOUT_S, exact=True)
        if ready != b"K":
            self._kill()
            raise KernelError("worker_kernel_filter_missing", "the kernel sent no ready byte")
        theirs = child_seccomp.status_fields(self.proc.pid)
        ours = child_seccomp.status_fields("self")
        want = str(int(ours.get("Seccomp_filters", "0")) + 1)
        if theirs.get("NoNewPrivs") != "1" or theirs.get("Seccomp_filters") != want:
            self._kill()
            raise KernelError(
                "worker_kernel_filter_missing",
                f"the kernel's filter is not stacked on this process's ({theirs})",
            )
        self._call(
            {
                "op": "init",
                "dir": str(self.dir),
                "names": self.names,
                "edit": self.edit,
                "notes": self.notes,
                "rebind": self.rebind,
            },
            CONTROL_TIMEOUT_S,
        )

    @property
    def pid(self) -> int | None:
        return None if self.proc is None else self.proc.pid

    def _kill(self) -> None:
        if self.proc is None:
            return
        with contextlib.suppress(ProcessLookupError, PermissionError):
            os.killpg(self.proc.pid, signal.SIGKILL)
        self.proc.wait()
        for stream in (self.proc.stdin, self.proc.stdout):
            if stream is not None:
                stream.close()
        self.proc = None

    def close(self) -> None:
        self._kill()

    def restart(self) -> None:
        self._kill()
        self.restarts += 1
        self.start()

    # ----------------------------------------------------------------------------- the pipe

    def _read(self, n: int, deadline: float, *, exact: bool = False) -> bytes | None:
        """``exact`` bytes, or one line; ``None`` on timeout, EOF or an oversized answer."""
        assert self.proc is not None and self.proc.stdout is not None
        fd = self.proc.stdout.fileno()
        with selectors.DefaultSelector() as sel:
            sel.register(fd, selectors.EVENT_READ)
            while True:
                if exact and len(self._pending) >= n:
                    out, self._pending = self._pending[:n], self._pending[n:]
                    return out
                if not exact and b"\n" in self._pending:
                    line, self._pending = self._pending.split(b"\n", 1)
                    return line
                left = deadline - time.monotonic()
                if left <= 0 or not sel.select(left):
                    return None
                chunk = os.read(fd, 1 << 16)
                if not chunk or len(self._pending) + len(chunk) > MAX_ANSWER_BYTES:
                    return None
                self._pending += chunk

    def _send(self, msg: Mapping[str, Any]) -> bool:
        assert self.proc is not None and self.proc.stdin is not None
        try:
            self.proc.stdin.write((json.dumps(msg) + "\n").encode())
            self.proc.stdin.flush()
        except (BrokenPipeError, OSError):
            return False
        return True

    def _call(self, msg: Mapping[str, Any], timeout_s: float) -> dict[str, Any] | None:
        if self.proc is None or not self._send(msg):
            return None
        line = self._read(0, time.monotonic() + timeout_s)
        if line is None:
            return None
        try:
            answer = json.loads(line)
        except ValueError:
            return None
        return answer if isinstance(answer, dict) else None

    # ----------------------------------------------------------------------------- the API

    def exec(self, code: str) -> ExecResult:
        """Run ``code``; its output and the deltas it made to ``W`` (checked), or a restart."""
        answer = self._call({"op": "exec", "code": code}, self.timeout_s)
        if answer is None or answer.get("ok") is not True:
            self.restart()
            from wsp_worker.miwsp_engine import prompts

            return ExecResult(prompts.KERNEL_RESTARTED, restarted=True)
        notes = answer.get("notes")
        if isinstance(notes, list):
            self.notes = [str(x)[:4_000] for x in notes][-200:]
        return ExecResult(str(answer.get("output", ""))[:100_000], self._deltas(answer))

    def _deltas(self, answer: Mapping[str, Any]) -> dict[str, tuple[np.ndarray, np.ndarray]]:
        values = self._values()
        out: dict[str, tuple[np.ndarray, np.ndarray]] = {}
        raw = answer.get("deltas")
        for item in raw if isinstance(raw, list) else []:
            if not isinstance(item, dict) or item.get("name") not in values or not self.edit:
                continue
            size = values[item["name"]].size
            idx, vals = item.get("idx"), item.get("values")
            if not isinstance(idx, list) or not isinstance(vals, list) or len(idx) != len(vals):
                continue
            if not all(
                isinstance(i, int) and not isinstance(i, bool) and 0 <= i < size for i in idx
            ):
                continue
            if not all(isinstance(v, int | float) and not isinstance(v, bool) for v in vals):
                continue
            out[item["name"]] = (
                np.asarray(idx, dtype=np.int64),
                np.asarray(vals, dtype=np.float64),
            )
        return out

    def sync(self, names: list[str]) -> None:
        """Rewrite ``names`` from the store's accepted values, and re-map them in the kernel."""
        if not names:
            return
        self.write_files(names)
        if self._call({"op": "sync", "names": names}, CONTROL_TIMEOUT_S) is None:
            self.restart()
