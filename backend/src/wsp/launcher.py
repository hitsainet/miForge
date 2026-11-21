"""``LocalProcessLauncher``: the WSP worker as a local subprocess, for dev and tests only
(FTID §2.1, §3.2; FR-016.10; R3A-1, P43, PF-42).

It runs the **real supervisor** (``python -m wsp_worker supervisor``) with:

- ``WSP_ENVIRONMENT=ci``, so the supervisor's PID-1 check is off. Not being PID 1, the supervisor
  skips ``kill(-1)`` and the IPC wipe by its own gates: those would reach every process and IPC
  object of the developer's uid;
- a fresh ``WSP_WORKER_ROOT`` from ``tempfile.mkdtemp()`` holding the marker file the supervisor
  requires before it empties anything under it (OVN-wsp-forge-16), removed afterwards;
- an explicit, minimal environment (never this process's: the backend's environment holds secrets
  the worker must never see, D20);
- the loopback interface and an ephemeral port, which the supervisor reports on its first stdout
  line.

**No containment is claimed here** (FR-016.10 step 11): the worker pod is the containment. It is
refused when the backend's ``environment == "prod"`` (``wsp_launcher_refused_in_production``,
P43). There is no Job launcher and no Kubernetes client (D14).
"""

from __future__ import annotations

import json
import os
import shutil
import signal
import subprocess
import sys
import tempfile
from collections.abc import Iterator
from contextlib import contextmanager, suppress
from dataclasses import dataclass
from pathlib import Path

from core.errors import refuse

#: The supervisor refuses a ``WSP_WORKER_ROOT`` without this file (``wsp_worker/supervisor.py``).
ROOT_MARKER = ".wsp-worker-root"
#: backend/src: the directory holding ``wsp_worker``.
SRC = Path(__file__).resolve().parents[1]
START_TIMEOUT_S = 30


@dataclass
class LocalWorker:
    process: subprocess.Popen[bytes]
    port: int
    root: Path

    @property
    def url(self) -> str:
        return f"http://127.0.0.1:{self.port}"


def make_worker_root() -> Path:
    """A fresh worker root under the system temp directory, with the supervisor's marker."""
    root = Path(tempfile.mkdtemp(prefix="wsp-worker-"))
    (root / ROOT_MARKER).write_text("wsp worker root: emptied and removed after the session\n")
    return root


def tree(pid: int) -> set[int]:
    """``pid`` and every process below it by parent link."""
    parent: dict[int, int] = {}
    for entry in os.listdir("/proc"):
        if entry.isdigit():
            try:
                raw = Path(f"/proc/{entry}/stat").read_text()
            except OSError:
                continue
            parent[int(entry)] = int(raw[raw.rfind(")") + 2 :].split()[1])
    found = {pid}
    grew = True
    while grew:
        grew = False
        for child, ppid in parent.items():
            if ppid in found and child not in found:
                found.add(child)
                grew = True
    return found


def kill_tree(process: subprocess.Popen[bytes]) -> None:
    """SIGKILL a process this code started, and everything below it by parent link."""
    if process.poll() is None:
        for pid in sorted(tree(process.pid) - {process.pid}):
            with suppress(OSError):
                os.kill(pid, signal.SIGKILL)
        process.kill()
    process.wait(timeout=30)
    for stream in (process.stdout, process.stderr, process.stdin):
        if stream is not None:
            stream.close()


class LocalProcessLauncher:
    def __init__(
        self,
        environment: str,
        *,
        test_script: str | None = None,
        python: str = sys.executable,
    ) -> None:
        if environment == "prod":
            raise refuse(
                "wsp_launcher_refused_in_production",
                "LocalProcessLauncher is for dev and tests; in production the driver pushes to the "
                "worker Service (WSP_WORKER_URL).",
            )
        self.environment = environment
        self.test_script = test_script
        self.python = python

    def start(self) -> LocalWorker:
        root = make_worker_root()
        env = {
            "PATH": os.environ.get("PATH", "/usr/local/bin:/usr/bin:/bin"),
            "PYTHONPATH": str(SRC),
            "WSP_ENVIRONMENT": "ci",
            "WSP_WORKER_PORT": "0",
            "WSP_WORKER_ROOT": str(root),
            "OMP_NUM_THREADS": "1",
            "OPENBLAS_NUM_THREADS": "1",
            "MKL_NUM_THREADS": "1",
        }
        if self.test_script:
            env["WSP_TEST_SCRIPT"] = self.test_script
        process = subprocess.Popen(
            [self.python, "-m", "wsp_worker", "supervisor"],
            env=env,
            cwd=str(SRC),
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        assert process.stdout is not None
        line = process.stdout.readline()
        if not line:
            err = process.stderr.read().decode(errors="replace") if process.stderr else ""
            kill_tree(process)
            shutil.rmtree(root, ignore_errors=True)
            raise refuse("worker_push_failed", f"the local supervisor did not start: {err[-2000:]}")
        return LocalWorker(process, int(json.loads(line)["port"]), root)

    def stop(self, worker: LocalWorker) -> None:
        kill_tree(worker.process)
        shutil.rmtree(worker.root, ignore_errors=True)

    @contextmanager
    def session(self) -> Iterator[LocalWorker]:
        worker = self.start()
        try:
            yield worker
        finally:
            self.stop(worker)
