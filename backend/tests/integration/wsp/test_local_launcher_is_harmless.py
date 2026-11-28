"""``LocalProcessLauncher`` harms nothing of the developer's (FR-016.10's AC; R3A-1; FTASKS 4.4b).

A full session through the launcher runs the real supervisor, which is not PID 1 here, so its
``kill(-1)`` and IPC wipe are skipped by their ``getpid() == 1`` gates. This test plants what those
two would destroy, runs a session, and requires all of it to survive:

- a same-uid sentinel process in its own session (``kill(-1)`` would kill it);
- a sentinel file in the real ``/tmp`` (a wipe of ``/tmp`` would remove it);
- a SysV shared-memory segment (the IPC wipe would remove it).

The test removes only what it created. Control C10ab (dropping the gate) kills every process of the
test's uid, so it runs only inside the image job's throwaway container, never on a workstation.
"""

from __future__ import annotations

import ctypes
import json
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import pytest

from core.errors import Refusal
from tests.support import wsp_supervisor as ws
from tests.support import wsp_worker_env
from wsp.launcher import LocalProcessLauncher
from wsp_worker import ipc_wipe, spec


@pytest.fixture(autouse=True)
def _seccomp() -> None:
    wsp_worker_env.require_seccomp()


def test_a_full_session_leaves_the_developers_processes_files_and_ipc_alone() -> None:
    libc = ctypes.CDLL(None, use_errno=True)
    sentinel = subprocess.Popen(
        [sys.executable, "-I", "-c", "import time; time.sleep(300)"], start_new_session=True
    )
    fd, name = tempfile.mkstemp(prefix="wsp-sentinel-", dir="/tmp")
    sentinel_file = Path(name)
    segment = libc.shmget(0, 4096, 0o1600)  # IPC_PRIVATE, IPC_CREAT | 0600
    assert segment >= 0
    launcher = LocalProcessLauncher("ci", test_script=str(ws.SCRIPTED))
    try:
        with launcher.session() as worker:
            sup = ws.LocalSupervisor(worker.process, worker.port, worker.root)
            boot = sup.ready().json()["boot_id"]
            data = b"heads"
            push = ws.push_doc(
                boot, actions=["write_result"], files=[ws.file_entry("heads.safetensors", data)]
            )
            assert sup.session(push, [("heads.safetensors", data)]).status == 202
            sup.wait_state("finished")
            members, digest = sup.result()
            assert json.loads(members["supervisor.json"])["exit_code"] == 0
            ack = {"boot_id": boot, "run_id": "wsr_test", "result_sha256": digest}
            assert sup.post_json("/ack", ack).status == 202
            assert worker.process.wait(timeout=15) == spec.EXIT_OK
            root = worker.root
        assert not root.exists()
        assert ws.signal_alive(sentinel.pid)
        assert sentinel_file.exists()
        assert segment in ipc_wipe.list_sysv("shm")
    finally:
        sentinel.kill()
        sentinel.wait(timeout=10)
        import os

        os.close(fd)
        sentinel_file.unlink(missing_ok=True)
        libc.shmctl(segment, 0, None)


@pytest.mark.parametrize("bad", ["/", "/tmp", "/dev/shm", "", "relative"])
def test_a_bad_worker_root_is_refused_before_anything_is_touched(bad: str) -> None:
    assert not Path("/", ".wsp-worker-root").exists() and not Path("/tmp/.wsp-worker-root").exists()
    proc = ws.start(
        Path(bad) if bad else Path(""), expect_listening=False, extra_env={"WSP_WORKER_ROOT": bad}
    ).process
    code, refused = ws.refusal(proc)
    assert code == spec.EXIT_REFUSED and refused["refused"] == "worker_root_refused"


def test_a_root_resolving_outside_itself_is_refused(tmp_path: Path) -> None:
    from wsp.launcher import make_worker_root

    root = make_worker_root()
    outside = tmp_path / "outside"
    outside.mkdir()
    (outside / "canary").write_text("keep")
    (root / "out").symlink_to(outside)
    try:
        code, refused = ws.refusal(ws.start(root, expect_listening=False).process)
        assert code == spec.EXIT_REFUSED and refused["refused"] == "worker_root_refused"
        assert (outside / "canary").read_text() == "keep"
    finally:
        shutil.rmtree(root, ignore_errors=True)


def test_the_launcher_is_refused_in_production() -> None:
    """P43 (control C10n)."""
    with pytest.raises(Refusal) as caught:
        LocalProcessLauncher("prod")
    assert caught.value.code == "wsp_launcher_refused_in_production"


def test_the_launcher_passes_no_environment_of_its_own(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """D20: the backend's environment (which holds secrets) never reaches the supervisor. The
    interpreter is replaced by a script that records its environment and answers the listening
    line, so ``start`` runs its real path; the supervisor itself is undumpable and unreadable."""
    monkeypatch.setenv("WSP_GATEWAY_SERVICE_TOKEN", "must-not-leak")
    fake = tmp_path / "python"
    fake.write_text(
        '#!/bin/sh\nenv > "$WSP_WORKER_ROOT/env.txt"\necho \'{"event": "listening", "port": 1}\'\n'
        "exec sleep 60\n"
    )
    fake.chmod(0o700)
    launcher = LocalProcessLauncher("ci", python=str(fake))
    worker = launcher.start()
    try:
        env = dict(
            line.split("=", 1) for line in (worker.root / "env.txt").read_text().splitlines()
        )
    finally:
        launcher.stop(worker)
    assert "WSP_GATEWAY_SERVICE_TOKEN" not in env
    assert {k for k in env if k not in ("PWD", "SHLVL", "_")} == {
        "PATH",
        "PYTHONPATH",
        "WSP_ENVIRONMENT",
        "WSP_WORKER_PORT",
        "WSP_WORKER_ROOT",
        "OMP_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "MKL_NUM_THREADS",
    }
    assert env["WSP_ENVIRONMENT"] == "ci"
    assert not worker.root.exists()
