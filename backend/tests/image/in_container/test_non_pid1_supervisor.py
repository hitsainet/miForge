"""INSIDE the worker image: a supervisor that is not PID 1 kills nothing of its uid and wipes no
IPC object (R3A-1; FTASKS 4.4b, 4.6; control C10ab).

This is the ``LocalProcessLauncher`` guarantee
(``integration/wsp/test_local_launcher_is_harmless.py``) proved where breaking it is safe: in a
throwaway container with its own PID namespace (never ``--pid host``), where pytest is PID 1 and
``kill(-1)`` from uid 10002 can reach only this container's processes. Control C10ab drops the
supervisor's ``getpid() == 1`` gate on ``kill(-1)``: the sentinel below then dies, and this test
goes red. On a workstation the same mutant would kill every process of the developer's uid, so it
is never run there.

Run by ``tests/image/test_wsp_worker_image.py::test_in_container_suite`` as
``python -I -m pytest`` in the image's ``test`` target (``WSP_IN_IMAGE_TESTS=1``); anywhere else it
skips. Standard library and pytest only; nothing of ``backend/tests`` but this directory and the
scripted children is mounted.
"""

from __future__ import annotations

import ctypes
import hashlib
import http.client
import json
import os
import secrets
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import pytest

pytestmark = pytest.mark.skipif(
    os.environ.get("WSP_IN_IMAGE_TESTS") != "1",
    reason="runs only inside the WSP worker image (tests/image/test_wsp_worker_image.py starts it)",
)

SCRIPTED = "/opt/wsp-tests/wsp_children/scripted.py"


def _alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    return Path(f"/proc/{pid}/stat").read_text().rsplit(")", 1)[1].split()[0] != "Z"


def _request(
    port: int, method: str, path: str, body: bytes = b"", ctype: str = ""
) -> tuple[int, bytes, dict[str, str]]:
    conn = http.client.HTTPConnection("127.0.0.1", port, timeout=30)
    try:
        conn.request(
            method, path, body=body or None, headers={"Content-Type": ctype} if ctype else {}
        )
        r = conn.getresponse()
        return r.status, r.read(), {k.lower(): v for k, v in r.getheaders()}
    finally:
        conn.close()


def _session_body(boot: str) -> tuple[str, bytes]:
    push = {
        "boot_id": boot,
        "run_id": "wsr_test",
        "mode": "session",
        "spec": {
            "time_budget_s": 30,
            "grace_s": 5,
            "pull_timeout_s": 30,
            "test_actions": ["write_result"],
        },
        "files": [],
        "gateway": None,
    }
    boundary = "wsp" + secrets.token_hex(8)
    body = (
        f'--{boundary}\r\nContent-Disposition: form-data; name="session"\r\n\r\n'.encode()
        + json.dumps(push).encode()
        + f"\r\n--{boundary}--\r\n".encode()
    )
    return f"multipart/form-data; boundary={boundary}", body


def test_a_non_pid1_supervisor_harms_nothing_of_its_uid() -> None:
    assert os.getpid() == 1, "this test must be the container's PID 1"
    libc = ctypes.CDLL(None, use_errno=True)
    sentinel = subprocess.Popen(
        [sys.executable, "-I", "-c", "import time; time.sleep(300)"], start_new_session=True
    )
    segment = libc.shmget(0, 4096, 0o1600)
    assert segment >= 0
    root = Path(tempfile.mkdtemp(prefix="wsp-worker-", dir="/tmp"))
    (root / ".wsp-worker-root").write_text("in-image launcher root\n")
    env = {
        "PATH": "/usr/local/bin:/usr/bin:/bin",
        "WSP_ENVIRONMENT": "ci",
        "WSP_WORKER_PORT": "0",
        "WSP_WORKER_ROOT": str(root),
        "WSP_TEST_SCRIPT": SCRIPTED,
    }
    sup = subprocess.Popen(
        [sys.executable, "-I", "-m", "wsp_worker", "supervisor"],
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    try:
        assert sup.stdout is not None
        line = sup.stdout.readline()
        assert line, sup.stderr.read() if sup.stderr else b""
        port = int(json.loads(line)["port"])
        status, body, _ = _request(port, "GET", "/ready")
        boot = json.loads(body)["boot_id"]
        ctype, payload = _session_body(boot)
        status, body, _ = _request(port, "POST", "/session", payload, ctype)
        assert status == 202, body
        end = time.monotonic() + 30
        while json.loads(_request(port, "GET", "/status")[1])["state"] != "finished":
            assert time.monotonic() < end
            time.sleep(0.1)
        status, result, headers = _request(port, "GET", "/result")
        assert status == 200
        digest = hashlib.sha256(result).hexdigest()
        assert headers["x-content-sha256"] == digest
        ack = json.dumps({"boot_id": boot, "run_id": "wsr_test", "result_sha256": digest})
        assert _request(port, "POST", "/ack", ack.encode(), "application/json")[0] == 202
        assert sup.wait(timeout=15) == 0
        assert _alive(sentinel.pid), "a non-PID-1 supervisor killed a process of its uid"
        assert segment in _segments(), "a non-PID-1 supervisor removed a SysV segment of its uid"
    finally:
        for proc in (sup, sentinel):
            if proc.poll() is None:
                proc.kill()
                proc.wait(timeout=10)
        libc.shmctl(ctypes.c_int(segment), 0, None)
        shutil.rmtree(root, ignore_errors=True)


def _segments() -> list[int]:
    rows = Path("/proc/sysvipc/shm").read_text().splitlines()
    head = rows[0].split()
    sid, uid = head.index("shmid"), head.index("uid")
    return [int(r.split()[sid]) for r in rows[1:] if int(r.split()[uid]) == os.getuid()]
