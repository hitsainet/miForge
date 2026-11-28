"""Driving a real WSP supervisor as a local process (feature 016; FTID §3.2, PF-42).

The supervisor runs as a subprocess of the test, never PID 1, so it skips ``kill(-1)`` and the IPC
wipe by its own ``getpid() == 1`` gates; its wipe roots live under a fresh launcher root
(``wsp.launcher.make_worker_root``). Tests only ever signal the process they started (and, at
cleanup, the tree under it by parent link).
"""

from __future__ import annotations

import hashlib
import http.client
import io
import json
import os
import secrets
import signal
import subprocess
import sys
import tarfile
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

SRC = Path(__file__).resolve().parents[2] / "src"
CHILDREN = Path(__file__).resolve().parent / "wsp_children"
SCRIPTED = CHILDREN / "scripted.py"
NO_FILTER = CHILDREN / "no_filter.py"


def encode_session(push: dict[str, Any], files: list[tuple[str, bytes]]) -> tuple[str, bytes]:
    """``(content type, body)`` of a ``POST /session``: the push, then one part per file."""
    boundary = "wsp" + secrets.token_hex(12)
    parts = [
        f'--{boundary}\r\nContent-Disposition: form-data; name="session"\r\n'
        "Content-Type: application/json\r\n\r\n".encode() + json.dumps(push).encode() + b"\r\n"
    ]
    for name, data in files:
        parts.append(
            f'--{boundary}\r\nContent-Disposition: form-data; name="file"; filename="{name}"\r\n'
            "Content-Type: application/octet-stream\r\n\r\n".encode() + data + b"\r\n"
        )
    parts.append(f"--{boundary}--\r\n".encode())
    return f"multipart/form-data; boundary={boundary}", b"".join(parts)


def file_entry(name: str, data: bytes, role: str = "heads") -> dict[str, Any]:
    return {
        "name": name,
        "role": role,
        "sha256": hashlib.sha256(data).hexdigest(),
        "bytes": len(data),
    }


@dataclass
class Reply:
    status: int
    body: bytes
    headers: dict[str, str] = field(default_factory=dict)

    def json(self) -> Any:
        return json.loads(self.body)


@dataclass
class LocalSupervisor:
    process: subprocess.Popen[bytes]
    port: int
    root: Path

    def request(
        self,
        method: str,
        path: str,
        body: bytes | None = None,
        headers: dict[str, str] | None = None,
        timeout: float = 30,
    ) -> Reply:
        conn = http.client.HTTPConnection("127.0.0.1", self.port, timeout=timeout)
        try:
            conn.request(method, path, body=body, headers=headers or {})
            r = conn.getresponse()
            return Reply(r.status, r.read(), {k.lower(): v for k, v in r.getheaders()})
        finally:
            conn.close()

    def ready(self) -> Reply:
        return self.request("GET", "/ready")

    def status(self) -> dict[str, Any]:
        reply = self.request("GET", "/status")
        assert reply.status == 200, reply
        return dict(reply.json())

    def post_json(self, path: str, doc: dict[str, Any]) -> Reply:
        return self.request(
            "POST", path, json.dumps(doc).encode(), {"Content-Type": "application/json"}
        )

    def session(self, push: dict[str, Any], files: list[tuple[str, bytes]] | None = None) -> Reply:
        ctype, body = encode_session(push, files or [])
        return self.request("POST", "/session", body, {"Content-Type": ctype})

    def wait_state(self, state: str, timeout: float = 30) -> dict[str, Any]:
        end = time.monotonic() + timeout
        while True:
            current = self.status()
            if current["state"] == state:
                return current
            assert time.monotonic() < end, f"never reached {state}: {current}"
            time.sleep(0.05)

    def result(self) -> tuple[dict[str, bytes], str]:
        reply = self.request("GET", "/result")
        assert reply.status == 200, reply
        digest = reply.headers["x-content-sha256"]
        assert hashlib.sha256(reply.body).hexdigest() == digest
        with tarfile.open(fileobj=io.BytesIO(reply.body)) as tar:
            members = {m.name: tar.extractfile(m).read() for m in tar.getmembers()}  # type: ignore[union-attr]
        return members, digest

    def wait_exit(self, timeout: float = 15) -> int:
        return self.process.wait(timeout=timeout)

    def stop(self) -> None:
        """Kill what this test started: the supervisor and the tree under it by parent link."""
        from wsp.launcher import kill_tree

        kill_tree(self.process)


def push_doc(
    boot_id: str,
    *,
    actions: list[str] | None = None,
    files: list[dict[str, Any]] | None = None,
    mode: str = "session",
    time_budget_s: float = 30,
    run_id: str = "wsr_test",
    gateway: dict[str, Any] | None = None,
    **spec_extra: Any,
) -> dict[str, Any]:
    return {
        "boot_id": boot_id,
        "run_id": run_id,
        "mode": mode,
        "spec": {
            "time_budget_s": time_budget_s,
            "grace_s": 5,
            "pull_timeout_s": 30,
            "test_actions": actions or [],
            **spec_extra,
        },
        "files": files or [],
        "gateway": gateway,
    }


def start(
    root: Path,
    *,
    script: Path | None = SCRIPTED,
    environment: str = "ci",
    port: int = 0,
    extra_env: dict[str, str] | None = None,
    expect_listening: bool = True,
) -> LocalSupervisor:
    """Start ``python -m wsp_worker supervisor`` with an explicit, minimal environment."""
    env = {
        "PATH": os.environ.get("PATH", "/usr/bin:/bin"),
        "PYTHONPATH": str(SRC),
        "WSP_ENVIRONMENT": environment,
        "WSP_WORKER_PORT": str(port),
        "WSP_WORKER_ROOT": str(root),
        **({"WSP_TEST_SCRIPT": str(script)} if script else {}),
        **(extra_env or {}),
    }
    env = {k: v for k, v in env.items() if v is not None}
    proc = subprocess.Popen(
        [sys.executable, "-m", "wsp_worker", "supervisor"],
        env=env,
        cwd=str(SRC),
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    if not expect_listening:
        return LocalSupervisor(proc, 0, root)
    assert proc.stdout is not None
    line = proc.stdout.readline()
    if not line:
        err = proc.stderr.read().decode() if proc.stderr else ""
        proc.wait(timeout=10)
        raise AssertionError(f"supervisor did not start (exit {proc.returncode}): {err}")
    return LocalSupervisor(proc, int(json.loads(line)["port"]), root)


def refusal(proc: subprocess.Popen[bytes], timeout: float = 15) -> tuple[int, dict[str, Any]]:
    """The exit code and the JSON refusal a supervisor that refused to start wrote to stderr."""
    _, err = proc.communicate(timeout=timeout)
    lines = [ln for ln in err.decode().splitlines() if ln.startswith("{")]
    return proc.returncode, (json.loads(lines[-1]) if lines else {"stderr": err.decode()})


def signal_alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    state = Path(f"/proc/{pid}/stat").read_text().rsplit(")", 1)[1].split()[0]
    return state != "Z"


__all__ = [
    "LocalSupervisor",
    "Reply",
    "encode_session",
    "file_entry",
    "push_doc",
    "refusal",
    "signal",
    "signal_alive",
    "start",
]
