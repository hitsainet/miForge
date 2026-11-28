"""INSIDE the worker image: the FR-018.14 probe in an engine session's kernel grandchild (feature
018; FTASKS 10.2). Started by ``tests/image/test_engine_probe_image.py`` as the container's own
process (PID 1, the pod's flags). Everything is made HERE, with the image's own numpy, safetensors
and worker package: the bundle (``image_bundle.write_bundle``, the in-image bundle writer), the
child line (``image_bundle.child_line``), a fresh token and the probe's masked needles. The host
runs only pytest and the standard library (CI's harness), and never sees the token.

For each placement it runs the REAL child entry point (``python -I -c <boot> run``, as the
supervisor starts it: the ready byte, then the line on stdin) against a scripted OpenAI endpoint on
this container's loopback. The editor's first reply runs the probe as ``python`` code (in the
grandchild); the second submits. Placements:

- ``pid1_child``: the session is a child of PID 1 (this process), as under the pod's supervisor;
- ``non_pid1_child``: the session's parent is an intermediate process, not PID 1.

Prints one line per placement: ``PROBE-RESULT <placement> <parent pid> <probe json>``, then
``TOKEN-SHA256 <hex>`` (the host checks the token itself never reached the log).
"""

from __future__ import annotations

import base64
import hashlib
import json
import os
import secrets
import subprocess
import sys
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

BOOT = (
    "import sys, runpy; sys.path.insert(0, '/opt/wsp'); "
    "runpy.run_module('wsp_worker', run_name='__main__')"
)
RELAY = "import subprocess, sys; sys.exit(subprocess.call(sys.argv[1:]))"


class Endpoint:
    """Two replies per session: the probe as ``python``, then ``submit``."""

    def __init__(self, code: str) -> None:
        self.code = code
        self.served = 0
        outer = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *_args: Any) -> None:
                return

            def do_POST(self) -> None:
                self.rfile.read(int(self.headers.get("content-length") or 0))
                data = json.dumps(outer.reply()).encode()
                self.send_response(200)
                self.send_header("content-type", "application/json")
                self.send_header("content-length", str(len(data)))
                self.end_headers()
                self.wfile.write(data)

        self.server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.port = self.server.server_address[1]
        threading.Thread(target=self.server.serve_forever, daemon=True).start()

    def reply(self) -> dict[str, Any]:
        name, args = (
            ("python", {"code": self.code})
            if self.served % 2 == 0
            else ("submit", {"answer": "probed"})
        )
        self.served += 1
        call = {
            "id": f"c{self.served}",
            "type": "function",
            "function": {"name": name, "arguments": json.dumps(args)},
        }
        return {
            "id": f"cmpl{self.served}",
            "object": "chat.completion",
            "choices": [
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": None, "tool_calls": [call]},
                    "finish_reason": "tool_calls",
                }
            ],
            "usage": {"prompt_tokens": 10, "completion_tokens": 5},
        }


def needles(token: str) -> list[list[Any]]:
    """The probe's masked needles (``tests/support/wsp_engine.needles``, which the image lacks)."""
    out = []
    for form in (
        token.encode(),
        base64.b64encode(token.encode()),
        json.dumps(token)[1:-1].encode(),
    ):
        key = 0x5A
        out.append([[b ^ key for b in form[:8]], key, hashlib.sha256(form).hexdigest(), len(form)])
    return out


def run(placement: str, token: str, endpoint: Endpoint) -> None:
    import numpy as np
    from image_bundle import child_line, write_bundle

    root = Path("/tmp") / f"probe-{placement}"
    tensors = {
        "weight": np.linspace(-1.0, 1.0, 8, dtype=np.float32),
        "bias": np.zeros(1, dtype=np.float32),
    }
    manifest = write_bundle(root / "work", tensors, "A linear probe (the FR-018.14 probe case).")
    line = child_line(root, manifest, "inspect", "Describe the probe.")
    line["spec"]["request"]["provider"] = {"id": "miwsp_engine", "version": "0.1.0"}
    line["gateway"] = {"ip": "127.0.0.1", "port": endpoint.port, "token": token}
    practice = [
        line["paths"]["work"] + "/" + f["name"] for f in manifest["files"] if f["role"] != "heads"
    ]
    args = {"needles": needles(token), "practice_paths": practice}
    source = (Path(__file__).resolve().parent / "engine_probe.py").read_text()
    endpoint.code = f"{source}\nimport json as _j\nprint('PROBE' + _j.dumps(probe(**{args!r})))"
    env = {
        "PATH": "/usr/local/bin:/usr/bin:/bin",
        "HOME": "/tmp",
        "PYTHONNOUSERSITE": "1",
        "OMP_NUM_THREADS": "1",
        "OPENBLAS_NUM_THREADS": "1",
        "MKL_NUM_THREADS": "1",
        "WSP_MODE": "run",
        "WSP_PROBE_CANARY": token,
    }
    child = [sys.executable, "-I", "-c", BOOT, "run"]
    argv = child if placement == "pid1_child" else [sys.executable, "-I", "-c", RELAY, *child]
    proc = subprocess.Popen(
        argv,
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        env=env,
        cwd=line["paths"]["work"],
        start_new_session=True,
    )
    assert proc.stdout is not None
    ready = os.read(proc.stdout.fileno(), 1)
    _stdout, stderr = proc.communicate(json.dumps(line).encode() + b"\n", timeout=240)
    if ready != b"R" or proc.returncode != 0:
        print(f"PROBE-FAILED {placement} {proc.returncode} {stderr.decode()[-2000:]}", flush=True)
        return
    outputs = [
        str(m.get("content"))
        for m in map(json.loads, (root / "out" / "transcript.jsonl").read_text().splitlines())
        if m.get("role") == "tool" and "PROBE" in str(m.get("content"))
    ]
    if not outputs:
        print(f"PROBE-FAILED {placement} no-probe-output", flush=True)
        return
    found = json.loads(outputs[0].split("PROBE", 1)[1].strip().splitlines()[0])
    found["practice_count"] = str(len(practice))
    print(f"PROBE-RESULT {placement} {os.getpid()} {json.dumps(found)}", flush=True)


def main() -> int:
    sys.path.insert(0, str(Path(__file__).resolve().parent))  # image_bundle, engine_probe
    token = "wsp_tok_" + secrets.token_hex(24)
    endpoint = Endpoint("")
    for placement in ("pid1_child", "non_pid1_child"):
        run(placement, token, endpoint)
    endpoint.server.shutdown()
    print(f"TOKEN-SHA256 {hashlib.sha256(token.encode()).hexdigest()}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
