"""A WSP test child (``WSP_TEST_SCRIPT``; honoured only outside prod): the real preamble, then the
actions the push's ``spec.test_actions`` lists. Run as ``python -I scripted.py <mode> <root>``."""

from __future__ import annotations

import json
import os
import signal
import socket
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, sys.argv[2])
from wsp_worker.__main__ import child_preamble, read_child_line

child_preamble()
# `-I` keeps this script's own directory off sys.path; the image job's actions live beside it.
sys.path.insert(1, str(Path(__file__).resolve().parent))
from image_actions import IMAGE_ACTIONS  # noqa: E402

line = read_child_line()
out = Path(str(line["paths"]["out"]))  # type: ignore[index]
tmp = Path(str(line["paths"]["tmp"]))  # type: ignore[index]
spec = line["spec"]
assert isinstance(spec, dict)
code = 0
for action in spec.get("test_actions", []):
    name, _, arg = str(action).partition(":")
    if name == "write_result":
        (out / "result.json").write_text(json.dumps({"stop_reason": "submitted"}))
    elif name == "write_transcript":
        lines = [{"role": "user", "content": "hi"}, {"role": "assistant", "content": "done"}]
        (out / "transcript.jsonl").write_text("".join(json.dumps(x) + "\n" for x in lines))
    elif name == "print":
        print(arg, flush=True)
    elif name == "dump_env":
        # What the supervisor passed at exec: /proc/self/environ, not os.environ (CPython's own
        # locale coercion adds LC_CTYPE in-process, PEP 538).
        raw = Path("/proc/self/environ").read_bytes().split(b"\0")
        env = dict(item.decode().split("=", 1) for item in raw if item)
        (out / "env.json").write_text(json.dumps(env))
    elif name == "dump_time_left":
        (out / "time_left.json").write_text(json.dumps(line["time_left_s"]))  # type: ignore[index]
    elif name == "dump_status":
        (out / "status.txt").write_text(Path("/proc/self/status").read_text())
    elif name == "dump_limits":
        # Review OR2-016-5: the file-size limit the preamble set, as the child sees it.
        import resource

        (out / "limits.json").write_text(
            json.dumps({"fsize": list(resource.getrlimit(resource.RLIMIT_FSIZE))})
        )
    elif name == "read_parent_environ":
        try:
            (out / "parent_environ.txt").write_bytes(
                Path(f"/proc/{os.getppid()}/environ").read_bytes()
            )
        except OSError as exc:
            (out / "parent_environ_error.txt").write_text(type(exc).__name__)
    elif name == "symlink":
        os.symlink("/proc/self/environ", out / "transcript.jsonl")
    elif name == "fifo":
        os.mkfifo(out / "fifo")
    elif name == "hardlink":
        (out / "a.txt").write_text("x")
        os.link(out / "a.txt", out / "b.txt")
    elif name == "oversize":
        with open(out / "big.bin", "wb") as fh:
            fh.truncate(int(arg))
    elif name == "write_termination":
        (tmp / "termination-log").write_text(arg)
    elif name == "plant_supervisor_json":
        (out / "supervisor.json").write_text("{}")
    elif name == "ignore_sigalrm_loop":
        signal.signal(signal.SIGALRM, signal.SIG_IGN)
        signal.alarm(1)
        while True:
            time.sleep(0.05)
    elif name == "setsid_sleeper":
        sleeper = subprocess.Popen(
            [sys.executable, "-I", "-c", "import time; time.sleep(600)"], start_new_session=True
        )
        (tmp / "sleeper.pid").write_text(str(sleeper.pid))
    elif name == "post_session":
        with socket.create_connection(("127.0.0.1", int(arg)), timeout=5) as s:
            body = b'{"x": 1}'
            s.sendall(
                b"POST /session HTTP/1.0\r\nContent-Type: multipart/form-data; boundary=b\r\n"
                + f"Content-Length: {len(body)}\r\n\r\n".encode()
                + body
            )
            (out / "post_session.txt").write_bytes(s.recv(4096)[:64])
    elif name in IMAGE_ACTIONS:
        IMAGE_ACTIONS[name](out, tmp, arg)
    elif name == "sleep":
        time.sleep(float(arg))
    elif name == "exit":
        code = int(arg)
sys.stdout.flush()
os._exit(code)
