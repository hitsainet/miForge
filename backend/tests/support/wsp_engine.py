"""Helpers for the ``miwsp_engine`` tests (feature 018): a child line shaped as the supervisor
writes it, and two ways to run the engine under the worker's interpreter.

- :func:`run_case`: ``wsp_children/engine_cases.py`` (the real child preamble, then
  ``miwsp_engine.run`` with test options such as a short kernel timeout);
- :func:`run_main`: the REAL entry point, ``python -I -c <boot> run``, exactly as the supervisor
  starts the child: the ready byte, then the line on stdin (FR-018.2's dispatch).

Both skip by name without the worker's interpreter or libseccomp (``wsp_worker_env``), and fail
under ``MIFORGE_REQUIRE_WSP_WORKER=1``.
"""

from __future__ import annotations

import base64
import hashlib
import json
import os
import shutil
import subprocess
from pathlib import Path
from typing import Any

from tests.support import wsp_worker_env
from tests.support.wsp_fixtures import FixtureBundle
from wsp.weight_ops.hashing import file_sha256

CHILDREN = Path(__file__).resolve().parent / "wsp_children"
CASES = CHILDREN / "engine_cases.py"
BOOT = (
    "import sys, runpy; sys.path.insert(0, {root!r}); "
    "runpy.run_module('wsp_worker', run_name='__main__')"
)
#: The child's environment (``spec.CHILD_ENV`` plus ``WSP_MODE``), as the supervisor passes it.
CHILD_ENV = {
    "PATH": "/usr/local/bin:/usr/bin:/bin",
    "HOME": "/tmp",
    "PYTHONNOUSERSITE": "1",
    "OMP_NUM_THREADS": "1",
    "OPENBLAS_NUM_THREADS": "1",
    "MKL_NUM_THREADS": "1",
}


def line_for(
    tmp: Path,
    fx: FixtureBundle,
    kind: str,
    gateway: dict[str, Any] | None,
    **over: Any,
) -> dict[str, Any]:
    """A child line for ``fx`` in fresh ``work/out/tmp`` dirs under ``tmp``."""
    work, out, scratch = tmp / "work", tmp / "out", tmp / "tmp"
    for d in (work, out, scratch):
        d.mkdir(parents=True)
    files = []
    for entry in fx.manifest["files"]:
        shutil.copyfile(fx.root / entry["name"], work / entry["name"])
        files.append(
            {
                "name": entry["name"],
                "role": entry["role"],
                "sha256": file_sha256(work / entry["name"]),
                "bytes": entry["bytes"],
            }
        )
    for name, role, source in over.get("extra", []):
        shutil.copyfile(source, work / name)
        files.append(
            {
                "name": name,
                "role": role,
                "sha256": file_sha256(work / name),
                "bytes": (work / name).stat().st_size,
            }
        )
    request: dict[str, Any] = {
        "kind": kind,
        "task": "Repair the probe.",
        "parent": {"kind": "bundle", "ref": fx.manifest["bundle_sha256"]},
        "provider": {"id": "miwsp_engine", "version": "0.1.0"},
        "budgets": {
            "tokens": over.get("tokens", 100_000),
            "time_s": over.get("time_s", 600),
            "max_steps": over.get("max_steps", 10),
            "forward_passes": over.get("forward_passes", 0),
            "cost_usd": over.get("cost_usd", 1.0),
        },
        "change_limits": {
            "max_changed_entries": over.get("max_changed_entries"),
            "allowed_tensors": over.get("allowed_tensors"),
            "frozen_tensors": over.get("frozen_tensors", []),
        },
        "thinking": over.get("thinking"),
        "observation_tools": over.get("observation_tools", []),
    }
    if "primary_sets" in over:
        request["primary_sets"] = over["primary_sets"]
    if "thinking_tokens" in over:
        request["thinking_tokens"] = over["thinking_tokens"]
    # FU-8 (2026-11-25): the request freezes python only as "off" (absent means on); the tests
    # written before it pass python=False.
    if over.get("python") in (False, "off"):
        request["python"] = "off"
    # Paging (2026-11-21): the request freezes it only as "off" (absent means on). These tests
    # default to off, so every test written before paging runs unchanged; ``paging="on"`` gives
    # the production default shape, the key absent.
    if over.get("paging", "off") == "off":
        request["paging"] = "off"
    editor = {
        "kind": over.get("editor_kind", "openai_compatible"),
        "model": "m",
        "serving": over.get("serving", "miserver"),
        "prices": over.get(
            "prices", {"input": None, "output": None, "cache_write": None, "cache_read": None}
        ),
        **over.get("editor", {}),
    }
    mode = over.get("mode", "session")
    spec: dict[str, Any] = {
        "time_budget_s": 600,
        "grace_s": 60,
        "pull_timeout_s": 60,
        "request": request,
        "manifest": fx.manifest,
        "editor": editor,
    }
    if "replay_of_mode" in over:
        spec["replay_of_mode"] = over["replay_of_mode"]
    return {
        "mode": mode,
        "run_id": "wsp_000000000001",
        "spec": spec,
        "files": files,
        "gateway": None if mode == "replay" else gateway,
        "paths": {"work": str(work), "out": str(out), "tmp": str(scratch)},
        # The supervisor writes the time its kill leaves (spec.child_line, fix 2026-11-21).
        "time_left_s": float(over.get("time_left_s", over.get("time_s", 600))),
    }


def _python() -> str:
    wsp_worker_env.require_seccomp()
    return wsp_worker_env.worker_python(("numpy", "httpx2"))


def run_case(tmp: Path, line: dict[str, Any], **options: Any) -> dict[str, Any]:
    """``engine_cases.py``: the real preamble, then the engine with ``options``."""
    python = _python()
    extra_env = options.pop("_env", {})
    (tmp / "line.json").write_text(json.dumps(line))
    (tmp / "options.json").write_text(json.dumps(options))
    args = [str(wsp_worker_env.SRC), str(tmp / "line.json"), str(tmp / "options.json")]
    done = subprocess.run(
        [python, "-I", str(CASES), *args],
        env={**CHILD_ENV, **extra_env},
        capture_output=True,
        timeout=300,
        check=False,
    )
    assert done.returncode == 0, done.stderr.decode()[-4000:]
    stdout = done.stdout.decode()
    assert stdout.startswith("R"), stdout[:200]  # the preamble's ready byte
    stdout = stdout[1:]
    last = json.loads(stdout.strip().splitlines()[-1])
    return {**last, "stdout": stdout, "stderr": done.stderr.decode()}


def run_main(
    tmp: Path, line: dict[str, Any], *, env: dict[str, str] | None = None, mode: str = "run"
) -> dict[str, Any]:
    """The real child entry point, as the supervisor starts it: ready byte, then the line."""
    python = _python()
    proc = subprocess.Popen(
        [python, "-I", "-c", BOOT.format(root=str(wsp_worker_env.SRC)), mode],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        env={**CHILD_ENV, "WSP_MODE": mode, **(env or {})},
        cwd=line["paths"]["work"],
        start_new_session=True,
    )
    assert proc.stdout is not None and proc.stdin is not None
    ready = os.read(proc.stdout.fileno(), 1)
    assert ready == b"R", ready
    stdout, stderr = proc.communicate(json.dumps(line).encode() + b"\n", timeout=300)
    return {"exit": proc.returncode, "stdout": stdout.decode(), "stderr": stderr.decode()}


def needles(token: str) -> list[list[Any]]:
    """The probe's masked needles for the token's raw, base64 and JSON-escaped forms."""
    out = []
    for form in (
        token.encode(),
        base64.b64encode(token.encode()),
        json.dumps(token)[1:-1].encode(),
    ):
        key = 0x5A
        out.append([[b ^ key for b in form[:8]], key, hashlib.sha256(form).hexdigest(), len(form)])
    return out


def probe_code(token: str, practice_paths: list[str]) -> str:
    """The probe as editor code: its source, then one printed JSON line."""
    source = (CHILDREN / "engine_probe.py").read_text()
    args = {"needles": needles(token), "practice_paths": practice_paths}
    return f"{source}\nimport json as _j\nprint('PROBE' + _j.dumps(probe(**{args!r})))"


def result(tmp: Path) -> dict[str, Any]:
    loaded: dict[str, Any] = json.loads((tmp / "out" / "result.json").read_text())
    return loaded


def transcript(tmp: Path) -> list[dict[str, Any]]:
    return [json.loads(x) for x in (tmp / "out" / "transcript.jsonl").read_text().splitlines()]


def tool_outputs(tmp: Path) -> list[str]:
    """Every tool or result message's content, in order, from the transcript."""
    return [
        str(m["content"])
        for m in transcript(tmp)
        if m.get("role") == "tool"
        or (m.get("role") == "user" and str(m.get("content", "")).startswith("Result of "))
    ]
