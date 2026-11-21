"""The one place a WSP worker test decides what it can run on this machine (feature 016).

The worker (``backend/src/wsp_worker``) is installed alone in its own image with its own lock,
``backend/requirements-wsp-worker.txt`` (``weightspace[openai,anthropic]``, ``anthropic``,
``openai``), none of which the backend's venv holds. Its tests therefore run their throwaway
children under **the worker's interpreter**, not the test process's:

- ``MIFORGE_WSP_WORKER_PYTHON`` names it (a venv built from ``requirements-wsp-worker.txt``, or
  the worker image's ``python``). Unset, the test process's own interpreter is used when it can
  import the worker's dependencies.
- A test whose interpreter lacks a dependency, or whose machine lacks libseccomp or is not
  x86-64, **skips by name, saying which and how to supply it**. ``MIFORGE_REQUIRE_WSP_WORKER=1``
  turns every such skip into a FAILURE, so the CI job that builds the worker venv cannot go
  green by skipping what it exists to run (the shape of ``MIFORGE_REQUIRE_GPU``,
  ``tests/support/gpu_node.py``, and of ``*_REQUIRE_CROSS_REPO_CHECKS``).

The pieces that need only the standard library and numpy (the filter's denials, the supervisor
API) run under the backend's interpreter and skip only for libseccomp or the architecture.
"""

from __future__ import annotations

import ctypes.util
import os
import platform
import subprocess
import sys
from pathlib import Path
from typing import NoReturn

import pytest

#: Set to "1" where the worker's environment is promised: an absent piece then fails.
REQUIRE_ENV = "MIFORGE_REQUIRE_WSP_WORKER"
#: The worker's interpreter (a venv from requirements-wsp-worker.txt, or the image's python).
PYTHON_ENV = "MIFORGE_WSP_WORKER_PYTHON"
#: backend/src, where ``wsp_worker`` lives.
SRC = Path(__file__).resolve().parents[2] / "src"
#: The worker's own dependencies that the backend venv does not hold.
WORKER_ONLY_MODULES = ("weightspace", "anthropic", "openai")


def _current_test() -> str:
    return os.environ.get("PYTEST_CURRENT_TEST", "this test").rsplit(" (", 1)[0]


def absent(reason: str) -> NoReturn:
    """Skip the running test by name - or fail it under ``MIFORGE_REQUIRE_WSP_WORKER=1``."""
    what = f"WSP worker: {_current_test()} cannot run here: {reason}."
    if os.environ.get(REQUIRE_ENV) == "1":
        pytest.fail(
            f"{what} {REQUIRE_ENV}=1 says this machine provides it, so this is a failure.",
            pytrace=False,
        )
    pytest.skip(
        f"{what} Build a venv from backend/requirements-wsp-worker.txt and set {PYTHON_ENV} to "
        f"its python (CI sets {REQUIRE_ENV}=1 there)."
    )


def require_seccomp() -> None:
    """Return when this machine can load the child's filter (x86-64 and libseccomp)."""
    if platform.machine() != "x86_64":
        absent(f"the child's seccomp filter is x86-64 only; this is {platform.machine()}")
    if ctypes.util.find_library("seccomp") is None:
        absent("libseccomp is not installed (ctypes.util.find_library('seccomp') is None)")


def _imports(python: str, modules: tuple[str, ...]) -> str | None:
    """``None`` when ``python -I`` imports every module, else the failure text."""
    code = "import " + ", ".join(modules)
    r = subprocess.run([python, "-I", "-c", code], capture_output=True, text=True, timeout=120)
    return None if r.returncode == 0 else (r.stderr.strip().splitlines() or ["?"])[-1]


def worker_python(modules: tuple[str, ...] = WORKER_ONLY_MODULES) -> str:
    """The interpreter to run worker children under, holding ``modules``; else :func:`absent`."""
    named = os.environ.get(PYTHON_ENV)
    python = named or sys.executable
    if named and not Path(named).is_file():
        absent(f"{PYTHON_ENV}={named} does not exist")
    failure = _imports(python, modules)
    if failure is not None:
        where = f"{PYTHON_ENV}={named}" if named else f"this interpreter ({sys.executable})"
        absent(f"{where} cannot import {', '.join(modules)}: {failure}")
    return python
