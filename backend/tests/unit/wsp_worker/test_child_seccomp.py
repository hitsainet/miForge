"""The WSP child's seccomp filter, loaded for real in a throwaway child process (FR-016.10a;
FTASKS 4.4a; R3A-8).

Never in this process: a seccomp filter cannot be removed and covers every thread. Each test
starts ``seccomp_probe_child.py``, which installs ``wsp_worker.child_seccomp`` into itself, runs
every probe (all harmless even against a broken filter; see its docstring) and prints the
verdicts.

Two loads:

- **the production load** (``EPERM``): every denied call answers ``EPERM``, every same call aimed
  at the caller succeeds, an outbound ``connect`` and a numpy matmul work;
- **a discriminating load** with a distinctive errno (``ENOTRECOVERABLE``), so a denial by the
  filter cannot be mistaken for the kernel's own ``EPERM``: on a workstation, signalling the
  host's PID 1 (root's init) is ``EPERM`` with no filter at all, so only this load proves the
  ``kill``/``tgkill``/``rt_*`` rules and the truncated-pid case bite here. In the worker's
  container PID 1 is the supervisor, same uid, and the image job proves it live.

The ``anthropic``/``openai`` construction and the scripted weightspace session need the worker's
own interpreter (``tests/support/wsp_worker_env.py``): they skip by name without it, and fail
under ``MIFORGE_REQUIRE_WSP_WORKER=1``. The image test that ``find_library("seccomp")``
resolves in the worker image belongs to the image job (FTASKS 8.5).
"""

from __future__ import annotations

import errno
import json
import os
import socket
import subprocess
import sys
import threading
from pathlib import Path
from typing import Any

import pytest

from tests.support import wsp_worker_env

PROBE = Path(__file__).with_name("seccomp_probe_child.py")
DISCRIMINATING = errno.ENOTRECOVERABLE
DENIED = (
    "bind_inet",
    "bind_unix",
    "listen",
    "kill_pid1",
    "kill_pid1_truncated",
    "tgkill_pid1",
    "pidfd_open_pid1",
    "pidfd_open_pid1_truncated",
    "rt_sigqueueinfo_pid1",
    "rt_tgsigqueueinfo_pid1",
    "tkill_self",
    "pidfd_send_signal",
    "prlimit64_parent_thread",
    "prlimit64_sibling_thread",
    "sched_setaffinity_sibling",
    "sched_setscheduler_sibling",
    "sched_setparam_sibling",
    "sched_setattr_sibling",
    "setpriority_pgrp",
    "setpriority_user",
    "setpriority_process_sibling",
    "ioprio_set_pgrp",
    "ioprio_set_process_sibling",
    "ptrace_traceme",
    "process_vm_readv_self",
    "process_vm_writev_self",
    "io_uring_setup",
    "io_uring_enter",
    "io_uring_register",
)
ALLOWED = (
    "prlimit64_self",
    "sched_setaffinity_self",
    "sched_setscheduler_self",
    "sched_setparam_self",
    "sched_setattr_self",
    "setpriority_self",
    "ioprio_set_self",
    "pidfd_open_self",
    "kill_self_signal0",
    "connect_outbound",
    "numpy_matmul",
)


def _probe(
    python: str, *, deny_errno: int = errno.EPERM, extras: bool = False, install: bool = True
) -> dict[str, Any]:
    """Run the probe child; serve one outbound ``connect`` for it meanwhile."""
    server = socket.create_server(("127.0.0.1", 0))
    server.settimeout(60)
    got: list[bytes] = []

    def accept() -> None:
        try:
            conn, _ = server.accept()
            with conn:
                got.append(conn.recv(16))
        except OSError:
            pass

    t = threading.Thread(target=accept, daemon=True)
    t.start()
    config = {
        "parent_tid": threading.get_native_id(),
        "connect_port": server.getsockname()[1],
        "deny_errno": deny_errno,
        "extras": extras,
        "install": install,
    }
    try:
        r = subprocess.run(
            [python, "-I", str(PROBE), str(wsp_worker_env.SRC), json.dumps(config)],
            capture_output=True,
            text=True,
            timeout=180,
            start_new_session=True,
            env={
                "PATH": os.environ.get("PATH", "/usr/bin:/bin"),
                "OMP_NUM_THREADS": "1",
                "OPENBLAS_NUM_THREADS": "1",
                "MKL_NUM_THREADS": "1",
            },
        )
    finally:
        t.join(timeout=10)
        server.close()
    assert r.returncode == 0, f"probe child exited {r.returncode}: {r.stderr[-3000:]}"
    out: dict[str, Any] = json.loads(r.stdout.strip().splitlines()[-1])
    out["connected"] = got
    return out


@pytest.fixture(scope="module")
def production() -> dict[str, Any]:
    wsp_worker_env.require_seccomp()
    return _probe(sys.executable)


@pytest.fixture(scope="module")
def discriminating() -> dict[str, Any]:
    wsp_worker_env.require_seccomp()
    return _probe(sys.executable, deny_errno=DISCRIMINATING)


def test_the_filter_is_installed_on_top_of_the_parents(production: dict[str, Any]) -> None:
    before, after = production["status_before"], production["status_after"]
    assert after["NoNewPrivs"] == "1"
    assert after["Seccomp"] == "2"
    assert int(after["Seccomp_filters"]) == int(before["Seccomp_filters"]) + 1


@pytest.mark.parametrize("probe", DENIED)
def test_each_denied_call_returns_eperm(production: dict[str, Any], probe: str) -> None:
    assert production["denied"][probe] == "EPERM", production["denied"]


@pytest.mark.parametrize("probe", DENIED)
def test_each_denial_is_the_filters_own(discriminating: dict[str, Any], probe: str) -> None:
    """With a distinctive errno, every denial comes from the filter, not the kernel's checks."""
    assert discriminating["denied"][probe] == errno.errorcode[DISCRIMINATING], discriminating[
        "denied"
    ]


@pytest.mark.parametrize("probe", ALLOWED)
def test_the_same_calls_on_the_caller_succeed(production: dict[str, Any], probe: str) -> None:
    assert production["allowed"][probe] == "ok", production["allowed"]


def test_outbound_connect_reaches_the_listener(production: dict[str, Any]) -> None:
    assert production["connected"] == [b"hello"]


def test_without_the_filter_the_probes_are_not_denied() -> None:
    """The probes discriminate: unfiltered, the calls the filter must refuse go through (the
    pid-1 signals excepted, which the host's own permission check refuses on a workstation)."""
    wsp_worker_env.require_seccomp()
    out = _probe(sys.executable, install=False)
    kernel_refuses_here = {
        "kill_pid1",
        "kill_pid1_truncated",
        "tgkill_pid1",
        "rt_sigqueueinfo_pid1",
        "rt_tgsigqueueinfo_pid1",
    }
    unfiltered = {
        k: v
        for k, v in out["denied"].items()
        if k not in kernel_refuses_here | {"setpriority_user"}
    }
    assert all(v != "EPERM" for v in unfiltered.values()), unfiltered
    assert out["denied"]["setpriority_user"] == "not_run_rule_absent"


def test_rules_compare_pids_under_the_32_bit_mask() -> None:
    from wsp_worker import child_seccomp as cs

    pid_rules = [r for r in cs.rules() if r.syscall in cs.PID1_DENIED]
    assert {r.syscall for r in pid_rules} == set(cs.PID1_DENIED)
    for rule in pid_rules:
        assert rule.args == ((0, cs.SCMP_CMP_MASKED_EQ, 0xFFFFFFFF, 1),)


def test_the_worker_libraries_run_under_the_filter() -> None:
    """R3A-8: client construction and a scripted weightspace session under the real filter."""
    wsp_worker_env.require_seccomp()
    python = wsp_worker_env.worker_python()
    out = _probe(python, extras=True)
    assert out["allowed"]["anthropic_openai_clients"] == "ok", out["allowed"]
    assert out["allowed"]["weightspace_scripted_session"] == "ok", out["allowed"]
    assert out["allowed"]["numpy_matmul"] == "ok"
