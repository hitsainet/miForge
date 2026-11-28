"""The throwaway child of ``test_child_seccomp.py`` (never collected: no ``test_`` prefix).

Run as ``python -I seccomp_probe_child.py <backend/src> <config json>``. It installs the WSP
child's filter into ITSELF (never into the test process: a filter cannot be removed), runs every
probe, and prints one JSON object: probe name -> ``"ok"`` or the errno's name.

Every probe is harmless even against a filter a control has broken: signals are signal 0 (an
existence check), scheduling calls re-apply the target's current value to a thread of this
child, ``prlimit64`` only reads, and the ``PRIO_USER`` probe (which, unfiltered, would renice
every process of the uid) runs only after the ``PRIO_PGRP`` probe proved the same rule denies.
"""

from __future__ import annotations

import ctypes
import errno
import json
import os
import socket
import sys
import threading

SRC, CONFIG = sys.argv[1], json.loads(sys.argv[2])
sys.path.insert(0, SRC)

from wsp_worker import child_seccomp  # noqa: E402

NR = {
    "bind": 49,
    "listen": 50,
    "kill": 62,
    "ptrace": 101,
    "rt_sigqueueinfo": 129,
    "getpriority": 140,
    "setpriority": 141,
    "sched_setparam": 142,
    "sched_setscheduler": 144,
    "tkill": 200,
    "sched_setaffinity": 203,
    "sched_getaffinity": 204,
    "tgkill": 234,
    "ioprio_set": 251,
    "ioprio_get": 252,
    "rt_tgsigqueueinfo": 297,
    "prlimit64": 302,
    "process_vm_readv": 310,
    "process_vm_writev": 311,
    "sched_setattr": 314,
    "pidfd_send_signal": 424,
    "io_uring_setup": 425,
    "io_uring_enter": 426,
    "io_uring_register": 427,
    "pidfd_open": 434,
}
PRIO_PROCESS, PRIO_PGRP, PRIO_USER = 0, 1, 2
IOPRIO_WHO_PROCESS, IOPRIO_WHO_PGRP = 1, 2
RLIMIT_NOFILE = 7

libc = ctypes.CDLL(None, use_errno=True)
libc.syscall.restype = ctypes.c_long


def sc(name: str, *args: object) -> int:
    """The raw system call: its non-negative result, or ``-errno``."""
    conv = [a if not isinstance(a, int) else ctypes.c_ulong(a & 0xFFFFFFFFFFFFFFFF) for a in args]
    ctypes.set_errno(0)
    r = libc.syscall(ctypes.c_long(NR[name]), *conv)
    return int(r) if r >= 0 else -ctypes.get_errno()


def verdict(r: int) -> str:
    return "ok" if r >= 0 else errno.errorcode.get(-r, str(-r))


def py(fn) -> str:  # type: ignore[no-untyped-def]
    try:
        fn()
        return "ok"
    except OSError as exc:
        return errno.errorcode.get(exc.errno or 0, str(exc.errno))
    except Exception as exc:  # the probe reports, the test decides
        return f"{type(exc).__name__}: {exc}"[:300]


# A sibling thread of this child: the "anything but the caller" target that is still ours.
ready, stop = threading.Event(), threading.Event()
sibling: dict[str, int] = {}


def _park() -> None:
    sibling["tid"] = threading.get_native_id()
    ready.set()
    stop.wait()


threading.Thread(target=_park, daemon=True).start()
ready.wait()
tid = sibling["tid"]
parent_tid = int(CONFIG["parent_tid"])
me = os.getpid()

before = child_seccomp.status_fields()
if CONFIG.get("install", True):
    child_seccomp.install(deny_errno=int(CONFIG.get("deny_errno", errno.EPERM)))
after = child_seccomp.status_fields()

out: dict[str, object] = {"status_before": before, "status_after": after}
mask = ctypes.create_string_buffer(128)
sc("sched_getaffinity", 0, 128, mask)
nice = sc("getpriority", PRIO_PROCESS, 0)  # the raw call answers 20 - nice
nice_now = 20 - nice
param = ctypes.c_int(0)
siginfo = ctypes.create_string_buffer(128)
old = ctypes.create_string_buffer(16)


class SchedAttr(ctypes.Structure):
    _fields_ = [
        ("size", ctypes.c_uint32),
        ("policy", ctypes.c_uint32),
        ("flags", ctypes.c_uint64),
        ("nice", ctypes.c_int32),
        ("priority", ctypes.c_uint32),
        ("runtime", ctypes.c_uint64),
        ("deadline", ctypes.c_uint64),
        ("period", ctypes.c_uint64),
    ]


attr = SchedAttr(48, 0, 0, nice_now, 0, 0, 0, 0)
ioprio = sc("ioprio_get", IOPRIO_WHO_PROCESS, 0)
ioprio = max(ioprio, 0)

# --- denied ----------------------------------------------------------------------------
denied: dict[str, str] = {}
denied["bind_inet"] = py(lambda: socket.socket().bind(("127.0.0.1", 0)))
denied["bind_unix"] = py(lambda: socket.socket(socket.AF_UNIX).bind(f"\0wsp-probe-{me}"))
denied["listen"] = py(lambda: socket.socket().listen(1))
denied["kill_pid1"] = verdict(sc("kill", 1, 0))
denied["kill_pid1_truncated"] = verdict(sc("kill", 0x100000001, 0))
denied["tgkill_pid1"] = verdict(sc("tgkill", 1, 1, 0))
denied["pidfd_open_pid1"] = verdict(sc("pidfd_open", 1, 0))
denied["pidfd_open_pid1_truncated"] = verdict(sc("pidfd_open", 0x100000001, 0))
denied["rt_sigqueueinfo_pid1"] = verdict(sc("rt_sigqueueinfo", 1, 0, siginfo))
denied["rt_tgsigqueueinfo_pid1"] = verdict(sc("rt_tgsigqueueinfo", 1, 1, 0, siginfo))
denied["tkill_self"] = verdict(sc("tkill", tid, 0))
own_pidfd = sc("pidfd_open", me, 0)
denied["pidfd_send_signal"] = verdict(sc("pidfd_send_signal", max(own_pidfd, 0), 0, 0, 0))
denied["prlimit64_parent_thread"] = verdict(sc("prlimit64", parent_tid, RLIMIT_NOFILE, 0, old))
denied["prlimit64_sibling_thread"] = verdict(sc("prlimit64", tid, RLIMIT_NOFILE, 0, old))
denied["sched_setaffinity_sibling"] = verdict(sc("sched_setaffinity", tid, 128, mask))
denied["sched_setscheduler_sibling"] = verdict(
    sc("sched_setscheduler", tid, 0, ctypes.byref(param))
)
denied["sched_setparam_sibling"] = verdict(sc("sched_setparam", tid, ctypes.byref(param)))
denied["sched_setattr_sibling"] = verdict(sc("sched_setattr", tid, ctypes.byref(attr), 0))
denied["setpriority_pgrp"] = verdict(sc("setpriority", PRIO_PGRP, 0, nice_now))
denied["setpriority_process_sibling"] = verdict(sc("setpriority", PRIO_PROCESS, tid, nice_now))
denied["ioprio_set_pgrp"] = verdict(sc("ioprio_set", IOPRIO_WHO_PGRP, 0, ioprio))
denied["ioprio_set_process_sibling"] = verdict(sc("ioprio_set", IOPRIO_WHO_PROCESS, tid, ioprio))
denied["ptrace_traceme"] = verdict(sc("ptrace", 0, 0, 0, 0))
empty = ctypes.create_string_buffer(16)
denied["process_vm_readv_self"] = verdict(sc("process_vm_readv", me, empty, 0, empty, 0, 0))
denied["process_vm_writev_self"] = verdict(sc("process_vm_writev", me, empty, 0, empty, 0, 0))
uring_params = ctypes.create_string_buffer(120)
denied["io_uring_setup"] = verdict(sc("io_uring_setup", 1, uring_params))
denied["io_uring_enter"] = verdict(sc("io_uring_enter", 0xFFFFFFFF, 0, 0, 0, 0, 0))
denied["io_uring_register"] = verdict(sc("io_uring_register", 0xFFFFFFFF, 0, 0, 0))
# PRIO_USER names every process of the uid: only after PRIO_PGRP proved the arg0 rule denies.
if denied["setpriority_pgrp"] == errno.errorcode[int(CONFIG.get("deny_errno", errno.EPERM))]:
    denied["setpriority_user"] = verdict(sc("setpriority", PRIO_USER, 0, nice_now))
else:
    denied["setpriority_user"] = "not_run_rule_absent"
out["denied"] = denied

# --- allowed: the same calls aimed at the caller --------------------------------------
allowed: dict[str, str] = {}
allowed["prlimit64_self"] = verdict(sc("prlimit64", 0, RLIMIT_NOFILE, 0, old))
allowed["sched_setaffinity_self"] = verdict(sc("sched_setaffinity", 0, 128, mask))
allowed["sched_setscheduler_self"] = verdict(sc("sched_setscheduler", 0, 0, ctypes.byref(param)))
allowed["sched_setparam_self"] = verdict(sc("sched_setparam", 0, ctypes.byref(param)))
allowed["sched_setattr_self"] = verdict(sc("sched_setattr", 0, ctypes.byref(attr), 0))
allowed["setpriority_self"] = verdict(sc("setpriority", PRIO_PROCESS, 0, nice_now))
allowed["ioprio_set_self"] = verdict(sc("ioprio_set", IOPRIO_WHO_PROCESS, 0, ioprio))
allowed["pidfd_open_self"] = verdict(own_pidfd)
allowed["kill_self_signal0"] = verdict(sc("kill", me, 0))


def _connect() -> None:
    with socket.create_connection(("127.0.0.1", int(CONFIG["connect_port"])), timeout=5) as s:
        s.sendall(b"hello")


allowed["connect_outbound"] = py(_connect)


def _matmul() -> None:
    import numpy as np

    a = np.arange(64, dtype=np.float64).reshape(8, 8)
    if float((a @ a).sum()) != float(np.einsum("ij,jk->", a, a)):
        raise OSError(errno.EIO, "matmul disagrees")


allowed["numpy_matmul"] = py(_matmul)

if CONFIG.get("extras"):

    def _clients() -> None:
        # The engine's own HTTP client (httpx2, with truststore's SSL context), both kinds, as a
        # session builds it. Until 018-P4 this case built the anthropic and openai SDK clients and
        # ran a scripted weightspace session; both left the worker with weightspace (FR-018.18).
        from wsp_worker.miwsp_engine.client import GatewayClient

        gateway = {"ip": "127.0.0.1", "port": 9, "token": "probe"}
        for kind in ("anthropic", "openai_compatible"):
            client = GatewayClient(kind, "probe", gateway)
            if not client.headers():
                raise OSError(errno.EIO, f"{kind} client built no headers")
            client.close()

    allowed["engine_http_clients"] = py(_clients)

out["allowed"] = allowed
stop.set()
sys.stdout.write(json.dumps(out) + "\n")
sys.stdout.flush()
