"""The WSP child's own seccomp filter (FR-016.10a; FTDD §8.1; P59, PF-56, R3A-8).

The child (an editor session, a program or a replay) shares the supervisor's uid and network
namespace, so the supervisor's API and its process are reachable from it. This filter is what
stops the child taking either over. It is installed by the child itself, **before** it reads its
spec or the gateway token from stdin (``wsp_worker/__main__.py``), and the supervisor checks it
landed (``NoNewPrivs: 1`` and a ``Seccomp_filters`` count one above its own) before it writes
anything to the child. It stacks on the pod's ``RuntimeDefault`` profile; the kernel enforces the
stricter of the two for every call.

Default action ``ALLOW``; x86-64 only (any other ABI kills the process); ``PR_SET_NO_NEW_PRIVS``
and TSYNC (every thread). Denied with ``EPERM``:

- ``bind`` and ``listen``: the child needs only outbound ``connect`` (to the gateway);
- signals to PID 1: ``kill``, ``tgkill``, ``pidfd_open``, ``rt_sigqueueinfo`` and
  ``rt_tgsigqueueinfo`` with arg0 == 1; ``tkill`` and ``pidfd_send_signal`` always;
- scheduling and limits on anything but the caller: ``prlimit64``, ``sched_setaffinity``,
  ``sched_setscheduler``, ``sched_setparam`` and ``sched_setattr`` unless arg0 == 0;
  ``setpriority`` unless ``which == PRIO_PROCESS`` and ``who == 0``; ``ioprio_set`` unless
  ``which == IOPRIO_WHO_PROCESS`` and ``who == 0`` (one ``SCMP_CMP_NE`` deny rule per argument,
  because the conditions inside one rule are joined by AND);
- ``ptrace``, ``process_vm_readv``, ``process_vm_writev``;
- ``io_uring_setup``, ``io_uring_enter``, ``io_uring_register`` (OVN-wsp-forge-13: io_uring's
  ``IORING_OP_BIND``/``IORING_OP_LISTEN``/``IORING_OP_SOCKET`` would bind without ``bind``; the
  pod's RuntimeDefault profile is not relied on to refuse it).

**Pids are compared under the mask ``0xFFFFFFFF``** (``SCMP_CMP_MASKED_EQ``). The kernel truncates
a 64-bit argument to the 32-bit ``pid_t``, so a plain ``SCMP_CMP_EQ`` against 1 would let
``syscall(SYS_kill, 0x100000001, sig)`` through to PID 1. The "unless arg0 == 0" rules compare all
64 bits with ``SCMP_CMP_NE``: a value whose low half is 0 but whose high half is not is refused
too, which only ever refuses more.

The mechanism is miServer's (``miserver/rules/runner_seccomp.py``): libseccomp through a few
``ctypes`` calls, no Python dependency. Anything that fails raises
:class:`SeccompUnavailableError`, and the child exits before reading stdin, so the supervisor's
filter check refuses the session (``worker_filter_missing``): it fails closed.

**The grandchild's layer** (feature 018, FR-018.14; FTDD §6.5): ``miwsp_engine`` runs the editor's
Python in a grandchild, a fresh interpreter that installs this filter **plus**
:data:`GRANDCHILD_RULES` (``socket``, ``socketpair``, ``connect``) as its own stacked layer before
it reads anything, so editor code can open no socket at all, not even to the gateway. The child's
own rule set (:func:`rules`) is unchanged by it.

⚠ Never call :func:`install` in a supervisor, backend or test process: a seccomp filter cannot be
removed and applies to every thread. Tests install it in a throwaway child process only.
"""

from __future__ import annotations

import ctypes
import ctypes.util
import errno
from dataclasses import dataclass
from typing import Any

#: libseccomp's ABI (seccomp.h; stable across the 2.x library).
SCMP_ACT_ALLOW = 0x7FFF0000
SCMP_ACT_KILL_PROCESS = 0x80000000
SCMP_ARCH_X86_64 = 0xC000003E  # AUDIT_ARCH_X86_64
SCMP_FLTATR_ACT_BADARCH = 2
SCMP_FLTATR_CTL_NNP = 3
SCMP_FLTATR_CTL_TSYNC = 4
SCMP_CMP_NE = 1
SCMP_CMP_EQ = 4
SCMP_CMP_MASKED_EQ = 7
NR_SCMP_ERROR = -1

PR_SET_NO_NEW_PRIVS = 38
PRIO_PROCESS = 0
IOPRIO_WHO_PROCESS = 1
#: The kernel's ``pid_t`` is 32 bits; every pid comparison masks to it (R3A-8).
PID_MASK = 0xFFFFFFFF
#: PID 1 in the worker's namespace: the supervisor.
SUPERVISOR_PID = 1


def act_errno(code: int) -> int:
    return 0x00050000 | (code & 0x0000FFFF)


#: Refused outright.
DENIED: tuple[str, ...] = (
    "bind",
    "listen",
    "tkill",
    "pidfd_send_signal",
    "ptrace",
    "process_vm_readv",
    "process_vm_writev",
    "io_uring_setup",
    "io_uring_enter",
    "io_uring_register",
)
#: The grandchild's extra denials (feature 018): no socket of any kind, no outbound connect.
GRANDCHILD_RULES: tuple[str, ...] = ("socket", "socketpair", "connect")
#: Refused when arg0, as a pid, is 1 (the supervisor).
PID1_DENIED: tuple[str, ...] = (
    "kill",
    "tgkill",
    "pidfd_open",
    "rt_sigqueueinfo",
    "rt_tgsigqueueinfo",
)
#: Refused unless arg0 == 0 (the caller itself).
SELF_ONLY: tuple[str, ...] = (
    "prlimit64",
    "sched_setaffinity",
    "sched_setscheduler",
    "sched_setparam",
    "sched_setattr",
)
#: Refused unless ``(arg0, arg1) == (which, 0)``: (syscall, the one allowed ``which``).
WHO_SELF_ONLY: tuple[tuple[str, int], ...] = (
    ("setpriority", PRIO_PROCESS),
    ("ioprio_set", IOPRIO_WHO_PROCESS),
)


class SeccompUnavailableError(Exception):
    """The filter could not be built or installed. ``code`` is a short fixed string."""

    def __init__(self, code: str) -> None:
        super().__init__(code)
        self.code = code


class _ArgCmp(ctypes.Structure):
    _fields_ = [
        ("arg", ctypes.c_uint),
        ("op", ctypes.c_int),
        ("datum_a", ctypes.c_uint64),
        ("datum_b", ctypes.c_uint64),
    ]


@dataclass(frozen=True)
class Rule:
    """One libseccomp rule: ``action`` on ``syscall`` when every comparison holds."""

    syscall: str
    action: int
    #: ``(arg index, op, datum_a, datum_b)``.
    args: tuple[tuple[int, int, int, int], ...] = ()


def rules(deny_errno: int = errno.EPERM) -> tuple[Rule, ...]:
    """The whole filter, as data (what :func:`install` loads and what the tests compare).

    ``deny_errno`` is ``EPERM`` everywhere but one test, which loads the filter with a distinctive
    errno so a denial by the filter cannot be confused with the kernel's own ``EPERM`` (a
    non-root process signalling the host's PID 1 gets ``EPERM`` with no filter at all)."""
    eperm = act_errno(deny_errno)
    out: list[Rule] = [Rule(name, eperm) for name in DENIED]
    out += [
        Rule(name, eperm, ((0, SCMP_CMP_MASKED_EQ, PID_MASK, SUPERVISOR_PID),))
        for name in PID1_DENIED
    ]
    out += [Rule(name, eperm, ((0, SCMP_CMP_NE, 0, 0),)) for name in SELF_ONLY]
    for name, which in WHO_SELF_ONLY:
        out.append(Rule(name, eperm, ((0, SCMP_CMP_NE, which, 0),)))
        out.append(Rule(name, eperm, ((1, SCMP_CMP_NE, 0, 0),)))
    return tuple(out)


def _library() -> Any:
    name = ctypes.util.find_library("seccomp") or "libseccomp.so.2"
    try:
        lib = ctypes.CDLL(name, use_errno=True)
    except OSError as exc:
        raise SeccompUnavailableError("libseccomp_missing") from exc
    try:
        return _bind(lib)
    except AttributeError as exc:
        raise SeccompUnavailableError("libseccomp_symbol_missing") from exc


def _bind(lib: Any) -> Any:
    lib.seccomp_init.restype = ctypes.c_void_p
    lib.seccomp_init.argtypes = [ctypes.c_uint32]
    lib.seccomp_release.argtypes = [ctypes.c_void_p]
    lib.seccomp_release.restype = None
    lib.seccomp_arch_native.restype = ctypes.c_uint32
    lib.seccomp_attr_set.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.c_uint32]
    lib.seccomp_syscall_resolve_name.argtypes = [ctypes.c_char_p]
    lib.seccomp_syscall_resolve_name.restype = ctypes.c_int
    lib.seccomp_rule_add_exact_array.argtypes = [
        ctypes.c_void_p,
        ctypes.c_uint32,
        ctypes.c_int,
        ctypes.c_uint,
        ctypes.POINTER(_ArgCmp),
    ]
    lib.seccomp_load.argtypes = [ctypes.c_void_p]
    return lib


def _check(rc: int, code: str) -> None:
    if rc < 0:
        raise SeccompUnavailableError(code)


def _build(lib: Any, deny_errno: int = errno.EPERM, extra_denied: tuple[str, ...] = ()) -> Any:
    """A filter context holding every rule, plus ``extra_denied`` refused outright (not loaded).
    The caller releases it."""
    if lib.seccomp_arch_native() != SCMP_ARCH_X86_64:
        raise SeccompUnavailableError("not_x86_64")
    ctx = lib.seccomp_init(SCMP_ACT_ALLOW)
    if not ctx:
        raise SeccompUnavailableError("seccomp_init_failed")
    try:
        _check(
            lib.seccomp_attr_set(ctx, SCMP_FLTATR_ACT_BADARCH, SCMP_ACT_KILL_PROCESS),
            "attr_badarch_failed",
        )
        _check(lib.seccomp_attr_set(ctx, SCMP_FLTATR_CTL_NNP, 1), "attr_nnp_failed")
        _check(lib.seccomp_attr_set(ctx, SCMP_FLTATR_CTL_TSYNC, 1), "attr_tsync_failed")
        extra = tuple(Rule(name, act_errno(deny_errno)) for name in extra_denied)
        for rule in rules(deny_errno) + extra:
            nr = lib.seccomp_syscall_resolve_name(rule.syscall.encode("ascii"))
            if nr == NR_SCMP_ERROR:
                raise SeccompUnavailableError(f"unknown_syscall_{rule.syscall}")
            args = (_ArgCmp * max(1, len(rule.args)))(*[_ArgCmp(*a) for a in rule.args])
            _check(
                lib.seccomp_rule_add_exact_array(ctx, rule.action, nr, len(rule.args), args),
                f"rule_add_failed_{rule.syscall}",
            )
    except BaseException:
        lib.seccomp_release(ctx)
        raise
    return ctx


def check() -> None:
    """Build the whole filter WITHOUT loading it (an image build runs this)."""
    lib = _library()
    lib.seccomp_release(_build(lib))


def install(
    library: Any | None = None,
    *,
    deny_errno: int = errno.EPERM,
    extra_denied: tuple[str, ...] = (),
) -> None:
    """Set ``PR_SET_NO_NEW_PRIVS``, then build and LOAD the filter into this process, every
    thread (irreversible). Raises :class:`SeccompUnavailableError` when anything fails; the kernel
    applies a filter whole or not at all. ``extra_denied`` adds calls refused outright (the
    grandchild passes :data:`GRANDCHILD_RULES`)."""
    libc = ctypes.CDLL(None, use_errno=True)
    if libc.prctl(PR_SET_NO_NEW_PRIVS, 1, 0, 0, 0) != 0:
        raise SeccompUnavailableError("no_new_privs_failed")
    lib = library if library is not None else _library()
    ctx = _build(lib, deny_errno, extra_denied)
    try:
        _check(lib.seccomp_load(ctx), "seccomp_load_failed")
    finally:
        lib.seccomp_release(ctx)


def status_fields(pid: int | str = "self") -> dict[str, str]:
    """``NoNewPrivs``, ``Seccomp`` and ``Seccomp_filters`` from ``/proc/<pid>/status`` (a field
    the kernel does not print is absent: ``Seccomp_filters`` needs Linux 5.9)."""
    out: dict[str, str] = {}
    with open(f"/proc/{pid}/status", encoding="ascii", errors="replace") as fh:
        for line in fh:
            key, _, value = line.partition(":")
            if key in ("NoNewPrivs", "Seccomp", "Seccomp_filters"):
                out[key] = value.strip()
    return out
