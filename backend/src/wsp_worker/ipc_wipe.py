"""SysV and POSIX IPC removal for the supervisor, **as PID 1 only** (FR-016.10 step 5; R1B-M1,
R3A-1).

A session's child can leave a SysV shared-memory segment, semaphore set or message queue, or a
POSIX message queue, behind for the next session to find: they live in the pod's IPC namespace,
not in any directory the wipe clears. As PID 1 the supervisor removes every one this uid owns and
then verifies none remain.

**Outside PID 1** (``LocalProcessLauncher``: dev and tests) these objects belong to the developer,
and removing "every one this uid owns" would destroy theirs. So :func:`wipe_all` refuses unless
``os.getpid() == 1`` - a second gate behind the supervisor's own - and only the primitives
(:func:`list_sysv`, :func:`remove_sysv`) run in tests, on objects the test created.
"""

from __future__ import annotations

import ctypes
import os
from pathlib import Path

IPC_RMID = 0
SYSVIPC = Path("/proc/sysvipc")
MQUEUE = Path("/dev/mqueue")
KINDS = ("shm", "sem", "msg")
_ID_COLUMN = {"shm": "shmid", "sem": "semid", "msg": "msqid"}

_libc = ctypes.CDLL(None, use_errno=True)


class IpcWipeRefusedError(RuntimeError):
    """The wipe was asked for outside PID 1, or /dev/mqueue is not mounted."""


def list_sysv(kind: str, uid: int | None = None) -> list[int]:
    """The ids of every SysV object of ``kind`` owned by ``uid`` (default: this process's)."""
    if kind not in KINDS:
        raise ValueError(kind)
    uid = os.getuid() if uid is None else uid
    lines = (SYSVIPC / kind).read_text().splitlines()
    header = lines[0].split()
    id_col, uid_col = header.index(_ID_COLUMN[kind]), header.index("uid")
    ids: list[int] = []
    for line in lines[1:]:
        cols = line.split()
        if len(cols) > max(id_col, uid_col) and int(cols[uid_col]) == uid:
            ids.append(int(cols[id_col]))
    return ids


def remove_sysv(kind: str, ident: int) -> None:
    """``IPC_RMID`` one SysV object by id (``shmctl``/``semctl``/``msgctl``)."""
    if kind == "shm":
        rc = _libc.shmctl(ctypes.c_int(ident), IPC_RMID, None)
    elif kind == "sem":
        rc = _libc.semctl(ctypes.c_int(ident), 0, IPC_RMID)
    elif kind == "msg":
        rc = _libc.msgctl(ctypes.c_int(ident), IPC_RMID, None)
    else:
        raise ValueError(kind)
    if rc != 0:
        err = ctypes.get_errno()
        raise OSError(err, f"IPC_RMID on {kind} {ident}: {os.strerror(err)}")


def list_mqueues() -> list[str]:
    return sorted(os.listdir(MQUEUE))


def mqueue_mounted() -> bool:
    try:
        with open("/proc/self/mounts", encoding="ascii", errors="replace") as fh:
            return any(line.split()[1:3] == [str(MQUEUE), "mqueue"] for line in fh)
    except OSError:
        return False


def remaining() -> list[str]:
    """What this uid still owns: ``kind:id`` for SysV, ``mq:name`` for POSIX queues."""
    left = [f"{kind}:{i}" for kind in KINDS for i in list_sysv(kind)]
    return left + [f"mq:{name}" for name in list_mqueues()]


def wipe_all() -> list[str]:
    """As PID 1: remove every SysV object of this uid and every POSIX queue; return what remains.

    Refuses outside PID 1 and when ``/dev/mqueue`` is not mounted (it cannot see queues it cannot
    list; R2A-7)."""
    if os.getpid() != 1:
        raise IpcWipeRefusedError("the IPC wipe runs only as PID 1")
    if not mqueue_mounted():
        raise IpcWipeRefusedError("/dev/mqueue is not mounted")
    for kind in KINDS:
        for ident in list_sysv(kind):
            try:
                remove_sysv(kind, ident)
            except OSError:
                continue  # verified below
    unlink = _libc.mq_unlink
    unlink.argtypes = [ctypes.c_char_p]
    for name in list_mqueues():
        unlink(("/" + name).encode())
    return remaining()
