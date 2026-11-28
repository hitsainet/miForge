"""The FR-018.14 probe (feature 018; FTASKS 6.5, 6.6). Self-contained: its SOURCE is sent as the
editor's ``python`` code and runs in the kernel grandchild; ``engine_cases.py`` also imports it
to run in the session process (the negative control, which must find the token).

It never holds the token: each form (raw, base64, JSON-escaped) arrives as an XOR-masked prefix,
the full form's sha256 and its length. A search finds the prefix, then hashes the full-length
window at that place, so the needle it carries can never match itself.

Checks: this process's environment, argv, reachable objects and readable memory; the parent's
``/proc/<ppid>/{environ, mem, fd, map_files}``; opening each practice file; opening a socket of
each kind. Each answers ``found``/``absent`` (a search), ``refused:<error>``, ``missing`` or
``opened``.
"""

from __future__ import annotations

from typing import Any


def probe(needles: list[list[Any]], practice_paths: list[str]) -> dict[str, str]:
    import gc
    import hashlib
    import os
    import socket
    import stat

    forms = [(bytes(b ^ key for b in masked), sha, length) for masked, key, sha, length in needles]

    def hit(blob: bytes) -> bool:
        for prefix, sha, length in forms:
            start = blob.find(prefix)
            while start != -1:
                if hashlib.sha256(blob[start : start + length]).hexdigest() == sha:
                    return True
                start = blob.find(prefix, start + 1)
        return False

    def search(name: str, read: Any) -> str:
        try:
            return "found" if hit(read()) else "absent"
        except OSError as exc:
            return f"refused:{type(exc).__name__}"

    out: dict[str, str] = {}

    def read(path: str) -> bytes:
        with open(path, "rb") as fh:
            return fh.read()

    out["environ"] = search(
        "environ",
        lambda: b"\0".join(f"{k}={v}".encode() for k, v in os.environ.items())
        + read("/proc/self/environ"),
    )
    out["cmdline"] = search("cmdline", lambda: read("/proc/self/cmdline"))

    def objects() -> bytes:
        parts: list[bytes] = []
        for obj in gc.get_objects():
            items: Any = ()
            if obj is parts:
                continue  # this scan's own growing list
            if isinstance(obj, dict):
                items = [*obj.keys(), *obj.values()]
            elif isinstance(obj, list | tuple):
                items = tuple(obj)
            # CPython untracks small containers holding only atoms (a ``{"token": ...}`` dict),
            # so gc never yields them: look one level into each container's containers.
            nested = [
                x
                for c in items
                if isinstance(c, dict | list | tuple) and c is not parts
                for x in ([*c.keys(), *c.values()] if isinstance(c, dict) else tuple(c))
            ]
            for item in [*items, *nested]:
                if isinstance(item, str):
                    parts.append(item.encode("utf-8", "surrogatepass"))
                elif isinstance(item, bytes | bytearray):
                    parts.append(bytes(item))
        return b"\n".join(parts)

    out["objects"] = search("objects", objects)

    def memory() -> bytes:
        found = []
        budget = 1 << 30
        with open("/proc/self/maps") as maps, open("/proc/self/mem", "rb", 0) as mem:
            for row in maps:
                fields = row.split()
                lo, hi = (int(x, 16) for x in fields[0].split("-"))
                path = fields[5] if len(fields) > 5 else ""
                if "r" not in fields[1] or path in ("[vvar]", "[vsyscall]", "[vvar_vclock]"):
                    continue
                step, overlap = 1 << 24, 4096
                at = lo
                while at < hi and budget > 0:
                    size = min(hi - at, step + overlap)
                    budget -= size
                    try:
                        mem.seek(at)
                        chunk = mem.read(size)
                    except (OSError, ValueError, OverflowError):
                        break
                    if hit(chunk):
                        found.append(path)
                        break
                    at += step
        return b"found" if found else b""

    def scanned() -> str:
        try:
            return "found" if memory() else "absent"
        except OSError as exc:
            return f"refused:{type(exc).__name__}"

    out["memory"] = scanned()
    ppid = os.getppid()
    out["parent_environ"] = search("parent_environ", lambda: read(f"/proc/{ppid}/environ"))
    try:
        with open(f"/proc/{ppid}/mem", "rb", 0):
            out["parent_mem"] = "opened"
    except OSError as exc:
        out["parent_mem"] = f"refused:{type(exc).__name__}"
    for where in ("fd", "map_files"):
        try:
            names = os.listdir(f"/proc/{ppid}/{where}")
        except OSError as exc:
            out[f"parent_{where}"] = f"refused:{type(exc).__name__}"
            continue
        opened = []
        for name in names:
            target = f"/proc/{ppid}/{where}/{name}"
            try:
                # Regular files only, never blocking: a pipe or socket read would hang the probe.
                fd = os.open(target, os.O_RDONLY | os.O_NONBLOCK)
            except OSError:
                continue
            try:
                if stat.S_ISREG(os.fstat(fd).st_mode) and hit(os.read(fd, 1 << 24)):
                    out["parent_fd_token"] = "found"
                opened.append(name)
            except OSError:
                pass
            finally:
                os.close(fd)
        out[f"parent_{where}"] = f"opened:{len(opened)}" if opened else "refused:none-opened"
    for i, path in enumerate(practice_paths):
        try:
            with open(path, "rb"):
                out[f"practice_{i}"] = "opened"
        except FileNotFoundError:
            out[f"practice_{i}"] = "missing"
        except OSError as exc:
            out[f"practice_{i}"] = f"refused:{type(exc).__name__}"
    for name, make in (
        ("socket_inet", lambda: socket.socket(socket.AF_INET, socket.SOCK_STREAM)),
        ("socket_unix", lambda: socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)),
        ("socketpair", lambda: socket.socketpair()),
    ):
        try:
            made = make()
            for s in made if isinstance(made, tuple) else (made,):
                s.close()
            out[name] = "opened"
        except OSError as exc:
            out[name] = f"refused:{type(exc).__name__}"
    return out
