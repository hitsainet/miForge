"""The kernel grandchild's main: the editor's Python, in its own fresh interpreter (feature 018;
FR-018.14; FTDD §6.5; FTID §3.4; F18-D10).

Its first acts, **before it reads anything**:

1. restore a fresh interpreter's signal dispositions;
2. install the child's seccomp filter plus ``child_seccomp.GRANDCHILD_RULES`` (``socket``,
   ``socketpair``, ``connect``) as its own stacked layer; a failure exits 71 with no ready byte;
3. move the protocol pipes off fds 0 and 1 (editor code that writes to fd 1 reaches
   ``/dev/null``, never the protocol);
4. write the ready byte ``K``.

Then one JSON object per line, in and out:

- ``{"op": "init", "dir", "names", "edit", "notes", "rebind"}``: map each tensor ``dir/t<i>.npy``
  as ``R`` (``mmap_mode="r"``: a read-only map, so setting ``writeable`` raises) and, in edit
  modes, as ``W`` (``mmap_mode="c"``: copy-on-write, so nothing reaches the file). With
  ``rebind`` (2026-11-27, run ``wsp_ba149c57f611``), every ``exec`` ends by binding the reserved
  names (:data:`RESERVED`) back to these objects when the code assigned or deleted one, and its
  output says so (:data:`RESERVED_NOTE`): that run's ``W = np.array(R['weight'])`` left ``W`` a
  plain array, so every later ``W['weight'][:] = ...`` raised an IndexError and no Python write
  could land for the rest of the session. A list assigned to ``notes`` becomes the notes' items.
  Without it (a replay of a run recorded before, by its transcript header) nothing is re-bound,
  as before;
- ``{"op": "exec", "code"}`` answers
  ``{"ok", "output", "deltas": [{"name", "idx", "values"}], "notes"}``: run the code, print
  the last expression, and report every entry of ``W`` whose BITS differ from ``R`` (the store's
  own comparison);
- ``{"op": "sync", "names"}``: re-map those tensors after the parent rewrote their files (an
  accepted write, or a refused one, which reverts ``W``).

This process is untrusted: the parent checks everything it says (``kernel.py``) and applies every
delta through the change limits. It holds no token, no socket and no practice data.
"""

from __future__ import annotations

import ast
import contextlib
import io
import json
import os
import sys
import traceback
from collections.abc import Iterator, MutableMapping
from pathlib import Path
from typing import Any

from wsp_worker.miwsp_engine import prompts

MAX_OUTPUT = 12_000
EXIT_FILTER_FAILED = 71
MAX_LINE = 64 * 1024 * 1024
#: The names the kernel binds and an ``exec`` cannot take away (with ``rebind``), in this order.
RESERVED = ("R", "W", "notes", "np")
#: The line an ``exec`` output ends with when it assigned or deleted a reserved name: an editor
#: text, so it lives (and is pinned) in ``prompts``.
RESERVED_NOTE = prompts.RESERVED_REBOUND
_MISSING = object()


class _Maps(MutableMapping[str, Any]):
    """``R`` or ``W``: a fixed set of names. ``W[name] = value`` writes INTO the map."""

    def __init__(self, arrays: dict[str, Any], *, writable: bool) -> None:
        self._arrays, self._writable = arrays, writable

    def __getitem__(self, name: str) -> Any:
        return self._arrays[name]

    def __setitem__(self, name: str, value: Any) -> None:
        if not self._writable:
            raise TypeError("R is read-only; write through W")
        if name not in self._arrays:
            raise KeyError(f"unknown tensor {name!r}")
        import numpy as np

        target = self._arrays[name]
        target[...] = np.broadcast_to(np.asarray(value, dtype=target.dtype), target.shape)

    def __delitem__(self, name: str) -> None:
        raise TypeError("tensors cannot be removed")

    def __iter__(self) -> Iterator[str]:
        return iter(self._arrays)

    def __len__(self) -> int:
        return len(self._arrays)

    def __repr__(self) -> str:
        return f"{'W' if self._writable else 'R'}({sorted(self._arrays)})"


class Kernel:
    def __init__(self) -> None:
        import numpy as np

        self.np = np
        self.dir = Path(".")
        self.names: list[str] = []
        self.r: dict[str, Any] = {}
        self.w: dict[str, Any] = {}
        self.edit = False
        self.notes: list[Any] = []
        self.ns: dict[str, Any] = {}
        self.rebind = False
        self.reserved: dict[str, Any] = {}

    def _map(self, names: list[str]) -> None:
        np = self.np
        for name in names:
            path = self.dir / f"t{self.names.index(name)}.npy"
            self.r[name] = np.load(path, mmap_mode="r", allow_pickle=False)
            if self.edit:
                self.w[name] = np.load(path, mmap_mode="c", allow_pickle=False)

    def init(self, msg: dict[str, Any]) -> dict[str, Any]:
        self.dir = Path(str(msg["dir"]))
        self.names = [str(n) for n in msg["names"]]
        self.edit = bool(msg["edit"])
        self.notes = list(msg.get("notes") or [])
        self._map(self.names)
        self.ns = {"__name__": "__miwsp__", "np": self.np, "R": _Maps(self.r, writable=False)}
        self.ns["notes"] = self.notes
        if self.edit:
            self.ns["W"] = _Maps(self.w, writable=True)
        self.rebind = msg.get("rebind") is True
        self.reserved = {name: self.ns[name] for name in RESERVED if name in self.ns}
        return {"ok": True}

    def restore_reserved(self) -> list[str]:
        """Bind every reserved name the code assigned or deleted back to the kernel's object; the
        names, in :data:`RESERVED` order. A list assigned to ``notes`` becomes the notes' items
        (the notes list is the one object the parent reads them from)."""
        undone = []
        for name, obj in self.reserved.items():
            value = self.ns.get(name, _MISSING)
            if value is obj:
                continue
            if name == "notes" and isinstance(value, list):
                self.notes[:] = list(value)
            self.ns[name] = obj
            undone.append(name)
        return undone

    def sync(self, msg: dict[str, Any]) -> dict[str, Any]:
        self._map([str(n) for n in msg["names"] if n in self.names])
        return {"ok": True}

    def exec(self, msg: dict[str, Any]) -> dict[str, Any]:
        buf = io.StringIO()
        try:
            tree = ast.parse(str(msg["code"]), mode="exec")
            last = None
            tail = tree.body[-1] if tree.body else None
            if isinstance(tail, ast.Expr):
                tree.body.pop()
                last = ast.Expression(tail.value)
            with contextlib.redirect_stdout(buf), contextlib.redirect_stderr(buf):
                exec(compile(tree, "<python>", "exec"), self.ns)
                if last is not None:
                    value = eval(compile(last, "<python>", "eval"), self.ns)
                    if value is not None:
                        print(repr(value))
        except BaseException:  # the editor's own failure (SystemExit included), reported as text
            lines = traceback.format_exc().splitlines()
            buf.write("\n".join(x for x in lines if 'File "' not in x or "<python>" in x))
        out = buf.getvalue()
        if len(out) > MAX_OUTPUT:
            half = MAX_OUTPUT // 2
            out = (
                out[:half]
                + f"\n... [{len(out) - MAX_OUTPUT:,} chars truncated] ...\n"
                + out[-half:]
            )
        undone = self.restore_reserved() if self.rebind else []
        if undone:
            names = ", ".join(undone)
            note = RESERVED_NOTE.format(
                names=names, are="is" if len(undone) == 1 else "are", them=names
            )
            out = (out.rstrip("\n") + "\n" if out else "") + note
        notes = self.ns.get("notes")
        return {
            "ok": True,
            "output": out or "(no output)",
            "deltas": self.deltas(),
            "notes": [str(x) for x in notes] if isinstance(notes, list) else [],
        }

    def deltas(self) -> list[dict[str, Any]]:
        np = self.np
        out = []
        for name in self.names if self.edit else []:
            w, r = self.w[name], self.r[name]
            uint = np.dtype(f"u{w.dtype.itemsize}")
            flat_w = np.asarray(w).reshape(-1)
            changed = np.flatnonzero(flat_w.view(uint) != np.asarray(r).reshape(-1).view(uint))
            if changed.size:
                out.append(
                    {
                        "name": name,
                        "idx": changed.tolist(),
                        "values": flat_w[changed].astype(np.float64).tolist(),
                    }
                )
        return out


def _preamble() -> tuple[int, int]:
    from wsp_worker import child_seccomp, spec

    spec.restore_fresh_dispositions()
    try:
        child_seccomp.install(extra_denied=child_seccomp.GRANDCHILD_RULES)
    except child_seccomp.SeccompUnavailableError:
        os._exit(EXIT_FILTER_FAILED)
    proto_in, proto_out = os.dup(0), os.dup(1)
    null = os.open(os.devnull, os.O_RDWR)
    os.dup2(null, 0)
    os.dup2(null, 1)
    os.close(null)
    os.write(proto_out, b"K")
    return proto_in, proto_out


def main() -> None:
    proto_in, proto_out = _preamble()
    reader = os.fdopen(proto_in, "rb", buffering=0)
    kernel = Kernel()
    handlers = {"init": kernel.init, "exec": kernel.exec, "sync": kernel.sync}
    pending = b""
    while True:
        while b"\n" not in pending:
            chunk = reader.read(1 << 16)
            if not chunk:
                os._exit(0)
            pending += chunk
            if len(pending) > MAX_LINE:
                os._exit(65)
        raw, pending = pending.split(b"\n", 1)
        msg = json.loads(raw)
        try:
            answer = handlers[str(msg.get("op"))](msg)
        except Exception as exc:  # a malformed message from the parent: answer, keep serving
            answer = {"ok": False, "error": f"{type(exc).__name__}: {exc}"}
        data = (json.dumps(answer) + "\n").encode()
        view = memoryview(data)
        while view:
            view = view[os.write(proto_out, view) :]
        sys.stdout.flush()


if __name__ == "__main__":
    main()
