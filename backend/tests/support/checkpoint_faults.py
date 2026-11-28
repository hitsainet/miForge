"""Two fakes: a syscall observer and a fault injector  (FTDD section 10; MUT-1, MUT-2).

**The observer exists because the outcome is identical until the power fails.** A test that
asserted only the result - "the directory is there and `latest` points at it" - would pass
against a writer that renames before it ``fsync``s, which is the one thing FR-005.3 is
about. So the order is *observed*: ``os.fsync``, ``os.rename`` and ``os.replace`` are
wrapped and recorded, and the assertion is over the sequence.

**The injector exists because the dangerous path is the failure path.** AC-005.3 is about
what is left behind when a save is cut short, and the interesting cut points are *between*
two operations the writer performs: after the weights, and between the ``fsync``s and the
rename. A test cannot pull the plug, so the injector raises at a named point.

*The foundation's ``test_atomic_dir.py`` already observes the primitive's own order. These
fakes are for the composed path - the writer, the service, the resave - where the order has
to hold through three layers of calls.*
"""

from __future__ import annotations

import os
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

__all__ = ["BrokenWriter", "InjectedFaultError", "SyscallLog", "observe_syscalls"]


@dataclass
class SyscallLog:
    """The durability-relevant syscalls, in the order they happened."""

    calls: list[tuple[str, str]] = field(default_factory=list)

    def names(self) -> list[str]:
        return [name for name, _ in self.calls]

    def first_index(self, name: str) -> int:
        """Where ``name`` first appears, or -1. Used to assert *order*, not counts."""
        for index, (call, _) in enumerate(self.calls):
            if call == name:
                return index
        return -1

    def fsynced_directories(self) -> list[str]:
        return [target for name, target in self.calls if name == "fsync_dir"]

    def fsynced_files(self) -> list[str]:
        return [target for name, target in self.calls if name == "fsync_file"]


@contextmanager
def observe_syscalls(monkeypatch: Any) -> Iterator[SyscallLog]:
    """Record ``fsync``/``rename``/``replace`` for the duration of the block.

    ``fsync`` is split into a file and a directory variant by looking at the descriptor,
    because *"the directory was ``fsync``ed too"* is its own assertion (MUT-2) and a single
    ``fsync`` count cannot make it.
    """
    log = SyscallLog()
    real_fsync, real_rename, real_replace = os.fsync, os.rename, os.replace

    def fsync(fd: int) -> None:
        try:
            target = os.readlink(f"/proc/self/fd/{fd}")
        except OSError:  # pragma: no cover - /proc is present on this platform
            target = str(fd)
        kind = "fsync_dir" if Path(target).is_dir() else "fsync_file"
        log.calls.append((kind, target))
        real_fsync(fd)

    def rename(src: Any, dst: Any) -> None:
        log.calls.append(("rename", f"{src} -> {dst}"))
        real_rename(src, dst)

    def replace(src: Any, dst: Any) -> None:
        log.calls.append(("replace", f"{src} -> {dst}"))
        real_replace(src, dst)

    monkeypatch.setattr(os, "fsync", fsync)
    monkeypatch.setattr(os, "rename", rename)
    monkeypatch.setattr(os, "replace", replace)
    try:
        yield log
    finally:
        monkeypatch.setattr(os, "fsync", real_fsync)
        monkeypatch.setattr(os, "rename", real_rename)
        monkeypatch.setattr(os, "replace", real_replace)


class InjectedFaultError(RuntimeError):
    """What the injector raises. A distinct type so a test cannot pass on a real error."""


@dataclass
class BrokenWriter:
    """A model whose ``save_pretrained`` fails at a chosen point.

    ``fail_after`` is one of:

    * ``"nothing"`` - fail before writing anything;
    * ``"weights"`` - write ``model.safetensors`` and then fail, which is AC-005.3's first
      case: the bytes are on disk in a staging directory and the manifest never happens;
    * ``"config"`` - write the weights and ``config.json``, then fail.

    The point of each is the same question: **what is discoverable afterwards?** The answer
    has to be nothing - no step directory, no temp directory in any listing or reindex, and
    ``latest`` still resolving to the previous step.
    """

    fail_after: str = "weights"
    #: What a successful ``save_pretrained`` would have written, so the staging directory
    #: looks exactly as it would at the moment of the fault.
    weights: bytes = b"not-really-tensors"
    config: dict[str, Any] = field(default_factory=lambda: {"model_type": "test"})

    def save_pretrained(self, directory: Any, **_: Any) -> None:
        import json

        target = Path(directory)
        if self.fail_after == "nothing":
            raise InjectedFaultError("injected before any file was written")
        (target / "model.safetensors").write_bytes(self.weights)
        if self.fail_after == "weights":
            raise InjectedFaultError("injected after the weights were written")
        (target / "config.json").write_text(json.dumps(self.config))
        if self.fail_after == "config":
            raise InjectedFaultError("injected after the config was written")
        raise InjectedFaultError(f"unknown fault point {self.fail_after!r}")
