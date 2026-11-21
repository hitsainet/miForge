"""The part-import job's completion line can actually be logged  (F10 task 4.11).

``extra={"created": ...}`` collides with ``LogRecord.created``, so ``logging`` raised
``KeyError: "Attempt to overwrite 'created' in LogRecord"`` - AFTER the part was imported,
failing the job over its own success line. Invisible to every test that ran the job with the
logger above INFO, because a disabled level never builds the record. Found 2026-10-18 while
sweeping for the same collision F11's resume record had (``name``).
"""

from __future__ import annotations

import logging
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any

import pytest


@dataclass
class _Imported:
    part_id: str = "part-1"
    part_kind: str = "corpus"
    created: bool = True
    manifest_sha256: str = "a" * 64
    splits: tuple[str, ...] = ("train",)
    caveats: tuple[dict[str, Any], ...] = ()
    licence_classes: tuple[str, ...] = field(default_factory=tuple)


def test_the_completion_line_logs_at_info(monkeypatch, caplog) -> None:
    import db.session
    from services.parts import acquire, tasks

    @contextmanager
    def scope():  # type: ignore[no-untyped-def]
        yield None

    monkeypatch.setattr(db.session, "session_scope", scope)
    monkeypatch.setattr(acquire, "from_hub", lambda: _Imported())
    caplog.set_level(logging.INFO, logger="miforge.parts.tasks")

    summary = tasks.part_import("hub", "org/repo")

    assert summary["created"] is True
    [line] = [r for r in caplog.records if getattr(r, "code", None) == "part_import_job_complete"]
    assert line.part_id == "part-1"  # type: ignore[attr-defined]
    assert line.part_created is True  # type: ignore[attr-defined]


@pytest.mark.parametrize("reserved", ["created", "name", "message"])
def test_a_reserved_key_really_does_raise(reserved: str, caplog) -> None:
    """The premise, checked: if logging ever stopped refusing these, the test above would
    be pinning nothing."""
    caplog.set_level(logging.INFO)
    with pytest.raises(KeyError):
        logging.getLogger("probe").info("x", extra={reserved: 1})
