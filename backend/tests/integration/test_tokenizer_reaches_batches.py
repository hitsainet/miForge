"""The acquired tokenizer reaches the batches through the PRODUCTION driver  (ADR-016 #1).

Found 2026-10-17 preparing F11 14.4: the run driver builds the signal source's ``RunContext``
before acquisition and gives it no tokenizer; every batch takes ``context.tokenizer``; the
objectives refuse ``tokenizer_unavailable`` without one. Nothing bound the acquired tokenizer,
so every production run - load, create and attach - would have refused at step 1. Every test
injected a tokenizer into its own ``RunContext`` or used a stub source that never tokenizes.

These drive ``tasks.drive_run`` with a source carrying a production-shaped ``RunContext``
(``tokenizer=None``) and assert what the source sees at ``next()``. Removing the
``bind_tokenizer`` call in ``tasks.py`` turns them red.
"""

# ruff: noqa: F811 - `rig` is a pytest fixture imported by name.
from __future__ import annotations

from typing import Any

from training import tasks
from training.acquisition.base import AcquiredModel
from training.signals.base import RunContext

from .test_step_loop import RECORD, Batch, drive, rig  # noqa: F401 - `rig` is a fixture


class _ContextSource:
    """A source shaped like production: a RunContext built before acquisition."""

    def __init__(self, tokenizer: Any = None) -> None:
        self.context = RunContext(run_id="run-1", tokenizer=tokenizer)
        self.seen: list[Any] = []

    def next(self) -> Batch:
        self.seen.append(self.context.tokenizer)
        return Batch()


def test_the_acquired_tokenizer_is_what_every_batch_sees(rig) -> None:  # type: ignore[no-untyped-def]
    source = _ContextSource()
    drive(rig, batch_source=source, max_steps=3)
    assert source.seen == [rig["tokenizer"]] * 3


def test_an_explicitly_supplied_tokenizer_is_kept(rig) -> None:  # type: ignore[no-untyped-def]
    mine = object()
    source = _ContextSource(tokenizer=mine)
    drive(rig, batch_source=source, max_steps=1)
    assert source.seen == [mine]


def test_bind_tokenizer_leaves_a_source_without_a_context_alone() -> None:
    class Stub:
        pass

    assert tasks.bind_tokenizer(Stub(), object()) is False
    assert tasks.bind_tokenizer(_ContextSource(), None) is False


def test_an_attached_acquisition_carries_the_serving_tokenizer(rig, monkeypatch) -> None:  # type: ignore[no-untyped-def]
    """The attach mode supplied ``tokenizer=None``; now the serving model's tokenizer."""
    serving = object()
    monkeypatch.setattr(
        tasks, "acquire_for_run", lambda spec: AcquiredModel(rig["model"], serving, RECORD)
    )
    source = _ContextSource()
    drive(rig, batch_source=source, max_steps=1)
    assert source.seen == [serving]
