"""The trainer writes its attachment's live state into the run record (operator Q1, 2026-10-16).

Driven through ``tasks.drive_run`` and the real ``AttachAcquirer.acquire`` over the fake
serving surface, with the run's ``session_scope`` over the test database - so what is
asserted is the row the read-only API serves. Before this, ``run_state.remember`` had **no
production caller** (``acquire_for_run`` passes no session), so even the base URL the UI
keys the chip on was never written.
"""

# ruff: noqa: F811 - `rig` is a pytest fixture imported by name; parameters receive it.

from __future__ import annotations

from contextlib import contextmanager
from typing import Any, ClassVar

import sqlalchemy as sa

from models.run import Run
from tests.support.fake_weights_surface import FakeWeightsSurface
from tests.support.fixtures_hotupdate import BASE_URL
from tests.support.jobs import session_for
from training import tasks

from .test_step_loop import RECORD, RUN, drive, rig  # noqa: F401 - `rig` is a fixture


def _acquirer(rig, monkeypatch):  # type: ignore[no-untyped-def]
    from tests.support.fixtures_hotupdate import configure
    from training.acquisition.base import AcquisitionMode, AcquisitionSpec, Availability
    from training.hotupdate import attach as attach_module
    from training.hotupdate import rebuild

    configure(monkeypatch)
    surface = FakeWeightsSurface()
    acquirer = attach_module.AttachAcquirer(transport=surface.transport())
    monkeypatch.setattr(type(acquirer), "available", lambda self: Availability(available=True))

    class Base:
        device_uuid = "GPU-test"
        tensors: ClassVar[dict[str, Any]] = {}

    monkeypatch.setattr(rebuild, "build", lambda response, **kw: Base())
    monkeypatch.setattr(rebuild, "check_build_identity", lambda response: None)
    monkeypatch.setattr(attach_module, "build_module", lambda response, base: rig["model"])
    monkeypatch.setattr(
        attach_module.AttachAcquirer,
        "_tokenizer_loader",
        lambda self, resident, live, *directory: (lambda: "serving-tokenizer"),
    )
    monkeypatch.setattr(attach_module, "acquisition_record", lambda **kw: RECORD)
    monkeypatch.setattr(attach_module.AttachAcquirer, "_verifier", lambda self, *a: None)
    monkeypatch.setattr(
        tasks,
        "acquire_for_run",
        lambda spec: acquirer.acquire(AcquisitionSpec(mode=AcquisitionMode.ATTACH)),
    )
    return acquirer, surface


def _drive_with_scope(rig, **overrides):  # type: ignore[no-untyped-def]
    """Drive with a ``session_scope`` that shares the run's session.

    The suite's in-memory SQLite is ONE connection (``StaticPool``), so a second session
    cannot open a transaction while the run's is open - an infrastructure limit, not the
    behaviour under test (production scopes are separate PostgreSQL transactions). Sharing
    the run's session keeps every write visible to the row read afterwards.
    """
    with session_for(rig["engine"]) as session:

        @contextmanager
        def scope():  # type: ignore[no-untyped-def]
            yield session
            session.flush()

        return drive(rig, session=session, session_scope=scope, **overrides)


def _row(rig) -> Any:  # type: ignore[no-untyped-def]
    table = Run.__table__
    with session_for(rig["engine"]) as session:
        return session.execute(
            sa.select(
                table.c.hotupdate_base_url, table.c.attachment_id, table.c.hotupdate_status
            ).where(table.c.id == RUN)
        ).one()


class TestTheRunRecordCarriesTheAttachment:
    def test_the_attachment_and_its_state_are_written_and_end_detached(
        self, rig, monkeypatch
    ) -> None:
        acquirer, surface = _acquirer(rig, monkeypatch)
        try:
            _drive_with_scope(rig, max_steps=2)
        finally:
            if acquirer.heartbeat is not None:
                acquirer.heartbeat.stop()
        base_url, attachment_id, status = _row(rig)
        assert base_url == BASE_URL
        assert attachment_id == "attachment-0001"
        assert status is not None
        assert status["resident_model"] == surface.resident_model
        # The last write is the detach's: nothing is attached for this run any more.
        assert status["attached"] is False
        assert status["weights_state"] == "clean"
        assert status["holder"] is None

    def test_state_is_published_while_running(self, rig, monkeypatch) -> None:
        """At bind, after the first step (then throttled), and at detach."""
        acquirer, _surface = _acquirer(rig, monkeypatch)
        from training.hotupdate import run_state

        writes: list[dict[str, Any]] = []
        real = run_state.record_status

        def spy(session, run_id, record, **kw):  # type: ignore[no-untyped-def]
            writes.append(record)
            return real(session, run_id, record, **kw)

        monkeypatch.setattr(run_state, "record_status", spy)
        try:
            _drive_with_scope(rig, max_steps=3)
        finally:
            if acquirer.heartbeat is not None:
                acquirer.heartbeat.stop()
        assert len(writes) == 3, writes  # bind, first step (others throttled), detach
        assert [w["attached"] for w in writes] == [True, True, False]
