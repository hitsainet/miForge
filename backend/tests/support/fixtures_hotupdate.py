"""Fixtures for §5.10, round-tripped through the real writers and readers.

**ADR-016's rule, and this feature's biggest trap:** *fixtures whose fields agree by
construction are how a suite stays green over a real bug.* A hand-built step record with
the two §5.10 fields set by hand proves the assertion, not the code - so the step-record
fixtures here go through F1's ``run_log.step_block`` and F11's ``timings.fill``, and the
client fixtures go through the real :class:`WeightsClient` pointed at the fake transport.

**The fake is reached through the configured base URL**, never by patching ``httpx``:
patching bypasses the headers, the refusal mapping and the no-retry rule, which are the
three things the client exists for.
"""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass
from typing import Any

import pytest

from core.config import reset_settings
from tests.support.fake_weights_surface import FakeWeightsSurface
from training.hotupdate import settings as settings_module
from training.hotupdate.client import WeightsClient
from training.hotupdate.timings import StepTimings

BASE_URL = "http://millm.test:8000"
TOKEN = "trainer-token"
LABEL = "miforge-nightly"


def configure(monkeypatch: pytest.MonkeyPatch, **overrides: str) -> None:
    """Configure §5.10 as a deployment would. **Live settings, reset afterwards.**"""
    env = {
        "MILLM_BASE_URL": BASE_URL,
        "MILLM_TRAINER_TOKEN": TOKEN,
        "MILLM_AUDIT_LABEL": LABEL,
    }
    env.update(overrides)
    for key, value in env.items():
        if value == "":
            monkeypatch.delenv(key, raising=False)
        else:
            monkeypatch.setenv(key, value)
    reset_settings()


@dataclass
class Harness:
    """A fake surface and a real client talking to it."""

    fake: FakeWeightsSurface
    client: WeightsClient
    timings: StepTimings

    def attach(self) -> dict[str, Any]:
        """Attach and remember the id, as the acquirer does **before any other call**."""
        response = self.client.call("attach")
        self.client.remember_attachment(str(response["attachment_id"]))
        return response


@pytest.fixture
def hotupdate_env(monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    configure(monkeypatch)
    yield
    reset_settings()


@pytest.fixture
def surface() -> FakeWeightsSurface:
    return FakeWeightsSurface()


@pytest.fixture
def harness(hotupdate_env: None, surface: FakeWeightsSurface) -> Iterator[Harness]:
    cfg = settings_module.load()
    client = WeightsClient(settings=cfg, transport=surface.transport())
    timings = StepTimings(
        watchdog_ms=cfg.commit_watchdog_ms,
        creep_fraction=cfg.commit_creep_fraction,
        run_id="hot-run",
    )
    try:
        yield Harness(fake=surface, client=client, timings=timings)
    finally:
        client.close()


class RecordingOptimizer:
    """A stand-in for F1's optimizer that records **when** its ``step`` ran.

    Not a mock of the model and not a mock of the optimizer's *effect*: it records the
    order of events so the bracket's stage order is assertable. ADR-016 forbids mocking the
    optimizer where the assertion is about what the optimizer did; here the assertion is
    about what surrounds it, and the surround is real.
    """

    def __init__(self, events: list[str], *, fail: BaseException | None = None) -> None:
        self.events = events
        self.fail = fail
        self.steps = 0
        self.param_groups: list[dict[str, Any]] = [{"lr": 1e-5}]

    def step(self) -> None:
        self.events.append("write")
        self.steps += 1
        if self.fail is not None:
            raise self.fail

    def zero_grad(self, set_to_none: bool = True) -> None:
        self.events.append("zero_grad")

    def state_dict(self) -> dict[str, Any]:
        return {"steps": self.steps}
