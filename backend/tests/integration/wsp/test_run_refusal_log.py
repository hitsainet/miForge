"""The ``wsp_run_refused`` log line (feature 020; FTID §12; FTASKS 9.1): the backend logs it at
filing and at push, from named fields only (the code, where, the endpoint or run), never text.
"""

from __future__ import annotations

import logging
from typing import Any

import pytest

from core.errors import Refusal
from tests.integration.wsp.test_dispatch import (
    FakeWorker,
    World,
    _hosted_anthropic,
    _queue_before_020,
)
from tests.integration.wsp.test_dispatch import world as world


def _lines(caplog: pytest.LogCaptureFixture) -> list[Any]:
    return [r for r in caplog.records if r.getMessage() == "wsp_run_refused"]


def test_a_refusal_at_filing_is_logged_once_with_its_code(
    world: World, caplog: pytest.LogCaptureFixture
) -> None:
    _hosted_anthropic(world, "ep-old", context_window_tokens=None)
    with caplog.at_level(logging.INFO, logger="wsp.runs"), pytest.raises(Refusal):
        world.queue("ep-old")
    [line] = _lines(caplog)
    assert (line.code, line.at, line.endpoint_id) == (
        "endpoint_window_required",
        "filing",
        "ep-old",
    )


def test_a_refusal_at_push_is_logged_once_with_its_code(
    world: World, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    _hosted_anthropic(world, "ep-old", price_cache_write_per_mtok=None)
    run_id = _queue_before_020(world, monkeypatch, "ep-old")
    with caplog.at_level(logging.INFO, logger="wsp.dispatch"):
        world.drive(FakeWorker())
    [line] = _lines(caplog)
    assert (line.code, line.at, line.run_id) == (
        "wsp_endpoint_cache_prices_required",
        "push",
        run_id,
    )
