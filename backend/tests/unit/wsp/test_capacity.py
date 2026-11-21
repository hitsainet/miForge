"""``wsp.capacity``: the one comparison of a push against the worker's ``/work`` (2026-11-20).

The wiring (freeze, dispatch, the pushed watchdog) is proved in ``test_freeze.py``,
``tests/integration/wsp/test_dispatch.py`` and ``test_wsp_approvals.py``; this file pins the
arithmetic and the message.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from pydantic import ValidationError

from core.errors import Refusal
from wsp import capacity
from wsp.capacity import EDITOR_HEADROOM_BYTES, GIB, WATCHDOG_MARGIN_BYTES
from wsp.settings import WSPSettings


def settings(work: int = 64 * GIB) -> WSPSettings:
    return WSPSettings(wsp_worker_work_bytes=work)


def test_the_stated_values() -> None:
    assert (EDITOR_HEADROOM_BYTES, WATCHDOG_MARGIN_BYTES) == (4 * GIB, 4 * GIB)
    assert WSPSettings().wsp_worker_work_bytes == 64 * GIB
    assert capacity.watchdog_bytes(settings()) == 60 * GIB


def test_the_watchdog_follows_the_setting() -> None:
    assert capacity.watchdog_bytes(settings(100 * GIB)) == 96 * GIB


def test_the_boundary_is_push_plus_headroom_at_the_watchdog() -> None:
    largest = 60 * GIB - EDITOR_HEADROOM_BYTES
    assert capacity.refusal(largest, settings()) is None
    found = capacity.refusal(largest + 1, settings())
    assert found is not None and found[0] == "wsp_bundle_too_large_for_worker"


def test_the_message_names_both_sizes_and_what_to_raise() -> None:
    push = 57 * GIB
    found = capacity.refusal(push, settings())
    assert found is not None
    message = found[1]
    assert str(push) in message and str(60 * GIB) in message
    assert "WSP_WORKER_WORK_BYTES" in message and "wsp-worker.yaml" in message
    assert "sizeLimit" in message


def test_require_fits_raises_with_the_figures() -> None:
    with pytest.raises(Refusal) as err:
        capacity.require_fits(57 * GIB, settings())
    assert err.value.code == "wsp_bundle_too_large_for_worker"
    assert err.value.details == {
        "push_bytes": 57 * GIB,
        "session_limit_bytes": 60 * GIB,
        "setting": "WSP_WORKER_WORK_BYTES",
    }
    capacity.require_fits(1, settings())  # a fitting push passes silently


def test_manifest_bytes_sums_every_file() -> None:
    manifest = {"files": [{"bytes": 3}, {"bytes": 34_881_633_695}]}
    assert capacity.manifest_bytes(manifest) == 34_881_633_698


def test_files_bytes_reads_each_size_and_skips_a_missing_file(tmp_path: Path) -> None:
    sparse = tmp_path / "practice.bin"
    with sparse.open("wb") as handle:
        handle.truncate(57 * GIB)  # sparse: no disk used
    small = tmp_path / "a.json"
    small.write_bytes(b"12345")
    files = [("practice.bin", sparse), ("a.json", small), ("gone", tmp_path / "gone")]
    assert capacity.files_bytes(files) == 57 * GIB + 5


def test_a_setting_with_no_room_for_a_push_is_refused_at_start() -> None:
    with pytest.raises(ValidationError):
        WSPSettings(wsp_worker_work_bytes=capacity.MIN_WORK_BYTES)


def test_the_refusal_is_a_413_like_the_import_size_refusal() -> None:
    from api.app import status_for

    assert status_for("wsp_bundle_too_large_for_worker") == status_for("wsp_bundle_too_large")
    assert status_for("wsp_bundle_too_large_for_worker") == 413


def test_the_headroom_is_the_childs_largest_file() -> None:
    """The headroom is stated as one file at the child's ``RLIMIT_FSIZE``; if that bound moves,
    this does."""
    from wsp_worker.__main__ import CHILD_FSIZE

    assert EDITOR_HEADROOM_BYTES == CHILD_FSIZE
