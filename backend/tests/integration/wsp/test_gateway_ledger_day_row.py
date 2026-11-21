"""The UTC day row is created without a race (review OR1-016-10).

``_locked_day`` used to SELECT ``FOR UPDATE`` and then INSERT when nothing was there. Two external
runs' first reserves of a UTC day could both find no row; the second INSERT then waited on the
first's uncommitted row and failed with a unique violation, which the internal app answered 500.
The race needs row locks, so its test runs on PostgreSQL (the serial job); SQLite serialises
writers, and there the sequential case is pinned.
"""

from __future__ import annotations

import datetime as dt
import threading
import time
from pathlib import Path
from typing import Any

import pytest
import sqlalchemy as sa
from sqlalchemy.orm import Session

from tests.integration.wsp.wsp_db import engine_at_head
from wsp import gateway_ledger

DAY = dt.date(2026, 11, 16)


def _days(engine: sa.Engine) -> int:
    with engine.connect() as conn:
        return int(conn.execute(sa.text("SELECT COUNT(*) FROM mf_wsp_spend_days")).scalar_one())


def test_the_day_row_is_made_once_and_found_again(tmp_path: Path) -> None:
    engine = engine_at_head(tmp_path)
    try:
        for _ in range(2):
            with Session(engine) as session:
                assert gateway_ledger._locked_day(session, DAY).utc_day == DAY
                session.commit()
        assert _days(engine) == 1
    finally:
        engine.dispose()


def test_two_first_reserves_of_a_day_both_get_the_one_row(tmp_path: Path) -> None:
    engine = engine_at_head(tmp_path)
    try:
        if engine.dialect.name != "postgresql":
            pytest.skip("needs row locks: runs in the serial PostgreSQL job")
        first = Session(engine)
        gateway_ledger._locked_day(first, DAY)  # inserted and locked, not committed
        outcome: dict[str, Any] = {}

        def rival() -> None:
            with Session(engine) as second:
                try:
                    outcome["day"] = gateway_ledger._locked_day(second, DAY).utc_day
                    second.commit()
                except Exception as exc:  # the defect: a unique violation
                    outcome["error"] = exc

        thread = threading.Thread(target=rival, daemon=True)
        thread.start()
        time.sleep(1.0)
        assert thread.is_alive(), f"the rival did not wait on the first row: {outcome}"
        first.commit()
        first.close()
        thread.join(30)
        assert not thread.is_alive()
        assert outcome == {"day": DAY}
        assert _days(engine) == 1
    finally:
        engine.dispose()
