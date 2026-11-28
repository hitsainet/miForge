"""Migration ``0033_wsp_prompt_evidence`` (2026-11-21): a usage row written before it has no
evidence (all three null), each CHECK refuses its bad value, and the round trip works. Runs on
SQLite, and on PostgreSQL when ``MIFORGE_TEST_DATABASE_URL`` names one.
"""

from __future__ import annotations

from typing import Any

import pytest
import sqlalchemy as sa
from alembic import command

from tests.integration.wsp.wsp_db import NOW, bundle_row, config, engine_at_head, insert, run_row

EVIDENCE = "0033_wsp_prompt_evidence"
BEFORE = "0032_wsp_thinking"
COLUMNS = ("sent_bytes", "prompt_messages", "conversation_sha256")


def _columns(engine: sa.Engine) -> set[str]:
    return {c["name"] for c in sa.inspect(engine).get_columns("mf_wsp_usage")}


def _usage(conn: sa.Connection, seq: int, **extra: Any) -> None:
    insert(
        conn,
        "mf_wsp_usage",
        run_id="wsp_000000000001",
        seq=seq,
        endpoint_id="ep-local",
        input_tokens=10,
        output_tokens=5,
        cost_usd=0,
        at=NOW,
        **extra,
    )


@pytest.fixture()
def at_0032(tmp_path):
    engine = engine_at_head(tmp_path)
    url = engine.url.render_as_string(hide_password=False)
    engine.dispose()
    command.downgrade(config(url), BEFORE)
    old = sa.create_engine(url)
    assert not set(COLUMNS) & _columns(old)
    with old.begin() as conn:
        bundle_row(conn)
        run_row(conn)
        _usage(conn, 0)
    old.dispose()
    yield url


def test_an_earlier_row_has_no_evidence_and_a_new_one_keeps_it(at_0032: str) -> None:
    command.upgrade(config(at_0032), EVIDENCE)
    engine = sa.create_engine(at_0032)
    assert set(COLUMNS) <= _columns(engine)
    with engine.begin() as conn:
        _usage(conn, 1, sent_bytes=33_100, prompt_messages=30, conversation_sha256="a" * 64)
        rows = conn.execute(
            sa.text(
                "SELECT sent_bytes, prompt_messages, conversation_sha256 FROM mf_wsp_usage "
                "ORDER BY seq"
            )
        ).all()
    assert [tuple(r) for r in rows] == [(None, None, None), (33_100, 30, "a" * 64)]
    engine.dispose()


@pytest.mark.parametrize(("column", "bad"), [("sent_bytes", 0), ("prompt_messages", -1)])
def test_each_check_refuses_its_bad_value(at_0032: str, column: str, bad: int) -> None:
    command.upgrade(config(at_0032), EVIDENCE)
    engine = sa.create_engine(at_0032)
    with pytest.raises(sa.exc.IntegrityError), engine.begin() as conn:
        _usage(conn, 2, **{column: bad})
    engine.dispose()


def test_the_round_trip_works(at_0032: str) -> None:
    command.upgrade(config(at_0032), EVIDENCE)
    command.downgrade(config(at_0032), BEFORE)
    engine = sa.create_engine(at_0032)
    assert not set(COLUMNS) & _columns(engine)
    engine.dispose()
    command.upgrade(config(at_0032), EVIDENCE)
