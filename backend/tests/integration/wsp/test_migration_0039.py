"""H-14: migration ``0039_wsp_hosted`` (feature 020; FR-020.1, FR-020.4, FR-020.5, FR-020.20;
F20-D8, D12, D13, D22). Runs on SQLite, and on PostgreSQL when ``MIFORGE_TEST_DATABASE_URL`` names
one (the serial PostgreSQL job).

- FPRD FR-020.5's four-row case: the history modes, the thinking values, and the local row's
  fingerprint unchanged byte for byte;
- every CHECK of ``mf_wsp_endpoints`` present after the rebuild (SQLite) and the re-created CHECK
  (PostgreSQL), compared with the list read at ``0038`` (FTASKS 3.1), and each still refusing;
- the rows, the referencing run and the usage columns survive; the new columns' CHECKs refuse;
- the downgrade refuses with an effort row and works without one.
"""

from __future__ import annotations

import re
from typing import Any

import pytest
import sqlalchemy as sa
from alembic import command

from tests.integration.wsp.wsp_db import NOW, bundle_row, config, engine_at_head, insert, run_row
from wsp.models import WspEndpoint
from wsp.runs import endpoint_fingerprint

HOSTED = "0039_wsp_hosted"
BEFORE = "0038_wsp_pool"

#: ``mf_wsp_endpoints``' CHECK names at ``0038``, read from a SQLite database migrated there and
#: from the model (FTASKS 3.1), written out here independently of the migration.
CHECKS_AT_0038 = {
    "ck_mf_wsp_endpoints_kind_known",
    "ck_mf_wsp_endpoints_external_has_prices",
    "ck_mf_wsp_endpoints_locality_known",
    "ck_mf_wsp_endpoints_output_cap_field_known",
    "ck_mf_wsp_endpoints_serving_known",
    "ck_mf_wsp_endpoints_action_protocol_known",
    "ck_mf_wsp_endpoints_context_window_positive",
    "ck_mf_wsp_endpoints_thinking_known",
    "ck_mf_wsp_endpoints_thinking_tokens_positive",
    "ck_mf_wsp_endpoints_output_cap_known",
    "ck_mf_wsp_endpoints_paging_known",
    "ck_mf_wsp_endpoints_output_ceiling_at_least_256",
    "ck_mf_wsp_endpoints_registration_window_positive",
    "ck_mf_wsp_endpoints_protocol_test_result_known",
    "ck_mf_wsp_endpoints_protocol_tested_action_protocol_known",
}
CHECKS_AT_0039 = CHECKS_AT_0038 | {"ck_mf_wsp_endpoints_history_mode_known"}


def _endpoint(conn: sa.Connection, endpoint_id: str, **values: Any) -> None:
    row: dict[str, Any] = {
        "id": endpoint_id,
        "kind": "anthropic",
        "base_url": "https://api.anthropic.com",
        "model": "claude-x",
        "locality": "external",
        "price_in_per_mtok": 3,
        "price_out_per_mtok": 15,
        "output_cap_field": "max_tokens",
        "serving": "generic",
        "key_ciphertext": b"sealed",
        "key_fingerprint": "abcd1234",
        "created_at": NOW,
        "created_by": "operator",
        "thinking": "off",
    }
    row.update(values)
    insert(conn, "mf_wsp_endpoints", **row)


def _four_rows(conn: sa.Connection) -> None:
    """FR-020.5's AC: two hosted Anthropic rows (one bare, one complete), a hosted
    OpenAI-compatible row and a local row, every one at thinking ``off``."""
    _endpoint(conn, "ep-a-bare")
    _endpoint(
        conn,
        "ep-a-full",
        context_window_tokens=200_000,
        price_cache_write_per_mtok=3.75,
        price_cache_read_per_mtok=0.3,
    )
    _endpoint(
        conn,
        "ep-oa-hosted",
        kind="openai_compatible",
        base_url="https://llm.example.com/v1",
        output_cap_field="max_completion_tokens",
    )
    _endpoint(
        conn,
        "ep-local",
        kind="openai_compatible",
        base_url="http://127.0.0.1:8000/v1",
        locality="local",
        price_in_per_mtok=None,
        price_out_per_mtok=None,
        output_cap_field="max_completion_tokens",
        key_ciphertext=None,
        key_fingerprint=None,
    )


def _rows(engine: sa.Engine) -> dict[str, dict[str, Any]]:
    with engine.connect() as conn:
        found = conn.execute(sa.text("SELECT * FROM mf_wsp_endpoints ORDER BY id")).mappings()
        return {r["id"]: dict(r) for r in found}


def _fingerprint(values: dict[str, Any]) -> str:
    known = {c.name for c in WspEndpoint.__table__.columns}
    return endpoint_fingerprint(WspEndpoint(**{k: v for k, v in values.items() if k in known}))


def _check_names(engine: sa.Engine) -> set[str]:
    with engine.connect() as conn:
        if engine.dialect.name == "postgresql":
            return set(
                conn.execute(
                    sa.text(
                        "SELECT conname FROM pg_constraint "
                        "WHERE conrelid = 'mf_wsp_endpoints'::regclass AND contype = 'c'"
                    )
                ).scalars()
            )
        sql = conn.execute(
            sa.text(
                "SELECT sql FROM sqlite_master WHERE type = 'table' AND name = 'mf_wsp_endpoints'"
            )
        ).scalar_one()
    return set(re.findall(r"CONSTRAINT (ck_\w+) CHECK", sql))


def _primary_key(engine: sa.Engine) -> list[str]:
    return list(sa.inspect(engine).get_pk_constraint("mf_wsp_endpoints")["constrained_columns"])


@pytest.fixture()
def at_0038(tmp_path) -> Any:
    """A database at ``0038`` holding the four rows, a run on the bare hosted row (a reference
    the SQLite rebuild must keep) and one call on it."""
    engine = engine_at_head(tmp_path)
    url = engine.url.render_as_string(hide_password=False)
    engine.dispose()
    command.downgrade(config(url), BEFORE)
    old = sa.create_engine(url)
    assert "history_mode" not in {
        c["name"] for c in sa.inspect(old).get_columns("mf_wsp_endpoints")
    }
    assert _check_names(old) == CHECKS_AT_0038
    with old.begin() as conn:
        _four_rows(conn)
        bundle_row(conn)
        run_row(conn, endpoint_id="ep-a-bare", endpoint_locality="external")
        insert(
            conn,
            "mf_wsp_usage",
            run_id="wsp_000000000001",
            seq=0,
            endpoint_id="ep-a-bare",
            input_tokens=10,
            output_tokens=2,
            cost_usd=0,
            at=NOW,
        )
    before = _rows(old)
    old.dispose()
    yield url, before


def test_the_four_rows_take_their_history_modes_and_thinking(at_0038) -> None:
    url, before = at_0038
    command.upgrade(config(url), HOSTED)
    engine = sa.create_engine(url)
    try:
        after = _rows(engine)
        assert {k: v["history_mode"] for k, v in after.items()} == {
            "ep-a-bare": "append_only",
            "ep-a-full": "append_only",
            "ep-local": "paging",
            "ep-oa-hosted": "paging",
        }
        assert {k: v["thinking"] for k, v in after.items()} == {
            "ep-a-bare": "default",
            "ep-a-full": "default",
            "ep-local": "off",
            "ep-oa-hosted": "off",
        }
        # Nothing else moved: windows and prices are left as they were (F20-D8, F20-D12).
        for endpoint_id, old in before.items():
            new = after[endpoint_id]
            assert {k: new[k] for k in old if k != "thinking"} == {
                k: v for k, v in old.items() if k != "thinking"
            }, endpoint_id
        assert after["ep-a-bare"]["context_window_tokens"] is None
        assert after["ep-a-bare"]["price_cache_read_per_mtok"] is None
        # The local row's fingerprint is byte for byte what it was at 0038; a hosted Anthropic
        # row's moves with its history mode (a run queued on it fails at push).
        assert _fingerprint(after["ep-local"]) == _fingerprint(before["ep-local"])
        assert _fingerprint(after["ep-a-full"]) != _fingerprint(before["ep-a-full"])
        with engine.connect() as conn:
            assert conn.execute(sa.text("SELECT endpoint_id FROM mf_wsp_runs")).scalar_one() == (
                "ep-a-bare"
            )
            usage = conn.execute(
                sa.text("SELECT cache_read_tokens, cache_write_tokens FROM mf_wsp_usage")
            ).one()
        assert tuple(usage) == (None, None)  # a call from before 020: not recorded
    finally:
        engine.dispose()


def test_every_check_survives_and_still_refuses(at_0038) -> None:
    url, _before = at_0038
    command.upgrade(config(url), HOSTED)
    engine = sa.create_engine(url)
    try:
        assert _check_names(engine) == CHECKS_AT_0039
        assert _primary_key(engine) == ["id"]
        bad = [
            {"kind": "mystery"},
            {"locality": "orbit"},
            {"output_cap_field": "max_words"},
            {"serving": "elsewhere"},
            {"price_in_per_mtok": None},  # external_has_prices
            {"action_protocol": "smoke"},
            {"context_window_tokens": 0},
            {"thinking": "xhigh"},
            {"thinking_tokens": 0},
            {"output_cap": "nah"},  # within VARCHAR(4): PostgreSQL checks the length first
            {"paging": "nah"},
            {"output_ceiling_tokens": 255},
            {"registration_window_tokens": 0},
            {"protocol_test_result": "unsure"},
            {"protocol_tested_action_protocol": "smoke"},
            {"history_mode": "rewrite"},
        ]
        for n, values in enumerate(bad):
            with pytest.raises(sa.exc.IntegrityError), engine.begin() as conn:
                _endpoint(conn, f"ep-bad-{n}", **values)
        with engine.begin() as conn:  # the control: each effort is now stored
            for effort in ("low", "medium", "high"):
                _endpoint(conn, f"ep-{effort}", thinking=effort, history_mode="append_only")
        with pytest.raises(sa.exc.IntegrityError), engine.begin() as conn:
            _endpoint(conn, "ep-dup", id="ep-low")  # the primary key
    finally:
        engine.dispose()


@pytest.mark.parametrize("column", ["cache_read_tokens", "cache_write_tokens"])
def test_the_usage_cache_columns_refuse_a_negative_count(tmp_path, column: str) -> None:
    engine = engine_at_head(tmp_path)
    call = {
        "run_id": "wsp_000000000001",
        "endpoint_id": "ep",
        "input_tokens": 1,
        "output_tokens": 1,
        "cost_usd": 0,
        "at": NOW,
    }
    with engine.begin() as conn:
        bundle_row(conn)
        run_row(conn)
        insert(conn, "mf_wsp_usage", seq=0, **call, **{column: 900})  # the control
    with pytest.raises(sa.exc.IntegrityError), engine.begin() as conn:
        insert(conn, "mf_wsp_usage", seq=1, **call, **{column: -1})
    engine.dispose()


def test_the_downgrade_refuses_while_an_endpoint_uses_an_effort(at_0038) -> None:
    url, _before = at_0038
    command.upgrade(config(url), HOSTED)
    engine = sa.create_engine(url)
    with engine.begin() as conn:
        conn.execute(
            sa.text("UPDATE mf_wsp_endpoints SET thinking = 'medium' WHERE id = 'ep-a-full'")
        )
    engine.dispose()
    with pytest.raises(RuntimeError) as refused:
        command.downgrade(config(url), BEFORE)
    assert "1 endpoint(s) use a thinking effort" in str(refused.value)
    check = sa.create_engine(url)
    try:
        assert _rows(check)["ep-a-full"]["history_mode"] == "append_only"  # nothing was dropped
    finally:
        check.dispose()


def test_the_downgrade_works_without_an_effort_and_keeps_every_row(at_0038) -> None:
    url, before = at_0038
    command.upgrade(config(url), HOSTED)
    command.downgrade(config(url), BEFORE)
    engine = sa.create_engine(url)
    try:
        assert _check_names(engine) == CHECKS_AT_0038
        columns = {c["name"] for c in sa.inspect(engine).get_columns("mf_wsp_endpoints")}
        assert "history_mode" not in columns
        usage = {c["name"] for c in sa.inspect(engine).get_columns("mf_wsp_usage")}
        assert not {"cache_read_tokens", "cache_write_tokens"} & usage
        after = _rows(engine)
        assert set(after) == set(before)
        # F20-D13's off-to-default is not reverted: default is valid under the old CHECK.
        assert after["ep-a-full"]["thinking"] == "default"
        with pytest.raises(sa.exc.IntegrityError), engine.begin() as conn:
            _endpoint(conn, "ep-effort", thinking="low")  # the old CHECK is back
    finally:
        engine.dispose()
    command.upgrade(config(url), "head")  # and the round trip goes up again
