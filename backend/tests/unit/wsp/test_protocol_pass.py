"""The protocol-test pass and its record (feature 019 P2; S-1, FTASKS 9.3-9.5; FR-019.12; F19-D9,
D23, D25, D26). Every case ends its run through ``runs.transition``, the production writer, so
removing the ``record`` line there is red here; the rows are real (a migrated database, SQLite or
the PostgreSQL ``MIFORGE_TEST_DATABASE_URL`` names).
"""

from __future__ import annotations

from collections.abc import Iterator
from typing import Any

import pytest
import sqlalchemy as sa
from sqlalchemy.orm import Session, sessionmaker

from tests.integration.wsp.wsp_db import NOW, bundle_row, engine_at_head, insert, run_row
from wsp import endpoints, protocol_tests, runs
from wsp.models import WspEndpoint

EP = "ep-gen"


@pytest.fixture()
def db(tmp_path: Any) -> Iterator[sessionmaker[Session]]:
    engine = engine_at_head(tmp_path)
    with engine.begin() as conn:
        insert(
            conn,
            "mf_wsp_endpoints",
            id=EP,
            kind="openai_compatible",
            base_url="http://vllm.inference.svc:8000/v1",
            model="qwen",
            locality="local",
            output_cap_field="max_completion_tokens",
            serving="generic",
            action_protocol="structured",
            created_at=NOW,
            created_by="operator",
        )
        bundle_row(conn)
    yield sessionmaker(bind=engine, expire_on_commit=False)
    engine.dispose()


_seq = iter(range(1, 10_000))


def _test_run(
    factory: sessionmaker[Session],
    *,
    marked: bool = True,
    status: str = "running",
    refused_turns: int | None = 0,
    calls: tuple[str, ...] = ("served",),
    caller_ref: str | None = None,
) -> str:
    run_id = f"wsp_{next(_seq):012d}"
    with factory() as s:
        conn = s.connection()
        run_row(
            conn,
            run_id,
            endpoint_id=EP,
            operation="inspect",
            status=status,
            refused_turns=refused_turns,
            action_protocol="structured" if status != "queued" else None,
            protocol_test_endpoint_id=EP if marked else None,
            caller_ref=caller_ref,
        )
        for n, outcome in enumerate(calls):
            insert(
                conn,
                "mf_wsp_usage",
                run_id=run_id,
                seq=n,
                endpoint_id=EP,
                input_tokens=1,
                output_tokens=1,
                cost_usd=0,
                at=NOW,
                outcome=outcome,
            )
        s.commit()
    return run_id


def _end(factory: sessionmaker[Session], run_id: str, to: str, frm: str = "running") -> None:
    with factory() as s:
        extra: dict[str, Any] = {"outcome": "report"} if to == "completed" else {}
        assert runs.transition(s, run_id, frm, to, **extra)
        s.commit()


def _endpoint(factory: sessionmaker[Session]) -> WspEndpoint:
    with factory() as s:
        row = s.get(WspEndpoint, EP)
        assert row is not None
        s.expunge(row)
        return row


def test_a_completed_clean_test_with_a_served_call_passes(db: sessionmaker[Session]) -> None:
    run_id = _test_run(db)
    _end(db, run_id, "completed")
    row = _endpoint(db)
    assert (row.protocol_test_result, row.protocol_test_run_id) == ("passed", run_id)
    assert row.protocol_test_at is not None
    assert (row.protocol_tested_base_url, row.protocol_tested_model) == (row.base_url, "qwen")
    assert row.protocol_tested_action_protocol == "structured"
    state = protocol_tests.pass_state(row)
    assert (state.current, state.cleared, state.required) == (True, False, True)


@pytest.mark.parametrize(
    ("over", "to"),
    [
        ({"refused_turns": 1}, "completed"),
        ({"calls": ("upstream_error",)}, "completed"),
        ({"calls": ()}, "completed"),
        ({}, "failed"),  # worker_lost and every other failure
        ({}, "cancelled"),
    ],
    ids=["one-refused-turn", "no-served-call", "no-call", "failed", "cancelled"],
)
def test_anything_else_fails(db: sessionmaker[Session], over: dict[str, Any], to: str) -> None:
    run_id = _test_run(db, **over)
    _end(db, run_id, to)
    assert _endpoint(db).protocol_test_result == "failed"


def test_a_failed_re_test_clears_a_pass_only_when_it_ends(db: sessionmaker[Session]) -> None:
    """F19-D9: the latest test is the record, written at its end: while the re-test is queued or
    running the earlier pass stands; its failure then replaces it."""
    first = _test_run(db)
    _end(db, first, "completed")
    second = _test_run(db, refused_turns=2)
    assert _endpoint(db).protocol_test_result == "passed"  # running: the pass stands
    _end(db, second, "completed")
    row = _endpoint(db)
    assert (row.protocol_test_result, row.protocol_test_run_id) == ("failed", second)
    assert not protocol_tests.pass_state(row).current


def test_a_cancelled_queued_test_fails_and_clears_a_pass(db: sessionmaker[Session]) -> None:
    """F19-D23: a queued test never pushed (no pushed protocol) fails, and clears the pass."""
    first = _test_run(db)
    _end(db, first, "completed")
    queued = _test_run(db, status="queued", calls=())
    _end(db, queued, "cancelled", frm="queued")
    row = _endpoint(db)
    assert (row.protocol_test_result, row.protocol_test_run_id) == ("failed", queued)
    assert row.protocol_tested_action_protocol == "structured"  # the endpoint's, as it stood


def test_a_run_that_is_not_a_test_records_nothing_whatever_its_caller_ref(
    db: sessionmaker[Session],
) -> None:
    run_id = _test_run(db, marked=False, caller_ref=f"protocol_test:{EP}")
    _end(db, run_id, "completed")
    row = _endpoint(db)
    assert (row.protocol_test_result, row.protocol_test_run_id) == (None, None)


def test_a_protocol_patch_clears_the_pass_on_read(db: sessionmaker[Session]) -> None:
    run_id = _test_run(db)
    _end(db, run_id, "completed")
    with db() as s:
        endpoints.update_endpoint(s, EP, {"action_protocol": "tool_calls"})
        s.commit()
    state = protocol_tests.pass_state(_endpoint(db))
    assert (state.result, state.current, state.cleared, state.required) == (
        "passed",
        False,
        True,
        False,
    )
    with db() as s:
        endpoints.update_endpoint(s, EP, {"action_protocol": "structured"})
        s.commit()
    assert protocol_tests.pass_state(_endpoint(db)).current  # the tested values hold again


def test_storing_a_result_moves_no_fingerprint(db: sessionmaker[Session]) -> None:
    """F19-D27: the record is outside the fingerprint, so it never fails a queued run."""
    before = runs.endpoint_fingerprint(_endpoint(db))
    run_id = _test_run(db)
    _end(db, run_id, "completed")
    assert runs.endpoint_fingerprint(_endpoint(db)) == before


def test_only_the_protocol_test_route_marks_a_run(db: sessionmaker[Session]) -> None:
    """TID §4.2's deviation, pinned: no CHECK ties the marker to ``endpoint_id``; ``create_run``
    writes it from ``Prepared.protocol_test`` alone, which only ``protocol_test.prepare`` sets."""
    import ast
    from pathlib import Path

    src = Path(runs.__file__).resolve().parents[1]
    setters = []
    for path in src.rglob("*.py"):
        tree = ast.parse(path.read_text())
        for node in ast.walk(tree):
            if (
                isinstance(node, ast.keyword)
                and node.arg == "protocol_test"
                and isinstance(node.value, ast.Constant)
                and node.value.value is True
            ):
                setters.append(path.relative_to(src).as_posix())
    assert setters == ["wsp/protocol_test.py"]
    with db() as s:
        mismatched = sa.text(
            "SELECT COUNT(*) FROM mf_wsp_runs WHERE protocol_test_endpoint_id IS NOT NULL "
            "AND protocol_test_endpoint_id <> endpoint_id"
        )
        assert s.scalar(mismatched) == 0
