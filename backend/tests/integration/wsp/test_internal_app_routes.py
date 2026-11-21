"""The internal application and the gateway ledger (feature 016; FTASKS 5.1, 6.3a, 6.7a, 6.7b
in part; P38, P49, P50, P61, D16, D19, R1B-A2, R3A-13, R3A-18).

The app is ``create_internal_app()`` itself, bound to a migrated database as the process engine.
"""

from __future__ import annotations

import datetime as dt
import logging
import threading
from collections.abc import Iterator
from decimal import Decimal
from typing import Any

import pytest
import sqlalchemy as sa
from fastapi.testclient import TestClient

from api.route_table import mounted_api_routes
from tests.integration.wsp.test_dispatch import FakeWorker, World
from tests.integration.wsp.test_dispatch import world as dispatch_world  # noqa: F401
from wsp import gateway_ledger, runs
from wsp.settings import get_wsp_settings

SERVICE = "s" * 48
AUTH = {"Authorization": f"Bearer {SERVICE}"}
PREFIX = "/wsp-internal/v1/gateway"
EXPECTED = {
    ("GET", f"{PREFIX}/sessions"),
    ("GET", f"{PREFIX}/sessions/by-token/{{token_sha256}}"),
    ("GET", f"{PREFIX}/endpoints/{{endpoint_id}}"),
    ("POST", f"{PREFIX}/reserve"),
    ("POST", f"{PREFIX}/settle"),
    # Live progress (2026-11-20): a call the gateway refused, for the run's call timeline.
    ("POST", f"{PREFIX}/refused"),
}


@pytest.fixture()
def inner(dispatch_world: World, monkeypatch: pytest.MonkeyPatch) -> Iterator[Any]:  # noqa: F811
    from api import wsp_internal_app
    from db import session as db_session

    monkeypatch.setenv("WSP_GATEWAY_SERVICE_TOKEN", SERVICE)
    get_wsp_settings.cache_clear()
    monkeypatch.setattr(db_session, "_engine", dispatch_world.engine)
    monkeypatch.setattr(db_session, "_session_factory", None)
    wsp_internal_app._negative.clear()
    client = TestClient(wsp_internal_app.create_internal_app(), raise_server_exceptions=False)
    yield dispatch_world, client
    get_wsp_settings.cache_clear()


def _running(world: World, endpoint: str = "ep-local", **budgets: Any) -> tuple[str, str]:
    """A pushed run and the PLAINTEXT token the push carried."""
    world.body = {**world.body, "budgets": {**world.body["budgets"], **budgets}}
    run_id = world.queue(endpoint)
    worker = FakeWorker()
    tokens: list[str] = []
    real = runs.transition

    world.settings = world.settings.model_copy(update={"wsp_gateway_url": "http://127.0.0.1:8080"})
    original_push = worker.push

    def push(document: dict[str, Any], files: Any) -> dict[str, Any]:
        tokens.append(document["gateway"]["token"])
        return original_push(document, files)

    worker.push = push  # type: ignore[method-assign]
    assert world.drive(worker) == [f"pushed:{run_id}"]
    assert real is runs.transition
    return run_id, tokens[0]


def _sha(token: str) -> str:
    from wsp.tokens import digest

    return digest(token)


# ----------------------------------------------------------------------- the surface


def test_the_live_route_table_is_the_gateway_set_only(inner: Any) -> None:
    from api.wsp_internal_app import create_internal_app

    live = {
        (method, route.path)
        for route in mounted_api_routes(create_internal_app())
        for method in route.methods
    }
    assert live, "the scan saw no route at all"
    assert live == EXPECTED


def test_the_public_app_has_no_internal_path() -> None:
    from api.app import create_app

    paths = {route.path for route in mounted_api_routes(create_app())}
    assert paths and not any("/wsp-internal" in path for path in paths)


@pytest.mark.parametrize("headers", [{}, {"Authorization": "Bearer wrong"}])
def test_every_route_requires_the_service_token(inner: Any, headers: dict[str, str]) -> None:
    _world, client = inner
    response = client.get(f"{PREFIX}/sessions", headers=headers)
    assert response.status_code == 401
    assert response.json()["error"]["code"] == "wsp_gateway_token_invalid"


# ----------------------------------------------------------------------- sessions


def test_a_session_minted_after_the_gateway_started_resolves_by_token(inner: Any) -> None:
    world, client = inner
    run_id, token = _running(world)
    listed = client.get(f"{PREFIX}/sessions", headers=AUTH).json()["sessions"]
    assert [s["run_id"] for s in listed] == [run_id]
    found = client.get(f"{PREFIX}/sessions/by-token/{_sha(token)}", headers=AUTH)
    assert found.status_code == 200, found.text
    assert found.json()["run_id"] == run_id and found.json()["spend"] == {"usd": 0.0, "tokens": 0}


def test_a_terminal_runs_token_is_refused_by_token_and_at_reserve(inner: Any) -> None:
    world, client = inner
    run_id, token = _running(world)
    with world.factory() as session:
        runs.cancel(session, run_id)
        session.commit()
    assert client.get(f"{PREFIX}/sessions/by-token/{_sha(token)}", headers=AUTH).status_code == 404
    reserved = client.post(f"{PREFIX}/reserve", headers=AUTH, json=_reserve(run_id, token))
    assert reserved.status_code == 409
    assert reserved.json()["error"]["code"] == "wsp_session_not_active"


def test_a_pushing_runs_first_reserve_succeeds(inner: Any) -> None:
    """R3A-18: the child may call before the driver marks the run running."""
    world, client = inner
    run_id, token = _running(world)
    with world.factory() as session:
        session.execute(
            sa.text("UPDATE mf_wsp_runs SET status = 'pushing' WHERE run_id = :r"), {"r": run_id}
        )
        session.commit()
    assert (
        client.post(f"{PREFIX}/reserve", headers=AUTH, json=_reserve(run_id, token)).status_code
        == 200
    )


def test_the_plaintext_token_is_in_no_row_and_no_answer(inner: Any, caplog: Any) -> None:
    world, client = inner
    caplog.set_level(logging.DEBUG)
    run_id, token = _running(world)
    answers = [
        client.get(f"{PREFIX}/sessions", headers=AUTH).text,
        client.get(f"{PREFIX}/sessions/by-token/{_sha(token)}", headers=AUTH).text,
        client.post(f"{PREFIX}/reserve", headers=AUTH, json=_reserve(run_id, token)).text,
    ]
    with world.engine.connect() as conn:
        dump = " ".join(
            str(row)
            for table in ("mf_wsp_runs", "mf_wsp_reservations", "mf_approvals")
            for row in conn.execute(sa.text(f"SELECT * FROM {table}"))
        )
    assert token not in dump and _sha(token) in dump
    assert all(token not in answer for answer in answers)
    assert token not in caplog.text


def test_the_endpoint_document_carries_the_ciphertext_never_a_key(inner: Any) -> None:
    world, client = inner
    with world.engine.begin() as conn:
        conn.execute(
            sa.text("UPDATE mf_wsp_endpoints SET key_ciphertext = :c WHERE id = 'ep-ext'"),
            {"c": b"\x00sealed"},
        )
    doc = client.get(f"{PREFIX}/endpoints/ep-ext", headers=AUTH).json()
    assert doc["key_ciphertext"] == "AHNlYWxlZA==" and doc["locality"] == "external"
    assert "key" not in {k for k in doc if k != "key_ciphertext"} and "api_key" not in doc


# ----------------------------------------------------------------------- the ledger


def _reserve(
    run_id: str, token: str, usd: str = "0.01", tokens_: int = 10, external: bool = False
) -> dict[str, Any]:
    return {
        "run_id": run_id,
        "token_sha256": _sha(token),
        "worst_case_usd": usd,
        "worst_case_tokens": tokens_,
        "external": external,
    }


def test_one_request_in_flight_per_token(inner: Any) -> None:
    world, client = inner
    run_id, token = _running(world)
    first = client.post(f"{PREFIX}/reserve", headers=AUTH, json=_reserve(run_id, token))
    second = client.post(f"{PREFIX}/reserve", headers=AUTH, json=_reserve(run_id, token))
    assert first.status_code == 200 and second.status_code == 409
    assert second.json()["error"]["code"] == "wsp_request_in_flight"
    settled = client.post(
        f"{PREFIX}/settle",
        headers=AUTH,
        json={
            "reservation_id": first.json()["reservation_id"],
            "usage": {"input_tokens": 3, "output_tokens": 4, "cost_usd": 0},
        },
    )
    assert settled.status_code == 200
    assert (
        client.post(f"{PREFIX}/reserve", headers=AUTH, json=_reserve(run_id, token)).status_code
        == 200
    )


def test_settling_a_closed_reservation_is_refused_and_appends_nothing(inner: Any) -> None:
    world, client = inner
    run_id, token = _running(world)
    rid = client.post(f"{PREFIX}/reserve", headers=AUTH, json=_reserve(run_id, token)).json()[
        "reservation_id"
    ]
    body = {"reservation_id": rid, "usage": {"input_tokens": 1, "output_tokens": 1, "cost_usd": 0}}
    assert client.post(f"{PREFIX}/settle", headers=AUTH, json=body).status_code == 200
    again = client.post(f"{PREFIX}/settle", headers=AUTH, json=body)
    assert again.status_code == 409 and again.json()["error"]["code"] == "wsp_reservation_closed"
    with world.engine.connect() as conn:
        assert conn.execute(sa.text("SELECT COUNT(*) FROM mf_wsp_usage")).scalar_one() == 1


def test_eight_concurrent_requests_keep_the_total_within_the_token_cap(inner: Any) -> None:
    """R1B-A2: maximal usage from eight threads never passes ``budgets.tokens``."""
    world, client = inner
    run_id, token = _running(world, tokens=100)
    _long_sqlite_busy_timeout(world.engine)
    barrier = threading.Barrier(8)
    statuses: list[int] = []

    def one() -> None:
        barrier.wait()
        for _ in range(10):
            r = client.post(
                f"{PREFIX}/reserve", headers=AUTH, json=_reserve(run_id, token, tokens_=30)
            )
            statuses.append(r.status_code)
            if r.status_code == 200:
                settled = client.post(
                    f"{PREFIX}/settle",
                    headers=AUTH,
                    json={
                        "reservation_id": r.json()["reservation_id"],
                        "usage": {"input_tokens": 10, "output_tokens": 20, "cost_usd": 0},
                    },
                )
                statuses.append(settled.status_code)

    threads = [threading.Thread(target=one) for _ in range(8)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    with world.factory() as session:
        _usd, used = gateway_ledger.spent(session, run_id)
    assert 0 < used <= 100
    # Every answer is a success or a named refusal; a 500 (e.g. SQLite "database is locked") is a
    # defect, not noise (the earlier form ignored it and flaked once under a loaded host).
    assert set(statuses) <= {200, 409, 429}, sorted(set(statuses))


def _long_sqlite_busy_timeout(engine: sa.Engine) -> None:
    """Why this test cannot flake on SQLite (FTASKS change log, wsp-forge3).

    SQLite serialises the ledger with a database write lock taken by each transaction's FIRST
    statement (``gateway_ledger._write_lock``), so a waiting writer sleeps in SQLite's busy
    handler. pysqlite's default handler gives up after 5 s and raises "database is locked": on a
    host shared by several full suites, 8 threads x 20 short transactions could queue one writer
    past 5 s. With a 120 s handler a writer fails only if the WHOLE serial sequence (160
    transactions of a few ms each) takes longer than that. No first statement is a read, so no
    lock upgrade (which SQLite refuses at once, without waiting) can occur. PostgreSQL uses row
    locks and is unaffected.
    """
    if engine.dialect.name != "sqlite":
        return

    @sa.event.listens_for(engine, "connect")
    def _busy(dbapi_connection: Any, _record: Any) -> None:
        cursor = dbapi_connection.cursor()
        cursor.execute("PRAGMA busy_timeout = 120000")
        cursor.close()

    engine.dispose()  # pooled connections were opened without the pragma


def test_the_daily_external_cap_holds_until_utc_midnight(inner: Any) -> None:
    """D19: two external sessions crossing $50 in one UTC day; a local one is unaffected."""
    world, _client = inner
    settings = world.settings.model_copy(update={"wsp_external_spend_max_usd_per_day": 50})
    day = gateway_ledger.utcnow()
    first, token_a = _running(world, "ep-ext", cost_usd=40)
    with world.factory() as session:
        rid = gateway_ledger.reserve(
            session,
            run_id=first,
            token_sha256=_sha(token_a),
            worst_case_usd=Decimal("30"),
            worst_case_tokens=1,
            external=True,
            settings=settings,
            now=day,
        )
        gateway_ledger.settle(session, reservation_id=rid, usage={"cost_usd": "30"}, now=day)
        session.commit()
        with pytest.raises(Exception) as refused:
            gateway_ledger.reserve(
                session,
                run_id=first,
                token_sha256=_sha(token_a),
                worst_case_usd=Decimal("21"),
                worst_case_tokens=1,
                external=True,
                settings=settings,
                now=day,
            )
        assert refused.value.envelope()["error"]["code"] in {
            "wsp_daily_spend_cap_reached",
            "wsp_budget_exhausted",
        }
        session.rollback()
        # the next UTC day starts fresh
        nxt = day + dt.timedelta(days=1)
        session.execute(
            sa.text("UPDATE mf_wsp_runs SET token_expires_at = :e WHERE run_id = :r"),
            {"e": nxt + dt.timedelta(hours=1), "r": first},
        )
        rid2 = gateway_ledger.reserve(
            session,
            run_id=first,
            token_sha256=_sha(token_a),
            worst_case_usd=Decimal("5"),
            worst_case_tokens=1,
            external=True,
            settings=settings,
            now=nxt,
        )
        assert rid2 > rid


def test_the_daily_cap_refuses_across_two_sessions(inner: Any) -> None:
    world, _client = inner
    settings = world.settings.model_copy(update={"wsp_external_spend_max_usd_per_day": 50})
    first, token_a = _running(world, "ep-ext", cost_usd=40)
    with world.factory() as session:
        rid = gateway_ledger.reserve(
            session,
            run_id=first,
            token_sha256=_sha(token_a),
            worst_case_usd=Decimal("35"),
            worst_case_tokens=1,
            external=True,
            settings=settings,
        )
        gateway_ledger.settle(session, reservation_id=rid, usage={"cost_usd": "35"})
        session.commit()
        runs.transition(session, first, "running", "completed", outcome="report")
        session.commit()
    second, token_b = _running(world, "ep-ext", cost_usd=40)
    with world.factory() as session:
        with pytest.raises(Exception) as refused:
            gateway_ledger.reserve(
                session,
                run_id=second,
                token_sha256=_sha(token_b),
                worst_case_usd=Decimal("20"),
                worst_case_tokens=1,
                external=True,
                settings=settings,
            )
        assert refused.value.envelope()["error"]["code"] == "wsp_daily_spend_cap_reached"


def test_a_terminal_transition_settles_open_reservations_at_worst_case(inner: Any) -> None:
    world, client = inner
    run_id, token = _running(world)
    rid = client.post(
        f"{PREFIX}/reserve", headers=AUTH, json=_reserve(run_id, token, usd="0.5", tokens_=40)
    ).json()["reservation_id"]
    with world.factory() as session:
        runs.cancel(session, run_id)
        session.commit()
        _usd, used = gateway_ledger.spent(session, run_id)
    assert used == 40
    late = client.post(
        f"{PREFIX}/settle", headers=AUTH, json={"reservation_id": rid, "usage": {"cost_usd": 0}}
    )
    assert late.status_code == 409 and late.json()["error"]["code"] == "wsp_reservation_closed"


def test_a_reserve_racing_a_terminal_settle_completes_without_a_deadlock(inner: Any) -> None:
    """R3A-18 (6.7b): reserve takes the run's lock, then the day's; the terminal transition
    takes the run's lock (its UPDATE) and then settles at worst case. The same order on both
    paths, so on PostgreSQL neither waits on the other forever. Run under
    ``MIFORGE_TEST_DATABASE_URL`` for the PostgreSQL reading (STATUS.md records it)."""
    world, client = inner
    run_id, token = _running(world, "ep-ext", cost_usd=5, tokens=10_000_000)
    errors: list[BaseException] = []
    answers: list[int] = []
    start = threading.Barrier(2)

    def reserve() -> None:
        try:
            start.wait(timeout=10)
            for _ in range(20):
                r = client.post(
                    f"{PREFIX}/reserve",
                    headers=AUTH,
                    json=_reserve(run_id, token, usd="0.01", tokens_=10, external=True),
                )
                answers.append(r.status_code)
                if r.status_code == 200:
                    client.post(
                        f"{PREFIX}/settle",
                        headers=AUTH,
                        json={
                            "reservation_id": r.json()["reservation_id"],
                            "usage": {"cost_usd": 0.001, "input_tokens": 1, "output_tokens": 1},
                        },
                    )
        except BaseException as exc:
            errors.append(exc)

    def finish() -> None:
        try:
            start.wait(timeout=10)
            with world.factory() as session:
                runs.cancel(session, run_id)
                session.commit()
        except BaseException as exc:
            errors.append(exc)

    threads = [threading.Thread(target=reserve), threading.Thread(target=finish)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=60)
    assert not any(t.is_alive() for t in threads), "a deadlock: a thread never finished"
    assert errors == []
    assert set(answers) <= {200, 409}, answers  # 409 wsp_session_not_active once cancelled
    assert 500 not in answers
    with world.factory() as session:
        assert (
            session.scalar(
                sa.select(sa.func.count())
                .select_from(sa.table("mf_wsp_reservations", sa.column("settled_at")))
                .where(sa.column("settled_at").is_(None))
            )
            == 0
        )  # nothing left open after the terminal transition


def test_the_terminal_transition_waits_for_the_run_lock_before_any_day_lock(inner: Any) -> None:
    """C15v made deterministic (R3A-18): the racing test above rarely interleaves, so here a
    reserve-shaped transaction HOLDS the run's lock while the terminal transition starts, then
    takes the day's lock. With one lock order (run, then day) on both paths that succeeds; a
    terminal path that took the day's lock first would deadlock here, and PostgreSQL would abort
    one side. PostgreSQL only: SQLite has no row locks."""
    import time

    world, _client = inner
    if world.engine.dialect.name != "postgresql":
        pytest.skip("needs row locks: runs in the serial PostgreSQL job")
    run_id, _token = _running(world, "ep-ext", cost_usd=5, tokens=10_000_000)
    errors: list[BaseException] = []
    holder = world.factory()
    gateway_ledger._locked_run(holder, run_id)  # the reserve path's first lock, held

    def finish() -> None:
        try:
            with world.factory() as session:
                runs.cancel(session, run_id)
                session.commit()
        except BaseException as exc:
            errors.append(exc)

    thread = threading.Thread(target=finish, daemon=True)
    thread.start()
    time.sleep(1.0)
    assert thread.is_alive(), f"the terminal transition did not wait for the run lock: {errors}"
    try:
        gateway_ledger._locked_day(holder, gateway_ledger.utcnow().date())  # its second lock
        holder.commit()
    except BaseException as exc:
        errors.append(exc)
        holder.rollback()
    finally:
        holder.close()
    thread.join(30)
    assert not thread.is_alive() and errors == []


def test_a_write_commits_before_its_answer_is_sent(
    inner: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    """016-P2 (found measuring 16.5): over a REAL server, a settle sent the moment its reserve
    answered must find the reservation. With FastAPI's default dependency scope the commit ran
    after the answer, so the settle could arrive first: refused, the response withheld, the
    session ended. The commit is delayed here so the order is deterministic, not a race."""
    import contextlib

    import httpx

    from api.wsp_internal_app import create_internal_app
    from db import session as db_session
    from tests.support.served import served

    world, _ = inner
    if world.engine.dialect.name == "sqlite":
        pytest.skip(
            "commit order needs PostgreSQL: SQLite serialises the two writers (or shares the "
            "test's one connection), so the late commit cannot show; the PostgreSQL job runs it"
        )
    run_id, token = _running(world)
    real_scope = db_session.session_scope

    @contextlib.contextmanager
    def slow_commit() -> Iterator[Any]:
        with real_scope() as session:
            yield session
            import time

            time.sleep(0.3)  # before session_scope's own commit

    monkeypatch.setattr(db_session, "session_scope", slow_commit)
    with served(create_internal_app()) as url, httpx.Client(base_url=url, headers=AUTH) as http:
        reserved = http.post(f"{PREFIX}/reserve", json=_reserve(run_id, token))
        assert reserved.status_code == 200, reserved.text
        settled = http.post(
            f"{PREFIX}/settle",
            json={
                "reservation_id": reserved.json()["reservation_id"],
                "usage": {"input_tokens": 1, "output_tokens": 1, "cost_usd": 0},
            },
        )
    assert settled.status_code == 200, settled.text


def test_every_internal_route_ends_its_transaction_before_answering() -> None:
    """The wiring, read from the LIVE app: every route's database session dependency is
    function-scoped (it closes, committing, before the response is sent). Since F-1 (2026-11-19)
    the internal app's routes take the shared ``api.deps.SessionDep``, so the dependency looked
    for is ``session_dependency`` itself."""
    from api.deps import session_dependency
    from api.route_table import declared_route
    from api.wsp_internal_app import create_internal_app

    found = 0
    for route in mounted_api_routes(create_internal_app()):
        for dep in declared_route(route).dependant.dependencies:
            if dep.call is session_dependency:
                found += 1
                assert dep.scope == "function", route.path
    assert found == len(EXPECTED) == 6
