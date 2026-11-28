"""A run on a miServer endpoint that states no window waits for its model (2026-11-27).

Run ``wsp_4fbf5a18758e`` (phase-2 E1) was pushed 18 s before miServer finished loading its editor
model. miServer's ``GET /api/health/inference`` omits ``context_length`` while no model is
resident, the read came back empty, the push resolved 8,192 ``assumed`` for a 126,976-token model,
and the run failed ``editor_context_window_too_small`` with no editor request sent. Now the claim
reads the window before it consumes anything and leaves such a run queued; it fails
``editor_not_resident`` only after ``wsp_editor_resident_wait_s``; a push whose own read is empty
puts the run back in the queue; and no path resolves an assumed window for it
(``0xcc/reviews/wsp_window_read_not_resident_2026-11-27.md``).

The miServer here is a real HTTP stub on 127.0.0.1 read through the production reader
(``window_read.read_window_answer`` -> ``wsp.tasks.read_served_window``), answering what
miServer's idle and loaded ``/api/health/inference`` answer
(``miserver/services/inference_service.py``
``get_backend_info``: the ``context_length`` key is absent while nothing is loaded).
"""

from __future__ import annotations

import datetime as dt
import json
import threading
import time
from collections.abc import Iterator
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

import pytest

from tests.integration.wsp.test_dispatch import FakeWorker, World, _at, _j, build_world
from tests.integration.wsp.wsp_db import NOW, insert
from wsp import dispatch, endpoints, window_read
from wsp.models import WspEndpoint
from wsp.settings import get_wsp_settings

#: miServer's idle answer: the serial backend's description, no ``context_length`` key.
IDLE = {
    "backend": "serial",
    "description": "Serial queue",
    "capabilities": {"streaming": True},
    "limitations": [],
}
#: The editor model of run ``wsp_4fbf5a18758e``, once loaded.
LOADED_WINDOW = 126_976
T0 = dt.datetime(2026, 11, 27, 20, 21, tzinfo=dt.UTC)


class MiServer:
    """``GET /api/health/inference`` on 127.0.0.1, idle until :meth:`load` is called."""

    def __init__(self) -> None:
        self.answer: dict[str, Any] = dict(IDLE)
        self.paths: list[str] = []
        outer = self

        class Handler(BaseHTTPRequestHandler):
            def do_GET(self) -> None:
                outer.paths.append(self.path)
                status = 200 if self.path == "/api/health/inference" else 404
                data = json.dumps(outer.answer if status == 200 else {}).encode()
                self.send_response(status)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(data)))
                self.end_headers()
                self.wfile.write(data)

            def log_message(self, *args: Any) -> None:
                return

        self.httpd = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.port = self.httpd.server_address[1]
        threading.Thread(target=self.httpd.serve_forever, daemon=True).start()

    def load(self, window: int = LOADED_WINDOW) -> None:
        self.answer = {**IDLE, "context_length": window}

    def unload(self) -> None:
        self.answer = dict(IDLE)

    def reads(self) -> int:
        return self.paths.count("/api/health/inference")

    def close(self) -> None:
        self.httpd.shutdown()
        self.httpd.server_close()


@pytest.fixture()
def miserver() -> Iterator[MiServer]:
    server = MiServer()
    yield server
    server.close()


class GenericServer:
    """A vLLM-shaped server on 127.0.0.1 answering every path with ``status``; with 200, its
    ``/v1/models`` lists ``MODEL`` at ``window`` (``None``: listed without a window)."""

    MODEL = "lfm2.5-1.2b-instruct"

    def __init__(self, status: int = 200, window: int | None = 32_768) -> None:
        self.status, self.window = status, window
        self.paths: list[str] = []
        #: Requests answered with ``status`` before every later one gets a 503 (``None``: never).
        self.answers_left: int | None = None
        #: Debt D5: seconds each request waits before any answer (a black-holed host when longer
        #: than the reader's timeout).
        self.hang_s = 0.0
        outer = self

        class Handler(BaseHTTPRequestHandler):
            def do_GET(self) -> None:
                outer.paths.append(self.path)
                if outer.hang_s:
                    time.sleep(outer.hang_s)
                if outer.answers_left is not None:
                    if outer.answers_left == 0:
                        outer.status = 503
                    else:
                        outer.answers_left -= 1
                body: dict[str, Any] = {"error": "no"}
                if outer.status == 200 and self.path == "/v1/models":
                    body = {"data": [{"id": outer.MODEL, "max_model_len": outer.window}]}
                data = json.dumps(body).encode()
                if outer.status != 200:
                    code = outer.status
                else:  # the models list answers; /props (llama.cpp's) is not this server's
                    code = 200 if self.path == "/v1/models" else 404
                self.send_response(code)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(data)))
                try:
                    self.end_headers()
                    self.wfile.write(data)
                except OSError:
                    return  # the reader gave up on a hung request (D5)

            def log_message(self, *args: Any) -> None:
                return

        self.httpd = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.port = self.httpd.server_address[1]
        threading.Thread(target=self.httpd.serve_forever, daemon=True).start()

    def close(self) -> None:
        self.httpd.shutdown()
        self.httpd.server_close()


@pytest.fixture()
def generic() -> Iterator[GenericServer]:
    server = GenericServer(status=503)
    yield server
    server.close()


def _closed_port() -> int:
    """A port nothing listens on (bound, then released), so a connection is refused."""
    import socket

    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return int(s.getsockname()[1])


@pytest.fixture()
def world(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, miserver: MiServer, generic: GenericServer
) -> Iterator[World]:
    built = build_world(tmp_path, monkeypatch)
    with built.engine.begin() as conn:
        # A miServer endpoint that states no window, at the stub; a generic one that states its
        # window, for the run that must not be held back; and (D1) two generic ones that state
        # none: at a port nothing listens on, and at a server answering ``generic.status``.
        for endpoint_id, url, serving, window in (
            ("ep-stub", f"http://127.0.0.1:{miserver.port}/v1", "miserver", None),
            ("ep-gen", "http://miserver-backend.miserver.svc:8001/v1", "generic", 32_768),
            ("ep-down", f"http://127.0.0.1:{_closed_port()}/v1", "generic", None),
            ("ep-vllm", f"http://127.0.0.1:{generic.port}/v1", "generic", None),
        ):
            insert(
                conn,
                "mf_wsp_endpoints",
                id=endpoint_id,
                kind="openai_compatible",
                base_url=url,
                model="lfm2.5-1.2b-instruct",
                locality="local",
                context_window_tokens=window,
                output_cap_field="max_tokens",
                serving=serving,
                created_at=NOW,
                created_by="operator",
            )
    yield built
    built.engine.dispose()
    get_wsp_settings.cache_clear()


def _drive(world: World, worker: Any, at: dt.datetime, **kwargs: Any) -> list[str]:
    """One pass at ``at`` with the PRODUCTION reader, unless a test passes its own."""
    kwargs.setdefault("read_window", window_read.read_window_answer)
    return world.drive(worker, now=at, **kwargs)


def test_a_run_waits_queued_until_miserver_reports_a_window_then_runs_reported(
    world: World, miserver: MiServer
) -> None:
    """The live case: nothing resident at the first pass, so nothing is consumed and the run stays
    queued; once the model is loaded the next pass claims it and the push resolves the REPORTED
    126,976, never 8,192 ``assumed``."""
    run_id = world.queue("ep-stub")
    worker = FakeWorker()
    assert _drive(world, worker, T0) == [f"editor_not_resident:{run_id}"]
    row = world.run(run_id)
    assert row["status"] == "queued" and row["boot_id"] is None and row["started_at"] is None
    assert _at(row["editor_wait_since"]) == T0
    assert world.boots() == 0 and worker.pushed is None
    assert worker.names() == ["ready"]  # read, not pushed, not recycled
    assert miserver.reads() == 1

    miserver.load()
    assert _drive(world, worker, T0 + dt.timedelta(seconds=18)) == [f"pushed:{run_id}"]
    row = world.run(run_id)
    assert row["status"] == "running"
    assert (row["window_tokens"], row["window_source"], row["served_context_tokens"]) == (
        LOADED_WINDOW,
        "reported",
        LOADED_WINDOW,
    )
    assert worker.pushed is not None
    editor = worker.pushed["spec"]["editor"]
    assert (editor["window_tokens"], editor["window_source"]) == (LOADED_WINDOW, "reported")
    assert editor["window_warnings"] == []
    # The claim's residency read, then the push's own read (F19-D11: the push resolves its read).
    assert miserver.reads() == 3
    # The wait stays on the row as the record of when it began.
    assert _at(row["editor_wait_since"]) == T0


def test_the_wait_is_bounded_and_ends_in_editor_not_resident(world: World) -> None:
    """Within ``wsp_editor_resident_wait_s`` of the first empty read the run waits; at it, the run
    fails ``editor_not_resident``, with a message in the suite's error layout (what, ``fix:``,
    ``next:`` as a JSON list), and nothing was ever consumed."""
    wait = world.settings.wsp_editor_resident_wait_s
    assert wait == 900  # the default this test documents
    run_id = world.queue("ep-stub")
    worker = FakeWorker()
    assert _drive(world, worker, T0) == [f"editor_not_resident:{run_id}"]
    almost = T0 + dt.timedelta(seconds=wait - 1)
    assert _drive(world, worker, almost) == [f"editor_not_resident:{run_id}"]
    assert world.run(run_id)["status"] == "queued"
    assert _drive(world, worker, T0 + dt.timedelta(seconds=wait)) == [
        f"refused:{run_id}:editor_not_resident"
    ]
    row = world.run(run_id)
    assert (row["status"], row["stop_reason"], row["outcome"]) == ("failed", "error", "failed")
    failure = _j(row["failure"])
    assert failure["code"] == "editor_not_resident"
    what, fix, nxt = failure["message"].split("\n")
    assert "no context window for endpoint ep-stub for 900 s" in what
    assert "never pushed" in what
    assert fix.startswith("fix: load lfm2.5-1.2b-instruct in miServer")
    assert "context_window_tokens" in fix
    assert nxt.startswith("next: ")
    steps = json.loads(nxt.removeprefix("next: "))
    assert [s["tool"] for s in steps] == [
        "miserver_load_model",
        "forge_wsp_list_endpoints",
        "forge_wsp_start_run",
    ]
    assert all(set(s) == {"tool", "args", "why"} for s in steps)
    assert len(failure["message"]) <= 2000  # the run contract's Failure.message bound
    assert world.boots() == 0 and worker.pushed is None
    assert row["window_source"] is None  # no window was ever resolved for it


def test_a_waiting_run_holds_back_neither_a_worker_nor_the_run_behind_it(world: World) -> None:
    """The oldest run waits; the next one (another endpoint, its window stated) is claimed and
    pushed in the same pass, on the worker the waiting run did not take."""
    waiting = world.queue("ep-stub")
    other = world.queue("ep-gen")
    worker = FakeWorker()
    assert _drive(world, worker, T0) == [f"editor_not_resident:{waiting}", f"pushed:{other}"]
    assert world.run(waiting)["status"] == "queued"
    assert world.run(other)["status"] == "running"


def test_a_push_whose_own_read_is_empty_puts_the_run_back_in_the_queue(
    world: World, miserver: MiServer
) -> None:
    """The model goes away between the claim's read and the push's read: the push resolves
    nothing, sends nothing, and the run is queued again with a new wait; its boot stays consumed
    (P49), so the next pass recycles that worker instead of pushing to it."""
    run_id = world.queue("ep-stub")
    worker = FakeWorker()
    answers = iter([LOADED_WINDOW, None])

    def reader(endpoint: WspEndpoint) -> int | None:
        return next(answers)

    later = T0 + dt.timedelta(seconds=5)
    assert _drive(world, worker, later, read_window=reader) == [f"requeued:{run_id}"]
    row = world.run(run_id)
    assert row["status"] == "queued"
    assert (row["boot_id"], row["worker_index"], row["schedule_key"]) == (None, None, None)
    assert (row["started_at"], row["push_heartbeat_at"], row["window_source"]) == (None, None, None)
    assert _at(row["editor_wait_since"]) == later
    assert worker.pushed is None and world.boots() == 1
    # The next pass: miServer still has nothing resident; the consumed boot is recycled.
    actions = _drive(world, worker, later + dt.timedelta(seconds=5))
    assert f"recycled:{worker.boot}" in actions
    assert world.run(run_id)["status"] == "queued"


@pytest.mark.parametrize(
    ("serving", "stated", "style", "waits"),
    [
        ("miserver", None, "session", True),
        ("miserver", None, "program", True),
        ("miserver", None, "replay", False),
        ("miserver", None, "program_rerun", False),
        ("miserver", 32_768, "session", False),
        ("generic", None, "session", False),
    ],
)
def test_only_a_miserver_endpoint_stating_no_window_waits_and_only_for_an_editor_run(
    serving: str, stated: int | None, style: str, waits: bool
) -> None:
    endpoint = WspEndpoint(
        id="ep",
        kind="openai_compatible",
        base_url="http://miserver-backend.miserver.svc:8000/v1",
        model="m",
        locality="local",
        output_cap_field="max_tokens",
        serving=serving,
        context_window_tokens=stated,
    )
    assert window_read.awaits_resident(endpoint, style) is waits
    if waits:
        with pytest.raises(window_read.NotResidentError):
            window_read.resolve(endpoint, None, style=style)
    else:
        resolved = window_read.resolve(endpoint, None, style=style)
        assert resolved.window_source == ("stated" if stated else "assumed")


def test_a_hosted_miserver_shaped_endpoint_never_waits() -> None:
    """An external endpoint is never read and must state its window (``run_refusal``)."""
    endpoint = WspEndpoint(
        id="ep",
        kind="openai_compatible",
        base_url="https://api.example.com/v1",
        model="m",
        locality="external",
        output_cap_field="max_tokens",
        serving="miserver",
        context_window_tokens=None,
    )
    assert window_read.awaits_resident(endpoint, "session") is False


def test_the_endpoint_list_never_predicts_an_assumed_window_for_a_miserver_endpoint() -> None:
    """``window_now`` and ``window_if_read_fails`` read ``waits`` with tokens 0 (none) for a
    miServer endpoint that states no window and has no run's read; a generic one keeps 8,192
    ``assumed``."""

    def endpoint(serving: str) -> WspEndpoint:
        return WspEndpoint(
            id="ep",
            kind="openai_compatible",
            base_url="http://miserver-backend.miserver.svc:8000/v1",
            model="m",
            locality="local",
            output_cap_field="max_tokens",
            serving=serving,
            context_window_tokens=None,
        )

    facts = endpoints.window_facts(endpoint("miserver"), None)
    assert facts["window_now"] == {"tokens": 0, "source": "waits", "run_id": None, "at": None}
    assert facts["window_if_read_fails"] == {"tokens": 0, "source": "waits", "no_answer": "waits"}
    facts = endpoints.window_facts(endpoint("generic"), None)
    assert facts["window_now"] == {"tokens": 8192, "source": "assumed", "run_id": None, "at": None}
    assert facts["window_if_read_fails"] == {
        "tokens": 8192,
        "source": "assumed",
        "no_answer": "waits",
    }


def test_the_production_pass_reads_with_the_production_reader_by_default() -> None:
    """``dispatch.drive``'s default reads nothing, so a pass wired without the reader makes every
    such run wait and then fail by name, never push it onto an assumed window."""
    assert dispatch.drive.__kwdefaults__["read_window"] is dispatch.no_window


# ------------------------------------------- D1 (operator, 2026-11-27): a generic server's answer


def test_d1_a_generic_server_that_does_not_answer_waits_then_fails(world: World) -> None:
    """Connection refused on every request: the run waits like a miServer one, then fails
    ``editor_not_resident`` with the no-answer message; nothing is consumed."""
    wait = world.settings.wsp_editor_resident_wait_s
    run_id = world.queue("ep-down")
    worker = FakeWorker()
    assert _drive(world, worker, T0) == [f"editor_not_resident:{run_id}"]
    assert world.run(run_id)["status"] == "queued" and world.boots() == 0
    assert _drive(world, worker, T0 + dt.timedelta(seconds=wait)) == [
        f"refused:{run_id}:editor_not_resident"
    ]
    failure = _j(world.run(run_id)["failure"])
    assert failure["code"] == "editor_not_resident"
    what, fix, nxt = failure["message"].split("\n")
    assert "gave no answer to the window read for 900 s" in what and "ep-down" in what
    assert fix.startswith("fix: start the server") and "context_window_tokens" in fix
    steps = json.loads(nxt.removeprefix("next: "))
    assert [s["tool"] for s in steps] == ["forge_wsp_list_endpoints", "forge_wsp_start_run"]
    assert worker.pushed is None and world.boots() == 0


@pytest.mark.parametrize("status", [500, 502, 503])
def test_d1_a_5xx_is_no_answer_and_the_run_runs_once_the_server_answers(
    world: World, generic: GenericServer, status: int
) -> None:
    generic.status = status
    run_id = world.queue("ep-vllm")
    worker = FakeWorker()
    assert _drive(world, worker, T0) == [f"editor_not_resident:{run_id}"]
    generic.status, generic.window = 200, 65_536
    assert _drive(world, worker, T0 + dt.timedelta(seconds=30)) == [f"pushed:{run_id}"]
    row = world.run(run_id)
    assert (row["window_tokens"], row["window_source"]) == (65_536, "reported")


@pytest.mark.parametrize("status", [401, 403, 404])
def test_d1_a_generic_server_that_answers_without_a_window_is_assumed(
    world: World, generic: GenericServer, status: int
) -> None:
    """F19-D10 kept: a keyed server's 401 (the backend sends no key), a 403 or a 404 is an
    answer that states no window, so the run is pushed on 8,192 ``assumed`` with its warning."""
    generic.status = status
    run_id = world.queue("ep-vllm")
    worker = FakeWorker()
    assert _drive(world, worker, T0) == [f"pushed:{run_id}"]
    row = world.run(run_id)
    assert (row["window_tokens"], row["window_source"]) == (8192, "assumed")
    assert worker.pushed is not None
    assert worker.pushed["spec"]["editor"]["window_warnings"] == ["window_assumed"]


def test_d1_a_listed_model_without_a_window_is_assumed(
    world: World, generic: GenericServer
) -> None:
    generic.status, generic.window = 200, None  # a LoRA card: max_model_len null
    run_id = world.queue("ep-vllm")
    assert _drive(world, FakeWorker(), T0) == [f"pushed:{run_id}"]
    assert world.run(run_id)["window_source"] == "assumed"


def test_d1_a_push_whose_read_gets_no_answer_requeues(world: World) -> None:
    run_id = world.queue("ep-vllm")
    worker = FakeWorker()
    answers = iter([16_384, window_read.NOT_ANSWERED])
    assert _drive(world, worker, T0, read_window=lambda e: next(answers)) == [f"requeued:{run_id}"]
    assert world.run(run_id)["status"] == "queued" and worker.pushed is None


def test_d1_the_real_push_read_that_gets_no_answer_requeues_and_sends_nothing(
    world: World, generic: GenericServer
) -> None:
    """The production reader at both reads: the claim's read is answered (32,768), then the server
    stops answering (503) before the push reads it, so the push puts the run back in the queue,
    pushes nothing and starts a new wait; the next pass recycles the consumed boot and the run
    waits on the no-answer clock, then fails with the no-answer message."""
    wait = world.settings.wsp_editor_resident_wait_s
    generic.status, generic.answers_left = 200, 1
    run_id = world.queue("ep-vllm")
    worker = FakeWorker()
    assert _drive(world, worker, T0) == [f"requeued:{run_id}"]
    # The claim's read (answered), then the push's: /v1/models and /props, both 503.
    assert generic.paths == ["/v1/models", "/v1/models", f"/props?model={generic.MODEL}"]
    row = world.run(run_id)
    assert (row["status"], row["boot_id"], row["window_source"]) == ("queued", None, None)
    assert _at(row["editor_wait_since"]) == T0 and worker.pushed is None
    actions = _drive(world, worker, T0 + dt.timedelta(seconds=5))
    assert f"recycled:{worker.boot}" in actions and world.run(run_id)["status"] == "queued"
    worker.boot = "f" * 32  # the recycled worker comes back with a new boot
    assert _drive(world, worker, T0 + dt.timedelta(seconds=30)) == [f"editor_not_resident:{run_id}"]
    assert _drive(world, worker, T0 + dt.timedelta(seconds=wait)) == [
        f"refused:{run_id}:editor_not_resident"
    ]
    failure = _j(world.run(run_id)["failure"])
    assert "gave no answer to the window read" in failure["message"]
    assert worker.pushed is None


def test_d1_the_endpoint_list_says_which_case_applies() -> None:
    def endpoint(serving: str, stated: int | None = None, url: str = "") -> WspEndpoint:
        return WspEndpoint(
            id="ep",
            kind="openai_compatible",
            base_url=url or "http://miserver-backend.miserver.svc:8000/v1",
            model="m",
            locality="external" if url.startswith("https://") else "local",
            output_cap_field="max_tokens",
            serving=serving,
            context_window_tokens=stated,
        )

    fails = endpoints.window_facts(endpoint("generic"), None)["window_if_read_fails"]
    assert fails == {"tokens": 8192, "source": "assumed", "no_answer": "waits"}
    fails = endpoints.window_facts(endpoint("miserver"), None)["window_if_read_fails"]
    assert fails == {"tokens": 0, "source": "waits", "no_answer": "waits"}
    fails = endpoints.window_facts(endpoint("generic", 32_768), None)["window_if_read_fails"]
    assert fails == {"tokens": 32_768, "source": "stated", "no_answer": "stated"}
    hosted = endpoint("generic", None, "https://api.anthropic.com")
    assert endpoints.window_facts(hosted, None)["window_if_read_fails"]["no_answer"] == "not_read"


def test_d1_the_production_pass_waits_for_a_generic_server_that_does_not_answer(
    world: World, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Through ``wsp.tasks.drive``: the pass the beat runs uses the reader that tells no answer
    from no window (``read_window_answer``); with the folding reader the run would be pushed on
    an assumed window (control C24)."""
    from tests.integration.wsp.test_dispatch import _drive_task

    run_id = world.queue("ep-down")
    worker, answer = _drive_task(world, monkeypatch)
    assert answer["actions"] == [f"editor_not_resident:{run_id}"], answer
    assert worker.pushed is None and world.run(run_id)["status"] == "queued"


@pytest.mark.parametrize(
    ("serving", "read", "waits"),
    [
        ("generic", window_read.NOT_ANSWERED, True),
        ("generic", None, False),
        ("generic", 8192, False),
        ("miserver", window_read.NOT_ANSWERED, True),
        ("miserver", None, True),
        ("miserver", 8192, False),
    ],
)
def test_d1_the_wait_rule(serving: str, read: Any, waits: bool) -> None:
    endpoint = WspEndpoint(
        id="ep",
        kind="openai_compatible",
        base_url="http://miserver-backend.miserver.svc:8000/v1",
        model="m",
        locality="local",
        output_cap_field="max_tokens",
        serving=serving,
        context_window_tokens=None,
    )
    assert window_read.waits(endpoint, "session", read) is waits
    assert window_read.waits(endpoint, "replay", read) is False


# ------------------------------------- D5 (2026-11-28): a server that gives no answer is held


def _timed(world: World, worker: Any, at: dt.datetime, **kwargs: Any) -> tuple[list[str], float]:
    started = time.monotonic()
    actions = _drive(world, worker, at, **kwargs)
    return actions, time.monotonic() - started


def test_d5_a_black_holed_server_is_read_once_per_hold_not_once_per_pass(
    world: World, generic: GenericServer, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A server that accepts the connection and never answers (the read times out): without the
    hold every 5 s pass paid the timeout and logged ``wsp_window_unread``. With it the claim reads
    at T0, then at +10 (hold 10 s), then at +30 (hold 20 s); every pass between makes no request
    and pays no timeout, and the run waits as before."""
    timeout = 1.0
    monkeypatch.setattr(window_read, "READ_TIMEOUT_S", timeout)
    generic.status, generic.hang_s = 200, timeout * 3
    run_id = world.queue("ep-vllm")
    worker, backoff = FakeWorker(), window_read.NoAnswerBackoff()
    waiting = [f"editor_not_resident:{run_id}"]
    expected = [(0, 1), (5, 0), (9.9, 0), (10, 1), (15, 0), (25, 0), (29.9, 0), (30, 1)]
    for offset, requests in expected:
        before = len(generic.paths)
        actions, took = _timed(world, worker, T0 + dt.timedelta(seconds=offset), backoff=backoff)
        assert actions == waiting, offset
        # A timed-out /v1/models is the origin not answering: /props is not asked (D1).
        assert len(generic.paths) - before == requests, (offset, generic.paths)
        if requests:
            assert took >= timeout, (offset, took)
        else:
            assert took < timeout, (offset, took)
    assert generic.paths == ["/v1/models"] * 3
    assert world.run(run_id)["status"] == "queued" and worker.pushed is None


def test_d5_a_recovered_server_is_read_within_the_longest_hold(
    world: World, generic: GenericServer
) -> None:
    """The hold doubles to at most 60 s, and any answer ends it: a server that comes back is
    read, and its run claimed, at most 60 s later, on the window it reports."""
    run_id = world.queue("ep-vllm")  # the fixture's server answers 503: no answer (D1)
    worker, backoff = FakeWorker(), window_read.NoAnswerBackoff()
    for offset in (0, 10, 30, 70, 130):  # holds of 10, 20, 40, 60, then 60 s
        reads = len(generic.paths)
        assert _drive(world, worker, T0 + dt.timedelta(seconds=offset), backoff=backoff) == [
            f"editor_not_resident:{run_id}"
        ]
        assert len(generic.paths) > reads, offset
    generic.status, generic.window = 200, 65_536  # the server comes back at +131
    reads = len(generic.paths)
    assert _drive(world, worker, T0 + dt.timedelta(seconds=189), backoff=backoff) == [
        f"editor_not_resident:{run_id}"
    ]
    assert len(generic.paths) == reads  # still held: 60 s from +130
    assert _drive(world, worker, T0 + dt.timedelta(seconds=190), backoff=backoff) == [
        f"pushed:{run_id}"
    ]
    row = world.run(run_id)
    assert (row["window_tokens"], row["window_source"]) == (65_536, "reported")


def test_d5_the_wait_still_ends_on_time_while_held(world: World) -> None:
    """A held endpoint's run waits on its last no answer, so at ``wsp_editor_resident_wait_s`` it
    fails ``editor_not_resident`` even though that pass made no read."""
    wait = world.settings.wsp_editor_resident_wait_s
    reads: list[str] = []

    def counting(endpoint: WspEndpoint) -> window_read.WindowAnswer:
        reads.append(endpoint.id)
        return window_read.read_window_answer(endpoint)

    run_id = world.queue("ep-down")
    worker, backoff = FakeWorker(), window_read.NoAnswerBackoff()
    for offset in (0, wait - 5):
        assert _drive(
            world, worker, T0 + dt.timedelta(seconds=offset), backoff=backoff, read_window=counting
        ) == [f"editor_not_resident:{run_id}"]
    assert reads == ["ep-down", "ep-down"]
    assert _drive(
        world, worker, T0 + dt.timedelta(seconds=wait), backoff=backoff, read_window=counting
    ) == [f"refused:{run_id}:editor_not_resident"]
    assert reads == ["ep-down", "ep-down"]  # held: the failure came without a third read
    failure = _j(world.run(run_id)["failure"])
    assert "gave no answer to the window read for 900 s" in failure["message"]


def test_d5_the_hold_doubles_to_its_cap_and_any_answer_ends_it() -> None:
    backoff = window_read.NoAnswerBackoff()
    ep = WspEndpoint(id="ep", base_url="http://127.0.0.1:1/v1", model="m", serving="generic")
    at = T0
    holds = []
    for _ in range(6):
        backoff.record(ep, window_read.NOT_ANSWERED, at)
        hold = next(s for s in range(1, 120) if not backoff.held(ep, at + dt.timedelta(seconds=s)))
        holds.append(hold)
        at += dt.timedelta(seconds=hold)
    assert holds == [10, 20, 40, 60, 60, 60]
    assert (window_read.NO_ANSWER_HOLD_S, window_read.NO_ANSWER_HOLD_MAX_S) == (10.0, 60.0)
    backoff.record(ep, window_read.NOT_ANSWERED, at)
    moved = WspEndpoint(id="ep", base_url="http://127.0.0.1:2/v1", model="m", serving="generic")
    assert backoff.held(ep, at) and not backoff.held(moved, at)  # a changed address is read
    for answer in (None, 32_768):
        backoff.record(ep, window_read.NOT_ANSWERED, at)
        backoff.record(ep, answer, at)
        assert not backoff.held(ep, at)
    backoff.record(ep, window_read.NOT_ANSWERED, at)  # after an answer it starts again at 10 s
    assert backoff.held(ep, at + dt.timedelta(seconds=9))
    assert not backoff.held(ep, at + dt.timedelta(seconds=10))


def test_d5_the_production_pass_holds_a_server_that_does_not_answer(
    world: World, generic: GenericServer, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Through ``wsp.tasks.drive`` (its real clock): two passes in a row read a 503 server once
    (``/v1/models`` and ``/props``), not twice: the beat's pass carries ``tasks.CLAIM_BACKOFF``."""
    from tests.integration.wsp.test_dispatch import _drive_task
    from wsp import tasks

    monkeypatch.setattr(tasks, "CLAIM_BACKOFF", window_read.NoAnswerBackoff())
    run_id = world.queue("ep-vllm")
    for _ in range(2):
        worker, answer = _drive_task(world, monkeypatch)
        assert answer["actions"] == [f"editor_not_resident:{run_id}"], answer
    assert generic.paths == ["/v1/models", f"/props?model={generic.MODEL}"]
    assert worker.pushed is None and world.run(run_id)["status"] == "queued"


# ----------------------------- D3 (2026-11-28): the wait is on the run's row, progress and MCP read


def test_d3_a_waiting_run_says_so_in_its_row_its_progress_and_forge_wsp_get_run(
    world: World, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A run waiting for its editor model carries ``editor_wait_since`` (when the claim first read
    no window) in its list row, its record and its progress read, and the MCP read hands the real
    route's answer on, typed by the answer contract. A run that never waited serves null."""
    import asyncio

    import httpx
    import jsonschema

    from mcp_server.config import MCPSettings
    from mcp_server.health_gate import HealthGate
    from mcp_server.server import OUTPUT_SCHEMAS_PATH, build_server
    from tests.integration.wsp.test_run_cause import _client

    waiting = world.queue("ep-stub")
    other = world.queue("ep-gen")  # states its window: claimed, never waits
    assert _drive(world, FakeWorker(), T0) == [
        f"editor_not_resident:{waiting}",
        f"pushed:{other}",
    ]
    api = _client(world, monkeypatch)
    since = "2026-11-27T20:21:00Z"
    rows = {r["run_id"]: r for r in api.get("/api/v1/wsp/runs").json()["runs"]}
    assert (rows[waiting]["editor_wait_since"], rows[other]["editor_wait_since"]) == (since, None)
    progress = api.get(f"/api/v1/wsp/runs/{waiting}/progress").json()
    assert (progress["status"], progress["editor_wait_since"]) == ("queued", since)
    assert api.get(f"/api/v1/wsp/runs/{other}/progress").json()["editor_wait_since"] is None
    record = api.get(f"/api/v1/wsp/runs/{waiting}/record").json()
    assert record["row"]["editor_wait_since"] == since

    async def open_gate(self: Any, *_: str) -> None:
        return None

    monkeypatch.setattr(HealthGate, "unavailable", open_gate)
    mcp, client = build_server(MCPSettings(auth_token="x" * 32))
    client._http = httpx.AsyncClient(
        transport=httpx.ASGITransport(app=api.app),
        base_url="http://forge.test/api/v1",
        headers={"X-Miforge-Agent": client.identity},
    )
    answer = asyncio.run(mcp.call_tool("forge_wsp_get_run", {"run_id": waiting}))
    served = answer.structured_content
    assert served is not None and served["row"]["editor_wait_since"] == since
    schemas = json.loads(OUTPUT_SCHEMAS_PATH.read_text(encoding="utf-8"))
    schema = schemas["typed"]["forge_wsp_get_run"]
    jsonschema.Draft202012Validator(schema).validate(served)
    # Typed, and always sent: the contract's row requires it, as a string or null.
    row = schema["$defs"]["WspRunRow"]
    assert "editor_wait_since" in row["required"]
    assert row["properties"]["editor_wait_since"] == {
        "anyOf": [{"type": "string"}, {"type": "null"}]
    }
