"""F11's fake weights surface answers in miLLM's REAL response shapes  (suite decision D39).

**Why this file exists.** On 2026-10-17, before the first live F11 call, the deployed F11 client
was found unable to drive the real surface on four counts: it never unwrapped miLLM's
``{success, data, error}`` envelope, it read a CUDA IPC handle off every *parameter* while miLLM
puts them on ``storages[]``, it read a status holder miLLM never sends, and it required a
``config`` the attach response did not carry. Every F11 test was green throughout, because the
fake surface they ran against had been written from prose and agreed with the reader by
construction.

**The authority is miLLM's code, not a copy of it.** ``docs/schemas/millm-weights-api-v1.json``
is GENERATED in miLLM from its registered ``/api/weights/*`` routes' ``response_model``
declarations (``millm/api/schemas/weights_contract.py``) and pinned there by a test that
regenerates it. miForge vendors it byte-identical (``test_schema_sync.py``; skips loudly when
miLLM is not checked out, fails under ``MIFORGE_REQUIRE_CROSS_REPO_CHECKS=1``). Here, every
response the fake gives on the wire - successes **and** refusals, across the states F11 drives
it through - is validated against that contract.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import httpx
import pytest

jsonschema = pytest.importorskip("jsonschema")

from tests.support.fake_weights_surface import ATTACHMENT_HEADER, FakeWeightsSurface  # noqa: E402
from training.hotupdate.client import OPS  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[3]
CONTRACT = json.loads(
    (REPO_ROOT / "docs" / "schemas" / "millm-weights-api-v1.json").read_text("utf-8")
)


def _validator(method: str, path: str) -> Any:
    route = f"{method} {path}"
    assert route in CONTRACT["responses"], f"{route} is not a route miLLM registers"
    schema = {**CONTRACT["responses"][route], "$defs": CONTRACT["$defs"]}
    return jsonschema.Draft202012Validator(schema)


class _Recorder:
    """Every wire response the fake gives, with the route it answered."""

    def __init__(self, fake: FakeWeightsSurface) -> None:
        self.fake = fake
        self.seen: list[tuple[str, str, int, Any]] = []
        self.http = httpx.Client(transport=fake.transport(), base_url="http://fake")
        self.attachment: str | None = None

    def call(self, op: str, body: Any = None) -> httpx.Response:
        method, path, _ = OPS[op]
        headers = {"Authorization": f"Bearer {self.fake.valid_token}"}
        if self.attachment:
            headers[ATTACHMENT_HEADER] = self.attachment
        response = self.http.request(method, path, json=body, headers=headers)
        self.seen.append((method, path, response.status_code, response.json()))
        return response


def _drive(recorder: _Recorder) -> None:
    """The states F11 drives the surface through: idle, attached, quiesced, writing, torn,
    late commit-end, save, load, lineage, a stale foreign holder, and refusals."""
    r = recorder
    r.call("status")
    attach = r.call("attach").json()["data"]
    r.attachment = attach["attachment_id"]
    r.call("heartbeat")
    r.call("status")
    r.call("quiesce")
    r.call("status")
    r.call("commit-begin")
    r.call("status")
    r.call("commit-end")
    r.call("resume")
    r.call("save", {"manifest": {"run_id": "r1", "step": 1}})
    r.call("lineage")
    r.call("quiesce")
    r.call("commit-begin")
    r.fake.trip_commit_watchdog()
    r.call("status")  # torn, commit still open
    r.call("commit-end")  # late, overran
    r.call("resume")
    r.call("load", {"checkpoint": "step-000001"})
    r.call("commit-end")  # refused: no open commit
    r.call("detach")
    r.attachment = None
    r.fake.make_stale("trainer:the-dead-worker")
    r.call("status")
    r.call("attach")  # refused: ALREADY_ATTACHED, stale


@pytest.fixture(scope="module")
def recorded() -> list[tuple[str, str, int, Any]]:
    recorder = _Recorder(FakeWeightsSurface())
    _drive(recorder)
    return recorder.seen


def test_the_scenario_reached_every_op_the_client_sends(recorded) -> None:  # type: ignore[no-untyped-def]
    """A conformance check over a subset would pass while an unexercised op drifted."""
    called = {(method, path) for method, path, _, _ in recorded}
    assert {(m, p) for m, p, _ in OPS.values()} <= called


def test_every_op_the_client_sends_is_a_route_miLLM_registers() -> None:
    missing = [f"{m} {p}" for m, p, _ in OPS.values() if f"{m} {p}" not in CONTRACT["responses"]]
    assert missing == []


def test_the_scenario_produced_refusals_too(recorded) -> None:  # type: ignore[no-untyped-def]
    assert any(status >= 400 for _, _, status, _ in recorded)


def test_every_response_validates_against_miLLMs_contract(recorded) -> None:  # type: ignore[no-untyped-def]
    failures = []
    for method, path, status, body in recorded:
        errors = sorted(_validator(method, path).iter_errors(body), key=str)
        if errors:
            failures.append(f"{method} {path} -> {status}: {errors[0].message}")
    assert failures == []


def test_success_carries_an_object_and_a_refusal_carries_an_error(recorded) -> None:  # type: ignore[no-untyped-def]
    """The envelope's `data` is nullable in the schema, so state the two halves directly."""
    for method, path, status, body in recorded:
        if status < 400:
            assert body["success"] is True and isinstance(body["data"], dict), (method, path)
            assert body["error"] is None
        else:
            assert body["success"] is False and body["data"] is None, (method, path)
            assert isinstance(body["error"]["code"], str)
