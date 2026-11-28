"""The bake-off harness's load-bearing lines (feature 016; FTASKS 10.2, 12.2; FR-016.24).

``scripts/wsp_bakeoff/`` is throwaway and never built by CI, but ``0xcc/reviews/016_bakeoff.md``
states results that rest on two of its modules, both numpy, torch and the standard library only, so
they run here without either comparator library:

- ``bakeoff/common.py``: the ONE bundle conversion (every file's hash checked, then the conversion
  proven against the bundle's parity scores: "conversion parity 0.0 over 8 rows"), the split (the
  stub scores the bundle's validation rows, which neither editor sees), the stub ``evaluate``
  (refuses a change to a frozen tensor; its verdict follows the sign of the difference), and the
  record's endpoint (no key, no header values).
- ``bakeoff/proxy.py``: the metering proxy ("library = proxy" tokens; "refused or steered answers:
  none"). It forwards only ``POST /v1/chat/completions`` and ``GET /v1/models``, refuses streaming,
  sends the endpoint's model, and adds the gateway's headers and body (``X-miLLM-Load-Policy:
  refuse``, an empty steering set: FTDD P28, P47) to every forwarded call.

12.2 mutated each of these lines and none was caught by any test (all fourteen survived); this file
is the result, re-run against each mutation as its negative control
(``0xcc/reviews/016_P1_review.md``, Round 3). The two library runners (``run_wsplab.py``,
``run_dspy_wsp.py``) import their library and are not tested here.

The public mirror strips ``scripts/`` (sync-to-clean.yml), so this file skips on the mirror's
snapshot only (``mirror_snapshot()``); the private CI is the gate and runs it.
"""

from __future__ import annotations

import json
import sys
import threading
import urllib.error
import urllib.request
from collections.abc import Iterator
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from types import ModuleType
from typing import Any

import numpy as np
import pytest

from tests.support.mirror_snapshot import mirror_snapshot
from tests.support.wsp_fixtures import make_bundle, reference_row_score

ROOT = Path(__file__).resolve().parents[4]
HARNESS = ROOT / "scripts" / "wsp_bakeoff"
#: make_fixture.py's parameters: the bundle the dry run and the real run used.
PARAMS: dict[str, Any] = {"rule": "mean", "d_model": 8, "rows": 64, "seed": 0}

pytestmark = pytest.mark.skipif(
    mirror_snapshot(ROOT), reason="the public mirror strips scripts/; checked in the private repo"
)


@pytest.fixture(scope="module")
def harness() -> Iterator[tuple[ModuleType, ModuleType]]:
    sys.path.insert(0, str(HARNESS))
    try:
        from bakeoff import common, proxy

        yield common, proxy
    finally:
        sys.path.remove(str(HARNESS))


@pytest.fixture()
def bundle_dir(tmp_path: Path) -> Path:
    root = tmp_path / "bundle"
    make_bundle(root, **PARAMS).write_manifest()
    return root


def _manifest(root: Path) -> dict[str, Any]:
    loaded: dict[str, Any] = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    return loaded


# ---------------------------------------------------------------------------- the conversion


def test_the_conversion_is_exact_at_parity_on_the_bake_off_bundle(harness, bundle_dir, tmp_path):
    common, _ = harness
    task = common.convert_bundle(bundle_dir, tmp_path / "converted")
    assert task.parity["rows"] == 8 and task.parity["max_abs_error"] <= 1e-5, task.parity
    assert sorted(task.editable) == ["bias", "weight"] and len(task.labels) == 64


def test_a_file_whose_hash_differs_from_the_manifest_is_refused(harness, bundle_dir, tmp_path):
    """The labels do not enter the parity check, so only the hash check can catch this."""
    common, _ = harness
    entry = next(f for f in _manifest(bundle_dir)["files"] if f["role"] == "practice_labels")
    path = bundle_dir / entry["name"]
    labels = np.load(path)
    np.save(path, 1 - labels)
    with pytest.raises(ValueError, match="sha256 differs from the manifest"):
        common.convert_bundle(bundle_dir, tmp_path / "converted")


def test_a_conversion_that_misses_the_bundles_parity_scores_is_refused(
    harness, bundle_dir, tmp_path
):
    common, _ = harness
    manifest = _manifest(bundle_dir)
    parity = manifest["function"]["parity"]
    parity["scores"] = [s + 1.0 for s in parity["scores"]]
    (bundle_dir / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(ValueError, match="parity check"):
        common.convert_bundle(bundle_dir, tmp_path / "converted")


def test_the_stub_scores_the_validation_rows_and_the_editor_never_sees_them(
    harness, bundle_dir, tmp_path
):
    common, _ = harness
    task = common.convert_bundle(bundle_dir, tmp_path / "converted")
    entry = next(
        f for f in _manifest(bundle_dir)["files"] if f["role"] == "practice_validation_mask"
    )
    mask = np.load(bundle_dir / entry["name"]).astype(bool)
    assert task.heldout_rows == [int(i) for i in np.flatnonzero(mask)]
    assert task.practice_rows == [int(i) for i in np.flatnonzero(~mask)]
    assert len(task.heldout_rows) == 16 and not set(task.heldout_rows) & set(task.practice_rows)


# ---------------------------------------------------------------------------- the stub evaluate


def test_the_stub_refuses_a_change_to_a_frozen_tensor(harness, bundle_dir, tmp_path):
    common, _ = harness
    task = common.convert_bundle(bundle_dir, tmp_path / "converted")
    edited = common.merge_edits(task, {"mean": task.heads["mean"] + np.float32(1.0)})
    with pytest.raises(ValueError, match="frozen tensor"):
        common.stub_evaluate(task, edited)


def test_the_stub_verdict_follows_the_sign_of_the_difference(harness, bundle_dir, tmp_path):
    """Accuracy on the held-out rows, recomputed here from the fixture's pure-Python reference
    (not the harness's torch head): an all-fire and a never-fire edit."""
    common, _ = harness
    task = common.convert_bundle(bundle_dir, tmp_path / "converted")
    fixture = make_bundle(tmp_path / "again", **PARAMS)
    t = fixture.tensors
    weight, mean, std = (list(map(float, t[n].values64())) for n in ("weight", "mean", "std"))
    bias = float(t["bias"].values64()[0])

    def score(row: int) -> float:
        tokens = fixture.activations[fixture.offsets[row] : fixture.offsets[row + 1]]
        toks = [[float(x) for x in tok] for tok in tokens.astype(np.float64)]
        return reference_row_score("mean", toks, weight, bias, mean, std, None, 1.0, 16)

    held = task.heldout_rows
    positive = np.asarray(fixture.labels)[held] == 1
    base = float(
        np.mean([(score(r) >= task.threshold) == p for r, p in zip(held, positive, strict=True)])
    )
    verdicts = set()
    for shift, accuracy in ((1000.0, float(positive.mean())), (-1000.0, float((~positive).mean()))):
        edited = common.merge_edits(task, {"bias": task.heads["bias"] + shift})
        result = common.stub_evaluate(task, edited)
        assert result["base_accuracy"] == pytest.approx(base) and result["changed_tensors"] == [
            "bias"
        ]
        assert result["edited_accuracy"] == pytest.approx(accuracy)
        assert result["difference"] == pytest.approx(accuracy - base)
        sign = np.sign(accuracy - base)
        assert result["verdict"] == {1: "improved", -1: "regressed", 0: "unchanged"}[int(sign)]
        verdicts.add(result["verdict"])
    assert verdicts & {"improved", "regressed"}, "both edits left the accuracy unchanged"


def test_the_records_endpoint_carries_no_key_and_no_header_value(harness):
    common, _ = harness
    endpoint = common.Endpoint(
        base_url="http://editor/v1",
        model="m",
        api_key="sk-planted-key",
        extra_headers={"X-miLLM-Load-Policy": "planted-header-value"},
        extra_body={},
    )
    public = json.dumps(endpoint.public())
    assert "sk-planted-key" not in public and "planted-header-value" not in public, public
    assert "X-miLLM-Load-Policy" in public


# ---------------------------------------------------------------------------- the proxy


class _Upstream:
    """A stand-in OpenAI-compatible endpoint that records what reached it."""

    def __init__(self) -> None:
        self.seen: list[dict[str, Any]] = []
        self.answer_headers: dict[str, str] = {"X-miLLM-Steering": "none"}
        upstream = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *args: Any) -> None:
                pass

            def _answer(self, body: dict[str, Any]) -> None:
                raw = json.dumps(body).encode()
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(raw)))
                for k, v in upstream.answer_headers.items():
                    self.send_header(k, v)
                self.end_headers()
                self.wfile.write(raw)

            def do_GET(self) -> None:
                upstream.seen.append({"method": "GET", "path": self.path})
                self._answer({"data": [{"id": "m"}]})

            def do_POST(self) -> None:
                raw = self.rfile.read(int(self.headers.get("Content-Length") or 0))
                upstream.seen.append(
                    {
                        "method": "POST",
                        "path": self.path,
                        "headers": {k.lower(): v for k, v in self.headers.items()},
                        "body": json.loads(raw),
                    }
                )
                self._answer(
                    {
                        "choices": [{"message": {"content": "ok"}, "finish_reason": "stop"}],
                        "usage": {"prompt_tokens": 11, "completion_tokens": 7},
                    }
                )

        self.server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()

    @property
    def base_url(self) -> str:
        return f"http://127.0.0.1:{self.server.server_address[1]}/v1"

    def close(self) -> None:
        self.server.shutdown()
        self.server.server_close()


@pytest.fixture()
def proxied(harness, tmp_path) -> Iterator[tuple[Any, _Upstream]]:
    common, proxy = harness
    upstream = _Upstream()
    endpoint = common.Endpoint(
        base_url=upstream.base_url,
        model="Qwen2.5-7B-Instruct",
        api_key="local",
        extra_headers={"X-miLLM-Origin": "judge", "X-miLLM-Load-Policy": "refuse"},
        extra_body={"steering": {"features": []}},
    )
    try:
        with proxy.MeteringProxy(endpoint, tmp_path / "proxy.jsonl") as running:
            yield running, upstream
    finally:
        upstream.close()


def _call(url: str, body: dict[str, Any] | None = None) -> int:
    data = None if body is None else json.dumps(body).encode()
    request = urllib.request.Request(
        url,
        data=data,
        headers={"Content-Type": "application/json"},
        method="GET" if data is None else "POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=10) as answer:
            return int(answer.status)
    except urllib.error.HTTPError as refused:
        return int(refused.code)


CHAT = {"model": "the-librarys-own-name", "messages": [{"role": "user", "content": "hi"}]}


def test_a_chat_call_is_forwarded_with_the_endpoints_model_headers_and_body(proxied):
    running, upstream = proxied
    assert _call(running.base_url + "/chat/completions", CHAT) == 200
    (seen,) = upstream.seen
    assert seen["path"] == "/v1/chat/completions"
    assert seen["body"]["model"] == "Qwen2.5-7B-Instruct"
    assert seen["body"]["steering"] == {"features": []}
    assert seen["headers"]["x-millm-load-policy"] == "refuse"
    assert seen["headers"]["x-millm-origin"] == "judge"


def test_the_proxy_counts_tokens_from_the_endpoints_usage(proxied):
    running, _ = proxied
    for _ in range(2):
        assert _call(running.base_url + "/chat/completions", CHAT) == 200
    totals = running.meter.totals()
    assert (totals["calls"], totals["prompt_tokens"], totals["completion_tokens"]) == (2, 22, 14)
    assert totals["steered_answers"] == [] and totals["refused_requests"] == []


def test_a_steered_answer_is_recorded(proxied):
    running, upstream = proxied
    upstream.answer_headers = {"X-miLLM-Steering": "cluster:planted"}
    assert _call(running.base_url + "/chat/completions", CHAT) == 200
    assert running.meter.totals()["steered_answers"] == ["cluster:planted"]


@pytest.mark.parametrize(
    ("path", "body"),
    [
        ("/v1/responses", CHAT),
        ("/api/health", None),
        ("/v1/chat/completions", {**CHAT, "stream": True}),
    ],
    ids=["post-responses-api", "get-other-path", "streaming"],
)
def test_anything_else_is_refused_recorded_and_never_forwarded(proxied, path, body):
    running, upstream = proxied
    assert _call(running.base_url.removesuffix("/v1") + path, body) == 400
    assert upstream.seen == []
    assert len(running.meter.totals()["refused_requests"]) == 1
