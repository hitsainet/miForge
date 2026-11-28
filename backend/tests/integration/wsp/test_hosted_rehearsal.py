"""H-15: the whole-run hosted rehearsal (feature 020; FR-020.23, FR-020.24, FR-020.27, FR-020.28;
F20-D11, F20-D20; TDD §6.7). Permanent, in CI on every push, with no network and no key.

The full topology: the REAL driver pushes to the REAL supervisor (``LocalProcessLauncher`` under
the worker's interpreter), whose child runs the REAL ``miwsp_engine``; the child's gateway address
is the REAL gateway (``create_gateway_app``), served over HTTP on a loopback port
(``tests/support/gateway_server.py``), with its real internal application and sealed keys; and the
gateway's upstream is the scripted ``api.anthropic.com`` and hosted OpenAI-compatible server
(``tests/support/hosted_upstream.py``), which refuse what the Messages API refuses.

The runs (TDD §6.7):
1. Anthropic ``create`` at effort ``medium``: thinking, caching, a handoff, a recall across it, a
   lost reply, the final turn (another tool refused, then the retry) and a submit;
2. run 3's accepted edit, then a cost cap the worst-case reservation reaches first:
   ``cost_budget``, and the edit kept as a candidate (FR-020.18's contrast);
3. Anthropic, a refusal on turn 4 after an accepted edit: ``editor_refused``, no candidate;
4. the replay of run 1: it reproduces;
5. OpenAI-compatible: ``reasoning_effort`` on every request and a replay; then a refusal;
6. a sensitive bundle on the hosted endpoint: refused at the start, the upstream sees nothing.

Every run is checked by TDD §6.7's list (:func:`check_run`). The figures FTASKS 8.5 records are
written to ``rehearsal_figures.json`` in the test's directory and printed.

Skips by name without the worker's interpreter or libseccomp; fails under
``MIFORGE_REQUIRE_WSP_WORKER=1`` (CI sets it, TD-9).
"""

from __future__ import annotations

import itertools
import json
import time
from collections.abc import Iterator
from decimal import Decimal
from pathlib import Path
from typing import Any

import httpx
import pytest
import sqlalchemy as sa
from fastapi.testclient import TestClient

from tests.integration.wsp.test_dispatch import World, _j, build_world
from tests.integration.wsp.test_internal_app_routes import SERVICE
from tests.integration.wsp.test_runs_lifecycle import _store
from tests.integration.wsp.wsp_db import NOW, insert
from tests.support import wsp_worker_env
from tests.support.fake_miworkbench import evaluate_heldout_sets
from tests.support.gateway_server import serve
from tests.support.hosted_upstream import ANTHROPIC, OPENAI, HostedUpstream, tokens
from tests.support.wsp_drive import drive_inline
from wsp import dispatch, runs, tasks
from wsp.settings import get_wsp_settings
from wsp_gateway import meter, sealing
from wsp_gateway.app import Gateway, create_gateway_app
from wsp_gateway.settings import GatewaySettings
from wsp_worker.miwsp_engine import prefix, prompts

OA_HOST = "llm.example.com"
#: The accepted edit runs 2 and 3 share: within every limit of a ``create``.
EDIT = "W['bias'][:] = R['bias'] * 2.0"
KEY = "sk-rehearsal-PLACEHOLDER-not-a-key"
#: A window small enough that one long result fills it: the next request hands off.
WINDOW = 24_000
PRICES = {"in": 3, "out": 15, "write": Decimal("3.75"), "read": Decimal("0.3")}
EDIT = "W['weight'][:] = R['weight'] * 2.0; W['bias'][:] = R['bias'] * 2.0; print('x' * 15000)"


class Stage:
    """The world, the served gateway, the two upstreams, and the figures."""

    def __init__(self, world: World, anth: HostedUpstream, oa: HostedUpstream) -> None:
        self.world, self.anth, self.oa = world, anth, oa
        self.figures: dict[str, Any] = {}

    def upstream(self, endpoint_id: str) -> HostedUpstream:
        return self.anth if endpoint_id == "ep-claude" else self.oa

    def start(self, endpoint_id: str, kind: str, steps: int, **budgets: Any) -> str:
        world = self.world
        world.body = {
            **world.body,
            "mode": "session",
            "budgets": {"tokens": 1_000_000, "time_s": 180, "max_steps": steps, **budgets},
        }
        return world.queue(endpoint_id, kind=kind)

    def drive(self) -> list[str]:
        """One pass that pushes, the session to its end, then the pass that collects (as
        ``test_end_to_end_probe._drive_real``)."""
        from wsp.launcher import LocalProcessLauncher
        from wsp.worker_client import HttpWorker

        world = self.world
        launcher = LocalProcessLauncher(environment="ci", python=wsp_worker_env.worker_python())
        with launcher.session() as local:
            client = HttpWorker(local.url)
            try:

                def once() -> list[str]:
                    return drive_inline(
                        world.factory,
                        client,
                        world.settings,
                        read_sensitivity=world.read,
                        files_for=tasks.run_files,
                        read_window=dispatch.no_window,
                    )

                actions = once()
                assert actions and actions[-1].startswith("pushed:"), actions
                deadline = time.monotonic() + 170
                while client.status()["state"] != "finished":
                    assert time.monotonic() < deadline, "the session never finished"
                    time.sleep(0.2)
                actions += once()
            finally:
                client.close()
            assert local.process.wait(timeout=30) == 0
        return actions

    def transcript(self, run_id: str) -> list[dict[str, Any]]:
        stored = runs.run_dir(self.world.settings, run_id) / "transcript.jsonl"
        return [json.loads(x) for x in stored.read_text(encoding="utf-8").splitlines()]

    def calls(self, run_id: str) -> list[dict[str, Any]]:
        with self.world.engine.connect() as conn:
            rows = conn.execute(
                sa.text("SELECT * FROM mf_wsp_usage WHERE run_id = :r ORDER BY seq"), {"r": run_id}
            ).mappings()
            return [dict(r) for r in rows]


def _endpoint(conn: sa.Connection, endpoint_id: str, **values: Any) -> None:
    row: dict[str, Any] = {
        "id": endpoint_id,
        "kind": "anthropic",
        "base_url": "https://api.anthropic.com",
        "model": "claude-rehearsal",
        "locality": "external",
        "price_in_per_mtok": PRICES["in"],
        "price_out_per_mtok": PRICES["out"],
        "price_cache_write_per_mtok": float(PRICES["write"]),
        "price_cache_read_per_mtok": float(PRICES["read"]),
        "output_cap_field": "max_tokens",
        "serving": "generic",
        "context_window_tokens": WINDOW,
        "thinking": "medium",
        "history_mode": "append_only",
        "created_at": NOW,
        "created_by": "operator",
    }
    row.update(values)
    insert(conn, "mf_wsp_endpoints", **row)


def _seal(world: World, public: str, endpoint_id: str) -> None:
    from api.wsp_internal_app import endpoint_document

    with world.factory() as session:
        row = runs.live_endpoint(session, endpoint_id)
        assert row is not None
        row.key_ciphertext = sealing.seal(
            public, KEY.encode(), sealing.aad_digest(endpoint_document(row))
        )
        row.key_fingerprint = sealing.fingerprint(KEY)
        session.commit()


@pytest.fixture()
def stage(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Iterator[Stage]:
    from api import wsp_internal_app
    from db import session as db_session
    from tests.integration.wsp.test_internal_app_routes import _long_sqlite_busy_timeout

    wsp_worker_env.require_seccomp()
    monkeypatch.setenv("WSP_EXTERNAL_HOST_ALLOWLIST", f"api.anthropic.com,{OA_HOST}")
    monkeypatch.setenv("WSP_GATEWAY_SERVICE_TOKEN", SERVICE)
    world = build_world(
        tmp_path,
        monkeypatch,
        bundle_id="wsb_0123456789ab",
        heldout_sets=evaluate_heldout_sets(),
        labels_from_scores=True,
    )
    _long_sqlite_busy_timeout(world.engine)
    _store(world, tmp_path)
    with world.engine.begin() as conn:
        _endpoint(conn, "ep-claude")
        _endpoint(
            conn,
            "ep-oa",
            kind="openai_compatible",
            base_url=f"https://{OA_HOST}/v1",
            model="oa-rehearsal",
            output_cap_field="max_completion_tokens",
            context_window_tokens=200_000,
        )
    private, public = sealing.generate_keypair()
    _seal(world, public, "ep-claude")
    _seal(world, public, "ep-oa")
    monkeypatch.setattr(db_session, "_engine", world.engine)
    monkeypatch.setattr(db_session, "_session_factory", None)
    wsp_internal_app._negative.clear()
    get_wsp_settings.cache_clear()
    internal = TestClient(wsp_internal_app.create_internal_app(), raise_server_exceptions=False)
    internal.headers.update({"Authorization": f"Bearer {SERVICE}"})
    anth, oa = HostedUpstream(ANTHROPIC), HostedUpstream(OPENAI)

    def route(request: httpx.Request) -> httpx.Response:
        return (oa if request.url.host == OA_HOST else anth).handle(request)

    gateway = Gateway(
        GatewaySettings(
            wsp_internal_url="http://testserver",
            wsp_gateway_service_token=SERVICE,
            wsp_gateway_private_key=private,
            wsp_external_host_allowlist=f"api.anthropic.com,{OA_HOST}",
        ),
        internal=internal,
        upstream_client=httpx.Client(transport=httpx.MockTransport(route)),
        resolver=lambda host: ["93.184.216.34"],
    )
    gateway.registry.rebuild()
    served = serve(create_gateway_app(gateway))
    monkeypatch.setenv("WSP_GATEWAY_URL", served.url)
    get_wsp_settings.cache_clear()
    world.settings = get_wsp_settings()
    built = Stage(world, anth, oa)
    try:
        yield built
    finally:
        served.stop()
        world.engine.dispose()
        get_wsp_settings.cache_clear()
        out = tmp_path / "rehearsal_figures.json"
        out.write_text(json.dumps(built.figures, indent=1, default=str))
        print("REHEARSAL FIGURES", json.dumps(built.figures, default=str))


# ----------------------------------------------------------------------------- the checks


def _static_tokens(body: dict[str, Any]) -> int:
    normal = prefix.normal(body)
    return tokens([{"system": normal["system"], "tools": normal["tools"]}, []])


def check_run(stage: Stage, run_id: str, upstream: HostedUpstream, first: int) -> dict[str, Any]:
    """TDD §6.7's list for the run whose requests start at ``upstream.requests[first]``."""
    seen = upstream.requests[first:]
    assert seen, "the upstream saw no request"
    assert all(s.status == 200 for s in seen), [s.message for s in seen if s.status != 200]
    # 1. The append-only rule on every request, read from the upstream's own records...
    by_conversation: dict[int, list[Any]] = {}
    for s in seen:
        by_conversation.setdefault(s.conversation, []).append(s)
    for convo in by_conversation.values():
        for a, b in itertools.pairwise(convo):
            assert prefix.check(a.body, b.body) is None
    # ... and from the gateway's stored digests: consecutive served calls of one conversation.
    every = stage.calls(run_id)
    # A call the gateway refused before its reservation (the cost cap) never reached the upstream.
    calls = [c for c in every if c["outcome"] != "refused"]
    refused = [c["refusal_code"] for c in every if c["outcome"] == "refused"]
    assert len(calls) == len(seen), (len(calls), len(seen))
    for (ca, sa_), (cb, sb) in zip(
        zip(calls, seen, strict=True), list(zip(calls, seen, strict=True))[1:], strict=False
    ):
        if sa_.conversation != sb.conversation or "outcome" not in ca:
            continue
        if ca["outcome"] == "served" and cb["outcome"] == "served":
            witnessed = meter.conversation_key(sb.body, ca["prompt_messages"])
            assert witnessed == ca["conversation_sha256"]
    # 2. Cache reads after the first request of each conversation (Anthropic).
    if upstream.personality == ANTHROPIC:
        for convo in by_conversation.values():
            static = _static_tokens(convo[0].body)
            served = [s for s in convo if s.usage]
            for previous, s in itertools.pairwise(served):
                total = sum(
                    previous.usage[k]
                    for k in (
                        "input_tokens",
                        "cache_read_input_tokens",
                        "cache_creation_input_tokens",
                    )
                )
                assert s.usage["cache_read_input_tokens"] >= total, (previous.usage, s.usage)
            # A conversation's first request reads nothing, or at least the static prefix (an
            # earlier run's identical opening can be in the cache, as it would be on the API).
            opening = served[0].usage["cache_read_input_tokens"]
            assert opening == 0 or opening >= static, (opening, static)
        assert all(s.body["cache_control"] == {"type": "ephemeral"} for s in seen)
        assert all(s.body["thinking"]["type"] == "adaptive" for s in seen)
    # 3. The cost is the sum of the calls; each served call is priced from the upstream's usage.
    cost = sum(Decimal(str(c["cost_usd"])) for c in every)
    with stage.world.factory() as session:
        totals = runs.metered_totals(session, [run_id])[run_id]
    assert Decimal(str(totals["cost_usd"])) == pytest.approx(cost, abs=Decimal("0.000002"))
    for c, s in zip(calls, seen, strict=True):
        if c["outcome"] == "served" and upstream.personality == ANTHROPIC:
            u = s.usage
            priced = (
                Decimal(u["input_tokens"]) * PRICES["in"]
                + Decimal(u["cache_read_input_tokens"]) * PRICES["read"]
                + Decimal(u["cache_creation_input_tokens"]) * PRICES["write"]
                + Decimal(u["output_tokens"]) * PRICES["out"]
            ) / 1_000_000
            assert Decimal(str(c["cost_usd"])) == pytest.approx(priced, abs=Decimal("0.000002"))
            assert (c["cache_read_tokens"], c["cache_write_tokens"]) == (
                u["cache_read_input_tokens"],
                u["cache_creation_input_tokens"],
            )
    # 4. One timing event per request; the figures.
    lines = stage.transcript(run_id)
    timing = [e for e in lines if e.get("event") == "timing"]
    assert len(timing) == len(every)  # one per request the engine made, refused ones too
    gaps = [e["gap_s"] for e in timing if e["gap_s"] is not None]
    assert all(g >= 0 for g in gaps)
    # 5. The run's document validates against the vendored contract (H-16, FR-020.27).
    from wsp import evaluate
    from wsp.schemas import validate_run

    with stage.world.factory() as session:
        validate_run(evaluate.document_of(session, runs.get_run(session, run_id)))
    return {
        "requests": len(seen),
        "conversations": len(by_conversation),
        "longest_request_s": max(e["duration_s"] for e in timing),
        "longest_gap_s": max(gaps, default=None),
        "cost_usd": str(cost),
        "refused_by_the_gateway": refused,
    }


# ----------------------------------------------------------------------------- the runs

RUN_1 = [
    {"calls": [("python", {"code": EDIT})]},  # an accepted edit, and a long result (page p1)
    # The next request no longer fits the window: the engine hands off before it.
    {"calls": [("recall", {"page": "p1", "offset": 12_000})]},  # across the handoff
    {"drop": True},  # a lost reply
    {"calls": [("list_tensors", {})]},  # the one retry, with the notice appended
    {"calls": [("read_tensor", {"name": "weight"})]},  # the final turn: not run
    {"calls": [("submit", {"answer": "programmed at twice the scale", "confidence": 0.6})]},
]


def test_the_hosted_rehearsal(stage: Stage) -> None:
    # ---------------------------------------------------------------- run 1
    stage.anth.scripts = list(RUN_1)
    run_1 = stage.start("ep-claude", "create", steps=4)
    actions = stage.drive()
    row = stage.world.run(run_1)
    assert f"collected:{run_1}:completed" in actions, (actions, row["failure"])
    assert (row["status"], row["outcome"], row["stop_reason"]) == (
        "completed",
        "candidate",
        "max_steps",
    ), row["failure"]
    figures = check_run(stage, run_1, stage.anth, 0)
    seen = stage.anth.requests
    assert len(seen) == 6 and figures["conversations"] == 2
    # The handoff: the second request starts a new conversation that reads the cached static
    # prefix (the system block's marker) and carries the summary with its page listing.
    assert seen[1].conversation != seen[0].conversation
    assert seen[1].usage["cache_read_input_tokens"] == _static_tokens(seen[1].body) > 0
    lines = stage.transcript(run_1)
    [handoff] = [e for e in lines if e.get("event") == "handoff"]
    assert handoff["conversation"] == 2
    recalled = [m for m in lines if m.get("role") == "tool"][1]["content"]
    assert recalled.endswith("\n[end of page p1]")
    # The lost reply was survived, the notice merged into the turn it was lost on.
    assert [e["code"] for e in lines if e.get("event") == "lost_reply"] == [
        "wsp_gateway_upstream_failed"
    ]
    assert prompts.LOST_REPLY in json.dumps(seen[3].body["messages"][-1])
    # The final turn kept every tool; the other tool it asked for ran nothing.
    names = [[t["name"] for t in s.body["tools"]] for s in seen]
    assert names[4] == names[5] == names[0]
    assert prompts.FINAL_SUBMIT_ONLY in json.dumps(seen[5].body["messages"][-1])
    # Effort medium and summarized adaptive thinking on every request; the version and no beta.
    assert {json.dumps(s.body.get("output_config")) for s in seen} == {'{"effort": "medium"}'}
    assert {s.headers["anthropic-version"] for s in seen} == {"2026-10-21"}
    stage.figures["run_1"] = figures

    # ---------------------------------------------------------------- run 2: the cost cap
    first = len(stage.anth.requests)
    # Each turn adds about 2 KB, which the worst case prices as cache writes at USD 3.75 per
    # MTok plus the output ceiling: the reservation outgrows a USD 0.06 cap long before the
    # metered spend (cache reads at USD 0.30) gets near it.
    #
    # FR-020.18's contrast (2026-11-25): its first turn makes run 3's accepted edit, within the
    # run's limits, so a budget stop keeps it as a partial patch (F18-D16 withholds only an edit
    # over a limit). The same edit ended by a refusal (run 3) keeps nothing: the cause, not the
    # session, decides.
    grow = "print('y' * 2000)"
    stage.anth.scripts = [{"calls": [("python", {"code": EDIT})]}] + [
        {"calls": [("python", {"code": f"{grow}; {n}"})]} for n in range(30)
    ]
    run_2 = stage.start("ep-claude", "create", steps=30, cost_usd=0.06)
    stage.drive()
    row = stage.world.run(run_2)
    assert row["stop_reason"] == "cost_budget", (row["stop_reason"], row["failure"])
    assert (row["status"], row["outcome"]) == ("completed", "candidate"), (
        row["outcome"],
        row["outcome_reason"],
        row["failure"],
    )
    assert row["patch_sha256"] is not None
    with stage.world.engine.connect() as conn:
        kept = conn.execute(
            sa.text(
                "SELECT patch_sha256, changed_entries, tensors FROM mf_wsp_candidates"
                " WHERE run_id = :r"
            ),
            {"r": run_2},
        ).all()
    [(patch_sha, changed, tensors)] = kept  # exactly one candidate
    assert patch_sha == row["patch_sha256"]
    assert changed > 0 and "bias" in json.dumps(_j(tensors)), (changed, tensors)
    figures = check_run(stage, run_2, stage.anth, first)
    figures["cost_cap_usd"] = 0.06
    figures["failure"] = _j(row["failure"])
    stage.figures["run_2"] = figures

    # ---------------------------------------------------------------- run 3: a refusal
    first = len(stage.anth.requests)
    stage.anth.scripts = [
        {"calls": [("python", {"code": EDIT})]},
        {"calls": [("list_tensors", {})]},
        {"calls": [("describe_tensor", {"name": "weight"})]},
        {"refusal": "cyber"},
        {"calls": [("submit", {"answer": "never asked for"})]},
    ]
    run_3 = stage.start("ep-claude", "create", steps=10)
    stage.drive()
    row = stage.world.run(run_3)
    rejected = [s.message for s in stage.anth.requests[first:] if s.status != 200]
    assert (row["status"], row["outcome"]) == ("failed", "failed"), (row["failure"], rejected)
    assert _j(row["failure"])["code"] == "editor_refused", (row["failure"], rejected)
    assert row["patch_sha256"] is None
    assert len(stage.anth.requests) - first == 4  # no nudge, no fifth request
    assert [e for e in stage.transcript(run_3) if e.get("event") == "refusal"] == [
        {"event": "refusal", "turn": 4, "category": "cyber"}
    ]
    with stage.world.engine.connect() as conn:
        assert (
            conn.execute(
                sa.text("SELECT COUNT(*) FROM mf_wsp_candidates WHERE run_id = :r"), {"r": run_3}
            ).scalar_one()
            == 0
        )
    stage.figures["run_3"] = check_run(stage, run_3, stage.anth, first)

    # ---------------------------------------------------------------- run 4: the replay of run 1
    before = len(stage.anth.requests)
    with stage.world.factory() as session:
        replay = runs.create_replay(session, run_1, "operator", stage.world.settings).run_id
        session.commit()
    actions = stage.drive()
    row = stage.world.run(replay)
    assert f"collected:{replay}:completed" in actions, (actions, row["failure"])
    assert row["replay_reproduced"] in (True, 1), (row["failure"], row["replay_reproduced"])
    assert len(stage.anth.requests) == before  # a replay sends nothing anywhere

    # ---------------------------------------------------------------- run 5: OpenAI-compatible
    # A result above the page threshold on turn 2, then four more turns in the same conversation:
    # a paging session would stub it out at turn 6 (review survivor R6a: the rehearsal's forced
    # page-out control needs a session where one is due).
    stage.oa.scripts = [
        {"calls": [("list_tensors", {})]},
        {"calls": [("python", {"code": EDIT.replace("15000", "3000")})]},
        {"calls": [("describe_tensor", {"name": "weight"})]},
        {"calls": [("describe_tensor", {"name": "bias"})]},
        {"calls": [("read_tensor", {"name": "bias"})]},
        {"calls": [("submit", {"answer": "done", "confidence": 0.5})]},
        {"calls": [("submit", {"answer": "done", "confidence": 0.5})]},
    ]
    run_5 = stage.start("ep-oa", "create", steps=10)
    stage.drive()
    row = stage.world.run(run_5)
    assert row["status"] == "completed", row["failure"]
    stage.figures["run_5"] = check_run(stage, run_5, stage.oa, 0)
    assert len(stage.oa.requests) >= 6
    assert {s.body.get("reasoning_effort") for s in stage.oa.requests} == {"medium"}
    assert all("cache_control" not in json.dumps(s.body) for s in stage.oa.requests)
    second = [s for s in stage.oa.requests if s.usage][1]
    assert second.usage["prompt_tokens_details"]["cached_tokens"] > 0
    with stage.world.factory() as session:
        replay_5 = runs.create_replay(session, run_5, "operator", stage.world.settings).run_id
        session.commit()
    stage.drive()
    assert stage.world.run(replay_5)["replay_reproduced"] in (True, 1)
    first = len(stage.oa.requests)
    stage.oa.scripts = [{"calls": [("list_tensors", {})]}, {"refusal": None}]
    run_5b = stage.start("ep-oa", "create", steps=10)
    stage.drive()
    row = stage.world.run(run_5b)
    assert _j(row["failure"])["code"] == "editor_refused"
    assert len(stage.oa.requests) - first == 2

    # ---------------------------------------------------------------- run 6: a sensitive bundle
    from core.errors import Refusal

    counted = len(stage.anth.requests) + len(stage.oa.requests)
    with stage.world.engine.connect() as conn:
        manifest = _j(conn.execute(sa.text("SELECT manifest FROM mf_wsp_bundles")).scalar_one())
    manifest = {**manifest, "bundle_id": "wsb_sensitive00001", "bundle_sha256": "e" * 64}
    manifest["sensitivity"] = {**manifest["sensitivity"], "value": "sensitive"}
    with stage.world.engine.begin() as conn:
        insert(
            conn,
            "mf_wsp_bundles",
            bundle_sha256="e" * 64,
            owner="miworkbench",
            bundle_id="wsb_sensitive00001",
            bundle_kind="linear_probe",
            artifact_kind="probe",
            artifact_id="p",
            detector_digest=None,
            model="m",
            layer=12,
            sensitivity_value="sensitive",
            manifest=json.dumps(manifest),
            bytes=1,
            import_status="imported",
            received_at=NOW,
            received_by="operator",
        )
    stage.world.body = {
        **stage.world.body,
        "sensitivity": "sensitive",
        "bundle": {
            "owner": "miworkbench",
            "bundle_id": "wsb_sensitive00001",
            "bundle_sha256": "e" * 64,
        },
    }
    with pytest.raises(Refusal) as refused:
        stage.start("ep-claude", "repair", steps=5)
    assert refused.value.code == "wsp_sensitive_data_external_endpoint"
    assert len(stage.anth.requests) + len(stage.oa.requests) == counted

    # FTASKS 8.5: prefix.check's time on the largest bodies seen, and on one near 3.4 MB.
    largest = max(stage.anth.requests, key=lambda s: len(json.dumps(s.body)))
    began = time.perf_counter()
    prefix.check(largest.body, largest.body)
    seen_s = time.perf_counter() - began
    big = dict(largest.body)
    big["messages"] = [
        *largest.body["messages"],
        *[{"role": "user", "content": "y" * 10_000}, {"role": "assistant", "content": "z"}] * 340,
    ]
    began = time.perf_counter()
    prefix.check(big, big)
    stage.figures["prefix_check_s"] = {
        "largest_seen_bytes": len(json.dumps(largest.body)),
        "largest_seen_s": round(seen_s, 4),
        "synthetic_bytes": len(json.dumps(big)),
        "synthetic_s": round(time.perf_counter() - began, 4),
    }
