"""Frozen identifiers keep working, through production code (FR-015.2/.3/.4/.27; FTID §3.7).

The suite renames miLLM to miServer and miStudio to miWorkbench, and freezes every identifier
another program or a stored row depends on (suite R1). The deferred-renames register lists them;
census exports the entries with miForge locations to ``docs/legacy-identifiers.md``, each naming
the ONE test here that exercises it. These tests run the real code that reads or sends each
identifier - renaming one turns its test red.

The meta tests make the register and this file agree: every entry has its registered test, no
retired entry keeps one, and a history-only entry matches nothing outside history.
"""

from __future__ import annotations

import json
import re
import subprocess
from pathlib import Path
from typing import Any

import pytest

from tests.support import legacy_identifiers as li
from tests.support.approval_world import World, definition
from tests.support.frozen_exercisers import REGISTRY, frozen

REPO_ROOT = li.REPO_ROOT
SCHEMAS = REPO_ROOT / "docs" / "schemas"
WEIGHTS = "millm-weights-api-v1.json"
SCORING = "millm-scoring-api-v1.json"
EXPORT = li.load()


def _entry(entry_id: str) -> dict[str, Any]:
    entry = EXPORT.entry(entry_id)
    if entry is None:
        if entry_id in EXPORT.withheld:
            pytest.skip(f"{entry_id} withheld from the public variant")
        pytest.fail(f"{entry_id} is not in {EXPORT.path}")
    return entry


# ------------------------------------------------------------------------------- meta tests


@pytest.mark.parametrize(
    "entry_id",
    [e["id"] for e in EXPORT.entries] + [w for w in EXPORT.withheld if w.startswith("DR-")],
)
def test_every_entry_has_its_test(entry_id: str) -> None:
    entry = _entry(entry_id)
    flags = set(entry["flags"])
    assert "pending" not in flags, (
        f"{entry_id} is pending: its exercise is not written yet (census --allow-pending). "
        f"Write the test, register it with @frozen({entry_id!r}), add its id to census "
        f"EXERCISE and regenerate the export."
    )
    exercise = entry["exercise"]
    if exercise is None:
        assert flags & {
            "history_only",
            "text_only",
        }, f"{entry_id} has no exercise and is neither history_only nor text_only (CF-29)"
        assert entry_id not in REGISTRY, f"{entry_id} is history-only yet registers a test"
        return
    if exercise.startswith("pytest:"):
        assert REGISTRY.get(entry_id) == exercise, (
            f"{entry_id}: the export names {exercise} but the registered test is "
            f"{REGISTRY.get(entry_id)}"
        )
        path, name = exercise.removeprefix("pytest:").split("::", 1)
        assert (REPO_ROOT / path).is_file(), f"{entry_id}: {path} does not exist"
        assert callable(globals().get(name)) or _defines(
            REPO_ROOT / path, name
        ), f"{entry_id}: {name} is not defined in {path}"
    elif exercise.startswith("vitest:"):
        assert exercise.removeprefix("vitest:").startswith("frontend/"), exercise
        assert entry_id not in REGISTRY
    else:  # pragma: no cover - the reader refuses other prefixes
        pytest.fail(f"{entry_id}: unknown exercise prefix {exercise!r}")


def _defines(path: Path, name: str) -> bool:
    return re.search(rf"^def {re.escape(name)}\(", path.read_text("utf-8"), re.M) is not None


def test_registry_lists_only_exported_entries() -> None:
    exported = {e["id"] for e in EXPORT.entries} | set(EXPORT.withheld)
    assert REGISTRY, "no @frozen test registered: the decorator registry is empty"
    assert (
        set(REGISTRY) <= exported
    ), f"tests registered for entries the export does not list: {set(REGISTRY) - exported}"


def test_no_exerciser_for_a_retired_entry() -> None:
    retired = {e["id"] for e in EXPORT.entries if e["status"] == "retired"}
    assert not (set(REGISTRY) & retired), f"retired entries keep a test: {set(REGISTRY) & retired}"


def _live_files() -> list[str]:
    out = subprocess.run(
        [
            "git",
            "ls-files",
            "-z",
            "--",
            ".",
            ":(exclude)AGENT.md",
            ":(exclude,glob)0xcc/**",
            ":(exclude,glob)**/docs/legacy-identifiers*.md",
            # Root-anchored too: see brand_scan.EXPORT_EXCLUSIONS (git 2.43 prefix case).
            ":(exclude,glob)docs/legacy-identifiers*.md",
        ],
        cwd=REPO_ROOT,
        capture_output=True,
        check=True,
    ).stdout.decode()
    return [p for p in out.split("\0") if p]


def test_history_only_patterns_match_nothing_live() -> None:
    entries = [e for e in EXPORT.entries if "history_only" in e["flags"]]
    files = _live_files()
    assert len(files) > 100, "git ls-files found almost nothing; the scan would be vacuous"
    hits: list[str] = []
    for entry in entries:
        pattern = re.compile(entry["pattern"])
        for rel in files:
            path = REPO_ROOT / rel
            if not path.is_file():
                continue
            try:
                text = path.read_text("utf-8")
            except UnicodeDecodeError:
                continue
            for number, line in enumerate(text.splitlines(), start=1):
                if pattern.search(line):
                    hits.append(f"{entry['id']}: {rel}:{number}: {line.strip()[:120]}")
    assert not hits, "history-only identifiers are live outside history:\n" + "\n".join(hits)


# ---------------------------------------------------------------------- one test per entry


def _contract(name: str) -> dict[str, Any]:
    return json.loads((SCHEMAS / name).read_text("utf-8"))


@frozen("DR-04")
def test_dr04_schema_ids_resolve() -> None:
    """Both vendored contracts resolve by their frozen ``$id`` through a JSON-schema registry."""
    from referencing import Registry, Resource

    resources = [Resource.from_contents(_contract(n)) for n in (WEIGHTS, SCORING)]
    registry: Registry[Any] = Registry().with_resources((r.id(), r) for r in resources)  # type: ignore[misc]
    literals = _entry("DR-04")["literals"]
    assert literals, "the DR-04 record lists no literal; the test would check nothing"
    for literal in literals:
        uri = f"https://{literal}"
        resolved = registry.resolver().lookup(uri).contents
        assert resolved["$id"] == uri
        assert literal.rsplit("/", 1)[1] in (WEIGHTS, SCORING)


@frozen("DR-05")
def test_dr05_contracts_load_by_file_name() -> None:
    """The code that validates the fakes opens each contract BY ITS FILE NAME."""
    import importlib

    jsonschema = pytest.importorskip("jsonschema")
    literals = set(_entry("DR-05")["literals"])
    assert literals == {WEIGHTS, SCORING}
    conforms = importlib.import_module("tests.contract.test_fake_weights_surface_conforms")
    # The weights contract, as the fake-surface conformance test loads it, validates a real
    # answer of the fake surface.
    from tests.support.fake_weights_surface import FakeWeightsSurface

    recorder = conforms._Recorder(FakeWeightsSurface())
    recorder.call("status")
    method, path, _status, body = recorder.seen[0]
    conforms._validator(method, path).validate(body)
    assert conforms.CONTRACT == _contract(WEIGHTS)
    # The scoring contract's 409 refusal, read from the file by name.
    refusal = _contract(SCORING)["scoring_precision"]["refusal"]
    jsonschema.Draft202012Validator.check_schema({"type": "object"})
    assert refusal["status"] == 409 and refusal["code"] == "scoring_precision_unavailable"
    for name in literals:
        assert (SCHEMAS / name).is_file(), f"{name} is gone from docs/schemas/"


def _scoring_provider(weights_module: Any, reason: str, reason_header: str) -> Any:
    """``/v1/completions`` as miLLM serves it; float32 refused, the reason ONLY in the header."""
    import httpx
    import torch

    from providers.openai_compatible import OpenAICompatibleProvider

    refusal = _contract(SCORING)["scoring_precision"]["refusal"]

    def handle(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        if body.get("scoring_precision") == "float32":
            return httpx.Response(
                refusal["status"],
                json={
                    "error": {
                        "message": "float32 scoring is unavailable for the resident model",
                        "type": refusal["type"],
                        "param": refusal["param"],
                        "code": refusal["code"],
                    }
                },
                headers={reason_header: reason},
            )
        ids = torch.tensor([[ord(c) % 16 for c in body["prompt"]]])
        with torch.no_grad():
            logits = weights_module(ids)[0][0, -1].float()
        allowed = torch.tensor(body["allowed_token_ids"])
        values = torch.log_softmax(logits[allowed], dim=0)
        top = {f"token_id:{int(i)}": float(v) for i, v in zip(allowed, values, strict=True)}
        first = next(iter(top))
        return httpx.Response(
            200,
            json={
                "id": "cmpl",
                "object": "text_completion",
                "created": 0,
                "model": body["model"],
                "choices": [
                    {
                        "index": 0,
                        "text": "",
                        "finish_reason": "length",
                        "logprobs": {
                            "tokens": [first],
                            "token_logprobs": [top[first]],
                            "top_logprobs": [top],
                            "text_offset": [0],
                        },
                    }
                ],
                "usage": {"prompt_tokens": 1, "completion_tokens": 1},
            },
        )

    return OpenAICompatibleProvider(
        base_url="http://millm.test:8000",
        model="tiny",
        connect_timeout_s=5.0,
        read_timeout_s=5.0,
        client=httpx.Client(transport=httpx.MockTransport(handle)),
    )


@frozen("DR-08")
def test_dr08_headers_sent_and_read(harness: Any) -> None:
    """``X-miLLM-Attachment`` is SENT by the real client; ``X-miLLM-Scoring-Precision-Reason`` is
    READ by the real float32 verifier. Payload and call count, not "was called"."""
    torch = pytest.importorskip("torch")
    from training.hotupdate import verify

    literals = _entry("DR-08")["literals"]
    assert "X-miLLM-Attachment" in literals and "X-miLLM-Scoring-Precision-Reason" in literals
    sent = "X-miLLM-Attachment"
    attachment = harness.attach()["attachment_id"]
    harness.client.call("heartbeat")
    assert harness.fake.call_count("heartbeat") == 1
    (headers,) = harness.fake.headers_of("heartbeat")
    assert headers.get(sent.lower()) == attachment, f"the heartbeat did not carry {sent}"

    reason_header = _contract(SCORING)["scoring_precision"]["refusal"]["reason_header"]
    assert reason_header in literals
    reason = _contract(SCORING)["scoring_precision"]["refusal"]["reasons"][0]

    class Tiny(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            torch.manual_seed(0)
            self.emb = torch.nn.Embedding(16, 8)
            self.head = torch.nn.Linear(8, 16, bias=False)

        def forward(self, input_ids: Any) -> tuple[Any]:
            return (self.head(self.emb(input_ids)),)

    class Tokenizer:
        def __call__(self, text: str, add_special_tokens: bool = True) -> dict[str, list[int]]:
            return {"input_ids": [ord(c) % 16 for c in text]}

    module = Tiny()
    checker = verify.Verifier(
        module=module,
        tokenizer_loader=Tokenizer,
        provider=_scoring_provider(module, reason, reason_header),
        prompt="The capital of France is",
        token_set_size=8,
        tolerance=5e-4,
        tolerance_is_provisional=False,
        precision="float32",
        native_tolerance=0.5,
    )
    result = checker()
    assert (
        result.precision_fallback_reason == reason
    ), f"the verifier did not read {reason_header}: recorded {result.precision_fallback_reason!r}"
    # Every DR-08 literal miForge carries is still spelled exactly so in code or contract.
    haystack = (
        "\n".join(p.read_text("utf-8") for p in (REPO_ROOT / "backend" / "src").rglob("*.py"))
        + json.dumps(_contract(SCORING))
        + json.dumps(_contract(WEIGHTS))
    )
    for literal in literals:
        assert literal in haystack, f"{literal} is no longer sent, read or declared"


@frozen("DR-17")
@pytest.mark.parametrize("actor", ["agent:mistudio-mcp", "agent:miworkbench-mcp"])
def test_dr17_both_actor_ids_are_agents(world: World, actor: str) -> None:
    """Both MCP actor ids are agents: a pending approval, the id kept verbatim, nothing run."""
    if actor == "agent:mistudio-mcp":
        assert actor in _entry("DR-17")["literals"]
    response = world.client.post(
        "/api/v1/runs/define", json=definition(), headers={"X-Miforge-Agent": actor}
    )
    assert response.status_code == 202, response.text
    rows = world.approvals()
    assert [row["status"] for row in rows] == ["pending"]
    assert [row["requested_by"] for row in rows] == [actor]
    assert world.enqueued == []


@frozen("DR-39")
def test_dr39_settings_keys_and_codes(world: World, monkeypatch: pytest.MonkeyPatch) -> None:
    """``millm_base_url`` as a settings key, a history row and an attach summary field; the
    codes ``millm_not_configured`` and ``loaded_in_millm`` as other programs read them."""
    import sqlalchemy as sa

    from api import health as health_module
    from core.config import reset_settings
    from services import approval_gates
    from services.purge import service as purge
    from tests.support.db import session_for
    from tests.support.jobs import insert_run
    from training import acquisition

    # Health with no address: the integration code.
    for probe in ("postgres", "redis"):
        monkeypatch.setitem(
            health_module.DEPENDENCY_PROBES, probe, lambda: {"status": "ok", "detail": None}
        )
    from services.legacy_names import store

    monkeypatch.setattr(store, "read_recent", lambda **_: store.StoredUses())
    monkeypatch.setattr(store, "record", lambda *a, **k: 0)
    assert health_module.health_payload()["integrations"]["millm"]["code"] == "millm_not_configured"
    # The same code as the attach mode's refusal cause, which agents match on.
    from core.errors import Refusal
    from training.acquisition.base import AcquisitionMode, AcquisitionSpec
    from training.hotupdate.attach import AttachAcquirer

    with pytest.raises(Refusal) as refused:
        AttachAcquirer().preflight(AcquisitionSpec(mode=AcquisitionMode.ATTACH))
    assert refused.value.details["cause"] == "millm_not_configured"

    # The settings key, through the operator's PUT and GET, and its history row.
    put = world.client.put(
        "/api/v1/settings/millm_base_url", json={"value": "http://millm.test:8000"}
    )
    assert put.status_code == 200, put.text
    got = world.client.get("/api/v1/settings")
    assert got.status_code == 200
    assert "millm_base_url" in json.dumps(got.json())
    with world.engine.connect() as conn:
        keys = conn.execute(sa.text("SELECT key FROM mf_setting_changes")).scalars().all()
    assert keys == ["millm_base_url"]

    # The attach approval summary names the field.
    assert "millm_base_url" in approval_gates.attach_target()

    # A checkpoint miLLM serves blocks a purge with the frozen code.
    monkeypatch.setattr(
        acquisition,
        "attachment_status",
        lambda: {"configured": True, "reachable": True, "resident_model": "base:gone:step-2"},
    )
    with session_for(world.engine) as session:
        insert_run(session, "gone", status="completed")
    with session_for(world.engine) as session:
        (plan,) = purge.plan_runs(session, ["gone"])
    codes = [b["code"] for b in plan.get("blockers", [])]
    assert "loaded_in_millm" in codes, plan
    reset_settings()
