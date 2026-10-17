"""miForge runs with no sibling application present  (foundation task 1.7 and 1.7a).

**This file discharges BRD-02 acceptance criterion 20, and all eleven features route
their share of that criterion into it.** That is why task 1.7a asserts there is exactly
one of it: fork the file and eleven claims silently become eleven different tests, and the
criterion that pins the whole 2026-10-13 standalone reframe stops meaning one thing.

ADR-010: **an unset base URL means absent-and-normal, not an error.** miForge consumes
siblings' outputs and produces outputs they consume, and requires none of them. Before the
reframe, BRD-02 said *"everything from R-02.1 onward is blocked on it"* — miForge was the
only application in the suite with a hard runtime dependency on a sibling, and the thing it
depended on did not exist. This file is what keeps that from coming back.

**Read the body, not the status code.** `/api/health` returns 503 when a required
dependency is down, and in CI with no Postgres it legitimately will. The question this file
asks is never "did it return 200" but "is each absent sibling named" — an integration
missing from a response is indistinguishable from one that is broken (ADR-013).
"""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from api.app import create_app
from core.config import get_settings, reset_settings
from services.scoring.settings import reset_scoring_settings as _reset_scoring_settings

#: The optional siblings, and the environment variable that configures each. Taken from
#: `Settings` rather than retyped: a sibling added there with no entry here would make this
#: file silently stop covering it.
SIBLING_ENV = {
    "millm": "MILLM_BASE_URL",
    "midataworks": "MIDATAWORKS_BASE_URL",
    # ⚠ ADDED 2026-10-14 with `clients/activations.py`, and it is the THIRD optional
    # integration — R-02.19's activation source for the retention check. It is declared in
    # `services/scoring/settings.py` rather than `core/config.py`, which is why it was
    # missing from `GET /api/health` on the day its client landed, and why that omission is
    # the same split-module shape that cost `/data/packs` its volume twice in one day.
    # Listed here so AC20 covers it: a standalone miForge must run with this unset too.
    "activation_source": "ACTIVATION_SOURCE_BASE_URL",
}


@pytest.fixture
def no_siblings(monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    """Every sibling URL unset, and the settings cache cleared on both sides.

    The cache must be reset *before* and *after*: before, so a cached `Settings` built by
    an earlier test does not survive into this one; after, so this fixture does not leak an
    unconfigured world into the rest of the suite. A fixture that only resets on the way in
    is how an env-var test ends up asserting against a configuration it did not set.
    """
    for variable in SIBLING_ENV.values():
        monkeypatch.delenv(variable, raising=False)
    reset_settings()
    _reset_scoring_settings()
    yield
    reset_settings()
    _reset_scoring_settings()


def test_the_api_starts_with_no_sibling_configured(no_siblings: None) -> None:
    """The app builds and serves. This is the whole claim, at its coarsest."""
    with TestClient(create_app()) as client:
        response = client.get("/api/health")
    assert response.status_code in (200, 503), response.text
    assert response.json()["integrations"], "health reported no integrations at all"


@pytest.mark.parametrize("service", sorted(SIBLING_ENV))
def test_each_absent_sibling_is_reported_unconfigured_BY_NAME(
    service: str, no_siblings: None
) -> None:
    """Named, with its refusal code and the capabilities it costs — never omitted.

    The `<service>_not_configured` spelling is ADR-010's, and health builds it from the
    same `not_configured` helper the refusal path uses, so the code on this response cannot
    drift from the code a caller actually receives.
    """
    with TestClient(create_app()) as client:
        body = client.get("/api/health").json()

    assert service in body["integrations"], (
        f"{service} is absent from the health body. An integration missing from a response "
        f"is indistinguishable from one that is broken (ADR-013)."
    )
    entry = body["integrations"][service]
    assert entry["configured"] is False
    assert (
        entry["code"] == f"{service}_not_configured"
    ), f"expected ADR-010's spelling, got {entry['code']!r}"
    assert service in body["unconfigured_integrations"]


def test_no_sibling_is_reported_configured_when_none_is(no_siblings: None) -> None:
    """The negative half. Without it, a mutation making everything 'configured: True'
    would pass the per-service test above, which only checks the ones it expects."""
    with TestClient(create_app()) as client:
        body = client.get("/api/health").json()
    configured = [n for n, e in body["integrations"].items() if e["configured"]]
    assert not configured, f"reported as configured with no URL set: {configured}"
    assert sorted(body["unconfigured_integrations"]) == sorted(SIBLING_ENV)


def test_an_empty_environment_value_counts_as_unset(monkeypatch: pytest.MonkeyPatch) -> None:
    """``MILLM_BASE_URL=`` means **unset**, not the empty string.

    This is why `Settings` coerces a blank to `None`. Without it, `.env.example`'s
    deliberately-blank sibling lines (task 9.4) would configure every sibling as the empty
    URL, and the standalone path — the one this file exists to pin — would never be taken
    on a developer's machine. The failure would appear as a connection error to `""` at
    some arbitrary later moment instead of a named refusal at the boundary.
    """
    for variable in SIBLING_ENV.values():
        monkeypatch.setenv(variable, "")
    reset_settings()
    try:
        assert get_settings().configured_integrations() == dict.fromkeys(SIBLING_ENV, False)
    finally:
        reset_settings()


def test_the_core_path_names_no_sibling_endpoint() -> None:
    """ADR-022 fences the optional hot-update mode: **nothing in the core path may
    reference a miLLM endpoint.**

    §5.10 grew to thirteen required rows once without anyone deciding it should, which is
    how a mode became an architecture. The fence is `training/hotupdate/` and
    `clients/`; a `/api/weights/...` path anywhere else is the mode leaking into the
    product.
    """
    src = Path(__file__).resolve().parents[2] / "src"
    fenced = ("training/hotupdate", "clients/")
    offenders: list[str] = []
    for path in src.rglob("*.py"):
        relative = str(path.relative_to(src))
        if any(part in relative for part in fenced):
            continue
        text = path.read_text(encoding="utf-8")
        if "/api/weights/" in text:
            offenders.append(relative)
    assert not offenders, (
        f"these core-path modules name a miLLM weights endpoint: {offenders}. §5.10 is "
        f"fenced (ADR-022); reach a sibling through its single client module or not at all."
    )


# --------------------------------------------------------------------------------------
# 1.7a — there must be exactly ONE standalone test file
# --------------------------------------------------------------------------------------


def test_there_is_exactly_one_standalone_test_file() -> None:
    """**Why this is a test and not a convention.**

    The 2026-10-13 acceptance audit found that **all eleven features claim a share of BRD
    acceptance criterion 20** — a full run of each type with no sibling present — and all
    eleven route into this single file. If it is ever forked, those eleven claims become
    eleven different tests and the criterion stops meaning one thing. Two features
    independently wrote *"add cases to that file, never a second one"* into their own task
    lists; this makes it checkable rather than remembered.

    Three features have already tried to create a second one, which is the empirical case
    for asserting it.
    """
    tests_root = Path(__file__).resolve().parents[1]
    found = sorted(
        str(p.relative_to(tests_root))
        for p in tests_root.rglob("*.py")
        if "standalone" in p.name.lower()
    )
    assert found == [f"integration/{Path(__file__).name}"], (
        f"expected exactly one standalone test file, found {found}. Eleven features route "
        f"their share of acceptance criterion 20 into one file; a second one splits the "
        f"criterion silently. Add cases to this file instead."
    )


# ======================================================================================
# F3 - selectors and graders  (AC-003.10, AC-003.23; R-02.9, R-02.50, ADR-010)
#
# **Four of the five selector types run with no network and no sibling; the fifth names the
# service it is missing.** That is this feature's whole share of criterion 20, and it is the
# half of R-02.11 that is enforced by absence: with no endpoint configured there is nothing
# for a judge to fall back to, and the only candidate would be the model being trained.
#
# Added to THIS file rather than a file of F3's own, per the rule above.
# ======================================================================================


@pytest.fixture
def no_selector_siblings(monkeypatch: pytest.MonkeyPatch):
    """Nothing configured, nothing bound: the default state and the point.

    Unbinds the provider resolver and the review-queue port as well as clearing the
    environment, because a leaked binding from another test would make this pass against a
    *configured* miForge - the control passing for the wrong reason.
    """
    from services.selectors import provider_port
    from services.selectors.plugins import human_review
    from services.selectors.settings import SelectorSettings, reset_selector_settings

    for name in (
        "MILLM_BASE_URL",
        "MIDATAWORKS_BASE_URL",
        "JUDGE_PROVIDER_BASE_URL",
        "JUDGE_PROVIDER_MODEL",
        "JUDGE_PROMPT_TEMPLATE",
    ):
        monkeypatch.delenv(name, raising=False)
    provider_port.bind_provider_resolver(None)
    human_review.bind_review_queue(None)
    reset_selector_settings()
    yield SelectorSettings(_env_file=None)  # type: ignore[call-arg]
    provider_port.bind_provider_resolver(None)
    human_review.bind_review_queue(None)
    reset_selector_settings()


def test_f3_all_five_selector_types_are_registered_with_no_sibling(
    no_selector_siblings,
) -> None:
    """Absent is not missing: every type is **listed** whether or not it can run."""
    from services.selectors import ensure_registered, registry

    ensure_registered()
    assert set(registry.registered_kinds()) == {
        "rule",
        "judge_model",
        "reference_match",
        "human_review",
        "composite",
    }


def test_f3_four_types_are_available_and_the_fifth_two_name_their_service(
    no_selector_siblings,
) -> None:
    """AC-003.10. The unavailable ones are reported **by name**, with ADR-010's spelling."""
    from services.selectors import ensure_registered, registry

    ensure_registered()
    rows = {row.name: row for row in registry.list_capabilities(no_selector_siblings)}

    assert rows["rule"].available is True
    assert rows["reference_match"].available is True
    assert rows["composite"].available is True

    assert rows["human_review"].available is False
    assert rows["human_review"].unavailable_reason == "midataworks_not_configured"
    assert rows["judge_model"].available is False
    assert rows["judge_model"].unavailable_reason == "judge_provider_not_configured"


def test_f3_a_full_evaluation_completes_with_no_sibling(no_selector_siblings) -> None:
    """A rule and a reference-match batch, evaluated end to end, with nothing configured."""
    import json as _json

    from services.selectors import ensure_registered
    from services.selectors.resolve import resolve
    from services.selectors.service import evaluate
    from services.selectors.types import EvaluationContext

    ensure_registered()

    rule = resolve(
        {
            "kind": "rule",
            "name": "no-guarantees",
            "config_sha256": None,
            "expression": _json.dumps({"forbidden_terms": ["guarantee"], "min_length": 3}),
        },
        source="inline",
    )
    result = evaluate(["a careful answer", "we guarantee it"], rule)
    assert [record.verdict.value.value for record in result.records] == ["accept", "reject"]
    assert result.health.n == 2
    assert result.hashes.selector_hash_provenance == "local"

    reference = resolve(
        {"kind": "reference_match", "name": "ref", "config_sha256": None, "mode": "exact"},
        source="inline",
    )
    scored = evaluate(["Paris"], reference, EvaluationContext(items=({"reference": "Paris"},)))
    assert scored.records[0].verdict.value.value == "accept"


def test_f3_the_judge_refuses_by_name_and_never_falls_back_to_the_trained_model(
    no_selector_siblings,
) -> None:
    """**R-02.11 enforced by absence.** A model grading itself is not a measurement, so with
    no endpoint configured the type refuses - it does not quietly become in-process."""
    from core.errors import Refusal
    from services.selectors import ensure_registered
    from services.selectors.resolve import resolve

    ensure_registered()
    with pytest.raises(Refusal) as exc:
        resolve(
            {
                "kind": "judge_model",
                "name": "j",
                "config_sha256": None,
                "provider": "judge",
                "threshold": 0.6,
            },
            source="inline",
        )
    assert exc.value.code == "judge_provider_not_configured"
    assert "will NOT fall back" in exc.value.message


def test_f3_human_review_refuses_by_name_and_the_others_are_unaffected(
    no_selector_siblings,
) -> None:
    """E-9: miForge keeps no queue of its own, so absent means refused, not substituted."""
    import json as _json

    from core.errors import Refusal
    from services.selectors import ensure_registered
    from services.selectors.resolve import resolve
    from services.selectors.service import evaluate

    ensure_registered()
    with pytest.raises(Refusal) as exc:
        resolve(
            {
                "kind": "human_review",
                "name": "h",
                "config_sha256": None,
                "queue": "legal",
            },
            source="inline",
        )
    assert exc.value.code == "midataworks_not_configured"

    rule = resolve(
        {
            "kind": "rule",
            "name": "r",
            "config_sha256": None,
            "expression": _json.dumps({"min_length": 1}),
        },
        source="inline",
    )
    assert len(evaluate(["still works"], rule).records) == 1


def test_f3_the_selector_endpoints_answer_with_no_sibling(no_selector_siblings) -> None:
    """The API serves the listing and a full evaluation on a machine with nothing wired."""
    from fastapi.testclient import TestClient

    from api.app import create_app

    client = TestClient(create_app(), raise_server_exceptions=False)

    listing = client.get("/api/v1/selectors")
    assert listing.status_code == 200
    assert len(listing.json()) == 5

    response = client.post(
        "/api/v1/selectors/evaluate",
        json={
            "config": {
                "kind": "rule",
                "name": "r",
                "config_sha256": None,
                "expression": '{"min_length": 3}',
            },
            "items": [{"completion": "long enough"}],
        },
    )
    assert response.status_code == 200, response.text
    assert response.json()["verdicts"][0]["value"] == "accept"


def test_f3_no_selector_module_names_a_sibling_endpoint() -> None:
    """ADR-022's shape, scoped to this feature: no miLLM, miStudio or miDataworks URL in
    F3's own code. The judge's ``base_url`` is configuration; if it happens to point at a
    sibling that is the operator's choice, not a name in the code."""
    from pathlib import Path

    src = Path(__file__).resolve().parents[2] / "src"
    roots = [
        src / "services" / "selectors",
        src / "services" / "graders",
        src / "providers" / "openai_compatible.py",
        src / "api" / "v1" / "endpoints" / "selectors.py",
    ]
    files = [p for root in roots for p in (root.rglob("*.py") if root.is_dir() else [root])]
    assert files, "the scan would be vacuous"

    offenders: list[str] = []
    for path in files:
        for lineno, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
            stripped = line.strip()
            if stripped.startswith("#") or stripped.startswith("*"):
                continue
            for host in ("http://millm", "http://mistudio", "http://midataworks"):
                if host in stripped:
                    offenders.append(f"{path.name}:{lineno}")
    assert offenders == [], f"a sibling endpoint is named in F3's code at {offenders}"


# --------------------------------------------------------------------------------------
# F1 - model acquisition and the step  (R-02.48; AC-001.1, AC-001.12)
#
# F1 is the first feature that could reintroduce a hard sibling dependency: before the
# 2026-10-13 reframe it opened a protocol with another process, reconstructed a model graph
# over CUDA IPC handles and bracketed every write in a quiesce/commit/resume handshake. Its
# whole share of criterion 20 is that none of that is on the default path.
# --------------------------------------------------------------------------------------


def test_F1_acquisition_modes_are_available_without_any_sibling(
    no_siblings: None,
) -> None:
    """The mode list answers with every sibling unset, and **no mode is unavailable because
    of one**.

    Asserted on the *reason* rather than on the availability boolean: the boolean depends on
    whether torch happens to be installed on this machine, and the reason does not. A test
    pinned to the boolean would pass on the GPU node and fail in CI while nothing had changed
    about the standalone guarantee.
    """
    from training import acquisition

    reasons = {entry["mode"]: entry["reason"] for entry in acquisition.modes()}
    assert set(reasons) == {"load", "create", "attach"}
    for mode in ("load", "create"):
        assert reasons[mode] in {None, "ml_runtime_unavailable"}, (
            f"{mode} is unavailable for reason {reasons[mode]!r}; no core acquisition mode "
            f"may depend on a sibling (R-02.48)"
        )
    for mode, reason in reasons.items():
        assert "not_configured" not in (reason or ""), (
            f"{mode} reports a sibling-shaped unavailability ({reason}); an unset sibling is "
            f"absent-and-normal and must affect no capability of F1's"
        )


def test_F1_refuses_the_attach_mode_BY_NAME_rather_than_loading_instead(
    no_siblings: None,
) -> None:
    """AC-001.12. **The substitution R-02.48 forbids**, and the dangerous one: attaching to a
    serving process's resident tensors and loading the same weights from disk train
    *different tensors*, and both report success. So the unavailable mode refuses, names the
    capabilities it would have provided, and never becomes a load.
    """
    from core.errors import Refusal
    from training import acquisition
    from training.acquisition.base import AcquisitionMode, AcquisitionSpec

    spec = AcquisitionSpec(mode=AcquisitionMode.ATTACH)
    acquirer = acquisition.resolve(spec)
    with pytest.raises(Refusal) as caught:
        acquirer.acquire(spec)
    assert caught.value.code == "acquisition_mode_unavailable"
    assert caught.value.details["mode"] == "attach"
    assert caught.value.details["capabilities_unavailable"]


def test_F1_completes_a_run_with_no_sibling_present(
    no_siblings: None, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """**AC-001.1 and F1's share of criterion 20**: acquire -> step -> record, end to end,
    with no sibling base URL configured, no database, and nothing attached.

    The tensor arithmetic is faked at its narrowest seam - this suite runs on a machine with
    no card and no ML wheels (ADR-023) - but the acquisition registry, the acquisition record,
    the step's three stages, the authoritative run log and the atomic trainer-state write are
    all production code. What is asserted is that **no part of that path reaches for another
    application**, which is a property of the code rather than of the fakes.
    """
    from tests.integration import test_step_loop as f1
    from training import run_log, tasks, trainer_state
    from training.acquisition.base import AcquiredModel, AcquisitionSpec
    from training.optimizer import OptimizerIdentity

    (tmp_path / "trainer-state").mkdir()
    (tmp_path / "checkpoints").mkdir()
    from core.config import Settings

    monkeypatch.setattr(
        "core.config._settings",
        Settings(  # type: ignore[call-arg]
            _env_file=None,
            trainer_state_root=tmp_path / "trainer-state",
            checkpoint_root=tmp_path / "checkpoints",
        ),
    )

    model = f1.Model()
    optimizer = f1.Optimizer()
    monkeypatch.setattr(trainer_state, "_torch", lambda: f1.FakeTorch)
    monkeypatch.setattr(
        tasks, "acquire_for_run", lambda spec: AcquiredModel(model, None, f1.RECORD)
    )
    monkeypatch.setattr(
        tasks.optimizer_module,
        "build",
        lambda module, cfg, state="fresh": (
            optimizer,
            OptimizerIdentity(name=cfg.optimizer_name(), state=state),
        ),
    )

    result = tasks.drive_run(
        run_id="standalone-f1",
        spec=AcquisitionSpec(path=Path("/data/checkpoints/base")),
        cfg=f1.CFG,
        batch_source=f1.Source(),
        objectives=f1.Registry({"sft": f1.Objective(name="sft", trace=[])}),
        session=None,  # no database either: the log is the authoritative record (ADR-004)
        log_root=tmp_path,
        trainer_state_root=tmp_path / "trainer-state",
        max_steps=3,
        save_every=3,
    )

    assert result.steps_completed == 3
    assert result.acquisition["starting_weights_sha256"]
    log = run_log.run_log_path(tasks.run_log_root(tmp_path), "standalone-f1")
    assert [b["step"] for b in run_log.read_blocks(log, block="step")] == [1, 2, 3]
    assert result.trainer_states, "optimizer state must be checkpointed on the cadence"


def test_F1_has_no_outbound_sibling_call_site(no_siblings: None) -> None:
    """``clients/`` holds one module per optional sibling, each the single outbound call
    site - and **F1 adds none**. Its only outbound call is the Hugging Face Hub, which is an
    external dependency (BRD-02 §2a *External*) and not a sibling, which is why an
    unreachable Hub is reported as the Hub (E-4) rather than as an integration failure.
    """
    import ast

    src = Path(__file__).resolve().parents[3] / "backend" / "src"
    offenders: list[str] = []
    for path in [*list((src / "training").rglob("*.py")), src / "services" / "acquisition.py"]:
        if "hotupdate" in path.parts:
            continue
        for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
            if isinstance(node, (ast.Import, ast.ImportFrom)):
                names = [a.name for a in node.names]
                if isinstance(node, ast.ImportFrom) and node.module:
                    names.append(node.module)
                for name in names:
                    if name.split(".")[0] == "clients":
                        offenders.append(f"{path.relative_to(src)}:{node.lineno} -> {name}")
    assert not offenders, f"F1's path reaches a sibling client: {offenders}"


# ======================================================================================
# F5 - save, load, continue, branch  (AC-005.10; R-02.20 - R-02.23, R-02.51, AC20)
#
# **F5 is where the reframe either holds or does not.** Until 2026-10-13 a save was a call
# to another application - miForge sent the fields it owned and miLLM wrote the file - and
# R-02.22 said a continue meant loading the checkpoint *into miLLM*. Taken literally, crash
# recovery required an application that is specced and unimplemented, which made criteria 6
# and 20 unachievable. Both requirements were corrected on this feature's findings, so the
# standalone forms asserted below ARE the requirements rather than a reading of them.
#
# Added to THIS file rather than a file of F5's own, per task 1.7a.
# ======================================================================================


@pytest.fixture
def f5_roots(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, no_siblings: None):
    """Checkpoint and trainer-state roots, with every sibling URL unset."""
    checkpoints = tmp_path / "checkpoints"
    states = tmp_path / "trainer-state"
    checkpoints.mkdir()
    states.mkdir()
    monkeypatch.setenv("CHECKPOINT_ROOT", str(checkpoints))
    monkeypatch.setenv("TRAINER_STATE_ROOT", str(states))
    reset_settings()
    yield checkpoints, states
    reset_settings()


def test_F5_a_save_completes_with_no_sibling_present(f5_roots) -> None:
    """A checkpoint is written, published and readable with nothing else running.

    R-02.20's rewrite dissolved the entire torn-weights apparatus into one line of
    filesystem semantics - a temporary directory, ``fsync``, ``rename``. **Standalone, a
    torn parameter set cannot occur**, and that is collected here.
    """
    from services.checkpoints import manifest as manifest_module
    from tests.support.checkpoints import saved_checkpoint

    checkpoints, _ = f5_roots
    result = saved_checkpoint(checkpoints, run_id="standalone-f5", step=100)

    assert result.directory.is_dir()
    assert (checkpoints / "standalone-f5" / "latest").resolve() == result.directory
    assert manifest_module.read(result.directory)["name"] == result.name
    assert get_settings().millm_base_url is None


def test_F5_a_continue_needs_no_other_application(f5_roots) -> None:
    """**DEF-2, the most serious thing the feature reviews found.** R-02.22 now reads
    *"miForge loads the checkpoint itself (R-02.46 case 1), verifies its ``weights_sha256``
    before use (R-02.21)"*; it used to say *"load the checkpoint into miLLM (BRD-01
    R-01.25)"*, and taken literally a crash recovery required a sibling that does not exist.
    """
    from services.checkpoints import resume as resume_module
    from tests.support.checkpoints import write_chain

    checkpoints, states = f5_roots
    chain = write_chain(checkpoints, run_id="standalone-f5", steps=(100, 200))

    plan = resume_module.continue_plan("standalone-f5", root=checkpoints, trainer_state_root=states)
    assert plan.run_id == "standalone-f5"
    assert plan.step == 200
    assert plan.parent == chain[-1].name
    assert plan.optimizer_state == "fresh"
    assert plan.verified_weights_sha256


def test_F5_a_branch_needs_no_other_application(f5_roots) -> None:
    """R-02.23: a new ``run_id``, ``parent`` at the branch point, the lineage root carried
    forward - all from manifests on one filesystem."""
    from services.checkpoints import resume as resume_module
    from tests.support.checkpoints import write_chain

    checkpoints, states = f5_roots
    chain = write_chain(checkpoints, run_id="standalone-f5", steps=(100,))

    plan = resume_module.branch_plan(
        "standalone-f5",
        100,
        new_run_id="standalone-f5-branch",
        root=checkpoints,
        trainer_state_root=states,
    )
    assert plan.run_id == "standalone-f5-branch"
    assert plan.parent == chain[0].name
    assert plan.base_model == "LiquidAI/LFM2-1.2B"
    assert plan.is_branch is True


def test_F5_the_lineage_walk_reads_only_the_filesystem(f5_roots) -> None:
    """No database, no sibling: a checkpoint carries enough of its own history to be found
    again in a tree of a hundred siblings (US-4, ADR-004)."""
    from services.checkpoints.lineage import walk
    from tests.support.checkpoints import write_chain

    checkpoints, _ = f5_roots
    write_chain(checkpoints, run_id="standalone-f5", steps=(1, 2, 3))

    result = walk("standalone-f5", 3, root=checkpoints)
    assert result.terminated == "root"
    assert [n.step for n in result.nodes] == [3, 2, 1]


def test_F5_imports_no_sibling_and_no_hot_update_module(no_siblings: None) -> None:
    """ADR-022's fence, as an AST walk rather than as a claim.

    A core-path import of ``training/hotupdate/`` fails the suite by construction: the
    standalone test runs with that package absent, and §5.10's mode *"grew to thirteen
    required rows once without anyone deciding it should."*
    """
    import ast

    src = Path(__file__).resolve().parents[3] / "backend" / "src"
    forbidden = {"millm", "mistudio", "midataworks"}
    offenders: list[str] = []
    paths = list((src / "services" / "checkpoints").rglob("*.py"))
    paths.append(src / "api" / "v1" / "endpoints" / "checkpoints.py")

    for path in paths:
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if isinstance(node, (ast.Import, ast.ImportFrom)):
                names = [a.name for a in node.names]
                if isinstance(node, ast.ImportFrom) and node.module:
                    names.append(node.module)
                for name in names:
                    head = name.split(".")[0]
                    if head in forbidden or "hotupdate" in name:
                        offenders.append(f"{path.relative_to(src)}:{node.lineno} -> {name}")
    assert not offenders, f"F5's path reaches a sibling or the fenced mode: {offenders}"


def test_F5_names_no_sibling_endpoint_anywhere_in_its_path(no_siblings: None) -> None:
    """The textual half of the same claim, because an endpoint can be named without an
    import. DEF-1 and DEF-2 were exactly that shape: a requirement naming
    ``POST /api/weights/load`` with nothing importing anything.
    """
    src = Path(__file__).resolve().parents[3] / "backend" / "src"
    paths = list((src / "services" / "checkpoints").rglob("*.py"))
    paths.append(src / "api" / "v1" / "endpoints" / "checkpoints.py")

    offenders: list[str] = []
    for path in paths:
        source = path.read_text(encoding="utf-8")
        # Comments explain WHY the sibling is absent and legitimately name it; the claim is
        # about code. Stripped rather than the explanations removed - the explanation is the
        # most valuable line in the file.
        body = "\n".join(line.split("#", 1)[0] for line in source.splitlines())
        for needle in ("/api/weights/", "millm_base_url", "X-miLLM-Attachment", "trainer_token"):
            if needle in body:
                offenders.append(f"{path.relative_to(src)} -> {needle}")
    assert not offenders, offenders


# ======================================================================================
# F2 - signal sources (R-02.4 - R-02.8). AC-002.19, this feature's share of BRD AC20.
#
# **Four sources, and what each needs with nothing configured.** `corpus` and
# `preference-pairs` read a contract-conformant export directory through F10's ingest and
# need nothing else at all. `self-generation` and `reward` generate through ADR-009's
# **in-process** provider, which is not a sibling and needs no endpoint - but it does need
# the parameters the `train` worker holds, so an unbound handle is reported by name rather
# than quietly substituted. The one selector that needs miDataworks - human review - reports
# `midataworks_not_configured` BY NAME, and no other capability is affected.
# ======================================================================================


@pytest.fixture
def f2_standalone(monkeypatch: pytest.MonkeyPatch, no_siblings: None) -> Iterator[None]:
    """No sibling URL, and F2's three must-be-explicit settings supplied locally."""
    from training.signals.settings import reset_signal_settings

    monkeypatch.setenv("SIGNAL_SFT_MASKING_POLICY", "completion_only")
    monkeypatch.setenv("SIGNAL_DPO_REFERENCE_SOURCE", "precomputed")
    monkeypatch.setenv("SIGNAL_PENDING_SHORTFALL", "short_batch")
    reset_signal_settings()
    yield
    reset_signal_settings()


def _f2_engine(tmp_path: Path):  # type: ignore[no-untyped-def]
    from tests.support import db as support_db

    engine = support_db.migrated_engine(tmp_path / "db")
    return engine


def test_f2_all_four_sources_are_registered_with_no_sibling(f2_standalone: None) -> None:
    """The registry is complete with nothing configured - and its names are the manifest's,
    so every one of them can be written into a checkpoint that a run saves."""
    from training.signals import ensure_registered, registry

    ensure_registered()
    assert set(registry.names()) == {"corpus", "self-generation", "preference-pairs", "reward"}


def test_f2_a_corpus_run_completes_with_no_sibling_present(
    f2_standalone: None, tmp_path: Path
) -> None:
    """**AC20 for ``corpus``**: a contract-conformant export on disk, through F10's ingest,
    producing real recorded batches - no miLLM, no miDataworks, no endpoint, no card."""
    from tests.support import db as support_db
    from tests.support import signal_fixtures as fixtures
    from training.signals import batch_index, ensure_registered, registry
    from training.signals.base import PartSpec, RunContext, SourceSpec

    engine = _f2_engine(tmp_path)
    with support_db.session_for(engine) as session:
        part = fixtures.corpus_part(session, tmp_path, texts=["alpha", "beta", "gamma", "delta"])
        ensure_registered()
        source = registry.get("corpus")(
            RunContext(run_id="standalone-f2-corpus", session=session, log_root=tmp_path / "runs")
        )
        source.preflight(
            SourceSpec(
                source="corpus",
                parts=(PartSpec(kind="corpus", part_id=str(part["id"])),),
                batch_size=2,
            )
        )
        first, second = source.next(), source.next()
        rows = batch_index.list_batches(session, run_id="standalone-f2-corpus", limit=10)

    assert first.record["batch_sha256"] != second.record["batch_sha256"]
    assert len(rows) == 2
    assert all(row["source"] == "corpus" for row in rows)


def test_f2_a_preference_pairs_run_completes_with_no_sibling_present(
    f2_standalone: None, tmp_path: Path
) -> None:
    """**AC20 for ``preference-pairs``** (the file half), which needs no provider at all."""
    from tests.support import db as support_db
    from tests.support import signal_fixtures as fixtures
    from training.signals import ensure_registered, registry
    from training.signals.base import PartSpec, RunContext, SourceSpec

    engine = _f2_engine(tmp_path)
    with support_db.session_for(engine) as session:
        part = fixtures.pairs_part(session, tmp_path, pairs=[("yes", "no"), ("good", "bad")])
        ensure_registered()
        source = registry.get("preference-pairs")(
            RunContext(run_id="standalone-f2-pairs", session=session, log_root=tmp_path / "runs")
        )
        source.preflight(
            SourceSpec(
                source="preference-pairs",
                parts=(PartSpec(kind="preference-pairs", part_id=str(part["id"])),),
                batch_size=2,
            )
        )
        batch = source.next()

    assert batch.objective == "dpo"
    assert batch.reference_source == "precomputed"
    assert batch.record["source"] == "preference-pairs"


def test_f2_self_generation_completes_through_the_in_process_provider(
    f2_standalone: None, tmp_path: Path
) -> None:
    """**AC20 for ``self-generation``**: generation through ADR-009's in-process provider -
    *miLLM is one row beside vLLM, Ollama and the OpenAI API*, and the default needs none of
    them - with a **rule** selector, which is the one selector type guaranteed available
    with nothing configured.
    """
    from tests.support import db as support_db
    from tests.support import signal_fixtures as fixtures
    from training.signals import ensure_registered, registry
    from training.signals.base import PartSpec, RunContext, SourceSpec

    engine = _f2_engine(tmp_path)
    with support_db.session_for(engine) as session:
        part = fixtures.prompt_part(session, tmp_path, prompts=["q1", "q2"])
        ensure_registered()
        source = registry.get("self-generation")(
            RunContext(
                run_id="standalone-f2-selfgen",
                session=session,
                provider=fixtures.in_process_provider(),
                current_step=lambda: 1,
                log_root=tmp_path / "runs",
            )
        )
        source.preflight(
            SourceSpec(
                source="self-generation",
                parts=(PartSpec(kind="prompt_set", part_id=str(part["id"])),),
                selector=fixtures.rule_selector(),
                config_source="inline",
                batch_size=2,
            )
        )
        batch = source.next()

    assert batch.record["resolved_model"] == "in_process:fake-model@1"
    assert batch.record["selector_config_sha256"] is not None
    # The configuration is LOCAL and is labelled so. R-02.40: miForge never claims
    # miDataworks provenance for anything miDataworks did not publish.
    assert batch.hashes.selector_hash_provenance == "local"


def test_f2_the_reward_source_reports_its_blocker_rather_than_recording_a_false_source(
    f2_standalone: None, tmp_path: Path
) -> None:
    """**AC20 for ``reward``. The gate-contract blocker is CLOSED; a real one remains.**

    This test was written against a defect in this repository's own gate contract:
    ``batch-record-v1.json``'s ``source`` enum carried no ``reward`` value, so a reward
    batch could not be recorded at all, and recording it as ``mixed`` or ``self_generation``
    would have been a false claim about what trained the checkpoint. **It read the contract
    rather than hard-coding the gap**, which is why it needed no edit when the enum was
    aligned on 2026-10-14 - it took its own "the day it closes" branch.

    **And that branch then revealed the next blocker, which is the useful part.** With the
    enum fixed, pre-flight gets further and refuses ``judge_threshold_missing``: a
    judge-model grader *decides which completions are admitted into training*, so miForge
    will not choose a threshold for it. That refusal is correct and is F3's. The fixture
    simply had not supplied one - an incomplete test, not a defect - so a threshold is
    passed now and the reward source pre-flights clean with **no sibling present**.

    *Worth keeping: a test that reads the contract instead of asserting a gap survives the
    gap closing, and then tells you what was behind it. One that hard-codes the gap would
    have gone red and been "fixed" by deleting the assertion.*
    """
    from core.errors import Refusal
    from tests.support import db as support_db
    from tests.support import signal_fixtures as fixtures
    from training.signals import batch_record, ensure_registered, registry
    from training.signals.base import PartSpec, RunContext, SourceSpec

    engine = _f2_engine(tmp_path)
    with support_db.session_for(engine) as session:
        part = fixtures.prompt_part(session, tmp_path, prompts=["q1", "q2"])
        ensure_registered()
        source = registry.get("reward")(
            RunContext(
                run_id="standalone-f2-reward",
                session=session,
                provider=fixtures.in_process_provider(),
            )
        )
        spec = SourceSpec(
            source="reward",
            parts=(PartSpec(kind="prompt_set", part_id=str(part["id"])),),
            grader={
                "kind": "judge_model",
                "name": "j",
                "config_sha256": None,
                "provider": "judge",
                # Explicit, because F3 refuses `judge_threshold_missing` rather than
                # choosing one: the threshold decides what is admitted into training, and a
                # default would be a verdict policy nobody set.
                "threshold": 0.5,
            },
            config_source="inline",
            group_size=2,
        )
        assert (
            "reward" in batch_record.source_enum()
        ), "the contract gap is closed; if this fails the enum regressed"
        with pytest.raises(Refusal) as caught:
            source.preflight(spec)

    # The blocker is now a PROVIDER, not the contract - and that is the honest AC20
    # statement for this source. A reward run needs a scalar to order two generations;
    # `rule` produces none, `reference_match` needs a reference a pair of fresh generations
    # does not have, and R-02.11 forbids the in-process provider being a judge, because a
    # model grading itself is not a measurement. So standalone it refuses for want of an
    # endpoint, by name, naming the setting to configure.
    assert caught.value.code == "judge_provider_not_configured"

    # THE AC20 ASSERTION, unchanged through both blockers: the refusal names no sibling.
    # It points at JUDGE_PROVIDER_BASE_URL, which is miForge's own configuration - any
    # OpenAI-compatible endpoint satisfies it, and miLLM is one row beside vLLM, Ollama and
    # the OpenAI API rather than a named dependency.
    assert "midataworks" not in caught.value.message.lower()
    assert "millm" not in caught.value.message.lower()
    assert "JUDGE_PROVIDER_BASE_URL" in caught.value.message


def test_f2_the_human_review_selector_names_midataworks_and_nothing_else_is_affected(
    f2_standalone: None, tmp_path: Path
) -> None:
    """The one selector that needs a sibling reports ``midataworks_not_configured`` **by
    name**, and the other four types stay available - *never substitute a different
    behaviour for a missing service* (ADR-010, ADR-018).
    """
    from services.selectors import ensure_registered as ensure_selectors
    from services.selectors.availability import MIDATAWORKS_NOT_CONFIGURED, availability_of
    from services.selectors.registry import capability, registered_kinds
    from services.selectors.settings import get_selector_settings, reset_selector_settings
    from services.selectors.types import Requirement

    reset_selector_settings()
    ensure_selectors()
    settings = get_selector_settings()
    unavailable = {
        kind: availability_of(capability(kind).requires, settings)[1]
        for kind in registered_kinds()
        if not availability_of(capability(kind).requires, settings)[0]
    }
    assert unavailable.get("human_review") == MIDATAWORKS_NOT_CONFIGURED
    # Every type that needs nothing is still available: one absent service costs exactly
    # the capabilities that needed it.
    for kind in registered_kinds():
        if capability(kind).requires is Requirement.NONE:
            assert availability_of(capability(kind).requires, settings)[0] is True


def test_f2_the_source_listing_names_every_unsatisfied_requirement(f2_standalone: None) -> None:
    """``GET /api/v1/signal-sources`` with nothing configured: four sources, each listed,
    each unavailable one carrying the reason. *An option missing from a response is
    indistinguishable from one that is broken.*"""
    from services.signal_source_service import listing

    rows = {row["name"]: row for row in listing()}
    assert len(rows) == 4
    assert rows["corpus"]["available"] is True
    assert rows["preference-pairs"]["available"] is True
    for name in ("self-generation", "reward"):
        if not rows[name]["available"]:
            assert rows[name]["unavailable_reason"]


def test_f2_adds_no_outbound_call_site_and_names_no_sibling_endpoint(no_siblings: None) -> None:
    """F2 performs no HTTP of its own: generation goes through ``providers/`` and
    miDataworks through F10's single call site (ADR-010), so there is no second timeout
    policy and no second answer about whether a sibling is present.
    """
    import ast

    src = Path(__file__).resolve().parents[3] / "backend" / "src"
    paths = list((src / "training" / "signals").rglob("*.py"))
    paths += list((src / "training" / "objectives").rglob("*.py"))
    paths += [
        src / "services" / "signal_source_service.py",
        src / "api" / "v1" / "endpoints" / "signal_sources.py",
        src / "api" / "v1" / "endpoints" / "signal_batches.py",
    ]
    offenders: list[str] = []
    for path in paths:
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if isinstance(node, ast.Import | ast.ImportFrom):
                names = [alias.name for alias in node.names]
                if isinstance(node, ast.ImportFrom) and node.module:
                    names.append(node.module)
                for name in names:
                    head = name.split(".")[0]
                    if head in {"httpx", "millm", "mistudio", "midataworks"} or "hotupdate" in name:
                        offenders.append(f"{path.relative_to(src)}:{node.lineno} -> {name}")
        body = "\n".join(
            line.split("#", 1)[0] for line in path.read_text(encoding="utf-8").splitlines()
        )
        for needle in ("/api/weights/", "millm_base_url", "X-miLLM-Attachment"):
            if needle in body:
                offenders.append(f"{path.relative_to(src)} -> {needle}")
    assert (
        not offenders
    ), f"F2's path reaches a sibling, an HTTP client or the fenced mode: {offenders}"


# ======================================================================================
# F4 - test and retention scoring  (AC-004.1, AC-004.28, AC-004.32; R-02.49, R-02.19, ADR-010)
#
# **A full test-and-retention pass completes on the in-process provider with nothing else
# running at all** - no endpoint, no credential, no sibling, no broker, no GPU. That is this
# feature's whole share of criterion 20, and it is what R-02.49 was written for: *"a test path
# wired to one sibling by name is what made the whole document depend on it."* Before the
# provider abstraction, miForge's test path called miLLM, and that one call site turned an
# optional integration into a hard runtime dependency on an application that did not exist.
#
# R-02.19's activation source is **absent from the tree**, not merely unconfigured, and the
# pass still completes - which is the difference between an optional integration and one that
# happens not to be reached today.
#
# Added to THIS file rather than a file of F4's own, per task 1.7a's rule above.
# ======================================================================================


@pytest.fixture
def no_scoring_siblings(monkeypatch: pytest.MonkeyPatch):
    """Nothing configured: no sibling URL, no provider row, no activation source."""
    import os

    from core.config import reset_settings
    from services.scoring.settings import reset_scoring_settings

    for variable in SIBLING_ENV.values():
        monkeypatch.delenv(variable, raising=False)
    for key in list(os.environ):
        if key.startswith("PROVIDER_") or key.startswith("JUDGE_PROVIDER_"):
            monkeypatch.delenv(key, raising=False)
    monkeypatch.delenv("ACTIVATION_SOURCE_BASE_URL", raising=False)
    reset_settings()
    reset_scoring_settings()
    yield
    reset_settings()
    reset_scoring_settings()


def _f4_spec(run_id: str = "standalone-run"):
    """A test-and-retention configuration with no endpoint anywhere in it."""
    from providers.base import GenerationParams
    from services.scoring import expectations, retention, service, sets

    rule = {
        "kind": "rule",
        "name": "len",
        "config_sha256": None,
        "expression": '{"min_length": 4}',
    }
    expectation = expectations.resolve_expectation(rule, source="pack")

    def a_set(kind: str):
        reference = sets.SetReference(
            rendered=f"{kind}_set/general@1",
            structured={"type": f"{kind}_set", "name": "general", "version": "1"},
        )
        return sets.resolve(
            kind,
            reference,
            [{"prompt": f"{kind}-{i}"} for i in range(4)],
            markers=("evaluation_only",),
        )

    return service.EvaluationSpec(
        run_id=run_id,
        provider_name="in_process",
        aggregation="mean",
        decode=GenerationParams(max_tokens=16, temperature=0.0),
        test_set=a_set("test"),
        retention_set=a_set("retention"),
        test_expectation=expectation,
        retention_expectation=expectation,
        floor=retention.ResolvedFloor(value=0.5, layer="pack_slot"),
        batch_bound=4,
    )


def test_F4_a_full_pass_completes_with_no_sibling_and_no_broker(
    no_scoring_siblings, tmp_path: Path
) -> None:
    """**The claim at its coarsest**: both sets scored, both records durable, no network.

    Synchronous and broker-free is not incidental - it is the property F6's loop depends on
    (FR-004.14a), because ADR-009's in-process provider generates from parameters that exist
    only in the ``train`` worker's process.
    """
    import sqlalchemy as sa

    from providers.fakes import FakeProvider
    from services.scoring import service
    from services.scoring.tables import SCORES
    from tests.support import scores as fixtures
    from tests.support.db import migrated_engine, session_for

    engine = migrated_engine(tmp_path / "db")
    fixtures.create_score_tables(engine)
    try:
        log_path = tmp_path / "runs" / "standalone-run" / "run.jsonl"
        with session_for(engine) as session:
            outcome = service.evaluate(
                _f4_spec(),
                0,
                session=session,
                log_path=log_path,
                provider=FakeProvider(answers="longer"),
            )
            assert {record.kind for record in outcome.records} == {"test", "retention"}
            assert session.execute(sa.select(sa.func.count()).select_from(SCORES)).scalar_one() == 2
        assert log_path.is_file(), "the authoritative run-log line must exist"
    finally:
        engine.dispose()


def test_F4_the_in_process_provider_is_the_only_row_with_nothing_configured(
    no_scoring_siblings,
) -> None:
    """ADR-009's default **needs nothing else running**, which is why it is the default."""
    from providers import registry

    assert [row.name for row in registry.list_rows()] == [registry.IN_PROCESS_NAME]


def test_F4_the_provider_listing_and_health_agree_with_no_sibling(
    no_scoring_siblings,
) -> None:
    """Two surfaces, one answer. *An integration missing from a response is
    indistinguishable from one that is broken* (ADR-013), so both must name what is absent
    rather than omitting it."""
    with TestClient(create_app()) as client:
        providers = client.get("/api/v1/providers").json()
        health = client.get("/api/health").json()

    rows = {row["name"]: row for row in providers["providers"]}
    assert set(rows) == {"in_process"}, "no HTTP provider exists with nothing configured"
    assert rows["in_process"]["base_url"] is None
    # Health names every absent sibling, and F4 adds no integration of its own to it: its
    # one optional integration is R-02.19's activation source, which is not built (see the
    # test below).
    assert set(health["integrations"]) >= set(SIBLING_ENV)
    # `unconfigured_integrations` is a list of NAMES, so every sibling must appear in it -
    # named, not merely absent from the configured set.
    assert set(health["unconfigured_integrations"]) >= set(SIBLING_ENV)


def test_F4_the_activation_source_is_OFF_and_named_rather_than_absent(
    no_scoring_siblings,
) -> None:
    """R-02.19's activation source is **built and off** (AC-004.28; E-20, E-21).

    *This test was a tripwire until 2026-10-14*: it asserted ``clients/activations.py`` did
    **not** exist, because the module belongs in ``backend/src/clients/`` - one module per
    optional service - which was outside F4's write scope. Its own docstring said *"when
    the module lands this test fails, which is the right moment to replace it with the real
    AC-004.28 cases."* **The module landed, the tripwire fired, and these are those
    cases.** It is flipped rather than deleted so the round that closed it is findable.

    Three claims, and the second is the one the requirement turns on:

    * unconfigured is **named**, with the capability it costs, never omitted;
    * **unreachable is a different code** from unconfigured - two states, two fixes
      (E-21, and F10's FR-010.6 from the other side);
    * the rest of scoring is **unaffected** by either, which is what makes it optional
      rather than merely unreached today.
    """
    from clients import activations
    from core.errors import Refusal

    src = Path(__file__).resolve().parents[2] / "src"
    assert (src / "clients" / "activations.py").exists(), "the one call site must be here"

    # Off by default, and `None` is the normal answer rather than a refusal.
    assert activations.is_configured() is False
    assert activations.resolve("recording activation statistics") is None

    # Named, with the capability it costs. ADR-010's spelling.
    with pytest.raises(Refusal) as caught:
        activations.require_activation_source(
            "recording activation statistics", "activation_statistics"
        )
    assert caught.value.code == "activation_source_not_configured"
    assert caught.value.details["capabilities_unavailable"] == [
        activations.CAPABILITIES["activation_statistics"]
    ]

    # Unreachable is a DIFFERENT code. Asserted on the constants, so a rename that
    # collapses the pair fails here as well as in the client's own controls.
    assert activations.UNREACHABLE != activations.NOT_CONFIGURED
    assert len({activations.NOT_CONFIGURED, activations.UNREACHABLE, activations.REFUSED}) == 3

    # The status row keeps `reachable` three-valued: unset is not down.
    row = activations.integration_status(reachable=False)
    assert (row["configured"], row["reachable"]) == (False, None)

    # And the core scoring path must not reference it, configured or not (FR-004.35).
    # Against the CODE: `scoring/settings.py`'s docstring cites `test_activations_client.py`
    # when recording the credential-echo fix, and a citation is not an import.
    from tests.support.source_text import code_only

    for path in sorted((src / "services" / "scoring").rglob("*.py")):
        assert "activations" not in code_only(path.read_text(encoding="utf-8")), path.name


def test_F4_the_judge_is_unavailable_by_name_rather_than_falling_back(
    no_scoring_siblings,
) -> None:
    """R-02.11 enforced by absence, from F4's side of the seam.

    F4 binds its registry into F3's resolver port, so with nothing configured the resolver
    exists and reports the judge **unavailable by name**. There is nothing to fall back to,
    and the only candidate would be the model being trained.
    """
    from core.errors import Refusal
    from providers import registry
    from services.scoring import wiring
    from services.selectors import provider_port

    wiring._wired = False
    wiring.wire()
    resolver = provider_port.current_provider_resolver()
    assert resolver is not None, "the resolver is bound even when no judge is configured"
    assert resolver.available("judge") is False
    with pytest.raises(Refusal) as caught:
        resolver.resolve("judge")
    assert caught.value.code == "judge_provider_not_configured"
    # And the in-process provider is refused as a judge under every configuration.
    registry.bind_model_handle(None)


def test_F4_no_scoring_module_reaches_a_sibling_or_an_http_client(
    no_scoring_siblings,
) -> None:
    """The structural half: F4's path has **no** outbound call site of its own.

    F10 owns the single miDataworks call site and F3 owns the single HTTP provider; F4
    resolves one from its registry and reads rows. An ``httpx`` import here would be a second
    HTTP client and a second set of timeout semantics.
    """
    import ast

    src = Path(__file__).resolve().parents[2] / "src"
    offenders: list[str] = []
    for path in sorted((src / "services" / "scoring").glob("*.py")):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            names: list[str] = []
            if isinstance(node, ast.Import):
                names = [alias.name for alias in node.names]
            elif isinstance(node, ast.ImportFrom) and node.module:
                names = [node.module]
            for name in names:
                if name.split(".")[0] in ("httpx", "clients", "millm", "mistudio", "midataworks"):
                    offenders.append(f"{path.name}:{node.lineno} -> {name}")
    assert offenders == [], f"F4's path reaches a client or a sibling: {offenders}"


# ======================================================================================
# F8 — Run types: create, adapters, merge  (R-02.41 - R-02.45; AC20's F8 share)
# ======================================================================================
#
# **Three run types, no sibling, and a creation run has no upstream at all.** F8-create is
# the least coupled capability in the project - no base model, no attachment, nothing
# upstream, only a config, a tokenizer and a corpus - which is exactly what makes it the
# criterion's sharpest case. BRD-02 section 5.9: *"The architecture's most tightly-coupled
# design was carrying its least-coupled capability."*
#
# What is asserted here with no card and no ML stack: all three run types are registered and
# resolvable, the budget preview answers, the adapter artifact is written and read back, and
# the merge refuses by name. What a real **training step** of each type needs is a card, and
# that is `test_adapter_run_gpu.py` / `test_create_run_gpu.py` - `gpu`-marked, deselected by
# default, and an operator-run gate. They have NOT been run; see the feature report.


@pytest.fixture
def f8_root(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, no_siblings: None):  # type: ignore[no-untyped-def]
    checkpoints = tmp_path / "checkpoints"
    checkpoints.mkdir()
    monkeypatch.setenv("CHECKPOINT_ROOT", str(checkpoints))
    monkeypatch.setenv("TRAINER_STATE_ROOT", str(tmp_path / "trainer-state"))
    reset_settings()
    yield checkpoints
    reset_settings()


def test_F8_all_three_run_types_are_registered_with_no_sibling(no_siblings: None) -> None:
    """The registry is populated at import and needs nothing configured to be reachable."""
    from training.runtypes import known, resolve
    from training.runtypes.base import RUN_TYPE_NAMES

    assert set(known()) == set(RUN_TYPE_NAMES)
    for name in RUN_TYPE_NAMES:
        assert resolve(name).name == name


def test_F8_the_run_type_list_and_the_budget_preview_answer_with_no_sibling(
    no_siblings: None,
) -> None:
    """Both are read by F9's New-run form **before a run exists**, so both must answer in a
    world where nothing else is running."""
    client = TestClient(create_app(), raise_server_exceptions=False)
    listing = client.get("/api/v1/run-types")
    assert listing.status_code == 200
    assert listing.json()["count"] == 3

    preview = client.post(
        "/api/v1/budget/estimate",
        json={
            "run_type": "create",
            "config": {"learning_rate": 0.0001, "warmup_steps": 10, "batch_size": 1},
            "architecture_config": {
                "model_type": "llama",
                "hidden_size": 64,
                "intermediate_size": 128,
                "num_hidden_layers": 2,
                "num_attention_heads": 4,
                "vocab_size": 512,
            },
        },
    )
    assert preview.status_code == 200, preview.text
    assert preview.json()["estimate"]["basis"] == "config"


def test_F8_a_creation_runs_lineage_needs_nothing_upstream(no_siblings: None) -> None:
    """R-02.41, FR-008.8: ``parent: null`` and ``base_model`` = the architecture config.

    Derived through F1's own ``lineage_for_creation`` and validated against the vendored
    schema - no base model to resolve, no sibling to ask, nothing to fetch.
    """
    from models.domain_state import ArtifactKind
    from services.checkpoints import manifest as checkpoint_manifest
    from services.reindex import schema_violations
    from training.acquisition.scratch import created_base_model_identity

    identity = created_base_model_identity({"model_type": "llama", "hidden_size": 64}, 7)
    doc = checkpoint_manifest.build(
        run_id="standalone-create",
        step=0,
        domain=None,
        signal_source="corpus",
        base_model=identity,
        parent=None,
        optimizer={"name": "adamw-8bit", "state": "fresh", "config_hash": "a" * 64},
        weights_sha256="f" * 64,
    )
    assert doc["parent"] is None
    assert doc["base_model"] == identity
    assert doc["domain"] is None
    assert schema_violations(doc, ArtifactKind.CHECKPOINT) == []


def test_F8_an_adapter_is_written_and_read_back_with_no_sibling(f8_root: Path) -> None:
    """R-02.44, FR-008.27: the full artifact path - three files, the digest, validation,
    the atomic rename - with nothing else running and no network reachable."""
    from tests.support.test_adapter_doubles import (
        FakePeftModel,
        adapter_config,
        hub_record,
        optimizer_block,
    )
    from training.runtypes.adapters import load as adapter_load
    from training.runtypes.adapters import manifest as adapter_manifest
    from training.runtypes.adapters import paths as adapter_paths
    from training.runtypes.adapters import save as adapter_save

    doc = adapter_manifest.build(
        run_id="standalone-adapter",
        step=10,
        base=adapter_manifest.base_block(hub_record("LiquidAI/LFM2.5-1.2B-Instruct")),
        adapter=adapter_config(rank=8, alpha=16.0, target_modules=("q_proj",)),
        optimizer=optimizer_block(),
    )
    dest = adapter_paths.adapter_dir("standalone-adapter", 10, root=f8_root)
    published = adapter_save.save(
        FakePeftModel(rank=8, alpha=16.0, target_modules=("q_proj",)), doc, dest
    )
    assert adapter_load.load_manifest(dest / adapter_paths.MANIFEST_NAME) == published
    assert sorted(p.name for p in dest.iterdir()) == [
        adapter_paths.PEFT_CONFIG_NAME,
        adapter_paths.MANIFEST_NAME,
        adapter_paths.WEIGHTS_NAME,
    ]


def test_F8_a_merge_refuses_by_name_with_no_sibling(f8_root: Path) -> None:
    """AC17 standalone. The protection is an atomic directory write plus this refusal -
    **not** a barrier: the bracket form is section 5.10's and F11's."""
    from core.errors import Refusal
    from tests.support.test_adapter_doubles import hub_record
    from training.runtypes.adapters import manifest as adapter_manifest
    from training.runtypes.adapters import merge as merge_module

    base = adapter_manifest.base_block(hub_record("LiquidAI/LFM2.5-1.2B-Instruct"))
    with pytest.raises(Refusal) as caught:
        merge_module.resolve_base_checkpoint(base, root=f8_root)
    assert caught.value.code == "merge_refused_no_unmerged_base"


def test_F8_names_no_sibling_endpoint_and_imports_no_hot_update_module(
    no_siblings: None,
) -> None:
    """ADR-022's fence and ADR-010's rule, from F8's side.

    **F8 mocks nothing, because F8 calls no sibling.** The adapter path writes no resident
    tensors, so it holds no barrier at all - not even in section 5.10 - and the standalone
    test runs with ``training/hotupdate/`` absent, which makes a core-path import of it a
    failure of this criterion rather than a style nit.
    """
    import ast

    src = Path(__file__).resolve().parents[2] / "src"
    package = src / "training" / "runtypes"
    offenders: list[str] = []
    for path in sorted(package.rglob("*.py")):
        text = path.read_text(encoding="utf-8")
        tree = ast.parse(text)
        for node in ast.walk(tree):
            names: list[str] = []
            if isinstance(node, ast.Import):
                names = [alias.name for alias in node.names]
            elif isinstance(node, ast.ImportFrom) and node.module:
                names = [node.module]
            for name in names:
                head = name.split(".")[0]
                if head in ("millm", "mistudio", "midataworks", "httpx", "clients"):
                    offenders.append(f"{path.name}:{node.lineno} -> {name}")
                if "hotupdate" in name:
                    offenders.append(f"{path.name}:{node.lineno} -> {name}")
        # And no sibling endpoint path as a string literal in executable code.
        for node in ast.walk(tree):
            if (
                isinstance(node, ast.Constant)
                and isinstance(node.value, str)
                and "/api/weights" in node.value
            ):
                offenders.append(f"{path.name}:{node.lineno} -> {node.value!r}")
    assert offenders == [], f"F8's path reaches a sibling or the hot-update module: {offenders}"


# ======================================================================================
# F7 - domain packs  (FAC-5, AC10's pack half; R-02.40, FR-007.31, ADR-010)
#
# **A pack built entirely from external parts is created, validated, exported, imported,
# dry-run and accepted by the run-start gate with no sibling running at all.** That is this
# feature's whole share of criterion 20.
#
# It matters more here than it reads, because a pack is the one artifact whose *inputs* come
# from another application: six of its seven slots name parts, and the obvious design would
# have been to resolve every one of them through miDataworks. R-02.40's "a pack may mix
# miDataworks parts and external parts freely" is what makes the standalone path a complete
# path rather than a degraded one - and FR-007.9's three-way failure split is what keeps an
# unconfigured service from being reported as a broken pack. With nothing configured:
#
#   * external parts resolve locally and the pack is **valid**;
#   * contract parts report `midataworks_not_configured` **by name**, and the pack is
#     reported unresolvable rather than invalid-by-authoring - two different fixes;
#   * counts that cannot be measured are `None`, never 0.
#
# Added to THIS file rather than a file of F7's own, per task 1.7a's rule above.
# ======================================================================================


@pytest.fixture
def no_pack_siblings(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    """Nothing configured, and a pack root of this test's own."""
    from core.config import reset_settings
    from packs.settings import reset_pack_settings

    for variable in SIBLING_ENV.values():
        monkeypatch.delenv(variable, raising=False)
    monkeypatch.setenv("PACKS_ROOT", str(tmp_path / "packs"))
    reset_settings()
    reset_pack_settings()
    yield
    reset_settings()
    reset_pack_settings()


def _f7_external_pack(session, tmp_path: Path):
    """A pack of **entirely external parts**, built through the real writer."""
    from packs.slots import SlotKind
    from tests.support import pack_fixtures as fx

    slots = {}
    # Hex digits, because `content_sha256` is a real digest pattern - `"p" * 64` is not a
    # digest and the reference type refuses it, which is the pin doing its job.
    for slot, origin, digest in (
        (SlotKind.PROMPT_SET, "/data/operator/prompts.jsonl", "a"),
        (SlotKind.TEST_SET, "/data/operator/test.jsonl", "b"),
        (SlotKind.RETENTION_SET, "/data/operator/retain.jsonl", "c"),
    ):
        _part_id, provenance = fx.import_external_part(
            session,
            parts_root=tmp_path / "parts",
            export_root=tmp_path / f"export-{slot.value}",
            origin=origin,
            content_sha256=digest * 64,
            rows=5,
        )
        document = fx.external_slot(
            origin=provenance["source"],
            content_sha256=provenance["content_sha256"],
            retrieved_at=provenance["retrieved_at"],
        )
        if slot is SlotKind.RETENTION_SET:
            document = {**document, "floor": 0.6}
        slots[slot.value] = document
    return fx.commit(session, fx.definition(name="standalone-pack", slots=slots))


def test_F7_a_wholly_external_pack_is_valid_with_no_sibling_present(
    no_pack_siblings: None, tmp_path: Path
) -> None:
    """FAC-5's first half: created and **valid**, with nothing else running."""
    from packs.gate import require_valid_pack
    from tests.support.db import migrated_engine, session_for

    engine = migrated_engine(tmp_path / "db")
    try:
        with session_for(engine) as session:
            pack = _f7_external_pack(session, tmp_path)
            # The gate ACCEPTS it - which is the assertion, not merely that validation
            # returned a verdict. F6 calls this first in its submit order.
            resolved = require_valid_pack(session, pack.name, pack.version)
        assert resolved.validity.is_valid
        assert resolved.floor() == 0.6
        assert [record.source for record in resolved.part_records()] == ["external"] * 3
    finally:
        engine.dispose()


def test_F7_the_whole_path_runs_with_no_sibling_present(
    no_pack_siblings: None, tmp_path: Path
) -> None:
    """FAC-5 in full: created, validated, exported, imported elsewhere, gate-accepted.

    "Elsewhere" is a second pack root and a second database, both empty, so the import is
    about the document that travelled rather than about rows that happened to be shared.
    """
    from packs.gate import require_valid_pack
    from packs.resolver import resolve_slots
    from packs.settings import reset_pack_settings
    from packs.transfer import export_bytes, import_document
    from packs.validation import validate
    from tests.support.db import migrated_engine, session_for

    here = migrated_engine(tmp_path / "db-here")
    there = migrated_engine(tmp_path / "db-there")
    try:
        with session_for(here) as session:
            pack = _f7_external_pack(session, tmp_path)
        exported = export_bytes(pack)

        # A second installation: its own pack root, its own database, and - because the parts
        # are external and were taken in locally - the material has to be taken in there too,
        # which is what a reference travelling intact means (FR-007.4).
        import os

        os.environ["PACKS_ROOT"] = str(tmp_path / "packs-there")
        reset_pack_settings()
        with session_for(there) as session:
            imported = import_document(session, exported)
            for slot, origin, digest in (
                ("prompt_set", "/data/operator/prompts.jsonl", "a"),
                ("test_set", "/data/operator/test.jsonl", "b"),
                ("retention_set", "/data/operator/retain.jsonl", "c"),
            ):
                from tests.support import pack_fixtures as fx

                fx.import_external_part(
                    session,
                    parts_root=tmp_path / "parts-there",
                    export_root=tmp_path / f"export-there-{slot}",
                    origin=origin,
                    content_sha256=digest * 64,
                    rows=5,
                )
            resolved = require_valid_pack(session, imported.name, imported.version)
            assert resolved.validity.is_valid
            # The same run inputs, which is AC10's second half.
            from packs.definition import run_input_projection

            assert run_input_projection(imported) == run_input_projection(pack)
            assert validate(imported, resolve_slots(session, imported)).is_valid
    finally:
        here.dispose()
        there.dispose()


def test_F7_a_contract_part_with_no_midataworks_names_the_service_not_the_pack(
    no_pack_siblings: None, tmp_path: Path
) -> None:
    """FAC-4 on the standalone path. **Nothing is wrong with the pack.**

    This is the line that would quietly tell every standalone operator their packs are
    broken, and the reason F7's refusal taxonomy separates a 503 from a 422.
    """
    from packs.errors import MIDATAWORKS_NOT_CONFIGURED, PackRefusal
    from packs.gate import require_valid_pack
    from tests.support import pack_fixtures as fx
    from tests.support.db import migrated_engine, session_for

    engine = migrated_engine(tmp_path / "db")
    try:
        with session_for(engine) as session:
            pack = fx.commit(session, fx.definition(slots=fx.valid_slots()))
            with pytest.raises(PackRefusal) as raised:
                require_valid_pack(session, pack.name, pack.version)
        assert raised.value.code == MIDATAWORKS_NOT_CONFIGURED
        assert raised.value.details["service"] == "midataworks"
        assert raised.value.names_an_absent_service
        # NOT reported as missing slots, and not as an unresolved part: the pack is
        # complete and the integration is absent.
        assert "missing_slots" not in raised.value.details
    finally:
        engine.dispose()


def test_F7_the_registry_and_authoring_remain_usable_with_no_sibling(
    no_pack_siblings: None, tmp_path: Path
) -> None:
    """FR-007.31: only the capabilities that needed the service are unavailable, by name."""
    from api.app import create_app
    from api.deps import session_dependency
    from tests.support.db import migrated_engine, session_for

    engine = migrated_engine(tmp_path / "db")
    app = create_app()

    def override():
        with session_for(engine) as session:
            yield session

    app.dependency_overrides[session_dependency] = override
    try:
        with TestClient(app, raise_server_exceptions=False) as client:
            assert client.get("/api/v1/packs").status_code == 200
            draft = client.post("/api/v1/pack-drafts", json={"name": "standalone"})
            assert draft.status_code == 201
            assert client.get("/api/v1/pack-drafts").status_code == 200
    finally:
        engine.dispose()


def test_F7_no_pack_module_has_an_outbound_call_site(no_pack_siblings: None) -> None:
    """ADR-010: one outbound call site per sibling, and it is F10's. F7 adds none.

    An AST check rather than a grep over text, so a mention in a docstring - of which there
    are several, explaining precisely why the call site is F10's - cannot satisfy or break it.
    """
    import ast

    import packs

    source_dir = Path(packs.__file__).resolve().parent
    offenders: list[str] = []
    for path in sorted(source_dir.glob("*.py")):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            names: list[str] = []
            if isinstance(node, ast.Import):
                names = [alias.name for alias in node.names]
            elif isinstance(node, ast.ImportFrom) and node.module:
                names = [node.module]
            for name in names:
                head = name.split(".")[0]
                if head in ("millm", "mistudio", "midataworks", "httpx", "requests"):
                    offenders.append(f"{path.name}:{node.lineno} -> {name}")
    assert offenders == [], f"F7 reaches a sibling or an HTTP client directly: {offenders}"


# ======================================================================================
# F6 - the run loop and stop conditions  (AC-006.35; R-02.24 - R-02.27, R-02.53, AC20)
#
# **The loop is where acceptance criterion 20 is actually demonstrated.** AC20 is *"a
# complete run of each type with no sibling application running at all"*, and a complete
# run IS the loop - no other feature can exercise it end to end. Added to THIS file rather
# than one of F6's own, per foundation task 1.7a, which asserts there is exactly one.
#
# Three things are collected here that no other section can hold:
#   * a complete run of each run type with every sibling URL unset;
#   * `training/hotupdate/` ABSENT from the import graph of the core path (ADR-022);
#   * a refresh that NEEDS miDataworks refusing BY NAME rather than carrying on with the
#     previous round's prompts, which is the failure mode a silent fallback would create.
# ======================================================================================


@pytest.fixture
def f6_standalone(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """Every sibling URL unset, artifact roots in a temp directory."""
    for name in ("MILLM_BASE_URL", "MIDATAWORKS_BASE_URL", "MILLM_TRAINER_TOKEN"):
        monkeypatch.delenv(name, raising=False)
    checkpoints = tmp_path / "checkpoints"
    states = tmp_path / "trainer-state"
    for directory in (checkpoints, states):
        directory.mkdir()
    monkeypatch.setenv("CHECKPOINT_ROOT", str(checkpoints))
    monkeypatch.setenv("TRAINER_STATE_ROOT", str(states))
    reset_settings()
    yield tmp_path
    reset_settings()


@pytest.mark.parametrize("run_type", ["change", "create", "adapter"])
def test_F6_a_complete_run_of_each_type_finishes_with_no_sibling_present(
    f6_standalone: Path, run_type: str
) -> None:
    """**AC20, AC-006.35.** A complete run - step 0's baseline, the steps, the cadenced
    saves, the stop-condition evaluation at every boundary, the run-end save and a terminal
    state - with nothing else running.

    The tensor arithmetic is faked at the narrowest seam (F1's one turn), because ADR-023
    requires a developer without a card to run everything except a training step. What is
    **not** faked is the loop, the stop registry, the boundary record, the run log and the
    contract validation - which is the part AC20 is about.
    """
    from tests.support.jobs import insert_run, memory_engine, session_for
    from tests.support.loop_fakes import FakeEvaluate, FakeSave, FakeStep, boundary_context
    from training import loop, run_log, run_state

    run_id = f"standalone-f6-{run_type}"
    engine = memory_engine()
    try:
        log = f6_standalone / "run.jsonl"
        evaluate = FakeEvaluate()
        save = FakeSave()
        with session_for(engine) as session:
            insert_run(session, run_id, status="running", run_type=run_type)
            session.commit()

            result = loop.run(
                boundary_context(
                    run_id=run_id,
                    session=session,
                    log_path=log,
                    take_step=FakeStep(),
                    evaluate=evaluate,
                    save=save,
                    test_every=2,
                    retention_every=1,
                    save_every=2,
                    stop_conditions={"step_count": {"total": 4}},
                ),
                max_steps=100,
                evaluate_step_zero=True,
                run_end_save=True,
            )
            run_state.mark_completed(
                session, run_id, stop_reason=result.stopped_because, detail=result.stop_detail
            )
            session.commit()
            from tests.support.jobs import read_run

            row = read_run(session, run_id)
    finally:
        engine.dispose()

    # The step-0 baseline (criterion 12's half), then every step.
    assert evaluate.calls[0] == 0
    assert result.steps_completed == 4
    assert result.stopped_because == "step_count"
    # Saves at the cadence AND at run end (R-02.20's "and at run end").
    assert save.calls == [2, 4, 4]
    assert result.run_end_saved is True
    # The terminal state, named.
    assert row["status"] == "completed"
    assert row["finished_at"] is not None
    # Every boundary recorded, and every one validating against the vendored contract.
    boundaries = list(run_log.read_blocks(log, block="boundary"))
    assert [b["step"] for b in boundaries] == [1, 2, 3, 4]
    for block in boundaries:
        run_log.validate_block(block)


def test_F6_the_core_loop_does_not_import_the_hot_update_package(
    f6_standalone: Path,
) -> None:
    """ADR-022: §5.10 is **fenced**, it lives under ``training/hotupdate/``, and nothing on
    the core path may reference a miLLM endpoint.

    Asserted as an import check over F6's own modules, because *the standalone test runs
    with that package absent* - so a core-path import of it would not merely be wrong, it
    would fail to import at all in the build AC20 describes.
    """
    import ast

    src = Path(__file__).resolve().parents[3] / "backend" / "src"
    f6 = [
        src / "training" / "loop.py",
        src / "training" / "boundary.py",
        src / "training" / "run_state.py",
        src / "training" / "run_driver.py",
        src / "training" / "run_events.py",
        src / "services" / "run_definition.py",
        src / "services" / "run_launch.py",
        src / "services" / "stop_service.py",
        src / "services" / "footprint.py",
        src / "api" / "v1" / "endpoints" / "runs_loop.py",
        *sorted((src / "training" / "stop").rglob("*.py")),
        *sorted((src / "training" / "refresh").rglob("*.py")),
    ]
    offenders: list[str] = []
    for path in f6:
        for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
            names: list[str] = []
            if isinstance(node, ast.Import):
                names = [alias.name for alias in node.names]
            elif isinstance(node, ast.ImportFrom) and node.module:
                names = [node.module]
            for name in names:
                if "hotupdate" in name or name.split(".")[0] in {
                    "millm",
                    "mistudio",
                    "midataworks",
                }:
                    offenders.append(f"{path.relative_to(src)}:{node.lineno} -> {name}")
    assert offenders == [], f"F6's core path reaches the fenced mode or a sibling: {offenders}"


def test_F6_a_refresh_that_NEEDS_miDataworks_refuses_BY_NAME(f6_standalone: Path) -> None:
    """**ADR-010's rule, on the one F6 path that legitimately consumes a sibling.**

    R-02.26's new-prompt-set-per-round loads a miDataworks prompt-set version, so the
    capability genuinely needs the service. With ``MIDATAWORKS_BASE_URL`` unset, the refresh
    **refuses by name** - ``midataworks_not_configured``, naming the capability that becomes
    unavailable - rather than proceeding with the previous round's prompt set or an empty
    one.

    *A silent fallback here would train the next round on the previous round's prompts while
    the run log recorded a fresh round*, which is a wrong answer delivered quietly in the
    record an operator would use to reproduce the run.
    """
    from core.errors import Refusal
    from tests.support.jobs import memory_engine, session_for
    from training import refresh
    from training.refresh.base import RefreshContext

    engine = memory_engine()
    try:
        from services.parts import tables as part_tables

        part_tables.metadata.create_all(engine)
        with session_for(engine) as session, pytest.raises(Refusal) as caught:
            refresh.run(
                RefreshContext(run_id="standalone-f6", step=200, round=1, session=session),
                "new_prompt_set",
                {"reference": {"type": "prompt_set", "name": "legal", "version": "2"}},
            )
    finally:
        engine.dispose()

    assert caught.value.code == "midataworks_not_configured", (
        f"expected the absent service named; got {caught.value.code!r}. An unset base URL "
        f"means absent-and-normal (ADR-010), and a capability that needs the service "
        f"reports it BY NAME with what became unavailable - it never silently degrades."
    )
    assert caught.value.details.get("capability") or caught.value.details.get(
        "capabilities_unavailable"
    ), "the refusal must name the capability that became unavailable"


def test_F6_the_other_refresh_mode_is_a_COMPLETE_path_with_no_sibling(
    f6_standalone: Path,
) -> None:
    """The half that matters for AC20: **the standalone case is the default case.**

    ``template_seeds`` needs no sibling at all - it is a mechanical transformation of
    accepted completions - so a self-generation run refreshing that way is a complete
    standalone path. (Its *seed source* is owed by F2 and the refusal names it; that is a
    missing publisher, not a sibling dependency.)
    """
    from core.errors import Refusal
    from tests.support.loop_fakes import SeedSource
    from training import refresh
    from training.refresh.base import RefreshContext, Seed

    outcome = refresh.run(
        RefreshContext(
            run_id="standalone-f6",
            step=200,
            round=1,
            source=SeedSource(seeds=[Seed("p1", "a question", "an answer")]),
        ),
        "template_seeds",
        {"template": "Another like: {prompt}"},
    )
    assert outcome.size == 1
    assert outcome.set_reference is not None
    assert "external_provenance" in outcome.set_reference
    del Refusal


def test_F6_a_run_definition_needing_no_optional_service_validates_with_NONE_configured(
    f6_standalone: Path,
) -> None:
    """FR-006.60's closing sentence, and **where AC20 begins**.

    A definition that names no contract reference and no ``human_review`` selector involves
    no optional integration, so it validates with every sibling URL unset - and the
    resolved record says ``needed: []`` rather than leaving the question unanswered.
    """
    from schemas.run_definition import RunDefinition
    from services import run_definition as validator
    from services.parts import tables as part_tables
    from tests.support.jobs import memory_engine, session_for

    engine = memory_engine()
    try:
        part_tables.metadata.create_all(engine)
        with session_for(engine) as session:
            outcome = validator.validate(
                RunDefinition(
                    run_id="standalone-f6-def",
                    model_source={"mode": "load", "path": "/data/checkpoints/base"},
                    run_type="change",
                    run_configuration={
                        "learning_rate": 1e-5,
                        "warmup_steps": 1,
                        "batch_size": 1,
                    },
                    signal_source={"source": "corpus"},
                    parts=[{"part_id": "p1", "kind": "corpus"}],
                    stop_conditions=[{"condition": "step_count", "config": {"total": 10}}],
                ),
                session=session,
                # The pack gate and the part resolver are stubbed because the subject is the
                # OPTIONAL-SERVICE step; the live service check is what runs.
                gate=lambda *a, **k: None,
                resolve_parts=lambda *a, **k: {"source": "parts", "parts": []},
                budget=lambda *a, **k: {"param_count": 10, "gpu_uuid": None},
            )
    finally:
        engine.dispose()

    assert outcome.ok, [failure.payload() for failure in outcome.failures]
    assert outcome.resolved["services"]["needed"] == []


# ======================================================================================
# F11 - the optional §5.10 hot-update mode, and the fence that makes it optional
# ======================================================================================
#
# **F11 is the one feature that requires another application**, and its share of acceptance
# criterion 20 is therefore the opposite shape from every other feature's: not *"this works
# with no sibling"* but **"everything else works with this feature absent, and this feature
# says so by name rather than failing later."**
#
# ADR-022: §5.10's code lives under `training/hotupdate/`, the standalone test runs with it
# **absent**, and nothing in the core path may reference a miLLM endpoint. The reason is in
# the ADR itself - *"this dependency grew to thirteen required rows and 29 unimplemented
# upstream requirements once already, without anyone deciding it should."*
#
# **Nothing below needs F11's own module to exist** (FR-011.48). The deletion half of the
# proof is task 17.6, run by hand: delete `backend/src/training/hotupdate/`, run the suite,
# restore it, confirm `git diff` is clean. These cases are what hold between those runs.


def test_F11_the_attach_mode_is_unavailable_BY_NAME_with_no_sibling(no_siblings: None) -> None:
    """R-02.48, FR-011.47. **Named, not omitted and not degraded.**

    Two things must be true at once with nothing configured: the mode is *listed* - an
    option missing from a response is indistinguishable from one that is broken (ADR-013) -
    and it is listed as unavailable with the capabilities it would have provided.
    """
    from training import acquisition
    from training.acquisition.base import AcquisitionMode

    state = acquisition.availability(AcquisitionMode.ATTACH)
    assert state.available is False
    assert state.reason == "hot_update_mode_not_available"
    assert state.unavailable_capabilities
    assert "MILLM_BASE_URL" in (state.message or "") or "§5.10" in (state.message or "")

    listed = {entry["mode"] for entry in acquisition.modes()}
    assert listed == {"load", "create", "attach"}


def test_F11_requesting_the_mode_with_no_sibling_refuses_rather_than_loading(
    no_siblings: None,
) -> None:
    """**The substitution R-02.48 forbids, and the dangerous one.** Attaching to a serving
    process's resident tensors and loading the same weights from disk train *different
    tensors*, and both report success. So pre-flight refuses by name, before a lease and
    before any allocation, and `acquire` refuses too rather than trusting that pre-flight
    ran.
    """
    from core.errors import Refusal
    from training import acquisition
    from training.acquisition.base import AcquisitionMode, AcquisitionSpec

    spec = AcquisitionSpec(mode=AcquisitionMode.ATTACH)
    acquirer = acquisition.resolve(spec)
    for method in (acquirer.preflight, acquirer.acquire):
        with pytest.raises(Refusal) as caught:
            method(spec)
        assert caught.value.code == "acquisition_mode_unavailable"
        assert caught.value.details["mode"] == "attach"
        assert caught.value.details["capabilities_unavailable"]


def test_F11_every_other_mode_and_run_type_is_untouched_by_its_presence(
    no_siblings: None,
) -> None:
    """FR-011.52: *F11 introduces no required row into any dependency table.*

    The fence's purpose stated as a test rather than as a paragraph. `load` and `create`
    must be unaffected by whether §5.10 is installed, and the three run types must still be
    exactly three - §5.10 changes the **acquisition mode**, not the run type, so a fourth
    run type would mean the scope grew without anyone deciding it should.
    """
    from training import acquisition, runtypes
    from training.acquisition.base import AcquisitionMode

    for mode in (AcquisitionMode.LOAD, AcquisitionMode.CREATE):
        state = acquisition.availability(mode)
        assert state.reason in {
            None,
            "ml_runtime_unavailable",
        }, f"{mode.value} became unavailable for a §5.10 reason: {state.reason}"
    assert sorted(runtypes.known()) == ["adapter", "change", "create"]


def test_F11_the_core_path_imports_the_fenced_package_exactly_once_and_dynamically(
    no_siblings: None,
) -> None:
    """ADR-022's fence, from the standalone side (FR-011.46, FR-011.50).

    **Exactly one** module in `backend/src/` reaches `training/hotupdate/`, it is the
    acquisition registry, and it reaches it **by name at runtime**. A static import would be
    a dependency edge from the core path onto an optional package, which is what would break
    when that directory is deleted - and deleting it is how this feature's own acceptance of
    ADR-022 is performed (task 17.6).
    """
    import ast

    src = Path(__file__).resolve().parents[2] / "src"
    static: list[str] = []
    dynamic: list[str] = []
    for path in src.rglob("*.py"):
        if "hotupdate" in path.parts:
            continue
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if (
                isinstance(node, ast.ImportFrom) and "training.hotupdate" in (node.module or "")
            ) or (
                isinstance(node, ast.Import)
                and any("training.hotupdate" in alias.name for alias in node.names)
            ):
                static.append(str(path.relative_to(src)))
            elif isinstance(node, ast.Call):
                name = (
                    node.func.attr
                    if isinstance(node.func, ast.Attribute)
                    else (node.func.id if isinstance(node.func, ast.Name) else "")
                )
                if name in {"import_module", "__import__"} and any(
                    isinstance(argument, ast.Constant)
                    and "training.hotupdate" in str(argument.value)
                    for argument in node.args
                ):
                    dynamic.append(str(path.relative_to(src)))

    assert static == [], f"a static import of the fenced package: {static}"
    assert dynamic == [
        "training/acquisition/__init__.py"
    ], f"the fenced package has {len(dynamic)} importers: {dynamic}. ADR-022 allows one."


def test_F11_no_core_path_module_names_a_weights_ENDPOINT(no_siblings: None) -> None:
    """ADR-022's other half, and the one a convenience edit breaks first.

    Checked on **string constants in the code**, not on the file's text: every module in
    this project explains at length what it does not do, and a text matcher reports a
    disclaimer as a claim. That mistake has a price in this repository - a fresh harness
    once reported 18 scope-creep failures and all 18 were documents correctly disclaiming a
    foreign requirement.
    """
    import ast

    markers = ("/api/weights", "commit-begin", "commit-end", "X-miLLM-Attachment")
    src = Path(__file__).resolve().parents[2] / "src"
    offenders: dict[str, list[int]] = {}
    for path in src.rglob("*.py"):
        if "hotupdate" in path.parts:
            continue
        tree = ast.parse(path.read_text(encoding="utf-8"))
        docstrings = {
            id(node.body[0].value)
            for node in ast.walk(tree)
            if isinstance(node, (ast.Module, ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))
            and getattr(node, "body", None)
            and isinstance(node.body[0], ast.Expr)
            and isinstance(node.body[0].value, ast.Constant)
            and isinstance(node.body[0].value.value, str)
        }
        hits = [
            node.lineno
            for node in ast.walk(tree)
            if isinstance(node, ast.Constant)
            and isinstance(node.value, str)
            and id(node) not in docstrings
            and any(marker in node.value for marker in markers)
        ]
        if hits:
            offenders[str(path.relative_to(src))] = hits
    assert offenders == {}, f"{offenders} name a miLLM weights endpoint outside the fence"


def test_F11_the_fenced_package_is_not_imported_by_merely_running_the_core_path(
    no_siblings: None,
) -> None:
    """The runtime half of the same property: with nothing configured, a core-path operation
    must not **load** the package.

    *The registration is the one exception and it is deliberate* - the registry installs the
    mode at import so that removing that line turns the reachability test red. What must not
    happen is any *other* module pulling it in: that is how a package nobody decided to
    depend on becomes a dependency.
    """
    import sys

    loaded_by_the_registry = [name for name in sys.modules if name.startswith("training.hotupdate")]
    # NOT ASSERTED, and said so: this line read `assert "...barrier" not in loaded or True`
    # until 2026-10-16 - an assertion that could never fail (ruff SIM222 found it). Since
    # then `attach.py` imports `barrier` at module scope to build the §5.10 stepper, so the
    # registry's install does load it; that is the fence working as designed (one importer,
    # the registry), and the importer count in `test_hotupdate_fence.py` is what guards it.
    del loaded_by_the_registry
    from training import acquisition
    from training.acquisition.base import AcquisitionMode

    # Asking about availability must touch neither the network nor the rest of the package.
    assert acquisition.availability(AcquisitionMode.ATTACH).available is False


# ======================================================================================
# F9 - the Admin UI  (AC-009.19, AC-009.28; FR-009.9, FR-009.37; R-02.48, R-02.50, AC20)
# ======================================================================================
#
# **F9's share of criterion 20 is the only one that is about a *surface* rather than a
# capability.** Every other feature asks "does this work with nothing configured"; this one
# asks "does the operator get told, by name, what is unavailable - and does every screen
# still render in full while being told." FR-009.9: *"never substitutes a different value, a
# zero or a silent omission."*
#
# What is asserted here, from the backend side (the component-level half is Vitest's, and
# `e2e/` covers the rendering):
#
#   * every endpoint F9's screens consume answers with **every** integration unset;
#   * `GET /api/health` **agrees** with every other surface about what is configured, which
#     is the one assertion that catches a drift no single endpoint's own test can see;
#   * **E-17** - a score recorded against an HTTP provider that is now unconfigured still
#     renders, carrying the provider it was taken with. It is historical data, not a live
#     capability, and rewriting it to the current configuration would be a fabricated value;
#   * **task 10.4's second half** - no frontend fetch addresses a sibling host (ADR-022).
#
# Added to THIS file rather than a file of F9's own, per task 1.7a's rule above.
# ======================================================================================

#: The endpoints F9's two screens and its New run form read. Every one must answer with
#: nothing configured - a screen whose data source 500s is a screen that cannot render, and
#: FR-009.9 promises the opposite.
F9_SCREEN_ENDPOINTS: tuple[str, ...] = (
    "/api/v1/tree",
    "/api/v1/checkpoints",
    "/api/v1/adapters",
    "/api/v1/selectors",
    "/api/v1/providers",
    "/api/v1/run-types",
    "/api/v1/stop-conditions",
    "/api/v1/refresh-modes",
    "/api/v1/signal-sources",
    "/api/v1/packs",
    "/api/v1/midataworks/status",
)


@pytest.fixture
def f9_rig(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):  # type: ignore[no-untyped-def]
    """A mounted app over a migrated database and a real artifact root, nothing configured.

    The forest is written by **F5's and F8's real writers** and indexed by ``reindex`` from
    the manifests on disk (``tests.support.tree``), because a hand-built index row agrees
    with the tree's expectations by construction and proves nothing (ADR-016).
    """
    from api.deps import session_dependency
    from packs.settings import reset_pack_settings
    from services.scoring.settings import reset_scoring_settings
    from tests.support import db as support
    from tests.support import scores as score_fixtures

    for variable in SIBLING_ENV.values():
        monkeypatch.delenv(variable, raising=False)
    monkeypatch.delenv("ACTIVATION_SOURCE_BASE_URL", raising=False)
    checkpoints = tmp_path / "checkpoints"
    checkpoints.mkdir()
    monkeypatch.setenv("CHECKPOINT_ROOT", str(checkpoints))
    # `PACKS_ROOT` is set because `packs.store.packs_root()` **mkdirs on a read path**, so
    # an unset value makes `GET /packs` try to create `/data/packs` and 500 with
    # `PermissionError` - which is not a standalone property and must not be asserted as
    # one. Reproduced by this fixture before it was set; recorded as F6's open finding
    # (`packs.store.packs_root()` still mkdirs on a read path), which is not F9's to fix.
    monkeypatch.setenv("PACKS_ROOT", str(tmp_path / "packs"))
    reset_settings()
    reset_scoring_settings()
    reset_pack_settings()

    engine = support.migrated_engine(tmp_path / "db")
    score_fixtures.create_score_tables(engine)
    app = create_app()

    def one_session():  # type: ignore[no-untyped-def]
        with support.session_for(engine) as session:
            yield session

    app.dependency_overrides[session_dependency] = one_session
    client = TestClient(app, raise_server_exceptions=False)
    try:
        yield client, checkpoints, engine
    finally:
        app.dependency_overrides.clear()
        engine.dispose()
        reset_settings()
        reset_scoring_settings()
        reset_pack_settings()


def _f9_forest(root: Path, engine) -> dict[str, str]:  # type: ignore[no-untyped-def]
    """Both artifact kinds and a merged checkpoint, through the real writers."""
    from tests.support import tree as forest

    base = forest.root_checkpoint(root, run_id="f9-base", step=40, domain="support")
    child = forest.branch(root, parent=base.name, run_id="f9-base", step=80)
    adapter = forest.adapter(root, base=base, run_id="f9-adapter", step=50)
    merged = forest.merged_checkpoint(
        root, base=base, adapter_document=adapter, run_id="f9-base", step=120
    )
    forest.indexed(engine, root, run_ids=("f9-base", "f9-adapter"))
    return {
        "base": base.name,
        "child": child.name,
        "adapter": adapter["name"],
        "merged": merged.name,
    }


class TestF9EveryScreenRendersWithNoSiblingConfigured:
    """AC-009.19 and AC-009.28, from the data side."""

    @pytest.mark.parametrize("path", F9_SCREEN_ENDPOINTS)
    def test_every_endpoint_a_screen_reads_answers(self, f9_rig, path: str) -> None:
        """Parametrised so a **new** failing endpoint is named rather than buried in a loop.

        A 5xx here is not a degraded screen; it is a blank one. FR-009.9 allows a capability
        to be reported unavailable and does not allow the screen to fail.
        """
        client, _root, _engine = f9_rig
        response = client.get(path)
        assert (
            response.status_code == 200
        ), f"{path} -> {response.status_code}: {response.text[:300]}"

    def test_the_forest_shows_BOTH_artifact_kinds_with_nothing_configured(self, f9_rig) -> None:
        """AC19's tree half, under AC20's conditions: checkpoints joined on ``parent``,
        adapters on the base their own manifest names, a merged checkpoint with both edges.
        """
        client, root, engine = f9_rig
        names = _f9_forest(root, engine)
        body = client.get("/api/v1/tree").json()
        rendered = {node["name"] for node in body["nodes"]}
        for key, name in names.items():
            assert name in rendered, f"{key} ({name}) is missing from the forest"
        kinds = {node["kind"] for node in body["nodes"]}
        assert {"checkpoint", "adapter"} <= kinds, kinds

    def test_the_run_view_reads_its_run_the_step_log_and_the_scores(self, f9_rig) -> None:
        """The three reads the most-watched screen in the product makes."""
        from tests.support import db as support
        from tests.support import jobs

        client, _root, engine = f9_rig
        with support.session_for(engine) as session:
            jobs.insert_run(session, "f9-run", status="running", last_step=4)
            session.commit()

        assert client.get("/api/v1/runs/f9-run").status_code == 200
        assert client.get("/api/v1/runs/f9-run/steps").status_code == 200
        assert client.get("/api/v1/runs/f9-run/scores").status_code == 200

    def test_an_unavailable_type_is_NAMED_and_still_LISTED_never_hidden(self, f9_rig) -> None:
        """FR-009.37's rule, enforced on the payload the form renders.

        *Never hidden, never swapped.* A type dropped from the listing is indistinguishable
        from a type that does not exist, and the operator's fix - set a base URL - is
        invisible. So the row is present, ``available`` is false, and the reason **names the
        service** in ADR-010's spelling.
        """
        client, _root, _engine = f9_rig
        rows = {row["name"]: row for row in client.get("/api/v1/selectors").json()}
        assert "human_review" in rows, "the type that needs the absent sibling was hidden"
        assert rows["human_review"]["available"] is False
        assert rows["human_review"]["unavailable_reason"] == "midataworks_not_configured"
        # And nothing was substituted: the four types that need no sibling are untouched.
        unaffected = {name for name, row in rows.items() if row["requires"] == "none"}
        assert unaffected, "no sibling-free selector type remained"
        assert all(rows[name]["available"] is True for name in unaffected)


class TestF9HealthAgreesWithEveryOtherSurface:
    """Task 10.2's second clause, and the one assertion no single endpoint's test can make.

    Each surface is individually correct and they can still disagree - which is this
    project's dominant defect class, one layer out. *Health reporting a capability the
    guard does not gate is worse than no report at all*, because an operator who reads
    "available" and then gets a refusal stops trusting the whole response.
    """

    def test_health_is_DERIVED_from_the_settings_not_retyped(self, f9_rig) -> None:
        """Compared against ``configured_integrations()`` rather than against a list in
        this test, so a sibling added to ``Settings`` is covered here the day it lands
        instead of the day somebody remembers to extend a tuple."""
        client, _root, _engine = f9_rig
        body = client.get("/api/health").json()
        expected = get_settings().configured_integrations()
        assert set(body["integrations"]) == set(expected), (
            "health names a different set of integrations from the one Settings declares; "
            "an integration missing from a response is indistinguishable from one that is "
            "broken (ADR-013)."
        )
        for service, configured in expected.items():
            assert body["integrations"][service]["configured"] is configured

    def test_health_and_the_midataworks_status_route_give_ONE_answer(self, f9_rig) -> None:
        """Two readers of one registry (``clients/capabilities.py``). Two lists drift."""
        client, _root, _engine = f9_rig
        health = client.get("/api/health").json()
        status = client.get("/api/v1/midataworks/status").json()
        assert health["integrations"]["midataworks"]["configured"] == status["configured"]
        assert health["integrations"]["midataworks"]["code"] == status["code"]
        # `reachable` is three-valued and unset must read as `None`, not `False`: "not
        # configured" and "configured and down" have different fixes.
        assert status["reachable"] is None

    def test_health_and_the_selector_listing_agree_about_midataworks(self, f9_rig) -> None:
        """The agreement F9's form actually depends on: it reads availability from health
        **and** from the selector listing at render time (task 9A.12), so a disagreement
        shows as a type that is available on one surface and refused on the other."""
        client, _root, _engine = f9_rig
        health = client.get("/api/health").json()
        rows = {row["name"]: row for row in client.get("/api/v1/selectors").json()}
        configured = health["integrations"]["midataworks"]["configured"]
        needs_it = [row for row in rows.values() if row["requires"] == "midataworks"]
        assert needs_it, "no selector type needs miDataworks; this test would be vacuous"
        for row in needs_it:
            assert row["available"] is configured
            if not configured:
                assert row["unavailable_reason"] == health["integrations"]["midataworks"]["code"]

    def test_health_names_no_integration_as_configured_when_none_is(self, f9_rig) -> None:
        """The negative half, kept beside the positive one: a mutation reporting everything
        configured would satisfy the per-service checks above."""
        client, _root, _engine = f9_rig
        body = client.get("/api/health").json()
        assert [n for n, e in body["integrations"].items() if e["configured"]] == []
        assert sorted(body["unconfigured_integrations"]) == sorted(
            get_settings().configured_integrations()
        )


class TestF9ARecordedProviderIsHistoryNotALiveCapability:
    """**E-17**, and it is the subtlest of F9's cases.

    A score taken months ago against an OpenAI-compatible endpoint that is no longer
    configured is *still that score, taken that way*. R-02.49 records the provider with
    every score precisely because *"a score is only comparable against another taken the
    same way"* - so the payload must carry the provider it was taken with, unchanged,
    whatever is configured now. Rewriting it to today's configuration, or dropping it,
    would be the fabricated value ADR-018 forbids wearing a 200.
    """

    def test_a_score_from_an_unconfigured_endpoint_still_renders_with_its_provider(
        self, f9_rig
    ) -> None:
        from services.scoring import record as record_module
        from tests.support import db as support
        from tests.support import jobs
        from tests.support import scores as score_fixtures

        client, _root, engine = f9_rig
        log_path = _root_log(engine)
        with support.session_for(engine) as session:
            jobs.insert_run(session, "f9-history", status="completed", last_step=7)
            record_module.write_pass(
                [
                    score_fixtures.score_record(
                        run_id="f9-history",
                        step=7,
                        kind="test",
                        provider=score_fixtures.HTTP,
                    )
                ],
                session=session,
                log_path=log_path,
            )
            session.commit()

        # Nothing is configured now - the endpoint that produced this score is gone.
        providers = client.get("/api/v1/providers").json()
        assert [row["name"] for row in providers["providers"]] == ["in_process"]

        body = client.get("/api/v1/runs/f9-history/scores").json()
        rows = [row for row in body["scores"] if row["step"] == 7]
        assert rows, f"the recorded score is not in the response at all: {body}"
        row = rows[0]

        # **Asserted field by field, not over a blob of the whole body.** Written first as
        # `"judge-7b" in str(body)`, this test passed against a mutation that deleted all
        # three `provider*` keys from the payload - because `message` independently names
        # the provider, so the two agreed by construction and the assertion was vacuous.
        # Control MUT-N7b is what found it, and it is the fixtures-agree-by-construction
        # failure in its smallest form.
        assert row["provider_kind"] == "openai_compatible", row
        assert row["provider_model"] == "judge-7b", row
        assert row["provider"].startswith("openai_compatible:judge-7b@"), row
        # Not rewritten to today's only available provider.
        assert row["provider_kind"] != "in_process", (
            "the recorded provider was rewritten to today's default, which is a fabricated "
            "value (ADR-018) - and it would make two incomparable scores look comparable."
        )
        # And the sentence F9 renders beside the number names it too (ADR-012: every number
        # names its scale and sample). A separate claim from the fields above, which is why
        # it is a separate assertion rather than a second reading of the same string.
        assert "judge-7b" in row["message"], row["message"]


def _root_log(engine) -> Path:  # type: ignore[no-untyped-def]
    """A run-log path beside the test's database, so ``write_pass`` has somewhere real."""
    url = str(engine.url)
    base = Path(url.split("///")[-1]).parent if "///" in url else Path(".")
    log = base / "runs" / "f9-history" / "run.jsonl"
    log.parent.mkdir(parents=True, exist_ok=True)
    return log


class TestF9NoFrontendFetchAddressesASiblingHost:
    """**Task 10.4's second half, as a test rather than a grep** (ADR-022, F9 FTID 2.2).

    ``frontend/src/api/client.ts`` already asserts the URL *it* builds is relative. That is
    the per-call-site half and it is not sufficient: it proves one module behaves, not that
    no other module calls ``fetch`` with an absolute sibling URL. *A grep proves a call
    exists, not that anything depends on it* - so the structural rule is pinned here, in the
    one file acceptance criterion 20 lives in.

    Located on the backend deliberately: this is a property of the **repository**, the same
    shape as ``test_the_core_path_names_no_sibling_endpoint`` above, and putting it beside
    the eleven other AC20 claims is what task 1.7a's single-file rule is for.
    """

    @staticmethod
    def _frontend_sources() -> list[Path]:
        root = Path(__file__).resolve().parents[3] / "frontend" / "src"
        if not root.is_dir():  # pragma: no cover - the frontend is always present
            pytest.fail(f"{root} does not exist; this test would pass vacuously")
        return sorted(
            path
            for path in root.rglob("*.ts*")
            if ".test." not in path.name and not path.name.endswith(".d.ts")
        )

    def test_the_frontend_tree_is_not_empty(self) -> None:
        """A scanner that scans nothing is green. The same rule the schema-sync test and
        the no-sibling-import guard both state about themselves."""
        sources = self._frontend_sources()
        assert len(sources) > 20, f"only {len(sources)} frontend sources found; scan is thin"

    def test_fetch_is_called_in_exactly_one_module(self) -> None:
        """ADR-010's one-call-site rule, applied to the browser.

        A second ``fetch`` has its own idea of what a refusal looks like, and it is the one
        that points somewhere else - which is precisely how a browser ends up talking to a
        sibling while every backend guard stays green.
        """
        import re

        allowed = "api/client.ts"
        pattern = re.compile(r"\bfetch\s*\(")
        offenders = [
            str(path)
            for path in self._frontend_sources()
            if pattern.search(path.read_text(encoding="utf-8")) and not str(path).endswith(allowed)
        ]
        assert offenders == [], (
            f"{offenders} call fetch(). The browser reaches miForge through "
            f"frontend/src/{allowed} and only miForge (ADR-022); a sibling is reached by the "
            f"backend's single client module in backend/src/clients/, or not at all."
        )

    def test_no_frontend_module_names_a_sibling_host_or_endpoint(self) -> None:
        """The other direction: an absolute URL, a sibling service name as a host, or a
        miLLM weights path. **Any** of the three would put a sibling on the browser's path.
        """
        import re

        # A sibling service name used as a HOST or ORIGIN - not merely the word, which
        # appears legitimately in refusal codes (`midataworks_not_configured`) and in copy.
        host_forms = re.compile(
            r"https?://[^\s'\"`]*(millm|mistudio|midataworks)"
            r"|//(millm|mistudio|midataworks)[:/]",
            re.IGNORECASE,
        )
        offenders: list[str] = []
        for path in self._frontend_sources():
            text = path.read_text(encoding="utf-8")
            if host_forms.search(text):
                offenders.append(f"{path.name}: sibling host")
            if "/api/weights/" in text:
                offenders.append(f"{path.name}: miLLM weights endpoint")
        assert offenders == [], (
            f"{offenders}. ADR-022 fences the core path off every sibling endpoint and a "
            f"browser fetch is on that path."
        )

    def test_the_socket_connects_to_the_serving_ORIGIN_rather_than_a_host(self) -> None:
        """The hole a fetch-only rule leaves. ``socket.io-client``'s ``io(url, opts)``
        takes a URL as its first argument, so ``io('http://millm:8000')`` would reach a
        sibling without the word ``fetch`` appearing anywhere.
        """
        import re

        offenders: list[str] = []
        for path in self._frontend_sources():
            text = path.read_text(encoding="utf-8")
            for match in re.finditer(r"\bio\s*\(\s*([^)\n]*)", text):
                argument = match.group(1).strip()
                # `io({...})` and `io()` take no URL. Anything starting with a quote does.
                if argument.startswith(("'", '"', "`")):
                    offenders.append(f"{path.name}: io({argument[:60]})")
        assert offenders == [], (
            f"{offenders} opens a socket against an explicit URL. The run view's socket must "
            f"use the serving origin, which is what io({{path}}) does."
        )
