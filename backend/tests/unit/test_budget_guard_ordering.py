"""The budget guard runs **before** anything is allocated  (FR-008.16; ADR-008, ADR-016 #1).

**M1's target.** The mutation is *"move the* ``budget_hook.check(...)`` *call to after*
``acquirer.acquire()``", and this file is what must go red.

## Why this file exists at all

F1's ``test_budget_guard.py`` opens by listing *"2. The guard runs before* ``acquire()``.
*Asserted with the payload and the call count"* - and **no such test is in that file**, nor
anywhere else in the suite (checked by searching for one before writing this). A docstring
promising a capability is not a requirement defining one, and *a capability is not shipped
until a test FAILS when its wiring is removed*. So the assertion is made here, in F8's own
file, rather than reported and left for someone else: F8 is the feature whose requirement
the ordering is (R-02.42, *"refuse before the run starts"*).

## The three things asserted, because two of them are not enough

* **Ordering** - the acquirer's own pre-flight, then the guard, and ``acquire()`` **not at
  all** during submit.
* **Payload** - the guard receives the spec *as pre-flight enriched it*, which is the only
  version that carries a parameter count. A guard handed the raw spec would refuse every
  load for having no input.
* **Call count** - exactly one. *"Was called" passes against a call sending the wrong
  arguments*, and it also passes against a guard called twice with the second call
  overwriting the first verdict.
"""

from __future__ import annotations

from typing import Any

import pytest

from core.errors import Refusal
from services import acquisition as acquisition_service
from training import acquisition as registry
from training.acquisition.base import (
    AcquisitionMode,
    AcquisitionRecord,
    AcquisitionSpec,
    Availability,
    SourceKind,
)
from training.budget import PassThroughGuard
from training.run_config import RunConfig


def cfg() -> RunConfig:
    return RunConfig(learning_rate=1e-4, warmup_steps=10, batch_size=2)


class RecordingAcquirer:
    """An acquirer that writes every call into a shared log, in order."""

    def __init__(self, log: list[str]) -> None:
        self.log = log

    @property
    def mode(self) -> AcquisitionMode:
        return AcquisitionMode.LOAD

    def available(self) -> Availability:
        return Availability(available=True)

    def preflight(self, spec: AcquisitionSpec) -> AcquisitionSpec:
        self.log.append("acquirer.preflight")
        # Pre-flight is where the parameter count becomes known - which is precisely why
        # the guard runs after it and before `acquire`.
        return spec.model_copy(update={"param_count": 1_200_000_000})

    def acquire(self, spec: AcquisitionSpec) -> Any:
        self.log.append("acquirer.acquire")
        return AcquisitionRecord(
            mode=AcquisitionMode.LOAD,
            source_kind=SourceKind.HUB,
            starting_weights_sha256="a" * 64,
            base_model="org/model",
        )


class SpyGuard:
    """A guard that records its calls. **F1's** ``PassThroughGuard`` shape, kept deliberately.

    F1's class is not deleted and this one mirrors it: *"keep the pass-through class, which
    the ordering test uses as a spy"* - and its loud log line documents the window in which
    runs were admitted unmeasured.
    """

    name = "spy"

    def __init__(self, log: list[str]) -> None:
        self.log = log
        self.calls: list[tuple[AcquisitionSpec, RunConfig]] = []

    def check(self, spec: AcquisitionSpec, config: RunConfig) -> None:
        self.log.append("guard.check")
        self.calls.append((spec, config))


@pytest.fixture
def wired(monkeypatch: pytest.MonkeyPatch) -> tuple[list[str], SpyGuard]:
    log: list[str] = []
    acquirer = RecordingAcquirer(log)
    monkeypatch.setattr(registry, "resolve", lambda spec: acquirer)
    monkeypatch.setattr(registry, "availability", lambda mode: Availability(available=True))
    return log, SpyGuard(log)


def spec() -> AcquisitionSpec:
    return AcquisitionSpec(mode=AcquisitionMode.LOAD, repo="org/model", revision="main")


class TestTheGuardPrecedesTheAllocation:
    def test_preflight_calls_the_guard_and_NOT_acquire(
        self, wired: tuple[list[str], SpyGuard]
    ) -> None:
        log, guard = wired
        acquisition_service.preflight(spec(), cfg(), guard=guard)
        assert log == ["acquirer.preflight", "guard.check"]
        assert (
            "acquirer.acquire" not in log
        ), "acquisition IS the first allocation, so a budget check after it is worthless"

    def test_the_guard_is_called_EXACTLY_ONCE(self, wired: tuple[list[str], SpyGuard]) -> None:
        """A second call would overwrite the first verdict, and "was called" would pass."""
        log, guard = wired
        acquisition_service.preflight(spec(), cfg(), guard=guard)
        assert len(guard.calls) == 1
        assert log.count("guard.check") == 1

    def test_the_PAYLOAD_is_the_ENRICHED_spec_not_the_raw_one(
        self, wired: tuple[list[str], SpyGuard]
    ) -> None:
        """The enrichment is the parameter count. A guard handed the raw spec would refuse
        every load for carrying neither input - the right behaviour against the wrong
        argument, which is what makes payload assertions necessary."""
        _, guard = wired
        raw = spec()
        assert raw.param_count is None
        acquisition_service.preflight(raw, cfg(), guard=guard)
        seen_spec, seen_cfg = guard.calls[0]
        assert seen_spec.param_count == 1_200_000_000
        assert seen_spec is not raw
        assert seen_cfg is not None and seen_cfg.batch_size == 2

    def test_the_SECOND_argument_is_the_run_config(self, wired: tuple[list[str], SpyGuard]) -> None:
        """Two-argument, F1's shape, adopted rather than proposed: ``cfg`` is where the
        batch size and the optimizer precision the terms need actually live."""
        _, guard = guard_and_log = wired
        acquisition_service.preflight(spec(), cfg(), guard=guard)
        _, seen_cfg = guard.calls[0]
        assert isinstance(seen_cfg, RunConfig)
        assert guard_and_log[0].index("guard.check") == 1

    def test_a_refusing_guard_stops_the_submit_before_the_lease_or_the_enqueue(
        self, wired: tuple[list[str], SpyGuard], monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """AC13: refused **before it starts**, not failed later by the allocator."""
        log, _ = wired
        enqueued: list[str] = []

        class Refusing:
            name = "refusing"

            def check(self, s: AcquisitionSpec, c: RunConfig) -> None:
                log.append("guard.check")
                raise Refusal("budget_exceeded", "too big", {"estimate_bytes": 1, "free_bytes": 0})

        monkeypatch.setattr(acquisition_service, "resolve_submit_device", lambda request: None)
        with pytest.raises(Refusal) as caught:
            acquisition_service.submit(
                session=object(),
                run_id="run-1",
                spec=spec(),
                cfg=cfg(),
                guard=Refusing(),
                enqueue=enqueued.append,
            )
        assert caught.value.code == "budget_exceeded"
        assert enqueued == [], "nothing is enqueued on a refusal"
        assert "acquirer.acquire" not in log


class TestAcquireHappensInTheWorkerAfterSubmit:
    """The structural half: ``acquire()`` is reached only from the worker's entry point, so
    the guard's call site in submit is necessarily earlier."""

    def test_the_only_acquire_call_site_is_the_worker_task(self) -> None:
        import ast
        from pathlib import Path

        src = Path(__file__).resolve().parents[3] / "backend" / "src"
        sites: list[str] = []
        for path in sorted(src.rglob("*.py")):
            if "hotupdate" in path.parts or path.name in {"base.py", "__init__.py"}:
                continue
            for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
                if (
                    isinstance(node, ast.Call)
                    and isinstance(node.func, ast.Attribute)
                    and node.func.attr == "acquire"
                    # `acquirer.acquire(...)`, not `lease_module.acquire(...)`. The GPU
                    # lease shares the verb and is taken inside submit *after* the guard,
                    # which is correct and must not be mistaken for an allocation of
                    # parameters - a check that flagged it would be a check about the
                    # wrong call that happened to be red.
                    and isinstance(node.func.value, ast.Name)
                    and node.func.value.id == "acquirer"
                ):
                    sites.append(f"{path.relative_to(src)}:{node.lineno}")
        assert sites and all(
            site.startswith("training/tasks.py") for site in sites
        ), f"acquirer.acquire() must be reached only from the worker task; found {sites}"

    def test_submit_runs_preflight_before_the_lease(
        self, wired: tuple[list[str], SpyGuard], monkeypatch: pytest.MonkeyPatch
    ) -> None:
        log, guard = wired
        monkeypatch.setattr(acquisition_service, "resolve_submit_device", lambda request: None)
        enqueued: list[str] = []
        acquisition_service.submit(
            session=object(),
            run_id="run-1",
            spec=spec(),
            cfg=cfg(),
            guard=guard,
            enqueue=enqueued.append,
        )
        assert log == ["acquirer.preflight", "guard.check"]
        assert enqueued == ["run-1"]


class TestF1sPassThroughIsKeptAsTheSpy:
    def test_the_pass_through_class_still_exists_and_records_its_calls(self) -> None:
        """F8 must not delete it: it is what documents the window in which runs were
        admitted unmeasured, and another feature's test uses it."""
        guard = PassThroughGuard()
        raw = spec()
        guard.check(raw, cfg())
        assert guard.calls == [(raw, cfg())]

    def test_the_real_guard_is_bound_when_a_device_resolves(self) -> None:
        """F8's work at this seam is *confirm, not rebuild*: the operator pulled R-02.42's
        estimator forward into F1 on 2026-10-14, so the binding already exists."""
        from core.gpu import Device
        from training.budget_guard import ComputedBudgetGuard

        device = Device(uuid="GPU-" + "0" * 32, index=0, name="NVIDIA GeForce RTX 3090")
        assert isinstance(acquisition_service.guard_for(device), ComputedBudgetGuard)
        assert isinstance(acquisition_service.guard_for(None), PassThroughGuard)
