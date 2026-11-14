"""The step-loop task is reachable, and the submit order is the requirement.

**The reachability half.** ``celery -A workers.celery_app:celery_app worker`` imports *that
module and nothing else*, so a task defined in a module no one imports is a fully
implemented, fully unit-tested capability that **does not exist for any caller**. That is the
sibling's sixteen unregistered MCP tools, in this package: every one of those tests passed by
importing the module directly, so the suite was green and the documentation said ✅ while the
feature did not exist. So this file asserts the task is in the **live registry after Celery's
own default-module import**, and the registration is one line in ``TASK_MODULES`` - delete it
and this file goes red.

**The ordering half.** ADR-008: *the budget check runs before any allocation.* Acquisition
**is** the first allocation, so the guard's call site is the last moment the check means
anything. Asserted with **the payload and the call count**, because "the guard was called" is
satisfied by a call made after the allocation it was meant to precede - which is exactly the
mutation the sweep targets.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from core.errors import Refusal
from core.gpu import Device
from training.acquisition.base import AcquisitionMode, AcquisitionSpec
from training.run_config import RunConfig

CFG = RunConfig(learning_rate=1e-5, warmup_steps=1, batch_size=1, gradient_accumulation=1)
LLAMA = {
    "model_type": "llama",
    "hidden_size": 64,
    "num_hidden_layers": 2,
    "num_attention_heads": 4,
    "vocab_size": 256,
}


@pytest.fixture
def create_spec(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> AcquisitionSpec:
    """A ``create`` spec that gets **past the acquirer's own pre-flight.**

    A creation run names the tokenizer its checkpoint will ship (2026-10-16), and
    ``ScratchAcquirer.preflight`` refuses ``tokenizer_required`` without one - **before**
    the budget guard, which is correct by the ordering this module's own comments state
    (most specific first; the acquirer's gate runs first). So a spec with no tokenizer no
    longer reaches the guard at all, and these ordering cases have to carry one to be
    about ordering.
    """
    from core.config import Settings, reset_settings
    from tests.support.tiny_hf_model import write_tokenizer_json_only

    checkpoints = tmp_path / "checkpoints"
    checkpoints.mkdir()
    monkeypatch.setattr(
        "core.config._settings",
        Settings(_env_file=None, checkpoint_root=checkpoints, parts_root=checkpoints),  # type: ignore[call-arg]
    )
    tokenizer = write_tokenizer_json_only(checkpoints / "tok", vocab_size=LLAMA["vocab_size"])
    yield AcquisitionSpec(mode=AcquisitionMode.CREATE, config=dict(LLAMA), tokenizer_path=tokenizer)
    reset_settings()


class TestTheTaskIsInTheLiveRegistry:
    def test_the_task_is_registered_after_celerys_own_module_import(self) -> None:
        """**Through Celery's loader, not by importing the module here.** Importing it
        directly is precisely the test that passed for sixteen unregistered tools."""
        from workers.celery_app import celery_app

        celery_app.loader.import_default_modules()
        assert "miforge.train.run_step_loop" in celery_app.tasks, (
            "the task is not in the live registry: a worker started with "
            "`-A workers.celery_app:celery_app` would never import its module, so the task "
            "would not exist for any caller. Check TASK_MODULES."
        )

    def test_the_registration_line_is_what_makes_it_reachable(self) -> None:
        """The removal test, as a permanent assertion rather than a manual control: the
        module must be *named* in the configuration the worker reads."""
        from workers.celery_app import TASK_MODULES, celery_app

        assert "training.tasks" in TASK_MODULES
        assert (
            tuple(celery_app.conf.include) == TASK_MODULES
        ), "TASK_MODULES must be the `include` Celery actually uses, not a parallel list"

    def test_it_routes_to_the_train_queue(self) -> None:
        """ADR-006's route table is **explicit**, and a task matching no entry lands on
        ``default`` - where a worker with no card picks it up and fails."""
        from training.tasks import TASK_RUN_STEP_LOOP, queue_of

        assert queue_of() == "train"
        assert queue_of(TASK_RUN_STEP_LOOP) == "train"

    def test_the_name_this_features_DOCUMENTS_use_would_route_NOWHERE(self) -> None:
        """**A document defect, pinned.** All four of F1's documents name the task
        ``miforge.training.run_step_loop``. The foundation's table is keyed
        ``miforge.train.*`` / ``miforge.generate.*`` / ``miforge.maintenance.*`` /
        ``miforge.index.*`` / ``miforge.packs.*``, and ADR-006 names the areas after the
        queues they contend on - so the documented name matches no entry and would land on
        ``default``.

        Asserted so that nobody "corrects" the code back to the documents.
        """
        from workers.celery_app import queue_for

        assert queue_for("miforge.training.run_step_loop") is None
        assert queue_for("miforge.train.run_step_loop") == "train"

    def test_the_route_is_resolved_through_celerys_own_router(self) -> None:
        """A second implementation of the glob match is a second answer, and the one a
        hand-rolled test checked would not be the one the broker uses."""
        import ast

        source = Path(
            Path(__file__).resolve().parents[3] / "backend" / "src" / "training" / "tasks.py"
        ).read_text(encoding="utf-8")
        assert "queue_for" in source
        for node in ast.walk(ast.parse(source)):
            # No literal queue name may be compared against: that would be the second copy.
            if isinstance(node, ast.Compare):
                for operand in [node.left, *node.comparators]:
                    assert not (
                        isinstance(operand, ast.Constant) and operand.value == "train"
                    ), "the queue must be resolved, not asserted against a literal"


class TestTheTaskRefusesRatherThanReportingAPhantomRun:
    """**Updated 2026-10-14, when F6 filled the seam this class was written about.**

    It asserted ``signal_source_unavailable`` with ``owed_by == ["F2 ...", "F6 ..."]`` -
    F1's honest placeholder for a task whose collaborators did not exist, because *a task
    that silently returned "done" would report a run that never happened*. F2 and F6 are
    both built now and ``run_step_loop`` drives
    :func:`training.run_driver.drive_recorded_run`, so the old assertion pinned a
    placeholder rather than a property. **Rewritten rather than deleted**: the property it
    was protecting is still live and is now stronger - the task still refuses by name
    rather than reporting a run that did not happen, for the case that can actually occur.
    """

    def test_a_run_with_no_recorded_definition_is_refused_by_name(self) -> None:
        """R-02.53 requires a recorded definition before a run is enqueued.

        A row created by F1's pre-R-02.53 ``POST /api/v1/runs`` has none, and the driver
        **refuses** it rather than running it unbounded on the node's only card. That
        refusal is what makes the ungated submit path unable to produce a run that trains,
        so there is exactly one path to a running run and R-02.53's gate sits on it.
        """
        from tests.support.jobs import insert_run, memory_engine, session_for
        from training import run_driver

        engine = memory_engine()
        with session_for(engine) as session:
            insert_run(session, "run-x", status="pending")
            session.commit()
            with pytest.raises(Refusal) as caught:
                run_driver.drive_recorded_run("run-x", session=session)
        assert caught.value.code == "run_definition_missing"
        assert "POST /api/v1/runs/define" in caught.value.message

    def test_the_task_delegates_to_the_driver_rather_than_refusing(self) -> None:
        """The seam is filled, asserted on the source rather than by reading it.

        Without this the previous test would still pass against a task that refused
        unconditionally - the refusal would be the task's, not the driver's, and nothing
        would notice that the loop was unreachable from production.
        """
        import ast

        source = (
            Path(__file__).resolve().parents[3] / "backend" / "src" / "training" / "tasks.py"
        ).read_text(encoding="utf-8")
        tree = ast.parse(source)
        task = next(
            node
            for node in ast.walk(tree)
            if isinstance(node, ast.FunctionDef) and node.name == "run_step_loop"
        )
        called = {
            node.func.attr
            for node in ast.walk(task)
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
        }
        assert "drive_recorded_run" in called, (
            "run_step_loop must drive F6's loop; if this is gone the whole loop is "
            "unreachable from production and every unit test of it still passes"
        )

    def test_the_driver_is_a_separate_plain_function_so_F1s_path_is_testable(self) -> None:
        """Two layers, deliberately: the task resolves a batch source, and ``drive_run``
        takes one as an argument. That split is what makes acquire -> record -> step ->
        record -> trainer state exercisable without a broker and without F2."""
        from training import tasks

        assert callable(tasks.drive_run)
        import inspect

        assert "batch_source" in inspect.signature(tasks.drive_run).parameters


class TestTheBudgetGuardRunsBeforeAnyAllocation:
    """ADR-008, FR-001.8, and the mutation that deletes the call site."""

    def test_the_guard_is_called_once_with_the_enriched_spec_and_the_config(
        self, create_spec: AcquisitionSpec
    ) -> None:
        from services import acquisition as service

        class Spy:
            def __init__(self) -> None:
                self.calls: list[tuple[Any, Any]] = []

            def check(self, spec: AcquisitionSpec, cfg: RunConfig) -> None:
                self.calls.append((spec, cfg))

        spy = Spy()
        spec = create_spec
        try:
            service.preflight(spec, CFG, guard=spy)
        except Refusal as exc:
            # On a machine with no ML runtime the mode is unavailable and pre-flight refuses
            # *after* the guard - which is itself the ordering under test.
            assert exc.code == "acquisition_mode_unavailable"

        assert (
            len(spy.calls) == 1
        ), f"the guard must be called exactly once per submit, was called {len(spy.calls)}"
        seen_spec, seen_cfg = spy.calls[0]
        # THE PAYLOAD. A guard handed an un-enriched spec would have neither a parameter
        # count nor a parsed config and would refuse every run as unestimable - or, with the
        # neither-input rule mutated to a pass, would permit every run.
        assert seen_spec.architecture_config == LLAMA
        assert seen_cfg is CFG

    def test_a_load_hands_the_guard_a_PARAMETER_COUNT_not_a_config(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from core.config import Settings, reset_settings
        from services import acquisition as service
        from tests.unit.test_acquisition_local_dir import build_model_dir

        checkpoints = tmp_path / "checkpoints"
        checkpoints.mkdir()
        monkeypatch.setattr(
            "core.config._settings",
            Settings(_env_file=None, checkpoint_root=checkpoints, parts_root=checkpoints),  # type: ignore[call-arg]
        )
        directory = build_model_dir(checkpoints)

        seen: list[AcquisitionSpec] = []

        class Spy:
            def check(self, spec: AcquisitionSpec, cfg: RunConfig) -> None:
                seen.append(spec)

        try:
            service.preflight(AcquisitionSpec(path=directory), CFG, guard=Spy())
        except Refusal as exc:
            assert exc.code == "acquisition_mode_unavailable"
        reset_settings()

        assert len(seen) == 1
        assert seen[0].param_count == 32 * 8 + 8 * 8
        assert (
            seen[0].architecture_config is None
        ), "exactly one input: both present is refused as ambiguous (F8 FR-008.10)"

    def test_the_guard_refusing_stops_the_submit_BEFORE_the_lease(
        self, create_spec: AcquisitionSpec
    ) -> None:
        """A refused budget must take no lease: a lease held by a run that never started
        blocks the card until its expiry, for nothing."""
        from services import acquisition as service

        class Refusing:
            def check(self, spec: AcquisitionSpec, cfg: RunConfig) -> None:
                raise Refusal("budget_exceeded", "too big", {"estimate": {}})

        with pytest.raises(Refusal) as caught:
            service.preflight(create_spec, CFG, guard=Refusing())
        # And `budget_exceeded` rather than the runtime gate's refusal: the three gates are
        # ordered most-specific-first, so the number the operator can change is named before
        # the installation they would otherwise go and fix for a run that would not fit.
        assert caught.value.code == "budget_exceeded"

    def test_the_bound_guard_is_the_COMPUTED_one_when_a_device_was_resolved(self) -> None:
        """**R-02.42 pulled forward by operator decision (2026-10-14).** F1's OQ-1 asked
        whether the estimator moves into this wave or whether a pass-through ships for two
        waves; this is the binding that answers it, and F8's work here becomes *confirm, not
        rebuild*."""
        from services.acquisition import guard_for
        from training.budget import PassThroughGuard
        from training.budget_guard import ComputedBudgetGuard

        device = Device(uuid="GPU-1", name="card", index=0, free_memory_bytes=24 * 1024**3)
        assert isinstance(guard_for(device), ComputedBudgetGuard)
        # With no card there is no allocation to guard, and the pass-through logs loudly.
        assert isinstance(guard_for(None), PassThroughGuard)

    def test_the_call_site_exists_in_the_service_rather_than_in_the_acquirer(self) -> None:
        """The guard must precede ``acquire()``, and the only place that can be arranged is
        the layer that calls both. A guard invoked *inside* an acquirer would run after the
        acquirer had already decided to allocate."""
        import ast

        path = (
            Path(__file__).resolve().parents[3] / "backend" / "src" / "services" / "acquisition.py"
        )
        tree = ast.parse(path.read_text(encoding="utf-8"))
        preflight = next(
            node
            for node in ast.walk(tree)
            if isinstance(node, ast.FunctionDef) and node.name == "preflight"
        )
        checks = [
            node
            for node in ast.walk(preflight)
            if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr == "check"
        ]
        assert len(checks) == 1, "exactly one budget check in the submit path"


class TestNoDeviceIsResolvedTwice:
    def test_a_resolved_device_is_passed_to_the_lease_rather_than_re_resolved(self) -> None:
        """Resolving twice invites the two answers to differ, and the second answer is the
        one the work actually ran on while the first is the one recorded."""
        import ast

        path = (
            Path(__file__).resolve().parents[3] / "backend" / "src" / "services" / "acquisition.py"
        )
        tree = ast.parse(path.read_text(encoding="utf-8"))
        submit = next(
            node
            for node in ast.walk(tree)
            if isinstance(node, ast.FunctionDef) and node.name == "submit"
        )
        lease_calls = [
            node
            for node in ast.walk(submit)
            if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr == "acquire"
        ]
        assert len(lease_calls) == 1
        assert any(
            kw.arg == "device" for kw in lease_calls[0].keywords
        ), "the lease must be handed the already-resolved device"

    def test_this_feature_adds_no_device_resolution_of_its_own(self) -> None:
        """ADR-008, and the hard-coded-GPU ratchet's allowlist stays empty: an index inverted
        once on this node and the work silently ran on the other card; a copied bus address
        unbound the root NVMe."""
        import ast

        src = Path(__file__).resolve().parents[3] / "backend" / "src"
        for path in [*(src / "training").rglob("*.py"), src / "services" / "acquisition.py"]:
            if "hotupdate" in path.parts:
                continue
            for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
                if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
                    assert node.func.attr != "run" or "nvidia" not in ast.dump(
                        node
                    ), f"{path.name} shells out to nvidia-smi; core/gpu.py owns that"
