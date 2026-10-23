"""The Celery task is a wrapper, and three callers reach one implementation  (FR-004.14a).

**Where a scoring pass runs is decided by whose VRAM it generates from, not by whether it is
generation.** FR-004.14a settled a conflict three feature documents each stated consistently
and contradicted each other on:

* the in-cadence pass F6's loop awaits runs in the ``train`` worker, **synchronously**, by a
  direct call to :func:`services.scoring.service.evaluate` - ADR-009's in-process provider
  generates from parameters that exist **only** in that process, a ``generate`` worker would
  need a second copy of them on the leased card, and a cross-queue ``.get()`` from a
  ``--pool=solo -c 1`` task blocks the only worker thread on a queue it cannot drain;
* ``generate`` carries exactly the HTTP-provider pass and the out-of-cadence route.

So the task must be a wrapper. *Two implementations would mean a pass run through the worker
and a pass run in the loop could produce different numbers* - and nothing would say which was
right.

**Two deviations are asserted here rather than left to a reader**: the task's name and the
module's location. Both are consequences of files this feature does not own, and both are
reported.
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

SRC = Path(__file__).resolve().parents[2] / "src"
TASKS = SRC / "services" / "scoring" / "tasks.py"


class TestTheTaskIsRegisteredAndRouted:
    def test_it_is_in_the_live_celery_registry(self) -> None:
        """Importing the module registers it; a task registered nowhere that matters is a
        fully-tested capability that does not exist for any caller."""
        from services.scoring.tasks import TASK_EVALUATE
        from workers.celery_app import celery_app

        assert TASK_EVALUATE in celery_app.tasks

    def test_it_resolves_to_the_generate_queue(self) -> None:
        """``test_janitor_wiring`` fails for any registered task that resolves to no route -
        *"a task with no route silently lands on default, where a worker with no card picks
        up a training task and fails."*"""
        from services.scoring.tasks import TASK_EVALUATE
        from workers.celery_app import QUEUE_GENERATE, TASK_ROUTES

        matched = [
            route["queue"]
            for pattern, route in TASK_ROUTES.items()
            if TASK_EVALUATE.startswith(pattern.rstrip("*"))
        ]
        assert matched == [QUEUE_GENERATE]

    def test_the_name_is_NOT_the_one_this_features_documents_give_it(self) -> None:
        """The deviation, pinned so it is not "corrected" back into a broken state.

        F4's FTASKS and FTID name the task ``miforge.scoring.evaluate``.
        ``workers.celery_app.TASK_ROUTES`` has **no** ``miforge.scoring.*`` entry, so that
        name resolves to no route and the wiring test goes red. Adding the route is a
        one-line edit to a file this feature does not own; the name is a constant here so
        the rename is one line on this side too, whenever the coordinator adds it.
        """
        from services.scoring.tasks import TASK_EVALUATE

        assert TASK_EVALUATE == "miforge.generate.evaluate"
        assert TASK_EVALUATE != "miforge.scoring.evaluate"

    def test_the_worker_imports_this_module(self) -> None:
        """**WIRED 2026-10-14**, and it closed two gaps with one line.

        A worker starts as ``celery -A workers.celery_app:celery_app`` and imports
        ``TASK_MODULES`` and nothing else, so an enqueued evaluation reached a worker that
        did not have the task. F10 hit the identical shape within a day of that tuple's
        docstring being written about it.

        **The second gap is the one worth knowing:** importing ``services.scoring.tasks``
        is also what binds F3's ``ProviderResolver``, because ``services/scoring/wiring.py``
        runs as an import side effect of the package. Without this line a ``train`` or
        ``generate`` worker had no resolver bound at all, so the judge-model selector
        refused *whatever the operator configured* - which is the gap F4 found called in
        fourteen test files and nowhere in ``backend/src/``.
        """
        from workers.celery_app import TASK_MODULES

        assert "services.scoring.tasks" in TASK_MODULES, (
            "the worker does not import F4's task module, so an enqueued evaluation "
            "reaches a worker without the task AND no ProviderResolver is bound there."
        )

    def test_the_task_body_is_one_delegating_call(self) -> None:
        tree = ast.parse(TASKS.read_text(encoding="utf-8"))
        function = next(
            node
            for node in ast.walk(tree)
            if isinstance(node, ast.FunctionDef) and node.name == "evaluate_task"
        )
        statements = [node for node in function.body if not isinstance(node, ast.Expr)]
        assert len(statements) <= 2, (
            "the task has grown logic of its own; it is a wrapper, and a second "
            "implementation is how a pass run through the worker and a pass run in the loop "
            "produce different numbers"
        )

    def test_it_imports_no_scoring_primitive(self) -> None:
        """It must not reach aggregation, the floor comparison or the record directly."""
        tree = ast.parse(TASKS.read_text(encoding="utf-8"))
        imported = {
            node.module
            for node in ast.walk(tree)
            if isinstance(node, ast.ImportFrom) and node.module
        }
        forbidden = {
            "services.scoring.aggregate",
            "services.scoring.retention",
            "services.scoring.record",
            "services.scoring.expectations",
        }
        assert imported & forbidden == set()

    def test_it_declares_no_queue_of_its_own_beyond_the_existing_one(self) -> None:
        from services.scoring.tasks import QUEUE_GENERATE
        from workers.celery_app import QUEUES

        assert QUEUE_GENERATE in QUEUES


class TestThreeCallersOneImplementation:
    def test_the_route_the_task_and_the_loop_all_reach_one_function(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Asserted on the **call count** of the service function, not on "something was
        called"."""
        from services.scoring import entrypoint, service, tasks

        calls: list[tuple[int, bool]] = []

        def spy(session, **kwargs):
            calls.append((kwargs["step"], bool(kwargs.get("out_of_cadence"))))
            return {"block": "evaluation"}

        monkeypatch.setattr(entrypoint, "_evaluate_with", spy)
        del service

        tasks.evaluate_task("run-1", 3)
        assert len(calls) == 1, "the task reaches the one entry point exactly once"
        assert calls[0] == (3, True), "an out-of-cadence pass is recorded as one"

    def test_the_task_delegates_to_the_entrypoint_module(self) -> None:
        source = TASKS.read_text(encoding="utf-8")
        assert "entrypoint.evaluate_run" in source

    def test_the_service_is_callable_with_no_broker(self) -> None:
        """The property F6's loop depends on, and an AC20 case: no Redis, no result backend,
        no second worker."""
        import inspect

        from services.scoring import service

        assert not inspect.iscoroutinefunction(service.evaluate)
        # No Celery **import**, and no enqueue. Asserted over the imports rather than over
        # the source text, because the module's own docstring explains at length why the
        # Celery task is a wrapper - and a guard that forbade the explanation would make the
        # reasoning unwritable in the place it applies.
        tree = ast.parse(inspect.getsource(service))
        imported = {
            name
            for node in ast.walk(tree)
            for name in (
                [alias.name for alias in node.names]
                if isinstance(node, ast.Import)
                else ([node.module] if isinstance(node, ast.ImportFrom) and node.module else [])
            )
        }
        assert not any(name.split(".")[0] in ("celery", "workers") for name in imported)
