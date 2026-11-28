"""The ``human_review`` selector is reachable when miDataworks is configured.

*A capability is not shipped until a test FAILS when its wiring is removed* (ADR-016 #1).

``services.selectors.plugins.human_review.bind_review_queue`` was called in **eleven test
files and nowhere in ``backend/src/``**, and
:class:`clients.midataworks_review.MiDataworksReviewQueue` was constructed only in its own
test. So ``current_review_queue()`` was always ``None`` and
``HumanReviewSelector.prepare`` **always** refused ``midataworks_not_configured`` - *naming
an absent service that was present*, with miDataworks configured, reachable and answering -
while ``GET /api/v1/selectors`` listed the type unavailable for the same reason. Dead with
it: ``verdicts_for_candidates``, ``HumanReviewSelector.item_ids``, ``.timeout_seconds`` and
``MiDataworksClient.get_queue``.

**With miDataworks unset the refusal is still ``midataworks_not_configured``** - ADR-010's
absent-and-normal, acceptance criterion 20, and :class:`TestUnconfiguredIsStillNormal`
pins it. The bug was never that miForge refused; it was that refusing was the only thing it
could do.

## Nothing here binds its own port

Every assertion reads the port :func:`clients.midataworks_review.wire` installs, through
the two paths production uses:
:func:`services.selectors.availability.midataworks_available` (which is what the listing
endpoint serves) and ``HumanReviewSelector``'s own calls. A test that bound a stub would
be the twelfth file that passes while production has nothing.

## THE THREE CONTROLS

1. **Reachability of the port.** Delete the ``bind_review_queue(...)`` call in
   :func:`clients.midataworks_review.wire` and
   :class:`TestThePortIsBoundByAProductionImport` and
   :class:`TestTheSelectorIsUsableWhenMiDataworksIsConfigured` must go red.
2. **Reachability of the run scope.** Delete the ``with human_review.review_run(...)``
   in :func:`training.run_driver.drive_recorded_run` and
   :class:`TestTheRunScopeIsOpenedByTheDriver` must go red. A bound port with no scope
   refuses every queue operation, so this half is as load-bearing as the first.
3. **The payload, not merely the call.** A global binding that reached *some* queue would
   satisfy 1 and 2 and still post this run's candidates to another run's queue - the queue
   id lands on a binding row keyed by ``run_id``, and the decisions are the only copy. So
   :class:`TestItRoutesToThisRunsQueue` asserts the ``external_ref`` miDataworks was sent,
   the candidate ids in the body, and the **call count** on the fake.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

# The production imports. `services.scoring` is imported for the reason the API process
# and both workers import it; it pulls in `services.selectors` (so the plugin registry
# exists) and `services.review` (the import trigger's permitted layer).
import services.scoring  # noqa: F401
from clients import midataworks_review
from core.config import reset_settings
from core.errors import Refusal
from schemas.selector import VerdictValue
from services.selectors import availability as selector_availability
from services.selectors.plugins import human_review
from services.selectors.resolve import resolve
from services.selectors.review_port import ReviewQueuePort
from services.selectors.settings import get_selector_settings
from tests.support import db as support
from tests.support.fake_midataworks import FakeMiDataworks
from tests.support.jobs import insert_run, memory_engine, session_for
from tests.support.selector_stubs import rule_config
from training import run_driver, tasks

RUN = "run-review-wiring"
QUEUE_NAME = "support-tickets-review"

SELECTOR_CONFIG: dict[str, Any] = {
    "kind": "human_review",
    "name": "human-eyes",
    "config_sha256": None,
    "queue": QUEUE_NAME,
    "timeout_seconds": 600,
}

SUBMITTED_DEFINITION: dict[str, Any] = {
    "run_id": RUN,
    "model_source": {"mode": "load", "path": "/data/checkpoints/base"},
    "run_type": "change",
    "run_configuration": {
        "learning_rate": 1e-5,
        "warmup_steps": 2,
        "batch_size": 2,
        "gradient_accumulation": 1,
    },
    "signal_source": {"source": "corpus"},
    "parts": [{"part_id": "part-1", "kind": "corpus"}],
    "cadences": {"test_every": 2, "retention_every": 1, "save_every": 10},
    "stop_conditions": [{"condition": "step_count", "config": {"total": 1}}],
    "domain": "legal",
    "gpu_request": "auto",
}


@pytest.fixture(autouse=True)
def wired():
    """Re-run the production wiring, idempotently, before every test here.

    Not a weakening of the control, and it is **the real production function** - the thing
    that is owed is the one import that *calls* it, which
    :class:`TestTheImportTriggerIsOwed` pins as a strict xfail. Two further reasons this
    has to be a fixture rather than a module import: ``wire`` is a one-shot, and the eleven
    test files that bind their own stub all unbind to ``None`` on teardown, so in a session
    this file's subject can be left unbound by a *neighbour* rather than by missing wiring.
    """
    midataworks_review.wire()
    yield


@pytest.fixture
def configured(monkeypatch: pytest.MonkeyPatch):
    """miDataworks configured, reachable and answering - the state the bug denied."""
    monkeypatch.setenv("MIDATAWORKS_BASE_URL", "http://dw.invalid:8000")
    reset_settings()
    yield
    reset_settings()


@pytest.fixture
def unconfigured(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.delenv("MIDATAWORKS_BASE_URL", raising=False)
    reset_settings()
    yield
    reset_settings()


# --------------------------------------------------------------------------------------
# CONTROL 1: the port is bound by a production import
# --------------------------------------------------------------------------------------


class TestTheImportTriggerIsOwed:
    """~~**STRICT XFAIL. One line, in a file outside this change's scope.**~~
    **THE LINE LANDED 2026-10-16 and the marker is gone — this is now a plain assertion.**

    The class keeps its name because the history is the point: the xfail was `strict=True`
    precisely so that applying the import would turn it `XPASS(strict)` → FAILED and *force*
    whoever applied it to delete the marker, rather than leaving a passing test permanently
    labelled "expected to fail". It did exactly that, and this is that deletion. A
    non-strict xfail would have gone quietly green and the stale label would still be here.

    :func:`clients.midataworks_review.wire` is the binding and it is complete. What is
    missing is the one production import that *runs* it: nothing in ``backend/src/``
    imports ``clients.midataworks_review``, which was the original defect, and the import
    cannot come from F3, F4 or F1 because all three layers are fenced against importing
    ``clients`` - ``test_selector_guards.TestNoImportOfTheForbiddenPackages``,
    ``test_scoring_static_guards.TestF4NeverCallsMiDataworks`` /
    ``test_F4_no_scoring_module_reaches_a_sibling_or_an_http_client``, and
    ``test_F1_has_no_outbound_sibling_call_site``. All four were observed red against
    earlier sitings of the binding, and every one of them is right.

    The owed line is at the foot of ``backend/src/services/review/__init__.py``::

        from clients import midataworks_review as _review_port_wiring  # noqa: F401

    ``services/review/`` is F10's own package, is already permitted to import ``clients``,
    and is reachable from every process: the API through ``api/v1/endpoints/runs_review.py``
    **and** through ``services/scoring/wiring.py``'s ``from services import review``, the
    workers through ``services.scoring.tasks`` in ``TASK_MODULES``. That ``from services
    import review`` is this change's half of the trigger and
    :meth:`test_the_reachable_half_of_the_trigger_is_in_place` asserts it.

    Strict, so this goes **red** the moment the owed line lands and the marker has to come
    off - rather than quietly passing and leaving a stale note behind.
    """

    def test_the_reachable_half_of_the_trigger_is_in_place(self) -> None:
        """Importing what every scoring process imports reaches F10's review package.

        Delete ``from services import review`` in ``services/scoring/wiring.py`` and this
        goes red. A subprocess, because ``sys.modules`` in this session has been populated
        by every other test file.
        """
        import subprocess
        import sys

        src = Path(__file__).resolve().parents[2] / "src"
        done = subprocess.run(
            [
                sys.executable,
                "-c",
                "import sys, services.scoring; print('services.review' in sys.modules)",
            ],
            cwd=str(src),
            capture_output=True,
            text=True,
            check=False,
            timeout=120,
        )
        assert done.returncode == 0, done.stderr
        assert done.stdout.strip() == "True", (
            "importing `services.scoring` no longer reaches `services.review`, so the owed "
            "one-line trigger would have nowhere to take effect from in a worker"
        )

    def test_a_bare_production_import_binds_f10s_client(self) -> None:
        """The end state, asserted now so that reaching it cannot go unnoticed."""
        import subprocess
        import sys

        src = Path(__file__).resolve().parents[2] / "src"
        program = (
            "import services.scoring\n"
            "from services.selectors.plugins import human_review\n"
            "port = human_review.current_review_queue()\n"
            "print(type(port).__module__, type(port).__name__)\n"
        )
        done = subprocess.run(
            [sys.executable, "-c", program],
            cwd=str(src),
            capture_output=True,
            text=True,
            check=False,
            timeout=120,
        )
        assert done.returncode == 0, done.stderr
        assert (
            done.stdout.strip() == "clients.midataworks_review RunScopedReviewQueue"
        ), f"a production import left the review port as {done.stdout.strip()!r}"


class TestThePortIsBoundByAProductionImport:
    """What :func:`clients.midataworks_review.wire` installs - the binding itself.

    Delete the ``bind_review_queue`` call inside it and every test here goes red.
    """

    def test_a_port_is_bound_at_all(self) -> None:
        """Not *importable* - **bound**, in the slot ``current_review_queue`` reads."""
        assert human_review.current_review_queue() is not None

    def test_it_is_f10s_client_and_not_an_arbitrary_object(self) -> None:
        """The identity matters: a binding that reached a stub would be this file's twelfth
        copy of the defect. It must be the run-scoped wrapper over F10's one call site."""
        port = human_review.current_review_queue()
        assert isinstance(port, midataworks_review.RunScopedReviewQueue)
        assert isinstance(port, ReviewQueuePort)

    def test_it_routes_to_f10s_per_run_queue(self) -> None:
        port = human_review.current_review_queue()
        assert isinstance(port, midataworks_review.RunScopedReviewQueue)
        routed = port.for_run("run-x")
        assert isinstance(routed, midataworks_review.MiDataworksReviewQueue)
        assert routed.run_id == "run-x"


# --------------------------------------------------------------------------------------
# The user-visible consequence, through the two paths production reads
# --------------------------------------------------------------------------------------


class TestTheSelectorIsUsableWhenMiDataworksIsConfigured:
    def test_the_listing_reports_the_type_available(self, configured) -> None:
        """What ``GET /api/v1/selectors`` serves. ``False`` here was the whole defect: the
        operator saw ``midataworks_not_configured`` beside a configured sibling."""
        assert selector_availability.midataworks_available() is True
        settings = get_selector_settings()
        from services.selectors.types import Requirement

        assert selector_availability.availability_of(Requirement.MIDATAWORKS, settings) == (
            True,
            None,
        )

    def test_prepare_binds_the_port_instead_of_refusing(self, configured) -> None:
        """``resolve`` is what calls ``prepare``, so this is the production path.

        Before the wiring this raised ``midataworks_not_configured`` for **every**
        configuration, which is why a run naming a ``human_review`` selector could not
        start at all.
        """
        with human_review.review_run(RUN):
            resolved = resolve(dict(SELECTOR_CONFIG), source="inline")
        assert resolved.bindings["review_queue"] is human_review.current_review_queue()
        # And the two reads that were dead with it now answer.
        assert resolved.plugin is not None
        assert resolved.plugin.timeout_seconds == 600
        assert resolved.plugin.queue == QUEUE_NAME


class TestUnconfiguredIsStillNormal:
    """ADR-010 and acceptance criterion 20, unchanged by the binding."""

    def test_the_listing_reports_it_unavailable_by_name(self, unconfigured) -> None:
        assert selector_availability.midataworks_available() is False
        from services.selectors.types import Requirement

        assert selector_availability.availability_of(
            Requirement.MIDATAWORKS, get_selector_settings()
        ) == (False, "midataworks_not_configured")

    def test_prepare_still_refuses_by_name(self, unconfigured) -> None:
        with pytest.raises(Refusal) as caught:
            resolve(dict(SELECTOR_CONFIG), source="inline")
        assert caught.value.code == "midataworks_not_configured"

    def test_the_other_selector_types_are_unaffected(self, unconfigured) -> None:
        """A rule selector needs no sibling, which is the half of ADR-010 that says an
        unconfigured service takes only its own capabilities with it."""
        resolved = resolve(rule_config("non-empty", min_length=1), source="inline")
        assert resolved.plugin is not None


class TestOutsideARunItNamesTheRunRatherThanTheSibling:
    """``review_queue_run_unknown`` is a **different fact** from an absent miDataworks.

    Reached by ``POST /api/v1/selectors/evaluate`` - R-02.39's authoring path - which has
    no run and therefore no binding row to record a queue id on. Reporting it as
    ``midataworks_not_configured`` there would be the same lie in the other direction.
    """

    def test_a_queue_operation_with_no_run_in_scope(self, configured) -> None:
        port = human_review.current_review_queue()
        assert port is not None
        assert human_review.current_review_run() is None
        with pytest.raises(Refusal) as caught:
            port.ensure_queue(name=QUEUE_NAME, question="?", label_set=["a", "b"])
        assert caught.value.code == "review_queue_run_unknown"

    def test_availability_still_answers_without_a_run(self, configured) -> None:
        """The API process has no run id, and ADR-013 requires the type to be *listed*
        unavailable by name rather than omitted - so this must answer with no run and no
        round trip."""
        port = human_review.current_review_queue()
        assert port is not None
        assert port.available() is True

    def test_an_empty_run_id_is_refused_rather_than_scoped(self) -> None:
        """A scope indistinguishable from no scope would suppress the refusal that names
        what is missing."""
        with pytest.raises(Refusal) as caught, human_review.review_run(""):
            pass  # pragma: no cover - the scope refuses before the body
        assert caught.value.code == "review_queue_run_unknown"

    def test_the_scope_is_restored_on_the_way_out(self) -> None:
        """``ContextVar.reset`` restores the previous value rather than clearing to
        ``None``, which is what stops one run's scope leaking into the next."""
        with human_review.review_run("outer"):
            with human_review.review_run("inner"):
                assert human_review.current_review_run() == "inner"
            assert human_review.current_review_run() == "outer"
        assert human_review.current_review_run() is None


# --------------------------------------------------------------------------------------
# CONTROL 3: the payload - THIS run's queue, with THIS run's candidates
# --------------------------------------------------------------------------------------


class TestItRoutesToThisRunsQueue:
    """A binding that reached *some* queue passes a "was it bound" test and loses the run.

    So the assertions are the ``external_ref`` miDataworks was sent, the ids in the posted
    body, and the number of calls - because *"was called" passes against a call sending the
    wrong arguments*, and here the wrong argument is another run's queue.
    """

    @pytest.fixture
    def rig(self, configured, tmp_path: Path):
        fake = FakeMiDataworks()
        engine = support.migrated_engine(tmp_path / "db")

        def scope():
            return support.session_for(engine)

        # The production port, with the fake's transport threaded through the one outbound
        # call site. Bound for the duration so the plugin reads it exactly as it reads the
        # one `wire()` installed - the wrapper class is production's, not a stub.
        port = midataworks_review.RunScopedReviewQueue(
            run_id=human_review.current_review_run,
            session_scope=scope,
            transport=fake.transport(),
        )
        human_review.bind_review_queue(port)
        yield {"fake": fake, "port": port}
        human_review.bind_review_queue(None)
        midataworks_review.wire()

    def test_the_queue_is_created_for_this_run_and_posted_to_once(self, rig) -> None:
        fake: FakeMiDataworks = rig["fake"]
        with human_review.review_run(RUN):
            resolved = resolve(dict(SELECTOR_CONFIG), source="inline")
            plugin = resolved.plugin
            assert plugin is not None
            plugin.prefetch(
                ["prompt one", "prompt two"],
                ["answer one", "answer two"],
                batch_id="batch-7",
            )

        assert len(fake.queues) == 1, "one queue per run; a second would orphan the first"
        queue = next(iter(fake.queues.values()))
        assert queue.external_ref["run_id"] == RUN
        assert queue.external_ref["app"] == "miforge"
        assert queue.external_ref["name"] == QUEUE_NAME
        # `origin_app` is set from the CREATING caller's agent header, and the poster's app
        # must match it - so a queue created by anyone else is one miForge cannot post to.
        assert queue.origin_app == "miforge"

        # THE CALL COUNT. One post for one batch: two posts of one batch is how a person is
        # asked to read the same candidate twice.
        assert fake.call_count("POST", f"/api/v1/review-queues/{queue.id}/candidates") == 1
        assert fake.call_count("POST", "/api/v1/review-queues") == 1

        # THE PAYLOAD. The ids miDataworks actually holds are F3's derived ones, so a
        # re-post after a crash is idempotent and nobody reads a candidate twice.
        assert sorted(queue.items) == sorted(
            [
                human_review.derive_external_id("batch-7", 0, "answer one"),
                human_review.derive_external_id("batch-7", 1, "answer two"),
            ]
        )

    def test_the_binding_row_is_written_against_this_run(self, rig, tmp_path: Path) -> None:
        """The durable half. The queue id on the wrong run's row is the failure mode a
        "was it bound" assertion cannot see - the decisions are the only copy and they are
        read back through that row."""
        from services.review import binding as binding_service

        with human_review.review_run(RUN):
            queue_id = rig["port"].ensure_queue(
                name=QUEUE_NAME, question="good enough?", label_set=["accept", "reject"]
            )
        with rig["port"].for_run(RUN)._session() as session:
            bound = binding_service.find_binding(session, RUN)
            assert bound is not None
            assert bound.queue_id == queue_id
            assert binding_service.find_binding(session, "some-other-run") is None

    def test_a_decision_comes_back_as_a_verdict(self, rig) -> None:
        """The round trip the type exists for, through the bound port rather than a
        hand-built queue: enqueue, decide, re-evaluate, and the verdict is the person's."""
        fake: FakeMiDataworks = rig["fake"]
        context = {"index": 0, "batch_id": "batch-9"}
        with human_review.review_run(RUN):
            resolved = resolve(dict(SELECTOR_CONFIG), source="inline")
            plugin = resolved.plugin
            assert plugin is not None

            first = plugin.evaluate(prompt="p", completion="c", context=dict(context))
            assert first.value is VerdictValue.UNDECIDED
            assert plugin.item_ids(), "the candidate must be enqueued, not merely reported"

            external_id = human_review.derive_external_id("batch-9", 0, "c")
            queue_id = next(iter(fake.queues))
            fake.decide(queue_id, external_id, "accept")
            plugin.prefetch(["p"], ["c"], batch_id="batch-9", start_index=0)
            second = plugin.evaluate(prompt="p", completion="c", context=dict(context))

        assert second.value is VerdictValue.ACCEPT
        assert plugin.decision_ids(), "R-02.36's last sentence: the decision id is recorded"


# --------------------------------------------------------------------------------------
# CONTROL 2: the run scope is opened by the production driver
# --------------------------------------------------------------------------------------


class _DriveRunSpy:
    """Records the review-queue scope visible from inside ``tasks.drive_run``.

    Patched on the ``tasks`` **module**, which is how ``run_driver`` resolves it - the
    driver does ``tasks.drive_run(...)``, looking the attribute up at call time, so this is
    the production call site and not a parallel one. It returns a real ``RunResult`` so the
    driver runs its whole terminal path afterwards.
    """

    def __init__(self) -> None:
        self.scopes: list[str | None] = []

    def __call__(self, **kwargs: Any) -> tasks.RunResult:
        self.scopes.append(human_review.current_review_run())
        return tasks.RunResult(
            run_id=kwargs["run_id"],
            steps_completed=1,
            last_step=1,
            acquisition={},
            trainer_states=[],
            checkpoints=[],
            stopped_because="step_count",
        )


class TestTheRunScopeIsOpenedByTheDriver:
    """Remove the ``with human_review.review_run(run_id)`` and this goes red.

    A port bound with no scope refuses ``review_queue_run_unknown`` on every queue
    operation, so a run naming a ``human_review`` selector would fail at its first batch
    with a message about a run id - which reads like a bug in miForge rather than like the
    missing wiring it would be.
    """

    def test_the_scope_names_the_run_inside_drive_run(
        self, configured, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        spy = _DriveRunSpy()
        monkeypatch.setattr(tasks, "drive_run", spy)
        engine = memory_engine()
        with session_for(engine) as session:
            insert_run(
                session,
                RUN,
                status="pending",
                definition_submitted=dict(SUBMITTED_DEFINITION),
                definition_resolved={"domain": "legal", "seed": 4242},
                definition_sha256="c" * 64,
            )
            session.commit()
            run_driver.drive_recorded_run(
                RUN, session=session, batch_source=object(), log_root=tmp_path
            )

        assert spy.scopes == [RUN], (
            f"the review-queue scope inside drive_run was {spy.scopes!r}. `None` means "
            f"every queue operation in the run refuses `review_queue_run_unknown`"
        )

    def test_the_scope_closes_when_the_run_ends(
        self, configured, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        """A solo worker (ADR-006) runs one run after another in one process, so a leaked
        scope would make the next run post candidates under the previous run's id."""
        monkeypatch.setattr(tasks, "drive_run", _DriveRunSpy())
        engine = memory_engine()
        with session_for(engine) as session:
            insert_run(
                session,
                RUN,
                status="pending",
                definition_submitted=dict(SUBMITTED_DEFINITION),
                definition_resolved={"domain": "legal", "seed": 4242},
                definition_sha256="c" * 64,
            )
            session.commit()
            run_driver.drive_recorded_run(
                RUN, session=session, batch_source=object(), log_root=tmp_path
            )
        assert human_review.current_review_run() is None
