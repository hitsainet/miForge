"""Pruning end to end: the directory, the index row, and the walk that must survive  (OQ-4).

Everything here goes through the **real** writer and the **real** index builder. ADR-016:
*fixtures whose fields agree by construction are how a suite stays green over a real bug* -
and the bug this file exists to catch is a lineage walk that stops one hop short, which a
hand-built fixture would reproduce faithfully in both directions.

Three things are proved, in ascending order of what they cost if they are wrong:

1. **ADR-004 still holds afterwards.** A prune deletes a directory *and* its index row, and
   ``reindex`` rebuilt from the surviving manifests answers identically. A prune that took
   only the directory would leave the index claiming a checkpoint nobody can load.
2. **The process can die anywhere in a deletion.** The rename happens first, so a killed
   sweep leaves bytes that are not discoverable - never a half-emptied directory that reads
   as a whole checkpoint (acceptance criterion 8, arriving through the delete path).
3. **The lineage clause works against the real walk.** ``lineage.walk`` from a branch run
   reaches its root rather than reporting ``truncated``, after a sweep that both space
   clauses would have taken its branch point in. That is the control; the mutation that
   removes the ancestry check must turn it red.
"""

from __future__ import annotations

import shutil
from pathlib import Path

import pytest

from core.atomic_dir import LATEST, is_temp_dir
from core.errors import Refusal
from models.domain_state import ArtifactKind
from services.checkpoints import index as index_module
from services.checkpoints import lineage, pruning
from services.checkpoints.lifecycle import KEEP_LINEAGE_ANCESTOR, LifecyclePolicy, plan
from services.checkpoints.paths import checkpoint_dir, run_dir
from services.reindex import drop_index, list_checkpoints, reindex
from tests.support import checkpoints as fixtures
from tests.support import jobs
from tests.support.db import migrated_engine, session_for


@pytest.fixture
def engine(tmp_path: Path):
    engine = migrated_engine(tmp_path / "db")
    yield engine
    engine.dispose()


@pytest.fixture
def session(engine):
    with session_for(engine) as session:
        yield session


@pytest.fixture
def root(tmp_path: Path) -> Path:
    path = tmp_path / "checkpoints"
    path.mkdir()
    return path


def write_run(
    root: Path,
    run_id: str,
    steps: tuple[int, ...],
    *,
    parent: str | None = None,
    base_model: str = "LiquidAI/LFM2-1.2B",
) -> list:
    """One run's successive saves, **production-shaped**.

    Every save of a run carries the **same** ``parent``: ``training/runtypes/change.py``
    passes ``parent=ctx.record.parent``, the acquisition record's parent, which is fixed for
    the whole run - the checkpoint it continued or branched from, or ``None`` for a run off
    a base model or a from-scratch creation.

    **This is why the policy frees anything at all, and it is easy to mis-predict from
    ``tests.support.checkpoints.write_chain``**, which chains each save to the previous one.
    That fixture exists to exercise the lineage *walk* and so makes every save but the last
    an ancestor; under this policy such a run is kept entire, correctly. A real root's
    parent set is the set of *branch points*, which is why ~20-30 checkpoints survive a
    hundred saves rather than ~99. One test below pins the chained shape deliberately.
    """
    return [
        fixtures.saved_checkpoint(
            root,
            run_id=run_id,
            step=step,
            parent=parent,
            base_model=base_model,
            payload=f"tensors-{run_id}-{step}".encode(),
        )
        for step in steps
    ]


def index_everything(session, root: Path) -> None:
    """Index every published checkpoint through the production upsert.

    Through :func:`services.checkpoints.index.upsert_from_manifest`, which builds its row
    with ``reindex``'s own builder - so the rows a prune deletes are the rows a rebuild
    would create, rather than two things that happen to agree today.
    """
    from services.reindex import scan_root

    for found in scan_root(root):
        if isinstance(found, Refusal):
            raise found
        if found.kind is ArtifactKind.CHECKPOINT:
            index_module.upsert_from_manifest(session, found.path, root=root)
    session.commit()


def decide(session, root: Path, policy: LifecyclePolicy):
    from workers.checkpoint_lifecycle import active_run_ids

    collected = pruning.collect(root)
    return plan(
        collected.artifacts,
        policy,
        latest_names=collected.latest_names,
        active_run_ids=active_run_ids(session),
    )


class TestADirectoryAndItsRowGoTogether:
    def test_a_prune_removes_both_and_reindex_then_agrees(self, session, root: Path) -> None:
        """**ADR-004's claim, after a deletion.** The index is a rebuildable projection, so
        the test is the structural one: prune, snapshot the responses, drop the index,
        rebuild from the surviving manifests, and assert the responses are *identical*."""
        write_run(root, "run-a", tuple(range(100, 1100, 100)))
        jobs.insert_run(session, "run-a", status="completed", finished_at=fixtures_now())
        index_everything(session, root)
        assert len(list_checkpoints(session)) == 10

        decided = decide(session, root, LifecyclePolicy(keep_last_n=2, keep_every_m=5))
        report = pruning.execute(session, decided, root=root, dry_run=False)

        assert report.failed == [] and report.orphaned == []
        assert len(report.deleted) == len(decided.delete) > 0

        for decision in decided.delete:
            assert not (root / decision.relative_path).exists(), decision.name
        after_prune = list_checkpoints(session)
        assert {row["name"] for row in after_prune} == {d.name for d in decided.keep}

        drop_index(session)
        reindex(session, root=root)
        session.commit()
        assert list_checkpoints(session) == after_prune

    def test_nothing_is_left_behind_on_disk_that_a_reader_can_see(
        self, session, root: Path
    ) -> None:
        """A prune that moved a directory aside and stopped would leave bytes that
        ``reindex`` walks. The only surviving entries under a run are step directories,
        ``latest``, and nothing else."""
        write_run(root, "run-a", tuple(range(100, 900, 100)))
        jobs.insert_run(session, "run-a", status="completed", finished_at=fixtures_now())
        index_everything(session, root)

        decided = decide(session, root, LifecyclePolicy(keep_last_n=1, keep_every_m=4))
        pruning.execute(session, decided, root=root, dry_run=False)

        entries = sorted(p.name for p in run_dir("run-a", root=root).iterdir())
        assert not any(is_temp_dir(name) for name in entries)
        assert entries == sorted([LATEST, *(d.relative_path.split("/")[-1] for d in decided.keep)])


class TestTheLineageClauseAgainstTheRealWalk:
    """**The control.** Mutating the ancestry check out of ``lifecycle.plan`` must turn
    every test in this class red."""

    def test_a_branch_point_survives_and_the_branchs_walk_still_reaches_its_root(
        self, session, root: Path
    ) -> None:
        """``run-b`` branched from step 200 of ``run-a`` - a step that is neither among the
        last N nor an Mth save, so **both space clauses would delete it**.

        The assertion is on the real walk, not on the plan: R-02.29 walks by ``parent`` and
        R-02.23 is what makes a branch possible, so the thing that must still be true after
        a prune is that the chain reaches ``root``. Pruned, it would report ``truncated``
        at the branch point - honestly, which is the whole problem: the record would say
        only that the history stops there.
        """
        chain = write_run(root, "run-a", (100, 200, 300, 400, 500))
        branch_point = chain[1]
        fixtures.saved_checkpoint(
            root, run_id="run-b", step=600, parent=branch_point.name, payload=b"branched"
        )
        jobs.insert_run(session, "run-a", status="completed", finished_at=fixtures_now())
        jobs.insert_run(session, "run-b", status="completed", finished_at=fixtures_now())
        index_everything(session, root)

        before = lineage.walk("run-b", 600, root=root)
        assert before.terminated == "root"

        decided = decide(session, root, LifecyclePolicy(keep_last_n=1, keep_every_m=1000))
        kept = {d.name: d for d in decided.keep}
        assert branch_point.name in kept
        assert kept[branch_point.name].reasons == (KEEP_LINEAGE_ANCESTOR,), (
            "if N or M were also keeping it, this test would pass with the ancestry "
            "check deleted and would be proving nothing"
        )

        pruning.execute(session, decided, root=root, dry_run=False)

        after = lineage.walk("run-b", 600, root=root)
        assert (
            after.terminated == "root"
        ), "the branch's walk was truncated by the prune: the lineage clause did not hold"
        assert [n.name for n in after.nodes] == [n.name for n in before.nodes]

    def test_the_branchs_walk_reaches_its_root_after_a_prune_and_nothing_else_is_asserted(
        self, session, root: Path
    ) -> None:
        """The same scenario asserting **only the outcome**, with no claim about the plan.

        The test above checks the decision first, so under a mutation it fails on the plan
        and never reaches the walk - which proves the clause is gone but not what it costs.
        This one prunes and asks ``lineage.walk`` one question, so the red says
        ``truncated`` and names the hop that disappeared. *Pin the property, not the
        mechanism: a future policy that protected branch points some other way must keep
        this green.*
        """
        chain = write_run(root, "run-a", (100, 200, 300, 400, 500))
        fixtures.saved_checkpoint(
            root, run_id="run-b", step=600, parent=chain[1].name, payload=b"branched"
        )
        jobs.insert_run(session, "run-a", status="completed", finished_at=fixtures_now())
        jobs.insert_run(session, "run-b", status="completed", finished_at=fixtures_now())
        index_everything(session, root)

        decided = decide(session, root, LifecyclePolicy(keep_last_n=1, keep_every_m=1000))
        pruning.execute(session, decided, root=root, dry_run=False)

        walk = lineage.walk("run-b", 600, root=root)
        assert (walk.terminated, walk.terminated_at) == (
            "root",
            None,
        ), f"the prune truncated the branch's lineage at {walk.terminated_at!r}"
        # Two hops, not three: in production shape every save of ``run-a`` carries the
        # SAME parent (``None`` here, off a base model), so step 200 *is* a root. Written
        # as [600, 200, 100] first - the chained shape - which is the mis-prediction
        # ``write_run``'s docstring exists to stop.
        assert [n.step for n in walk.nodes] == [600, 200]

    def test_a_linearly_chained_run_is_kept_entire_and_that_is_correct(
        self, session, root: Path
    ) -> None:
        """``write_chain`` chains each save to the previous one, so **every save but the
        last is a parent** and the policy keeps all of them.

        Pinned rather than avoided, because it is the shape a reader predicts production to
        have and it is not: ``training/runtypes/change.py`` gives every save of a run the
        **same** ``parent``. If a future change made a save's parent the previous save, this
        test stays green while the volume stops being freed - which is the honest failure,
        and a note in the report rather than a silent one.
        """
        fixtures.write_chain(root, run_id="run-chained", steps=tuple(range(100, 900, 100)))
        jobs.insert_run(session, "run-chained", status="completed", finished_at=fixtures_now())
        index_everything(session, root)

        decided = decide(session, root, LifecyclePolicy(keep_last_n=1, keep_every_m=1000))
        assert decided.delete == ()
        assert sum(1 for d in decided.keep if KEEP_LINEAGE_ANCESTOR in d.reasons) == 7

    def test_an_adapters_base_survives_so_the_adapter_can_still_be_rebound(
        self, session, root: Path
    ) -> None:
        """Edge 2, through the **real** adapter writer. An adapter names its base, so an
        adapter creates ancestry - and BRD-01 R-01.27 has a consumer verify that base by
        the hash in the adapter's own manifest, which is unresolvable once the base is
        gone."""
        from services.checkpoints import manifest as manifest_module
        from tests.support.test_adapter_doubles import (
            FakePeftModel,
            adapter_config,
            checkpoint_record,
            optimizer_block,
        )
        from training.runtypes.adapters import manifest as adapter_manifest
        from training.runtypes.adapters import paths as adapter_paths
        from training.runtypes.adapters import save as adapter_save

        chain = write_run(root, "run-a", (100, 200, 300, 400, 500))
        base = chain[1]
        # The base's hash as the REAL reader computes it. `base_block` refuses a null here
        # when the base is a checkpoint, because null asserts "the base was never a saved
        # checkpoint" - a false assertion that would make the adapter unrebindable.
        base_doc = manifest_module.read(checkpoint_dir("run-a", 200, root=root))
        base_hash = manifest_module.digest(base_doc)

        doc = adapter_manifest.build(
            run_id="run-b",
            step=40,
            base=adapter_manifest.base_block(
                checkpoint_record(base.name), manifest_sha256=base_hash
            ),
            adapter=adapter_config(),
            optimizer=optimizer_block(),
        )
        adapter_save.save(FakePeftModel(), doc, adapter_paths.adapter_dir("run-b", 40, root=root))
        jobs.insert_run(session, "run-a", status="completed", finished_at=fixtures_now())
        jobs.insert_run(session, "run-b", status="completed", finished_at=fixtures_now())
        index_everything(session, root)

        decided = decide(session, root, LifecyclePolicy(keep_last_n=1, keep_every_m=1000))
        kept = {d.name: d for d in decided.keep}
        assert kept[base.name].reasons == (KEEP_LINEAGE_ANCESTOR,)

        pruning.execute(session, decided, root=root, dry_run=False)
        assert checkpoint_dir("run-a", 200, root=root).is_dir()
        # And the adapter itself is untouched: adapters are never pruned.
        assert (adapter_paths.adapter_dir("run-b", 40, root=root)).is_dir()


class TestTheThingsThatAreNeverTouched:
    def test_latest_still_resolves_after_a_prune(self, session, root: Path) -> None:
        from services.checkpoints.paths import resolve_latest

        write_run(root, "run-a", tuple(range(100, 900, 100)))
        jobs.insert_run(session, "run-a", status="completed", finished_at=fixtures_now())
        index_everything(session, root)

        decided = decide(session, root, LifecyclePolicy(keep_last_n=1, keep_every_m=1000))
        pruning.execute(session, decided, root=root, dry_run=False)

        assert resolve_latest("run-a", root=root) == 800
        assert (run_dir("run-a", root=root) / LATEST).resolve().is_dir()

    def test_a_running_run_loses_nothing(self, session, root: Path) -> None:
        """*Never delete while a run is actively writing.* The row is ``running``, so the
        whole run is immune - the save publishes with one ``rename`` and a finer check
        would still race it."""
        write_run(root, "run-a", tuple(range(100, 900, 100)))
        jobs.insert_run(session, "run-a", status="running", heartbeat_at=fixtures_now())
        index_everything(session, root)

        decided = decide(session, root, LifecyclePolicy(keep_last_n=1, keep_every_m=1000))
        assert decided.delete == ()
        report = pruning.execute(session, decided, root=root, dry_run=False)
        assert report.deleted == []
        assert len(list_checkpoints(session)) == 8


class TestTheGates:
    def test_a_dry_run_deletes_nothing_and_says_so(self, session, root: Path) -> None:
        write_run(root, "run-a", tuple(range(100, 900, 100)))
        jobs.insert_run(session, "run-a", status="completed", finished_at=fixtures_now())
        index_everything(session, root)

        decided = decide(session, root, LifecyclePolicy(keep_last_n=1, keep_every_m=1000))
        assert len(decided.delete) > 0
        report = pruning.execute(session, decided, root=root)  # the DEFAULT

        assert report.dry_run is True and report.deleted == []
        assert len(list_checkpoints(session)) == 8
        for decision in decided.delete:
            assert (root / decision.relative_path).is_dir()

    def test_dry_run_is_the_default_of_the_parameter_itself(self) -> None:
        """Not merely of the setting. ``execute(session, plan, root=...)`` with no third
        thought must not delete - because ``False`` means delete, and a caller added later
        will take the default."""
        import inspect

        assert inspect.signature(pruning.execute).parameters["dry_run"].default is True

    def test_the_sweep_plans_and_reports_while_disabled(
        self, session, root: Path, monkeypatch
    ) -> None:
        """**Disabled still computes the plan**, because *"what would this delete"* is the
        question an operator needs answered before turning it on. A sweep that declined
        without saying what it would have done would be a feature nobody can evaluate."""
        import core.config as config_module
        from workers.checkpoint_lifecycle import sweep

        write_run(root, "run-a", tuple(range(100, 900, 100)))
        jobs.insert_run(session, "run-a", status="completed", finished_at=fixtures_now())
        index_everything(session, root)
        monkeypatch.setattr(
            config_module,
            "_settings",
            config_module.Settings(
                checkpoint_lifecycle_enabled=False,
                checkpoint_lifecycle_keep_last_n=1,
                checkpoint_lifecycle_keep_every_m=1000,
            ),
        )

        report = sweep(session, root=root)
        assert report.outcome == "disabled"
        assert report.to_delete == 6 and report.deleted == []
        assert len(list_checkpoints(session)) == 8

    def test_enabled_and_not_dry_run_is_the_only_combination_that_deletes(
        self, session, root: Path, monkeypatch
    ) -> None:
        import core.config as config_module
        from workers.checkpoint_lifecycle import sweep

        write_run(root, "run-a", tuple(range(100, 900, 100)))
        jobs.insert_run(session, "run-a", status="completed", finished_at=fixtures_now())
        index_everything(session, root)

        for enabled, dry_run, expected in (
            (False, True, "disabled"),
            (False, False, "disabled"),
            (True, True, "dry_run"),
        ):
            monkeypatch.setattr(
                config_module,
                "_settings",
                config_module.Settings(
                    checkpoint_lifecycle_enabled=enabled,
                    checkpoint_lifecycle_dry_run=dry_run,
                    checkpoint_lifecycle_keep_last_n=1,
                    checkpoint_lifecycle_keep_every_m=1000,
                ),
            )
            report = sweep(session, root=root)
            assert report.outcome == expected and report.deleted == []
            assert len(list_checkpoints(session)) == 8

        monkeypatch.setattr(
            config_module,
            "_settings",
            config_module.Settings(
                checkpoint_lifecycle_enabled=True,
                checkpoint_lifecycle_dry_run=False,
                checkpoint_lifecycle_keep_last_n=1,
                checkpoint_lifecycle_keep_every_m=1000,
            ),
        )
        report = sweep(session, root=root)
        assert report.outcome == "executed" and len(report.deleted) == 6
        assert len(list_checkpoints(session)) == 2


class TestDyingInTheMiddleOfADeletion:
    def test_the_staged_name_is_one_every_discovery_path_already_hides(self) -> None:
        """**The invariant the whole crash story rests on.** A fresh ``.prune-`` prefix
        would not be in ``is_temp_dir``, so a staged directory would be walked by
        ``reindex`` and re-indexed as a live checkpoint - half-deleted, on a killed
        sweep."""
        assert is_temp_dir(pruning.staging_name())
        assert pruning.PRUNE_PREFIX.startswith(".tmp-save-")
        assert (
            "prune" in pruning.PRUNE_PREFIX
        ), "the shared prefix must still say which of the two killed writers left it"

    def test_a_failed_rmtree_leaves_bytes_that_nothing_can_find(
        self, session, root: Path, monkeypatch
    ) -> None:
        """The rename happened, the row is gone, and the bytes are still there. Reported,
        not raised: nothing can see them, so the next sweep's ``rmtree`` or an operator's
        ``rm -rf`` finishes the job."""
        write_run(root, "run-a", (100, 200, 300, 400))
        jobs.insert_run(session, "run-a", status="completed", finished_at=fixtures_now())
        index_everything(session, root)
        decided = decide(session, root, LifecyclePolicy(keep_last_n=1, keep_every_m=1000))
        assert len(decided.delete) == 2

        def refuse_to_remove(path, *args, **kwargs):
            raise OSError(16, "Device or resource busy")

        monkeypatch.setattr(shutil, "rmtree", refuse_to_remove)
        report = pruning.execute(session, decided, root=root, dry_run=False)

        assert len(report.orphaned) == 2
        assert {name for name, _ in report.orphaned} == {d.name for d in decided.delete}
        # De-indexed, and invisible to a rebuild: the bytes are under a hidden name.
        assert {row["name"] for row in list_checkpoints(session)} == {d.name for d in decided.keep}
        drop_index(session)
        reindex(session, root=root)
        session.commit()
        assert {row["name"] for row in list_checkpoints(session)} == {d.name for d in decided.keep}
        staged = [p.name for p in run_dir("run-a", root=root).iterdir() if is_temp_dir(p.name)]
        assert len(staged) == 2

    def test_a_directory_already_gone_is_repaired_rather_than_reported_as_failure(
        self, session, root: Path
    ) -> None:
        """The state a sweep killed between the rename and the commit leaves behind: the
        row is the only thing left, and removing it **is** the repair."""
        write_run(root, "run-a", (100, 200, 300, 400))
        jobs.insert_run(session, "run-a", status="completed", finished_at=fixtures_now())
        index_everything(session, root)
        decided = decide(session, root, LifecyclePolicy(keep_last_n=1, keep_every_m=1000))

        gone = decided.delete[0]
        shutil.rmtree(root / gone.relative_path)
        report = pruning.execute(session, decided, root=root, dry_run=False)

        assert report.failed == []
        assert gone.name in report.deleted
        assert gone.name not in {row["name"] for row in list_checkpoints(session)}

    def test_one_undeletable_directory_does_not_stop_the_others(
        self, session, root: Path, monkeypatch
    ) -> None:
        """The one-subsystem-kills-everything shape ``reindex`` was fixed for. A permission
        fault on one directory must not abandon the other twenty-nine."""
        import os as os_module

        write_run(root, "run-a", (100, 200, 300, 400, 500, 600))
        jobs.insert_run(session, "run-a", status="completed", finished_at=fixtures_now())
        index_everything(session, root)
        decided = decide(session, root, LifecyclePolicy(keep_last_n=1, keep_every_m=1000))
        assert len(decided.delete) == 4
        doomed = decided.delete[1]

        real_rename = os_module.rename

        def rename_except_one(src, dst):
            if str(src).endswith(doomed.relative_path.split("/")[-1]):
                raise PermissionError(13, "Permission denied")
            return real_rename(src, dst)

        monkeypatch.setattr(os_module, "rename", rename_except_one)
        report = pruning.execute(session, decided, root=root, dry_run=False)

        assert [name for name, _ in report.failed] == [doomed.name]
        assert len(report.deleted) == 3
        assert doomed.name in {row["name"] for row in list_checkpoints(session)}, (
            "a directory that could not be staged must keep its index row: the checkpoint "
            "is still there and still loadable"
        )


class TestAnUnreadableManifestStopsEverything:
    def test_the_whole_sweep_refuses_and_deletes_nothing(self, session, root: Path) -> None:
        """**Fail closed** (ADR-013). The ancestor set is built from manifest fields, so a
        manifest nobody can read is three edges nobody can see - and one of them may be the
        only thing naming a checkpoint about to go. The prune would detach a subtree while
        every check passed."""
        write_run(root, "run-a", tuple(range(100, 900, 100)))
        (checkpoint_dir("run-a", 300, root=root) / "manifest.json").write_text("{not json")

        with pytest.raises(Refusal) as caught:
            pruning.collect(root)
        assert caught.value.code == "checkpoint_lifecycle_unreadable_manifest"

    def test_the_sweep_reports_the_refusal_instead_of_raising_on_a_schedule(
        self, session, root: Path, monkeypatch
    ) -> None:
        """A scheduled task that raises becomes a traceback every hour and the next sweep
        has the same problem. The report names the code so the log says what to fix."""
        import core.config as config_module
        from workers.checkpoint_lifecycle import sweep

        write_run(root, "run-a", tuple(range(100, 900, 100)))
        jobs.insert_run(session, "run-a", status="completed", finished_at=fixtures_now())
        index_everything(session, root)
        (checkpoint_dir("run-a", 300, root=root) / "manifest.json").write_text("{not json")
        monkeypatch.setattr(
            config_module,
            "_settings",
            config_module.Settings(
                checkpoint_lifecycle_enabled=True,
                checkpoint_lifecycle_dry_run=False,
                checkpoint_lifecycle_keep_last_n=1,
                checkpoint_lifecycle_keep_every_m=1000,
            ),
        )

        report = sweep(session, root=root)
        assert report.outcome == "refused"
        assert report.refusal_code == "checkpoint_lifecycle_unreadable_manifest"
        assert report.deleted == []
        assert len(list_checkpoints(session)) == 8


class TestWhatTheSweepLogs:
    def test_it_logs_every_checkpoint_kept_because_something_descends_from_it(
        self, session, root: Path, monkeypatch, caplog
    ) -> None:
        """*Log what was kept and why, not only what was deleted.* The question asked
        afterwards is always *"why is step 200 of a finished run still on my volume"*, and
        the answer has to be in the log at the moment the decision was made."""
        import logging

        import core.config as config_module
        from workers.checkpoint_lifecycle import sweep

        chain = write_run(root, "run-a", (100, 200, 300, 400, 500))
        fixtures.saved_checkpoint(root, run_id="run-b", step=600, parent=chain[1].name)
        jobs.insert_run(session, "run-a", status="completed", finished_at=fixtures_now())
        jobs.insert_run(session, "run-b", status="completed", finished_at=fixtures_now())
        index_everything(session, root)
        monkeypatch.setattr(
            config_module,
            "_settings",
            config_module.Settings(
                checkpoint_lifecycle_keep_last_n=1, checkpoint_lifecycle_keep_every_m=1000
            ),
        )

        with caplog.at_level(logging.INFO, logger="miforge.checkpoints.lifecycle"):
            report = sweep(session, root=root)

        assert report.protected_ancestors == [chain[1].name]
        kept_lines = [
            r
            for r in caplog.records
            if r.__dict__.get("code") == "checkpoint_kept_lineage_ancestor"
        ]
        assert [r.__dict__["checkpoint"] for r in kept_lines] == [chain[1].name]
        plan_lines = [
            r for r in caplog.records if r.__dict__.get("code") == "checkpoint_lifecycle_plan"
        ]
        assert len(plan_lines) == 1
        assert plan_lines[0].__dict__["kept_by_reason"][KEEP_LINEAGE_ANCESTOR] == 1
        assert plan_lines[0].__dict__["policy"] == {"keep_last_n": 1, "keep_every_m": 1000}


def fixtures_now():
    from workers.run_record import utcnow

    return utcnow()
