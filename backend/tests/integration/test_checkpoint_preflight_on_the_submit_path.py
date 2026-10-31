"""F5's resume cases, on the path that actually starts a run  (**task 7.8**; R-02.21, R-02.22).

**The gap.** F5's cases all exist and all run - as
:func:`services.checkpoints.resume.resume_plan`, which ``POST /runs/{run_id}/continue``
calls, so a *planned* continue refuses correctly. But that route only **returns a plan**.
The path that creates an ``mf_runs`` row and enqueues ``miforge.train.run_step_loop`` is
:func:`services.run_launch.launch` -> :mod:`services.run_definition` step 5 ->
:func:`services.acquisition.preflight`, and **nothing on it verified a checkpoint**. The
worker then re-ran only the *acquirer's* preflight, which checks that the weight files are
present and the dtype is loadable, and **never** that the bytes hash to what the manifest
records.

So a run defined with ``model_source = {"mode": "load", "path": <a corrupted checkpoint>}``
was accepted, leased a card, and trained on corrupted weights with no refusal anywhere -
the silently-wrong answer ADR-013 forbids, on the one path R-02.21 was amended to protect.

**The worker's half of that was still open when this file was first written, and is now
closed** (2026-10-16). Submit verifying is not sufficient on its own: its verdict is minutes
old by the time the worker allocates, and a continue is exactly the case where the bytes were
written hours earlier - a half-finished rsync, a bad disk, a restore from the wrong backup
all land in that window. :func:`training.tasks.drive_run` therefore calls
:func:`training.tasks.verify_checkpoint_source` **before** :func:`training.tasks.
acquire_for_run`, which is where the tensors are read.
:class:`TestTheWorkerRunsTheSameCheck` is the reachability half - the call, the payload and
the call count - and ``test_continuity_across_a_resume.py``'s
``TestTheWorkerVerifiesTheCheckpointBeforeReadingATensor`` is the refusal half, which also
asserts the acquirer was never reached. **One implementation serves both paths**: both call
:func:`services.acquisition.preflight_checkpoint_source`, and the test below drives each in
turn through one spy so a second spelling of the check would show up as a missing call rather
than as silence.

**Every assertion below checks the refusal CODE**, not that something raised: a bare
``pytest.raises`` passes against a ``KeyError`` from a typo in the checker, which reads as
protection and protects nothing.

## What is deliberately *not* refused, and why that is half the work

A ``load`` may name a plain Hugging Face directory - R-02.46 case 1 covers both - and such a
directory has no ``manifest.json`` and therefore no recorded hash to verify against.
Refusing it ``checkpoint_not_found`` would break the commonest load there is. So
:func:`services.acquisition.is_miforge_checkpoint` discriminates on the manifest's presence,
and :class:`TestWhatIsLeftAlone` pins both halves - because a gate that refuses too much is
found by its users and a gate that refuses too little is not found at all.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from core.canonical import sha256_of_file, write_canonical
from core.config import Settings, reset_settings
from core.errors import Refusal
from schemas.trainer_state import REQUIRED_FILES
from services import acquisition as acquisition_service
from services.checkpoints import manifest as manifest_module
from services.checkpoints.paths import checkpoint_dir
from services.checkpoints.writer import WEIGHTS_NAME

# Imported rather than re-spelled. `write_safetensors` is the safetensors writer this suite
# already round-trips through the production reader; a second copy of the `<u64 len><JSON
# header><payload>` layout here would be a second definition of the format, and the two
# would disagree the first time either moved.
from tests.unit.test_acquisition_dtype_gate import write_safetensors
from training.acquisition.base import AcquisitionMode, AcquisitionSpec
from training.run_config import RunConfig

CFG = RunConfig(
    learning_rate=1e-4,
    warmup_steps=0,
    batch_size=1,
    # 32-bit, so `cfg.config_hash()` and anything downstream never needs `bitsandbytes`.
    optimizer_state_precision="32bit",
)


@pytest.fixture
def rooted(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """Point every configured root at ``tmp_path``. Yields ``(checkpoints, trainer_state)``.

    The roots are **created**, which matters for more than tidiness:
    :func:`services.acquisition.preflight_artifact_roots` skips an absent root and says so,
    so a fixture that left them absent would make the atomicity assertions below pass
    without the probe ever running.
    """
    checkpoints = tmp_path / "checkpoints"
    parts = tmp_path / "parts"
    states = tmp_path / "trainer-state"
    for directory in (checkpoints, parts, states):
        directory.mkdir()
    monkeypatch.setattr(
        "core.config._settings",
        Settings(
            _env_file=None,  # type: ignore[call-arg]
            checkpoint_root=checkpoints,
            parts_root=parts,
            trainer_state_root=states,
        ),
    )
    acquisition_service.reset_artifact_root_atomicity_cache()
    yield checkpoints, states
    reset_settings()
    # The atomicity verdict is memoised per process (see `preflight_artifact_roots`), and
    # every test here repoints the roots at a fresh `tmp_path`. Leaving the cache populated
    # would let one test's verdict answer the next test's question - which is the
    # construction-green failure, arriving through a fixture.
    acquisition_service.reset_artifact_root_atomicity_cache()


def loadable_checkpoint(
    root: Path,
    *,
    run_id: str = "run-a",
    step: int = 500,
    corrupt_weights: bool = False,
    manifest_text: str | None = None,
) -> Path:
    """A directory that is **both** a standard HF model and a miForge checkpoint.

    Both halves are needed because this test drives the *whole* of
    :func:`services.acquisition.preflight`: the acquirer's own preflight runs first and
    reads a real ``safetensors`` header for the dtype gate, so a planted byte string would
    refuse before F5's cases were ever reached - and the test would then pass for the wrong
    reason, which is this project's signature false measurement.

    The manifest's ``weights_sha256`` is computed from the bytes on disk by the **production
    hasher**, so a passing verification is a real agreement rather than two values a fixture
    made equal.
    """
    directory = checkpoint_dir(run_id, step, root=root)
    directory.mkdir(parents=True)
    (directory / "config.json").write_text(json.dumps({"model_type": "llama"}))
    # Tokenizer files AT THE ROOT (R-02.51, acceptance criterion 23).
    (directory / "tokenizer.json").write_text("{}")
    weights = write_safetensors(
        directory / WEIGHTS_NAME,
        {"embed.weight": ("F16", [32, 8]), "layer.weight": ("F16", [8, 8])},
    )

    if manifest_text is not None:
        (directory / manifest_module.MANIFEST_NAME).write_text(manifest_text)
        return directory

    doc = manifest_module.build(
        run_id=run_id,
        step=step,
        domain=None,
        signal_source="corpus",
        base_model="LiquidAI/LFM2-1.2B",
        parent=None,
        optimizer={"name": "adamw-32bit", "state": "fresh", "config_hash": None},
        weights_sha256=sha256_of_file(weights),
    )
    write_canonical(directory / manifest_module.MANIFEST_NAME, doc)

    if corrupt_weights:
        # AFTER the manifest, so the recorded hash is the hash of the good bytes and the
        # bytes on disk are not. That is the real failure: a bit-rotted shard, a truncated
        # copy, a restore from a torn backup.
        weights.write_bytes(weights.read_bytes() + b"rot")
    return directory


def plant_trainer_state(root: Path, run_id: str, step: int, *, complete: bool) -> Path:
    """Trainer state built from F1's own ``REQUIRED_FILES``, not a second spelling of them."""
    path = root / run_id / f"step-{step:06d}"
    path.mkdir(parents=True, exist_ok=True)
    for name in REQUIRED_FILES if complete else REQUIRED_FILES[:-1]:
        (path / name).write_bytes(b"state")
    return path


def load_spec(directory: Path) -> AcquisitionSpec:
    return AcquisitionSpec(mode=AcquisitionMode.LOAD, path=directory, gpu_request="auto")


def run_preflight(directory: Path, *, run_id: str | None = None) -> Any:
    """The whole of F1's submit-path preflight, as ``run_definition`` step 5 calls it."""
    return acquisition_service.preflight(spec=load_spec(directory), cfg=CFG, run_id=run_id)


class TestTheSubmitPathRefusesACorruptedCheckpoint:
    """R-02.21: verified before a single tensor is used, refusing having changed nothing."""

    def test_a_hash_mismatch_refuses_by_name(self, rooted) -> None:
        checkpoints, _states = rooted
        directory = loadable_checkpoint(checkpoints, corrupt_weights=True)

        with pytest.raises(Refusal) as caught:
            run_preflight(directory, run_id="run-a")

        assert caught.value.code == "checkpoint_hash_mismatch"
        # ADR-013: the message says what to do next, and names BOTH digests so the operator
        # can tell a rotted shard from a rewritten manifest.
        assert "restore" in caught.value.message.lower()
        assert caught.value.details["expected"] != caught.value.details["actual"]

    def test_it_refuses_with_no_run_id_too(self, rooted) -> None:
        """The define path does not know the run id yet - the verification still runs.

        :func:`services.run_launch.launch` mints the id *after* validation, so if the hash
        check depended on a run id it would be absent on the one path that enqueues the
        task. Only the **trainer-state** case needs the id, and its absence is reported.
        """
        checkpoints, _states = rooted
        directory = loadable_checkpoint(checkpoints, corrupt_weights=True)

        with pytest.raises(Refusal) as caught:
            run_preflight(directory, run_id=None)
        assert caught.value.code == "checkpoint_hash_mismatch"

    def test_an_unparseable_manifest_refuses_by_name(self, rooted) -> None:
        checkpoints, _states = rooted
        directory = loadable_checkpoint(checkpoints, manifest_text="{not json")

        with pytest.raises(Refusal) as caught:
            run_preflight(directory, run_id="run-a")
        assert caught.value.code == "checkpoint_manifest_invalid"

    def test_a_manifest_missing_a_required_field_refuses_by_name(self, rooted) -> None:
        """A manifest off disk is untrusted input until validated, like a request body."""
        checkpoints, _states = rooted
        directory = loadable_checkpoint(checkpoints)
        doc = json.loads((directory / manifest_module.MANIFEST_NAME).read_text())
        del doc["weights_sha256"]
        (directory / manifest_module.MANIFEST_NAME).write_text(json.dumps(doc))

        with pytest.raises(Refusal) as caught:
            run_preflight(directory, run_id="run-a")
        assert caught.value.code == "checkpoint_manifest_invalid"

    def test_a_torn_trainer_state_for_the_continued_step_refuses_by_name(self, rooted) -> None:
        """The one case that is not a judgement call.

        Continuing from a partial optimizer state does not crash - it *runs*, and produces a
        loss discontinuity that looks exactly like a real regression, so the time is lost
        twice.
        """
        checkpoints, states = rooted
        directory = loadable_checkpoint(checkpoints, run_id="run-a", step=500)
        plant_trainer_state(states, "run-a", 500, complete=False)

        with pytest.raises(Refusal) as caught:
            run_preflight(directory, run_id="run-a")
        assert caught.value.code == "trainer_state_incomplete"

    def test_a_branch_is_NOT_refused_by_the_source_runs_torn_state(self, rooted) -> None:
        """And this is why the run id has to be passed rather than inferred.

        Trainer state is addressed by ``run_id`` (R-02.15), so a branch is ``fresh`` **by
        construction** (FR-005.38) and will never read the source run's directory. Refusing
        it over a state it is not going to touch would block legitimate work - the
        false-refusal direction, which is the expensive one for a gate like this.
        """
        checkpoints, states = rooted
        directory = loadable_checkpoint(checkpoints, run_id="run-a", step=500)
        plant_trainer_state(states, "run-a", 500, complete=False)

        enriched, _availability = run_preflight(directory, run_id="a-new-run")
        assert enriched.mode is AcquisitionMode.LOAD

    def test_every_refusal_this_fold_adds_is_named_in_the_constant(self) -> None:
        """So a refusal added to F5's planner without being considered here is visible.

        The set is asserted against :func:`services.checkpoints.resume.resume_plan`'s own
        documented ``Raises:`` list, read out of the docstring rather than restated - every
        restated name in this project has eventually drifted.
        """
        from services.checkpoints import resume as resume_module

        documented = resume_module.resume_plan.__doc__ or ""
        for code in acquisition_service.CHECKPOINT_PREFLIGHT_REFUSALS:
            assert code in documented, (
                f"{code!r} is folded in here but is not in resume_plan's documented "
                f"refusals, so one of the two has drifted"
            )


class TestWhatIsLeftAlone:
    """A gate that refuses too much is the other way to get this wrong."""

    def test_a_plain_hugging_face_directory_is_not_treated_as_a_checkpoint(self, rooted) -> None:
        """No ``manifest.json``, so no recorded hash exists to verify against.

        This is the commonest ``load`` there is - R-02.46 case 1 - and refusing it
        ``checkpoint_not_found`` would make the gate unusable.
        """
        checkpoints, _states = rooted
        directory = checkpoints / "a-downloaded-model"
        directory.mkdir()
        (directory / "config.json").write_text(json.dumps({"model_type": "llama"}))
        (directory / "tokenizer.json").write_text("{}")
        write_safetensors(directory / WEIGHTS_NAME, {"embed.weight": ("F16", [32, 8])})

        assert acquisition_service.is_miforge_checkpoint(directory) is False
        enriched, _availability = run_preflight(directory, run_id="run-z")
        assert enriched.mode is AcquisitionMode.LOAD

    def test_a_create_run_is_untouched(self, rooted) -> None:
        """``create`` has no source to verify, and the fold must not invent one."""
        del rooted
        spec = AcquisitionSpec(
            mode=AcquisitionMode.CREATE,
            # `config`, inline. `architecture_config` is the read-only property over it.
            config={
                "model_type": "llama",
                "hidden_size": 16,
                "num_hidden_layers": 1,
                "num_attention_heads": 1,
                "vocab_size": 64,
            },
            gpu_request="auto",
        )
        assert acquisition_service.preflight_checkpoint_source(spec) is None

    def test_a_good_checkpoint_passes_and_reports_what_it_checked(self, rooted) -> None:
        """The happy path, and it must *say* what it verified.

        A checker that returns nothing on success cannot be distinguished from one that did
        not run - which is the whole false-measurement family this project tracks.
        """
        checkpoints, states = rooted
        directory = loadable_checkpoint(checkpoints, run_id="run-a", step=500)
        plant_trainer_state(states, "run-a", 500, complete=True)

        report = acquisition_service.preflight_checkpoint_source(
            load_spec(directory), run_id="run-a"
        )
        assert report is not None
        assert report["source_run_id"] == "run-a"
        assert report["step"] == 500
        assert report["addressed_by"] == "run_and_step"
        assert report["trainer_state_checked"] is True
        assert report["optimizer_state"] == "restored"
        assert len(report["verified_weights_sha256"]) == 64

    def test_without_a_run_id_the_trainer_state_case_is_reported_unchecked(self, rooted) -> None:
        """Reported, never silently passed.

        The same directory is a continue for one caller and a branch point for another, and
        the two want opposite verdicts on a torn state - so without a run id there is no
        answer, and saying so is the honest form. A skip that names its reason is what
        ``test_schema_sync.py`` does by classing a contract ``Internal`` **by name** rather
        than giving it an empty vendor list, which would make its assertion pass vacuously.
        """
        checkpoints, _states = rooted
        directory = loadable_checkpoint(checkpoints)

        report = acquisition_service.preflight_checkpoint_source(load_spec(directory))
        assert report is not None
        assert report["trainer_state_checked"] is False
        assert "no run_id was supplied" in report["trainer_state_not_checked_because"]
        # The verification still happened: that is the case that must never be conditional.
        assert len(report["verified_weights_sha256"]) == 64


class TestTheFoldIsReachableFromTheSubmitPath:
    """ADR-016 #1: *a capability is not shipped until a test FAILS when its wiring is removed.*

    The refusal tests above already go through the real :func:`services.acquisition.preflight`
    rather than calling the checker directly, so deleting the call site turns five of them
    red. These two add what a refusal test cannot show: the **payload** and the **call
    count** - *"was called" passes against a call sending the wrong arguments*.
    """

    def test_preflight_calls_the_checker_once_with_the_spec_it_enriched(
        self, rooted, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        checkpoints, _states = rooted
        directory = loadable_checkpoint(checkpoints, run_id="run-a", step=500)
        calls: list[dict[str, Any]] = []

        def _record(spec: Any, **kwargs: Any) -> None:
            calls.append({"spec": spec, **kwargs})

        monkeypatch.setattr(acquisition_service, "preflight_checkpoint_source", _record)
        enriched, _availability = run_preflight(directory, run_id="run-a")

        assert len(calls) == 1, f"the checker ran {len(calls)} times, not once"
        # THE ENRICHED spec, not the submitted one: the acquirer's preflight is what
        # confines the path to a configured root, and checking the unconfined path would
        # hash a file outside the roots the operator configured.
        assert calls[0]["spec"] is enriched
        assert calls[0]["run_id"] == "run-a"
        # The configured optimizer hash rides along, so a changed configuration is logged
        # with BOTH values rather than silently carried (FR-005.34).
        assert calls[0]["configured_config_hash"] == CFG.config_hash()

    def test_preflight_probes_the_artifact_roots_once_and_before_the_budget_guard(
        self, rooted, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Written because its absence was a hole in *this* file, found by mutating it.

        Every atomicity assertion in :class:`TestTheArtifactRootsAtomicityCase` calls
        :func:`services.acquisition.preflight_artifact_roots` **directly**, so deleting the
        call from :func:`services.acquisition.preflight` turned nothing red - the capability
        was unit-tested and unreachable, which is the exact defect ADR-016 #1 exists for and
        which this project has shipped three times. Caught by running the mutation rather
        than by reading the file.

        The **order** is asserted as well as the call: the probe is cheap and applies to
        every mode, while the budget guard's answer is useless for a run that cannot publish
        a checkpoint at all.
        """
        checkpoints, _states = rooted
        directory = loadable_checkpoint(checkpoints, run_id="run-a", step=500)
        order: list[str] = []

        def _record_probe(**kwargs: Any) -> dict[str, str]:
            order.append("roots")
            assert kwargs == {}, f"preflight passed the probe arguments it does not own: {kwargs}"
            return {}

        class _Spy:
            def check(self, spec: Any, cfg: Any) -> None:
                del spec, cfg
                order.append("budget")

        monkeypatch.setattr(acquisition_service, "preflight_artifact_roots", _record_probe)
        acquisition_service.preflight(
            spec=load_spec(directory), cfg=CFG, run_id="run-a", guard=_Spy()
        )

        assert order.count("roots") == 1, f"the probe ran {order.count('roots')} times, not once"
        assert order == ["roots", "budget"], (
            f"the artifact-root probe and the budget guard ran in the order {order}; a run "
            f"whose roots cannot publish atomically should be refused before it is budgeted"
        )

    def test_submit_passes_the_run_id_through(
        self, rooted, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """``submit`` is the one caller that knows the run id, so it must pass it.

        Without this the trainer-state case would be unreachable on *every* path, and the
        fold would ship a check nothing could ever run - the unreachable-capability defect
        this project has shipped three times.
        """
        del rooted
        seen: list[Any] = []

        def _record(spec: Any, cfg: Any, **kwargs: Any) -> tuple[Any, Any]:
            seen.append(kwargs.get("run_id"))
            raise Refusal("stop_here", "preflight recorded; the rest of submit is not under test")

        monkeypatch.setattr(acquisition_service, "preflight", _record)
        with pytest.raises(Refusal):
            acquisition_service.submit(
                session=None,
                run_id="run-xyz",
                spec=AcquisitionSpec(
                    mode=AcquisitionMode.LOAD, path=Path("/tmp"), gpu_request="auto"
                ),
                cfg=CFG,
                enqueue=lambda _run_id: None,
            )
        assert seen == ["run-xyz"]


class TestTheArtifactRootsAtomicityCase:
    """FR-005.5 on the task path, which never had it.

    ``create_app`` probes the checkpoint root and **logs** - deliberately, so an unwritable
    mount does not take down ``/api/health`` and every screen that touches no artifact. A
    **Celery worker never calls** ``create_app``, so on the task path nothing probed
    anything and the first discovery was ``atomic_dir`` refusing at a save point hours into
    the run. Refusing at submit is R-02.48's own rule.
    """

    def test_an_existing_root_is_actually_probed(self, rooted) -> None:
        checkpoints, states = rooted
        report = acquisition_service.preflight_artifact_roots()
        assert report == {"checkpoint_root": "atomic", "trainer_state_root": "atomic"}
        # The probe cleans up after itself; a leftover directory would be indexed as a
        # checkpoint by `reindex` on the next sweep.
        assert [p.name for p in checkpoints.iterdir()] == []
        assert [p.name for p in states.iterdir()] == []

    def test_the_verdict_is_probed_once_per_process_not_once_per_submit(self, rooted) -> None:
        """Because the probe directory is named after the **pid**, not the call.

        ``verify_root_is_atomic`` creates ``.atomicity-probe-<pid>``, which is unique per
        process because it was written as a startup check. Called per submit it is not
        unique per call, so two concurrent submits in one uvicorn worker would race on
        ``mkdir`` and the loser would be refused ``checkpoint_root_unwritable`` **for
        colliding with itself** - a false refusal that appears only under load, which is
        precisely what a single-threaded suite cannot see. The memoisation and its lock are
        what close that, so the call count is asserted rather than assumed.
        """
        checkpoints, _states = rooted
        probes: list[Path] = []

        from core import atomic_dir as atomic_dir_module

        original = atomic_dir_module.verify_root_is_atomic

        def _counting(root: Path) -> Path:
            probes.append(root)
            return original(root)

        atomic_dir_module.verify_root_is_atomic = _counting  # type: ignore[assignment]
        try:
            first = acquisition_service.preflight_artifact_roots()
            second = acquisition_service.preflight_artifact_roots()
        finally:
            atomic_dir_module.verify_root_is_atomic = original  # type: ignore[assignment]

        assert first == second == {"checkpoint_root": "atomic", "trainer_state_root": "atomic"}
        assert len(probes) == 2, (
            f"the roots were probed {len(probes)} times across two calls; expected one probe "
            f"per root and then the cached verdict. Probes: {probes}"
        )
        assert checkpoints in probes

    def test_a_refusal_is_NOT_cached_so_a_fixed_mount_is_believed(self, tmp_path: Path) -> None:
        """An operator who fixes the mount must not have to restart the process.

        Caching the failure would make the refusal outlive its cause, and the only way back
        would be a restart - which turns a one-command fix into an outage.
        """
        root = tmp_path / "locked"
        root.mkdir()
        root.chmod(0o500)
        try:
            for _attempt in range(2):
                with pytest.raises(Refusal) as caught:
                    acquisition_service.preflight_artifact_roots(
                        checkpoint_root=root, trainer_state_root=root
                    )
                assert caught.value.code == "checkpoint_root_unwritable"
            root.chmod(0o700)
            assert acquisition_service.preflight_artifact_roots(
                checkpoint_root=root, trainer_state_root=root
            ) == {"checkpoint_root": "atomic", "trainer_state_root": "atomic"}
        finally:
            root.chmod(0o700)
            acquisition_service.reset_artifact_root_atomicity_cache()

    def test_an_unwritable_root_refuses_by_name(self, tmp_path: Path) -> None:
        root = tmp_path / "locked"
        root.mkdir()
        root.chmod(0o500)
        try:
            with pytest.raises(Refusal) as caught:
                acquisition_service.preflight_artifact_roots(
                    checkpoint_root=root, trainer_state_root=root
                )
            assert caught.value.code == "checkpoint_root_unwritable"
        finally:
            root.chmod(0o700)

    def test_an_absent_root_is_reported_not_silently_passed(self, tmp_path: Path) -> None:
        """Because the probe *creates* the root, and a request handler must not.

        On a default configuration that would mkdir ``/data/checkpoints`` from an API
        process and refuse every submit with ``checkpoint_root_unwritable`` - which is
        exactly the 38-collection-error failure ``verify_root_is_atomic``'s own comment
        records. So the verdict is ``not_checked_root_absent`` and the root is left alone.
        """
        absent = tmp_path / "nowhere"
        report = acquisition_service.preflight_artifact_roots(
            checkpoint_root=absent, trainer_state_root=absent
        )
        assert report == {
            "checkpoint_root": "not_checked_root_absent",
            "trainer_state_root": "not_checked_root_absent",
        }
        assert not absent.exists(), "the probe created a root it only meant to inspect"


class TestTheWorkerRunsTheSameCheck:
    """The worker's re-check, closed 2026-10-16. **The call, the payload, the call count.**

    ``test_submit_passes_the_run_id_through`` above is this assertion for the submit path.
    This is the same assertion for the path that actually allocates - and it is the one that
    matters to a continue, because submit's verdict is minutes old by the time the worker
    reads a tensor and the bytes were written hours before that.

    Both tests spy on :func:`services.acquisition.preflight_checkpoint_source` rather than on
    a wrapper, so a second copy of the check appearing anywhere would show up here as a
    **missing** call. *A grep proves a call exists; a spy on the shared implementation proves
    both callers depend on it.*
    """

    @staticmethod
    def _spy(recorded: list[dict[str, Any]]) -> Any:
        """A checker that records its whole payload and then stops the caller dead.

        It raises rather than returns, so neither path proceeds into acquisition - the
        acquirer would need a real ``transformers`` model and the subject here is the call,
        not what follows it. The sentinel code is asserted, because a test that accepted any
        ``Refusal`` would pass against a ``TypeError`` from a renamed keyword.
        """

        def _checker(spec: Any, **kwargs: Any) -> None:
            recorded.append({"spec": spec, **kwargs})
            raise Refusal("stop_here", "payload recorded; the rest of the caller is not under test")

        return _checker

    def test_the_worker_re_verifies_with_the_spec_the_run_id_and_the_config_hash(
        self, rooted, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        checkpoints, states = rooted
        directory = loadable_checkpoint(checkpoints)
        recorded: list[dict[str, Any]] = []
        monkeypatch.setattr(acquisition_service, "preflight_checkpoint_source", self._spy(recorded))

        from training import tasks

        with pytest.raises(Refusal) as caught:
            tasks.drive_run(
                run_id="run-a",
                spec=load_spec(directory),
                cfg=CFG,
                batch_source=None,  # never reached: the checker above raises first
                max_steps=1,
                log_root=tmp_path,
                trainer_state_root=states,
            )

        assert caught.value.code == "stop_here"
        assert len(recorded) == 1, (
            f"the worker ran the checkpoint check {len(recorded)} times. Once per run is "
            f"the contract: it hashes every weight byte, so a per-step or per-save call "
            f"would put seconds of SHA-256 on the step path and blow FR-006.50's 5 % budget."
        )
        call = recorded[0]
        assert call["spec"].path == directory
        assert call["run_id"] == "run-a", (
            "without the run id the trainer-state case is reported UNCHECKED rather than "
            "decided, so the worker would verify the hash and never learn whether its own "
            "optimizer state is restorable - which is the decision drive_run then honours."
        )
        assert call["configured_config_hash"] == CFG.config_hash(), (
            "the configured optimizer hash must travel, or FR-005.34's "
            "`optimizer_config_hash_changed` warning can never fire: a restored AdamW under "
            "a changed learning-rate schedule is a legitimate act and an unexplained loss "
            "discontinuity if nothing records it."
        )
        assert call["trainer_state_root"] == states

    def test_both_paths_go_through_ONE_checker_not_two(
        self, rooted, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Submit and the worker, driven in turn through the same spy.

        *A translation layer between two spellings of one check is a second source of
        truth*, and this project has paid for that four times. If either path grew its own
        copy, the list below would be one entry short - which is a louder failure than two
        checks that happen to agree today.
        """
        checkpoints, states = rooted
        directory = loadable_checkpoint(checkpoints)
        recorded: list[dict[str, Any]] = []
        monkeypatch.setattr(acquisition_service, "preflight_checkpoint_source", self._spy(recorded))

        from training import tasks

        with pytest.raises(Refusal):
            run_preflight(directory, run_id="run-a")
        with pytest.raises(Refusal):
            tasks.drive_run(
                run_id="run-a",
                spec=load_spec(directory),
                cfg=CFG,
                batch_source=None,
                max_steps=1,
                log_root=tmp_path,
                trainer_state_root=states,
            )

        assert [call["run_id"] for call in recorded] == ["run-a", "run-a"], (
            f"expected the submit path and the worker to reach the same checker once each; "
            f"got {recorded!r}. A missing entry means that path has its own copy of the "
            f"verification."
        )
