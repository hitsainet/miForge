"""AC12 on a card: a creation run, with nothing else running.  ``gpu``-marked.

**Deselected by default** and **not run for this increment** - see
``test_adapter_run_gpu.py``'s header for the full statement of why, and treat these as an
operator-run gate at feature acceptance rather than as green.

AC12 is *"a creation run with no other application running"*: a random-init model
instantiated in **miForge's own process**, a step-0 test score recorded as the baseline, and
a save producing a loadable checkpoint with ``parent: null`` and ``base_model`` set to the
config. The parts of that which need no card are already covered on CPU -
``test_create_run_type.py`` for the lineage and the refusals,
``test_standalone_without_siblings.py`` for AC20's share. **What needs a card is the
instantiation and the step**, and that is what is below.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from tests.support.test_adapter_doubles import TINY_CONFIG
from training.runtypes import resolve
from training.runtypes.base import SaveContext

pytestmark = pytest.mark.gpu

CREATE = resolve("create")


@pytest.fixture
def checkpoints(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """The checkpoint root, set **once** for the whole case.

    Set in a fixture rather than inside each test because the tokenizer directory has to
    live inside a configured model root and the spec resolves it at pre-flight: two tests
    each setting their own root after the spec was built is how one of them ends up
    resolving against the other's.
    """
    from core.config import reset_settings

    root = tmp_path / "checkpoints"
    root.mkdir()
    monkeypatch.setenv("CHECKPOINT_ROOT", str(root))
    reset_settings()
    yield root
    reset_settings()


@pytest.fixture
def tokenizer_dir(checkpoints: Path) -> Path:
    """A **real** tokenizer, sized from the config, inside the configured root.

    *This fixture used to be ``MIFORGE_TEST_TOKENIZER`` and a ``pytest.skip``*, which meant
    AC12's artifact half and AC23's create half were proved on **no machine anywhere** -
    and the reason the env var was needed is that nothing could hand a creation run a
    tokenizer at all. The reference is on the spec now, so the gate uses the product's own
    mechanism instead of reaching around it.
    """
    from tests.support.tiny_hf_model import write_tiny_tokenizer

    directory = checkpoints / "given-tokenizer"
    directory.mkdir()
    write_tiny_tokenizer(directory, vocab_size=TINY_CONFIG["vocab_size"])
    return directory


@pytest.fixture
def spec(tokenizer_dir: Path) -> Any:
    """The spec, **pinned to the resolved card by UUID**, naming its tokenizer.

    *This passed no ``gpu_request`` until 2026-10-16, i.e. ``"auto"``.* Acquisition now
    places the model on the leased card itself, and with several cards visible ``"auto"`` is
    refused by name (``gpu_not_pinned``) rather than resolved to the first enumerated device:
    the lease was taken on one specific UUID at submit, and placing elsewhere puts two runs
    on one card with both leases claiming exclusivity. So the gate pins the card the same way
    production must.
    """
    from core.errors import Refusal
    from core.gpu import resolve_device
    from training.acquisition.base import AcquisitionMode, AcquisitionSpec

    try:
        device = resolve_device("auto")
    except Refusal as exc:
        pytest.skip(f"no CUDA device resolved ({exc.code}); this case needs the GPU node")
    return AcquisitionSpec(
        mode=AcquisitionMode.CREATE,
        config=dict(TINY_CONFIG),
        seed=13,
        gpu_request=device.uuid,
        tokenizer_path=tokenizer_dir,
    )


class TestARealCreationRun:
    def test_the_model_is_instantiated_in_MIFORGES_OWN_PROCESS(self, spec: Any) -> None:
        """R-02.41, FR-008.8: no base model, no attachment, no sibling, nothing upstream."""
        from training.acquisition.scratch import ScratchAcquirer

        acquired = ScratchAcquirer().acquire(ScratchAcquirer().preflight(spec))
        assert acquired.module is not None
        assert acquired.record.parent is None
        assert acquired.record.base_model.startswith("config:")
        assert "+seed=13" in acquired.record.base_model

    def test_the_seed_makes_the_starting_hash_REPRODUCIBLE(self, spec: Any) -> None:
        """R-02.47 requires the content hash of the weights the run started from, and F1
        folds the seed into the identity so two creations from one config with different
        seeds have different ``base_model``. FPRD D9 records that no requirement names a
        seed field; this asserts what F1 shipped instead."""
        from training.acquisition.scratch import ScratchAcquirer

        first = ScratchAcquirer().acquire(ScratchAcquirer().preflight(spec)).record
        second = ScratchAcquirer().acquire(ScratchAcquirer().preflight(spec)).record
        assert first.starting_weights_sha256 == second.starting_weights_sha256
        assert first.base_model == second.base_model

        other = spec.model_copy(update={"seed": 14})
        third = ScratchAcquirer().acquire(ScratchAcquirer().preflight(other)).record
        assert third.base_model != first.base_model

    def test_prepare_returns_EVERY_parameter_and_no_base_digest(self, spec: Any) -> None:
        from training.acquisition.scratch import ScratchAcquirer

        acquired = ScratchAcquirer().acquire(ScratchAcquirer().preflight(spec))
        assert acquired.tokenizer is not None, "the spec named one; the acquisition loaded it"
        plan = CREATE.prepare(acquired)
        assert len(plan.trainable) == len(list(acquired.module.parameters()))
        assert plan.base_digest is None, "a creation run is supposed to change its parameters"

    def test_a_save_writes_a_loadable_checkpoint_with_the_creation_lineage(
        self, spec: Any, checkpoints: Path
    ) -> None:
        """AC12's artifact half, through F5's writer - F8 writes no checkpoint manifest."""
        from tests.support.checkpoints import create_feature_tables
        from tests.support.jobs import insert_run, memory_engine, session_for
        from training.acquisition.scratch import ScratchAcquirer

        engine = memory_engine()
        create_feature_tables(engine)

        acquired = ScratchAcquirer().acquire(ScratchAcquirer().preflight(spec))
        with session_for(engine) as session:
            insert_run(session, "run-create", status="running")
            outcome = CREATE.save(
                SaveContext(
                    run_id="run-create",
                    step=0,
                    session=session,
                    module=acquired.module,
                    domain=None,
                    signal_source="corpus",
                    record=acquired.record,
                    optimizer={"name": "adamw-8bit", "state": "fresh", "config_hash": "a" * 64},
                    tokenizer=acquired.tokenizer,
                    root=checkpoints,
                )
            )
            session.commit()

        manifest = json.loads((outcome.directory / "manifest.json").read_text(encoding="utf-8"))
        assert manifest["parent"] is None
        assert manifest["base_model"] == acquired.record.base_model
        assert manifest["domain"] is None, "written as null, not omitted"
        assert "domain" in manifest

    def test_the_checkpoint_loads_in_a_process_that_cannot_import_miforge(
        self, spec: Any, checkpoints: Path
    ) -> None:
        """The strongest form of the output contract (AC23's machinery, reused): the load
        happens in a subprocess whose ``sys.path`` has no miForge on it at all.

        **Also covered on the CPU now**, by
        ``test_create_run_ships_a_tokenizer.py::test_the_checkpoint_LOADS_IN_A_PROCESS_
        THAT_CANNOT_IMPORT_MIFORGE``, because neither instantiating a 2-layer model nor
        loading the result needs a card - and while this was the only copy, criterion 23
        was proved for a created checkpoint nowhere at all.
        """
        from tests.support.checkpoints import create_feature_tables
        from tests.support.conformance import assert_loads_in_a_clean_process
        from tests.support.jobs import insert_run, memory_engine, session_for
        from training.acquisition.scratch import ScratchAcquirer

        engine = memory_engine()
        create_feature_tables(engine)

        acquired = ScratchAcquirer().acquire(ScratchAcquirer().preflight(spec))
        with session_for(engine) as session:
            insert_run(session, "run-create", status="running")
            outcome = CREATE.save(
                SaveContext(
                    run_id="run-create",
                    step=0,
                    session=session,
                    module=acquired.module,
                    domain=None,
                    signal_source="corpus",
                    record=acquired.record,
                    optimizer={"name": "adamw-8bit", "state": "fresh", "config_hash": "a" * 64},
                    tokenizer=acquired.tokenizer,
                    root=checkpoints,
                )
            )
            session.commit()

        result = assert_loads_in_a_clean_process(outcome.directory)
        assert result["generated"].strip()
        assert result["miforge_modules"] == []
