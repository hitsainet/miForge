"""The attached run's tokenizer source: never a served ``<base>:<run>:step-N`` name.

**F11 window 3 (2026-10-18), FTASKS 18.4.** Every D52 resume died ``run_loop_error`` before
R-02.2: ``AttachAcquirer._tokenizer_loader`` took ``HOTUPDATE_VERIFY_TOKENIZER or
resident_model``, the deployment never sets the former, and after a checkpoint load the
resident is named ``LiquidAI/LFM2.5-1.2B-Instruct:f11w2-r4:step-100`` - which
``AutoTokenizer.from_pretrained`` rejects (``HFValidationError``). The resume's own
precondition is that the resident IS the checkpoint, so every resume hit it. Every test of
the attach path injected the loader, so the line had never run.

These tests run the REAL loader and the REAL ``from_pretrained`` (observed and called
through) against a genuinely loadable tokenizer written at a checkpoint's root under a
``CHECKPOINT_READ_ROOTS`` entry, offline (``HF_HUB_OFFLINE=1``), asserting the source
``from_pretrained`` received and how many times it was called.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, ClassVar

import pytest

from core.config import reset_settings
from core.errors import Refusal
from services.checkpoints.paths import checkpoint_dir
from tests.support.fake_weights_surface import FakeWeightsSurface
from tests.support.fixtures_hotupdate import configure
from tests.support.tiny_hf_model import write_tiny_tokenizer

pytest.importorskip("transformers")
pytest.importorskip("tokenizers")

BASE = "LiquidAI/LFM2.5-1.2B-Instruct"
RUN = "f11w2-r4"
STEP = 100
QUALIFIED = f"{BASE}:{RUN}:step-{STEP}"


@pytest.fixture
def roots(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> dict[str, Any]:
    """A primary root (miForge's) and a read root holding miLLM's checkpoint of step 100."""
    primary = tmp_path / "checkpoints"
    millm = tmp_path / "millm-checkpoints"
    primary.mkdir()
    millm.mkdir()
    directory = checkpoint_dir(RUN, STEP, root=millm)
    directory.mkdir(parents=True)
    write_tiny_tokenizer(directory)
    configure(
        monkeypatch,
        CHECKPOINT_ROOT=str(primary),
        CHECKPOINT_READ_ROOTS=str(millm),
        HOTUPDATE_VERIFY_TOKENIZER="",
    )
    monkeypatch.setenv("HF_HUB_OFFLINE", "1")
    yield {"primary": primary, "millm": millm, "directory": directory}
    reset_settings()


@pytest.fixture
def sources(monkeypatch: pytest.MonkeyPatch) -> list[str]:
    """Every source the REAL ``AutoTokenizer.from_pretrained`` receives (called through)."""
    import transformers

    real = transformers.AutoTokenizer.from_pretrained.__func__
    seen: list[str] = []

    def observed(cls: Any, source: Any, *args: Any, **kwargs: Any) -> Any:
        seen.append(str(source))
        return real(cls, source, *args, **kwargs)

    monkeypatch.setattr(transformers.AutoTokenizer, "from_pretrained", classmethod(observed))
    return seen


def _live() -> Any:
    from training.hotupdate import settings as settings_module

    reset_settings()
    return settings_module.load()


def _loader(resident: str, directory: Path | None = None) -> Any:
    from training.hotupdate import attach as attach_module

    return attach_module.AttachAcquirer()._tokenizer_loader(resident, _live(), directory)


class TestTheDefectIsReproducibleHere:
    def test_the_served_checkpoint_name_is_not_loadable_by_from_pretrained(self) -> None:
        """Positive check that this environment can SEE the defect: the name window 3 passed
        fails here as it did live, so a green below is not the network being absent."""
        from transformers import AutoTokenizer

        with pytest.raises(Exception) as caught:
            AutoTokenizer.from_pretrained(QUALIFIED)
        assert "Repo id" in str(caught.value) or "repo" in str(caught.value).lower()


class TestTheRealLoader:
    def test_a_checkpoint_resident_loads_from_its_directory_under_a_read_root(
        self, roots, sources
    ) -> None:
        tokenizer = _loader(QUALIFIED)()
        # Payload and count: the directory, once, and never the qualified name.
        assert sources == [str(roots["directory"])]
        assert QUALIFIED not in sources
        # A real, working tokenizer - the one miLLM wrote.
        assert tokenizer("t5 t6")["input_ids"] == [5, 6]

    def test_a_resume_directory_wins_over_the_name_lookup(self, roots, sources) -> None:
        """The directory the identity check verified is the one read - even when the run
        also has a directory in the primary root, which the name lookup would pick first."""
        shadow = checkpoint_dir(RUN, STEP, root=roots["primary"])
        shadow.mkdir(parents=True)  # the run exists in the primary root too, with no tokenizer
        _loader(QUALIFIED, roots["directory"])()
        assert sources == [str(roots["directory"])]

    def test_the_name_lookup_alone_would_pick_the_primary_root(self, roots, sources) -> None:
        """The control for the test above: without the resume directory the primary root is
        chosen (first root holding the run) and, having no tokenizer, refused by name."""
        checkpoint_dir(RUN, STEP, root=roots["primary"]).mkdir(parents=True)
        with pytest.raises(Refusal) as caught:
            _loader(QUALIFIED)()
        assert caught.value.code == "verification_tokenizer_unavailable"
        assert sources == []

    def test_the_configured_tokenizer_wins(self, roots, sources, monkeypatch, tmp_path) -> None:
        configured = tmp_path / "operator-tokenizer"
        configured.mkdir()
        write_tiny_tokenizer(configured)
        monkeypatch.setenv("HOTUPDATE_VERIFY_TOKENIZER", str(configured))
        _loader(QUALIFIED, roots["directory"])()
        assert sources == [str(configured)]

    def test_an_unreadable_checkpoint_refuses_by_name_and_loads_nothing(
        self, roots, sources, monkeypatch
    ) -> None:
        # The read root is not mounted in this worker: nothing holds the run.
        monkeypatch.setenv("CHECKPOINT_READ_ROOTS", "")
        with pytest.raises(Refusal) as caught:
            _loader(QUALIFIED)()
        assert caught.value.code == "verification_tokenizer_unavailable"
        assert caught.value.details["directory_exists"] is False
        assert sources == [], "a qualified name must never reach from_pretrained"

    def test_a_checkpoint_without_tokenizer_files_refuses(self, roots, sources) -> None:
        for path in roots["directory"].iterdir():
            path.unlink()
        with pytest.raises(Refusal) as caught:
            _loader(QUALIFIED)()
        assert caught.value.code == "verification_tokenizer_unavailable"
        assert caught.value.details["directory"] == str(roots["directory"])
        assert sources == []

    def test_a_plain_served_name_is_still_passed_through(self, roots, monkeypatch) -> None:
        """Not a checkpoint: the served name is the only name there is (unchanged)."""
        import transformers

        seen: list[str] = []
        monkeypatch.setattr(
            transformers.AutoTokenizer,
            "from_pretrained",
            classmethod(lambda cls, source, **kw: seen.append(source)),
        )
        _loader(BASE)()
        assert seen == [BASE]


class TestTheAcquisitionUsesTheRealLoader:
    """Through ``AttachAcquirer.acquire`` with a checkpoint resident and NO loader injected:
    removing the acquisition's tokenizer line, or loading from the served name, is red."""

    def _acquirer(self, monkeypatch: pytest.MonkeyPatch, surface: Any = None) -> tuple[Any, Any]:
        from training.acquisition.base import Availability
        from training.hotupdate import attach as attach_module
        from training.hotupdate import rebuild

        surface = surface or FakeWeightsSurface(resident_model=QUALIFIED)
        acquirer = attach_module.AttachAcquirer(transport=surface.transport())
        monkeypatch.setattr(type(acquirer), "available", lambda self: Availability(available=True))

        class Base:
            device_uuid = "GPU-test"
            tensors: ClassVar[dict[str, Any]] = {}

        monkeypatch.setattr(rebuild, "build", lambda response, **kw: Base())
        monkeypatch.setattr(rebuild, "check_build_identity", lambda response: None)
        monkeypatch.setattr(attach_module, "build_module", lambda response, base: object())
        monkeypatch.setattr(attach_module, "acquisition_record", lambda **kw: None)
        monkeypatch.setattr(
            attach_module.AttachAcquirer, "_verify_provider", lambda self, *a: object()
        )
        return acquirer, surface

    @staticmethod
    def _spec() -> Any:
        from training.acquisition.base import AcquisitionMode, AcquisitionSpec

        return AcquisitionSpec(mode=AcquisitionMode.ATTACH)

    def test_the_acquired_tokenizer_is_the_checkpoints_and_the_verifier_shares_it(
        self, roots, sources, monkeypatch
    ) -> None:
        acquirer, _ = self._acquirer(monkeypatch)
        acquired = acquirer.acquire(self._spec())
        try:
            assert acquired.tokenizer is not None
            assert acquired.tokenizer("t7")["input_ids"] == [7]
            # R-02.2 encodes with the SAME object the batches are encoded with - one load.
            assert acquired.verifier.tokenizer_loader() is acquired.tokenizer
            assert sources == [str(roots["directory"])]
        finally:
            if acquirer.heartbeat is not None:
                acquirer.heartbeat.stop()

    def test_an_unreadable_tokenizer_detaches_the_attachment(
        self, roots, sources, monkeypatch
    ) -> None:
        monkeypatch.setenv("CHECKPOINT_READ_ROOTS", "")
        reset_settings()
        acquirer, surface = self._acquirer(monkeypatch)
        with pytest.raises(Refusal) as caught:
            acquirer.acquire(self._spec())
        assert caught.value.code == "verification_tokenizer_unavailable"
        assert sources == []
        # The acquisition failed AFTER attach, so it released the slot it took.
        assert [op for op in surface.ops() if op != "status"][-1] == "detach"

    def test_a_resume_reads_the_directory_it_verified_not_a_name_lookup(
        self, roots, sources, monkeypatch
    ) -> None:
        """A path-form resume of the read root's checkpoint, while the run ALSO has a
        directory in the primary root (no tokenizer): the acquisition must load from the
        directory the identity check verified by manifest-file hash. Dropping the resume
        directory from the acquisition's loader call falls back to the name lookup, which
        picks the primary root and refuses - red."""
        import hashlib
        import json

        from core.canonical import write_canonical
        from services.checkpoints import manifest as manifest_module
        from training.acquisition.base import AcquisitionMode, AcquisitionSpec

        config = {
            "model_type": "llama",
            "hidden_size": 8,
            "num_attention_heads": 2,
            "num_hidden_layers": 1,
        }
        directory = roots["directory"]
        (directory / "config.json").write_text(json.dumps(config))
        doc = manifest_module.build(
            run_id=RUN,
            step=STEP,
            domain="legal",
            signal_source="corpus",
            base_model=BASE,
            parent=None,
            optimizer={"name": "adamw-32bit", "state": "fresh", "config_hash": None},
            weights_sha256="0" * 64,
        )
        write_canonical(directory / manifest_module.MANIFEST_NAME, doc)
        manifest_hash = hashlib.sha256(
            (directory / manifest_module.MANIFEST_NAME).read_bytes()
        ).hexdigest()
        assert doc["name"] == QUALIFIED
        checkpoint_dir(RUN, STEP, root=roots["primary"]).mkdir(parents=True)  # the shadow

        surface = FakeWeightsSurface(
            resident_model=QUALIFIED, manifest_sha256=manifest_hash, steps_since_load=0
        )
        surface.config = dict(config)
        acquirer, _ = self._acquirer(monkeypatch, surface)
        acquired = acquirer.acquire(
            AcquisitionSpec(mode=AcquisitionMode.ATTACH, resume_from={"path": str(directory)})
        )
        try:
            assert acquirer.resume_verified is not None
            assert acquirer.resume_verified.stage == "attach"
            assert sources == [str(directory)]
            assert acquired.tokenizer("t9")["input_ids"] == [9]
        finally:
            if acquirer.heartbeat is not None:
                acquirer.heartbeat.stop()
