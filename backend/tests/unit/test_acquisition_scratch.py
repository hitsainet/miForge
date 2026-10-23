"""Creating a model from an architecture config  (R-02.46 case 2; FR-001.6, E-5, AC-001.3).

Three things are settled here, and two of them were open questions in this feature's own
documents:

* **The config is parsed and validated before anything is instantiated.** R-02.42 as
  corrected estimates *"from the architecture config, before ``instantiate``"*, because **a
  from-scratch run has no resident model to measure** - so the guard fires too late for the
  run type that most needs it unless the config comes first.
* **The seed** (FR-001.11, OQ-2). R-02.47 requires the content hash of the weights a run
  started from, for *both* modes, and no BRD-02 requirement records a seed - so a random
  initialisation's hash said *which* init happened without letting anyone reproduce it. The
  operator recorded a seed (2026-10-14), following ``GenerationParams.seed``'s precedent:
  **null is an assertion that no seed was set**, and therefore that the hash is not
  reproducible.
* **The lineage is ``parent: null`` with ``base_model`` = the config** (R-02.47), which the
  vendored schema already allowed - no schema change was ever needed.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from core.canonical import sha256_of
from core.errors import Refusal
from training.acquisition.base import AcquisitionMode, AcquisitionSpec
from training.acquisition.digest import config_identity, created_weights_sha256
from training.acquisition.scratch import (
    ScratchAcquirer,
    assert_architecture_known,
    created_base_model_identity,
    parse_config,
)


class _Tokenizer:
    """A tokenizer double **that can express the vocabulary defect.**

    ``len`` is what :func:`~training.acquisition.scratch.assert_loaded_vocab` reads, so a
    double without it would make every acquire-path assertion vacuous on the one check
    that stands between a 32k tokenizer and ``IndexError`` - *a double that cannot express
    the defect makes its assertion vacuous* (F11's MUT-19a, this project's own).
    """

    def __init__(self, size: int = 256) -> None:
        self._size = size

    def __len__(self) -> int:
        return self._size


LLAMA = {
    "model_type": "llama",
    "hidden_size": 64,
    "num_hidden_layers": 2,
    "num_attention_heads": 4,
    "vocab_size": 256,
}


def spec(**overrides) -> AcquisitionSpec:
    base = {"mode": AcquisitionMode.CREATE, "config": dict(LLAMA)}
    base.update(overrides)
    return AcquisitionSpec(**base)


@pytest.fixture
def tokenizer_dir(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """A tokenizer directory **inside a configured model root**, sized for ``LLAMA``.

    Inside a root because a submitted path is an input even on a LAN and
    ``resolve_within_roots`` confines it - the same policy the ``load`` half uses, reused
    rather than relaxed for the tokenizer.
    """
    from core.config import reset_settings
    from tests.support.tiny_hf_model import write_tokenizer_json_only

    root = tmp_path / "checkpoints"
    monkeypatch.setenv("CHECKPOINT_ROOT", str(root))
    reset_settings()
    directory = write_tokenizer_json_only(root / "tok", vocab_size=LLAMA["vocab_size"])
    yield directory
    reset_settings()


class TestConfigParsing:
    def test_an_inline_config_parses_and_names_its_architecture(self) -> None:
        parsed = parse_config(spec())
        assert parsed.architecture == "llama"
        assert parsed.data == LLAMA

    def test_a_config_file_parses(self, tmp_path: Path) -> None:
        path = tmp_path / "arch.json"
        path.write_text(json.dumps(LLAMA))
        parsed = parse_config(spec(config=None, config_path=path))
        assert parsed.data == LLAMA

    def test_architectures_is_accepted_as_well_as_model_type(self) -> None:
        """Configs in the wild use both forms, so neither is declared canonical."""
        parsed = parse_config(spec(config={"architectures": ["LlamaForCausalLM"]}))
        assert parsed.architecture == "LlamaForCausalLM"

    def test_a_missing_file_refuses_by_name(self, tmp_path: Path) -> None:
        with pytest.raises(Refusal) as caught:
            parse_config(spec(config=None, config_path=tmp_path / "absent.json"))
        assert caught.value.code == "architecture_config_invalid"
        assert "absent.json" in caught.value.details["path"]

    def test_unparseable_json_refuses_before_any_allocation(self, tmp_path: Path) -> None:
        path = tmp_path / "arch.json"
        path.write_text("{nope")
        with pytest.raises(Refusal) as caught:
            parse_config(spec(config=None, config_path=path))
        assert caught.value.code == "architecture_config_invalid"
        assert "before any allocation" in caught.value.message

    def test_a_config_naming_no_architecture_refuses_and_NAMES_THE_FIELD(self) -> None:
        """E-5's requirement, and ADR-018's: **no default.** A guessed architecture produces
        a confident estimate of a model that does not exist, and reaches the allocator with
        the budget guard already satisfied."""
        with pytest.raises(Refusal) as caught:
            parse_config(spec(config={"hidden_size": 64}))
        assert caught.value.code == "architecture_config_invalid"
        assert caught.value.details["field"] == "architectures"
        assert "guessing" in caught.value.message

    def test_a_quantized_architecture_refuses(self) -> None:
        """A quantized architecture cannot describe a full-parameter trainable model, and
        refusing at the config is cheaper than refusing after instantiation."""
        with pytest.raises(Refusal) as caught:
            parse_config(spec(config={**LLAMA, "quantization_config": {"bits": 4}}))
        assert caught.value.code == "architecture_config_invalid"
        assert caught.value.details["field"] == "quantization_config"

    def test_a_spec_with_no_config_at_all_refuses(self) -> None:
        with pytest.raises(Refusal) as caught:
            parse_config(spec().model_copy(update={"config": None, "config_path": None}))
        assert caught.value.code == "architecture_config_invalid"
        assert "did not ask for" in caught.value.message


class TestArchitectureValidation:
    def test_an_unknown_architecture_is_not_refused_when_transformers_is_absent(self) -> None:
        """**The two facts are kept apart.** "This architecture is unknown" and "this
        machine has no ML runtime" have different fixes, and ``available()`` already reports
        the second by name - so conflating them would send the operator to the wrong one."""
        try:
            import transformers  # noqa: F401

            installed = True
        except ImportError:
            installed = False

        if installed:  # pragma: no cover - depends on the machine
            with pytest.raises(Refusal):
                assert_architecture_known("not-a-real-architecture")
        else:
            assert assert_architecture_known("not-a-real-architecture") is None


class TestTheSeedMakesTheStartingHashMeaningful:
    """FR-001.11 / OQ-2, settled by the operator on 2026-10-14."""

    def test_two_seeds_produce_different_starting_hashes(self) -> None:
        """Which is the entire point: without a seed the hash identified *which*
        initialisation happened but let nobody reproduce it."""
        a = created_weights_sha256(config=LLAMA, seed=1)
        b = created_weights_sha256(config=LLAMA, seed=2)
        assert a != b

    def test_the_same_config_and_seed_produce_the_same_hash(self) -> None:
        assert created_weights_sha256(config=LLAMA, seed=7) == created_weights_sha256(
            config=dict(reversed(list(LLAMA.items()))), seed=7
        ), "canonical hashing must be stable under key reordering (ADR-005)"

    def test_no_seed_is_recorded_as_null_rather_than_omitted(self) -> None:
        """``GenerationParams.seed``'s precedent in this codebase: *null is an assertion
        that no seed was set*, which makes the result irreproducible - and that is worth
        recording as a fact rather than hiding."""
        digest = created_weights_sha256(config=LLAMA, seed=None)
        assert digest == sha256_of({"created_from": {"config": LLAMA, "seed": None}})
        assert digest != created_weights_sha256(
            config=LLAMA, seed=0
        ), "an absent seed and a seed of zero are different facts"

    def test_the_seed_is_inside_the_base_model_identity(self) -> None:
        """**Why it has to be in the string.** ``checkpoint-manifest-v1.json``'s
        ``base_model`` is a plain ``string`` and the manifest is closed to unknown keys, so
        there is no separate field for a seed and adding one would be a schema change in two
        repos. Folding it in means two creations from one config with different seeds have
        different ``base_model`` values, so the required ``starting_weights_sha256`` is
        explicable from the manifest alone.
        """
        with_seed = created_base_model_identity(LLAMA, 42)
        without = created_base_model_identity(LLAMA, None)
        assert with_seed != without
        assert with_seed == f"config:llama@{config_identity(LLAMA)}+seed=42"
        assert without.endswith("+seed=none"), (
            "an absent seed marker would read as an older identity format; the explicit "
            "'none' is the assertion that the initialisation is not reproducible"
        )


class TestPreflightHandsTheConfigToTheBudgetGuard:
    def test_preflight_yields_a_config_and_NO_parameter_count(self, tokenizer_dir: Path) -> None:
        """R-02.42 estimates from the config precisely because there is nothing to measure
        yet; a count here could only come from building the model the guard exists to veto."""
        enriched = ScratchAcquirer().preflight(spec(tokenizer_path=tokenizer_dir))
        assert enriched.architecture_config == LLAMA
        assert enriched.param_count is None

    def test_preflight_VALIDATES_THE_ARCHITECTURE_against_the_live_library(
        self, monkeypatch: pytest.MonkeyPatch, tokenizer_dir: Path
    ) -> None:
        """**A test finding from the mutation sweep (M28), written up as a regression.**

        ``assert_architecture_known`` was unit-tested and **nothing asserted it was wired
        into ``preflight``**: deleting the call left the suite green, and an architecture
        ``transformers`` does not know would then have reached the allocator with the budget
        guard already satisfied - E-5's refusal, silently removed.

        Asserted by **the argument and the call count**, because on a machine without
        ``transformers`` the check is deliberately a no-op (it must not conflate "unknown
        architecture" with "no ML runtime"), so there is no observable refusal to assert
        here and only the wiring can be pinned.
        """
        from training.acquisition import scratch as scratch_module

        seen: list[str] = []
        monkeypatch.setattr(scratch_module, "assert_architecture_known", seen.append)
        ScratchAcquirer().preflight(spec(tokenizer_path=tokenizer_dir))
        assert seen == [
            "llama"
        ], "preflight must validate the architecture the config names, before instantiation"

    def test_an_unknown_architecture_is_refused_at_preflight_when_the_library_is_present(
        self, monkeypatch: pytest.MonkeyPatch, tokenizer_dir: Path
    ) -> None:
        """The observable half, with the library's answer stubbed at the one seam.

        Stubbed rather than skipped so the refusal path is covered in CI, where there is no
        ``transformers`` - otherwise E-5's most important case would only ever run on the
        GPU node.
        """
        from training.acquisition import scratch as scratch_module

        def refuse_everything(architecture: str) -> None:
            raise Refusal(
                "architecture_config_invalid",
                f"transformers does not know the architecture {architecture!r}",
                {"field": "architectures", "architecture": architecture},
            )

        monkeypatch.setattr(scratch_module, "assert_architecture_known", refuse_everything)
        with pytest.raises(Refusal) as caught:
            ScratchAcquirer().preflight(spec(tokenizer_path=tokenizer_dir))
        assert caught.value.code == "architecture_config_invalid"
        assert caught.value.details["architecture"] == "llama"

    def test_a_config_path_is_resolved_into_the_spec_so_the_guard_can_read_it(
        self, tmp_path: Path, tokenizer_dir: Path
    ) -> None:
        """``architecture_config`` deliberately does not touch the filesystem, so preflight
        has to be what turns a path into data - otherwise the guard's "exactly one input"
        rule would depend on whether a file happened to be readable."""
        path = tmp_path / "arch.json"
        path.write_text(json.dumps(LLAMA))
        enriched = ScratchAcquirer().preflight(
            spec(config=None, config_path=path, tokenizer_path=tokenizer_dir)
        )
        assert enriched.architecture_config == LLAMA
        assert enriched.config_path is None

    def test_a_create_spec_may_not_also_name_a_model_source(self) -> None:
        """A created model is initialised, never loaded; a spec naming both is refused rather
        than resolved by precedence."""
        from pydantic import ValidationError

        with pytest.raises(ValidationError, match="initialised, never loaded"):
            AcquisitionSpec(mode=AcquisitionMode.CREATE, config=dict(LLAMA), path=Path("/tmp/x"))


class TestTheLineageOfACreation:
    """AC-001.3, via the real record rather than a hand-built literal."""

    def test_the_record_is_parent_null_with_base_model_the_config(
        self, monkeypatch: pytest.MonkeyPatch, tokenizer_dir: Path
    ) -> None:
        from training.acquisition import scratch as scratch_module

        monkeypatch.setattr(
            scratch_module, "_instantiate", lambda parsed, seed, tok: (object(), _Tokenizer())
        )
        record = ScratchAcquirer().acquire(spec(seed=11, tokenizer_path=tokenizer_dir)).record

        assert record.parent is None, "R-02.47: a creation has no parent"
        assert record.base_model == created_base_model_identity(LLAMA, 11)
        assert record.config_identity == config_identity(LLAMA)
        assert record.seed == 11
        assert record.starting_weights_sha256 == created_weights_sha256(config=LLAMA, seed=11)
        assert record.source_path is None and record.source_repo is None

    def test_the_tokenizer_is_not_invented_and_IS_the_one_the_spec_NAMED(
        self, monkeypatch: pytest.MonkeyPatch, tokenizer_dir: Path
    ) -> None:
        """miForge does not train a tokenizer (BRD-02 section 8), so a creation is *given*
        one - and until 2026-10-16 **nothing could give it one.**

        *This case read ``assert acquired.tokenizer is None`` and passed*, which is the
        shape worth remembering: it pinned the defect instead of preventing it. A
        ``create`` run then trained for however long it was configured for and died at its
        first save point on ``AttributeError: 'NoneType' object has no attribute
        'save_pretrained'``. "Not invented" is still the rule; what changed is that the
        reference exists, so the assertion can be that the tokenizer is **the named one**
        rather than that there is none.
        """
        from training.acquisition import scratch as scratch_module

        seen: list[Path] = []

        def fake(parsed, seed, tokenizer_directory):
            seen.append(tokenizer_directory)
            return object(), _Tokenizer()

        monkeypatch.setattr(scratch_module, "_instantiate", fake)
        acquired = ScratchAcquirer().acquire(spec(tokenizer_path=tokenizer_dir))
        assert acquired.tokenizer is not None
        assert seen == [tokenizer_dir], "the tokenizer comes from the directory the spec named"
        assert acquired.record.tokenizer_source == str(tokenizer_dir)

    def test_the_realised_parameter_count_is_recorded_when_it_can_be_read(
        self, monkeypatch: pytest.MonkeyPatch, tokenizer_dir: Path
    ) -> None:
        """Recorded so the guard's **estimate** from the config can be checked against the
        realised figure. An estimator that is wrong is otherwise invisible: the run either
        fits or dies in the allocator, and neither says the estimate was off."""
        from training.acquisition import scratch as scratch_module

        class FakeParam:
            def numel(self) -> int:
                return 1000

        class FakeModule:
            def parameters(self):
                return [FakeParam(), FakeParam()]

        monkeypatch.setattr(
            scratch_module, "_instantiate", lambda parsed, seed, tok: (FakeModule(), _Tokenizer())
        )
        count = ScratchAcquirer().acquire(spec(tokenizer_path=tokenizer_dir)).record.param_count
        assert count == 2000


# ===================================================== the tokenizer a create run ships


class TestTheTokenizerReferenceIsTheSUPPLYMECHANISM:
    """**The defect this class exists for, stated once.**

    ``_instantiate`` returned ``(module, None)`` and said *"the caller supplies it"*.
    **No caller in ``backend/src/`` existed** - ``AcquisitionSpec`` carried no field for the
    reference, and ``CreateRunType.prepare``, which does raise ``tokenizer_required``, has
    no production caller (``training/run_driver.py`` records that correction). So a
    ``create`` run trained for however long it was configured for, reached its first save
    point and raised ``AttributeError: 'NoneType' object has no attribute
    'save_pretrained'`` inside F5's writer, having produced nothing. Scoring raised
    ``TypeError: 'NoneType' object is not callable`` one level over.

    **A ``None`` guard would have been the wrong fix** and these cases are what distinguish
    the two: the assertions below are about a tokenizer being *supplied*, not about an
    absence being *reported tidily*.
    """

    def test_a_creation_with_NO_tokenizer_is_refused_BY_NAME(self) -> None:
        """R-02.48 / ADR-013. The refusal is in ``preflight``, so it lands at submit -
        ``TestTheRefusalIsONTHESUBMITPATH`` is what proves that rather than this."""
        with pytest.raises(Refusal) as caught:
            ScratchAcquirer().preflight(spec())
        assert caught.value.code == "tokenizer_required"
        assert caught.value.details["fields"] == ["tokenizer_path", "tokenizer_repo"]
        # The fields are named in `details` for an API or agent caller. The MESSAGE is read
        # by the person at the New run form (2026-10-20), which now has a Tokenizer field, so
        # it says what to choose in plain words rather than spelling the wire names.
        message = caught.value.message
        assert "tokenizer" in message.lower()
        assert "folder on this host" in message and "Hugging Face repository" in message
        assert "`" not in message

    def test_acquire_ALSO_refuses_rather_than_handing_back_a_None_tokenizer(self) -> None:
        """``acquire`` is a public method. A caller that skipped preflight must be told so,
        not handed the ``tokenizer=None`` that caused the original crash."""
        with pytest.raises(Refusal) as caught:
            ScratchAcquirer().acquire(spec())
        assert caught.value.code == "tokenizer_required"

    def test_naming_BOTH_spellings_is_refused_rather_than_resolved_by_precedence(self) -> None:
        from pydantic import ValidationError

        with pytest.raises(ValidationError, match="naming two is refused"):
            AcquisitionSpec(
                mode=AcquisitionMode.CREATE,
                config=dict(LLAMA),
                tokenizer_path=Path("/tmp/a"),
                tokenizer_repo="org/tok",
            )

    def test_a_LOAD_may_not_name_a_tokenizer(self) -> None:
        """A load's tokenizer is the model directory's own, and
        ``require_tokenizer_at_root`` already requires it to be there. Two answers about
        which tokenizer one artifact carries is worse than one."""
        from pydantic import ValidationError

        with pytest.raises(ValidationError, match="takes no tokenizer reference"):
            AcquisitionSpec(path=Path("/tmp/model"), tokenizer_path=Path("/tmp/tok"))

    def test_a_tokenizer_revision_without_a_repo_is_refused_rather_than_IGNORED(self) -> None:
        """A silently dropped revision means a different tokenizer than the one asked for,
        and nothing in the record would say so."""
        from pydantic import ValidationError

        with pytest.raises(ValidationError, match="revision of nothing"):
            AcquisitionSpec(
                mode=AcquisitionMode.CREATE, config=dict(LLAMA), tokenizer_revision="main"
            )

    def test_a_path_outside_the_configured_roots_is_refused(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The same confinement the ``load`` half uses, reused rather than relaxed: a
        submitted path is an input even on a LAN, and the worker that reads it holds the
        card."""
        from core.config import reset_settings

        monkeypatch.setenv("CHECKPOINT_ROOT", str(tmp_path / "checkpoints"))
        reset_settings()
        outside = tmp_path / "elsewhere"
        outside.mkdir()
        (outside / "tokenizer.json").write_text("{}")
        try:
            with pytest.raises(Refusal) as caught:
                ScratchAcquirer().preflight(spec(tokenizer_path=outside))
        finally:
            reset_settings()
        assert caught.value.code == "tokenizer_source_outside_root"
        assert "tokenizers" in caught.value.message, (
            "'miForge reads models from ...' on a create run names a field the payload "
            "does not contain; a create has no model source"
        )

    def test_a_directory_with_no_tokenizer_files_is_refused(self, tokenizer_dir: Path) -> None:
        empty = tokenizer_dir.parent / "empty"
        empty.mkdir()
        with pytest.raises(Refusal) as caught:
            ScratchAcquirer().preflight(spec(tokenizer_path=empty))
        assert caught.value.code == "tokenizer_source_incomplete"
        assert caught.value.details["missing"]

    def test_a_tokenizer_SUBDIRECTORY_is_refused_with_its_own_message(
        self, tokenizer_dir: Path
    ) -> None:
        """ADR-015's own historical defect: the files are present and one level down, and
        "tokenizer missing" would send the operator looking for the wrong thing."""
        from tests.support.tiny_hf_model import write_tokenizer_json_only

        parent = tokenizer_dir.parent / "nested"
        parent.mkdir()
        write_tokenizer_json_only(parent / "tokenizer", vocab_size=LLAMA["vocab_size"])
        with pytest.raises(Refusal) as caught:
            ScratchAcquirer().preflight(spec(tokenizer_path=parent))
        assert caught.value.code == "tokenizer_not_at_root"
        assert caught.value.details["found_in"].endswith("tokenizer")

    def test_a_path_that_is_not_a_directory_is_refused(self, tokenizer_dir: Path) -> None:
        f = tokenizer_dir.parent / "a-file"
        f.write_text("{}")
        with pytest.raises(Refusal) as caught:
            ScratchAcquirer().preflight(spec(tokenizer_path=f))
        assert caught.value.code == "tokenizer_source_incomplete"

    def test_the_resolved_reference_is_RECORDED_so_the_vocabulary_is_traceable(
        self, monkeypatch: pytest.MonkeyPatch, tokenizer_dir: Path
    ) -> None:
        """The checkpoint manifest is closed to unknown keys and cannot carry this, so the
        acquisition record is the only place a created model's vocabulary is written down.
        A ``base_model`` of ``config:llama@...`` says nothing about which tokenizer it was
        trained against."""
        from training.acquisition import scratch as scratch_module

        monkeypatch.setattr(
            scratch_module, "_instantiate", lambda parsed, seed, tok: (object(), _Tokenizer())
        )
        enriched = ScratchAcquirer().preflight(spec(tokenizer_path=tokenizer_dir))
        record = ScratchAcquirer().acquire(enriched).record
        assert record.tokenizer_source == str(tokenizer_dir)
        assert record.tokenizer_vocab_size == LLAMA["vocab_size"]


class TestTheVocabularyGate:
    """``LLAMA`` declares ``vocab_size: 256``. A 32k tokenizer against it raises
    ``IndexError: index out of range in self`` from the embedding lookup at the first
    generation - a crash about an index, hours after a submit that looked fine, naming
    neither the tokenizer nor the config."""

    def test_a_tokenizer_TOO_BIG_for_the_config_is_refused_by_name_naming_BOTH_numbers(
        self, tokenizer_dir: Path
    ) -> None:
        from tests.support.tiny_hf_model import write_tokenizer_json_only

        big = write_tokenizer_json_only(tokenizer_dir.parent / "big", vocab_size=32000)
        with pytest.raises(Refusal) as caught:
            ScratchAcquirer().preflight(spec(tokenizer_path=big))
        assert caught.value.code == "tokenizer_vocab_mismatch"
        assert caught.value.details["tokenizer_vocab_size"] == 32000
        assert caught.value.details["config_vocab_size"] == 256
        assert "IndexError" in caught.value.message, (
            "the message must name the failure it is preventing, because that is the "
            "symptom an operator who ignores it will actually see"
        )

    def test_a_tokenizer_SMALLER_than_the_config_is_allowed(self, tokenizer_dir: Path) -> None:
        """Surplus embedding rows are wasteful, not wrong - and refusing would stop a
        perfectly serviceable run."""
        from tests.support.tiny_hf_model import write_tokenizer_json_only

        small = write_tokenizer_json_only(tokenizer_dir.parent / "small", vocab_size=64)
        enriched = ScratchAcquirer().preflight(spec(tokenizer_path=small))
        assert enriched.tokenizer_vocab_size == 64

    def test_an_ADDED_TOKEN_above_the_base_vocabulary_counts(self, tokenizer_dir: Path) -> None:
        """An added token's id indexes the same embedding table, and real tokenizers
        routinely place them above the base vocabulary. Counting only ``model.vocab``
        would let a 257th id through on a 256-row table."""
        import json as _json

        from training.acquisition.scratch import tokenizer_vocab_ceiling

        directory = tokenizer_dir.parent / "added"
        directory.mkdir()
        (directory / "tokenizer.json").write_text(
            _json.dumps(
                {
                    "added_tokens": [{"id": 999, "content": "<extra>"}],
                    "model": {"type": "WordLevel", "vocab": {"a": 0, "b": 1}},
                }
            ),
            encoding="utf-8",
        )
        assert tokenizer_vocab_ceiling(directory) == 1000
        with pytest.raises(Refusal) as caught:
            ScratchAcquirer().preflight(spec(tokenizer_path=directory))
        assert caught.value.code == "tokenizer_vocab_mismatch"

    def test_a_UNREADABLE_vocabulary_is_REPORTED_not_treated_as_agreement(
        self, tokenizer_dir: Path, caplog: pytest.LogCaptureFixture
    ) -> None:
        """A SentencePiece ``tokenizer.model`` needs the library to parse. The *early*
        refusal is lost; the refusal is not - ``assert_loaded_vocab`` runs against the
        loaded tokenizer in ``acquire``. **A skip that names its reason is the honest
        form**; a silent pass would report a check that never happened."""
        import logging

        directory = tokenizer_dir.parent / "sentencepiece"
        directory.mkdir()
        (directory / "tokenizer.model").write_bytes(b"\x00not-parseable-without-the-library")

        with caplog.at_level(logging.WARNING, logger="miforge.acquisition.scratch"):
            enriched = ScratchAcquirer().preflight(spec(tokenizer_path=directory))
        assert enriched.tokenizer_vocab_size is None
        record = next(
            r for r in caplog.records if getattr(r, "code", None) == "tokenizer_vocab_not_checked"
        )
        # **THE REASON, not only the code.** A surviving mutation (M7 of this round's sweep)
        # renamed the `reason` key and the suite stayed green: the code alone cannot tell
        # *"this config states no vocab_size"* from *"this tokenizer's vocabulary needs the
        # library to read"*, and those are two different facts with two different fixes.
        # A skip that does not name its reason is the dishonest form of a skip.
        assert record.reason == "tokenizer_vocabulary_not_readable_without_the_library"
        assert record.tokenizer_source == str(directory)

    def test_a_config_with_NO_vocab_size_is_reported_as_its_OWN_reason(
        self, tokenizer_dir: Path, caplog: pytest.LogCaptureFixture
    ) -> None:
        """The other unchecked branch, and the reason has to distinguish them. Both say
        *"not checked"*; only the reason says which file the operator should look at."""
        import logging

        no_vocab = {k: v for k, v in LLAMA.items() if k != "vocab_size"}
        with caplog.at_level(logging.WARNING, logger="miforge.acquisition.scratch"):
            enriched = ScratchAcquirer().preflight(
                AcquisitionSpec(
                    mode=AcquisitionMode.CREATE,
                    config=no_vocab,
                    tokenizer_path=tokenizer_dir,
                )
            )
        assert enriched.tokenizer_vocab_size is None
        record = next(
            r for r in caplog.records if getattr(r, "code", None) == "tokenizer_vocab_not_checked"
        )
        assert record.reason == "config_states_no_vocab_size"

    def test_the_LOADED_tokenizers_length_is_checked_too_and_that_is_the_FACT(
        self, monkeypatch: pytest.MonkeyPatch, tokenizer_dir: Path
    ) -> None:
        """The files are a prediction of the number the embedding lookup will be handed;
        ``len(tokenizer)`` is that number. This is the only check the unreadable case has,
        so a double without ``__len__`` would make it vacuous."""
        from training.acquisition import scratch as scratch_module

        monkeypatch.setattr(
            scratch_module,
            "_instantiate",
            lambda parsed, seed, tok: (object(), _Tokenizer(size=32000)),
        )
        with pytest.raises(Refusal) as caught:
            ScratchAcquirer().acquire(spec(tokenizer_path=tokenizer_dir))
        assert caught.value.code == "tokenizer_vocab_mismatch"
        assert caught.value.details["tokenizer_vocab_size"] == 32000

    def test_the_REAL_tokenizer_writers_output_reads_as_the_size_it_was_built_for(
        self, tokenizer_dir: Path
    ) -> None:
        """**The control that keeps the hand-written fixture honest.**

        ``write_tokenizer_json_only`` agrees with the ceiling reader by construction, which
        is this project's named reason a suite stays green over a real bug. So the ceiling
        reader is also run over :func:`tests.support.tiny_hf_model.write_tiny_tokenizer`'s
        output - the **real** ``PreTrainedTokenizerFast.save_pretrained`` - and required to
        compute the size that writer was asked for. If a ``tokenizers`` format change ever
        makes the vocabulary unreadable, this goes red instead of a run doing so.
        """
        pytest.importorskip("tokenizers")
        pytest.importorskip("transformers")
        from tests.support.tiny_hf_model import write_tiny_tokenizer
        from training.acquisition.scratch import tokenizer_vocab_ceiling

        real = tokenizer_dir.parent / "real"
        real.mkdir()
        write_tiny_tokenizer(real, vocab_size=256)
        assert tokenizer_vocab_ceiling(real) == 256


class TestTheHubSpellingOfTheTokenizerReference:
    """``tokenizer_repo`` mirrors a ``load``'s ``repo``/``revision`` pair, and reuses
    ``training.acquisition.hub``'s own two functions rather than making a second outbound
    Hub call - every Hub call in miForge is in that module (its own docstring), and a
    convenience here would break that for one field.

    The Hub is stubbed at those two seams, so these run in CI and **do not reach the
    network behind anyone's back**.
    """

    @pytest.fixture
    def stubbed_hub(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> dict[str, Any]:
        from tests.support.tiny_hf_model import write_tokenizer_json_only
        from training.acquisition import hub as hub_module

        snapshot = write_tokenizer_json_only(tmp_path / "snapshot", vocab_size=LLAMA["vocab_size"])
        seen: dict[str, Any] = {"resolved": [], "downloaded": [], "allow_patterns": []}

        def resolve_revision(repo: str, revision: str) -> str:
            seen["resolved"].append((repo, revision))
            return "c" * 40

        def download_snapshot(repo: str, commit: str, *, allow_patterns: Any = None) -> Path:
            seen["downloaded"].append((repo, commit))
            seen["allow_patterns"].append(allow_patterns)
            return snapshot

        monkeypatch.setattr(hub_module, "resolve_revision", resolve_revision)
        monkeypatch.setattr(hub_module, "download_snapshot", download_snapshot)
        return seen

    def test_the_revision_is_resolved_to_a_COMMIT_before_anything_is_fetched(
        self, stubbed_hub: dict[str, Any]
    ) -> None:
        """Same rule as the model half: a branch that moves between resolution and
        download must not change which tokenizer the checkpoint ships."""
        enriched = ScratchAcquirer().preflight(
            spec(tokenizer_repo="org/tok", tokenizer_revision="main")
        )
        assert stubbed_hub["resolved"] == [("org/tok", "main")]
        assert stubbed_hub["downloaded"] == [("org/tok", "c" * 40)]
        assert enriched.tokenizer_revision_resolved == "c" * 40
        assert enriched.tokenizer_revision == "main", "the request is never overwritten"

    def test_the_fetch_is_FILTERED_to_the_tokenizer_files(
        self, stubbed_hub: dict[str, Any]
    ) -> None:
        """A tokenizer named by a model repository must not pull its weights: the fetch
        asks for the tokenizer's files only. The patterns admit every at-the-root candidate
        and its ``tokenizer/`` twin (so the subdirectory refusal still has something to
        see), and no weight file."""
        from huggingface_hub.utils import filter_repo_objects

        from training.acquisition.local_dir import TOKENIZER_CANDIDATES
        from training.acquisition.scratch import TOKENIZER_FILE_PATTERNS

        ScratchAcquirer().preflight(spec(tokenizer_repo="org/model"))
        assert stubbed_hub["allow_patterns"] == [TOKENIZER_FILE_PATTERNS]

        listing = [
            *TOKENIZER_CANDIDATES,
            *(f"tokenizer/{name}" for name in TOKENIZER_CANDIDATES),
            "merges.txt",
            "model.safetensors",
            "model-00001-of-00004.safetensors",
            "model.safetensors.index.json",
            "pytorch_model.bin",
            "original/consolidated.00.pth",
            "flax_model.msgpack",
        ]
        admitted = set(filter_repo_objects(listing, allow_patterns=list(TOKENIZER_FILE_PATTERNS)))
        assert set(TOKENIZER_CANDIDATES) <= admitted
        assert {f"tokenizer/{name}" for name in TOKENIZER_CANDIDATES} <= admitted
        assert admitted.isdisjoint(
            {
                "model.safetensors",
                "model-00001-of-00004.safetensors",
                "model.safetensors.index.json",
                "pytorch_model.bin",
                "original/consolidated.00.pth",
                "flax_model.msgpack",
            }
        )

    def test_an_omitted_revision_uses_the_NAMED_default_rather_than_a_blank(
        self, stubbed_hub: dict[str, Any]
    ) -> None:
        ScratchAcquirer().preflight(spec(tokenizer_repo="org/tok"))
        assert stubbed_hub["resolved"] == [("org/tok", "main")]

    def test_the_record_names_the_repo_AT_THE_RESOLVED_COMMIT(
        self, monkeypatch: pytest.MonkeyPatch, stubbed_hub: dict[str, Any]
    ) -> None:
        """Not the requested branch: the commit is what the checkpoint's tokenizer came
        from, and recording the branch would claim a precision the request did not have."""
        from training.acquisition import scratch as scratch_module

        monkeypatch.setattr(
            scratch_module, "_instantiate", lambda parsed, seed, tok: (object(), _Tokenizer())
        )
        acquirer = ScratchAcquirer()
        enriched = acquirer.preflight(spec(tokenizer_repo="org/tok", tokenizer_revision="main"))
        record = acquirer.acquire(enriched).record
        assert record.tokenizer_source == f"hf:org/tok@{'c' * 40}"

    def test_the_VOCABULARY_gate_runs_on_a_fetched_snapshot(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A Hub tokenizer gets the vocabulary check, because both spellings reduce to
        *"a local directory"* and there is one implementation of it."""
        from tests.support.tiny_hf_model import write_tokenizer_json_only
        from training.acquisition import hub as hub_module

        big = write_tokenizer_json_only(tmp_path / "big-snapshot", vocab_size=32000)
        monkeypatch.setattr(hub_module, "resolve_revision", lambda repo, rev: "d" * 40)
        monkeypatch.setattr(hub_module, "download_snapshot", lambda repo, commit, **_: big)

        with pytest.raises(Refusal) as caught:
            ScratchAcquirer().preflight(spec(tokenizer_repo="org/big"))
        assert caught.value.code == "tokenizer_vocab_mismatch"

    def test_the_AT_THE_ROOT_gate_runs_on_a_fetched_snapshot_TOO(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """**A test finding from this round's sweep (M14).** Deleting
        ``assert_tokenizer_at_root`` from the Hub branch left the suite green: the case
        above asserts the *vocabulary* refusal, and ``assert_vocab_agrees`` runs separately
        in ``preflight``, so it fired either way.

        What was therefore unprotected: a Hub repo with **no tokenizer at its root** would
        be accepted at submit and then raise inside ``AutoTokenizer.from_pretrained`` -
        which is the late-crash class this whole mechanism exists to remove, reintroduced
        one spelling over.
        """
        from training.acquisition import hub as hub_module

        snapshot = tmp_path / "model-only-snapshot"
        snapshot.mkdir()
        (snapshot / "config.json").write_text("{}", encoding="utf-8")
        monkeypatch.setattr(hub_module, "resolve_revision", lambda repo, rev: "e" * 40)
        monkeypatch.setattr(hub_module, "download_snapshot", lambda repo, commit, **_: snapshot)

        with pytest.raises(Refusal) as caught:
            ScratchAcquirer().preflight(spec(tokenizer_repo="org/model-only"))
        assert caught.value.code == "tokenizer_source_incomplete"
        assert caught.value.details["missing"]

    def test_a_fetched_snapshot_with_a_tokenizer_SUBDIRECTORY_is_refused(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The other half of M14: the subdirectory case has its own message because the
        fix is a move rather than a download, and that distinction is as useful for a Hub
        snapshot as for a local directory."""
        from tests.support.tiny_hf_model import write_tokenizer_json_only
        from training.acquisition import hub as hub_module

        snapshot = tmp_path / "nested-snapshot"
        snapshot.mkdir()
        write_tokenizer_json_only(snapshot / "tokenizer", vocab_size=LLAMA["vocab_size"])
        monkeypatch.setattr(hub_module, "resolve_revision", lambda repo, rev: "f" * 40)
        monkeypatch.setattr(hub_module, "download_snapshot", lambda repo, commit, **_: snapshot)

        with pytest.raises(Refusal) as caught:
            ScratchAcquirer().preflight(spec(tokenizer_repo="org/nested"))
        assert caught.value.code == "tokenizer_not_at_root"
