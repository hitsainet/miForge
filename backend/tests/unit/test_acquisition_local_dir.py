"""Loading from a local directory, and every refusal it owns  (FR-001.3; E-2, E-5).

**Real directories with real files throughout.** The fixtures build a Hugging Face layout on
disk and the production code reads it: a mocked ``Path`` would prove the mock, and the whole
point of this acquirer is what it does with a directory it did not create.

The refusals are the valuable half. Each asserts the code **and** that the message names the
thing to fix - a missing file by name, a tokenizer in the wrong place, a path outside the
roots - because a refusal whose message does not say what to do fails ADR-013's intent even
when the code is right.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from core.config import Settings, reset_settings
from core.errors import Refusal
from tests.unit.test_acquisition_dtype_gate import write_safetensors
from training.acquisition.base import AcquisitionSpec
from training.acquisition.local_dir import (
    CONFIG_FILE,
    LocalDirAcquirer,
    param_count_from_headers,
    read_config,
    require_tokenizer_at_root,
    resolve_within_roots,
    weight_files,
)


@pytest.fixture
def rooted(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Point the configured model roots at ``tmp_path`` and return the checkpoint root."""
    checkpoints = tmp_path / "checkpoints"
    parts = tmp_path / "parts"
    checkpoints.mkdir()
    parts.mkdir()
    settings = Settings(
        _env_file=None,  # type: ignore[call-arg]
        checkpoint_root=checkpoints,
        parts_root=parts,
    )
    monkeypatch.setattr("core.config._settings", settings)
    yield checkpoints
    reset_settings()


def build_model_dir(
    root: Path,
    *,
    name: str = "model-000001",
    dtype: str = "F16",
    config: dict | None = None,
    tokenizer: str | None = "tokenizer.json",
    weights: bool = True,
) -> Path:
    """A standard Hugging Face directory: config, shards and the tokenizer **at the root**."""
    directory = root / name
    directory.mkdir(parents=True)
    (directory / CONFIG_FILE).write_text(
        json.dumps(config if config is not None else {"model_type": "llama"})
    )
    if weights:
        write_safetensors(
            directory / "model.safetensors",
            {"embed.weight": (dtype, [32, 8]), "layer.weight": (dtype, [8, 8])},
        )
    if tokenizer is not None:
        (directory / tokenizer).write_text("{}")
    return directory


class TestPathConfinement:
    """TDD §8: a submitted path is an input even on a LAN."""

    def test_a_path_inside_a_configured_root_resolves(self, rooted: Path) -> None:
        directory = build_model_dir(rooted)
        assert resolve_within_roots(directory) == directory.resolve()

    def test_a_path_outside_every_root_is_refused_naming_the_roots(
        self, rooted: Path, tmp_path: Path
    ) -> None:
        with pytest.raises(Refusal) as caught:
            resolve_within_roots(tmp_path / "elsewhere")
        assert caught.value.code == "model_source_outside_root"
        assert caught.value.details["roots"], "the refusal must name where models may live"

    def test_traversal_out_of_a_root_is_refused_after_resolution(self, rooted: Path) -> None:
        """Checked against the **resolved** path, so ``../..`` and a symlink out are both
        caught - a worker that holds the card must not read an arbitrary path."""
        with pytest.raises(Refusal) as caught:
            resolve_within_roots(rooted / ".." / ".." / "etc")
        assert caught.value.code == "model_source_outside_root"

    def test_a_symlink_pointing_out_of_a_root_is_refused(
        self, rooted: Path, tmp_path: Path
    ) -> None:
        outside = tmp_path / "outside"
        outside.mkdir()
        link = rooted / "sneaky"
        link.symlink_to(outside)
        with pytest.raises(Refusal) as caught:
            resolve_within_roots(link)
        assert caught.value.code == "model_source_outside_root"


class TestTheDirectoryMustBeComplete:
    """E-2: refuse, **naming the missing file**. No partial load, no fabricated default."""

    def test_a_missing_config_is_refused_by_name(self, rooted: Path) -> None:
        directory = rooted / "empty"
        directory.mkdir()
        with pytest.raises(Refusal) as caught:
            read_config(directory)
        assert caught.value.code == "model_source_incomplete"
        assert caught.value.details["missing"] == CONFIG_FILE

    def test_an_unparseable_config_names_the_file(self, rooted: Path) -> None:
        directory = rooted / "broken"
        directory.mkdir()
        (directory / CONFIG_FILE).write_text("{not json")
        with pytest.raises(Refusal) as caught:
            read_config(directory)
        assert caught.value.code == "architecture_config_invalid"
        assert CONFIG_FILE in caught.value.details["path"]

    def test_a_missing_weight_file_is_refused_by_name(self, rooted: Path) -> None:
        directory = build_model_dir(rooted, weights=False)
        with pytest.raises(Refusal) as caught:
            weight_files(directory)
        assert caught.value.code == "model_source_incomplete"
        assert ".safetensors" in caught.value.details["missing"]

    def test_a_pickled_weight_file_is_refused_with_the_reason(self, rooted: Path) -> None:
        """Not a gap in support - a **refusal with a reason**: a pickled file's dtypes
        cannot be checked without unpickling it, which executes arbitrary code in the
        training worker and allocates before the budget guard has run."""
        directory = build_model_dir(rooted, weights=False)
        (directory / "pytorch_model.bin").write_bytes(b"\x80\x04")
        with pytest.raises(Refusal) as caught:
            weight_files(directory)
        assert caught.value.code == "model_weights_format_unsupported"
        assert "safetensors" in caught.value.message
        assert "pytorch_model.bin" in caught.value.details["found"]


class TestTheTokenizerIsAtTheRoot:
    """ADR-015, R-02.51 - the project's own historical defect, in the reader half."""

    @pytest.mark.parametrize(
        "name", ["tokenizer.json", "tokenizer_config.json", "tokenizer.model", "vocab.json"]
    )
    def test_any_standard_tokenizer_file_at_the_root_satisfies_it(
        self, name: str, rooted: Path
    ) -> None:
        """Several forms load perfectly well, so requiring one exact name would refuse
        models that work."""
        require_tokenizer_at_root(build_model_dir(rooted, tokenizer=name))

    def test_a_tokenizer_SUBDIRECTORY_is_refused_with_its_own_message(self, rooted: Path) -> None:
        """**The defect R-02.51 records.** As originally specced a checkpoint put the
        tokenizer in ``tokenizer/``, and such a directory cannot be loaded by
        ``transformers``, vLLM, TGI or miLLM's own loader - it broke R-02.21 and acceptance
        criterion 2 before anything was built.

        Its own code, because the fix is a *move*: telling the operator "tokenizer missing"
        when the files are one level down sends them looking for the wrong thing.
        """
        directory = build_model_dir(rooted, tokenizer=None)
        nested = directory / "tokenizer"
        nested.mkdir()
        (nested / "tokenizer.json").write_text("{}")
        with pytest.raises(Refusal) as caught:
            require_tokenizer_at_root(directory)
        assert caught.value.code == "model_tokenizer_not_at_root"
        assert "root" in caught.value.message
        assert caught.value.details["found_in"].endswith("tokenizer")

    def test_an_absent_tokenizer_is_refused_and_says_miForge_does_not_train_one(
        self, rooted: Path
    ) -> None:
        directory = build_model_dir(rooted, tokenizer=None)
        with pytest.raises(Refusal) as caught:
            require_tokenizer_at_root(directory)
        assert caught.value.code == "model_source_incomplete"
        assert "does not train a tokenizer" in caught.value.message


class TestTheParameterCountComesFromTheHeaders:
    def test_shapes_are_multiplied_across_every_shard(self, rooted: Path) -> None:
        """**From the headers, not from a loaded model.** ADR-008 requires the budget check
        to precede allocation, so counting a loaded model's parameters would invert the
        ordering R-02.42 exists to establish."""
        directory = build_model_dir(rooted)
        assert param_count_from_headers(weight_files(directory)) == 32 * 8 + 8 * 8

    def test_scalar_buffers_are_not_counted_as_parameters(self, rooted: Path) -> None:
        directory = rooted / "with-scalar"
        directory.mkdir()
        (directory / CONFIG_FILE).write_text(json.dumps({"model_type": "llama"}))
        path = write_safetensors(directory / "model.safetensors", {"w": ("F16", [4, 4])})
        assert param_count_from_headers([path]) == 16


class TestPreflightDoesTheWholeGateAndAllocatesNothing:
    def test_a_good_directory_preflights_and_yields_a_parameter_count(self, rooted: Path) -> None:
        """The count is what the budget guard reads, so preflight must produce it."""
        directory = build_model_dir(rooted)
        enriched = LocalDirAcquirer().preflight(AcquisitionSpec(path=directory))
        assert enriched.param_count == 32 * 8 + 8 * 8
        assert enriched.path == directory.resolve()
        assert (
            enriched.architecture_config is None
        ), "a load supplies a count, not a config: the budget guard takes exactly one"

    def test_quantized_weights_are_refused_at_preflight(self, rooted: Path) -> None:
        """AC-001.4: refused **before any VRAM is allocated**, which is what preflight
        being a separate method from ``acquire`` buys."""
        directory = build_model_dir(rooted, dtype="I8")
        with pytest.raises(Refusal) as caught:
            LocalDirAcquirer().preflight(AcquisitionSpec(path=directory))
        assert caught.value.code == "model_quantized"
        assert "FP16 or BF16" in caught.value.message

    def test_a_quantization_block_is_refused_at_preflight(self, rooted: Path) -> None:
        directory = build_model_dir(
            rooted, config={"model_type": "llama", "quantization_config": {"bits": 4}}
        )
        with pytest.raises(Refusal) as caught:
            LocalDirAcquirer().preflight(AcquisitionSpec(path=directory))
        assert caught.value.code == "model_quantized"

    def test_a_tokenizer_SUBDIRECTORY_is_refused_AT_PREFLIGHT(self, rooted: Path) -> None:
        """**A test finding from the mutation sweep (M25), written up as a regression.**

        ``require_tokenizer_at_root`` was fully unit-tested above and **nothing asserted it
        was wired into ``preflight``**: deleting the call from the acquirer left the whole
        suite green. That is the weaker form of ADR-016 #1 exactly as CLAUDE.md describes it
        - *a grep proves a call exists, not that anything depends on it* - and the
        consequence is the project's own recurring defect: a checkpoint whose tokenizer sits
        in ``tokenizer/`` cannot be loaded by ``transformers``, vLLM, TGI or miLLM's own
        loader, which breaks R-02.51 and acceptance criterion 2.

        So this asserts the **refusal through the acquirer**, not through the helper.
        """
        directory = build_model_dir(rooted, tokenizer=None)
        nested = directory / "tokenizer"
        nested.mkdir()
        (nested / "tokenizer.json").write_text("{}")

        with pytest.raises(Refusal) as caught:
            LocalDirAcquirer().preflight(AcquisitionSpec(path=directory))
        assert caught.value.code == "model_tokenizer_not_at_root"

    def test_an_absent_tokenizer_is_refused_AT_PREFLIGHT(self, rooted: Path) -> None:
        """The other half of the same wiring, so the guard cannot be narrowed to one case."""
        directory = build_model_dir(rooted, tokenizer=None)
        with pytest.raises(Refusal) as caught:
            LocalDirAcquirer().preflight(AcquisitionSpec(path=directory))
        assert caught.value.code == "model_source_incomplete"
        assert "does not train a tokenizer" in caught.value.message

    def test_preflight_runs_the_tokenizer_check_once_on_the_resolved_directory(
        self, rooted: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The payload and the call count, not "was called" (ADR-016 #1).

        A check invoked on the *submitted* path rather than the resolved one would pass this
        suite while confining nothing, which is why the argument is asserted too.
        """
        from training.acquisition import local_dir

        seen: list[Path] = []
        monkeypatch.setattr(local_dir, "require_tokenizer_at_root", seen.append)
        directory = build_model_dir(rooted)
        LocalDirAcquirer().preflight(AcquisitionSpec(path=directory))
        assert seen == [directory.resolve()]

    def test_a_missing_path_refuses_rather_than_defaulting(self) -> None:
        spec = AcquisitionSpec(mode="load", path=Path("/tmp"))
        with pytest.raises(Refusal) as caught:
            LocalDirAcquirer().preflight(spec.model_copy(update={"path": None}))
        assert caught.value.code == "model_source_missing"
        assert "did not name" in caught.value.message

    def test_a_file_where_a_directory_belongs_is_refused(self, rooted: Path) -> None:
        notdir = rooted / "a-file"
        notdir.write_text("")
        with pytest.raises(Refusal) as caught:
            LocalDirAcquirer().preflight(AcquisitionSpec(path=notdir))
        assert caught.value.code == "model_source_incomplete"

    def test_acquire_without_preflight_refuses_rather_than_crashing(self) -> None:
        """``acquire`` is a public method, so a caller that skipped preflight is told so."""
        spec = AcquisitionSpec(path=Path("/tmp")).model_copy(update={"path": None})
        with pytest.raises(Refusal) as caught:
            LocalDirAcquirer().acquire(spec)
        assert caught.value.code == "acquisition_preflight_skipped"


class TestRemoteCodeIsOff:
    def test_from_pretrained_is_called_with_trust_remote_code_false(
        self, rooted: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """**Asserted rather than trusted to the library default**, which has changed before.

        A remote-code architecture would execute arbitrary Python inside the worker that
        holds the GPU lease. The payload is asserted, not merely that the call happened - a
        test that only checked invocation would pass against a call sending the wrong
        arguments.
        """
        from training.acquisition import local_dir

        calls: list[dict] = []

        class FakeAuto:
            @staticmethod
            def from_pretrained(path, **kwargs):
                calls.append({"path": str(path), **kwargs})
                return object()

        class FakeTorch:
            float16 = "float16"
            bfloat16 = "bfloat16"

        monkeypatch.setattr(local_dir, "_ml_imports", lambda: (FakeTorch, FakeAuto, FakeAuto))
        directory = build_model_dir(rooted)
        local_dir._load_pretrained(directory, "F16")

        assert len(calls) == 2, f"expected a model load and a tokenizer load, got {calls}"
        for call in calls:
            assert call["trust_remote_code"] is False, (
                "trust_remote_code must be passed explicitly as False: a Hub config could "
                "otherwise run arbitrary Python in the worker holding the card"
            )
        # The tokenizer is read FROM THE ROOT. A `tokenizer/` subdirectory is the defect
        # R-02.51 records, and this is the assertion that the loader does not look there.
        assert {call["path"] for call in calls} == {str(directory)}
        assert calls[0]["dtype"] == "float16"
        assert "torch_dtype" not in calls[0], "transformers 5 deprecated torch_dtype for dtype"

    def test_the_REAL_loader_honours_the_dtype_it_is_given(self, tmp_path: Path) -> None:
        """Through the installed transformers, not a fake: the fake above records whatever
        keyword it is handed, so it cannot tell a renamed argument from a working one. The
        tiny model is SAVED in bf16 and loaded asking for F16 - transformers 5's default is
        the config's own dtype, so an ignored or misspelled keyword reads back bf16."""
        pytest.importorskip("transformers")
        import torch

        from tests.support.tiny_hf_model import write_tiny_model
        from training.acquisition import local_dir

        directory = write_tiny_model(tmp_path / "tiny")
        module, tokenizer = local_dir._load_pretrained(directory, "F16")
        assert {p.dtype for p in module.parameters()} == {torch.float16}
        assert tokenizer("t5 t6")["input_ids"] == [5, 6]
