"""The output contract, as a file set  (FR-005.2, 11, 19-23; AC-005.1, AC-005.14).

R-02.51's promise is that a checkpoint loads in software that never heard of miForge. That
promise is kept or broken by **which files are in the directory**, so the assertions here
are over the file set rather than over the writer's intentions:

* an **equality**, not a subset - a subset assertion passes against a directory carrying an
  extra miForge file, and against optimizer state, which breaks criterion 23 without
  breaking anything a test of the save path would notice;
* **the tokenizer's files at the root** - as first specified a ``tokenizer/`` subdirectory
  could not be loaded by ``transformers``, vLLM, TGI or miLLM's own loader, and it broke
  R-02.21 and criterion 2 before any code existed. This test is what stops a later "tidy";
* **``manifest.json`` last**, after the hash, so a manifest never describes files that are
  not there yet;
* **no absolute path, hostname, URL or credential** anywhere in the written manifest -
  scanned, not inspected (AC-005.14, ADR-020).
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import pytest

from core.atomic_dir import LATEST
from core.canonical import sha256_of_file
from core.errors import Refusal
from services.checkpoints import writer as writer_module
from services.checkpoints.paths import checkpoint_dir
from tests.support.checkpoint_faults import observe_syscalls
from tests.support.checkpoints import TinyModel, TinyTokenizer, fields, saved_checkpoint


class TestTheDirectoryIsAStandardHuggingFaceModelDirectory:
    def test_the_file_set_is_exactly_the_hf_files_plus_the_manifest(self, tmp_path: Path):
        result = saved_checkpoint(tmp_path)
        assert sorted(p.name for p in result.directory.iterdir()) == [
            "config.json",
            "manifest.json",
            "model.safetensors",
            "special_tokens_map.json",
            "tokenizer.json",
            "tokenizer_config.json",
        ]

    def test_tokenizer_files_are_at_the_root_and_a_subdirectory_is_refused(self, tmp_path: Path):
        """The defect that broke the output contract before any code existed."""
        with pytest.raises(Refusal) as caught:
            saved_checkpoint(tmp_path, tokenizer=TinyTokenizer(subdirectory="tokenizer"))
        assert caught.value.code == "checkpoint_contains_subdirectory"
        assert "tokenizer" in caught.value.details["entry"]

    def test_nothing_is_published_when_the_file_set_is_refused(self, tmp_path: Path):
        """The refusal happens inside the atomic block, so the destination never appears."""
        with pytest.raises(Refusal):
            saved_checkpoint(tmp_path, tokenizer=TinyTokenizer(subdirectory="tokenizer"))
        assert not checkpoint_dir("run-a", 500, root=tmp_path).exists()
        assert list((tmp_path / "run-a").iterdir()) == []

    def test_trainer_state_inside_a_checkpoint_is_refused_by_its_own_name(self, tmp_path: Path):
        """A checkpoint that gains optimizer tensors stops being a standard HF directory,
        which breaks criterion 23 **without breaking anything the save path would notice** -
        hence a named refusal rather than a generic "unexpected file"."""
        directory = tmp_path / "ckpt"
        directory.mkdir()
        for name in ("model.safetensors", "config.json", "manifest.json", "optimizer.pt"):
            (directory / name).write_text("x")
        with pytest.raises(Refusal) as caught:
            writer_module.assert_file_set(directory)
        assert caught.value.code == "checkpoint_trainer_state_inside"

    def test_an_unexpected_file_is_refused(self, tmp_path: Path):
        """``manifest.json`` is the ONE non-standard file; a second makes the artifact
        something only miForge understands."""
        directory = tmp_path / "ckpt"
        directory.mkdir()
        for name in ("model.safetensors", "config.json", "manifest.json", "miforge-notes.txt"):
            (directory / name).write_text("x")
        with pytest.raises(Refusal) as caught:
            writer_module.assert_file_set(directory)
        assert caught.value.code == "checkpoint_file_set_unexpected"
        assert caught.value.details["unexpected"] == ["miforge-notes.txt"]

    def test_a_sharded_weights_file_is_accepted(self, tmp_path: Path):
        """``save_pretrained`` shards a large model; the allowlist has to know that or a
        real save of a real model would be refused by its own guard."""
        directory = tmp_path / "ckpt"
        directory.mkdir()
        for name in (
            "model.safetensors",
            "model-00001-of-00002.safetensors",
            "model.safetensors.index.json",
            "config.json",
            "manifest.json",
        ):
            (directory / name).write_text("x")
        assert "model-00001-of-00002.safetensors" in writer_module.assert_file_set(directory)

    def test_safe_serialization_is_passed_not_defaulted(self, tmp_path: Path):
        """``TinyModel`` asserts it: a ``pytorch_model.bin`` is a pickle, and R-02.51
        promises an artifact a consumer can load without executing it."""
        saved_checkpoint(tmp_path)  # the fixture's assertion is the test


class TestTheHashAndTheManifestsPlaceInTheOrder:
    def test_weights_sha256_is_the_digest_of_the_written_file(self, tmp_path: Path):
        result = saved_checkpoint(tmp_path, payload=b"distinct-bytes")
        assert result.manifest["weights_sha256"] == sha256_of_file(
            result.directory / "model.safetensors"
        )

    def test_the_manifest_is_written_after_the_weights_it_describes(self, tmp_path: Path):
        """Asserted through the *content*: the manifest names a hash of bytes that are
        already on disk, which is only possible if it was written afterwards."""
        result = saved_checkpoint(tmp_path, payload=b"abc")
        doc = json.loads((result.directory / "manifest.json").read_text())
        assert doc["weights_sha256"] == sha256_of_file(result.directory / "model.safetensors")

    def test_two_different_parameter_sets_hash_differently(self, tmp_path: Path):
        """The premise of the resave rule: without this, "the same step with different
        weights" would be undetectable."""
        a = saved_checkpoint(tmp_path, run_id="r1", payload=b"one")
        b = saved_checkpoint(tmp_path, run_id="r2", payload=b"two")
        assert a.manifest["weights_sha256"] != b.manifest["weights_sha256"]


class TestTheCommitOrderHoldsThroughTheWriter:
    def test_every_file_and_the_directory_are_fsynced_before_the_rename(
        self, tmp_path: Path, monkeypatch
    ):
        """**MUT-1 and MUT-2's target, through the composed path.** The outcome is
        identical under every ordering until the power fails, so the *observed* order is
        what is asserted - a result-based test passes against a writer that renames first.

        The foundation's own test covers the primitive; this one covers the writer stack
        above it, because an ordering guarantee that holds in isolation and not in
        composition is not a guarantee."""
        with observe_syscalls(monkeypatch) as calls:
            result = saved_checkpoint(tmp_path)

        rename_at = calls.first_index("rename")
        assert rename_at >= 0, "nothing was renamed, so nothing was published atomically"
        file_syncs = [i for i, (name, _) in enumerate(calls.calls) if name == "fsync_file"]
        dir_syncs = [i for i, (name, _) in enumerate(calls.calls) if name == "fsync_dir"]
        assert file_syncs and max(file_syncs) < rename_at, "a file was fsynced after the rename"
        assert [i for i in dir_syncs if i < rename_at], (
            "the temp DIRECTORY was never fsynced before the rename - the mutation no "
            "outcome-based test can catch, because the directory entry is visible either "
            "way and only an untimely power loss tells the difference"
        )
        assert result.directory.is_dir()

    def test_latest_is_swapped_after_the_rename(self, tmp_path: Path, monkeypatch):
        with observe_syscalls(monkeypatch) as calls:
            saved_checkpoint(tmp_path)
        # Matched on the *target*, not on "the first replace": `write_canonical` uses
        # `os.replace` for manifest.json inside the staging directory, so a positional
        # assertion would be asserting something else entirely and would stay green when
        # `latest` moved first.
        swaps = [
            i
            for i, (name, target) in enumerate(calls.calls)
            if name == "replace" and target.endswith(f"/{LATEST}")
        ]
        assert swaps, "`latest` was never swapped"
        assert (
            calls.first_index("rename") < swaps[0]
        ), "`latest` moved before the directory it points at existed"


class TestLatestIsRelativeSoACopyKeepsItsMeaning:
    def test_latest_resolves_to_the_published_step(self, tmp_path: Path):
        saved_checkpoint(tmp_path, step=500)
        assert (tmp_path / "run-a" / LATEST).readlink() == Path("step-000500")

    def test_latest_survives_a_directory_copy(self, tmp_path: Path):
        """**MUT-6's target.** ADR-004 promises *a checkpoint copied to another machine
        keeps its meaning*. An absolute symlink breaks on the copy and - worse - silently
        resolves to the **source** host's path if that path happens to exist, which is a
        quietly wrong answer rather than a visible failure."""
        saved_checkpoint(tmp_path, step=500)
        elsewhere = tmp_path / "elsewhere"
        shutil.copytree(tmp_path / "run-a", elsewhere, symlinks=True)

        link = elsewhere / LATEST
        assert link.is_symlink()
        assert (
            not link.readlink().is_absolute()
        ), "an absolute `latest` points back at the machine the checkpoint was written on"
        assert link.resolve() == (elsewhere / "step-000500").resolve()
        assert (link / "manifest.json").is_file()


class TestVerification:
    def test_verify_returns_the_digest_when_it_matches(self, tmp_path: Path):
        result = saved_checkpoint(tmp_path)
        assert (
            writer_module.verify_weights(result.directory, result.manifest["weights_sha256"])
            == result.manifest["weights_sha256"]
        )

    def test_a_mismatch_refuses_naming_both_digests_and_changes_nothing(self, tmp_path: Path):
        result = saved_checkpoint(tmp_path)
        before = (result.directory / "manifest.json").read_bytes()
        (result.directory / "model.safetensors").write_bytes(b"tampered")

        with pytest.raises(Refusal) as caught:
            writer_module.verify_weights(result.directory, result.manifest["weights_sha256"])
        assert caught.value.code == "checkpoint_hash_mismatch"
        assert caught.value.details["expected"] != caught.value.details["actual"]
        assert (result.directory / "manifest.json").read_bytes() == before, (
            "the manifest was repaired to match the bytes it found - which destroys the "
            "only record of what was actually written (FR-005.27)"
        )

    def test_missing_weights_are_checkpoint_not_found_rather_than_an_empty_hash(
        self, tmp_path: Path
    ):
        result = saved_checkpoint(tmp_path)
        (result.directory / "model.safetensors").unlink()
        with pytest.raises(Refusal) as caught:
            writer_module.verify_weights(result.directory, result.manifest["weights_sha256"])
        assert caught.value.code == "checkpoint_not_found"


class TestNothingSensitiveReachesTheArtifact:
    def test_a_written_manifest_contains_no_path_host_url_or_credential(self, tmp_path: Path):
        """**AC-005.14, by scanning the written file rather than by inspection.** A manifest
        that named the machine it was written on would stop being portable the moment it was
        copied, and ADR-020 forbids a secret in a manifest at all."""
        result = saved_checkpoint(tmp_path, test=None, retention=None)
        doc = json.loads((result.directory / "manifest.json").read_text())
        assert writer_module.contains_secret_shaped_values(doc) == []

    def test_the_scan_can_actually_fail(self, tmp_path: Path):
        """The negative control for the control: a scanner that always returns ``[]``
        would make the assertion above meaningless."""
        assert writer_module.contains_secret_shaped_values(
            {"base_model": "/data/checkpoints/run-a", "x": {"api_key": "s3cret"}}
        )
        assert writer_module.contains_secret_shaped_values({"a": "http://millm.internal/v1"})


class TestTheWriterRefusesToOverwriteAPublishedArtifact:
    def test_a_second_write_at_the_same_step_refuses_through_the_primitive(self, tmp_path: Path):
        """The step is part of the identity, so an overwrite means two parameter sets
        claiming one name. The *decision* about what to do instead is the service's
        (FR-005.6) - the writer is past the point of deciding."""
        saved_checkpoint(tmp_path)
        with pytest.raises(Refusal) as caught:
            writer_module.write(
                TinyModel(payload=b"different"),
                TinyTokenizer(),
                dest=checkpoint_dir("run-a", 500, root=tmp_path),
                fields=fields(),
            )
        assert caught.value.code == "artifact_already_exists"
