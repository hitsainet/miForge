"""The adapter write: three files, one digest, the agreement, and the atomicity.

Everything here runs with **no card and no ML stack**: ``peft`` is faked at the one place it
touches the filesystem (``save_pretrained``), and the atomicity, the streamed digest, the
PEFT/manifest agreement check and schema validation are the real code.

**M5's, M15's and M13's targets.**
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from core.canonical import sha256_of_file
from core.errors import Refusal
from models.domain_state import ArtifactKind
from services.reindex import schema_violations
from tests.support.checkpoint_faults import observe_syscalls
from tests.support.test_adapter_doubles import (
    FakePeftModel,
    adapter_config,
    hub_record,
    optimizer_block,
)
from training.runtypes.adapters import manifest as adapter_manifest
from training.runtypes.adapters import paths as adapter_paths
from training.runtypes.adapters import save as adapter_save

HUB_REPO = "LiquidAI/LFM2.5-1.2B-Instruct"


def document(step: int = 40, **overrides: Any) -> dict[str, Any]:
    values: dict[str, Any] = {
        "run_id": "run-7",
        "step": step,
        "base": adapter_manifest.base_block(hub_record(HUB_REPO)),
        "adapter": adapter_config(rank=8, alpha=16.0, target_modules=("q_proj",)),
        "optimizer": optimizer_block(),
    }
    values.update(overrides)
    return adapter_manifest.build(**values)


def model(**overrides: Any) -> FakePeftModel:
    defaults: dict[str, Any] = {"rank": 8, "alpha": 16.0, "target_modules": ("q_proj",)}
    defaults.update(overrides)
    return FakePeftModel(**defaults)


@pytest.fixture
def dest(tmp_path: Path) -> Path:
    return adapter_paths.adapter_dir("run-7", 40, root=tmp_path)


class TestAPublishedAdapterIsAStandardPeftDirectoryPlusOneFile:
    def test_all_THREE_files_are_present(self, dest: Path) -> None:
        """``adapter_config.json`` is the one the normative layout prose omits and
        ``PeftConfig.from_pretrained`` reads - FPRD D5, the same shape as the
        tokenizer-subdirectory defect."""
        adapter_save.save(model(), document(), dest)
        assert sorted(p.name for p in dest.iterdir()) == sorted(
            [
                adapter_paths.WEIGHTS_NAME,
                adapter_paths.PEFT_CONFIG_NAME,
                adapter_paths.MANIFEST_NAME,
            ]
        )

    def test_the_manifest_is_the_only_non_standard_file_and_is_additive(self, dest: Path) -> None:
        published = adapter_save.save(model(), document(), dest)
        peft_config = json.loads((dest / adapter_paths.PEFT_CONFIG_NAME).read_text())
        assert peft_config["peft_type"] == "LORA"
        assert published["kind"] == "miforge.adapter-manifest"

    def test_the_published_manifest_validates_against_the_real_schema(self, dest: Path) -> None:
        adapter_save.save(model(), document(), dest)
        on_disk = json.loads((dest / adapter_paths.MANIFEST_NAME).read_text())
        assert schema_violations(on_disk, ArtifactKind.ADAPTER) == []

    def test_latest_points_at_the_step_directory(self, dest: Path) -> None:
        adapter_save.save(model(), document(), dest)
        latest = dest.parent / "latest"
        assert latest.is_symlink()
        # Relative, so a copied tree keeps its meaning (ADR-004).
        assert Path(latest.readlink()) == Path("step-000040")


class TestTheDigestIsOfTheFileOnDisk:
    """M5's target. BRD-01 R-01.27 verifies an adapter's hash the way R-01.14 verifies a
    checkpoint's, and until the field existed that verification had nothing to read."""

    def test_weights_sha256_matches_a_streamed_hash_of_the_written_file(self, dest: Path) -> None:
        published = adapter_save.save(model(), document(), dest)
        assert published["weights_sha256"] == sha256_of_file(dest / adapter_paths.WEIGHTS_NAME)

    def test_two_different_adapters_get_two_different_digests(self, tmp_path: Path) -> None:
        """A digest that did not depend on the bytes would pass every assertion above."""
        first = adapter_save.save(
            model(payload=b"one"), document(40), adapter_paths.adapter_dir("r", 40, root=tmp_path)
        )
        second = adapter_save.save(
            model(payload=b"two"), document(41), adapter_paths.adapter_dir("r", 41, root=tmp_path)
        )
        assert first["weights_sha256"] != second["weights_sha256"]

    def test_a_manifest_WITHOUT_the_digest_fails_validation(self) -> None:
        """So dropping the line that sets it cannot ship: the schema refuses the document
        and the write fails rather than publishing an unverifiable adapter."""
        doc = document()
        assert "weights_sha256" not in doc
        with pytest.raises(Refusal) as caught:
            adapter_save.adapter_schema.validate(doc)
        assert caught.value.code == "adapter_manifest_invalid"

    def test_a_non_hex_digest_fails_validation(self) -> None:
        doc = document()
        doc["weights_sha256"] = "not-a-digest"
        with pytest.raises(Refusal):
            adapter_save.adapter_schema.validate(doc)


class TestTheTwoFilesMustAgree:
    """The defect **neither schema can see**: miLLM's Feature 033 loader cross-checks
    ``r``/``lora_alpha`` against ``rank``/``alpha`` and refuses on a mismatch, so a
    disagreeing directory validates against both schemas, loads under plain ``peft``, and
    fails acceptance criterion 15."""

    @pytest.mark.parametrize(
        ("override", "field"),
        [({"r": 16}, "rank"), ({"lora_alpha": 32.0}, "alpha")],
    )
    def test_a_disagreement_is_refused_and_nothing_is_published(
        self, dest: Path, override: dict[str, Any], field: str
    ) -> None:
        with pytest.raises(Refusal) as caught:
            adapter_save.save(model(config_overrides=override), document(), dest)
        assert caught.value.code == "adapter_config_manifest_disagree"
        assert field in (caught.value.details or {})["disagreements"]
        assert not dest.exists(), "a refused write publishes nothing"

    def test_a_disagreeing_directory_WOULD_HAVE_validated_against_both_schemas(
        self, tmp_path: Path
    ) -> None:
        """The reason the check has to exist in code: proof that neither contract catches it.

        The manifest with ``rank: 8`` validates; ``adapter_config.json`` is PEFT's own file
        and this project vendors no schema for it. So a writer deriving the two numbers
        separately produces an artifact both contracts accept and miLLM refuses.
        """
        doc = document()
        doc["weights_sha256"] = "a" * 64
        assert schema_violations(doc, ArtifactKind.ADAPTER) == []
        staging = tmp_path / "staging"
        staging.mkdir()
        model(config_overrides={"r": 16}).save_pretrained(staging)
        peft_config = json.loads((staging / adapter_paths.PEFT_CONFIG_NAME).read_text())
        assert peft_config["r"] != doc["adapter"]["rank"]

    def test_an_agreeing_pair_passes(self, tmp_path: Path) -> None:
        staging = tmp_path / "ok"
        staging.mkdir()
        doc = document()
        doc["weights_sha256"] = "a" * 64
        model().save_pretrained(staging)
        assert adapter_save.assert_config_agrees(staging, doc)["r"] == 8


class TestTheMissingPeftConfigIsRefused:
    """M15's target: dropping ``adapter_config.json`` from the writer."""

    def test_a_directory_without_the_peft_config_is_refused(self, dest: Path) -> None:
        with pytest.raises(Refusal) as caught:
            adapter_save.save(model(write_peft_config=False), document(), dest)
        assert caught.value.code == "adapter_config_missing"
        assert "PeftConfig.from_pretrained" in caught.value.message
        assert not dest.exists()

    def test_a_save_that_writes_no_tensors_is_refused(self, dest: Path) -> None:
        class NoTensors:
            def save_pretrained(self, directory: Any) -> None:
                Path(directory, adapter_paths.PEFT_CONFIG_NAME).write_text("{}")

        with pytest.raises(Refusal) as caught:
            adapter_save.save(NoTensors(), document(), dest)
        assert caught.value.code == "adapter_weights_not_written"
        assert not dest.exists()


class TestTheWriteIsAtomicAndValidationPrecedesTheRename:
    """M13's shape, applied to the adapter. A directory write cannot tear - which is what
    replaces the torn-weights apparatus for this artifact (ADR-015)."""

    def test_a_refusal_inside_the_block_leaves_NO_discoverable_directory(self, dest: Path) -> None:
        with pytest.raises(Refusal):
            adapter_save.save(model(config_overrides={"r": 99}), document(), dest)
        assert not dest.exists()
        assert not (dest.parent / "latest").exists()
        leftovers = [p.name for p in dest.parent.iterdir()] if dest.parent.exists() else []
        assert leftovers == [], f"the staging directory must be removed; found {leftovers}"

    def test_a_failed_save_leaves_an_EARLIER_adapter_and_its_latest_untouched(
        self, tmp_path: Path
    ) -> None:
        good = adapter_paths.adapter_dir("run-7", 40, root=tmp_path)
        adapter_save.save(model(), document(40), good)
        before = (good.parent / "latest").readlink()

        bad = adapter_paths.adapter_dir("run-7", 41, root=tmp_path)
        with pytest.raises(Refusal):
            adapter_save.save(model(write_peft_config=False), document(41), bad)

        assert not bad.exists()
        assert (good.parent / "latest").readlink() == before
        assert (good / adapter_paths.MANIFEST_NAME).is_file()

    def test_the_syscall_ORDER_is_fsync_files_then_fsync_dir_then_rename(
        self, dest: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Asserted through **F5's** syscall observer, not a second one. The outcome is
        identical under every ordering until the power fails, which is exactly why the
        order is what is asserted."""
        with observe_syscalls(monkeypatch) as log:
            adapter_save.save(model(), document(), dest)
        kinds = [kind for kind, _ in log.calls]
        assert "fsync_file" in kinds
        first_dir = kinds.index("fsync_dir")
        rename = kinds.index("rename")
        assert kinds.index("fsync_file") < first_dir < rename
        # The `latest` swap is an ``os.replace`` and comes AFTER the rename. The *last*
        # replace, not the first: ``core.canonical.write_canonical`` publishes the manifest
        # inside the staging directory with a temp-file-then-replace of its own, so an
        # earlier replace is the manifest's and not the pointer's. Asserting the first one
        # would have been an assertion about the wrong syscall that happened to be true.
        assert rename < max(index for index, kind in enumerate(kinds) if kind == "replace")

    def test_overwriting_a_published_adapter_is_refused(self, dest: Path) -> None:
        """The step is part of the identity, so an overwrite means two different adapters
        claiming one name. The primitive's refusal, not a second rule."""
        adapter_save.save(model(), document(), dest)
        with pytest.raises(Refusal) as caught:
            adapter_save.save(model(payload=b"different"), document(), dest)
        assert caught.value.code == "artifact_already_exists"


class TestF8OwnsNoAtomicityImplementationOfItsOwn:
    """Task 9.4a's ratchet, allowlist **empty**. *Two implementations of this guarantee are
    two answers*, and an F8-local temp-dir dance would be the third."""

    def test_no_f8_module_calls_rename_replace_or_fsync(self) -> None:
        import ast

        package = Path(__file__).resolve().parents[3] / "backend/src/training/runtypes"
        offenders: list[str] = []
        for path in sorted(package.rglob("*.py")):
            tree = ast.parse(path.read_text(encoding="utf-8"))
            for node in ast.walk(tree):
                if (
                    isinstance(node, ast.Attribute)
                    and node.attr
                    in {
                        "rename",
                        "replace",
                        "fsync",
                        "symlink",
                    }
                    and isinstance(node.value, ast.Name)
                    and node.value.id in {"os", "shutil"}
                ):
                    offenders.append(f"{path.name}:{node.lineno}:{node.attr}")
        assert not offenders, f"F8 must call core.atomic_dir, not reimplement it: {offenders}"

    def test_the_saver_uses_the_shared_primitive(self) -> None:
        source = (
            Path(__file__).resolve().parents[3] / "backend/src/training/runtypes/adapters/save.py"
        ).read_text(encoding="utf-8")
        assert "from core.atomic_dir import atomic_dir" in source
        assert "with atomic_dir(dest)" in source
