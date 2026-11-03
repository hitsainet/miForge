"""AC17: a merge is refused when no checkpoint of the unmerged base exists.

**M7's target.** A merge **cannot be undone by a second merge** - once the base has moved
the adapter's contribution cannot be subtracted back out - so the refusal is the whole point
of the operation, and it runs **before any tensor arithmetic**, not before the write.

The strict, fail-closed reading of FPRD D15 is what is implemented: a base loaded from a Hub
repo at a pinned revision is still *obtainable* but is **not a checkpoint**, and the refusal
does not admit it. The alternative is recorded for the operator, not taken here.
"""

from __future__ import annotations

import inspect
from pathlib import Path
from typing import Any

import pytest

from core.canonical import sha256_of, write_canonical
from core.errors import Refusal
from services.checkpoints import manifest as checkpoint_manifest
from tests.support.checkpoints import saved_checkpoint
from tests.support.test_adapter_doubles import (
    adapter_config,
    checkpoint_record,
    hub_record,
    optimizer_block,
)
from training.runtypes.adapters import manifest as adapter_manifest
from training.runtypes.adapters import merge as merge_module
from training.runtypes.adapters import paths as adapter_paths

HUB_REPO = "LiquidAI/LFM2.5-1.2B-Instruct"


def write_adapter(
    root: Path, *, base: dict[str, Any], run_id: str = "run-7", step: int = 40
) -> dict[str, Any]:
    """An adapter manifest on disk, through the real assembler."""
    directory = adapter_paths.adapter_dir(run_id, step, root=root)
    directory.mkdir(parents=True, exist_ok=True)
    doc = adapter_manifest.build(
        run_id=run_id,
        step=step,
        base=base,
        adapter=adapter_config(),
        optimizer=optimizer_block(),
    )
    doc["weights_sha256"] = "a" * 64
    write_canonical(directory / adapter_paths.MANIFEST_NAME, doc)
    (directory / adapter_paths.WEIGHTS_NAME).write_bytes(b"lora")
    return doc


class TestTheRefusalWhenNoUnmergedBaseSurvives:
    def test_a_HUB_base_is_refused_because_it_is_not_a_checkpoint(self, tmp_path: Path) -> None:
        """FPRD D15's strict reading. Still obtainable, but not a checkpoint."""
        base = adapter_manifest.base_block(hub_record(HUB_REPO))
        with pytest.raises(Refusal) as caught:
            merge_module.resolve_base_checkpoint(base, root=tmp_path)
        assert caught.value.code == "merge_refused_no_unmerged_base"

    def test_the_refusal_SAYS_WHAT_TO_DO_NEXT(self, tmp_path: Path) -> None:
        """ADR-021: what to do next, not how the reader should feel."""
        base = adapter_manifest.base_block(hub_record(HUB_REPO))
        with pytest.raises(Refusal) as caught:
            merge_module.resolve_base_checkpoint(base, root=tmp_path)
        assert "Save a checkpoint of the base first" in caught.value.message

    def test_a_checkpoint_base_whose_DIRECTORY_IS_GONE_is_refused(self, tmp_path: Path) -> None:
        base = adapter_manifest.base_block(checkpoint_record(), manifest_sha256="9" * 64)
        with pytest.raises(Refusal) as caught:
            merge_module.resolve_base_checkpoint(base, root=tmp_path)
        assert caught.value.code == "merge_refused_no_unmerged_base"
        assert "expected_path" in (caught.value.details or {}), (
            "the refusal must name the path it looked in, which is how an operator finds "
            "out whether the checkpoint was never saved or was moved"
        )

    def test_an_unparseable_base_name_is_refused_rather_than_searched_for(
        self, tmp_path: Path
    ) -> None:
        with pytest.raises(Refusal) as caught:
            merge_module.resolve_base_checkpoint(
                {"name": "not-a-qualified-name", "manifest_sha256": "9" * 64}, root=tmp_path
            )
        assert caught.value.code == "merge_refused_no_unmerged_base"

    def test_a_checkpoint_whose_MANIFEST_HASH_DIFFERS_is_refused(self, tmp_path: Path) -> None:
        """A different parameter set under the same name. Merging onto it would produce a
        checkpoint whose lineage is a lie - and the hash is the only thing that notices."""
        written = saved_checkpoint(tmp_path, run_id="run-base", step=40)
        with pytest.raises(Refusal) as caught:
            merge_module.resolve_base_checkpoint(
                {"name": written.manifest["name"], "manifest_sha256": "0" * 64}, root=tmp_path
            )
        assert caught.value.code == "merge_refused_no_unmerged_base"
        details = caught.value.details or {}
        assert details["recorded_manifest_sha256"] == "0" * 64
        assert details["found_manifest_sha256"] != "0" * 64


class TestTheAdmissionWhenAnUnmergedBaseDoesSurvive:
    def test_a_saved_checkpoint_base_resolves(self, tmp_path: Path) -> None:
        written = saved_checkpoint(tmp_path, run_id="run-base", step=40)
        digest = sha256_of(written.manifest)
        resolved = merge_module.resolve_base_checkpoint(
            {"name": written.manifest["name"], "manifest_sha256": digest}, root=tmp_path
        )
        assert resolved.name == written.manifest["name"]
        assert resolved.directory == written.directory
        assert resolved.manifest_sha256 == digest

    def test_the_resolved_base_carries_the_lineage_ROOT_not_the_parent(
        self, tmp_path: Path
    ) -> None:
        """``base_model`` is *"the root model this lineage descends from"*, so the merged
        checkpoint's ``base_model`` is the base's own - carrying it forward rather than
        substituting the base, which would terminate every lineage walk one hop short."""
        written = saved_checkpoint(
            tmp_path, run_id="run-base", step=40, base_model="LiquidAI/LFM2-1.2B"
        )
        resolved = merge_module.resolve_base_checkpoint(
            {"name": written.manifest["name"], "manifest_sha256": sha256_of(written.manifest)},
            root=tmp_path,
        )
        assert resolved.base_model == "LiquidAI/LFM2-1.2B"
        assert resolved.base_model != resolved.name

    def test_the_hash_compared_is_of_the_CANONICAL_bytes_like_reindexs(
        self, tmp_path: Path
    ) -> None:
        """One hashing path (ADR-005). Hashing the *file* would include its whitespace and
        two readers would disagree silently."""
        written = saved_checkpoint(tmp_path, run_id="run-base", step=40)
        on_disk = checkpoint_manifest.read(written.directory)
        assert sha256_of(on_disk) == sha256_of(written.manifest)


class TestTheRefusalRunsBeforeAnyTensorArithmetic:
    """FR-008.38: *"the check runs before any merged tensor is computed, not before the
    write."* Computing merged tensors for a merge that will be refused wastes the card and
    makes a refusal look like a failure."""

    def test_the_check_precedes_the_apply_in_the_merge_body(self) -> None:
        source = inspect.getsource(merge_module.merge)
        check = source.index("resolve_base_checkpoint")
        apply_call = source.index("apply_adapter(")
        assert check < apply_call, "the unmerged-base check must precede the arithmetic"

    def test_the_refusal_lives_in_its_OWN_NAMED_function(self) -> None:
        """So M7 is a one-line deletion with a test to catch it, rather than a condition
        buried in a longer body."""
        assert callable(merge_module.resolve_base_checkpoint)
        assert "merge_refused_no_unmerged_base" in inspect.getsource(
            merge_module.resolve_base_checkpoint
        )

    def test_a_refused_merge_reaches_peft_not_at_all(self, tmp_path: Path) -> None:
        """Proved by the refusal arriving on a machine where ``peft`` is not installed: if
        the arithmetic ran first, this would raise ``ml_runtime_unavailable`` instead."""
        write_adapter(tmp_path, base=adapter_manifest.base_block(hub_record(HUB_REPO)))
        with pytest.raises(Refusal) as caught:
            merge_module.merge(
                session=object(),
                adapter_name=adapter_paths.adapter_name(HUB_REPO, "run-7", 40),
                base_module=object(),
                tokenizer=object(),
                run_id="run-7",
                step=41,
                domain=None,
                signal_source="corpus",
                optimizer=optimizer_block(),
                root=tmp_path,
            )
        assert caught.value.code == "merge_refused_no_unmerged_base"


class TestTheAdapterNameIsResolvedNotGuessed:
    def test_an_unparseable_adapter_name_is_refused(self, tmp_path: Path) -> None:
        with pytest.raises(Refusal) as caught:
            merge_module.merge(
                session=object(),
                adapter_name="../../etc/passwd",
                base_module=object(),
                tokenizer=object(),
                run_id="run-7",
                step=41,
                domain=None,
                signal_source="corpus",
                optimizer=optimizer_block(),
                root=tmp_path,
            )
        assert caught.value.code == "adapter_manifest_missing"

    def test_a_traversal_in_the_run_id_is_refused_by_the_confined_path(
        self, tmp_path: Path
    ) -> None:
        """Path confinement is F5's ``run_dir``, reused. A traversal is **refused, not
        normalised** - sanitising would compose *some* path and operate on it."""
        with pytest.raises(Refusal):
            adapter_paths.adapter_dir("../escape", 1, root=tmp_path)


class TestTheBaseMismatchIsRefused:
    """E-13: a merge is only defined against the base the adapter names."""

    def test_a_held_base_that_differs_is_refused_before_the_unmerged_base_check(
        self, tmp_path: Path
    ) -> None:
        write_adapter(tmp_path, base=adapter_manifest.base_block(hub_record(HUB_REPO)))
        with pytest.raises(Refusal) as caught:
            merge_module.merge(
                session=object(),
                adapter_name=adapter_paths.adapter_name(HUB_REPO, "run-7", 40),
                base_module=object(),
                tokenizer=object(),
                run_id="run-7",
                step=41,
                domain=None,
                signal_source="corpus",
                optimizer=optimizer_block(),
                held_base_name="some/other-model",
                root=tmp_path,
            )
        assert caught.value.code == "adapter_base_mismatch"


class TestTheMergedCheckpointRecordsBOTHAncestors:
    """FR-008.37, M8. All three ``merged_adapter`` members, **and the schema now requires
    all three** - re-verified here by validation rather than by reading."""

    def _fields(self, merged_adapter: dict[str, Any] | None) -> dict[str, Any]:
        return {
            "run_id": "run-7",
            "step": 41,
            "domain": None,
            "signal_source": "corpus",
            "base_model": "LiquidAI/LFM2-1.2B",
            "parent": "LiquidAI/LFM2-1.2B:run-base:40",
            "optimizer": optimizer_block(),
            "weights_sha256": "f" * 64,
            "merged_adapter": merged_adapter,
        }

    def test_the_three_member_form_validates(self) -> None:
        doc = checkpoint_manifest.build(
            **self._fields({"name": "x:run-7:adapter-40", "step": 40, "manifest_sha256": "b" * 64})
        )
        assert doc["merged_adapter"]["manifest_sha256"] == "b" * 64
        assert doc["parent"] == "LiquidAI/LFM2-1.2B:run-base:40"

    def test_the_TWO_member_form_FAILS_the_real_schema(self) -> None:
        """FPRD D4's closure, verified by validation. ``manifest_sha256`` is required now,
        so a conforming merged checkpoint can no longer identify its adapter unverifiably."""
        with pytest.raises(Refusal):
            checkpoint_manifest.build(**self._fields({"name": "x:run-7:adapter-40", "step": 40}))

    def test_manifest_sha256_is_required_AND_NULLABLE_which_is_three_states(self) -> None:
        """A hash, present-and-``null`` (conforming, deliberately unverified), and the key
        **absent** (nonconforming). ``null`` is not collapsed into absent."""
        from models.domain_state import ArtifactKind
        from services.reindex import load_schema, schema_violations

        definition = load_schema(ArtifactKind.CHECKPOINT)["$defs"]["MergedAdapter"]
        assert "manifest_sha256" in definition["required"]
        assert {"type": "null"} in definition["properties"]["manifest_sha256"]["anyOf"]

        doc = checkpoint_manifest.build(
            **self._fields({"name": "x:run-7:adapter-40", "step": 40, "manifest_sha256": None})
        )
        assert schema_violations(doc, ArtifactKind.CHECKPOINT) == []
        assert doc["merged_adapter"]["manifest_sha256"] is None

    def test_merge_writes_all_three_from_the_adapter_it_read(self) -> None:
        source = inspect.getsource(merge_module.merge)
        assert '"name": doc["name"]' in source
        assert '"step": doc["step"]' in source
        assert '"manifest_sha256": manifest_sha256' in source


class TestTheAdapterSurvivesItsOwnMerge:
    def test_nothing_on_this_path_deletes_or_moves_an_adapter_directory(self) -> None:
        """FR-008.41. A merged adapter cannot be taken off, which is why the separate
        artifact stays - so a merge that consumed it would remove the only way back."""
        import ast

        source = (
            Path(__file__).resolve().parents[3] / "backend/src/training/runtypes/adapters/merge.py"
        ).read_text(encoding="utf-8")
        forbidden = {"rmtree", "unlink", "rmdir", "move", "rename"}
        for node in ast.walk(ast.parse(source)):
            if isinstance(node, ast.Attribute) and node.attr in forbidden:
                raise AssertionError(f"merge.py calls {node.attr} at line {node.lineno}")


class TestNoBarrierAnywhereOnTheMergePath:
    """FR-008.40, ADR-022. Standalone the protection is the atomic write plus the refusal;
    the bracket form is section 5.10's and F11's."""

    #: Everything that would mean the barrier had come back. Checked against **code**,
    #: not against prose: the module's docstring explains at length that there is no
    #: barrier, so a text grep over the whole file fails on its own explanation - which is
    #: a check that cannot pass rather than a check that bites.
    FORBIDDEN = ("quiesce", "commit-begin", "commit_begin", "commit-end", "/api/weights")

    def test_no_barrier_call_and_no_sibling_endpoint_appears_in_the_CODE(self) -> None:
        import ast

        source = (
            Path(__file__).resolve().parents[3] / "backend/src/training/runtypes/adapters/merge.py"
        ).read_text(encoding="utf-8")
        tree = ast.parse(source)
        docstrings = {
            id(node.body[0].value)
            for node in ast.walk(tree)
            if isinstance(node, ast.Module | ast.ClassDef | ast.FunctionDef)
            and node.body
            and isinstance(node.body[0], ast.Expr)
            and isinstance(node.body[0].value, ast.Constant)
            and isinstance(node.body[0].value.value, str)
        }
        offenders: list[str] = []
        for node in ast.walk(tree):
            if isinstance(node, ast.Constant) and isinstance(node.value, str):
                if id(node) in docstrings:
                    continue
                lowered = node.value.lower()
                offenders += [
                    f"string {term!r} at line {node.lineno}"
                    for term in self.FORBIDDEN
                    if term in lowered
                ]
            if isinstance(node, ast.Attribute) and node.attr in {
                "quiesce",
                "commit_begin",
                "commit_end",
                "resume",
            }:
                offenders.append(f"call .{node.attr} at line {node.lineno}")
        assert not offenders, offenders

    def test_no_f8_module_imports_the_hotupdate_package(self) -> None:
        """ADR-022's fence. The standalone test runs with ``training/hotupdate/`` absent, so
        a core-path import of it is not merely untidy - it breaks AC20."""
        import ast

        package = Path(__file__).resolve().parents[3] / "backend/src/training/runtypes"
        offenders: list[str] = []
        for path in sorted(package.rglob("*.py")):
            for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
                if isinstance(node, ast.ImportFrom) and "hotupdate" in (node.module or ""):
                    offenders.append(f"{path.name}:{node.lineno}")
                if isinstance(node, ast.Import):
                    offenders += [
                        f"{path.name}:{node.lineno}"
                        for alias in node.names
                        if "hotupdate" in alias.name
                    ]
        assert not offenders, offenders
