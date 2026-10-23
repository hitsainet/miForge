"""The manifest: provenance, the two key-set clauses, and the lineage root  (FR-005.7-16).

**The four worked examples are built through the real writer and read back through the real
reader**, not asserted against hand-written dicts: a fixture and a builder that are two views
of one mistake agree forever (ADR-016). The five negative controls are what prove each
assertion can fail - *a control whose expected result is stale is worse than no control*,
and two of these five had their expected result inverted by the schema change on
2026-10-13, so they are run against the **live** schema rather than against a list.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from core.errors import Refusal
from services.checkpoints import manifest as manifest_module
from services.checkpoints.manifest import ROOT, declared_keys
from services.checkpoints.paths import qualified_name, step_dir_name
from tests.support.checkpoints import fields, saved_checkpoint, score


def build(**overrides):
    """``build`` with the fixture's borrowed values, overridden where a test needs it."""
    base = fields(**{k: v for k, v in overrides.items() if k in _FIELD_NAMES})
    kwargs = base.as_kwargs()
    kwargs.pop("extra")
    kwargs.update({k: v for k, v in overrides.items() if k not in _FIELD_NAMES})
    kwargs.setdefault("weights_sha256", "c" * 64)
    return manifest_module.build(**kwargs)


_FIELD_NAMES = {
    "run_id",
    "step",
    "domain",
    "signal_source",
    "base_model",
    "parent",
    "optimizer_state",
    "optimizer_name",
    "config_hash",
    "test",
    "retention",
    "merged_adapter",
    "created_at",
}


# ======================================================================== the key sets


class TestTheDeclaredKeysComeFromTheSchema:
    def test_the_key_list_is_read_from_the_file_and_never_retyped(self):
        """A copied list agrees with itself and would pass the day the schema changes -
        which is how this task came to be wrong in the first place."""
        schema = json.loads(
            (
                Path(__file__).resolve().parents[3] / "docs/schemas/checkpoint-manifest-v1.json"
            ).read_text("utf-8")
        )
        root = schema["$defs"]["CheckpointManifestV1"]["properties"]
        assert declared_keys()[ROOT] == frozenset(root)
        assert declared_keys()["test"] == frozenset(schema["$defs"]["ScoreRecord"]["properties"])
        assert declared_keys()["optimizer"] == frozenset(
            schema["$defs"]["OptimizerIdentity"]["properties"]
        )
        assert declared_keys()["merged_adapter"] == frozenset(
            schema["$defs"]["MergedAdapter"]["properties"]
        )

    def test_provider_is_declared_and_NOT_required(self):
        """**The premise of clause (b).** Because the field is declared, a manifest without
        it validates clean; because it is not required, neither the validator nor the
        schema-sync test - which compares schema *files* - can see it go missing. A test is
        the only thing that catches the omission."""
        schema = json.loads(
            (
                Path(__file__).resolve().parents[3] / "docs/schemas/checkpoint-manifest-v1.json"
            ).read_text("utf-8")
        )
        record = schema["$defs"]["ScoreRecord"]
        assert "provider" in record["properties"], "the field the join depends on is gone"
        assert "provider" not in record["required"], (
            "if provider ever becomes required, clause (b) stops being the ONLY enforcement "
            "and this test should say so rather than silently keep passing"
        )


class TestTheKeySetAssertion:
    def test_manifest_rejects_undeclared_keys(self):
        """Clause (a). Defence in depth beside the schema's own ``additionalProperties:
        false`` - it fires first and names the key in miForge's vocabulary."""
        with pytest.raises(Refusal) as caught:
            build(extra={"tolerance": 0.01})
        assert caught.value.code == "checkpoint_manifest_invalid"
        assert "tolerance" in str(caught.value.details)

    def test_an_undeclared_key_inside_a_score_record_is_refused_too(self):
        """One level down. ``provide`` for ``provider`` is the realistic spelling."""
        with pytest.raises(Refusal) as caught:
            build(test={"set": "s", "score": 0.5, "n": 10, "provide": "x"})
        assert caught.value.code == "checkpoint_manifest_invalid"

    def test_manifest_preserves_every_supplied_key(self):
        """**Clause (b) - MUT-5's target, and the control that still bites.**

        A manifest that silently loses the ``provider`` F4 recorded validates clean, so the
        provenance R-02.16 demands would be gone with no error on any surface."""
        supplied = score(provider="openai-compatible:http://vllm")
        doc = build(test=supplied)
        assert doc["test"] == supplied
        assert doc["test"]["provider"] == "openai-compatible:http://vllm"

    def test_dropping_a_supplied_key_is_refused_by_name(self):
        """The assertion itself, exercised directly: a writer that dropped the key would
        hit this, and the refusal names what was lost."""
        doc = build(test=score())
        doc["test"].pop("provider")
        with pytest.raises(Refusal) as caught:
            manifest_module.assert_no_dropped_keys(doc, {"test": score()})
        assert caught.value.code == "checkpoint_manifest_dropped_keys"
        assert caught.value.details["dropped"] == ["provider"]

    def test_a_record_written_as_null_when_one_was_supplied_is_refused(self):
        doc = build(test=score())
        doc["test"] = None
        with pytest.raises(Refusal) as caught:
            manifest_module.assert_no_dropped_keys(doc, {"test": score()})
        assert caught.value.code == "checkpoint_manifest_dropped_record"

    def test_a_null_record_is_not_a_dropped_key(self):
        """``null`` means *no test ran at this step*, which is a statement. Only a record
        that was handed over and then lost is a breach."""
        doc = build(test=None)
        manifest_module.assert_no_dropped_keys(doc, {"test": None})


# ======================================================================== provenance


class TestEveryValueIsSuppliedAndNoneIsDefaulted:
    def test_manifest_writes_domain_explicitly(self):
        """**MUT-8's target.** The schema keeps ``domain`` required precisely so that *no
        domain* is asserted: null is a statement, an absent key is an omission."""
        assert build(domain=None)["domain"] is None
        assert "domain" in build(domain=None)

    def test_domain_has_no_default_in_the_signature(self):
        """A default would make the omission reachable. ``None`` must be passed."""
        with pytest.raises(TypeError):
            manifest_module.build(  # type: ignore[call-arg]
                run_id="r",
                step=1,
                signal_source="corpus",
                base_model="b",
                parent=None,
                optimizer={"name": "adamw", "state": "fresh"},
                weights_sha256="c" * 64,
            )

    def test_signal_source_is_refused_rather_than_defaulted(self):
        """F2 owns the value and F5 and F2 are concurrent. ``"corpus"`` here would be a
        fabricated provenance claim in the one artifact meant to outlive everything."""
        with pytest.raises(Refusal) as caught:
            build(signal_source="")
        assert caught.value.code == "save_refused_missing_field"
        assert caught.value.details["field"] == "signal_source"

    def test_an_unknown_signal_source_fails_the_save(self):
        with pytest.raises(Refusal) as caught:
            build(signal_source="vibes")
        assert caught.value.code == "checkpoint_manifest_invalid"

    def test_no_getattr_or_or_literal_defaults_near_manifest_assembly(self):
        """ADR-018, as a grep rather than as a review instruction: *in this feature a
        fabricated value is permanent.*"""
        source = Path(manifest_module.__file__).read_text("utf-8")
        block = source.split("doc: dict[str, Any] = {")[1].split("if extra:")[0]
        # Comments carry the *reason* a default is forbidden and necessarily contain the
        # words; the assertion is about code. Stripped rather than the words softened,
        # because "no `or` on this line on purpose" is exactly the comment worth keeping.
        body = "\n".join(line.split("#", 1)[0] for line in block.splitlines())
        for forbidden in ("getattr(", " or ", ".get("):
            assert forbidden not in body, (
                f"{forbidden!r} appears in the manifest assembly block. A default that "
                f"always fires is a fabricated value (ADR-018)."
            )


class TestScoreRecordsPassThroughVerbatim:
    def test_an_n_of_zero_fails_the_save(self):
        """E-6: a zero-prompt score is not a score. Refused, not coerced - and the schema's
        ``minimum: 1`` agrees, so both fire."""
        with pytest.raises(Refusal) as caught:
            build(test={"set": "s", "score": 0.5, "n": 0})
        assert caught.value.code == "checkpoint_manifest_invalid"

    def test_a_score_outside_zero_to_one_fails_the_save(self):
        with pytest.raises(Refusal):
            build(test={"set": "s", "score": 1.5, "n": 10})

    def test_a_missing_set_fails_the_save(self):
        with pytest.raises(Refusal) as caught:
            build(test={"score": 0.5, "n": 10})
        assert caught.value.details["missing"] == ["set"]

    def test_nothing_is_rounded_or_renormalised(self):
        supplied = score(value=0.123456789012345, n=7)
        assert build(test=supplied)["test"]["score"] == 0.123456789012345

    def test_a_provider_is_never_defaulted_in(self):
        """A record F4 supplied without one is written without one. A guessed provider is a
        fabricated provenance, and null is a statement."""
        doc = build(test=score(provider=None))
        assert "provider" not in doc["test"]


# ======================================================================== lineage


class TestBaseModelIsTheLineageRoot:
    def test_base_model_is_the_lineage_root(self):
        """**MUT-9's target.** For a continue or a branch the thing loaded *is* a
        checkpoint, so the root is carried forward from its own manifest - not set to the
        thing just loaded, which would root the tree at the parent and terminate every walk
        one hop short, on a tree that validates and looks plausible."""
        parent_manifest = {
            "name": "LiquidAI/LFM2-1.2B:run-a:step-500",
            "base_model": "LiquidAI/LFM2-1.2B",
        }
        root, parent = manifest_module.resolve_lineage(parent_manifest=parent_manifest)
        assert root == "LiquidAI/LFM2-1.2B"
        assert parent == "LiquidAI/LFM2-1.2B:run-a:step-500"

    def test_a_first_load_is_its_own_root_with_no_parent(self):
        assert manifest_module.resolve_lineage(loaded_identity="LiquidAI/LFM2-1.2B") == (
            "LiquidAI/LFM2-1.2B",
            None,
        )

    def test_a_create_run_roots_at_its_config_with_a_null_parent(self):
        """AC12: ``parent: null`` and ``base_model`` = the architecture config. No schema
        change was ever needed for it."""
        assert manifest_module.resolve_lineage(config_identity="sha256:abc") == ("sha256:abc", None)

    def test_a_checkpoint_with_no_base_model_refuses_rather_than_substituting_itself(self):
        """Substituting would root the tree at the parent. F1's function refuses; this is
        the second reader confirming it still does."""
        with pytest.raises(Refusal) as caught:
            manifest_module.resolve_lineage(
                parent_manifest={"name": "x:run-a:step-1", "base_model": ""}
            )
        assert caught.value.code == "checkpoint_base_model_missing"

    def test_naming_no_source_at_all_is_a_refusal_not_a_guess(self):
        with pytest.raises(Refusal) as caught:
            manifest_module.resolve_lineage()
        assert caught.value.code == "save_refused_missing_field"


class TestBothStepFormsComeFromTheOneInteger:
    @pytest.mark.parametrize("step", [0, 7, 500, 10000])
    def test_the_path_pads_to_six_and_the_name_does_not(self, step):
        """BRD-01 R-01.24 and the schema's own ``step`` description. **Nothing carries the
        step as a string of its own**, which is why both are functions of the integer."""
        assert step_dir_name(step) == f"step-{step:06d}"
        assert qualified_name("b", "r", step).endswith(f":step-{step}")

    def test_the_manifest_renders_the_unpadded_name(self):
        doc = build(step=500)
        assert doc["name"] == "LiquidAI/LFM2-1.2B:run-a:step-500"
        assert doc["step"] == 500 and isinstance(doc["step"], int)


# ======================================================================== the §5.4 examples


class TestTheFourWorkedExamplesValidate:
    """FPRD section 5.4, **written by the real writer and read back by the real reader**."""

    def test_A_a_mid_run_save_with_the_optimizer_restored(self, tmp_path: Path):
        result = saved_checkpoint(
            tmp_path,
            step=500,
            parent="LiquidAI/LFM2-1.2B:run-a:step-400",
            test=score(),
            retention=score("retention-general", value=0.72, floor=0.65),
        )
        back = manifest_module.read(result.directory)
        assert back == result.manifest
        assert back["optimizer"]["state"] == "restored"
        assert back["merged_adapter"] is None

    def test_B_a_create_runs_step_zero_baseline(self, tmp_path: Path):
        """AC12: ``step: 0``, ``domain: null``, ``parent: null``, ``base_model`` = the
        architecture config identity."""
        result = saved_checkpoint(
            tmp_path,
            run_id="run-create",
            step=0,
            domain=None,
            base_model="sha256:2f6c...-architecture-config",
            parent=None,
            optimizer_state="fresh",
            test=None,
        )
        back = manifest_module.read(result.directory)
        assert back["step"] == 0 and back["domain"] is None and back["parent"] is None
        assert result.directory.name == "step-000000"

    def test_C_a_merged_checkpoint_has_two_ancestors(self, tmp_path: Path):
        """R-02.45: the base via ``parent`` and the adapter via ``merged_adapter``."""
        result = saved_checkpoint(
            tmp_path,
            run_id="run-merge",
            step=12,
            parent="LiquidAI/LFM2-1.2B:run-a:step-500",
            merged_adapter={
                "name": "LiquidAI/LFM2-1.2B:run-lora:step-800",
                "step": 800,
                "manifest_sha256": "d" * 64,
            },
        )
        back = manifest_module.read(result.directory)
        assert back["parent"] and back["merged_adapter"]["name"]

    def test_C_manifest_sha256_has_three_states_not_two(self, tmp_path: Path):
        """``MergedAdapter.manifest_sha256`` is **required AND nullable**: a hash,
        present-and-null (conforming, deliberately unverified), or absent (nonconforming).
        Several documents collapsed null into absent; the schema does not."""
        saved_checkpoint(
            tmp_path,
            run_id="run-merge-null",
            step=1,
            merged_adapter={"name": "a:b:step-1", "step": 1, "manifest_sha256": None},
        )
        with pytest.raises(Refusal):
            saved_checkpoint(
                tmp_path,
                run_id="run-merge-absent",
                step=1,
                merged_adapter={"name": "a:b:step-1", "step": 1},
            )

    def test_D_a_branchs_first_save_carries_the_root_forward(self, tmp_path: Path):
        result = saved_checkpoint(
            tmp_path,
            run_id="run-branch",
            step=0,
            parent="LiquidAI/LFM2-1.2B:run-a:step-500",
            base_model="LiquidAI/LFM2-1.2B",
            optimizer_state="fresh",
        )
        back = manifest_module.read(result.directory)
        assert back["base_model"] == "LiquidAI/LFM2-1.2B"
        assert back["parent"] == "LiquidAI/LFM2-1.2B:run-a:step-500"
        assert back["optimizer"]["state"] == "fresh"


class TestTheFiveNegativeControls:
    """*A control whose expected result is stale is worse than none.* Two of these five
    were ``accepted`` until the schema closed its objects on 2026-10-13; they are run
    against the live schema rather than against a table."""

    def test_domain_removed_is_refused(self):
        doc = build()
        doc.pop("domain")
        with pytest.raises(Refusal):
            manifest_module.validate(doc)

    def test_an_unknown_optimizer_state_is_refused(self):
        with pytest.raises(Refusal):
            build(optimizer_state="stale")

    def test_n_zero_is_refused(self):
        with pytest.raises(Refusal):
            build(test={"set": "s", "score": 0.5, "n": 0})

    def test_an_undeclared_top_level_key_is_refused(self):
        with pytest.raises(Refusal):
            build(extra={"tolerance": 0.01})

    def test_a_misspelled_key_inside_a_score_record_is_refused(self):
        with pytest.raises(Refusal):
            build(test={"set": "s", "score": 0.5, "n": 10, "provide": "x"})


# ======================================================================== read


class TestReadValidatesBecauseADirectoryCanArriveFromAnywhere:
    def test_a_missing_manifest_is_checkpoint_not_found(self, tmp_path: Path):
        with pytest.raises(Refusal) as caught:
            manifest_module.read(tmp_path)
        assert caught.value.code == "checkpoint_not_found"

    def test_unparseable_json_is_refused_by_name(self, tmp_path: Path):
        (tmp_path / "manifest.json").write_text("{not json", encoding="utf-8")
        with pytest.raises(Refusal) as caught:
            manifest_module.read(tmp_path)
        assert caught.value.code == "checkpoint_manifest_invalid"

    def test_a_nonconforming_manifest_on_disk_is_refused(self, tmp_path: Path):
        """OQ-4: a checkpoint directory can arrive from anywhere, so a manifest read from
        disk is untrusted input until validated - exactly like a request body."""
        doc = build()
        doc["step"] = -1
        (tmp_path / "manifest.json").write_text(json.dumps(doc), encoding="utf-8")
        with pytest.raises(Refusal) as caught:
            manifest_module.read(tmp_path)
        assert caught.value.code == "checkpoint_manifest_invalid"
