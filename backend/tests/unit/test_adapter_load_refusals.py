"""The read path's refusals: an unknown scaling rule, a bad manifest, a wrong base.

**M6's target.** Note what the control's record must say: M6 goes red against **both** the
code check and the schema's enum, because the enum refuses a bad value and validation runs
first. That is a double red over one rule, not two independent protections - and the code
line is kept as defence in depth for two futures (an unvalidated document, and a v2 schema
vendored beside this one), **not** because the schema misses the present case.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from core.canonical import write_canonical
from core.errors import Refusal
from models.domain_state import ArtifactKind
from services.reindex import load_schema, schema_violations
from tests.support.test_adapter_doubles import adapter_config, hub_record, optimizer_block
from training.runtypes.adapters import load as adapter_load
from training.runtypes.adapters import manifest as adapter_manifest
from training.runtypes.adapters import paths as adapter_paths
from training.runtypes.adapters import schema as adapter_schema

HUB_REPO = "LiquidAI/LFM2.5-1.2B-Instruct"


def valid_document(**overrides: Any) -> dict[str, Any]:
    doc = adapter_manifest.build(
        run_id="run-7",
        step=40,
        base=adapter_manifest.base_block(hub_record(HUB_REPO)),
        adapter=adapter_config(),
        optimizer=optimizer_block(),
    )
    doc["weights_sha256"] = "a" * 64
    doc.update(overrides)
    return doc


def written(tmp_path: Path, doc: dict[str, Any]) -> Path:
    path = tmp_path / adapter_paths.MANIFEST_NAME
    write_canonical(path, doc)
    return path


class TestTheHappyPath:
    def test_a_valid_manifest_loads(self, tmp_path: Path) -> None:
        doc = valid_document()
        assert adapter_load.load_manifest(written(tmp_path, doc)) == doc

    def test_load_for_resolves_the_run_and_step_directory(self, tmp_path: Path) -> None:
        directory = adapter_paths.adapter_dir("run-7", 40, root=tmp_path)
        directory.mkdir(parents=True)
        write_canonical(directory / adapter_paths.MANIFEST_NAME, valid_document())
        assert adapter_load.load_for("run-7", 40, root=tmp_path)["step"] == 40


class TestTheScalingRuleIsRefusedAndNeverGuessed:
    """E-8, FR-008.23, M6."""

    def test_an_unrecognised_value_is_refused_BY_NAME(self) -> None:
        doc = valid_document()
        doc["adapter"]["scaling"] = "rslora"
        with pytest.raises(Refusal) as caught:
            adapter_load.assert_scaling_known(doc)
        assert caught.value.code == "adapter_scaling_unrecognised"
        assert (caught.value.details or {})["scaling"] == "rslora"

    def test_the_SCHEMA_also_refuses_it_and_the_control_record_must_say_so(self) -> None:
        """The double red M6 produces, made explicit so nobody records it as two
        independent protections."""
        doc = valid_document()
        doc["adapter"]["scaling"] = "rslora"
        assert schema_violations(doc, ArtifactKind.ADAPTER), "the enum must refuse it too"

    def test_an_ABSENT_scaling_key_means_alpha_over_rank_BY_CONTRACT(self) -> None:
        """Not a fabricated default: ``scaling`` is not in ``AdapterConfig.required`` and
        the schema declares ``"default": "alpha_over_r"``. Turning an absent key into a
        refusal would be a refusal the contract does not authorise."""
        definition = load_schema(ArtifactKind.ADAPTER)["$defs"]["AdapterConfig"]
        assert "scaling" not in definition["required"]
        assert definition["properties"]["scaling"]["default"] == "alpha_over_r"

        doc = valid_document()
        del doc["adapter"]["scaling"]
        assert schema_violations(doc, ArtifactKind.ADAPTER) == []
        assert adapter_load.assert_scaling_known(doc) == adapter_schema.scaling_default()

    def test_the_known_set_is_READ_from_the_schemas_enum(self) -> None:
        """So a vendored schema whose enum grew cannot leave the code quietly narrower."""
        enum = load_schema(ArtifactKind.ADAPTER)["$defs"]["AdapterConfig"]["properties"]["scaling"][
            "enum"
        ]
        assert adapter_schema.known_scaling() == frozenset(enum)

    def test_a_v2_document_is_rejected_by_schema_version_BEFORE_the_check_is_reached(
        self, tmp_path: Path
    ) -> None:
        """Which is why the code check is defence in depth rather than the enforcement:
        ``schema_version``'s ``const`` rejects a v2 document earlier."""
        doc = valid_document()
        doc["schema_version"] = "2"
        doc["adapter"]["scaling"] = "rslora"
        with pytest.raises(Refusal) as caught:
            adapter_load.load_manifest(written(tmp_path, doc))
        assert caught.value.code == "adapter_manifest_invalid"


class TestAMalformedManifestIsRefusedByName:
    def test_a_missing_file_is_refused(self, tmp_path: Path) -> None:
        with pytest.raises(Refusal) as caught:
            adapter_load.load_manifest(tmp_path / adapter_paths.MANIFEST_NAME)
        assert caught.value.code == "adapter_manifest_missing"

    def test_unparseable_json_is_refused(self, tmp_path: Path) -> None:
        path = tmp_path / adapter_paths.MANIFEST_NAME
        path.write_text("{not json", encoding="utf-8")
        with pytest.raises(Refusal) as caught:
            adapter_load.load_manifest(path)
        assert caught.value.code == "adapter_manifest_invalid"

    def test_a_json_array_is_refused(self, tmp_path: Path) -> None:
        path = tmp_path / adapter_paths.MANIFEST_NAME
        path.write_text("[]", encoding="utf-8")
        with pytest.raises(Refusal) as caught:
            adapter_load.load_manifest(path)
        assert caught.value.code == "adapter_manifest_invalid"

    def test_a_manifest_with_NO_DIGEST_is_refused(self, tmp_path: Path) -> None:
        """E-9. BRD-01 R-01.27 verifies an adapter's hash the way R-01.14 verifies a
        checkpoint's; without the field that verification has nothing to read and a
        corrupted adapter loads silently."""
        doc = valid_document()
        del doc["weights_sha256"]
        with pytest.raises(Refusal) as caught:
            adapter_load.load_manifest(written(tmp_path, doc))
        assert caught.value.code == "adapter_manifest_invalid"
        assert any("weights_sha256" in v for v in (caught.value.details or {})["violations"])

    def test_an_INVENTED_key_is_refused_because_the_objects_are_closed(
        self, tmp_path: Path
    ) -> None:
        """FPRD D6, closed 2026-10-13 - verified here by validation rather than by reading.
        ``additionalProperties: false`` on every object, so a misspelled field fails."""
        doc = valid_document()
        doc["domain"] = "contracts"
        with pytest.raises(Refusal) as caught:
            adapter_load.load_manifest(written(tmp_path, doc))
        assert caught.value.code == "adapter_manifest_invalid"

    def test_an_invented_key_INSIDE_a_score_record_is_refused_too(self, tmp_path: Path) -> None:
        doc = valid_document()
        doc["test"] = {
            "set": "s",
            "score": 0.5,
            "n": 10,
            "floor": None,
            "provider": "in_process",
            "made_up": 1,
        }
        with pytest.raises(Refusal):
            adapter_load.load_manifest(written(tmp_path, doc))

    def test_every_violation_is_reported_not_only_the_first(self, tmp_path: Path) -> None:
        """A writer that is wrong about two fields should be told about both."""
        doc = valid_document()
        del doc["weights_sha256"]
        doc["step"] = -1
        with pytest.raises(Refusal) as caught:
            adapter_load.load_manifest(written(tmp_path, doc))
        assert len((caught.value.details or {})["violations"]) >= 2


class TestAnAdapterCannotBeAppliedToADifferentBase:
    """E-10, E-13, FR-008.34. An adapter applied to a different base produces undefined
    behaviour rather than a worse answer, which is why this refuses instead of warning."""

    def test_the_matching_base_passes(self) -> None:
        adapter_load.assert_base_matches(valid_document(), HUB_REPO)

    def test_a_different_base_is_refused(self) -> None:
        with pytest.raises(Refusal) as caught:
            adapter_load.assert_base_matches(valid_document(), "some/other-model")
        assert caught.value.code == "adapter_base_mismatch"
        assert (caught.value.details or {})["adapter_base"] == HUB_REPO

    def test_the_shape_a_consumer_checks_is_target_modules(self) -> None:
        """The schema's own description names it as the field a consumer verifies against
        the resident base (BRD-01 R-01.27), so it must be present and non-empty."""
        definition = load_schema(ArtifactKind.ADAPTER)["$defs"]["AdapterConfig"]
        assert "target_modules" in definition["required"]
        assert definition["properties"]["target_modules"]["minItems"] == 1
        assert valid_document()["adapter"]["target_modules"]


class TestTheManifestWrittenByTheRealWriterIsReadByTheRealReader:
    """A round trip, not a hand-built dict whose fields agree with the writer by
    construction - which is how a suite stays green over a real bug (ADR-016)."""

    def test_round_trip_through_save_and_load(self, tmp_path: Path) -> None:
        from tests.support.test_adapter_doubles import FakePeftModel
        from training.runtypes.adapters import save as adapter_save

        dest = adapter_paths.adapter_dir("run-7", 40, root=tmp_path)
        doc = adapter_manifest.build(
            run_id="run-7",
            step=40,
            base=adapter_manifest.base_block(hub_record(HUB_REPO)),
            adapter=adapter_config(rank=8, alpha=16.0, target_modules=("q_proj",)),
            optimizer=optimizer_block(),
        )
        published = adapter_save.save(
            FakePeftModel(rank=8, alpha=16.0, target_modules=("q_proj",)), doc, dest
        )
        read_back = adapter_load.load_manifest(dest / adapter_paths.MANIFEST_NAME)
        assert read_back == published
        assert read_back == json.loads((dest / adapter_paths.MANIFEST_NAME).read_text())
