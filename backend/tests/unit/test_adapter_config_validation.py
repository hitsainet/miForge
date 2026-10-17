"""The adapter configuration's bounds, **read from the schema** (FR-008.22, E-7).

Every assertion below reads its bound out of ``adapter-manifest-v1.json``'s
``$defs.AdapterConfig`` at test time. A retyped list agrees with itself, which is the
fixture failure ADR-016 names - and it is not hypothetical in this chain: ``ScoreRecord``
had four properties when F8's documents were written and gained a fifth the same day.
"""

from __future__ import annotations

from typing import Any

import pytest

from core.errors import Refusal
from models.domain_state import ArtifactKind
from services.reindex import schema_violations
from training.runtypes.adapters import config as adapter_config_module
from training.runtypes.adapters import schema as adapter_schema
from training.runtypes.adapters.config import AdapterConfigIn, validate


def valid() -> dict[str, Any]:
    return {"rank": 8, "alpha": 16.0, "target_modules": ("q_proj", "v_proj"), "dropout": 0.05}


class TestTheBoundsComeFromTheSchema:
    def test_each_bound_matches_the_vendored_file(self) -> None:
        definition = adapter_schema.definition("AdapterConfig")["properties"]
        assert adapter_config_module.RANK_MINIMUM == definition["rank"]["minimum"]
        assert (
            adapter_config_module.ALPHA_EXCLUSIVE_MINIMUM == definition["alpha"]["exclusiveMinimum"]
        )
        assert (
            adapter_config_module.TARGET_MODULES_MIN_ITEMS
            == definition["target_modules"]["minItems"]
        )
        assert adapter_config_module.DROPOUT_MINIMUM == definition["dropout"]["minimum"]
        assert adapter_config_module.DROPOUT_MAXIMUM == definition["dropout"]["maximum"]

    def test_rank_uses_minimum_and_alpha_uses_exclusiveMinimum(self) -> None:
        """The asymmetry is the schema's and is preserved rather than flattened to "> 0"
        for both - which would admit ``rank: 0``."""
        assert adapter_config_module.RANK_MINIMUM == 1
        assert adapter_config_module.ALPHA_EXCLUSIVE_MINIMUM == 0
        validate({**valid(), "rank": 1})
        with pytest.raises(Refusal):
            validate({**valid(), "alpha": 0.0})

    def test_a_missing_definition_refuses_rather_than_relaxing_every_bound(self) -> None:
        with pytest.raises(Refusal) as caught:
            adapter_schema.definition("NoSuchDefinition")
        assert caught.value.code == "schema_definition_missing"


class TestEachViolationIsRefusedAndTheSchemaAgrees:
    """E-7, asserted twice: the edge model refuses, **and** the real schema refuses the
    manifest block that would have been written. Two layers, same answer."""

    @pytest.mark.parametrize(
        ("field", "value"),
        [
            ("rank", 0),
            ("rank", -1),
            ("alpha", 0.0),
            ("alpha", -3.0),
            ("target_modules", ()),
            ("dropout", -0.1),
            ("dropout", 1.5),
        ],
    )
    def test_the_edge_model_refuses(self, field: str, value: Any) -> None:
        with pytest.raises(Refusal) as caught:
            validate({**valid(), field: value})
        assert caught.value.code == "adapter_config_invalid"
        assert caught.value.details is not None
        assert "bounds" in caption_details(caught.value)

    @pytest.mark.parametrize(
        ("field", "value"),
        [("rank", 0), ("alpha", 0.0), ("target_modules", []), ("dropout", 1.5)],
    )
    def test_the_real_schema_refuses_the_same_block(self, field: str, value: Any) -> None:
        """The block an unvalidated config would have produced, against the live file."""
        block = {
            "rank": 8,
            "alpha": 16.0,
            "target_modules": ["q_proj"],
            "dropout": 0.05,
            "scaling": "alpha_over_r",
        }
        block[field] = value
        doc = {
            "kind": "miforge.adapter-manifest",
            "schema_version": "1",
            "name": "base:run:adapter-1",
            "run_id": "run",
            "step": 1,
            "base": {"name": "base", "manifest_sha256": None},
            "adapter": block,
            "weights_sha256": "a" * 64,
            "created_at": "2026-10-14T00:00:00Z",
        }
        assert schema_violations(doc, ArtifactKind.ADAPTER), f"{field}={value!r} must fail"


def caption_details(refusal: Refusal) -> dict[str, Any]:
    return refusal.details or {}


class TestTheConfigIsClosedAndFrozen:
    def test_an_unknown_key_is_refused_rather_than_ignored(self) -> None:
        """A misspelled ``lora_alpha`` would otherwise leave ``alpha`` at a value the
        operator did not choose and record it in a manifest as though they had."""
        with pytest.raises(Refusal) as caught:
            validate({**valid(), "lora_alpha": 32})
        assert caught.value.code == "adapter_config_invalid"

    def test_it_cannot_be_mutated_after_validation(self) -> None:
        config = validate(valid())
        from pydantic import ValidationError

        with pytest.raises(ValidationError):
            config.rank = 32  # type: ignore[misc]

    def test_an_already_validated_config_passes_through_unchanged(self) -> None:
        config = validate(valid())
        assert validate(config) is config


class TestTheManifestBlockAndTheScale:
    def test_the_block_carries_the_schemas_declared_scaling_default(self) -> None:
        block = validate(valid()).manifest_block()
        assert block["scaling"] == adapter_schema.scaling_default()
        assert block["scaling"] in adapter_schema.known_scaling()

    def test_the_block_validates_as_the_real_schemas_AdapterConfig(self) -> None:
        import jsonschema

        schema = adapter_schema.schema()
        validator = jsonschema.Draft202012Validator(
            {**schema["$defs"]["AdapterConfig"], "$defs": schema["$defs"]}
        )
        validator.validate(validate(valid()).manifest_block())

    def test_scaling_is_not_an_operator_input(self) -> None:
        """Nothing in this release implements a second rule, and an input the code ignores
        is worse than no input."""
        assert "scaling" not in AdapterConfigIn.model_fields

    def test_the_scale_is_alpha_over_rank_computed_in_one_place(self) -> None:
        assert validate({**valid(), "rank": 8, "alpha": 16.0}).scale == 2.0

    def test_dropout_defaults_to_the_schemas_own_default(self) -> None:
        config = validate({k: v for k, v in valid().items() if k != "dropout"})
        assert config.dropout == adapter_config_module.DROPOUT_DEFAULT
        assert config.dropout == adapter_schema.bound("AdapterConfig", "dropout", "default")
