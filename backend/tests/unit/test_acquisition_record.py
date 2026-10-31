"""The acquisition record and the two lineage fields  (R-02.47; FR-001.9 - FR-001.12).

**The defect this file exists for.** F1's FR-001.12, read literally, said a ``load`` sets
``base_model`` to the loaded model's identity. That is right for a first load and **wrong for
every continue and every branch**, because the thing loaded *is* a checkpoint - so the root
becomes the parent and **every lineage walk terminates one hop short of the real root**. On a
long chain of continues the tree is rooted at the previous step rather than at the model the
work started from.

It validates. It saves. The tree looks plausible. Only reading F5 against F1 found it, which
is why the carry-forward is a **named function** with a test pointed at it rather than a line
inside a larger one.

The schema's own description of the field is *"the root model this lineage descends from"*,
and BRD-01 R-01.16 walks *"back to ``base_model``"* - so the two sentences that make this a
defect are both in documents outside this feature.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from core.errors import Refusal
from training.acquisition.base import (
    AcquisitionMode,
    AcquisitionRecord,
    SourceKind,
    lineage_for_creation,
    lineage_for_first_load,
    lineage_for_loaded_checkpoint,
)

SCHEMA = Path(__file__).resolve().parents[3] / "docs" / "schemas" / "checkpoint-manifest-v1.json"


def record(**overrides) -> AcquisitionRecord:
    base = {
        "mode": AcquisitionMode.LOAD,
        "source_kind": SourceKind.LOCAL_DIR,
        "source_path": "/data/checkpoints/base",
        "starting_weights_sha256": "a" * 64,
        "base_model": "local:base@abcdef",
        "parent": None,
    }
    base.update(overrides)
    return AcquisitionRecord(**base)  # type: ignore[arg-type]


class TestTheCarryForward:
    """FR-001.12 as corrected. **The whole defect, in three assertions.**"""

    def test_loading_a_CHECKPOINT_carries_its_base_model_forward_unchanged(self) -> None:
        base_model, parent = lineage_for_loaded_checkpoint(
            checkpoint_name="run-1-000500", checkpoint_base_model="hf:org/model@abc123"
        )
        assert base_model == "hf:org/model@abc123", (
            "the loaded checkpoint's OWN base_model is the root; substituting the checkpoint "
            "would terminate every lineage walk one hop short of it"
        )
        assert parent == "run-1-000500"

    def test_a_chain_of_continues_keeps_ONE_root(self) -> None:
        """The failure the literal reading produces, exercised over four hops. Each step's
        ``parent`` advances; the root must not."""
        root = "hf:org/model@abc123"
        base_model = root
        for step in (100, 200, 300, 400):
            base_model, parent = lineage_for_loaded_checkpoint(
                checkpoint_name=f"run-1-{step:06d}", checkpoint_base_model=base_model
            )
            assert parent == f"run-1-{step:06d}"
        assert (
            base_model == root
        ), "after four continues the root is still the model the work started from"

    def test_a_FIRST_load_is_itself_the_root_and_has_no_parent(self) -> None:
        """A local Hugging Face directory or a Hub repo at a revision *is* the root, and the
        schema permits a null ``parent``."""
        assert lineage_for_first_load(model_identity="hf:org/model@abc") == (
            "hf:org/model@abc",
            None,
        )

    def test_a_creation_is_parent_null_with_base_model_the_config(self) -> None:
        """R-02.47, and the schema already allowed it - **no schema change was ever
        needed**."""
        assert lineage_for_creation(config_identity="config:llama@deadbeef+seed=1") == (
            "config:llama@deadbeef+seed=1",
            None,
        )

    def test_a_checkpoint_with_no_base_model_REFUSES_rather_than_substituting(self) -> None:
        """Substituting this checkpoint would root the tree at the parent - which is exactly
        the defect, arriving through a missing field instead of through a misreading."""
        with pytest.raises(Refusal) as caught:
            lineage_for_loaded_checkpoint(checkpoint_name="run-1-000500", checkpoint_base_model="")
        assert caught.value.code == "checkpoint_base_model_missing"
        assert "one hop short" in caught.value.message


class TestTheFieldsMatchTheVendoredSchema:
    """F1 derives the two lineage fields; **F5 writes them into a manifest** (R-02.20)."""

    def test_base_model_is_a_required_STRING_in_the_schema(self) -> None:
        """Which is why the ``create`` seed is folded into the identity string rather than
        added as a field: the manifest is closed to unknown keys, so a new field would be a
        schema change in two repos."""
        document = json.loads(SCHEMA.read_text(encoding="utf-8"))
        manifest = document["$defs"]["CheckpointManifestV1"]
        assert manifest["properties"]["base_model"]["type"] == "string"
        assert "base_model" in manifest["required"]
        assert manifest["additionalProperties"] is False

    def test_parent_is_nullable_and_required(self) -> None:
        """Null is an assertion - *a checkpoint saved directly off the base model* - and an
        absent key would be an omission."""
        document = json.loads(SCHEMA.read_text(encoding="utf-8"))
        manifest = document["$defs"]["CheckpointManifestV1"]
        assert {"type": "null"} in manifest["properties"]["parent"]["anyOf"]
        assert "parent" in manifest["required"]

    def test_the_schemas_description_is_what_makes_the_carry_forward_correct(self) -> None:
        """Pinned here so a future reader can check the claim against its source rather than
        re-reading - the method that caught every defect this project has found."""
        document = json.loads(SCHEMA.read_text(encoding="utf-8"))
        description = document["$defs"]["CheckpointManifestV1"]["properties"]["base_model"][
            "description"
        ]
        assert "root model this lineage descends from" in description

    def test_F1_writes_no_manifest(self) -> None:
        """R-02.20 is F5's. F1 hands over the two fields and writes nothing - a second
        manifest writer is a second answer about what a checkpoint claims."""
        import ast

        src = Path(__file__).resolve().parents[3] / "backend" / "src" / "training"
        offenders: list[str] = []
        for path in src.rglob("*.py"):
            if "hotupdate" in path.parts:
                continue
            for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
                if isinstance(node, ast.Constant) and node.value == "manifest.json":
                    offenders.append(f"{path.relative_to(src)}:{node.lineno}")
        assert not offenders, f"F1 must write no checkpoint manifest; found {offenders}"


#: Credential-shaped field names, by **segment**. ``tokenizer_source`` splits to
#: ``{"tokenizer", "source"}`` and matches nothing; ``hf_token`` splits to ``{"hf",
#: "token"}`` and matches. Plural forms are listed because ``access_tokens`` is as much a
#: credential as ``access_token``.
CREDENTIAL_SEGMENTS = frozenset(
    {"token", "tokens", "secret", "secrets", "password", "passwords", "key", "keys", "apikey"}
)


def _suspicious(names) -> set[str]:
    return {name for name in names if set(str(name).split("_")) & CREDENTIAL_SEGMENTS}


class TestTheRecordIsClosedAndCarriesNoSecret:
    def test_an_unknown_field_is_refused(self) -> None:
        from pydantic import ValidationError

        with pytest.raises(ValidationError):
            record(hf_token="secret")  # type: ignore[call-arg]

    def test_no_token_field_exists_anywhere_on_the_record(self) -> None:
        """ADR-020: nothing secret in a run log, and the record goes into one. A path, a
        repo, a revision and a hash are not secrets; a Hub token is.

        **The predicate matches name segments, not substrings, and it is proved able to
        fail below.** A bare ``"token" in name`` flagged ``tokenizer_source`` and
        ``tokenizer_vocab_size`` the day they were added (2026-10-16) - a *tokenizer* is a
        vocabulary, not a credential, so that was the sixth time in this project a fresh
        checker disagreed with correct work. Loosening a guard is only honest with a
        positive control attached, so the control is in the test rather than in a note
        about it.
        """
        assert _suspicious(AcquisitionRecord.model_fields) == set()

    def test_the_secret_name_check_IS_ABLE_TO_FAIL(self) -> None:
        """The negative control for the predicate above. Without this, segment matching
        could be narrowed to nothing and the suite would stay green."""
        assert _suspicious(["hf_token", "api_key", "hub_password", "client_secret"]) == {
            "hf_token",
            "api_key",
            "hub_password",
            "client_secret",
        }
        assert _suspicious(["tokenizer_source", "tokenizer_vocab_size", "source_revision"]) == set()

    def test_no_prompt_or_corpus_field_exists(self) -> None:
        """The acquisition record describes *weights*. Batch content belongs to F2's batch
        record, which carries identifiers and hashes and never text."""
        suspicious = {
            name
            for name in AcquisitionRecord.model_fields
            if any(word in name for word in ("prompt", "corpus", "completion", "text"))
        }
        assert suspicious == set()

    def test_the_record_is_frozen(self) -> None:
        """It is the run's lineage root, persisted before the first step; a record that could
        be edited afterwards is not a record."""
        import pydantic

        with pytest.raises(pydantic.ValidationError):
            record().mode = AcquisitionMode.CREATE  # type: ignore[misc]


class TestEachModeCarriesExactlyWhatRTwoFortySevenNamesForIt:
    """*"path or repo-and-revision for a load, config identity for a creation"* - both
    halves required, not one or the other."""

    def test_a_local_load_names_the_path(self) -> None:
        payload = record().model_dump(mode="json")
        assert payload["source_path"] == "/data/checkpoints/base"
        assert payload["source_repo"] is None
        assert payload["config_identity"] is None
        assert payload["starting_weights_sha256"]

    def test_a_hub_load_names_the_repo_AND_BOTH_revisions(self) -> None:
        payload = record(
            source_kind=SourceKind.HUB,
            source_path=None,
            source_repo="org/model",
            source_revision_requested="main",
            source_revision_resolved="c" * 40,
        ).model_dump(mode="json")
        assert payload["source_repo"] == "org/model"
        assert payload["source_revision_requested"] == "main"
        assert payload["source_revision_resolved"] == "c" * 40
        assert payload["source_revision_requested"] != payload["source_revision_resolved"], (
            "a branch and the commit it resolved to are different facts; keeping only one "
            "makes a moved branch a silent change of what was trained"
        )

    def test_a_creation_names_the_config_identity_and_the_seed(self) -> None:
        payload = record(
            mode=AcquisitionMode.CREATE,
            source_kind=SourceKind.CONFIG,
            source_path=None,
            config_identity="d" * 64,
            seed=42,
            base_model="config:llama@" + "d" * 64 + "+seed=42",
        ).model_dump(mode="json")
        assert payload["config_identity"] == "d" * 64
        assert payload["seed"] == 42
        assert payload["parent"] is None

    def test_a_null_seed_is_SERIALISED_rather_than_omitted(self) -> None:
        """``GenerationParams.seed``'s precedent: *null is an assertion that no seed was
        set*, which makes the starting hash irreproducible - and that is worth recording as
        a fact rather than hiding."""
        payload = record(mode=AcquisitionMode.CREATE, seed=None).model_dump(mode="json")
        assert "seed" in payload and payload["seed"] is None
