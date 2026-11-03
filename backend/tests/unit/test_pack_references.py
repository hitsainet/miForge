"""The two source kinds, and the absence that keeps provenance honest  (FR-007.6, FR-007.24).

FAC-6's first half: **no code path gives an external part a contract-reference shape.** The
mechanism is that no conversion exists, so the test asserts an **absence** - which is the
only form of assertion that can cover "nobody wrote the convenience function".

Mutation control (task 14.3a) for this file: set ``source="midataworks"`` on an external
reference. ``test_the_discriminator_decides_which_type_is_built`` and
``test_an_external_reference_cannot_be_built_with_contract_fields`` must go red.
"""

from __future__ import annotations

from datetime import UTC, datetime

import pytest
from pydantic import ValidationError

from packs import contracts
from packs.references import PART_REF_ADAPTER, ContractPartRef, ExternalPartRef

NOW = datetime(2026, 10, 14, tzinfo=UTC)


class TestTheDiscriminatedUnion:
    def test_the_discriminator_decides_which_type_is_built(self) -> None:
        contract = PART_REF_ADAPTER.validate_python(
            {"source": "midataworks", "type": "prompt-set", "name": "tickets", "version": "3"}
        )
        external = PART_REF_ADAPTER.validate_python(
            {
                "source": "external",
                "origin": "/data/mine.jsonl",
                "content_sha256": "a" * 64,
                "taken_in_at": "2026-10-14T00:00:00Z",
            }
        )
        assert isinstance(contract, ContractPartRef)
        assert isinstance(external, ExternalPartRef)

    def test_an_unknown_source_is_refused_rather_than_defaulted(self) -> None:
        with pytest.raises(ValidationError):
            PART_REF_ADAPTER.validate_python({"source": "midstudio", "type": "x"})

    def test_an_external_reference_cannot_be_built_with_contract_fields(self) -> None:
        """``extra="forbid"``: an import carrying extra keys refuses rather than narrowing.

        A narrowed import is a pack that claims to be the one exported and is not - and the
        keys most likely to arrive are exactly a contract reference's.
        """
        with pytest.raises(ValidationError):
            ExternalPartRef(
                origin="/data/mine.jsonl",
                content_sha256="a" * 64,
                taken_in_at=NOW,
                type="prompt-set",  # type: ignore[call-arg]
            )

    @pytest.mark.parametrize(
        "reference",
        [
            ContractPartRef(type="prompt-set", name="tickets", version="3"),
            ExternalPartRef(origin="/x", content_sha256="a" * 64, taken_in_at=NOW),
        ],
    )
    def test_a_pinned_reference_is_frozen(self, reference: object) -> None:
        """A pinned reference that can be mutated is not pinned (FR-007.2, FR-007.4)."""
        with pytest.raises(ValidationError):
            reference.type = "other"  # type: ignore[attr-defined]


class TestThereIsNoConversionPath:
    """FAC-6's first half, asserted as an **absence**.

    A test that checked "export does not promote" would cover one path. This covers every
    path, because a conversion that does not exist cannot be called from any of them.
    """

    FORBIDDEN = (
        "to_contract",
        "as_contract_reference",
        "from_external",
        "from_contract",
        "promote",
        "to_midataworks",
        "as_midataworks",
        "to_external",
    )

    @pytest.mark.parametrize("name", FORBIDDEN)
    def test_neither_type_offers_a_conversion(self, name: str) -> None:
        assert not hasattr(ContractPartRef, name), (
            f"ContractPartRef.{name} exists. R-02.40 forbids miForge claiming miDataworks "
            f"provenance for anything miDataworks did not publish, and the mechanism is "
            f"that there is nothing to call."
        )
        assert not hasattr(ExternalPartRef, name), f"ExternalPartRef.{name} exists."

    def test_the_contract_renderer_refuses_an_external_reference(self) -> None:
        """``as_contract_dict`` exists only on the contract type - the typing is the guard."""
        assert hasattr(ContractPartRef, "as_contract_dict")
        assert not hasattr(ExternalPartRef, "as_contract_dict")

    def test_neither_type_has_a_content_field(self) -> None:
        """FR-007.4 by **absence**: there is nowhere to put a copy of a part's rows.

        Copies fork. Two installations with "the same pack" would hold two datasets
        diverging silently, which is the failure the handoff contract exists to prevent.
        """
        for model in (ContractPartRef, ExternalPartRef):
            for field in model.model_fields:
                assert field not in {"rows", "content", "prompts", "text", "data", "body"}


class TestTheFieldsAreTheContractS:
    def test_the_contract_reference_fields_are_exactly_theirs(self) -> None:
        """Read from the vendored file, never a copied list - a copy agrees with itself."""
        theirs = contracts.at(contracts.BATCH_RECORD_SCHEMA, "$defs", "ContractReference")
        assert set(ContractPartRef.model_fields) - {"source"} == set(theirs["required"])
        assert set(ContractPartRef.model_fields) - {"source"} == set(theirs["properties"])

    def test_the_external_provenance_keys_are_exactly_theirs(self) -> None:
        theirs = contracts.at(contracts.BATCH_RECORD_SCHEMA, "$defs", "ExternalProvenance")
        produced = ExternalPartRef(
            origin="/x", content_sha256="a" * 64, taken_in_at=NOW
        ).as_external_dict()
        assert set(theirs["required"]) <= set(produced)
        assert set(produced) <= set(theirs["properties"])

    def test_there_is_no_local_version_field(self) -> None:
        """FPRD OQ-2's invented ordinal, **answered from the contract** instead.

        ``ExternalProvenance`` is closed on ``{source, content_sha256, retrieved_at,
        licence}``, so a pack document could not carry an ordinal, and one living only in
        this installation's database would not survive the export AC10 requires. The content
        hash is the pin.
        """
        assert "local_version" not in ExternalPartRef.model_fields

    def test_the_digest_must_be_a_digest(self) -> None:
        with pytest.raises(ValidationError):
            ExternalPartRef(origin="/x", content_sha256="not-a-digest", taken_in_at=NOW)

    def test_a_naive_timestamp_is_refused(self) -> None:
        """Every timestamp miForge writes is UTC-aware; a naive one is an unanswerable "when"."""
        with pytest.raises(ValidationError):
            ExternalPartRef(
                origin="/x",
                content_sha256="a" * 64,
                taken_in_at=datetime(2026, 10, 14),
            )
