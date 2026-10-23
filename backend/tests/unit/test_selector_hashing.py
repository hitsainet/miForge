"""Configuration identity  (AC-003.15 - AC-003.19; **R-02.10**, R-02.40, E-3).

Three things, each with a mutation control behind it:

* **the contract's hash is copied, byte-identical** - mutation control 2 recomputes it
  instead, and these tests must go red;
* **a local hash is labelled local** and never rendered as a contract reference - mutation
  control 5 mislabels it, and these tests must go red;
* **a composite publishes one scalar per kind**, its own.

Verification is tested as a *separate* thing from the recorded value, because the two
misreadings of R-02.10 pull in opposite directions: "copy it, therefore never check it"
would load a configuration whose bytes are known to disagree with its own manifest, and
"we digest it, therefore record our digest" mints a rival identity for a thing miDataworks
already named.
"""

from __future__ import annotations

import hashlib
import json

import pytest

from core.canonical import canonical_bytes, sha256_of
from core.errors import Refusal
from services.selectors import hashing
from services.selectors.types import SelectorForm
from tests.support.selector_stubs import (
    composite_config,
    handoff_version,
    resolved,
    rule_config,
)

CONTRACT_SHA = "a" * 64


class TestTheContractsHashIsCopied:
    def test_the_recorded_hash_is_byte_identical_to_the_contracts(self) -> None:
        """AC-003.15. **Mutation control 2 turns this red.**"""
        config = rule_config(min_length=3)
        result = hashing.contract_hash(config, handoff=handoff_version(CONTRACT_SHA))
        assert result.sha256 == CONTRACT_SHA
        assert result.provenance == "contract"

    def test_no_second_digest_is_recorded_beside_it(self) -> None:
        """R-02.10: miForge does not compute a second, competing hash of one configuration."""
        config = rule_config(min_length=3)
        result = hashing.contract_hash(config, handoff=handoff_version(CONTRACT_SHA))
        locally = sha256_of(hashing.body_for_hashing(config))
        assert locally != CONTRACT_SHA  # the two genuinely differ, so the test can fail
        assert result.sha256 != locally
        assert result.source is None

    def test_the_contract_reference_is_recorded_as_the_contract_states_it(self) -> None:
        result = hashing.contract_hash(
            rule_config(), handoff=handoff_version(CONTRACT_SHA, name="house", version=7)
        )
        assert result.contract_reference is not None
        assert result.contract_reference.name == "house"
        assert result.contract_reference.version == 7
        assert result.contract_reference.type == "selector_config"

    def test_an_incomplete_reference_is_omitted_rather_than_invented(self) -> None:
        """A partial reference rendered as complete is worse than none: a reader would try
        to resolve it."""
        handoff = handoff_version(CONTRACT_SHA)
        handoff["target"] = {"kind": "selector_config"}
        handoff["content"]["config"].pop("number")
        handoff["content"]["config"].pop("name")
        result = hashing.contract_hash(rule_config(), handoff=handoff)
        assert result.contract_reference is None
        assert result.sha256 == CONTRACT_SHA

    def test_a_version_with_no_config_hash_is_refused_not_hashed_locally(self) -> None:
        """Falling back to a local digest would silently convert a contract configuration
        into a local one and lose the provenance R-02.10 exists to preserve."""
        handoff = handoff_version(CONTRACT_SHA)
        handoff["content"]["config"].pop("sha256")
        with pytest.raises(Refusal) as exc:
            hashing.contract_hash(rule_config(), handoff=handoff)
        assert exc.value.code == "selector_config_contract_malformed"


class TestVerificationIsNotASecondHash:
    def test_disagreeing_bytes_are_refused_with_both_digests(self) -> None:
        """AC-003.16, E-3. Never recompute and proceed."""
        authored = b'{"min_length":3}'
        with pytest.raises(Refusal) as exc:
            hashing.contract_hash(
                rule_config(), handoff=handoff_version(CONTRACT_SHA), authored_bytes=authored
            )
        assert exc.value.code == "selector_config_hash_mismatch"
        assert exc.value.details["contract_declares"] == CONTRACT_SHA
        assert exc.value.details["computed"] == hashlib.sha256(authored).hexdigest()

    def test_agreeing_bytes_still_record_the_contracts_value(self) -> None:
        """The whole distinction: verification detects corruption, it does not mint an id."""
        authored = b'{"min_length":3}'
        digest = hashlib.sha256(authored).hexdigest()
        result = hashing.contract_hash(
            rule_config(), handoff=handoff_version(digest), authored_bytes=authored
        )
        assert result.sha256 == digest
        assert result.provenance == "contract"

    def test_a_configuration_citing_a_different_hash_is_refused(self) -> None:
        config = rule_config()
        config["config_sha256"] = "b" * 64
        with pytest.raises(Refusal) as exc:
            hashing.contract_hash(config, handoff=handoff_version(CONTRACT_SHA))
        assert exc.value.code == "selector_config_hash_mismatch"
        assert exc.value.details["configuration_cites"] == "b" * 64
        assert exc.value.details["contract_declares"] == CONTRACT_SHA


class TestALocalHashIsLabelledLocal:
    def test_provenance_is_local_and_the_source_is_recorded(self) -> None:
        """AC-003.17. **Mutation control 5 turns this red.**"""
        result = hashing.local_hash(rule_config(min_length=3), source="packs/legal/rule.json")
        assert result.provenance == "local"
        assert result.source == "packs/legal/rule.json"

    def test_a_local_hash_is_never_rendered_as_a_contract_reference(self) -> None:
        """R-02.40: miForge does not claim miDataworks provenance for anything miDataworks
        did not publish."""
        result = hashing.local_hash(rule_config(), source="inline")
        assert result.contract_reference is None

    def test_the_digest_goes_through_the_one_canonical_function(self) -> None:
        """ADR-005: two canonicalisers are two answers. Asserted by recomputing through
        ``core.canonical`` rather than by reading the module."""
        config = rule_config(min_length=3, forbidden_terms=["x"])
        result = hashing.local_hash(config, source="inline")
        expected = hashlib.sha256(canonical_bytes(hashing.body_for_hashing(config))).hexdigest()
        assert result.sha256 == expected

    def test_the_identity_field_is_excluded_from_its_own_digest(self) -> None:
        """A hash covering its own slot could never be reproduced by whoever authored it."""
        without = dict(rule_config(min_length=3))
        with_null = dict(without)
        with_null["config_sha256"] = None
        assert hashing.local_hash(without, source="x").sha256 == (
            hashing.local_hash(with_null, source="x").sha256
        )

    def test_a_local_hash_with_no_source_is_refused(self) -> None:
        """No default, so a local hash cannot be recorded without saying where it came
        from."""
        with pytest.raises(Refusal) as exc:
            hashing.local_hash(rule_config(), source="")
        assert exc.value.code == "selector_config_source_missing"

    def test_provenance_has_no_default_on_the_model(self) -> None:
        """ADR-018: a field defaulting to ``contract`` is a fabricated provenance claim the
        moment someone forgets to set it."""
        from services.selectors.types import ResolvedHash

        assert ResolvedHash.model_fields["provenance"].is_required()


class TestTheEntryPointDecidesProvenanceByInput:
    def test_a_handoff_gives_contract_provenance(self) -> None:
        result = hashing.resolve_hash(rule_config(), handoff=handoff_version(CONTRACT_SHA))
        assert result.provenance == "contract"

    def test_no_handoff_and_no_citation_gives_local(self) -> None:
        result = hashing.resolve_hash(rule_config(), source="inline")
        assert result.provenance == "local"

    def test_a_citation_without_a_version_is_still_the_contracts_value(self) -> None:
        """It IS the contract's hash; recomputing would mint a rival identity. But the
        reference is not claimed, because it is not known."""
        config = rule_config()
        config["config_sha256"] = CONTRACT_SHA
        result = hashing.resolve_hash(config, source="inline")
        assert (result.sha256, result.provenance) == (CONTRACT_SHA, "contract")
        assert result.contract_reference is None


class TestPublishingToTheWriters:
    def test_one_scalar_per_kind(self) -> None:
        """FR-003.16. ``checkpoint-manifest-v1.json`` holds a single nullable string each."""
        selector = hashing.local_hash(rule_config(), source="inline")
        published = hashing.publish(selector=selector, grader=None)
        assert published.selector_hash == selector.sha256
        assert published.grader_hash is None
        assert published.selector_hash_provenance == "local"

    def test_a_composite_publishes_its_own_hash_not_its_childrens(self) -> None:
        """AC-003.19. A list is not expressible in the manifest."""
        rule = rule_config("child-rule", min_length=3)
        ref = {
            "kind": "reference_match",
            "name": "child-ref",
            "config_sha256": None,
            "mode": "exact",
        }
        composite = composite_config("both", ["child-rule", "child-ref"], "all")
        selector = resolved(composite, catalog=[rule, ref])
        published = hashing.publish(selector=selector.hash, grader=None)

        own = hashing.local_hash(composite, source="inline").sha256
        child_hashes = {child.hash.sha256 for child in selector.children}
        assert published.selector_hash == own
        assert published.selector_hash not in child_hashes

    def test_the_manifest_fields_exist_in_the_vendored_schema(self) -> None:
        """Verified against the file rather than asserted from a document (FR-003.16)."""
        from pathlib import Path

        schema = json.loads(
            (
                Path(__file__).resolve().parents[3]
                / "docs"
                / "schemas"
                / "checkpoint-manifest-v1.json"
            ).read_text()
        )
        text = json.dumps(schema)
        assert '"selector_hash"' in text
        assert '"grader_hash"' in text


class TestTheResolverCarriesTheHash:
    def test_a_resolved_selector_exposes_its_hash_and_provenance(self) -> None:
        selector = resolved(rule_config(min_length=3))
        assert selector.config_sha256 == selector.hash.sha256
        assert selector.hash.provenance == "local"
        assert SelectorForm.SELECTOR in selector.capability.forms
