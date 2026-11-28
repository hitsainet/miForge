"""Configuration identity  (FR-003.14, FR-003.15, FR-003.16; **R-02.10**, ADR-005).

R-02.10 requires the selector's and grader's configuration hash to be recorded in every
batch and every checkpoint manifest, and **the hash recorded is the handoff contract's, not
one miForge invents.**

## Verification is not a second hash - and both misreadings are guarded here

Two readings of R-02.10 would each break something, so both are named:

* **"Copy it, therefore never check it."** That would mean loading a configuration whose
  bytes are known to disagree with its own manifest - a corrupted selector silently
  changing every verdict in a run whose manifest says it was something else. So where the
  authored bytes are in hand they are digested **only to compare**, and a mismatch refuses
  with both digests (E-3).
* **"We digest it, therefore just record our digest."** That mints a rival identity for a
  thing miDataworks already named, and then two repos disagree about what one configuration
  is. So the **recorded value is always the contract's**, verbatim.

The next reader will want to simplify one of these into the other. The comments and the
tests exist so that neither simplification is quiet. Mutation control 2 - recompute instead
of copying - must go red.

## Provenance is returned, never defaulted

:class:`services.selectors.types.ResolvedHash` has no default for ``provenance``, and no
function in this module takes ``provenance="contract"`` as a keyword default. ADR-018: a
default that always fires is a fabricated value, and R-02.40 forbids miForge claiming
miDataworks provenance for anything miDataworks did not publish. Mutation control 5
mislabels a local hash as ``contract`` and must go red.
"""

from __future__ import annotations

import hashlib
from typing import Any

from core.canonical import canonical_bytes, sha256_of
from core.errors import refuse
from services.selectors.types import (
    ContractReference,
    HashProvenance,
    PublishedHashes,
    ResolvedHash,
)

#: The one field of a selector configuration that is *about* the configuration's identity
#: rather than part of its body. Excluded from a locally computed digest, because a hash
#: that covered its own slot could never be reproduced by whoever authored the body.
_IDENTITY_FIELD = "config_sha256"


def body_for_hashing(config: dict[str, Any]) -> dict[str, Any]:
    """The configuration body as it is hashed: everything but its own hash field.

    A named function rather than an inline ``dict`` comprehension so the choice is visible
    and testable. Changing it changes every local digest in the product.
    """
    return {key: value for key, value in config.items() if key != _IDENTITY_FIELD}


def local_hash(config: dict[str, Any], *, source: str) -> ResolvedHash:
    """Hash a locally supplied configuration and **label it local** (FR-003.15).

    ``source`` is required and keyword-only: there is no default, so a local hash cannot be
    recorded without saying where it came from (R-02.40's honesty rule). Uses the
    foundation's one canonical-JSON function (ADR-005) - **no local canonicaliser, not even
    a two-line one**, because two canonicalisers are two answers and the digest is the
    thing being agreed about.

    Args:
        config: The configuration body, including its ``kind`` and ``name``.
        source: Where it came from - ``"inline"`` for F7's dry-run, a path or a URL for an
            external part (R-02.40).

    Returns:
        A :class:`ResolvedHash` with ``provenance="local"`` and **no contract reference**.
        A local hash is never rendered or recorded as a contract reference.
    """
    if not source:
        raise refuse(
            "selector_config_source_missing",
            "A locally supplied selector configuration must say where it came from "
            "before its hash can be recorded. Pass the path, URL or 'inline' as the "
            "source: miForge does not claim provenance it was not given.",
        )
    digest = sha256_of(body_for_hashing(config))
    return ResolvedHash(sha256=digest, provenance="local", contract_reference=None, source=source)


def contract_hash(
    config: dict[str, Any],
    *,
    handoff: dict[str, Any],
    authored_bytes: bytes | None = None,
) -> ResolvedHash:
    """Take the hash from the handoff contract, **unchanged** (FR-003.14).

    Reads ``content.config.sha256`` from a ``midataworks.dataset-version/v1`` version and
    returns it verbatim, with the contract reference beside it. The vendored schema's own
    description says miForge copies it; this function is that sentence.

    Args:
        config: The selector configuration body. If it carries ``config_sha256``, it must
            agree with the contract's - a configuration citing a different hash from the
            version that delivered it is a mismatch, not a preference.
        handoff: The handoff version document.
        authored_bytes: The authored configuration bytes, when miForge holds them. Digested
            **only to compare**; the recorded value stays the contract's.

    Raises:
        Refusal: ``selector_config_contract_malformed`` when the version does not carry a
            config hash - refused rather than falling back to a local digest, which would
            silently convert a contract configuration into a local one and lose the
            provenance that R-02.10 exists to preserve.
        Refusal: ``selector_config_hash_mismatch`` with **both digests**, for a disagreeing
            citation or disagreeing bytes.
    """
    content = handoff.get("content") or {}
    contract_config = content.get("config") or {}
    declared = contract_config.get("sha256")
    if not isinstance(declared, str) or not declared:
        raise refuse(
            "selector_config_contract_malformed",
            "The miDataworks handoff version carries no content.config.sha256, so there "
            "is no contract hash to record. Refusing rather than computing one: a locally "
            "minted digest recorded as contract-sourced would claim a provenance it "
            "does not have.",
            target=handoff.get("target"),
        )

    cited = config.get(_IDENTITY_FIELD)
    if isinstance(cited, str) and cited and cited != declared:
        raise refuse(
            "selector_config_hash_mismatch",
            "The selector configuration cites a different hash from the handoff version "
            "that delivered it. Reconcile the configuration with its manifest; miForge "
            "will not choose between two identities for one configuration.",
            configuration_cites=cited,
            contract_declares=declared,
        )

    if authored_bytes is not None:
        computed = hashlib.sha256(authored_bytes).hexdigest()
        if computed != declared:
            raise refuse(
                "selector_config_hash_mismatch",
                "The configuration bytes in hand do not hash to the sha256 the handoff "
                "version declares. Refusing: loading a configuration whose bytes disagree "
                "with its own manifest would change every verdict in the run while the "
                "manifest said it was something else. Re-fetch the configuration version.",
                computed=computed,
                contract_declares=declared,
            )

    target = handoff.get("target") or {}
    reference = _contract_reference(target, contract_config)
    # THE RECORDED VALUE IS THE CONTRACT'S. `declared`, not `computed`. If a future edit
    # makes this line return a digest computed here, R-02.10 is broken and mutation
    # control 2 is the test that says so.
    return ResolvedHash(
        sha256=declared, provenance="contract", contract_reference=reference, source=None
    )


def _contract_reference(
    target: dict[str, Any], contract_config: dict[str, Any]
) -> ContractReference | None:
    """``{type, name, version}`` as the **contract** states them (BRD-03 R-03.47).

    Returns ``None`` rather than inventing fields when the version does not name all
    three - an incomplete reference rendered as though it were complete is worse than
    none, because a reader would try to resolve it.
    """
    kind = target.get("kind") or target.get("type")
    name = target.get("name") or contract_config.get("name")
    version = target.get("version")
    if version is None:
        version = contract_config.get("number")
    if kind is None or name is None or version is None:
        return None
    return ContractReference(type=str(kind), name=str(name), version=version)


def publish(*, selector: ResolvedHash | None, grader: ResolvedHash | None) -> PublishedHashes:
    """What F2's batch record and F5's manifest write get (FR-003.16).

    **One scalar per kind.** A composite publishes **its own** hash, not a list of its
    children's: ``checkpoint-manifest-v1.json`` holds ``selector_hash`` and ``grader_hash``
    as single nullable strings - verified against the vendored file - so a list is not
    expressible there. Children are enumerated in the batch record instead.
    """
    return PublishedHashes(
        selector_hash=selector.sha256 if selector else None,
        grader_hash=grader.sha256 if grader else None,
        selector_hash_provenance=selector.provenance if selector else None,
        grader_hash_provenance=grader.provenance if grader else None,
        selector_contract_reference=selector.contract_reference if selector else None,
        grader_contract_reference=grader.contract_reference if grader else None,
    )


def resolve_hash(
    config: dict[str, Any],
    *,
    handoff: dict[str, Any] | None = None,
    source: str | None = None,
    authored_bytes: bytes | None = None,
) -> ResolvedHash:
    """The one entry point: contract hash when there is a contract, local otherwise.

    ``provenance`` is decided by **which input was given**, never by a parameter default -
    so there is no keyword a caller can forget that would silently claim contract
    provenance.

    Args:
        config: The selector configuration body.
        handoff: A miDataworks handoff version, when the configuration came from one.
        source: Where a local configuration came from. Required when ``handoff`` is absent.
        authored_bytes: The authored bytes, for verification against the contract.
    """
    if handoff is not None:
        return contract_hash(config, handoff=handoff, authored_bytes=authored_bytes)
    if config.get(_IDENTITY_FIELD):
        # The configuration cites a contract hash but no version was supplied to verify it
        # against. Record the citation as the contract's - it IS the contract's, and
        # recomputing would mint a rival identity - but refuse to pretend we know the
        # reference.
        return ResolvedHash(
            sha256=str(config[_IDENTITY_FIELD]),
            provenance="contract",
            contract_reference=None,
            source=None,
        )
    return local_hash(config, source=source or "")


def canonical_config_bytes(config: dict[str, Any]) -> bytes:
    """The bytes a local digest covers, exposed for tests and for F2's batch record.

    Goes through the foundation's :func:`core.canonical.canonical_bytes`, so what is hashed
    here and what is written anywhere else cannot drift.
    """
    return canonical_bytes(body_for_hashing(config))


HashProvenanceLiteral = HashProvenance
