"""The two source kinds, and the absence that keeps provenance honest  (task 2.1).

**FR-007.6, FR-007.24, R-02.35, R-02.40.** A filled slot holds a reference to a part and
the part's pinned version. It never holds the part's rows, prompts, text, rule terms or
judge template - and that is enforced by **absence**: neither type below has a content
field, so there is nowhere to put a copy. Copies fork. Two installations with "the same
pack" would hold two datasets diverging silently, which is the failure the handoff contract
exists to prevent.

## Why two types and not one type with a flag

A function that can be handed either and cannot tell them apart is how provenance gets
borrowed. So ``midataworks`` and ``external`` are **distinct models joined by a Pydantic
discriminated union**, and:

**There is no conversion function, in either direction.** No ``to_contract()``, no
``from_external()``, no ``promote()``, no ``as_contract_reference()``, no ``__init__``
accepting the other's fields. If a caller needs a contract reference it must already have
one. *That absence is the mechanism of FR-007.24* - R-02.40 forbids miForge claiming
miDataworks provenance for anything miDataworks did not publish, and a rule stated once in
prose is how two earlier defects in this project happened. A test asserts no attribute or
classmethod on either type converts to the other, and a negative control flips an external
reference's ``source`` to prove the test bites.

**:class:`ContractPartRef`'s three fields are exactly the vendored schema's
``$defs.ContractReference`` required list** - ``{type, name, version}``, all three required,
each non-empty - checked against the file at import below rather than recalled. A fourth
field here would be miForge inventing contract surface, which is the shape of this project's
recurring defect.
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Annotated, Any, Literal

from pydantic import AwareDatetime, BaseModel, ConfigDict, Field, TypeAdapter

from core.errors import refuse
from packs import contracts


class ContractPartRef(BaseModel):
    """A part published by miDataworks, cited by its contract reference (BRD-03 R-03.47).

    ``frozen`` because a pinned reference that can be mutated is not pinned (FR-007.2,
    FR-007.4), and ``extra="forbid"`` because an import carrying extra keys must be refused
    rather than silently narrowed - a narrowed import is a pack that claims to be the one
    exported and is not.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    source: Literal["midataworks"] = "midataworks"
    type: str = Field(min_length=1)
    name: str = Field(min_length=1)
    version: str = Field(min_length=1)

    def as_contract_dict(self) -> dict[str, str]:
        """``{type, name, version}`` - **their** shape, with no ``source`` key.

        This is not a conversion: it is the same reference, rendered for the field their
        contract defines. It is deliberately the only serialisation of this type that drops
        ``source``, because the discriminator is miForge's layering and is not part of
        miDataworks' reference.
        """
        return {"type": self.type, "name": self.name, "version": self.version}


class ExternalPartRef(BaseModel):
    """A part from outside the suite, carrying its own provenance (R-02.40).

    R-02.40 asks for exactly three things - where it came from, a content hash, and when it
    was taken in - and these are those three, in the shape the contract's
    ``ExternalProvenance`` has them.

    **There is no ``local_version``, and that answers FPRD OQ-2 from the contract rather than
    by invention.** The FPRD assigned a miForge-local ordinal so that FR-007.4's "every slot
    pins a version" would hold; ``ExternalProvenance`` is closed on
    ``{source, content_sha256, retrieved_at, licence}``, so there is nowhere in the artifact
    to put one, and a pin that lived only in this installation's database would not survive
    the export AC10 requires. **The content hash is the pin** - a stronger one than an
    ordinal, because it identifies the bytes rather than their arrival order. Re-ingesting
    the same origin with different bytes yields a *different* reference, which is exactly
    what the ordinal was invented to achieve.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    source: Literal["external"] = "external"
    #: The path or location **as the operator gave it, verbatim**. Not normalised, not
    #: symlink-resolved, not rewritten: R-02.40 asks where it came from, and a rewritten
    #: path is a different answer. It is operator-supplied text, so it is escaped on
    #: display and never interpreted.
    origin: str = Field(min_length=1)
    content_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    taken_in_at: AwareDatetime

    def as_external_dict(self) -> dict[str, Any]:
        """The provenance in the shape ``batch-record-v1.json``'s ``ExternalProvenance`` has.

        Their field names, not these: ``source`` for the origin and ``retrieved_at`` for the
        time. Mapped here, once, rather than at each writer - and ``licence`` is **omitted
        rather than guessed**, because the contract makes it nullable and an invented
        licence is worse than an absent one.

        ``licence`` is **omitted rather than guessed** for the same reason the ordinal does
        not exist: the contract makes it nullable, and an invented licence is worse than an
        absent one.
        """
        return {
            "source": self.origin,
            "content_sha256": self.content_sha256,
            "retrieved_at": self.taken_in_at.isoformat().replace("+00:00", "Z"),
        }


#: A reference of either kind. Discriminated on ``source``, so a caller must handle both -
#: a bare union would let one branch be forgotten and the forgotten branch is always the
#: external one.
PartRef = Annotated[ContractPartRef | ExternalPartRef, Field(discriminator="source")]

PART_REF_ADAPTER: TypeAdapter[ContractPartRef | ExternalPartRef] = TypeAdapter(PartRef)


def _assert_contract_fields_match_the_schema() -> None:
    """``ContractPartRef``'s fields are exactly their ``ContractReference``'s. **At import.**

    Raises:
        Refusal: ``contract_reference_fields_drifted`` naming the difference. Checked here
            rather than in a test because the failure mode is miForge inventing contract
            surface, and an import-time failure cannot be skipped.
    """
    theirs = contracts.at(contracts.BATCH_RECORD_SCHEMA, "$defs", "ContractReference")
    required = set(theirs.get("required", []))
    declared = set(theirs.get("properties", {}))
    ours = set(ContractPartRef.model_fields) - {"source"}
    if ours != required or ours != declared:
        raise refuse(
            "contract_reference_fields_drifted",
            f"batch-record-v1.json's ContractReference declares {sorted(declared)} "
            f"(required {sorted(required)}) and ContractPartRef carries {sorted(ours)}. "
            f"The contract is theirs: miForge cites it and adds nothing. Failing at import "
            f"rather than writing a reference with a field they do not accept.",
            contract_properties=sorted(declared),
            contract_required=sorted(required),
            ours=sorted(ours),
        )

    theirs_external = contracts.at(contracts.BATCH_RECORD_SCHEMA, "$defs", "ExternalProvenance")
    required_external = set(theirs_external.get("required", []))
    # A real instance, through the real model, so the check runs over what the method
    # actually emits rather than over a list of key names written twice.
    produced = set(
        ExternalPartRef(
            origin="(import-time self-check)",
            content_sha256="0" * 64,
            taken_in_at=datetime(2026, 1, 1, tzinfo=UTC),
        ).as_external_dict()
    )
    if not required_external <= produced:
        raise refuse(
            "contract_reference_fields_drifted",
            f"ExternalProvenance requires {sorted(required_external)} and this module "
            f"produces {sorted(produced)}. An external part that cannot state its own "
            f"provenance in the contract's shape could not appear in a batch record, and "
            f"nothing downstream could say where a behaviour came from.",
            contract_required=sorted(required_external),
            ours=sorted(produced),
        )


_assert_contract_fields_match_the_schema()
