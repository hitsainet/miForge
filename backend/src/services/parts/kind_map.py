"""Which contract targets miForge consumes, and which it refuses by name  (F10 task 3.4-3.6).

FR-010.15. **The enums are read from the vendored schema, never re-typed in Python.** That
one decision is the whole mechanism: when miDataworks adds a target kind, the
exhaustiveness check below fails **at import** - loudly, at startup - instead of the new
value falling through a default and being treated as something it is not. ADR-018's rule
in its structural form: *a default that always fires is a fabricated value.*

A hard-coded Python enum would simply not know.

## The two axes, both true, describing different things

BRD-03 R-03.48 lists **six bullets** of sets miForge consumes, and the sixth is "selector
and grader configurations" - two kinds in one bullet. The schema has **five**
``miforge_set_kind`` values; selector and grader configurations are not among them, because
they are their own ``target.kind`` values carrying ``content.content_kind: "config"``.

So "six set kinds" and "five ``miforge_set_kind`` values" are both correct and are not
about the same axis. This module says so in code rather than picking one and being half
wrong: :data:`MIFORGE_SET_KINDS` is the inner axis, :data:`CONSUMED_TARGET_KINDS` the outer.

## The part-kind vocabulary is the gate contract's

:data:`PART_KINDS` is exactly ``PartReference.kind``'s enum in
``docs/schemas/batch-record-v1.json`` - ``prompt_set``, ``corpus``, ``preference_pairs``,
``test_set``, ``retention_set``, ``selector_config``, ``grader_config``. Settled there
because the batch record is a gate contract, so this module maps **onto** it and declares
no vocabulary of its own.

*(Note for a reader comparing tables: ``models.pack.PART_KINDS`` spells its slot kinds with
hyphens - ``prompt-set`` - and includes ``selector``/``grader``. That is F7's pack-slot
vocabulary, a third axis again, and translating between the two is F7's business at the
pack boundary. This module stays on the contract's spelling.)*
"""

from __future__ import annotations

from dataclasses import dataclass

from core.errors import Refusal, refuse
from services.parts.contract_validator import DatasetVersion, enum_at

#: Every ``target.kind`` the vendored schema defines, read from the file.
SCHEMA_TARGET_KINDS: frozenset[str] = enum_at("$defs", "Target", "properties", "kind")

#: Every ``target.miforge_set_kind`` the vendored schema defines, read from the file.
#: ``null`` is permitted by the schema and is handled in :func:`resolve`, not here.
SCHEMA_MIFORGE_SET_KINDS: frozenset[str] = enum_at(
    "$defs", "Target", "properties", "miforge_set_kind"
)

#: miForge's part kinds - the gate contract's ``PartReference.kind`` enum. Stated as a
#: constant rather than read from ``batch-record-v1.json`` because that file is miForge's
#: own and a test asserts the two agree; reading it here would make this module's import
#: depend on a second schema load for no behavioural gain.
PART_KINDS: frozenset[str] = frozenset(
    {
        "prompt_set",
        "corpus",
        "preference_pairs",
        "test_set",
        "retention_set",
        "selector_config",
        "grader_config",
    }
)

#: ``target.kind`` -> miForge part kind, for the targets that *are* a part kind outright.
TARGET_KIND_TO_PART_KIND: dict[str, str] = {
    "selector_config": "selector_config",
    "grader_config": "grader_config",
}

#: ``target.miforge_set_kind`` -> miForge part kind, for a ``miforge_set`` target. The
#: mapping is the identity today and is written out anyway: the two vocabularies are
#: owned by different repositories, and an identity mapping that is *stated* survives one
#: of them renaming a value, while an implicit one becomes a ``KeyError`` in production.
MIFORGE_SET_KIND_TO_PART_KIND: dict[str, str] = {
    "prompt_set": "prompt_set",
    "corpus": "corpus",
    "preference_pairs": "preference_pairs",
    "test_set": "test_set",
    "retention_set": "retention_set",
}

#: ``target.kind`` -> why miForge does not consume it. **Explicit, with a reason.** These
#: are real targets for other consumers; they are simply not miForge's, and naming them
#: here is what turns "unknown" into "known and declined".
REFUSED_TARGET_KINDS: dict[str, str] = {
    "dataset": (
        "a general dataset target, which names no miForge set kind; publish it as a "
        "miforge_set to use it here"
    ),
    "trl_export": "a TRL training export, consumed by a TRL trainer rather than by miForge",
    "detector_role": "a miStudio detector role; miForge consumes no detector sets",
    "reward_bundle": (
        "a reward bundle for GRPO-style reward modelling, which miForge does not run"
    ),
}

#: The target kinds miForge *does* consume.
CONSUMED_TARGET_KINDS: frozenset[str] = frozenset({"miforge_set", *TARGET_KIND_TO_PART_KIND})

#: The ``miforge_set_kind`` values miForge consumes.
MIFORGE_SET_KINDS: frozenset[str] = frozenset(MIFORGE_SET_KIND_TO_PART_KIND)


def _assert_exhaustive() -> None:
    """Every schema value is mapped or explicitly refused. **Checked at import.**

    This is the mechanism the whole module exists for, so it runs when the module loads
    rather than in a test: a test can be skipped, and a feature flag cannot turn a startup
    failure into a silent miscategorisation.

    Raises:
        Refusal: ``kind_map_incomplete`` naming the unhandled values.
    """
    handled = CONSUMED_TARGET_KINDS | set(REFUSED_TARGET_KINDS)
    unhandled = SCHEMA_TARGET_KINDS - handled
    if unhandled:
        raise refuse(
            "kind_map_incomplete",
            f"midataworks-dataset-version-v1.json defines target kind(s) "
            f"{sorted(unhandled)} that services/parts/kind_map.py neither maps nor "
            f"refuses. miDataworks has added a target: decide explicitly whether miForge "
            f"consumes it. Failing at import is deliberate - the alternative is treating "
            f"it as something it is not.",
            unhandled=sorted(unhandled),
        )
    ghosts = handled - SCHEMA_TARGET_KINDS
    if ghosts:
        raise refuse(
            "kind_map_names_a_ghost",
            f"services/parts/kind_map.py names target kind(s) {sorted(ghosts)} that the "
            f"vendored schema does not define. Either the schema moved and this map is "
            f"stale, or the map invented a value; both need a human.",
            ghosts=sorted(ghosts),
        )

    unhandled_sets = SCHEMA_MIFORGE_SET_KINDS - MIFORGE_SET_KINDS
    if unhandled_sets:
        raise refuse(
            "kind_map_incomplete",
            f"midataworks-dataset-version-v1.json defines miforge_set_kind(s) "
            f"{sorted(unhandled_sets)} that this map does not cover. A new set kind is a "
            f"new signal source or a new scoring input, not a value to pass through.",
            unhandled=sorted(unhandled_sets),
        )
    set_ghosts = MIFORGE_SET_KINDS - SCHEMA_MIFORGE_SET_KINDS
    if set_ghosts:
        raise refuse(
            "kind_map_names_a_ghost",
            f"this map names miforge_set_kind(s) {sorted(set_ghosts)} the vendored schema "
            f"does not define.",
            ghosts=sorted(set_ghosts),
        )

    stray = set(MIFORGE_SET_KIND_TO_PART_KIND.values()) | set(TARGET_KIND_TO_PART_KIND.values())
    if not stray <= PART_KINDS:
        raise refuse(
            "kind_map_part_kind_unknown",
            f"this map produces part kind(s) {sorted(stray - PART_KINDS)} that are not in "
            f"the batch record's PartReference.kind enum. A part kind nothing downstream "
            f"can record is a part that cannot appear in a batch's provenance.",
            unknown=sorted(stray - PART_KINDS),
        )


_assert_exhaustive()


@dataclass(frozen=True, slots=True)
class ResolvedKind:
    """What a target resolves to.

    ``content_kind`` rides along because the two travel together: a ``selector_config``
    target carries ``content.content_kind == "config"`` and has no splits, while a
    ``miforge_set`` carries ``"rows"`` and has at least one. A caller that had to re-derive
    that pairing would be the second place it is decided.
    """

    part_kind: str
    target_kind: str
    miforge_set_kind: str | None
    content_kind: str

    @property
    def is_config(self) -> bool:
        return self.content_kind == "config"


def resolve(version: DatasetVersion) -> ResolvedKind:
    """Which miForge part kind this version is, or refuse **naming the kind**.

    Args:
        version: An already-validated document. Validation first, always: this function
            indexes into ``target`` and reads a ``content_kind``, both of which the schema
            has guaranteed are present.

    Returns:
        The resolved kind.

    Raises:
        Refusal: ``dataset_version_target_unsupported`` for a target kind miForge does not
            consume, naming it and why (E-10); ``dataset_version_miforge_set_kind_missing``
            for a ``miforge_set`` target whose ``miforge_set_kind`` is **null** (E-11).

    **E-11's refusal is defence in depth, and the feature documents are wrong about why.**
    The FPRD's E-11, the FTID's §3.5 and the FTASKS' 3.5 all say the schema *"requires the
    key present"* while *"a null value on this target is an omission in substance"* - which
    reads as: the schema permits it and this layer must catch it. **It does not permit it.**
    ``Target``'s third ``allOf`` branch is
    ``if kind == "miforge_set" then {"miforge_set_kind": {"type": "string"}}``, and
    ``type: "string"`` excludes ``null``, so such a document fails validation at
    ``/target/miforge_set_kind`` before it ever reaches here. Verified by parsing the file
    and by a test that asserts the schema refuses it - which is how the claim was found to
    be wrong, since reading the `if/then` and concluding "type only, so null is fine" is
    exactly the reasoning the documents record.

    The refusal is kept anyway, for one honest reason: :func:`resolve` is a public function
    and nothing in Python stops a future caller passing a document that was not validated.
    A refusal that is unreachable through the intended path and correct through every other
    one costs four lines. It is **documented as unreachable** rather than presented as the
    only guard, because a test written to the old belief would have been asserting against
    the wrong layer.
    """
    target_kind = version.target_kind
    set_kind = version.miforge_set_kind
    content_kind = version.content_kind

    if target_kind in REFUSED_TARGET_KINDS:
        raise refuse(
            "dataset_version_target_unsupported",
            f"This version's target is {target_kind!r}: {REFUSED_TARGET_KINDS[target_kind]}. "
            f"miForge consumes {sorted(CONSUMED_TARGET_KINDS)}. Refused by name rather "
            f"than coerced into a kind it is not.",
            target_kind=target_kind,
            consumed=sorted(CONSUMED_TARGET_KINDS),
            reason=REFUSED_TARGET_KINDS[target_kind],
        )

    if target_kind in TARGET_KIND_TO_PART_KIND:
        return ResolvedKind(
            part_kind=TARGET_KIND_TO_PART_KIND[target_kind],
            target_kind=target_kind,
            miforge_set_kind=set_kind,
            content_kind=content_kind,
        )

    if target_kind == "miforge_set":
        if set_kind is None:
            raise refuse(
                "dataset_version_miforge_set_kind_missing",
                "This version's target is `miforge_set` but its `miforge_set_kind` is "
                "null, so it does not say which kind of set it is. The schema requires "
                "the key and permits a null value, which is an omission in substance: "
                f"miForge will not guess between {sorted(MIFORGE_SET_KINDS)}. Ask "
                "miDataworks to republish the version with its set kind.",
                target_kind=target_kind,
                expected_one_of=sorted(MIFORGE_SET_KINDS),
            )
        if set_kind not in MIFORGE_SET_KIND_TO_PART_KIND:  # pragma: no cover - import guard
            # Unreachable while `_assert_exhaustive` holds and the document validated.
            # Kept because "unreachable" is a claim about two other things staying true.
            raise refuse(
                "dataset_version_target_unsupported",
                f"miforge_set_kind={set_kind!r} is not one miForge maps.",
                miforge_set_kind=set_kind,
            )
        return ResolvedKind(
            part_kind=MIFORGE_SET_KIND_TO_PART_KIND[set_kind],
            target_kind=target_kind,
            miforge_set_kind=set_kind,
            content_kind=content_kind,
        )

    raise refuse(  # pragma: no cover - guarded by _assert_exhaustive at import
        "dataset_version_target_unsupported",
        f"target kind {target_kind!r} is neither mapped nor refused by name.",
        target_kind=target_kind,
    )


def refusal_for_unsupported(target_kind: str) -> Refusal:
    """The refusal a caller can build without a document, for a listing filter."""
    return refuse(
        "dataset_version_target_unsupported",
        f"miForge does not consume target kind {target_kind!r}.",
        target_kind=target_kind,
        consumed=sorted(CONSUMED_TARGET_KINDS),
    )
