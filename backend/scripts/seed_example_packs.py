#!/usr/bin/env python3
"""Seed the two example packs  (task 4.6; FR-007.16, R-02.39).

**Run it, or do not have them.** This is a script and emphatically **not** a migration, and
that is the requirement rather than a convenience: FR-007.16 says example packs are
*"deletable with nothing depending on them"*, and a seeder that ran inside ``alembic
upgrade`` would put them back on the next deploy. An operator who deleted an example would
find it returned without having asked, which is the opposite of deletable. So a deleted
example **stays deleted** until somebody runs this again on purpose::

    python backend/scripts/seed_example_packs.py            # seed, and report what it did
    python backend/scripts/seed_example_packs.py --dry-run  # say what it would do

## What an example pack is for, and why these two

F7 shipped **no** example packs rather than seed unmarked ones indistinguishable from real
ones, because ``mf_packs`` had no ``is_example`` column until migration ``0015``. That was
the right call and it cost the product its onboarding: two worked examples are how an
operator who has never seen a pack learns what one is. The names and the human titles are
the mockup's own two (``ex-financial-supervision`` and ``ex-carburetor-rebuild``), taken from
there rather than invented here so the screenshots and the product agree.

**They differ in exactly one slot, deliberately.** One takes its signal from a corpus and the
other from preference pairs, so the pair shows the two signal shapes R-02.4 and R-02.6
describe rather than showing the same pack twice. Both fill six of the seven slots, which is
also the mockup's count, and both fill all three of R-02.38's required slots **with a
retention floor**, because a pack whose only interesting property was its own invalidity
would teach the wrong lesson.

## They do not resolve on a fresh installation, and that is the honest state

Every slot here is a **contract reference** into miDataworks, and nothing seeds the parts
themselves: those are miDataworks' (BRD-03 R-03.48) and this project does not rebuild them.
So on an installation with no miDataworks configured, the registry will show these two with
``Needs a service · midataworks not configured`` — R-02.50's convention, and the single most
important distinction in :mod:`packs.resolver`: *nothing is wrong with the pack.*

**Fabricating resolvable parts was the alternative and it is worse.** Writing rows into the
parts index so an example looked ``Valid`` would make a pack that an operator could start a
run from, and that run would train on invented data while every number it produced looked
real. An example that says what it needs is a better teacher than one that lies about being
ready.

## Idempotence, and the one thing it repairs

Running this twice creates nothing twice: a pack version whose ``pack.json`` is already on
disk is left alone (``store.write_committed`` refuses to overwrite a committed version, which
is FR-007.2 and not something to work around). The flag is a separate question, and the
script **restores it** if the file is there and the row is not marked. That is not
tidiness — it is the documented remedy for migration ``0015``'s downgrade, which drops
``is_example`` and which neither ``pack.json`` nor an export can restore, because the pack
document's top-level object is closed to unknown keys.

Three outcomes, reported separately rather than summed into "done": ``created``,
``flag_restored``, ``already_present``.
"""

from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path

if __package__ in (None, ""):  # pragma: no cover - executed only as a script
    # `python backend/scripts/seed_example_packs.py` from a clone, with no install. The
    # package root is `backend/src` (AGENT.md's path notation), so a bare run needs it on the
    # path; an installed deployment already has it and this is then a no-op.
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

# Imported after the path shim above, deliberately: a bare `python backend/scripts/...` run
# has no `backend/src` on the path until that block has run.
from sqlalchemy.orm import Session

from packs import registry, store
from packs.definition import FilledSlot, PackVersionDefinition
from packs.references import ContractPartRef
from packs.slots import SLOT_ORDER, SlotKind

#: ``created_at`` for a seeded pack. **Fixed, not ``now()``**, so two installations that seed
#: the same example hold the same bytes and therefore the same ``definition_sha256`` - which
#: is what makes an exported example comparable between installations. A wall-clock stamp
#: would make every seed a different document describing the same pack.
SEEDED_AT = datetime(2026, 10, 14, tzinfo=UTC)


def _ref(type_: str, name: str, version: str) -> ContractPartRef:
    """A miDataworks contract reference, ``{type, name, version}``.

    The ``type`` spellings are the mockup's (``prompts``, ``tests``, ``selectors``) because
    they are the only recorded examples of that field's vocabulary, and the field is
    miDataworks' - free-form in their contract. Nothing here depends on the spelling: a
    reference that does not resolve is reported as unresolved by name.
    """
    return ContractPartRef(type=type_, name=name, version=version)


@dataclass(frozen=True, slots=True)
class ExamplePack:
    """One example pack, declared rather than built imperatively."""

    name: str
    version: str
    domain: str | None
    title: str
    slots: dict[SlotKind, FilledSlot]

    def definition(self) -> PackVersionDefinition:
        """Through the **real** model, so the document is validated against the schema.

        :meth:`PackVersionDefinition.document` validates on build, so a seeded pack that did
        not conform would be refused here rather than written - and the writer below is
        ``registry.create``, which is the same path the authoring flow commits through. A
        seeder with its own JSON would be a second writer, and a fixture whose fields agree
        with its reader by construction is how a suite stays green over a real bug.
        """
        return PackVersionDefinition(
            name=self.name,
            version=self.version,
            domain=self.domain,
            slots={slot: self.slots.get(slot) for slot in SLOT_ORDER},
            created_at=SEEDED_AT,
        )


#: The two examples. Their one difference is the third slot: a corpus against preference
#: pairs. See the module docstring.
EXAMPLE_PACKS: tuple[ExamplePack, ...] = (
    ExamplePack(
        name="ex-financial-supervision",
        version="1",
        domain="financial-supervision",
        title="Financial supervision",
        slots={
            SlotKind.PROMPT_SET: FilledSlot(ref=_ref("prompts", "supervision-core", "1")),
            SlotKind.CORPUS: FilledSlot(ref=_ref("corpus", "supervisory-guidance", "1")),
            SlotKind.SELECTOR_CONFIG: FilledSlot(
                ref=_ref("selectors", "cites-the-regulation", "1")
            ),
            SlotKind.GRADER_CONFIG: FilledSlot(ref=_ref("graders", "answer-completeness", "1")),
            SlotKind.TEST_SET: FilledSlot(ref=_ref("tests", "supervision-holdout", "1")),
            # The floor is MIFORGE'S value, not miDataworks': they own the set, this project
            # owns what counts as an unacceptable regression on it (FR-007.5).
            SlotKind.RETENTION_SET: FilledSlot(
                ref=_ref("tests", "general-knowledge", "1"), floor=0.78
            ),
        },
    ),
    ExamplePack(
        name="ex-carburetor-rebuild",
        version="1",
        domain="small-engine-repair",
        title="Small-engine carburetor rebuild",
        slots={
            SlotKind.PROMPT_SET: FilledSlot(ref=_ref("prompts", "rebuild-procedures", "1")),
            SlotKind.PREFERENCE_PAIRS: FilledSlot(ref=_ref("pairs", "preferred-procedure", "1")),
            SlotKind.SELECTOR_CONFIG: FilledSlot(ref=_ref("selectors", "names-the-torque", "1")),
            SlotKind.GRADER_CONFIG: FilledSlot(ref=_ref("graders", "step-order", "1")),
            SlotKind.TEST_SET: FilledSlot(ref=_ref("tests", "rebuild-holdout", "1")),
            SlotKind.RETENTION_SET: FilledSlot(
                ref=_ref("tests", "general-knowledge", "1"), floor=0.80
            ),
        },
    ),
)


@dataclass(slots=True)
class SeedReport:
    """What the seeder did, in three lists rather than one count."""

    created: list[str] = field(default_factory=list)
    flag_restored: list[str] = field(default_factory=list)
    already_present: list[str] = field(default_factory=list)

    def lines(self, *, dry_run: bool = False) -> list[str]:
        """One line per pack, naming what happened to it - or what would.

        The tense is the report's own rather than a prefix glued on by the caller: *"would
        created"* is what a prefix produces, and a line an operator cannot read is a line
        they will not read.
        """
        out: list[str] = []
        for identifier in self.created:
            verb = "would create" if dry_run else "created"
            out.append(f"{verb}  {identifier}  (and mark is_example)")
        for identifier in self.flag_restored:
            out.append(
                f"repaired  {identifier}  (already on disk; is_example was not set and now is)"
            )
        for identifier in self.already_present:
            out.append(f"present  {identifier}  (nothing to do)")
        return out


def seed(session: Session, *, dry_run: bool = False) -> SeedReport:
    """Seed the examples. Returns what was done, per pack.

    Args:
        session: An open session. The caller owns the transaction: :func:`main` uses
            ``db.session.session_scope``, and a test passes its own so the seeding and the
            assertions see one transaction.
        dry_run: Report without writing. Nothing is created and no flag is set, so a
            ``--dry-run`` on a fresh installation reports both packs as ``created``: that
            is what *would* happen, which is the question being asked.

    The order is: write the file, index it, then mark the flag. That is ``registry.create``'s
    own order for a reason - the filesystem is authoritative (ADR-004), so it is written
    first - and the flag comes last because it is index-only state that has nowhere else to
    live.
    """
    report = SeedReport()
    for example in EXAMPLE_PACKS:
        identifier = f"{example.name}@{example.version}"
        # A writer may call `store.exists`, which resolves `packs_root()` and so creates the
        # root on demand. That is right here and would be wrong on a read path - the same
        # mkdir once aborted a whole reindex with PermissionError - and this script exists to
        # write packs.
        on_disk = store.exists(example.name, example.version)

        if not on_disk:
            if dry_run:
                report.created.append(identifier)
                continue
            registry.create(session, example.definition())
            registry.mark_example(session, example.name, example.version)
            report.created.append(identifier)
            continue

        if dry_run:
            report.already_present.append(identifier)
            continue
        # On disk already. The flag is the one thing worth repairing - see the module
        # docstring on migration 0015's downgrade. `mark_example` returns whether it changed
        # anything, so "restored" and "already marked" are distinguishable.
        changed = registry.mark_example(session, example.name, example.version)
        (report.flag_restored if changed else report.already_present).append(identifier)
    return report


def main(argv: Sequence[str] | None = None) -> int:
    """Seed, print what happened, and say what the registry will show."""
    parser = argparse.ArgumentParser(description=__doc__ or "", add_help=True)
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="say what would happen and write nothing",
    )
    args = parser.parse_args(argv)

    from db.session import session_scope

    with session_scope() as session:
        report = seed(session, dry_run=args.dry_run)

    for line in report.lines(dry_run=args.dry_run):
        print(line)
    print(
        "\nThese packs hold references into miDataworks and seed no parts: the parts are "
        "miDataworks' (BRD-03 R-03.48) and are not fabricated here. With no miDataworks "
        "configured the registry will show each as 'Needs a service', which is an absent "
        "service and not a fault in the pack. Both are deletable, and deleting one is "
        "permanent: nothing puts it back but this script."
    )
    return 0


if __name__ == "__main__":  # pragma: no cover - the CLI entry point
    raise SystemExit(main())
