"""The retention floor is enforced on the LIVE path, and ``require_floor`` is not it.

**This file exists because a reachability sweep reported ``packs.definition.require_floor``
as an unwired capability and the report was wrong.** Measured rather than read: a pack
committed with a filled retention slot and **no floor** is refused by
``pack_service.require_valid_pack`` with ``retention_floor_missing`` and
``details.slot == "retention_set"`` - byte for byte the refusal ``require_floor`` raises.
``packs/validation.py`` computes ``floor_missing``, :meth:`PackValidity.raise_if_invalid`
raises it, and that pair is wired at **both** ends: the run-start gate (step 2 of R-02.53's
validator) and ``packs.authoring.commit``.

So ``require_floor`` is a **second implementation of one rule**, not a missing one, and
wiring it would have added the defect AGENT.md records deleting: *"the validator runs first,
so it was unreachable for the two cases it existed to catch, and an unreachable guard reads
as protection while protecting nothing."* It is left in place and annotated; what was
genuinely missing is a test that the floor rule depends on the **wired** path - which is
what this file is.

## The control, and why it is on the gate rather than on the rule

ADR-016 #1 asks for a red when the **wiring** is removed, not when the rule is. Deleting
``validity.raise_if_invalid()`` from ``packs/gate.py::require_valid_pack`` already turns
``test_pack_gate.py`` red for ``pack_incomplete`` - but it did **not** turn anything red for
the floor, because no test drove a floorless pack through the live gate. That was the real
gap.

Observed with ``validity.raise_if_invalid()`` replaced by ``pass`` in
``packs/gate.py::require_valid_pack``: **1 failed** -
``test_the_run_start_gate_refuses_it_with_its_own_code_and_slot``, *"DID NOT RAISE"*.
Restored (``git diff`` clean for that path): 4 passed.

The caller tripwire's own measurement was controlled too, because a reachability test that
searches the wrong directory finds no callers and **passes**: with its root climbed one level
too far it fails loudly on the positive check (*"the production surface was not found at
/home/x-sean/app/miSuite/miForge/src"*, as it reads in the miSuite layout) rather than
reporting a clean result it never earned.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from packs.errors import RETENTION_FLOOR_MISSING, PackRefusal
from packs.service import pack_service
from packs.settings import reset_pack_settings
from packs.slots import SlotKind
from tests.support import pack_fixtures as fx
from tests.support.db import migrated_engine, session_for


@pytest.fixture
def packs_root(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("PACKS_ROOT", str(tmp_path / "packs"))
    reset_pack_settings()
    yield tmp_path / "packs"
    reset_pack_settings()


@pytest.fixture
def session(tmp_path: Path, packs_root: Path):
    engine = migrated_engine(tmp_path / "db")
    with session_for(engine) as session:
        yield session
    engine.dispose()


class TestAFloorlessRetentionSlotIsRefusedByTheLiveGate:
    def test_the_run_start_gate_refuses_it_with_its_own_code_and_slot(self, session) -> None:
        """E14 / R-02.38, through ``pack_service`` rather than by importing ``packs.gate``.

        The **payload** as well as the code: *"a fabricated floor of 0.0 passes every run
        forever while nothing says a thing"*, and a refusal that does not name the slot
        leaves the operator with nothing to fix.
        """
        pack = fx.commit(
            session,
            fx.definition(slots=fx.valid_slots(retention_set=fx.retention_slot(floor=None))),
        )
        with pytest.raises(PackRefusal) as raised:
            pack_service.require_valid_pack(session, pack.name, pack.version)
        assert raised.value.code == RETENTION_FLOOR_MISSING
        assert raised.value.details["slot"] == SlotKind.RETENTION_SET.value

    def test_a_floor_of_zero_is_a_floor_and_is_accepted(self, session, tmp_path: Path) -> None:
        """``0.0`` is a *stated* floor, and the distinction is the whole of E14: the refusal
        is for a floor that was never chosen, not for a low one. Collapsing ``None`` into
        ``0.0`` here would be the fabricated floor the refusal exists to prevent.

        All three slots point at the **one imported part**, which is the recipe
        ``test_pack_gate.py``'s own valid-pack test uses: a slot naming a part this
        installation does not hold is refused ``pack_part_unresolved``, which would make
        this assertion pass or fail for a reason that has nothing to do with the floor.
        """
        fx.import_contract_part(
            session,
            parts_root=tmp_path / "parts",
            export_root=tmp_path / "export",
            set_kind="prompt_set",
        )
        imported = ("prompt-set", "support-tickets", "3")
        pack = fx.commit(
            session,
            fx.definition(
                slots={
                    SlotKind.PROMPT_SET.value: fx.contract_slot(*imported),
                    SlotKind.TEST_SET.value: fx.contract_slot(*imported),
                    SlotKind.RETENTION_SET.value: {
                        "contract_reference": {
                            "type": imported[0],
                            "name": imported[1],
                            "version": imported[2],
                        },
                        "external": None,
                        "floor": 0.0,
                    },
                }
            ),
        )
        resolved = pack_service.require_valid_pack(session, pack.name, pack.version)
        assert resolved.floor() == 0.0
        assert resolved.validity.is_valid


class TestRequireFloorIsASecondImplementationAndNotAGuard:
    """A tripwire, so the next reader neither re-reports it nor wires a second rule."""

    def test_require_floor_still_has_no_production_caller(self) -> None:
        """If it ever gains one, there are two sources of truth for one rule and this says
        so. **Searched for the symbol in source, not for a spelling of a call** - the
        ``src/`` tree is the whole production surface."""
        # **Anchored on this test file and then NAMED**, which is what the neighbouring
        # reachability tests do (`test_review_queue_wiring.py`,
        # `test_hotupdate_adapter_read.py`). The first draft climbed `.parents[2]` from the
        # *production module* instead and landed on `backend/` - the `parents[N]`-climb
        # hazard AGENT.md records, where eight modules' depths compensated and all eight
        # computed the same wrong path while the suite stayed green.
        root = Path(__file__).resolve().parents[2] / "src"
        owner = (root / "packs" / "definition.py").resolve()
        # THE POSITIVE CHECK, and it is not decoration: with a wrong root `rglob` finds
        # nothing, `callers == []` passes, and the test reports "no caller" having searched
        # no files at all - a green from a check that never ran.
        assert owner.is_file(), f"the production surface was not found at {root}"
        searched = list(root.rglob("*.py"))
        assert len(searched) > 100, f"only {len(searched)} modules under {root}; wrong root"

        callers = [
            path
            for path in searched
            if path.resolve() != owner and "require_floor" in path.read_text(encoding="utf-8")
        ]
        assert callers == [], (
            "require_floor has gained a production caller, so one rule now has two "
            "implementations: this one and packs/validation.py's `floor_missing`, which is "
            "what the live gate and `authoring.commit` both raise. Delete one of them - and "
            f"if it is this one, delete this test with it. Callers: {callers}"
        )

    def test_it_agrees_exactly_with_the_rule_that_IS_wired(self) -> None:
        """The two implementations must not merely both refuse - they must refuse the same
        pack with the same code. A duplicate that drifted would be worse than either."""
        from packs.definition import require_floor
        from packs.validation import validate

        floorless = fx.definition(slots=fx.valid_slots(retention_set=fx.retention_slot(floor=None)))
        with pytest.raises(PackRefusal) as from_helper:
            require_floor(floorless)
        with pytest.raises(PackRefusal) as from_wired:
            validate(floorless, {}).raise_if_invalid()
        assert from_helper.value.code == from_wired.value.code == RETENTION_FLOOR_MISSING
