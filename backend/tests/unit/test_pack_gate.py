"""The run-start gate, on the live interface  (FR-007.10, FAC-10; ADR-016 #1).

*A capability is not shipped until a test FAILS when its wiring is removed.* So this file
asserts the gate **on the live** :class:`packs.service.PackService` rather than by importing
:mod:`packs.gate` - a sibling once shipped sixteen fully-implemented, unit-tested, documented
MCP tools that were never registered, and every one of those tests passed by importing the
module directly.

It also asserts the **payload and the call count**, not merely that the gate was called:
*"was called" passes against a call sending the wrong arguments.*

Mutation control (task 14.3d) for this file: delete ``require_valid_pack`` from
``PackService``. ``test_the_gate_is_on_the_live_service_interface`` must go red.

**What is still owed, and is not asserted here:** that F6's run-create path calls it. F6's
submit order puts the gate first - *"before any allocation"* - and its own task is the
reachability test for that seam. A grep proving a call exists is the weaker form and is not
sufficient, so this file does not pretend to cover it.
"""

from __future__ import annotations

import inspect
from pathlib import Path

import pytest

from packs import gate, require_valid_pack
from packs.errors import PACK_INCOMPLETE, PackRefusal
from packs.service import PackService, pack_service
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


class TestTheGateIsReachable:
    def test_the_gate_is_on_the_live_service_interface(self) -> None:
        """Read off the class, so a stub bound at the module level cannot satisfy it."""
        assert hasattr(PackService, "require_valid_pack")
        assert callable(PackService.require_valid_pack)
        assert hasattr(pack_service, "require_valid_pack")

    def test_the_gates_signature_is_the_one_f6_calls(self) -> None:
        """The documented signature, asserted - F6 calls ``(session, pack_id, version)``."""
        parameters = list(inspect.signature(PackService.require_valid_pack).parameters)
        assert parameters == ["self", "session", "pack_id", "version"]

    def test_it_is_exported_from_the_package(self) -> None:
        assert require_valid_pack is gate.require_valid_pack
        import packs

        assert "require_valid_pack" in packs.__all__

    def test_the_resolved_view_is_also_on_the_interface(self) -> None:
        """F2 and F4 consume this; three features must not write three resolvers."""
        for name in ("resolved_pack", "part_records", "lineage", "validity"):
            assert hasattr(PackService, name), f"PackService.{name} is missing"


class TestTheGateRefusesAnInvalidPack:
    def test_an_incomplete_pack_is_refused_with_its_payload(self, session) -> None:
        slots = fx.valid_slots()
        del slots[SlotKind.TEST_SET.value]
        pack = fx.commit(session, fx.definition(slots=slots))
        with pytest.raises(PackRefusal) as raised:
            pack_service.require_valid_pack(session, pack.name, pack.version)
        assert raised.value.code == PACK_INCOMPLETE
        # The PAYLOAD, not just the code: a refusal naming no slot fails AC10 even while
        # refusing correctly.
        assert raised.value.details["missing_slots"] == ["test_set"]

    def test_a_valid_pack_returns_the_resolved_view(self, session, tmp_path: Path) -> None:
        part = fx.import_contract_part(
            session,
            parts_root=tmp_path / "parts",
            export_root=tmp_path / "export",
            set_kind="prompt_set",
        )
        assert part
        pack = fx.commit(
            session,
            fx.definition(
                slots={
                    SlotKind.PROMPT_SET.value: fx.contract_slot(),
                    SlotKind.TEST_SET.value: fx.contract_slot("prompt-set", "support-tickets", "3"),
                    SlotKind.RETENTION_SET.value: {
                        "contract_reference": {
                            "type": "prompt-set",
                            "name": "support-tickets",
                            "version": "3",
                        },
                        "external": None,
                        "floor": 0.72,
                    },
                }
            ),
        )
        resolved = pack_service.require_valid_pack(session, pack.name, pack.version)
        assert resolved.validity.is_valid
        assert resolved.identity.pack_id == pack.name
        assert resolved.floor() == 0.72
        # The resolved view carries the part records F8 writes - produced from the resolution,
        # so three features cannot disagree about what a part was.
        assert len(resolved.part_records()) == 3

    def test_the_gate_re_resolves_rather_than_reading_the_stored_flag(self, session) -> None:
        """A cache hit on a deleted part is the failure the gate exists to catch.

        ``mf_packs.valid`` is set at index time; this test sets it to ``True`` behind the
        gate's back for a pack that is **not** valid, and the gate must still refuse.
        """
        from models.pack import Pack
        from packs.registry import row_id

        slots = fx.valid_slots()
        del slots[SlotKind.RETENTION_SET.value]
        pack = fx.commit(session, fx.definition(slots=slots))
        row = session.get(Pack, row_id(pack.name, pack.version))
        assert row is not None
        row.valid = True
        row.invalid_reason = None
        session.flush()

        with pytest.raises(PackRefusal) as raised:
            pack_service.require_valid_pack(session, pack.name, pack.version)
        assert raised.value.details["missing_slots"] == ["retention_set"]


class TestTheGateCountsItsCalls:
    def test_the_resolver_is_called_exactly_once_per_gate_call(
        self, session, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The call **count**, because a gate that resolved twice would be two answers.

        One batched resolution per gate call is also what §9's budget assumes; a per-slot
        resolution is the easiest way to lose it and the cheapest way to notice is here.
        """
        calls: list[tuple[str, str]] = []
        real = gate.resolve_slots

        def counting(session_arg, definition):
            calls.append((definition.name, definition.version))
            return real(session_arg, definition)

        monkeypatch.setattr(gate, "resolve_slots", counting)
        slots = fx.valid_slots()
        del slots[SlotKind.TEST_SET.value]
        pack = fx.commit(session, fx.definition(slots=slots))
        calls.clear()
        with pytest.raises(PackRefusal):
            pack_service.require_valid_pack(session, pack.name, pack.version)
        assert calls == [(pack.name, pack.version)]
