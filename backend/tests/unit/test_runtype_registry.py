"""The run-type registry: three registrations and **one loop**  (FR-008.1, FR-008.2, FR-008.19).

Resolved through :func:`training.runtypes.resolve`, never by importing the three modules.
*Importing proves a module parses; resolving proves a caller can reach it* - the lesson of
the sibling that shipped sixteen implemented, unit-tested, documented MCP tools which were
never registered with its server.

**M12's target.** Deleting a ``register(...)`` line must turn this file red.
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

from core.errors import Refusal
from models.run import Run
from services import runtypes_service
from training import runtypes
from training.runtypes.base import RUN_TYPE_NAMES, STEP, RunType

SRC = Path(__file__).resolve().parents[3] / "backend" / "src"


class TestTheThreeAreReachableThroughTheLiveRegistry:
    @pytest.mark.parametrize("name", RUN_TYPE_NAMES)
    def test_each_declared_run_type_resolves(self, name: str) -> None:
        assert runtypes.resolve(name).name == name

    def test_the_registry_holds_exactly_the_three(self) -> None:
        """Two-way, like the route manifest: a fourth registration nobody declared is as
        much a defect as a missing one, and ``attach`` is the one someone will add."""
        assert set(runtypes.known()) == set(RUN_TYPE_NAMES)

    def test_there_is_no_attach_run_type(self) -> None:
        """Section 5.10 changes the **acquisition mode**, not the run type (ADR-022)."""
        with pytest.raises(Refusal) as caught:
            runtypes.resolve("attach")
        assert caught.value.code == "run_type_unknown"

    def test_an_unknown_name_refuses_and_LISTS_what_is_known(self) -> None:
        """A bare ``KeyError`` reaching the API is a 500 where a 400 naming the three
        valid values belongs."""
        with pytest.raises(Refusal) as caught:
            runtypes.resolve("lora")
        assert caught.value.code == "run_type_unknown"
        assert caught.value.details is not None
        assert sorted(caught.value.details["known"]) == sorted(RUN_TYPE_NAMES)

    def test_a_duplicate_registration_raises_rather_than_replacing(self) -> None:
        """Overwriting would make the earlier registration's behaviour vanish silently."""
        with pytest.raises(RuntimeError, match="duplicate run type"):
            runtypes.register(runtypes.resolve("adapter"))

    def test_every_registered_type_satisfies_the_declared_protocol(self) -> None:
        for name in RUN_TYPE_NAMES:
            assert isinstance(runtypes.resolve(name), RunType), name


class TestThreeRunTypesShareOneLoop:
    """BRD-02 section 5.9's first sentence, asserted rather than commented."""

    def test_all_three_carry_the_SAME_step_object(self) -> None:
        """Identity, not equality. A run type with its own step function would pass an
        equality check on the name and have duplicated the loop."""
        steps = {name: runtypes.resolve(name).step for name in RUN_TYPE_NAMES}
        assert len(set(map(id, steps.values()))) == 1, steps
        assert steps["adapter"] is STEP

    def test_the_step_object_is_F1s_run_one_and_not_a_wrapper(self) -> None:
        from training.step import run_one

        assert STEP is run_one

    def test_change_and_create_return_the_same_trainable_set(self) -> None:
        """R-02.41's content is that step 0 is random - which F1's acquisition already did -
        so the trainable set is identical and that identity should stay visible."""
        from tests.support.test_adapter_doubles import FakeModule, FakeParameter

        module = FakeModule({"w": FakeParameter(), "b": FakeParameter()})
        acquired = type("Acquired", (), {"module": module, "tokenizer": object()})()
        change = runtypes.resolve("change").prepare(acquired)
        create = runtypes.resolve("create").prepare(acquired)
        assert change.trainable == create.trainable
        assert change.base_digest is None and create.base_digest is None


class TestNoFeatureBranchesOnRunTypeOutsideThisPackage:
    """A ratchet with an **empty allowlist**. ADR-022's shape, applied to a design rule.

    *"A* ``if run_type ==`` *inside the step is a defect in this feature, not a shortcut."*
    The registry exists so that adding F11's hot-update acquisition does not touch the step,
    the optimizer, the save path or the test path either.
    """

    #: Where branching on a run type is legitimate: this feature's own package, plus the
    #: service and endpoint that dispatch *into* it. Empty elsewhere, and it stays empty.
    ALLOWED = (
        Path("training/runtypes"),
        Path("services/runtypes_service.py"),
        Path("api/v1/endpoints/runtypes.py"),
    )

    def _offenders(self) -> list[str]:
        found: list[str] = []
        for path in sorted(SRC.rglob("*.py")):
            relative = path.relative_to(SRC)
            if any(allowed in (relative, *relative.parents) for allowed in self.ALLOWED):
                continue
            tree = ast.parse(path.read_text(encoding="utf-8"))
            for node in ast.walk(tree):
                if not isinstance(node, ast.Compare):
                    continue
                left = node.left
                name = (
                    left.attr
                    if isinstance(left, ast.Attribute)
                    else left.id if isinstance(left, ast.Name) else None
                )
                if name in {"run_type", "runtype"}:
                    found.append(f"{relative}:{node.lineno}")
        return found

    def test_nothing_outside_F8s_package_compares_a_run_type(self) -> None:
        assert not self._offenders(), (
            "a run type is a registry entry, not a conditional; found: " f"{self._offenders()}"
        )


class TestTheStoredSpellingAgreesWithTheLIVEConstraint:
    """``mf_runs.run_type``'s CHECK permits ``create|full|adapter|merge``; the registry's
    third name is ``change``.

    One mapping, asserted against the **constraint as the model declares it** rather than
    against a retyped list - a retyped list agrees with itself, which is the fixture failure
    ADR-016 names. The divergence itself is reported for the operator: two spellings of one
    concept is exactly the drift that costs this project sessions.
    """

    def _permitted(self) -> set[str]:
        import re

        for constraint in Run.__table_args__:
            text = getattr(constraint, "sqltext", None)
            if text is None:
                continue
            rendered = str(text)
            if "run_type IN" not in rendered:
                continue
            return set(re.findall(r"'([a-z_]+)'", rendered))
        raise AssertionError("mf_runs declares no run_type CHECK constraint to read")

    def test_every_registry_name_IS_a_permitted_value(self) -> None:
        """The assertion worth keeping, and the one that found the divergence.

        A registration whose name the CHECK constraint forbids is a run whose row cannot be
        written - discovered at the first submit rather than here. This is what caught
        ``change`` against a constraint that said ``full``, and it goes red if either side
        moves again.
        """
        permitted = self._permitted()
        for name in RUN_TYPE_NAMES:
            assert name in permitted, (
                f"the registry offers {name!r} and mf_runs.run_type permits "
                f"{sorted(permitted)}. A run of that type could be submitted and then fail "
                f"to persist."
            )

    def test_there_is_no_translation_between_the_two_vocabularies(self) -> None:
        """``DB_RUN_TYPE`` is gone, and its absence is asserted rather than assumed.

        F8 shipped ``{"change": "full", ...}`` because the constraint and the registry
        disagreed, and reported it instead of editing a migration it did not own. Migration
        ``0014`` settled it on ``change``, BRD-02 §4's own word, and the map was removed
        rather than reduced to an identity function: a translation table between two names
        for one concept is a second source of truth, which is the defect class behind the
        ``step-`` prefix, the three conflated signal vocabularies and an invented ``skip``
        decision asserted as a sibling's.

        If a mapping reappears, this goes red and asks which vocabulary is authoritative.
        """
        assert not hasattr(runtypes_service, "DB_RUN_TYPE"), (
            "a run-type translation map is back. Decide which spelling is authoritative and "
            "make the other one not exist, rather than keeping both in step by hand."
        )

    def test_the_api_reports_the_stored_value_as_the_name_itself(self) -> None:
        """``stored_as`` is retained and now equals ``name`` for every entry.

        It is kept rather than dropped because it is a client-visible promise that no
        translation happens - which is exactly what was not true before ``0014``.
        """
        entries = runtypes_service.run_types()
        assert entries, "the live registry is empty, so nothing here is being checked"
        for entry in entries:
            assert entry["stored_as"] == entry["name"]


class TestTheTwoVocabulariesCannotDrift:
    """``models.run.RUN_TYPES`` and the live registry, kept in step by a test not an import.

    **They drifted, and it took two agents to notice.** Migration ``0014`` renamed the
    permitted value ``full`` to ``change`` and updated the CHECK constraint; ``RUN_TYPES``
    still said ``full``. It was exported from ``models/__init__.py`` and read by **nothing**,
    so it was a dead second registry carrying a stale spelling — the same shape as the
    ``DB_RUN_TYPE`` translation map that migration removed, one file away. F6 found it and
    documented the workaround (*"'full' is the stale spelling in models/run.RUN_TYPES"*); F9
    found it exported and unused. Neither could fix it, because ``models/`` is not a feature
    agent's to edit.

    **Why a test rather than an import.** ``models/`` must not import ``training/`` — the
    models are the bottom layer and the registry sits above them, so the dependency would be
    inverted. The CHECK constraint is now *rendered from* ``RUN_TYPES``, which removes one of
    the three copies; this closes the gap between the remaining two.
    """

    def test_every_registered_run_type_is_permitted_by_the_model(self) -> None:
        from models.run import RUN_TYPES

        missing = set(RUN_TYPE_NAMES) - RUN_TYPES
        assert not missing, (
            f"the registry offers {sorted(missing)} and models.run.RUN_TYPES does not permit "
            f"them, so a run of that type could be submitted and then fail to persist. "
            f"RUN_TYPES renders the CHECK constraint, so this is the database refusing a row "
            f"the API accepted."
        )

    def test_the_model_permits_nothing_the_registry_cannot_produce_except_merge(self) -> None:
        """The other direction, with ``merge`` named as the one deliberate exception.

        A permitted value no registry can produce is dead vocabulary, and dead vocabulary is
        how ``full`` survived a rename. ``merge`` is excepted on the record rather than
        silently: a merge is an **operation**, not a run, and is recorded in ``mf_merges``.
        """
        from models.run import RUN_TYPES

        unreachable = RUN_TYPES - set(RUN_TYPE_NAMES) - {"merge"}
        assert not unreachable, (
            f"models.run.RUN_TYPES permits {sorted(unreachable)}, which no registered run "
            f"type produces. Either register it or remove it - a value nothing can write is "
            f"what let 'full' outlive migration 0014."
        )

    def test_the_check_constraint_is_rendered_from_the_set(self) -> None:
        """Pins the mechanism, not just today's agreement.

        If someone writes the constraint out as a literal again, the two can diverge again -
        so this asserts the constraint text contains exactly the members of ``RUN_TYPES``.
        """
        # The constraint is read here rather than borrowed from the other class's helper:
        # this test is about the rendering mechanism, so it must read the rendered text.
        import re

        from models.run import RUN_TYPES, Run

        permitted: set[str] = set()
        for constraint in Run.__table_args__:
            text = getattr(constraint, "sqltext", None)
            if text is None or "run_type IN" not in str(text):
                continue
            permitted = set(re.findall(r"'([a-z_]+)'", str(text)))
            break
        assert permitted, "mf_runs declares no run_type CHECK constraint to read"
        assert permitted == RUN_TYPES, (
            f"the CHECK constraint permits {sorted(permitted)} and RUN_TYPES holds "
            f"{sorted(RUN_TYPES)}. The constraint is supposed to be rendered from the set; a "
            f"literal has been written back in."
        )
