"""``next_for`` builds one ``next_step`` and refuses non-miForge names (FR-017.6)."""

from __future__ import annotations

import pytest

from services.next_step import next_for
from tests.support.result_language import validate


def test_the_shape_is_the_contracts() -> None:
    step = next_for("forge_get_run", {"run_id": "r1"}, "see it")
    assert step == {"tool": "forge_get_run", "args": {"run_id": "r1"}, "why": "see it"}
    validate("next_step", step)
    validate("next_step", next_for("forge_health", None, "check the backend"))


@pytest.mark.parametrize("name", ["dataworks_x", "miserver_status", "get_run", "", "Forge_get_run"])
def test_a_name_that_is_not_a_miforge_tool_is_refused(name: str) -> None:
    with pytest.raises(ValueError, match="forge_"):
        next_for(name, None, "why")


@pytest.mark.parametrize("why", ["", "x" * 301])
def test_a_why_outside_the_contracts_length_is_refused(why: str) -> None:
    with pytest.raises(ValueError, match="why"):
        next_for("forge_get_run", None, why)


def test_the_args_are_copied_not_shared() -> None:
    args = {"run_id": "r1"}
    step = next_for("forge_get_run", args, "see it")
    args["run_id"] = "changed"
    assert step["args"] == {"run_id": "r1"}
