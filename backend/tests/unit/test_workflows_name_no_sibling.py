"""CI names no sibling: no sibling checkout, folder or variable in any workflow value (FR-015.9).

Every workflow mention of a sibling today is a comment. A value that named one - a checkout
of ``../miLLM``, an env entry ``MIFORGE_MILLM_ROOT`` - would make the cross-repo checks depend
on a folder name the rename moves, and would make the public mirror's CI reach for a private
sibling. Keys and scalar values are scanned; comment lines inside ``run:`` scripts are not.
The repository gates naming miForge itself are not siblings and stay.
"""

from __future__ import annotations

import re
from collections.abc import Iterator
from typing import Any

import pytest
import yaml

from tests.support import siblings

WORKFLOWS = siblings.REPO_ROOT / ".github" / "workflows"
BRAND = re.compile(r"(?i)\bmi(llm|studio|suite|server|workbench)\b")
FOLDER = re.compile(r"\.\./mi[A-Z]\w+")
VARIABLES = frozenset((*siblings.MISERVER.env, *siblings.RETIRED_ENV, *siblings.MIDATAWORKS.env))


def _strings(node: Any, trail: str = "") -> Iterator[tuple[str, str]]:
    if isinstance(node, dict):
        for key, value in node.items():
            here = f"{trail}.{key}" if trail else str(key)
            yield here, str(key)
            if key == "run" and isinstance(value, str):
                kept = [ln for ln in value.splitlines() if not ln.lstrip().startswith("#")]
                yield here, "\n".join(kept)
            else:
                yield from _strings(value, here)
    elif isinstance(node, list):
        for index, item in enumerate(node):
            yield from _strings(item, f"{trail}[{index}]")
    elif isinstance(node, str):
        yield trail, node


def findings(text: str, name: str = "<planted>") -> list[str]:
    out = []
    for trail, value in _strings(yaml.safe_load(text)):
        for rule in (BRAND, FOLDER):
            for match in rule.finditer(value):
                out.append(f"{name} {trail}: {match.group(0)}")
        for variable in VARIABLES:
            if re.search(rf"\b{variable}\b", value):
                out.append(f"{name} {trail}: {variable}")
    return out


def test_no_sibling_in_workflow_values() -> None:
    files = sorted(WORKFLOWS.glob("*.yml"))
    assert len(files) >= 9, f"found only {len(files)} workflows"
    found = [f for path in files for f in findings(path.read_text("utf-8"), path.name)]
    assert not found, "workflows must not name a sibling:\n" + "\n".join(found)


@pytest.mark.parametrize(
    ("planted", "expected"),
    [
        ("jobs:\n  t:\n    env:\n      MIFORGE_MILLM_ROOT: ../miLLM\n", 3),
        ("jobs:\n  t:\n    steps:\n      - run: git clone x ../miServer\n", 2),
        ("jobs:\n  t:\n    steps:\n      - name: check miWorkbench\n", 1),
        (
            "jobs:\n  t:\n    steps:\n      - run: |\n"
            "          # ../miLLM is not needed\n          echo ok\n",
            0,
        ),
        (
            "on: push\njobs:\n  t:\n    if: github.repository == '"
            + "Onegai"
            + "shimas/miForge'\n",
            0,
        ),
    ],
    ids=["env-root", "clone", "step-name", "comment", "repository-guard"],
)
def test_scanner_self_test(planted: str, expected: int) -> None:
    assert len(findings(planted)) == expected, findings(planted)
