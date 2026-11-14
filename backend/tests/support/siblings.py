"""The one place a test finds a sibling checkout (feature 015, FR-015.8; FTID §3.6).

The suite's repositories are checked out side by side, and miLLM became miServer (suite ENH-009).
From P0 to P7 either folder name, and either generation of variable, found it; since P7 only the
miServer spellings do (``RETIRED_ENV`` lists the old variables, read by nothing here). Before
this module five test files each did the lookup their own way and disagreed: one honoured only
``MISTUDIO_REQUIRE_CROSS_REPO_CHECKS``, one read ``MILLM_REPO`` before ``MIFORGE_MILLM_ROOT``, and
``test_schema_sync.py`` skipped an absent vendor even when cross-repo checks were required.
``test_no_inline_sibling_lookup.py`` keeps this the only implementation.

Rules:

- **Variables first, in order.** The first SET variable wins, even when its path does not
  exist: a variable is the operator saying where the sibling is, so a wrong path must surface
  as "absent" rather than fall through to whatever happens to sit beside the checkout.
- **Then folders beside this checkout,** new name first; the first that exists wins. With none
  present the first folder name is reported (so a message names where it looked).
- **Absent means skip, loudly; under any require switch it means fail.**
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import NoReturn

import pytest


@dataclass(frozen=True)
class Sibling:
    """A sibling repository: its display name, the variables that move it, its folder names."""

    key: str
    name: str
    env: tuple[str, ...]
    folders: tuple[str, ...]


MISERVER = Sibling(
    "miserver",
    "miServer",
    env=("MIFORGE_MISERVER_ROOT", "MISERVER_REPO"),
    folders=("miServer",),
)
#: The variables that found miServer before suite P7 retired them (FTID §9.7). Never read: kept
#: so the inline-lookup and workflow guards still flag a test or workflow that names one.
RETIRED_ENV = ("MIFORGE_MILLM_ROOT", "MILLM_REPO")
MIDATAWORKS = Sibling(
    "midataworks",
    "miDataworks",
    env=("MIFORGE_MIDATAWORKS_ROOT", "MIDATAWORKS_REPO"),
    folders=("miDataworks",),
)

#: Any of these set to "1" turns an absent sibling into a failure (miSuite CLAUDE.md). The
#: miStudio spelling stays until miWorkbench's rename retires it (suite P7).
REQUIRE_SWITCHES = (
    "MIFORGE_REQUIRE_CROSS_REPO_CHECKS",
    "MIWORKBENCH_REQUIRE_CROSS_REPO_CHECKS",
    "MISTUDIO_REQUIRE_CROSS_REPO_CHECKS",
)

#: miForge's repository root (this file is backend/tests/support/siblings.py).
REPO_ROOT = Path(__file__).resolve().parents[3]


@dataclass(frozen=True)
class Located:
    """Where a sibling was looked for and found (or not): the path, the spelling that chose
    it, and every spelling tried in order, for the message when it is absent."""

    path: Path
    spelling: str
    tried: tuple[str, ...]


def locate(sibling: Sibling, *, base: Path | None = None) -> Located:
    """Find ``sibling``: the first set variable, else the first existing folder beside
    ``base`` (default: the directory holding this checkout), else the first folder name."""
    parent = REPO_ROOT.parent if base is None else base
    tried = (*sibling.env, *(f"../{folder}" for folder in sibling.folders))
    for name in sibling.env:
        value = os.environ.get(name)
        if value:
            return Located(Path(value).expanduser(), name, tried)
    for folder in sibling.folders:
        candidate = parent / folder
        if candidate.is_dir():
            return Located(candidate, f"../{folder}", tried)
    first = sibling.folders[0]
    return Located(parent / first, f"../{first}", tried)


def required() -> bool:
    """True when any require switch is "1"."""
    return any(os.environ.get(name) == "1" for name in REQUIRE_SWITCHES)


def absent(message: str) -> NoReturn:
    """Skip loudly; fail instead when cross-repo checks are required."""
    if required():
        pytest.fail(
            f"{message} - and cross-repo checks are required ({' / '.join(REQUIRE_SWITCHES)})"
        )
    pytest.skip(message)


def not_checked_out(sibling: Sibling, located: Located, consequence: str) -> str:
    """The message for an absent sibling: every spelling tried, then what goes unchecked."""
    return (
        f"{sibling.name} is not checked out at {located.path} "
        f"(tried {', '.join(located.tried)}); {consequence}"
    )


def miserver_config_py(root: Path) -> Path | None:
    """miServer's settings module, ``miserver/core/config.py`` under ``root``; ``None`` when absent.

    Since suite P8 (miServer feature 041) this is the only layout: the pre-P8 package fallback
    was dropped with the DR-20 retirement (015 FTASKS 14.5), so a checkout that predates the
    rename reads as absent - a loud skip, or a failure under a require switch - instead of being
    compared against a stale layout.
    """
    config = root / "miserver" / "core" / "config.py"
    return config if config.is_file() else None
