"""``@frozen(entry_id)``: the registry of frozen-identifier tests (feature 015, FTID §3.7).

Each deferred-renames register entry with a miForge location names ONE test in the census export
(``exercise``). The decorator records which test exercises which entry, and
``test_frozen_identifiers.py::test_every_entry_has_its_test`` requires the two to agree, so
deleting a registration (or the test) is a red, not a silently unexercised identifier.
"""

from __future__ import annotations

import inspect
from collections.abc import Callable
from pathlib import Path
from typing import Any, TypeVar

REPO_ROOT = Path(__file__).resolve().parents[3]

#: Entry id -> ``pytest:<path from the repository root>::<test name>``.
REGISTRY: dict[str, str] = {}

F = TypeVar("F", bound=Callable[..., Any])


def frozen(entry_id: str) -> Callable[[F], F]:
    def register(fn: F) -> F:
        source = Path(inspect.getsourcefile(fn) or "").resolve()
        node = f"pytest:{source.relative_to(REPO_ROOT).as_posix()}::{fn.__name__}"
        if entry_id in REGISTRY and REGISTRY[entry_id] != node:
            raise ValueError(f"{entry_id} is already exercised by {REGISTRY[entry_id]}")
        REGISTRY[entry_id] = node
        return fn

    return register
