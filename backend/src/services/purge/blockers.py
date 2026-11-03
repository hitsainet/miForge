"""Blocker checks contributed by optional features  (purge; FR-006.46's deletability rule).

The purge service must not import an optional feature's code: chaining is deletable by
design (``test_chain_deletability.py``), so the chain's own module registers its check here
when it is imported, and deleting the chain modules removes the check with the feature -
no chains, nothing for a chain to block.

A check takes ``(session, kind, item_id, mentions)`` - ``mentions(value)`` says whether a
stored definition names the item, by the purge service's own rules - and returns
``(code, message, details)`` for each blocker it finds.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from sqlalchemy.orm import Session

Blocker = tuple[str, str, dict[str, Any]]
Check = Callable[[Session, str, str, Callable[[Any], bool]], list[Blocker]]

_CHECKS: list[Check] = []


def register(check: Check) -> Check:
    """Add a check once. Returns it, so it can be used as a decorator."""
    if check not in _CHECKS:
        _CHECKS.append(check)
    return check


def registered() -> tuple[Check, ...]:
    return tuple(_CHECKS)


def contributed(
    session: Session, kind: str, item_id: str, mentions: Callable[[Any], bool]
) -> list[Blocker]:
    """Every registered check's blockers for one item."""
    found: list[Blocker] = []
    for check in _CHECKS:
        found.extend(check(session, kind, item_id, mentions))
    return found
