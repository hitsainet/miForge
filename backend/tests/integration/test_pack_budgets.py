"""The interactive budgets, measured on the path that changed  (task 14.5; FTDD §9, ADR-024).

ADR-024: **measure the path you changed**, and a number quoted without its topology is not a
measurement. These are **pack-path** numbers - a registry listing, a validity check, an
export - and none of them may be quoted from a training measurement. The under-5 %-of-step
overhead budget does not apply here because no step is involved.

**The dry-run's latency is deliberately not asserted.** It is the provider's, not miForge's;
asserting it would be asserting a stub's speed, and §8 says so explicitly.

Measured while writing this file, on the development machine, **against SQLite** (the
suite's default engine, per ``tests/support/db``): listing 50 packs **52 ms**, one pack's
validity **1.3 ms**, an export **0.35 ms**. The assertions below use the FTDD's budgets
rather than those figures, with the margin deliberate: a tight assertion over a timing
measurement is a flaky test, and PostgreSQL is a different engine with a different number -
which `tests/support/db` records as a known limitation rather than hiding.
"""

from __future__ import annotations

import time
from pathlib import Path

import pytest

from packs import registry, store
from packs.settings import reset_pack_settings
from packs.transfer import export_bytes, import_document
from tests.support import pack_fixtures as fx
from tests.support.db import migrated_engine, session_for

#: FTDD §9's budgets, in milliseconds.
LISTING_BUDGET_MS = 500
VALIDITY_BUDGET_MS = 200
TRANSFER_BUDGET_MS = 1000

PACK_COUNT = 50


@pytest.fixture
def fifty_packs(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("PACKS_ROOT", str(tmp_path / "packs"))
    monkeypatch.delenv("MIDATAWORKS_BASE_URL", raising=False)
    reset_pack_settings()
    engine = migrated_engine(tmp_path / "db")
    with session_for(engine) as session:
        for index in range(PACK_COUNT):
            fx.commit(session, fx.definition(name=f"pack-{index:02d}", slots=fx.valid_slots()))
    yield engine
    engine.dispose()
    reset_pack_settings()


def test_the_registry_lists_fifty_packs_inside_its_budget(fifty_packs) -> None:
    """Warm, as §8 specifies - the first call pays for the schema load and the validators."""
    with session_for(fifty_packs) as session:
        registry.listing(session)
        start = time.perf_counter()
        rows = registry.listing(session)
        elapsed_ms = (time.perf_counter() - start) * 1000
    assert len(rows) == PACK_COUNT
    assert elapsed_ms < LISTING_BUDGET_MS, f"{elapsed_ms:.0f} ms for {PACK_COUNT} packs"


def test_one_packs_validity_is_inside_its_budget(fifty_packs) -> None:
    """Excluding outbound resolution, which is what §8 says - here there is none to exclude."""
    with session_for(fifty_packs) as session:
        registry.get(session, "pack-00", "1")
        start = time.perf_counter()
        registry.get(session, "pack-00", "1")
        elapsed_ms = (time.perf_counter() - start) * 1000
    assert elapsed_ms < VALIDITY_BUDGET_MS, f"{elapsed_ms:.0f} ms"


def test_an_export_and_an_import_are_inside_their_budget(fifty_packs, tmp_path: Path) -> None:
    definition = store.read("pack-00", "1")
    start = time.perf_counter()
    exported = export_bytes(definition)
    export_ms = (time.perf_counter() - start) * 1000

    # Import into a second store, so the measurement is a real import rather than a refusal
    # for an existing version.
    import os

    os.environ["PACKS_ROOT"] = str(tmp_path / "packs-import")
    reset_pack_settings()
    engine = migrated_engine(tmp_path / "db-import")
    try:
        with session_for(engine) as session:
            start = time.perf_counter()
            import_document(session, exported)
            import_ms = (time.perf_counter() - start) * 1000
    finally:
        engine.dispose()

    assert export_ms < TRANSFER_BUDGET_MS, f"export {export_ms:.0f} ms"
    assert import_ms < TRANSFER_BUDGET_MS, f"import {import_ms:.0f} ms"


def test_a_pack_is_kilobytes_not_megabytes(fifty_packs) -> None:
    """FR-007.4, as a size: *a pack is bytes in the hundreds, not megabytes.*

    Not a performance assertion so much as the structural one it follows from - a pack that
    grew to megabytes would be a pack carrying copies of its parts.
    """
    assert len(export_bytes(store.read("pack-00", "1"))) < 4096


def test_the_listing_issues_one_resolution_per_pack_and_one_usage_query(
    fifty_packs, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The **call count** behind the budget, so a regression is caught before the clock is.

    A timing assertion with a 10x margin passes through a change that made the listing ten
    times slower; the shape it would have broken is this one - one batched part lookup per
    pack and **one** usage query for the whole listing, not one per row.
    """
    from packs import registry as registry_module

    lookups: list[int] = []
    usage_queries: list[int] = []
    real_lookup = registry_module.resolve_slots
    real_usage = registry_module.usage_map

    def counting_lookup(session, definition):
        lookups.append(1)
        return real_lookup(session, definition)

    def counting_usage(session):
        usage_queries.append(1)
        return real_usage(session)

    monkeypatch.setattr(registry_module, "resolve_slots", counting_lookup)
    monkeypatch.setattr(registry_module, "usage_map", counting_usage)
    with session_for(fifty_packs) as session:
        registry_module.listing(session)
    assert len(lookups) == PACK_COUNT
    assert len(usage_queries) == 1
