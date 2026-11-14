"""G4 (FR-012.20, MR-13): ``docs/mcp-contract.md`` is what the LIVE registry generates.

A stale file fails here; regenerating it is the fix. And each row's route agrees with the
ledger, so a generator that read a tool body wrongly cannot launder the error into the file.

The public mirror KEEPS this file (``sync-to-clean.yml`` exempts it beside ``docs/schemas/``),
so it is a hard failure when absent - pinned below against the workflow itself, because a
guard reading a file the mirror strips would turn the mirror's build red and block the image.
"""

from __future__ import annotations

import re
from pathlib import Path

from mcp_server.contract import CONTRACT_PATH, collect, generate_contract
from tests.support.mcp_ledger import EXPECTED_CALLS, full, template_matches

REPO_ROOT = Path(__file__).resolve().parents[3]
SYNC = REPO_ROOT / ".github" / "workflows" / "sync-to-clean.yml"


def test_the_contract_is_current() -> None:
    assert CONTRACT_PATH.is_file(), f"{CONTRACT_PATH} is missing; regenerate it"
    assert CONTRACT_PATH.read_text(encoding="utf-8") == generate_contract(), (
        "docs/mcp-contract.md differs from the live registry. Regenerate: cd backend/src && "
        'python -c "from mcp_server.contract import write_contract; write_contract()"'
    )


def test_every_row_names_the_route_the_ledger_names() -> None:
    rows = {row["name"]: row for rows in collect().values() for row in rows}
    assert len(rows) >= 80
    for tool, (method, path, _a, _r, _c) in EXPECTED_CALLS.items():
        endpoints = rows[tool]["endpoints"]
        assert len(endpoints) == 1, (tool, endpoints)
        row_method, row_template = endpoints[0].split(" ", 1)
        assert row_method == method, (tool, endpoints)
        assert template_matches(row_template, full(path)), (tool, row_template, path)


def test_the_mirror_keeps_the_contract() -> None:
    text = SYNC.read_text()
    (line,) = [each for each in text.splitlines() if "find docs/" in each]
    assert "! -name schemas" in line, "the scan read the wrong line"
    assert re.search(
        r"! -name mcp-contract\.md\b", line
    ), "sync-to-clean.yml strips docs/mcp-contract.md from the mirror, where this test runs"
