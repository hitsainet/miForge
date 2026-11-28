"""The vendored result-language contract is byte-identical to miAssist's (017 FR-017.1; ADR-014).

miAssist owns ``docs/schemas/mistudio-result-language-v1.json``; miForge keeps a copy every
shape test reads (``tests/support/result_language.py``). One changed byte fails here. miAssist
absent: a loud skip (``CROSS-REPO:``); absent under any require switch: a failure.
"""

from __future__ import annotations

import json

from tests.support.result_language import VENDORED
from tests.support.siblings import MIASSIST, absent, locate, not_checked_out

OWNER_PATH = ("docs", "schemas", "mistudio-result-language-v1.json")


def test_the_copy_is_byte_identical_to_miassists() -> None:
    located = locate(MIASSIST)
    owner = located.path.joinpath(*OWNER_PATH)
    if not owner.is_file():
        absent(
            "CROSS-REPO: "
            + not_checked_out(
                MIASSIST, located, "the vendored result-language contract is not compared"
            )
        )
    assert (
        VENDORED.read_bytes() == owner.read_bytes()
    ), f"{VENDORED} differs from {owner}: re-vendor it byte for byte (cp), never hand-edit."


def test_the_copy_is_the_contract_miforge_reads() -> None:
    document = json.loads(VENDORED.read_text(encoding="utf-8"))
    defs = document["$defs"]
    for name in ("approval", "next_step", "unavailable", "error_code", "approval_status"):
        assert name in defs, name
    assert {"executing", "failed"} <= set(defs["approval_status"]["enum"])
    assert "miforge" in defs["app_id"]["enum"]
