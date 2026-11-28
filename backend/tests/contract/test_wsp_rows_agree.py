"""The contract's WSP list rows agree with the row-key lists (021 FR-021.21; FTASKS 8.5).

``tests/support/wsp_list_keys.py`` is the list miAssist's cross-repo test reads today; the answer
contract replaces it (021-2 deletes both). Until then the two must not drift: the properties the
contract declares for an endpoint row and a run row EQUAL ``ENDPOINT_ROW_KEYS`` and
``RUN_ROW_KEYS``.
"""

from __future__ import annotations

import json

from schemas import answers_contract
from tests.support.wsp_list_keys import ENDPOINT_ROW_KEYS, RUN_ROW_KEYS


def _row(route: str, items: str) -> set[str]:
    contract = json.loads(answers_contract.CONTRACT_PATH.read_text(encoding="utf-8"))
    defs = contract["$defs"]
    answer = defs[contract["responses"][route]["200"]["$ref"].rsplit("/", 1)[-1]]
    ref = answer["properties"][items]["items"]["$ref"]
    return set(defs[ref.rsplit("/", 1)[-1]]["properties"])


def test_the_endpoint_row_is_the_endpoint_row_keys() -> None:
    assert _row("GET /api/v1/wsp/endpoints", "endpoints") == set(ENDPOINT_ROW_KEYS)


def test_the_run_row_is_the_run_row_keys() -> None:
    assert _row("GET /api/v1/wsp/runs", "runs") == set(RUN_ROW_KEYS)
