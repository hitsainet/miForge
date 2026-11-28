"""FR-021.13's acceptance: every contract route answers in exactly one router module.

Each ``test_<router>_answer.py`` here holds its own router's set equality; this test holds the
whole: every router that serves a contract route has its module, and the routes the modules
cover, together, are every key of ``responses`` and ``excluded``, each in one module only.
"""

from __future__ import annotations

import importlib
from pathlib import Path

from tests.support import answer_cases, answer_validator

HERE = Path(__file__).resolve().parent


def test_the_modules_cover_every_contract_route_once() -> None:
    for path in sorted(HERE.glob("test_*_answer.py")):
        importlib.import_module(f"tests.integration.answers.{path.stem}")
    contract = answer_validator.answer_check().contract
    every = set(contract["responses"]) | set(contract["excluded"])
    routers = {
        name.rsplit(".", 1)[0]
        for name, methods in answer_validator.endpoint_keys().items()
        if set(methods.values()) & every
    }
    assert set(answer_cases.CASES) == routers, sorted(routers ^ set(answer_cases.CASES))
    seen: dict[str, str] = {}
    for router, keys in answer_cases.CASES.items():
        for key in keys:
            assert key not in seen, f"{key} is covered by {seen[key]} and {router}"
            seen[key] = router
    assert set(seen) == every, sorted(every ^ set(seen))
