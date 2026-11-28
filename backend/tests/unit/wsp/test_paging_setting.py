"""The paging setting (operator, 2026-11-21; ``0xcc/reviews/018_paging_2026-11-21.md``):
``paging`` ``on`` | ``off``, per endpoint (default ``on``) with a run override, ``miwsp_engine``'s
only.

- The value in effect is frozen into the request as ``paging: "off"`` when off and absent when on
  (the ``output_cap`` pattern), so ``request_sha256`` and the approval carry it, and every
  default run stays valid under miWorkbench's vendored copy of the run contract from before it.
- The engine reads what was frozen: absent pages, ``off`` does not (the push carries the request
  unchanged, so a test of the freeze plus one of the engine's reading is the whole path).
- The intake stores the engine's four paging counters; the list rows serve them.
"""

from __future__ import annotations

from typing import Any

import pytest

from core.errors import Refusal
from tests.unit.wsp.test_thinking import MISERVER, body, frozen
from wsp import endpoints, runs
from wsp.fixture_bundle import make_bundle
from wsp.schemas import validate_frozen_request
from wsp_worker.miwsp_engine import paging as engine_paging


@pytest.fixture(scope="module")
def bundle(tmp_path_factory: Any) -> dict[str, Any]:
    return make_bundle(tmp_path_factory.mktemp("b"), rule="mean").manifest


def test_the_engine_freezes_off_and_leaves_on_absent(bundle: dict[str, Any]) -> None:
    b = body(bundle)
    on = frozen(bundle, b)
    assert "paging" not in on.request and engine_paging.requested(on.request)
    off = frozen(bundle, b, {**MISERVER, "paging": "off"})
    assert off.request["paging"] == "off" and not engine_paging.requested(off.request)
    validate_frozen_request(off.request)
    assert off.request_sha256 != on.request_sha256
    # The run overrides the endpoint, both ways.
    assert frozen(bundle, {**b, "paging": "off"}).request["paging"] == "off"
    back = frozen(bundle, {**b, "paging": "on"}, {**MISERVER, "paging": "off"})
    assert "paging" not in back.request


@pytest.mark.parametrize("value", ["sometimes", True, 0])
def test_a_bad_value_is_refused(bundle: dict[str, Any], value: Any) -> None:
    with pytest.raises(Refusal) as caught:
        frozen(bundle, body(bundle, paging=value))
    assert caught.value.code == "validation_failed"
    with pytest.raises(Refusal):
        endpoints.check_paging(value)


def test_it_is_patchable_and_not_sealed() -> None:
    from wsp_gateway import sealing

    assert "paging" in endpoints.PATCHABLE
    assert "paging" not in sealing.BOUND_FIELDS


def test_the_intake_stores_the_engines_paging_counters() -> None:
    engine = {
        "action_protocol": "structured",
        "handoffs": 0,
        "refused_turns": 0,
        "kernel_restarts": 0,
        "pages_out": 12,
        "recalls": 2,
        "ledger_rows": 30,
        "peak_prompt_tokens": 8_675,
    }
    counters = runs.result_counters({"forward_used": 4, "engine": engine})
    assert {k: counters[k] for k in ("pages_out", "recalls", "ledger_rows")} == {
        "pages_out": 12,
        "recalls": 2,
        "ledger_rows": 30,
    }
    assert counters["peak_prompt_tokens"] == 8_675
    # An engine from before paging writes none: the columns stay null.
    old = {k: v for k, v in engine.items() if k in runs.ENGINE_KEYS}
    assert runs.result_counters({"engine": old})["pages_out"] is None
    with pytest.raises(Refusal):
        runs.result_counters({"engine": {**engine, "pages_out": -1}})
