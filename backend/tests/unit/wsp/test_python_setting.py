"""018 FU-8 (F18-D15, FR-018.5 "a request can switch Python off"), with the run-contract revision
of 2026-11-25 (``0xcc/reviews/run_contract_r2_2026-11-25.md``).

- The start body's ``python`` (``on`` | ``off``; omitted or null, ``on``) is frozen into the
  request as ``python: "off"`` when off and absent when on (the ``paging`` pattern), so
  ``request_sha256``, the approval and a replay carry it, and every default run stays valid under
  a vendored copy of the run contract from before it.
- The engine reads what was frozen: ``off`` offers no ``python`` tool and starts no kernel; absent
  offers it. The push carries the request unchanged, so the freeze plus the engine's reading, as
  the endpoint RECEIVED the tool list, is the whole path.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from core.errors import Refusal
from tests.support.wsp_fixtures import FixtureBundle
from tests.support.wsp_fixtures import make_bundle as make_fixture
from tests.unit.wsp.test_thinking import body, frozen
from tests.unit.wsp_worker.miwsp_engine.test_engine_scripted import call, session
from wsp import freeze as freeze_mod
from wsp.fixture_bundle import make_bundle
from wsp.schemas import validate_frozen_request


@pytest.fixture(scope="module")
def bundle(tmp_path_factory: Any) -> dict[str, Any]:
    return make_bundle(tmp_path_factory.mktemp("b"), rule="mean").manifest


def test_off_is_frozen_and_on_is_absent(bundle: dict[str, Any]) -> None:
    default = frozen(bundle, body(bundle))
    assert "python" not in default.request
    on = frozen(bundle, body(bundle, python="on"))
    assert "python" not in on.request and on.request_sha256 == default.request_sha256
    off = frozen(bundle, body(bundle, python="off"))
    assert off.request["python"] == "off"
    validate_frozen_request(off.request)
    assert off.request_sha256 != default.request_sha256
    assert frozen(bundle, body(bundle, python=None)).request_sha256 == default.request_sha256


@pytest.mark.parametrize("value", ["sometimes", True, False, 0, "OFF"])
def test_a_bad_value_is_refused(bundle: dict[str, Any], value: Any) -> None:
    with pytest.raises(Refusal) as caught:
        frozen(bundle, body(bundle, python=value))
    assert caught.value.code == "validation_failed"
    assert "python is one of on, off" in caught.value.message


def test_the_contract_admits_only_off() -> None:
    with pytest.raises(Refusal):
        validate_frozen_request({"python": "on"})
    assert freeze_mod.PYTHON_VALUES == ("on", "off")


@pytest.fixture()
def fx(tmp_path: Path) -> FixtureBundle:
    return make_fixture(tmp_path / "bundle", rule="mean")


@pytest.mark.parametrize(("kind", "edit"), [("repair", True), ("create", True), ("inspect", False)])
def test_the_engine_offers_no_python_tool_when_the_request_says_off(
    tmp_path: Path, fx: FixtureBundle, kind: str, edit: bool
) -> None:
    ep, _run, _ = session(tmp_path / "off", fx, kind, [call("submit", answer="a")], python="off")
    assert "python" not in ep.tool_names(0)
    assert ("write_tensor" in ep.tool_names(0)) is edit
    ep_on, _, _ = session(tmp_path / "on", fx, kind, [call("submit", answer="a")], python=True)
    assert "python" in ep_on.tool_names(0)


def test_a_python_call_on_an_off_run_is_not_run(tmp_path: Path, fx: FixtureBundle) -> None:
    steps = [call("python", code="W['weight'][0] = 9.0"), call("submit", answer="a")]
    _, run, _ = session(tmp_path, fx, "repair", steps, python="off")
    assert not (run / "out" / "patch" / "patch.json").exists()
