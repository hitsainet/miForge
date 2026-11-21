"""FR-016.2: the evaluator reproduces a bundle's parity scores.

It refuses a bundle that does not agree.

The fixtures' parity scores come from ``tests/support/wsp_fixtures.reference_row_score``, an
independent
pure-Python reference, so a pass here is agreement between two implementations of miWorkbench's
definition, not an implementation agreeing with itself.
"""

from __future__ import annotations

import numpy as np
import pytest

from core.errors import Refusal
from tests.support.wsp_fixtures import RULES, make_bundle
from wsp.weight_ops.evaluator import parity, require_parity, score_rows
from wsp.weight_ops.practice import load_practice
from wsp.weight_ops.tensor import from_array


def _check(fx) -> object:
    practice = load_practice(fx.paths_by_role, fx.manifest["practice"])
    return parity(fx.manifest["function"], fx.tensors, practice.activations, practice.offsets)


@pytest.mark.parametrize("rule", RULES)
def test_each_rule_passes_on_its_own_fixture(rule: str, tmp_path) -> None:
    fx = make_bundle(
        tmp_path,
        rule=rule,
        seed=3,
        window=8 if rule == "rolling_mean_max" else None,
        tau=0.5 if rule == "softmax" else None,
    )
    report = _check(fx)
    assert report.ok, report
    require_parity(report)


@pytest.mark.parametrize("rule", RULES)
def test_a_float16_practice_set_passes(rule: str, tmp_path) -> None:
    fx = make_bundle(tmp_path, rule=rule, seed=5, practice_dtype="float16")
    assert _check(fx).ok


def test_a_permuted_weight_is_refused(tmp_path) -> None:
    fx = make_bundle(tmp_path, rule="mean", seed=1)
    w = fx.tensors["weight"].values64().astype(np.float32)
    fx.tensors["weight"] = from_array(np.roll(w, 1))
    report = _check(fx)
    assert not report.ok
    with pytest.raises(Refusal) as err:
        require_parity(report)
    assert err.value.code == "wsp_bundle_parity_failed"
    assert err.value.details["worst_row"] in fx.manifest["function"]["parity"]["rows"]


def test_rolled_parity_labels_are_refused(tmp_path) -> None:
    """The parity rows rolled by one against their scores: every score now names the wrong row."""
    fx = make_bundle(tmp_path, rule="mean", seed=2)
    rows = fx.manifest["function"]["parity"]["rows"]
    fx.manifest["function"]["parity"]["rows"] = rows[1:] + rows[:1]
    assert not _check(fx).ok


def test_the_degenerate_channel_is_zeroed_not_divided(tmp_path) -> None:
    """A channel with std 0 contributes nothing, whatever its weight (miWorkbench's rule)."""
    fx = make_bundle(tmp_path, rule="max", seed=4, degenerate_channel=True)
    practice = load_practice(fx.paths_by_role, fx.manifest["practice"])
    before = score_rows(
        fx.manifest["function"], fx.tensors, practice.activations, practice.offsets, range(8)
    )
    w = fx.tensors["weight"].values64().astype(np.float32)
    w[-1] = 1e6
    fx.tensors["weight"] = from_array(w)
    after = score_rows(
        fx.manifest["function"], fx.tensors, practice.activations, practice.offsets, range(8)
    )
    assert np.array_equal(before, after)


def test_ignoring_std_would_be_caught_on_the_max_rule(tmp_path) -> None:
    """The fixture's channels have non-unit scale, so a no-std evaluator gives different scores."""
    fx = make_bundle(tmp_path, rule="max", seed=6)
    fx.tensors.pop("std")
    assert not _check(fx).ok


def test_sae_function_is_refused_until_phase_4(tmp_path) -> None:
    fx = make_bundle(tmp_path, rule="mean", seed=7)
    fx.manifest["function"]["kind"] = "sae/v1"
    with pytest.raises(Refusal) as err:
        _check(fx)
    assert err.value.code == "wsp_phase_not_enabled"


def test_attention_without_a_query_is_refused(tmp_path) -> None:
    fx = make_bundle(tmp_path, rule="attention", seed=8)
    fx.tensors.pop("attention_query")
    with pytest.raises(Refusal) as err:
        _check(fx)
    assert err.value.code == "wsp_bundle_invalid"


def test_a_pickled_practice_file_is_refused(tmp_path) -> None:
    fx = make_bundle(tmp_path, rule="mean", seed=9)
    np.save(
        fx.paths_by_role["practice_labels"],
        np.array([object()] * 24, dtype=object),
        allow_pickle=True,
    )
    with pytest.raises(Refusal) as err:
        load_practice(fx.paths_by_role, fx.manifest["practice"])
    assert err.value.code == "wsp_bundle_format_refused"
