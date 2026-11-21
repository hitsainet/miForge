"""The random control (feature 016; FTASKS 14.1, 14.0a; FR-016.27; R1B-A1, R1B-A8)."""

from __future__ import annotations

import math

import numpy as np
import pytest

from wsp.weight_ops.control import control_seed, random_control
from wsp.weight_ops.tensor import Tensor, from_array, from_values

CONTENT = "3f" * 32


def _pair(dtype: str = "float32", seed: int = 0) -> tuple[dict[str, Tensor], dict[str, Tensor]]:
    """Mixed magnitudes: a sparse edit spanning 1e-6 to 1e3 on top of a mixed base."""
    rng = np.random.default_rng(seed)
    base_w = (rng.standard_normal((16, 24)) * np.logspace(-4, 3, 24)).astype(np.float32)
    base_b = rng.standard_normal(5).astype(np.float32)
    edit_w = base_w.copy()
    idx = rng.choice(base_w.size, size=40, replace=False)
    edit_w.reshape(-1)[idx] += (rng.standard_normal(40) * np.logspace(-6, 3, 40)).astype(np.float32)
    edit_b = base_b.copy()
    edit_b[2] += np.float32(0.5)
    base = {"weight": from_values(dtype, base_w), "bias": from_values(dtype, base_b)}
    edited = {"weight": from_values(dtype, edit_w), "bias": from_values(dtype, edit_b)}
    return base, edited


def _independent(base: Tensor, cast: Tensor) -> tuple[int, float]:
    """Recompute the cast control's figures with nothing from the module under test."""
    b = base.values64().reshape(-1).tolist()
    c = cast.values64().reshape(-1).tolist()
    diffs = [y - x for x, y in zip(b, c, strict=True)]
    return sum(1 for d in diffs if d != 0.0), math.sqrt(math.fsum(d * d for d in diffs))


@pytest.mark.parametrize("dtype", ["float32", "float16", "bfloat16"])
def test_pre_cast_count_and_norm_equal_the_candidates_exactly(dtype: str) -> None:
    base, edited = _pair(dtype)
    control = random_control(base, edited, ["weight", "bias"], CONTENT)
    assert control.pre_cast == control.candidate
    assert control.candidate.changed_entries == sum(
        int(np.count_nonzero(edited[n].values64() - base[n].values64())) for n in base
    )


@pytest.mark.parametrize("dtype", ["float32", "float16", "bfloat16"])
def test_the_recorded_post_cast_figures_equal_an_independent_recomputation(dtype: str) -> None:
    base, edited = _pair(dtype, seed=3)
    control = random_control(base, edited, ["bias", "weight"], CONTENT)
    counts, squares = 0, []
    for name in ("bias", "weight"):
        n, l2 = _independent(base[name], control.tensors[name])
        counts += n
        squares.append(l2 * l2)
    assert control.post_cast.changed_entries == counts
    assert control.post_cast.l2_delta == pytest.approx(math.sqrt(math.fsum(squares)), rel=1e-12)


def test_the_support_differs_and_the_signs_are_mixed() -> None:
    base, edited = _pair("float64")  # no cast loss: the deltas are exactly the moved ones
    control = random_control(base, edited, ["weight"], CONTENT)
    d_edit = (edited["weight"].values64() - base["weight"].values64()).reshape(-1)
    d_ctrl = (control.tensors["weight"].values64() - base["weight"].values64()).reshape(-1)
    assert set(np.flatnonzero(d_edit)) != set(np.flatnonzero(d_ctrl))
    # The seeded sign flip is not the identity: the multiset of deltas changes while the
    # multiset of magnitudes does not (C27 drops the flip, and the first assert goes red).
    moved = np.sort(d_ctrl[d_ctrl != 0])
    original = np.sort(d_edit[d_edit != 0])
    assert not np.array_equal(moved, original), "the signs were never flipped"
    assert np.array_equal(np.sort(np.abs(moved)), np.sort(np.abs(original)))


def test_the_same_content_gives_the_same_control_and_n_moves_it() -> None:
    base, edited = _pair()
    one = random_control(base, edited, ["weight", "bias"], CONTENT)
    two = random_control(base, edited, ["bias", "weight"], CONTENT)  # order does not matter
    assert all(one.tensors[n].same_bits(two.tensors[n]) for n in one.tensors)
    other = random_control(base, edited, ["weight", "bias"], CONTENT, n=1)
    assert other.seed == one.seed + 1
    assert not other.tensors["weight"].same_bits(one.tensors["weight"])
    third = random_control(base, edited, ["weight", "bias"], "40" * 32)
    assert not third.tensors["weight"].same_bits(one.tensors["weight"])


def test_the_seed_is_the_first_eight_hex_digits_plus_n() -> None:
    assert control_seed(CONTENT) == int("3f3f3f3f", 16)
    assert control_seed(CONTENT, 4) == int("3f3f3f3f", 16) + 4
    with pytest.raises(ValueError):
        control_seed("abc")


def test_only_the_patched_tensors_are_controlled() -> None:
    base, edited = _pair()
    control = random_control(base, edited, ["bias"], CONTENT)
    assert set(control.tensors) == {"bias"}


def test_a_float64_edit_round_trips_without_cast_loss() -> None:
    w = np.linspace(-1, 1, 12)
    e = w.copy()
    e[[1, 7]] += [0.25, -2.0]
    control = random_control({"w": from_array(w)}, {"w": from_array(e)}, ["w"], CONTENT)
    assert control.post_cast == control.pre_cast == control.candidate
