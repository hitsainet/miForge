"""FR-016.3: inspect is deterministic, its singular values are numpy's, and it is canonical JSON."""

from __future__ import annotations

import json

import numpy as np

from tests.support.wsp_fixtures import make_bundle
from wsp.weight_ops.hashing import jcs_sha256
from wsp.weight_ops.inspect import inspect_response, inspect_tensors
from wsp.weight_ops.tensor import from_array


def test_two_calls_return_byte_identical_bodies(tmp_path) -> None:
    fx = make_bundle(tmp_path, rule="attention", seed=1)
    first = inspect_response(fx.tensors, cosine_with=["attention_query"])
    second = inspect_response(
        dict(reversed(list(fx.tensors.items()))), cosine_with=["attention_query"]
    )
    assert first == second
    body, digest = first
    assert digest == jcs_sha256(json.loads(body))


def test_singular_values_match_numpy_within_1e_6_relative() -> None:
    rng = np.random.default_rng(0)
    m = rng.normal(size=(40, 64)).astype(np.float32)
    report = inspect_tensors({"W_dec": from_array(m)}, bundle_kind="sae")
    got = np.array(report["tensors"][0]["singular_values"])
    want = np.linalg.svd(m.astype(np.float64), compute_uv=False)[:32]
    assert got.shape == (32,)
    np.testing.assert_allclose(got, want, rtol=1e-6)


def test_probe_block_effective_norm_and_cosine(tmp_path) -> None:
    fx = make_bundle(tmp_path, rule="mean", seed=2)
    report = inspect_tensors(fx.tensors, cosine_with=["weight"])
    w = fx.tensors["weight"].values64()
    std = fx.tensors["std"].values64()
    effective = np.where(np.abs(std) <= 1e-6, 0.0, w / np.where(np.abs(std) <= 1e-6, 1.0, std))
    assert np.isclose(report["probe"]["effective_norm"], np.linalg.norm(effective), rtol=1e-12)
    cos = effective @ w / (np.linalg.norm(effective) * np.linalg.norm(w))
    assert np.isclose(report["probe"]["cosine_with"]["weight"], cos, rtol=1e-12)


def test_non_finite_values_are_null_not_nan() -> None:
    report = inspect_tensors({"weight": from_array(np.array([1.0, np.nan], dtype=np.float32))})
    body = json.dumps(report)
    assert "NaN" not in body
