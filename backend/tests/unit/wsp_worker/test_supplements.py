"""Validation-only practice supplements on the worker (revision 2026-11-25, FU-7a C2).

- ``wsp_worker.bundle.load`` reads a bundle's supplements beside practice, and refuses files
  without entries or entries without files;
- the engine's observation context carries them (wiring: remove ``supplement=`` from
  ``session.observation_context`` and the wiring test goes red);
- they are validation ONLY: no practice split, no program support set and no bar basis holds a
  supplement row (the practice arrays are exactly practice's, row for row);
- ``supplement_metrics`` 1 answers each supplement on its own, at each head's own judged bar,
  against an independent pure-Python reference (``reference_row_score`` and the order statistic
  written out here), never against the scorer it checks.

The engine's unlink of the supplement files is pinned by ``miwsp_engine/test_engine_security.py``,
whose fixture now carries a supplement and whose probed paths are every non-heads file.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pytest

from tests.support.wsp_fixtures import FixtureBundle, make_bundle, reference_row_score
from wsp_worker import bundle as bundle_mod
from wsp_worker import observation_tools as ot
from wsp_worker import program

SIZES = [12, 9]
TARGET = 0.25


def _files(fx: FixtureBundle) -> list[dict[str, Any]]:
    return [{k: e[k] for k in ("name", "role", "sha256")} for e in fx.manifest["files"]]


@pytest.fixture()
def fx(tmp_path: Path) -> FixtureBundle:
    return make_bundle(
        tmp_path / "b",
        rule="mean",
        rows=80,
        supplements=SIZES,
        supplement_positives=3,
        target_fpr=TARGET,
    )


def test_the_worker_reads_the_supplements_beside_practice(fx: FixtureBundle) -> None:
    loaded = bundle_mod.load(fx.root, fx.manifest, _files(fx))
    supplement = loaded.practice.supplement
    assert supplement is not None
    assert [e["rows"] for e in supplement.entries] == SIZES
    assert supplement.arrays["supplement_offsets"].shape == (sum(SIZES) + 1,)
    assert supplement.arrays["supplement_labels"].shape == (sum(SIZES),)
    assert supplement.activations.shape[0] == sum(e["tokens"] for e in supplement.entries)
    # Practice is exactly practice's: not one supplement row is in its arrays.
    assert loaded.practice.arrays["practice_labels"].shape == (fx.manifest["practice"]["rows"],)


def test_a_bundle_without_supplements_reads_none(tmp_path: Path) -> None:
    plain = make_bundle(tmp_path / "p", rule="mean")
    assert bundle_mod.load(plain.root, plain.manifest, _files(plain)).practice.supplement is None


def test_supplement_files_without_entries_are_refused(fx: FixtureBundle) -> None:
    manifest = {**fx.manifest, "practice": dict(fx.manifest["practice"])}
    del manifest["practice"]["supplements"]
    with pytest.raises(bundle_mod.BundleError, match=r"without practice\.supplements"):
        bundle_mod.load(fx.root, manifest, _files(fx))


def test_entries_without_every_supplement_file_are_refused(fx: FixtureBundle) -> None:
    manifest = {
        **fx.manifest,
        "files": [f for f in fx.manifest["files"] if f["role"] != "supplement_truncated_mask"],
    }
    pushed = [f for f in _files(fx) if f["role"] != "supplement_truncated_mask"]
    with pytest.raises(bundle_mod.BundleError, match="all five"):
        bundle_mod.load(fx.root, manifest, pushed)


def _session_view(tmp_path: Path, fx: FixtureBundle) -> ot.PracticeView:
    from tests.support.wsp_engine import line_for
    from wsp_worker.miwsp_engine import session as session_mod

    s = session_mod.Session(line_for(tmp_path / "run", fx, "repair", None, python=False))
    return session_mod.observation_context(s.bundle, s.store).view


def test_the_engine_view_carries_the_supplements(tmp_path: Path, fx: FixtureBundle) -> None:
    """Wiring: the engine's observation context hands the tools the bundle's supplements, with
    their own scorer over their own rows."""
    view = _session_view(tmp_path, fx)
    assert view.supplement is not None
    assert [e["view_id"] for e in view.supplement.entries] == [
        s["view_id"] for s in fx.manifest["practice"]["supplements"]
    ]
    assert view.supplement.scorer.rows == sum(SIZES)
    assert view.rows == fx.manifest["practice"]["rows"]


def test_no_practice_split_support_set_or_bar_basis_holds_a_supplement_row(
    tmp_path: Path, fx: FixtureBundle
) -> None:
    view = _session_view(tmp_path, fx)
    rows = fx.manifest["practice"]["rows"]
    for split in ot.SPLITS:
        assert int(view.split(split).max()) < rows
    assert int(view.judged_rows().max()) < rows
    loaded = bundle_mod.load(fx.root, fx.manifest, _files(fx))
    for kind in program.SUPPORT_KINDS:
        assert int(program.SupportSet(kind, loaded).rows.max()) < rows


# ------------------------------------------------------------------ supplement_metrics 1


def _reference_scores(
    fx: FixtureBundle,
    tensors: dict[str, np.ndarray],
    acts: np.ndarray,
    offsets: np.ndarray,
    rows: Any,
) -> list[float]:
    w = [float(v) for v in tensors["weight"]]
    m = [float(v) for v in tensors["mean"]]
    s = [float(v) for v in tensors["std"]]
    b = float(np.asarray(tensors["bias"]).reshape(-1)[0])
    out = []
    for r in rows:
        toks = [[float(x) for x in row] for row in acts[offsets[r] : offsets[r + 1]]]
        out.append(reference_row_score("mean", toks, w, b, m, s, None, 1.0, 16))
    return out


def _own_bar(negatives: list[float], target: float) -> float | None:
    """miWorkbench's ``calibrate``: the ``int(target * n)``-th highest negative, written out."""
    ordered = sorted(negatives, reverse=True)
    allowed = int(target * len(ordered))
    return ordered[allowed - 1] if allowed > 0 else None


def _expected(fx: FixtureBundle, tensors: dict[str, np.ndarray]) -> list[dict[str, Any]]:
    practice = fx.manifest["practice"]
    validation = np.load(fx.root / "practice_validation_mask.npy").astype(bool)
    truncated = np.load(fx.root / "practice_truncated_mask.npy").astype(bool)
    labels = np.load(fx.root / "practice_labels.npy")
    basis = [int(r) for r in np.flatnonzero(validation & ~truncated & (labels == 0))]
    bar = _own_bar(
        _reference_scores(fx, tensors, fx.activations.astype(np.float64), fx.offsets, basis),
        TARGET,
    )
    suffix = "f32" if practice["dtype"] == "float32" else "f16"
    acts = np.fromfile(fx.root / f"supplement_activations.{suffix}", dtype="<f4").reshape(
        -1, practice["d_model"]
    )
    offsets = np.load(fx.root / "supplement_offsets.npy")
    s_labels = np.load(fx.root / "supplement_labels.npy")
    s_truncated = np.load(fx.root / "supplement_truncated_mask.npy").astype(bool)
    out = []
    for entry in practice["supplements"]:
        span = range(entry["first_row"], entry["first_row"] + entry["rows"])
        rows = [r for r in span if not s_truncated[r]]
        scores = _reference_scores(fx, tensors, acts.astype(np.float64), offsets, rows)
        y = [int(s_labels[r]) for r in rows]
        fired = [bar is not None and sc >= bar for sc in scores]
        negatives = [f for f, label in zip(fired, y, strict=True) if label == 0]
        positives = [f for f, label in zip(fired, y, strict=True) if label == 1]
        out.append(
            {
                "bar": bar,
                "false_alarms": sum(negatives),
                "negatives": len(negatives),
                "positives": len(positives),
                "caught": sum(positives),
            }
        )
    return out


def _tool_answer(tmp_path: Path, fx: FixtureBundle, edit: Any) -> dict[str, Any]:
    view = _session_view(tmp_path, fx)
    parent = {n: t.values64() for n, t in fx.tensors.items()}
    edited = edit(parent)
    ctx = ot.Context(view, lambda: dict(edited), parent, seed=7)
    [spec] = ot.build([{"name": "supplement_metrics", "version": "1", "cost": 2}], ctx)
    return spec.fn()


@pytest.mark.parametrize("delta", [0.0, 0.7])
def test_supplement_metrics_matches_the_reference_per_supplement(
    tmp_path: Path, fx: FixtureBundle, delta: float
) -> None:
    def edit(p: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
        w = p["weight"].copy()
        w[0] += delta
        return {**p, "weight": w}

    answer = _tool_answer(tmp_path, fx, edit)
    parent = {n: t.values64() for n, t in fx.tensors.items()}
    assert answer["tool"] == "supplement_metrics" and answer["supplements"] == len(SIZES)
    assert answer["basis"] == "practice_validation_negatives"
    assert answer["target_false_alarm_rate"] == TARGET
    for side, tensors in (("now", edit(parent)), ("parent", parent)):
        for index, want in enumerate(_expected(fx, tensors)):
            got = answer["sources"][f"supplement_{index}"][side]
            assert got["bar"] == pytest.approx(want["bar"], abs=1e-4)
            assert got["false_alarms"] == want["false_alarms"], (side, index)
            assert (got["negatives"], got["positives"]) == (want["negatives"], want["positives"])
            recall = None if not want["positives"] else want["caught"] / want["positives"]
            assert (
                got["recall"] == pytest.approx(recall)
                if recall is not None
                else got["recall"] is None
            )
    for index, entry in enumerate(fx.manifest["practice"]["supplements"]):
        cell = answer["sources"][f"supplement_{index}"]
        assert (cell["rows"], cell["rows_counted"], cell["view_id"]) == (
            entry["rows"],
            entry["rows"] - entry["truncated_rows"],
            entry["view_id"],
        )


def test_supplement_metrics_on_a_bundle_without_supplements(tmp_path: Path) -> None:
    plain = make_bundle(tmp_path / "p", rule="mean")
    answer = _tool_answer(tmp_path, plain, lambda p: p)
    assert answer["supplements"] == 0 and answer["sources"] == {}


def test_supplement_metrics_reads_only_supplement_rows_through_its_own_scorer(
    tmp_path: Path, fx: FixtureBundle
) -> None:
    """The supplement scorer reads only its own rows, and the practice scorer reads only the
    bar basis (validation negatives not truncated): nothing else is touched."""
    view = _session_view(tmp_path, fx)
    assert view.supplement is not None
    touched: dict[str, set[int]] = {"practice": set(), "supplement": set()}

    def recorder(scorer: Any, key: str) -> None:
        real = scorer.read_rows

        def read_rows(first: int, stop: int) -> np.ndarray:
            touched[key].update(range(int(first), int(stop)))
            return real(first, stop)

        scorer.read_rows = read_rows

    recorder(view.scorer, "practice")
    recorder(view.supplement.scorer, "supplement")
    parent = {n: t.values64() for n, t in fx.tensors.items()}
    ctx = ot.Context(view, lambda: dict(parent), parent, seed=7)
    ot.supplement_metrics(ctx)
    assert touched["supplement"] <= set(range(sum(SIZES)))
    assert touched["practice"] <= {int(r) for r in view.judged_rows()}


def test_the_reference_case_is_not_vacuous(fx: FixtureBundle) -> None:
    """The golden case must cut a real bar and see false alarms, or every comparison above could
    pass on ``None`` and zeros."""
    parent = {n: t.values64() for n, t in fx.tensors.items()}
    expected = _expected(fx, parent)
    assert all(cell["bar"] is not None for cell in expected)
    assert sum(cell["false_alarms"] for cell in expected) > 0
    assert sum(cell["positives"] for cell in expected) == 3 * len(SIZES)
    assert [s["truncated_rows"] for s in fx.manifest["practice"]["supplements"]] == [1, 1]
