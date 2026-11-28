"""Validation-only practice supplements through the real import (revision 2026-11-25, FU-7a C2).

``wsp.bundles.request_import`` and ``import_bundle`` (the task's body) on a migrated database,
against the recording fake miWorkbench, on fixture bundles whose every hash is real
(``wsp.fixture_bundle.make_bundle(supplements=...)``). Each refusal is driven by a manifest that
is RESEALED after the change, so only the supplement check can catch it, never the digest.

Negative controls (``0xcc/reviews/fu7a_c2_2026-11-25.md``): delete ``check_supplements(...)`` from
``bundles.check_files`` and every refusal below goes red; the wiring test also counts the calls.
"""

from __future__ import annotations

import json
from collections.abc import Callable, Iterator
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import sqlalchemy as sa
from sqlalchemy.orm import sessionmaker

from tests.integration.wsp.wsp_db import engine_at_head
from tests.support.fake_miworkbench import BASE, FakeMiWorkbench
from tests.support.wsp_fixtures import make_bundle, row_ids_sha256, seal
from wsp import bundles
from wsp.settings import WSPSettings
from wsp.weight_ops.hashing import file_sha256

SIZES = [12, 9]


class Env:
    def __init__(self, tmp_path: Path, **kwargs: Any) -> None:
        self.bundle = make_bundle(tmp_path / "src", rule="max", **kwargs)
        self.fake = FakeMiWorkbench(self.bundle)
        self.engine = engine_at_head(tmp_path)
        self.factory = sessionmaker(bind=self.engine, expire_on_commit=False)
        self.settings = WSPSettings(
            miworkbench_api_url=BASE, wsp_bundle_root=tmp_path / "wsp", wsp_providers_available=""
        )

    @property
    def manifest(self) -> dict[str, Any]:
        return self.fake.manifest if self.fake.manifest is not None else self.bundle.manifest

    def run(self) -> tuple[str, dict[str, Any]]:
        m = self.manifest
        body = {
            "owner": "miworkbench",
            "bundle_id": m["bundle_id"],
            "bundle_sha256": m["bundle_sha256"],
        }
        with self.factory() as session:
            row, _ = bundles.request_import(
                session, body, received_by="operator", settings=self.settings
            )
            session.commit()
            outcome = bundles.import_bundle(
                session, row.bundle_sha256, settings=self.settings, transport=self.fake.transport()
            )
            session.commit()
        with self.engine.connect() as conn:
            got = conn.execute(sa.text("SELECT * FROM mf_wsp_bundles")).mappings().one()
        return outcome, dict(got)

    def edit(self, change: Callable[[dict[str, Any]], None]) -> None:
        """Change the manifest, then reseal it, so its digest is right and only a deeper check can
        refuse it."""
        manifest = json.loads(json.dumps(self.bundle.manifest))
        change(manifest)
        for entry in manifest["files"]:
            path = self.bundle.root / entry["name"]
            entry["sha256"], entry["bytes"] = file_sha256(path), path.stat().st_size
        self.fake.manifest = seal(manifest)


@pytest.fixture()
def env(tmp_path: Path) -> Iterator[Env]:
    e = Env(tmp_path, supplements=SIZES, supplement_positives=2)
    yield e
    e.engine.dispose()


def _failure(row: dict[str, Any]) -> dict[str, Any]:
    value = row["import_failure"]
    return json.loads(value) if isinstance(value, str) else value


def _supplements(m: dict[str, Any]) -> list[dict[str, Any]]:
    return m["practice"]["supplements"]  # type: ignore[no-any-return]


def test_a_bundle_with_two_supplements_is_imported_with_every_file(env: Env) -> None:
    outcome, row = env.run()
    assert outcome == "imported", _failure(row)
    stored = bundles.store_dir(env.settings, env.manifest["bundle_sha256"])
    roles = {e["role"] for e in env.manifest["files"]}
    assert {r for r in roles if r.startswith("supplement_")} == {
        "supplement_activations",
        "supplement_offsets",
        "supplement_positions",
        "supplement_labels",
        "supplement_truncated_mask",
    }
    for entry in env.manifest["files"]:
        assert file_sha256(stored / entry["name"]) == entry["sha256"]
    manifest = row["manifest"] if isinstance(row["manifest"], dict) else json.loads(row["manifest"])
    assert [s["rows"] for s in _supplements(manifest)] == SIZES


def test_a_bundle_without_supplements_is_unchanged(tmp_path: Path) -> None:
    """The additive half: the manifest has no ``supplements`` key and no supplement file."""
    e = Env(tmp_path)
    try:
        assert "supplements" not in e.bundle.manifest["practice"]
        assert not any(f["role"].startswith("supplement_") for f in e.bundle.manifest["files"])
        outcome, row = e.run()
        assert outcome == "imported", _failure(row)
    finally:
        e.engine.dispose()


def test_check_files_runs_the_supplement_check_once_with_the_manifest(
    env: Env, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Wiring, with payload and count: the import's ``check_files`` hands ``check_supplements``
    the manifest and the files by role, exactly once."""
    calls: list[tuple[dict[str, Any], dict[str, Path]]] = []
    real = bundles.check_supplements

    def spy(manifest: dict[str, Any], paths: dict[str, Path], tensors: Any) -> None:
        calls.append((manifest, paths))
        real(manifest, paths, tensors)

    monkeypatch.setattr(bundles, "check_supplements", spy)
    outcome, row = env.run()
    assert outcome == "imported", _failure(row)
    assert len(calls) == 1
    manifest, paths = calls[0]
    assert manifest["bundle_sha256"] == env.manifest["bundle_sha256"]
    assert {"supplement_activations", "supplement_labels"} <= set(paths)


def _shift_parity_score(m: dict[str, Any]) -> None:
    _supplements(m)[1]["parity"]["scores"][3] += 0.01


def _lie_label_counts(m: dict[str, Any]) -> None:
    counts = _supplements(m)[0]["label_counts"]
    counts["positive"], counts["negative"] = counts["positive"] + 1, counts["negative"] - 1


def _lie_tokens(m: dict[str, Any]) -> None:
    _supplements(m)[0]["tokens"] += 1
    _supplements(m)[1]["tokens"] -= 1


def _lie_truncated(m: dict[str, Any]) -> None:
    _supplements(m)[1]["truncated_rows"] += 1


def _gap_first_row(m: dict[str, Any]) -> None:
    _supplements(m)[1]["first_row"] += 1


def _wrong_row_ids(m: dict[str, Any]) -> None:
    _supplements(m)[0]["row_ids_sha256"] = row_ids_sha256("view-supplement-0", [1, 2, 3])


def _heldout_as_supplement(m: dict[str, Any]) -> None:
    _supplements(m)[0]["view_id"] = m["heldout_ref"]["sets"][0]["view_id"]


def _practice_as_supplement(m: dict[str, Any]) -> None:
    _supplements(m)[0]["view_id"] = m["practice"]["view_id"]


def _twice(m: dict[str, Any]) -> None:
    _supplements(m)[1]["view_id"] = _supplements(m)[0]["view_id"]


def _parity_past_the_end(m: dict[str, Any]) -> None:
    _supplements(m)[1]["parity"]["rows"][-1] = SIZES[1]


def _entries_without_files(m: dict[str, Any]) -> None:
    m["files"] = [f for f in m["files"] if f["role"] != "supplement_labels"]


def _files_without_entries(m: dict[str, Any]) -> None:
    del m["practice"]["supplements"]


#: case -> (the change, the refusal code, words of THAT check's refusal): the words pin which check
#: refused, so a case another check (or a crash) happens to catch cannot hide a missing one
#: (review control M6: with only the code, removing the first_row check stayed green because an
#: IndexError ended the import ``wsp_bundle_invalid`` too).
REFUSALS: dict[str, tuple[Callable[[dict[str, Any]], None], str, str]] = {
    "parity": (_shift_parity_score, "wsp_bundle_parity_failed", "view-supplement-1's parity"),
    "label_counts": (_lie_label_counts, "wsp_bundle_invalid", "label counts or truncated rows"),
    "tokens": (_lie_tokens, "wsp_bundle_invalid", "its tokens, label counts"),
    "truncated_rows": (_lie_truncated, "wsp_bundle_invalid", "label counts or truncated rows"),
    "first_row": (_gap_first_row, "wsp_bundle_invalid", "follow one another"),
    "row_ids": (_wrong_row_ids, "wsp_bundle_hash_mismatch", "supplements[0].row_ids_sha256"),
    "heldout_view": (_heldout_as_supplement, "wsp_bundle_invalid", "never practice"),
    "practice_view": (_practice_as_supplement, "wsp_bundle_invalid", "never practice"),
    "listed_twice": (_twice, "wsp_bundle_invalid", "listed twice"),
    "parity_row_past_the_end": (_parity_past_the_end, "wsp_bundle_invalid", "past the supplement"),
    "entries_without_files": (_entries_without_files, "wsp_bundle_invalid", "needs the files"),
    "files_without_entries": (
        _files_without_entries,
        "wsp_bundle_invalid",
        "no practice.supplements",
    ),
}


@pytest.mark.parametrize("case", sorted(REFUSALS))
def test_a_supplement_that_disagrees_with_its_files_is_refused(env: Env, case: str) -> None:
    change, code, words = REFUSALS[case]
    env.edit(change)
    outcome, row = env.run()
    assert outcome == f"failed:{code}", (case, _failure(row))
    assert words in _failure(row)["message"], (case, _failure(row))
    assert not bundles.store_dir(env.settings, env.manifest["bundle_sha256"]).exists()


def test_a_changed_supplement_activation_fails_parity(env: Env) -> None:
    """The bytes, not only the manifest: one supplement token row changed under a resealed
    manifest is caught by the supplement's own parity."""
    path = env.bundle.root / "supplement_activations.f32"
    acts = np.fromfile(path, dtype="<f4").reshape(-1, 8)
    acts[:, :] += 0.5
    acts.astype("<f4").tofile(path)
    env.edit(lambda m: None)
    outcome, row = env.run()
    assert outcome == "failed:wsp_bundle_parity_failed", _failure(row)
    assert "view-supplement-0" in _failure(row)["message"]


def test_a_label_outside_zero_and_one_is_refused(env: Env) -> None:
    path = env.bundle.root / "supplement_labels.npy"
    labels = np.load(path)
    labels[0] = 2
    np.save(path, labels)
    env.edit(lambda m: None)
    outcome, row = env.run()
    assert outcome == "failed:wsp_bundle_invalid", _failure(row)
