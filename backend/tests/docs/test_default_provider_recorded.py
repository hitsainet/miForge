"""The default WSP provider may become ``miwsp_engine`` only with the comparison on record (feature
018; FR-018.17; FTASKS 14.1; D27).

The EFFECTIVE default is what the deployed backend reads: ``WSPSettings.wsp_default_provider``
built from the environment the pod gets, i.e. the code default overridden by
``WSP_DEFAULT_PROVIDER`` in ``gitops/pilot/config.yaml``. It is computed by the settings class
itself, never by reading source text.

When that default is ``miwsp_engine``, ``0xcc/reviews/018_switch.md`` must exist and carry a
``switch`` block that marks every one of D27's four measures ``equal_or_better`` on both bundles
(the phase-1 fixture and the first phase-2 bundle)::

    ```switch
    phase1_fixture:
      sessions_completed: equal_or_better
      refused_turns: equal_or_better
      heldout_verdict: equal_or_better
      replay_bit_for_bit: equal_or_better
    phase2_bundle:
      ...
    ```

The block is the record's claim; the numbers behind it are the record's prose and tables.
Controls (FTASKS 14.1): the default flipped with no record goes red; a record with one measure
``worse`` goes red.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import pytest
import yaml

from tests.support.mirror_snapshot import mirror_snapshot

ROOT = Path(__file__).resolve().parents[3]
CONFIG = ROOT / "gitops" / "pilot" / "config.yaml"
RECORD = ROOT / "0xcc" / "reviews" / "018_switch.md"
ON_MIRROR = mirror_snapshot(ROOT)

ENGINE = "miwsp_engine"
BUNDLES = ("phase1_fixture", "phase2_bundle")
MEASURES = ("sessions_completed", "refused_turns", "heldout_verdict", "replay_bit_for_bit")
BLOCK = re.compile(r"^```switch\n(.*?)^```", re.MULTILINE | re.DOTALL)


def configmap_env(path: Path = CONFIG) -> dict[str, str]:
    """The ConfigMap's ``data`` (the environment every backend container gets)."""
    documents = [d for d in yaml.safe_load_all(path.read_text("utf-8")) if d]
    maps = [d for d in documents if d.get("kind") == "ConfigMap"]
    assert len(maps) == 1, f"{path}: expected one ConfigMap, found {len(maps)}"
    return {str(k): str(v) for k, v in (maps[0].get("data") or {}).items()}


def effective_default(env: dict[str, str], monkeypatch: pytest.MonkeyPatch) -> str:
    """``WSPSettings().wsp_default_provider`` under exactly ``env``'s WSP default (none given:
    the code's)."""
    from wsp.settings import WSPSettings

    monkeypatch.delenv("WSP_DEFAULT_PROVIDER", raising=False)
    if "WSP_DEFAULT_PROVIDER" in env:
        monkeypatch.setenv("WSP_DEFAULT_PROVIDER", env["WSP_DEFAULT_PROVIDER"])
    return str(WSPSettings().wsp_default_provider)


def switch_problems(record: Path) -> list[str]:
    """What the record lacks for the switch; empty when it shows all four measures on both
    bundles."""
    if not record.is_file():
        return [f"{record.name} does not exist"]
    found = BLOCK.search(record.read_text("utf-8"))
    if found is None:
        return [f"{record.name} has no ```switch block"]
    claim: Any = yaml.safe_load(found.group(1)) or {}
    problems = []
    for bundle in BUNDLES:
        measures = claim.get(bundle) if isinstance(claim, dict) else None
        if not isinstance(measures, dict):
            problems.append(f"{bundle}: missing")
            continue
        for measure in MEASURES:
            if measures.get(measure) != "equal_or_better":
                problems.append(f"{bundle}.{measure}: {measures.get(measure)!r}")
    return problems


@pytest.mark.skipif(
    ON_MIRROR, reason="the public mirror strips 0xcc/; checked in the private repository"
)
def test_the_engine_is_the_default_only_with_the_comparison_on_record(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    default = effective_default(configmap_env(), monkeypatch)
    if default != ENGINE:
        return  # weightspace (or another provider) is the default: nothing to show yet
    assert switch_problems(RECORD) == [], (
        f"the effective default is {ENGINE} but 0xcc/reviews/018_switch.md does not show it equal "
        "or better on all four D27 measures, on both bundles"
    )


def test_the_effective_default_is_a_provider_the_deployment_makes_available(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    env = configmap_env()
    available = [p.strip() for p in env.get("WSP_PROVIDERS_AVAILABLE", "").split(",") if p.strip()]
    assert effective_default(env, monkeypatch) in available, available


def test_the_configmap_overrides_the_code_default(monkeypatch: pytest.MonkeyPatch) -> None:
    """The reader above follows the pod's precedence: the ConfigMap's key wins over the code."""
    assert effective_default({"WSP_DEFAULT_PROVIDER": ENGINE}, monkeypatch) == ENGINE
    assert effective_default({}, monkeypatch) == "weightspace"


def test_a_record_with_one_measure_worse_is_not_a_switch(tmp_path: Path) -> None:
    good = {b: dict.fromkeys(MEASURES, "equal_or_better") for b in BUNDLES}
    record = tmp_path / "018_switch.md"
    record.write_text("# switch\n\n```switch\n" + yaml.safe_dump(good) + "```\n", "utf-8")
    assert switch_problems(record) == []
    worse = {**good, "phase2_bundle": {**good["phase2_bundle"], "heldout_verdict": "worse"}}
    record.write_text("```switch\n" + yaml.safe_dump(worse) + "```\n", "utf-8")
    assert switch_problems(record) == ["phase2_bundle.heldout_verdict: 'worse'"]
    record.write_text("no block here\n", "utf-8")
    assert switch_problems(record) == ["018_switch.md has no ```switch block"]
    assert switch_problems(tmp_path / "absent.md") == ["absent.md does not exist"]
