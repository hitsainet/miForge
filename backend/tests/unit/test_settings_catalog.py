"""Every setting says what it does and when a change takes effect  (013 G4; ENH-003 section 3.4).

The catalog is checked against the LIVE ``Settings`` model and the LIVE allow-list, so a new
field cannot ship undescribed and a key cannot move onto the allow-list while still telling the
operator it needs a restart.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import pytest

from core import settings_catalog
from core.config import RUNTIME_KEYS, Settings

RULES = Path(__file__).resolve().parents[3] / "manual" / "publish-rules.json"


def test_the_catalog_covers_exactly_the_settings() -> None:
    fields = set(Settings.model_fields)
    assert len(fields) > 30, "the scan read almost nothing"
    assert set(settings_catalog.CATALOG) == fields


def test_every_runtime_key_takes_effect_without_a_restart_and_no_other_key_does() -> None:
    assert RUNTIME_KEYS, "the allow-list is empty - the check would assert nothing"
    for key, info in settings_catalog.CATALOG.items():
        if key in RUNTIME_KEYS:
            assert info.effect in {"now", "next_sweep", "next_attached_run"}, key
        else:
            assert info.effect == "restart", key


#: Each runtime key's stated timing, and where that timing is exercised.
PINNED: list[tuple[str, str]] = [
    # Each timing is exercised by a test that observes it:
    # now - test_settings_history (health and log level at the write) and the worker's
    #       task_prerun refresh (test_settings_api's junction test);
    ("log_level", "now"),
    ("millm_base_url", "now"),
    ("midataworks_base_url", "now"),
    # next_sweep - policy_from_settings() read inside a real task (test_settings_api);
    ("checkpoint_lifecycle_enabled", "next_sweep"),
    ("checkpoint_lifecycle_dry_run", "next_sweep"),
    ("checkpoint_lifecycle_keep_last_n", "next_sweep"),
    ("checkpoint_lifecycle_keep_every_m", "next_sweep"),
    # next_attached_run - hotupdate_settings.load() at attach (test_settings_api).
    ("hotupdate_verification_tolerance", "next_attached_run"),
    ("hotupdate_verification_tolerance_float32", "next_attached_run"),
    ("hotupdate_request_timeout_seconds", "next_attached_run"),
    ("hotupdate_checkpoint_timeout_seconds", "next_attached_run"),
    ("hotupdate_commit_end_timeout_seconds", "next_attached_run"),
]


@pytest.mark.parametrize(("key", "effect"), PINNED)
def test_each_runtime_keys_stated_timing_is_pinned(key: str, effect: str) -> None:
    assert settings_catalog.info_for(key).effect == effect


def test_the_pinned_list_is_the_whole_allow_list() -> None:
    assert {key for key, _ in PINNED} == set(RUNTIME_KEYS)


def test_a_restart_names_the_pod_and_says_it_stops_jobs() -> None:
    payload = settings_catalog.effect_payload("database_url")
    assert payload["pod"] == "miforge-backend"
    assert "miforge-backend" in payload["text"]
    assert "stops running jobs" in payload["text"]
    assert "pod" not in settings_catalog.effect_payload("log_level")


def test_every_entry_has_a_label_and_a_sentence() -> None:
    for key, info in settings_catalog.CATALOG.items():
        assert info.label.strip() and "\n" not in info.label, key
        assert info.description.strip().endswith("."), key
        assert info.label != key, f"{key}: the label repeats the raw name"


def test_no_entry_carries_an_internal_identifier() -> None:
    """The manual's publish rules (one list for the manual and the app)."""
    rules = [
        re.compile(rule["pattern"], re.I if "i" in rule["flags"] else 0)
        for rule in json.loads(RULES.read_text(encoding="utf-8"))["forbidden"]
        if rule["category"] == "internal"
    ]
    assert len(rules) >= 3, "the scan read no rules"
    assert any(rule.search("see FR-012.7 (D5)") for rule in rules), "the rules catch nothing"
    for key, info in settings_catalog.CATALOG.items():
        text = " ".join((info.label, info.description, settings_catalog.EFFECT_TEXT[info.effect]))
        for rule in rules:
            assert not rule.search(text), (key, rule.pattern)


def test_every_setting_has_exactly_one_known_group_and_no_group_is_empty() -> None:
    """The Settings screen groups by this (2026-10-28); a field with no group would fall into
    the screen's catch-all, which is how a new setting would go unnoticed."""
    fields = set(Settings.model_fields)
    known = [group for group, _ in settings_catalog.GROUPS]
    assert len(known) == len(set(known)) >= 5
    assert set(settings_catalog.GROUP_OF) == fields
    assert set(settings_catalog.GROUP_OF.values()) == set(known)


def test_the_groups_are_served_in_their_fixed_order() -> None:
    served = settings_catalog.groups_payload()
    assert [group["id"] for group in served] == [
        "general",
        "storage",
        "jobs",
        "checkpoint_pruning",
        "connections",
        "hot_update",
    ]
    assert all(group["label"].strip() for group in served)


def test_settings_that_belong_together_share_a_group() -> None:
    """Spot checks by meaning, so a group map that agrees with itself by construction (every
    key in one group) cannot pass."""
    group = settings_catalog.GROUP_OF
    assert {group[key] for key in Settings.model_fields if key.startswith("hotupdate_")} == {
        "hot_update"
    }
    assert {
        group[key] for key in Settings.model_fields if key.startswith("checkpoint_lifecycle_")
    } == {"checkpoint_pruning"}
    assert group["millm_base_url"] == group["midataworks_base_url"] == "connections"
    assert group["database_url"] == group["checkpoint_root"] == "storage"
    assert group["log_level"] == "general"


def test_the_import_guard_refuses_an_ungrouped_setting(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delitem(settings_catalog.GROUP_OF, "log_level")
    with pytest.raises(RuntimeError, match="ungrouped \\['log_level'\\]"):
        settings_catalog._check()


def test_no_group_label_carries_an_internal_identifier() -> None:
    rules = [
        re.compile(rule["pattern"], re.I if "i" in rule["flags"] else 0)
        for rule in json.loads(RULES.read_text(encoding="utf-8"))["forbidden"]
        if rule["category"] == "internal"
    ]
    assert rules, "the scan read no rules"
    for _, label in settings_catalog.GROUPS:
        for rule in rules:
            assert not rule.search(label), (label, rule.pattern)


# ------------------------------------------------- feature 015 P4: the copy names miServer

#: The stored keys are frozen (suite R1, register entry DR-39); only what a person reads moves.
MISERVER_LABELS = {
    "millm_base_url": "miServer address",
    "millm_trainer_token": "miServer trainer token",
    "millm_audit_label": "Name in miServer's log",
}


def test_labels_name_miserver() -> None:
    """FR-015.21: a history row keyed ``millm_base_url`` reads "miServer address"."""
    for key, label in MISERVER_LABELS.items():
        assert key in settings_catalog.CATALOG, f"{key}: the stored key must not be renamed"
        assert settings_catalog.CATALOG[key].label == label
    old = re.compile(r"(?i)\bmi(llm|studio)\b")
    named = 0
    for key, info in settings_catalog.CATALOG.items():
        text = f"{info.label} {info.description}"
        assert not old.search(text), f"{key}: {text!r} names a product by its old name"
        named += "miServer" in text
    for group in settings_catalog.groups_payload():
        assert not old.search(" ".join(str(v) for v in group.values())), group
    assert named >= 10, f"only {named} entries name miServer: the catalogue was not read"


def test_approval_summary_names_miserver() -> None:
    """FR-015.21: the attach request an operator approves names the server by its name."""
    from services import approval_gates

    assert approval_gates.model_label({"mode": "attach"}) == "the model miServer is serving"
