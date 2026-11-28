"""The WSP queue limit per caller is 10, in the ConfigMap and in the code (operator 2026-11-20).

Production, 2026-11-20: the operator approved four runs filed by one agent and the fourth failed at
execution, ``wsp_queue_limit_reached`` "This caller already has 3 queued WSP runs (the limit is
3)". The limit is now 10. Two numbers must agree, read from the real files:
``WSP_MAX_QUEUED_PER_CALLER`` in ``gitops/pilot/config.yaml`` (what the backend pod reads, through
``envFrom``) and
``WSPSettings``' default; and the ConfigMap's value must REACH the setting when the process
reads its environment, as the pod does.
"""

from __future__ import annotations

import pytest

from tests.support.wsp_manifests import documents
from wsp.settings import WSPSettings, get_wsp_settings

LIMIT = 10


def config_data() -> dict[str, str]:
    (config,) = [d for d in documents("config.yaml") if d.get("kind") == "ConfigMap"]
    return dict(config["data"])


def test_the_config_map_sets_ten() -> None:
    assert config_data()["WSP_MAX_QUEUED_PER_CALLER"] == str(LIMIT)


def test_the_setting_default_equals_the_config_map() -> None:
    default = WSPSettings.model_fields["wsp_max_queued_per_caller"].default
    assert default == LIMIT == int(config_data()["WSP_MAX_QUEUED_PER_CALLER"])


def test_the_config_map_value_reaches_the_setting(monkeypatch: pytest.MonkeyPatch) -> None:
    """The pod's environment is the ConfigMap's data (``envFrom``): load all of it, as the pod
    does, and read the setting the backend reads. The control: a different value moves it."""
    for key, value in config_data().items():
        monkeypatch.setenv(key, str(value))
    get_wsp_settings.cache_clear()
    try:
        assert get_wsp_settings().wsp_max_queued_per_caller == LIMIT
        monkeypatch.setenv("WSP_MAX_QUEUED_PER_CALLER", "7")
        get_wsp_settings.cache_clear()
        assert get_wsp_settings().wsp_max_queued_per_caller == 7
    finally:
        get_wsp_settings.cache_clear()
