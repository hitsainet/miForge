"""Review round 2 of 015 P0: the audit label, now a SECRET_FIELD, is not printed by ``repr`` either.

``millm_trainer_token`` carries ``Field(repr=False)``; ``millm_audit_label`` did not, so
``repr(get_settings())`` printed the value the settings API now masks. The hot-update settings
dataclass printed both the token and the label (``test_hotupdate_secret_hygiene`` names the
risk - "repr is an output" - but asserts on ``unavailable_message`` instead).
"""

from __future__ import annotations

import pytest

from core.config import get_settings, reset_settings

# Hyphenated lowercase words: the repository secret scanner (test_no_secrets_in_artifacts)
# treats a token-named value with digits as a credential, and this one is a sentinel.
TOKEN = "trainer-token-sentinel-quokka"
LABEL = "audit-label-sentinel-81c2"


@pytest.fixture
def configured(monkeypatch: pytest.MonkeyPatch) -> None:
    for name in ("MILLM_TRAINER_TOKEN", "MISERVER_TRAINER_TOKEN", "MILLM_AUDIT_LABEL"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("MISERVER_TRAINER_TOKEN", TOKEN)
    monkeypatch.setenv("MISERVER_AUDIT_LABEL", LABEL)
    reset_settings()
    yield
    reset_settings()


def test_the_settings_repr_carries_no_secret_field(configured: None) -> None:
    settings = get_settings()
    assert settings.millm_audit_label == LABEL, "the fixture configured it"
    text = repr(settings) + str(settings)
    assert TOKEN not in text and LABEL not in text


def test_the_hot_update_settings_repr_carries_neither(configured: None) -> None:
    from training.hotupdate import settings as settings_module

    cfg = settings_module.load()
    assert cfg.trainer_token == TOKEN and cfg.audit_label == LABEL, "the fixture configured it"
    text = repr(cfg) + str(cfg)
    assert TOKEN not in text and LABEL not in text
