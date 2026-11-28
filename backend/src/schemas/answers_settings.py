"""The answers of the settings routes (feature 021, P3): ``api/v1/endpoints/settings.py``.

Documentation only (MF9): the routes keep ``response_model=None`` and serve what they build.
"""

from pydantic import BaseModel

from .answers_common import JsonValue, free_form, may_be_absent


class SettingEffect(BaseModel):
    """``core/settings_catalog.py:effect_payload``: when a change to the setting takes effect."""

    when: str
    text: str
    pod: str | None = may_be_absent(
        description="The pod whose restart applies it; sent only when ``when`` is restart."
    )


class SettingGroup(BaseModel):
    """``core/settings_catalog.py:groups_payload``: one group, in screen order."""

    id: str
    label: str


class SettingChange(BaseModel):
    """``api/v1/endpoints/settings.py:change_payload``: one history row."""

    id: int
    key: str
    action: str
    previous: JsonValue = free_form(
        "the value in force before the change, as GET /settings serves it; its type is the "
        "setting's own"
    )
    previous_source: str
    value: JsonValue = free_form(
        "the value in force after the change, as GET /settings serves it; its type is the "
        "setting's own"
    )
    source: str
    actor: str
    approval_id: str | None
    reason: str | None
    changed_at: str


class SettingEntry(BaseModel):
    """``api/v1/endpoints/settings.py:_payload``: one setting in the catalog."""

    value: JsonValue = free_form(
        "the setting's effective value, redacted where secret; its type is the setting's own "
        "(string, number, boolean, list or null)"
    )
    source: str
    runtime_writable: bool
    label: str
    description: str
    effect: SettingEffect
    group: str
    env_value: JsonValue = free_form(
        "the setting's environment value, redacted where secret; its type is the setting's own"
    )
    updated_at: str | None
    updated_by: str | None
    last_change: SettingChange | None


class SettingReadSettingsAnswer(BaseModel):
    """``api/v1/endpoints/settings.py:read_settings`` (``_payload``)."""

    settings: dict[str, SettingEntry]
    count: int
    runtime_keys: list[str]
    overrides: list[str]
    groups: list[SettingGroup]


class SettingReadHistoryAnswer(BaseModel):
    """``api/v1/endpoints/settings.py:read_history``."""

    changes: list[SettingChange]
    count: int


class SettingWriteSettingAnswer(BaseModel):
    """``api/v1/endpoints/settings.py:write_setting``."""

    key: str
    value: JsonValue = free_form(
        "the value now in force, redacted where secret; its type is the setting's own"
    )
    previous: JsonValue = free_form(
        "the value in force before the write, redacted where secret; its type is the setting's own"
    )
    source: str


class SettingResetSettingAnswer(BaseModel):
    """``api/v1/endpoints/settings.py:reset_setting``."""

    key: str
    value: JsonValue = free_form(
        "the environment value that applies again, redacted where secret; its type is the "
        "setting's own"
    )
    source: str
