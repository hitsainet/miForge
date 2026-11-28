"""F2's three must-be-explicit settings  (FTID §9; E-5, E-13, OQ-1).

**Three settings have no default and a run is refused without them**, because each decides
something no requirement states and no loss curve would reveal: what the cross-entropy is
masked over, what a direct preference objective is measured against, and what an unfilled
batch does. ADR-018: *a value that is quietly substituted is a fabricated value* - and
these three would be substituted for the meaning of the objective itself.

The empty-string test is the one that is easy to leave out: ``.env.example`` carries all
three **present and blank**, so a settings class that rejected ``""`` would make copying
that file a startup failure - loud, but loud about the wrong thing.
"""

from __future__ import annotations

import pytest

from core.errors import Refusal
from training.signals.settings import (
    SignalSettings,
    get_signal_settings,
    require_masking_policy,
    require_reference_source,
    require_shortfall_behaviour,
    reset_signal_settings,
)


@pytest.fixture(autouse=True)
def _clean() -> None:
    reset_signal_settings()
    yield
    reset_signal_settings()


class TestTheThreeRefusals:
    @pytest.mark.parametrize(
        "accessor,code,options",
        [
            (
                require_masking_policy,
                "sft_masking_policy_not_configured",
                ["completion_only", "whole_sequence"],
            ),
            (
                require_reference_source,
                "dpo_reference_source_not_configured",
                ["frozen_copy", "precomputed"],
            ),
            (
                require_shortfall_behaviour,
                "pending_shortfall_not_configured",
                ["refuse_batch", "short_batch"],
            ),
        ],
    )
    def test_an_unset_value_refuses_naming_the_setting_and_its_options(
        self, accessor, code: str, options: list[str]
    ) -> None:
        with pytest.raises(Refusal) as caught:
            accessor(SignalSettings())
        assert caught.value.code == code
        assert caught.value.details["options"] == options
        assert caught.value.details["setting"].startswith("SIGNAL_")

    @pytest.mark.parametrize(
        "variable,value,accessor",
        [
            ("SIGNAL_SFT_MASKING_POLICY", "whole_sequence", require_masking_policy),
            ("SIGNAL_DPO_REFERENCE_SOURCE", "frozen_copy", require_reference_source),
            ("SIGNAL_PENDING_SHORTFALL", "refuse_batch", require_shortfall_behaviour),
        ],
    )
    def test_a_configured_value_is_returned(
        self, variable: str, value: str, accessor, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv(variable, value)
        assert accessor(SignalSettings()) == value


class TestTheEnvFileCanBeCopied:
    @pytest.mark.parametrize(
        "variable",
        ["SIGNAL_SFT_MASKING_POLICY", "SIGNAL_DPO_REFERENCE_SOURCE", "SIGNAL_PENDING_SHORTFALL"],
    )
    def test_a_blank_value_reads_as_unset_rather_than_failing_validation(
        self, variable: str, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """``.env.example`` carries these three blank **on purpose**; copying it must leave
        the refusal to the run rather than breaking startup."""
        monkeypatch.setenv(variable, "")
        settings = SignalSettings()
        assert getattr(settings, variable.removeprefix("SIGNAL_").lower()) is None

    def test_every_setting_in_the_env_example_is_one_this_class_reads(self) -> None:
        """A documented knob that turns nothing is worse than an undocumented one, because
        someone will set it and believe it took effect. Read from the file, not retyped."""
        from pathlib import Path

        example = Path(__file__).resolve().parents[3] / ".env.example"
        declared = {f"SIGNAL_{name.upper()}" for name in SignalSettings.model_fields}
        in_file = {
            line.split("=", 1)[0].strip()
            for line in example.read_text("utf-8").splitlines()
            if line.startswith("SIGNAL_")
        }
        assert in_file <= declared, in_file - declared
        # And the three that must be explicit are all present, so an operator sees them.
        assert {
            "SIGNAL_SFT_MASKING_POLICY",
            "SIGNAL_DPO_REFERENCE_SOURCE",
            "SIGNAL_PENDING_SHORTFALL",
        } <= in_file


class TestTheCache:
    def test_the_settings_are_built_once_and_can_be_reset(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Not an ``lru_cache``: a cache that cannot be cleared makes an env-var fixture
        silently ineffective, which is how a standalone test ends up asserting against a
        configuration it did not set."""
        monkeypatch.setenv("SIGNAL_REWARD_GROUP_SIZE", "8")
        assert get_signal_settings().reward_group_size == 8
        monkeypatch.setenv("SIGNAL_REWARD_GROUP_SIZE", "16")
        assert get_signal_settings().reward_group_size == 8, "cached"
        reset_signal_settings()
        assert get_signal_settings().reward_group_size == 16

    def test_a_group_size_below_two_is_refused_at_the_type(self) -> None:
        """E-16 at the earliest possible layer: the configuration cannot even hold it."""
        from pydantic import ValidationError

        with pytest.raises(ValidationError):
            SignalSettings(reward_group_size=1)
