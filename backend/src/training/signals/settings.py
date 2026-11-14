"""F2's settings  (FTID §9; ADR-010, ADR-018, ADR-020).

**Three of these are unset-means-refuse, and that is the point of the module.** A silent
default for the masking policy changes the objective; a silent default for DPO's reference
source changes what the loss is measured against; a silent default for the shortfall
behaviour changes how a batch that could not fill is treated. ADR-018: *a value that is
quietly substituted is a fabricated value*, and these three are substitutions nobody would
see in a loss curve.

**Why these live here and not in :mod:`core.config`.** ``core/`` is shared state with one
owner and this feature does not edit it - the same reason F3 keeps
``services/selectors/settings.py``. Both classes read the same ``.env`` and the same
environment, and ``Settings.model_config`` sets ``extra="ignore"``, so these names do not
disturb it. The coordinator may fold them in later; nothing here is read except through
:func:`get_signal_settings`.

**Every timeout is explicit and never ``None``** - a default-less timeout is how an
endpoint outage becomes a hung step rather than a named refusal.
"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

from core.errors import refuse

#: What ``sft``'s cross-entropy may be taken over. ``completion_only`` is the value that
#: matters: training over the prompt tokens of a chat or prompt/completion example teaches
#: the model to **generate prompts**, which is a different objective wearing the same name.
MaskingPolicy = Literal["completion_only", "whole_sequence"]

#: Where DPO's reference log-probabilities come from (BRD-02 §9's own open question, OQ-3).
#: Both are written behind one interface so the answer is a configuration value rather than
#: a rewrite - and the frozen-copy option's transfer cost must be **measured on this node**,
#: where both PCIe links are x4 (ADR-024), not estimated from an x16 figure.
ReferenceSource = Literal["frozen_copy", "precomputed"]

#: What to do when a bounded wait leaves a self-generation batch unfilled (OQ-1).
#: **Neither value admits a pending completion** - that part is fixed by FR-002.13 and is
#: not configurable.
ShortfallBehaviour = Literal["refuse_batch", "short_batch"]


class SignalSettings(BaseSettings):
    """Configuration for the four sources and the three objectives."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        # Pydantic appends `input_value=...` to `str(ValidationError)`, so ANY validator on
        # this class would reproduce the value it rejects. `core/config.py:50` sets the same
        # flag for the same reason. Added 2026-10-16 as a PER-CLASS property of the error
        # rendering, not of the fields that happen to live here today: this class holds no
        # credential now, and the first validator added to it would leak whatever it refuses.
        hide_input_in_errors=True,
        case_sensitive=False,
        env_prefix="signal_",
    )

    # ------------------------------------------------------------------ generation
    #: Bounds provider load, matching F3's convention for judge calls. One convention.
    generation_concurrency: int = Field(default=4, ge=1, le=64)
    #: Explicit, and **never ``None``**. Applied per batch of candidates.
    generation_timeout_s: float = Field(default=120.0, gt=0.0)

    # ------------------------------------------------------------------ the three refusals
    #: ``None`` means **refuse** (E-5). See :func:`require_masking_policy`.
    sft_masking_policy: MaskingPolicy | None = None
    #: ``None`` means **refuse** (E-13). See :func:`require_reference_source`.
    dpo_reference_source: ReferenceSource | None = None
    #: ``None`` means **refuse** (OQ-1). See :func:`require_shortfall_behaviour`.
    pending_shortfall: ShortfallBehaviour | None = None

    # ------------------------------------------------------------------ DPO
    #: The strength of the preference term. Explicit because it changes what the loss
    #: means, and a loss figure is comparable with another only when both say their beta.
    dpo_beta: float = Field(default=0.1, gt=0.0)

    # ------------------------------------------------------------------ pending
    #: Bounds the **wait** for a human decision and produces only a **count**. There is no
    #: path from this deadline to an admission (FR-002.13, AC-002.4).
    pending_deadline_s: float = Field(default=0.0, ge=0.0)

    # ------------------------------------------------------------------ reward
    #: Completions per prompt. **Below two is refused** (E-16).
    reward_group_size: int = Field(default=4, ge=2)
    reward_groups_per_batch: int = Field(default=4, ge=1)
    #: Below this variance a group counts as having no spread - the GRPO zero-advantage
    #: case. Explicit because "zero" in floating point is a choice, and it matches the
    #: epsilon F3 uses for the same signal so the two cannot disagree about one group.
    reward_variance_epsilon: float = Field(default=1e-9, ge=0.0)

    # ------------------------------------------------------------------ bounds
    #: Checked **before** a generation call, refusing with the bound named rather than
    #: truncating: truncation changes the training example and the verdict **without
    #: changing any hash**, so the record would describe a batch that was never trained on.
    max_prompt_chars: int = Field(default=65_536, ge=1)
    max_completion_chars: int = Field(default=65_536, ge=1)
    #: The batch listing's page cap, named on refusal.
    batch_page_size_max: int = Field(default=200, ge=1)

    @field_validator(
        "sft_masking_policy", "dpo_reference_source", "pending_shortfall", mode="before"
    )
    @classmethod
    def _empty_string_is_unset(cls, value: Any) -> Any:
        """``SIGNAL_SFT_MASKING_POLICY=`` in a ``.env`` file means **unset**, not ``""``.

        Without this, copying ``.env.example`` - whose three deliberately-blank lines are
        the whole point of that section - would fail validation at **startup**, because an
        empty string is not one of the ``Literal`` values. The failure would be loud, which
        is something, but it would be loud in the wrong place: the file is correct and the
        operator would be told their configuration is malformed. The foundation's
        ``Settings`` carries the same validator for the sibling URLs, for the same reason.
        """
        if isinstance(value, str) and not value.strip():
            return None
        return value


_settings: SignalSettings | None = None


def get_signal_settings() -> SignalSettings:
    """The process-wide signal settings, built once.

    Not an ``lru_cache``: the tests need :func:`reset_signal_settings`, and a cache that
    cannot be cleared makes an env-var fixture silently ineffective - which is how a
    standalone test ends up asserting against a configuration it did not set.
    """
    global _settings
    if _settings is None:
        _settings = SignalSettings()
    return _settings


def reset_signal_settings() -> None:
    """Drop the cached settings. For tests that change the environment."""
    global _settings
    _settings = None


def require_masking_policy(settings: SignalSettings) -> str:
    """The masking policy, or **refuse naming the setting and both values** (E-5).

    R-02.4 says *"next-token cross-entropy (supervised fine-tuning)"* and **says nothing
    about masking**, while the choice changes the objective materially: a prompt/completion
    or chat example trained over the whole sequence teaches the model to produce prompts as
    well as answers. So the policy is explicit configuration, recorded on every batch, and
    absent means refused rather than guessed.
    """
    if settings.sft_masking_policy is None:
        raise refuse(
            "sft_masking_policy_not_configured",
            "SIGNAL_SFT_MASKING_POLICY is unset, so there is no answer to which tokens the "
            "cross-entropy is taken over. Set it to 'completion_only' (the loss ignores "
            "prompt tokens) or 'whole_sequence' (every token counts). Refusing rather than "
            "defaulting: whole-sequence training on prompt/completion data teaches the "
            "model to generate prompts, and no loss curve says so.",
            setting="SIGNAL_SFT_MASKING_POLICY",
            options=["completion_only", "whole_sequence"],
        )
    return settings.sft_masking_policy


def require_reference_source(settings: SignalSettings) -> str:
    """DPO's reference-logprob source, or **refuse naming both options** (E-13).

    BRD-02 §9 leaves *how* open - *"keep a frozen copy in CPU RAM ... or cache reference
    logprobs per batch before the run?"* - and leaving it **open** is not leaving it
    **absent**. A direct preference objective computed against the policy itself is not a
    preference objective at all; it would still produce a falling loss.
    """
    if settings.dpo_reference_source is None:
        raise refuse(
            "dpo_reference_source_not_configured",
            "SIGNAL_DPO_REFERENCE_SOURCE is unset, so a direct preference objective has no "
            "reference to measure against. Set it to 'frozen_copy' (a frozen parameter set "
            "in host memory - measure the transfer on this node, both links are x4) or "
            "'precomputed' (reference log-probabilities cached per batch before the run, "
            "invalid the moment the reference changes).",
            setting="SIGNAL_DPO_REFERENCE_SOURCE",
            options=["frozen_copy", "precomputed"],
        )
    return settings.dpo_reference_source


def require_shortfall_behaviour(settings: SignalSettings) -> str:
    """What an unfilled batch does, or **refuse naming both options** (OQ-1).

    **Neither option admits a pending completion.** The spike was about which of the two
    permitted behaviours is the default, never about whether silence can become consent:
    FR-002.13 fixes that, and this refusal exists so the remaining choice is made by an
    operator rather than by whoever wrote the first default.
    """
    if settings.pending_shortfall is None:
        raise refuse(
            "pending_shortfall_not_configured",
            "SIGNAL_PENDING_SHORTFALL is unset, so there is no answer to what happens when "
            "a batch cannot fill - candidates pending review, or a prompt set exhausted. "
            "Set it to 'refuse_batch' (stop, naming the shortfall) or 'short_batch' (train "
            "on what was admitted and record the shortfall, which changes the effective "
            "batch size step to step). Neither admits a pending completion.",
            setting="SIGNAL_PENDING_SHORTFALL",
            options=["refuse_batch", "short_batch"],
        )
    return settings.pending_shortfall
