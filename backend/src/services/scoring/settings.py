"""F4's settings  (FTID section 9; ADR-010, ADR-018, ADR-024).

**Every value is explicit and nothing that changes a number is defaulted silently.** The
cadences, the decode parameters and the batch bound all change what a score means, so each
has a stated default here that a benchmark report can quote (ADR-024) rather than a literal
buried at a call site.

**Why these live here and not in :mod:`core.config`.** ``core/`` is shared state with one
owner and this feature does not edit it - the same reason F3 put ``SelectorSettings`` beside
its consumer. Both classes read the same ``.env`` and the same process environment, and
``Settings.model_config`` sets ``extra="ignore"``, so these names do not disturb it.

**``retention_every`` defaults to 1 whenever a floor is set** (FR-004.26). That is not a
preference: a breach can only be observed at a retention evaluation, so with a cadence of N
up to **N-1 steps of training occur between a breach happening and being seen**. The default
is therefore the safe one and a coarser cadence reports its exposure window as a number
(:func:`exposure_window_steps`) rather than leaving it implied. BRD-02 acceptance criterion 3
asks for *"every step"* and ADR-024 advises a cadence; DEF-8 records the tension and this is
the reading that makes the safety property the default.
"""

from __future__ import annotations

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class ScoringSettings(BaseSettings):
    """Cadence, decode and bound configuration. Instantiate via the accessor below."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        case_sensitive=False,
        # PYDANTIC ECHOES THE REJECTED VALUE INTO ITS ERROR MESSAGE BY DEFAULT.
        # `str(ValidationError)` appends `input_value=...`, so the credential refusal below
        # for `activation_source_base_url` **reproduced the very credential it was
        # refusing** - and the refusal is reachable from a request path, not only at
        # startup: the same changeset added `activation_source` to
        # `core.config.configured_integrations()`, which `api/health.py` calls, so
        # `GET /api/health` wrote the credential into the application log in plain text.
        #
        # `core.config.Settings` sets this same flag for this same reason, and its comment
        # is the long form: **a validation error is an output** - the one path nobody
        # thinks of as one. This class reads the same `.env` and declares the third sibling
        # base URL, so it needed the flag too and did not have it. *Found by executing the
        # refusal and reading the error string, not by reading the field beside the one
        # that already had the flag.*
        #
        # `repr=False` on a field does NOT cover this: it governs `repr()`, not Pydantic's
        # error rendering.
        hide_input_in_errors=True,
    )

    # ------------------------------------------------------------------ the provider
    #: The provider a run scores through unless its definition names another. ``in_process``
    #: is **the default that makes acceptance criterion 20 hold**: it needs no endpoint, no
    #: credential and no sibling.
    scoring_provider: str = "in_process"

    # ------------------------------------------------------------------ decode
    #: **Greedy by default** (FR-004.5). Criterion 2 asks another server to reproduce a
    #: recorded number, and it is written against exactly this case: *"with deterministic
    #: decoding and the same provider, exactly."* A sampled pass is recorded as **not
    #: reproducible, with the reason**, rather than being quietly compared anyway.
    scoring_decode_temperature: float = Field(default=0.0, ge=0.0, le=2.0)
    scoring_decode_top_p: float = Field(default=1.0, gt=0.0, le=1.0)
    scoring_decode_max_tokens: int = Field(default=256, gt=0)
    #: Comma-separated stop sequences. Empty means none, which is a stated default.
    scoring_decode_stop: str = ""

    # ------------------------------------------------------------------ cadence
    #: Steps between test evaluations. ``1`` means every step.
    test_every: int = Field(default=1, ge=1)
    #: Steps between retention evaluations. See the module docstring: **1 when a floor is
    #: set** is the safe default, and FR-004.26 reports the window for anything coarser.
    retention_every: int = Field(default=1, ge=1)

    # ------------------------------------------------------------------ bounds
    #: Prompts sent to a provider in one call. Beside a live optimizer state, so bounded
    #: explicitly rather than "as many as fit" - the free figure moves while a run steps.
    scoring_batch_bound: int = Field(default=8, ge=1)
    #: The most prompts one evaluation will score. An unbounded pass over a judge is a
    #: request that outlives its client, and over the in-process provider it is a
    #: generation pass that dominates every step (ADR-024).
    #:
    #: **This is not the effective ceiling** - see :func:`effective_max_prompts`. F3's
    #: batch evaluation has its own synchronous bound, and the lower of the two governs.
    scoring_max_prompts: int = Field(default=2000, ge=1)

    # ------------------------------------------------------------------ R-02.19, optional
    #: **UNSET BY DEFAULT**, and unset means absent-and-normal (ADR-010, R-02.50). R-02.19's
    #: activation statistics are an opt-in sibling integration reached through a single
    #: ``clients/`` call site; nothing on the core scoring path imports it, and the
    #: standalone harness runs with the module absent.
    activation_source_base_url: str | None = None

    @field_validator("activation_source_base_url", mode="before")
    @classmethod
    def _empty_string_is_unset(cls, value: object) -> object:
        """``ACTIVATION_SOURCE_BASE_URL=`` means **unset**, not the empty string.

        **This field had neither of the two validators `core/config.py` gives a sibling base
        URL, and it is a sibling base URL** — it just lives in a different settings object,
        which is exactly why it was missed. Without this, `.env.example`'s deliberately-blank
        line configured the source as the empty URL: `GET /api/health` reported
        `activation_source: configured`, and the failure surfaced later as a connection
        error to `""` instead of a named refusal at the boundary. *Found by running the
        standalone case, which went red the moment health learned to name this integration —
        not by reading the field.*
        """
        if isinstance(value, str) and not value.strip():
            return None
        return value

    @field_validator("activation_source_base_url")
    @classmethod
    def _a_base_url_carries_no_credentials(cls, value: str | None) -> str | None:
        """**Refuse** ``user:password@`` in this URL rather than mask it.

        The third sibling URL, and the only one that had no such guard. A base URL is a
        *location*; this architecture keeps credentials in three separate places and none of
        them is a URL. AGENT.md records what masking costs: a credential in
        `millm_base_url` *"defeated every field-by-field mask"*, reaching any log line or
        traceback, because masking hides the symptom and leaves the credential live on the
        wire — sent from a field no reviewer reads as sensitive.

        **The rejected value is not echoed — and that took two things, not one.** This
        message carries the **host only**, but a message that names no secret is not
        sufficient: Pydantic appends the rejected value as `input_value=...` to
        `str(ValidationError)` regardless of what the validator said, which is how the
        refusal added for `millm_trainer_token` reproduced the very credential it was
        refusing. `hide_input_in_errors=True` on `model_config` above is the half that
        closes it, and it is pinned by the secret-absent assertion in
        `test_activations_client.py` rather than by this sentence.

        *This docstring claimed "`hide_input_in_errors` is set on `core.config`'s Settings,
        not on this class … so this raises with the host only" — which read as a reason the
        flag was unnecessary here, while the flag's absence was the leak. The host-only
        message was true; the conclusion drawn from it was false, and the error string
        contained the credential.*
        """
        if value is None:
            return None
        from urllib.parse import urlsplit

        parts = urlsplit(value)
        if parts.username or parts.password:
            raise ValueError(
                f"ACTIVATION_SOURCE_BASE_URL carries credentials in the URL (host "
                f"{parts.hostname!r}). A base URL is a location; send identity in a header. "
                f"Refused rather than masked, because a mask hides the symptom and leaves "
                f"the credential live on the wire."
            )
        return value

    def decode_stop(self) -> tuple[str, ...]:
        """The stop sequences as a tuple. Empty is a stated default, not a missing value."""
        return tuple(
            part for part in (p.strip() for p in self.scoring_decode_stop.split(",")) if part
        )


_settings: ScoringSettings | None = None


def get_scoring_settings() -> ScoringSettings:
    """The process-wide scoring settings, built once.

    Not an ``lru_cache``: the tests need :func:`reset_scoring_settings`, and a cache that
    cannot be cleared makes an env-var fixture silently ineffective - which is how a
    standalone test ends up asserting against a configuration it did not set.
    """
    global _settings
    if _settings is None:
        _settings = ScoringSettings()
    return _settings


def reset_scoring_settings() -> None:
    """Drop the cached settings. For tests that change the environment."""
    global _settings
    _settings = None


def exposure_window_steps(retention_every: int) -> int:
    """How many steps of training can pass between a breach occurring and being seen.

    ``retention_every - 1`` (FR-004.26, AC-004.15). **Reported as a number wherever a floor
    is shown, never implied**: with a cadence of 10, nine steps of training can run on a
    model that has already fallen below its floor, and an operator who was not told that
    will read the floor as a guarantee it is not.
    """
    if retention_every < 1:
        raise ValueError("a cadence is at least one step")
    return retention_every - 1


def effective_max_prompts(settings: ScoringSettings | None = None) -> int:
    """The ceiling a set is actually resolved against: **the lower of two bounds.**

    Found by benchmarking, not by reading. F3's ``evaluate_max_completions`` defaults to
    **200** and its service refuses above it; this feature's own ceiling defaulted to 2000.
    A 201-prompt test set therefore resolved fine, generated 201 completions, and then failed
    at F3 with ``evaluation_too_large`` - whose message is about *"an unbounded synchronous
    evaluation over a judge … a request that outlives its client."*

    Three things were wrong with that, in rising order of cost:

    1. the noun. A cadence-driven pass in the ``train`` worker has no client to outlive, so
       the message sends an operator looking for a request timeout;
    2. the timing. It failed **after** generating, so the expensive half was paid and
       discarded - and R-02.48's rule is that a run which cannot proceed says so *before it
       starts*;
    3. the drift. **Two features held two numbers for one bound.** Reading F3's rather than
       restating it is the same discipline as reading the ScoreRecord key set out of its
       schema: the lower bound governs, and neither side can move without the other seeing it.

    **The deeper question is reported, not decided here:** whether a scoring pass should
    *chunk* through F3's batch service the way it already chunks generation through the
    provider. That would lift the ceiling altogether, but F3 returns **one** health record
    and **one** pair of configuration hashes per call, so chunking means deciding what a
    per-chunk health signal means - which is F3's to answer, not F4's to assume.
    """
    from services.selectors.settings import get_selector_settings

    settings = settings or get_scoring_settings()
    return min(settings.scoring_max_prompts, get_selector_settings().evaluate_max_completions)
