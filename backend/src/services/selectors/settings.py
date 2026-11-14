"""F3's settings  (FTID §9; ADR-010, ADR-018, ADR-020).

**Every endpoint URL is unset by default, and unset means absent-and-normal** (ADR-010).
That is the mechanism by which acceptance criterion 20 - a full run with no sibling present
- is true rather than asserted: with nothing configured, four of the five selector types
work and the fifth names the service it is missing.

**Why these live here and not in :mod:`core.config`.** ``core/`` is shared state with one
owner and this feature does not edit it. The two classes read the same ``.env`` and the
same process environment, and ``Settings.model_config`` sets ``extra="ignore"``, so the
names below do not disturb it. The coordinator may fold them into ``core/config.py``
later; nothing in this feature reads them except through :func:`get_selector_settings`.

**One setting here is a stand-in for a missing contract field and is marked as such:**
``judge_prompt_template``. See its comment.

**A credential in ``judge_provider_base_url`` is refused, not masked**, and the refusal does
not reproduce it - see :meth:`SelectorSettings._a_base_url_carries_no_credentials` and
``hide_input_in_errors`` in ``model_config``. Both protections are ``core/config.py``'s and
were simply absent here, which is why the enumeration in
``tests/unit/test_all_base_urls_refuse_credentials.py`` now discovers the field rather than
relying on anyone remembering this class exists.
"""

from __future__ import annotations

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class SelectorSettings(BaseSettings):
    """Configuration for the five selector types. Instantiate via the accessor below."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        case_sensitive=False,
        # PYDANTIC ECHOES THE REJECTED VALUE INTO ITS ERROR MESSAGE BY DEFAULT, and this
        # model holds a credential. `str(ValidationError)` appends `input_value=...`, so a
        # validation failure on `judge_provider_api_key` printed the key, and the refusal
        # added below for a credential-bearing base URL would have reproduced the very
        # credential it was refusing. A startup validation error is logged, so that is a
        # leak through the error path - the one path nobody thinks of as an output.
        #
        # `repr=False` on the field does NOT cover this: it governs `repr()`, not Pydantic's
        # error rendering. Copied deliberately from `core/config.py:50`, where this exact
        # pair of leaks was found and closed; this class was written before that and had
        # neither half.
        hide_input_in_errors=True,
    )

    # ------------------------------------------------------------------ the judge
    # UNSET BY DEFAULT, and deliberately so. A default base URL would turn an absent
    # endpoint into a connection error at an arbitrary later moment instead of a named
    # refusal at the boundary - and R-02.11 makes the consequence worse here than
    # elsewhere, because the only thing available to fall back to is the model being
    # trained, which is a model grading itself.
    judge_provider_base_url: str | None = None
    judge_provider_model: str | None = None
    judge_provider_api_key: str | None = Field(default=None, repr=False)

    #: Separate connect and read timeouts, both explicit and **never ``None``**. A missing
    #: timeout is how a judge outage becomes a hung step rather than a named refusal.
    judge_connect_timeout_s: float = 5.0
    judge_read_timeout_s: float = 30.0

    #: The §9 performance lever. Bounded concurrency, not an unbounded gather: unbounded
    #: concurrency against a llama.cpp server on a CPU node is how a batch becomes a
    #: timeout storm.
    judge_concurrency: int = Field(default=4, ge=1, le=64)

    #: **A STAND-IN FOR A FIELD THE CONTRACT DOES NOT HAVE - reported, not invented.**
    #: FR-003.5 says *"the prompt template and the accept threshold come from the
    #: configuration"*, and ``selector-config-v1.json``'s ``JudgeModelSelector`` is
    #: ``{kind, name, config_sha256, provider, threshold}`` with ``additionalProperties:
    #: false``. There is no template field, so there is nowhere in the contract for a
    #: template to live. F3 does not add one (FR-003.2).
    #:
    #: The alternative was a hard-coded template in the plugin, which is precisely the
    #: ADR-018 shape this project keeps paying for: a default that always fires is a
    #: fabricated value, and a judge template is a verdict policy. So it is **explicit
    #: operator configuration, unset by default**, and the judge type refuses
    #: ``judge_template_not_configured`` without it rather than judging with a template
    #: nobody chose.
    judge_prompt_template: str | None = None

    # ------------------------------------------------------------------ health signals
    #: A 1-of-1 batch is not evidence of anything, so no signal is raised below this.
    selection_min_n_for_signals: int = Field(default=5, ge=1)
    #: Below this variance a grader's scores count as having no spread - the GRPO
    #: zero-advantage case. Explicit because "zero" in floating point is a choice.
    grader_variance_epsilon: float = Field(default=1e-9, ge=0.0)

    # ------------------------------------------------------------------ bounds
    #: The synchronous endpoint's cap. A dry-run is five prompts, not five thousand; an
    #: unbounded synchronous endpoint over a judge is a request that outlives its client.
    evaluate_max_completions: int = Field(default=200, ge=1)
    #: miDataworks caps text at 65,536 characters - verified in that repo, not assumed.
    #: Exceeding it **refuses with the bound named** rather than truncating, because
    #: truncation changes the verdict.
    selector_max_text_chars: int = Field(default=65_536, ge=1)
    #: miDataworks accepts at most 500 candidates per post - verified.
    review_batch_max: int = Field(default=500, ge=1, le=500)
    #: Pattern-length ceiling for a rule regex, applied at ``prepare``. A pathological
    #: pattern is refused once per configuration, not run against every completion.
    rule_max_pattern_chars: int = Field(default=512, ge=1)

    @field_validator("judge_provider_base_url")
    @classmethod
    def _a_base_url_carries_no_credentials(cls, value: str | None) -> str | None:
        """**Refuse** a judge base URL with ``user:password@`` in it, rather than mask it.

        This class had **zero validators**, so the guard ``core.config`` carries for
        ``millm_base_url`` and ``midataworks_base_url`` did not exist here - the
        split-module blindness this project has now paid for three times (``packs_root``
        twice, declared in ``packs/settings.py`` and missed by every list written against
        ``core/config.py``; and the base-URL guard, hand-parametrised over *one* settings
        class). The structural answer is not another remembered copy of this method: it is
        ``tests/unit/test_all_base_urls_refuse_credentials.py``, which **enumerates** every
        ``BaseSettings`` subclass reachable from ``backend/src`` and requires each
        ``*_base_url`` field to refuse. That test is what makes the next instance fail in
        CI rather than in production.

        A base URL is a *location*. The credential for this endpoint has its own field -
        ``judge_provider_api_key``, which carries ``repr=False`` - and identity travels
        separately on all three of this system's surfaces. ``http://user:token@host`` is not
        a supported deployment; it is a credential arriving through a field nothing treats
        as secret, and masking it on the way out would hide the symptom while leaving the
        credential live on the wire (ADR-018's silent-success failure).

        **The message names the HOST, never the URL**, because a refusal is an output: it
        reaches a log line and the operator's terminal. ``hide_input_in_errors=True`` above
        closes the other half, which ``repr=False`` does not cover.
        """
        if value is None:
            return value
        from urllib.parse import urlsplit

        try:
            parts = urlsplit(value)
        except ValueError:  # pragma: no cover - urlsplit is tolerant
            return value
        # `username` is set for `//user@host` and `//user:pw@host` alike. Both are refused:
        # a bare username is not a secret, but it means the URL is being used to carry
        # identity, which is the pattern this refusal exists to keep out. Same reading as
        # `core.config`, deliberately - one convention.
        if parts.username or parts.password:
            raise ValueError(
                "a judge base URL must not carry credentials. Got a URL with embedded user "
                f"information for host {parts.hostname or '<none>'!r}; configure the host "
                "only (https://host:port) and set the credential in "
                "JUDGE_PROVIDER_API_KEY, which is held with repr=False and never "
                "serialised. A credential in this field is not masked in logs, in repr or "
                "in a recorded provider identity, so it is refused rather than accepted "
                "and hidden."
            )
        return value


_settings: SelectorSettings | None = None


def get_selector_settings() -> SelectorSettings:
    """The process-wide selector settings, built once.

    Not an ``lru_cache``: the tests need :func:`reset_selector_settings`, and a cache that
    cannot be cleared makes an env-var fixture silently ineffective - which is how a
    standalone test ends up asserting against a configuration it did not set.
    """
    global _settings
    if _settings is None:
        _settings = SelectorSettings()
    return _settings


def reset_selector_settings() -> None:
    """Drop the cached settings. For tests that change the environment."""
    global _settings
    _settings = None
