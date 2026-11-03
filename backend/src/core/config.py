"""Settings, from the environment  (task 2.2; ADR-010, ADR-020).

**Every optional sibling URL is unset by default, and unset means absent-and-normal.**
That is ADR-010, and it is the mechanism by which acceptance criterion 20 - a full run of
each type with no sibling present - is true rather than asserted. miDataworks' own
``.env.example`` says it in one line: *"Optional siblings: leave either unset and
miDataworks runs standalone."* miForge copies the spirit.

**No secret is in code, and none reaches a manifest or a run log** (ADR-020). The §5.10
``trainer`` token is read from the environment and carries an audit label for miLLM's log;
:meth:`Settings.redacted` is what anything user-facing or persisted must go through.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Literal

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

#: R-02.2's provisional tolerance, as a module constant so that ``hotupdate/settings.py``
#: can report whether the value in force is still the unmeasured starting point without
#: either side importing the other. **The fence runs the other way**: ADR-022 forbids the
#: core path importing ``training/hotupdate/``, which is why this number lives here and is
#: read from there (SEAM 2).
PROVISIONAL_VERIFICATION_TOLERANCE = 1e-2
#: The NATIVE (bf16) comparison's bar since suite decision D41 (2026-10-17): 0.5, PROVISIONAL.
#: Window 2 measured the clean cross-process bf16 gap at 0.059-0.221 and a grossly wrong graph at
#: >= 2.29, so R-02.2's 1e-2 refused every correct reconstruction. It is the default because it
#: is what the float32 mode falls back to when miLLM cannot score in float32 (a 409
#: ``scoring_precision_unavailable``, e.g. a bitsandbytes model): a fallback onto 1e-2 would turn
#: "float32 unavailable" into "every attached run refused" (F11 14.3b, 2026-10-18).
D41_NATIVE_VERIFICATION_TOLERANCE = 0.5
#: The FLOAT32 comparison's bar, MEASURED (F11 14.3b, 2026-10-18; suite D61). On the RTX 3090,
#: LFM2.5-1.2B resident in bf16, miLLM's float32 scoring (a float32 copy on cuda, miLLM
#: FR-25.15, main ce8046b) against miForge's CPU float32 reference: clean max gap 1.96e-5 over 8
#: prompts and 4.96e-5 over 32; deliberately perturbed rebuilds - one block's w2 x1.001 3.62e-3,
#: w2 x1.01 0.0175, out_proj x1.01 0.0688; the native bf16 gap on the same prompts 0.139-0.378.
#: 5e-4 is 10x the largest clean gap and 7x under the smallest perturbation. CAVEAT: one model.
FLOAT32_VERIFICATION_TOLERANCE = 5e-4


class Settings(BaseSettings):
    """Process configuration. Instantiate once via :func:`get_settings`."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        # A typo'd variable is a silent default, so names are checked case-insensitively
        # but never guessed at.
        case_sensitive=False,
        # PYDANTIC ECHOES THE REJECTED VALUE INTO ITS ERROR MESSAGE BY DEFAULT, and this
        # model holds secrets. `str(ValidationError)` appends `input_value=...`, so a
        # validation failure on `millm_trainer_token` printed the token, and the refusal
        # added below for a credential-bearing base URL reproduced the very credential it
        # was refusing. A startup validation error is logged, so that is a leak through the
        # error path - the one path nobody thinks of as an output.
        #
        # `repr=False` on the field does NOT cover this: it governs `repr(Settings)`, not
        # Pydantic's error rendering. Found by the leak test for foundation task 2.5, which
        # asserted the refusal message does not contain the secret it refused.
        hide_input_in_errors=True,
    )

    # ---------------------------------------------------------------- core services
    database_url: str = "postgresql+psycopg://miforge:miforge@localhost:5432/miforge"
    redis_url: str = "redis://localhost:6379/0"

    # ---------------------------------------------------------------- artifact roots
    # ADR-004: the filesystem is the source of truth. These are the three PVCs of
    # foundation task 8.5.
    checkpoint_root: Path = Path("/data/checkpoints")
    trainer_state_root: Path = Path("/data/trainer-state")
    parts_root: Path = Path("/data/parts")
    #: Extra checkpoint roots miForge READS and never writes, comma-separated absolute
    #: paths; empty by default. D47 (2026-10-18): in hot-update mode miLLM writes the
    #: attached run's checkpoints under ITS checkpoint directory, so continuing that run
    #: needs to find them there (the deployment mounts it read-only). Searched after
    #: ``checkpoint_root``, in order, by :func:`services.checkpoints.paths.root_holding_run`.
    checkpoint_read_roots: str = ""

    # ---------------------------------------------------------------- optional siblings
    # UNSET BY DEFAULT. Do not give these a default value "for convenience": a default
    # base URL turns an absent sibling into a connection error at an arbitrary later
    # moment instead of a named refusal at the boundary, and it would quietly break the
    # standalone test that pins the whole 2026-10-13 reframe.
    millm_base_url: str | None = None
    midataworks_base_url: str | None = None

    #: miDataworks' REST surface identifies the caller with `X-Dataworks-Agent:
    #: agent:<name>` and has **no bearer gate** - verified in that repo's `core/` and
    #: `api/`. Three surfaces, three credentials, none shared: this header, miDataworks'
    #: MCP bearer token, and miLLM's `trainer` token below.
    midataworks_agent: str = "agent:miforge"

    # ---------------------------------------------------------------- section 5.10 only
    # The optional hot-update mode. ADR-022 fences it: nothing in the core path may read
    # these, and the standalone test runs with the mode absent.
    millm_trainer_token: str | None = Field(default=None, repr=False)
    millm_audit_label: str | None = None

    #: **R-02.2's verification tolerance, and it is PROVISIONAL** - the one value in this
    #: block with a default rather than ``None``, because the requirement supplies one.
    #: 1x10^-2 in log-probability space, as the *maximum absolute difference* over the
    #: compared token set, to be replaced before §5.10's acceptance by the observed spread
    #: across repeated identical calls on unchanged weights times a margin. Everything that
    #: records a comparison labels it provisional until that measurement exists: *a
    #: provisional bar that stops reading as provisional is a widened tolerance shipping
    #: without anyone deciding to widen it.*
    #: *Default 0.5 since 2026-10-18* (D41's value; it was R-02.2's 1e-2, which the window had
    #: to override by environment because it refused every clean bf16 reconstruction).
    hotupdate_verification_tolerance: float = D41_NATIVE_VERIFICATION_TOLERANCE
    #: Whether the tolerance in force was MEASURED as R-02.2 asks. **False by default, and a
    #: value other than 0.01 does not make it true**: suite decision D41 (2026-10-17) set 0.5
    #: from one session's numbers and it stays provisional until the operator says otherwise.
    hotupdate_verification_tolerance_measured: bool = False
    #: An override for the §5.10 heartbeat interval. **Unset is the normal case**: the
    #: interval comes from the attach response's hint, because the staleness timeout it
    #: must beat is miLLM's configuration and a local constant drifts from it in silence.
    hotupdate_heartbeat_interval_seconds: float | None = None
    hotupdate_request_timeout_seconds: float = 10.0
    #: `save` and `load` write or verify a whole checkpoint (2.4 GB for a 1.2B model) before
    #: answering, which outlasted the 10 s request timeout on the GPU node: the client then
    #: reported "unknown whether miLLM accepted" for a save miLLM had completed, and the run
    #: failed (found live 2026-10-18, F11 window 2, at the 13th save of a run).
    hotupdate_checkpoint_timeout_seconds: float = 600.0
    #: Deliberately the longest. A late ``commit-end`` is accepted by miLLM (BRD-01
    #: R-01.23a) and beats abandoning the bracket, which leaves the parameter set ``torn``
    #: and serving refused.
    hotupdate_commit_end_timeout_seconds: float = 30.0
    #: miLLM's R-01.22 commit watchdog. Held here only so a commit duration can be judged
    #: against the limit it will be cut off at - *a duration without its limit cannot be
    #: judged.* The value is the other repository's configuration: **800 ms since D45
    #: (2026-10-18)**, ten times the p99.9 commit of miForge's own real trainer (F11 14.4, run
    #: f11w2-r2: 1,000 steps, LFM2.5-1.2B-Instruct, 8-bit AdamW, RTX 3090: p99.9 71.0 ms),
    #: rounded up to the next 100 ms. Before that, 200 ms from synthetic commits (miLLM
    #: `db44fea`, p99.9 15.9 ms)
    #: and the provisional 2,000 ms. Mirror miLLM's `WEIGHTS_COMMIT_WATCHDOG_MS`; a cross-repo
    #: test pins the two equal (tests/unit/test_hotupdate_timings.py).
    hotupdate_commit_watchdog_ms: float = 800.0
    #: Warn when a commit exceeds this fraction of the watchdog. *A run whose commits creep
    #: toward the watchdog is the early warning that the watchdog is set too tight.*
    hotupdate_commit_creep_fraction: float = 0.5
    #: R-02.2's reconstruction check (F11 FR-011.9 - FR-011.13), run once before step 1.
    #: The fixed prompt; the token set is the rebuilt model's top-N next tokens for it,
    #: scored on both sides over exactly that set (N <= 20, the endpoint's alternatives cap).
    hotupdate_verify_prompt: str = "The capital of France is"
    hotupdate_verify_token_set_size: int = Field(default=15, ge=1, le=20)
    #: Where the tokenizer for the check comes from: a local directory or a Hub repo id.
    #: Unset: a checkpoint resident (``<base>:<run>:step-N``) reads the tokenizer files at
    #: that checkpoint's root (``attach.verify_tokenizer_source``), any other resident its
    #: served name. The tokenizer must be the SERVING model's - miForge's attach holds none.
    hotupdate_verify_tokenizer: str | None = None
    #: **Which precision R-02.2's comparison runs in** (F11 14.3b, D41's follow-up, D61).
    #: ``float32`` (the default since 2026-10-18): both sides in float32 - miForge over a float32
    #: copy of the rebuilt parameters, miLLM asked to score in float32 (``scoring_precision``,
    #: miLLM FR-25.15) and REQUIRED to confirm it by the echo; judged against
    #: :attr:`hotupdate_verification_tolerance_float32`. Sees a ~1 % reconstruction error.
    #: When miLLM REFUSES float32 by name (409 ``scoring_precision_unavailable``: a
    #: bitsandbytes/GGUF model, memory, device, busy) the check falls back to the native
    #: comparison under D41's provisional bar, logs the reason at WARNING and records
    #: ``requested_precision``/``precision_fallback_reason`` in the result - never silently.
    #: An answer WITHOUT the echo is still refused (that would be a silent downgrade).
    #: ``native``: both sides in the served dtype (bf16), judged against
    #: :attr:`hotupdate_verification_tolerance` (D41's provisional 0.5). Catches missing or
    #: failed tensors and gross errors; **cannot** see a ~1 % reconstruction error, which
    #: lands inside bf16's cross-process kernel spread (window 2: clean 0.059-0.221, 1 %
    #: error 0.072-0.220).
    hotupdate_verification_precision: Literal["native", "float32"] = "float32"
    #: The float32 mode's tolerance, **MEASURED 2026-10-18** (see
    #: :data:`FLOAT32_VERIFICATION_TOLERANCE` for the numbers: clean <= 4.96e-5, smallest
    #: perturbation 3.62e-3; one model, LFM2.5-1.2B on the 3090). Was R-02.2's provisional 1e-2.
    hotupdate_verification_tolerance_float32: float = Field(
        default=FLOAT32_VERIFICATION_TOLERANCE, gt=0
    )
    hotupdate_verification_tolerance_float32_measured: bool = True
    #: Where miForge's float32 forward runs. ``cpu`` by default: a float32 copy of a 1.2B
    #: model is ~4.8 GB, and the card is shared with the model being served.
    hotupdate_verify_float32_device: str = "cpu"
    #: How many times a step is retried after a refusal that arrived BEFORE commit-begin
    #: (BRD AC7: nothing was written, so the weights are unchanged). Never applies after
    #: commit-begin - an interrupted commit is never retried (R-02.31).
    hotupdate_precommit_retries: int = Field(default=1, ge=0, le=5)

    # ---------------------------------------------------------------- jobs
    #: ADR-007. A MINIMUM interval between writes, not a maximum gap - a generation-bound
    #: test pass or a 2.4 GB save legitimately exceeds it. Anything deriving a liveness
    #: threshold from this number is wrong; read the janitor's threshold instead.
    heartbeat_min_interval_seconds: int = 60
    #: The janitor's lapse threshold. Provisional until foundation 6.4b's longest-observed
    #: gap exists, because nothing else in the product can produce a defensible value.
    janitor_lapse_seconds: int = 900
    gpu_lease_seconds: int = 3600

    # ------------------------------------------------- the checkpoint lifecycle (OQ-4)
    # The operator's decision, 2026-10-16: keep the last N saves, every Mth save, and
    # unconditionally every checkpoint that is a parent in the lineage tree.
    #
    # NAMED `checkpoint_lifecycle_*`, NOT `retention_*`. `retention` already means
    # R-02.17's held-out set in this codebase - it is a `ScoreRecord` on every manifest and
    # a column on both index tables - and a second meaning for it would be a conflated
    # vocabulary of the kind that has already cost this project three corrections.
    #
    #: **Off by default**, and the sweep says so every time it declines. miStudio shipped
    #: its own step-granular checkpoint retention "disabled + dry-run" deliberately, after
    #: a feature that had deleted nothing was the only reason 423 rows and 78 GB were
    #: recoverable. A policy that starts deleting on upgrade is a policy nobody consented
    #: to on the installation it first runs on.
    checkpoint_lifecycle_enabled: bool = False
    #: **Dry run by default, and ``False`` means DELETE.** Keep that asymmetry in mind
    #: reading the validator below: a boolean setting must fail to its DEFAULT, not to
    #: False - miStudio's durable lesson from the same feature.
    checkpoint_lifecycle_dry_run: bool = True
    #: N - how many of a run's most recent saves survive regardless. Ordinal over the
    #: run's actual saves, not arithmetic on the step number (see `lifecycle.py`).
    checkpoint_lifecycle_keep_last_n: int = 10
    #: M - keep every Mth save, counted from the oldest, so a run's first save (the one a
    #: branch most often points at) is always among them. With `save_every=100` over
    #: 10,000 steps these two defaults keep about twenty checkpoints of a hundred, plus
    #: every branch point - which is OQ-4's sizing against an 800Gi volume.
    checkpoint_lifecycle_keep_every_m: int = 10

    # ---------------------------------------------------------------- F12 approvals
    #: How long an agent's pending request waits for the operator before it expires and can
    #: never execute (FR-012.14). 24 h, as miDataworks' queue.
    approval_ttl_hours: float = Field(default=24.0, gt=0)

    # ---------------------------------------------------------------- misc
    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR"] = "INFO"
    environment: Literal["dev", "ci", "prod"] = "dev"

    @field_validator(
        "millm_base_url",
        "midataworks_base_url",
        # §5.10's optional override, for the same reason and with a sharper failure: a
        # `float | None` field given the empty string does not fall back to its default, it
        # **fails validation at startup** - so `.env.example`'s deliberately-blank line
        # would stop the whole process rather than leave the mode unconfigured. Found by
        # setting the variable to "" and starting a Settings object, not by reading.
        "hotupdate_heartbeat_interval_seconds",
        # Blank means "resolve from the resident" (see above), never the empty-string path.
        "hotupdate_verify_tokenizer",
        mode="before",
    )
    @classmethod
    def _empty_string_is_unset(cls, value: Any) -> Any:
        """``MILLM_BASE_URL=`` in a ``.env`` file means **unset**, not the empty string.

        Without this, `.env.example`'s deliberately-blank lines (task 9.4) would configure
        every sibling as the empty URL, and the standalone path - the one pinned by
        acceptance criterion 20 - would never be taken on a developer's machine.
        """
        if isinstance(value, str) and not value.strip():
            return None
        return value

    @field_validator("checkpoint_read_roots")
    @classmethod
    def _read_roots_are_absolute(cls, value: str) -> str:
        for part in (p.strip() for p in value.split(",")):
            if part and not Path(part).is_absolute():
                raise ValueError(f"CHECKPOINT_READ_ROOTS entries must be absolute: {part!r}")
        return value

    @property
    def checkpoint_read_root_paths(self) -> list[Path]:
        """``checkpoint_read_roots`` parsed: absolute paths, blanks dropped, order kept."""
        return [Path(p.strip()) for p in self.checkpoint_read_roots.split(",") if p.strip()]

    @field_validator("millm_base_url", "midataworks_base_url")
    @classmethod
    def _a_base_url_carries_no_credentials(cls, value: str | None) -> str | None:
        """**Refuse** a sibling base URL with ``user:password@`` in it, rather than mask it.

        A base URL is a *location*. This architecture keeps credentials in three separate
        places and none of them is a URL: miDataworks' REST surface uses the
        ``X-Dataworks-Agent`` header, its MCP server a bearer token, and miLLM the
        ``trainer`` token above - *three surfaces, three credentials, none shared*. So
        ``http://user:token@host`` is not a supported deployment; it is a credential
        arriving through a field nothing treats as secret.

        **Masking it in the log would be the wrong fix, and that was the first thing tried.**
        ``millm_base_url`` has no ``repr=False`` and is not in :meth:`redacted`'s mask list,
        so a credential embedded here reaches any log line, traceback or settings dump that
        touches the object. Stripping it on the way out would hide the symptom and leave the
        credential live on the wire, sent from a field no reviewer reads as sensitive - the
        silent-success failure ADR-018 forbids. Refusing at startup is loud, is fixable in
        one edit, and cannot be half-applied.

        Found by the leak test for foundation task 2.5 rather than by review: nothing calls
        ``redacted()`` yet, so this was latent and would have become a real leak the first
        time anything logged the settings.
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
        # identity, which is the pattern this refusal exists to keep out.
        if parts.username or parts.password:
            raise ValueError(
                "a sibling base URL must not carry credentials. Got a URL with "
                "embedded user information; configure the host only "
                "(https://host:port) and supply identity separately - "
                "MILLM_TRAINER_TOKEN for miLLM, MIDATAWORKS_AGENT for miDataworks' REST "
                "surface. A credential in this field is not masked in logs or in repr, so "
                "it is refused rather than accepted and hidden."
            )
        return value

    @field_validator("midataworks_agent")
    @classmethod
    def _agent_matches_their_pattern(cls, value: str) -> str:
        """miDataworks refuses an identity that does not match ``^agent:[a-z0-9][a-z0-9-]{0,47}$``.

        Checked here so a misconfiguration fails at startup with a readable message rather
        than as a rejected call on the first part resolution.
        """
        import re

        if not re.fullmatch(r"agent:[a-z0-9][a-z0-9-]{0,47}", value):
            raise ValueError(
                f"midataworks_agent={value!r} does not match miDataworks' required form "
                f"'agent:<name>' (^agent:[a-z0-9][a-z0-9-]{{0,47}}$)"
            )
        return value

    @field_validator(
        "checkpoint_lifecycle_enabled",
        "checkpoint_lifecycle_dry_run",
        mode="before",
    )
    @classmethod
    def _a_lifecycle_flag_fails_to_its_default(cls, value: Any, info: Any) -> Any:
        """An unparseable value for either flag becomes **the field's default**, loudly.

        **Not False, and not a startup crash.** miStudio's checkpoint-lifecycle feature
        left this as a durable lesson: *a boolean setting must fail to its DEFAULT, not to
        False* - because for ``dry_run``, ``False`` means delete. The two failure modes a
        reader expects are both wrong here:

        * Pydantic's own behaviour is to raise, which stops the whole process over a
          typo'd flag on a field whose safe value is already the default. A deployment
          that cannot start because someone wrote ``DRY_RUN=maybe`` is an outage bought
          for nothing.
        * Falling through to ``False`` is the one outcome that must never happen:
          ``CHECKPOINT_LIFECYCLE_DRY_RUN=ture`` would then *delete checkpoints*, silently,
          with the operator believing the opposite.

        So an unrecognised value is replaced by the declared default and **logged as a
        warning naming the field and the value**. The empty string takes the same path:
        ``CHECKPOINT_LIFECYCLE_DRY_RUN=`` in a ``.env`` file means "I did not set this",
        which is the default, not ``False``.

        Recognised spellings are passed straight through to Pydantic, so the accepted
        vocabulary stays Pydantic's rather than becoming a second one declared here.
        """
        if isinstance(value, bool) or value is None:
            return value
        if not isinstance(value, str):
            return value
        text = value.strip().lower()
        if text in {"1", "true", "t", "yes", "y", "on", "0", "false", "f", "no", "n", "off"}:
            return value
        field_name = getattr(info, "field_name", "")
        default = cls.model_fields[field_name].default if field_name else None
        import logging

        logging.getLogger("miforge.config").warning(
            "unparseable boolean for a checkpoint-lifecycle flag; using the default",
            extra={
                "code": "checkpoint_lifecycle_flag_unparseable",
                "field": field_name,
                "value": value,
                "using": default,
            },
        )
        return default

    # ---------------------------------------------------------------- derived
    def configured_integrations(self) -> dict[str, bool]:
        """Which optional integrations are configured. Feeds ``GET /api/health``.

        Reported by name either way, because "not configured" is a fact a reader needs,
        and an integration that is simply missing from a response is indistinguishable
        from one that is broken.
        """
        from services.scoring.settings import get_scoring_settings

        return {
            "millm": self.millm_base_url is not None,
            "midataworks": self.midataworks_base_url is not None,
            # ⚠ THE THIRD SIBLING LIVES IN ANOTHER SETTINGS OBJECT, which is why it was
            # missing from here and therefore from `GET /api/health` on the day
            # `clients/activations.py` landed. Exactly the split-module omission that cost
            # `/data/packs` its volume twice in one day: `packs_root` is declared in
            # `packs/settings.py` and every list written against `core/config.py` missed it.
            # Read through the owning settings object rather than re-declared here, so one
            # field cannot become two.
            #
            # Imported inside the method on purpose: `services/` imports `core.config`, so a
            # module-scope import closes a cycle.
            "activation_source": get_scoring_settings().activation_source_base_url is not None,
        }

    def redacted(self) -> dict[str, Any]:
        """Settings safe to log, serve or persist. **The only form that may leave here.**"""
        data = self.model_dump(mode="json")
        for secret in ("millm_trainer_token",):
            if data.get(secret) is not None:
                data[secret] = "***"
        return data


#: **The settings an operator may change at runtime** through ``PUT /api/v1/settings/{key}``
#: (F12, FR-012.8; operator decision D5, 2026-10-18). Behaviour knobs that are safe to change
#: between jobs and are read through ``core.runtime_settings.effective_settings()`` by every
#: reader, so a stored override actually reaches the code that uses it. Enabling checkpoint
#: pruning is effectively a delete, which is why every write is gated ``config_write``.
#:
#: ``hotupdate_commit_watchdog_ms`` and ``hotupdate_commit_creep_fraction`` are deliberately
#: NOT here (FTID I-5): they mirror miLLM's own configuration and a cross-repo test pins the
#: watchdog equal, so changing one side at runtime would break that pin.
RUNTIME_KEYS: frozenset[str] = frozenset(
    {
        "checkpoint_lifecycle_enabled",
        "checkpoint_lifecycle_dry_run",
        "checkpoint_lifecycle_keep_last_n",
        "checkpoint_lifecycle_keep_every_m",
        "millm_base_url",
        "midataworks_base_url",
        "log_level",
        "hotupdate_request_timeout_seconds",
        "hotupdate_checkpoint_timeout_seconds",
        "hotupdate_commit_end_timeout_seconds",
        "hotupdate_verification_tolerance",
        "hotupdate_verification_tolerance_float32",
    }
)

#: **Never writable, named** (D5): the database and broker URLs, every artifact root, the
#: trainer token and the environment. Everything not in ``RUNTIME_KEYS`` is refused anyway;
#: this list exists so a test can try each of these by name and so a later edit that moves
#: one of them into ``RUNTIME_KEYS`` fails ``test_settings_api.py`` rather than shipping.
NEVER_WRITABLE: frozenset[str] = frozenset(
    {
        "database_url",
        "redis_url",
        "checkpoint_root",
        "trainer_state_root",
        "parts_root",
        "checkpoint_read_roots",
        "millm_trainer_token",
        "environment",
    }
)

#: Fields whose value is a secret and is never served (``GET /settings`` shows
#: ``"<redacted>"``). URL fields carrying a password are redacted separately.
SECRET_FIELDS: frozenset[str] = frozenset({"millm_trainer_token"})

_settings: Settings | None = None


def get_settings() -> Settings:
    """The process-wide settings, built once.

    Not an ``lru_cache`` on purpose: the tests need :func:`reset_settings`, and a cache
    that cannot be cleared makes an env-var fixture silently ineffective - which is how a
    standalone test ends up asserting against a configuration it did not set.
    """
    global _settings
    if _settings is None:
        _settings = Settings()
    return _settings


def reset_settings() -> None:
    """Drop the cached settings. For tests that change the environment."""
    global _settings
    _settings = None
