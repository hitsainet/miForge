"""Provider configuration, and why a row is configuration rather than code  (FR-004.3; ADR-010).

**miLLM is a row, not a name.** R-02.49's own rationale is that *"a test path wired to one
sibling by name is what made the whole document depend on it"* - so an HTTP provider here is
a ``base_url``, a model identity and an optional credential, and miLLM, vLLM, Ollama and the
OpenAI API are **configuration values, not code paths.** There is no ``if kind == "millm"``
anywhere in this package and the scanner in
``backend/tests/unit/test_provider_registry_guard.py`` proves it, with a self-test proving
the scanner sees each form it forbids.

## Why an environment scan rather than named fields

``pydantic-settings`` wants one field per setting, and the shape FR-004.3 requires is *a row
per configured endpoint, named by the operator* - an unbounded set of names. A fixed field
list would mean miForge shipping a name for every endpoint anyone might configure, which is
the named-dependency shape this requirement exists to remove. So the rows are discovered by
prefix from the environment:

    PROVIDER_<NAME>_BASE_URL      required to make a row exist at all
    PROVIDER_<NAME>_MODEL         required beside it; the pair half-set is REFUSED
    PROVIDER_<NAME>_API_KEY       optional - a locally served model needs none
    PROVIDER_<NAME>_CONNECT_TIMEOUT_S / _READ_TIMEOUT_S
    PROVIDER_<NAME>_CONCURRENCY

**Unset means absent-and-normal** (ADR-010): with nothing configured there are no HTTP rows
and the in-process provider is the only one, which is exactly the standalone case
acceptance criterion 20 pins.

## A credential in the base URL is refused, not masked

``PROVIDER_<NAME>_BASE_URL`` carrying ``user:password@`` is refused at the boundary. The
credential belongs in ``PROVIDER_<NAME>_API_KEY``, which the row holds and
:meth:`HttpProviderRow.redacted` reports only as ``has_api_key``. See
:func:`refuse_credentials_in_base_url` for why this surface is the worst place in the
codebase for the gap to have existed: ``base_url`` is serialised **verbatim** into the
provider identity recorded on every score, so the leak **persisted to disk** rather than
scrolling past in a log.

## The pair half-set is a refusal, not a default

``PROVIDER_JUDGE_BASE_URL`` with no ``..._MODEL`` is refused by name. A default model would
be the fabricated value ADR-018 forbids in the place it costs most: the model identity is
**recorded on every score** (R-02.16), so a guessed one makes every number it touches
incomparable while looking fine.

## Timeouts are explicit and retries stay off

Both timeouts have stated defaults here and are always sent to the provider, which takes
them as required keyword arguments with none of its own. *A default-less timeout is how an
outage becomes a hang.* Client retries stay off regardless of configuration, for miStudio's
recorded reason: *"a retry inside the client hides a job-level failure as latency: the run
would appear to hang rather than report that the judge is down."*
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass

from core.errors import refuse

#: The environment prefix a provider row is discovered by.
ENV_PREFIX = "PROVIDER_"

#: The suffix that makes a row exist. Everything else is optional or defaulted.
ENV_BASE_URL = "_BASE_URL"

#: A provider name an operator may use. Lower-cased, and deliberately the same shape
#: miDataworks requires of an agent name - a vocabulary that is already in this codebase
#: beats a second one.
NAME_PATTERN = re.compile(r"^[a-z0-9][a-z0-9_-]{0,47}$")

#: Stated defaults. Named constants rather than inline literals so a benchmark report can
#: quote them: ADR-024 - a figure without its parameters is not a measurement.
DEFAULT_CONNECT_TIMEOUT_S = 5.0
DEFAULT_READ_TIMEOUT_S = 60.0
DEFAULT_CONCURRENCY = 4


def refuse_credentials_in_base_url(
    base_url: str, *, variable: str, name: str | None = None
) -> None:
    """Refuse a provider base URL carrying ``user:password@``. **Refuse, never mask.**

    This is ``core.config``'s guard applied to the surface that needed it most, and it was
    missing here. The reason it is worse here than there:
    :meth:`HttpProviderRow.redacted` is documented as *"the only form that may leave this
    process"* and it includes ``base_url`` **verbatim**. So a credential embedded in
    ``PROVIDER_<NAME>_BASE_URL`` left through ``GET /api/v1/providers`` and - the part that
    does not wash out - into the provider identity recorded on **every score** and into run
    log provenance. A leaked log line is bounded by log retention; a leaked artifact
    **persists to disk** and a checkpoint is built to be copied off-box.

    **Masking was not an option**, for ADR-018's reason and this file's own: *"masking hides
    the symptom and leaves the credential live on the wire"*, sent from a field no reviewer
    reads as sensitive. A base URL is a *location*. Identity travels separately on all three
    of this system's surfaces, and this package already has a field for it
    (``PROVIDER_<NAME>_API_KEY``), which :meth:`HttpProviderRow.redacted` deliberately
    reports only as ``has_api_key``.

    **The message names the HOST and never the URL.** A refusal is an output - it reaches a
    log line, an error envelope and the operator's terminal - so a guard whose message
    reproduces what it refused has moved the credential rather than stopped it. That is the
    mistake ``core/config.py:50`` records: the first version of *its* refusal was echoed
    back by Pydantic's ``input_value=``, which is why ``hide_input_in_errors=True`` is set
    there. Nothing here renders through Pydantic, so the message is the whole surface and it
    is built from ``hostname`` alone.

    Args:
        base_url: The configured value. Not interpolated into any message.
        variable: The environment variable to name in the refusal, so the operator knows
            which of an unbounded set of rows to fix.
        name: The provider key, when one has been resolved.

    Raises:
        Refusal: ``inference_provider_misconfigured``.
    """
    from urllib.parse import urlsplit

    try:
        parts = urlsplit(base_url)
    except ValueError:  # pragma: no cover - urlsplit is tolerant
        return
    # `username` is set for `//user@host` and `//user:pw@host` alike, and both are refused:
    # a bare username carries no secret, but it means the URL is being used to carry
    # identity, which is the pattern this refusal exists to keep out. Same reading as
    # `core.config._a_base_url_carries_no_credentials`, deliberately.
    if not (parts.username or parts.password):
        return
    details: dict[str, object] = {"variable": variable, "host": parts.hostname or "<none>"}
    if name is not None:
        details["name"] = name
    raise refuse(
        "inference_provider_misconfigured",
        f"{variable} carries embedded user information and is refused. Configure the host "
        f"only (https://{parts.hostname or 'host'}:port) and set the credential in "
        f"{ENV_PREFIX}<NAME>_API_KEY, which is held on the row and never serialised. A "
        f"credential here is not refused for tidiness: base_url is included verbatim in the "
        f"serialisable form, so it would reach GET /api/v1/providers, every score's recorded "
        f"provider identity and the run log - and those persist to disk.",
        **details,
    )


@dataclass(frozen=True, slots=True)
class HttpProviderRow:
    """One configured OpenAI-compatible endpoint.

    Frozen, and **the credential is held here and nowhere else**: it never reaches a
    :class:`providers.base.ProviderIdentity`, which has no field for one, and never reaches
    a record, a log line or an API response (ADR-020, E-23). :meth:`redacted` is the only
    form that may leave this process.
    """

    name: str
    base_url: str
    model: str
    api_key: str | None = None
    connect_timeout_s: float = DEFAULT_CONNECT_TIMEOUT_S
    read_timeout_s: float = DEFAULT_READ_TIMEOUT_S
    concurrency: int = DEFAULT_CONCURRENCY

    def __post_init__(self) -> None:
        """Refuse a credential-bearing ``base_url`` **on construction**.

        Here rather than in :func:`http_provider_rows` on purpose. A check in the env scan
        guards the one path that exists today; a check on the type guards every path there
        will ever be, including a row built directly by another module or a test. *A guard
        that can be bypassed by constructing the object another way is a convention, not a
        guard* - and this is the type whose serialisable form persists to disk.
        """
        refuse_credentials_in_base_url(
            self.base_url,
            variable=f"{ENV_PREFIX}{self.name.upper()}{ENV_BASE_URL}",
            name=self.name,
        )

    def redacted(self) -> dict[str, object]:
        """The serialisable form. **No credential, not even redacted-by-value.**

        A ``"***"`` placeholder would tell a reader a credential exists, which is a fact
        the listing has no reason to carry; ``has_api_key`` says the same thing without
        implying the value is somewhere in the payload.

        ``base_url`` is included **verbatim** and that is correct - it is a location, and a
        listing that will not say where a provider is is useless. It is only safe because
        :func:`refuse_credentials_in_base_url` runs in :meth:`__post_init__`, so a row
        carrying a credential in its URL cannot exist to be serialised.
        """
        return {
            "name": self.name,
            "base_url": self.base_url,
            "model": self.model,
            "has_api_key": self.api_key is not None,
            "connect_timeout_s": self.connect_timeout_s,
            "read_timeout_s": self.read_timeout_s,
            "concurrency": self.concurrency,
        }


def http_provider_rows(environ: dict[str, str] | None = None) -> list[HttpProviderRow]:
    """Every configured HTTP provider row, from the **live** environment.

    Read live on every call rather than cached at import, as F3's capability listing is, so
    an operator who configures an endpoint sees it in ``GET /api/v1/providers`` without a
    restart of this layer.

    Raises:
        Refusal: ``inference_provider_misconfigured`` for a base URL carrying a credential,
            for a base URL with no model, for a name that is not a usable identifier, or
            for a non-numeric timeout. All four
            are refused at the boundary rather than at the first prompt - R-02.48's rule
            that *"a run that cannot proceed says so before it starts, not partway
            through"*, applied to the configuration a run would be started with.
    """
    source = os.environ if environ is None else environ
    rows: list[HttpProviderRow] = []
    for key, value in sorted(source.items()):
        if not key.startswith(ENV_PREFIX) or not key.endswith(ENV_BASE_URL):
            continue
        raw_name = key[len(ENV_PREFIX) : -len(ENV_BASE_URL)]
        base_url = (value or "").strip()
        if not base_url:
            # An empty value means unset, exactly as `core.config` treats a blank sibling
            # URL: `.env.example`'s deliberately-blank lines must not configure a provider
            # at the empty URL, or the standalone path is never taken on a developer's
            # machine.
            continue
        name = raw_name.lower()
        # BEFORE the name and model checks, and the same helper :meth:`HttpProviderRow.
        # __post_init__` calls - one guard, two call sites, not two guards. Called here as
        # well so the refusal can name the variable with the operator's own spelling, and so
        # a credential takes precedence over the missing-model refusal: with both wrong, the
        # credential is the one that must be reported first, because the operator's next
        # action for the other one is to set a model and ship the leak.
        refuse_credentials_in_base_url(base_url, variable=key, name=name)
        if not NAME_PATTERN.match(name):
            raise refuse(
                "inference_provider_misconfigured",
                f"{key} names the provider {raw_name!r}, which is not a usable provider "
                f"key: it must match {NAME_PATTERN.pattern}. A selector configuration "
                f"names a provider by this key, so an unusable one cannot be resolved.",
                variable=key,
                name=raw_name,
            )
        model = (source.get(f"{ENV_PREFIX}{raw_name}_MODEL") or "").strip()
        if not model:
            raise refuse(
                "inference_provider_misconfigured",
                f"{key} is set but {ENV_PREFIX}{raw_name}_MODEL is not. Both are needed: "
                f"the model identity is recorded on every score this provider produces, "
                f"and miForge will not choose one - a guessed model name makes every "
                f"number it touches incomparable while looking correct.",
                variable=key,
                name=name,
                missing=f"{ENV_PREFIX}{raw_name}_MODEL",
            )
        rows.append(
            HttpProviderRow(
                name=name,
                base_url=base_url,
                model=model,
                api_key=(source.get(f"{ENV_PREFIX}{raw_name}_API_KEY") or None),
                connect_timeout_s=_float(
                    source, f"{ENV_PREFIX}{raw_name}_CONNECT_TIMEOUT_S", DEFAULT_CONNECT_TIMEOUT_S
                ),
                read_timeout_s=_float(
                    source, f"{ENV_PREFIX}{raw_name}_READ_TIMEOUT_S", DEFAULT_READ_TIMEOUT_S
                ),
                concurrency=int(
                    _float(source, f"{ENV_PREFIX}{raw_name}_CONCURRENCY", DEFAULT_CONCURRENCY)
                ),
            )
        )
    return rows


def _float(source: dict[str, str] | os._Environ[str], key: str, default: float) -> float:
    """A numeric setting, **refused rather than defaulted** when it is present and wrong.

    An unparseable timeout silently falling back to the default is the shape ADR-018 names:
    the operator set a number, it was discarded, and nothing said so. An *absent* value is
    different and takes the stated default.
    """
    raw = source.get(key)
    if raw is None or not str(raw).strip():
        return default
    try:
        return float(raw)
    except ValueError as exc:
        raise refuse(
            "inference_provider_misconfigured",
            f"{key}={raw!r} is not a number. Refusing rather than falling back to "
            f"{default}: a discarded setting the operator chose is a configuration they "
            f"believe is in force.",
            variable=key,
            value=str(raw),
        ) from exc


@dataclass(frozen=True, slots=True)
class InProcessTuning:
    """The in-process provider's two numbers, both explicit (ADR-024).

    Named and defaulted here rather than inside the provider so a **benchmark report can
    quote them**: ADR-024's rule is that a figure without its parameters is not a
    measurement, and the KV coefficient and the batch bound are the two parameters a
    scoring-pass figure depends on. Both come from the live environment so a bound can be
    lowered on a card that is tighter than expected without a code change.
    """

    kv_bytes_per_token: int = 128 * 1024
    batch_bound: int = 8


def in_process_tuning(environ: dict[str, str] | None = None) -> InProcessTuning:
    """The in-process provider's tuning, from the **live** environment.

    Lives beside the HTTP rows rather than in ``services/scoring/`` so that
    :mod:`providers.registry` has no import into ``services/`` - the layering is one-way,
    and a lazy import inside a function is a dependency that exists while being hard to
    see.
    """
    source = os.environ if environ is None else environ
    defaults = InProcessTuning()
    return InProcessTuning(
        kv_bytes_per_token=int(
            _float(
                source,
                f"{ENV_PREFIX}IN_PROCESS_KV_BYTES_PER_TOKEN",
                float(defaults.kv_bytes_per_token),
            )
        ),
        batch_bound=int(
            _float(source, f"{ENV_PREFIX}IN_PROCESS_BATCH_BOUND", float(defaults.batch_bound))
        ),
    )
