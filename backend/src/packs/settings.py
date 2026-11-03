"""F7's settings  (FTID §9; ADR-003, ADR-004, ADR-010).

**One setting, and it exists because of a decision the gate contract made.**
``pack-definition-v1.json`` says, in its own description, that a pack definition *"is an
on-disk file, ``pack.json``"* - not a database-only record. ADR-003 puts the pack registry
in PostgreSQL while ADR-004 makes the filesystem authoritative for produced artifacts, and
packs were the **one** artifact with no on-disk manifest, so losing the database lost every
unexported pack and ``reindex`` could not rebuild them. The contract closed that by giving a
pack a file; this setting is where that file lives.

**Why these live here and not in :mod:`core.config`.** ``core/`` is shared state with one
owner and this feature does not edit it (PPRD §6). The two classes read the same ``.env``
and the same process environment, and ``Settings.model_config`` sets ``extra="ignore"``, so
the name below does not disturb it. The coordinator may fold it into ``core/config.py``
later; nothing in this feature reads it except through :func:`get_pack_settings`.
"""

from __future__ import annotations

from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


class PackSettings(BaseSettings):
    """Where pack definitions live, and how big a dry-run sample is."""

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
    )

    #: The root of the pack store. A fourth artifact root beside ``checkpoint_root``,
    #: ``trainer_state_root`` and ``parts_root`` (foundation task 8.5's PVCs). A pack is
    #: references, not data - a few kilobytes - so this is a small directory, and paying
    #: one small writer keeps ADR-004 universal instead of carving out an exception every
    #: future reader has to remember.
    packs_root: Path = Path("/data/packs")

    #: The directory operator-supplied files are read from, for external-part ingestion
    #: (R-02.40). **Unset by default, and unset means ingestion refuses by name** - reading
    #: whatever path arrives is a security decision, so this fails closed until someone
    #: makes it (ADR-013's "refuse with a reason", ADR-018's "a default that always fires is
    #: a fabricated value"). It is a local directory, **not** a sibling: configuring it does
    #: not make miForge depend on another application, and the standalone case (AC20) sets
    #: it to a directory on the same machine.
    external_ingest_root: Path | None = None

    #: The dry-run's sample size (R-02.39: *"a sample of the pack's own prompts"*). Five by
    #: default, which is the number the mockup shows and the number AC11's copy counts over
    #: ("Accepted 3 of 5"). Bounded deliberately: the dry-run is synchronous and
    #: in-request, and an unbounded sample over a judge is a request that outlives its
    #: client. If a bigger sample is ever wanted, that is a decision, not a default.
    dry_run_sample_size: int = 5


_settings: PackSettings | None = None


def get_pack_settings() -> PackSettings:
    """The process-wide pack settings, built once.

    Not an ``lru_cache``: the tests need :func:`reset_pack_settings`, and a cache that
    cannot be cleared makes an env-var fixture silently ineffective - which is how a
    standalone test ends up asserting against a configuration it did not set.
    """
    global _settings
    if _settings is None:
        _settings = PackSettings()
    return _settings


def reset_pack_settings() -> None:
    """Drop the cached settings. For tests that change the environment."""
    global _settings
    _settings = None
