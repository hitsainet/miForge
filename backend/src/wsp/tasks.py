"""WSP Celery tasks (feature 016; FTDD §7; P60, PF-53, PF-57).

``miforge.wsp.drive`` runs every 5 s from beat (``expires=4``) on its own queue ``wsp``, consumed
only by the solo ``worker-wsp`` container, so a hung worker call never blocks the maintenance
janitor, the chain sweep or pruning (R2A-13). Standalone by default: with ``WSP_WORKER_URL`` or
``WSP_GATEWAY_URL`` unset the pass does nothing and returns ``wsp_not_configured``, and logs why
when a provider is enabled (review OR3-016-2).
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from workers.celery_app import (
    QUEUE_DEFAULT,
    QUEUE_WSP,
    TASK_WSP_DRIVE,
    TASK_WSP_IMPORT_BUNDLE,
    celery_app,
)
from wsp import dispatch
from wsp.models import WspBundle, WspEndpoint, WspRun
from wsp.settings import WSPSettings, get_wsp_settings, not_configured_message

log = logging.getLogger(__name__)

#: How often a skipped pass repeats its warning (the beat runs every 5 s).
NOT_CONFIGURED_WARN_EVERY_S = 600.0
_last_not_configured_warning: list[float] = []


def _warn_not_configured(settings: WSPSettings) -> None:
    """Review OR3-016-2: a skipped pass is silent in a standalone install (no provider enabled),
    and says why, at most every ten minutes, when a provider IS enabled but a URL is unset."""
    import time

    if not settings.providers_available():
        return
    now = time.monotonic()
    if _last_not_configured_warning and now - _last_not_configured_warning[0] < (
        NOT_CONFIGURED_WARN_EVERY_S
    ):
        return
    _last_not_configured_warning[:] = [now]
    log.warning(
        not_configured_message(settings.missing_urls()),
        extra={"code": "wsp_not_configured", "missing": settings.missing_urls()},
    )


def bundle_files(run: WspRun, bundle: WspBundle) -> list[tuple[str, Path]]:
    """The bundle's stored files, pushed to the worker in a fixed order (OVN-wsp-forge-27):
    ``<WSP_BUNDLE_ROOT>/bundles/<bundle_sha256>/``, regular files only, by name."""
    root = Path(get_wsp_settings().wsp_bundle_root) / "bundles" / bundle.bundle_sha256
    if not root.is_dir():
        return []
    return sorted((p.name, p) for p in root.iterdir() if p.is_file() and not p.is_symlink())


def run_files(run: WspRun, bundle: WspBundle) -> list[tuple[str, Path]]:
    """What the driver pushes for ``run``: the bundle's files, plus, for a replay, the ORIGINAL
    run's stored transcript, and for a program rerun, the stored program (FR-016.21, .23). The
    driver checks each against the digest the database holds before it consumes a boot."""
    from wsp import runs

    settings = get_wsp_settings()
    files = bundle_files(run, bundle)
    if run.style == "replay" and run.replay_of:
        files.append(
            (dispatch.TRANSCRIPT_NAME, runs.run_dir(settings, run.replay_of) / "transcript.jsonl")
        )
    elif run.style == "program_rerun" and run.program_sha256:
        files.append((dispatch.PROGRAM_NAME, runs.program_path(settings, run.program_sha256)))
    return files


def read_sensitivity(run: WspRun, bundle: WspBundle) -> str:
    """miWorkbench's live value (P41); any failure reads as ``unknown``, which is sensitive."""
    from clients.miworkbench import read_bundle_sensitivity

    return read_bundle_sensitivity(bundle.bundle_id)


#: miServer's health route that reports the resident model's window (``context_length``).
SERVED_WINDOW_PATH = "/api/health/inference"
SERVED_WINDOW_TIMEOUT_S = 5.0


def read_served_window(endpoint: WspEndpoint) -> int | None:
    """The window miServer reports for its resident model (feature 018, FR-018.12; FTASKS 9.1).

    ``GET <scheme>://<host:port>/api/health/inference`` (the endpoint's base URL without ``/v1``),
    5 s, from the BACKEND: the worker reaches only the gateway, which serves no such route. The
    answer carries ``context_length`` as a top-level integer and NO model key (read live
    2026-11-18: 32,768 with Qwen2.5-7B-Instruct resident), so the value cannot be matched to
    ``endpoint.model``. It is used anyway, because the gateway sends miServer
    ``X-miLLM-Load-Policy: refuse``: a session whose model is not the resident one is refused
    ``editor_model_not_resident`` at its first request, so a session that runs at all runs on the
    model this window belongs to (018-P2 deviation P2-D3). ``None`` on any failure; the caller
    never fails a push on it."""
    from urllib.parse import urlsplit

    import httpx

    parts = urlsplit(endpoint.base_url)
    url = f"{parts.scheme}://{parts.netloc}{SERVED_WINDOW_PATH}"
    with httpx.Client(timeout=SERVED_WINDOW_TIMEOUT_S, follow_redirects=False) as client:
        response = client.get(url)
    if response.status_code != 200:
        return None
    body = response.json()
    value = body.get("context_length") if isinstance(body, dict) else None
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        return None
    return value


@celery_app.task(name=TASK_WSP_DRIVE, queue=QUEUE_WSP)  # type: ignore[untyped-decorator]
def drive() -> dict[str, Any]:
    """One driver pass."""
    from db.session import get_session_factory
    from wsp.worker_client import HttpWorker

    settings = get_wsp_settings()
    if not settings.configured():
        _warn_not_configured(settings)
        return {"skipped": "wsp_not_configured"}
    worker = HttpWorker(settings.wsp_worker_url)
    try:
        done = dispatch.drive(
            get_session_factory(),
            worker,
            settings,
            read_sensitivity=read_sensitivity,
            files_for=run_files,
            read_served_window=read_served_window,
        )
    finally:
        worker.close()
    if done.actions:
        log.info("wsp drive pass", extra={"code": "wsp_drive", "actions": done.actions})
    return {"actions": done.actions}


@celery_app.task(name=TASK_WSP_IMPORT_BUNDLE, queue=QUEUE_DEFAULT)  # type: ignore[untyped-decorator]
def import_bundle_task(bundle_sha256: str) -> dict[str, Any]:
    """Pull, check and store one bundle (FR-016.1); the row ends ``imported`` or ``failed``."""
    from db.session import session_scope
    from wsp.bundles import import_bundle

    with session_scope() as session:
        outcome = import_bundle(session, bundle_sha256, settings=get_wsp_settings())
    log.info("wsp bundle import", extra={"code": "wsp_import", "outcome": outcome})
    return {"bundle_sha256": bundle_sha256, "outcome": outcome}
