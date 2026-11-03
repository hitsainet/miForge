"""How a part's bytes arrive  (F10 tasks 4.1-4.3; FR-010.13, FR-010.19).

**Grouped by how bytes arrive, which is also the axis on which the availability guard
differs.** That is deliberate: the "needs the sibling" distinction lives in one place
instead of being re-reasoned at each call.

===================  ==========================================  ================
Path                 Needs miDataworks configured?               Guard
===================  ==========================================  ================
``from_dir``         **no** - a manifest is self-describing      none
``from_http``        yes                                         ``part_import_http``
``from_hub``         **no** - deferred, see below                none (when built)
===================  ==========================================  ================

**Why ``from_dir`` needs no guard, and why that is the point.** A
``midataworks.dataset-version/v1`` document says what it is, where it came from, under what
licence and with what caveats. So a contract-conformant part can be taken in **with
miDataworks absent** - which is the whole dividend of a vendored contract and the reason
"optional" is this feature rather than a caveat on it.

## OQ-1, resolved: the directory path is built first, the Hub path is recorded debt

miDataworks' export service states that a miForge set *"is delivered by publishing its
version to a private repository (miDataworks FR-008.68); the local directory is for
inspection"*, which makes the Hub the **intended** channel. It is also the channel that
works unconfigured, which is a strong argument for it. Against that: it needs an
``HF_TOKEN`` and outbound network access, which a LAN-only install may not have, while a
mounted directory needs neither and a Hub download produces *exactly the same directory*
that ``from_dir`` then takes in.

So the decision taken here is: **``from_dir`` and ``from_http`` are built; ``from_hub`` is
recorded as debt**, because it is a *fetch* in front of ``from_dir`` rather than a third
ingest path, and nothing in this feature's acceptance criteria needs it. The debt is
written in the feature report, not left implicit. :func:`from_hub` exists as a named
refusal rather than as an absence, so a caller discovers the gap by name instead of by
``AttributeError``.
"""

from __future__ import annotations

import logging
from collections.abc import Mapping
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any

from sqlalchemy.orm import Session

from clients.midataworks import MiDataworksClient
from core.errors import refuse
from services.parts.contract_validator import MANIFEST_FILENAME
from services.parts.ingest import IngestResult, ingest
from services.parts.tables import SOURCE_CONTRACT, SOURCE_EXTERNAL

log = logging.getLogger("miforge.parts.acquire")


def from_dir(
    session: Session,
    directory: Path,
    *,
    parts_root: Path,
    source: str = SOURCE_CONTRACT,
    external_provenance: Mapping[str, Any] | None = None,
) -> IngestResult:
    """Take a part in from a directory an operator copied or mounted.

    **No availability guard**: this path works with miDataworks unconfigured, and a guard
    here would be the first step back toward the hard dependency the 2026-10-13 reframe
    removed.

    The directory is never written to. ``source`` defaults to the contract channel because
    a conformant export *is* miDataworks' material however it arrived; pass
    ``external_files`` with an ``external_provenance`` for material from outside the suite
    (R-02.40).

    Raises:
        Refusal: ``part_directory_missing``, then any of ingest's named refusals.
    """
    if not directory.is_dir():
        raise refuse(
            "part_directory_missing",
            f"{directory} is not a directory. Point at the export directory itself - the "
            f"one containing {MANIFEST_FILENAME}.",
            directory=str(directory),
        )
    return ingest(
        session,
        directory,
        parts_root=parts_root,
        source=source,
        external_provenance=external_provenance,
        origin=str(directory),
    )


def from_http(
    session: Session,
    export_id: str,
    *,
    parts_root: Path,
    client: MiDataworksClient | None = None,
) -> IngestResult:
    """Take a part in over miDataworks' export routes.

    Behind ``require_midataworks("importing a part over HTTP", "part_import_http")``,
    which :class:`clients.midataworks.MiDataworksClient` applies in its constructor - so an
    unconfigured install refuses at the boundary rather than failing later with a ``None``
    base URL.

    **The same validator as every other path** (FR-010.11). There is no trusted route: a
    version that fails the contract is a defect wherever it came from, and a bypass for
    "parts we fetched ourselves" is how an invalid part gets a part row.

    The export's files are downloaded into a temporary directory and handed to the same
    :func:`services.parts.ingest.ingest` as a mounted directory, so the two paths produce
    **identical** rows by construction rather than by two code paths agreeing.

    Raises:
        Refusal: ``midataworks_not_configured`` when unset; ``midataworks_unreachable`` /
            ``_unauthorized`` / ``_refused`` per the client's status table;
            ``export_has_no_files`` when the export record names none.
    """
    owned = client is None
    api = client or MiDataworksClient(
        action="importing a part over HTTP", capability="part_import_http"
    )
    try:
        record = api.get_export(export_id)
        names = _file_names(record, export_id)
        with TemporaryDirectory(prefix="miforge-part-fetch-") as scratch:
            staged = Path(scratch)
            for name in names:
                target = staged / name
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_bytes(api.fetch_export_file(export_id, name))
            return ingest(
                session,
                staged,
                parts_root=parts_root,
                source=SOURCE_CONTRACT,
                origin=f"miDataworks export {export_id}",
            )
    finally:
        if owned:
            api.close()


def _file_names(record: Mapping[str, Any], export_id: str) -> list[str]:
    """The export's filenames, with the manifest guaranteed to be among them.

    Their ``export_file`` route allows ``{f["path"] for f in row.files} | {MANIFEST_FILE}``
    - the manifest is servable whether or not it is listed in ``files`` - so it is added
    here rather than assumed present. Verified by reading ``export_service.export_file``.
    """
    files = record.get("files") or []
    names = [
        str(entry["path"]) for entry in files if isinstance(entry, Mapping) and entry.get("path")
    ]
    if MANIFEST_FILENAME not in names:
        names.append(MANIFEST_FILENAME)
    if len(names) == 1:
        raise refuse(
            "export_has_no_files",
            f"miDataworks export {export_id} lists no files, so there is nothing to "
            f"import. Check that the export completed.",
            export_id=export_id,
            status=record.get("status"),
        )
    return names


def from_hub(*_args: object, **_kwargs: object) -> IngestResult:
    """**Not built.** Recorded debt, refused by name rather than missing.

    Raises:
        Refusal: ``acquisition_path_not_built``, naming the two paths that are.

    See the module docstring for OQ-1's resolution. A Hub fetch is a download in front of
    :func:`from_dir`, not a third ingest path, so building it later changes nothing about
    the contract, the validator or the index.
    """
    raise refuse(
        "acquisition_path_not_built",
        "Importing a part straight from a Hugging Face dataset repository is not built in "
        "this release. Clone or download the repository and import the directory "
        "(`source=dir`), which is the same path a Hub fetch would hand to - or import over "
        "miDataworks' export routes (`source=http`) when it is configured.",
        built=["dir", "http"],
        requested="hub",
    )


#: The acquisition paths a request may name, and whether each needs the sibling. Read by
#: the import endpoint so the route does not carry its own copy of this table.
PATHS: dict[str, bool] = {"dir": False, "http": True, "hub": False}

__all__ = [
    "PATHS",
    "SOURCE_CONTRACT",
    "SOURCE_EXTERNAL",
    "from_dir",
    "from_http",
    "from_hub",
]
