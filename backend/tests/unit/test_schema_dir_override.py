"""Every module finds the vendored contracts when the package is **installed**, not just run.

**Why this exists: the backend container could not start, and the whole suite was green.**

Eight modules in this tree resolve ``docs/schemas/`` from ``__file__`` with their own
``parents[N]`` arithmetic. Every one is correct when the package is run from the repository —
and wrong the moment it is pip-installed, because ``parents[N]`` counts up from the module's
location and that location moves. From
``/usr/local/lib/python3.11/site-packages/packs/contracts.py`` the climb lands on
``/usr/local/lib/docs/schemas``, which does not exist.

The depths happen to compensate across the eight: a module two levels deep uses
``parents[4]`` and one level deep uses ``parents[3]``, so from site-packages they all
computed the *same* wrong path. One failure, eight sites.

The backend image died on import with::

    core.errors.Refusal: schema_not_vendored: batch-record-v1.json is not present at
    /usr/local/lib/docs/schemas/batch-record-v1.json

**No test in the suite could have caught it**, and that is the point of this file. Every test
runs from the repository, where the relative climb resolves. It surfaced only because the
operator made Docker Compose a deployment path on 2026-10-14, the deploy profile was actually
brought up, and the backend container was watched exiting — the UI served a clean 200 while
``/api/health`` returned 502 through nginx.

``MIFORGE_SCHEMA_DIR`` was already the override **five** of the eight honoured, so the fix
adopted that convention rather than inventing one: the other three were taught to honour it
and the image sets it to ``/app/docs/schemas``.

**What this file asserts is the override, not the default.** A test that only checked the
repository path would pass today and pass again the next time someone adds a ninth
``parents[N]``. Pointing the variable at a directory and requiring every module to look there
is the only form that fails when a module ignores it.
"""

from __future__ import annotations

import importlib
import json
import shutil
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[3]
VENDORED = REPO_ROOT / "docs" / "schemas"

#: ``(module, callable-or-attribute name)`` for every site that resolves the directory.
#: Listed explicitly rather than discovered, because the failure mode is a module **nobody
#: remembered** — and a discovery mechanism that missed one would reproduce exactly the gap
#: this file exists to close. A ninth site added without a line here is caught by
#: :func:`test_no_module_computes_the_directory_without_honouring_the_override` below.
RESOLVERS: tuple[tuple[str, str], ...] = (
    ("packs.contracts", "schema_dir"),
    ("training.run_log", "schema_path"),
    ("training.signals.batch_record", "schema_path"),
    ("services.selectors.resolve", "schema_dir"),
    ("services.scoring.manifest_payload", "schema_dir"),
    ("services.reindex", "schema_dir"),
    ("services.parts.contract_validator", "schema_dir"),
)

#: The eighth site, and it has **no named resolver** — ``services.scoring.sets`` reads the
#: variable inline inside the function that needs it. There is nothing to call, so it cannot
#: be parametrised above; its coverage is
#: :func:`test_no_module_computes_the_directory_without_honouring_the_override`, which greps
#: the source and so catches it along with any ninth site added the same way. Named here so
#: the count in this file's docstring (eight) matches the rows (seven) on purpose rather than
#: by an oversight a reader has to resolve.
UNNAMED_RESOLVER_SITES: tuple[str, ...] = ("backend/src/services/scoring/sets.py",)


@pytest.fixture
def relocated(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """The contracts at a path no ``parents[N]`` climb could ever produce.

    ``tmp_path`` is under ``/tmp``, nowhere near this repository or site-packages, so a module
    that ignores the variable cannot find them by accident — which is what makes a pass here
    mean something.
    """
    destination = tmp_path / "relocated-schemas"
    shutil.copytree(VENDORED, destination)
    monkeypatch.setenv("MIFORGE_SCHEMA_DIR", str(destination))
    return destination


class TestEverySiteHonoursTheOverride:
    def test_the_vendored_directory_is_where_we_think(self) -> None:
        """The positive control. Without it every assertion below could pass vacuously."""
        assert VENDORED.is_dir(), f"no vendored contracts at {VENDORED}"
        assert (VENDORED / "batch-record-v1.json").is_file()

    @pytest.mark.parametrize(("module_name", "attribute"), RESOLVERS)
    def test_the_site_resolves_to_the_override(
        self, module_name: str, attribute: str, relocated: Path
    ) -> None:
        """Each of the eight, individually named, so a failure says which one."""
        module = importlib.import_module(module_name)
        resolver = getattr(module, attribute, None)
        assert resolver is not None, (
            f"{module_name} has no {attribute!r}. If it was renamed, update RESOLVERS; if the "
            f"module stopped resolving the directory, remove its row — but do not delete the "
            f"row to make a red go away, because that is how the eighth site went unnoticed."
        )
        resolved = Path(resolver() if callable(resolver) else resolver)
        # Some sites return the directory and some the file inside it; both are accepted, and
        # what is asserted is that the override is on the path.
        assert str(relocated) in str(resolved), (
            f"{module_name}.{attribute} resolved to {resolved}, ignoring MIFORGE_SCHEMA_DIR. "
            f"It computes the directory from __file__ with its own parents[N] climb, which is "
            f"right from the repository and lands on /usr/local/lib/docs/schemas once the "
            f"package is pip-installed — the exact reason the backend image could not start."
        )

    def test_a_schema_actually_loads_through_the_override(self, relocated: Path) -> None:
        """Resolution is not the whole claim; the file has to be readable and parse.

        A path that points at the right directory and a loader that still reads elsewhere
        would pass the assertions above. This one goes through the real loader.
        """
        from packs import contracts

        contracts._cache.clear()
        reference = contracts.at(contracts.BATCH_RECORD_SCHEMA, "$defs", "ContractReference")
        assert set(reference["required"]) == {"type", "name", "version"}
        contracts._cache.clear()

    def test_an_override_pointing_nowhere_refuses_by_name(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The refusal has to name the path tried, or the next person debugs blind.

        This is the message that diagnosed the container crash: it printed the computed path,
        which is what made an eight-site arithmetic bug a five-minute fix instead of a hunt.
        """
        from core.errors import Refusal
        from packs import contracts

        monkeypatch.setenv("MIFORGE_SCHEMA_DIR", str(tmp_path / "absent"))
        contracts._cache.clear()
        with pytest.raises(Refusal) as caught:
            contracts.load(contracts.BATCH_RECORD_SCHEMA)
        contracts._cache.clear()
        assert caught.value.code == "schema_not_vendored"
        assert "absent" in caught.value.message, "the refusal does not name the path it tried"


class TestTheImageSetsIt:
    """The override is useless if the image does not set it, which is how this broke."""

    def test_the_backend_dockerfile_sets_the_variable(self) -> None:
        dockerfile = (REPO_ROOT / "backend" / "Dockerfile").read_text("utf-8")
        assert "MIFORGE_SCHEMA_DIR" in dockerfile, (
            "backend/Dockerfile does not set MIFORGE_SCHEMA_DIR. Every module then falls back "
            "to a parents[N] climb that, from site-packages, points at a directory that does "
            "not exist — and the container dies on import before serving one request."
        )

    def test_it_points_at_where_the_dockerfile_copies_them(self) -> None:
        """The two lines have to agree, and nothing else checks that they do."""
        dockerfile = (REPO_ROOT / "backend" / "Dockerfile").read_text("utf-8")
        assert "COPY" in dockerfile and "docs/schemas/" in dockerfile
        assert "MIFORGE_SCHEMA_DIR=/app/docs/schemas" in dockerfile, (
            "the variable and the COPY destination disagree. The image's WORKDIR is /app and "
            "the COPY lands at /app/docs/schemas; pointing the variable elsewhere reproduces "
            "the original failure with a different path in the error message."
        )


def test_no_module_computes_the_directory_without_honouring_the_override() -> None:
    """**The ratchet.** A ninth site added the old way fails here, not in production.

    Greps the source for a ``parents[N] / "docs" / "schemas"`` climb and requires the same
    file to mention ``MIFORGE_SCHEMA_DIR``. The climb itself is fine — it is the right
    default for a developer running from the repository — and what must never happen again is
    a climb with no override beside it.
    """
    import re

    climb = re.compile(r'parents\[\d+\]\s*/\s*"docs"\s*/\s*"schemas"')
    offenders: list[str] = []
    for path in sorted((REPO_ROOT / "backend" / "src").rglob("*.py")):
        text = path.read_text("utf-8")
        if climb.search(text) and "MIFORGE_SCHEMA_DIR" not in text:
            offenders.append(str(path.relative_to(REPO_ROOT)))
    assert not offenders, (
        "these modules compute the vendored-schema directory from __file__ and do not honour "
        "MIFORGE_SCHEMA_DIR, so they will look in /usr/local/lib/docs/schemas once the "
        f"package is installed and the container will not start: {offenders}"
    )


def test_the_relocated_copy_is_byte_identical(relocated: Path) -> None:
    """The fixture must not be the thing under test.

    If ``copytree`` altered a byte, a schema-conformance failure elsewhere in this file would
    look like a resolution bug. Cheap to rule out, and this project has lost time to a
    measurement that was itself broken more than once.
    """
    for original in sorted(VENDORED.glob("*.json")):
        copy = relocated / original.name
        assert copy.read_bytes() == original.read_bytes(), original.name
        json.loads(copy.read_text("utf-8"))
