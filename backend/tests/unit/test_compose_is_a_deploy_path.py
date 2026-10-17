"""`docker compose` can deploy the whole application, not just run it for a developer.

**Operator decision, 2026-10-14:** *"All of the apps need to be able to deploy independently,
and they all need to be able to deploy with Docker compose."* ADR-023's *"dev only"* is
amended accordingly. That turned the compose file from a convenience into a supported
deployment path, and a real cold start found **four** defects in it that no test in this suite
could see — every one found by running `docker compose --profile deploy up` and reading the
result, not by reading the file.

1. **The frontend image could not be built outside CI.** Its context is `./frontend` while
   its Dockerfile does `COPY LICENSE`, and the repository-root LICENSE is not in that
   context. CI injected it with a copy step; `docker build ./frontend` failed everywhere
   else. Fixed by a committed byte-identical `frontend/LICENSE`
   (`test_license_in_build_contexts.py`).
2. **The backend container died on import.** Eight modules resolved the vendored-schema
   directory from `__file__`, correct from the repository and `/usr/local/lib/docs/schemas`
   once pip-installed. Fixed by `MIFORGE_SCHEMA_DIR`
   (`test_schema_dir_override.py`).
3. **Nothing ran the migrations.** The image carried neither `alembic.ini` nor
   `backend/alembic/`, so a cold start served against an empty database: `/api/health`
   reported postgres **ok** — it checks connectivity, not schema — while every data endpoint
   returned 500 with `relation "mf_runs" does not exist`. Health green and the API wholly
   broken is the worst pair to ship.
4. **`/data/packs` had no volume**, and this one was data loss rather than an outage.
   `pack.json` became the authoritative record of a pack version the same day, so a
   committed pack lived in the container's writable layer and vanished on the next
   recreate — taking the authoritative copy and leaving the database index pointing at packs
   that no longer existed. It surfaced as a 500 only by luck, because `/data` is root-owned
   and the process runs as uid 10001, so the mkdir-on-demand was refused. Had `/data` been
   writable, the stack would have looked healthy and discarded packs silently.

**Defect 4 is the one worth generalising, and this file's central assertion does.** The three
roots the compose volumes covered are declared in `core/config.py`; `packs_root` is declared
in `packs/settings.py` — **a different module**. A volume list written against one module
cannot see the other, and nothing connected them. So the assertion below reads the roots from
**both** modules and requires a volume for each, which is what makes a fifth root added
anywhere fail here rather than in production.

These are static assertions over `docker-compose.yml`. They cannot prove the stack serves —
only a real `up` does that, and it was done: every endpoint answered 200 from a cold start
with empty volumes and no manual step. What they do is stop the four from coming back.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

yaml = pytest.importorskip("yaml")

REPO_ROOT = Path(__file__).resolve().parents[3]
COMPOSE = REPO_ROOT / "docker-compose.yml"

#: Services that read or write produced artifacts, and therefore need every artifact root
#: mounted. `migrate` is excluded deliberately: it touches the database only.
ARTIFACT_SERVICES: tuple[str, ...] = ("backend", "celery-train")


@pytest.fixture(scope="module")
def compose() -> dict[str, Any]:
    assert COMPOSE.is_file(), f"no compose file at {COMPOSE}"
    document = yaml.safe_load(COMPOSE.read_text("utf-8"))
    assert document.get("services"), "the compose file declares no services"
    return document


def _artifact_roots() -> dict[str, str]:
    """``{setting name: container path}`` for every artifact root, from **both** modules.

    Read from the settings classes rather than listed here, because a hard-coded list is
    exactly what missed `packs_root`: it lives in `packs/settings.py` while the other three
    live in `core/config.py`, and the compose volumes had been written against one module.
    """
    from core.config import Settings
    from packs.settings import PackSettings

    roots: dict[str, str] = {}
    for model in (Settings, PackSettings):
        for name, field in model.model_fields.items():
            if not name.endswith("_root"):
                continue
            default = field.default
            # `external_ingest_root` defaults to None - genuinely optional, no volume owed.
            if isinstance(default, Path) and str(default).startswith("/data/"):
                roots[name] = str(default)
    return roots


class TestTheArtifactRootsAllHaveVolumes:
    def test_the_settings_actually_declare_some(self) -> None:
        """The positive control: without roots, every assertion below passes vacuously."""
        roots = _artifact_roots()
        assert len(roots) >= 4, (
            f"expected at least the four known artifact roots, found {sorted(roots)}. If a "
            f"root moved off /data the assertions below stop covering it."
        )
        assert "packs_root" in roots, (
            "packs_root is the one that was missed, and it is missed by reading only "
            "core/config.py - it is declared in packs/settings.py"
        )

    @pytest.mark.parametrize("service", ARTIFACT_SERVICES)
    def test_every_root_is_a_named_volume_on_the_service(
        self, compose: dict[str, Any], service: str
    ) -> None:
        """A root without a volume is data in the container's writable layer.

        ADR-004 makes the filesystem the source of truth for produced artifacts, so an
        unmounted root is not a missing convenience - it is the authoritative copy living
        somewhere a `docker compose up -d --force-recreate` deletes.
        """
        definition = compose["services"].get(service)
        assert definition is not None, f"compose has no {service!r} service"
        mounted = {
            entry.split(":")[1]
            for entry in definition.get("volumes", [])
            if isinstance(entry, str) and ":" in entry
        }
        missing = {name: path for name, path in _artifact_roots().items() if path not in mounted}
        assert not missing, (
            f"{service} has no volume for {missing}. Artifacts written there live in the "
            f"container's writable layer and are deleted on the next recreate - and for "
            f"packs_root that is the authoritative copy of every committed pack, because "
            f"pack.json is the record and the database is only an index over it."
        )

    def test_the_mounts_are_named_volumes_not_host_bind_mounts(
        self, compose: dict[str, Any]
    ) -> None:
        """A bind mount would put artifacts where `git clean` can delete them.

        The compose file says so in terms, and this is the assertion behind the comment.
        """
        declared = set(compose.get("volumes") or {})
        for service in ARTIFACT_SERVICES:
            for entry in compose["services"][service].get("volumes", []):
                if not isinstance(entry, str) or ":" not in entry:
                    continue
                source, _, _ = entry.partition(":")
                if entry.split(":")[1].startswith("/data/"):
                    assert source in declared, (
                        f"{service} mounts {entry!r} from {source!r}, which is not a "
                        f"declared named volume - a host path puts a checkpoint where a "
                        f"`git clean` can delete it"
                    )


class TestTheSchemaIsMigratedBeforeAnythingServes:
    def test_a_one_shot_migrate_service_exists(self, compose: dict[str, Any]) -> None:
        migrate = compose["services"].get("migrate")
        assert migrate is not None, (
            "no `migrate` service. Without one a cold `up` serves against an empty database "
            "and every data endpoint 500s while /api/health reports postgres ok, because "
            "health checks connectivity and not schema."
        )
        command = migrate.get("command")
        rendered = " ".join(command) if isinstance(command, list) else str(command)
        assert (
            "alembic" in rendered and "upgrade" in rendered
        ), f"the migrate service does not run an alembic upgrade: {rendered!r}"
        assert migrate.get("restart") in ("no", None), (
            "a restart policy on a task that exits 0 turns a completed migration into a "
            "crash-loop"
        )

    @pytest.mark.parametrize("service", ARTIFACT_SERVICES)
    def test_the_service_waits_for_the_migration_to_SUCCEED(
        self, compose: dict[str, Any], service: str
    ) -> None:
        """`service_completed_successfully`, not a plain `depends_on`.

        A plain dependency waits for the container to *start*; a migration that failed would
        then let the backend serve against a half-migrated schema, which is worse than not
        starting at all.
        """
        depends = compose["services"][service].get("depends_on") or {}
        assert isinstance(
            depends, dict
        ), f"{service} uses the list form of depends_on, which cannot express a condition"
        assert (
            depends.get("migrate", {}).get("condition") == "service_completed_successfully"
        ), f"{service} does not wait for the migration to succeed: {depends.get('migrate')}"

    def test_the_image_carries_what_a_migration_needs(self) -> None:
        """The image had neither file, so the migrate service would have had nothing to run."""
        dockerfile = (REPO_ROOT / "backend" / "Dockerfile").read_text("utf-8")
        assert "alembic.ini" in dockerfile, "the image has no alembic.ini"
        assert "backend/alembic/" in dockerfile, "the image has no migration scripts"


class TestTheDeployProfileServesTheBuiltArtifact:
    def test_the_deploy_frontend_is_built_not_a_dev_server(self, compose: dict[str, Any]) -> None:
        """The dev service runs Vite from source and cannot deploy anywhere.

        It needs the source tree bind-mounted and serves through Vite's dev server, so a
        deploy profile pointing at it would be a deployment of a development server.
        """
        frontend = compose["services"].get("frontend")
        assert frontend is not None, "no `frontend` service in the deploy profile"
        assert "deploy" in (
            frontend.get("profiles") or []
        ), "the built frontend is not behind the `deploy` profile"
        assert frontend.get("build"), (
            "the deploy frontend has no build: it must be the published image's Dockerfile, "
            "not node:alpine running `npm run dev`"
        )
        assert "node:" not in str(
            frontend.get("image", "")
        ), "the deploy frontend is a node image, which means it is the dev server"
        assert not any(
            isinstance(v, str) and v.startswith("./") for v in frontend.get("volumes", [])
        ), (
            "the deploy frontend bind-mounts the source over the image's own dist/. This is "
            "the mistake a compose OVERRIDE file would have made silently, because Compose "
            "merges volumes by appending and cannot remove the dev service's bind mount - "
            "which is why the two UIs are profile-selected services in one file instead."
        )

    def test_it_waits_for_the_backend_to_be_HEALTHY(self, compose: dict[str, Any]) -> None:
        """nginx resolves its literal upstream **once**, at config load.

        With a plain `depends_on` it can load before the backend has an address and then 502
        every request forever - while reporting healthy, because its own healthcheck
        deliberately hits nginx's `/health` and not the API. Observed exactly that way.
        """
        depends = compose["services"]["frontend"].get("depends_on") or {}
        assert isinstance(depends, dict), "the deploy frontend uses the conditionless form"
        assert (
            depends.get("backend", {}).get("condition") == "service_healthy"
        ), f"the deploy frontend does not wait for backend health: {depends.get('backend')}"

    def test_the_dev_server_is_behind_its_own_profile(self, compose: dict[str, Any]) -> None:
        """Neither UI is a default, so a reader always knows which one is running."""
        dev = compose["services"].get("frontend-dev")
        assert dev is not None, "the Vite dev server service is gone"
        assert "dev" in (dev.get("profiles") or []), "the dev UI is not profile-gated"

    def test_no_sibling_base_url_is_configured(self, compose: dict[str, Any]) -> None:
        """Standalone by construction (ADR-010, acceptance criterion 20).

        Setting either URL here would quietly take the deployed stack off the no-sibling
        path that criterion 20 pins. Verified live as well: a cold start reports both
        siblings `*_not_configured` by name, with the capabilities that need them listed
        unavailable.
        """
        for name, definition in compose["services"].items():
            environment = definition.get("environment") or {}
            items = (
                environment.items()
                if isinstance(environment, dict)
                else (entry.split("=", 1) for entry in environment if "=" in entry)
            )
            for key, value in items:
                if key in ("MILLM_BASE_URL", "MIDATAWORKS_BASE_URL"):
                    assert not str(value).strip(), (
                        f"{name} configures {key}={value!r}. Unset means absent-and-normal; "
                        f"setting it here takes the deployed stack off the standalone path."
                    )
