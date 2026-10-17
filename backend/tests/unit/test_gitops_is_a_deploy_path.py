"""The GitOps manifests can actually serve the application, not merely describe it.

**This file is the Kubernetes half of `test_compose_is_a_deploy_path.py`, and it exists
because three of that file's four defects were still live in `gitops/pilot/` after Compose
was fixed.** Compose was made a supported deployment path on 2026-10-14 and a real cold
start found four defects in it that no test could see. The same day, `gitops/pilot/`
carried:

1. **No Ingress at all** — and that was the operator's reported 404. `miforge-frontend` was
   a bare `ClusterIP`, so every request arriving at the ingress-nginx controller matched no
   rule and was answered by the controller's **default backend**, which serves a bare 404.
   `frontend/nginx.conf`'s own header had recorded the gap — *"miForge HAS NO INGRESS"* —
   and nothing turned the note into a manifest. miLLM, miStudio and miDataworks all have
   one; miForge was the only app in the suite that could not be reached from outside.
2. **Nothing ran the migrations.** This is Compose defect 3, unfixed here.
   `backend/Dockerfile` deliberately has no `SERVICE_TYPE` entrypoint, unlike miStudio's
   and miDataworks' — whose entrypoints run ``alembic upgrade head`` for the API before
   uvicorn — so the API container ran the image `CMD` and no process in the pod migrated
   anything. A fresh namespace serves against an empty schema while ``/api/health``
   reports postgres **ok**, because it checks connectivity and not schema. Health green
   and every data endpoint 500 with ``relation "mf_runs" does not exist``.
3. **`/data/packs` had no volume.** This is Compose defect 4, unfixed here, and it is data
   loss rather than an outage: ``pack-definition-v1.json`` makes ``pack.json`` the
   authoritative on-disk record of a pack version, so a committed pack lived in the
   container's writable layer — and ``strategy: Recreate`` means **every deploy**
   destroyed it, leaving the database index pointing at packs that no longer existed.

**Defect 3 is the one worth generalising, and it is generalised the same way the Compose
test generalises it.** The three roots the manifests covered are declared in
``core/config.py``; ``packs_root`` is declared in ``packs/settings.py`` — *a different
module*. A volume list written against one module cannot see the other, and nothing
connected them. ``packs/settings.py``'s own docstring even called it "a fourth artifact
root beside ``checkpoint_root``, ``trainer_state_root`` and ``parts_root`` (foundation task
8.5's PVCs)", naming a file that had no such PVC — a claim its cited target did not
support, in the direction that costs an artifact. So the assertions below read the roots
from **both** modules and require a volume, a mount and a chown for each, which is what
makes a fifth root added anywhere fail here rather than in production.

**What these assertions are and are not.** They are static reads of the manifests and of
`kustomization.yaml`. They cannot prove the cluster serves — only a real sync can, and it
could not be run from this workstation (no kubeconfig; ``kubectl`` reaches nothing). What
they do is make each of the three a permanent regression, and in particular they assert
that the Ingress **is listed in the kustomization**, because Kustomize applies only what
`resources:` names: a manifest present on disk and absent from that list is never applied,
and the symptom is the same 404 — with the fix apparently in place.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

yaml = pytest.importorskip("yaml")

REPO_ROOT = Path(__file__).resolve().parents[3]
GITOPS = REPO_ROOT / "gitops" / "pilot"

#: The hostnames every rule set must carry. Derived from the three working deployments
#: rather than from one: miStudio's README states the convention ("ingress rules for both
#: local (`hitsai.local`) and tunneled (`hitsai.net`) access"), miLLM and miStudio each
#: carry the two `k8s-` forms, and miDataworks carries a bare third form as well.
#:
#: ⚠ The `.net` entry is the load-bearing one and the reason this list is asserted at all.
#: A Cloudflared tunnel presents the PUBLIC hostname, so an Ingress carrying only the
#: `.local` name is a correct-looking Ingress with an unchanged 404 — the second half of
#: the same defect, and the one most likely to be reintroduced by someone copying miLLM's
#: first rule and stopping there.
REQUIRED_HOSTS: frozenset[str] = frozenset(
    {
        "k8s-miforge.hitsai.local",
        "k8s-miforge.hitsai.net",
        "miforge.hitsai.net",
    }
)

BACKEND_SERVICE = "miforge-backend"
FRONTEND_SERVICE = "miforge-frontend"


def _documents(name: str) -> list[dict[str, Any]]:
    path = GITOPS / name
    assert path.is_file(), f"no manifest at {path}"
    documents = list(yaml.safe_load_all(path.read_text("utf-8")))
    assert all(d is not None for d in documents), (
        f"{name} contains an empty YAML document — usually a stray '---'. Harmless to "
        f"`kubectl apply`, but it means a document boundary is not where it looks."
    )
    return documents


@pytest.fixture(scope="module")
def kustomization() -> dict[str, Any]:
    return _documents("kustomization.yaml")[0]


@pytest.fixture(scope="module")
def ingresses() -> list[dict[str, Any]]:
    return [d for d in _documents("ingress.yaml") if d.get("kind") == "Ingress"]


@pytest.fixture(scope="module")
def backend_deployment() -> dict[str, Any]:
    for document in _documents("backend.yaml"):
        if document.get("kind") == "Deployment":
            return document
    pytest.fail("backend.yaml declares no Deployment")


@pytest.fixture(scope="module")
def pod_spec(backend_deployment: dict[str, Any]) -> dict[str, Any]:
    return backend_deployment["spec"]["template"]["spec"]


def _rules(ingress: dict[str, Any]) -> dict[str, dict[str, str]]:
    """``{host: {path: "service:port"}}`` for one Ingress."""
    out: dict[str, dict[str, str]] = {}
    for rule in ingress["spec"]["rules"]:
        paths = {}
        for entry in rule["http"]["paths"]:
            service = entry["backend"]["service"]
            paths[entry["path"]] = f"{service['name']}:{service['port']['number']}"
        out[rule["host"]] = paths
    return out


def _artifact_roots() -> dict[str, str]:
    """``{setting name: container path}`` for every artifact root, from **both** modules.

    The central assertion of this file depends on this function being wrong only in the
    safe direction. It reads the live settings classes rather than a hand-written list, so
    a root added to either module appears here automatically — which is exactly what did
    not happen when ``packs_root`` was added to a module no manifest list had read.
    """
    from core.config import Settings
    from packs.settings import PackSettings

    roots: dict[str, str] = {}
    for model, names in (
        (Settings, ("checkpoint_root", "trainer_state_root", "parts_root")),
        (PackSettings, ("packs_root",)),
    ):
        for name in names:
            field = model.model_fields[name]
            roots[name] = str(field.default)
    return roots


class TestTheIngressExistsAndIsApplied:
    """The 404. Both halves: the object, and its listing in the kustomization."""

    def test_an_ingress_manifest_exists(self, ingresses: list[dict[str, Any]]) -> None:
        assert ingresses, (
            "gitops/pilot/ingress.yaml declares no Ingress. Without one the frontend "
            "Service is a bare ClusterIP, nothing outside the cluster can reach it, and "
            "ingress-nginx answers every request from its default backend with a 404."
        )

    def test_it_is_listed_in_the_kustomization(self, kustomization: dict[str, Any]) -> None:
        """⚠ The half that is easy to forget and produces an identical symptom.

        Kustomize applies only what ``resources:`` names. ``storage.yaml`` is deliberately
        unlisted and says so in its own header; an unlisted ``ingress.yaml`` is not a
        decision, it is a file ArgoCD never applies.
        """
        resources = kustomization.get("resources") or []
        assert "ingress.yaml" in resources, (
            "ingress.yaml is not in kustomization.yaml's `resources:`, so ArgoCD never "
            "applies it. The manifest would exist in git and the cluster would still 404 "
            f"— which looks exactly like the fix being in place. resources={resources}"
        )

    def test_the_ingress_class_is_the_one_this_cluster_serves(
        self, ingresses: list[dict[str, Any]]
    ) -> None:
        """A wrong class is picked up by no controller — indistinguishable from no Ingress.

        ``public`` is microk8s' bundled ingress-nginx, and all three siblings use it.
        """
        for ingress in ingresses:
            assert ingress["spec"].get("ingressClassName") == "public", (
                f"{ingress['metadata']['name']} does not set ingressClassName: public. "
                f"miLLM, miStudio and miDataworks all do; a class no controller serves "
                f"means the object is created, reports no address, and routes nothing."
            )

    def test_everything_is_in_the_miforge_namespace(self, ingresses: list[dict[str, Any]]) -> None:
        for ingress in ingresses:
            assert ingress["metadata"].get("namespace") == "miforge", (
                f"{ingress['metadata']['name']} is not in the miforge namespace, so its "
                f"`backend.service.name` references resolve in the wrong namespace and "
                f"the rule points at nothing."
            )


class TestEveryHostnameTheTunnelMightPresentIsRouted:
    """The second half of the 404, and the half an Ingress alone does not fix."""

    def test_the_http_ingress_serves_every_required_host(
        self, ingresses: list[dict[str, Any]]
    ) -> None:
        http = next(i for i in ingresses if i["metadata"]["name"] == "miforge-ingress")
        hosts = set(_rules(http))
        missing = REQUIRED_HOSTS - hosts
        assert not missing, (
            f"miforge-ingress does not route {sorted(missing)}. A host rule matches only "
            f"the `Host:` header that arrives, and the Cloudflared tunnel presents the "
            f"PUBLIC name — so dropping the `.hitsai.net` forms leaves a correct-looking "
            f"Ingress and an unchanged 404."
        )

    def test_no_rule_omits_its_host(self, ingresses: list[dict[str, Any]]) -> None:
        """⚠ A host-less rule is not a convenient catch-all; it is a suite-wide hazard.

        In ingress-nginx a rule with no ``host`` becomes the default server for the whole
        class, so miForge would answer for every hostname no other Ingress claims —
        including anything meant for miLLM or miStudio whose own rule stopped matching.
        That turns one app's outage into cross-app mis-routing. None of the three siblings
        does it. If the tunnel presents a fourth name, add the name.
        """
        for ingress in ingresses:
            for rule in ingress["spec"]["rules"]:
                assert rule.get("host"), (
                    f"{ingress['metadata']['name']} has a rule with no `host:`, which "
                    f"makes miForge the default server for the entire `public` class."
                )

    def test_every_host_is_routed_identically(self, ingresses: list[dict[str, Any]]) -> None:
        """Fixing one host and not its sibling is the suite's most repeated anti-pattern.

        miStudio's own ingress says so in a comment, about the internet-facing half. The
        rules are written longhand here — matching the three deployments proven to work in
        this cluster rather than using a YAML anchor — so this assertion is what supplies
        the guarantee an anchor would have given, with no deployment risk.
        """
        for ingress in ingresses:
            rules = _rules(ingress)
            distinct = {tuple(sorted(paths.items())) for paths in rules.values()}
            assert len(distinct) == 1, (
                f"{ingress['metadata']['name']}'s hosts are not routed identically: "
                f"{rules}. One host was changed and the others were not."
            )


class TestThePathsReachTheServicesThatServeThem:
    def test_api_goes_to_the_backend_and_everything_else_to_the_frontend(
        self, ingresses: list[dict[str, Any]]
    ) -> None:
        http = next(i for i in ingresses if i["metadata"]["name"] == "miforge-ingress")
        for host, paths in _rules(http).items():
            assert paths.get("/api") == f"{BACKEND_SERVICE}:8000", (
                f"{host} does not route /api to {BACKEND_SERVICE}:8000 (got "
                f"{paths.get('/api')!r})."
            )
            assert paths.get("/") == f"{FRONTEND_SERVICE}:80", (
                f"{host} does not route / to {FRONTEND_SERVICE}:80 (got {paths.get('/')!r}). "
                f"Port 80 on the Service, 8080 in the container — the image is "
                f"nginx-unprivileged and cannot bind a privileged port."
            )

    def test_one_api_prefix_covers_the_whole_http_surface(self) -> None:
        """Checked against `api/app.py`, which is why there are no `/v1` or `/docs` rules.

        miLLM's Ingress carries five backend rules because its app serves those paths at
        the root. miForge's serves everything under ``/api``, so copying that list would
        have added four rules routing nothing. If a route ever escapes ``/api`` this test
        fails and the Ingress needs a rule — which is the useful direction for it to fail.
        """
        app_source = (REPO_ROOT / "backend" / "src" / "api" / "app.py").read_text("utf-8")
        for declaration in (
            'openapi_url="/api/openapi.json"',
            'docs_url="/api/docs"',
            "redoc_url=None",
            'include_router(health_router, prefix="/api")',
        ):
            assert declaration in app_source, (
                f"api/app.py no longer contains {declaration!r}. The Ingress routes only "
                f"/api to the backend; a path that moved outside it is now unreachable "
                f"from outside the cluster."
            )

    def test_the_socket_io_path_has_its_own_ingress_with_the_websocket_annotation(
        self, ingresses: list[dict[str, Any]]
    ) -> None:
        """⚠ Without this the live channel works by accident and degrades silently.

        ``/socket.io`` would otherwise fall under ``/`` to the frontend pod, whose nginx
        does proxy it — so it would appear to work, with the upgrade declared by nothing.
        Socket.IO then falls back to HTTP long-polling and the run view looks slow rather
        than dead, which is the harder failure to diagnose.
        """
        websocket = next(
            (i for i in ingresses if i["metadata"]["name"] == "miforge-websocket-ingress"),
            None,
        )
        assert websocket is not None, (
            "there is no miforge-websocket-ingress. All three siblings split the "
            "WebSocket path into its own object."
        )
        annotations = websocket["metadata"].get("annotations", {})
        assert (
            annotations.get("nginx.ingress.kubernetes.io/websocket-services") == BACKEND_SERVICE
        ), "the websocket-services annotation must name the backend Service"
        for host, paths in _rules(websocket).items():
            assert paths.get("/socket.io") == f"{BACKEND_SERVICE}:8000", (
                f"{host} does not route /socket.io to the backend (got "
                f"{paths.get('/socket.io')!r})."
            )

    def test_no_upgrade_headers_are_set_by_hand(self) -> None:
        """The warning all three siblings carry, honoured rather than rediscovered.

        > "Do NOT add configuration-snippet with Upgrade headers — it causes header
        >  duplication."
        """
        source = (GITOPS / "ingress.yaml").read_text("utf-8")
        offending = [
            line
            for line in source.splitlines()
            if "configuration-snippet" in line and not line.lstrip().startswith("#")
        ]
        assert not offending, (
            f"ingress.yaml sets a configuration-snippet: {offending}. "
            f"`websocket-services` already handles the upgrade; adding the headers "
            f"duplicates them and the handshake fails."
        )

    def test_the_socket_io_path_is_read_from_the_module_that_mounts_it(self) -> None:
        source = (REPO_ROOT / "backend" / "src" / "api" / "sockets.py").read_text("utf-8")
        assert 'socketio_path="socket.io"' in source, (
            "api/sockets.py no longer mounts Socket.IO at 'socket.io', so the "
            "miforge-websocket-ingress rule points at a path nothing serves."
        )


class TestTheSchemaIsMigratedBeforeAnythingServes:
    """Compose defect 3, in the cluster. Green health over a dead API."""

    def test_an_init_container_runs_the_migration(self, pod_spec: dict[str, Any]) -> None:
        init = {c["name"]: c for c in pod_spec.get("initContainers", [])}
        assert "migrate" in init, (
            "the backend pod has no `migrate` init container, so nothing in the cluster "
            "runs `alembic upgrade head`. backend/Dockerfile has no SERVICE_TYPE "
            "entrypoint (unlike miStudio's and miDataworks', which migrate for the API), "
            "so the API container runs uvicorn and only uvicorn. A fresh namespace then "
            "serves against an empty schema while /api/health reports postgres ok — it "
            "checks connectivity, not schema."
        )
        assert init["migrate"]["command"] == ["alembic", "upgrade", "head"], (
            f"the migrate init container does not run the migration: "
            f"{init['migrate'].get('command')!r}"
        )

    def test_it_runs_before_every_application_container(self, pod_spec: dict[str, Any]) -> None:
        """An init container, not a Job: the API cannot start beside it.

        That is ``depends_on: service_completed_successfully`` expressed in Kubernetes,
        which is the shape ``docker-compose.yml`` says it wanted ("it is the shape
        Kubernetes wants too (an init container or a Job)").
        """
        names = [c["name"] for c in pod_spec.get("initContainers", [])]
        assert "migrate" in names, "migrate must be an initContainer, not a container"
        assert "migrate" not in [c["name"] for c in pod_spec["containers"]], (
            "migrate appears among the app containers, where it would run beside the API "
            "rather than before it"
        )

    def test_it_cannot_race_another_migration(self, backend_deployment: dict[str, Any]) -> None:
        """Measured, not assumed: parallel migration of one database is 29 errors.

        ``backend-tests.yml`` records ``-n auto`` against one database as 29 errors and 1
        failure where serial gives 458 passed. ``replicas: 1`` plus ``Recreate`` is what
        makes exactly one of these ever run — the same guarantee that makes exactly one
        Celery Beat safe.
        """
        spec = backend_deployment["spec"]
        assert spec.get("replicas") == 1, (
            f"replicas is {spec.get('replicas')!r}; two pods would each run the migrate "
            f"init container against the same database."
        )
        assert spec.get("strategy", {}).get("type") == "Recreate", (
            "strategy is not Recreate, so a rolling update would start a new pod (and a "
            "second migration) while the old one still runs."
        )

    def test_the_image_carries_what_a_migration_needs(self) -> None:
        """The init container exits non-zero without these, and nothing starts.

        They were added to the image on 2026-10-14 for the Compose fix; before that this
        init container could not have worked either.
        """
        dockerfile = (REPO_ROOT / "backend" / "Dockerfile").read_text("utf-8")
        for required in ("alembic.ini", "backend/alembic/"):
            assert required in dockerfile, (
                f"backend/Dockerfile does not COPY {required}, so `alembic upgrade head` "
                f"cannot run in the image and the pod never leaves Init."
            )

    def test_the_migration_touches_the_database_and_nothing_else(
        self, pod_spec: dict[str, Any]
    ) -> None:
        """No artifact volumes and no GPU env, so the claim is checkable not asserted."""
        migrate = next(c for c in pod_spec["initContainers"] if c["name"] == "migrate")
        assert not migrate.get("volumeMounts"), (
            f"the migrate init container mounts {migrate.get('volumeMounts')}; a "
            f"migration touches the database only, and withholding the volumes is what "
            f"makes that true rather than stated."
        )
        env = {e["name"] for e in migrate.get("env", [])}
        assert "DATABASE_URL" in env, "the migration has no DATABASE_URL"
        assert not env & {
            "NVIDIA_VISIBLE_DEVICES",
            "NVIDIA_DRIVER_CAPABILITIES",
        }, "the migrate init container is given GPU environment it has no use for"

    def test_the_password_is_expanded_before_it_is_used(self, pod_spec: dict[str, Any]) -> None:
        """Kubernetes' ``$(VAR)`` only sees variables defined EARLIER in the same list.

        The file header says so for the app containers; the init container is subject to
        the same rule, and an unexpanded ``$(POSTGRES_PASSWORD)`` is a literal in the DSN
        — so the migration fails to authenticate rather than failing to find a variable.
        """
        migrate = next(c for c in pod_spec["initContainers"] if c["name"] == "migrate")
        order = [e["name"] for e in migrate["env"]]
        assert order.index("POSTGRES_PASSWORD") < order.index(
            "DATABASE_URL"
        ), f"POSTGRES_PASSWORD must be declared before DATABASE_URL; got {order}"


class TestEveryArtifactRootHasDurableStorage:
    """Compose defect 4, in the cluster — and the reason it reads two settings modules."""

    def test_the_settings_actually_declare_some(self) -> None:
        """A positive control. An empty mapping would make every assertion below vacuous.

        This is the shape the schema-sync test guards against by naming internal contracts
        rather than giving them an empty vendor list: a check that performs nothing looks
        exactly like a check that passed.
        """
        roots = _artifact_roots()
        assert len(roots) >= 4, (
            f"expected at least four artifact roots across core.config and packs.settings, "
            f"found {roots}. If a root was removed, remove its volume too; if this import "
            f"stopped seeing one, every assertion below just stopped testing anything."
        )
        assert "packs_root" in roots, (
            "packs_root is no longer visible to this test, which is precisely how its "
            "volume went missing in the first place."
        )

    def test_every_root_is_a_persistent_volume_claim_on_the_pod(
        self, pod_spec: dict[str, Any]
    ) -> None:
        claims = {
            v["name"]: v["persistentVolumeClaim"]["claimName"]
            for v in pod_spec["volumes"]
            if "persistentVolumeClaim" in v
        }
        mounted_paths = {
            m["mountPath"]: m["name"]
            for container in pod_spec["containers"]
            for m in container.get("volumeMounts", [])
        }
        for setting, path in _artifact_roots().items():
            assert path in mounted_paths, (
                f"{setting} is {path} and no container mounts it. ADR-004 makes every "
                f"artifact root AUTHORITATIVE, so an unmounted one lives in the "
                f"container's writable layer and `strategy: Recreate` destroys it on "
                f"every deploy. Mounted paths: {sorted(mounted_paths)}"
            )
            volume = mounted_paths[path]
            assert volume in claims, (
                f"{path} is mounted from volume {volume!r}, which is not a "
                f"persistentVolumeClaim. An emptyDir or hostPath here would come up "
                f"healthy and lose the artifact."
            )

    def test_every_claim_is_declared_in_storage_yaml(self, pod_spec: dict[str, Any]) -> None:
        """An absent claim leaves the pod Pending, which is the INTENDED failure.

        Louder than a DirectoryOrCreate hostPath coming up healthy against an empty
        directory — the failure miStudio hit twice and Neuronpedia sat on for 19 days.
        """
        declared = {
            d["metadata"]["name"]
            for d in _documents("storage.yaml")
            if d.get("kind") == "PersistentVolumeClaim"
        }
        referenced = {
            v["persistentVolumeClaim"]["claimName"]
            for v in pod_spec["volumes"]
            if "persistentVolumeClaim" in v
        }
        missing = referenced - declared
        assert not missing, (
            f"the backend pod references claims {sorted(missing)} that storage.yaml does "
            f"not declare, so `kubectl apply -f gitops/pilot/storage.yaml` cannot create "
            f"them and the pod sits Pending forever."
        )

    def test_the_volume_root_is_chowned_for_the_application_user(
        self, pod_spec: dict[str, Any]
    ) -> None:
        """⚠ Without this the directory is kubelet-created and root-owned.

        The process runs as uid 10001, and ``packs/settings.py`` mkdirs its root on
        demand — right for a writer, wrong on a read path, and reindex already hit
        ``PermissionError`` from exactly that. A mounted volume the process cannot write
        is not storage.
        """
        owner = next(c for c in pod_spec["initContainers"] if c["name"] == "data-volume-owner")
        script = "\n".join(str(part) for part in owner["command"])
        for setting, path in _artifact_roots().items():
            assert path in script, (
                f"{setting}'s root {path} is not chowned by the data-volume-owner init "
                f"container, so it is created root-owned and uid 10001 cannot write to it."
            )

    def test_the_config_map_names_every_root(self) -> None:
        """Documentation, asserted — an operator should not read Python to find the data.

        It is also the list whose incompleteness hid the pack store: the ConfigMap header
        claimed every key was a ``core/config.py`` field, so nobody looked in a second
        module.
        """
        config = _documents("config.yaml")[0]["data"]
        for setting, path in _artifact_roots().items():
            key = setting.upper()
            assert key in config, (
                f"config.yaml does not declare {key}. It equals the field default, so it "
                f"configures nothing — but every list that omitted this root is how its "
                f"volume came to be missing."
            )
            assert config[key] == path, (
                f"config.yaml sets {key}={config[key]!r} while {setting} defaults to "
                f"{path!r}, and the volume is mounted at the default. One of the two is "
                f"serving from a path with no volume behind it."
            )
