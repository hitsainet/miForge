"""Running the WSP worker IMAGE the way its pod runs, and driving it from outside (feature 016;
FR-016.25; FTDD §8.1; FTID §3.2; FTASKS 4.3a, 4.4, 8.5).

**The harness is the backend.** It talks HTTP to the supervisor from the host, so the supervisor
really is PID 1 of its container. Each test gets a fresh :class:`Pod`:

- a long-lived **sandbox** container holding the network namespace (on its own ``--internal``
  network: no route out) and a shareable IPC namespace, as a pod's pause container does. The worker
  joins both (``--network``/``--ipc container:<sandbox>``), so SysV objects, POSIX queues and
  sockets in ``TIME_WAIT`` outlive a worker restart exactly as they outlive a container restart in
  the pod;
- **named volumes** for ``/work``, ``/out``, ``/tmp`` and ``/dev/shm``, never ``--tmpfs``: Docker
  recreates a ``--tmpfs`` on every start, while a pod's ``emptyDir`` lives as long as the pod
  (P62, R2A-7). ``/out``, ``/tmp`` and ``/dev/shm`` are memory-backed (the pod's ``medium:
  Memory``) and the sandbox mounts them too, so they are not unmounted between worker runs;
  ``/work`` is on disk, owned by uid 10002 through Docker's copy-up of the image's ``/work``;
- ``/tmp/termination-log`` **bind-mounted from a host file**, as the kubelet mounts the
  ``terminationMessagePath``: the supervisor cannot unlink it, only truncate it;
- the pod's flags: ``--read-only``, ``--user 10002:10002``, ``--cap-drop ALL``, ``--security-opt
  no-new-privileges``, ``--pids-limit 256``, the pod's exact env, and ``--restart always``.
  Between sessions the harness ``docker wait``s for the container's OWN exit and requires exit 0
  and a new ``boot_id``; it never runs ``docker restart``, which would kill a mutant that refused
  to exit (C10o).

The tests are mounted read-only (``backend/tests/support/wsp_children`` and
``backend/tests/image/in_container``); nothing of them is in the image.

Environment:

- ``MIFORGE_WSP_WORKER_IMAGE``: the image's ``test`` target (required to run anything here);
  ``MIFORGE_WSP_WORKER_PROD_IMAGE``: its ``prod`` target.
- ``MIFORGE_REQUIRE_WSP_IMAGE=1`` (set by ``backend/scripts/wsp_worker_image_job.py``) turns every
  skip into a FAILURE, so the job cannot go green by skipping what it exists to run.
- ``MIFORGE_WSP_IMAGE_PREFIX``: the name prefix of every container, volume and network created
  (default ``wsp-image``); every one also carries the label ``miforge.wsp-image-test=<run id>`` and
  is removed by exact name when its test ends.
- ``MIFORGE_WSP_IMAGE_SOURCE`` (controls only): a ``wsp_worker`` directory bind-mounted read-only
  over the image's ``/opt/wsp/wsp_worker``, so a mutated source is run without a rebuild. CI never
  sets it.
"""

from __future__ import annotations

import hashlib
import http.client
import io
import json
import os
import secrets
import shutil
import subprocess
import tarfile
import tempfile
import time
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, NoReturn

import pytest

from tests.support.wsp_supervisor import Reply, encode_session, push_doc

IMAGE_ENV = "MIFORGE_WSP_WORKER_IMAGE"
PROD_IMAGE_ENV = "MIFORGE_WSP_WORKER_PROD_IMAGE"
REQUIRE_ENV = "MIFORGE_REQUIRE_WSP_IMAGE"
PREFIX_ENV = "MIFORGE_WSP_IMAGE_PREFIX"
SOURCE_ENV = "MIFORGE_WSP_IMAGE_SOURCE"
RUN_ID_ENV = "MIFORGE_WSP_IMAGE_RUN_ID"
LABEL = "miforge.wsp-image-test"
RUN_ID = os.environ.get(RUN_ID_ENV) or secrets.token_hex(4)

BACKEND = Path(__file__).resolve().parents[2]
CHILDREN = BACKEND / "tests" / "support" / "wsp_children"
IN_CONTAINER = Path(__file__).resolve().parent / "in_container"
REQUIREMENTS = BACKEND / "requirements-wsp-worker.txt"
MOUNT_ROOT = "/opt/wsp-tests"
SCRIPTED = f"{MOUNT_ROOT}/wsp_children/scripted.py"
NO_FILTER = f"{MOUNT_ROOT}/wsp_children/no_filter.py"
PACKAGE_IN_IMAGE = "/opt/wsp/wsp_worker"

UID = 10002
PORT = 9000
PIDS_LIMIT = 256
#: The pod's container env, exactly (``gitops/pilot/wsp-worker.yaml``), with ``ci`` for prod:
#: ``WSP_TEST_SCRIPT`` and ``WSP_TEST_HOOKS`` are refused in prod (FTID §3.2).
POD_ENV = {
    "NVIDIA_VISIBLE_DEVICES": "void",
    "WSP_WORKER_PORT": str(PORT),
    "WSP_ENVIRONMENT": "ci",
    "OMP_NUM_THREADS": "1",
    "OPENBLAS_NUM_THREADS": "1",
    "MKL_NUM_THREADS": "1",
}
#: Pod mount -> (memory-backed?, tmpfs size). The pod's emptyDirs: /work on disk, the rest memory.
VOLUMES = {
    "/work": (False, ""),
    "/out": (True, "128m"),
    "/tmp": (True, "64m"),
    "/dev/shm": (True, "64m"),
}


def absent(reason: str) -> NoReturn:
    """Skip the running test by name, or fail it under ``MIFORGE_REQUIRE_WSP_IMAGE=1``."""
    test = os.environ.get("PYTEST_CURRENT_TEST", "this test").rsplit(" (", 1)[0]
    what = f"WSP worker image: {test} cannot run here: {reason}."
    if os.environ.get(REQUIRE_ENV) == "1":
        pytest.fail(f"{what} {REQUIRE_ENV}=1 says it must, so this is a failure.", pytrace=False)
    pytest.skip(
        f"{what} Run `python backend/scripts/wsp_worker_image_job.py` (it builds the image and "
        f"sets {IMAGE_ENV}); CI's `wsp-worker-image` job does, with {REQUIRE_ENV}=1."
    )


def docker(
    *args: str, timeout: float = 120, check: bool = True, input: str | None = None
) -> subprocess.CompletedProcess[str]:
    r = subprocess.run(
        ["docker", *args], capture_output=True, text=True, timeout=timeout, input=input
    )
    if check and r.returncode != 0:
        raise AssertionError(f"docker {' '.join(args[:3])} ... failed: {r.stderr.strip()}")
    return r


def require_image(env: str = IMAGE_ENV) -> str:
    image = os.environ.get(env)
    if not image:
        absent(f"{env} is not set")
    if shutil.which("docker") is None:
        absent("the docker CLI is not installed")
    if docker("image", "inspect", image, check=False).returncode != 0:
        absent(f"{env}={image} is not a local image")
    return image


def prefix() -> str:
    return os.environ.get(PREFIX_ENV, "wsp-image")


def label_args() -> list[str]:
    return ["--label", f"{LABEL}={RUN_ID}"]


def worker_run_args(
    *,
    name: str,
    sandbox: str,
    image: str,
    volumes: dict[str, str],
    termlog: str,
    env: dict[str, str],
    restart: str = "always",
    pids_limit: int | None = PIDS_LIMIT,
    init: bool = False,
    mqueue: bool = True,
    source: str | None = None,
    entrypoint: str | None = None,
    command: tuple[str, ...] = ("supervisor",),
    extra_mounts: tuple[str, ...] = (),
) -> list[str]:
    """``docker run`` arguments for one worker container with the pod's flags (FTDD §8.1).

    Pinned flag by flag by ``unit/wsp/test_wsp_image_job.py``; every live test starts its worker
    through here, so a flag dropped here is dropped for all of them."""
    args = ["run", "-d", "--name", name, *label_args()]
    args += ["--network", f"container:{sandbox}", "--ipc", f"container:{sandbox}"]
    args += ["--restart", restart, "--read-only", "--user", f"{UID}:{UID}"]
    args += ["--cap-drop", "ALL", "--security-opt", "no-new-privileges"]
    # The pod's limits are cpu 2 and memory 6Gi; 2g here keeps a shared CI host safe.
    args += ["--cpus", "2", "--memory", "2g"]
    if pids_limit is not None:
        args += ["--pids-limit", str(pids_limit)]
    if init:
        args.append("--init")
    for path in VOLUMES:
        # `volume-nocopy`: Docker would otherwise copy the IMAGE's directory (its owner and mode,
        # e.g. /tmp's root 1777) onto the volume at first mount; the pod's emptyDir is the
        # volume's own (uid 10002 here, fsGroup in the pod).
        args += ["--mount", f"type=volume,src={volumes[path]},dst={path},volume-nocopy"]
    args += ["--mount", f"type=bind,src={termlog},dst=/tmp/termination-log"]
    args += ["--mount", f"type=bind,src={CHILDREN},dst={MOUNT_ROOT}/wsp_children,readonly"]
    args += ["--mount", f"type=bind,src={IN_CONTAINER},dst={MOUNT_ROOT}/in_container,readonly"]
    if source:
        args += ["--mount", f"type=bind,src={source},dst={PACKAGE_IN_IMAGE},readonly"]
    for mount in extra_mounts:
        args += ["--mount", mount]
    if not mqueue:
        # Only for the refusal test: a tmpfs over the IPC namespace's mqueue mount.
        args += ["--mount", "type=tmpfs,dst=/dev/mqueue"]
    for key, value in env.items():
        args += ["-e", f"{key}={value}"]
    if entrypoint:
        args += ["--entrypoint", entrypoint]
    return [*args, image, *command]


# ----------------------------------------------------------------------------- one pod


@dataclass
class Pod:
    """A sandbox, its network and volumes, and at most one worker container at a time."""

    image: str
    base: str = field(default_factory=lambda: f"{prefix()}-{RUN_ID}-{secrets.token_hex(3)}")
    ip: str = ""
    termdir: Path = field(default_factory=lambda: Path(tempfile.mkdtemp(prefix="wsp-termlog-")))
    _created: list[tuple[str, str]] = field(default_factory=list)

    @property
    def sandbox(self) -> str:
        return f"{self.base}-sandbox"

    @property
    def worker(self) -> str:
        return f"{self.base}-worker"

    @property
    def termlog(self) -> Path:
        return self.termdir / "termination-log"

    def volume(self, path: str) -> str:
        return f"{self.base}-{path.strip('/').replace('/', '-')}"

    def create(self) -> Pod:
        self.termdir.chmod(0o755)
        self.termlog.write_text("")
        self.termlog.chmod(0o666)  # the kubelet's file is writable by the container's uid
        net = f"{self.base}-net"
        docker("network", "create", "--internal", *label_args(), net)
        self._created.append(("network", net))
        mounts: list[str] = []
        for path, (memory, size) in VOLUMES.items():
            vol = self.volume(path)
            tmpfs = f"o=uid={UID},gid={UID},mode=0700,size={size}"
            opts = (
                [
                    "--driver",
                    "local",
                    "--opt",
                    "type=tmpfs",
                    "--opt",
                    "device=tmpfs",
                    "--opt",
                    tmpfs,
                ]
                if memory
                else []
            )
            docker("volume", "create", *opts, *label_args(), vol)
            self._created.append(("volume", vol))
            # The sandbox holds every volume, so a memory volume outlives a worker restart (as an
            # emptyDir outlives a container restart); /work keeps the image's path for copy-up.
            keep = path if path == "/work" else f"/keep{path}"
            mounts += ["--mount", f"type=volume,src={vol},dst={keep}"]
        docker(
            "run", "-d", "--name", self.sandbox, *label_args(), "--network", net,
            "--ipc", "shareable", "--read-only", "--user", f"{UID}:{UID}", "--cap-drop", "ALL",
            "--security-opt", "no-new-privileges", "--pids-limit", "8", *mounts,
            "--entrypoint", "python", self.image, "-I", "-c", "import time; time.sleep(86400)",
        )  # fmt: skip
        self._created.append(("container", self.sandbox))
        inspect = docker("inspect", "-f", "{{json .NetworkSettings.Networks}}", self.sandbox)
        self.ip = json.loads(inspect.stdout)[net]["IPAddress"]
        return self

    def start(
        self,
        *,
        script: str | None = SCRIPTED,
        env: dict[str, str] | None = None,
        wait_ready: bool = True,
        **kwargs: Any,
    ) -> Worker:
        """Start the worker container (removing a previous one by its exact name)."""
        docker("rm", "-f", self.worker, check=False)
        full = {**POD_ENV, **({"WSP_TEST_SCRIPT": script} if script else {}), **(env or {})}
        args = worker_run_args(
            name=self.worker,
            sandbox=self.sandbox,
            image=self.image,
            volumes={p: self.volume(p) for p in VOLUMES},
            termlog=str(self.termlog),
            env=full,
            source=os.environ.get(SOURCE_ENV) or None,
            **kwargs,
        )
        docker(*args)
        if ("container", self.worker) not in self._created:
            self._created.append(("container", self.worker))
        w = Worker(self)
        if wait_ready:
            w.wait_ready()
        return w

    def remove(self) -> None:
        for kind, name in reversed(self._created):
            if kind == "container":
                docker("rm", "-f", "-v", name, check=False)
        for kind, name in reversed(self._created):
            if kind != "container":
                docker(kind, "rm", name, check=False)
        shutil.rmtree(self.termdir, ignore_errors=True)


@contextmanager
def pod(image: str) -> Iterator[Pod]:
    p = Pod(image)
    try:
        yield p.create()
    finally:
        p.remove()


# ----------------------------------------------------------------------------- one worker


@dataclass
class Worker:
    pod: Pod

    @property
    def name(self) -> str:
        return self.pod.worker

    def request(
        self,
        method: str,
        path: str,
        body: bytes | None = None,
        headers: dict[str, str] | None = None,
        timeout: float = 30,
    ) -> Reply:
        conn = http.client.HTTPConnection(self.pod.ip, PORT, timeout=timeout)
        try:
            conn.request(method, path, body=body, headers=headers or {})
            r = conn.getresponse()
            return Reply(r.status, r.read(), {k.lower(): v for k, v in r.getheaders()})
        finally:
            conn.close()

    def ready(self) -> Reply:
        return self.request("GET", "/ready", timeout=5)

    def wait_ready(self, timeout: float = 30, not_boot: str | None = None) -> str:
        """The ``boot_id`` once ``/ready`` answers 200 (and differs from ``not_boot``)."""
        end = time.monotonic() + timeout
        last: object = None
        while time.monotonic() < end:
            try:
                r = self.ready()
                last = r
                if r.status == 200 and r.json()["boot_id"] != not_boot:
                    return str(r.json()["boot_id"])
            except OSError as exc:
                last = exc
            time.sleep(0.1)
        raise AssertionError(f"{self.name}: no ready boot in {timeout}s ({last}); {self.logs()}")

    def boot(self) -> str:
        return str(self.ready().json()["boot_id"])

    def status(self) -> dict[str, Any]:
        r = self.request("GET", "/status")
        assert r.status == 200, r
        return dict(r.json())

    def post_json(self, path: str, doc: dict[str, Any]) -> Reply:
        return self.request(
            "POST", path, json.dumps(doc).encode(), {"Content-Type": "application/json"}
        )

    def session(self, push: dict[str, Any], files: list[tuple[str, bytes]] | None = None) -> Reply:
        ctype, body = encode_session(push, files or [])
        return self.request("POST", "/session", body, {"Content-Type": ctype})

    def wait_state(self, state: str, timeout: float = 60) -> dict[str, Any]:
        end = time.monotonic() + timeout
        current: object = None
        while True:
            try:
                current = self.status()
            except OSError as exc:  # the connection cap closes a request at accept when full
                current = exc
            if isinstance(current, dict) and current["state"] == state:
                return current
            assert time.monotonic() < end, f"never reached {state}: {current}"
            time.sleep(0.1)

    def result_raw(self) -> tuple[bytes, str]:
        """``GET /result`` as the backend's driver reads it: the tar's bytes and their sha256."""
        r = self.request("GET", "/result")
        assert r.status == 200, r
        digest = r.headers["x-content-sha256"]
        assert hashlib.sha256(r.body).hexdigest() == digest
        return r.body, digest

    def result(self) -> tuple[dict[str, bytes], str]:
        body, digest = self.result_raw()
        with tarfile.open(fileobj=io.BytesIO(body)) as tar:
            members = {m.name: tar.extractfile(m).read() for m in tar.getmembers()}  # type: ignore[union-attr]
        return members, digest

    def run(
        self, actions: list[str], **push_kw: Any
    ) -> tuple[str, dict[str, bytes], str, dict[str, Any]]:
        """One session: ``(boot_id, members, sha256, supervisor.json)`` at ``finished``."""
        boot = self.boot()
        reply = self.session(push_doc(boot, actions=actions, **push_kw))
        assert reply.status == 202, reply
        self.wait_state("finished", timeout=float(push_kw.get("time_budget_s", 30)) + 60)
        members, digest = self.result()
        return boot, members, digest, json.loads(members["supervisor.json"])

    def ack(self, boot: str, digest: str, run_id: str = "wsr_test") -> Reply:
        return self.post_json("/ack", {"boot_id": boot, "run_id": run_id, "result_sha256": digest})

    def exit_on(self, trigger: Callable[[], object], timeout: float = 30) -> int:
        """Start ``docker wait`` BEFORE the trigger, then return the container's own exit code.

        Never ``docker restart``: a mutant that refuses to exit shows here as a timeout."""
        waiter = subprocess.Popen(
            ["docker", "wait", self.name], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True
        )
        try:
            time.sleep(0.5)  # let `docker wait` attach before the exit it waits for
            trigger()
            out, err = waiter.communicate(timeout=timeout)
        except subprocess.TimeoutExpired:
            waiter.kill()
            waiter.communicate()
            raise AssertionError(f"{self.name} did not exit within {timeout}s") from None
        assert waiter.returncode == 0, err
        return int(out.strip())

    def wait_exit(self, timeout: float = 30) -> int:
        """For ``--restart no`` containers: the exit code once stopped."""
        r = docker("wait", self.name, timeout=timeout)
        return int(r.stdout.strip())

    def restart_count(self) -> int:
        return int(docker("inspect", "-f", "{{.RestartCount}}", self.name).stdout.strip())

    def logs(self) -> str:
        r = docker("logs", self.name, check=False)
        return r.stdout + r.stderr

    def host_pid(self) -> int:
        return int(docker("inspect", "-f", "{{.State.Pid}}", self.name).stdout.strip())

    def processes(self) -> list[str]:
        """Every process in the container, from the host (``docker top`` adds none)."""
        lines = docker("top", self.name).stdout.strip().splitlines()
        return lines[1:]

    def exec_python(
        self, code: str, user: str = f"{UID}:{UID}"
    ) -> subprocess.CompletedProcess[str]:
        return docker("exec", "--user", user, self.name, "python", "-I", "-c", code, check=False)

    def refusal(self) -> dict[str, Any]:
        lines = [ln for ln in self.logs().splitlines() if ln.startswith("{")]
        return dict(json.loads(lines[-1])) if lines else {"logs": self.logs()}


def run_once(image: str, *args: str, timeout: float = 120) -> subprocess.CompletedProcess[str]:
    """``docker run --rm`` of ``python -I`` in the image: no supervisor, no pod (image facts)."""
    return docker(
        "run", "--rm", *label_args(), "--network", "none", "--read-only", "--user", f"{UID}:{UID}",
        "--cap-drop", "ALL", "--security-opt", "no-new-privileges", "--entrypoint", "python",
        image, "-I", *args, timeout=timeout, check=False,
    )  # fmt: skip


__all__ = [
    "IMAGE_ENV",
    "NO_FILTER",
    "POD_ENV",
    "PROD_IMAGE_ENV",
    "SCRIPTED",
    "Pod",
    "Worker",
    "absent",
    "docker",
    "pod",
    "push_doc",
    "require_image",
    "run_once",
    "worker_run_args",
]
