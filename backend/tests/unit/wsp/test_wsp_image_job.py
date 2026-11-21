"""The private-CI image job's shape, its driver and its layer check (feature 016; FR-016.25;
FTASKS 4.3a; R1B-M4, R3A-12, R3A-23, P62).

The job is three things, each pinned here by behaviour rather than by reading text:

- the workflow job ``wsp-worker-image`` in ``backend-tests.yml`` runs the driver, with no
  ``continue-on-error`` anywhere (a job that cannot fail does not gate);
- the driver ``backend/scripts/wsp_worker_image_job.py`` builds ``prod`` then ``test``, runs the
  layer check, then pytest with the require switch, and returns the first failure's code; the
  calls are recorded with their arguments and counted;
- the layer check refuses each way ``test`` could be more than ``prod`` plus test tooling;
- ``tests/image/harness.py::worker_run_args`` carries the pod's flags, which every live test uses.
"""

from __future__ import annotations

import importlib.util
import io
import os
import re
import sys
import tarfile
from pathlib import Path
from types import ModuleType
from typing import Any

import pytest
import yaml

from tests.image import harness

REPO = Path(__file__).resolve().parents[4]
WORKFLOW = REPO / ".github" / "workflows" / "backend-tests.yml"
SCRIPT = REPO / "backend" / "scripts" / "wsp_worker_image_job.py"


def _job_module() -> ModuleType:
    spec = importlib.util.spec_from_file_location("wsp_worker_image_job", SCRIPT)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules["wsp_worker_image_job"] = module
    spec.loader.exec_module(module)
    return module


job = _job_module()


# ----------------------------------------------------------------------------- the workflow


def _workflow() -> dict[str, Any]:
    doc: dict[str, Any] = yaml.safe_load(WORKFLOW.read_text())
    doc["on"] = doc.pop(True, doc.get("on"))
    return doc


def test_the_job_runs_on_every_push_to_main_and_every_pull_request() -> None:
    doc = _workflow()
    assert "wsp-worker-image" in doc["jobs"]
    assert doc["on"]["push"]["branches"] == ["main"] and "pull_request" in doc["on"]
    assert "if" not in doc["jobs"]["wsp-worker-image"], "a condition could skip the job"


def test_the_job_runs_the_driver_and_nothing_can_turn_a_failure_green() -> None:
    j = _workflow()["jobs"]["wsp-worker-image"]
    assert "continue-on-error" not in j
    assert all("continue-on-error" not in s for s in j["steps"])
    runs = [s["run"].strip() for s in j["steps"] if "run" in s]
    assert runs == [
        "pip install --no-deps -r backend/requirements-wsp-worker-test.txt",
        "python backend/scripts/wsp_worker_image_job.py --prefix wsp-ci",
    ]
    assert j["permissions"] == {"contents": "read"}
    assert isinstance(j["timeout-minutes"], int)


def test_the_test_tooling_is_pinned_to_the_backend_lock() -> None:
    pins = job.pinned(job.TEST_REQUIREMENTS.read_text())
    lock = {
        job.canon(m.group(1)): m.group(2)
        for m in re.finditer(
            r"^([A-Za-z0-9_.\-]+)==(\S+)", (REPO / "constraints.txt").read_text(), re.M
        )
    }
    assert "pytest" in pins
    assert {k: lock.get(k) for k in pins} == pins


# ----------------------------------------------------------------------------- the driver


class Recorder:
    def __init__(self, codes: dict[str, int] | None = None) -> None:
        self.calls: list[tuple[list[str], dict[str, str] | None]] = []
        self.codes = codes or {}
        self.layers: list[tuple[str, str]] = []
        self.problems: list[str] = []
        self.cleaned: list[str] = []

    def run(self, cmd: list[str], env: dict[str, str] | None = None) -> int:
        self.calls.append((list(cmd), env))
        joined = " ".join(cmd)
        return next((c for key, c in self.codes.items() if key in joined), 0)

    def layer_problems(self, prod: str, test: str) -> list[str]:
        self.layers.append((prod, test))
        return self.problems

    def cleanup(self, run_id: str) -> None:
        self.cleaned.append(run_id)


@pytest.fixture
def rec(monkeypatch: pytest.MonkeyPatch) -> Recorder:
    r = Recorder()
    monkeypatch.setattr(job, "run", r.run)
    monkeypatch.setattr(job, "layer_problems", r.layer_problems)
    monkeypatch.setattr(job, "cleanup", r.cleanup)
    return r


def test_the_driver_builds_checks_then_tests_with_the_require_switch(rec: Recorder) -> None:
    assert job.main(["--prefix", "pfx"]) == 0
    cmds = [c for c, _ in rec.calls]
    build = ["docker", "build", "-f", "backend/Dockerfile.wsp-worker", "--target"]
    assert cmds[:2] == [
        [*build, "prod", "-t", "pfx/wsp-worker:prod", "."],
        [*build, "test", "-t", "pfx/wsp-worker:test", "."],
    ]
    assert not any("push" in " ".join(c) for c in cmds)
    assert rec.layers == [("pfx/wsp-worker:prod", "pfx/wsp-worker:test")]
    assert len(cmds) == 3
    pytest_cmd, env = rec.calls[2]
    assert pytest_cmd[1:3] == ["-m", "pytest"] and pytest_cmd[-1] == "backend/tests/image"
    assert env is not None
    assert env["MIFORGE_REQUIRE_WSP_IMAGE"] == "1"
    assert env["MIFORGE_WSP_WORKER_IMAGE"] == "pfx/wsp-worker:test"
    assert env["MIFORGE_WSP_WORKER_PROD_IMAGE"] == "pfx/wsp-worker:prod"
    assert "MIFORGE_WSP_IMAGE_SOURCE" not in env
    assert len(rec.cleaned) == 1 and rec.cleaned[0] == env["MIFORGE_WSP_IMAGE_RUN_ID"]


def test_a_failing_test_run_fails_the_job(rec: Recorder) -> None:
    rec.codes = {"pytest": 1}
    assert job.main(["--tests", "placeholder.py"]) == 1
    assert rec.calls[-1][0][-1] == "placeholder.py"
    assert len(rec.cleaned) == 1


def test_a_failing_layer_check_stops_before_the_tests(rec: Recorder) -> None:
    rec.problems = ["a file that is neither test tooling nor the allow-list: etc/evil"]
    assert job.main([]) != 0
    assert not any("pytest" in " ".join(c) for c, _ in rec.calls)
    assert len(rec.cleaned) == 1


@pytest.mark.parametrize("target", ["prod", "test"])
def test_a_failing_build_stops_the_job(rec: Recorder, target: str) -> None:
    rec.codes = {f"--target {target}": 2}
    assert job.main([]) == 2
    assert rec.layers == [] and not any("pytest" in " ".join(c) for c, _ in rec.calls)


# ----------------------------------------------------------------------------- the layer check

SITE = "usr/local/lib/python3.13/site-packages"
PINS = {"pytest": "9.1.1", "pluggy": "1.7.0"}


def _dist(name: str, version: str, files: list[str]) -> tuple[dict[str, str], dict[str, bytes]]:
    info = f"{SITE}/{name}-{version}.dist-info"
    record = [*files, f"{name}-{version}.dist-info/RECORD"]
    entries = {os.path.normpath(f"{SITE}/{f}"): "file" for f in record}
    entries[info] = "dir"
    return entries, {f"{info}/RECORD": "\n".join(f"{r},," for r in record).encode()}


def _good() -> list[Any]:
    a, ad = _dist("pytest", "9.1.1", ["pytest/__init__.py", "../../../bin/pytest"])
    b, bd = _dist("pluggy", "1.7.0", ["pluggy/__init__.py"])
    layer = job.Layer(
        {**a, **b, f"{SITE}/pytest": "dir", f"{SITE}/pluggy": "dir", "tmp": "dir"}, {**ad, **bd}
    )
    return [layer]


BASE = {
    "tmp",
    "usr",
    "usr/local",
    "usr/local/bin",
    "usr/local/lib",
    "usr/local/lib/python3.13",
    SITE,
}


def _check(extra: list[Any], prod: list[str] | None = None, test: list[str] | None = None,
           allowlist: bytes | None = None) -> list[str]:  # fmt: skip
    prod = prod if prod is not None else ["a", "b"]
    test = test if test is not None else ["a", "b", "c"]
    problems: list[str] = job.check_layers(prod, test, extra, BASE, PINS, allowlist)
    return problems


def test_prod_plus_test_tooling_passes() -> None:
    assert _check(_good()) == []


def test_a_test_image_not_built_on_prod_is_refused() -> None:
    assert any("prefix" in p for p in _check(_good(), test=["a", "x", "c"]))


def test_an_extra_file_is_refused() -> None:
    extra = _good()
    extra[0].entries["etc/cron.d/evil"] = "file"
    assert _check(extra) == [
        "a file that is neither test tooling nor the allow-list: etc/cron.d/evil"
    ]


def test_a_deletion_of_a_prod_file_is_refused() -> None:
    extra = _good()
    extra[0].entries[
        "usr/local/lib/python3.13/site-packages/wsp_worker.pth".replace(
            "wsp_worker.pth", ".wh.wsp_worker.pth"
        )
    ] = "file"
    assert any("whiteout" in p for p in _check(extra))


def test_a_new_empty_directory_is_refused() -> None:
    extra = _good()
    extra[0].entries["opt/smuggled"] = "dir"
    assert _check(extra) == ["a new directory with no allowed file in it: opt/smuggled"]


def test_an_unpinned_distribution_is_refused() -> None:
    extra = _good()
    c, cd = _dist("requests", "2.0", ["requests/__init__.py"])
    extra[0].entries.update(c)
    extra[0].data.update(cd)
    assert any("not exactly" in p for p in _check(extra))


def test_the_allow_list_is_required_once_it_exists_and_must_match() -> None:
    assert any("allowlist.py is missing" in p for p in _check(_good(), allowlist=b"BLOCKS = ()\n"))
    extra = _good()
    extra.append(job.Layer({"opt/wsp/wsp_gateway": "dir", job.ALLOWLIST_IN_IMAGE: "file"},
                           {job.ALLOWLIST_IN_IMAGE: b"BLOCKS = ()\n"}))  # fmt: skip
    assert _check(extra, test=["a", "b", "c", "d"], allowlist=b"BLOCKS = ()\n") == []
    assert any("differs" in p for p in _check(extra, test=["a", "b", "c", "d"], allowlist=b"X\n"))
    # Before 6.5 the file must not be there at all.
    assert any("neither" in p for p in _check(extra, test=["a", "b", "c", "d"]))


def test_read_layer_reads_kinds_and_records() -> None:
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w") as t:
        for name, kind, body in [
            ("./usr/", tarfile.DIRTYPE, b""),
            (f"{SITE}/x-1.dist-info/RECORD", tarfile.REGTYPE, b"x/__init__.py,,\n"),
            ("usr/.wh.gone", tarfile.REGTYPE, b""),
            ("usr/link", tarfile.SYMTYPE, b""),
        ]:
            info = tarfile.TarInfo(name)
            info.type, info.size = kind, len(body)
            t.addfile(info, io.BytesIO(body))
    buf.seek(0)
    with tarfile.open(fileobj=buf) as t:
        layer = job.read_layer(t)
    assert layer.entries == {
        "usr": "dir",
        f"{SITE}/x-1.dist-info/RECORD": "file",
        "usr/.wh.gone": "file",
        "usr/link": "link",
    }
    assert layer.data == {f"{SITE}/x-1.dist-info/RECORD": b"x/__init__.py,,\n"}


# ----------------------------------------------------------------------------- the pod's flags


def _args(**kw: Any) -> list[str]:
    volumes = {p: f"v{i}" for i, p in enumerate(harness.VOLUMES)}
    return harness.worker_run_args(
        name="w", sandbox="sb", image="img", volumes=volumes, termlog="/host/termlog",
        env=dict(harness.POD_ENV), **kw,
    )  # fmt: skip


def _pairs(args: list[str]) -> list[tuple[str, str]]:
    return [(args[i], args[i + 1]) for i in range(len(args) - 1) if args[i].startswith("-")]


def test_the_worker_runs_with_the_pods_flags() -> None:
    args = _args()
    pairs = _pairs(args)
    for flag in [
        ("--network", "container:sb"),
        ("--ipc", "container:sb"),
        ("--restart", "always"),
        ("--user", "10002:10002"),
        ("--cap-drop", "ALL"),
        ("--security-opt", "no-new-privileges"),
        ("--pids-limit", "256"),
    ]:
        assert flag in pairs, flag
    assert "--read-only" in args
    mounts = [v for k, v in pairs if k == "--mount"]
    for i, path in enumerate(harness.VOLUMES):
        assert f"type=volume,src=v{i},dst={path},volume-nocopy" in mounts
    assert "type=bind,src=/host/termlog,dst=/tmp/termination-log" in mounts
    assert all(m.endswith(",readonly") for m in mounts if "/opt/wsp-tests" in m)
    assert not any(m.startswith("type=tmpfs") for m in mounts)
    for forbidden in ("--tmpfs", "--pid", "--privileged", "--cap-add", "--userns"):
        assert forbidden not in args
    env = [v for k, v in pairs if k == "-e"]
    assert sorted(env) == sorted(f"{k}={v}" for k, v in harness.POD_ENV.items())
    assert args[-2:] == ["img", "supervisor"]


def test_only_the_refusal_test_replaces_dev_mqueue() -> None:
    mounts = [v for k, v in _pairs(_args(mqueue=False, restart="no")) if k == "--mount"]
    assert [m for m in mounts if m.startswith("type=tmpfs")] == ["type=tmpfs,dst=/dev/mqueue"]


def test_the_pod_env_is_the_manifests() -> None:
    """The harness's env is the Deployment's literal list (with ``ci`` for ``prod``)."""
    from tests.support.wsp_manifests import one

    pod = one("Deployment", "miforge-wsp-worker")["spec"]["template"]["spec"]
    (container,) = pod["containers"]
    manifest = {e["name"]: e["value"] for e in container["env"]}
    assert {**manifest, "WSP_ENVIRONMENT": "ci"} == harness.POD_ENV
