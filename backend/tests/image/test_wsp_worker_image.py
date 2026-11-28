"""What the WSP worker image holds and what a session's child can see in it (feature 016;
FR-016.9, FR-016.25; FTASKS 8.5; WR-12).

Image facts are read from the **prod** target (what the mirror publishes;
``MIFORGE_WSP_WORKER_PROD_IMAGE``), with ``python -I`` in a throwaway container. The environment and
filesystem dump is a fixture session on the **test** target, with the pod's flags (``harness.py``).
Skips by name outside the image job; ``MIFORGE_REQUIRE_WSP_IMAGE=1`` makes a skip a failure.

The README example ports of ``weightspace`` 0.3.0 (FTASKS 8.5) ran here through its adapter until
018-P4, and went with it (FR-018.18). The engine runs a real session in this image in
``test_engine_probe_image.py``.
"""

from __future__ import annotations

import json
import re
from collections.abc import Iterator

import pytest

from tests.image import harness as h
from tests.support.wsp_expected import assert_child_env
from tests.support.wsp_supervisor import push_doc
from wsp import result_archive

#: FR-016.9: none of these may be importable in the worker image.
FORBIDDEN = ("torch", "sqlalchemy", "psycopg", "psycopg2", "redis", "celery")
#: FR-018.18: the retired provider and the SDKs it alone pulled in.
RETIRED = ("weightspace", "anthropic", "openai")
#: FR-016.25: the only places a session's child may create a file. ``/dev/mqueue`` is the pod's
#: IPC namespace, which the child cannot be kept out of and the supervisor wipes as PID 1
#: (OVN-wsp-forge-18).
WRITABLE_ALLOWED = ("/work", "/out", "/tmp", "/dev/shm", "/dev/mqueue")


@pytest.fixture(scope="module")
def prod() -> str:
    return h.require_image(h.PROD_IMAGE_ENV)


@pytest.fixture(scope="module")
def test_image() -> str:
    return h.require_image()


@pytest.fixture
def pod(test_image: str) -> Iterator[h.Pod]:
    with h.pod(test_image) as p:
        yield p


def _pins() -> dict[str, str]:
    pins: dict[str, str] = {}
    for raw in h.REQUIREMENTS.read_text().splitlines():
        line = raw.split("#", 1)[0].strip()
        if line:
            m = re.fullmatch(r"([A-Za-z0-9_.\-]+)(?:\[[^\]]*\])?==(\S+)", line)
            assert m, line
            pins[m.group(1).lower()] = m.group(2)
    return pins


@pytest.mark.parametrize("module", FORBIDDEN)
def test_forbidden_modules_cannot_be_imported(prod: str, module: str) -> None:
    r = h.run_once(prod, "-c", f"import {module}")
    assert r.returncode != 0, f"{module} is importable in the worker image"
    assert "ModuleNotFoundError" in r.stderr, r.stderr


def test_pinned_versions(prod: str) -> None:
    """The engine's own dependencies are exactly their lock lines (FR-018.18: ``httpx2`` is named
    directly in the lock since ``weightspace`` and its SDKs left the image), read from the
    installed distributions in the prod image, not from module attributes."""
    names = ("httpx2", "httpcore2", "numpy", "safetensors")
    r = h.run_once(
        prod,
        "-c",
        "import json, importlib.metadata as m\n"
        f"print(json.dumps({{n: m.version(n) for n in {names!r}}}))\n",
    )
    assert r.returncode == 0, r.stderr
    pins = _pins()
    assert json.loads(r.stdout) == {n: pins[n] for n in names}


@pytest.mark.parametrize("module", RETIRED)
def test_retired_provider_and_its_sdks_are_not_in_the_image(prod: str, module: str) -> None:
    """FR-018.18: ``weightspace`` and the two SDKs only it pulled in are gone from the image."""
    r = h.run_once(prod, "-c", f"import {module}")
    assert r.returncode != 0, f"{module} is importable in the worker image"
    assert "ModuleNotFoundError" in r.stderr, r.stderr


def test_libseccomp_resolves(prod: str) -> None:
    r = h.run_once(prod, "-c", "import ctypes.util; print(ctypes.util.find_library('seccomp'))")
    assert r.returncode == 0 and r.stdout.strip().startswith("libseccomp.so"), r.stdout + r.stderr


def test_the_loader_refuses_pt(prod: str) -> None:
    """WR-12: a pickled format is refused by name before any read (the image has no torch)."""
    r = h.run_once(
        prod,
        "-c",
        "from pathlib import Path\n"
        "from wsp_worker import bundle\n"
        "try:\n"
        "    bundle.read_heads(Path('/work/heads.pt'))\n"
        "except bundle.BundleError as e:\n"
        "    print(e.code)\n",
    )
    assert r.stdout.strip() == "wsp_bundle_format_refused", r.stdout + r.stderr


def test_prod_has_no_pytest(prod: str) -> None:
    """P62: the published target carries no test tooling."""
    r = h.run_once(prod, "-c", "import pytest")
    assert "ModuleNotFoundError" in r.stderr, r.stdout + r.stderr


def test_environment_dump(pod: h.Pod) -> None:
    """FR-016.25: a fixture session's CHILD has exactly the allow-listed environment, no secrets
    mount, and can create files only under the pod's writable mounts (control C25: a fake
    ``DATABASE_URL`` in the child env turns this red)."""
    _, members, _, record = pod.start().run(["dump_env", "dump_writable", "write_result"])
    assert record["exit_code"] == 0
    env = json.loads(members["env.json"])
    assert_child_env(env)
    seen = json.loads(members["writable.json"])
    assert seen["secrets"] == []
    outside = [
        p
        for p in seen["writable"]
        if not any(p == root or p.startswith(root + "/") for root in WRITABLE_ALLOWED)
    ]
    assert outside == [], outside
    assert {"/work", "/out", "/tmp", "/dev/shm"} <= set(seen["writable"]), seen["writable"]


def test_in_container_suite(test_image: str, pod: h.Pod) -> None:
    """``tests/image/in_container/`` under pytest INSIDE the test target, as PID 1 of a throwaway
    container with the pod's flags (its own PID namespace, never ``--pid host``): a supervisor
    that is not PID 1 harms nothing of its uid (C10ab)."""
    suite = ("-I", "-m", "pytest", "-p", "no:cacheprovider", "-q", "-rs")
    w = pod.start(
        entrypoint="python",
        command=(*suite, f"{h.MOUNT_ROOT}/in_container"),
        env={"WSP_IN_IMAGE_TESTS": "1"},
        restart="no",
        wait_ready=False,
    )
    code = w.wait_exit(timeout=300)
    logs = w.logs()
    assert code == 0, logs
    # Every in-container test ran and passed: the count is derived from the files (it read a
    # literal 1 until wsp-forge3 added two allowlist tests, and the job was not re-run).
    expected = sum(
        len(re.findall(r"^(?:    )?def test_", f.read_text(), flags=re.M))
        for f in h.IN_CONTAINER.glob("test_*.py")
    )
    # Two since 018-P4 (FTASKS 15.5): the engine's real-body test and the non-PID-1 supervisor;
    # weightspace's real-body file, which would have skipped there, was deleted.
    assert expected >= 2, expected
    assert re.search(rf"\b{expected} passed\b", logs) and "skipped" not in logs, logs
    assert "failed" not in logs and "error" not in logs.lower(), logs


def test_intake_accepts_a_real_worker_tar(pod: h.Pod) -> None:
    """R3A-14: the tar the REAL supervisor builds in the image, as PID 1 with the pod's flags,
    passes the intake's hostile-archive reader (``wsp.result_archive``, the module
    ``wsp.runs.intake_result`` reads every result through; it is standard library only so this
    host, which has pytest and nothing else, can import it). Its local twin, through
    ``intake_result`` itself, is ``test_dispatch.py::test_one_pass_end_to_end_against_the_real_
    supervisor``. Control ``C-8.5-intake-members`` (``logs/stdout.txt`` dropped from
    ``RESULT_MEMBERS``) turns this red."""
    w = pod.start()
    boot = w.boot()
    run_id = "wsr_intake_image"
    actions = ["write_transcript", "print:hello", "write_result"]
    reply = w.session(push_doc(boot, actions=actions, run_id=run_id))
    assert reply.status == 202, reply
    w.wait_state("finished", timeout=90)
    raw, _digest = w.result_raw()
    members = result_archive.read_result_archive(raw)
    assert {"supervisor.json", "result.json", "transcript.jsonl", "logs/stdout.txt"} <= set(members)
    record = result_archive.check_record(members, run_id, boot)
    assert record["exit_code"] == 0 and record["deadline_hit"] is False
    result_archive.check_transcript(members["transcript.jsonl"])
    result = result_archive.json_object(members["result.json"], "result_json")
    assert result == {"stop_reason": "submitted"}
    assert b"hello" in members["logs/stdout.txt"]
